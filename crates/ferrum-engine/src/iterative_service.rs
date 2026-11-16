//! One bounded iterative worker for typed model requests and outputs.
//! Runtime grants and all continuation/retirement owners remain here.

use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryReservation};
use ferrum_types::RequestCorrelation;
use serde::Serialize;
use std::{
    any::Any,
    collections::VecDeque,
    error::Error,
    fmt,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex, MutexGuard,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use uuid::Uuid;

use ferrum_types::IterativeBatchPolicy as BatchPolicy;
pub type GeneratorError = Box<dyn Error + Send + 'static>;

/// A batch rejection has no pending device work or recovery owners and leaves
/// its generator usable. It carries only a diagnostic, not an arbitrary error
/// whose cleanup obligations could be lost. All other errors are fatal and
/// retain their original owner until explicit service retirement.
#[derive(Debug)]
pub enum BatchError {
    /// Only preparation may return this variant. No device work was submitted.
    Capacity {
        proof: DeviceCapacityExceeded,
        diagnostic: String,
    },
    /// Explicit Host admission pressure; distinct from device capacity search.
    Overloaded(String),
    /// Iterative preparation only: the accepted request can resume when the
    /// original subscribed Host account refunds capacity. No native work or
    /// recovery owner is held by this result. The worker checks both the typed
    /// temporary pressure and account identity before retaining the request.
    AwaitHostCapacity {
        account: HostMemoryAccount,
        pressure: ferrum_runtime::HostMemoryError,
    },
    /// Completion only: decoder admission did not submit work and every
    /// allocation rollback/release was acknowledged. A live survivor may
    /// release a required native pin by advancing; there is no external native
    /// capacity subscription and an idle worker must reject this finitely.
    AwaitDenoisingProgress {
        stage: ExecutionStageKey,
        diagnostic: String,
    },
    /// Initialization admission only, before any native work/recovery owner.
    /// Live denoising members may release capacity; idle workers reject this
    /// finitely because this protocol has no external native-release wake.
    InitializationAwaitDenoisingProgress {
        diagnostic: String,
    },
    /// Typed request input rejection, with no native work/recovery obligation.
    InvalidInput(String),
    Rejected(String),
    Fatal(GeneratorError),
}
impl From<GeneratorError> for BatchError {
    fn from(error: GeneratorError) -> Self {
        Self::Fatal(error)
    }
}
impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity { diagnostic, .. }
            | Self::Overloaded(diagnostic)
            | Self::AwaitDenoisingProgress { diagnostic, .. }
            | Self::InitializationAwaitDenoisingProgress { diagnostic } => f.write_str(diagnostic),
            Self::AwaitHostCapacity { pressure, .. } => fmt::Display::fmt(pressure, f),
            Self::InvalidInput(message) => f.write_str(message),
            Self::Rejected(message) => f.write_str(message),
            Self::Fatal(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl Error for BatchError {}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ServiceConfig {
    pub batching: BatchPolicy,
    pub maximum_pending: usize,
    pub maximum_pending_bytes: usize,
    pub max_num_seqs: usize,
    pub request_batch_max_wait_ms: u64,
}

/// An owned input. Byte counts include only backing retained by this input;
/// the worker separately charges its actual queue/job control allocation.
pub trait ServiceRequest: Send + 'static {
    fn pending_bytes(&self) -> usize;
    /// Uncharged input backing; pre-admitted Host buffers must not be charged twice.
    fn admission_bytes(&self) -> usize;
    fn kind(&self) -> &'static str;
    fn record_admission(&self, timing: AdmissionTiming) {
        tracing::debug!(target: "ferrum_engine::iterative_service::timing",
            request_kind = self.kind(), batch_index = timing.batch_index,
            batch_size = timing.batch_size, queue_wait_us = timing.queue_wait_us,
            queue_and_prepare_us = timing.queue_and_prepare_us,
            preparation_us = timing.preparation_us, "request admitted to execution");
    }
}

/// Observed Host intervals for an admitted request. Payload metadata remains
/// with the request adapter, which may record its own stable observation schema.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionTiming {
    pub batch_index: usize,
    pub batch_size: usize,
    pub queue_wait_us: u64,
    pub queue_and_prepare_us: u64,
    pub preparation_us: u64,
}

fn host_admission_error(error: ferrum_runtime::HostMemoryError) -> BatchError {
    if matches!(error, ferrum_runtime::HostMemoryError::Capacity { .. }) {
        BatchError::Overloaded(format!("Host admission: {error}"))
    } else {
        BatchError::Rejected(format!("Host preparation: {error}"))
    }
}

/// A single-use, generator-owned preparation. The default empty owner keeps
/// existing generators usable; capacity-aware generators retain compiled plans.
#[derive(Default)]
pub struct BatchPreparation(
    pub(crate) Option<Box<dyn Any + Send>>,
    pub(crate) Option<batch_efficiency::ExecutionContext>,
    pub(crate) Option<DeviceCapacityReport>,
);
impl BatchPreparation {
    pub fn new<T: Any + Send>(prepared: T) -> Self {
        Self(Some(Box::new(prepared)), None, None)
    }
    /// Borrow the original typed preparation for nonblocking admission. A
    /// foreign or already consumed owner is rejected without replacing it.
    pub fn get_mut<T: Any + Send>(&mut self) -> Result<&mut T, BatchError> {
        self.0
            .as_mut()
            .and_then(|owner| owner.downcast_mut::<T>())
            .ok_or_else(|| {
                BatchError::Rejected("image preparation belongs to another generator".into())
            })
    }
    /// Optional compiled demand for keeping the workload's weights resident.
    /// Exceeding this advisory boundary does not make staged execution unsafe.
    pub fn with_resident_capacity(mut self, report: DeviceCapacityReport) -> Self {
        self.2 = Some(report);
        self
    }
    pub fn into_inner<T: Any + Send>(self) -> Result<T, BatchError> {
        self.0
            .and_then(|owner| owner.downcast::<T>().ok())
            .map(|owner| *owner)
            .ok_or_else(|| {
                BatchError::Rejected("image preparation belongs to another generator".into())
            })
    }
}

/// Opaque adapter-produced identity for metadata reuse. Equality partitions
/// planning/cost history; it is never a proof of batch compatibility or admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterDomainKey([u8; 32]);
impl AdapterDomainKey {
    pub const fn from_fingerprint(fingerprint: [u8; 32]) -> Self {
        Self(fingerprint)
    }
}

/// Already-held protocol-independent resources that overlap engine execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct PreparationResources {
    pub output_credit_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceErrorKind {
    InvalidRequest,
    QueueFull,
    Stopped,
    Cancelled,
    Generation,
    WorkerPanic,
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct ServiceError {
    kind: ServiceErrorKind,
    message: String,
    correlation: Option<RequestCorrelation>,
}
impl ServiceError {
    pub fn kind(&self) -> ServiceErrorKind {
        self.kind
    }
    pub fn correlation(&self) -> Option<RequestCorrelation> {
        self.correlation
    }
}
impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl Error for ServiceError {}
pub(crate) fn service_error(kind: ServiceErrorKind, message: impl Into<String>) -> ServiceError {
    ServiceError {
        kind,
        message: message.into(),
        correlation: None,
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ServiceSnapshot {
    pub accepting: bool,
    pub active_requests: usize,
    pub pending_requests: usize,
    pub preparing_requests: usize,
    pub pending_bytes: usize,
    pub maximum_pending_bytes: usize,
    pub rejected_requests: u64,
    pub executed_batches: u64,
    pub executed_requests: u64,
    pub last_batch_size: usize,
    pub capacity_reductions: u64,
    pub batching: BatchPolicy,
    pub maximum_pending: usize,
    pub max_num_seqs: usize,
}

pub enum WorkerFailure {
    Generation(GeneratorError),
    Panic(Box<dyn Any + Send + 'static>),
    Concurrent {
        execution: Box<WorkerFailure>,
        preparation: Box<WorkerFailure>,
    },
}
impl fmt::Debug for WorkerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(error) => f.debug_tuple("Generation").field(error).finish(),
            Self::Concurrent {
                execution,
                preparation,
            } => f
                .debug_struct("Concurrent")
                .field("execution", execution)
                .field("preparation", preparation)
                .finish(),
            Self::Panic(_) => f.write_str("Panic(original payload retained)"),
        }
    }
}

impl fmt::Display for WorkerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
impl std::error::Error for WorkerFailure {}

/// Execution authority owned by one worker. Request interpretation belongs to
/// its adapter; scheduling and lifecycle never inspect model/modality payloads.
pub(crate) trait BatchWorker: Send + 'static {
    type Request: ServiceRequest;
    type Output: Send + 'static;
    fn lifecycle_observer(&self) -> Option<LifecycleObserverHandle> {
        None
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn DenoisingGenerator<Request = Self::Request, Output = Self::Output>> {
        None
    }
    fn preparation_worker(
        &self,
    ) -> Option<Box<dyn BatchWorker<Request = Self::Request, Output = Self::Output>>> {
        None
    }
    fn retained_output_bytes(&self) -> u64 {
        0
    }
    fn adapter_domain_key(&self) -> Option<AdapterDomainKey> {
        None
    }
    fn capacity_search_context(
        &self,
        _requests: &[&Self::Request],
    ) -> Option<CapacitySearchContext> {
        None
    }
    fn batch_execution_feedback(
        &self,
        _outputs: &[Self::Output],
    ) -> Option<BatchExecutionFeedback> {
        None
    }
    fn execution_output_bytes(
        &self,
        _requests: &[&Self::Request],
    ) -> Result<Option<usize>, BatchError> {
        Ok(None)
    }
    fn preparation_bytes(&self, _request: &Self::Request) -> Result<usize, BatchError> {
        Ok(0)
    }
    fn prepare(
        &mut self,
        _requests: &mut [&mut Self::Request],
    ) -> Result<BatchPreparation, BatchError> {
        Ok(BatchPreparation::default())
    }
    fn try_admit_prepared(
        &mut self,
        _prepared: &mut BatchPreparation,
        _resources: PreparationResources,
    ) -> Result<bool, BatchError> {
        Ok(true)
    }
    fn execute(
        &mut self,
        prepared: BatchPreparation,
        requests: Vec<Self::Request>,
    ) -> Result<Vec<Self::Output>, BatchError>;
    fn compatible(&self, a: &Self::Request, b: &Self::Request) -> bool;
    fn maximum_batch_size(&self, request: &Self::Request, configured: usize) -> usize;
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        Ok(())
    }
    /// Adapter-owned evidence from the one normal retirement. The core does
    /// not interpret this metadata or use it as live resource authority.
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        self.shutdown()?;
        Ok(None)
    }
}
pub(crate) struct Queued<R: ServiceRequest, O> {
    pub(crate) trace: trace::RequestTrace,
    pub(crate) request: R,
    pub(crate) adapter: Option<Arc<AdapterDomainKey>>,
    pub(crate) submitted: Instant,
    pub(crate) cancelled: Arc<AtomicBool>,
    pub(crate) response: oneshot::Sender<Result<O, ServiceError>>,
    pub(crate) output: Option<QueuedOutput>,
    pub(crate) _prompt_reservation: Option<HostMemoryReservation>,
    pub(crate) preprocessing_bytes: usize,
    pub(crate) preparation_duration: Duration,
    /// A reduced FIFO tail must finish before newer arrivals enlarge its batch.
    pub(crate) preparation_group: Option<Arc<()>>,
}
impl<R: ServiceRequest, O> Queued<R, O> {
    pub(crate) fn pending_bytes(&self) -> usize {
        self.request
            .pending_bytes()
            .saturating_add(std::mem::size_of::<Self>())
            .saturating_add(self.preprocessing_bytes)
    }
}
pub(crate) struct State<R: ServiceRequest, O> {
    pub(crate) queue: VecDeque<Queued<R, O>>,
    // Field order frees the deque before refunding its backing reservation.
    pub(crate) queue_backing: Option<HostMemoryReservation>,
    pub(crate) accepting: bool,
    pub(crate) closing: bool,
    pub(crate) active: usize,
    pub(crate) preparing: usize,
    pub(crate) preparing_bytes: usize,
    pub(crate) rejected: u64,
    pub(crate) executed_batches: u64,
    pub(crate) executed_requests: u64,
    pub(crate) last_batch_size: usize,
    pub(crate) capacity_reductions: u64,
}
impl<R: ServiceRequest, O> State<R, O> {
    pub(crate) fn reserve_queue_slot(
        &mut self,
        host: Option<&HostMemoryAccount>,
        maximum_pending: usize,
    ) -> Result<(), ServiceError> {
        let full = || {
            service_error(
                ServiceErrorKind::QueueFull,
                "cannot grow the image request queue within its Host budget",
            )
        };
        // A prepared candidate can return its tail while newer jobs arrive.
        // Its slots remain available, so requeue never allocates implicitly.
        let required = self
            .queue
            .len()
            .checked_add(self.preparing)
            .and_then(|slots| slots.checked_add(1))
            .ok_or_else(full)?;
        if required <= self.queue.capacity() {
            return Ok(());
        }
        let mut capacity = self
            .queue
            .capacity()
            .saturating_mul(2)
            .max(required)
            .min(maximum_pending);
        // Reserve the complete replacement while the old allocation remains
        // charged. This covers allocation/copy peak, not just the net growth.
        let reserve = |capacity: usize| {
            let bytes = capacity
                .checked_mul(std::mem::size_of::<Queued<R, O>>())
                .filter(|bytes| *bytes <= isize::MAX as usize)
                .ok_or_else(full)?;
            host.map(|host| host.reserve(HostMemoryClass::InputPreprocess, bytes))
                .transpose()
                .map_err(|error| {
                    service_error(
                        ServiceErrorKind::QueueFull,
                        format!("image queue backing Host admission failed: {error}"),
                    )
                })
        };
        let reservation = match reserve(capacity) {
            Ok(reservation) => reservation,
            Err(_) if capacity > required => {
                // Amortized growth is optional: admit the actual next slot if
                // the geometric spare capacity would exceed the Host budget.
                capacity = required;
                reserve(capacity)?
            }
            Err(error) => return Err(error),
        };
        let mut queue = VecDeque::new();
        queue.try_reserve_exact(capacity).map_err(|_| full())?;
        // Keep the charged capacity equal to the retained allocation; never
        // accept an allocator-provided larger deque on the smaller grant.
        if queue.capacity() != capacity {
            return Err(full());
        }
        queue.extend(self.queue.drain(..));
        self.queue = queue;
        self.queue_backing = reservation;
        Ok(())
    }
}
pub(crate) struct Shared<R: ServiceRequest, O> {
    pub(crate) trace: Option<LifecycleObserverHandle>,
    pub(crate) state: Mutex<State<R, O>>,
    pub(crate) changed: Condvar,
    pub(crate) resources: Arc<ResourceSignal>,
    pub(crate) config: ServiceConfig,
    pub(crate) host: Option<HostMemoryAccount>,
    pub(crate) adapter: Option<Arc<AdapterDomainKey>>,
}
impl<R: ServiceRequest, O> Shared<R, O> {
    pub(crate) fn lock(&self) -> MutexGuard<'_, State<R, O>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(crate) fn stop(&self) {
        let mut state = self.lock();
        state.accepting = false;
        state.closing = true;
        self.changed.notify_all();
        self.resources.notify();
    }
    fn wake_cancellation(&self) {
        // Resource callbacks never acquire the queue mutex. Publish their
        // wake before synchronizing with the declared collection wait.
        self.resources.notify();
        // Pair with the waiter's predicate check and atomic unlock/wait.
        // Atomic cancellation alone cannot prevent a lost Condvar wake.
        let _state = self.lock();
        self.changed.notify_all();
    }
}

/// A request's cancellation authority, independent of its result waiter.
///
/// Clones refer to the original request and worker wake. Dropping a handle does
/// not cancel the request; cancellation remains cooperative at the worker's
/// existing safe boundaries and does not revoke an already published result.
#[derive(Clone)]
pub struct JobCancellationHandle {
    cancelled: Arc<AtomicBool>,
    wake: std::task::Waker,
}
impl JobCancellationHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.wake.wake_by_ref();
    }
}

// An external cancellation handle must not keep a retired service's queue,
// outputs or Host parent alive. The worker and result Job retain those owners.
struct CancellationWake<R: ServiceRequest, O>(std::sync::Weak<Shared<R, O>>);
impl<R: ServiceRequest, O: Send + 'static> std::task::Wake for CancellationWake<R, O> {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        if let Some(shared) = self.0.upgrade() {
            shared.wake_cancellation();
        }
    }
}

pub(crate) struct Job<R: ServiceRequest, O> {
    pub(crate) correlation: RequestCorrelation,
    pub(crate) receiver: Option<oneshot::Receiver<Result<O, ServiceError>>>,
    pub(crate) cancelled: Arc<AtomicBool>,
    pub(crate) shared: Arc<Shared<R, O>>,
}
impl<R: ServiceRequest, O> Job<R, O> {
    pub(crate) fn cancellation_handle(&self) -> JobCancellationHandle
    where
        O: Send + 'static,
    {
        JobCancellationHandle {
            cancelled: self.cancelled.clone(),
            wake: std::task::Waker::from(Arc::new(CancellationWake(Arc::downgrade(&self.shared)))),
        }
    }
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.shared.wake_cancellation();
    }
    pub(crate) async fn wait(mut self) -> Result<O, ServiceError> {
        match self.receiver.take() {
            Some(receiver) => receiver.await.unwrap_or_else(|_| {
                Err(service_error(
                    ServiceErrorKind::Stopped,
                    "image worker response channel closed",
                ))
            }),
            None => Err(service_error(
                ServiceErrorKind::Stopped,
                "image response already consumed",
            )),
        }
    }
    /// Only after the finite worker has returned: no runtime blocking or
    /// second execution loop is needed to collect its published terminal.
    pub(crate) fn completed(mut self) -> Result<O, ServiceError> {
        self.receiver
            .take()
            .and_then(|mut receiver| receiver.try_recv().ok())
            .unwrap_or_else(|| {
                Err(service_error(
                    ServiceErrorKind::Stopped,
                    "iterative worker returned without a terminal response",
                ))
            })
    }
}
impl<R: ServiceRequest, O> Drop for Job<R, O> {
    fn drop(&mut self) {
        self.cancel();
    }
}

pub(crate) struct ServiceCore<R: ServiceRequest, O: Send + 'static> {
    pub(crate) shared: Arc<Shared<R, O>>,
    pub(crate) join: Mutex<Option<JoinHandle<Retirement<R, O>>>>,
}
impl<R: ServiceRequest, O: Send + 'static> ServiceCore<R, O> {
    #[cfg(test)]
    pub(crate) fn start(
        worker: Box<dyn BatchWorker<Request = R, Output = O>>,
        config: impl Into<ServiceConfig>,
    ) -> Result<Self, ServiceError> {
        Self::start_with_host(worker, config, None)
    }
    #[cfg(any(test, feature = "cuda-fast-h3"))]
    pub(crate) fn start_with_host(
        worker: Box<dyn BatchWorker<Request = R, Output = O>>,
        config: impl Into<ServiceConfig>,
        host: Option<HostMemoryAccount>,
    ) -> Result<Self, ServiceError> {
        let service = Self::unstarted(worker.as_ref(), config, host);
        service.start_worker(worker)?;
        Ok(service)
    }
    pub(crate) fn unstarted(
        worker: &dyn BatchWorker<Request = R, Output = O>,
        config: impl Into<ServiceConfig>,
        host: Option<HostMemoryAccount>,
    ) -> Self {
        let config = config.into();
        let adapter = worker.adapter_domain_key().map(Arc::new);
        let trace = worker.lifecycle_observer();
        let shared = Arc::new(Shared {
            trace,
            state: Mutex::new(State {
                queue: VecDeque::new(),
                queue_backing: None,
                accepting: true,
                closing: false,
                active: 0,
                preparing: 0,
                preparing_bytes: 0,
                rejected: 0,
                executed_batches: 0,
                executed_requests: 0,
                last_batch_size: 0,
                capacity_reductions: 0,
            }),
            changed: Condvar::new(),
            resources: Arc::new(ResourceSignal::default()),
            config,
            host,
            adapter,
        });
        Self {
            shared,
            join: Mutex::new(None),
        }
    }
    pub(crate) fn start_worker(
        &self,
        worker: Box<dyn BatchWorker<Request = R, Output = O>>,
    ) -> Result<(), ServiceError> {
        let thread_shared = self.shared.clone();
        let join = thread::Builder::new()
            .name("ferrum-iterative-worker".into())
            .spawn(move || worker_loop(worker, thread_shared))
            .map_err(|e| {
                service_error(
                    ServiceErrorKind::Stopped,
                    format!("cannot start image worker: {e}"),
                )
            })?;
        *self.join.lock().unwrap_or_else(|e| e.into_inner()) = Some(join);
        Ok(())
    }
    /// Seal ingress and drain already admitted work using the same worker as
    /// serving. Sealing deliberately does not request cancellation (`closing`).
    pub(crate) fn run_to_completion(
        &self,
        worker: Box<dyn BatchWorker<Request = R, Output = O>>,
    ) -> Retirement<R, O> {
        self.shared.lock().accepting = false;
        self.shared.changed.notify_all();
        self.shared.resources.notify();
        worker_loop(worker, self.shared.clone())
    }

    pub(crate) fn snapshot(&self) -> ServiceSnapshot {
        let state = self.shared.lock();
        ServiceSnapshot {
            accepting: state.accepting,
            active_requests: state.active,
            pending_requests: state.queue.len() + state.preparing,
            preparing_requests: state.preparing,
            pending_bytes: state
                .queue
                .iter()
                .map(|job| job.pending_bytes())
                .sum::<usize>()
                + state.preparing_bytes,
            maximum_pending_bytes: self.shared.config.maximum_pending_bytes,
            rejected_requests: state.rejected,
            executed_batches: state.executed_batches,
            executed_requests: state.executed_requests,
            last_batch_size: state.last_batch_size,
            capacity_reductions: state.capacity_reductions,
            batching: self.shared.config.batching,
            maximum_pending: self.shared.config.maximum_pending,
            max_num_seqs: self.shared.config.max_num_seqs,
        }
    }
    #[cfg(test)]
    pub(crate) fn try_submit(&self, request: impl Into<R>) -> Result<Job<R, O>, ServiceError> {
        self.try_submit_request(request.into())
    }
    #[cfg(test)]
    pub(crate) fn try_submit_request(&self, request: R) -> Result<Job<R, O>, ServiceError> {
        self.try_submit_request_with_output(request, None)
    }
    #[cfg(test)]
    pub(crate) fn try_submit_request_with_output(
        &self,
        request: R,
        output: Option<QueuedOutput>,
    ) -> Result<Job<R, O>, ServiceError> {
        self.try_submit_request_with_output_and_correlation(
            request,
            output,
            RequestCorrelation::new(None),
        )
    }
    pub(crate) fn submission_error(
        &self,
        correlation: RequestCorrelation,
        mut error: ServiceError,
    ) -> ServiceError {
        trace::submission_rejected(self.shared.trace.as_ref(), correlation, error.kind());
        error.correlation = Some(correlation);
        error
    }
    pub(crate) fn try_submit_request_with_output_and_correlation(
        &self,
        request: R,
        output: Option<QueuedOutput>,
        correlation: RequestCorrelation,
    ) -> Result<Job<R, O>, ServiceError> {
        (|| {
            let mut state = self.shared.lock();
            prune_cancelled(&mut state.queue);
            if !state.accepting {
                return Err(service_error(
                    ServiceErrorKind::Stopped,
                    "image service is stopped",
                ));
            }
            let queued_bytes = state
                .queue
                .iter()
                .map(|job| job.pending_bytes())
                .sum::<usize>();
            if state.queue.len() + state.preparing >= self.shared.config.maximum_pending
                || queued_bytes
                    .saturating_add(state.preparing_bytes)
                    .saturating_add(request.pending_bytes())
                    .saturating_add(std::mem::size_of::<Queued<R, O>>())
                    > self.shared.config.maximum_pending_bytes
            {
                state.rejected = state.rejected.saturating_add(1);
                return Err(service_error(
                    ServiceErrorKind::QueueFull,
                    "image pending queue count or Host byte budget is full",
                ));
            }
            let prompt_reservation = self
                .shared
                .host
                .as_ref()
                .map(|host| {
                    host.reserve(
                        HostMemoryClass::InputPreprocess,
                        request
                            .admission_bytes()
                            .saturating_add(std::mem::size_of::<Queued<R, O>>()),
                    )
                })
                .transpose()
                .map_err(|error| {
                    state.rejected = state.rejected.saturating_add(1);
                    service_error(
                        ServiceErrorKind::QueueFull,
                        format!("image queue Host admission failed: {error}"),
                    )
                })?;
            if let Err(error) = state.reserve_queue_slot(
                self.shared.host.as_ref(),
                self.shared.config.maximum_pending,
            ) {
                state.rejected = state.rejected.saturating_add(1);
                return Err(error);
            }
            let (response, receiver) = oneshot::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            let trace = trace::RequestTrace::new(correlation, self.shared.trace.clone());
            let submitted = Instant::now();
            trace.queued(submitted);
            state.queue.push_back(Queued {
                trace,
                request,
                adapter: self.shared.adapter.clone(),
                submitted,
                cancelled: cancelled.clone(),
                response,
                output,
                _prompt_reservation: prompt_reservation,
                preprocessing_bytes: 0,
                preparation_duration: Duration::ZERO,
                preparation_group: None,
            });
            self.shared.changed.notify_one();
            Ok(Job {
                correlation,
                receiver: Some(receiver),
                cancelled,
                shared: self.shared.clone(),
            })
        })()
        .map_err(|error| self.submission_error(correlation, error))
    }
    pub(crate) fn take_join(&self) -> Option<JoinHandle<Retirement<R, O>>> {
        self.join.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
    pub(crate) async fn shutdown(&self) -> Result<Retirement<R, O>, ServiceError> {
        self.shared.stop();
        let join = self.take_join().ok_or_else(|| {
            service_error(
                ServiceErrorKind::Shutdown,
                "image worker shutdown already claimed",
            )
        })?;
        tokio::task::spawn_blocking(move || join_worker(join))
            .await
            .map_err(|e| {
                service_error(
                    ServiceErrorKind::Shutdown,
                    format!("image join task failed: {e}"),
                )
            })
    }
}
impl<R: ServiceRequest, O: Send + 'static> Drop for ServiceCore<R, O> {
    fn drop(&mut self) {
        self.shared.stop();
        if let Some(join) = self.take_join() {
            drop(join_worker(join));
        }
    }
}

pub(crate) struct Retirement<R: ServiceRequest, O: Send + 'static> {
    pub(crate) worker: Option<Box<dyn BatchWorker<Request = R, Output = O>>>,
    pub(crate) failure: Option<WorkerFailure>,
    pub(crate) evidence: Option<Box<dyn Any + Send>>,
}
impl<R: ServiceRequest, O: Send + 'static> Drop for Retirement<R, O> {
    fn drop(&mut self) {
        if self.failure.is_some() {
            // No generic backend can prove an opaque failed owner is quiescent.
            // Explicit recovery consumes these fields; unattended Drop fails closed.
            std::mem::forget(self.worker.take());
            std::mem::forget(self.failure.take());
        }
    }
}
fn join_worker<R: ServiceRequest, O: Send + 'static>(
    join: JoinHandle<Retirement<R, O>>,
) -> Retirement<R, O> {
    join.join().unwrap_or_else(|panic| Retirement {
        worker: None,
        failure: Some(WorkerFailure::Panic(panic)),
        evidence: None,
    })
}

pub(crate) fn prune_cancelled<R: ServiceRequest, O>(queue: &mut VecDeque<Queued<R, O>>) {
    let mut index = 0;
    while index < queue.len() {
        if queue[index].cancelled.load(Ordering::Acquire) {
            if let Some(job) = queue.remove(index) {
                let _ = job.response.send(Err(job.trace.error(
                    ServiceErrorKind::Cancelled,
                    "image request cancelled before execution",
                )));
            }
        } else {
            index += 1;
        }
    }
}
pub(crate) fn reject_pending<R: ServiceRequest, O>(
    state: &mut State<R, O>,
    kind: ServiceErrorKind,
) {
    for job in state.queue.drain(..) {
        let _ = job.response.send(Err(job
            .trace
            .error(kind, "image service stopped before execution")));
    }
    // Terminal shutdown releases both the retained allocation and its charge.
    state.queue = VecDeque::new();
    state.queue_backing.take();
}

pub(crate) mod output_gate;
pub use output_gate::{OutputGate, OutputGateError, OutputLease};
pub(crate) use output_gate::{QueuedOutput, ResourceSignal};
pub(crate) mod scheduler;
pub(crate) mod trace;
pub(crate) use scheduler::retire_worker;
use scheduler::worker_loop;
pub(crate) mod steps;
pub use steps::{
    DenoisingBatch, DenoisingExitReason, DenoisingRequest, InitializationCohortCompletion,
    InitializationCohortMember, InitializationCohortOutcome, InitializationCohortProgress,
    InitializationOutcome, InitializationProgress, RequestInitialization,
};
pub(crate) use steps::{
    DenoisingGenerator, InitializationCohortGenerator, InitializationCohortRequest,
};

#[cfg(test)]
mod tests;

pub(crate) mod capacity;
pub use capacity::{DeviceCapacityExceeded, DeviceCapacityReport};
pub(crate) mod capacity_hint;
pub use capacity_hint::CapacitySearchContext;
pub(crate) mod batch_efficiency;
pub use batch_efficiency::{BatchExecutionFeedback, BatchStageFeedback, ExecutionStageKey};

pub mod observations;
pub use observations::{LifecycleObserver, LifecycleObserverHandle};
