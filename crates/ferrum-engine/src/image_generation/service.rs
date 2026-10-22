//! Bounded request batching around one injected image generator.
//!
//! The worker calls `generate_batch` once per compatible FIFO batch. The current
//! Qwen adapter advertises batch one and cold component residency: this is a G2
//! integration path, not a throughput-qualified serving implementation. A future
//! resident/batched generator uses the same queue without pretending a loop of
//! single-image executions is a batch. Pending prompts are bounded separately
//! from the generator's shared Host tensor account.

use super::{ImageGenerationOutput, ImageGenerationRequest, QwenImage21Pipeline};
use ferrum_runtime::{DeviceRuntime, HostMemoryAccount};
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

type GeneratorError = Box<dyn Error + Send + 'static>;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImageServiceConfig {
    pub maximum_pending: usize,
    pub maximum_prompt_bytes: usize,
    pub max_num_seqs: usize,
    pub request_batch_max_wait_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageGeneratorCapabilities {
    pub maximum_batch_size: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub dimension_multiple: u32,
}

/// Implementations own their runtime and use one Host account for all jobs.
/// Successful outputs correspond exactly to input order. On failure the boxed
/// error must retain its original cleanup/recovery owners; formatting it is not
/// a replacement for returning it. Compatibility must be deterministic and may
/// only restrict the common width/height grouping enforced by this service.
pub trait ImageGenerator: Send + 'static {
    fn host_account(&self) -> &HostMemoryAccount;
    /// Metadata-only validation: this method must not initialize work or create
    /// recovery obligations. Execution failures belong to `generate_batch`.
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError>;
    fn compatible(&self, _a: &ImageGenerationRequest, _b: &ImageGenerationRequest) -> bool {
        true
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, GeneratorError>;
}

impl<R: DeviceRuntime> ImageGenerator for QwenImage21Pipeline<R> {
    fn host_account(&self) -> &HostMemoryAccount {
        self.host_account()
    }

    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        let scale = self
            .assets
            .spatial_scale()
            .map_err(|e| -> GeneratorError { e.into() })?;
        let multiple = scale.checked_mul(2).ok_or_else(|| {
            Box::new(service_error(
                ImageServiceErrorKind::InvalidRequest,
                "image dimension multiple overflows",
            )) as GeneratorError
        })?;
        let maximum = scale.checked_mul(2048).ok_or_else(|| {
            Box::new(service_error(
                ImageServiceErrorKind::InvalidRequest,
                "image dimension maximum overflows",
            )) as GeneratorError
        })?;
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 1,
            maximum_width: self.config.maximum_width.min(maximum),
            maximum_height: self.config.maximum_height.min(maximum),
            dimension_multiple: multiple,
        })
    }

    fn generate_batch(
        &mut self,
        mut requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, GeneratorError> {
        if requests.len() != 1 {
            return Err(Box::new(service_error(ImageServiceErrorKind::Generation,
                "current Qwen G2 adapter requires exactly one request; real batched execution is not implemented")));
        }
        self.generate(requests.remove(0))
            .map(|output| vec![output])
            .map_err(|error| Box::new(error) as GeneratorError)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageServiceErrorKind {
    InvalidRequest,
    QueueFull,
    Stopped,
    Cancelled,
    Generation,
    WorkerPanic,
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct ImageServiceError {
    kind: ImageServiceErrorKind,
    message: String,
}
impl ImageServiceError {
    pub fn kind(&self) -> ImageServiceErrorKind {
        self.kind
    }
}
impl fmt::Display for ImageServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl Error for ImageServiceError {}
fn service_error(kind: ImageServiceErrorKind, message: impl Into<String>) -> ImageServiceError {
    ImageServiceError {
        kind,
        message: message.into(),
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImageServiceSnapshot {
    pub accepting: bool,
    pub active_requests: usize,
    pub pending_requests: usize,
    pub maximum_pending: usize,
    pub max_num_seqs: usize,
}

/// Only this owner constructs the worker; callers can share it through `Arc`.
/// Call `shutdown` explicitly and retain its result for error recovery. Dropping
/// a still-live service joins synchronously; it never detaches active GPU work.
pub struct ImageService {
    core: ServiceCore<ImageGenerationOutput>,
    host: HostMemoryAccount,
    capabilities: ImageGeneratorCapabilities,
}

impl ImageService {
    pub fn new<P: ImageGenerator>(
        generator: P,
        config: ImageServiceConfig,
    ) -> Result<Self, ImageServiceError> {
        let capabilities = generator.capabilities().map_err(|error| {
            service_error(ImageServiceErrorKind::InvalidRequest, error.to_string())
        })?;
        validate_config(config, capabilities)?;
        let host = generator.host_account().clone();
        Ok(Self {
            core: ServiceCore::start(Box::new(GeneratorAdapter(generator)), config)?,
            host,
            capabilities,
        })
    }

    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn capabilities(&self) -> ImageGeneratorCapabilities {
        self.capabilities
    }
    pub fn snapshot(&self) -> ImageServiceSnapshot {
        self.core.snapshot()
    }
    pub fn is_accepting(&self) -> bool {
        self.snapshot().accepting
    }

    pub fn validate_request(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<(), ImageServiceError> {
        let caps = self.capabilities;
        if request.n != 1
            || request.prompt.len() > self.core.shared.config.maximum_prompt_bytes
            || request.width == 0
            || request.height == 0
            || request.width > caps.maximum_width
            || request.height > caps.maximum_height
            || request.width % caps.dimension_multiple != 0
            || request.height % caps.dimension_multiple != 0
        {
            return Err(service_error(ImageServiceErrorKind::InvalidRequest,
                "request exceeds image capability: n=1, prompt bytes, positive bounded dimensions and exact dimension multiple are required"));
        }
        Ok(())
    }

    pub fn try_submit(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageJob, ImageServiceError> {
        self.validate_request(&request)?;
        self.core
            .try_submit(request)
            .map(|inner| ImageJob { inner })
    }

    /// Stops admission, discards queued requests, lets active work reach its
    /// ordinary completion/retirement path, then joins the sole worker.
    /// Cancellation of this future does not cancel that join or active work.
    pub async fn shutdown(&self) -> Result<ImageServiceShutdown, ImageServiceError> {
        self.core
            .shutdown()
            .await
            .map(|inner| ImageServiceShutdown { inner })
    }
}

pub struct ImageJob {
    inner: Job<ImageGenerationOutput>,
}
impl ImageJob {
    /// Cancellation suppresses delivery, not active backend execution.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    pub async fn wait(self) -> Result<ImageGenerationOutput, ImageServiceError> {
        self.inner.wait().await
    }
}

/// Original error or unwind payload, not a string replacement. `Generation`
/// supports downcasting to `ImageGenerationError<R>` for typed retirement.
pub enum ImageWorkerFailure {
    Generation(GeneratorError),
    Panic(Box<dyn Any + Send + 'static>),
}
impl fmt::Debug for ImageWorkerFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generation(error) => f.debug_tuple("Generation").field(error).finish(),
            Self::Panic(_) => f.write_str("Panic(original payload retained)"),
        }
    }
}

/// Keep this opaque generator owner alive while recovering the original error.
/// It contains the original runtime, provider registry, assets and Host parent.
pub struct ImageGeneratorOwner {
    _worker: Box<dyn BatchWorker<Output = ImageGenerationOutput>>,
}

#[must_use = "retain the joined failure owner, or explicitly take it for typed recovery"]
pub struct ImageServiceShutdown {
    inner: Retirement<ImageGenerationOutput>,
}
impl ImageServiceShutdown {
    pub fn failure(&self) -> Option<&ImageWorkerFailure> {
        self.inner.failure.as_ref()
    }
    pub fn has_failure(&self) -> bool {
        self.inner.failure.is_some()
    }
    /// Explicit ownership transfer for recovery. The caller must retain the
    /// generator until any original typed error has been safely retired. If a
    /// shutdown receipt with a failure is dropped without this transfer, its
    /// owners are conservatively quarantined (leaked), never silently released.
    pub fn into_parts(mut self) -> (Option<ImageGeneratorOwner>, Option<ImageWorkerFailure>) {
        (
            self.inner
                .worker
                .take()
                .map(|worker| ImageGeneratorOwner { _worker: worker }),
            self.inner.failure.take(),
        )
    }
}

fn validate_config(
    config: ImageServiceConfig,
    caps: ImageGeneratorCapabilities,
) -> Result<(), ImageServiceError> {
    let invalid = || {
        service_error(
            ImageServiceErrorKind::InvalidRequest,
            "invalid bounded image service configuration or unsupported batch size",
        )
    };
    if config.maximum_pending == 0
        || config.maximum_prompt_bytes == 0
        || config.max_num_seqs == 0
        || config.max_num_seqs > caps.maximum_batch_size
        || caps.dimension_multiple == 0
        || caps.maximum_width < caps.dimension_multiple
        || caps.maximum_height < caps.dimension_multiple
    {
        return Err(invalid());
    }
    config
        .maximum_pending
        .checked_add(config.max_num_seqs)
        .and_then(|n| n.checked_mul(config.maximum_prompt_bytes))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(invalid)?;
    config
        .maximum_pending
        .checked_mul(std::mem::size_of::<Queued<ImageGenerationOutput>>())
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(invalid)?;
    Instant::now()
        .checked_add(Duration::from_millis(config.request_batch_max_wait_ms))
        .ok_or_else(invalid)?;
    Ok(())
}

// Generic only inside this module so scheduler tests can use small scalar
// outputs; production fixes the output to the managed image artifact above.
trait BatchWorker: Send + 'static {
    type Output: Send + 'static;
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool;
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<Self::Output>, GeneratorError>;
}
struct GeneratorAdapter<P>(P);
impl<P: ImageGenerator> BatchWorker for GeneratorAdapter<P> {
    type Output = ImageGenerationOutput;
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        self.0.compatible(a, b)
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<Self::Output>, GeneratorError> {
        self.0.generate_batch(requests)
    }
}

struct Queued<O> {
    request: ImageGenerationRequest,
    submitted: Instant,
    cancelled: Arc<AtomicBool>,
    response: oneshot::Sender<Result<O, ImageServiceError>>,
}
struct State<O> {
    queue: VecDeque<Queued<O>>,
    accepting: bool,
    closing: bool,
    active: usize,
}
struct Shared<O> {
    state: Mutex<State<O>>,
    changed: Condvar,
    config: ImageServiceConfig,
}
impl<O> Shared<O> {
    fn lock(&self) -> MutexGuard<'_, State<O>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn stop(&self) {
        let mut state = self.lock();
        state.accepting = false;
        state.closing = true;
        self.changed.notify_all();
    }
}
struct Job<O> {
    receiver: Option<oneshot::Receiver<Result<O, ImageServiceError>>>,
    cancelled: Arc<AtomicBool>,
    shared: Arc<Shared<O>>,
}
impl<O> Job<O> {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.shared.changed.notify_all();
    }
    async fn wait(mut self) -> Result<O, ImageServiceError> {
        match self.receiver.take() {
            Some(receiver) => receiver.await.unwrap_or_else(|_| {
                Err(service_error(
                    ImageServiceErrorKind::Stopped,
                    "image worker response channel closed",
                ))
            }),
            None => Err(service_error(
                ImageServiceErrorKind::Stopped,
                "image response already consumed",
            )),
        }
    }
}
impl<O> Drop for Job<O> {
    fn drop(&mut self) {
        self.cancel();
    }
}

struct ServiceCore<O: Send + 'static> {
    shared: Arc<Shared<O>>,
    join: Mutex<Option<JoinHandle<Retirement<O>>>>,
}
impl<O: Send + 'static> ServiceCore<O> {
    fn start(
        worker: Box<dyn BatchWorker<Output = O>>,
        config: ImageServiceConfig,
    ) -> Result<Self, ImageServiceError> {
        // Reserve queue metadata once, before any request/backend call.
        let mut queue = VecDeque::new();
        queue
            .try_reserve_exact(config.maximum_pending)
            .map_err(|_| {
                service_error(
                    ImageServiceErrorKind::InvalidRequest,
                    "cannot reserve bounded image request queue",
                )
            })?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue,
                accepting: true,
                closing: false,
                active: 0,
            }),
            changed: Condvar::new(),
            config,
        });
        let thread_shared = shared.clone();
        let join = thread::Builder::new()
            .name("ferrum-image-worker".into())
            .spawn(move || worker_loop(worker, thread_shared))
            .map_err(|e| {
                service_error(
                    ImageServiceErrorKind::Stopped,
                    format!("cannot start image worker: {e}"),
                )
            })?;
        Ok(Self {
            shared,
            join: Mutex::new(Some(join)),
        })
    }

    fn snapshot(&self) -> ImageServiceSnapshot {
        let state = self.shared.lock();
        ImageServiceSnapshot {
            accepting: state.accepting,
            active_requests: state.active,
            pending_requests: state.queue.len(),
            maximum_pending: self.shared.config.maximum_pending,
            max_num_seqs: self.shared.config.max_num_seqs,
        }
    }
    fn try_submit(&self, request: ImageGenerationRequest) -> Result<Job<O>, ImageServiceError> {
        let mut state = self.shared.lock();
        prune_cancelled(&mut state.queue);
        if !state.accepting {
            return Err(service_error(
                ImageServiceErrorKind::Stopped,
                "image service is stopped",
            ));
        }
        if state.queue.len() >= self.shared.config.maximum_pending {
            return Err(service_error(
                ImageServiceErrorKind::QueueFull,
                "image pending queue is full",
            ));
        }
        let (response, receiver) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        state.queue.push_back(Queued {
            request,
            submitted: Instant::now(),
            cancelled: cancelled.clone(),
            response,
        });
        self.shared.changed.notify_one();
        Ok(Job {
            receiver: Some(receiver),
            cancelled,
            shared: self.shared.clone(),
        })
    }
    fn take_join(&self) -> Option<JoinHandle<Retirement<O>>> {
        self.join.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
    async fn shutdown(&self) -> Result<Retirement<O>, ImageServiceError> {
        self.shared.stop();
        let join = self.take_join().ok_or_else(|| {
            service_error(
                ImageServiceErrorKind::Shutdown,
                "image worker shutdown already claimed",
            )
        })?;
        tokio::task::spawn_blocking(move || join_worker(join))
            .await
            .map_err(|e| {
                service_error(
                    ImageServiceErrorKind::Shutdown,
                    format!("image join task failed: {e}"),
                )
            })
    }
}
impl<O: Send + 'static> Drop for ServiceCore<O> {
    fn drop(&mut self) {
        self.shared.stop();
        if let Some(join) = self.take_join() {
            drop(join_worker(join));
        }
    }
}

struct Retirement<O: Send + 'static> {
    worker: Option<Box<dyn BatchWorker<Output = O>>>,
    failure: Option<ImageWorkerFailure>,
}
impl<O: Send + 'static> Drop for Retirement<O> {
    fn drop(&mut self) {
        if self.failure.is_some() {
            // No generic backend can prove an opaque failed owner is quiescent.
            // Explicit recovery consumes these fields; unattended Drop fails closed.
            std::mem::forget(self.worker.take());
            std::mem::forget(self.failure.take());
        }
    }
}
fn join_worker<O: Send + 'static>(join: JoinHandle<Retirement<O>>) -> Retirement<O> {
    join.join().unwrap_or_else(|panic| Retirement {
        worker: None,
        failure: Some(ImageWorkerFailure::Panic(panic)),
    })
}

fn prune_cancelled<O>(queue: &mut VecDeque<Queued<O>>) {
    let mut index = 0;
    while index < queue.len() {
        if queue[index].cancelled.load(Ordering::Acquire) {
            if let Some(job) = queue.remove(index) {
                let _ = job.response.send(Err(service_error(
                    ImageServiceErrorKind::Cancelled,
                    "image request cancelled before execution",
                )));
            }
        } else {
            index += 1;
        }
    }
}
fn reject_pending<O>(state: &mut State<O>, kind: ImageServiceErrorKind) {
    for job in state.queue.drain(..) {
        let _ = job.response.send(Err(service_error(
            kind,
            "image service stopped before execution",
        )));
    }
}

fn take_batch<O: Send + 'static>(
    worker: &dyn BatchWorker<Output = O>,
    shared: &Shared<O>,
) -> Option<Vec<Queued<O>>> {
    let mut state = shared.lock();
    loop {
        prune_cancelled(&mut state.queue);
        if state.closing {
            reject_pending(&mut state, ImageServiceErrorKind::Shutdown);
            return None;
        }
        let Some(first) = state.queue.front() else {
            state = shared
                .changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
            continue;
        };
        let mut count = 1;
        let mut incompatible = false;
        for job in state
            .queue
            .iter()
            .skip(1)
            .take(shared.config.max_num_seqs.saturating_sub(1))
        {
            if job.request.width != first.request.width
                || job.request.height != first.request.height
                || !worker.compatible(&first.request, &job.request)
            {
                incompatible = true;
                break;
            }
            count += 1;
        }
        let deadline = first.submitted.checked_add(Duration::from_millis(
            shared.config.request_batch_max_wait_ms,
        ));
        let remaining =
            deadline.and_then(|deadline| deadline.checked_duration_since(Instant::now()));
        if count == shared.config.max_num_seqs
            || incompatible
            || remaining.is_none()
            || remaining == Some(Duration::ZERO)
        {
            state.active = count;
            return Some(state.queue.drain(..count).collect());
        }
        state = shared
            .changed
            .wait_timeout(state, remaining.unwrap_or(Duration::ZERO))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

fn worker_loop<O: Send + 'static>(
    mut worker: Box<dyn BatchWorker<Output = O>>,
    shared: Arc<Shared<O>>,
) -> Retirement<O> {
    let run = catch_unwind(AssertUnwindSafe(|| {
        while let Some(batch) = take_batch(worker.as_ref(), &shared) {
            let mut requests = Vec::with_capacity(batch.len());
            let mut responses = Vec::with_capacity(batch.len());
            for job in batch {
                requests.push(job.request);
                responses.push((job.cancelled, job.response));
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| worker.generate_batch(requests)));
            let outcome = match outcome {
                Ok(Ok(outputs)) if outputs.len() == responses.len() => Ok(outputs),
                Ok(Ok(_)) => Err(ImageWorkerFailure::Generation(Box::new(service_error(
                    ImageServiceErrorKind::Generation,
                    "generator returned a different number of outputs than batch requests",
                )))),
                Ok(Err(error)) => Err(ImageWorkerFailure::Generation(error)),
                Err(panic) => Err(ImageWorkerFailure::Panic(panic)),
            };
            match outcome {
                Ok(outputs) => {
                    for ((cancelled, response), output) in responses.into_iter().zip(outputs) {
                        let result = if cancelled.load(Ordering::Acquire) {
                            Err(service_error(
                                ImageServiceErrorKind::Cancelled,
                                "image response cancelled; active execution completed",
                            ))
                        } else {
                            Ok(output)
                        };
                        let _ = response.send(result);
                    }
                    shared.lock().active = 0;
                }
                Err(failure) => {
                    let (kind, message) = match &failure {
                        ImageWorkerFailure::Generation(error) => (ImageServiceErrorKind::Generation,
                            catch_unwind(AssertUnwindSafe(|| error.to_string()))
                                .unwrap_or_else(|_| "image generation failed; error formatter panicked, original owner retained".to_owned())),
                        ImageWorkerFailure::Panic(_) => (ImageServiceErrorKind::WorkerPanic, "image worker panicked; original owners retained".to_owned()),
                    };
                    let mut state = shared.lock();
                    state.accepting = false;
                    state.closing = true;
                    state.active = 0;
                    reject_pending(&mut state, ImageServiceErrorKind::Stopped);
                    for (_, response) in responses {
                        let _ = response.send(Err(service_error(kind, message.clone())));
                    }
                    return Some(failure);
                }
            }
        }
        None
    }));
    let failure = match run {
        Ok(failure) => failure,
        Err(panic) => Some(ImageWorkerFailure::Panic(panic)),
    };
    {
        let mut state = shared.lock();
        state.accepting = false;
        state.closing = true;
        state.active = 0;
        reject_pending(&mut state, ImageServiceErrorKind::Stopped);
    }
    Retirement {
        worker: Some(worker),
        failure,
    }
}

#[cfg(test)]
mod tests;
