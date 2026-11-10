//! Optional step protocol. The production worker observes boundaries only
//! after real completed forwards; fixed-batch generators need not implement it.
use super::*;
use crate::image_generation::denoising::{
    ContinuousBatch, ContinuousRequest, PreparedContinuousRequest,
};
use crate::image_generation::{
    ImageDecoderBatchPolicy, ImageGenerationError, ImageGenerationFailure,
};
mod cohorts;
pub use cohorts::{
    ImageInitializationCohortCompletion, ImageInitializationCohortGenerator,
    ImageInitializationCohortMember, ImageInitializationCohortOutcome,
    ImageInitializationCohortProgress, ImageInitializationCohortRequest,
};
mod initialization;
pub use initialization::{
    ImageInitializationOutcome, ImageInitializationProgress, ImageRequestInitialization,
};

/// Why a fenced denoising continuation retired, including a Ready request
/// withdrawn before its first cohort. Completed means solver completion;
/// downstream image decoding or delivery can still fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDenoisingExitReason {
    Completed,
    Cancelled,
    Failed,
}

/// A quiescent request continuation. Submitted work belongs to a step batch or
/// the original typed fatal error, never to a queued prompt/recovery String.
pub struct ImageDenoisingRequest(Box<dyn Any + Send>);
impl ImageDenoisingRequest {
    pub fn new<T: Any + Send>(value: T) -> Self {
        Self(Box::new(value))
    }
    pub fn get<T: Any + Send>(&self) -> Result<&T, ImageBatchError> {
        self.0.downcast_ref().ok_or_else(|| {
            ImageBatchError::Rejected("denoising request belongs to another generator".into())
        })
    }
    pub fn get_mut<T: Any + Send>(&mut self) -> Result<&mut T, ImageBatchError> {
        self.0.downcast_mut().ok_or_else(|| {
            ImageBatchError::Rejected("denoising request belongs to another generator".into())
        })
    }
    pub fn into_inner<T: Any + Send>(self) -> Result<T, ImageBatchError> {
        self.0.downcast::<T>().map(|v| *v).map_err(|_| {
            ImageBatchError::Rejected("denoising request belongs to another generator".into())
        })
    }
}
/// Exact, admitted membership. Pause this owner at its completion fence before
/// starting another model stage, removing members, or discarding/replacing it.
#[must_use = "pause/retire an admitted step owner before changing its membership"]
pub struct ImageDenoisingBatch(Box<dyn Any + Send>);
impl ImageDenoisingBatch {
    pub fn new<T: Any + Send>(value: T) -> Self {
        Self(Box::new(value))
    }
    pub fn get<T: Any + Send>(&self) -> Result<&T, ImageBatchError> {
        self.0.downcast_ref().ok_or_else(|| {
            ImageBatchError::Rejected("denoising batch belongs to another generator".into())
        })
    }
    pub fn get_mut<T: Any + Send>(&mut self) -> Result<&mut T, ImageBatchError> {
        self.0.downcast_mut().ok_or_else(|| {
            ImageBatchError::Rejected("denoising batch belongs to another generator".into())
        })
    }
}

/// Declared Full-T2I capability. Returning this typed interface opts into real
/// request-local continuation semantics; no model name or client name selects
/// a scheduling path. Editing/prefix execution remains on the whole-batch API.
pub trait ImageDenoisingGenerator: Send {
    type Output: Send + 'static;
    /// Metadata-only capability: initializing another request can replace the
    /// resident denoiser with an auxiliary model. Auto scheduling may finish a
    /// known final solver step before starting that model switch.
    fn initialization_uses_auxiliary_model(&self) -> bool {
        false
    }
    /// Remaining committed solver transitions, without submitting work. None
    /// preserves ordinary admission for generators without this observation.
    fn remaining_denoising_steps(&self, _request: &ImageDenoisingRequest) -> Option<usize> {
        None
    }
    /// Explicit finite-cohort initialization. None preserves the original
    /// singleton consumer and admission behavior without model-name dispatch.
    fn initialization_cohort_generator(
        &mut self,
    ) -> Option<&mut dyn ImageInitializationCohortGenerator> {
        None
    }
    /// Attach observation metadata to the exact preparation. This must not
    /// submit work or derive identity from a seed, client name or prompt.
    fn bind_request_correlation(
        &mut self,
        _prepared: &mut ImageBatchPreparation,
        _correlation: ImageRequestCorrelation,
    ) {
    }
    /// Observation only, after the worker has fenced and chosen an actual
    /// exit from denoising. A Ready request may exit with zero committed steps
    /// before joining its first cohort. Opaque pending initialization is not a
    /// denoising request. This must not submit work or change the outcome.
    fn record_request_exit(
        &self,
        _request: &ImageDenoisingRequest,
        _reason: ImageDenoisingExitReason,
    ) {
    }
    /// Metadata-only declaration. Whole-batch editing support does not imply
    /// that a generator can preserve an editing request across real steps.
    fn supports_image_request(&self, request: ImageRequestRef<'_>) -> bool {
        matches!(request, ImageRequestRef::Generate(_))
    }
    /// Metadata only: no device submissions or pending recovery owners.
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError>;
    fn prepare_image_request(
        &mut self,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        match request {
            ImageRequestRef::Generate(request) => self.prepare_request(request),
            ImageRequestRef::Edit(_) => Err(ImageBatchError::Rejected(
                "step generation has not declared image editing".into(),
            )),
        }
    }
    /// Nonblocking Host admission. `false` preserves the exact preparation.
    fn try_admit_request(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError>;
    /// Optional auxiliary-model admission at an already fenced step boundary.
    /// The default keeps the original admission behavior. A generator that
    /// retires an installed cohort must mark that batch suspended and restore
    /// its real native state before another denoising forward is permitted.
    fn try_admit_request_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
        _batch: Option<&mut ImageDenoisingBatch>,
        _requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError> {
        self.try_admit_request(prepared, resources)
    }
    fn batch_is_suspended(&self, _batch: &ImageDenoisingBatch) -> bool {
        false
    }
    /// True only while this original initialization still owns an auxiliary
    /// native phase that prevents restoring a suspended denoiser. Metadata or
    /// Host-only downstream waits do not block restoration.
    fn initialization_blocks_denoising(
        &self,
        _initialization: &ImageRequestInitialization,
    ) -> bool {
        false
    }
    /// Restore at an actual auxiliary retirement boundary, without advancing
    /// any solver. Errors must preserve the suspended batch and request owners.
    fn restore_suspended_batch(
        &mut self,
        _batch: &mut ImageDenoisingBatch,
        _requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        Ok(())
    }
    /// TE/noise/persistent initialization. An existing batch must be paused.
    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError>;
    fn start_image_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        match request {
            ImageRequestRef::Generate(request) => self.start_request(prepared, request),
            ImageRequestRef::Edit(_) => Err(ImageBatchError::Rejected(
                "step generation has not declared image editing".into(),
            )),
        }
    }
    /// Starts one initialization owner at a completed batch boundary. Existing
    /// generators keep the original synchronous, already-quiescent start path.
    fn begin_image_initialization(
        &mut self,
        prepared: ImageBatchPreparation,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageInitializationOutcome, ImageBatchError> {
        self.start_image_request(prepared, request)
            .map(ImageInitializationOutcome::Ready)
    }
    /// Advances at most one initialization wave, then returns at its original
    /// execution boundary. Pending owners never enter the denoising cohort.
    /// Pending(false) must register `wake` with its actual external wait before
    /// returning. Cancellation/closing must retain the original owner until a
    /// quiescent Cancelled, Rejected or Ready is returned. Once Pending has been issued,
    /// any error must be Fatal; it never authorizes dropping that opaque owner.
    fn advance_image_initialization(
        &mut self,
        _initialization: &mut ImageRequestInitialization,
        _request: ImageRequestRef<'_>,
        _cancelled: bool,
        _wake: &std::task::Waker,
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        Err(ImageBatchError::Fatal(Box::new(service_error(
            ImageServiceErrorKind::Generation,
            "generator returned pending initialization without its continuation",
        ))))
    }
    /// Advance the original initialization with the current paused membership.
    /// Implementations may prepare immutable membership metadata during this
    /// work, but must retain the original admission and publication boundaries.
    /// The default preserves generators without such preparation.
    fn advance_image_initialization_at_boundary(
        &mut self,
        initialization: &mut ImageRequestInitialization,
        request: ImageRequestRef<'_>,
        cancelled: bool,
        wake: &std::task::Waker,
        _batch: Option<&mut ImageDenoisingBatch>,
        _requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        self.advance_image_initialization(initialization, request, cancelled, wake)
    }
    /// Compile and physically admit this exact membership before publishing it.
    /// `None` may delay an optional join; it must not consume old step credit.
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError>;
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError>;
    /// One actual shared DiT evaluation followed by atomic request-local solver
    /// successors. Completion flags correspond exactly to input member order.
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError>;
    /// The batch is paused. A generator may exchange its existing work grant
    /// for decoder work while preserving the surviving cohort's forward credit.
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError>;
    /// `None` retains the complete request and fenced work owner for an
    /// observable resource change. It must not submit an unacknowledged retry.
    /// The worker can advance survivors while completed output is blocked.
    fn try_finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Option<Self::Output>, ImageBatchError> {
        self.finish_request(request, batch).map(Some)
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    fn step_error(&mut self, failure: ImageGenerationFailure<R>) -> ImageBatchError {
        if let ImageGenerationFailure::Preflight(error) = failure {
            return preparation_error(error);
        }
        self.execution.stopped = !matches!(
            failure,
            ImageGenerationFailure::AllocationRejected { .. }
                | ImageGenerationFailure::CapacityRejected { .. }
                | ImageGenerationFailure::InitializationCapacityRejected { .. }
        );
        qwen_batch_error(
            self.is_stopped(),
            ImageGenerationError {
                failure,
                completed_stages: vec![],
                completed_steps: vec![],
            },
        )
    }
    pub(super) fn declared_denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        // Preserve the explicitly selected whole-decoder diagnostic mode.
        (!self.execution.preparation_only
            && self.execution.config.decoder_batch_policy == ImageDecoderBatchPolicy::SingleImage)
            .then_some(self)
    }
}
impl<R: DeviceRuntime> ImageDenoisingGenerator for QwenImage21Pipeline<R> {
    type Output = ImageGenerationOutput;
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.prepare_continuous_request(request)
            .map(ImageBatchPreparation::new)
            .map_err(preparation_error)
    }
    fn try_admit_request(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        let p = prepared
            .0
            .as_mut()
            .and_then(|p| p.downcast_mut::<PreparedContinuousRequest<R>>())
            .ok_or_else(|| ImageBatchError::Rejected("foreign continuation preparation".into()))?;
        self.try_admit_continuous_request_with_output_credit(p, resources.output_credit_bytes)
            .map_err(preparation_error)
    }
    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        let p = prepared.into_inner::<PreparedContinuousRequest<R>>()?;
        self.start_continuous_request(p, request)
            .map(ImageDenoisingRequest::new)
            .map_err(|e| self.step_error(e))
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<ContinuousRequest<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        let prepared = if let Some(previous) = previous {
            self.try_replace_continuous_batch(
                previous.get_mut::<ContinuousBatch<R>>()?,
                &mut requests,
            )
        } else {
            self.try_prepare_continuous_batch(&mut requests)
        };
        match prepared {
            // Both variants prove a clean, pre-submission rollback. The core
            // restored the old work authority, so an optional join can wait
            // while its existing cohort continues actual solver progress.
            Err(
                ImageGenerationFailure::AllocationRejected { .. }
                | ImageGenerationFailure::CapacityRejected { .. }
                | ImageGenerationFailure::InitializationCapacityRejected { .. },
            ) => Ok(None),
            Err(ImageGenerationFailure::Initialization {
                source:
                    ferrum_execution::component::ComponentInitializationError::Contract(
                        ferrum_runtime::VNextError::DeviceCapacityUnavailable(_),
                    ),
                ..
            }) => Ok(None),
            Err(ImageGenerationFailure::Initialization {
                plan,
                source:
                    ferrum_execution::component::ComponentInitializationError::Pool {
                        source: ferrum_runtime::VNextError::DeviceCapacityUnavailable(_),
                        transaction,
                        ..
                    },
            }) => {
                // Component initialization creates pool backings before any
                // weight upload or inference submission. Release the actual
                // committed journal; only acknowledged cleanup permits retry.
                match transaction.release() {
                    Ok(released) => {
                        drop(released);
                        Ok(None)
                    }
                    Err(source) => Err(self.step_error(
                        ImageGenerationFailure::InitializationRelease { plan, source },
                    )),
                }
            }
            other => other
                .map(|v| v.map(ImageDenoisingBatch::new))
                .map_err(|e| self.step_error(e)),
        }
    }
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        let batch = batch.get_mut::<ContinuousBatch<R>>()?;
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<ContinuousRequest<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        batch
            .pause(self, &mut requests)
            .map_err(|e| self.step_error(e))
    }
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        let batch = batch.get_mut::<ContinuousBatch<R>>()?;
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<ContinuousRequest<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        batch
            .advance(self, &mut requests)
            .map_err(|e| self.step_error(e))?;
        Ok(requests.iter().map(|r| r.is_complete()).collect())
    }
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.finish_continuous_request_borrowed(
            request.get_mut::<ContinuousRequest<R>>()?,
            batch.get::<ContinuousBatch<R>>()?,
        )
        .map_err(|e| self.step_error(e))
    }
}

struct Running<O> {
    job: Queued<O>,
    state: ImageDenoisingRequest,
    complete: bool,
}
struct Pending<O> {
    job: Queued<O>,
    prepared: Option<ImageBatchPreparation>,
    state: Option<ImageDenoisingRequest>,
    initialization: Option<ImageRequestInitialization>,
    initialization_trace: Option<u64>,
}
impl<O> Pending<O> {
    fn new(job: Queued<O>) -> Self {
        Self {
            job,
            prepared: None,
            state: None,
            initialization: None,
            initialization_trace: None,
        }
    }
}
struct FailedOwners {
    source: GeneratorError,
    _batch: Option<ImageDenoisingBatch>,
    _requests: Vec<ImageDenoisingRequest>,
    _initializations: Vec<ImageRequestInitialization>,
}
impl fmt::Debug for FailedOwners {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageDenoisingFailure")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for FailedOwners {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}
impl Error for FailedOwners {}

fn reject_job<O>(job: Queued<O>, kind: ImageServiceErrorKind, message: &str) {
    // Return output/prompt credit before waking a caller that can submit again.
    let Queued {
        trace,
        response,
        output,
        _prompt_reservation,
        ..
    } = job;
    drop((output, _prompt_reservation));
    let _ = response.send(Err(trace.error(kind, message)));
}
fn pending_removed<O>(shared: &Shared<O>, job: &Queued<O>) {
    let mut s = shared.lock();
    s.preparing -= 1;
    s.preparing_bytes -= job.pending_bytes();
}
fn reject_pending_step<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    pending: Pending<O>,
    shared: &Shared<O>,
    kind: ImageServiceErrorKind,
    message: &str,
) {
    // A Ready continuation is already quiescent, even if actual cohort
    // admission never succeeded. Observe its one terminal withdrawal before
    // returning its original persistent/native and queue owners. Metadata and
    // opaque initialization have no initialized denoising request to retire.
    if pending.initialization.is_none() {
        if let Some(request) = pending.state.as_ref() {
            let reason = if matches!(
                kind,
                ImageServiceErrorKind::Cancelled | ImageServiceErrorKind::Shutdown
            ) {
                ImageDenoisingExitReason::Cancelled
            } else {
                ImageDenoisingExitReason::Failed
            };
            worker
                .denoising_generator()
                .expect("declared step capability remains stable")
                .record_request_exit(request, reason);
        }
    }
    pending_removed(shared, &pending.job);
    drop((pending.state, pending.prepared));
    reject_job(pending.job, kind, message);
}
fn fatal<O: Send + 'static>(
    source: GeneratorError,
    batch: Option<ImageDenoisingBatch>,
    running: Vec<Running<O>>,
    pending: VecDeque<Pending<O>>,
    shared: &Shared<O>,
) -> ImageWorkerFailure {
    let mut owners = Vec::new();
    let mut initializations = Vec::new();
    for r in running {
        owners.push(r.state);
        reject_job(
            r.job,
            ImageServiceErrorKind::Generation,
            "step generation failed; original recovery owners retained",
        );
    }
    for p in pending {
        if let Some(initialization) = p.initialization {
            initializations.push(initialization);
        }
        if let Some(state) = p.state {
            owners.push(state);
        }
        drop(p.prepared);
        reject_job(
            p.job,
            ImageServiceErrorKind::Generation,
            "step generation failed; original recovery owners retained",
        );
    }
    let mut state = shared.lock();
    state.active = 0;
    state.preparing = 0;
    state.preparing_bytes = 0;
    state.accepting = false;
    state.closing = true;
    reject_pending(&mut state, ImageServiceErrorKind::Stopped);
    ImageWorkerFailure::Generation(Box::new(FailedOwners {
        source,
        _batch: batch,
        _requests: owners,
        _initializations: initializations,
    }))
}
fn source(error: ImageBatchError) -> GeneratorError {
    match error {
        ImageBatchError::Fatal(e) => e,
        other => Box::new(other),
    }
}
fn pause<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    batch: &mut Option<ImageDenoisingBatch>,
    running: &mut [Running<O>],
) -> Result<(), ImageBatchError> {
    if let Some(batch) = batch {
        worker
            .denoising_generator()
            .expect("declared step capability remains stable")
            .pause_batch(
                batch,
                &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
            )?;
    }
    Ok(())
}
fn output_credit<O>(shared: &Shared<O>, candidate: &Queued<O>) -> Result<usize, ImageBatchError> {
    std::iter::once(candidate)
        .filter_map(|j| j.output.as_ref())
        .filter(|o| o.lease.is_some())
        .try_fold(0usize, |sum, o| {
            sum.checked_add(
                shared
                    .host
                    .as_ref()
                    .map_or(0, |h| o.gate.host_output_bytes(h)),
            )
            .ok_or_else(|| {
                ImageBatchError::Overloaded("step output credit extent overflows".into())
            })
        })
}
fn try_output<O: Send + 'static>(
    worker: &dyn BatchWorker<Output = O>,
    job: &mut Queued<O>,
    shared: &Shared<O>,
) -> Result<bool, ImageBatchError> {
    if job.output.as_ref().is_some_and(|o| o.lease.is_some()) {
        return Ok(true);
    }
    if !super::scheduler::output_prefix_available(worker, std::slice::from_ref(job), 0, shared)? {
        return Ok(false);
    }
    if let Some(output) = &mut job.output {
        match output.gate.try_acquire() {
            Ok(Some(lease)) => output.lease = Some(lease),
            Ok(None) => return Ok(false),
            Err(ImageOutputGateError::Capacity(e)) => return Err(ImageBatchError::Overloaded(e)),
            Err(ImageOutputGateError::Failed(e)) => return Err(ImageBatchError::Rejected(e)),
        }
    }
    Ok(true)
}
fn deliver<O>(shared: &Shared<O>, job: Queued<O>, output: O) {
    let Queued {
        trace,
        cancelled,
        response,
        output: gate,
        _prompt_reservation,
        ..
    } = job;
    if cancelled.load(Ordering::Acquire) {
        drop((output, gate, _prompt_reservation));
        shared.lock().active -= 1;
        let _ = response.send(Err(trace.error(
            ImageServiceErrorKind::Cancelled,
            "image request cancelled at a completed step",
        )));
        return;
    }
    if let Some(QueuedOutput {
        lease: Some(lease),
        response,
        ..
    }) = gate
    {
        let _ = response.send(lease);
    }
    drop(_prompt_reservation);
    shared.lock().active -= 1;
    let delivered = response.send(Ok(output)).is_ok();
    trace.finished(if delivered {
        super::trace::ImageTraceRequestOutcome::Completed
    } else {
        super::trace::ImageTraceRequestOutcome::Cancelled
    });
}
fn pull<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    running: &[Running<O>],
    pending: &mut VecDeque<Pending<O>>,
    shared: &Shared<O>,
) -> bool {
    if !pending.is_empty() {
        return true;
    }
    let mut state = shared.lock();
    if state.closing {
        return false;
    }
    prune_cancelled(&mut state.queue);
    let Some(first) = state.queue.front() else {
        return false;
    };
    if !worker.denoising_generator().is_some_and(|generator| {
        generator.supports_image_request(first.request.image_request_ref())
    }) || running
        .first()
        .is_some_and(|r| !super::scheduler::compatible_queued(worker, &r.job, first))
    {
        return false;
    }
    let job = state.queue.pop_front().unwrap();
    job.trace.dequeued();
    state.preparing += 1;
    state.preparing_bytes += job.pending_bytes();
    pending.push_back(Pending::new(job));
    true
}

fn finish_tail_before_auxiliary_initialization<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    policy: ImageBatchPolicy,
    running: &[Running<O>],
) -> bool {
    if policy != ImageBatchPolicy::Auto || running.is_empty() {
        return false;
    }
    let generator = worker.denoising_generator().unwrap();
    generator.initialization_uses_auxiliary_model()
        && running
            .iter()
            .all(|request| generator.remaining_denoising_steps(&request.state) == Some(1))
}

/// One compatible generation stream. Returning to the original worker loop
/// leaves editing/incompatible FIFO work on its existing whole-batch path.
/// Only this execution worker can publish an admitted membership replacement.
pub(super) fn execute<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    shared: &Arc<Shared<O>>,
    initial: Vec<Queued<O>>,
) -> Result<(), ImageWorkerFailure> {
    let mut cohort = None;
    let result = (|| {
        let mut pending = initial
            .into_iter()
            .map(Pending::new)
            .collect::<VecDeque<_>>();
        let mut running: Vec<Running<O>> = Vec::new();
        let mut batch = None;
        let mut finishing = VecDeque::new();
        let mut survivor_forward_due = false;
        let mut auxiliary_survivor_forward_due = false;
        loop {
            let observed = shared.resources.epoch();
            let closing = shared.lock().closing;
            let suspended = batch.as_ref().is_some_and(|batch| {
                worker
                    .denoising_generator()
                    .unwrap()
                    .batch_is_suspended(batch)
            });
            if let Err(error) = if suspended {
                Ok(())
            } else {
                completion::retire_exits(worker, shared, &mut batch, &mut running, &mut finishing)
            } {
                return Err(fatal(
                    source(error),
                    batch,
                    completion::failure_owners(running, finishing),
                    pending,
                    shared,
                ));
            }
            // Give an already completed output one attempt before ordinary
            // initialization replaces its working set. Existing native owners
            // and cancellation must still reach their original retirement.
            // A real completion wait falls through to the unchanged progress
            // path below; it is not a reason to suppress initialization.
            if !suspended
                && !closing
                && !finishing.is_empty()
                && cohort.is_none()
                && pending.iter().any(|p| p.initialization.is_some())
                && !pending.iter().any(|p| {
                    p.initialization.as_ref().is_some_and(|initialization| {
                        p.job.cancelled.load(Ordering::Acquire)
                            || worker
                                .denoising_generator()
                                .unwrap()
                                .initialization_blocks_denoising(initialization)
                    })
                })
            {
                if let Err(error) = completion::drain_finished(
                    worker,
                    shared,
                    &mut batch,
                    &mut running,
                    &mut finishing,
                    true,
                ) {
                    return Err(fatal(
                        source(error),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ));
                }
            }
            let (cohort_tick, cohort_finished) = match cohorts::advance(
                worker,
                shared,
                &mut batch,
                &mut running,
                &mut pending,
                &mut cohort,
            ) {
                Ok(tick) => tick,
                Err(error) => {
                    return Err(fatal(
                        source(error),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ))
                }
            };
            auxiliary_survivor_forward_due |= cohort_finished && !running.is_empty();
            let initialization_tick = if cohort_tick.advanced || cohort_finished {
                cohort_tick
            } else {
                match initialization::advance(
                    worker,
                    shared,
                    &mut batch,
                    &mut running,
                    &mut pending,
                ) {
                    Ok(tick) => tick,
                    Err(error) => {
                        return Err(fatal(
                            source(error),
                            batch,
                            completion::failure_owners(running, finishing),
                            pending,
                            shared,
                        ))
                    }
                }
            };
            let initializing =
                cohort.is_some() || pending.iter().any(|p| p.initialization.is_some());
            match restore_at_boundary(worker, &mut batch, &mut running, &pending, cohort.as_ref()) {
                Ok(true) => (),
                Ok(false) => {
                    if !initialization_tick.made_progress {
                        shared.resources.wait(observed);
                    }
                    continue;
                }
                Err(error) => {
                    return Err(fatal(
                        source(error),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ))
                }
            }
            if suspended {
                if let Err(error) = completion::retire_exits(
                    worker,
                    shared,
                    &mut batch,
                    &mut running,
                    &mut finishing,
                ) {
                    return Err(fatal(
                        source(error),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ));
                }
            }
            if let Err(error) = completion::drain_finished(
                worker,
                shared,
                &mut batch,
                &mut running,
                &mut finishing,
                initializing,
            ) {
                return Err(fatal(
                    source(error),
                    batch,
                    completion::failure_owners(running, finishing),
                    pending,
                    shared,
                ));
            }
            // The initialization consumer above withdraws quiescent metadata and
            // drives native cancellation/closing until an actual terminal boundary.
            if running.is_empty() && finishing.is_empty() {
                drop(batch.take());
            }
            if running.is_empty() {
                // Idle initialization must still be able to become the first
                // runnable cohort without waiting for a nonexistent forward.
                survivor_forward_due = false;
                auxiliary_survivor_forward_due = false;
            }

            // Resource-blocked joins never wait while an authorized request can
            // advance. The subscription covers the idle wait without lost wakes.
            let mut gate_subscription = None;
            let mut initialization_made_progress = initialization_tick.made_progress;
            let mut cohort_collection_handled = false;
            // A final real forward releases resident requests and lets the next
            // queued prefix initialize together. Starting an auxiliary model
            // now would suspend that tail and split the prefix into singletons.
            // Existing initialization/Ready owners still advance and join; this
            // only defers starting new work for one actual solver transition.
            let finish_tail = finish_tail_before_auxiliary_initialization(
                worker,
                shared.config.batching,
                &running,
            );
            // Rejections and late cancellations consume an attempt too. New
            // arrivals cannot replenish this turn's budget before a real forward.
            if !closing
                && !finish_tail
                && !initialization_tick.advanced
                && !survivor_forward_due
                && !auxiliary_survivor_forward_due
                && finishing.is_empty()
                && running.len() < shared.config.max_num_seqs
            {
                match cohorts::collect(worker, shared, &running, &mut pending, &mut cohort) {
                    Ok(collected) => {
                        cohort_collection_handled = collected;
                        initialization_made_progress |= collected;
                        if collected && cohort.is_some() {
                            // Freeze, admit and begin at this completed fence.
                            // Deferring begin until after another forward can
                            // finish a short survivor before its peer starts.
                            // A newly collected group has no native owner, so
                            // this performs admission/begin, not a token wave.
                            match cohorts::advance(
                                worker,
                                shared,
                                &mut batch,
                                &mut running,
                                &mut pending,
                                &mut cohort,
                            ) {
                                Ok((tick, _)) => initialization_made_progress |= tick.made_progress,
                                Err(error) => {
                                    return Err(fatal(
                                        source(error),
                                        batch,
                                        completion::failure_owners(running, finishing),
                                        pending,
                                        shared,
                                    ));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        return Err(fatal(
                            source(error),
                            batch,
                            completion::failure_owners(running, finishing),
                            pending,
                            shared,
                        ))
                    }
                }
            }
            let mut candidate_attempts = shared.config.max_num_seqs.saturating_sub(running.len());
            while !closing
                && (!finish_tail || pending.front().is_some_and(|p| p.state.is_some()))
                && (!initialization_tick.advanced
                    || pending.front().is_some_and(|p| p.state.is_some()))
                && !survivor_forward_due
                && (!cohort_collection_handled
                    || pending.front().is_some_and(|p| p.state.is_some()))
                && (cohort.is_none() || pending.front().is_some_and(|p| p.state.is_some()))
                && (!auxiliary_survivor_forward_due
                    || pending.front().is_some_and(|p| p.state.is_some()))
                && finishing.is_empty()
                && candidate_attempts > 0
                && running.len() < shared.config.max_num_seqs
                && pull(worker, &running, &mut pending, shared)
            {
                candidate_attempts -= 1;
                if let Err(e) = pause(worker, &mut batch, &mut running) {
                    return Err(fatal(
                        source(e),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ));
                }
                let mut candidate = pending.pop_front().unwrap();
                gate_subscription = candidate
                    .job
                    .output
                    .as_ref()
                    .map(|o| o.gate.subscribe_capacity(shared.resources.waker()));
                let attempt = (|| -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
                    if shared.lock().closing || candidate.job.cancelled.load(Ordering::Acquire) {
                        return Ok(None);
                    }
                    if !try_output(worker, &mut candidate.job, shared)? {
                        return Ok(None);
                    }
                    if candidate.initialization.is_some() {
                        return Ok(None);
                    }
                    if candidate.state.is_none() {
                        cohorts::prepare_member(worker, &mut candidate)?;
                        let resources = ImagePreparationResources {
                            output_credit_bytes: output_credit(shared, &candidate.job)?,
                        };
                        if !worker
                            .denoising_generator()
                            .unwrap()
                            .try_admit_request_at_boundary(
                                candidate.prepared.as_mut().unwrap(),
                                resources,
                                batch.as_mut(),
                                &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
                            )?
                        {
                            return Ok(None);
                        }
                        candidate.job.trace.initialization_started();
                        let span = candidate.job.trace.phase_started(
                        crate::image_generation::continuous_trace::ImageTracePhase::Initialization,
                    );
                        let initialized = worker
                            .denoising_generator()
                            .unwrap()
                            .begin_image_initialization(
                                candidate.prepared.take().unwrap(),
                                candidate.job.request.image_request_ref(),
                            );
                        match initialized {
                            Ok(ImageInitializationOutcome::Ready(request)) => {
                                candidate.job.trace.phase_finished(span, crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed);
                                candidate.state = Some(request)
                            }
                            Ok(ImageInitializationOutcome::Pending(initialization)) => {
                                candidate.initialization_trace = span;
                                candidate.initialization = Some(initialization);
                                // This finite state transition must lead to the
                                // first advance, even on an otherwise idle worker.
                                initialization_made_progress = true;
                                return Ok(None);
                            }
                            Err(error) => {
                                candidate.job.trace.phase_finished(span, crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed);
                                return Err(error);
                            }
                        }
                    }
                    if shared.lock().closing || candidate.job.cancelled.load(Ordering::Acquire) {
                        // Initialization is quiescent here. Shutdown does not
                        // publish a preparing request as newly active work.
                        return Ok(None);
                    }
                    let mut members = running
                        .iter_mut()
                        .map(|r| &mut r.state)
                        .chain(std::iter::once(candidate.state.as_mut().unwrap()))
                        .collect::<Vec<_>>();
                    worker
                        .denoising_generator()
                        .unwrap()
                        .try_prepare_batch(&mut members, batch.as_mut())
                })();
                match attempt {
                    Ok(Some(next)) => {
                        // Fresh compiled/device/work grants succeeded; now the
                        // queue entry and exact membership become active together.
                        pending_removed(shared, &candidate.job);
                        {
                            let mut s = shared.lock();
                            s.active += 1;
                            s.executed_requests = s.executed_requests.saturating_add(1);
                        }
                        running.push(Running {
                            job: candidate.job,
                            state: candidate.state.take().unwrap(),
                            complete: false,
                        });
                        drop(candidate.prepared);
                        batch = Some(next);
                    }
                    Ok(None)
                        if candidate.initialization.is_none()
                            && (shared.lock().closing
                                || candidate.job.cancelled.load(Ordering::Acquire)) =>
                    {
                        let closing = shared.lock().closing;
                        reject_pending_step(
                            worker,
                            candidate,
                            shared,
                            if closing {
                                ImageServiceErrorKind::Shutdown
                            } else {
                                ImageServiceErrorKind::Cancelled
                            },
                            if closing {
                                "image service is shutting down"
                            } else {
                                "image request cancelled before joining a step"
                            },
                        );
                    }
                    Ok(None) if running.is_empty() && candidate.state.is_some() => {
                        // There is no runnable cohort to restore, and no generic
                        // native-capacity subscription in this protocol. Give a
                        // finite result instead of waiting on an unobservable
                        // device release. Pure Host/gate waits remain subscribed.
                        reject_pending_step(
                            worker,
                            candidate,
                            shared,
                            ImageServiceErrorKind::QueueFull,
                            "initial denoising cohort cannot obtain native/work capacity",
                        );
                    }
                    Ok(None) => {
                        pending.push_front(candidate);
                        break;
                    }
                    Err(ImageBatchError::Capacity { proof, .. }) if !running.is_empty() => {
                        tracing::debug!(target: "ferrum.image.capacity",
                            candidate_request_id = %candidate.job.trace.correlation.request_id,
                            candidate_batch_size = running.len() + 1,
                            complete_batch = proof.report.complete_batch,
                            required_bytes = proof.report.required_bytes,
                            capacity_bytes = proof.report.capacity_bytes,
                            retained_static_bytes = proof.report.retained_static_bytes,
                            active_stage_peak_bytes = proof.report.active_stage_peak_bytes,
                            retained_dynamic_allowance_bytes = proof.report.retained_dynamic_allowance_bytes,
                            native_allowance_bytes = proof.report.native_allowance_bytes,
                            "image continuation candidate exceeds device capacity");
                        let mut state = shared.lock();
                        state.capacity_reductions = state.capacity_reductions.saturating_add(1);
                        drop(state);
                        pending.push_front(candidate);
                        break;
                    }
                    Err(ImageBatchError::InitializationAwaitDenoisingProgress { diagnostic }) => {
                        if !running.is_empty() {
                            pending.push_front(candidate);
                            break;
                        }
                        reject_pending_step(
                            worker,
                            candidate,
                            shared,
                            ImageServiceErrorKind::QueueFull,
                            &diagnostic,
                        );
                    }
                    Err(ImageBatchError::Fatal(e)) => {
                        pending.push_front(candidate);
                        return Err(fatal(
                            e,
                            batch,
                            completion::failure_owners(running, finishing),
                            pending,
                            shared,
                        ));
                    }
                    Err(e) => {
                        let (kind, message) = super::scheduler::rejection(e)?;
                        reject_pending_step(worker, candidate, shared, kind, &message);
                    }
                }
            }
            match restore_at_boundary(worker, &mut batch, &mut running, &pending, cohort.as_ref()) {
                Ok(true) => (),
                Ok(false) => {
                    if !initialization_made_progress {
                        shared.resources.wait(observed);
                    }
                    continue;
                }
                Err(error) => {
                    return Err(fatal(
                        source(error),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ))
                }
            }
            if running.is_empty() {
                if pending.is_empty() && finishing.is_empty() && cohort.is_none() {
                    return Ok(());
                }
                if !initialization_made_progress
                    && (!finishing.is_empty()
                        || cohort.is_some()
                        || pending.iter().any(|p| p.initialization.is_some())
                        || (!shared.lock().closing
                            && pending
                                .front()
                                .is_some_and(|p| !p.job.cancelled.load(Ordering::Acquire))))
                {
                    shared.resources.wait(observed);
                }
                drop(gate_subscription);
                continue;
            }
            if running
                .iter()
                .any(|r| r.job.cancelled.load(Ordering::Acquire))
            {
                // Retirement may free another slot, but cancelling a newly
                // admitted member must not renew joins ahead of its survivor.
                survivor_forward_due = true;
                continue;
            }
            let result = worker.denoising_generator().unwrap().advance_batch(
                batch.as_mut().unwrap(),
                &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
            );
            match result {
                Ok(flags) if flags.len() == running.len() => {
                    survivor_forward_due = false;
                    auxiliary_survivor_forward_due = false;
                    for (r, complete) in running.iter_mut().zip(flags) {
                        r.complete = complete;
                    }
                    let mut s = shared.lock();
                    s.executed_batches = s.executed_batches.saturating_add(1);
                    s.last_batch_size = running.len();
                }
                Ok(_) => {
                    return Err(fatal(
                        Box::new(service_error(
                            ImageServiceErrorKind::Generation,
                            "step completion count differs from exact membership",
                        )),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ))
                }
                Err(e) => {
                    return Err(fatal(
                        source(e),
                        batch,
                        completion::failure_owners(running, finishing),
                        pending,
                        shared,
                    ))
                }
            }
            drop(gate_subscription);
        }
    })();
    result.map_err(|failure| cohorts::retain_failure(failure, cohort))
}

mod completion;

/// A suspended batch cannot observe completion, accept a new membership or
/// fake an advance while an original auxiliary owner remains native-live.
fn restore_at_boundary<O: Send + 'static>(
    worker: &mut dyn BatchWorker<Output = O>,
    batch: &mut Option<ImageDenoisingBatch>,
    running: &mut [Running<O>],
    pending: &VecDeque<Pending<O>>,
    cohort: Option<&cohorts::Cohort<O>>,
) -> Result<bool, ImageBatchError> {
    let Some(batch) = batch.as_mut() else {
        return Ok(true);
    };
    if cohorts::blocks_denoising(worker, cohort)? {
        return Ok(false);
    }
    let generator = worker.denoising_generator().unwrap();
    if !generator.batch_is_suspended(batch) {
        return Ok(true);
    }
    if pending
        .iter()
        .filter_map(|p| p.initialization.as_ref())
        .any(|initialization| generator.initialization_blocks_denoising(initialization))
    {
        return Ok(false);
    }
    generator.restore_suspended_batch(
        batch,
        &mut running.iter_mut().map(|r| &mut r.state).collect::<Vec<_>>(),
    )?;
    if generator.batch_is_suspended(batch) {
        return Err(ImageBatchError::Fatal(Box::new(service_error(
            ImageServiceErrorKind::Generation,
            "denoising restoration returned before leaving the suspended phase",
        ))));
    }
    Ok(true)
}
#[cfg(test)]
mod tests;
