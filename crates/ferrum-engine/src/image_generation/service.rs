//! Bounded request batching around one injected image generator.
//!
//! The worker calls one generation or editing entrypoint per compatible FIFO
//! batch. Each generator declares its editing ceiling separately; old custom
//! generators keep single-request edits. Pending prompts are bounded separately
//! from the generator's shared Host tensor account.

use super::{ImageEditRequest, ImageGenerationOutput, ImageGenerationRequest, QwenImage21Pipeline};
use ferrum_models::qwen_image21::image_input::{
    png_dimensions_hint, ImageInputLimits, QwenImage21ImageProcessor,
};
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

/// A batch rejection has no pending device work or recovery owners and leaves
/// its generator usable. It carries only a diagnostic, not an arbitrary error
/// whose cleanup obligations could be lost. All other errors are fatal and
/// retain their original owner until explicit service retirement.
#[derive(Debug)]
pub enum ImageBatchError {
    /// Typed request input rejection, with no native work/recovery obligation.
    InvalidInput(String),
    Rejected(String),
    Fatal(GeneratorError),
}
impl From<GeneratorError> for ImageBatchError {
    fn from(error: GeneratorError) -> Self {
        Self::Fatal(error)
    }
}
impl fmt::Display for ImageBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Rejected(message) => f.write_str(message),
            Self::Fatal(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl Error for ImageBatchError {}

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
/// Successful outputs correspond exactly to input order. Fatal errors must
/// retain their original cleanup/recovery owners; formatting them is not a
/// replacement for returning them. `Rejected` is only valid when the generator
/// remains usable and no recovery obligation exists. Compatibility must be deterministic and may
/// only restrict the common width/height grouping enforced by this service.
pub trait ImageGenerator: Send + 'static {
    fn host_account(&self) -> &HostMemoryAccount;
    /// Metadata-only validation: this method must not initialize work or create
    /// recovery obligations. Execution failures belong to `generate_batch`.
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError>;
    fn supports_edits(&self) -> bool {
        false
    }
    fn edit(
        &mut self,
        _request: ImageEditRequest,
    ) -> Result<ImageGenerationOutput, ImageBatchError> {
        Err(ImageBatchError::Rejected(
            "image editing is not supported by this generator".into(),
        ))
    }
    /// Existing generators retain their single-request editing contract.
    fn maximum_edit_batch_size(&self) -> usize {
        1
    }
    /// Scheduling only; full input validation remains in `edit_batch`.
    fn compatible_edits(&self, _a: &ImageEditRequest, _b: &ImageEditRequest) -> bool {
        false
    }
    fn edit_batch(
        &mut self,
        mut requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        if requests.len() != 1 {
            return Err(ImageBatchError::Rejected(
                "this generator supports only single-request editing".into(),
            ));
        }
        self.edit(requests.pop().unwrap())
            .map(|output| vec![output])
    }
    fn compatible(&self, _a: &ImageGenerationRequest, _b: &ImageGenerationRequest) -> bool {
        true
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError>;
    /// Explicitly retire retained sessions after normal worker shutdown.
    /// Errors must retain original recovery owners. A fatal generation error
    /// bypasses this hook so its recovery authority is not disturbed.
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        Ok(())
    }
}

impl<R: DeviceRuntime> ImageGenerator for QwenImage21Pipeline<R> {
    fn supports_edits(&self) -> bool {
        QwenImage21Pipeline::supports_edits(self)
    }
    fn edit(
        &mut self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationOutput, ImageBatchError> {
        match QwenImage21Pipeline::edit(self, request) {
            Ok(output) => Ok(output),
            Err(error) => Err(qwen_batch_error(self.is_stopped(), error)),
        }
    }
    fn maximum_edit_batch_size(&self) -> usize {
        self.config.maximum_batch_size
    }
    fn compatible_edits(&self, a: &ImageEditRequest, b: &ImageEditRequest) -> bool {
        self.supports_edits() && compatible_edit_geometry(a, b)
    }
    fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        match QwenImage21Pipeline::edit_batch(self, requests) {
            Ok(outputs) => Ok(outputs),
            Err(error) => Err(qwen_batch_error(self.is_stopped(), error)),
        }
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        QwenImage21Pipeline::shutdown(self)
            .map(|_| ())
            .map_err(|error| Box::new(error) as GeneratorError)
    }
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
            maximum_batch_size: self.config.maximum_batch_size,
            maximum_width: self.config.maximum_width.min(maximum),
            maximum_height: self.config.maximum_height.min(maximum),
            dimension_multiple: multiple,
        })
    }

    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        match QwenImage21Pipeline::generate_batch(self, requests) {
            Ok(outputs) => Ok(outputs),
            Err(error) => Err(qwen_batch_error(self.is_stopped(), error)),
        }
    }
}

fn compatible_edit_geometry(a: &ImageEditRequest, b: &ImageEditRequest) -> bool {
    let geometry = |request: &ImageEditRequest| {
        let (width, height) =
            png_dimensions_hint(request.image.bytes(), ImageInputLimits::default())?;
        QwenImage21ImageProcessor::condition_dimensions(width, height).ok()
    };
    match (geometry(a), geometry(b)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn qwen_batch_error<R: DeviceRuntime>(
    stopped: bool,
    error: super::ImageGenerationError<R>,
) -> ImageBatchError {
    // Preflight can also originate inside a stage. The pipeline marks any run
    // failure stopped, even when its variant happens to be Preflight. Only the
    // pre-run, empty-receipt path is safe to discard after returning a message.
    if !stopped && error.completed_stages.is_empty() && error.completed_steps.is_empty() {
        if let super::ImageGenerationFailure::Preflight(source) = &error.failure {
            use ferrum_models::qwen_image21::image_input::ImageInputError;
            if matches!(
                source.downcast_ref::<ImageInputError>(),
                Some(
                    ImageInputError::Invalid(_)
                        | ImageInputError::Unsupported(_)
                        | ImageInputError::Limit(_)
                        | ImageInputError::Png(_)
                )
            ) {
                return ImageBatchError::InvalidInput(format!("image input: {source:#}"));
            }
            return ImageBatchError::Rejected(format!("image preflight: {source:#}"));
        }
    }
    ImageBatchError::Fatal(Box::new(error))
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
    supports_edits: bool,
    maximum_edit_batch_size: usize,
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
        let supports_edits = generator.supports_edits();
        let edit_cap = generator.maximum_edit_batch_size();
        if edit_cap == 0 || edit_cap > capabilities.maximum_batch_size {
            return Err(service_error(
                ImageServiceErrorKind::InvalidRequest,
                "editing batch ceiling must be positive and within the generator batch capability",
            ));
        }
        Ok(Self {
            core: ServiceCore::start(Box::new(GeneratorAdapter(generator)), config)?,
            host,
            capabilities,
            supports_edits,
            maximum_edit_batch_size: if supports_edits {
                edit_cap.min(config.max_num_seqs)
            } else {
                0
            },
        })
    }

    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn capabilities(&self) -> ImageGeneratorCapabilities {
        self.capabilities
    }
    pub fn supports_edits(&self) -> bool {
        self.supports_edits
    }
    /// Effective editing ceiling after service admission limits, or zero when
    /// the generator does not support editing.
    pub fn maximum_edit_batch_size(&self) -> usize {
        self.maximum_edit_batch_size
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

    pub fn validate_edit_request(
        &self,
        request: &ImageEditRequest,
    ) -> Result<(), ImageServiceError> {
        self.validate_request(&request.generation)?;
        if !self.supports_edits
            || !self.host.owns(&request.image)
            || request.image.class() != ferrum_runtime::HostMemoryClass::InputPreprocess
            || request.image.is_empty()
            || request.image.len() > ImageInputLimits::default().maximum_encoded_bytes()
        {
            return Err(service_error(ImageServiceErrorKind::InvalidRequest,
                "editing requires supported single-image capability and a bounded PNG backing from this service Host account"));
        }
        Ok(())
    }

    /// Edits share the worker and retirement authority. Compatible edits may
    /// batch up to their own declared ceiling and never mix with generation.
    pub fn try_submit_edit(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageJob, ImageServiceError> {
        self.validate_edit_request(&request)?;
        self.core
            .try_submit_request(QueuedRequest::Edit(request))
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
    ) -> Result<Vec<Self::Output>, ImageBatchError>;
    fn edit(&mut self, _request: ImageEditRequest) -> Result<Self::Output, ImageBatchError> {
        Err(ImageBatchError::Rejected(
            "image editing is not supported by this worker".into(),
        ))
    }
    fn maximum_edit_batch_size(&self) -> usize {
        1
    }
    fn compatible_edits(&self, _a: &ImageEditRequest, _b: &ImageEditRequest) -> bool {
        false
    }
    fn edit_batch(
        &mut self,
        mut requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        if requests.len() != 1 {
            return Err(ImageBatchError::Rejected(
                "this worker supports only single-request editing".into(),
            ));
        }
        self.edit(requests.pop().unwrap())
            .map(|output| vec![output])
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        Ok(())
    }
}
struct GeneratorAdapter<P>(P);
impl<P: ImageGenerator> BatchWorker for GeneratorAdapter<P> {
    type Output = ImageGenerationOutput;
    fn maximum_edit_batch_size(&self) -> usize {
        self.0.maximum_edit_batch_size()
    }
    fn compatible_edits(&self, a: &ImageEditRequest, b: &ImageEditRequest) -> bool {
        self.0.compatible_edits(a, b)
    }
    fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        let batch_size = requests.len();
        let started = Instant::now();
        let outcome = self.0.edit_batch(requests);
        trace_request_timing("edit", batch_size, started, outcome.as_deref());
        outcome
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        self.0.shutdown()
    }
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        self.0.compatible(a, b)
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        let batch_size = requests.len();
        let started = Instant::now();
        let outcome = self.0.generate_batch(requests);
        trace_request_timing("generate", batch_size, started, outcome.as_deref());
        outcome
    }
}

fn trace_request_timing(
    request_kind: &'static str,
    batch_size: usize,
    started: Instant,
    outcome: Result<&[ImageGenerationOutput], &ImageBatchError>,
) {
    // Keep the existing total_generate_us field for both request kinds. It is
    // the generator call's host wall duration, excluding queueing and encoding.
    let total_generate_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
    if !tracing::enabled!(target: "ferrum_engine::image_generation::timing", tracing::Level::DEBUG)
    {
        return;
    }
    match outcome {
        Ok(outputs) => {
            // Each output carries the same batch stages. Record them once,
            // before cancellation or response fanout can discard an output.
            // Stage durations are host wall durations, not GPU kernel timings.
            tracing::debug!(
                target: "ferrum_engine::image_generation::timing",
                request_kind,
                job = ?outputs.first().map(|output| output.receipt.job),
                batch_size,
                output_count = outputs.len(),
                total_generate_us,
                "image batch generation completed"
            );
            if let Some(output) = outputs.first() {
                for (stage_index, stage) in output.receipt.stages.iter().enumerate() {
                    let execution = &stage.execution;
                    let plan = &execution.plan;
                    tracing::debug!(
                        target: "ferrum_engine::image_generation::timing",
                        request_kind,
                        job = output.receipt.job,
                        batch_size,
                        stage_index,
                        stage = ?plan.stage,
                        stage_batch_size = plan.batch_size,
                        plan_cache_hit = plan.plan_cache_hit,
                        session_reused = execution.session_reused,
                        weights_reused = execution.weights_reused,
                        completed_executions = execution.completed_executions,
                        compile_us = plan.compile_wall_duration_us,
                        init_us = execution.initialization_wall_duration_us,
                        execute_us = ?execution.execution_wall_durations_us,
                        close_us = stage.close_wall_duration_us,
                        "image batch stage timing"
                    );
                }
            }
        }
        Err(error) => {
            let outcome = match error {
                ImageBatchError::InvalidInput(_) => "invalid_input",
                ImageBatchError::Rejected(_) => "rejected",
                ImageBatchError::Fatal(_) => "fatal",
            };
            tracing::debug!(
                target: "ferrum_engine::image_generation::timing",
                request_kind,
                batch_size,
                total_generate_us,
                outcome,
                "image batch generation failed"
            );
        }
    }
}

enum QueuedRequest {
    Generate(ImageGenerationRequest),
    Edit(ImageEditRequest),
}

impl QueuedRequest {
    fn generation(&self) -> &ImageGenerationRequest {
        match self {
            Self::Generate(request) => request,
            Self::Edit(request) => &request.generation,
        }
    }
}

struct Queued<O> {
    request: QueuedRequest,
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
        self.try_submit_request(QueuedRequest::Generate(request))
    }
    fn try_submit_request(&self, request: QueuedRequest) -> Result<Job<O>, ImageServiceError> {
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
        let maximum_batch = match &first.request {
            QueuedRequest::Generate(_) => shared.config.max_num_seqs,
            QueuedRequest::Edit(_) => shared
                .config
                .max_num_seqs
                .min(worker.maximum_edit_batch_size().max(1)),
        };
        let mut count = 1;
        let mut incompatible = false;
        for job in state
            .queue
            .iter()
            .skip(1)
            .take(maximum_batch.saturating_sub(1))
        {
            let compatible = job.request.generation().width == first.request.generation().width
                && job.request.generation().height == first.request.generation().height
                && match (&first.request, &job.request) {
                    (QueuedRequest::Generate(a), QueuedRequest::Generate(b)) => {
                        worker.compatible(a, b)
                    }
                    (QueuedRequest::Edit(a), QueuedRequest::Edit(b)) => {
                        worker.compatible_edits(a, b)
                    }
                    _ => false,
                };
            if !compatible {
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
        if count == maximum_batch
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
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                if matches!(requests.first(), Some(QueuedRequest::Edit(_))) {
                    let edits = requests
                        .into_iter()
                        .map(|request| match request {
                            QueuedRequest::Edit(request) => Ok(request),
                            QueuedRequest::Generate(_) => Err(ImageBatchError::Rejected(
                                "mixed editing batch is unsupported".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    worker.edit_batch(edits)
                } else {
                    let generated = requests
                        .into_iter()
                        .map(|request| match request {
                            QueuedRequest::Generate(request) => Ok(request),
                            QueuedRequest::Edit(_) => Err(ImageBatchError::Rejected(
                                "mixed editing batch is unsupported".into(),
                            )),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    worker.generate_batch(generated)
                }
            }));
            let outcome = match outcome {
                Ok(Ok(outputs)) if outputs.len() == responses.len() => Ok(outputs),
                Ok(Ok(_)) => Err(ImageWorkerFailure::Generation(Box::new(service_error(
                    ImageServiceErrorKind::Generation,
                    "generator returned a different number of outputs than batch requests",
                )))),
                Ok(Err(
                    error @ (ImageBatchError::Rejected(_) | ImageBatchError::InvalidInput(_)),
                )) => {
                    let (message, kind) = match error {
                        ImageBatchError::InvalidInput(message) => {
                            (message, ImageServiceErrorKind::InvalidRequest)
                        }
                        ImageBatchError::Rejected(message) => {
                            (message, ImageServiceErrorKind::Generation)
                        }
                        ImageBatchError::Fatal(_) => unreachable!(),
                    };
                    shared.lock().active = 0;
                    for (cancelled, response) in responses {
                        let error = if cancelled.load(Ordering::Acquire) {
                            service_error(
                                ImageServiceErrorKind::Cancelled,
                                "image response cancelled; batch preflight rejected",
                            )
                        } else {
                            // A preflight rejection can be a server-side plan
                            // problem as well as a request problem. Do not turn
                            // every owner-free rejection into HTTP 400.
                            service_error(kind, message.clone())
                        };
                        let _ = response.send(Err(error));
                    }
                    // Do not reopen admission or clear `closing`: shutdown may
                    // have been requested while this batch was being prepared.
                    continue;
                }
                Ok(Err(ImageBatchError::Fatal(error))) => {
                    Err(ImageWorkerFailure::Generation(error))
                }
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
    let mut failure = match run {
        Ok(failure) => failure,
        Err(panic) => Some(ImageWorkerFailure::Panic(panic)),
    };
    if failure.is_none() {
        failure = match catch_unwind(AssertUnwindSafe(|| worker.shutdown())) {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(ImageWorkerFailure::Generation(error)),
            Err(panic) => Some(ImageWorkerFailure::Panic(panic)),
        };
    }
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
mod edit_tests;
#[cfg(test)]
mod tests;
