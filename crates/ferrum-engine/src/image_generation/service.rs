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
use ferrum_runtime::{DeviceRuntime, HostMemoryAccount, HostMemoryClass, HostMemoryReservation};
use ferrum_types::ImageBatchPolicy;
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
    /// Only preparation may return this variant. No device work was submitted.
    Capacity {
        proof: super::ImageDeviceCapacityExceeded,
        diagnostic: String,
    },
    /// Explicit Host admission pressure; distinct from device capacity search.
    Overloaded(String),
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
            Self::Capacity { diagnostic, .. } | Self::Overloaded(diagnostic) => {
                f.write_str(diagnostic)
            }
            Self::InvalidInput(message) => f.write_str(message),
            Self::Rejected(message) => f.write_str(message),
            Self::Fatal(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl Error for ImageBatchError {}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImageServiceConfig {
    pub batching: ImageBatchPolicy,
    pub maximum_pending: usize,
    pub maximum_pending_bytes: usize,
    pub maximum_prompt_bytes: usize,
    pub max_num_seqs: usize,
    pub request_batch_max_wait_ms: u64,
}

/// A single-use, generator-owned preparation. The default empty owner keeps
/// existing generators usable; capacity-aware generators retain compiled plans.
#[derive(Default)]
pub struct ImageBatchPreparation(Option<Box<dyn Any + Send>>);
impl ImageBatchPreparation {
    pub fn new<T: Any + Send>(prepared: T) -> Self {
        Self(Some(Box::new(prepared)))
    }
    pub fn into_inner<T: Any + Send>(self) -> Result<T, ImageBatchError> {
        self.0
            .and_then(|owner| owner.downcast::<T>().ok())
            .map(|owner| *owner)
            .ok_or_else(|| {
                ImageBatchError::Rejected("image preparation belongs to another generator".into())
            })
    }
}

/// Preprocessed reference backing survives candidate reduction and requeueing.
#[derive(Default)]
pub struct ImageEditPreparation(Option<Box<dyn Any + Send>>, usize);
impl ImageEditPreparation {
    pub fn new<T: Any + Send>(prepared: T, retained_host_bytes: usize) -> Self {
        Self(Some(Box::new(prepared)), retained_host_bytes)
    }
    pub fn get<T: Any + Send>(&self) -> Result<&T, ImageBatchError> {
        self.0
            .as_ref()
            .and_then(|owner| owner.downcast_ref::<T>())
            .ok_or_else(|| {
                ImageBatchError::Rejected("edit preparation belongs to another generator".into())
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImageGeneratorCapabilities {
    pub maximum_batch_size: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub dimension_multiple: u32,
}

/// Already-held protocol-independent resources that overlap engine execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct ImagePreparationResources {
    pub output_credit_bytes: usize,
}

#[derive(Clone, Copy)]
pub enum ImageRequestRef<'a> {
    Generate(&'a ImageGenerationRequest),
    Edit(&'a ImageEditRequest),
}
impl ImageRequestRef<'_> {
    fn generation(&self) -> &ImageGenerationRequest {
        match self {
            Self::Generate(request) => request,
            Self::Edit(request) => &request.generation,
        }
    }
}

/// Implementations own their runtime and use one Host account for all jobs.
/// Successful outputs correspond exactly to input order. Fatal errors must
/// retain their original cleanup/recovery owners; formatting them is not a
/// replacement for returning them. `Rejected` is only valid when the generator
/// remains usable and no recovery obligation exists. Compatibility must be deterministic and may
/// only restrict the common width/height grouping enforced by this service.
pub trait ImageGenerator: Send + 'static {
    /// Exact physical Output peak from typed request/model metadata. This runs
    /// before plan compilation to select a currently affordable FIFO prefix.
    /// Legacy generators may leave the extra proof unavailable.
    fn execution_output_bytes(
        &self,
        _requests: &[ImageRequestRef<'_>],
    ) -> Result<Option<usize>, ImageBatchError> {
        Ok(None)
    }
    /// Bound the retained preprocessed reference before allocating it. The
    /// queue reserves this many waiting bytes across preparation and requeue.
    fn edit_preparation_bytes(
        &self,
        _request: &ImageEditRequest,
    ) -> Result<usize, ImageBatchError> {
        Ok(0)
    }
    /// Preparation must never submit device work. Only a typed capacity error
    /// permits the scheduler to reduce a candidate and prepare it again.
    fn prepare_generation(
        &mut self,
        _requests: &[ImageGenerationRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        Ok(ImageBatchPreparation::default())
    }
    fn preprocess_edit(
        &mut self,
        _request: &ImageEditRequest,
    ) -> Result<ImageEditPreparation, ImageBatchError> {
        Ok(ImageEditPreparation::default())
    }
    fn prepare_edits(
        &mut self,
        _requests: &[(&ImageEditRequest, &ImageEditPreparation)],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        Ok(ImageBatchPreparation::default())
    }
    /// Obtain execution credit for an already compiled preparation. `false`
    /// means externally held output capacity can be released; the same token
    /// remains valid for a later retry, without compiling or decoding again.
    fn try_admit_prepared(
        &mut self,
        _prepared: &mut ImageBatchPreparation,
        _resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        Ok(true)
    }
    fn execute_generation(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        if prepared.0.is_some() {
            return Err(ImageBatchError::Rejected(
                "generator did not consume its prepared plan".into(),
            ));
        }
        self.generate_batch(requests)
    }
    fn execute_edits(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        if prepared.0.is_some() {
            return Err(ImageBatchError::Rejected(
                "generator did not consume its prepared edit plan".into(),
            ));
        }
        self.edit_batch(requests)
    }
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
    fn execution_output_bytes(
        &self,
        requests: &[ImageRequestRef<'_>],
    ) -> Result<Option<usize>, ImageBatchError> {
        self.service_execution_output_bytes(requests)
            .map(Some)
            .map_err(preparation_error)
    }
    fn try_admit_prepared(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.try_admit_service_prepared(prepared, &resources)
    }
    fn prepare_generation(
        &mut self,
        requests: &[ImageGenerationRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.prepare_service_generation(requests)
    }
    fn edit_preparation_bytes(&self, request: &ImageEditRequest) -> Result<usize, ImageBatchError> {
        self.prepare_service_edit_bytes(request)
    }
    fn preprocess_edit(
        &mut self,
        request: &ImageEditRequest,
    ) -> Result<ImageEditPreparation, ImageBatchError> {
        self.preprocess_service_edit(request)
    }
    fn prepare_edits(
        &mut self,
        requests: &[(&ImageEditRequest, &ImageEditPreparation)],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.prepare_service_edits(requests)
    }
    fn execute_generation(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        self.execute_service_generation(prepared, requests)
    }
    fn execute_edits(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        self.execute_service_edits(prepared, requests)
    }
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

fn preparation_error(error: anyhow::Error) -> ImageBatchError {
    if let Some(capacity) = error.downcast_ref::<super::ImageDeviceCapacityExceeded>() {
        return ImageBatchError::Capacity {
            proof: *capacity,
            diagnostic: format!("image preflight: {error:#}"),
        };
    }
    use ferrum_models::qwen_image21::image_input::ImageInputError;
    if error.chain().any(|source| {
        matches!(
            source.downcast_ref::<ferrum_runtime::HostMemoryError>(),
            Some(ferrum_runtime::HostMemoryError::Capacity { .. })
        )
    }) {
        return ImageBatchError::Overloaded(format!("image Host admission: {error:#}"));
    }
    if matches!(
        error.downcast_ref::<ImageInputError>(),
        Some(
            ImageInputError::Invalid(_)
                | ImageInputError::Unsupported(_)
                | ImageInputError::Limit(_)
                | ImageInputError::Png(_)
        )
    ) {
        ImageBatchError::InvalidInput(format!("image input: {error:#}"))
    } else {
        ImageBatchError::Rejected(format!("image preparation: {error:#}"))
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
    pub preparing_requests: usize,
    pub pending_bytes: usize,
    pub maximum_pending_bytes: usize,
    pub rejected_requests: u64,
    pub executed_batches: u64,
    pub executed_requests: u64,
    pub last_batch_size: usize,
    pub capacity_reductions: u64,
    pub batching: ImageBatchPolicy,
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
    /// Fixed ownership cost of one production request before prompt, encoded
    /// reference and preprocessing backing. Queue storage is admitted separately
    /// because deque slots coexist with requests moved into an executing batch.
    pub fn minimum_pending_request_bytes() -> usize {
        std::mem::size_of::<Queued<ImageGenerationOutput>>()
    }

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
            core: ServiceCore::start_with_host(
                Box::new(GeneratorAdapter(generator)),
                config,
                Some(host.clone()),
            )?,
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
        self.core.try_submit(request).map(|inner| ImageJob {
            inner,
            output: None,
        })
    }

    /// Queue only the request and gate metadata. Output credit is acquired by
    /// the worker for the executable FIFO prefix, immediately before prepare.
    pub fn try_submit_with_output(
        &self,
        request: ImageGenerationRequest,
        gate: Arc<dyn ImageOutputGate>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.validate_request(&request)?;
        self.submit_with_output(QueuedRequest::Generate(request), gate)
    }

    fn submit_with_output(
        &self,
        request: QueuedRequest,
        gate: Arc<dyn ImageOutputGate>,
    ) -> Result<ImageJob, ImageServiceError> {
        let (response, receiver) = oneshot::channel();
        let output = QueuedOutput {
            gate,
            lease: None,
            response,
        };
        self.core
            .try_submit_request_with_output(request, Some(output))
            .map(|inner| ImageJob {
                inner,
                output: Some(receiver),
            })
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
            .try_submit_request(QueuedRequest::Edit(request, None))
            .map(|inner| ImageJob {
                inner,
                output: None,
            })
    }

    pub fn try_submit_edit_with_output(
        &self,
        request: ImageEditRequest,
        gate: Arc<dyn ImageOutputGate>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.validate_edit_request(&request)?;
        self.submit_with_output(QueuedRequest::Edit(request, None), gate)
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
    output: Option<oneshot::Receiver<ImageOutputLease>>,
}
impl ImageJob {
    /// Cancellation suppresses delivery, not active backend execution.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    pub async fn wait(self) -> Result<ImageGenerationOutput, ImageServiceError> {
        self.inner.wait().await
    }
    pub async fn wait_with_output(
        self,
    ) -> Result<(ImageGenerationOutput, ImageOutputLease), ImageServiceError> {
        let output = self.inner.wait().await?;
        let receiver = self.output.ok_or_else(|| {
            service_error(
                ImageServiceErrorKind::Generation,
                "image job has no output admission gate",
            )
        })?;
        let lease = receiver.await.map_err(|_| {
            service_error(
                ImageServiceErrorKind::Generation,
                "image output admission owner was not delivered",
            )
        })?;
        Ok((output, lease))
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
        || config.maximum_pending_bytes == 0
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
    fn execution_output_bytes(
        &self,
        _requests: &[&QueuedRequest],
    ) -> Result<Option<usize>, ImageBatchError> {
        Ok(None)
    }
    fn preparation_bytes(&self, _request: &QueuedRequest) -> Result<usize, ImageBatchError> {
        Ok(0)
    }
    fn prepare(
        &mut self,
        _requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        Ok(ImageBatchPreparation::default())
    }
    fn try_admit_prepared(
        &mut self,
        _prepared: &mut ImageBatchPreparation,
        _resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        Ok(true)
    }
    fn execute(
        &mut self,
        _prepared: ImageBatchPreparation,
        requests: Vec<QueuedRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        if matches!(requests.first(), Some(QueuedRequest::Edit(..))) {
            self.edit_batch(
                requests
                    .into_iter()
                    .map(|request| match request {
                        QueuedRequest::Edit(request, _) => request,
                        _ => unreachable!("compatibility was checked before preparation"),
                    })
                    .collect(),
            )
        } else {
            self.generate_batch(
                requests
                    .into_iter()
                    .map(|request| match request {
                        QueuedRequest::Generate(request) => request,
                        _ => unreachable!("compatibility was checked before preparation"),
                    })
                    .collect(),
            )
        }
    }
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
    fn execution_output_bytes(
        &self,
        requests: &[&QueuedRequest],
    ) -> Result<Option<usize>, ImageBatchError> {
        let requests: Vec<_> = requests
            .iter()
            .map(|request| match request {
                QueuedRequest::Generate(request) => ImageRequestRef::Generate(request),
                QueuedRequest::Edit(request, _) => ImageRequestRef::Edit(request),
            })
            .collect();
        self.0.execution_output_bytes(&requests)
    }
    fn try_admit_prepared(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.0.try_admit_prepared(prepared, resources)
    }
    fn preparation_bytes(&self, request: &QueuedRequest) -> Result<usize, ImageBatchError> {
        match request {
            QueuedRequest::Generate(_) => Ok(0),
            QueuedRequest::Edit(request, _) => self.0.edit_preparation_bytes(request),
        }
    }
    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        if matches!(requests.first(), Some(QueuedRequest::Edit(..))) {
            for request in requests.iter_mut() {
                if let QueuedRequest::Edit(request, input @ None) = &mut **request {
                    let bound = self.0.edit_preparation_bytes(request)?;
                    let prepared = self.0.preprocess_edit(request)?;
                    if prepared.1 > bound {
                        return Err(ImageBatchError::Rejected(
                            "edit preprocessing exceeded its declared retained Host bound".into(),
                        ));
                    }
                    *input = Some(prepared);
                }
            }
            let requests = requests
                .iter()
                .map(|request| match request {
                    QueuedRequest::Edit(request, Some(input)) => (request, input),
                    _ => unreachable!("compatible edit candidate"),
                })
                .collect::<Vec<_>>();
            self.0.prepare_edits(&requests)
        } else {
            let prompt_bytes = requests
                .iter()
                .map(|request| request.generation().prompt.len())
                .sum::<usize>();
            let _prompt_charge = if prompt_bytes > 0 {
                Some(
                    self.0
                        .host_account()
                        .reserve(HostMemoryClass::InputPreprocess, prompt_bytes)
                        .map_err(|error| preparation_error(error.into()))?,
                )
            } else {
                None
            };
            let requests = requests
                .iter()
                .map(|request| request.generation().clone())
                .collect::<Vec<_>>();
            self.0.prepare_generation(&requests)
        }
    }
    fn execute(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<QueuedRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        let batch_size = requests.len();
        let started = Instant::now();
        let edit = matches!(requests.first(), Some(QueuedRequest::Edit(..)));
        let outcome = if edit {
            self.0.execute_edits(
                prepared,
                requests
                    .into_iter()
                    .map(|request| match request {
                        QueuedRequest::Edit(request, _) => request,
                        _ => unreachable!("compatible edit candidate"),
                    })
                    .collect(),
            )
        } else {
            self.0.execute_generation(
                prepared,
                requests
                    .into_iter()
                    .map(|request| match request {
                        QueuedRequest::Generate(request) => request,
                        _ => unreachable!("compatible generation candidate"),
                    })
                    .collect(),
            )
        };
        trace_request_timing(
            if edit { "edit" } else { "generate" },
            batch_size,
            started,
            outcome.as_deref(),
        );
        outcome
    }
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
                        width = output.receipt.width,
                        height = output.receipt.height,
                        batch_size,
                        stage_index,
                        stage = ?plan.stage,
                        role = ?plan.role,
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
                ImageBatchError::Capacity { .. } => "capacity",
                ImageBatchError::Overloaded(_) => "overloaded",
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
    Edit(ImageEditRequest, Option<ImageEditPreparation>),
}

impl QueuedRequest {
    fn generation(&self) -> &ImageGenerationRequest {
        match self {
            Self::Generate(request) => request,
            Self::Edit(request, _) => &request.generation,
        }
    }
    fn pending_bytes(&self) -> usize {
        self.generation()
            .prompt
            .capacity()
            .saturating_add(match self {
                Self::Edit(request, _) => request.image.backing_bytes(),
                Self::Generate(_) => 0,
            })
            .saturating_add(ImageService::minimum_pending_request_bytes())
    }
}

struct Queued<O> {
    request: QueuedRequest,
    submitted: Instant,
    cancelled: Arc<AtomicBool>,
    response: oneshot::Sender<Result<O, ImageServiceError>>,
    output: Option<QueuedOutput>,
    _prompt_reservation: Option<HostMemoryReservation>,
    preprocessing_bytes: usize,
    preparation_duration: Duration,
}
impl<O> Queued<O> {
    fn pending_bytes(&self) -> usize {
        self.request
            .pending_bytes()
            .saturating_add(self.preprocessing_bytes)
    }
}
struct State<O> {
    queue: VecDeque<Queued<O>>,
    // Field order frees the deque before refunding its backing reservation.
    queue_backing: Option<HostMemoryReservation>,
    accepting: bool,
    closing: bool,
    active: usize,
    preparing: usize,
    preparing_bytes: usize,
    rejected: u64,
    executed_batches: u64,
    executed_requests: u64,
    last_batch_size: usize,
    capacity_reductions: u64,
}
impl<O> State<O> {
    fn reserve_queue_slot(
        &mut self,
        host: Option<&HostMemoryAccount>,
        maximum_pending: usize,
    ) -> Result<(), ImageServiceError> {
        let full = || {
            service_error(
                ImageServiceErrorKind::QueueFull,
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
                .checked_mul(std::mem::size_of::<Queued<O>>())
                .filter(|bytes| *bytes <= isize::MAX as usize)
                .ok_or_else(full)?;
            host.map(|host| host.reserve(HostMemoryClass::InputPreprocess, bytes))
                .transpose()
                .map_err(|error| {
                    service_error(
                        ImageServiceErrorKind::QueueFull,
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
struct Shared<O> {
    state: Mutex<State<O>>,
    changed: Condvar,
    resources: Arc<ResourceSignal>,
    config: ImageServiceConfig,
    host: Option<HostMemoryAccount>,
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
        self.resources.notify();
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
        self.shared.resources.notify();
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
    #[cfg(test)]
    fn start(
        worker: Box<dyn BatchWorker<Output = O>>,
        config: ImageServiceConfig,
    ) -> Result<Self, ImageServiceError> {
        Self::start_with_host(worker, config, None)
    }
    fn start_with_host(
        worker: Box<dyn BatchWorker<Output = O>>,
        config: ImageServiceConfig,
        host: Option<HostMemoryAccount>,
    ) -> Result<Self, ImageServiceError> {
        let shared = Arc::new(Shared {
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
    fn try_submit(&self, request: ImageGenerationRequest) -> Result<Job<O>, ImageServiceError> {
        self.try_submit_request(QueuedRequest::Generate(request))
    }
    fn try_submit_request(&self, request: QueuedRequest) -> Result<Job<O>, ImageServiceError> {
        self.try_submit_request_with_output(request, None)
    }
    fn try_submit_request_with_output(
        &self,
        request: QueuedRequest,
        output: Option<QueuedOutput>,
    ) -> Result<Job<O>, ImageServiceError> {
        let mut state = self.shared.lock();
        prune_cancelled(&mut state.queue);
        if !state.accepting {
            return Err(service_error(
                ImageServiceErrorKind::Stopped,
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
                > self.shared.config.maximum_pending_bytes
        {
            state.rejected = state.rejected.saturating_add(1);
            return Err(service_error(
                ImageServiceErrorKind::QueueFull,
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
                        .generation()
                        .prompt
                        .capacity()
                        .saturating_add(ImageService::minimum_pending_request_bytes()),
                )
            })
            .transpose()
            .map_err(|error| {
                state.rejected = state.rejected.saturating_add(1);
                service_error(
                    ImageServiceErrorKind::QueueFull,
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
        state.queue.push_back(Queued {
            request,
            submitted: Instant::now(),
            cancelled: cancelled.clone(),
            response,
            output,
            _prompt_reservation: prompt_reservation,
            preprocessing_bytes: 0,
            preparation_duration: Duration::ZERO,
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
    // Terminal shutdown releases both the retained allocation and its charge.
    state.queue = VecDeque::new();
    state.queue_backing.take();
}

mod output_budget;
mod output_gate;
mod preparation;
pub use output_gate::{ImageOutputGate, ImageOutputGateError, ImageOutputLease};
use output_gate::{QueuedOutput, ResourceSignal};
mod scheduler;
use scheduler::worker_loop;

#[cfg(test)]
mod auto_tests;
#[cfg(test)]
mod edit_auto_tests;
#[cfg(test)]
mod edit_tests;
#[cfg(test)]
mod output_gate_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod timing_tests;
