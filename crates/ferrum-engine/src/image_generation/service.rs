//! Bounded request batching around one injected image generator.
//!
//! The worker calls one generation or editing entrypoint per compatible FIFO
//! batch. Each generator declares its editing ceiling separately; old custom
//! generators keep single-request edits. Pending prompts are bounded separately
//! from the generator's shared Host tensor account.

use super::continuous_trace::ImageContinuousTraceHandle;
use super::{ImageEditRequest, ImageGenerationOutput, ImageGenerationRequest, QwenImage21Pipeline};
use ferrum_models::qwen_image21::image_input::{
    png_dimensions_hint, ImageInputLimits, QwenImage21ImageProcessor,
};
#[cfg(test)]
use ferrum_runtime::HostMemoryReservation;
use ferrum_runtime::{DeviceRuntime, HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use ferrum_types::image_trace::ImageRequestCorrelation;
use ferrum_types::ImageBatchPolicy;
use serde::Serialize;
use std::{
    any::Any,
    sync::Arc,
    time::{Duration, Instant},
};
#[cfg(test)]
use std::{
    collections::VecDeque,
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Condvar, Mutex,
    },
};
use tokio::sync::oneshot;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImageServiceConfig {
    pub batching: ImageBatchPolicy,
    pub maximum_pending: usize,
    pub maximum_pending_bytes: usize,
    pub maximum_prompt_bytes: usize,
    pub max_num_seqs: usize,
    pub request_batch_max_wait_ms: u64,
}

use crate::iterative_service::{service_error, GeneratorError};
pub use crate::iterative_service::{
    BatchError as ImageBatchError, BatchPreparation as ImageBatchPreparation,
    PreparationResources as ImagePreparationResources, ServiceError as ImageServiceError,
    ServiceErrorKind as ImageServiceErrorKind, ServiceSnapshot as ImageServiceSnapshot,
    WorkerFailure as ImageWorkerFailure,
};
type Queued<O> = crate::iterative_service::Queued<QueuedRequest, O>;
#[cfg(test)]
type State<O> = crate::iterative_service::State<QueuedRequest, O>;
#[cfg(test)]
type Shared<O> = crate::iterative_service::Shared<QueuedRequest, O>;
type Job<O> = crate::iterative_service::Job<QueuedRequest, O>;
type ServiceCore<O> = crate::iterative_service::ServiceCore<QueuedRequest, O>;
type Retirement<O> = crate::iterative_service::Retirement<QueuedRequest, O>;
impl From<ImageServiceConfig> for crate::iterative_service::ServiceConfig {
    fn from(c: ImageServiceConfig) -> Self {
        Self {
            batching: c.batching,
            maximum_pending: c.maximum_pending,
            maximum_pending_bytes: c.maximum_pending_bytes,
            max_num_seqs: c.max_num_seqs,
            request_batch_max_wait_ms: c.request_batch_max_wait_ms,
        }
    }
}
impl From<ImageGenerationRequest> for QueuedRequest {
    fn from(request: ImageGenerationRequest) -> Self {
        Self::Generate(request)
    }
}
impl crate::iterative_service::ServiceRequest for QueuedRequest {
    fn record_admission(&self, timing: crate::iterative_service::AdmissionTiming) {
        let request = self.generation();
        tracing::debug!(target: "ferrum_engine::image_generation::timing",
            batch_index = timing.batch_index, batch_size = timing.batch_size,
            seed = request.seed, width = request.width, height = request.height,
            request_kind = self.kind(), queue_wait_us = timing.queue_wait_us,
            queue_and_prepare_us = timing.queue_and_prepare_us,
            preparation_us = timing.preparation_us, "image request admitted to execution");
    }

    fn pending_bytes(&self) -> usize {
        self.pending_bytes()
            .saturating_sub(ImageService::minimum_pending_request_bytes())
    }
    fn admission_bytes(&self) -> usize {
        self.generation().prompt.capacity()
    }
    fn kind(&self) -> &'static str {
        match self {
            Self::Generate(_) => "generate",
            Self::Edit(..) => "edit",
        }
    }
}

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

/// Immutable image adapter descriptor. The generic worker receives only its
/// opaque domain fingerprint; strength and image composition stay here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageAdapterBatchIdentity {
    pub effective_identity: String,
    pub strength_bits: u32,
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
    /// Observation of the original optional collector, without retaining a
    /// runtime or creating a second clock or diagnostic allocation.
    fn continuous_trace_handle(&self) -> Option<ImageContinuousTraceHandle> {
        None
    }
    /// Opt into the completed-step protocol. Default generators retain their
    /// existing whole-batch behavior and cancellation/delivery contract.
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        None
    }
    /// Optional plan-only worker sharing model/provider identity and Host account.
    /// It must never submit device execution or admit execution resources.
    fn preparation_worker(&self) -> Option<Box<dyn ImageGenerator>> {
        None
    }
    fn adapter_batch_identity(&self) -> Option<ImageAdapterBatchIdentity> {
        None
    }
    /// Optional metadata for reusing a previous capacity-search boundary. This
    /// is only a search hint: preparation and execution admission still run.
    /// A context must describe every row's complete capacity shape and current
    /// available device ceiling. Return `None` for heterogeneous/unknown shapes.
    fn capacity_search_context(
        &self,
        _requests: &[ImageRequestRef<'_>],
    ) -> Option<ImageCapacitySearchContext> {
        None
    }
    /// Successful, completed device work only. The service measures execution
    /// wall time separately; these observations explain cold installations and
    /// the weights this batch leaves missing for the next invocation.
    fn batch_execution_feedback(
        &self,
        _outputs: &[ImageGenerationOutput],
    ) -> Option<ImageBatchExecutionFeedback> {
        None
    }
    /// Exact physical Output peak from typed request/model metadata. This runs
    /// before plan compilation to select a currently affordable FIFO prefix.
    /// Legacy generators may leave the extra proof unavailable.
    fn execution_output_bytes(
        &self,
        _requests: &[ImageRequestRef<'_>],
    ) -> Result<Option<usize>, ImageBatchError> {
        Ok(None)
    }
    /// Physical Output backing retained by this generator for its complete
    /// worker lifetime, on `host_account()`'s original parent. This is an
    /// observation of an actual owner, not an additional admission request.
    /// Install such diagnostics before constructing the service; their fixed
    /// backing must not disappear or grow while the worker can admit work.
    fn retained_output_bytes(&self) -> u64 {
        0
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
    /// Ordered input image capacity, independent of concurrent request capacity.
    fn maximum_reference_images(&self) -> usize {
        usize::from(self.supports_edits())
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
    /// Receipts from the same native close, for finite entrypoints that publish
    /// them. Legacy generators keep their original shutdown hook.
    fn shutdown_receipts(&mut self) -> Result<Vec<super::ImageStageReceipt>, GeneratorError> {
        self.shutdown()?;
        Ok(Vec::new())
    }
}

impl<R: DeviceRuntime> ImageGenerator for QwenImage21Pipeline<R> {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        self.declared_denoising_generator()
    }
    fn preparation_worker(&self) -> Option<Box<dyn ImageGenerator>> {
        self.preparation_fork()
            .map(|p| Box::new(p) as Box<dyn ImageGenerator>)
    }
    fn adapter_batch_identity(&self) -> Option<ImageAdapterBatchIdentity> {
        self.execution
            .sources
            .assets
            .receipt()
            .lora
            .as_ref()
            .map(|lora| ImageAdapterBatchIdentity {
                effective_identity: lora.identity.clone(),
                strength_bits: lora.strength.to_bits(),
            })
    }
    fn capacity_search_context(
        &self,
        requests: &[ImageRequestRef<'_>],
    ) -> Option<ImageCapacitySearchContext> {
        self.service_capacity_search_context(requests)
            .ok()
            .flatten()
    }
    fn batch_execution_feedback(
        &self,
        outputs: &[ImageGenerationOutput],
    ) -> Option<ImageBatchExecutionFeedback> {
        self.service_batch_execution_feedback(outputs)
    }
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
    fn maximum_reference_images(&self) -> usize {
        QwenImage21Pipeline::maximum_reference_images(self)
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
        self.execution.config.maximum_batch_size
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
    fn shutdown_receipts(&mut self) -> Result<Vec<super::ImageStageReceipt>, GeneratorError> {
        QwenImage21Pipeline::shutdown(self).map_err(|error| Box::new(error) as GeneratorError)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.host_account()
    }

    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        let scale = self
            .execution
            .sources
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
            maximum_batch_size: self.execution.config.maximum_batch_size,
            maximum_width: self.execution.config.maximum_width.min(maximum),
            maximum_height: self.execution.config.maximum_height.min(maximum),
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

pub(in crate::image_generation) fn preparation_error(error: anyhow::Error) -> ImageBatchError {
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
        request
            .images()
            .map(|image| {
                let (width, height) =
                    png_dimensions_hint(image.bytes(), ImageInputLimits::default())?;
                QwenImage21ImageProcessor::condition_dimensions(width, height).ok()
            })
            .collect::<Option<Vec<_>>>()
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
    if !stopped
        && matches!(
            error.failure,
            super::ImageGenerationFailure::AllocationRejected { .. }
                | super::ImageGenerationFailure::CapacityRejected { .. }
                | super::ImageGenerationFailure::InitializationCapacityRejected { .. }
        )
    {
        return ImageBatchError::Rejected(error.to_string());
    }
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

/// Only this owner constructs the worker; callers can share it through `Arc`.
/// Call `shutdown` explicitly and retain its result for error recovery. Dropping
/// a still-live service joins synchronously; it never detaches active GPU work.
pub struct ImageService {
    core: ServiceCore<ImageGenerationOutput>,
    host: HostMemoryAccount,
    capabilities: ImageGeneratorCapabilities,
    supports_edits: bool,
    maximum_reference_images: usize,
    maximum_edit_batch_size: usize,
    retained_output_bytes: u64,
    maximum_prompt_bytes: usize,
    continuous_trace: Option<ImageContinuousTraceHandle>,
    adapter: Option<ImageAdapterBatchIdentity>,
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
        let worker = GeneratorAdapter(Box::new(generator));
        let service = Self::prepare(&worker, config)?;
        service.core.start_worker(Box::new(worker))?;
        Ok(service)
    }

    /// Shared image ingress and worker metadata, before choosing threaded
    /// serving or a finite inline drain of the same lifecycle.
    fn prepare(
        worker: &GeneratorAdapter,
        config: ImageServiceConfig,
    ) -> Result<Self, ImageServiceError> {
        let generator = &worker.0;
        let capabilities = generator.capabilities().map_err(|error| {
            service_error(ImageServiceErrorKind::InvalidRequest, error.to_string())
        })?;
        Self::prepare_with_capabilities(worker, config, capabilities)
    }
    fn prepare_with_capabilities(
        worker: &GeneratorAdapter,
        config: ImageServiceConfig,
        capabilities: ImageGeneratorCapabilities,
    ) -> Result<Self, ImageServiceError> {
        let generator = &worker.0;
        validate_config(config, capabilities)?;
        let host = generator.host_account().clone();
        let retained_output_bytes = generator.retained_output_bytes();
        let continuous_trace = generator.continuous_trace_handle();
        let adapter = generator.adapter_batch_identity();
        let supports_edits = generator.supports_edits();
        let maximum_reference_images = generator.maximum_reference_images();
        if supports_edits && maximum_reference_images == 0 {
            return Err(service_error(
                ImageServiceErrorKind::InvalidRequest,
                "editing capability requires a positive input image capacity",
            ));
        }
        let edit_cap = generator.maximum_edit_batch_size();
        if edit_cap == 0 || edit_cap > capabilities.maximum_batch_size {
            return Err(service_error(
                ImageServiceErrorKind::InvalidRequest,
                "editing batch ceiling must be positive and within the generator batch capability",
            ));
        }
        Ok(Self {
            core: ServiceCore::unstarted(worker, config, Some(host.clone())),
            host,
            capabilities,
            supports_edits,
            maximum_reference_images: if supports_edits {
                maximum_reference_images
            } else {
                0
            },
            maximum_edit_batch_size: if supports_edits {
                edit_cap.min(config.max_num_seqs)
            } else {
                0
            },
            retained_output_bytes,
            maximum_prompt_bytes: config.maximum_prompt_bytes,
            continuous_trace,
            adapter,
        })
    }

    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn continuous_trace_handle(&self) -> Option<ImageContinuousTraceHandle> {
        self.continuous_trace.clone()
    }
    /// Original generator-owned Output backing that response admission must
    /// retain when proving that releasing ordinary caller outputs can help.
    pub fn retained_output_bytes(&self) -> u64 {
        self.retained_output_bytes
    }
    pub fn adapter_batch_identity(&self) -> Option<&ImageAdapterBatchIdentity> {
        self.adapter.as_ref()
    }
    pub fn capabilities(&self) -> ImageGeneratorCapabilities {
        self.capabilities
    }
    pub fn supports_edits(&self) -> bool {
        self.supports_edits
    }
    pub fn maximum_reference_images(&self) -> usize {
        self.maximum_reference_images
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
            || request.prompt.len() > self.maximum_prompt_bytes
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
        self.try_submit_with_client_attempt(request, None)
    }
    pub fn try_submit_with_client_attempt(
        &self,
        request: ImageGenerationRequest,
        client_attempt_id: Option<Uuid>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.submit_request(QueuedRequest::Generate(request), None, client_attempt_id)
    }

    /// Queue only the request and gate metadata. Output credit is acquired by
    /// the worker for the executable FIFO prefix, immediately before prepare.
    pub fn try_submit_with_output(
        &self,
        request: ImageGenerationRequest,
        gate: Arc<dyn ImageOutputGate>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.try_submit_with_output_and_client_attempt(request, gate, None)
    }
    pub fn try_submit_with_output_and_client_attempt(
        &self,
        request: ImageGenerationRequest,
        gate: Arc<dyn ImageOutputGate>,
        client_attempt_id: Option<Uuid>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.submit_request(
            QueuedRequest::Generate(request),
            Some(gate),
            client_attempt_id,
        )
    }

    fn submit_request(
        &self,
        request: QueuedRequest,
        gate: Option<Arc<dyn ImageOutputGate>>,
        client_attempt_id: Option<Uuid>,
    ) -> Result<ImageJob, ImageServiceError> {
        let correlation = ImageRequestCorrelation::new(client_attempt_id);
        let validated = match &request {
            QueuedRequest::Generate(request) => self.validate_request(request),
            QueuedRequest::Edit(request, _) => self.validate_edit_request(request),
        };
        validated.map_err(|error| self.core.submission_error(correlation, error))?;
        let (output, receiver) = match gate {
            Some(gate) => {
                let (response, receiver) = oneshot::channel();
                (
                    Some(QueuedOutput {
                        gate,
                        lease: None,
                        response,
                    }),
                    Some(receiver),
                )
            }
            None => (None, None),
        };
        self.core
            .try_submit_request_with_output_and_correlation(request, output, correlation)
            .map(|inner| ImageJob {
                inner,
                output: receiver,
            })
    }

    pub fn validate_edit_request(
        &self,
        request: &ImageEditRequest,
    ) -> Result<(), ImageServiceError> {
        self.validate_request(&request.generation)?;
        if !self.supports_edits
            || request.image_count() > self.maximum_reference_images
            || request.images().any(|image| {
                !self.host.owns(image)
                    || image.class() != ferrum_runtime::HostMemoryClass::InputPreprocess
                    || image.is_empty()
                    || image.len() > ImageInputLimits::default().maximum_encoded_bytes()
            })
        {
            return Err(service_error(ImageServiceErrorKind::InvalidRequest,
                "editing requires an admitted ordered image list within this generator's declared input capacity"));
        }
        Ok(())
    }

    /// Edits share the worker and retirement authority. Compatible edits may
    /// batch up to their own declared ceiling and never mix with generation.
    pub fn try_submit_edit(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageJob, ImageServiceError> {
        self.try_submit_edit_with_client_attempt(request, None)
    }
    pub fn try_submit_edit_with_client_attempt(
        &self,
        request: ImageEditRequest,
        client_attempt_id: Option<Uuid>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.submit_request(QueuedRequest::Edit(request, None), None, client_attempt_id)
    }

    pub fn try_submit_edit_with_output(
        &self,
        request: ImageEditRequest,
        gate: Arc<dyn ImageOutputGate>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.try_submit_edit_with_output_and_client_attempt(request, gate, None)
    }
    pub fn try_submit_edit_with_output_and_client_attempt(
        &self,
        request: ImageEditRequest,
        gate: Arc<dyn ImageOutputGate>,
        client_attempt_id: Option<Uuid>,
    ) -> Result<ImageJob, ImageServiceError> {
        self.submit_request(
            QueuedRequest::Edit(request, None),
            Some(gate),
            client_attempt_id,
        )
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
    pub fn correlation(&self) -> ImageRequestCorrelation {
        self.inner.correlation
    }
    /// Keep cancellation authority while another task owns `wait`.
    /// Dropping this handle does not cancel the image job.
    pub fn cancellation_handle(&self) -> crate::iterative_service::JobCancellationHandle {
        self.inner.cancellation_handle()
    }
    /// Step-capable generators withdraw at a completed denoising boundary.
    /// Whole-batch generators preserve delivery-only cancellation.
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
/// Keep this opaque generator owner alive while recovering the original error.
/// It contains the original runtime, provider registry, assets and Host parent.
pub struct ImageGeneratorOwner {
    _worker: Box<
        dyn crate::iterative_service::BatchWorker<
            Request = QueuedRequest,
            Output = ImageGenerationOutput,
        >,
    >,
}

#[must_use = "retain the joined failure owner, or explicitly take it for typed recovery"]
pub struct ImageServiceShutdown {
    inner: Retirement<ImageGenerationOutput>,
}
impl ImageServiceShutdown {
    /// Real adapter receipts from normal shutdown, absent when that hook did
    /// not complete. This metadata does not release a failed native owner.
    pub fn stage_receipts(&self) -> Option<&[super::ImageStageReceipt]> {
        self.inner
            .evidence
            .as_ref()?
            .downcast_ref::<Vec<super::ImageStageReceipt>>()
            .map(Vec::as_slice)
    }
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

// Legacy image adapter surface. Production and existing semantic fixtures
// are bridged to the single generic worker in iterative_adapter.
pub(crate) trait BatchWorker: Send + 'static {
    fn continuous_trace_handle(&self) -> Option<ImageContinuousTraceHandle> {
        None
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = Self::Output>> {
        None
    }
    fn preparation_worker(&self) -> Option<Box<dyn BatchWorker<Output = Self::Output>>> {
        None
    }
    type Output: Send + 'static;
    fn retained_output_bytes(&self) -> u64 {
        0
    }
    fn adapter_batch_identity(&self) -> Option<ImageAdapterBatchIdentity> {
        None
    }
    fn capacity_search_context(
        &self,
        _requests: &[&QueuedRequest],
    ) -> Option<ImageCapacitySearchContext> {
        None
    }
    fn batch_execution_feedback(
        &self,
        _outputs: &[Self::Output],
    ) -> Option<ImageBatchExecutionFeedback> {
        None
    }
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
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        self.shutdown()?;
        Ok(None)
    }
}
struct GeneratorAdapter(Box<dyn ImageGenerator>);
impl BatchWorker for GeneratorAdapter {
    fn continuous_trace_handle(&self) -> Option<ImageContinuousTraceHandle> {
        self.0.continuous_trace_handle()
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = Self::Output>> {
        self.0.denoising_generator()
    }
    fn preparation_worker(&self) -> Option<Box<dyn BatchWorker<Output = Self::Output>>> {
        self.0
            .preparation_worker()
            .map(|p| Box::new(GeneratorAdapter(p)) as Box<dyn BatchWorker<Output = Self::Output>>)
    }
    type Output = ImageGenerationOutput;
    fn retained_output_bytes(&self) -> u64 {
        self.0.retained_output_bytes()
    }
    fn adapter_batch_identity(&self) -> Option<ImageAdapterBatchIdentity> {
        self.0.adapter_batch_identity()
    }
    fn capacity_search_context(
        &self,
        requests: &[&QueuedRequest],
    ) -> Option<ImageCapacitySearchContext> {
        let requests = requests
            .iter()
            .map(|request| match request {
                QueuedRequest::Generate(request) => ImageRequestRef::Generate(request),
                QueuedRequest::Edit(request, _) => ImageRequestRef::Edit(request),
            })
            .collect::<Vec<_>>();
        self.0.capacity_search_context(&requests)
    }
    fn batch_execution_feedback(
        &self,
        outputs: &[Self::Output],
    ) -> Option<ImageBatchExecutionFeedback> {
        self.0.batch_execution_feedback(outputs)
    }
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
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        self.0
            .shutdown_receipts()
            .map(|receipts| Some(Box::new(receipts) as Box<dyn Any + Send>))
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
                ImageBatchError::AwaitHostCapacity { .. } => "host_capacity_wait",
                ImageBatchError::AwaitDenoisingProgress { .. } => "native_completion_capacity",
                ImageBatchError::InitializationAwaitDenoisingProgress { .. } => {
                    "native_initialization_capacity"
                }
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

pub(crate) enum QueuedRequest {
    Generate(ImageGenerationRequest),
    Edit(ImageEditRequest, Option<ImageEditPreparation>),
}

impl QueuedRequest {
    fn image_request_ref(&self) -> ImageRequestRef<'_> {
        match self {
            Self::Generate(request) => ImageRequestRef::Generate(request),
            Self::Edit(request, _) => ImageRequestRef::Edit(request),
        }
    }
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
                Self::Edit(request, _) => request.encoded_backing_bytes().saturating_add(
                    request
                        .references
                        .capacity()
                        .saturating_mul(std::mem::size_of::<ManagedHostBytes>()),
                ),
                Self::Generate(_) => 0,
            })
            .saturating_add(ImageService::minimum_pending_request_bytes())
    }
}

pub(crate) mod capacity_hint;
#[cfg(test)]
use crate::iterative_service::trace;
pub use capacity_hint::ImageCapacitySearchContext;
pub(crate) mod batch_efficiency;
pub use batch_efficiency::{ImageBatchExecutionFeedback, ImageBatchStageFeedback};
mod output_budget;
use crate::iterative_service::output_gate;
mod preparation;
#[cfg(test)]
use crate::iterative_service::scheduler;
#[cfg(test)]
use crate::iterative_service::scheduler::worker_loop;
use output_gate::QueuedOutput;
#[cfg(test)]
use output_gate::ResourceSignal;
pub use output_gate::{
    OutputGate as ImageOutputGate, OutputGateError as ImageOutputGateError,
    OutputLease as ImageOutputLease,
};
mod steps;
pub use steps::{
    ImageDenoisingBatch, ImageDenoisingExitReason, ImageDenoisingGenerator, ImageDenoisingRequest,
    ImageInitializationCohortCompletion, ImageInitializationCohortGenerator,
    ImageInitializationCohortMember, ImageInitializationCohortOutcome,
    ImageInitializationCohortProgress, ImageInitializationCohortRequest,
    ImageInitializationOutcome, ImageInitializationProgress, ImageRequestInitialization,
};

#[cfg(test)]
mod auto_tests;
#[cfg(test)]
mod edit_auto_tests;
#[cfg(test)]
mod edit_tests;
#[cfg(test)]
mod output_gate_tests;
#[cfg(test)]
mod retained_output_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod timing_tests;

mod iterative_adapter;

mod observation;
mod run_once;
pub use run_once::ImageServiceRun;
