use super::*;
use crate::image_generation::denoising::{
    ContinuousBatch, ContinuousRequest, PreparedContinuousRequest,
};
use crate::image_generation::{
    ImageDecoderBatchPolicy, ImageGenerationError, ImageGenerationFailure,
};
pub use crate::iterative_service::{
    DenoisingBatch as ImageDenoisingBatch, DenoisingExitReason as ImageDenoisingExitReason,
    DenoisingRequest as ImageDenoisingRequest,
    InitializationCohortCompletion as ImageInitializationCohortCompletion,
    InitializationCohortMember as ImageInitializationCohortMember,
    InitializationCohortOutcome as ImageInitializationCohortOutcome,
    InitializationCohortProgress as ImageInitializationCohortProgress,
    InitializationOutcome as ImageInitializationOutcome,
    InitializationProgress as ImageInitializationProgress,
    RequestInitialization as ImageRequestInitialization,
};
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

#[derive(Clone, Copy)]
pub struct ImageInitializationCohortRequest<'a> {
    pub request_id: Uuid,
    pub request: ImageRequestRef<'a>,
    /// A fresh whole-cohort snapshot, sticky once true, including closing.
    pub cancelled: bool,
}
/// Finite, sealed initialization is explicit generator behavior. The default
/// denoising-generator method returns None and preserves its singleton path.
/// Each cohort owns one original Host phase and native/recovery owner. No join
/// or member replacement is permitted after preparation.
pub trait ImageInitializationCohortGenerator: Send {
    /// Zero selects the original singleton protocol for this preparation.
    /// The worker additionally bounds this by its existing service lane limit.
    fn maximum_initialization_cohort_members(
        &mut self,
        prepared: &mut ImageBatchPreparation,
    ) -> Result<usize, ImageBatchError>;
    /// Metadata only; false leaves the next exact FIFO preparation untouched.
    fn initialization_cohort_compatible(
        &mut self,
        first: &mut ImageBatchPreparation,
        next: &mut ImageBatchPreparation,
    ) -> Result<bool, ImageBatchError>;
    /// No device submission or hidden admission. Validate all members before
    /// consuming their preparations; preserve exact ingress/source identities.
    fn prepare_initialization_cohort(
        &mut self,
        members: Vec<ImageInitializationCohortMember>,
    ) -> Result<ImageBatchPreparation, ImageBatchError>;
    /// Authorize the aggregate real shared phase once. Downstream work may
    /// instead transfer as Pending after retirement. false preserves preparation
    /// and any old runnable batch. If old
    /// native state is suspended, an ordinary refusal must restore it first.
    fn try_admit_initialization_cohort_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        batch: Option<&mut ImageDenoisingBatch>,
        survivors: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError>;
    /// A nonfatal error proves no possible native work or recovery owner and
    /// leaves the preparation safe to drop. Otherwise retain the original
    /// owner in Fatal. The worker retains this preparation in fatal custody.
    fn begin_initialization_cohort(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        members: &[ImageInitializationCohortRequest<'_>],
    ) -> Result<ImageRequestInitialization, ImageBatchError>;
    /// At most one actual execution/maintenance/close wave per call. A false
    /// made_progress registers wake on the original external wait. Completions
    /// may arrive over multiple calls but are withheld until finished. Pending
    /// handoffs are allowed only with finished=true after shared retirement.
    /// Returning Ready instead requires those real states to fit together.
    fn advance_initialization_cohort(
        &mut self,
        initialization: &mut ImageRequestInitialization,
        members: &[ImageInitializationCohortRequest<'_>],
        wake: &std::task::Waker,
    ) -> Result<ImageInitializationCohortProgress, ImageBatchError>;
    /// false is legal only after the shared auxiliary native owner no longer
    /// blocks real restoration. Per-member EOS alone is insufficient.
    fn initialization_cohort_blocks_denoising(
        &self,
        initialization: &ImageRequestInitialization,
    ) -> bool;
}

#[cfg(test)]
use crate::iterative_service::steps::initialization;
#[cfg(test)]
use crate::iterative_service::steps::{pause, restore_at_boundary, FailedOwners, Pending, Running};
#[cfg(test)]
mod cohorts {
    pub(super) type CohortFailure =
        crate::iterative_service::steps::cohorts::CohortFailure<super::super::QueuedRequest>;
}
#[cfg(test)]
mod tests;
