//! Compatibility boundary for existing image generators. The common worker
//! sees an owned image input as its associated Request and never branches on it.
use super::*;
use crate::iterative_service as core;

// Preserve erased legacy preparation workers without another executor or queue.
impl<O: Send + 'static> BatchWorker for Box<dyn BatchWorker<Output = O>> {
    type Output = O;
    fn continuous_trace_handle(&self) -> Option<ImageContinuousTraceHandle> {
        BatchWorker::continuous_trace_handle(self.as_ref())
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = Self::Output>> {
        BatchWorker::denoising_generator(self.as_mut())
    }
    fn preparation_worker(&self) -> Option<Box<dyn BatchWorker<Output = Self::Output>>> {
        BatchWorker::preparation_worker(self.as_ref())
    }
    fn retained_output_bytes(&self) -> u64 {
        BatchWorker::retained_output_bytes(self.as_ref())
    }
    fn adapter_batch_identity(&self) -> Option<ImageAdapterBatchIdentity> {
        BatchWorker::adapter_batch_identity(self.as_ref())
    }
    fn capacity_search_context(
        &self,
        _requests: &[&QueuedRequest],
    ) -> Option<ImageCapacitySearchContext> {
        BatchWorker::capacity_search_context(self.as_ref(), _requests)
    }
    fn batch_execution_feedback(
        &self,
        _outputs: &[Self::Output],
    ) -> Option<ImageBatchExecutionFeedback> {
        BatchWorker::batch_execution_feedback(self.as_ref(), _outputs)
    }
    fn execution_output_bytes(
        &self,
        _requests: &[&QueuedRequest],
    ) -> Result<Option<usize>, ImageBatchError> {
        BatchWorker::execution_output_bytes(self.as_ref(), _requests)
    }
    fn preparation_bytes(&self, _request: &QueuedRequest) -> Result<usize, ImageBatchError> {
        BatchWorker::preparation_bytes(self.as_ref(), _request)
    }
    fn prepare(
        &mut self,
        _requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        BatchWorker::prepare(self.as_mut(), _requests)
    }
    fn try_admit_prepared(
        &mut self,
        _prepared: &mut ImageBatchPreparation,
        _resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        BatchWorker::try_admit_prepared(self.as_mut(), _prepared, _resources)
    }
    fn execute(
        &mut self,
        _prepared: ImageBatchPreparation,
        requests: Vec<QueuedRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        BatchWorker::execute(self.as_mut(), _prepared, requests)
    }
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        BatchWorker::compatible(self.as_ref(), a, b)
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        BatchWorker::generate_batch(self.as_mut(), requests)
    }
    fn edit(&mut self, _request: ImageEditRequest) -> Result<Self::Output, ImageBatchError> {
        BatchWorker::edit(self.as_mut(), _request)
    }
    fn maximum_edit_batch_size(&self) -> usize {
        BatchWorker::maximum_edit_batch_size(self.as_ref())
    }
    fn compatible_edits(&self, _a: &ImageEditRequest, _b: &ImageEditRequest) -> bool {
        BatchWorker::compatible_edits(self.as_ref(), _a, _b)
    }
    fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        BatchWorker::edit_batch(self.as_mut(), requests)
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        BatchWorker::shutdown(self.as_mut())
    }
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        BatchWorker::shutdown_evidence(self.as_mut())
    }
}

impl<T: BatchWorker> core::BatchWorker for T {
    type Request = QueuedRequest;
    type Output = T::Output;
    fn lifecycle_observer(&self) -> Option<core::LifecycleObserverHandle> {
        BatchWorker::continuous_trace_handle(self)
            .map(|handle| Arc::new(handle) as core::LifecycleObserverHandle)
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn core::DenoisingGenerator<Request = Self::Request, Output = Self::Output>>
    {
        if BatchWorker::denoising_generator(self).is_some() {
            Some(self)
        } else {
            None
        }
    }
    fn preparation_worker(
        &self,
    ) -> Option<Box<dyn core::BatchWorker<Request = Self::Request, Output = Self::Output>>> {
        BatchWorker::preparation_worker(self).map(|worker| {
            Box::new(worker)
                as Box<dyn core::BatchWorker<Request = QueuedRequest, Output = Self::Output>>
        })
    }
    fn retained_output_bytes(&self) -> u64 {
        BatchWorker::retained_output_bytes(self)
    }
    fn adapter_domain_key(&self) -> Option<core::AdapterDomainKey> {
        BatchWorker::adapter_batch_identity(self)
            .map(|identity| core::AdapterDomainKey::from(&identity))
    }
    fn capacity_search_context(
        &self,
        _requests: &[&Self::Request],
    ) -> Option<ImageCapacitySearchContext> {
        BatchWorker::capacity_search_context(self, _requests)
    }
    fn batch_execution_feedback(
        &self,
        _outputs: &[Self::Output],
    ) -> Option<ImageBatchExecutionFeedback> {
        BatchWorker::batch_execution_feedback(self, _outputs)
    }
    fn execution_output_bytes(
        &self,
        _requests: &[&Self::Request],
    ) -> Result<Option<usize>, ImageBatchError> {
        BatchWorker::execution_output_bytes(self, _requests)
    }
    fn preparation_bytes(&self, _request: &Self::Request) -> Result<usize, ImageBatchError> {
        BatchWorker::preparation_bytes(self, _request)
    }
    fn prepare(
        &mut self,
        _requests: &mut [&mut Self::Request],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        BatchWorker::prepare(self, _requests)
    }
    fn try_admit_prepared(
        &mut self,
        _prepared: &mut ImageBatchPreparation,
        _resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        BatchWorker::try_admit_prepared(self, _prepared, _resources)
    }
    fn execute(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<Self::Request>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        BatchWorker::execute(self, prepared, requests)
    }
    fn compatible(&self, a: &Self::Request, b: &Self::Request) -> bool {
        a.generation().width == b.generation().width
            && a.generation().height == b.generation().height
            && match (a, b) {
                (QueuedRequest::Generate(a), QueuedRequest::Generate(b)) => {
                    BatchWorker::compatible(self, a, b)
                }
                (QueuedRequest::Edit(a, _), QueuedRequest::Edit(b, _)) => {
                    BatchWorker::compatible_edits(self, a, b)
                }
                _ => false,
            }
    }
    fn maximum_batch_size(&self, request: &Self::Request, configured: usize) -> usize {
        match request {
            QueuedRequest::Generate(_) => configured,
            QueuedRequest::Edit(..) => {
                configured.min(BatchWorker::maximum_edit_batch_size(self).max(1))
            }
        }
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        BatchWorker::shutdown(self)
    }
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        BatchWorker::shutdown_evidence(self)
    }
}

impl<T: BatchWorker> core::DenoisingGenerator for T {
    type Request = QueuedRequest;
    type Output = T::Output;
    fn initialization_uses_auxiliary_model(&mut self) -> bool {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .initialization_uses_auxiliary_model()
    }
    fn remaining_denoising_steps(&mut self, _request: &ImageDenoisingRequest) -> Option<usize> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .remaining_denoising_steps(_request)
    }
    fn initialization_cohort_generator(
        &mut self,
    ) -> Option<&mut dyn core::InitializationCohortGenerator<Request = Self::Request>> {
        if BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .is_some()
        {
            Some(self)
        } else {
            None
        }
    }
    fn bind_request_correlation(
        &mut self,
        _prepared: &mut ImageBatchPreparation,
        _correlation: ImageRequestCorrelation,
    ) {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .bind_request_correlation(_prepared, _correlation)
    }
    fn record_request_exit(
        &mut self,
        _request: &ImageDenoisingRequest,
        _reason: ImageDenoisingExitReason,
    ) {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .record_request_exit(_request, _reason)
    }
    fn supports_request(&mut self, request: &Self::Request) -> bool {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .supports_image_request(request.image_request_ref())
    }
    fn prepare_request(
        &mut self,
        request: &Self::Request,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .prepare_image_request(request.image_request_ref())
    }
    fn try_admit_request(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .try_admit_request(prepared, resources)
    }
    fn try_admit_request_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
        _batch: Option<&mut ImageDenoisingBatch>,
        _requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .try_admit_request_at_boundary(prepared, resources, _batch, _requests)
    }
    fn batch_is_suspended(&mut self, _batch: &ImageDenoisingBatch) -> bool {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .batch_is_suspended(_batch)
    }
    fn initialization_blocks_denoising(
        &mut self,
        _initialization: &ImageRequestInitialization,
    ) -> bool {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .initialization_blocks_denoising(_initialization)
    }
    fn restore_suspended_batch(
        &mut self,
        _batch: &mut ImageDenoisingBatch,
        _requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .restore_suspended_batch(_batch, _requests)
    }
    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &Self::Request,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .start_image_request(prepared, request.image_request_ref())
    }
    fn begin_initialization(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &Self::Request,
    ) -> Result<ImageInitializationOutcome, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .begin_image_initialization(prepared, request.image_request_ref())
    }
    fn advance_initialization(
        &mut self,
        _initialization: &mut ImageRequestInitialization,
        _request: &Self::Request,
        _cancelled: bool,
        _wake: &std::task::Waker,
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .advance_image_initialization(
                _initialization,
                _request.image_request_ref(),
                _cancelled,
                _wake,
            )
    }
    fn advance_initialization_at_boundary(
        &mut self,
        initialization: &mut ImageRequestInitialization,
        request: &Self::Request,
        cancelled: bool,
        wake: &std::task::Waker,
        _batch: Option<&mut ImageDenoisingBatch>,
        _requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .advance_image_initialization_at_boundary(
                initialization,
                request.image_request_ref(),
                cancelled,
                wake,
                _batch,
                _requests,
            )
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .try_prepare_batch(requests, previous)
    }
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .pause_batch(batch, requests)
    }
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .advance_batch(batch, requests)
    }
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .finish_request(request, batch)
    }
    fn try_finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Option<Self::Output>, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .expect("declared image step capability remains stable")
            .try_finish_request(request, batch)
            .map_err(|error| match error {
                // Only the legacy decoder protocol declares a clean, zero-
                // submission deferral. Other stage signals remain fatal just
                // as they were before extracting the common worker.
                error @ ImageBatchError::AwaitDenoisingProgress { stage, .. }
                    if stage != crate::image_generation::ImageStage::Decoder.into() =>
                {
                    ImageBatchError::Fatal(Box::new(error))
                }
                error => error,
            })
    }
}

impl<T: BatchWorker> core::InitializationCohortGenerator for T {
    type Request = QueuedRequest;
    fn maximum_initialization_cohort_members(
        &mut self,
        prepared: &mut ImageBatchPreparation,
    ) -> Result<usize, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .maximum_initialization_cohort_members(prepared)
    }
    fn initialization_cohort_compatible(
        &mut self,
        first: &mut ImageBatchPreparation,
        next: &mut ImageBatchPreparation,
    ) -> Result<bool, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .initialization_cohort_compatible(first, next)
    }
    fn prepare_initialization_cohort(
        &mut self,
        members: Vec<ImageInitializationCohortMember>,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .prepare_initialization_cohort(members)
    }
    fn try_admit_initialization_cohort_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        batch: Option<&mut ImageDenoisingBatch>,
        survivors: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError> {
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .try_admit_initialization_cohort_at_boundary(prepared, batch, survivors)
    }
    fn begin_initialization_cohort(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        members: &[core::InitializationCohortRequest<'_, Self::Request>],
    ) -> Result<ImageRequestInitialization, ImageBatchError> {
        let members = members
            .iter()
            .map(|member| ImageInitializationCohortRequest {
                request_id: member.request_id,
                request: member.request.image_request_ref(),
                cancelled: member.cancelled,
            })
            .collect::<Vec<_>>();
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .begin_initialization_cohort(prepared, &members)
    }
    fn advance_initialization_cohort(
        &mut self,
        initialization: &mut ImageRequestInitialization,
        members: &[core::InitializationCohortRequest<'_, Self::Request>],
        wake: &std::task::Waker,
    ) -> Result<ImageInitializationCohortProgress, ImageBatchError> {
        let members = members
            .iter()
            .map(|member| ImageInitializationCohortRequest {
                request_id: member.request_id,
                request: member.request.image_request_ref(),
                cancelled: member.cancelled,
            })
            .collect::<Vec<_>>();
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .advance_initialization_cohort(initialization, &members, wake)
    }
    fn initialization_cohort_blocks_denoising(
        &mut self,
        initialization: &ImageRequestInitialization,
    ) -> bool {
        BatchWorker::denoising_generator(self)
            .and_then(|generator| generator.initialization_cohort_generator())
            .expect("declared image cohort capability remains stable")
            .initialization_cohort_blocks_denoising(initialization)
    }
}

impl From<crate::image_generation::ImageStage> for core::ExecutionStageKey {
    fn from(stage: crate::image_generation::ImageStage) -> Self {
        Self::new(stage.name())
    }
}

impl From<&ImageAdapterBatchIdentity> for core::AdapterDomainKey {
    fn from(identity: &ImageAdapterBatchIdentity) -> Self {
        use sha2::{Digest, Sha256};
        let mut fingerprint = Sha256::new();
        fingerprint.update(b"ferrum-image-adapter-domain-v1");
        fingerprint.update((identity.effective_identity.len() as u64).to_le_bytes());
        fingerprint.update(identity.effective_identity.as_bytes());
        fingerprint.update(identity.strength_bits.to_le_bytes());
        Self::from_fingerprint(fingerprint.finalize().into())
    }
}
