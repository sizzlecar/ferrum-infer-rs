//! The declared H3 capability consumes the existing step worker. Its native
//! owner, original Host account and typed failure retirement stay unchanged.
use super::{
    clean_capacity_failure, ContinuousBatch as H3Batch, ContinuousRequest as H3Request,
    PH3ImagePipeline,
};
use crate::image_generation::{
    service::{
        preparation_error, ImageBatchError, ImageBatchPreparation, ImageDenoisingBatch,
        ImageDenoisingGenerator, ImageDenoisingRequest, ImageGenerator, ImageGeneratorCapabilities,
        ImageInitializationOutcome, ImageInitializationProgress, ImagePreparationResources,
        ImageRequestInitialization, ImageRequestRef,
    },
    ImageEditRequest, ImageGenerationError, ImageGenerationFailure, ImageGenerationOutput,
    ImageGenerationRequest,
};
use ferrum_runtime::{DeviceRuntime, HostMemoryAccount};
type GeneratorError = Box<dyn std::error::Error + Send + 'static>;

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn h3_step_error(&mut self, failure: ImageGenerationFailure<R>) -> ImageBatchError {
        if let ImageGenerationFailure::Preflight(source) = failure {
            return preparation_error(source);
        }
        self.execution.stopped = true;
        ImageBatchError::Fatal(Box::new(ImageGenerationError {
            failure,
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        }))
    }
    pub(super) fn h3_initialization_error(
        &mut self,
        error: ImageGenerationError<R>,
    ) -> ImageBatchError {
        if self.is_stopped() {
            return ImageBatchError::Fatal(Box::new(error));
        }
        if error.completed_stages.is_empty() && error.completed_steps.is_empty() {
            if let ImageGenerationFailure::Preflight(source) = error.failure {
                return preparation_error(source);
            }
        }
        let diagnostic = error.to_string();
        match clean_initialization_capacity(error.failure) {
            Ok(()) => ImageBatchError::Overloaded(diagnostic),
            Err(failure) => {
                self.execution.stopped = true;
                ImageBatchError::Fatal(Box::new(ImageGenerationError {
                    failure,
                    completed_stages: error.completed_stages,
                    completed_steps: error.completed_steps,
                }))
            }
        }
    }
}

impl<R: DeviceRuntime> ImageDenoisingGenerator for PH3ImagePipeline<R> {
    type Output = ImageGenerationOutput;
    fn initialization_uses_auxiliary_model(&self) -> bool {
        self.execution.sources.assets.has_prompt_expansion()
    }
    fn remaining_denoising_steps(&self, request: &ImageDenoisingRequest) -> Option<usize> {
        self.execution
            .sources
            .assets
            .schedule
            .steps()
            .len()
            .checked_sub(request.get::<H3Request<R>>().ok()?.step_index())
    }
    fn initialization_cohort_generator(
        &mut self,
    ) -> Option<&mut dyn crate::image_generation::service::ImageInitializationCohortGenerator> {
        self.execution
            .sources
            .assets
            .has_prompt_expansion()
            .then_some(self)
    }
    fn bind_request_correlation(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        correlation: ferrum_types::ImageRequestCorrelation,
    ) {
        self.bind_initialization_correlation(prepared, correlation);
    }
    fn record_request_exit(
        &self,
        request: &ImageDenoisingRequest,
        reason: crate::image_generation::service::ImageDenoisingExitReason,
    ) {
        if let Ok(request) = request.get::<H3Request<R>>() {
            self.trace_request_exit(request, reason);
        }
    }
    fn supports_image_request(&self, request: ImageRequestRef<'_>) -> bool {
        matches!(request, ImageRequestRef::Generate(_))
            || !self.execution.sources.assets.has_prompt_expansion()
            || self.execution.sources.assets.has_prompt_expansion_visual()
    }
    fn prepare_image_request(
        &mut self,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.prepare_initialization(request)
    }
    fn start_image_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.start_initialization_blocking(prepared, request)
    }
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.prepare_initialization(ImageRequestRef::Generate(request))
    }
    fn try_admit_request(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.admit_initialization(prepared, resources.output_credit_bytes)
    }
    fn try_admit_request_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
        batch: Option<&mut ImageDenoisingBatch>,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError> {
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<H3Request<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        let batch = batch
            .map(|batch| batch.get_mut::<H3Batch<R>>())
            .transpose()?;
        self.admit_initialization_at_boundary(
            prepared,
            resources.output_credit_bytes,
            batch,
            &mut requests,
        )
    }
    fn batch_is_suspended(&self, batch: &ImageDenoisingBatch) -> bool {
        batch.get::<H3Batch<R>>().is_ok_and(|batch| batch.suspended)
    }
    fn initialization_blocks_denoising(&self, initialization: &ImageRequestInitialization) -> bool {
        self.initialization_has_auxiliary_owner(initialization)
    }
    fn restore_suspended_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        let batch = batch.get_mut::<H3Batch<R>>()?;
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<H3Request<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        self.restore_continuous_batch(batch, &mut requests)
            .map_err(|failure| self.suspension_error(failure))
    }
    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.start_initialization_blocking(prepared, ImageRequestRef::Generate(request))
    }
    fn begin_image_initialization(
        &mut self,
        prepared: ImageBatchPreparation,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageInitializationOutcome, ImageBatchError> {
        self.begin_initialization(prepared, request)
    }
    fn advance_image_initialization(
        &mut self,
        initialization: &mut ImageRequestInitialization,
        request: ImageRequestRef<'_>,
        cancelled: bool,
        wake: &std::task::Waker,
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        self.advance_initialization(initialization, request, cancelled, wake)
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<H3Request<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        let result = match previous {
            Some(previous) => {
                self.try_replace_continuous_batch(previous.get_mut::<H3Batch<R>>()?, &mut requests)
            }
            None => self.try_prepare_continuous_batch(&mut requests),
        };
        result
            .map(|batch| batch.map(ImageDenoisingBatch::new))
            .map_err(|source| self.h3_step_error(source))
    }
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        let batch = batch.get_mut::<H3Batch<R>>()?;
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<H3Request<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        batch
            .pause(self, &mut requests)
            .map_err(|source| self.h3_step_error(source))
    }
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        let batch = batch.get_mut::<H3Batch<R>>()?;
        let mut requests = requests
            .iter_mut()
            .map(|r| r.get_mut::<H3Request<R>>())
            .collect::<Result<Vec<_>, _>>()?;
        batch
            .advance(self, &mut requests)
            .map_err(|source| self.h3_step_error(source))?;
        Ok(requests.iter().map(|r| r.is_complete()).collect())
    }
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.finish_continuous_request_borrowed(
            request.get_mut::<H3Request<R>>()?,
            batch.get_mut::<H3Batch<R>>()?,
        )
        .map_err(|source| self.h3_step_error(source))
    }
    fn try_finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Option<Self::Output>, ImageBatchError> {
        match self.finish_continuous_request_borrowed(
            request.get_mut::<H3Request<R>>()?,
            batch.get_mut::<H3Batch<R>>()?,
        ) {
            Ok(output) => Ok(Some(output)),
            Err(ImageGenerationFailure::Preflight(source))
                if source.chain().any(|error| {
                    error
                        .downcast_ref::<ferrum_runtime::HostMemoryError>()
                        .is_some_and(ferrum_runtime::HostMemoryError::is_temporary_capacity)
                }) =>
            {
                Ok(None)
            }
            Err(source) => {
                let diagnostic = source.to_string();
                match super::native_completion::clean_decoder_admission(source) {
                    Ok(()) => Err(ImageBatchError::AwaitDenoisingProgress {
                        stage: crate::image_generation::ImageStage::Decoder,
                        diagnostic,
                    }),
                    Err(source) => Err(self.h3_step_error(source)),
                }
            }
        }
    }
}

impl<R: DeviceRuntime> ImageGenerator for PH3ImagePipeline<R> {
    fn continuous_trace_handle(
        &self,
    ) -> Option<crate::image_generation::continuous_trace::ImageContinuousTraceHandle> {
        self.execution
            .continuous_trace
            .as_ref()
            .map(|trace| trace.handle())
    }
    fn retained_output_bytes(&self) -> u64 {
        self.execution.retained_diagnostic_output_bytes() as u64
    }
    fn supports_edits(&self) -> bool {
        !self.execution.sources.assets.has_prompt_expansion()
            || self.execution.sources.assets.has_prompt_expansion_visual()
    }
    fn maximum_edit_batch_size(&self) -> usize {
        self.execution.config.maximum_batch_size
    }
    fn compatible_edits(&self, _a: &ImageEditRequest, _b: &ImageEditRequest) -> bool {
        true
    }
    fn edit(
        &mut self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationOutput, ImageBatchError> {
        let mut outputs = self
            .edit_continuous_batch(vec![request])
            .map_err(|source| self.h3_initialization_error(source))?;
        Ok(outputs
            .pop()
            .expect("one admitted H3 edit returns one image"))
    }
    fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        self.edit_continuous_batch(requests)
            .map_err(|source| self.h3_initialization_error(source))
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        (!self.execution.preparation_only).then_some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        PH3ImagePipeline::host_account(self)
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        let assets = &self.execution.sources.assets;
        let [_, height_patch, width_patch] = assets.denoiser.transformer().patch_size();
        let mut a = height_patch;
        let mut b = width_patch;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let multiple = height_patch
            .checked_div(a)
            .and_then(|n| n.checked_mul(width_patch))
            .and_then(|n| n.checked_mul(assets.spatial_scale()))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| -> GeneratorError {
                Box::new(std::io::Error::other(
                    "H3 pixel dimension multiple overflows",
                ))
            })?;
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: self.execution.config.maximum_batch_size,
            maximum_width: self.execution.config.maximum_width,
            maximum_height: self.execution.config.maximum_height,
            dimension_multiple: multiple,
        })
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        self.generate_continuous_batch(requests)
            .map_err(|source| self.h3_initialization_error(source))
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        PH3ImagePipeline::shutdown(self)
            .map(|_| ())
            .map_err(|source| Box::new(source) as GeneratorError)
    }
}

fn clean_initialization_capacity<R: DeviceRuntime>(
    failure: ImageGenerationFailure<R>,
) -> Result<(), ImageGenerationFailure<R>> {
    match failure {
        ImageGenerationFailure::Pipeline {
            source,
            stages,
            weights,
            host_scopes,
            host_backing,
        } if stages.is_empty() => match clean_initialization_capacity(*source) {
            Ok(()) => {
                drop((weights, host_scopes, host_backing));
                Ok(())
            }
            Err(source) => Err(ImageGenerationFailure::Pipeline {
                source: Box::new(source),
                stages,
                weights,
                host_scopes,
                host_backing,
            }),
        },
        failure => clean_capacity_failure(failure),
    }
}
