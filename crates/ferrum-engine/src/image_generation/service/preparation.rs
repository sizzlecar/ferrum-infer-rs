//! Pipeline and request provenance for the public two-phase service contract.
//! Tokens retain immutable source owners, never bare addresses or payload copies.

use super::*;
use crate::image_generation::{
    prepare::PreparedRequest, ImageGenerationError, ImageGenerationFailure,
};
use ferrum_models::qwen_image21::image_input::QwenImage21ConditionInputs;
use ferrum_runtime::ManagedHostBytes;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
struct GenerationIdentity {
    prompt_sha256: [u8; 32],
    width: u32,
    height: u32,
    seed: u64,
    n: u32,
}
impl From<&ImageGenerationRequest> for GenerationIdentity {
    fn from(request: &ImageGenerationRequest) -> Self {
        Self {
            // Content identity avoids retaining another potentially 1 MiB
            // prompt copy outside the queue's Host charge.
            prompt_sha256: Sha256::digest(request.prompt.as_bytes()).into(),
            width: request.width,
            height: request.height,
            seed: request.seed,
            n: request.n,
        }
    }
}

#[derive(Debug, Clone)]
struct EditIdentity {
    generation: GenerationIdentity,
    image: ManagedHostBytes,
}
impl From<&ImageEditRequest> for EditIdentity {
    fn from(request: &ImageEditRequest) -> Self {
        Self {
            generation: GenerationIdentity::from(&request.generation),
            image: request.image.clone(),
        }
    }
}
impl PartialEq for EditIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation && self.image.same_region(&other.image)
    }
}

/// Every source Arc stays alive until validation, preventing address reuse.
/// The generic payload permits hardware-independent tests of this exact guard.
struct BoundPreparation<T, I> {
    pipeline: Arc<()>,
    requests: Vec<I>,
    payload: T,
}
impl<T, I: PartialEq> BoundPreparation<T, I> {
    fn new(pipeline: &Arc<()>, requests: Vec<I>, payload: T) -> Self {
        Self {
            pipeline: pipeline.clone(),
            requests,
            payload,
        }
    }
    fn validate(&self, pipeline: &Arc<()>, requests: &[I]) -> Result<(), ImageBatchError> {
        if !Arc::ptr_eq(&self.pipeline, pipeline) || self.requests != requests {
            return Err(ImageBatchError::Rejected(
                "image preparation pipeline, request order or input identity differs".into(),
            ));
        }
        Ok(())
    }
    fn into_payload(self, pipeline: &Arc<()>, requests: &[I]) -> Result<T, ImageBatchError> {
        self.validate(pipeline, requests)?;
        Ok(self.payload)
    }
}

type GenerationPreparation<R> = BoundPreparation<PreparedRequest<R>, GenerationIdentity>;
type EditBatchPreparation<R> = BoundPreparation<PreparedRequest<R>, EditIdentity>;
type EditInputPreparation = BoundPreparation<QwenImage21ConditionInputs, EditIdentity>;

fn validate_output_budget(expected: usize, actual: usize) -> Result<(), ImageBatchError> {
    if expected != actual {
        return Err(ImageBatchError::Rejected(format!(
            "image output demand differs between metadata and compiled plan: {expected} != {actual}"
        )));
    }
    Ok(())
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    fn service_resident_preference(
        &self,
        prepared: &PreparedRequest<R>,
    ) -> Result<Option<super::super::ImageDeviceCapacityReport>, ImageBatchError> {
        if self.config.residency != super::super::ImageResidencyPolicy::Adaptive {
            return Ok(None);
        }
        let mut stages = vec![
            &prepared.text_stage,
            &prepared.noise_stage,
            &prepared.denoise_stage,
            &prepared.decode_stage,
        ];
        if let Some(edit) = &prepared.edit {
            stages.extend([&edit.vision_stage, &edit.encoder_stage]);
        }
        if let Some(prefix) = &prepared.prefix_decode {
            stages.push(&prefix.stage);
        }
        self.resident_preference_with_prefix(
            &stages,
            prepared
                .prefix_decode
                .as_ref()
                .map(|p| (&prepared.denoise_stage, &p.stage)),
        )
        .map(Some)
        .map_err(preparation_error)
    }

    fn validate_service_availability(&self) -> Result<(), ImageBatchError> {
        if self.stopped || (!self.preparation_only && self.is_stopped()) {
            return Err(ImageBatchError::Fatal(Box::new(
                ImageGenerationError::<R> {
                    failure: ImageGenerationFailure::Unavailable,
                    completed_stages: Vec::new(),
                    completed_steps: Vec::new(),
                },
            )));
        }
        Ok(())
    }
    pub(super) fn prepare_service_generation(
        &mut self,
        requests: &[ImageGenerationRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.validate_service_availability()?;
        let identities = requests.iter().map(GenerationIdentity::from).collect();
        let output_bytes = self
            .service_execution_output_bytes(
                &requests
                    .iter()
                    .map(ImageRequestRef::Generate)
                    .collect::<Vec<_>>(),
            )
            .map_err(preparation_error)?;
        let prepared = self
            .prepare_batch_with_host_admission(requests, None, false)
            .map_err(preparation_error)?;
        validate_output_budget(output_bytes, prepared.host_budget.output)?;
        let resident = self.service_resident_preference(&prepared)?;
        let mut preparation = ImageBatchPreparation::new(BoundPreparation::new(
            &self.service_preparation_identity,
            identities,
            prepared,
        ));
        if let Some(report) = resident {
            preparation = preparation.with_resident_capacity(report);
        }
        Ok(preparation)
    }

    pub(super) fn prepare_service_edit_bytes(
        &self,
        request: &ImageEditRequest,
    ) -> Result<usize, ImageBatchError> {
        let (width, height) =
            png_dimensions_hint(request.image.bytes(), ImageInputLimits::default()).ok_or_else(
                || ImageBatchError::InvalidInput("invalid bounded PNG header".into()),
            )?;
        let assets = self
            .assets
            .editing
            .as_ref()
            .ok_or_else(|| ImageBatchError::Rejected("editing assets are missing".into()))?;
        assets
            .processor
            .prepared_backing_bytes(&self.host, width, height)
            .map_err(|error| preparation_error(error.into()))
    }

    pub(super) fn preprocess_service_edit(
        &mut self,
        request: &ImageEditRequest,
    ) -> Result<ImageEditPreparation, ImageBatchError> {
        self.validate_service_availability()?;
        let input =
            QwenImage21Pipeline::preprocess_edit(self, request).map_err(preparation_error)?;
        let retained_host_bytes = (|| -> anyhow::Result<usize> {
            let patches = input.vision_patches().bytes().backing_bytes();
            let vae = input.vae_rgba().bytes().backing_bytes();
            patches
                .checked_add(vae)
                .ok_or_else(|| anyhow::anyhow!("edit input retained bytes overflow"))
        })()
        .map_err(preparation_error)?;
        Ok(ImageEditPreparation::new(
            BoundPreparation::new(
                &self.service_preparation_identity,
                vec![EditIdentity::from(request)],
                input,
            ),
            retained_host_bytes,
        ))
    }

    pub(super) fn prepare_service_edits(
        &mut self,
        requests: &[(&ImageEditRequest, &ImageEditPreparation)],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.validate_service_availability()?;
        let output_bytes = self
            .service_execution_output_bytes(
                &requests
                    .iter()
                    .map(|(request, _)| ImageRequestRef::Edit(request))
                    .collect::<Vec<_>>(),
            )
            .map_err(preparation_error)?;
        let mut identities = Vec::with_capacity(requests.len());
        let mut inputs = Vec::with_capacity(requests.len());
        for (request, input) in requests {
            let identity = EditIdentity::from(*request);
            let input = input.get::<EditInputPreparation>()?;
            input.validate(
                &self.service_preparation_identity,
                std::slice::from_ref(&identity),
            )?;
            identities.push(identity);
            inputs.push(input.payload.clone());
        }
        // Borrowing embedded generation requests requires a contiguous owned
        // projection. Charge those temporary prompt copies before cloning.
        let prompt_bytes = requests
            .iter()
            .try_fold(0usize, |bytes, (request, _)| {
                bytes
                    .checked_add(request.generation.prompt.len())
                    .ok_or_else(|| anyhow::anyhow!("edit prompt preparation bytes overflow"))
            })
            .map_err(preparation_error)?;
        let _prompt_charge = if prompt_bytes > 0 {
            Some(
                self.host
                    .reserve(HostMemoryClass::InputPreprocess, prompt_bytes)
                    .map_err(|error| preparation_error(error.into()))?,
            )
        } else {
            None
        };
        let generation: Vec<_> = requests
            .iter()
            .map(|(request, _)| request.generation.clone())
            .collect();
        let prepared = self
            .prepare_preprocessed_edits_with_host_admission(&generation, &inputs, false)
            .map_err(preparation_error)?;
        validate_output_budget(output_bytes, prepared.host_budget.output)?;
        let resident = self.service_resident_preference(&prepared)?;
        let mut preparation = ImageBatchPreparation::new(BoundPreparation::new(
            &self.service_preparation_identity,
            identities,
            prepared,
        ));
        if let Some(report) = resident {
            preparation = preparation.with_resident_capacity(report);
        }
        Ok(preparation)
    }

    /// Retain one validated plan and its immutable weight owners while waiting
    /// for output backings belonging to earlier executions to be consumed.
    /// Queue inputs, pinned weights and this candidate's encoding grant cannot
    /// make progress independently; their combined hard boundary is checked
    /// before returning a temporary shortage.
    pub(super) fn try_admit_service_prepared(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: &ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        if self.preparation_only {
            return Err(ImageBatchError::Rejected(
                "plan-only worker cannot execute or admit resources".into(),
            ));
        }
        self.validate_service_availability()?;
        let owner = prepared.0.as_mut().ok_or_else(|| {
            ImageBatchError::Rejected("image preparation belongs to another generator".into())
        })?;
        let payload = if owner.is::<GenerationPreparation<R>>() {
            let bound = owner.downcast_mut::<GenerationPreparation<R>>().unwrap();
            if !Arc::ptr_eq(&self.service_preparation_identity, &bound.pipeline) {
                return Err(ImageBatchError::Rejected(
                    "image preparation belongs to another pipeline".into(),
                ));
            }
            &mut bound.payload
        } else if let Some(bound) = owner.downcast_mut::<EditBatchPreparation<R>>() {
            if !Arc::ptr_eq(&self.service_preparation_identity, &bound.pipeline) {
                return Err(ImageBatchError::Rejected(
                    "edit preparation belongs to another pipeline".into(),
                ));
            }
            &mut bound.payload
        } else {
            return Err(ImageBatchError::Rejected(
                "image preparation belongs to another generator".into(),
            ));
        };
        let mut stages = vec![
            &payload.text_stage,
            &payload.noise_stage,
            &payload.denoise_stage,
            &payload.decode_stage,
        ];
        if let Some(edit) = &payload.edit {
            stages.extend([&edit.vision_stage, &edit.encoder_stage]);
        }
        if let Some(prefix) = &payload.prefix_decode {
            stages.push(&prefix.stage);
        }
        self.validate_capacity_with_prefix(
            &stages,
            payload
                .prefix_decode
                .as_ref()
                .map(|p| (&payload.denoise_stage, &p.stage)),
        )
        .map_err(preparation_error)?;
        if payload.host_scope.is_some() {
            return Ok(true);
        }
        let claims = payload.host_budget.claims();
        self.host
            .check_scope_after_output_release(&claims, resources.output_credit_bytes)
            .map_err(|error| preparation_error(error.into()))?;
        match self.host.reserve_scope(&claims) {
            Ok(scope) => {
                payload.host_scope = Some(scope);
                Ok(true)
            }
            Err(error) if error.is_temporary_capacity() => {
                // Queue admission can race the first snapshot. Recheck the
                // non-releasable owners before parking on output consumption.
                self.host
                    .check_scope_after_output_release(&claims, resources.output_credit_bytes)
                    .map_err(|error| preparation_error(error.into()))?;
                Ok(false)
            }
            Err(error) => Err(preparation_error(error.into())),
        }
    }

    pub(super) fn execute_service_generation(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        if self.preparation_only {
            return Err(ImageBatchError::Rejected(
                "plan-only worker cannot execute or admit resources".into(),
            ));
        }
        self.validate_service_availability()?;
        let identities: Vec<_> = requests.iter().map(GenerationIdentity::from).collect();
        let prepared = prepared
            .into_inner::<GenerationPreparation<R>>()?
            .into_payload(&self.service_preparation_identity, &identities)?;
        self.execute_prepared(prepared, &requests)
            .map_err(|error| qwen_batch_error(self.is_stopped(), error))
    }

    pub(super) fn execute_service_edits(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        if self.preparation_only {
            return Err(ImageBatchError::Rejected(
                "plan-only worker cannot execute or admit resources".into(),
            ));
        }
        self.validate_service_availability()?;
        let identities: Vec<_> = requests.iter().map(EditIdentity::from).collect();
        let prepared = prepared
            .into_inner::<EditBatchPreparation<R>>()?
            .into_payload(&self.service_preparation_identity, &identities)?;
        drop(identities);
        let generation: Vec<_> = requests
            .into_iter()
            .map(|request| request.generation)
            .collect();
        self.execute_prepared(prepared, &generation)
            .map_err(|error| qwen_batch_error(self.is_stopped(), error))
    }
}

#[cfg(test)]
mod tests;
