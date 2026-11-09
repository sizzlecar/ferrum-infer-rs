//! Explicit per-image edit streams. Native Philox is intentionally distinct
//! from the reference CPU Torch generator; neither call continues global RNG.
use super::PH3ImagePipeline;
use crate::image_generation::{
    ImageGenerationFailure, ImageStage, ImageStageAttachment, ImageStageDeclaration,
    ImageStagePlanRole, ImageStageReceipt, PreparedImageStage,
};
use anyhow::{ensure, Result};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily, NormalNoisePrecision};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageEditNoisePolicy {
    /// Fresh seed 42 for the complete posterior, and the request seed for
    /// complete patch-order reference augmentation. No tile-specific draws.
    #[default]
    NativeCudaPhiloxPosterior42ReferenceSeedV1,
}
impl PH3ImageEditNoisePolicy {
    pub const fn identity(self) -> &'static str {
        match self {
            Self::NativeCudaPhiloxPosterior42ReferenceSeedV1 => {
                "p_h3_image.edit_noise.native_cuda_philox.posterior42.patch_reference_seed.v1"
            }
        }
    }
    pub const fn posterior_seed(self) -> u64 {
        42
    }
    pub const fn augmentation(self) -> f64 {
        0.999
    }
}

pub(super) struct CompiledEditNoise<R: DeviceRuntime> {
    pub config: NormalNoiseConfig,
    pub stage: PreparedImageStage<R>,
}
impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn compile_edit_noise(&self, elements: u64) -> Result<CompiledEditNoise<R>> {
        let config =
            NormalNoiseConfig::new_with_precision(1, 1, elements, NormalNoisePrecision::F32)?;
        let family = TypedFamilyRegistration::new(NormalNoiseFamily::new_with_precision(
            NormalNoisePrecision::F32,
            self.options.noise_profile.as_str(),
        )?)
        .prepare_with_profile(&serde_json::to_value(&config)?, &self.options.noise_profile)?;
        let stage = self.execution.compile_stage(
            ImageStageDeclaration {
                stage: ImageStage::Noise,
                role: ImageStagePlanRole::WholeStage,
                batch_size: 1,
                weight_owner: None,
                attachment: ImageStageAttachment::MayInitializeStored,
            },
            family,
            &config.compile_options()?,
        )?;
        Ok(CompiledEditNoise { config, stage })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_edit_noise(
        &mut self,
        prepared: &CompiledEditNoise<R>,
        seed: u64,
        destination: &HostMemoryAccount,
        tensor: ResolvedTensorSpec,
        output_id: ProgramValueId,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        let work = self.execution.execution_host().clone();
        let validate = || -> Result<()> {
            ensure!(
                work.same_parent(destination)
                    && tensor.element_type() == ElementType::F32
                    && tensor.layout() == &ResolvedTensorLayout::Contiguous
                    && tensor.minimum_storage_bytes()?
                        == prepared.config.output_tensor()?.minimum_storage_bytes()?,
                "edit noise requires one exact F32 draw and the existing Host parent"
            );
            Ok(())
        };
        validate().map_err(ImageGenerationFailure::Preflight)?;
        let state = super::noise::managed_state(&work, PhiloxSeedOffset::fresh(seed))
            .map_err(ImageGenerationFailure::Preflight)?;
        let active = self
            .execution
            .start_stage(prepared.stage.clone(), job, receipts)?;
        let Some(actor) = self.execution.serial_actor.as_mut() else {
            return Err(active.processing(anyhow::anyhow!("H3 serial actor is closed")));
        };
        let (active, outputs) = active.execute_on_actor(actor, job, vec![state])?;
        let copy = || -> Result<_> {
            ensure!(
                outputs.len() == 1,
                "edit noise requires exactly one readback"
            );
            let output = &outputs[0];
            ensure!(
                output.value_id().as_str() == NormalNoiseFamily::OUTPUT_ID
                    && output.tensor() == &prepared.config.output_tensor()?
                    && work.owns(output.bytes())
                    && output.bytes().class() == HostMemoryClass::Output
                    && output.bytes().len() == usize::try_from(tensor.minimum_storage_bytes()?)?,
                "edit noise readback differs from its declared output or owner"
            );
            let mut bytes = destination
                .allocate_zeroed(HostMemoryClass::InputPreprocess, output.bytes().len())?;
            bytes.as_mut().copy_from_slice(output.bytes().bytes());
            Ok(ManagedHostTensor::new(output_id, tensor, bytes.freeze())?)
        };
        let sample = match copy() {
            Ok(sample) => sample,
            Err(source) => return Err(active.processing(source)),
        };
        drop(outputs);
        self.execution.finish_stage(active, receipts)?;
        Ok(sample)
    }
}
