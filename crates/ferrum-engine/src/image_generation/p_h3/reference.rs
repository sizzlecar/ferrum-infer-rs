//! Reference-image encoding on the existing image executor. The caller owns
//! the complete, immutable RGB image and posterior noise; tile normalization,
//! raw-neighbour stitching and posterior sampling remain in the VAE consumer.
use super::PH3ImagePipeline;
use crate::image_generation::{
    ImageGenerationFailure, ImagePipelineSourceAdapter, ImageStage, ImageStageAttachment,
    ImageStageDeclaration, ImageStagePlanRole, ImageStageReceipt, PreparedImageStage,
};
use anyhow::{ensure, Result};
use ferrum_models::p_h3_image_vae::{
    PH3ImageVaePipelineError, PH3ImageVaePixelDomain, PreparedPH3ImageVaeEncoder,
};
use ferrum_runtime::*;
use std::time::Instant;

pub(super) struct CompiledReferenceEncoder<R: DeviceRuntime> {
    pub recipe: PreparedPH3ImageVaeEncoder,
    pub stage: PreparedImageStage<R>,
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    /// Metadata and plans only. The reference canvas has its own geometry;
    /// neither target-generation dimensions nor a model name select it.
    pub(super) fn compile_reference_encoder(
        &self,
        width: u32,
        height: u32,
    ) -> Result<CompiledReferenceEncoder<R>> {
        ensure!(!self.is_stopped(), "H3 image pipeline is stopped");
        let recipe = self.execution.sources.assets.vae.prepare_encoder(
            1,
            usize::try_from(width)?,
            usize::try_from(height)?,
            self.options.vae_spatial_policy,
            self.options.vae_encoder,
        )?;
        let stage = self.execution.compile_stage(
            ImageStageDeclaration {
                stage: ImageStage::VaeEncoder,
                role: ImageStagePlanRole::WholeStage,
                batch_size: 1,
                weight_owner: self.execution.sources.weight_owner(ImageStage::VaeEncoder),
                attachment: ImageStageAttachment::MayInitializeStored,
            },
            recipe.family().clone(),
            recipe.compile_options(),
        )?;
        Ok(CompiledReferenceEncoder { recipe, stage })
    }

    /// Runs real complete encoder tiles and finishes the provided-noise
    /// posterior. Static source preparation and Host grants use the caller's
    /// current shared execution scope. The returned latent is F32 [1,C,H,W];
    /// adding the DiT's T=1 view requires no payload copy or dtype conversion.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_reference_encoder(
        &mut self,
        prepared: &CompiledReferenceEncoder<R>,
        pixels: &ManagedHostTensor,
        posterior_noise: &ManagedHostTensor,
        job: u64,
        request: &RequestIdentity,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        if self.is_stopped() {
            return Err(ImageGenerationFailure::Unavailable);
        }
        let host = self.execution.execution_host().clone();
        let validate = || -> Result<()> {
            ensure!(
                prepared.stage.declaration().stage == ImageStage::VaeEncoder
                    && prepared.stage.declaration().batch_size == 1
                    && prepared.stage.family().fingerprint()?
                        == prepared.recipe.family().fingerprint()?,
                "reference encoder stage differs from its validated VAE recipe"
            );
            for (tensor, expected) in [
                (pixels, prepared.recipe.pixel_input_spec()?),
                (posterior_noise, prepared.recipe.posterior_noise_spec()?),
            ] {
                ensure!(
                    tensor.tensor() == &expected
                        && host.owns(tensor.bytes())
                        && tensor.bytes().class() == HostMemoryClass::InputPreprocess
                        && tensor.bytes().len() as u64 == expected.minimum_storage_bytes()?,
                    "reference encoder requires exact original F32 RGB/noise on its Host parent"
                );
            }
            Ok(())
        };
        validate().map_err(ImageGenerationFailure::Preflight)?;
        let mut active = self
            .execution
            .start_stage(prepared.stage.clone(), job, receipts)?;
        let Some(actor) = self.execution.serial_actor.as_mut() else {
            return Err(active.processing(anyhow::anyhow!("H3 serial actor is closed")));
        };
        let output = (|| -> std::result::Result<_, PH3ImageVaePipelineError<R>> {
            let mut consumer = prepared
                .recipe
                .bind(
                    &mut active.session,
                    pixels,
                    PH3ImageVaePixelDomain::UnitRgb,
                    posterior_noise,
                    request.clone(),
                    host.clone(),
                    host.clone(),
                )
                .map_err(PH3ImageVaePipelineError::Preparation)?;
            loop {
                let run = RunId::new(format!(
                    "run.image.h3.{job}.reference.encode.{}",
                    consumer.completed_tiles()
                ))
                .map_err(anyhow::Error::from)
                .map_err(PH3ImageVaePipelineError::Preparation)?;
                let started = Instant::now();
                let result = consumer.advance(actor, run);
                active
                    .receipt
                    .execution_wall_durations_us
                    .push(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
                active.receipt.completed_executions = consumer.executed_tiles();
                if let Some(profile) = consumer.take_execution_profile() {
                    active.receipt.device_profiles.push(profile);
                }
                if result? {
                    break;
                }
            }
            consumer
                .finish(
                    ProgramValueId::new("value.image.h3.reference.latent")
                        .map_err(anyhow::Error::from)
                        .map_err(PH3ImageVaePipelineError::Preparation)?,
                )
                .map_err(PH3ImageVaePipelineError::Preparation)
        })();
        let latent = match output {
            Ok(latent) => latent,
            Err(PH3ImageVaePipelineError::Preparation(source)) => {
                return Err(active.processing(source));
            }
            Err(PH3ImageVaePipelineError::Execution(source)) => {
                return Err(ImageGenerationFailure::Execution { source, active });
            }
        };
        self.execution.finish_stage(active, receipts)?;
        Ok(latent)
    }
}
