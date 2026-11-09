//! Device-free stage recipes. Membership changes compile a new steady recipe;
//! request-local conditioning, sampling state and Host grants stay with callers.
use super::trace::metadata_phase;
use super::{
    conditioning::PreparedConditioning, noise::JointNoiseLayout, request::BatchGeometry,
    PH3ImageInitialNoisePolicy, PH3ImagePipeline,
};
use crate::image_generation::{
    ImageGenerationRequest, ImagePipelineSourceAdapter, ImageStage, ImageStageAttachment,
    ImageStageDeclaration, ImageStagePlanRole, PreparedImageStage,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    normal_noise::{NormalNoiseFamily, NormalNoisePrecision},
    p_h3_image::{PH3ImageDenoiserPhase, PreparedPH3ImageDenoiser},
    p_h3_image_vae::PreparedPH3ImageVaeDecoderTiles,
};
use ferrum_runtime::*;

pub(super) struct CompiledConditioning<R: DeviceRuntime> {
    pub recipe: PreparedConditioning,
    pub visual: Option<PreparedImageStage<R>>,
    pub encoder: PreparedImageStage<R>,
    pub connector: PreparedImageStage<R>,
}
pub(super) struct CompiledNoise<R: DeviceRuntime> {
    pub layout: JointNoiseLayout,
    pub execution: InitialNoiseExecution<R>,
}
pub(super) enum InitialNoiseExecution<R: DeviceRuntime> {
    NativeCuda(PreparedImageStage<R>),
    CpuMt19937,
}
impl<R: DeviceRuntime> CompiledNoise<R> {
    pub fn stage(&self) -> Option<&PreparedImageStage<R>> {
        match &self.execution {
            InitialNoiseExecution::NativeCuda(stage) => Some(stage),
            InitialNoiseExecution::CpuMt19937 => None,
        }
    }
}
pub(super) struct CompiledDenoiser<R: DeviceRuntime> {
    pub recipe: PreparedPH3ImageDenoiser,
    pub stage: PreparedImageStage<R>,
}
pub(super) struct CompiledDenoising<R: DeviceRuntime> {
    pub install: CompiledDenoiser<R>,
    pub refine: CompiledDenoiser<R>,
    pub steady: CompiledDenoiser<R>,
}
pub(super) type CompiledSteady<R> = CompiledDenoiser<R>;
pub(super) struct CompiledDecoder<R: DeviceRuntime> {
    pub recipe: PreparedPH3ImageVaeDecoderTiles,
    pub stages: Vec<PreparedImageStage<R>>,
}
impl<R: DeviceRuntime> CompiledDecoder<R> {
    pub fn stage(&self, tile_count: usize) -> Result<&PreparedImageStage<R>> {
        self.recipe
            .shapes()
            .iter()
            .position(|shape| shape.tile_count() == tile_count)
            .and_then(|index| self.stages.get(index))
            .context("decoder group has no compiled actual-shape stage")
    }
    pub fn shares_weights(&self) -> bool {
        self.stages.len() > 1
    }
}
pub(super) struct PreparedGeneration<R: DeviceRuntime> {
    pub conditioning: CompiledConditioning<R>,
    pub noise: CompiledNoise<R>,
    pub denoising: CompiledDenoising<R>,
    pub decoder: CompiledDecoder<R>,
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn compile_recipe(
        &self,
        stage: ImageStage,
        attachment: ImageStageAttachment,
        batch_size: usize,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
    ) -> Result<PreparedImageStage<R>> {
        self.execution.compile_stage(
            ImageStageDeclaration {
                stage,
                role: ImageStagePlanRole::WholeStage,
                batch_size,
                weight_owner: self.execution.sources.weight_owner(stage),
                attachment,
            },
            family,
            options,
        )
    }

    pub(super) fn prepare_generation(
        &self,
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
    ) -> Result<PreparedGeneration<R>> {
        let assets = &self.execution.sources.assets;
        let recipe = assets.prepare_text_conditioning_with_family(
            request,
            identity,
            &self.execution.config,
            |provider, config, profile| {
                self.execution.prepare_family(
                    ImageStage::TextEncoder,
                    ImageStagePlanRole::WholeStage,
                    1,
                    provider,
                    config,
                    profile,
                )
            },
            |provider, config, profile| {
                self.execution.prepare_family(
                    ImageStage::TextConnector,
                    ImageStagePlanRole::WholeStage,
                    1,
                    provider,
                    config,
                    profile,
                )
            },
        )?;
        self.prepare_generation_with_conditioning(request, recipe)
    }

    /// Both presentation paths share every later noise/refiner/DiT/decoder
    /// recipe. An edit's caller must also supply its real reference encoder.
    pub(super) fn prepare_generation_with_conditioning(
        &self,
        request: &ImageGenerationRequest,
        recipe: PreparedConditioning,
    ) -> Result<PreparedGeneration<R>> {
        let assets = &self.execution.sources.assets;
        ensure!(
            (recipe.geometry.target.latent_height() as u64).checked_mul(assets.spatial_scale())
                == Some(u64::from(request.height))
                && (recipe.geometry.target.latent_width() as u64)
                    .checked_mul(assets.spatial_scale())
                    == Some(u64::from(request.width)),
            "conditioning geometry differs from generation dimensions"
        );
        let conditioning =
            metadata_phase("compile_conditioning", || self.compile_conditioning(recipe))?;
        let recipe = &conditioning.recipe;
        let layout =
            JointNoiseLayout::new(&recipe.geometry.image_sample, &recipe.geometry.audio_sample)?;
        let execution = match self.options.initial_noise {
            PH3ImageInitialNoisePolicy::NativeCudaPhiloxJointImageAudioV1 => {
                let noise_family =
                    TypedFamilyRegistration::new(NormalNoiseFamily::new_with_precision(
                        NormalNoisePrecision::F32,
                        self.options.noise_profile.as_str(),
                    )?)
                    .prepare_with_profile(
                        &serde_json::to_value(&layout.config)?,
                        &self.options.noise_profile,
                    )?;
                InitialNoiseExecution::NativeCuda(self.compile_recipe(
                    ImageStage::Noise,
                    ImageStageAttachment::MayInitializeStored,
                    1,
                    noise_family,
                    &layout.config.compile_options()?,
                )?)
            }
            PH3ImageInitialNoisePolicy::CpuMt19937ImageAudioV1 => {
                layout.validate_cpu()?;
                InitialNoiseExecution::CpuMt19937
            }
        };
        let noise = CompiledNoise { layout, execution };
        let geometry = metadata_phase("denoiser_geometry", || {
            BatchGeometry::new(
                assets.denoiser.transformer(),
                vec![&recipe.geometry],
                self.execution.config.maximum_batch_size,
                assets.limits.denoiser.max_metadata_bytes,
            )
        })?;
        let install = metadata_phase("denoiser_install_plan", || {
            self.compile_denoiser_phase(&geometry, PH3ImageDenoiserPhase::InstallFullWeights)
        })?;
        let refine = metadata_phase("denoiser_refine_plan", || {
            self.compile_denoiser_phase(&geometry, PH3ImageDenoiserPhase::RefineFreshText)
        })?;
        let steady = metadata_phase("denoiser_steady_plan", || {
            self.compile_steady_batch(&geometry)
        })?;
        drop(geometry);
        let decoder_recipe = metadata_phase("decoder_recipe", || {
            assets.vae.prepare_decoder_with_tile_batch(
                1,
                usize::try_from(request.width)?,
                usize::try_from(request.height)?,
                self.options.vae_spatial_policy,
                self.options.vae_decoder,
                self.options.vae_decoder_tile_batch,
            )
        })?;
        let multiple_shapes = decoder_recipe.shapes().len() > 1;
        let decoder = decoder_recipe
            .shapes()
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let attachment = if !multiple_shapes {
                    ImageStageAttachment::MayInitializeStored
                } else if index == 0 {
                    ImageStageAttachment::InitializeShared
                } else {
                    ImageStageAttachment::RequireShared
                };
                self.compile_recipe(
                    ImageStage::Decoder,
                    attachment,
                    usize::try_from(shape.physical_batch())?,
                    shape.family().clone(),
                    shape.compile_options(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedGeneration {
            conditioning,
            noise,
            denoising: CompiledDenoising {
                install,
                refine,
                steady,
            },
            decoder: CompiledDecoder {
                recipe: decoder_recipe,
                stages: decoder,
            },
        })
    }

    pub(super) fn compile_conditioning(
        &self,
        recipe: PreparedConditioning,
    ) -> Result<CompiledConditioning<R>> {
        let visual = recipe
            .prompts
            .visual()
            .map(|visual| -> Result<_> {
                self.compile_recipe(
                    ImageStage::VisionEncoder,
                    ImageStageAttachment::MayInitializeStored,
                    1,
                    visual.family.clone(),
                    &visual.options,
                )
            })
            .transpose()?;
        let encoder = self.compile_recipe(
            ImageStage::TextEncoder,
            ImageStageAttachment::MayInitializeStored,
            1,
            recipe.encoder.clone(),
            &recipe.encoder_options,
        )?;
        let connector = self.compile_recipe(
            ImageStage::TextConnector,
            ImageStageAttachment::MayInitializeStored,
            1,
            recipe.connector.clone(),
            &recipe.connector_options,
        )?;
        Ok(CompiledConditioning {
            recipe,
            visual,
            encoder,
            connector,
        })
    }

    pub(super) fn compile_steady_batch(
        &self,
        geometry: &BatchGeometry<'_>,
    ) -> Result<CompiledSteady<R>> {
        self.compile_denoiser_phase(geometry, PH3ImageDenoiserPhase::DenoiseRefinedText)
    }
    fn compile_denoiser_phase(
        &self,
        geometry: &BatchGeometry<'_>,
        phase: PH3ImageDenoiserPhase,
    ) -> Result<CompiledDenoiser<R>> {
        let (stage, attachment) = match phase {
            PH3ImageDenoiserPhase::InstallFullWeights => {
                (ImageStage::Denoiser, ImageStageAttachment::InitializeShared)
            }
            PH3ImageDenoiserPhase::RefineFreshText => {
                (ImageStage::TextRefiner, ImageStageAttachment::RequireShared)
            }
            PH3ImageDenoiserPhase::DenoiseRefinedText => {
                (ImageStage::Denoiser, ImageStageAttachment::RequireShared)
            }
        };
        let recipe = geometry.prepare_with_family(
            &self.execution.sources.assets.denoiser,
            phase,
            |provider, config, profile| {
                self.execution.prepare_family(
                    stage,
                    ImageStagePlanRole::WholeStage,
                    geometry.request_count(),
                    provider,
                    config,
                    profile,
                )
            },
        )?;
        ensure!(
            !recipe.request_ids().is_empty(),
            "H3 stage requires an exact nonempty membership"
        );
        let compiled = self.compile_recipe(
            stage,
            attachment,
            recipe.request_ids().len(),
            recipe.family().clone(),
            recipe.compile_options(),
        )?;
        Ok(CompiledDenoiser {
            recipe,
            stage: compiled,
        })
    }
}
