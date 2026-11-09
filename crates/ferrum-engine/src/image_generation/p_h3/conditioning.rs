//! A text request's immutable component recipes. Compilation and execution
//! remain with the shared image executor, including both physical weight owners.
use super::{request::RequestGeometry, vision::PreparedVision, PH3ImageAssets};
use crate::image_generation::{ImageGenerationRequest, ImagePipelineConfig};
use anyhow::{ensure, Result};
use ferrum_models::p_h3_image::{
    PH3ImageReferenceImage, PH3ImageTextConditioningConfig, PH3ImageTextConditioningFamily,
    PH3ImageTextConnectorProjectionConfig, PH3ImageTextConnectorProjectionFamily,
    PH3ImageTextPromptBatch,
};
use ferrum_models::qwen3_vl_vision::comfy::Qwen3VLVisionComfyPixelPolicy;
use ferrum_runtime::*;
use std::collections::BTreeMap;

pub(super) struct PreparedConditioning {
    pub prompts: PreparedPrompts,
    pub config: PH3ImageTextConditioningConfig,
    pub geometry: RequestGeometry,
    pub encoder: PreparedModelFamily,
    pub encoder_options: ProgramPlanCompileOptions,
    pub connector: PreparedModelFamily,
    pub connector_options: ProgramPlanCompileOptions,
}
pub(super) enum PreparedPrompts {
    Text(PH3ImageTextPromptBatch),
    Image(PreparedVision),
}
impl PreparedPrompts {
    pub fn text(&self) -> &PH3ImageTextPromptBatch {
        match self {
            Self::Text(p) => p,
            Self::Image(p) => p.prompt.text(),
        }
    }
    pub fn visual(&self) -> Option<&PreparedVision> {
        match self {
            Self::Text(_) => None,
            Self::Image(p) => Some(p),
        }
    }
}
impl PH3ImageAssets {
    /// Validate the prompt and all component identities before granting device
    /// or tensor memory. No chat wrapping, silent token truncation or resizing.
    pub(super) fn prepare_text_conditioning(
        &self,
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
        pipeline: &ImagePipelineConfig,
    ) -> Result<PreparedConditioning> {
        self.prepare_text_conditioning_with_family(
            request,
            identity,
            pipeline,
            prepare_uncached::<PH3ImageTextConditioningFamily>,
            prepare_uncached::<PH3ImageTextConnectorProjectionFamily>,
        )
    }

    pub(super) fn prepare_text_conditioning_with_family(
        &self,
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
        pipeline: &ImagePipelineConfig,
        encoder: impl FnOnce(
            PH3ImageTextConditioningFamily,
            &PH3ImageTextConditioningConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
        connector: impl FnOnce(
            PH3ImageTextConnectorProjectionFamily,
            &PH3ImageTextConnectorProjectionConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
    ) -> Result<PreparedConditioning> {
        let prompts = self.tokenizer.prepare(&[request.prompt.as_str()])?;
        ensure!(
            prompts.valid_lengths().len() == 1,
            "one request must retain one valid conditioning prefix"
        );
        let geometry =
            self.text_request_geometry(request, identity, pipeline, prompts.valid_lengths()[0])?;
        self.conditioning_recipe(PreparedPrompts::Text(prompts), geometry, encoder, connector)
    }

    /// Reference geometry must come from the independently prepared real VAE
    /// encoder. Dimensions describe the RGBA8 presentation source after any
    /// explicitly selected node-level outer resize. `pixels`
    /// declares only Qwen-VL's subsequent patch resize/normalization profile.
    /// This supplies actual visual conditioning and Image tags; it does not
    /// create or stand in for the reference latent payload.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_image_conditioning(
        &self,
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
        pipeline: &ImagePipelineConfig,
        source_width: u32,
        source_height: u32,
        reference: PH3ImageReferenceImage,
        pixels: Qwen3VLVisionComfyPixelPolicy,
    ) -> Result<PreparedConditioning> {
        self.prepare_image_conditioning_with_family(
            request,
            identity,
            pipeline,
            source_width,
            source_height,
            reference,
            pixels,
            prepare_uncached::<PH3ImageTextConditioningFamily>,
            prepare_uncached::<PH3ImageTextConnectorProjectionFamily>,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_image_conditioning_with_family(
        &self,
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
        pipeline: &ImagePipelineConfig,
        source_width: u32,
        source_height: u32,
        reference: PH3ImageReferenceImage,
        pixels: Qwen3VLVisionComfyPixelPolicy,
        encoder: impl FnOnce(
            PH3ImageTextConditioningFamily,
            &PH3ImageTextConditioningConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
        connector: impl FnOnce(
            PH3ImageTextConnectorProjectionFamily,
            &PH3ImageTextConnectorProjectionConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
    ) -> Result<PreparedConditioning> {
        let visual =
            PreparedVision::new(self, &request.prompt, source_width, source_height, pixels)?;
        let geometry = RequestGeometry::new(
            request,
            identity,
            pipeline,
            self.spatial_scale(),
            self.denoiser.transformer(),
            visual.prompt.token_tags().to_vec(),
            Some(reference),
        )?;
        self.conditioning_recipe(PreparedPrompts::Image(visual), geometry, encoder, connector)
    }
    fn conditioning_recipe(
        &self,
        prompts: PreparedPrompts,
        geometry: RequestGeometry,
        prepare_encoder: impl FnOnce(
            PH3ImageTextConditioningFamily,
            &PH3ImageTextConditioningConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
        prepare_connector: impl FnOnce(
            PH3ImageTextConnectorProjectionFamily,
            &PH3ImageTextConnectorProjectionConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
    ) -> Result<PreparedConditioning> {
        let mut config = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
            self.text.clone(),
            prompts.text().layout().clone(),
            &self.sources.text_encoder,
            &self.connector,
        )?;
        if let Some(visual) = prompts.visual() {
            config = config.with_visual_inputs(visual.prompt.visual_rows().clone())?;
        }
        let encoder = prepare_checked(
            PH3ImageTextConditioningFamily::new(self.limits.text)?,
            &config,
            &NumericalProfileId::new(PH3ImageTextConditioningFamily::cuda_profile_id(&config))
                .map_err(anyhow::Error::msg)?,
            prepare_encoder,
        )?;
        let encoder_options = config.compile_options()?;
        let projection = config.projection_config()?;
        let connector = prepare_checked(
            PH3ImageTextConnectorProjectionFamily::new(self.limits.connector)?,
            &projection,
            &NumericalProfileId::new(PH3ImageTextConnectorProjectionFamily::CUDA_PROFILE_ID)
                .map_err(anyhow::Error::msg)?,
            prepare_connector,
        )?;
        let connector_options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            ProgramValueId::new(PH3ImageTextConnectorProjectionFamily::INPUT_ID)?,
            projection.input_spec(config.encoder().text().layout().rows())?,
        )]))?;
        Ok(PreparedConditioning {
            prompts,
            config,
            geometry,
            encoder,
            encoder_options,
            connector,
            connector_options,
        })
    }
}

fn prepare_uncached<P: ModelFamilyProvider>(
    provider: P,
    config: &P::Config,
    profile: &NumericalProfileId,
) -> Result<PreparedModelFamily> {
    Ok(TypedFamilyRegistration::new(provider)
        .prepare_with_profile(&serde_json::to_value(config)?, profile)?)
}

fn prepare_checked<P: ModelFamilyProvider>(
    provider: P,
    config: &P::Config,
    profile: &NumericalProfileId,
    prepare: impl FnOnce(P, &P::Config, &NumericalProfileId) -> Result<PreparedModelFamily>,
) -> Result<PreparedModelFamily> {
    let family_id = provider.family_id().clone();
    let metadata_ids = provider.external_metadata_ids();
    let family = prepare(provider, config, profile)?;
    ensure!(
        family.family_id() == &family_id
            && metadata_ids.contains(family.external_metadata_id())
            && family.canonical_config() == &serde_json::to_value(config)?
            && &family.numerical_profile().id == profile,
        "conditioning family differs from its original geometry/profile/source domain"
    );
    Ok(family)
}
