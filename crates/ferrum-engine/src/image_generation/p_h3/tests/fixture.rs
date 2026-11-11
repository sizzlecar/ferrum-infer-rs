//! Complete original-format assets for the production H3 loader. These
//! deterministic small checkpoints grant no inference or capacity evidence.
use super::super::{PH3ImageAssetLimits, PH3ImageAssets, PH3ImageMetadata, PH3ImageSources};
use anyhow::Result;
use ferrum_models::test_support::{
    TinyPH3ImageConditioningFixture, TinyPH3ImageDiTFixture, TinyPH3ImageVaeFixture,
};
use serde_json::json;

pub(in crate::image_generation::p_h3) struct CompleteFixture {
    pub conditioning: TinyPH3ImageConditioningFixture,
    pub denoiser: TinyPH3ImageDiTFixture,
    pub vae: TinyPH3ImageVaeFixture,
    pub connector: Vec<u8>,
    index: Vec<u8>,
    scheduler: Vec<u8>,
}
impl CompleteFixture {
    pub fn new() -> Result<Self> {
        let conditioning = TinyPH3ImageConditioningFixture::new()?;
        let mut connector: serde_json::Value =
            serde_json::from_slice(conditioning.connector_metadata())?;
        // The released connector includes this provenance field. Retain it
        // through the same loader as its mathematical configuration.
        connector["_diffusers_version"] = json!("0.42.0.dev0");
        let mut index = json!({
            "_class_name":"PH3ImageModularPipeline", "_blocks_class_name":"PH3ImageBlocks",
            "reference_min_pixels":1024, "reference_max_pixels":4096,
            "canvas_min_pixels":1024, "vision_pixels":1024,
            "sample_sigmas":[1.0,0.5,0.0]
        });
        for (slot, library, class) in [
            (
                "text_encoder",
                "transformers",
                "Qwen3VLForConditionalGeneration",
            ),
            ("text_connector", "diffusers", "PH3ImageTextConnector"),
            ("tokenizer", "transformers", "Qwen2TokenizerFast"),
            ("processor", "transformers", "Qwen3VLProcessor"),
            ("transformer", "diffusers", "MiniMaxH3Transformer3DModel"),
            ("vae", "diffusers", "AutoencoderKLPH3Image"),
            ("scheduler", "diffusers", "MiniMaxH3Scheduler"),
            ("audio_scheduler", "diffusers", "MiniMaxH3Scheduler"),
        ] {
            index[slot] = json!([library, class, {
                "type_hint":[library,class],
                "pretrained_model_name_or_path":"local-original-format-fixture",
                "subfolder":slot,"variant":null,"revision":null
            }]);
        }
        Ok(Self {
            conditioning,
            denoiser: TinyPH3ImageDiTFixture::with_image_channels(24)?,
            vae: TinyPH3ImageVaeFixture::new()?,
            connector: serde_json::to_vec(&connector)?,
            index: serde_json::to_vec(&index)?,
            scheduler: serde_json::to_vec(&json!({
                "_class_name":"MiniMaxH3Scheduler", "shift":1.0,
                "shift_provided_sigmas":false
            }))?,
        })
    }
    pub fn metadata(&self) -> PH3ImageMetadata<'_> {
        PH3ImageMetadata {
            model_index: &self.index,
            text_encoder: self.conditioning.text_encoder_metadata(),
            text_connector: &self.connector,
            transformer: self.denoiser.transformer_metadata(),
            vae: self.vae.metadata(),
            scheduler: &self.scheduler,
            audio_scheduler: &self.scheduler,
            tokenizer: self.conditioning.tokenizer_json(),
            tokenizer_config: self.conditioning.tokenizer_config(),
        }
    }
    pub fn sources(&self) -> PH3ImageSources {
        PH3ImageSources {
            text_encoder: self.conditioning.source(),
            transformer: self.denoiser.source(),
            vae: self.vae.source(),
        }
    }
    pub fn limits(&self) -> PH3ImageAssetLimits {
        PH3ImageAssetLimits {
            text: self.conditioning.text_limits(),
            vision: self.conditioning.vision_limits(),
            connector: self.conditioning.connector_limits(),
            denoiser: self.denoiser.limits(),
            vae: self.vae.limits(),
            ..PH3ImageAssetLimits::default()
        }
    }
    pub fn assets(&self) -> Result<PH3ImageAssets> {
        PH3ImageAssets::prepare(self.metadata(), self.sources(), self.limits())
    }
    #[cfg(feature = "cuda")]
    pub fn assets_with_cuda_prompt_expansion(
        &self,
        expansion: &super::super::expansion::MergedExpansionFixture,
        options: super::super::PH3ImagePromptExpansionOptions,
    ) -> Result<PH3ImageAssets> {
        // Preserve the same complete original H3 preparation. The unique
        // expansion fixture attaches its frozen BF16 source and actual CUDA
        // HF-vision profile through the public product asset API.
        expansion.attach_cuda(self.assets()?, options)
    }
}
