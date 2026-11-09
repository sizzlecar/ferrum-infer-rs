//! Registration declarations from the pinned modular model index. These are
//! data to validate, never instructions to import code or resolve a remote path.
//! The composition root supplies every metadata file and mapped weight source.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct SourceReference {
    type_hint: [String; 2],
    pretrained_model_name_or_path: String,
    subfolder: String,
    variant: Option<String>,
    revision: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Declaration(String, String, SourceReference);
impl Declaration {
    fn validate(&self, slot: &str, library: &str, class: &str) -> Result<()> {
        ensure!(
            self.0 == library
                && self.1 == class
                && self.2.type_hint[0] == library
                && self.2.type_hint[1] == class,
            "{slot} registration differs from its declared native component contract"
        );
        ensure!(
            self.2.subfolder == slot
                && !self.2.pretrained_model_name_or_path.is_empty()
                && self.2.variant.is_none()
                && self
                    .2
                    .revision
                    .as_ref()
                    .is_none_or(|revision| !revision.is_empty()),
            "{slot} source registration has an unsupported folder, variant or empty identity"
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct ComponentDeclarations {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_blocks_class_name")]
    blocks_class: String,
    pub reference_min_pixels: u64,
    pub reference_max_pixels: u64,
    pub canvas_min_pixels: u64,
    pub vision_pixels: u64,
    text_encoder: Declaration,
    text_connector: Declaration,
    tokenizer: Declaration,
    processor: Declaration,
    transformer: Declaration,
    vae: Declaration,
    scheduler: Declaration,
    audio_scheduler: Declaration,
    prompt_upsampler: Option<Declaration>,
    prompt_upsampler_processor: Option<Declaration>,
}
impl ComponentDeclarations {
    pub fn parse(bytes: &[u8], max_bytes: usize) -> Result<Self> {
        ensure!(
            max_bytes > 0 && !bytes.is_empty() && bytes.len() <= max_bytes,
            "modular model index exceeds its nonempty metadata capacity"
        );
        // Parse original bytes, preserving duplicate-field rejection.
        let result: Self = serde_json::from_slice(bytes)?;
        ensure!(
            result.class == "PH3ImageModularPipeline" && result.blocks_class == "PH3ImageBlocks",
            "model index does not declare the supported image component graph"
        );
        for (slot, declaration, library, class) in [
            (
                "text_encoder",
                &result.text_encoder,
                "transformers",
                "Qwen3VLForConditionalGeneration",
            ),
            (
                "text_connector",
                &result.text_connector,
                "diffusers",
                "PH3ImageTextConnector",
            ),
            (
                "tokenizer",
                &result.tokenizer,
                "transformers",
                "Qwen2TokenizerFast",
            ),
            (
                "processor",
                &result.processor,
                "transformers",
                "Qwen3VLProcessor",
            ),
            (
                "transformer",
                &result.transformer,
                "diffusers",
                "MiniMaxH3Transformer3DModel",
            ),
            ("vae", &result.vae, "diffusers", "AutoencoderKLPH3Image"),
            (
                "scheduler",
                &result.scheduler,
                "diffusers",
                "MiniMaxH3Scheduler",
            ),
            (
                "audio_scheduler",
                &result.audio_scheduler,
                "diffusers",
                "MiniMaxH3Scheduler",
            ),
        ] {
            declaration.validate(slot, library, class)?;
        }
        match (&result.prompt_upsampler, &result.prompt_upsampler_processor) {
            (Some(model), Some(processor)) => {
                model.validate(
                    "prompt_upsampler",
                    "transformers",
                    "Qwen3_5ForConditionalGeneration",
                )?;
                processor.validate(
                    "prompt_upsampler_processor",
                    "transformers",
                    "AutoProcessor",
                )?;
            }
            (None, None) => (),
            _ => anyhow::bail!("prompt expansion requires both declared model and processor"),
        }
        ensure!(
            result.reference_min_pixels > 0
                && result.reference_min_pixels <= result.reference_max_pixels
                && result.reference_max_pixels <= isize::MAX as u64 / 12
                && result.canvas_min_pixels > 0
                && result.canvas_min_pixels <= isize::MAX as u64 / 12
                && result.vision_pixels > 0
                && result.vision_pixels <= isize::MAX as u64 / 12,
            "model image-area declarations must be ordered, positive and addressable as RGB F32"
        );
        Ok(result)
    }
    /// Declaration availability does not claim that a prompt model is loaded.
    pub fn declares_prompt_expansion(&self) -> bool {
        self.prompt_upsampler.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn declaration(slot: &str, library: &str, class: &str) -> Declaration {
        Declaration(
            library.into(),
            class.into(),
            SourceReference {
                type_hint: [library.into(), class.into()],
                pretrained_model_name_or_path: "caller-resolved-model".into(),
                subfolder: slot.into(),
                variant: None,
                revision: None,
            },
        )
    }
    fn fixture() -> ComponentDeclarations {
        ComponentDeclarations {
            class: "PH3ImageModularPipeline".into(),
            blocks_class: "PH3ImageBlocks".into(),
            reference_min_pixels: 1 << 20,
            reference_max_pixels: 8 << 20,
            canvas_min_pixels: 4 << 20,
            vision_pixels: 1_000_000,
            text_encoder: declaration(
                "text_encoder",
                "transformers",
                "Qwen3VLForConditionalGeneration",
            ),
            text_connector: declaration("text_connector", "diffusers", "PH3ImageTextConnector"),
            tokenizer: declaration("tokenizer", "transformers", "Qwen2TokenizerFast"),
            processor: declaration("processor", "transformers", "Qwen3VLProcessor"),
            transformer: declaration("transformer", "diffusers", "MiniMaxH3Transformer3DModel"),
            vae: declaration("vae", "diffusers", "AutoencoderKLPH3Image"),
            scheduler: declaration("scheduler", "diffusers", "MiniMaxH3Scheduler"),
            audio_scheduler: declaration("audio_scheduler", "diffusers", "MiniMaxH3Scheduler"),
            prompt_upsampler: Some(declaration(
                "prompt_upsampler",
                "transformers",
                "Qwen3_5ForConditionalGeneration",
            )),
            prompt_upsampler_processor: Some(declaration(
                "prompt_upsampler_processor",
                "transformers",
                "AutoProcessor",
            )),
        }
    }
    fn parse(value: &ComponentDeclarations) -> Result<ComponentDeclarations> {
        ComponentDeclarations::parse(&serde_json::to_vec(value)?, 4096)
    }
    #[test]
    fn source_registration_is_checked_as_data_and_bounded_before_deserialization() {
        let source = fixture();
        let bytes = serde_json::to_vec(&source).unwrap();
        let parsed = ComponentDeclarations::parse(&bytes, bytes.len()).unwrap();
        assert!(parsed.declares_prompt_expansion());
        assert_eq!(
            (parsed.reference_min_pixels, parsed.reference_max_pixels),
            (1 << 20, 8 << 20)
        );
        assert!(ComponentDeclarations::parse(&bytes, bytes.len() - 1).is_err());
        let duplicate = format!(
            "{{\"canvas_min_pixels\":1,{}",
            std::str::from_utf8(&bytes).unwrap().trim_start_matches('{')
        );
        assert!(ComponentDeclarations::parse(duplicate.as_bytes(), 4096).is_err());
        let mut source = source;
        source.text_encoder.2.type_hint[1] = "DifferentEncoder".into();
        assert!(parse(&source).is_err());
        source = fixture();
        source.vae.2.subfolder = "../other".into();
        assert!(parse(&source).is_err());
        source = fixture();
        source.transformer.2.variant = Some("different_precision".into());
        assert!(parse(&source).is_err());
    }
    #[test]
    fn image_area_and_optional_prompt_component_pair_are_validated() {
        let mut source = fixture();
        source.reference_max_pixels = source.reference_min_pixels - 1;
        assert!(parse(&source).is_err());
        source = fixture();
        source.vision_pixels = 0;
        assert!(parse(&source).is_err());
        source = fixture();
        source.prompt_upsampler = None;
        assert!(parse(&source).is_err());
        source.prompt_upsampler_processor = None;
        assert!(!parse(&source).unwrap().declares_prompt_expansion());
    }
}
