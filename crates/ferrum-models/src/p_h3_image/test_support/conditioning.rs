//! Complete original-format tiny conditioning source, including all four
//! visual exits. The payload builders are shared with the model tests.
use super::*;
use crate::qwen3_vl_text::{Qwen3VLTextArchitecture, Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use crate::qwen3_vl_vision::{comfy::Qwen3VLVisionComfyLimits, Qwen3VLVisionArchitecture};
use anyhow::{ensure, Result};
use half::bf16;

/// Original INT8 embedding, independently selected W4/W8 text projections,
/// F32 merged connector and original BF16 main/deepstack visual weights.
/// These synthetic assets are for execution tests, not image-quality claims.
pub struct TinyPH3ImageConditioningFixture {
    _directory: tempfile::TempDir,
    source: Arc<ComfyCanonicalWeightSource>,
    text_encoder: Vec<u8>,
    connector: Vec<u8>,
    tokenizer: Vec<u8>,
    tokenizer_config: Vec<u8>,
}
impl TinyPH3ImageConditioningFixture {
    pub fn new() -> Result<Self> {
        use super::super::text_conditioning::test_fixture as text;
        use crate::qwen3_vl_vision::comfy::test_fixture as visual;

        let mut metadata = text::metadata();
        metadata["text_config"]["max_position_embeddings"] = serde_json::json!(512);
        metadata["text_config"]["rope_scaling"]["mrope_section"] = serde_json::json!([24, 20, 20]);
        metadata["vision_config"] = visual::metadata()["vision_config"].clone();
        let text_encoder = serde_json::to_vec(&metadata)?;
        let layout = Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?;
        let seed_config = text::config(layout.clone());
        let mut tensors = text::original(&seed_config);
        let original = visual::Original::new(&visual::config(2, 2));
        for (name, values) in original.values {
            let bytes = values
                .into_iter()
                .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            ensure!(
                tensors
                    .insert(
                        name.clone(),
                        (
                            safetensors::Dtype::BF16,
                            original.shapes[&name].clone(),
                            bytes
                        )
                    )
                    .is_none(),
                "conditioning fixture has a duplicate original tensor {name}"
            );
        }
        let connector = serde_json::to_vec(&serde_json::json!({
            "_class_name":"PH3ImageTextConnector", "hidden_state_indices":[1,2,3,4,5],
            "text_hidden_size":256, "out_channels":4
        }))?;
        let (_directory, source) = text::archive(&tensors);
        let source = Arc::new(source);
        let declared = PH3ImageTextConnectorConfig::from_hf_json(&connector)?;
        let config = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
            Qwen3VLTextArchitecture::from_hf_value(&metadata)?,
            layout,
            &source,
            &declared,
        )?;
        PH3ImageTextWeightAdapter::new(&config, source.clone(), text::limits())?;
        PH3ImageTextConnectorCheckpoint::new_with_layout(
            &declared,
            source.archive(),
            Self::connector_limit(),
            PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
        )?;
        let vision = Qwen3VLVisionArchitecture::from_hf_value(&metadata)?;
        let vconfig = crate::qwen3_vl_vision::comfy::Qwen3VLVisionComfyConfig::new(
            vision,
            crate::qwen3_vl_vision::Qwen3VLVisionGrid::new(1, 2, 2)?,
        )?;
        crate::qwen3_vl_vision::comfy::Qwen3VLVisionComfyWeightAdapter::new(
            &vconfig,
            source.clone(),
            visual::limits(),
        )?;
        let (tokenizer, tokenizer_config) = tokenizer_source()?;
        Ok(Self {
            _directory,
            source,
            text_encoder,
            connector,
            tokenizer,
            tokenizer_config,
        })
    }
    pub fn source(&self) -> Arc<ComfyCanonicalWeightSource> {
        self.source.clone()
    }
    pub fn text_encoder_metadata(&self) -> &[u8] {
        &self.text_encoder
    }
    pub fn connector_metadata(&self) -> &[u8] {
        &self.connector
    }
    pub fn tokenizer_json(&self) -> &[u8] {
        &self.tokenizer
    }
    pub fn tokenizer_config(&self) -> &[u8] {
        &self.tokenizer_config
    }
    pub fn text_limits(&self) -> PH3ImageTextConditioningLimits {
        super::super::text_conditioning::test_fixture::limits()
    }
    pub fn vision_limits(&self) -> Qwen3VLVisionComfyLimits {
        crate::qwen3_vl_vision::comfy::test_fixture::limits()
    }
    pub fn connector_limits(&self) -> PH3ImageTextConnectorProjectionLimits {
        Self::connector_limit()
    }
    fn connector_limit() -> PH3ImageTextConnectorProjectionLimits {
        PH3ImageTextConnectorProjectionLimits {
            max_projection_bytes: 1 << 20,
        }
    }
}

fn tokenizer_source() -> Result<(Vec<u8>, Vec<u8>)> {
    let vocabulary = [
        "<Picture",
        "1>:",
        "cat",
        "after",
        "[UNK]",
        "Edit",
        "a",
        "dog",
        "red",
        "blue",
        "scene",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|endoftext|>",
    ]
    .into_iter()
    .enumerate()
    .map(|(id, token)| (token.to_owned(), id as u32))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| anyhow::anyhow!("fixture tokenizer model: {e}"))?;
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.add_special_tokens(&[
        tokenizers::AddedToken::from("<|endoftext|>", true),
        tokenizers::AddedToken::from("<|vision_start|>", true),
        tokenizers::AddedToken::from("<|vision_end|>", true),
        tokenizers::AddedToken::from("<|image_pad|>", true),
        tokenizers::AddedToken::from("<|video_pad|>", true),
    ]);
    Ok((
        tokenizer
            .to_string(false)
            .map_err(|e| anyhow::anyhow!("fixture tokenizer source: {e}"))?
            .into_bytes(),
        br#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"<|endoftext|>","bos_token":null}"#
            .to_vec(),
    ))
}
