use super::*;
use ferrum_models::qwen3_vl_vision::{
    Qwen3VLVisionArchitecture, Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid,
};
use ferrum_models::qwen_image21::image_input::QwenImage21ImageProcessor;
use ferrum_models::qwen_image21::{QwenImage21VaeEncoderConfig, QwenImage21VaeEncoderFamily};

pub(crate) struct EditAssets {
    pub vision: Qwen3VLVisionArchitecture,
    pub encoder: QwenImage21VaeEncoderConfig,
    pub processor: QwenImage21ImageProcessor,
    pub required_weight_bytes: [u64; 2],
}
impl QwenImage21Assets {
    /// Explicit capability preparation. Validates both additional components
    /// against the same mapped checkpoint before constructing any device state.
    pub fn with_image_editing(
        mut self,
        text_config: &[u8],
        vae_config: &[u8],
        processor_config: &[u8],
    ) -> anyhow::Result<Self> {
        ensure!(
            text_config.len() <= 4 * 1024 * 1024
                && vae_config.len() <= 4 * 1024 * 1024
                && processor_config.len() <= 4 * 1024 * 1024,
            "editing metadata exceeds bound"
        );
        let vision =
            Qwen3VLVisionArchitecture::from_hf_value(&serde_json::from_slice(text_config)?)?;
        let processor = QwenImage21ImageProcessor::from_json(processor_config)?;
        let config =
            Qwen3VLVisionConfig::new(vision.clone(), vec![Qwen3VLVisionGrid::new(1, 2, 2)?])?;
        ensure!(
            vision.output_size() == self.text.hidden_size()
                && vision.merge_size() == processor.merge_size() as u64
                && vision.patch_size() == processor.patch_size() as u64
                && vision.temporal_patch_size() == processor.temporal_patch_size() as u64,
            "vision, processor and language architecture differ"
        );
        let encoder = QwenImage21VaeEncoderConfig::from_hf_json(vae_config, 1, 16, 16)?;
        ensure!(
            encoder.latent_channels() == 64
                && encoder.spatial_scale() == u64::from(self.spatial_scale()?),
            "encoder and decoder latent contracts differ"
        );
        let vision_schema = Qwen3VLVisionFamily::new()?.weights(&config)?;
        let encoder_schema = QwenImage21VaeEncoderFamily::new()?.weights(&encoder)?;
        let required_weight_bytes = [
            validate_archive(
                vision_schema.stored_schema()?,
                self.sources.text_encoder.as_ref(),
            )?,
            validate_archive(encoder_schema.stored_schema()?, self.sources.vae.as_ref())?,
        ];
        let positions = self
            .sources
            .text_encoder
            .tensor(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)?;
        ensure!(
            positions.element_type() == Some(ElementType::Bf16)
                && positions.shape() == [vision.position_embeddings(), vision.hidden_size()],
            "vision position table differs from architecture"
        );
        for (name, bytes) in [
            ("editing/text_encoder", text_config),
            ("editing/vae", vae_config),
            ("editing/processor", processor_config),
        ] {
            self.receipt
                .metadata_sha256
                .insert(name.into(), format!("{:x}", Sha256::digest(bytes)));
        }
        self.editing = Some(EditAssets {
            vision,
            encoder,
            processor,
            required_weight_bytes,
        });
        Ok(self)
    }

    pub fn editing_weight_bytes(&self) -> Option<[u64; 2]> {
        self.editing.as_ref().map(|e| e.required_weight_bytes)
    }
}
impl EditAssets {
    pub fn encoder_config(
        &self,
        height: u32,
        width: u32,
    ) -> anyhow::Result<QwenImage21VaeEncoderConfig> {
        let e = &self.encoder;
        Ok(QwenImage21VaeEncoderConfig::new(
            e.latent_channels(),
            e.base_channels(),
            e.dim_mult().to_vec(),
            e.num_res_blocks(),
            e.temporal_downsample().to_vec(),
            1,
            u64::from(height),
            u64::from(width),
        )?)
    }
}
