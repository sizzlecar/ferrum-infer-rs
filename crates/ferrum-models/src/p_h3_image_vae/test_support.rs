//! Complete original encoder/decoder source shared with full pipeline tests.
use super::*;
use super::{
    decoder::{test_fixture as decoder_fixture, PH3ImageVaeDecoderExecutionConfig},
    decoder_block::PH3ImageVaeDecoderBlockExecutionPreset,
    encoder::{test_fixture as encoder_fixture, PH3ImageVaeEncoderConfig},
    encoder_vertical::PH3ImageVaeEncoderArchitecture,
};
use anyhow::Result;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use safetensors::Dtype;
use std::sync::Arc;
pub(super) fn limits() -> PH3ImageVaeAdapterLimits {
    PH3ImageVaeAdapterLimits {
        max_metadata_bytes: 64 << 20,
        max_stored_bytes: 256 << 20,
        max_plan_nodes: 8192,
    }
}
pub(super) fn metadata() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "_class_name":"AutoencoderKLPH3Image","in_channels":3,"latent_channels":24,"out_channels":3,
        "block_out_channels":[4,8,8,16,16,32],"layers_per_block":2,
        "spatial_downsample_factors":[2,2,2,2,1,1],"temporal_downsample_factors":[1,2,2,1,1,1],
        "norm_num_groups":2,"norm_eps":0.000001,"spatial_padding_mode":"reflect",
        "decoder_num_layers":2,"decoder_num_attention_heads":4,"decoder_attention_head_dim":64,
        "decoder_num_register_tokens":4,"decoder_ffn_mult":4,"decoder_rope_dim_ratio":0.75,
        "decoder_rope_theta":100,"decoder_norm_eps":0.00001
    }))
    .unwrap()
}
/// Synthetic original-format CNN encoder plus INT8 transformer decoder.
/// The retained TempDir and source Arc stay with this fixture's owner.
pub struct TinyPH3ImageVaeFixture {
    _directory: tempfile::TempDir,
    source: Arc<ComfyCanonicalWeightSource>,
    metadata: Vec<u8>,
    encoder_bytes: u64,
    decoder_bytes: u64,
}
impl TinyPH3ImageVaeFixture {
    pub fn new() -> Result<Self> {
        let metadata = metadata();
        let ec = PH3ImageVaeEncoderConfig::new(
            PH3ImageVaeEncoderArchitecture::from_hf_json(&metadata).unwrap(),
            1,
            32,
            32,
        )
        .unwrap();
        let dc = PH3ImageVaeDecoderExecutionConfig::new(
            PH3ImageVaeDecoderConfig::from_hf_json(&metadata).unwrap(),
            1,
            2,
            2,
            PH3ImageVaeDecoderBlockExecutionPreset::F32A8FullAttention,
        )
        .unwrap();
        let mut tensors = encoder_fixture::originals(&ec);
        // An independent physical-byte inventory from original source tensors:
        // every CNN temporal kernel contributes one last-time tap, not all KT.
        let encoder_bytes = tensors
            .values()
            .map(|(_, shape, bytes)| {
                if shape.len() == 5 {
                    (bytes.len() / shape[2]) as u64
                } else {
                    bytes.len() as u64
                }
            })
            .sum();
        let decoder = decoder_fixture::originals(&dc);
        let decoder_bytes = decoder
            .values()
            .map(|(dtype, _, bytes)| match dtype {
                Dtype::BF16 => (bytes.len() * 2) as u64,
                Dtype::I8 | Dtype::F32 => bytes.len() as u64,
                Dtype::U8 => 0, // .comfy_quant metadata is checked, not a device weight.
                other => panic!("unexpected released fixture storage {other:?}"),
            })
            .sum();
        tensors.extend(decoder);
        for (name, value) in [("latents_mean", 0.125_f32), ("latents_std", 0.875_f32)] {
            tensors.insert(
                name.into(),
                (
                    Dtype::F32,
                    vec![24],
                    (0..24)
                        .flat_map(|i| (value + i as f32 / 64. + 0.000123).to_le_bytes())
                        .collect(),
                ),
            );
        }
        let (directory, source) = decoder_fixture::archive(&tensors);

        let source = Arc::new(source);
        PH3ImageVaeAdapter::new(source.clone(), &metadata, limits())?;
        Ok(Self {
            _directory: directory,
            source,
            metadata,
            encoder_bytes,
            decoder_bytes,
        })
    }
    pub fn source(&self) -> Arc<ComfyCanonicalWeightSource> {
        self.source.clone()
    }
    pub fn metadata(&self) -> &[u8] {
        &self.metadata
    }
    pub fn limits(&self) -> PH3ImageVaeAdapterLimits {
        limits()
    }
    pub fn encoder_stored_bytes(&self) -> u64 {
        self.encoder_bytes
    }
    pub fn decoder_stored_bytes(&self) -> u64 {
        self.decoder_bytes
    }
    #[cfg(test)]
    pub(super) fn into_parts(
        self,
    ) -> (tempfile::TempDir, Arc<ComfyCanonicalWeightSource>, u64, u64) {
        (
            self._directory,
            self.source,
            self.encoder_bytes,
            self.decoder_bytes,
        )
    }
}
