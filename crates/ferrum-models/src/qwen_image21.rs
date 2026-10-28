//! Qwen-Image 2.1 Transformer metadata, typed partial families and T2I graph.
//!
//! Detached safetensors headers establish declared structure, not payload
//! authenticity. The full transformer consumes explicit text/latent inputs;
//! text encoding, the denoising loop and VAE decoding remain separate stages.
//! Weight payloads use the existing runtime weight-source interface.

mod attention;
mod block0;
mod block0_mlp;
mod conditioning;
mod config;
mod headers;
pub mod image_input;
mod image_projection;
pub mod pipeline_adapters;
mod qk_norm;
pub mod sampling_schedule;
pub mod t2i_prompt;
mod transformer;
mod vae_decoder;
mod vae_encoder;
mod vae_stem;
mod weights;

pub use attention::{
    QwenImage21AttentionConfig, QwenImage21AttentionKeyRun, QwenImage21AttentionLayout,
    QwenImage21AttentionSegment, QwenImage21Block0AttentionFamily,
};
pub use block0::{QwenImage21Block0Broadcast, QwenImage21Block0Config, QwenImage21Block0Family};
pub use block0_mlp::QwenImage21Block0MlpFamily;
pub use config::QwenImage21Config;
pub use headers::{
    HeaderEvidence, PayloadVerification, QwenImage21Structure, ShardStructure, TensorStructure,
};
pub use image_projection::{
    QwenImage21ImageProjectionFamily, IMAGE_PROJECTION_FAMILY_ID, IMAGE_PROJECTION_INPUT_ID,
    IMAGE_PROJECTION_NODE_ID, IMAGE_PROJECTION_OUTPUT_ID, IMAGE_PROJECTION_PROFILE_ID,
    IMAGE_PROJECTION_WEIGHT_ID,
};
pub use qk_norm::{QwenImage21Block0QkNormFamily, QwenImage21QkFrontend};
pub use transformer::{
    QwenImage21ModelTimestep, QwenImage21PrefixCachePort, QwenImage21PrefixPart,
    QwenImage21ReferenceImage, QwenImage21RotaryFrequencies, QwenImage21T2iLayout,
    QwenImage21Timestep256, QwenImage21TransformerConfig, QwenImage21TransformerFamily,
    QwenImage21TransformerPhase,
};
pub use vae_decoder::{
    QwenImage21VaeDecoderConfig, QwenImage21VaeDecoderFamily, QwenImage21VaeDecoderWeight,
};
pub use vae_encoder::{
    QwenImage21VaeEncoderConfig, QwenImage21VaeEncoderFamily, QwenImage21VaeEncoderWeight,
};
pub use vae_stem::{QwenImage21VaeStemConfig, QwenImage21VaeStemFamily, QwenImage21VaeStemWeight};
pub use weights::{ExpectedTensor, ExpectedWeights, LoraTarget, LoraTargetSlice};

use ferrum_types::{FerrumError, Result};

pub const TRANSFORMER_CLASS: &str = "QwenImage21Transformer2DModel";
pub const FAMILY_ID: &str = "family.qwen_image21.transformer";

/// Metadata parsing and inventory limits, not a bound on process RSS or future
/// device allocations. Parser scratch and container overhead are additional.
#[derive(Debug, Clone, Copy)]
pub struct StructureLimits {
    metadata_file_bytes: u64,
    total_metadata_bytes: u64,
    tensors: usize,
    shards: usize,
}

impl StructureLimits {
    pub fn new(
        metadata_file_bytes: u64,
        total_metadata_bytes: u64,
        tensors: usize,
        shards: usize,
    ) -> Result<Self> {
        if metadata_file_bytes == 0
            || total_metadata_bytes == 0
            || metadata_file_bytes >= isize::MAX as u64
            || total_metadata_bytes >= isize::MAX as u64
            || tensors == 0
            || shards == 0
        {
            return Err(invalid(
                "structure limits must be positive and leave room for a host read sentinel",
            ));
        }
        Ok(Self {
            metadata_file_bytes,
            total_metadata_bytes,
            tensors,
            shards,
        })
    }
    pub const fn metadata_file_bytes(self) -> u64 {
        self.metadata_file_bytes
    }
    pub const fn total_metadata_bytes(self) -> u64 {
        self.total_metadata_bytes
    }
    pub const fn maximum_tensors(self) -> usize {
        self.tensors
    }
    pub const fn maximum_shards(self) -> usize {
        self.shards
    }

    fn check_metadata(self, label: &str, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() || bytes.len() as u64 > self.metadata_file_bytes {
            return Err(invalid(format!(
                "{label} is empty or exceeds the metadata file limit"
            )));
        }
        Ok(())
    }
}

impl Default for StructureLimits {
    fn default() -> Self {
        Self {
            metadata_file_bytes: 4 * 1024 * 1024,
            total_metadata_bytes: 32 * 1024 * 1024,
            tensors: 65536,
            shards: 64,
        }
    }
}

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::model(message)
}

#[cfg(test)]
mod tests;
