//! Checked P-H3 image VAE decoder architecture and released Comfy weight map.
//!
//! This module describes source tensors; it does not register an executable
//! family or claim backend support. P-H3 uses a ViT decoder, unlike the Qwen
//! convolutional VAE. Latent denormalization, spatial tiling, mixed precision
//! execution and pixel processing remain separate pipeline responsibilities.

mod config;
mod decode_tiles;
mod decoder;
mod decoder_block;
mod encode_layout;
mod encode_tiles;
mod encoder;
mod encoder_vertical;
mod image_layout;
mod latent_statistics;
mod pipeline_adapter;
#[cfg(any(test, feature = "test-support"))]
pub(crate) mod test_support;
mod tiling;
mod weights;

pub use config::PH3ImageVaeDecoderConfig;
pub use pipeline_adapter::{
    PH3ImageVaeAdapter, PH3ImageVaeAdapterLimits, PH3ImageVaeDecoderConsumer,
    PH3ImageVaeDecoderHostPhase, PH3ImageVaeDecoderHostPhaseBudget,
    PH3ImageVaeDecoderHostRequirements, PH3ImageVaeDecoderPreset, PH3ImageVaeDecoderTilesConsumer,
    PH3ImageVaeEncoderConsumer, PH3ImageVaeEncoderHostPhase, PH3ImageVaeEncoderHostPhaseBudget,
    PH3ImageVaeEncoderHostRequirements, PH3ImageVaeEncoderPreset, PH3ImageVaePipelineError,
    PH3ImageVaePixelDomain, PH3ImageVaeSpatialPolicy, PH3ImageVaeTileBatch, PH3ImageVaeTileGroup,
    PH3ImageVaeWeightSource, PreparedPH3ImageVaeDecoder, PreparedPH3ImageVaeDecoderTiles,
    PreparedPH3ImageVaeEncoder,
};
pub use weights::{PH3ImageVaeDecoderWeight, PH3ImageVaeWeightEncoding};

pub const PH3_IMAGE_VAE_CLASS: &str = "AutoencoderKLPH3Image";

fn invalid(field: &str, reason: impl Into<String>) -> ferrum_runtime::VNextError {
    ferrum_runtime::VNextError::InvalidModelConfig {
        family_id: "family.p_h3_image.vae.decoder".into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
