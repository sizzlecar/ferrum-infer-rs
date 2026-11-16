//! Borrowed serialized NVFP4 weights, distinct from MXFP4 and Comfy W4A8.
//!
//! Source: FastVideo `866a3cdb` `layers/quantization/nvfp4_config.py`,
//! `models/encoders/minimax_h3_checkpoint_nvfp4.py`, and
//! `models/vaes/minimax_h3_nvfp4_checkpoint.py`. Two low-nibble-first E2M1
//! values occupy each byte; groups of 16 use E4M3FN scales in FlashInfer's
//! 128x4 layout. This module validates source storage and supplies a scalar
//! mathematical reference. It does not quantize activations, implement a GPU
//! operation, or claim runtime or reference-framework numerical parity.

mod geometry;
mod rounding;
mod source;
pub use geometry::{decode_e2m1, decode_e4m3_scale, Nvfp4Geometry};
pub use rounding::{encode_e2m1_rne, encode_e4m3_scale_rne};
pub use source::{
    Nvfp4Matrix, Nvfp4MatrixNames, Nvfp4MatrixSpec, Nvfp4SafetensorsSource, Nvfp4SourceLimits,
};

pub use ferrum_runtime::NVFP4_E2M1_E4M3_128X4_FORMAT_ID;

fn invalid(reason: impl std::fmt::Display) -> ferrum_types::FerrumError {
    ferrum_types::FerrumError::model(format!("NVFP4: {reason}"))
}

#[cfg(test)]
mod tests;
