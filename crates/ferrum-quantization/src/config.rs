//! Quantization configuration parsed from model metadata.
//!
//! Populated by `WeightLoader` implementations from sources like
//! `quantize_config.json` (GPTQ/AWQ) or a GGUF header.

use serde::{Deserialize, Serialize};

/// The quantization scheme in use, if any.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum QuantMethod {
    /// No quantization — dense fp32/fp16/bf16 weights.
    #[default]
    None,
    /// GPTQ: int4/int8 group-wise with scales + zeros, asymmetric.
    Gptq,
    /// AWQ: int4 group-wise, similar to GPTQ but different packing.
    Awq,
    /// GGUF: k-quants and legacy quants embedded in a single-file format.
    Gguf,
}

/// Combined quantization config.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct QuantConfig {
    // AutoGPTQ writes `quant_method`; some older / hand-written packs use
    // `method`. Accept both. Default to None if absent (which would have
    // been an error before — but in practice every real quant pack has
    // one of the two).
    #[serde(rename = "quant_method", alias = "method", default)]
    pub method: QuantMethod,
    /// Bit-width (typically 4 or 8 for GPTQ/AWQ).
    #[serde(default)]
    pub bits: u32,
    /// Group size for group-wise scales (typically 128).
    #[serde(default)]
    pub group_size: usize,
    /// Whether to use descending activation order (GPTQ only).
    #[serde(default)]
    pub desc_act: bool,
    /// Whether scales use symmetric quantization.
    #[serde(default)]
    pub sym: bool,
}
