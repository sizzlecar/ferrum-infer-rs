//! Comfy text and explicit visual conditioning over the existing component owners.
//!
//! Token IDs -> canonical INT8 embedding -> every quantized decoder layer run
//! in one device program. Only the five selected F32 taps are read back, once.
//! The existing charged Host connector adapter then standardizes those taps,
//! executes the reusable Stored F32 projection, and replaces each first token
//! with its sink. The explicit visual profile replaces image slots and adds
//! three DeepStack features before the same five taps. Visual readbacks are
//! copied to genuine Input owners; this is an explicit Host boundary, not a
//! fully device-resident connector or a complete editing pipeline by itself.
//!
//! Source semantics: ComfyUI 56f179f6c50dad0c4c860ae4da427415c8e86415,
//! `comfy/text_encoders/minimax.py` and `qwen3vl.py`: raw unweighted text,
//! no chat template/BOS/EOS, minimum one PAD token, taps 13/19/25/31/36,
//! final tap normalized, embedding orig_dtype BF16 -> output F32, connector
//! output affine `addcmul`. Numerical profiles remain independent staged
//! implementations, not an upstream Torch binary-parity claim.

use anyhow::{ensure, Context, Result};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    PH3ImageConditioningPrecision, PH3ImageF32HiddenState, PH3ImageTextConnectorCheckpoint,
    PH3ImageTextConnectorCheckpointLayout, PH3ImageTextConnectorConfig,
    PH3ImageTextConnectorManagedAdapter, PH3ImageTextConnectorProjectionConfig,
    PH3ImageTextConnectorProjectionFamily, PH3ImageTextConnectorProjectionLimits,
};
use crate::qwen3_vl_text::{
    Qwen3VLTextArchitecture, Qwen3VLTextComfyCheckpoint, Qwen3VLTextComfyConfig,
    Qwen3VLTextComfyFamily, Qwen3VLTextComfyLimits, Qwen3VLTextConfig, Qwen3VLTextEncoderFamily,
    Qwen3VLTextKeyRun, Qwen3VLTextLayout,
};

mod config;
mod controller;
mod family;
mod host_requirements;
#[cfg(any(test, feature = "test-support"))]
pub(in crate::p_h3_image) mod test_fixture;
mod tokenizer;
mod weight_source;
pub use config::{PH3ImageTextConditioningConfig, PH3ImageTextConditioningLimits};
pub use controller::{
    PH3ImageTextConditioning, PH3ImageTextConditioningError, PH3ImageTextConditioningOutput,
    PH3ImageTextConditioningRun,
};
pub use family::{PH3ImageTextConditioningCheckpoint, PH3ImageTextConditioningFamily};
pub use host_requirements::{
    PH3ImageTextConditioningHostPhase, PH3ImageTextConditioningHostPhaseBudget,
    PH3ImageTextConditioningHostRequirements,
};
pub use tokenizer::{
    PH3ImageEditPrompt, PH3ImageTextPromptBatch, PH3ImageTextTokenizer,
    PH3ImageTextTokenizerLimits, PH3_IMAGE_EDIT_PRESENTATION_PROFILE,
    PH3_IMAGE_EDIT_ROTARY_PROFILE,
};
pub use weight_source::{PH3ImageTextWeightAdapter, PH3ImageTextWeightSource};

pub const PH3_IMAGE_TEXT_HIDDEN_STATES: [u64; 5] = [13, 19, 25, 31, 36];
pub const PH3_IMAGE_TEXT_PRESENTATION_PROFILE: &str =
    "p_h3_image.comfy.text.raw_no_bos_no_eos.right_padded_valid_prefix.v1";
pub const PH3_IMAGE_TEXT_ROTARY_PROFILE: &str =
    "p_h3_image.text.mrope.text_positions.f32_libm-0.2.15.v1";

fn invalid(message: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: PH3ImageTextConditioningFamily::FAMILY_ID.into(),
        field: "text_conditioning".into(),
        reason: message.into(),
    }
}

#[cfg(test)]
mod tests;
