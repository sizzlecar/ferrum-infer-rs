//! Request facts read from the product's already-tokenized sequence.
use super::*;

/// The original input, including tokens inserted by the installed tokenizer.
/// Recompute/continuation work does not replace this digest with generated text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CalibrationRequestEvidence {
    pub original_input_tokens: usize,
    /// SHA256 of `ferrum-token-ids:u32-le-v1\0` followed by ordered u32-LE ids.
    pub original_input_tokens_sha256: [u8; 32],
}

impl CalibrationRequestEvidence {
    pub(super) fn capture(sequence: &SequenceState) -> Self {
        Self {
            original_input_tokens: sequence.input_tokens.len(),
            original_input_tokens_sha256: crate::continuous_engine::token_ids_digest(
                &sequence.input_tokens,
            )
            .into(),
        }
    }
}
