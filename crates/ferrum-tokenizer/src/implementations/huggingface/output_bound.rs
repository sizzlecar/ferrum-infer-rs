use super::{byte_level_token_bytes, DecodedTextBound, DecoderWrapper, HfTokenizer};
use std::num::NonZeroUsize;

pub(super) fn byte_level_decoded_text_bound(
    tokenizer: &HfTokenizer,
    vocabulary: &[Option<String>],
    semantic_markers: &[(u32, &'static str)],
) -> Option<DecodedTextBound> {
    // A Sequence containing ByteLevel may also contain Replace or other
    // context-dependent expansion. Merely finding ByteLevel is not a proof.
    if !matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_))) {
        return None;
    }
    // ByteLevel maps each token to raw bytes (or its original UTF-8 spelling if
    // any character lacks a byte-alphabet mapping), concatenates them, then
    // uses from_utf8_lossy. Each invalid input byte expands to at most the
    // three-byte replacement character. Include every added vocabulary token.
    // Splitting at semantic markers only splits this same bounded byte stream;
    // each replacement marker is explicitly included in the per-token bound.
    let mut maximum = 0;
    for token in vocabulary.iter().flatten() {
        maximum = maximum.max(byte_level_token_bytes(token).len().checked_mul(3)?);
    }
    for (_, canonical) in semantic_markers {
        maximum = maximum.max(canonical.len());
    }
    NonZeroUsize::new(maximum).map(DecodedTextBound::new)
}
