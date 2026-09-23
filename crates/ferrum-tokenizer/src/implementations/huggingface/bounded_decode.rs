//! Allocation-free equivalent of tokenizers 0.21.4's direct ByteLevel decoder.
//!
//! HF first looks up added/model tokens, filters its added-vocabulary special
//! set, maps whole token spellings to bytes, concatenates, then applies UTF-8
//! lossy decoding. Ferrum additionally flushes segments at semantic markers.

use super::{byte_level_char_bytes, DecoderWrapper, HfTokenizer, HuggingFaceTokenizer};
use ferrum_interfaces::tokenizer::{BoundedDecodeBound, BoundedDecodeError};
use ferrum_types::TokenId;
use std::{collections::HashMap, num::NonZeroUsize};

pub(super) struct ByteLevelBoundedDecoder {
    bound: BoundedDecodeBound,
    // Initialize the shared alphabet during construction, never during decode.
    alphabet: &'static HashMap<char, u8>,
}

impl ByteLevelBoundedDecoder {
    pub(super) fn new(
        tokenizer: &HfTokenizer,
        vocabulary: &[Option<String>],
        semantic_markers: &[(u32, &'static str)],
    ) -> Option<Self> {
        if !matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_))) {
            return None;
        }

        // A reverse table made from vocabulary spellings can be ambiguous for
        // malformed/aliased IDs. Prove the borrowed table has exactly HF's
        // added-first lookup semantics, including holes and every known ID.
        for (id, token) in vocabulary.iter().enumerate() {
            if tokenizer.id_to_token(u32::try_from(id).ok()?).as_deref() != token.as_deref() {
                return None;
            }
        }
        let model_vocabulary = tokenizer.get_vocab(false);
        if model_vocabulary
            .values()
            .any(|id| *id as usize >= vocabulary.len())
            || tokenizer
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .keys()
                .any(|id| *id as usize >= vocabulary.len())
        {
            return None;
        }

        let alphabet = byte_level_char_bytes();
        let mut max_raw_bytes = 0;
        for token in vocabulary.iter().flatten() {
            // HF falls back for the WHOLE token if any character is unmapped.
            let bytes = if token
                .chars()
                .all(|character| alphabet.contains_key(&character))
            {
                token.chars().count()
            } else {
                token.len()
            };
            max_raw_bytes = max_raw_bytes.max(bytes);
        }
        // Every invalid raw byte expands to at most one three-byte replacement.
        // Marker tokens replace their raw surface and flush a segment; the sum
        // of these per-token maxima therefore bounds the complete output.
        let mut max_text_bytes = max_raw_bytes.checked_mul(3)?;
        for (_, canonical) in semantic_markers {
            max_text_bytes = max_text_bytes.max(canonical.len());
        }
        Some(Self {
            bound: BoundedDecodeBound::new(NonZeroUsize::new(max_text_bytes)?, max_raw_bytes),
            alphabet,
        })
    }

    pub(super) fn bound(&self) -> BoundedDecodeBound {
        self.bound
    }

    pub(super) fn token_bytes_bound(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.bound.max_scratch_bytes_per_token())
    }
}

/// Raw context-free vocabulary surface, before lossy UTF-8 decoding, special
/// filtering, or semantic marker substitutions. Equivalent to the existing
/// direct ByteLevel `token_bytes` path, with no intermediate Vec allocation.
pub(super) fn token_bytes_into(
    tokenizer: &HuggingFaceTokenizer,
    id: TokenId,
    output: &mut [u8],
) -> Result<Option<usize>, BoundedDecodeError> {
    let decoder = tokenizer
        .bounded_decoder
        .as_ref()
        .ok_or(BoundedDecodeError::Unsupported)?;
    let Some(token) = tokenizer
        .id_to_token
        .get(id.get() as usize)
        .and_then(Option::as_deref)
    else {
        return Ok(None);
    };
    let mapped = token
        .chars()
        .all(|character| decoder.alphabet.contains_key(&character));
    let required = if mapped {
        token.chars().count()
    } else {
        token.len()
    };
    if output.len() < required {
        return Err(BoundedDecodeError::InsufficientOutput {
            required,
            available: output.len(),
        });
    }
    if mapped {
        for (slot, character) in output.iter_mut().zip(token.chars()) {
            *slot = decoder.alphabet[&character];
        }
    } else {
        // If any character is unmapped, HF and the existing raw API preserve
        // the entire UTF-8 spelling, including otherwise mapped characters.
        output[..required].copy_from_slice(token.as_bytes());
    }
    Ok(Some(required))
}

pub(super) fn decode_into(
    tokenizer: &HuggingFaceTokenizer,
    tokens: &[TokenId],
    skip_special: bool,
    scratch: &mut [u8],
    output: &mut String,
) -> Result<(), BoundedDecodeError> {
    let decoder = tokenizer
        .bounded_decoder
        .as_ref()
        .ok_or(BoundedDecodeError::Unsupported)?;
    let required = decoder.bound.requirements(tokens.len())?;
    if scratch.len() < required.scratch_bytes {
        return Err(BoundedDecodeError::InsufficientScratch {
            required: required.scratch_bytes,
            available: scratch.len(),
        });
    }
    if output.capacity() < required.text_bytes {
        return Err(BoundedDecodeError::InsufficientOutput {
            required: required.text_bytes,
            available: output.capacity(),
        });
    }

    // No fallible operation or allocation follows preflight. At most n tokens
    // each add max_raw_bytes to scratch and max_text_bytes to output; both
    // products and their simultaneous sum were checked before either write.
    output.clear();
    let mut raw_len = 0;
    let added = tokenizer.tokenizer.get_added_vocabulary();
    for id in tokens {
        if skip_special {
            if let Some((_, canonical)) = tokenizer
                .semantic_markers
                .iter()
                .find(|(marker, _)| *marker == id.get())
            {
                append_lossy(&scratch[..raw_len], output);
                raw_len = 0;
                output.push_str(canonical);
                continue;
            }
        }
        let Some(token) = tokenizer
            .id_to_token
            .get(id.get() as usize)
            .and_then(Option::as_deref)
        else {
            // HF filters unknown IDs; it does not substitute the UNK spelling.
            continue;
        };
        if skip_special && added.is_special_token(token) {
            continue;
        }
        if token
            .chars()
            .all(|character| decoder.alphabet.contains_key(&character))
        {
            for character in token.chars() {
                scratch[raw_len] = decoder.alphabet[&character];
                raw_len += 1;
            }
        } else {
            let end = raw_len + token.len();
            scratch[raw_len..end].copy_from_slice(token.as_bytes());
            raw_len = end;
        }
    }
    append_lossy(&scratch[..raw_len], output);
    Ok(())
}

/// Same invalid-sequence boundaries as `String::from_utf8_lossy`, writing
/// directly into the already authorized output capacity, without a Cow/String.
fn append_lossy(mut bytes: &[u8], output: &mut String) {
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                output.push_str(valid);
                break;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                output.push_str(std::str::from_utf8(&bytes[..valid]).expect("validated prefix"));
                output.push('\u{fffd}');
                match error.error_len() {
                    Some(invalid) => bytes = &bytes[valid + invalid..],
                    None => break, // One replacement for the incomplete suffix.
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
