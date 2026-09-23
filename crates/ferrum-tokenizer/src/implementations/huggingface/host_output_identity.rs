//! Construction-time identity of the real HF wrapper and immutable content.
use super::{HfTokenizer, SpecialTokens};
use serde_json::Value;
use sha2::{Digest, Sha256};

// Changes to HF dependency behavior, wrapper decode/cache behavior, raw-token
// mapping or bounded decoder implementation must revise this version domain.
const IMPLEMENTATION: &[u8] =
    b"ferrum.hf-host-output.v2/tokenizers-0.21.4/legacy-cache-1000/bounded-bytelevel-v1/prepared-completion-v1\0";

pub(super) fn content(
    tokenizer: &HfTokenizer,
    vocabulary: &[Option<String>],
    markers: &[(u32, &'static str)],
    byte_level: bool,
    bounded_byte_level: bool,
) -> Option<[u8; 32]> {
    // Serialization is a cold initialization cost, never an admission or token
    // loop cost. Errors mean Unknown; labels or sizes are not fallback proof.
    let source = tokenizer.to_string(false).ok()?;
    let configuration: Value = serde_json::from_str(&source).ok()?;
    let vocabulary = serde_json::to_value(vocabulary).ok()?;
    let markers = serde_json::to_value(markers).ok()?;
    let mut digest = Sha256::new();
    digest.update(IMPLEMENTATION);
    hash_value(&mut digest, &configuration);
    // Include the actual borrowed lookup table as malformed/aliased vocabularies
    // need not have an unambiguous reverse mapping from serialized model vocab.
    hash_value(&mut digest, &vocabulary);
    hash_value(&mut digest, &markers);
    digest.update([u8::from(byte_level), u8::from(bounded_byte_level)]);
    Some(digest.finalize().into())
}

pub(super) fn resolved(
    content: Option<[u8; 32]>,
    special_tokens: &SpecialTokens,
) -> Option<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.hf-host-output.resolved-special-policy.v1\0");
    digest.update(content?);
    hash_value(&mut digest, &serde_json::to_value(special_tokens).ok()?);
    Some(digest.finalize().into())
}

/// Typed, length-delimited canonical hashing. Object insertion order does not
/// affect identity; array order and numeric representations remain observable.
fn hash_value(digest: &mut Sha256, value: &Value) {
    match value {
        Value::Null => digest.update([0]),
        Value::Bool(value) => digest.update([1, u8::from(*value)]),
        Value::Number(value) => {
            digest.update([2]);
            hash_bytes(digest, value.to_string().as_bytes());
        }
        Value::String(value) => {
            digest.update([3]);
            hash_bytes(digest, value.as_bytes());
        }
        Value::Array(values) => {
            digest.update([4]);
            digest.update((values.len() as u64).to_le_bytes());
            for value in values {
                hash_value(digest, value);
            }
        }
        Value::Object(values) => {
            digest.update([5]);
            digest.update((values.len() as u64).to_le_bytes());
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            for key in keys {
                hash_bytes(digest, key.as_bytes());
                hash_value(digest, &values[key]);
            }
        }
    }
}

fn hash_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

#[cfg(test)]
mod tests;
