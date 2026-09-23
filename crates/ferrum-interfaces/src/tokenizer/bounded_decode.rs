//! Decode capability with caller-owned, preauthorized working storage.

use super::DecodedTextBound;
use std::num::NonZeroUsize;

/// An explicit equivalence between ordinary incremental decoding and two
/// bounded full decodes with skip_special=true. A full-decode bound alone
/// cannot establish incremental semantics for an arbitrary tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundedIncrementalDecodePolicy {
    /// The delta is full.strip_prefix(previous); rewriting the prefix errors.
    StrictDecodedPrefix,
}

impl BoundedIncrementalDecodePolicy {
    /// Borrow the exact delta without allocating an additional String. None
    /// means the legacy operation must report a changed-prefix error.
    pub fn delta<'a>(self, previous: &str, full: &'a str) -> Option<&'a str> {
        match self {
            Self::StrictDecodedPrefix => full.strip_prefix(previous),
        }
    }
}

/// A declaration that `decode_bounded_into` uses only the supplied output and
/// scratch storage, borrowed immutable vocabulary, and constant-size stack
/// state. It must not allocate token copies, temporary strings, caches, or other
/// input-sized storage. This capability does not change ordinary `decode`.
///
/// Bounds cover either `skip_special` value and the entire input, including
/// retained tokens. Callers must reserve before allocating and charge the actual
/// buffer capacities if their allocator returns more than requested. Persistent
/// tokenizer storage, allocator metadata, protocol assembly and transport copies
/// are separate accounting domains; these bounds do not include them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedDecodeBound {
    text: DecodedTextBound,
    max_scratch_bytes_per_token: usize,
}

impl BoundedDecodeBound {
    /// Declare bounds proven from the implementation and immutable vocabulary.
    /// A decoder with no input-sized scratch may declare zero scratch bytes.
    pub const fn new(
        max_utf8_bytes_per_token: NonZeroUsize,
        max_scratch_bytes_per_token: usize,
    ) -> Self {
        Self {
            text: DecodedTextBound::new(max_utf8_bytes_per_token),
            max_scratch_bytes_per_token,
        }
    }

    pub const fn text_bound(self) -> DecodedTextBound {
        self.text
    }

    pub const fn max_scratch_bytes_per_token(self) -> usize {
        self.max_scratch_bytes_per_token
    }

    /// Overflow in either component or their sum is an error, never an
    /// unlimited, wrapped, or saturating reservation.
    pub fn requirements(
        self,
        tokens: usize,
    ) -> Result<BoundedDecodeRequirements, BoundedDecodeError> {
        let text_bytes = self
            .text
            .max_decoded_bytes(tokens)
            .ok_or(BoundedDecodeError::CapacityOverflow)?;
        let scratch_bytes = self
            .max_scratch_bytes_per_token
            .checked_mul(tokens)
            .ok_or(BoundedDecodeError::CapacityOverflow)?;
        let total_bytes = text_bytes
            .checked_add(scratch_bytes)
            .ok_or(BoundedDecodeError::CapacityOverflow)?;
        Ok(BoundedDecodeRequirements {
            text_bytes,
            scratch_bytes,
            total_bytes,
        })
    }
}

/// Minimum payload capacities to allocate before decoding. `total_bytes` is the
/// checked sum, not additional storage to allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedDecodeRequirements {
    pub text_bytes: usize,
    pub scratch_bytes: usize,
    pub total_bytes: usize,
}

/// Errors carry no allocated diagnostic strings. On error the decoder must
/// leave both supplied buffers unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BoundedDecodeError {
    #[error("tokenizer does not support bounded decoding")]
    Unsupported,
    #[error("bounded decode capacity arithmetic overflow")]
    CapacityOverflow,
    #[error("bounded decode needs {required} scratch bytes, but has {available}")]
    InsufficientScratch { required: usize, available: usize },
    #[error("bounded decode needs {required} output bytes, but has {available}")]
    InsufficientOutput { required: usize, available: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_decode_requirements_include_both_live_buffers() {
        let bound = BoundedDecodeBound::new(NonZeroUsize::new(9).unwrap(), 3);
        assert_eq!(bound.text_bound().max_decoded_bytes(2), Some(18));
        assert_eq!(bound.max_scratch_bytes_per_token(), 3);
        assert_eq!(
            bound.requirements(2),
            Ok(BoundedDecodeRequirements {
                text_bytes: 18,
                scratch_bytes: 6,
                total_bytes: 24,
            })
        );
        assert_eq!(bound.requirements(0).unwrap().total_bytes, 0);
    }

    #[test]
    fn bounded_decode_rejects_component_and_combined_overflow() {
        let text = BoundedDecodeBound::new(NonZeroUsize::new(2).unwrap(), 0);
        let scratch = BoundedDecodeBound::new(NonZeroUsize::new(1).unwrap(), 2);
        let combined = BoundedDecodeBound::new(NonZeroUsize::new(1).unwrap(), 1);
        for bound in [text, scratch, combined] {
            assert_eq!(
                bound.requirements(usize::MAX / 2 + 1),
                Err(BoundedDecodeError::CapacityOverflow)
            );
        }
        assert_eq!(
            combined.requirements(usize::MAX / 2).unwrap().total_bytes,
            usize::MAX - 1
        );
    }
}
