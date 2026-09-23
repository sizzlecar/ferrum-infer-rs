//! Text expansion capability, separate from output protocol/transport budgets.

use std::num::NonZeroUsize;

/// The declaring tokenizer guarantees that decoding any sequence of `n` token
/// IDs produces at most `n * max_utf8_bytes_per_token` UTF-8 bytes, for either
/// value of `skip_special`. Empty input must decode to empty text.
///
/// This is a complete-text bound, not a bound on the next visible event. UTF-8,
/// stop-prefix and protocol buffering can release many earlier tokens at once.
/// It also does not guarantee prefix-stable incremental decoding or bound JSON
/// escaping, protocol envelopes, copies, or allocator overhead. Those costs
/// require separate, explicit reservations at their owning layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedTextBound {
    max_utf8_bytes_per_token: NonZeroUsize,
}

impl DecodedTextBound {
    /// Declare a bound established from the decoder's algorithm and vocabulary.
    /// Sampling observed strings is insufficient to establish this capability.
    pub const fn new(max_utf8_bytes_per_token: NonZeroUsize) -> Self {
        Self {
            max_utf8_bytes_per_token,
        }
    }

    pub const fn max_utf8_bytes_per_token(self) -> NonZeroUsize {
        self.max_utf8_bytes_per_token
    }

    /// Maximum complete decoded payload; overflow is unsupported, never an
    /// unlimited or saturating capacity reservation.
    pub fn max_decoded_bytes(self, tokens: usize) -> Option<usize> {
        self.max_utf8_bytes_per_token.get().checked_mul(tokens)
    }

    /// Bound on bytes not yet emitted from a verified, unchanged decoded prefix.
    /// The caller must establish prefix stability and count only bytes from the
    /// tokenizer, not injected protocol text. Existing retained storage already
    /// owns credit; moving it to an output event transfers that credit.
    pub fn max_unemitted_bytes(
        self,
        committed_tokens: usize,
        emitted_prefix_bytes: usize,
    ) -> Option<usize> {
        self.max_decoded_bytes(committed_tokens)?
            .checked_sub(emitted_prefix_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_tokens_are_part_of_the_next_event_budget() {
        let bound = DecodedTextBound::new(NonZeroUsize::new(12).unwrap());
        assert_eq!(bound.max_decoded_bytes(0), Some(0));
        assert_eq!(bound.max_unemitted_bytes(4, 12), Some(36));
        assert!(bound.max_unemitted_bytes(4, 12).unwrap() > 12);
        assert_eq!(bound.max_unemitted_bytes(1, 13), None);
    }

    #[test]
    fn text_budget_overflow_cannot_authorize_output() {
        let bound = DecodedTextBound::new(NonZeroUsize::new(3).unwrap());
        assert_eq!(
            bound.max_decoded_bytes(usize::MAX / 3),
            Some(usize::MAX / 3 * 3)
        );
        assert_eq!(bound.max_decoded_bytes(usize::MAX / 3 + 1), None);
        assert_eq!(bound.max_unemitted_bytes(usize::MAX, 0), None);
    }
}
