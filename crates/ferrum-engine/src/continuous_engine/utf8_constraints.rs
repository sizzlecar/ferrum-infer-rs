//! Compact raw-byte constraints. One immutable three-byte entry per scanned
//! vocabulary token, shared with the existing tokenizer policy cache.
use super::*;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TokenTransition {
    valid: u8,
    complete: u8,
    known: bool,
}

impl TokenTransition {
    pub(super) fn new(bytes: Option<&[u8]>) -> Self {
        let Some(bytes) = bytes else {
            return Self::default();
        };
        let mut result = Self {
            known: true,
            ..Self::default()
        };
        for state in 0..8 {
            if let Some(next) = advance(state, bytes) {
                result.valid |= 1 << state;
                if next == 0 {
                    result.complete |= 1 << state;
                }
            }
        }
        result
    }

    fn permits(self, state: u8, terminal: bool) -> bool {
        !self.known || (if terminal { self.complete } else { self.valid }) & (1 << state) != 0
    }
}

#[derive(Debug)]
pub(super) struct CachedTransitions {
    pub(super) tokenizer: Weak<dyn Tokenizer + Send + Sync>,
    pub(super) tokens: Box<[TokenTransition]>,
}

impl CachedTransitions {
    pub(super) fn apply(
        &self,
        tokenizer: &dyn Tokenizer,
        pending: &[u8],
        stops: &HashSet<u32>,
        required_delimiter: Option<u32>,
        logits: &mut [f32],
    ) -> Result<()> {
        // Callers may supply a different tokenizer in legacy/test paths. Its
        // byte policy must never borrow this table's authority.
        if self.tokenizer.strong_count() == 0
            || self.tokenizer.as_ptr().cast::<()>()
                != (tokenizer as *const dyn Tokenizer).cast::<()>()
        {
            return Err(FerrumError::tokenizer(
                "sampling tokenizer differs from its admitted byte policy",
            ));
        }
        if pending.len() > 3 {
            return Err(FerrumError::internal(
                "committed UTF-8 pending state exceeds three bytes",
            ));
        }
        let state = advance(0, pending)
            .ok_or_else(|| FerrumError::internal("committed UTF-8 pending state is invalid"))?;
        if !pending.is_empty() && state == 0 {
            return Err(FerrumError::internal(
                "committed UTF-8 pending state is already complete",
            ));
        }
        for (id, (logit, token)) in logits.iter_mut().zip(self.tokens.iter()).enumerate() {
            if required_delimiter == Some(id as u32) {
                continue;
            }
            // Only a valid-but-incomplete token needs the stop-set lookup.
            // Complete tokens and invalid continuations settle from the bits.
            if logit.is_finite()
                && (!token.permits(state, false)
                    || (!token.permits(state, true) && stops.contains(&(id as u32))))
            {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

// Accept; 1/2/3 unrestricted continuation bytes; E0/ED/F0/F4 restricted first
// continuation. Reject overlongs, surrogates, and values above U+10FFFF.
fn advance(mut state: u8, bytes: &[u8]) -> Option<u8> {
    for &byte in bytes {
        state = match (state, byte) {
            (0, 0..=0x7f) => 0,
            (0, 0xc2..=0xdf) => 1,
            (0, 0xe0) => 4,
            (0, 0xed) => 5,
            (0, 0xe1..=0xec | 0xee..=0xef) => 2,
            (0, 0xf0) => 6,
            (0, 0xf1..=0xf3) => 3,
            (0, 0xf4) => 7,
            (1..=3, 0x80..=0xbf) => state - 1,
            (4, 0xa0..=0xbf) | (5, 0x80..=0x9f) => 1,
            (6, 0x90..=0xbf) | (7, 0x80..=0x8f) => 2,
            _ => return None,
        };
    }
    Some(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_table_matches_rust_utf8_prefix_validation() {
        // Includes every byte after every distinguishable live prefix, plus
        // complete and mixed tokens. This checks the table's semantic states.
        let prefixes: &[&[u8]] = &[
            &[],
            &[0xc2],
            &[0xe1],
            &[0xf1],
            &[0xe0],
            &[0xed],
            &[0xf0],
            &[0xf4],
        ];
        for prefix in prefixes {
            let state = advance(0, prefix).unwrap();
            for byte in 0..=255 {
                let mut combined = prefix.to_vec();
                combined.push(byte);
                let (valid, complete) = match std::str::from_utf8(&combined) {
                    Ok(_) => (true, true),
                    Err(error) => (error.error_len().is_none(), false),
                };
                let token = TokenTransition::new(Some(&[byte]));
                assert_eq!(token.permits(state, false), valid, "{combined:?}");
                assert_eq!(token.permits(state, true), complete, "{combined:?}");
            }
        }
        assert!(TokenTransition::new(Some(&[0x94, 0xa5, b'x', 0xe4])).permits(2, false));
        assert!(!TokenTransition::new(Some(&[0x94, 0xa5, b'x', 0xe4])).permits(2, true));
        assert!(!TokenTransition::new(Some(&[0xed, 0xa0, 0x80])).permits(0, false));
    }
}
