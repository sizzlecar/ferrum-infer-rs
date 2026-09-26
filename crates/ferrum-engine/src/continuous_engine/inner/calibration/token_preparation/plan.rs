//! Declaration-only byte trajectory validation. This module grants no live
//! sequence, submission, cost-sample, or source-membership authority.
use ferrum_interfaces::Tokenizer;
use ferrum_types::{FerrumError, Result, TokenId};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

/// One immutable diagnostic prefix. A future session installer must separately
/// bind its cohort slot and validate the actual admitted request's constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPrefixTokensV1 {
    pub tokenizer_policy_sha256: [u8; 32],
    pub token_ids: Vec<TokenId>,
    pub release_generated: usize,
}

/// Static declaration evidence only. In particular, expected pending bytes are
/// never permission to write a SequenceState or construct a live cost receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCalibrationPrefixTokensV1 {
    declaration: CalibrationPrefixTokensV1,
    expected_pending_bytes: Vec<u8>,
}

impl CalibrationPrefixTokensV1 {
    pub fn validate_for_tokenizer(
        &self,
        tokenizer: &dyn Tokenizer,
        maximum_output: NonZeroUsize,
    ) -> Result<ValidatedCalibrationPrefixTokensV1> {
        self.validate_length(maximum_output)?;
        if tokenizer.host_output_policy_identity() != Some(self.tokenizer_policy_sha256) {
            return Err(FerrumError::invalid_request(
                "calibration prefix tokenizer policy differs from its declaration",
            ));
        }
        let raw_bound = tokenizer.bounded_token_bytes_bound().ok_or_else(|| {
            FerrumError::unsupported("calibration prefix requires bounded raw token bytes")
        })?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(raw_bound.get()).map_err(|_| {
            FerrumError::invalid_request("calibration prefix raw byte buffer exceeds capacity")
        })?;
        scratch.resize(raw_bound.get(), 0);
        let mut pending = Vec::new();
        for &token in &self.token_ids {
            if usize::from(token) >= tokenizer.vocab_size() {
                return Err(FerrumError::invalid_request(
                    "calibration prefix token exceeds actual vocabulary",
                ));
            }
            let written = tokenizer
                .token_bytes_bounded_into(token, &mut scratch)
                .map_err(|_| {
                    FerrumError::invalid_request("calibration prefix raw token bytes unavailable")
                })?
                .ok_or_else(|| {
                    FerrumError::invalid_request("calibration prefix contains an unknown token")
                })?;
            let bytes = scratch.get(..written).ok_or_else(|| {
                FerrumError::invalid_request("calibration prefix tokenizer exceeded its byte bound")
            })?;
            Self::advance(&mut pending, bytes)?;
        }
        Ok(self.validated(pending))
    }

    fn validate_length(&self, maximum_output: NonZeroUsize) -> Result<()> {
        if self.token_ids.is_empty()
            || self.release_generated != self.token_ids.len()
            || self.release_generated >= maximum_output.get()
        {
            return Err(FerrumError::invalid_request(
                "calibration prefix must release at its exact length before normal output ends",
            ));
        }
        Ok(())
    }

    fn advance(pending: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
        // Use the exact committed-output validator, including replacement and
        // mojibake rejection; do not create a second approximate UTF-8 DFA.
        *pending = crate::continuous_engine::advance_pending_utf8_fragment(pending, bytes)
            .map_err(|()| {
                FerrumError::invalid_request("calibration prefix has an invalid UTF-8 trajectory")
            })?;
        Ok(())
    }

    fn validated(&self, pending: Vec<u8>) -> ValidatedCalibrationPrefixTokensV1 {
        ValidatedCalibrationPrefixTokensV1 {
            declaration: self.clone(),
            expected_pending_bytes: pending,
        }
    }
}

impl ValidatedCalibrationPrefixTokensV1 {
    pub fn declaration(&self) -> &CalibrationPrefixTokensV1 {
        &self.declaration
    }

    pub fn expected_pending_bytes(&self) -> &[u8] {
        &self.expected_pending_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaration(tokens: usize) -> CalibrationPrefixTokensV1 {
        CalibrationPrefixTokensV1 {
            tokenizer_policy_sha256: [17; 32],
            token_ids: (0..tokens as u32).map(TokenId::new).collect(),
            release_generated: tokens,
        }
    }

    #[test]
    fn release_requires_an_exact_nonempty_prefix_and_a_normal_suffix() {
        let maximum = NonZeroUsize::new(4).unwrap();
        for tokens in [0, 4, 5] {
            assert!(declaration(tokens).validate_length(maximum).is_err());
        }
        let mut valid = declaration(3);
        valid.validate_length(maximum).unwrap();
        valid.release_generated = 2;
        assert!(valid.validate_length(maximum).is_err());
    }

    #[test]
    fn real_committed_byte_validator_retains_and_completes_pending() {
        let mut pending = Vec::new();
        CalibrationPrefixTokensV1::advance(&mut pending, b"plain").unwrap();
        assert!(pending.is_empty());
        for fragment in [&[0xf0][..], &[0x9f][..], &[0x94][..]] {
            CalibrationPrefixTokensV1::advance(&mut pending, fragment).unwrap();
        }
        assert_eq!(pending, [0xf0, 0x9f, 0x94]);
        let static_plan = declaration(4).validated(pending.clone());
        assert_eq!(static_plan.expected_pending_bytes(), pending);
        CalibrationPrefixTokensV1::advance(&mut pending, &[0xa5]).unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn invalid_utf8_and_forbidden_replacement_cannot_be_predeclared() {
        for bytes in [
            &[0x80][..],
            &[0xc0, 0x80][..],
            &[0xed, 0xa0, 0x80][..],
            &[0xf4, 0x90, 0x80, 0x80][..],
            "\u{fffd}".as_bytes(),
        ] {
            assert!(CalibrationPrefixTokensV1::advance(&mut Vec::new(), bytes).is_err());
        }
        let mut pending = vec![0xe1];
        assert!(CalibrationPrefixTokensV1::advance(&mut pending, b"x").is_err());
    }
}
