//! Borrowed GDN control validation for compact binding preparation, without a
//! payload allocation. Full execution retains its original single-pass encoder.
//! These values prove control encoding bounds, not resource or device authority.

#[derive(Debug)]
pub(crate) struct SequenceControl<'a> {
    token_counts: &'a [u64],
    capacity: usize,
}

impl<'a> SequenceControl<'a> {
    pub(crate) fn validate(token_counts: &'a [u64]) -> Result<Self, String> {
        if token_counts.is_empty() {
            return Err("attention sequence control cannot be empty".to_owned());
        }
        let mut cumulative = 0_u64;
        let capacity = token_counts
            .len()
            .checked_add(1)
            .and_then(|entries| entries.checked_mul(4))
            .ok_or_else(|| "attention sequence control size overflows".to_owned())?;
        for tokens in token_counts {
            if *tokens == 0 {
                return Err("attention sequence control contains an empty sequence".to_owned());
            }
            cumulative = cumulative
                .checked_add(*tokens)
                .ok_or_else(|| "attention sequence control token count overflows".to_owned())?;
            u32::try_from(cumulative)
                .map_err(|_| "attention sequence control exceeds u32".to_owned())?;
        }
        Ok(Self {
            token_counts,
            capacity,
        })
    }

    pub(crate) fn into_bytes(self) -> Box<[u8]> {
        let mut control = Vec::with_capacity(self.capacity);
        control.extend_from_slice(&0_u32.to_le_bytes());
        let mut cumulative = 0_u64;
        for tokens in self.token_counts {
            // The immutable slice is the same slice checked by validate().
            cumulative += *tokens;
            let cumulative = u32::try_from(cumulative).expect("validated GDN cumulative count");
            control.extend_from_slice(&cumulative.to_le_bytes());
        }
        control.into_boxed_slice()
    }
}

#[derive(Debug)]
pub(crate) struct TokenSequenceIndices<'a> {
    token_counts: &'a [u64],
    capacity: usize,
}

impl<'a> TokenSequenceIndices<'a> {
    pub(crate) fn validate(token_counts: &'a [u64]) -> Result<Self, String> {
        let total_tokens = token_counts.iter().try_fold(0_u64, |total, tokens| {
            total
                .checked_add(*tokens)
                .ok_or_else(|| "attention token sequence index count overflows".to_owned())
        })?;
        let capacity = usize::try_from(total_tokens)
            .map_err(|_| "attention token sequence index count exceeds usize".to_owned())?
            .checked_mul(4)
            .ok_or_else(|| "attention token sequence index bytes overflow".to_owned())?;
        // Preserve the check even for an empty sequence: its index still has to
        // be representable under the original encoding contract.
        for (sequence, _) in token_counts.iter().enumerate() {
            u32::try_from(sequence)
                .map_err(|_| "attention sequence index exceeds u32".to_owned())?;
        }
        Ok(Self {
            token_counts,
            capacity,
        })
    }

    pub(crate) fn into_bytes(self) -> Box<[u8]> {
        let mut indices = Vec::with_capacity(self.capacity);
        for (sequence, tokens) in self.token_counts.iter().enumerate() {
            let sequence = u32::try_from(sequence).expect("validated GDN sequence index");
            for _ in 0..*tokens {
                indices.extend_from_slice(&sequence.to_le_bytes());
            }
        }
        indices.into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::{SequenceControl, TokenSequenceIndices};

    #[test]
    fn gdn_sequence_control_preserves_unequal_sequence_wire_bytes() {
        let counts = [2, 1, 3];
        assert_eq!(
            &*SequenceControl::validate(&counts).unwrap().into_bytes(),
            &[0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0]
        );
        assert_eq!(
            &*TokenSequenceIndices::validate(&counts)
                .unwrap()
                .into_bytes(),
            &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0]
        );
    }

    #[test]
    fn gdn_sequence_control_preserves_empty_and_prefix_failure_rules() {
        assert_eq!(
            SequenceControl::validate(&[]).unwrap_err(),
            "attention sequence control cannot be empty"
        );
        assert_eq!(
            SequenceControl::validate(&[0, u64::MAX]).unwrap_err(),
            "attention sequence control contains an empty sequence"
        );
        assert_eq!(
            SequenceControl::validate(&[u64::from(u32::MAX), 1, 0]).unwrap_err(),
            "attention sequence control exceeds u32"
        );
        // The inclusive u32 endpoint is legal and emits only two entries.
        assert_eq!(
            &*SequenceControl::validate(&[u64::from(u32::MAX)])
                .unwrap()
                .into_bytes(),
            &[0, 0, 0, 0, 255, 255, 255, 255]
        );
    }

    #[test]
    fn gdn_token_sequence_indices_preserve_zero_length_sequence_positions() {
        assert!(TokenSequenceIndices::validate(&[])
            .unwrap()
            .into_bytes()
            .is_empty());
        assert_eq!(
            &*TokenSequenceIndices::validate(&[0, 1, 0, 2])
                .unwrap()
                .into_bytes(),
            &[1, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0]
        );
    }

    #[test]
    fn gdn_token_sequence_indices_check_counts_before_payload_allocation() {
        assert_eq!(
            TokenSequenceIndices::validate(&[u64::MAX, 1]).unwrap_err(),
            "attention token sequence index count overflows"
        );
        let too_many_bytes = u64::try_from(usize::MAX / 4).unwrap() + 1;
        assert_eq!(
            TokenSequenceIndices::validate(&[too_many_bytes]).unwrap_err(),
            "attention token sequence index bytes overflow"
        );
        if usize::BITS < u64::BITS {
            let too_many_tokens = u64::try_from(usize::MAX).unwrap() + 1;
            assert_eq!(
                TokenSequenceIndices::validate(&[too_many_tokens]).unwrap_err(),
                "attention token sequence index count exceeds usize"
            );
        }
    }
}
