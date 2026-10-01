//! Unique-history bounds from observed token counts. No future token IDs and
//! no executable sampling policy are constructed by this numerical protocol.
use super::ExecutionCostRouteUnknown;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FutureRepetitionRangeV3 {
    minimum: u64,
    maximum: u64,
}
impl FutureRepetitionRangeV3 {
    /// Full-generation history: each new token adds zero or one unique ID.
    /// The first generated token necessarily adds one. Vocabulary is a real
    /// producer/admission limit, not a fitted or benchmark-selected capacity.
    pub fn from_observed_history(
        known_unique: u64,
        known_generated: u64,
        future_generated: u64,
        vocabulary: u64,
    ) -> Result<Self, ExecutionCostRouteUnknown> {
        let invalid = ExecutionCostRouteUnknown::InvalidInput;
        if vocabulary == 0
            || vocabulary > u64::from(u32::MAX)
            || known_unique > vocabulary
            || known_unique > known_generated
            || (known_generated == 0) != (known_unique == 0)
        {
            return Err(invalid);
        }
        let added = future_generated
            .checked_sub(known_generated)
            .ok_or(invalid)?;
        let maximum = known_unique
            .checked_add(added)
            .ok_or(invalid)?
            .min(vocabulary);
        let minimum = if known_unique == 0 && added > 0 {
            1
        } else {
            known_unique
        };
        Ok(Self { minimum, maximum })
    }
    pub fn minimum(self) -> u64 {
        self.minimum
    }
    pub fn maximum(self) -> u64 {
        self.maximum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn unique_range_contains_every_actual_short_vocabulary_continuation() {
        let vocab = 3u64;
        for prefix in [vec![], vec![0], vec![0, 0], vec![0, 1], vec![0, 1, 2]] {
            let known = prefix.iter().copied().collect::<BTreeSet<u64>>();
            for added in 0..=4u32 {
                let range = FutureRepetitionRangeV3::from_observed_history(
                    known.len() as u64,
                    prefix.len() as u64,
                    prefix.len() as u64 + u64::from(added),
                    vocab,
                )
                .unwrap();
                let mut observed = BTreeSet::new();
                for encoded in 0..vocab.pow(added) {
                    let mut tokens = known.clone();
                    let mut value = encoded;
                    for _ in 0..added {
                        tokens.insert(value % vocab);
                        value /= vocab;
                    }
                    observed.insert(tokens.len() as u64);
                }
                assert_eq!(observed.first().copied(), Some(range.minimum()));
                assert_eq!(observed.last().copied(), Some(range.maximum()));
            }
        }
    }
    #[test]
    fn unique_range_rejects_invalid_history_regression_and_arithmetic_overflow() {
        for args in [
            (0, 1, 2, 3),
            (2, 1, 2, 3),
            (1, 2, 1, 3),
            (1, 1, 2, 0),
            (4, 4, 5, 3),
        ] {
            assert!(
                FutureRepetitionRangeV3::from_observed_history(args.0, args.1, args.2, args.3)
                    .is_err()
            );
        }
        assert_eq!(
            FutureRepetitionRangeV3::from_observed_history(3, 100, 200, 3).unwrap(),
            FutureRepetitionRangeV3 {
                minimum: 3,
                maximum: 3
            }
        );
    }
}
