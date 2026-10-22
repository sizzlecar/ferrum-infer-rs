//! Lossless retained encoding only. The validated distribution keeps every
//! rank in order; no quantile, rounding, clipping or support rule changes here.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TimeEncoding {
    Nanoseconds,
    SecondsNanoseconds,
}

impl TimeEncoding {
    pub fn select(scenarios: &[CostScenario]) -> Self {
        if scenarios
            .iter()
            .all(|scenario| u64::try_from(scenario.service_time.as_nanos()).is_ok())
        {
            Self::Nanoseconds
        } else {
            Self::SecondsNanoseconds
        }
    }

    fn words_per_rank(self) -> usize {
        match self {
            Self::Nanoseconds => 1,
            Self::SecondsNanoseconds => 2,
        }
    }

    pub fn payload_bytes(self, ranks: usize) -> Option<usize> {
        ranks
            .checked_mul(self.words_per_rank())?
            .checked_mul(std::mem::size_of::<u64>())
    }

    /// Called only after selection and the checked retained-byte guard, with
    /// that same owned distribution. Its times cannot change between the two.
    pub fn pack(self, scenarios: Vec<CostScenario>) -> Box<[u64]> {
        match self {
            Self::Nanoseconds => scenarios
                .into_iter()
                .map(|scenario| {
                    u64::try_from(scenario.service_time.as_nanos())
                        .expect("selected lossless nanosecond representation")
                })
                .collect(),
            Self::SecondsNanoseconds => {
                let words = scenarios
                    .len()
                    .checked_mul(2)
                    .expect("checked retained payload size");
                let mut encoded = Vec::with_capacity(words);
                for scenario in scenarios {
                    encoded.push(scenario.service_time.as_secs());
                    encoded.push(u64::from(scenario.service_time.subsec_nanos()));
                }
                encoded.into_boxed_slice()
            }
        }
    }

    pub fn get(self, encoded: &[u64], rank: usize) -> Option<Duration> {
        match self {
            Self::Nanoseconds => encoded.get(rank).copied().map(Duration::from_nanos),
            Self::SecondsNanoseconds => {
                // A word index is never a rank. Keep overflow/out-of-range and
                // malformed-pair checks even though only pack constructs these.
                if encoded.len() % 2 != 0 {
                    return None;
                }
                let offset = rank.checked_mul(2)?;
                let seconds = *encoded.get(offset)?;
                let nanos = u32::try_from(*encoded.get(offset.checked_add(1)?)?).ok()?;
                (nanos < 1_000_000_000).then(|| Duration::new(seconds, nanos))
            }
        }
    }
}
