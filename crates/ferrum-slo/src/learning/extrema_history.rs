//! Actual causal training witnesses, bounded independently of sample rate.

use super::{worker::Observation, LearningConfig};
use crate::model::FEATURE_COUNT;
use std::time::{Duration, Instant};

pub(super) const MAX_EXTREMA: usize = 2 * FEATURE_COUNT;

pub(super) struct ExtremaHistory {
    samples: Vec<Observation>,
    promoted_through: u64,
}

pub(super) fn fresh(observation: &Observation, now: Instant, ttl: Duration) -> bool {
    now.checked_duration_since(observation.completed_at)
        .is_some_and(|age| age < ttl)
}

/// Indices refer to the supplied order. Equal values select the newer actual
/// producer sequence, so only a genuinely newer observation renews evidence.
pub(super) fn extremal_indices<'a>(
    observations: impl IntoIterator<Item = &'a Observation>,
) -> [Option<usize>; MAX_EXTREMA] {
    let mut witnesses: [Option<(usize, &'a Observation)>; MAX_EXTREMA] = [None; MAX_EXTREMA];
    for (index, observation) in observations.into_iter().enumerate() {
        for axis in 0..FEATURE_COUNT {
            for (slot, minimum) in [(2 * axis, true), (2 * axis + 1, false)] {
                let replace = witnesses[slot].is_none_or(|(_, prior)| {
                    let value = observation.sample.features.values()[axis];
                    let old = prior.sample.features.values()[axis];
                    (if minimum { value < old } else { value > old })
                        || (value == old && observation.sequence > prior.sequence)
                });
                if replace {
                    witnesses[slot] = Some((index, observation));
                }
            }
        }
    }
    witnesses.map(|witness| witness.map(|(index, _)| index))
}

impl ExtremaHistory {
    pub fn new(config: &LearningConfig) -> Self {
        Self {
            samples: Vec::with_capacity(config.extrema_history_capacity()),
            promoted_through: 0,
        }
    }

    pub fn samples(&self) -> &[Observation] {
        &self.samples
    }

    /// The caller supplies only the record just before the current immutable
    /// validation/calibration suffix. Current suffix records cannot evict it.
    pub fn promote(&mut self, config: &LearningConfig, observation: Observation, now: Instant) {
        if config.extrema_history_capacity() == 0 {
            return;
        }
        let ttl = Duration::from_millis(config.model_ttl_ms);
        self.samples.retain(|sample| fresh(sample, now, ttl));
        if observation.sequence <= self.promoted_through {
            return;
        }
        self.promoted_through = observation.sequence;
        if !fresh(&observation, now, ttl)
            || !observation.sample.service_seconds.is_finite()
            || observation.sample.service_seconds <= 0.0
        {
            return;
        }
        let indices = extremal_indices(self.samples.iter().chain(std::iter::once(&observation)));
        let new_index = self.samples.len();
        let include_new = indices.contains(&Some(new_index));
        let mut index = 0;
        self.samples.retain(|_| {
            let retain = indices.contains(&Some(index));
            index += 1;
            retain
        });
        // At most eighteen distinct indices survive. Remove dominated records
        // before the push so this never grows beyond the validated allocation.
        if include_new {
            self.samples.push(observation);
        }
    }
}

#[cfg(test)]
#[path = "extrema_history_tests.rs"]
mod tests;
