//! Bounded actual-width representatives and chronological fitting partitions.

use super::{worker::Observation, LearningConfig, TrainingHistoryPolicy};
use crate::model::ModelError;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub(super) struct WidthHistory {
    samples: VecDeque<Observation>,
}

fn width(observation: &Observation, maximum: usize) -> Option<usize> {
    let width = observation.sample.features.batch_size();
    (width >= 1.0 && width <= maximum as f64 && width.fract() == 0.0).then_some(width as usize)
}

fn fresh(observation: &Observation, now: Instant, ttl: Duration) -> bool {
    now.checked_duration_since(observation.completed_at)
        .is_some_and(|age| age < ttl)
}

impl WidthHistory {
    pub fn new(config: &LearningConfig) -> Self {
        Self {
            samples: VecDeque::with_capacity(
                config
                    .width_history_capacity()
                    .expect("validated width history bound"),
            ),
        }
    }

    pub fn observe(&mut self, config: &LearningConfig, observation: Observation, now: Instant) {
        let TrainingHistoryPolicy::WidthStratified { samples_per_width } = config.training_history
        else {
            return;
        };
        let ttl = Duration::from_millis(config.model_ttl_ms);
        self.samples.retain(|sample| fresh(sample, now, ttl));
        let Some(observed_width) = width(&observation, config.collection.max_participants) else {
            return;
        };
        if !fresh(&observation, now, ttl)
            || !observation.sample.service_seconds.is_finite()
            || observation.sample.service_seconds <= 0.0
        {
            return;
        }
        let matching = |sample: &&Observation| {
            width(sample, config.collection.max_participants) == Some(observed_width)
        };
        if self.samples.iter().filter(matching).count() >= samples_per_width {
            let oldest = self
                .samples
                .iter()
                .position(|sample| {
                    width(sample, config.collection.max_participants) == Some(observed_width)
                })
                .expect("counted width representative");
            self.samples.remove(oldest);
        }
        // Input has passed the owner's monotone sequence/actual-work checks.
        // Reuse neither changes its timestamp nor introduces another label.
        self.samples.push_back(observation);
    }
}

pub(super) fn fit_observations(
    config: &LearningConfig,
    window: &VecDeque<Observation>,
    history: &WidthHistory,
    now: Instant,
) -> Result<Vec<Observation>, ModelError> {
    let start =
        window
            .len()
            .checked_sub(config.required_samples())
            .ok_or(ModelError::InvalidData(
                "insufficient chronological fit history",
            ))?;
    if config.training_history == TrainingHistoryPolicy::Recent {
        return Ok(window.iter().skip(start).copied().collect());
    }
    let suffix_start = window.len() - config.validation_samples - config.calibration_samples;
    let validation_start = window[suffix_start].sequence;
    let ttl = Duration::from_millis(config.model_ttl_ms);
    let mut candidates = Vec::with_capacity(window.len() + history.samples.len());
    candidates.extend(
        window
            .iter()
            .chain(&history.samples)
            .filter(|sample| sample.sequence < validation_start && fresh(sample, now, ttl))
            .copied(),
    );
    candidates.sort_unstable_by_key(|sample| sample.sequence);
    candidates.dedup_by_key(|sample| sample.sequence);
    if candidates
        .iter()
        .any(|sample| width(sample, config.collection.max_participants).is_none())
    {
        return Err(ModelError::InvalidData("invalid observed width"));
    }
    // Order depends only on actual width and producer sequence, never labels,
    // prediction errors, calibration outcomes or held-out model responses.
    candidates.sort_unstable_by(|a, b| {
        a.sample
            .features
            .batch_size()
            .total_cmp(&b.sample.features.batch_size())
            .then_with(|| b.sequence.cmp(&a.sequence))
    });
    let mut groups = Vec::with_capacity(config.collection.max_participants);
    let mut first = 0;
    while first < candidates.len() {
        let group_width = candidates[first].sample.features.batch_size();
        let mut end = first + 1;
        while end < candidates.len() && candidates[end].sample.features.batch_size() == group_width
        {
            end += 1;
        }
        groups.push((first, end));
        first = end;
    }
    let mut selected = Vec::with_capacity(config.required_samples());
    let mut level = 0;
    while selected.len() < config.training_samples {
        let before = selected.len();
        for &(first, end) in &groups {
            if first + level < end {
                selected.push(candidates[first + level]);
                if selected.len() == config.training_samples {
                    break;
                }
            }
        }
        if selected.len() == before {
            return Err(ModelError::InvalidData(
                "insufficient fresh stratified training history",
            ));
        }
        level += 1;
    }
    selected.sort_unstable_by_key(|sample| sample.sequence);
    selected.extend(window.iter().skip(suffix_start).copied());
    Ok(selected)
}

#[cfg(test)]
#[path = "width_history_tests.rs"]
mod tests;
