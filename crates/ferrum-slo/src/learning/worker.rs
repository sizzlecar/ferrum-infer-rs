use super::{snapshot, LearningConfig};
use crate::{
    calibration::OnlineCalibrator,
    model::{
        ExecutionIdentity, FeatureDomain, FittedModel, ModelError, ModelFamily, TrainingSample,
        WorkFeatures,
    },
    queue::Shared,
    worker::{sample_work, validate_sample, Excluded},
};
use ferrum_execution::execution_attempt::ExecutionAttemptSummary;
use std::{
    collections::VecDeque,
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub(super) struct Observation {
    pub(super) sample: TrainingSample,
    pub(super) completed_at: Instant,
    pub(super) sequence: u64,
}

struct Active {
    model: Arc<FittedModel>,
    model_revision: u64,
    model_expires_at: Instant,
    calibration: OnlineCalibrator<FeatureDomain>,
    /// Calibration is strictly later than all fitting/parameter-selection facts.
    fitting_through: u64,
    calibration_after: Option<Instant>,
}

pub(super) struct Domain {
    key: FeatureDomain,
    window: VecDeque<Observation>,
    width_history: super::width_history::WidthHistory,
    seen: u64,
    since_attempt: usize,
    active: Option<Active>,
}

pub(super) struct Learner {
    config: LearningConfig,
    identity: Arc<ExecutionIdentity>,
    family: ModelFamily,
    domains: Vec<Domain>,
    revision: u64,
    model_revision: u64,
}

impl Learner {
    pub fn restore(
        config: LearningConfig,
        identity: ExecutionIdentity,
        family: ModelFamily,
        restored: super::state::RestoredState,
    ) -> Self {
        let mut learner = Self::new(config, identity, family);
        learner.revision = restored.revision;
        for domain in restored.domains {
            learner.model_revision = learner.model_revision.max(domain.model_revision);
            learner.domains.push(Domain {
                key: domain.model.domain(),
                window: VecDeque::with_capacity(learner.config.window_samples),
                width_history: super::width_history::WidthHistory::new(&learner.config),
                seen: 0,
                since_attempt: 0,
                active: Some(Active {
                    model: domain.model,
                    model_revision: domain.model_revision,
                    model_expires_at: domain.model_expires_at,
                    calibration: domain.calibration,
                    fitting_through: 0,
                    calibration_after: Some(restored.loaded_at),
                }),
            });
        }
        learner
    }

    pub fn new(config: LearningConfig, identity: ExecutionIdentity, family: ModelFamily) -> Self {
        let domains = Vec::with_capacity(config.max_domains);
        Self {
            config,
            identity: Arc::new(identity),
            family,
            domains,
            revision: 0,
            model_revision: 0,
        }
    }

    pub fn apply(
        &mut self,
        shared: &Shared,
        summary: &ExecutionAttemptSummary,
        now: Instant,
    ) -> bool {
        if let Some(delay) = now.checked_duration_since(summary.completed_at) {
            shared
                .counters
                .feedback_delay_samples
                .fetch_add(1, Ordering::Relaxed);
            record_duration(
                &shared.counters.feedback_delay_total_ns,
                &shared.counters.feedback_delay_max_ns,
                delay,
            );
        }
        let wall = match validate_sample(summary, now) {
            Ok(wall) => wall,
            Err(Excluded::Ineligible) => {
                shared
                    .counters
                    .excluded_attempts
                    .fetch_add(1, Ordering::Relaxed);
                return false;
            }
            Err(Excluded::Invalid) => {
                shared
                    .counters
                    .invalid_attempts
                    .fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };
        let work: Vec<_> = summary.participants.iter().map(sample_work).collect();
        let Ok(features) = WorkFeatures::extract(&work, self.config.collection.max_participants)
        else {
            shared
                .counters
                .invalid_attempts
                .fetch_add(1, Ordering::Relaxed);
            return false;
        };
        self.observe(
            shared,
            TrainingSample {
                features,
                service_seconds: wall.as_secs_f64(),
            },
            summary.completed_at,
            now,
        )
    }

    fn observe(
        &mut self,
        shared: &Shared,
        sample: TrainingSample,
        completed_at: Instant,
        now: Instant,
    ) -> bool {
        let key = sample.features.domain();
        let index = if let Some(index) = self.domains.iter().position(|d| d.key == key) {
            index
        } else {
            if self.domains.len() == self.config.max_domains {
                shared.counters.domain_limit.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            self.domains.push(Domain {
                key,
                window: VecDeque::with_capacity(self.config.window_samples),
                width_history: super::width_history::WidthHistory::new(&self.config),
                seen: 0,
                since_attempt: 0,
                active: None,
            });
            self.domains.len() - 1
        };
        let domain = &mut self.domains[index];
        if domain
            .window
            .back()
            .is_some_and(|last| completed_at < last.completed_at)
        {
            // Late execution facts remain in product statistics but cannot
            // retroactively enter this causal learning sequence.
            shared
                .counters
                .excluded_attempts
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let Some(sequence) = domain.seen.checked_add(1) else {
            return false;
        };
        domain.seen = sequence;
        let mut changed = false;
        let mut predicted = false;
        // Predict with the previous immutable base before fitting/updating.
        if let Some(active) = &mut domain.active {
            if sequence > active.fitting_through
                && completed_at < active.model_expires_at
                && active
                    .calibration_after
                    .is_none_or(|cutoff| completed_at >= cutoff)
            {
                match active.model.predict(&self.identity, &sample.features) {
                    Ok(base) => {
                        predicted = true;
                        changed = active.calibration.observe(
                            active.model_revision,
                            key,
                            self.config.bucket(sample.features.work_tokens()),
                            sample.service_seconds,
                            base.risk_base_seconds,
                            completed_at,
                        );
                    }
                    Err(ModelError::UnsupportedDomain)
                        if now < active.model_expires_at
                            && now.checked_duration_since(completed_at).is_some_and(|age| {
                                age < Duration::from_millis(self.config.calibration_ttl_ms)
                            })
                            && sample.service_seconds.is_finite()
                            && sample.service_seconds > 0.0 =>
                    {
                        // Match the disjoint calibration suffix: a verified
                        // execution may qualify its measured features without
                        // changing coefficients, scaling, or model freshness.
                        // Validation precedes the bounded copy. Publish it only
                        // with an accepted residual from this same risk head.
                        if let Ok((qualified, base)) = active
                            .model
                            .qualify_observed_copy(&self.identity, &sample.features)
                        {
                            changed = active.calibration.observe(
                                active.model_revision,
                                key,
                                self.config.bucket(sample.features.work_tokens()),
                                sample.service_seconds,
                                base.risk_base_seconds,
                                completed_at,
                            );
                            if changed {
                                active.model = Arc::new(qualified);
                            }
                        }
                        // The previous published envelope was still Unknown.
                        // Do not retrospectively count this as a known worker
                        // prediction or alter the pre-execution frozen ticket.
                    }
                    Err(_) => {}
                }
            }
        }
        if predicted {
            shared
                .counters
                .worker_base_prediction_known
                .fetch_add(1, Ordering::Relaxed);
        } else {
            shared
                .counters
                .worker_base_prediction_unknown
                .fetch_add(1, Ordering::Relaxed);
        }
        if changed {
            shared
                .counters
                .calibration_updates
                .fetch_add(1, Ordering::Relaxed);
        }
        if domain.window.len() == self.config.window_samples {
            domain.window.pop_front();
        }
        let observation = Observation {
            sample,
            completed_at,
            sequence,
        };
        domain.window.push_back(observation);
        domain.width_history.observe(&self.config, observation, now);
        domain.since_attempt = domain.since_attempt.saturating_add(1);
        shared
            .counters
            .trained_samples
            .fetch_add(1, Ordering::Relaxed);
        if domain.window.len() >= self.config.required_samples()
            && domain.since_attempt >= self.config.retrain_every
        {
            domain.since_attempt = 0;
            let Some(next_revision) = self.model_revision.checked_add(1) else {
                return changed;
            };
            shared
                .counters
                .model_fit_attempts
                .fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            let candidate = fit(
                &self.config,
                &self.identity,
                self.family,
                domain,
                next_revision,
                now,
            );
            record_duration(
                &shared.counters.model_fit_total_ns,
                &shared.counters.model_fit_max_ns,
                started.elapsed(),
            );
            match candidate {
                Ok(candidate) => {
                    shared.counters.model_fits.fetch_add(1, Ordering::Relaxed);
                    domain.active = Some(candidate);
                    self.model_revision = next_revision;
                    changed = true;
                }
                Err(_) => {
                    // Keep the previous coherent model/calibrator, with its
                    // original model expiry, rather than publishing half a fit.
                    shared
                        .counters
                        .training_failures
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        changed
    }

    pub fn snapshot(&mut self, now: Instant) -> Option<snapshot::Snapshot> {
        let revision = self.revision.checked_add(1)?;
        self.revision = revision;
        Some(self.current_snapshot(now))
    }

    pub fn current_snapshot(&self, now: Instant) -> snapshot::Snapshot {
        let mut domains = Vec::with_capacity(self.domains.len());
        for domain in &self.domains {
            if let Some(active) = &domain.active {
                domains.push(snapshot::Domain {
                    model: Arc::clone(&active.model),
                    model_revision: active.model_revision,
                    model_expires_at: active.model_expires_at,
                    calibration: active.calibration.snapshot(now),
                });
            }
        }
        snapshot::Snapshot {
            revision: self.revision,
            query: self.config.query,
            identity: self.identity.clone(),
            boundaries: self.config.work_bucket_boundaries.clone().into(),
            quantile: self.config.training.risk_quantile,
            domains,
            rejections: None,
        }
    }
}

fn record_duration(
    total: &std::sync::atomic::AtomicU64,
    maximum: &std::sync::atomic::AtomicU64,
    duration: Duration,
) {
    let nanos = duration.as_nanos().min(u64::MAX as u128) as u64;
    let _ = total.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_add(nanos))
    });
    maximum.fetch_max(nanos, Ordering::Relaxed);
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

fn fit(
    config: &LearningConfig,
    identity: &ExecutionIdentity,
    family: ModelFamily,
    domain: &Domain,
    revision: u64,
    now: Instant,
) -> Result<Active, ModelError> {
    let observations =
        super::width_history::fit_observations(config, &domain.window, &domain.width_history, now)?;
    let training_end = config.training_samples;
    let fitting_end = training_end + config.validation_samples;
    let samples: Vec<_> = observations[..fitting_end]
        .iter()
        .map(|o| o.sample)
        .collect();
    let mut model = FittedModel::train(
        family,
        identity.clone(),
        &samples[..training_end],
        &samples[training_end..],
        &config.training,
    )?;
    let evidence_time = match config.training_history {
        super::TrainingHistoryPolicy::Recent => observations[training_end - 1].completed_at,
        // Every retained support point must expire on its original clock.
        // A new fit must not use current labels to renew old width evidence.
        super::TrainingHistoryPolicy::WidthStratified { .. } => observations
            .iter()
            .map(|observation| observation.completed_at)
            .min()
            .expect("populated fit partitions"),
    };
    let model_expires_at = evidence_time
        .checked_add(Duration::from_millis(config.model_ttl_ms))
        .filter(|expires| *expires > now)
        .ok_or(ModelError::InvalidData("training evidence expired"))?;
    let fitting_through = observations[fitting_end - 1].sequence;
    let mut calibration = OnlineCalibrator::new(config.calibration_config(), revision)
        .map_err(ModelError::Configuration)?;
    for observation in &observations[fitting_end..] {
        if observation.sequence <= fitting_through {
            return Err(ModelError::InvalidData(
                "overlapping fit/calibration history",
            ));
        }
        // The suffix supplies out-of-fit residuals. Feature qualification does
        // not inspect its labels or change fitted coefficients/tree selection.
        let Ok(base) = model.qualify_observed_features(identity, &observation.sample.features)
        else {
            continue;
        };
        calibration.observe(
            revision,
            domain.key,
            config.bucket(observation.sample.features.work_tokens()),
            observation.sample.service_seconds,
            base.risk_base_seconds,
            observation.completed_at,
        );
    }
    let snapshot = calibration.snapshot(now);
    // A bucket outside the configured finite list forces same-domain global
    // qualification. Never install an uncalibrated new model next to old errors.
    if snapshot
        .correction(revision, &domain.key, u32::MAX, now)
        .is_none()
    {
        return Err(ModelError::InvalidData(
            "candidate lacks fresh out-of-fit calibration",
        ));
    }
    Ok(Active {
        model: Arc::new(model),
        model_revision: revision,
        model_expires_at,
        calibration,
        fitting_through,
        calibration_after: None,
    })
}
