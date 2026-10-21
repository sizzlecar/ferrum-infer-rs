//! Common asynchronous learning and immutable publication for both base models.
//! A published model and its signed residuals always have the same revision.
//! Final performance evaluation data are not inputs to this training interface.

mod extrema_history;
mod owner;
mod rejection;
mod snapshot;
mod state;
mod telemetry;
mod width_history;
mod worker;

pub use ferrum_types::TrainingHistoryPolicy;
pub use owner::{AutoFallbackReason, FamilyResolution, LearningEstimatorOwner};
pub use rejection::{LearningQueryRejectionStats, OutsideSupportWitness, RejectionWork};
pub use state::{LearningStateError, StateLoadReport, StateLoadStatus, StateSaveReport};
pub use telemetry::{DomainPredictionStats, LearningPredictionStats};

use crate::{
    calibration::CalibrationConfig,
    model::{ExecutionIdentity, ModelFamily, TrainingBudget},
    CollectionConfig, EstimatorError,
};
use ferrum_scheduler::cost_prediction::CostQuery;
use ferrum_types::CostQueryLimits;
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearningConfig {
    pub collection: CollectionConfig,
    pub query: CostQueryLimits,
    pub training: TrainingBudget,
    /// At most the nine phase/reuse combinations for one execution identity.
    pub max_domains: usize,
    pub window_samples: usize,
    pub training_history: TrainingHistoryPolicy,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub calibration_samples: usize,
    pub min_calibration_samples: usize,
    pub retrain_every: usize,
    pub calibration_ttl_ms: u64,
    pub model_ttl_ms: u64,
    /// Inclusive upper bounds on requested product tokens, not physical tokens.
    pub work_bucket_boundaries: Vec<usize>,
    /// Retained learning vectors/models and the one source-local rejection
    /// recorder, excluding collection, fit scratch, cold diagnostic copies and
    /// caller-retained old snapshots; allocator/RSS is measured independently.
    pub max_retained_bytes: usize,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            collection: CollectionConfig::default(),
            query: CostQueryLimits::default(),
            training: TrainingBudget::default(),
            max_domains: 9,
            window_samples: 512,
            training_history: TrainingHistoryPolicy::Recent,
            training_samples: 128,
            validation_samples: 32,
            calibration_samples: 32,
            min_calibration_samples: 16,
            retrain_every: 64,
            calibration_ttl_ms: 30_000,
            model_ttl_ms: 900_000,
            work_bucket_boundaries: vec![1, 16, 64, 256, 1024],
            max_retained_bytes: 32 * 1024 * 1024,
        }
    }
}

impl LearningConfig {
    pub fn validate(&self) -> Result<(), EstimatorError> {
        let invalid = |message: &str| EstimatorError::Configuration(message.into());
        self.collection.validate()?;
        self.training_history
            .validate()
            .map_err(|message| invalid(&message))?;
        self.query
            .validate()
            .map_err(EstimatorError::Configuration)?;
        self.training
            .validate()
            .map_err(|e| invalid(&e.to_string()))?;
        if self.max_domains == 0
            || self.max_domains > 9
            || self.window_samples > 4096
            || self.training_samples < 2
            || self.validation_samples == 0
            || self.min_calibration_samples == 0
            || self.calibration_samples < self.min_calibration_samples
            || self.retrain_every == 0
            || self.work_bucket_boundaries.len() > 8
            || self.work_bucket_boundaries.iter().any(|n| *n == 0)
            || self.work_bucket_boundaries.windows(2).any(|w| w[0] >= w[1])
            || self.collection.max_participants > 256
            || self.collection.queue_capacity > 4096
            || self.collection.max_queue_bytes > 32 * 1024 * 1024
            || self.query.max_participants > self.collection.max_participants
            || !CostQuery::fits(self.query.max_participants, self.query)
        {
            return Err(invalid("invalid learning sample/domain/query bounds"));
        }
        let fit = self
            .training_samples
            .checked_add(self.validation_samples)
            .filter(|n| *n <= self.training.max_samples)
            .ok_or_else(|| invalid("fit samples exceed training budget"))?;
        fit.checked_add(self.calibration_samples)
            .filter(|n| *n <= self.window_samples)
            .ok_or_else(|| invalid("disjoint partitions exceed the sample window"))?;
        self.calibration_config().validate().map_err(invalid)?;
        if self.model_ttl_ms == 0
            || Instant::now()
                .checked_add(Duration::from_millis(self.model_ttl_ms))
                .is_none()
            || Instant::now()
                .checked_add(Duration::from_millis(self.calibration_ttl_ms))
                .is_none()
        {
            return Err(invalid("invalid model or calibration TTL"));
        }
        let models = self
            .training
            .owned_bytes_bound(0)
            .map_err(|e| invalid(&e.to_string()))?;
        let calibration = crate::calibration::OnlineCalibrator::<crate::model::FeatureDomain>::retained_bytes_bound(self.calibration_config())
            .ok_or_else(|| invalid("calibration memory bound overflow"))?;
        let history_samples = self
            .width_history_capacity()
            .and_then(|count| count.checked_add(self.extrema_history_capacity()))
            .ok_or_else(|| invalid("width history bound overflow"))?;
        let rejection_bytes =
            rejection::Recorder::retained_bytes_bound(self.query.max_participants)
                .ok_or_else(|| invalid("rejection recorder memory bound overflow"))?;
        let per_domain = self
            .window_samples
            .checked_add(history_samples)
            .ok_or_else(|| invalid("width history bound overflow"))?
            .checked_mul(std::mem::size_of::<worker::Observation>())
            .and_then(|n| n.checked_add(models))
            .and_then(|n| n.checked_add(calibration))
            .and_then(|n| n.checked_add(std::mem::size_of::<worker::Domain>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<snapshot::Domain>()))
            .ok_or_else(|| invalid("retained memory bound overflow"))?;
        let retained = per_domain
            .checked_mul(self.max_domains)
            // The single worker can hold one temporary qualified model copy
            // before publishing it with its accepted same-version residual.
            .and_then(|n| n.checked_add(models))
            .and_then(|n| {
                n.checked_add(
                    2 * 64 * 1024
                        + self.work_bucket_boundaries.len() * std::mem::size_of::<usize>(),
                )
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<telemetry::Telemetry>()))
            .and_then(|n| n.checked_add(rejection_bytes))
            .ok_or_else(|| invalid("retained memory bound overflow"))?;
        if self.max_retained_bytes > 32 * 1024 * 1024 || retained > self.max_retained_bytes {
            return Err(invalid("learning retained memory budget exceeded"));
        }
        let fit_workspace = self
            .training
            .owned_bytes_bound(fit)
            .map_err(|e| invalid(&e.to_string()))?
            .checked_add(self.required_samples() * std::mem::size_of::<worker::Observation>())
            .and_then(|n| {
                self.width_training_scratch_bytes()
                    .and_then(|bytes| n.checked_add(bytes))
            })
            .and_then(|n| n.checked_add(fit * std::mem::size_of::<crate::model::TrainingSample>()))
            .and_then(|n| {
                n.checked_add(
                    self.collection.max_participants
                        * std::mem::size_of::<ferrum_scheduler::cost_prediction::CostWork>(),
                )
            })
            .ok_or_else(|| invalid("fit workspace bound overflow"))?;
        if fit_workspace > self.training.max_owned_bytes {
            return Err(invalid("learning fit memory budget exceeded"));
        }
        Ok(())
    }

    fn calibration_config(&self) -> CalibrationConfig {
        CalibrationConfig {
            max_domains: 1,
            buckets_per_domain: self.work_bucket_boundaries.len() + 1,
            window_samples: self.calibration_samples,
            min_samples: self.min_calibration_samples,
            ttl: Duration::from_millis(self.calibration_ttl_ms),
            quantile: self.training.risk_quantile,
        }
    }

    fn bucket(&self, work: f64) -> u32 {
        self.work_bucket_boundaries
            .partition_point(|limit| (*limit as f64) < work) as u32
    }

    fn required_samples(&self) -> usize {
        self.training_samples + self.validation_samples + self.calibration_samples
    }

    fn width_history_capacity(&self) -> Option<usize> {
        match self.training_history {
            TrainingHistoryPolicy::Recent => Some(0),
            TrainingHistoryPolicy::WidthStratified { samples_per_width }
            | TrainingHistoryPolicy::WidthAndExtrema { samples_per_width } => self
                .collection
                .max_participants
                .checked_mul(samples_per_width),
        }
    }

    fn extrema_history_capacity(&self) -> usize {
        if matches!(
            self.training_history,
            TrainingHistoryPolicy::WidthAndExtrema { .. }
        ) {
            extrema_history::MAX_EXTREMA
        } else {
            0
        }
    }

    fn width_training_scratch_bytes(&self) -> Option<usize> {
        if self.training_history == TrainingHistoryPolicy::Recent {
            return Some(0);
        }
        self.window_samples
            .checked_add(self.width_history_capacity()?)?
            .checked_add(self.extrema_history_capacity())?
            .checked_mul(std::mem::size_of::<worker::Observation>())?
            .checked_add(
                self.collection
                    .max_participants
                    .checked_mul(std::mem::size_of::<(usize, usize)>())?,
            )
            .and_then(|bytes| {
                let cursor_bytes = if self.extrema_history_capacity() == 0 {
                    0
                } else {
                    self.collection
                        .max_participants
                        .checked_mul(std::mem::size_of::<(usize, usize)>())?
                };
                bytes.checked_add(cursor_bytes)
            })
    }
}

pub struct LearningEstimator;

impl LearningEstimator {
    /// Restore the family only from a fully validated, fresh state bound to the
    /// actual execution identity. Cold fallback is explicit and makes no claim
    /// that Physical won a comparative benchmark for this configuration.
    pub fn spawn_auto_with_state(
        config: LearningConfig,
        identity: ExecutionIdentity,
        path: Option<&std::path::Path>,
    ) -> Result<LearningEstimatorOwner, LearningStateError> {
        config.validate()?;
        let restored = path
            .map(|path| state::load_selected(path, &config, &identity, None))
            .transpose()?
            .flatten();
        let report = restored.as_ref().map(|state| state.report).or_else(|| {
            path.map(|_| StateLoadReport {
                status: StateLoadStatus::Missing,
                loaded_domains: 0,
                expired_domains: 0,
                expired_residual_entries: 0,
            })
        });
        let (family, resolution) = match restored.as_ref() {
            Some(state) if state.report.status == StateLoadStatus::Loaded => (
                state.family,
                FamilyResolution::AutoRestoredBoundState {
                    family: state.family,
                },
            ),
            _ => (
                ModelFamily::Physical,
                FamilyResolution::AutoPhysicalFallback {
                    reason: match report.map(|report| report.status) {
                        Some(StateLoadStatus::Missing) => AutoFallbackReason::MissingState,
                        Some(StateLoadStatus::Stale) => AutoFallbackReason::StaleState,
                        _ => AutoFallbackReason::NoBoundStateProvided,
                    },
                },
            ),
        };
        let mut owner = Self::spawn_restored(
            config,
            identity,
            family,
            crate::worker::Hooks::default(),
            restored,
            report,
        )?;
        owner.resolution = resolution;
        Ok(owner)
    }

    pub fn spawn_with_state(
        config: LearningConfig,
        identity: ExecutionIdentity,
        family: ModelFamily,
        path: Option<&std::path::Path>,
    ) -> Result<LearningEstimatorOwner, LearningStateError> {
        config.validate()?;
        let restored = path
            .map(|path| state::load(path, &config, &identity, family))
            .transpose()?
            .flatten();
        let report = restored.as_ref().map(|state| state.report).or_else(|| {
            path.map(|_| StateLoadReport {
                status: StateLoadStatus::Missing,
                loaded_domains: 0,
                expired_domains: 0,
                expired_residual_entries: 0,
            })
        });
        Self::spawn_restored(
            config,
            identity,
            family,
            crate::worker::Hooks::default(),
            restored,
            report,
        )
        .map_err(Into::into)
    }

    pub fn spawn(
        config: LearningConfig,
        identity: ExecutionIdentity,
        family: ModelFamily,
    ) -> Result<LearningEstimatorOwner, EstimatorError> {
        Self::spawn_inner(config, identity, family, crate::worker::Hooks::default())
    }

    fn spawn_inner(
        config: LearningConfig,
        identity: ExecutionIdentity,
        family: ModelFamily,
        hooks: crate::worker::Hooks,
    ) -> Result<LearningEstimatorOwner, EstimatorError> {
        Self::spawn_restored(config, identity, family, hooks, None, None)
    }

    fn spawn_restored(
        config: LearningConfig,
        identity: ExecutionIdentity,
        family: ModelFamily,
        hooks: crate::worker::Hooks,
        restored: Option<state::RestoredState>,
        report: Option<StateLoadReport>,
    ) -> Result<LearningEstimatorOwner, EstimatorError> {
        config.validate()?;
        let (source, revisions) = snapshot::Source::new(
            config.query,
            identity.clone(),
            config.work_bucket_boundaries.clone(),
        );
        let source = Arc::new(source);
        let output = Arc::clone(&source);
        let telemetry = Arc::new(telemetry::Telemetry::new(
            family,
            config.training.risk_quantile,
        ));
        let worker_telemetry = telemetry.clone();
        let mut learner = match restored {
            Some(restored) => worker::Learner::restore(config.clone(), identity, family, restored),
            None => worker::Learner::new(config.clone(), identity, family),
        };
        if report.is_some_and(|report| report.status != StateLoadStatus::Missing) {
            let initial = learner.current_snapshot(Instant::now());
            let revision = initial.revision;
            source.install(initial);
            revisions.send_replace(revision);
        }
        let common = crate::spawn_owner(
            config.collection,
            source.clone(),
            move |shared, receiver, wake| {
                crate::worker::receive(Arc::clone(&shared), receiver, wake, hooks, |sample| {
                    let now = Instant::now();
                    if let Ok(features) = crate::model::WorkFeatures::extract_iter(
                        sample.participants.iter().map(crate::worker::sample_work),
                        shared.config.max_participants,
                    ) {
                        worker_telemetry.feedback(features, sample.completed_at, now);
                    }
                    if learner.apply(&shared, sample, now) {
                        if let Some(snapshot) = learner.snapshot(Instant::now()) {
                            let revision = snapshot.revision;
                            output.install(snapshot);
                            revisions.send_replace(revision);
                            shared
                                .counters
                                .published_snapshots
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                });
            },
        )?;
        Ok(LearningEstimatorOwner {
            common,
            source,
            config,
            family,
            resolution: FamilyResolution::Explicit { family },
            telemetry,
            load_report: report,
        })
    }
}
