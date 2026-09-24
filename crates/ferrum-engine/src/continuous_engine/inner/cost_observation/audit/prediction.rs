//! Retrospective actual-shape queries against one immutable pre-batch model.
//! This is neither pre-execution candidate coverage nor evidence of a benefit
//! from lookahead. Queue losses never receive fabricated query outcomes.
use super::super::profile::EngineCostSnapshot;
use super::{add_one, ReasonCount, TrainingDisposition, TrainingErrorReason};
use ferrum_scheduler::implementations::continuous::cost_model as model;
use serde::Serialize;

pub(super) const SCOPE: &str = "retrospective actual-shape queries at original receipt time against the model published before this training batch; all drained completed observations including training rejects; queued losses are unaudited and remain in the independent offered denominator; not pre-execution candidate coverage or lookahead benefit";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum PredictionUnknownReason {
    FingerprintMismatch,
    InvalidShape,
    UnobservedBucket,
    InsufficientSamples,
    OutsideObservedCoverage,
    StaleSamples,
    ClockMovedBackwards,
    NumericFeaturesMissing,
    NumericRowOrderUnsupported,
    OutsideJointNumericSupport,
    HostContentFeaturesMissing,
    BoundaryUnsupported,
}
impl PredictionUnknownReason {
    const ALL: [Self; 12] = [
        Self::FingerprintMismatch,
        Self::InvalidShape,
        Self::UnobservedBucket,
        Self::InsufficientSamples,
        Self::OutsideObservedCoverage,
        Self::StaleSamples,
        Self::ClockMovedBackwards,
        Self::NumericFeaturesMissing,
        Self::NumericRowOrderUnsupported,
        Self::OutsideJointNumericSupport,
        Self::HostContentFeaturesMissing,
        Self::BoundaryUnsupported,
    ];
}
impl From<model::CostUnknownReason> for PredictionUnknownReason {
    fn from(reason: model::CostUnknownReason) -> Self {
        use model::CostUnknownReason as R;
        match reason {
            R::FingerprintMismatch => Self::FingerprintMismatch,
            R::InvalidShape => Self::InvalidShape,
            R::UnobservedBucket => Self::UnobservedBucket,
            R::InsufficientSamples => Self::InsufficientSamples,
            R::OutsideObservedCoverage => Self::OutsideObservedCoverage,
            R::StaleSamples => Self::StaleSamples,
            R::ClockMovedBackwards => Self::ClockMovedBackwards,
            R::NumericFeaturesMissing => Self::NumericFeaturesMissing,
            R::NumericRowOrderUnsupported => Self::NumericRowOrderUnsupported,
            R::OutsideJointNumericSupport => Self::OutsideJointNumericSupport,
            R::HostContentFeaturesMissing => Self::HostContentFeaturesMissing,
            R::BoundaryUnsupported => Self::BoundaryUnsupported,
        }
    }
}

/// Fixed-size wire diagnostics: no shape, coverage Arc, clock origin or model
/// snapshot can accidentally be retained once this scalar record is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum PreUpdatePrediction {
    NotCompleted,
    NoPublishedModel,
    Unknown {
        reason: PredictionUnknownReason,
    },
    Known {
        typical_ns: u64,
        planning_ns: u64,
        model_version: u64,
        valid_for_ns: u64,
        observed_cost_ns: Option<u64>,
        underestimate_ns: Option<u64>,
    },
}

impl PreUpdatePrediction {
    /// Must run before observe() mutates the trainer; snapshot must be the same
    /// object for every observation in this batch, including rejected samples.
    pub fn query(
        snapshot: Option<&EngineCostSnapshot>,
        sample: &model::WaveCostObservation,
    ) -> Self {
        if sample.outcome != model::WaveObservationOutcome::Completed {
            return Self::NotCompleted;
        }
        let Some(snapshot) = snapshot else {
            return Self::NoPublishedModel;
        };
        match snapshot.predict(
            &sample.fingerprint,
            &sample.actual_shape,
            sample.boundary,
            sample.observed_at_ns,
        ) {
            model::CostPrediction::Unknown(reason) => Self::Unknown {
                reason: reason.into(),
            },
            model::CostPrediction::Known(value) => Self::Known {
                typical_ns: value.typical_ns,
                planning_ns: value.planning_ns,
                model_version: value.model_version,
                valid_for_ns: value.valid_for_ns,
                observed_cost_ns: None,
                underestimate_ns: None,
            },
        }
    }

    /// The old prediction is already fixed. The actual trainer's validation
    /// determines whether the measured cost is eligible for an error metric.
    /// InvalidTiming must never manufacture an underestimate, even if positive.
    pub fn with_comparison(mut self, cost: Option<u64>, disposition: TrainingDisposition) -> Self {
        let timing_validated = matches!(
            disposition,
            TrainingDisposition::Recorded
                | TrainingDisposition::Error {
                    reason: TrainingErrorReason::CapacityExceeded
                        | TrainingErrorReason::ArithmeticOverflow
                }
        );
        if timing_validated {
            if let (
                Some(cost),
                Self::Known {
                    planning_ns,
                    observed_cost_ns,
                    underestimate_ns,
                    ..
                },
            ) = (cost.filter(|cost| *cost > 0), &mut self)
            {
                *observed_cost_ns = Some(cost);
                *underestimate_ns = Some(cost.saturating_sub(*planning_ns));
            }
        }
        self
    }
}

pub(in crate::continuous_engine::inner::cost_observation) fn observed_cost(
    sample: &model::WaveCostObservation,
) -> Option<u64> {
    match sample.boundary {
        model::CostBoundary::PreparationToCommit
        | model::CostBoundary::PreparationToHostSettledV1 => Some(sample.timing.wall_total_ns),
        model::CostBoundary::DeviceOnly => sample.timing.device_elapsed_ns,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct PredictionAuditCounts {
    pub completed: u64,
    pub not_completed: u64,
    pub no_published_model: u64,
    pub unknown: [ReasonCount<PredictionUnknownReason>; 12],
    pub known: u64,
    pub compared: u64,
    pub underestimates: u64,
    pub max_underestimate_ns: u64,
}
impl PredictionAuditCounts {
    pub(super) fn filled(count: u64) -> Self {
        Self {
            completed: count,
            not_completed: count,
            no_published_model: count,
            unknown: PredictionUnknownReason::ALL.map(|reason| ReasonCount { reason, count }),
            known: count,
            compared: count,
            underestimates: count,
            max_underestimate_ns: count,
        }
    }
    pub(super) fn record(&mut self, prediction: PreUpdatePrediction, exhausted: &mut bool) {
        if prediction == PreUpdatePrediction::NotCompleted {
            add_one(&mut self.not_completed, exhausted);
            return;
        }
        add_one(&mut self.completed, exhausted);
        match prediction {
            PreUpdatePrediction::NotCompleted => unreachable!(),
            PreUpdatePrediction::NoPublishedModel => {
                add_one(&mut self.no_published_model, exhausted)
            }
            PreUpdatePrediction::Unknown { reason } => {
                add_one(&mut self.unknown[reason as usize].count, exhausted)
            }
            PreUpdatePrediction::Known {
                underestimate_ns, ..
            } => {
                add_one(&mut self.known, exhausted);
                if let Some(error) = underestimate_ns {
                    add_one(&mut self.compared, exhausted);
                    if error > 0 {
                        add_one(&mut self.underestimates, exhausted);
                    }
                    self.max_underestimate_ns = self.max_underestimate_ns.max(error);
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct PredictionAuditSnapshot {
    pub scope: &'static str,
    pub counts: PredictionAuditCounts,
    pub by_wave: [WavePredictionCounts; super::WAVE_COUNT],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct WavePredictionCounts {
    pub wave: &'static str,
    pub counts: PredictionAuditCounts,
}
impl PredictionAuditSnapshot {
    pub(super) fn filled(count: u64) -> Self {
        Self {
            scope: SCOPE,
            counts: PredictionAuditCounts::filled(count),
            by_wave: super::WAVE_NAMES.map(|wave| WavePredictionCounts {
                wave,
                counts: PredictionAuditCounts::filled(count),
            }),
        }
    }
    pub(super) fn record(
        &mut self,
        kind: model::WaveKind,
        prediction: PreUpdatePrediction,
        exhausted: &mut bool,
    ) {
        self.counts.record(prediction, exhausted);
        self.by_wave[super::wave_index(kind)]
            .counts
            .record(prediction, exhausted);
    }
}

#[cfg(test)]
mod tests;
