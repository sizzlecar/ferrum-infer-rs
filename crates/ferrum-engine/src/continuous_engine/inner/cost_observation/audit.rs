//! Fixed-cardinality diagnostics. Counts describe instrumented observations,
//! never infer unseen executor waves or calibrated coverage from retained data.
use super::CostSampleStats;
use ferrum_scheduler::implementations::continuous::{cost_model as model, cost_profile as profile};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod host_content;
mod prediction;
mod presubmit;
mod selected;
pub(super) use host_content::{HostContentAudit, HostContentEvaluation, HostContentRejection};
use prediction::PredictionAuditSnapshot;
pub(super) use prediction::{observed_cost, PreUpdatePrediction};
#[cfg(test)]
pub(super) use selected::SelectedUnknownReason;
pub(super) use selected::{
    SelectedNotCompleted, SelectedServingAudit, SelectedServingComparison,
    SelectedServingEvaluation,
};

pub(super) fn increment(value: &AtomicU64, exhausted: &AtomicBool) {
    if value
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .is_err()
    {
        exhausted.store(true, Ordering::Relaxed);
    }
}

pub(super) fn add_one(value: &mut u64, exhausted: &mut bool) {
    match value.checked_add(1) {
        Some(next) => *value = next,
        None => *exhausted = true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TrainingSkipReason {
    NotSubmitted,
    Deferred,
    FailedAfterSubmit,
    UnisolatedPartial,
    MissingDeviceTiming,
}
impl TrainingSkipReason {
    const ALL: [Self; 5] = [
        Self::NotSubmitted,
        Self::Deferred,
        Self::FailedAfterSubmit,
        Self::UnisolatedPartial,
        Self::MissingDeviceTiming,
    ];
}
impl From<model::ObservationSkipReason> for TrainingSkipReason {
    fn from(value: model::ObservationSkipReason) -> Self {
        use model::ObservationSkipReason as R;
        match value {
            R::NotSubmitted => Self::NotSubmitted,
            R::Deferred => Self::Deferred,
            R::FailedAfterSubmit => Self::FailedAfterSubmit,
            R::UnisolatedPartial => Self::UnisolatedPartial,
            R::MissingDeviceTiming => Self::MissingDeviceTiming,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TrainingErrorReason {
    InvalidSettings,
    FingerprintMismatch,
    InvalidShape,
    InvalidTiming,
    ClockMovedBackwards,
    CapacityExceeded,
    ArithmeticOverflow,
    VersionExhausted,
    MarginDecreaseRequiresRecalibration,
    ImportedClock,
    ImportedProfileContract,
}
impl TrainingErrorReason {
    const ALL: [Self; 11] = [
        Self::InvalidSettings,
        Self::FingerprintMismatch,
        Self::InvalidShape,
        Self::InvalidTiming,
        Self::ClockMovedBackwards,
        Self::CapacityExceeded,
        Self::ArithmeticOverflow,
        Self::VersionExhausted,
        Self::MarginDecreaseRequiresRecalibration,
        Self::ImportedClock,
        Self::ImportedProfileContract,
    ];
}
impl From<model::CostModelError> for TrainingErrorReason {
    fn from(value: model::CostModelError) -> Self {
        use model::CostModelError as E;
        match value {
            E::InvalidSettings(_) => Self::InvalidSettings,
            E::FingerprintMismatch => Self::FingerprintMismatch,
            E::InvalidShape(_) => Self::InvalidShape,
            E::InvalidTiming(_) => Self::InvalidTiming,
            E::ClockMovedBackwards => Self::ClockMovedBackwards,
            E::CapacityExceeded(_) => Self::CapacityExceeded,
            E::ArithmeticOverflow => Self::ArithmeticOverflow,
            E::VersionExhausted => Self::VersionExhausted,
            E::MarginDecreaseRequiresRecalibration => Self::MarginDecreaseRequiresRecalibration,
        }
    }
}
impl From<profile::CostProfileError> for TrainingErrorReason {
    fn from(value: profile::CostProfileError) -> Self {
        match value {
            profile::CostProfileError::Model(error)
            | profile::CostProfileError::Sample { reason: error, .. } => error.into(),
            profile::CostProfileError::Clock(_) => Self::ImportedClock,
            _ => Self::ImportedProfileContract,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum TrainingDisposition {
    Recorded,
    Skipped { reason: TrainingSkipReason },
    Error { reason: TrainingErrorReason },
    Unavailable,
}
impl TrainingDisposition {
    pub fn from_result(
        result: Option<Result<model::ObservationDisposition, TrainingErrorReason>>,
    ) -> Self {
        match result {
            Some(Ok(model::ObservationDisposition::Recorded)) => Self::Recorded,
            Some(Ok(model::ObservationDisposition::Skipped(reason))) => Self::Skipped {
                reason: reason.into(),
            },
            Some(Err(reason)) => Self::Error { reason },
            None => Self::Unavailable,
        }
    }
    pub const fn recorded(self) -> bool {
        matches!(self, Self::Recorded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct ReasonCount<R> {
    pub reason: R,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct TrainingCounts {
    pub recorded: u64,
    pub skipped: [ReasonCount<TrainingSkipReason>; 5],
    pub errors: [ReasonCount<TrainingErrorReason>; 11],
    pub unavailable: u64,
}
impl Default for TrainingCounts {
    fn default() -> Self {
        Self::filled(0)
    }
}
impl TrainingCounts {
    fn filled(count: u64) -> Self {
        Self {
            recorded: count,
            unavailable: count,
            skipped: TrainingSkipReason::ALL.map(|reason| ReasonCount { reason, count }),
            errors: TrainingErrorReason::ALL.map(|reason| ReasonCount { reason, count }),
        }
    }
    pub fn record(&mut self, disposition: TrainingDisposition, exhausted: &mut bool) {
        let count = match disposition {
            TrainingDisposition::Recorded => &mut self.recorded,
            TrainingDisposition::Skipped { reason } => &mut self.skipped[reason as usize].count,
            TrainingDisposition::Error { reason } => &mut self.errors[reason as usize].count,
            TrainingDisposition::Unavailable => &mut self.unavailable,
        };
        add_one(count, exhausted);
    }
}

pub(super) const WAVE_COUNT: usize = 5;
pub(super) const WAVE_NAMES: [&str; WAVE_COUNT] =
    ["decode", "prefill", "mixed", "restore", "maintenance"];
pub(super) fn wave_index(kind: model::WaveKind) -> usize {
    match kind {
        model::WaveKind::Decode => 0,
        model::WaveKind::Prefill => 1,
        model::WaveKind::Mixed => 2,
        model::WaveKind::Restore => 3,
        model::WaveKind::Maintenance => 4,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct WaveTrainingCounts {
    pub wave: &'static str,
    pub outcomes: TrainingCounts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine) struct TrainingAuditSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) selected_serving: Option<SelectedServingAudit>,
    pub(super) host_content: HostContentAudit,
    pub(super) consumed: u64,
    pub(super) outcomes: TrainingCounts,
    pub(super) by_wave: [WaveTrainingCounts; WAVE_COUNT],
    pub(super) pre_update_prediction: PredictionAuditSnapshot,
    pub(super) publish_attempts: u64,
    pub(super) published_snapshots: u64,
    pub(super) publish_errors: [ReasonCount<TrainingErrorReason>; 11],
    pub(super) counter_exhausted: bool,
}
impl Default for TrainingAuditSnapshot {
    fn default() -> Self {
        Self::filled(0)
    }
}
impl TrainingAuditSnapshot {
    fn filled(count: u64) -> Self {
        Self {
            selected_serving: None,
            host_content: HostContentAudit::filled(count),
            consumed: count,
            outcomes: TrainingCounts::filled(count),
            by_wave: WAVE_NAMES.map(|wave| WaveTrainingCounts {
                wave,
                outcomes: TrainingCounts::filled(count),
            }),
            pre_update_prediction: PredictionAuditSnapshot::filled(count),
            publish_attempts: count,
            published_snapshots: count,
            publish_errors: TrainingErrorReason::ALL.map(|reason| ReasonCount { reason, count }),
            counter_exhausted: count == u64::MAX,
        }
    }
    pub fn maximum_serialized_counts() -> Self {
        let mut value = Self::filled(u64::MAX);
        value.selected_serving = Some(SelectedServingAudit::filled(u64::MAX));
        // Reserve JSON's longer false spelling, not only maximum integers.
        value.counter_exhausted = false;
        value
    }
    pub fn training_rejected(&self) -> u64 {
        self.outcomes
            .skipped
            .iter()
            .map(|r| r.count)
            .chain(self.outcomes.errors.iter().map(|r| r.count))
            .fold(self.outcomes.unavailable, u64::saturating_add)
    }
    pub fn publish_rejected(&self) -> u64 {
        self.publish_errors
            .iter()
            .map(|r| r.count)
            .fold(0, u64::saturating_add)
    }
    pub(super) fn observe(&mut self, kind: model::WaveKind, disposition: TrainingDisposition) {
        add_one(&mut self.consumed, &mut self.counter_exhausted);
        self.outcomes
            .record(disposition, &mut self.counter_exhausted);
        self.by_wave[wave_index(kind)]
            .outcomes
            .record(disposition, &mut self.counter_exhausted);
    }
    pub(super) fn predict(&mut self, kind: model::WaveKind, prediction: PreUpdatePrediction) {
        self.pre_update_prediction
            .record(kind, prediction, &mut self.counter_exhausted);
    }
    pub(super) fn publish(&mut self, result: Result<(), TrainingErrorReason>) {
        add_one(&mut self.publish_attempts, &mut self.counter_exhausted);
        let count = match result {
            Ok(()) => &mut self.published_snapshots,
            Err(reason) => &mut self.publish_errors[reason as usize].count,
        };
        add_one(count, &mut self.counter_exhausted);
    }
}

/// A live snapshot is not atomic across producers and worker. Only a quiescent
/// final summary may be reconciled as a settled funnel, and neither establishes
/// that every physical executor path was instrumented.
#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine) struct ObservationFunnelSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prospective_capture: Option<super::prospective_capture::CaptureAuditSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_feedback: Option<super::selected_feedback::FeedbackAudit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_feedback: Option<super::selected_feedback::FeedbackAudit>,
    pub scope: &'static str,
    pub sink: CostSampleStats,
    pub training: TrainingAuditSnapshot,
    pub export: super::profile_export::ExportAuditSnapshot,
}

#[cfg(test)]
mod tests;
