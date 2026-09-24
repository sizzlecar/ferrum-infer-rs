//! Bounded retrospective audit of ordinary selected-model observations.
//! No training, residual update, validity decision or execution permission.
use super::*;
use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown;
use model::statistical::model::ModelUnknown;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum SelectedUnknownReason {
    Evidence(StatisticalEvidenceUnknown),
    InvalidSettings,
    InvalidSample,
    WrongFingerprint,
    WrongSource,
    PhaseLeakage,
    DuplicateRecord,
    Capacity,
    Clock,
    Stale,
    InsufficientFit,
    InsufficientResidual,
    FamilyMissing,
    JointSupport,
    Numerical,
}
impl SelectedUnknownReason {
    const ALL: [Self; 24] = {
        use StatisticalEvidenceUnknown as E;
        [
            Self::Evidence(E::MissingProducer),
            Self::Evidence(E::InvalidAlgorithm),
            Self::Evidence(E::InvalidWork),
            Self::Evidence(E::CommandMismatch),
            Self::Evidence(E::ExactBindingMismatch),
            Self::Evidence(E::MissingHostDomain),
            Self::Evidence(E::UnsupportedWave),
            Self::Evidence(E::UnsupportedReplay),
            Self::Evidence(E::Capacity),
            Self::Evidence(E::Overflow),
            Self::InvalidSettings,
            Self::InvalidSample,
            Self::WrongFingerprint,
            Self::WrongSource,
            Self::PhaseLeakage,
            Self::DuplicateRecord,
            Self::Capacity,
            Self::Clock,
            Self::Stale,
            Self::InsufficientFit,
            Self::InsufficientResidual,
            Self::FamilyMissing,
            Self::JointSupport,
            Self::Numerical,
        ]
    };
}
impl From<ModelUnknown> for SelectedUnknownReason {
    fn from(reason: ModelUnknown) -> Self {
        use ModelUnknown as M;
        match reason {
            M::Evidence(reason) => Self::Evidence(reason),
            M::InvalidSettings => Self::InvalidSettings,
            M::InvalidSample => Self::InvalidSample,
            M::WrongFingerprint => Self::WrongFingerprint,
            M::WrongSource => Self::WrongSource,
            M::PhaseLeakage => Self::PhaseLeakage,
            M::DuplicateRecord => Self::DuplicateRecord,
            M::Capacity => Self::Capacity,
            M::Clock => Self::Clock,
            M::Stale => Self::Stale,
            M::InsufficientFit => Self::InsufficientFit,
            M::InsufficientResidual => Self::InsufficientResidual,
            M::FamilyMissing => Self::FamilyMissing,
            M::JointSupport => Self::JointSupport,
            M::Numerical => Self::Numerical,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum SelectedNotCompleted {
    NotSubmitted,
    Deferred,
    FailedAfterSubmit,
    PartiallyCompleted,
}
impl SelectedNotCompleted {
    const ALL: [Self; 4] = [
        Self::NotSubmitted,
        Self::Deferred,
        Self::FailedAfterSubmit,
        Self::PartiallyCompleted,
    ];
}

/// Scalar diagnostic only: no retained request, model, buffer or execution lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct SelectedServingComparison {
    pub accepted_ordinal: u64,
    pub call_id: u64,
    pub family_signature: [u8; 32],
    pub terminal: bool,
    pub observed_at_ns: u64,
    pub consumed_at_ns: u64,
    pub model_version: u64,
    pub planning_ns: u64,
    pub actual_ns: u64,
    pub underestimate_ns: u64,
}

pub(in crate::continuous_engine::inner::cost_observation) enum SelectedServingEvaluation {
    NotCompleted(SelectedNotCompleted),
    InvalidActual(ModelUnknown),
    Complete {
        terminal: bool,
        // None is a genuinely absent startup publication, not zero cost.
        prediction: Option<Result<SelectedServingComparison, ModelUnknown>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct SelectedServingAudit {
    pub scope: &'static str,
    /// All selected-mode entries actually drained, including missing/invalid data.
    /// Pre-queue offers and drops remain in ObservationFunnelSnapshot.sink.
    pub drained_entries: u64,
    pub not_completed: [ReasonCount<SelectedNotCompleted>; 4],
    pub actual_unavailable: [ReasonCount<SelectedUnknownReason>; 24],
    pub constructable_actual: u64,
    pub constructable_terminal: u64,
    pub no_published_model: u64,
    pub prediction_unknown: [ReasonCount<SelectedUnknownReason>; 24],
    pub known_compared: u64,
    pub terminal_compared: u64,
    pub underestimates: u64,
    pub total_underestimate_ns: u64,
    pub max_underestimate_ns: u64,
}
impl Default for SelectedServingAudit {
    fn default() -> Self {
        Self::filled(0)
    }
}
impl SelectedServingAudit {
    pub(super) fn filled(count: u64) -> Self {
        Self {
            scope: "ordinary selected-model FIFO entries; retrospective complete actual host-settled waves against the immutable pre-drain model at real consumption time, not the pre-submit prediction, client-visible SLO or drift control; sink.entries_offered/published/dropped_* remain the full queue population; missing, failed, non-submitted and expired evidence never has zero measured error",
            drained_entries: count,
            not_completed: SelectedNotCompleted::ALL.map(|reason| ReasonCount { reason, count }),
            actual_unavailable: SelectedUnknownReason::ALL.map(|reason| ReasonCount { reason, count }),
            constructable_actual: count,
            constructable_terminal: count,
            no_published_model: count,
            prediction_unknown: SelectedUnknownReason::ALL.map(|reason| ReasonCount { reason, count }),
            known_compared: count,
            terminal_compared: count,
            underestimates: count,
            total_underestimate_ns: count,
            max_underestimate_ns: count,
        }
    }
    pub fn record(&mut self, result: SelectedServingEvaluation, exhausted: &mut bool) {
        add_one(&mut self.drained_entries, exhausted);
        let (terminal, prediction) = match result {
            SelectedServingEvaluation::NotCompleted(reason) => {
                add_one(&mut self.not_completed[reason as usize].count, exhausted);
                return;
            }
            SelectedServingEvaluation::InvalidActual(reason) => {
                record_reason(&mut self.actual_unavailable, reason, exhausted);
                return;
            }
            SelectedServingEvaluation::Complete {
                terminal,
                prediction,
            } => (terminal, prediction),
        };
        add_one(&mut self.constructable_actual, exhausted);
        if terminal {
            add_one(&mut self.constructable_terminal, exhausted);
        }
        let value = match prediction {
            None => {
                add_one(&mut self.no_published_model, exhausted);
                return;
            }
            Some(Err(reason)) => {
                record_reason(&mut self.prediction_unknown, reason, exhausted);
                return;
            }
            Some(Ok(value)) => value,
        };
        add_one(&mut self.known_compared, exhausted);
        if terminal {
            add_one(&mut self.terminal_compared, exhausted);
        }
        if value.underestimate_ns > 0 {
            add_one(&mut self.underestimates, exhausted);
            self.max_underestimate_ns = self.max_underestimate_ns.max(value.underestimate_ns);
            match self
                .total_underestimate_ns
                .checked_add(value.underestimate_ns)
            {
                Some(sum) => self.total_underestimate_ns = sum,
                None => *exhausted = true,
            }
        }
    }
}
fn record_reason(
    counts: &mut [ReasonCount<SelectedUnknownReason>; 24],
    reason: ModelUnknown,
    exhausted: &mut bool,
) {
    let reason = SelectedUnknownReason::from(reason);
    let count = counts
        .iter_mut()
        .find(|item| item.reason == reason)
        .expect("all typed selected reasons have a fixed audit counter");
    add_one(&mut count.count, exhausted);
}
