//! A separate denominator for the opt-in empirical host-settled population.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum HostContentRejection {
    MissingStages,
    Incomplete,
    IdentityMissing,
    DomainMissing,
    InvalidClock,
    LegacyFailure,
}
impl HostContentRejection {
    const ALL: [Self; 6] = [
        Self::MissingStages,
        Self::Incomplete,
        Self::IdentityMissing,
        Self::DomainMissing,
        Self::InvalidClock,
        Self::LegacyFailure,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct HostContentEvaluation {
    pub rejection: Option<HostContentRejection>,
    pub training: TrainingDisposition,
    pub pre_update_prediction: Option<PreUpdatePrediction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct HostContentAudit {
    pub scope: &'static str,
    pub offered_entries: u64,
    pub rejected: [ReasonCount<HostContentRejection>; 6],
    pub eligible: u64,
    pub outcomes: TrainingCounts,
    pub pre_update_prediction: PredictionAuditSnapshot,
}
impl HostContentAudit {
    pub(super) fn filled(count: u64) -> Self {
        Self {
            scope: "opt-in complete actual host-settled waves; every drained FIFO entry is offered, including absent/ineligible stages; legacy token-commit observations remain a separate population; retrospective immutable pre-batch predictions, not future-candidate coverage; content residuals are empirical, not worst-case bounds",
            offered_entries: count,
            rejected: HostContentRejection::ALL.map(|reason| ReasonCount { reason, count }),
            eligible: count,
            outcomes: TrainingCounts::filled(count),
            pre_update_prediction: PredictionAuditSnapshot::filled(count),
        }
    }
    pub fn record(
        &mut self,
        kind: Option<model::WaveKind>,
        result: HostContentEvaluation,
        exhausted: &mut bool,
    ) {
        add_one(&mut self.offered_entries, exhausted);
        if let Some(reason) = result.rejection {
            add_one(&mut self.rejected[reason as usize].count, exhausted);
            return;
        }
        add_one(&mut self.eligible, exhausted);
        self.outcomes.record(result.training, exhausted);
        if let (Some(kind), Some(prediction)) = (kind, result.pre_update_prediction) {
            self.pre_update_prediction
                .record(kind, prediction, exhausted);
        }
    }
}
