//! One computed finite-prefix comparison, owned by its planning round. These
//! proxies are neither measured counterfactual benefits nor SLO certificates.

use super::*;
use crate::joint_policy::{
    BoundedActionEvidence, BoundedDecision, BoundedSelectionReason, Unsupported,
};
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BoundedActionDiagnostic {
    Static,
    PrefillLimit { max_tokens: NonZeroUsize },
    DecodeOnly,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum BoundedDecisionOutcome {
    /// Scoring finished; the consumer has not yet retained or applied it.
    Computed,
    Retained,
    Applied,
    /// The selected edit failed its actual batch/frontier validation.
    Invalidated,
    RejectedAfterEvaluation {
        reason: Unsupported,
        cpu_budget_exhausted: bool,
        cost_expired: bool,
    },
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct BoundedDecisionDiagnostic {
    pub snapshot_revision: Option<u64>,
    pub lookahead_rounds: usize,
    pub max_catchup_rounds: usize,
    /// Earliest queried cost expiry relative to the comparison's frozen `now`.
    /// It is not a wall timestamp or a fresh validity guarantee at trace write.
    pub cost_valid_for: Duration,
    /// Index into both candidate_actions and alternatives. A computed choice
    /// is not evidence of application; consult outcome.
    pub selected_index: Option<usize>,
    /// Local quantile/paired-work selection rule, distinct from application.
    pub selection_reason: BoundedSelectionReason,
    pub static_score: f64,
    pub selected_score: Option<f64>,
    pub request_count: usize,
    pub observed_ranks: usize,
    pub historical_ttft_late_requests: usize,
    pub historical_visible_intervals: usize,
    pub historical_bad_visible_intervals: usize,
    pub static_action: BoundedActionDiagnostic,
    pub static_evidence: BoundedActionEvidence,
    pub candidate_actions: Vec<BoundedActionDiagnostic>,
    pub alternatives: Vec<BoundedActionEvidence>,
    pub outcome: BoundedDecisionOutcome,
}

impl BoundedDecisionDiagnostic {
    /// Additional dynamic payload beyond the already charged accumulator.
    /// Its evidence vector is moved here, not duplicated. The action labels
    /// coexist with the adapter's private action vector during evaluation.
    pub(crate) fn additional_bytes_bound(alternatives: usize) -> Option<usize> {
        alternatives.checked_mul(std::mem::size_of::<BoundedActionDiagnostic>())
    }

    pub(crate) fn new(
        snapshot_revision: Option<u64>,
        config: &SchedulerSloConfig,
        now: Instant,
        decision: BoundedDecision,
        candidate_actions: Vec<BoundedActionDiagnostic>,
    ) -> Self {
        Self {
            snapshot_revision,
            lookahead_rounds: config.planning.max_lookahead_rounds,
            max_catchup_rounds: config.planning.max_catchup_rounds,
            cost_valid_for: decision.valid_until.saturating_duration_since(now),
            selected_index: decision.selected,
            selection_reason: decision.selection_reason,
            static_score: decision.static_score,
            selected_score: decision.selected_score,
            request_count: decision.request_count,
            observed_ranks: decision.observed_ranks,
            historical_ttft_late_requests: decision.historical_ttft_late_requests,
            historical_visible_intervals: decision.historical_visible_intervals,
            historical_bad_visible_intervals: decision.historical_bad_visible_intervals,
            static_action: BoundedActionDiagnostic::Static,
            static_evidence: decision.static_evidence,
            candidate_actions,
            alternatives: decision.alternatives,
            outcome: BoundedDecisionOutcome::Computed,
        }
    }
}
