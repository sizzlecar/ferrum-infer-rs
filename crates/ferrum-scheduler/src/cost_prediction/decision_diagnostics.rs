//! Fixed-size counts of computed decisions and their actual consumer mapping.
use super::ComparisonKind;
use crate::joint_policy::{
    CandidateDecisionCounts, DecisionObservation, FeasibilityCounts, MetricClassCounts,
    UnresolvedDecisionWitness,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub(crate) enum DecisionConsumerOutcome {
    StaticFeasible,
    Alternative,
    AllKnownInfeasible,
    FeasibilityUnresolved,
    StoppingCoverage,
    RequiresAction,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct DecisionOutcomeCounts {
    pub static_feasible: u64,
    /// Selected by the policy, not necessarily committed by the resource owner.
    pub alternative: u64,
    pub all_known_infeasible: u64,
    pub feasibility_unresolved: u64,
    pub stopping_coverage: u64,
    pub requires_action: u64,
}

impl DecisionOutcomeCounts {
    fn record(&mut self, outcome: DecisionConsumerOutcome) {
        let count = match outcome {
            DecisionConsumerOutcome::StaticFeasible => &mut self.static_feasible,
            DecisionConsumerOutcome::Alternative => &mut self.alternative,
            DecisionConsumerOutcome::AllKnownInfeasible => &mut self.all_known_infeasible,
            DecisionConsumerOutcome::FeasibilityUnresolved => &mut self.feasibility_unresolved,
            DecisionConsumerOutcome::StoppingCoverage => &mut self.stopping_coverage,
            DecisionConsumerOutcome::RequiresAction => &mut self.requires_action,
        };
        *count = count.saturating_add(1);
    }
    fn add(&mut self, other: Self) {
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {$(
                self.$field = self.$field.saturating_add(other.$field);
            )+};
        }
        add!(
            static_feasible,
            alternative,
            all_known_infeasible,
            feasibility_unresolved,
            stopping_coverage,
            requires_action
        );
    }
}

/// Only final FeasibilityUnresolved consumer outcomes enter this denominator.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct UnresolvedDecisionCounts {
    pub decisions: u64,
    pub no_alternatives: u64,
    pub static_raw: FeasibilityCounts,
    pub static_effective: FeasibilityCounts,
    pub static_metrics: MetricClassCounts,
    /// Candidate counts use alternatives, not comparisons, as their grain.
    pub candidates: CandidateDecisionCounts,
    /// Nonexclusive comparison flags; never add as distinct failure causes.
    pub any_candidate_completion_unresolved: u64,
    pub any_partial_window_block: u64,
    pub any_static_window_debt_block: u64,
    pub any_target_time_block: u64,
}

impl UnresolvedDecisionCounts {
    fn record(&mut self, value: &DecisionObservation) {
        self.decisions = self.decisions.saturating_add(1);
        self.no_alternatives = self
            .no_alternatives
            .saturating_add(u64::from(value.alternatives_total == 0));
        self.static_raw.record(value.static_raw);
        self.static_effective.record(value.static_effective);
        self.static_metrics.record(value.static_metrics);
        self.candidates.add(value.candidates);
        self.any_candidate_completion_unresolved = self
            .any_candidate_completion_unresolved
            .saturating_add(u64::from(
                value.candidates.metrics.completion_failure.unresolved > 0,
            ));
        self.any_partial_window_block = self.any_partial_window_block.saturating_add(u64::from(
            value.candidates.gates.partial_window_unproven > 0,
        ));
        self.any_static_window_debt_block = self.any_static_window_debt_block.saturating_add(
            u64::from(value.candidates.gates.static_feasible_window_debt > 0),
        );
        self.any_target_time_block = self.any_target_time_block.saturating_add(u64::from(
            value.candidates.gates.target_time_unavailable > 0,
        ));
    }
    fn add(&mut self, other: &Self) {
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {$(
                self.$field = self.$field.saturating_add(other.$field);
            )+};
        }
        add!(
            decisions,
            no_alternatives,
            any_candidate_completion_unresolved,
            any_partial_window_block,
            any_static_window_debt_block,
            any_target_time_block
        );
        self.static_raw.add(other.static_raw);
        self.static_effective.add(other.static_effective);
        self.static_metrics.add(other.static_metrics);
        self.candidates.add(other.candidates);
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct DecisionTotals {
    /// aggregate.finish returned a decision; evaluator Ok has a wider scope.
    pub computed: u64,
    /// Decision passed the existing post-finish time guard.
    pub returned: u64,
    pub post_finish_time_rejected: u64,
    /// Actual dispatch choice/admission verdict mappings, before resource apply.
    pub consumed: u64,
    pub outcomes: DecisionOutcomeCounts,
    pub candidates: CandidateDecisionCounts,
    pub raw_effective_static_mismatch: u64,
    pub unresolved: UnresolvedDecisionCounts,
    /// Only a final FeasibilityUnresolved consumer result replaces this slot.
    pub latest_unresolved: Option<UnresolvedDecisionWitness>,
}

impl DecisionTotals {
    fn consume(&mut self, outcome: DecisionConsumerOutcome, value: &DecisionObservation) {
        self.consumed = self.consumed.saturating_add(1);
        self.outcomes.record(outcome);
        self.candidates.add(value.candidates);
        self.raw_effective_static_mismatch = self
            .raw_effective_static_mismatch
            .saturating_add(u64::from(value.static_raw != value.static_effective));
        if matches!(outcome, DecisionConsumerOutcome::FeasibilityUnresolved) {
            self.unresolved.record(value);
            if let Some(witness) = value.unresolved {
                self.latest_unresolved = Some(witness);
            }
        }
    }
    fn add(&mut self, other: &Self) {
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {$(
                self.$field = self.$field.saturating_add(other.$field);
            )+};
        }
        add!(
            computed,
            returned,
            post_finish_time_rejected,
            consumed,
            raw_effective_static_mismatch
        );
        self.outcomes.add(other.outcomes);
        self.candidates.add(other.candidates);
        self.unresolved.add(&other.unresolved);
        if other.latest_unresolved.is_some() {
            self.latest_unresolved = other.latest_unresolved;
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct JointDecisionDiagnostics {
    /// Fixed resident Rust payload only, excluding cold serialization and RSS.
    pub retained_bytes_bound: usize,
    pub admission: DecisionTotals,
    pub dispatch: DecisionTotals,
}

impl Default for JointDecisionDiagnostics {
    fn default() -> Self {
        Self {
            retained_bytes_bound: std::mem::size_of::<Self>(),
            admission: DecisionTotals::default(),
            dispatch: DecisionTotals::default(),
        }
    }
}

impl JointDecisionDiagnostics {
    fn totals(&mut self, kind: ComparisonKind) -> &mut DecisionTotals {
        match kind {
            ComparisonKind::Admission => &mut self.admission,
            ComparisonKind::Dispatch => &mut self.dispatch,
        }
    }
    pub(crate) fn computed(&mut self, kind: ComparisonKind) {
        let value = self.totals(kind);
        value.computed = value.computed.saturating_add(1);
    }
    pub(crate) fn returned(&mut self, kind: ComparisonKind) {
        let value = self.totals(kind);
        value.returned = value.returned.saturating_add(1);
    }
    pub(crate) fn post_finish_time_rejected(&mut self, kind: ComparisonKind) {
        let value = self.totals(kind);
        value.post_finish_time_rejected = value.post_finish_time_rejected.saturating_add(1);
    }
    pub(crate) fn consume(
        &mut self,
        kind: ComparisonKind,
        outcome: DecisionConsumerOutcome,
        value: &DecisionObservation,
    ) {
        self.totals(kind).consume(outcome, value);
    }
    pub(crate) fn add(&mut self, other: &Self) {
        self.admission.add(&other.admission);
        self.dispatch.add(&other.dispatch);
    }
}

#[cfg(test)]
mod tests;
