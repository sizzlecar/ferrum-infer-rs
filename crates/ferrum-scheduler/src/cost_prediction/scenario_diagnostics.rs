//! Aggregate only entered scenario evaluations, including failed traversals.
//! Cohort/capacity failures before the evaluator have a different denominator.

use crate::joint_policy::{EvaluationFailure, ScenarioDiagnostics};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ComparisonKind {
    Admission,
    Dispatch,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct JointEvaluationDiagnostics {
    pub admission: EvaluationTotals,
    pub dispatch: EvaluationTotals,
}

impl JointEvaluationDiagnostics {
    pub(crate) fn record(
        &mut self,
        kind: ComparisonKind,
        value: ScenarioDiagnostics,
        completed: bool,
    ) {
        let totals = match kind {
            ComparisonKind::Admission => &mut self.admission,
            ComparisonKind::Dispatch => &mut self.dispatch,
        };
        totals.add(EvaluationTotals {
            evaluations: 1,
            completed_traversals: u64::from(completed),
            failed_traversals: u64::from(!completed),
            provider_calls: value.provider_calls,
            simulation_queries: value.simulation_queries,
            cache_hits: value.cache_hits,
            cache_peak_bytes: value.cache_peak_bytes,
            elapsed_total_us: value.elapsed_us,
            elapsed_max_us: value.elapsed_us,
            rank_attempts: value.rank_attempts,
            rank_completed: value.rank_completed,
            rank_partial_static: value.rank_partial_static,
            rank_unresolved: value.rank_unresolved,
            max_rank_simulation_queries: value.max_rank_simulation_queries,
            observed_failures: value.failure_count,
            latest_first_failure: value.first_failure,
            latest_terminal_failure: value.terminal_failure,
        });
    }

    pub(crate) fn add(&mut self, value: Self) {
        self.admission.add(value.admission);
        self.dispatch.add(value.dispatch);
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct EvaluationTotals {
    pub evaluations: u64,
    pub completed_traversals: u64,
    pub failed_traversals: u64,
    /// Actual provider calls on every exit, including seed and failed calls.
    pub provider_calls: u64,
    /// Simulation snapshot calls, including exact-cache hits.
    pub simulation_queries: u64,
    pub cache_hits: u64,
    pub cache_peak_bytes: usize,
    /// Evaluator wall time, not OS CPU or the scheduler's budget charge.
    pub elapsed_total_us: u64,
    pub elapsed_max_us: u64,
    pub rank_attempts: usize,
    pub rank_completed: usize,
    pub rank_partial_static: usize,
    pub rank_unresolved: usize,
    pub max_rank_simulation_queries: usize,
    pub observed_failures: u64,
    /// Latest evaluation that observed a failure, with that evaluation's first
    /// failure. Neither this sample nor its reason is a cause-frequency table.
    pub latest_first_failure: Option<EvaluationFailure>,
    /// Latest failed traversal. Successful later traversals do not erase it.
    pub latest_terminal_failure: Option<EvaluationFailure>,
}

impl EvaluationTotals {
    fn add(&mut self, value: Self) {
        macro_rules! sum {
            ($($field:ident),+ $(,)?) => {$(
                self.$field = self.$field.saturating_add(value.$field);
            )+};
        }
        sum!(
            evaluations,
            completed_traversals,
            failed_traversals,
            provider_calls,
            simulation_queries,
            cache_hits,
            elapsed_total_us,
            rank_attempts,
            rank_completed,
            rank_partial_static,
            rank_unresolved,
            observed_failures,
        );
        self.cache_peak_bytes = self.cache_peak_bytes.max(value.cache_peak_bytes);
        self.elapsed_max_us = self.elapsed_max_us.max(value.elapsed_max_us);
        self.max_rank_simulation_queries = self
            .max_rank_simulation_queries
            .max(value.max_rank_simulation_queries);
        if value.latest_first_failure.is_some() {
            self.latest_first_failure = value.latest_first_failure;
        }
        if value.latest_terminal_failure.is_some() {
            self.latest_terminal_failure = value.latest_terminal_failure;
        }
    }
}
