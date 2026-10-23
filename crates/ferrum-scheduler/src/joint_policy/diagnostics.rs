//! Fixed-size, comparison-local observations. These explain evaluated work,
//! not the counterfactual outcome of an unevaluated candidate.

use super::Unsupported;
use parking_lot::Mutex;
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStage {
    #[default]
    Preparation,
    Seed,
    RankAssignment,
    Simulation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EvaluationPoint {
    pub stage: EvaluationStage,
    pub rank: Option<usize>,
    /// Zero is static; one and above identify alternatives in input order.
    pub candidate: Option<usize>,
    /// Zero-based trajectory wave and actual invocation group.
    pub round: Option<usize>,
    pub group: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureOrigin {
    #[default]
    Evaluation,
    Provider,
    ProviderPanic,
    ScenarioContract,
    RankAssignment,
    Simulation,
    PreQueryTime,
    PostQueryTime,
    ProviderQueryBudget,
    SimulationQueryBudget,
    SimulationTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EvaluationFailure {
    pub reason: Unsupported,
    pub origin: FailureOrigin,
    pub point: EvaluationPoint,
    pub provider_calls: u64,
    pub simulation_queries: u64,
    pub rank_simulation_queries: usize,
    pub query_limit: usize,
    pub elapsed_us: u64,
}

/// Always returned through the out parameter, including early failure. Actual
/// provider calls and a sticky budget boundary remain separate: cached
/// simulation work can exhaust its guard without making another provider call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ScenarioDiagnostics {
    /// Actual original provider calls, including seed and failed calls.
    pub provider_calls: u64,
    /// Any provider/simulation query-budget exhaustion, including an unsupported
    /// alternative in an otherwise successful traversal. An original provider's
    /// unspecified PlanningBudget also stops later queries conservatively; its
    /// internal work is unknown and is never added to provider_calls.
    pub query_budget_exhausted: bool,
    /// Candidate/rank traversals that exhausted bounded catch-up waves. A
    /// comparison may still complete or select another action. This does not
    /// imply exhaustion of the global query or elapsed-time budgets.
    pub candidate_horizon_exhaustions: u64,
    /// Calls to the simulation snapshot, including cache hits.
    pub simulation_queries: u64,
    pub cache_hits: u64,
    pub cache_peak_bytes: usize,
    pub elapsed_us: u64,
    pub rank_attempts: usize,
    /// Visited ranks whose static trajectory is complete. Alternatives can
    /// still be unsupported. Partial static and unassigned ranks are separate.
    pub rank_completed: usize,
    pub rank_partial_static: usize,
    pub rank_unresolved: usize,
    pub max_rank_simulation_queries: usize,
    /// Observed rejected trajectories/assignments, not all possible causes.
    pub failure_count: u64,
    /// First failure observed, including an unsupported alternative in an
    /// otherwise successful traversal. It is not a complete causal diagnosis.
    pub first_failure: Option<EvaluationFailure>,
    /// Present only when the complete evaluator returns Err.
    pub terminal_failure: Option<EvaluationFailure>,
}

struct State {
    report: ScenarioDiagnostics,
    point: EvaluationPoint,
    rank_queries: usize,
    last_failure: Option<EvaluationFailure>,
}

/// Only this evaluation owns this fixed-size lock. Never held across provider,
/// adapter, worker, or cache calls; no retained trajectory or request payload.
pub(super) struct Recorder {
    started: Instant,
    query_limit: usize,
    state: Mutex<State>,
}

impl Recorder {
    pub(super) fn new(query_limit: usize) -> Self {
        Self {
            started: Instant::now(),
            query_limit,
            state: Mutex::new(State {
                report: ScenarioDiagnostics::default(),
                point: EvaluationPoint::default(),
                rank_queries: 0,
                last_failure: None,
            }),
        }
    }

    fn elapsed_us(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    pub(super) fn point(&self, point: EvaluationPoint, rank_queries: usize) {
        let mut state = self.state.lock();
        state.point = point;
        state.rank_queries = rank_queries;
        state.report.max_rank_simulation_queries =
            state.report.max_rank_simulation_queries.max(rank_queries);
    }

    pub(super) fn provider_call(&self) {
        let mut state = self.state.lock();
        state.report.provider_calls = state.report.provider_calls.saturating_add(1);
    }

    pub(super) fn simulation_query(&self, rank_queries: usize) {
        let mut state = self.state.lock();
        state.report.simulation_queries = state.report.simulation_queries.saturating_add(1);
        state.rank_queries = rank_queries;
        state.report.max_rank_simulation_queries =
            state.report.max_rank_simulation_queries.max(rank_queries);
    }

    pub(super) fn cache_hit(&self) {
        let mut state = self.state.lock();
        state.report.cache_hits = state.report.cache_hits.saturating_add(1);
    }

    pub(super) fn cache_bytes(&self, bytes: usize) {
        let mut state = self.state.lock();
        state.report.cache_peak_bytes = state.report.cache_peak_bytes.max(bytes);
    }

    pub(super) fn rank_attempt(&self) {
        let mut state = self.state.lock();
        state.report.rank_attempts = state.report.rank_attempts.saturating_add(1);
    }

    pub(super) fn rank_finished(&self, unresolved: bool) {
        let mut state = self.state.lock();
        if unresolved {
            state.report.rank_unresolved = state.report.rank_unresolved.saturating_add(1);
        } else {
            state.report.rank_completed = state.report.rank_completed.saturating_add(1);
        }
    }

    pub(super) fn failures(&self) -> u64 {
        self.state.lock().report.failure_count
    }

    pub(super) fn rank_partial_static(&self) {
        let mut state = self.state.lock();
        state.report.rank_partial_static = state.report.rank_partial_static.saturating_add(1);
    }

    pub(super) fn failure(&self, reason: Unsupported, origin: FailureOrigin) {
        let elapsed_us = self.elapsed_us();
        let mut state = self.state.lock();
        if reason == Unsupported::BoundedHorizonExhausted {
            state.report.candidate_horizon_exhaustions =
                state.report.candidate_horizon_exhaustions.saturating_add(1);
        }
        state.report.query_budget_exhausted |= matches!(reason, Unsupported::QueryBudget)
            || matches!(
                origin,
                FailureOrigin::ProviderQueryBudget | FailureOrigin::SimulationQueryBudget
            )
            || (matches!(
                reason,
                Unsupported::Cost(crate::cost_prediction::CostUnknownReason::PlanningBudget)
            ) && origin == FailureOrigin::Provider);
        let failure = EvaluationFailure {
            reason,
            origin,
            point: state.point,
            provider_calls: state.report.provider_calls,
            simulation_queries: state.report.simulation_queries,
            rank_simulation_queries: state.rank_queries,
            query_limit: self.query_limit,
            elapsed_us,
        };
        state.report.failure_count = state.report.failure_count.saturating_add(1);
        state.report.first_failure.get_or_insert(failure);
        state.last_failure = Some(failure);
    }

    pub(super) fn fallback_failure(
        &self,
        prior_failures: u64,
        reason: Unsupported,
        origin: FailureOrigin,
    ) {
        if self.failures() == prior_failures {
            self.failure(reason, origin);
        }
    }

    pub(super) fn finish(&self, error: Option<Unsupported>) -> ScenarioDiagnostics {
        if let Some(reason) = error {
            let matches = self
                .state
                .lock()
                .last_failure
                .is_some_and(|failure| failure.reason == reason);
            if !matches {
                self.failure(reason, FailureOrigin::Evaluation);
            }
        }
        let mut state = self.state.lock();
        state.report.elapsed_us = self.elapsed_us();
        state.report.terminal_failure = error.and(state.last_failure);
        state.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_candidate_horizon_failure_does_not_poison_global_query_budget() {
        let recorder = Recorder::new(16);
        recorder.point(
            EvaluationPoint {
                stage: EvaluationStage::Simulation,
                rank: Some(0),
                candidate: Some(1),
                round: Some(3),
                group: Some(0),
            },
            4,
        );
        let before = recorder.failures();
        recorder.failure(
            Unsupported::BoundedHorizonExhausted,
            FailureOrigin::Simulation,
        );
        // The simulator's outer fallback must not count the same failure again.
        recorder.fallback_failure(
            before,
            Unsupported::BoundedHorizonExhausted,
            FailureOrigin::Simulation,
        );
        recorder.rank_finished(false);
        let completed = recorder.finish(None);
        assert_eq!(completed.candidate_horizon_exhaustions, 1);
        assert_eq!(completed.failure_count, 1);
        assert_eq!(completed.rank_completed, 1);
        assert!(!completed.query_budget_exhausted);
        assert_eq!(completed.terminal_failure, None);
        assert_eq!(
            completed.first_failure.unwrap().reason,
            Unsupported::BoundedHorizonExhausted
        );

        // A later global failure remains its own failure; it neither erases nor
        // duplicates the already observed candidate/rank boundary.
        recorder.failure(Unsupported::TimeBudget, FailureOrigin::SimulationTime);
        let timed_out = recorder.finish(Some(Unsupported::TimeBudget));
        assert_eq!(timed_out.candidate_horizon_exhaustions, 1);
        assert!(!timed_out.query_budget_exhausted);
        assert_eq!(timed_out.failure_count, 2);
        assert_eq!(
            timed_out.terminal_failure.unwrap().reason,
            Unsupported::TimeBudget
        );
        recorder.failure(
            Unsupported::QueryBudget,
            FailureOrigin::SimulationQueryBudget,
        );
        let quota = recorder.finish(Some(Unsupported::QueryBudget));
        assert!(quota.query_budget_exhausted);
        assert_eq!(quota.candidate_horizon_exhaustions, 1);
        assert_eq!(quota.failure_count, 3);
    }
}
