//! Read-only selection diagnostics. Estimator samples and fitting belong to the
//! independently owned reference worker, not a scheduler mutex.
use crate::cost_prediction::{PlanningStatistics, TimePlanningRound};
use serde::Serialize;

/// Keeps an Engine planning turn observable when admission returns, fails or is
/// cancelled before batch selection. It owns no resource or scheduling authority.
pub struct PlanningRoundObservation<'a> {
    scheduler: &'a super::ContinuousBatchScheduler,
    round: Option<TimePlanningRound>,
}

impl PlanningRoundObservation<'_> {
    pub fn as_mut(&mut self) -> Option<&mut TimePlanningRound> {
        self.round.as_mut()
    }

    /// End the observation before device execution, retaining the immutable
    /// selection facts. A raw caller may subsequently start another selection.
    pub fn finish(mut self) -> Option<TimePlanningRound> {
        self.record();
        let mut round = self.round.take();
        if let Some(round) = round.as_mut() {
            round.diagnostics_scope_owned = false;
        }
        round
    }

    fn record(&self) {
        if let Some(round) = self.round.as_ref() {
            self.scheduler.slo_diagnostics.lock().record(round);
        }
    }
}

impl Drop for PlanningRoundObservation<'_> {
    fn drop(&mut self) {
        self.record();
    }
}

impl super::ContinuousBatchScheduler {
    pub fn observe_planning_round(
        &self,
        mut round: Option<TimePlanningRound>,
    ) -> PlanningRoundObservation<'_> {
        if let Some(round) = round.as_mut() {
            round.diagnostics_scope_owned = true;
            round.selection_attempted = false;
        }
        PlanningRoundObservation {
            scheduler: self,
            round,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SchedulerSloSnapshot {
    pub enabled: bool,
    pub planning_rounds: u64,
    /// Created turns that ended before entering batch selection, including
    /// admission-only progress, errors and cancelled admission futures.
    pub planning_rounds_before_selection: u64,
    pub last_revision: Option<u64>,
    #[serde(flatten)]
    pub(crate) selections: PlanningStatistics,
}

impl SchedulerSloSnapshot {
    pub(super) fn record(&mut self, round: &TimePlanningRound) {
        self.enabled = true;
        self.planning_rounds += 1;
        self.planning_rounds_before_selection += u64::from(!round.selection_attempted);
        // The optional provider revision was captured outside this mutex at
        // turn creation. Drop/cancellation must not invoke provider callbacks.
        self.last_revision = round.diagnostic_revision;
        let value = &round.statistics;
        self.selections.joint_cost_snapshot_queries = self
            .selections
            .joint_cost_snapshot_queries
            .saturating_add(value.joint_cost_snapshot_queries);
        self.selections.joint_cost_cache_hits = self
            .selections
            .joint_cost_cache_hits
            .saturating_add(value.joint_cost_cache_hits);
        self.selections.joint_cost_cache_peak_bytes = self
            .selections
            .joint_cost_cache_peak_bytes
            .max(value.joint_cost_cache_peak_bytes);
        self.selections.joint_comparisons += value.joint_comparisons;
        self.selections.joint_interventions += value.joint_interventions;
        self.selections.joint_unknown_resources += value.joint_unknown_resources;
        self.selections.joint_unresolved += value.joint_unresolved;
        self.selections
            .joint_unsupported
            .add(value.joint_unsupported);
        self.selections.joint_admission_action_required = self
            .selections
            .joint_admission_action_required
            .saturating_add(value.joint_admission_action_required);
        self.selections.joint_known_infeasible += value.joint_known_infeasible;
        self.selections.known_queries += value.known_queries;
        self.selections.unknown_queries += value.unknown_queries;
        self.selections.held_candidates += value.held_candidates;
        self.selections.rejected_candidates += value.rejected_candidates;
        self.selections.shrunk_prefills += value.shrunk_prefills;
        self.selections.static_fallbacks += value.static_fallbacks;
        self.selections.budget_misses += value.budget_misses;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost_prediction::{
        CostExecution, CostPrediction, CostQuery, CostSnapshot, CostUnknownReason,
    };
    use ferrum_execution::model_executor::PrefillShapePolicy;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Instant;

    struct RevisionProbe {
        scheduler: Arc<super::super::ContinuousBatchScheduler>,
        calls: AtomicUsize,
        panic: bool,
    }

    impl CostSnapshot for RevisionProbe {
        fn revision(&self) -> u64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            // A valid read-only provider callback must not run under the
            // scheduler diagnostics lock, including during scope destruction.
            let _ = self.scheduler.slo_snapshot();
            assert!(!self.panic, "injected revision panic");
            7
        }

        fn query_limits(&self) -> ferrum_types::CostQueryLimits {
            ferrum_types::CostQueryLimits::default()
        }

        fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        }
    }

    #[tokio::test]
    async fn cancelled_planning_scope_records_without_provider_callbacks_on_drop() {
        for panic in [false, true] {
            let policy = "ttft:1000,tpot:100,itl:100".parse().unwrap();
            let scheduler = Arc::new(super::super::ContinuousBatchScheduler::new(
                ferrum_types::SchedulerConfig {
                    slo: Some(policy),
                    ..Default::default()
                },
            ));
            let provider = Arc::new(RevisionProbe {
                scheduler: scheduler.clone(),
                calls: AtomicUsize::new(0),
                panic,
            });
            let round = TimePlanningRound::new(
                provider.clone(),
                policy,
                PrefillShapePolicy::Identity,
                CostExecution::SingleInvocation,
            );
            let mut pending = Box::pin(async {
                let _observation = scheduler.observe_planning_round(Some(round));
                std::future::pending::<()>().await;
            });
            assert!(futures::poll!(pending.as_mut()).is_pending());
            drop(pending);
            let snapshot = scheduler.slo_snapshot().unwrap();
            assert_eq!(snapshot.planning_rounds, 1);
            assert_eq!(snapshot.planning_rounds_before_selection, 1);
            assert_eq!(snapshot.last_revision, (!panic).then_some(7));
            assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
        }
    }
}
