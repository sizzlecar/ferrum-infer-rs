//! Read-only selection diagnostics. Estimator samples and fitting belong to the
//! independently owned reference worker, not a scheduler mutex.
use crate::cost_prediction::{PlanningStatistics, TimePlanningRound};
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct SchedulerSloSnapshot {
    pub enabled: bool,
    pub planning_rounds: u64,
    pub last_revision: Option<u64>,
    #[serde(flatten)]
    pub(crate) selections: PlanningStatistics,
}

impl SchedulerSloSnapshot {
    pub(super) fn record(&mut self, round: &TimePlanningRound) {
        self.enabled = true;
        self.planning_rounds += 1;
        self.last_revision = Some(round.revision());
        let value = &round.statistics;
        self.selections.known_queries += value.known_queries;
        self.selections.unknown_queries += value.unknown_queries;
        self.selections.held_candidates += value.held_candidates;
        self.selections.rejected_candidates += value.rejected_candidates;
        self.selections.shrunk_prefills += value.shrunk_prefills;
        self.selections.static_fallbacks += value.static_fallbacks;
        self.selections.budget_misses += value.budget_misses;
    }
}
