//! Explicit maintenance shares the captured lane/plan authority with waves.
use super::*;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(in super::super) fn project_future_checkpoint(
        &self,
        view: &ExecutionCostRouteView,
        state: &ExecutionCostRouteState,
        query: FutureCheckpointCostQuery<'_>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<FutureCheckpointCostProjection> {
        if !budget.has_budget() {
            return ExecutionCostRouteAvailability::Unknown(U::BudgetExhausted);
        }
        if view.lane_id() != self.lane.id() {
            return ExecutionCostRouteAvailability::Unknown(U::StaleView);
        }
        if let Err(reason) = self.future_cost_policy() {
            return ExecutionCostRouteAvailability::Unknown(reason);
        }
        let plan = self.resolved_plan.execution_plan();
        let descriptor = self.runtime.descriptor();
        match query {
            FutureCheckpointCostQuery::Capture {
                source,
                span_start,
                boundary,
                prompt_tokens,
            } => self.plan_resources.project_future_checkpoint_capture(
                view,
                state,
                plan,
                descriptor,
                source,
                span_start,
                boundary,
                prompt_tokens,
                budget,
            ),
            FutureCheckpointCostQuery::Restore {
                checkpoint,
                target,
                prompt_tokens,
            } => self.plan_resources.project_future_checkpoint_restore(
                view,
                state,
                plan,
                descriptor,
                checkpoint,
                target,
                prompt_tokens,
                budget,
            ),
        }
    }
}
