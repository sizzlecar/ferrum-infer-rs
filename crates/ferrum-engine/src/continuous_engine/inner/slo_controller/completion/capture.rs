//! Transaction-local shared capture; completion remains independently usable.
use super::*;
use ferrum_interfaces::model_executor::ExecutorPlanningCapture;
use ferrum_interfaces::vnext::{
    ExecutionCostRouteAvailability, ExecutionCostRouteView, ResourcePlanningBudget,
};
#[cfg(test)]
mod tests;

pub(super) struct DraftCapture {
    pub started: std::time::Instant,
    pub reserve_percent: u8,
    pub forecast_limits: ResourcePlanningLimits,
    pub forecast_enabled: bool,
    pub preparation_wall: Option<std::time::Duration>,
    pub optional_started: Option<std::time::Instant>,
    pub optional_elapsed: Option<std::time::Duration>,
    pub clock_invalid: bool,
    pub phase: Option<ControllerOptionalPhase>,
    pub forecast: Option<ExecutionCostRouteAvailability<ExecutionCostRouteView>>,
}

impl DraftCapture {
    pub(super) fn completed_preparation_wall(
        &self,
        finished_at: std::time::Instant,
    ) -> Option<std::time::Duration> {
        if self.clock_invalid || self.optional_started.is_some() {
            return None;
        }
        finished_at
            .checked_duration_since(self.started)?
            .checked_sub(self.optional_elapsed.unwrap_or_default())
    }
}

pub(super) struct CaptureObserver<'a> {
    pub state: &'a mut DraftCapture,
    pub budget: Arc<ControllerBudget>,
    pub rows: Option<Vec<ExpectedWorkSelection>>,
    pub kind: ActualWaveKind,
    pub work: Option<Result<ExpectedWaveWork>>,
}

impl ResourcePlanningBudget for CaptureObserver<'_> {
    fn has_budget(&mut self) -> bool {
        self.state
            .phase
            .as_ref()
            .map_or_else(|| self.budget.poll(), ControllerOptionalPhase::poll)
    }
    fn diagnostic_checkpoint(&mut self, stage: &'static str) {
        self.budget.diagnostic_checkpoint(stage);
    }
}

impl ExecutorPlanningCapture for CaptureObserver<'_> {
    fn resource_budget(&mut self) -> &mut dyn ResourcePlanningBudget {
        self
    }
    fn completion_ready(&mut self, resources: &ResourcePlanningView) -> bool {
        let Some(mut rows) = self.rows.take() else {
            return false;
        };
        // This callback does not reenter engine/executor state or obtain locks.
        rows.sort_by_key(|row| resources.participants()[row.participant_index].authority());
        self.work = Some(ExpectedWaveWork::new(resources, self.kind, rows));
        let Some(elapsed) = slo_clock_now().checked_duration_since(self.state.started) else {
            self.state.clock_invalid = true;
            return false;
        };
        self.state.preparation_wall = Some(elapsed);
        if self.work.as_ref().is_some_and(Result::is_ok) && self.budget.poll() {
            self.state.phase = Some(
                self.budget
                    .completion_optional_phase(elapsed, self.state.reserve_percent),
            );
            let accepted = self.state.forecast_enabled
                && self
                    .state
                    .phase
                    .as_ref()
                    .is_some_and(ControllerOptionalPhase::poll);
            if accepted {
                let started = slo_clock_now();
                if started.checked_duration_since(self.state.started).is_none() {
                    self.state.clock_invalid = true;
                    return false;
                }
                self.state.optional_started = Some(started);
            }
            accepted
        } else {
            false
        }
    }
    fn forecast_budget(&mut self) -> &mut dyn ResourcePlanningBudget {
        self
    }
    fn forecast_finished(&mut self) {
        let elapsed = self
            .state
            .optional_started
            .take()
            .and_then(|started| slo_clock_now().checked_duration_since(started));
        if self.state.optional_elapsed.is_some() || elapsed.is_none() {
            self.state.clock_invalid = true;
        } else {
            self.state.optional_elapsed = elapsed;
        }
    }
}

impl CompletionDraft {
    pub(in crate::continuous_engine::inner::slo_controller) fn take_compatible_forecast(
        &mut self,
        queue: &PlanningQueueSnapshot,
        fences: &[EngineFence],
        limits: ResourcePlanningLimits,
    ) -> Option<ExecutionCostRouteAvailability<ExecutionCostRouteView>> {
        if self.selection.proof.queue.requests() != queue.requests()
            || self.selection.proof.queue.iteration() != queue.iteration()
            || self.selection.proof.queue.wake_epochs() != queue.wake_epochs()
            || self.selection.proof.fences.len() != fences.len()
            || !self
                .selection
                .proof
                .fences
                .iter()
                .zip(fences)
                .all(|(old, new)| {
                    old.key == new.key
                        && old.incarnation == new.incarnation
                        && old.generation == new.generation
                        && old.generated == new.generated
                        && old.context == new.context
                        && old.output == new.output
                        && old.cache_id == new.cache_id
                        && old.prefill_complete == new.prefill_complete
                        && old.prefill_tokens_processed == new.prefill_tokens_processed
                        && old.prefill_total == new.prefill_total
                        && old.logits_policy.same_captured_input(&new.logits_policy)
                })
            || self.forecast_limits != limits
        {
            return None;
        }
        self.forecast.take()
    }
}
