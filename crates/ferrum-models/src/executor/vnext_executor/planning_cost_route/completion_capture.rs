//! Completion-first capture with optional forecast metadata in the same lane.
use super::*;
use ferrum_interfaces::model_executor::{
    ExecutorCompletionPlanningCapture, ExecutorPlanningCapture,
};

struct CompletionBudget<'a>(&'a mut dyn ExecutorPlanningCapture);
impl ResourcePlanningBudget for CompletionBudget<'_> {
    fn has_budget(&mut self) -> bool {
        self.0.resource_budget().has_budget()
    }
    fn diagnostic_checkpoint(&mut self, stage: &'static str) {
        self.0.resource_budget().diagnostic_checkpoint(stage);
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(in crate::executor::vnext_executor) fn capture_completion_and_future_route(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        limits: ResourcePlanningLimits,
        forecast_limits: ResourcePlanningLimits,
        observer: &mut dyn ExecutorPlanningCapture,
    ) -> ResourcePlanningAvailability<ExecutorCompletionPlanningCapture> {
        use ResourcePlanningAvailability as A;
        let mut observer = CompletionBudget(observer);
        let captured = resource_planning::with_registry_sequences(
            &self.sequences,
            requests,
            limits,
            &mut observer,
            |sequences, sessions, observer| {
                // Preserve registry -> token frontier -> mask ledger -> lane
                // order. Unavailable optional metadata never rejects completion.
                let mut frontiers = (|| {
                    self.future_cost_policy()?;
                    let mut frontiers = Vec::new();
                    frontiers
                        .try_reserve_exact(sequences.len())
                        .map_err(|_| U::Capacity)?;
                    for (sequence, request) in sequences.iter().zip(requests) {
                        if !observer.has_budget() {
                            return Err(U::BudgetExhausted);
                        }
                        let tokens = if request.cache_id.is_some() {
                            sequence
                                .tokens
                                .try_lock()
                                .map(|tokens| tokens.len())
                                .ok_or(U::Resource(ResourcePlanningUnknown::BusyOrUnavailable))?
                        } else {
                            sequence.prefill_tokens_processed.load(Ordering::Acquire)
                        };
                        frontiers.push(u64::try_from(tokens).map_err(|_| U::InvalidInput)?);
                    }
                    Ok(frontiers)
                })();
                // An unsupported policy or unavailable frontier does not take
                // the optional ledger lock and cannot reject completion.
                let ledger = frontiers
                    .is_ok()
                    .then(|| self.product_token_mask_residency.try_lock())
                    .flatten();
                let result = self.plan_resources.completion_planning_capture(
                    sessions,
                    &self.lane,
                    limits,
                    forecast_limits,
                    observer.0,
                    &mut |budget| {
                        if !budget.has_budget() {
                            return Err(U::BudgetExhausted);
                        }
                        let frontiers = std::mem::replace(&mut frontiers, Err(U::InvalidInput))?;
                        let ledger = ledger
                            .as_ref()
                            .ok_or(U::Resource(ResourcePlanningUnknown::BusyOrUnavailable))?;
                        let token_masks =
                            masks::capture(ledger, self.io.token_mask_residency_eligible, budget)?;
                        Ok((frontiers, token_masks))
                    },
                    &mut |view, token_masks, _budget| {
                        self.future_cost_graph_policy(&view)?;
                        Ok(view.with_token_mask_residency(token_masks))
                    },
                );
                drop(ledger);
                result
            },
        );
        captured.unwrap_or_else(A::Unknown)
    }
}
