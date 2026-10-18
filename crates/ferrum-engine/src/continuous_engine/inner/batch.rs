use super::*;

impl EngineInner {
    pub(in crate::continuous_engine) async fn process_batch(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> Result<()> {
        self.process_token_batch(batch).await
    }

    pub(in crate::continuous_engine) fn execution_capacity_release_snapshot(
        &self,
    ) -> Result<ExecutionCapacityReleaseSnapshot> {
        let authorities = self
            .sequences
            .read()
            .iter()
            .filter_map(|(request_id, sequence)| {
                sequence
                    .execution_frontier
                    .as_ref()
                    .map(|frontier| (request_id.clone(), frontier.sequence().clone()))
            })
            .collect::<Vec<_>>();
        let mut capabilities = Vec::with_capacity(authorities.len());
        for (request_id, sequence) in authorities {
            let mut sources = Vec::new();
            if self
                .model_executor
                .write_token_capacity_release_sources(&sequence, &mut sources)?
            {
                capabilities.push((request_id, sources));
            }
        }
        Ok(ExecutionCapacityReleaseSnapshot::new(capabilities))
    }

    fn reconcile_capacity_yield_error(
        &self,
        transaction: &PressureYieldTransaction,
        victim_released: bool,
        attempted_width: usize,
        observed_free_blocks: Option<usize>,
        error: FerrumError,
    ) -> FerrumError {
        match self.scheduler.abort_execution_capacity_yield(
            transaction,
            victim_released,
            attempted_width,
            observed_free_blocks,
        ) {
            Ok((victim_requeued, aborted_ordinal, closed_ordinal)) => {
                self.write_scheduler_trace_event(serde_json::json!({
                    "event": "scheduler_pressure_yield_aborted",
                    "episode_id": transaction.episode_id().get(),
                    "handoff_generation": transaction.handoff_generation(),
                    "victim_request_id": transaction.victim_request_id(),
                    "progress_owner_id": transaction.progress_owner_id(),
                    "victim_released": victim_released,
                    "victim_requeued": victim_requeued,
                    "aborted_transition_ordinal": aborted_ordinal.get(),
                    "closed_transition_ordinal": closed_ordinal.get(),
                    "error": error.to_string(),
                    "scheduler": self.scheduler.trace_snapshot(),
                }));
                error
            }
            Err(reconcile_error) => FerrumError::internal(format!(
                "execution-capacity yield failed ({error}) and reconciliation failed ({reconcile_error})"
            )),
        }
    }

    pub(in crate::continuous_engine) async fn execute_capacity_yield(
        &self,
        transaction: &PressureYieldTransaction,
        attempted_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> Result<bool> {
        let armed = self
            .scheduler
            .arm_execution_capacity_yield(transaction)
            .map_err(|error| {
                self.reconcile_capacity_yield_error(
                    transaction,
                    false,
                    attempted_width,
                    observed_free_blocks,
                    error,
                )
            })?;
        let ticket = self
            .begin_execution_recompute(transaction.clone(), attempted_width, observed_free_blocks)
            .map_err(|error| {
                self.reconcile_capacity_yield_error(
                    transaction,
                    false,
                    attempted_width,
                    observed_free_blocks,
                    error,
                )
            })?;
        self.write_scheduler_trace_event(serde_json::json!({
            "event": "scheduler_pressure_release_fence_armed",
            "episode_id": transaction.episode_id().get(),
            "handoff_generation": transaction.handoff_generation(),
            "victim_request_id": transaction.victim_request_id(),
            "progress_owner_id": transaction.progress_owner_id(),
            "transition_ordinal": armed.get(),
            "physical_release_completed": false,
        }));
        // The independent close task retains the original slot and transaction if
        // this caller is cancelled. Its first result is not an Ended attestation.
        let first = ticket.initial_result().await;
        let terminal = self.drain_execution_closes();
        first.and(terminal)?;
        // Even an immediately completed release returns through the scheduler.
        // Do not grant an inline retry from a close-operation return value.
        Ok(false)
    }
}
