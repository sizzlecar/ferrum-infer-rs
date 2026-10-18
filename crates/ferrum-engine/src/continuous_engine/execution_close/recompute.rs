//! Recompute is a terminal purpose of the same exact-incarnation close owner.
//! It does not release device resources or create a second execution registry.

use super::*;

impl EngineInner {
    pub(in crate::continuous_engine) fn begin_execution_recompute(
        &self,
        transaction: PressureYieldTransaction,
        attempted_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> Result<ExecutionCloseTicket> {
        let request_id = transaction.victim_request_id().clone();
        let mut sequences = self.sequences.write();
        let failure_targets = self
            .scheduler
            .execution_capacity_yield_requests(&transaction)?
            .into_iter()
            .filter_map(|key| {
                sequences
                    .get(key.request_id())
                    .map(|sequence| RecomputeFailureTarget {
                        key,
                        frontier: sequence.execution_frontier.clone(),
                    })
            })
            .collect();
        let sequence = sequences
            .get_mut(&request_id)
            .ok_or_else(|| FerrumError::internal("capacity yield lost its outer request"))?;
        let frontier = sequence.execution_frontier.clone().ok_or_else(|| {
            FerrumError::internal("capacity yield requires the actual token incarnation")
        })?;
        if frontier.sequence().observed_end().is_some()
            || sequence.execution_close.is_some()
            || self
                .execution_closes
                .contains_incarnation(frontier.sequence())
        {
            return Err(FerrumError::internal(
                "capacity yield cannot replace a closing incarnation",
            ));
        }
        let readiness = self
            .scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&request_id))?
            .into_wake();
        let slot = sequence.request_slot.take();
        // No await separates installing the scheduling barrier and transferring
        // the existing slot into the independent close observation task.
        self.execution_closes.start(
            request_id,
            frontier.sequence().clone(),
            slot,
            Arc::clone(&self.model_executor),
            CloseStart::Execute(ExecutionEndReason::Cancelled),
            ClosePurpose::Recompute(RecomputeClose {
                frontier,
                transaction,
                attempted_width,
                observed_free_blocks,
                readiness,
                failure_targets,
                failure_error: None,
                failure_marked: false,
            }),
            Arc::clone(&self.work_notify),
        )
    }

    /// Product failures remain attached to the original close entry until
    /// each exact request has handed off its terminal obligations. Cancelling
    /// this future cannot discard the not-yet-notified peers.
    pub(in crate::continuous_engine) async fn complete_failed_execution_recomputes(
        &self,
    ) -> Result<()> {
        let mut first_error = None;
        let mut handled = false;
        while let Some((closing, target, error)) = self.execution_closes.next_recompute_failure() {
            handled = true;
            let current = {
                let sequences = self.sequences.read();
                self.scheduler
                    .execution_capacity_request_is_current(&target.key)
                    && sequences
                        .get(target.key.request_id())
                        .is_some_and(|sequence| {
                            sequence.execution_frontier.as_ref() == target.frontier.as_ref()
                        })
            };
            if current {
                let result = match &target.frontier {
                    Some(frontier) => {
                        self.fail_token_frontier(target.key.request_id(), frontier, error, None)
                            .await
                    }
                    None => {
                        self.complete_request_with_error(target.key.request_id(), error)
                            .await
                    }
                };
                if let Err(error) = result {
                    first_error.get_or_insert(error);
                }
            }
            // No await separates completed handoff and notification removal.
            // A stale key is acknowledged without touching its replacement.
            self.execution_closes
                .acknowledge_recompute_failure(&closing, &target.key);
        }
        if handled {
            if let Err(error) = self.drain_execution_closes() {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(super) fn finish_execution_recompute(
        &self,
        sequences: &mut HashMap<RequestId, SequenceState>,
        request_id: &RequestId,
        ended: &ExecutorSequence,
        recompute: RecomputeClose,
        slot: &mut Option<RequestSlotLease>,
    ) -> (bool, Result<()>) {
        let RecomputeClose {
            frontier,
            transaction,
            attempted_width,
            observed_free_blocks,
            readiness,
            failure_targets: _,
            failure_error: _,
            failure_marked: _,
        } = recompute;
        let reject_recompute = |error: FerrumError| {
            readiness.cancel();
            match self.scheduler.abort_execution_capacity_yield(
                &transaction, false, attempted_width, observed_free_blocks,
            ) {
                Ok(_) => error,
                Err(reconcile) => FerrumError::internal(format!(
                    "recompute finalization failed ({error}); pressure reconciliation failed ({reconcile})",
                )),
            }
        };
        if frontier.sequence() != ended {
            return (
                false,
                Err(reject_recompute(FerrumError::internal(
                    "recompute terminal proof belongs to another incarnation",
                ))),
            );
        }
        let live = sequences.get_mut(request_id).filter(|sequence| {
            sequence.execution_frontier.as_ref() == Some(&frontier)
                && !self.shutdown_started.load(Ordering::Acquire)
                && matches!(
                    self.scheduler.trace_phase(request_id),
                    Some(RequestPhase::Prefilling | RequestPhase::Decoding)
                )
        });
        let Some(sequence) = live else {
            // The product request was cancelled or replaced while the exact
            // old executor remained Held. Never revive its product identity.
            readiness.cancel();
            return (
                false,
                self.scheduler
                    .abort_execution_capacity_yield(
                        &transaction,
                        false,
                        attempted_width,
                        observed_free_blocks,
                    )
                    .map(|_| ()),
            );
        };
        if sequence.request_slot.is_some() || sequence.execution_close.is_some() {
            return (
                false,
                Err(reject_recompute(FerrumError::internal(
                    "recompute found conflicting product closure ownership",
                ))),
            );
        }
        sequence.execution_frontier = None;
        sequence.prefill_complete = false;
        sequence.prefill_tokens_processed = 0;
        sequence.phase = RequestPhase::Waiting;
        sequence.tokens_this_iteration = 0;
        sequence.preemption_count = sequence.preemption_count.saturating_add(1);
        sequence.request_slot = slot.take();
        // Logical output, RNG, sampling history and emitted offsets remain
        // untouched; a later admission rebuilds the complete causal context.
        readiness.mark_ready();

        let source_progress = if let Some(condition) = transaction.wait_condition().as_native() {
            let mut availability = Vec::new();
            self.model_executor
                .write_execution_capacity_snapshot(&mut availability)
                .and_then(|epochs| {
                    if epochs.is_none() {
                        return Err(FerrumError::internal(
                            "capacity yield lost its native source evidence",
                        ));
                    }
                    condition
                        .changed_since(&availability)
                        .map_err(|error| FerrumError::internal(error.to_string()))
                })
        } else {
            // The allocator query selected this exact reference owner. Its
            // real Ended above proves withdrawal, including a shared block
            // that still has another contributor. Preserve the same pressure
            // hold until the blocked owner progresses; only a fresh allocator
            // reservation can establish that enough blocks are now free.
            Ok(true)
        };
        match source_progress {
            Ok(true) => match self.scheduler.complete_execution_capacity_yield(
                &transaction,
                attempted_width,
                observed_free_blocks,
            ) {
                Ok(completion) => {
                    if completion.victim_requeued() {
                        self.total_preemptions.fetch_add(1, Ordering::Relaxed);
                        counter!("ferrum.engine.preemptions_total").increment(1);
                    }
                    self.write_scheduler_trace_event(serde_json::json!({
                        "event": "scheduler_pressure_release_fence_completed",
                        "episode_id": transaction.episode_id().get(),
                        "victim_request_id": request_id,
                        "progress_owner_id": transaction.progress_owner_id(),
                        "physical_release_completed": true,
                        "victim_requeued": completion.victim_requeued(),
                        "release_transition_ordinal": completion.release_transition_ordinal().get(),
                    }));
                    (completion.victim_requeued(), Ok(()))
                }
                Err(failure) => {
                    let (requeued, error) = failure.into_parts();
                    // The queue mutation is already known. Only reconcile
                    // the failed fence here; never requeue its victim twice.
                    let error = match self.scheduler.abort_execution_capacity_yield(
                        &transaction, false, attempted_width, observed_free_blocks,
                    ) {
                        Ok(_) => error,
                        Err(reconcile) => FerrumError::internal(format!(
                            "capacity yield finalization failed ({error}); reconciliation failed ({reconcile})",
                        )),
                    };
                    (requeued, Err(error))
                }
            },
            other => {
                // Ended is already established, so reconciliation may requeue
                // only the cleared incarnation. It never grants admission.
                let reconciliation = self.scheduler.abort_execution_capacity_yield(
                    &transaction,
                    true,
                    attempted_width,
                    observed_free_blocks,
                );
                let error = other.err().unwrap_or_else(|| {
                    FerrumError::internal("capacity yield did not advance the failed exact source")
                });
                match reconciliation {
                    Ok((requeued, _, _)) => (requeued, Err(error)),
                    Err(reconcile) => (false, Err(FerrumError::internal(format!(
                        "capacity yield source check failed ({error}); reconciliation failed ({reconcile})",
                    )))),
                }
            }
        }
    }
}
