use super::*;
use ferrum_execution::model_executor::{
    PlanRuntimePrefixRestoreDeferral, PlanRuntimePrefixRestoreInput,
    PlanRuntimePrefixRestoreOutcome,
};
use ferrum_execution::token_execution::ExecutorSequence;
use ferrum_interfaces::scheduler::PreparedPrefixRestore;
use ferrum_scheduler::implementations::continuous::PrefixRestoreCapacityStatus;

pub(in super::super) struct PendingPrefixRestore {
    handle: Arc<SequenceHandle>,
    prepared: PreparedPrefixRestore,
    tokens: Vec<TokenId>,
    maximum_sequence_tokens: usize,
    sequence: ExecutorSequence,
    deferral: PlanRuntimePrefixRestoreDeferral,
}

impl EngineInner {
    pub(super) fn discard_pending_prefix_restore(&self, request_id: &RequestId) {
        let pending = self.prefix_restore_pending.lock().remove(request_id);
        if let Some(pending) = pending {
            if let Err(error) = self
                .scheduler
                .abandon_prefix_restore_capacity(&pending.prepared)
            {
                warn!(%request_id, %error, "Could not abandon optional prefix restore gate");
            }
            // Drop the immutable device pin outside the container lock.
            drop(pending);
        }
    }

    pub(super) fn release_pending_prefix_restores_for_capacity_pressure(&self) {
        let pending = std::mem::take(&mut *self.prefix_restore_pending.lock());
        for (_, pending) in pending {
            if let Err(error) = self
                .scheduler
                .abandon_prefix_restore_capacity(&pending.prepared)
            {
                warn!(%error, "Could not release optional prefix restore under capacity pressure");
            }
            drop(pending);
        }
    }

    fn prefix_restore_resume_status(
        &self,
        prepared: &PreparedPrefixRestore,
        release: &ExecutionCapacityReleaseSnapshot,
    ) -> Result<PrefixRestoreCapacityStatus> {
        let mut availability = Vec::new();
        let epochs = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)?
            .ok_or_else(|| FerrumError::scheduler("prefix restore retry lost capacity epochs"))?;
        let wake = AdmissionWakeSnapshot::new(
            AdmissionWakeEpochs::new(
                epochs.coordinator_id,
                epochs.release_epoch,
                epochs.capacity_epoch,
                0,
            ),
            &availability,
        );
        self.scheduler
            .resume_prefix_restore_after_capacity(prepared, wake, release)
    }

    /// Planning owns the iteration gate. Each preparation additionally retains
    /// its original request handle across the device await and publication.
    pub(super) async fn restore_admitted_prefixes(&self) -> Result<()> {
        let candidates = self
            .sequences
            .snapshot()
            .into_iter()
            .filter_map(|handle| {
                let state = handle.lock();
                let sequence = state.as_ref()?;
                let eligible = self.sequences.is_current(&handle)
                    && !sequence.prefill_complete
                    && sequence.prefill_tokens_processed == 0
                    && sequence
                        .execution_frontier
                        .as_ref()
                        .is_some_and(|frontier| {
                            frontier.consumed_tokens() == 0
                                && frontier.sequence().observed_end().is_none()
                        });
                let prompt_tokens = sequence.prefill_context_len();
                drop(state);
                eligible.then_some((handle, prompt_tokens))
            })
            .collect::<Vec<_>>();

        let stale = self
            .prefix_restore_pending
            .lock()
            .keys()
            .filter(|id| {
                !candidates
                    .iter()
                    .any(|(candidate, _)| candidate.request_id() == *id)
            })
            .cloned()
            .collect::<Vec<_>>();
        for request_id in stale {
            self.discard_pending_prefix_restore(&request_id);
        }

        for (handle, prompt_tokens) in candidates {
            let request_id = handle.request_id().clone();
            let Some((tokens, maximum_sequence_tokens, incarnation)) = (|| {
                let state = handle.lock();
                if !self.sequences.is_current(&handle) {
                    return None;
                }
                state.as_ref().and_then(|sequence| {
                    Some((
                        sequence.prefill_context_tokens(),
                        sequence.model_maximum_sequence_tokens(),
                        sequence.execution_frontier.as_ref()?.sequence().clone(),
                    ))
                })
            })() else {
                continue;
            };
            let pending = self.prefix_restore_pending.lock().remove(&request_id);
            // This observation visits peer requests, so collect it before
            // taking the original request lock. Scheduler consumption below
            // still revalidates its exact preparation under that lock.
            let pending = pending
                .map(|pending| {
                    self.execution_capacity_release_snapshot()
                        .map(|release| (pending, release))
                })
                .transpose()?;
            let (prepared, retry, checkpoint) = {
                // Do not let cancellation remove this owner between scheduler
                // preparation and retaining its optional retry/checkpoint state.
                let state = handle.lock();
                if !self.sequences.is_current(&handle)
                    || !state.as_ref().is_some_and(|sequence| {
                        !sequence.prefill_complete
                            && sequence.prefill_tokens_processed == 0
                            && sequence
                                .execution_frontier
                                .as_ref()
                                .is_some_and(|frontier| {
                                    frontier.sequence() == &incarnation
                                        && frontier.consumed_tokens() == 0
                                })
                    })
                {
                    if let Some((pending, _)) = pending {
                        self.scheduler
                            .abandon_prefix_restore_capacity(&pending.prepared)?;
                    }
                    continue;
                }
                let (prepared, retry) = if let Some((pending, release)) = pending {
                    if !Arc::ptr_eq(&pending.handle, &handle)
                        || pending.tokens != tokens
                        || pending.maximum_sequence_tokens != maximum_sequence_tokens
                        || pending.sequence != incarnation
                    {
                        self.scheduler
                            .abandon_prefix_restore_capacity(&pending.prepared)?;
                        continue;
                    }
                    match self.prefix_restore_resume_status(&pending.prepared, &release)? {
                        PrefixRestoreCapacityStatus::Pending => {
                            self.prefix_restore_pending
                                .lock()
                                .insert(request_id, pending);
                            continue;
                        }
                        PrefixRestoreCapacityStatus::Fallback
                        | PrefixRestoreCapacityStatus::Stale => {
                            self.scheduler
                                .abandon_prefix_restore_capacity(&pending.prepared)?;
                            continue;
                        }
                        PrefixRestoreCapacityStatus::Retry => {
                            (pending.prepared, Some(pending.deferral))
                        }
                    }
                } else {
                    let Some(prepared) =
                        self.scheduler
                            .prepare_prefix_restore(&request_id, 0, prompt_tokens)?
                    else {
                        continue;
                    };
                    (prepared, None)
                };
                let checkpoint = retry
                    .is_none()
                    .then(|| self.take_rendezvous_checkpoint(&request_id))
                    .flatten();
                (prepared, retry, checkpoint)
            };
            let started = Instant::now();
            let output = match self
                .model_executor
                .try_restore_plan_runtime_prefix(PlanRuntimePrefixRestoreInput {
                    request_id: &request_id,
                    input_tokens: &tokens,
                    maximum_sequence_tokens,
                    checkpoint: checkpoint.as_deref(),
                    retry: retry.as_ref(),
                })
                .await
            {
                Ok(PlanRuntimePrefixRestoreOutcome::Restored(output)) => output,
                Ok(PlanRuntimePrefixRestoreOutcome::Unavailable) => continue,
                Ok(PlanRuntimePrefixRestoreOutcome::Deferred(deferral)) => {
                    let release = self.execution_capacity_release_snapshot()?;
                    let state = handle.lock();
                    if !self.sequences.is_current(&handle)
                        || !state.as_ref().is_some_and(|sequence| {
                            !sequence.prefill_complete
                                && sequence.prefill_tokens_processed == 0
                                && sequence
                                    .execution_frontier
                                    .as_ref()
                                    .is_some_and(|frontier| {
                                        frontier.sequence() == &incarnation
                                            && frontier.consumed_tokens() == 0
                                    })
                        })
                    {
                        self.scheduler.abandon_prefix_restore_capacity(&prepared)?;
                        continue;
                    }
                    let retained = self.scheduler.defer_prefix_restore_for_capacity(
                        &prepared,
                        deferral.capacity(),
                        &release,
                    )?;
                    self.write_scheduler_trace_event(serde_json::json!({
                        "event": "scheduler_prefix_restore_capacity",
                        "request_id": request_id,
                        "candidate_prefix_tokens": deferral.checkpoint().boundary(),
                        "retained": retained,
                        "stage": deferral.capacity().stage(),
                        "wait_condition": deferral.capacity().wait_condition(),
                        "typed_evidence": deferral.capacity().evidence(),
                    }));
                    if retained {
                        self.prefix_restore_pending.lock().insert(
                            request_id,
                            PendingPrefixRestore {
                                handle: Arc::clone(&handle),
                                prepared,
                                tokens,
                                maximum_sequence_tokens,
                                sequence: incarnation.clone(),
                                deferral,
                            },
                        );
                    } else {
                        self.scheduler.abandon_prefix_restore_capacity(&prepared)?;
                    }
                    continue;
                }
                Err(error) => {
                    drop(prepared);
                    self.complete_request_exact_frontier_detached(
                        &handle,
                        &incarnation,
                        FinishReason::Error,
                        Some(error),
                    )?;
                    continue;
                }
            };
            if let Err(error) = output.validate_for(&request_id, prompt_tokens, &incarnation) {
                drop(output);
                drop(prepared);
                self.complete_request_exact_frontier_detached(
                    &handle,
                    &incarnation,
                    FinishReason::Error,
                    Some(error),
                )?;
                continue;
            }
            let restored_tokens = output.restored_tokens();
            let committed = (|| {
                let mut state = handle.lock();
                if !self.sequences.is_current(&handle) {
                    return Err(FerrumError::internal(
                        "prefix restore outer request incarnation changed",
                    ));
                }
                let sequence = state.as_mut().ok_or_else(|| {
                    FerrumError::internal("prefix restore lost its outer sequence")
                })?;
                if sequence.prefill_tokens_processed != 0
                    || sequence.prefill_complete
                    || sequence.prefill_context_tokens() != tokens
                    || sequence.execution_frontier.as_ref().is_none_or(|frontier| {
                        frontier.sequence() != &incarnation || frontier.consumed_tokens() != 0
                    })
                {
                    return Err(FerrumError::internal(
                        "prefix restore outer sequence changed before publication",
                    ));
                }
                if output.token_sequence() != &incarnation {
                    return Err(FerrumError::internal(
                        "prefix publication belongs to a different token incarnation",
                    ));
                }
                // The actual native acknowledgement opens the execution gate.
                // Publish its frontier before a fallible scheduler transition:
                // failure closes this incarnation, never replays from zero.
                let frontier = output.acknowledge_tokens()?;
                if frontier.sequence() != &incarnation
                    || frontier.consumed_tokens() != restored_tokens
                {
                    return Err(FerrumError::internal(
                        "prefix acknowledgement returned conflicting token progress",
                    ));
                }
                sequence.execution_frontier = Some(frontier);
                sequence.prefill_tokens_processed = restored_tokens;
                self.scheduler
                    .commit_prefix_restored(prepared, restored_tokens)
            })();
            if let Err(error) = committed {
                self.complete_request_exact_frontier_detached(
                    &handle,
                    &incarnation,
                    FinishReason::Error,
                    Some(error),
                )?;
                continue;
            }
            self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.prefix_cache_hits").increment(1);
            counter!("ferrum.engine.prefix_cache_tokens_total").increment(restored_tokens as u64);
            self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.prefix_restore",
                ProfileEventKind::TimedSpan,
                ProfileStatus::Ok,
                Some(duration_to_us(started.elapsed())),
                BTreeMap::from([
                    (
                        "restored_tokens".to_string(),
                        serde_json::json!(restored_tokens),
                    ),
                    (
                        "prompt_tokens".to_string(),
                        serde_json::json!(prompt_tokens),
                    ),
                ]),
                BTreeMap::new(),
                None,
            );
        }
        Ok(())
    }
}
