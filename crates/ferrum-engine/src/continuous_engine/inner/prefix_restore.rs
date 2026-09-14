use super::*;
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreDeferral, PlanRuntimePrefixRestoreInput,
    PlanRuntimePrefixRestoreOutcome,
};
use ferrum_interfaces::scheduler::PreparedPrefixRestore;
use ferrum_scheduler::implementations::continuous::PrefixRestoreCapacityStatus;

pub(in super::super) struct PendingPrefixRestore {
    prepared: PreparedPrefixRestore,
    tokens: Vec<TokenId>,
    maximum_sequence_tokens: usize,
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
    ) -> Result<PrefixRestoreCapacityStatus> {
        let release = self.execution_capacity_release_snapshot()?;
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
            .resume_prefix_restore_after_capacity(prepared, wake, &release)
    }

    /// Called under the existing iteration lock, after admission and before
    /// publishing a batch. Submit/replacement uses the same lock, so one
    /// scheduler preparation cannot be applied to another outer sequence.
    pub(super) async fn restore_admitted_prefixes(&self) -> Result<()> {
        let candidates = self
            .sequences
            .read()
            .iter()
            .filter(|(_, sequence)| {
                !sequence.prefill_complete && sequence.prefill_tokens_processed == 0
            })
            .map(|(id, sequence)| (id.clone(), sequence.prefill_context_len()))
            .collect::<Vec<_>>();

        let stale = self
            .prefix_restore_pending
            .lock()
            .keys()
            .filter(|id| !candidates.iter().any(|(candidate, _)| candidate == *id))
            .cloned()
            .collect::<Vec<_>>();
        for request_id in stale {
            self.discard_pending_prefix_restore(&request_id);
        }

        for (request_id, prompt_tokens) in candidates {
            let Some((tokens, maximum_sequence_tokens)) =
                self.sequences.read().get(&request_id).map(|sequence| {
                    (
                        sequence.prefill_context_tokens(),
                        sequence.model_maximum_sequence_tokens(),
                    )
                })
            else {
                continue;
            };
            let pending = self.prefix_restore_pending.lock().remove(&request_id);
            let (prepared, retry) = if let Some(pending) = pending {
                if pending.tokens != tokens
                    || pending.maximum_sequence_tokens != maximum_sequence_tokens
                {
                    self.scheduler
                        .abandon_prefix_restore_capacity(&pending.prepared)?;
                    continue;
                }
                match self.prefix_restore_resume_status(&pending.prepared)? {
                    PrefixRestoreCapacityStatus::Pending => {
                        self.prefix_restore_pending
                            .lock()
                            .insert(request_id, pending);
                        continue;
                    }
                    PrefixRestoreCapacityStatus::Fallback | PrefixRestoreCapacityStatus::Stale => {
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
            let started = Instant::now();
            let checkpoint = retry
                .is_none()
                .then(|| self.take_rendezvous_checkpoint(&request_id))
                .flatten();
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
                                prepared,
                                tokens,
                                maximum_sequence_tokens,
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
                    self.complete_request_with_error(&request_id, error).await?;
                    continue;
                }
            };
            if let Err(error) = output.validate_for(&request_id, prompt_tokens) {
                drop(output);
                drop(prepared);
                self.complete_request_with_error(&request_id, error).await?;
                continue;
            }
            let restored_tokens = output.restored_tokens();
            let committed = (|| {
                let mut sequences = self.sequences.write();
                let sequence = sequences.get_mut(&request_id).ok_or_else(|| {
                    FerrumError::internal("prefix restore lost its outer sequence")
                })?;
                if sequence.prefill_tokens_processed != 0
                    || sequence.prefill_complete
                    || sequence.prefill_context_tokens() != tokens
                {
                    return Err(FerrumError::internal(
                        "prefix restore outer sequence changed before publication",
                    ));
                }
                self.scheduler
                    .commit_prefix_restored(prepared, restored_tokens)?;
                let update = sequence.commit_plan_runtime_prefill_chunk_resources(
                    Arc::clone(output.kv_cache()),
                    restored_tokens,
                    false,
                );
                // Both outer owners now agree. Only the native publication
                // may open the target execution gate. The closure also drops
                // that owner before error cleanup awaits request release.
                Ok((update, output.acknowledge()))
            })();
            let (update, acknowledged) = match committed {
                Ok(value) => value,
                Err(error) => {
                    self.complete_request_with_error(&request_id, error).await?;
                    continue;
                }
            };
            self.apply_model_cache_ref_update(&request_id, update);
            if let Err(error) = acknowledged {
                self.complete_request_with_error(&request_id, error).await?;
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
