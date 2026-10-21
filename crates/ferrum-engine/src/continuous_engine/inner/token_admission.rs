//! Admission owns its preparation while the executor awaits; scheduler and
//! outer sequence locks cover only selection and exact outcome publication.
use super::*;
use ferrum_execution::token_execution::ExecutionFrontier;
use ferrum_scheduler::implementations::continuous::{PrefixRequestKey, TokenAdmissionApply};

struct PreparedTokenAdmission<'a> {
    engine: &'a EngineInner,
    request: RequestId,
    handle: Option<Arc<SequenceHandle>>,
    key: Option<PrefixRequestKey>,
    frontier: Option<ExecutionFrontier>,
    newly_admitted: bool,
}

impl<'a> PreparedTokenAdmission<'a> {
    fn new(engine: &'a EngineInner, request: RequestId, key: Option<PrefixRequestKey>) -> Self {
        Self {
            engine,
            handle: engine.sequences.get(&request),
            request,
            key,
            frontier: None,
            newly_admitted: false,
        }
    }

    async fn abort(&mut self, error: FerrumError) -> Result<()> {
        let Some(key) = self.key.clone() else {
            return Err(FerrumError::internal(
                "unpublished admission lacks a scheduler identity",
            ));
        };
        let Some(handle) = self.handle.as_ref() else {
            return Ok(());
        };
        {
            let mut state = handle.lock();
            if !self.engine.sequences.is_current(handle) {
                return Ok(());
            }
            let Some(sequence) = state
                .as_mut()
                .filter(|sequence| sequence.pending_token_admission.as_ref() == Some(&key))
            else {
                return Ok(());
            };
            if self.newly_admitted && sequence.execution_frontier.is_none() {
                sequence.execution_frontier = self.frontier.take();
                self.newly_admitted = false;
            }
        }
        // Explicit-error completion atomically moves the exact frontier and
        // slot into the existing close registry before its first await.
        self.engine
            .complete_unpublished_admission(handle, &key, error)
            .await
    }

    fn publish(&mut self) -> Result<()> {
        let handle = self
            .handle
            .as_ref()
            .ok_or_else(|| FerrumError::internal("admission has no original outer request"))?;
        let mut state = handle.lock();
        if !self.engine.sequences.is_current(handle) {
            return Err(FerrumError::internal(
                "admission outer request incarnation changed",
            ));
        }
        let sequence = state
            .as_mut()
            .filter(|sequence| sequence.pending_token_admission == self.key)
            .ok_or_else(|| FerrumError::internal("admission outer request incarnation changed"))?;
        let frontier = self
            .frontier
            .as_ref()
            .ok_or_else(|| FerrumError::internal("admitted request has no execution frontier"))?;
        if self.newly_admitted {
            if sequence.execution_frontier.is_some() {
                return Err(FerrumError::internal(
                    "admission would replace an existing execution frontier",
                ));
            }
            sequence.execution_frontier = self.frontier.take();
            self.newly_admitted = false;
        } else if sequence.execution_frontier.as_ref() != Some(frontier) {
            return Err(FerrumError::internal(
                "repeated admission changed its execution frontier",
            ));
        }
        sequence.pending_token_admission = None;
        Ok(())
    }
}

impl Drop for PreparedTokenAdmission<'_> {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            let mut state = handle.lock();
            if let Some(sequence) = state.as_mut().filter(|sequence| {
                self.engine.sequences.is_current(handle)
                    && sequence.pending_token_admission == self.key
            }) {
                sequence.pending_token_admission = None;
                if self.newly_admitted && sequence.execution_frontier.is_none() {
                    // An error after preparation cannot lose the original outer
                    // owner or release its request slot. Ordinary terminal handling
                    // closes this exact frontier, including malformed progress.
                    sequence.execution_frontier = self.frontier.take();
                }
            }
        }
        if self.newly_admitted {
            if let Some(frontier) = self.frontier.take() {
                // The original outer owner disappeared or was replaced. This
                // fresh physical admission still needs independent retirement.
                if let Err(error) = self.engine.begin_execution_close(
                    self.request.clone(),
                    frontier.sequence().clone(),
                    None,
                    ExecutionEndReason::Cancelled,
                ) {
                    warn!(%error, "Could not register unpublished token admission");
                }
            }
        }
    }
}

/// Reuse the capacity snapshot allocation without keeping its cache mutex
/// locked across device preparation. This guard owns data, never a lock guard.
struct AdmissionAvailability<'a> {
    cache: &'a Mutex<Vec<CapacityAvailabilityEpoch>>,
    values: Vec<CapacityAvailabilityEpoch>,
}

impl<'a> AdmissionAvailability<'a> {
    fn take(cache: &'a Mutex<Vec<CapacityAvailabilityEpoch>>) -> Self {
        Self {
            cache,
            values: std::mem::take(&mut *cache.lock()),
        }
    }
}

impl Drop for AdmissionAvailability<'_> {
    fn drop(&mut self) {
        *self.cache.lock() = std::mem::take(&mut self.values);
    }
}

impl EngineInner {
    async fn probe_executor_prefill_admission(
        &self,
        request: &InferenceRequest,
        capture_trace: bool,
        pending: &mut PreparedTokenAdmission<'_>,
    ) -> ExecutorPrefillProbeResult {
        let Some((
            input_tokens,
            maximum_sequence_tokens,
            product_prompt_tokens,
            replayed_output_tokens,
            existing_frontier,
            closing,
        )) = pending.handle.as_ref().and_then(|handle| {
            let mut state = handle.lock();
            if !self.sequences.is_current(handle) {
                return None;
            }
            state.as_mut().and_then(|sequence| {
                if sequence.pending_token_admission.is_some()
                    && sequence.pending_token_admission != pending.key
                {
                    return None;
                }
                sequence.pending_token_admission = pending.key.clone();
                Some((
                    sequence.prefill_context_tokens(),
                    sequence.model_maximum_sequence_tokens(),
                    sequence.input_tokens.len(),
                    sequence.generated_tokens.len(),
                    sequence.execution_frontier.clone(),
                    sequence.execution_close.is_some(),
                ))
            })
        })
        else {
            let error = FerrumError::internal(format!(
                "request {} reached typed admission before its sequence state was published",
                request.id
            ));
            return ExecutorPrefillProbeResult {
                trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                    request_id: request.id.clone(),
                    input_token_count: None,
                    maximum_sequence_tokens: None,
                    evidence: ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                }),
                outcome: AdmissionProbeOutcome::Faulted(error),
                maintenance: None,
            };
        };
        let admission = match ExecutorPrefillAdmission::for_product_request(
            &request.id,
            &input_tokens,
            maximum_sequence_tokens,
            product_prompt_tokens,
            replayed_output_tokens,
        ) {
            Ok(admission) => admission,
            Err(error) => {
                return ExecutorPrefillProbeResult {
                    trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                        request_id: request.id.clone(),
                        input_token_count: Some(input_tokens.len()),
                        maximum_sequence_tokens: Some(maximum_sequence_tokens),
                        evidence: ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                    }),
                    outcome: AdmissionProbeOutcome::Faulted(error),
                    maintenance: None,
                };
            }
        };
        let capture_admission_trace = |evidence| {
            capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                request_id: request.id.clone(),
                input_token_count: Some(input_tokens.len()),
                maximum_sequence_tokens: Some(maximum_sequence_tokens),
                evidence,
            })
        };
        let adapter = match ferrum_interfaces::adapter::parse_lora_selection(&request.metadata) {
            Ok(adapter) => adapter,
            Err(error) => {
                return ExecutorPrefillProbeResult {
                    trace: capture_admission_trace(ExecutorPrefillAdmissionTraceEvidence::Faulted(
                        error.to_string(),
                    )),
                    outcome: AdmissionProbeOutcome::Faulted(error),
                    maintenance: None,
                }
            }
        };
        if let Some(frontier) = existing_frontier {
            if !closing
                && frontier.consumed_tokens() == 0
                && frontier.sequence().observed_end().is_none()
            {
                pending.frontier = Some(frontier);
                let receipt = ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                };
                return ExecutorPrefillProbeResult {
                    trace: capture_admission_trace(
                        ExecutorPrefillAdmissionTraceEvidence::Admitted(receipt.clone()),
                    ),
                    outcome: AdmissionProbeOutcome::Admitted(receipt),
                    maintenance: None,
                };
            }
            let error = FerrumError::internal(
                "admission cannot replace an advanced or ended token incarnation",
            );
            return ExecutorPrefillProbeResult {
                trace: capture_admission_trace(ExecutorPrefillAdmissionTraceEvidence::Faulted(
                    error.to_string(),
                )),
                outcome: AdmissionProbeOutcome::Faulted(error),
                maintenance: None,
            };
        }
        match self
            .model_executor
            .admit_tokens(ExecutionAdmission {
                context: admission,
                adapter,
            })
            .await
        {
            Ok(TokenAdmissionOutcome::Admitted(frontier)) => {
                let valid_initial =
                    frontier.consumed_tokens() == 0 && frontier.sequence().observed_end().is_none();
                // From Ready through queue acceptance/publication there is no
                // await. The guard immediately owns even malformed progress.
                pending.frontier = Some(frontier);
                pending.newly_admitted = true;
                if valid_initial {
                    let receipt = ExecutorPrefillAdmissionReceipt {
                        request_id: request.id.clone(),
                    };
                    ExecutorPrefillProbeResult {
                        trace: capture_admission_trace(
                            ExecutorPrefillAdmissionTraceEvidence::Admitted(receipt.clone()),
                        ),
                        outcome: AdmissionProbeOutcome::Admitted(receipt),
                        maintenance: None,
                    }
                } else {
                    let error = FerrumError::internal(
                        "token admission did not return an initial live incarnation",
                    );
                    ExecutorPrefillProbeResult {
                        trace: capture_admission_trace(
                            ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                        ),
                        outcome: AdmissionProbeOutcome::Faulted(error),
                        maintenance: None,
                    }
                }
            }
            Ok(TokenAdmissionOutcome::CapacityDeferred(deferred)) => ExecutorPrefillProbeResult {
                trace: capture_admission_trace(
                    ExecutorPrefillAdmissionTraceEvidence::CapacityDeferred(deferred),
                ),
                outcome: AdmissionProbeOutcome::CapacityDeferred,
                maintenance: None,
            },
            Ok(TokenAdmissionOutcome::Deferred(deferred)) => {
                if deferred.action() == DeferredAction::AwaitBackingGrowth {
                    self.model_executor.cancel_prefill_admission(&request.id);
                    let error = FerrumError::internal(format!(
                        "executor returned AwaitBackingGrowth for {} without retaining typed maintenance state",
                        request.id
                    ));
                    return ExecutorPrefillProbeResult {
                        trace: capture_admission_trace(
                            ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                        ),
                        outcome: AdmissionProbeOutcome::Faulted(error),
                        maintenance: None,
                    };
                }
                let outcome = AdmissionProbeOutcome::Deferred(AdmissionDeferral::from_admission(
                    &deferred, 0,
                ));
                ExecutorPrefillProbeResult {
                    trace: capture_admission_trace(
                        ExecutorPrefillAdmissionTraceEvidence::Deferred(deferred),
                    ),
                    outcome,
                    maintenance: None,
                }
            }
            Ok(TokenAdmissionOutcome::MaintenanceDeferred(deferred)) => {
                if deferred.request_id() != &request.id {
                    self.model_executor.cancel_prefill_admission(&request.id);
                    let error = FerrumError::internal(format!(
                        "executor maintenance deferral belongs to {}, expected {}",
                        deferred.request_id(),
                        request.id
                    ));
                    return ExecutorPrefillProbeResult {
                        trace: capture_admission_trace(
                            ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                        ),
                        outcome: AdmissionProbeOutcome::Faulted(error),
                        maintenance: None,
                    };
                }
                let observed = deferred.observed();
                let outcome = AdmissionProbeOutcome::Deferred(AdmissionDeferral::new(
                    DeferredAction::AwaitBackingGrowth,
                    AdmissionWakeEpochs::new(
                        observed.coordinator_id,
                        observed.release_epoch,
                        observed.capacity_epoch,
                        0,
                    ),
                    deferred.wait_condition().clone(),
                ));
                ExecutorPrefillProbeResult {
                    trace: capture_admission_trace(
                        ExecutorPrefillAdmissionTraceEvidence::MaintenanceDeferred(
                            deferred.clone(),
                        ),
                    ),
                    outcome,
                    maintenance: Some(deferred),
                }
            }
            Ok(TokenAdmissionOutcome::PermanentRejected(rejected)) => ExecutorPrefillProbeResult {
                trace: capture_admission_trace(
                    ExecutorPrefillAdmissionTraceEvidence::PermanentRejected(rejected.clone()),
                ),
                outcome: AdmissionProbeOutcome::PermanentRejected(rejected),
                maintenance: None,
            },
            Err(error) => ExecutorPrefillProbeResult {
                trace: capture_admission_trace(ExecutorPrefillAdmissionTraceEvidence::Faulted(
                    error.to_string(),
                )),
                outcome: AdmissionProbeOutcome::Faulted(error),
                maintenance: None,
            },
        }
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) async fn probe_executor_prefill_admission_for_test(
        &self,
        request: &InferenceRequest,
    ) -> bool {
        let mut pending = PreparedTokenAdmission::new(self, request.id.clone(), None);
        let result = self
            .probe_executor_prefill_admission(request, false, &mut pending)
            .await;
        matches!(result.outcome, AdmissionProbeOutcome::Admitted(_)) && pending.publish().is_ok()
    }

    pub(in crate::continuous_engine) async fn prepare_dynamic_admission_round(
        &self,
        maximum_admissions: usize,
        mut planning: Option<&mut TimePlanningRound>,
    ) -> Result<(usize, Vec<ExecutorPrefillMaintenanceDeferral>)> {
        if let Some(round) = planning.as_deref_mut() {
            // This path is shared by ordinary admission and fill-first retries.
            // Refresh membership, clocks and resource evidence after earlier
            // asynchronous admissions and their terminal cleanup;
            // the immutable cost/stopping snapshot remains fixed for the turn.
            self.capture_planning_request_facts(round, round.max_request_participants());
            self.scheduler.prepare_joint_admission_gate(
                &ferrum_interfaces::BatchHint {
                    max_batch_size: self.config.batching.max_batch_size,
                    max_tokens: self.config.batching.max_num_batched_tokens,
                    target_latency_ms: None,
                    available_memory: None,
                    resource_constraints:
                        ferrum_interfaces::scheduler::ResourceConstraints::default(),
                },
                round,
                Instant::now(),
            );
        }
        let mut availability = AdmissionAvailability::take(&self.dynamic_admission_availability);
        let epochs = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability.values)?;
        let wake = epochs.map(|epochs| {
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability.values,
            )
        });
        let capture_trace = self.scheduler_trace_jsonl.is_some();
        let mut records = Vec::<ExecutorSchedulerTraceRecord>::new();
        let mut maintenance =
            PendingPrefillMaintenance::new(self.model_executor.as_ref().as_ref(), Vec::new());
        let mut prefix_pressure = false;
        let prepared = async {
            let mut round = self
                .scheduler
                .begin_token_admission(maximum_admissions, wake)?;
            loop {
                let Some(candidate) = round.next(
                    &mut |observation| {
                        if capture_trace {
                            records.push(ExecutorSchedulerTraceRecord::AdmissionQueueObservation(
                                observation,
                            ));
                        }
                    },
                    planning.as_deref_mut(),
                )?
                else {
                    break;
                };
                let mut pending = PreparedTokenAdmission::new(
                    self,
                    candidate.request().id.clone(),
                    Some(candidate.key().clone()),
                );
                let result = self
                    .probe_executor_prefill_admission(
                        candidate.request(),
                        capture_trace,
                        &mut pending,
                    )
                    .await;
                if (self
                    .config
                    .scheduler
                    .prefix_rendezvous_max_wait_ms
                    .is_some()
                    || !self.prefix_restore_pending.lock().is_empty())
                    && matches!(&result.outcome, AdmissionProbeOutcome::Deferred(_))
                    && result.maintenance.is_none()
                {
                    prefix_pressure = true;
                }
                if let Some(trace) = result.trace {
                    records.push(ExecutorSchedulerTraceRecord::PrefillAdmission(trace));
                }
                match round.apply(candidate, result.outcome) {
                    TokenAdmissionApply::Applied { accepted } => {
                        if let Some(accepted) = accepted {
                            if let Err(error) = pending.publish() {
                                let rollback = self.scheduler.cancel_token_admission(&accepted);
                                let completion = pending.abort(error.clone()).await;
                                rollback?;
                                completion?;
                                return Err(error);
                            }
                        }
                        if let Some(deferral) = result.maintenance {
                            maintenance.pending.push(deferral);
                        }
                    }
                    TokenAdmissionApply::Unapplied { outcome, error } => {
                        let preparation_error = match outcome {
                            AdmissionProbeOutcome::Faulted(error) => Some(error),
                            _ => None,
                        };
                        if let Some(deferral) = result.maintenance {
                            self.model_executor
                                .cancel_prefill_admission(deferral.request_id());
                        }
                        // A cancelled queue ticket has no later scheduler
                        // completion. Retire its exact outer owner now; a same-ID
                        // replacement is outside this admission's authority.
                        pending
                            .abort(preparation_error.or_else(|| error.clone()).unwrap_or_else(
                                || {
                                    FerrumError::cancelled(
                                        "request was cancelled during executor admission",
                                    )
                                },
                            ))
                            .await?;
                        if let Some(error) = error {
                            return Err(error);
                        }
                    }
                }
            }
            round.finish()
        }
        .await;
        if prefix_pressure {
            self.release_prefix_rendezvous_for_capacity_pressure();
        }
        for record in records {
            match record {
                ExecutorSchedulerTraceRecord::PrefillAdmission(trace) => {
                    self.trace_executor_prefill_admission(trace)
                }
                ExecutorSchedulerTraceRecord::AdmissionQueueObservation(observation) => {
                    self.trace_executor_admission_queue_observation(observation)
                }
            }
        }
        // A later candidate error must not strand an earlier rejected owner.
        // Cleanup also retains its own failures, without replacing the primary
        // scheduling/preparation error when both operations fail.
        let completion = self.complete_typed_admission_failures().await;
        let receipt = match (prepared, completion) {
            (Ok(receipt), Ok(())) => receipt,
            (Ok(_), Err(error)) | (Err(error), Ok(())) => return Err(error),
            (Err(error), Err(cleanup_error)) => {
                warn!(%error, %cleanup_error, "Admission and terminal cleanup both failed");
                return Err(error);
            }
        };
        Ok((receipt.admitted(), maintenance.into_pending()))
    }
}

#[cfg(test)]
mod tests;
