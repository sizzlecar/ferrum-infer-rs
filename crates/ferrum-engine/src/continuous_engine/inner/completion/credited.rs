//! Engine-side projection and terminal handoff for a credited transport.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    HostSettledReceipt, HostTerminalStageV1, PendingHostRow,
};
use crate::continuous_engine::output_flow_runtime::{OutputDelta, OutputTerminalDecision};
use ferrum_interfaces::model_executor::ExecutorCompletionWork;
use ferrum_interfaces::output_flow::{BoundedOutputError, OutputCompletion, OutputHistory};

impl EngineInner {
    /// The physical loop has drained before this call. Closing a credited
    /// request never waits for its HTTP/stdout consumer, including on shutdown.
    pub(in crate::continuous_engine) async fn shutdown_credited_outputs(&self) -> Result<()> {
        let _iteration = self.iteration_lock.lock().await;
        let request_ids: Vec<_> = self
            .sequences
            .read()
            .iter()
            .filter_map(|(id, sequence)| sequence.credited_output.is_some().then(|| id.clone()))
            .collect();
        let mut first_error = None;
        for id in request_ids {
            if let Err(error) = self
                .complete_request_with_error(
                    &id,
                    FerrumError::cancelled("inference engine shut down"),
                )
                .await
            {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(in crate::continuous_engine) async fn complete_credited_output_failures(
        &self,
    ) -> Result<()> {
        use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
        let failed: Vec<_> = self
            .sequences
            .read()
            .iter()
            .filter_map(|(id, sequence)| {
                let output = sequence.credited_output.as_ref()?;
                (output.failure.is_some()
                    || matches!(output.port.readiness(), OutputReadinessState::Closing(_)))
                .then(|| {
                    (
                        id.clone(),
                        output
                            .failure
                            .as_ref()
                            .map(|error| error.message().to_owned())
                            .unwrap_or_else(|| {
                                "output owner closed before inference completed".to_owned()
                            }),
                    )
                })
            })
            .collect();
        for (id, error) in failed {
            self.complete_request_with_error(&id, FerrumError::backend(error))
                .await?;
        }
        Ok(())
    }

    /// Returns false only for the legacy output path. Even a token withheld by
    /// UTF-8/stop projection consumes its commit grant exactly once.
    pub(super) fn send_credited_text(
        &self,
        request_id: &RequestId,
        token: Option<TokenId>,
        terminal: Option<FinishReason>,
    ) -> bool {
        let snapshot = {
            let sequences = self.sequences.read();
            let Some(sequence) = sequences.get(request_id) else {
                return false;
            };
            if sequence.credited_output.is_none() {
                return false;
            }
            super::super::output_projection::StreamProjectionSnapshot::capture(sequence)
        };
        let projected = snapshot.project_fenced(self.tokenizer.as_ref(), terminal);
        let mut sequences = self.sequences.write();
        let Some(sequence) = sequences.get_mut(request_id) else {
            return true;
        };
        let delta = match projected.commit(sequence) {
            None => return true,
            Some(Ok(Some(text))) => text,
            Some(Ok(None)) => String::new(),
            Some(Err(error)) => {
                if let Some(output) = sequence.credited_output.as_mut() {
                    output.failure = Some(BoundedOutputError::new(&error.to_string()));
                    output.port.cancel();
                }
                return true;
            }
        };
        if !delta.is_empty() {
            record_text_prepared(sequence);
        }
        let Some(output) = sequence.credited_output.as_mut() else {
            return true;
        };
        let Some(grant) = output.grant.take() else {
            output.failure = Some(BoundedOutputError::new(
                "token projection has no pre-submission output grant",
            ));
            output.port.cancel();
            return true;
        };
        if output.tokens_before_grant.checked_add(1) != Some(sequence.generated_tokens.len()) {
            drop(grant);
            output.failure = Some(BoundedOutputError::new(
                "token projection exceeded its authorized frontier",
            ));
            output.port.cancel();
            return true;
        }
        output.accepted_ordinal = grant.committed(OutputDelta {
            text: delta,
            token,
            generated_tokens: sequence.generated_tokens.len(),
            created: created_seconds(),
        });
        true
    }

    pub(super) async fn complete_credited_request(
        &self,
        request_id: &RequestId,
        reason: FinishReason,
        error: Option<FerrumError>,
    ) -> Result<()> {
        self.complete_credited_request_inner(request_id, reason, error, None)
            .await
            .0
    }

    pub(in crate::continuous_engine::inner) async fn complete_credited_request_inner(
        &self,
        request_id: &RequestId,
        mut reason: FinishReason,
        error: Option<FerrumError>,
        mut pending: Option<PendingHostRow>,
    ) -> (Result<()>, Option<HostSettledReceipt>) {
        let pending_restore_removed = self.discard_pending_prefix_restore_observed(request_id);
        // Take the whole sequence, retaining its final port field. All costly
        // final decode work happens outside the global sequence map lock.
        let Some(mut sequence) = self.sequences.write().remove(request_id) else {
            return (Ok(()), None);
        };
        let owner_matched = pending
            .as_ref()
            .is_some_and(|pending| pending.matches_owner(&sequence));
        let generated_tokens = u64::try_from(sequence.generated_tokens.len()).ok();
        let mut failure = error
            .as_ref()
            .map(|error| BoundedOutputError::new(&error.to_string()))
            .or_else(|| {
                sequence
                    .credited_output
                    .as_mut()
                    .and_then(|output| output.failure.take())
            });
        if reason == FinishReason::Error && failure.is_none() {
            failure = Some(BoundedOutputError::new("inference failed"));
        }
        let text = if failure.is_none() {
            match sequence.decoded_output_text(self.tokenizer.as_ref(), Some(reason)) {
                Ok(text) => text.into_boxed_str().into_string(),
                Err(error) => {
                    failure = Some(BoundedOutputError::new(&error.to_string()));
                    String::new()
                }
            }
        } else {
            String::new()
        };
        let mut final_text = if failure.is_none() {
            match text.get(sequence.streamed_text_len..) {
                Some(delta) => delta.to_owned(),
                None => {
                    failure = Some(BoundedOutputError::new(
                        "terminal text rewrote an emitted prefix",
                    ));
                    String::new()
                }
            }
        } else {
            String::new()
        };
        if failure.is_none() && !final_text.is_empty() {
            record_text_prepared(&mut sequence);
            sequence.streamed_text_len = text.len();
        }
        let output = sequence.credited_output.as_mut().expect("credited route");
        if let Some(grant) = output.grant.take() {
            if failure.is_none()
                && output.tokens_before_grant.checked_add(1)
                    == Some(sequence.generated_tokens.len())
            {
                // A terminal token can bypass ordinary token streaming. Its
                // already-owned slot still carries that commit and its text.
                output.accepted_ordinal = grant.committed(OutputDelta {
                    text: std::mem::take(&mut final_text),
                    token: sequence.generated_tokens.last().copied(),
                    generated_tokens: sequence.generated_tokens.len(),
                    created: created_seconds(),
                });
            } else if failure.is_none()
                && sequence.generated_tokens.len() == output.tokens_before_grant
            {
                grant.return_unsubmitted();
            } else {
                drop(grant);
                failure.get_or_insert_with(|| {
                    BoundedOutputError::new("terminal output has an unaccounted commit")
                });
            }
        }
        if failure.is_some() {
            reason = FinishReason::Error;
            final_text.clear();
        }
        let trace = self.capture_sequence_terminal_token_trace(&sequence);
        let observation = super::super::slo_observation::SloTerminalObservation::capture(
            &sequence,
            reason,
            super::super::slo_clock_now(),
        );
        let error_observation = super::super::slo_observation::SloTerminalObservation::capture(
            &sequence,
            FinishReason::Error,
            super::super::slo_clock_now(),
        );
        let resources = sequence.take_completion_resources();
        let mut response = InferenceResponse {
            request_id: request_id.clone(),
            text,
            usage: TokenUsage::new(sequence.input_tokens.len(), sequence.generated_tokens.len()),
            tokens: std::mem::take(&mut sequence.generated_tokens),
            finish_reason: reason,
            latency_ms: sequence.start_time.elapsed().as_millis() as u64,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
            api_response: None,
            execution_evidence: None,
        };
        let admission_cancellation_work = if pending.is_some() {
            self.model_executor
                .cancel_prefill_admission_observed(request_id)
                .work
        } else {
            self.model_executor.cancel_prefill_admission(request_id);
            ExecutorCompletionWork::Unknown
        };
        let mut cache_completion_work = ExecutorCompletionWork::Unknown;
        let mut other_physical_resources = false;
        let physical_result = if reason == FinishReason::Error {
            other_physical_resources = !resources.physical.is_empty();
            self.release_sequence_physical_resources(request_id, resources.physical)
                .await;
            Ok(())
        } else if pending.is_some() {
            let (result, work, other) = self
                .complete_sequence_physical_resources_inner(
                    request_id,
                    resources.physical,
                    &response.usage,
                    true,
                )
                .await;
            cache_completion_work = work;
            other_physical_resources = other;
            result
        } else {
            self.complete_sequence_physical_resources(
                request_id,
                resources.physical,
                &response.usage,
            )
            .await
        };
        if let Err(error) = &physical_result {
            failure = Some(BoundedOutputError::new(&error.to_string()));
            final_text.clear();
            reason = FinishReason::Error;
            response.finish_reason = reason;
        }
        let scheduler_result = self.scheduler.complete(request_id.clone(), &response).await;
        let request_slot_closed = resources.request_slot.is_some();
        if let Some(slot) = resources.request_slot {
            slot.close(self);
        }
        if let Err(error) = &scheduler_result {
            failure = Some(BoundedOutputError::new(&error.to_string()));
            final_text.clear();
            reason = FinishReason::Error;
        }
        self.record_slo_terminal(
            request_id,
            if failure.is_some() {
                error_observation
            } else {
                observation
            },
        );
        self.write_sequence_terminal_token_trace(
            request_id,
            "completed",
            Some(reason),
            trace.as_ref(),
        );
        let output_failed = failure.is_some();
        let outcome = if let Some(failure) = failure {
            // Destroy the unused history while the actor still owns its grant.
            drop(response);
            OutputCompletion::Failed(failure)
        } else {
            OutputCompletion::Succeeded {
                history: Some(OutputHistory {
                    text: response.text,
                    tokens: response.tokens,
                }),
                reason,
                usage: response.usage,
            }
        };
        let output = sequence.credited_output.as_mut().expect("credited route");
        let through_output_ordinal = output.accepted_ordinal;
        let decision = OutputTerminalDecision {
            through_output_ordinal: output.accepted_ordinal,
            outcome,
            final_text,
            created: created_seconds(),
        };
        // Failure to hand off destroys decision/history before the engine's
        // port lifetime is released. No network wait or terminal allocation.
        let handoff = output.port.terminal(decision);
        let terminal_handoff_succeeded = handoff.is_ok();
        drop(handoff);
        if let Some(pending) = pending.as_mut() {
            pending.terminal_handed_off();
        }
        let receipt = if let Some(pending) = pending {
            Some(pending.settle(
                sequence,
                HostTerminalStageV1 {
                    finish_reason: reason,
                    generated_tokens: generated_tokens.unwrap_or(0),
                    through_output_ordinal,
                    output_failed,
                    physical_failed: physical_result.is_err(),
                    scheduler_failed: scheduler_result.is_err(),
                    terminal_handoff_succeeded,
                    pending_restore_removed,
                    admission_cancellation_work,
                    cache_completion_work,
                    other_physical_resources,
                    request_slot_closed,
                    owner_matched: owner_matched && generated_tokens.is_some(),
                },
            ))
        } else {
            drop(sequence);
            None
        };
        (physical_result.and(scheduler_result), receipt)
    }
}

fn created_seconds() -> u64 {
    chrono::Utc::now().timestamp().max(0) as u64
}

fn record_text_prepared(sequence: &mut SequenceState) {
    let now = Instant::now();
    if sequence.first_emit_at.is_none() {
        sequence.first_emit_at = Some(now);
        histogram!("ferrum.engine.ttft_seconds")
            .record(now.duration_since(sequence.start_time).as_secs_f64());
    } else if let Some(last) = sequence.last_emit_at {
        histogram!("ferrum.engine.itl_seconds").record(now.duration_since(last).as_secs_f64());
    }
    sequence.last_emit_at = Some(now);
    sequence.emitted_chunks = sequence.emitted_chunks.saturating_add(1);
}
