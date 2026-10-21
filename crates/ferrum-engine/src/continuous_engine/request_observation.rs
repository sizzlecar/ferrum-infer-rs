//! Product stopping observations. These are checked request facts, independent
//! of executor frontier commits, KV identity and wire finish-reason projection.

use super::*;
use ferrum_interfaces::request_observation::*;
use std::io::Write;

static NEXT_PRODUCER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_producer() -> Option<u64> {
    NEXT_PRODUCER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .ok()
}

struct BoundedIdentity {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedIdentity {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "effective stopping identity exceeds projection limit",
            ));
        }
        self.bytes.reserve_exact(bytes.len());
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn effective_rules(sequence: &SequenceState, limit: usize) -> Option<Arc<EffectiveStoppingRules>> {
    // Sorting provides stable full identity. Check vector extents before copies;
    // the serializer itself rejects oversized schemas/strings before allocation.
    if sequence
        .stop_token_ids
        .len()
        .checked_mul(std::mem::size_of::<u32>())?
        > limit
    {
        return None;
    }
    let mut stop_ids: Vec<u32> = sequence.stop_token_ids.iter().copied().collect();
    stop_ids.sort_unstable();
    let mut user_stop_ids: Vec<u32> = sequence.user_stop_token_ids.iter().copied().collect();
    user_stop_ids.sort_unstable();
    let mut eos_ids = sequence.model_eos_token_ids.clone();
    eos_ids.sort_unstable();
    let params = &sequence.sampling_params;
    // Grammar construction reads these chat contracts, but not message text.
    // Retaining the complete source identity avoids hash collisions and keeps
    // schema, grammar activation, tool envelope and response boundary isolated.
    let chat_contract = sequence
        .original_request
        .api_request
        .as_ref()
        .and_then(|api| match api {
            ferrum_types::ApiRequest::Chat(chat) => Some((
                &chat.tools,
                &chat.tool_choice,
                &chat.tool_call_protocol,
                &chat.legacy_functions,
                &chat.legacy_function_call,
                &chat.response_format,
            )),
            ferrum_types::ApiRequest::Completion(_) => None,
        });
    let identity = (
        1u8,
        stop_ids,
        user_stop_ids,
        eos_ids,
        &sequence.stop_text_seqs,
        &params.response_format,
        &params.structured_output_start,
        &params.response_completion_boundary,
        params.model_output_protocol,
        chat_contract,
        // Grammar construction receives this cap; do not merge constrained
        // grammars compiled under different length budgets.
        sequence
            .structured_output_processor
            .as_ref()
            .map(|_| params.max_tokens),
        // Sampling changes the stopping distribution. Seed is deliberately
        // omitted; it selects a draw under the same effective output policy.
        (
            params.temperature,
            params.top_p,
            params.top_k,
            params.repetition_penalty,
            params.presence_penalty,
            params.frequency_penalty,
            params.min_p,
            params.tfs,
            params.typical_p,
            &params.mirostat,
        ),
    );
    let mut writer = BoundedIdentity {
        bytes: Vec::with_capacity(limit.min(256)),
        limit,
    };
    serde_json::to_writer(&mut writer, &identity).ok()?;
    let mechanism = if sequence.model_eos_token_ids.is_empty()
        && sequence.user_stop_token_ids.is_empty()
        && sequence.stop_text_seqs.is_empty()
        && sequence.structured_output_processor.is_none()
        && params.response_completion_boundary == ResponseCompletionBoundary::Immediate
    {
        StoppingMechanism::CapOnly
    } else {
        StoppingMechanism::Observed
    };
    EffectiveStoppingRules::new(writer.bytes, mechanism, limit).map(Arc::new)
}

impl EngineInner {
    pub(super) fn capture_planning_request_facts(
        &self,
        round: &mut TimePlanningRound,
        max_members: usize,
    ) {
        let started = Instant::now();
        self.capture_current_planning_request_facts(round, max_members);
        round.charge_projection_time(started.elapsed());
    }

    fn capture_current_planning_request_facts(
        &self,
        round: &mut TimePlanningRound,
        max_members: usize,
    ) {
        use ferrum_scheduler::cost_prediction::{PlanningCapacity, RequestClockFacts};
        round.clear_request_facts();
        round.set_capacity(self.model_executor.paged_admission_capacity_snapshot().map(
            |snapshot| PlanningCapacity {
                block_size: snapshot.slots.block_size,
                total_blocks: snapshot.slots.total_blocks,
                free_blocks: snapshot.slots.free_blocks,
                admission_coverage: snapshot.coverage,
            },
        ));
        let Some(handles) = self.sequences.snapshot_bounded(max_members) else {
            return;
        };
        let mut facts = Vec::with_capacity(handles.len());
        let mut clocks = Vec::with_capacity(handles.len());
        let mut residents = Vec::with_capacity(handles.len());
        for handle in handles {
            let state = handle.lock();
            let Some(sequence) = state
                .as_ref()
                .filter(|_| self.sequences.is_current(&handle))
            else {
                return;
            };
            let Some(fact) = &sequence.stopping_observation else {
                return;
            };
            if sequence.execution_close.is_none()
                && sequence.pending_token_admission.is_none()
                && fact.cap == sequence.sampling_params.max_tokens
            {
                if let Some(frontier) = sequence.execution_frontier.as_ref() {
                    residents.push(
                        ferrum_execution::model_executor::ResidentSequenceCapacityRequest {
                            sequence: frontier.sequence().clone(),
                            // The final sampled token is not consumed by a
                            // subsequent invocation. Use the same execution
                            // ceiling as actual admission, including replay.
                            through_tokens: sequence.model_maximum_sequence_tokens(),
                        },
                    );
                }
            }
            let clock = if let Some(sender) = &sequence.stream_sender {
                let Some(clock) = sender.text_clock_snapshot(sequence.generated_tokens.len())
                else {
                    return;
                };
                clock
            } else {
                let captured_at = Instant::now();
                let Some(age) = captured_at.checked_duration_since(sequence.start_time) else {
                    return;
                };
                RequestClockFacts {
                    captured_at,
                    age,
                    first_text: None,
                    last_text: None,
                    max_text_interval: None,
                    nonempty_text_events: 0,
                    violating_text_intervals: 0,
                    itl_threshold: None,
                    generated_tokens: sequence.generated_tokens.len(),
                    streaming: false,
                }
            };
            facts.push((handle.request_id().clone(), fact.clone()));
            clocks.push((handle.request_id().clone(), clock));
        }
        if round.set_request_facts(facts) {
            if round.set_request_clocks(clocks)
                && round
                    .request_facts()
                    .is_some_and(|facts| residents.len() == facts.len())
            {
                // Query after releasing every sequence lock. The executor
                // proves actual ownership/growth for these exact incarnations;
                // a configured maximum or generic free byte count is insufficient.
                round.set_resident_capacity(
                    self.model_executor
                        .resident_cohort_capacity_snapshot(&residents),
                );
            }
        }
    }

    /// Called under the serialized initial product admission boundary, before
    /// scheduler/resource admission. Requeue/recompute never calls this method.
    pub(super) fn attach_request_stopping(&self, sequence: &mut SequenceState) {
        let Some(observer) = self.request_stopping_observer.as_ref() else {
            return;
        };
        let Some(producer) = self.request_stopping_producer else {
            return;
        };
        let Ok(admission) = self.request_stopping_admission.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |value| value.checked_add(1),
        ) else {
            return;
        };
        let lifecycle = RequestLifecycleId {
            producer,
            admission,
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observer.record_admission(lifecycle, sequence.start_time)
        }));
        let Ok(limits) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observer.limits()))
        else {
            return;
        };
        let Some(rules) = effective_rules(sequence, limits.max_rule_bytes) else {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                observer.record_drop(RequestObservationDrop::ProjectionLimit)
            }));
            return;
        };
        let observation = RequestStoppingObservation {
            lifecycle,
            admitted_at: sequence.start_time,
            revision: 1,
            observed_at: Instant::now(),
            rules,
            cap: sequence.sampling_params.max_tokens,
            generated_tokens: 0,
            verified_survival_through: 0,
            terminal: None,
            completion: None,
        };
        self.record_request_stopping(&observation);
        sequence.stopping_observation = Some(observation);
    }

    fn record_request_stopping(&self, observation: &RequestStoppingObservation) {
        if let Some(observer) = &self.request_stopping_observer {
            // Optional feedback has no authority to fail or replay inference.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                observer.try_record(observation)
            }));
        }
    }

    pub(super) fn observe_verified_survival(&self, sequence: &mut SequenceState) {
        let generated = sequence.generated_tokens.len();
        let Some(observation) = sequence.stopping_observation.as_mut() else {
            return;
        };
        if generated <= observation.verified_survival_through {
            return;
        }
        let Some(revision) = observation.revision.checked_add(1) else {
            return;
        };
        observation.revision = revision;
        observation.observed_at = Instant::now();
        observation.generated_tokens = generated;
        observation.verified_survival_through = generated;
        self.record_request_stopping(observation);
    }

    pub(super) fn retain_verified_stop(&self, sequence: &mut SequenceState, reason: FinishReason) {
        let generated = sequence.generated_tokens.len();
        let Some(observation) = sequence.stopping_observation.as_mut() else {
            return;
        };
        observation.generated_tokens = generated;
        observation.terminal = Some(terminal_reason(reason));
        if matches!(reason, FinishReason::EOS | FinishReason::Stop) {
            observation.verified_survival_through = generated.saturating_sub(1);
        } else if reason == FinishReason::Length {
            // sequence::stop_reason checks every natural terminator before cap.
            observation.verified_survival_through = generated;
        }
    }

    pub(super) fn observe_request_terminal(
        &self,
        sequence: &mut SequenceState,
        generated: usize,
        reason: FinishReason,
    ) -> Option<RequestStoppingObservation> {
        let Some(mut observation) = sequence.stopping_observation.take() else {
            return None;
        };
        let Some(revision) = observation.revision.checked_add(1) else {
            return None;
        };
        observation.revision = revision;
        observation.observed_at = Instant::now();
        observation.generated_tokens = generated;
        // A requested/wire finish reason alone is not stopping evidence.
        let reason = if matches!(
            reason,
            FinishReason::EOS | FinishReason::Stop | FinishReason::Length
        ) && observation.terminal != Some(terminal_reason(reason))
        {
            FinishReason::Error
        } else {
            reason
        };
        observation.terminal = Some(terminal_reason(reason));
        if matches!(
            reason,
            FinishReason::Error | FinishReason::Cancelled | FinishReason::ContentFilter
        ) {
            observation.completion = Some(RequestCompletionOutcome::Failed);
        }
        // A full terminal fact repairs missed middle notices. EOS/Stop have
        // passed the request stopping check; failures retain only verified data.
        if matches!(reason, FinishReason::EOS | FinishReason::Stop) {
            observation.verified_survival_through = generated.saturating_sub(1);
        } else if reason == FinishReason::Length {
            observation.verified_survival_through = generated;
        }
        self.record_request_stopping(&observation);
        Some(observation)
    }

    /// A successful stopping endpoint does not imply successful product close
    /// and delivery. This terminal enrichment owns no execution resources.
    pub(super) fn observe_product_completion(
        &self,
        mut observation: RequestStoppingObservation,
        success: bool,
    ) {
        if observation.completion.is_some() {
            return;
        }
        let Some(revision) = observation.revision.checked_add(1) else {
            return;
        };
        observation.revision = revision;
        observation.observed_at = Instant::now();
        observation.completion = Some(if success {
            RequestCompletionOutcome::Succeeded
        } else {
            RequestCompletionOutcome::Failed
        });
        self.record_request_stopping(&observation);
    }
}

fn terminal_reason(reason: FinishReason) -> RequestTerminalReason {
    match reason {
        FinishReason::EOS => RequestTerminalReason::Eos,
        FinishReason::Stop => RequestTerminalReason::Stop,
        FinishReason::Length => RequestTerminalReason::Length,
        FinishReason::Cancelled => RequestTerminalReason::Cancelled,
        _ => RequestTerminalReason::Error,
    }
}

#[cfg(test)]
mod tests;
