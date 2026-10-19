//! Commit actual executor progress before any fallible product publication.
//! Sampling failures are request-local; no branch may replay the completed
//! model work or prevent a later healthy participant from being committed.

use super::*;
use ferrum_execution::token_execution::{ExecutedTokenWork, ExecutionCompletion, ExecutionProduct};

struct SchedulerProgress {
    input_index: usize,
    executed: ExecutedTokenWork,
    frontier: ExecutionFrontier,
    product: Option<ExecutionProduct>,
    generated_tokens: Option<usize>,
}

impl EngineInner {
    pub(super) fn commit_token_completions(
        &self,
        inputs: &[&TokenBatchInput],
        completions: Vec<ExecutionCompletion>,
        timing: Option<TokenExecutionTiming>,
        attempt: &mut Option<TokenAttemptCapture>,
    ) -> TokenBatchEffects {
        let mut effects = TokenBatchEffects {
            ready: Vec::new(),
            failed: Vec::new(),
        };
        let mut progress = Vec::with_capacity(completions.len());
        {
            // The caller has already validated mapping, role, output policy
            // and duplicate/conflicting batch membership. Recheck the live
            // owner under the same lock used to publish every actual frontier.
            // Never overwrite a replacement incarnation with an old result.
            for completion in completions {
                let (input_index, next_frontier, executed, product) = completion.into_parts();
                let input = inputs[input_index];
                if let Some(attempt) = attempt.as_mut() {
                    attempt.completed(input_index, executed, false);
                }
                let mut state = input.handle.lock();
                let Some(sequence) = state.as_mut() else {
                    effects.failed.push((
                        Arc::clone(&input.handle),
                        FerrumError::internal("completed token work lost its outer request"),
                    ));
                    continue;
                };
                if !self.sequences.is_current(&input.handle)
                    || sequence.execution_frontier.as_ref() != Some(&input.frontier)
                    || sequence.execution_close.is_some()
                {
                    effects.failed.push((
                        Arc::clone(&input.handle),
                        FerrumError::internal(
                            "completed token work no longer matches the live outer frontier",
                        ),
                    ));
                    continue;
                }
                sequence.execution_frontier = Some(next_frontier.clone());
                if let Some(attempt) = attempt.as_mut() {
                    attempt.completed(input_index, executed, true);
                }
                match executed {
                    ExecutedTokenWork::Prefill { chunk, .. } => {
                        sequence.prefill_tokens_processed = chunk.end();
                        sequence.prefill_complete = chunk.is_final();
                        sequence.phase = if chunk.is_final() {
                            RequestPhase::Decoding
                        } else {
                            RequestPhase::Prefilling
                        };
                    }
                    ExecutedTokenWork::Decode { .. } => {
                        sequence.tokens_this_iteration += 1;
                    }
                }
                progress.push(SchedulerProgress {
                    input_index,
                    frontier: next_frontier,
                    executed,
                    product: Some(product),
                    generated_tokens: None,
                });
            }

            // All accepted physical frontiers are now saved. Product errors
            // below cannot restore a stale frontier or hide a peer's result.
            for pending in &mut progress {
                let input = inputs[pending.input_index];
                let executed = pending.executed;
                let product = pending
                    .product
                    .take()
                    .expect("each retained product is consumed exactly once");
                let mut state = input.handle.lock();
                let Some(sequence) = state.as_mut().filter(|sequence| {
                    self.sequences.is_current(&input.handle)
                        && sequence.execution_frontier.as_ref() == Some(&pending.frontier)
                        && sequence.execution_close.is_none()
                }) else {
                    effects.failed.push((
                        Arc::clone(&input.handle),
                        FerrumError::internal(
                            "completed token product lost its original live frontier",
                        ),
                    ));
                    continue;
                };
                if matches!(executed, ExecutedTokenWork::Decode { .. }) {
                    if let Some(timing) = timing {
                        sequence.close_decode_scheduling(timing.started_at);
                        sequence.record_decode_execution(timing.started_at, timing.completed_at);
                    }
                }
                let sampled = (|| -> Result<Option<TokenId>> {
                    if let Some(error) = sequence.output_accounting_error() {
                        return Err(error.clone());
                    }
                    let ExecutionProduct::Sampling(output) = product else {
                        return Ok(None);
                    };
                    if matches!(executed, ExecutedTokenWork::Prefill { .. }) {
                        sequence.reset_guided_processors()?;
                    }
                    let token = match output {
                        ExecutorSamplingOutput::GreedyToken(token) => {
                            sequence.validate_and_commit_model_greedy_argmax_token(
                                Some(self.tokenizer.as_ref()),
                                token,
                            )?;
                            token
                        }
                        ExecutorSamplingOutput::FullLogits(mut logits) => sequence
                            .sample_and_commit_with_processors_and_tokenizer(
                                &mut logits,
                                Some(self.tokenizer.as_ref()),
                            )?,
                        ExecutorSamplingOutput::NoSelectableFiniteLogit => {
                            return Err(FerrumError::model(
                                "engine sampling policies have no finite token",
                            ));
                        }
                    };
                    match executed {
                        ExecutedTokenWork::Prefill { .. } => {
                            sequence.record_decode_ready();
                            sequence.record_generated_token_commit();
                        }
                        ExecutedTokenWork::Decode { .. } => {
                            sequence.record_generated_token_commit_with_decode_postprocess(
                                timing.map(|timing| timing.completed_at),
                            );
                        }
                    }
                    if let Some(error) = sequence.output_accounting_error() {
                        return Err(error.clone());
                    }
                    Ok(Some(token))
                })();
                pending.generated_tokens = match sampled {
                    Ok(Some(token)) => {
                        effects.ready.push((Arc::clone(&input.handle), token));
                        Some(sequence.generated_tokens.len())
                    }
                    Ok(None) => None,
                    Err(error) => {
                        effects.failed.push((Arc::clone(&input.handle), error));
                        None
                    }
                };
            }
        }

        // No scheduler callback or output/cleanup await runs until all model
        // progress and all successful sampled tokens have been retained.
        for progress in progress {
            let input = &inputs[progress.input_index];
            // Keep the original owner live through synchronous scheduler
            // publication, so a reused RequestId cannot receive old progress.
            let state = input.handle.lock();
            if !state.as_ref().is_some_and(|sequence| {
                self.sequences.is_current(&input.handle)
                    && sequence.execution_frontier.as_ref() == Some(&progress.frontier)
                    && sequence.execution_close.is_none()
            }) {
                continue;
            }
            match progress.executed {
                ExecutedTokenWork::Prefill { chunk, .. } => {
                    // A deliberate shape projection is planned work, not a
                    // capacity-limited completion. Compare only with the work
                    // actually submitted to this executor.
                    let TokenWork::Prefill(requested) = input.work else {
                        unreachable!("validated prefill completion has prefill input")
                    };
                    let scheduler_result = self
                        .scheduler
                        .mark_prefill_chunk_processed_with_capacity_feedback(
                            &input.request_id,
                            input.context_tokens,
                            requested.tokens_to_process(),
                            chunk.tokens_to_process(),
                        );
                    let error = match scheduler_result {
                        Ok(promoted) if promoted == chunk.is_final() => None,
                        Ok(_) => Some(FerrumError::scheduler(
                            "scheduler prefill finality differs from completed token work",
                        )),
                        Err(error) => Some(error),
                    };
                    let scheduler_committed = error.is_none();
                    if let Some(error) = error {
                        if !effects
                            .failed
                            .iter()
                            .any(|(id, _)| Arc::ptr_eq(id, &input.handle))
                        {
                            effects.failed.push((Arc::clone(&input.handle), error));
                        }
                        effects
                            .ready
                            .retain(|(id, _)| !Arc::ptr_eq(id, &input.handle));
                    }
                    // These tokens really executed even if sampling or logical
                    // publication subsequently failed. They are not retries.
                    self.total_prefill_tokens
                        .fetch_add(chunk.tokens_to_process() as u64, Ordering::Relaxed);
                    counter!("ferrum.engine.prefill_tokens_total")
                        .increment(chunk.tokens_to_process() as u64);
                    if chunk.is_final() && progress.generated_tokens.is_some() {
                        counter!("ferrum.engine.prefills_total").increment(1);
                        if scheduler_committed {
                            self.scheduler.mark_first_token_emitted(&input.request_id);
                        }
                    }
                    if scheduler_committed {
                        self.trace_plan_runtime_prefill_completion(&input.request_id, chunk);
                    }
                }
                ExecutedTokenWork::Decode { .. } => {
                    if let Some(generated_tokens) = progress.generated_tokens {
                        self.scheduler
                            .update_decode_progress(&input.request_id, generated_tokens);
                        self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
                        counter!("ferrum.engine.decode_tokens_total").increment(1);
                    }
                }
            }
        }
        effects
    }
}
