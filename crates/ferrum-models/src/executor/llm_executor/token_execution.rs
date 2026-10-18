//! Direct token lifecycle for the legacy model-owned KV implementation.
//!
//! This registry retains immutable input and product frontiers, not a second
//! allocator. Physical preparation, reservation, execution and release all
//! remain inside the same model. The synchronous model guard spans a batch;
//! there is no Tensor input/output bridge and no await between participants.

use std::collections::HashMap;

use ferrum_execution::model_executor::ExecutorSamplingOutput;
use ferrum_execution::token_execution::*;

use super::*;
use crate::common::llm::{catch_host_execution, ModelTokenOutcome, ModelTokenWork};

#[derive(Default)]
pub(super) struct TokenSequences {
    live: HashMap<ExecutorSequence, TokenSequence>,
}

struct TokenSequence {
    request_id: RequestId,
    cache_id: String,
    prompt: Vec<u32>,
    maximum_tokens: usize,
    product_prompt_tokens: usize,
    replayed_output_tokens: usize,
    frontier: ExecutionFrontier,
    recovery_key: ExecutionRecoveryKey,
    adapter: Option<ActiveLoraAdapter>,
    prepared: bool,
    closing: bool,
}

impl TokenSequence {
    fn validate_work(&self, input: &ExecutionWork<'_>) -> Result<()> {
        if self.closing || input.expected_frontier() != &self.frontier {
            return Err(FerrumError::request_validation(
                "token execution requires the exact live frontier",
            ));
        }
        let range = input.work().range()?;
        if range.end() > self.maximum_tokens || range.end() > u32::MAX as usize {
            return Err(FerrumError::request_validation(
                "token work exceeds the admitted sequence capacity",
            ));
        }
        match input.work() {
            TokenWork::Prefill(chunk) if chunk.total_prompt_tokens() != self.prompt.len() => Err(
                FerrumError::request_validation("prefill work differs from the retained prompt"),
            ),
            TokenWork::Decode { position, .. } if position < self.prompt.len() => Err(
                FerrumError::request_validation("decode cannot precede final prefill"),
            ),
            _ => Ok(()),
        }
    }
}

impl LlmExecutor {
    pub(super) fn admit_token_sequence(
        &self,
        input: ExecutionAdmission<'_>,
    ) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        let mut sequences = self.token_sequences.lock();
        if sequences
            .live
            .values()
            .any(|entry| &entry.request_id == input.context.request_id)
        {
            return Err(FerrumError::request_validation(
                "request already has a live token incarnation",
            ));
        }
        let mut model = self.lock_model();
        if !model.supports_direct_token_execution() {
            return Err(FerrumError::unsupported(
                "model/backend has not implemented the direct token lifecycle",
            ));
        }
        let context = input.context;
        if context.maximum_sequence_tokens > model.kv_capacity()
            || context.maximum_sequence_tokens > u32::MAX as usize
            || context
                .input_tokens
                .iter()
                .any(|token| token.get() as usize >= model.config().vocab_size)
        {
            return Err(FerrumError::request_validation(
                "token admission exceeds model capacity or vocabulary",
            ));
        }
        let adapter = match input.adapter {
            Some(selection) => {
                if self.adapter_support != AdapterExecutionSupport::PerSequenceLora {
                    return Err(FerrumError::unsupported(
                        "configured model does not support per-sequence LoRA",
                    ));
                }
                let adapter = ActiveLoraAdapter {
                    name: selection.name.to_owned(),
                    path: selection.path.into(),
                };
                // Loading may cache reusable adapter weights, but must neither
                // bind an incarnation nor allocate a sequence cache.
                catch_host_execution(|| model.preload_lora_adapter(&adapter))??;
                Some(adapter)
            }
            None => None,
        };
        let sequence = self.token_issuer.issue_sequence()?;
        let frontier = self.token_issuer.frontier(&sequence, 0)?;
        let recovery_key = self.token_issuer.recovery_key(&sequence)?;
        sequences.live.insert(
            sequence,
            TokenSequence {
                request_id: context.request_id.clone(),
                cache_id: self.gen_cache_id(),
                prompt: context
                    .input_tokens
                    .iter()
                    .map(|token| token.get())
                    .collect(),
                maximum_tokens: context.maximum_sequence_tokens,
                product_prompt_tokens: context.product_prompt_tokens,
                replayed_output_tokens: context.replayed_output_tokens,
                frontier: frontier.clone(),
                recovery_key,
                adapter,
                prepared: false,
                closing: false,
            },
        );
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }

    pub(super) fn execute_token_batch(
        &self,
        inputs: &[ExecutionWork<'_>],
    ) -> Result<ExecutionBatchOutcome> {
        let batch = self.token_issuer.batch(inputs, self.info.vocab_size)?;
        let mut sequences = self.token_sequences.lock();
        // Reject foreign, stale or malformed members across the whole input
        // before any reservation or numerical operation mutates the model.
        for input in inputs {
            sequences
                .live
                .get(input.sequence())
                .ok_or_else(|| FerrumError::request_validation("token incarnation is not live"))?
                .validate_work(input)?;
        }
        let mut model = self.lock_model();
        let mut completed = Vec::with_capacity(inputs.len());
        let mut remaining = Vec::new();
        let mut causes = Vec::new();
        let mut rejected_completions = Vec::new();
        let mut contract_error = None;
        for (index, input) in inputs.iter().enumerate() {
            // No await or release of either guard occurs after the preflight.
            let Some(entry) = sequences.live.get_mut(input.sequence()) else {
                return Ok(contract_violation(
                    FerrumError::internal("validated token incarnation disappeared"),
                    completed,
                    remaining,
                    causes,
                    rejected_completions,
                ));
            };
            let range = match input.work().range() {
                Ok(range) => range,
                Err(error) => {
                    return Ok(contract_violation(
                        error,
                        completed,
                        remaining,
                        causes,
                        rejected_completions,
                    ))
                }
            };
            let preparation = catch_host_execution(|| -> Result<()> {
                if !entry.prepared {
                    model.prepare_kv_capacity(&entry.cache_id, entry.maximum_tokens);
                }
                model.reserve_kv_slots(&[KvSlotRequest {
                    cache_id: entry.cache_id.clone(),
                    target_len: range.end(),
                    admission_target_len: Some(entry.prompt.len().max(range.end())),
                }])?;
                if !entry.prepared {
                    model.set_lora_adapter_for_cache(&entry.cache_id, entry.adapter.clone())?;
                    entry.prepared = true;
                }
                Ok(())
            })
            .and_then(|result| result);
            let actual = match input.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            };
            let outcome = match preparation {
                Err(error) => ModelTokenOutcome::NotSubmitted(error),
                Ok(()) => {
                    let work = match input.work() {
                        TokenWork::Prefill(chunk) => ModelTokenWork::Prefill {
                            tokens: &entry.prompt[chunk.range()],
                            position: chunk.tokens_processed(),
                        },
                        TokenWork::Decode { token, position } => ModelTokenWork::Decode {
                            token: token.get(),
                            position: position as u32,
                        },
                    };
                    model.execute_token_work(&entry.cache_id, work)
                }
            };
            let (progress, error) = match outcome {
                ModelTokenOutcome::NotSubmitted(error) => (
                    // Zero execution of this call does not prove the original
                    // frontier is usable (a model KV precheck may disagree).
                    // Without an explicit preservation certificate, close it.
                    ExecutionProgress::NotSubmitted {
                        stage: NoSubmissionStage::BeforeSubmit,
                    },
                    error,
                ),
                ModelTokenOutcome::Terminal(Err(error)) => (
                    ExecutionProgress::Terminal {
                        disposition: TerminalModelDisposition::Failed,
                        executed_range: None,
                    },
                    error,
                ),
                ModelTokenOutcome::Terminal(Ok(logits)) => {
                    let product = match input.work() {
                        TokenWork::Prefill(chunk) if !chunk.is_final() => {
                            ExecutionProduct::Intermediate
                        }
                        _ => ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)),
                    };
                    match batch.completion(index, actual, product) {
                        Ok(completion) => {
                            entry.frontier = completion.next_frontier().clone();
                            completed.push(completion);
                            continue;
                        }
                        Err(rejected) => {
                            // The model really completed even though its output
                            // violated the product contract. Never replay it.
                            let (error, product) = rejected.into_parts();
                            contract_error.get_or_insert_with(|| error.clone());
                            rejected_completions.push(product);
                            (
                                ExecutionProgress::Terminal {
                                    disposition: TerminalModelDisposition::Successful,
                                    executed_range: Some(range),
                                },
                                error,
                            )
                        }
                    }
                }
            };
            let cause_index = causes.len();
            causes.push(error);
            let continuation =
                match self.close_token_locked(&mut sequences, &mut **model, input.sequence()) {
                    Ok(close) => {
                        if let Some(error) = close.error {
                            causes.push(error);
                        }
                        match close.disposition {
                            ExecutionCloseDisposition::Ended(ended) => {
                                ContinuationDisposition::Ended(ended)
                            }
                            ExecutionCloseDisposition::Held(held) => {
                                ContinuationDisposition::Held(held)
                            }
                        }
                    }
                    Err(error) => {
                        return Ok(contract_violation(
                            error,
                            completed,
                            remaining,
                            causes,
                            rejected_completions,
                        ))
                    }
                };
            match batch.participant(index, progress, continuation, cause_index) {
                Ok(participant) => remaining.push(participant),
                Err(rejected) => {
                    let mut outcome = rejected.into_outcome((completed, remaining, causes));
                    if let ExecutionBatchOutcome::ContractViolation(violation) = &mut outcome {
                        violation.rejected_completions = rejected_completions;
                    }
                    return Ok(outcome);
                }
            }
        }
        Ok(if let Some(error) = contract_error {
            contract_violation(error, completed, remaining, causes, rejected_completions)
        } else if remaining.is_empty() {
            batch
                .completed(completed)
                .unwrap_or_else(|rejected| rejected.into_outcome())
        } else {
            batch
                .interrupted(completed, remaining, causes)
                .unwrap_or_else(|rejected| rejected.into_outcome())
        })
    }

    pub(super) fn close_token_incarnation(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        // Checks issuer identity without treating an unknown cache name as an
        // already-ended sequence.
        self.token_issuer.frontier(sequence, 0)?;
        let mut sequences = self.token_sequences.lock();
        let entry = sequences
            .live
            .get(sequence)
            .ok_or_else(|| FerrumError::request_validation("token incarnation is not live"))?;
        if let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        {
            if input_tokens != entry.product_prompt_tokens
                || output_tokens < entry.replayed_output_tokens
                || input_tokens
                    .checked_add(output_tokens)
                    .is_none_or(|total| total > entry.maximum_tokens)
            {
                return Err(FerrumError::request_validation(
                    "token completion usage differs from immutable admission",
                ));
            }
        }
        self.close_token_locked(&mut sequences, &mut **self.lock_model(), sequence)
    }

    fn close_token_locked(
        &self,
        sequences: &mut TokenSequences,
        model: &mut dyn DecoderOnlyLLM,
        sequence: &ExecutorSequence,
    ) -> Result<ExecutionCloseOutcome> {
        let entry = sequences
            .live
            .get_mut(sequence)
            .ok_or_else(|| FerrumError::request_validation("token incarnation is not live"))?;
        entry.closing = true;
        match model.close_token_sequence(&entry.cache_id) {
            Ok(()) => {
                let ended = self.token_issuer.ended(sequence)?;
                sequences.live.remove(sequence);
                Ok(ExecutionCloseOutcome::ended(ended))
            }
            Err(error) => {
                // Keep the actual model cache and closing entry, independent
                // of the observer. An observation error explicitly reports
                // unreclaimed ownership; it is never a successful retirement.
                let observation_error = error.clone();
                let held = self
                    .token_issuer
                    .held(
                        entry.recovery_key.clone(),
                        async move { Err(observation_error) },
                    )?;
                Ok(ExecutionCloseOutcome::held(held).with_error(error))
            }
        }
    }
}

fn contract_violation(
    error: FerrumError,
    completed: Vec<ExecutionCompletion>,
    remaining: Vec<InterruptedParticipant>,
    causes: Vec<FerrumError>,
    rejected_completions: Vec<CompletionParts>,
) -> ExecutionBatchOutcome {
    ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation {
        error,
        completed,
        remaining,
        causes,
        rejected_completions,
        rejected_participants: Vec::new(),
        unassigned_products: Vec::new(),
    })
}

#[cfg(test)]
mod tests;
