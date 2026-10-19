//! Stub model executor for MVP testing and development

use async_trait::async_trait;
use ferrum_execution::{model_executor::ExecutorSamplingOutput, token_execution::*};
use ferrum_execution::{
    model_executor::{
        AttentionType, ExecutorCapabilities, ExecutorMemoryUsage, ExecutorState, ExecutorStatus,
        MemoryRequirements,
    },
    ModelExecutor,
};
use ferrum_types::{DataType, Device, ModelInfo, ModelType, Result};
use ferrum_types::{FerrumError, RequestId};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::debug;

struct StubSequence {
    request: RequestId,
    prompt: Arc<[ferrum_types::TokenId]>,
    maximum: usize,
    product_prompt: usize,
    frontier: ExecutionFrontier,
}

/// Stub model executor - MVP implementation
///
/// Returns dummy outputs to allow pipeline testing without real models.
pub struct StubModelExecutor {
    info: ModelInfo,
    issuer: ExecutorSequenceIssuer,
    sequences: Mutex<HashMap<ExecutorSequence, StubSequence>>,
    closing: AtomicBool,
}

impl StubModelExecutor {
    pub fn new(model_id: impl Into<ferrum_types::ModelId>, vocab_size: usize) -> Self {
        let info = ModelInfo {
            model_id: model_id.into(),
            model_type: ModelType::Custom("stub".into()),
            num_parameters: 1_000_000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size,
            max_sequence_length: 2048,
            dtype: DataType::FP16,
            device: Device::CPU,
            version: Some("mvp-stub".into()),
            license: Some("Apache-2.0".into()),
            metadata: std::collections::HashMap::new(),
        };

        debug!("Created StubModelExecutor: vocab={}", vocab_size);

        Self {
            info,
            issuer: ExecutorSequenceIssuer::new(),
            sequences: Mutex::new(HashMap::new()),
            closing: AtomicBool::new(false),
        }
    }

    fn close_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.issuer.frontier(sequence, 0)?;
        let mut sequences = self.sequences.lock();
        let Some(entry) = sequences.get(sequence) else {
            return sequence
                .observed_end()
                .map(ExecutionCloseOutcome::ended)
                .ok_or_else(|| FerrumError::request_validation("unknown stub incarnation"));
        };
        if let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        {
            if input_tokens != entry.product_prompt
                || input_tokens
                    .checked_add(output_tokens.saturating_sub(1))
                    .is_none_or(|end| end > entry.maximum)
            {
                return Err(FerrumError::request_validation(
                    "stub completion usage mismatch",
                ));
            }
        }
        sequences.remove(sequence);
        Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?))
    }
}

#[async_trait]
impl ModelExecutor for StubModelExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        use ferrum_execution::executor_lifecycle::ExecutorShutdownObserver;
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        let result = (|| {
            let mut sequences = self.sequences.lock();
            self.closing.store(true, Ordering::Release);
            let original: Vec<_> = sequences.keys().cloned().collect();
            for sequence in original {
                self.issuer.ended(&sequence)?;
                sequences.remove(&sequence);
            }
            Ok::<(), FerrumError>(())
        })();
        if let Err(error) = result {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        // Every stored incarnation was synchronous host state. Publish only
        // after its actual table and original executor have been released.
        let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
        drop(self);
        reporter.retired(None);
        observer
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        if input.adapter.is_some() {
            return Err(FerrumError::unsupported("stub does not implement adapters"));
        }
        if self.info.vocab_size == 0
            || input.context.maximum_sequence_tokens > self.info.max_sequence_length
            || input
                .context
                .input_tokens
                .iter()
                .any(|token| token.get() as usize >= self.info.vocab_size)
        {
            return Err(FerrumError::request_validation(
                "stub admission exceeds model limits",
            ));
        }
        let mut sequences = self.sequences.lock();
        if self.closing.load(Ordering::Acquire) {
            return Err(FerrumError::request_validation("stub executor is closing"));
        }
        if sequences
            .values()
            .any(|entry| &entry.request == input.context.request_id)
        {
            return Err(FerrumError::request_validation(
                "duplicate live stub request",
            ));
        }
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        sequences.insert(
            sequence,
            StubSequence {
                request: input.context.request_id.clone(),
                prompt: input.context.input_tokens.into(),
                maximum: input.context.maximum_sequence_tokens,
                product_prompt: input.context.product_prompt_tokens,
                frontier: frontier.clone(),
            },
        );
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let batch = self.issuer.batch(work, self.info.vocab_size)?;
        let mut sequences = self.sequences.lock();
        if self.closing.load(Ordering::Acquire) {
            return Err(FerrumError::request_validation("stub executor is closing"));
        }
        for input in work {
            let entry = sequences
                .get(input.sequence())
                .ok_or_else(|| FerrumError::request_validation("unknown stub incarnation"))?;
            if &entry.frontier != input.expected_frontier()
                || input.work().range()?.end() > entry.maximum
            {
                return Err(FerrumError::request_validation("stale stub frontier"));
            }
            match input.work() {
                TokenWork::Prefill(chunk) if chunk.total_prompt_tokens() == entry.prompt.len() => {}
                TokenWork::Decode { position, .. } if position >= entry.prompt.len() => {}
                _ => {
                    return Err(FerrumError::request_validation(
                        "stub work does not match admitted prompt",
                    ))
                }
            }
        }
        let prefill = matches!(work[0].work(), TokenWork::Prefill(_));
        if work
            .iter()
            .any(|input| matches!(input.work(), TokenWork::Prefill(_)) != prefill)
        {
            return Ok(batch.not_submitted(NoSubmissionReason::UnsupportedShape));
        }
        let mut completed = Vec::with_capacity(work.len());
        for (index, input) in work.iter().enumerate() {
            let executed = match input.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            };
            let product = match input.work() {
                TokenWork::Prefill(chunk) if !chunk.is_final() => ExecutionProduct::Intermediate,
                _ => ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(
                    vec![0.0; self.info.vocab_size],
                )),
            };
            match batch.completion(index, executed, product) {
                Ok(completion) => completed.push(completion),
                Err(error) => {
                    return Ok(
                        batch.not_submitted(NoSubmissionReason::Rejected(error.into_parts().0))
                    )
                }
            }
        }
        for completion in &completed {
            sequences
                .get_mut(completion.next_frontier().sequence())
                .expect("validated stub incarnation")
                .frontier = completion.next_frontier().clone();
        }
        Ok(batch
            .completed(completed)
            .unwrap_or_else(|error| error.into_outcome()))
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.close_tokens(sequence, reason)
    }

    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.close_tokens(sequence, ExecutionEndReason::Cancelled)
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 32,
            max_sequence_length: 2048,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32, DataType::FP16],
            supported_devices: vec![Device::CPU],
            memory_requirements: MemoryRequirements {
                typed_sequence_state: None,
                parameter_memory: 4 * 1024 * 1024, // 4MB
                activation_memory_per_token: 1024,
                kv_cache_memory_per_token: 512,
                overhead_memory: 1024 * 1024,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            state: ExecutorState::Ready,
            is_ready: true,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: 0,
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: 1024 * 1024,
                used_bytes: 512 * 1024,
                peak_bytes: 1024 * 1024,
                utilization_percent: 50.0,
            },
            last_operation: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_execution::model_executor::{
        ExecutorPrefillAdmission, LogitsReturnPolicy, PrefillChunk,
    };
    use ferrum_types::TokenId;

    fn assert_ended(outcome: ExecutionCloseOutcome, sequence: &ExecutorSequence) {
        assert!(outcome.error.is_none());
        let ExecutionCloseDisposition::Ended(ended) = outcome.disposition else {
            panic!("stub synchronous owner must end");
        };
        assert_eq!(ended.sequence(), sequence);
    }

    #[tokio::test]
    async fn stub_owned_shutdown_publishes_original_terminals_before_retirement() {
        use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorShutdownPhase};
        let owner = ExecutorOwner::new(Box::new(StubModelExecutor::new("close-stub", 16)));
        let scope = owner.scope().unwrap();
        let id = RequestId::new();
        let prompt = [TokenId::new(1), TokenId::new(2)];
        let TokenAdmissionOutcome::Admitted(frontier) = scope
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_diagnostic(&id, &prompt, 4),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("stub admission must succeed")
        };
        assert!(frontier.sequence().observed_end().is_none());
        drop(scope);
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            owner.begin_shutdown().wait(),
        )
        .await
        .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        assert_eq!(
            frontier.sequence().observed_end().unwrap().sequence(),
            frontier.sequence()
        );
        assert!(owner.scope().is_err());
    }

    #[tokio::test]
    async fn stub_tokens_chunk_then_decode_and_replacement_use_exact_incarnations() {
        let executor = StubModelExecutor::new("stub", 16);
        let request = RequestId::new();
        let prompt = [TokenId::new(1), TokenId::new(2)];
        let admission = || ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_diagnostic(&request, &prompt, 4),
            adapter: None,
        };
        let TokenAdmissionOutcome::Admitted(first) =
            executor.admit_tokens(admission()).await.unwrap()
        else {
            panic!("admission");
        };
        let mut frontier = first.clone();
        let policy = LogitsReturnPolicy::FullLogits;
        for work in [
            TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap()),
            TokenWork::Prefill(PrefillChunk::new(1, 1, 2).unwrap()),
            TokenWork::Decode {
                token: TokenId::new(0),
                position: 2,
            },
        ] {
            let inputs = [ExecutionWork::new(&frontier, work, &policy).unwrap()];
            let ExecutionBatchOutcome::Completed(output) =
                executor.execute_tokens(&inputs).await.unwrap()
            else {
                panic!("completed");
            };
            let completion = output.into_completions().remove(0);
            if matches!(work, TokenWork::Prefill(chunk) if !chunk.is_final()) {
                assert!(matches!(
                    completion.product(),
                    ExecutionProduct::Intermediate
                ));
            } else {
                let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) =
                    completion.product()
                else {
                    panic!("full logits");
                };
                assert_eq!(values, &[0.0; 16]);
            }
            frontier = completion.next_frontier().clone();
        }
        let outcome = executor
            .finish_tokens(
                first.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 2,
                    output_tokens: 2,
                },
            )
            .await
            .unwrap();
        assert_ended(outcome, first.sequence());
        let TokenAdmissionOutcome::Admitted(replacement) =
            executor.admit_tokens(admission()).await.unwrap()
        else {
            panic!("replacement");
        };
        assert_ne!(first.sequence(), replacement.sequence());
        assert_ended(
            executor.cancel_tokens(first.sequence()).unwrap(),
            first.sequence(),
        );
        assert!(replacement.sequence().observed_end().is_none());
        let stale = [ExecutionWork::new(
            &first,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &policy,
        )
        .unwrap()];
        assert!(executor.execute_tokens(&stale).await.is_err());
        assert_ended(
            executor.cancel_tokens(replacement.sequence()).unwrap(),
            replacement.sequence(),
        );
    }
}
