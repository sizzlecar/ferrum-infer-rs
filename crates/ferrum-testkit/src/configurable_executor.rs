//! Configurable model executor for testing stop sequences, EOS, and specific token patterns.
//!
//! Unlike MockModelExecutor (always biases token 42), this executor can be configured
//! to produce specific token sequences, emit EOS after N tokens, etc.

use crate::executor::token_state::TokenState;
use async_trait::async_trait;
use ferrum_execution::token_execution::*;
use ferrum_execution::{
    model_executor::{
        AttentionType, ExecutorCapabilities, ExecutorMemoryUsage, ExecutorState, ExecutorStatus,
        MemoryRequirements,
    },
    ModelExecutor,
};
use ferrum_types::{DataType, Device, ModelInfo, ModelType, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Model executor that produces a configurable sequence of tokens.
pub struct ConfigurableModelExecutor {
    info: ModelInfo,
    /// Token sequence to cycle through on decode.
    token_sequence: Vec<u32>,
    /// If set, emit this token (EOS) after this many decode steps.
    eos_after: Option<usize>,
    /// EOS token ID.
    eos_token: u32,
    decode_count: AtomicU64,
    tokens: TokenState,
}

impl ConfigurableModelExecutor {
    /// Create executor that cycles through the given token sequence.
    pub fn with_token_sequence(vocab_size: usize, tokens: Vec<u32>) -> Self {
        Self {
            info: mock_info(vocab_size),
            token_sequence: tokens,
            eos_after: None,
            eos_token: 2, // common EOS
            decode_count: AtomicU64::new(0),
            tokens: TokenState::default(),
        }
    }

    /// Create executor that emits EOS after `n` decode steps.
    pub fn with_eos_after(vocab_size: usize, n: usize, eos_token: u32) -> Self {
        Self {
            info: mock_info(vocab_size),
            token_sequence: vec![42], // default token before EOS
            eos_after: Some(n),
            eos_token,
            decode_count: AtomicU64::new(0),
            tokens: TokenState::default(),
        }
    }
}

#[async_trait]
impl ModelExecutor for ConfigurableModelExecutor {
    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        if self.token_sequence.is_empty() {
            return Err(ferrum_types::FerrumError::request_validation(
                "empty configurable output sequence",
            ));
        }
        self.tokens
            .admit(input, self.info.max_sequence_length, self.info.vocab_size)
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let outcome = self
            .tokens
            .execute(work, self.info.vocab_size, |work, prompt_len| {
                // Decode scripts restart per incarnation, not per batch or executor.
                // Final prefill emits sequence[0], preserving the original fixture.
                let token = match work {
                    TokenWork::Prefill(_) => self.token_sequence[0],
                    TokenWork::Decode { position, .. } => {
                        let step = position - prompt_len;
                        if self.eos_after.is_some_and(|limit| step >= limit) {
                            self.eos_token
                        } else {
                            self.token_sequence[step % self.token_sequence.len()]
                        }
                    }
                };
                let mut logits = vec![0.0; self.info.vocab_size];
                if let Some(value) = logits.get_mut(token as usize) {
                    *value = 10.0;
                }
                logits
            })?;
        if matches!(outcome, ExecutionBatchOutcome::Completed(_)) {
            self.decode_count.fetch_add(
                work.iter()
                    .filter(|input| matches!(input.work(), TokenWork::Decode { .. }))
                    .count() as u64,
                Ordering::Relaxed,
            );
        }
        Ok(outcome)
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.tokens.close(sequence, reason)
    }

    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.tokens.close(sequence, ExecutionEndReason::Cancelled)
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 256,
            max_sequence_length: 4096,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: true,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32],
            supported_devices: vec![Device::CPU],
            memory_requirements: MemoryRequirements {
                typed_sequence_state: None,
                parameter_memory: 0,
                activation_memory_per_token: 0,
                kv_cache_memory_per_token: 0,
                overhead_memory: 0,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            state: ExecutorState::Ready,
            is_ready: true,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: self.decode_count.load(Ordering::Relaxed),
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            last_operation: None,
        }
    }
}

fn mock_info(vocab_size: usize) -> ModelInfo {
    ModelInfo {
        model_id: "configurable-mock".into(),
        model_type: ModelType::Custom("mock".into()),
        num_parameters: 1_000_000,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 12,
        vocab_size,
        max_sequence_length: 4096,
        dtype: DataType::FP32,
        device: Device::CPU,
        version: Some("configurable-1.0".into()),
        license: None,
        metadata: HashMap::new(),
    }
}
