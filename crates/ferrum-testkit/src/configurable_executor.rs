//! Configurable model executor for testing stop sequences, EOS, and specific token patterns.
//!
//! Unlike MockModelExecutor (always biases token 42), this executor can be configured
//! to produce specific token sequences, emit EOS after N tokens, etc.

use crate::kv_cache::MockKvCacheHandle;
use crate::tensor::MockTensor;
use async_trait::async_trait;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
        ExecutorState, ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    ModelExecutor,
};
use ferrum_types::{DataType, Device, ModelInfo, ModelType, RequestId, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
        }
    }

    fn next_token_logits(&self) -> Vec<f32> {
        let step = self.decode_count.load(Ordering::Relaxed) as usize;
        let vocab_size = self.info.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];

        // Check if we should emit EOS
        if let Some(eos_n) = self.eos_after {
            if step >= eos_n {
                if (self.eos_token as usize) < vocab_size {
                    logits[self.eos_token as usize] = 10.0;
                }
                return logits;
            }
        }

        // Cycle through token sequence
        let token = self.token_sequence[step % self.token_sequence.len()];
        if (token as usize) < vocab_size {
            logits[token as usize] = 10.0;
        }
        logits
    }
}

#[async_trait]
impl ModelExecutor for ConfigurableModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let batch_size = input.batch_size();
        let seq_len = input.sequence_length();
        let vocab_size = self.info.vocab_size;

        // Prefill logits: bias first token from sequence
        let token = self.token_sequence[0];
        let mut logits_data = vec![0.0f32; batch_size * seq_len * vocab_size];
        for b in 0..batch_size {
            for s in 0..seq_len {
                let offset = (b * seq_len + s) * vocab_size;
                if offset + token as usize >= logits_data.len() {
                    continue;
                }
                logits_data[offset + token as usize] = 10.0;
            }
        }
        let logits =
            MockTensor::from_f32(logits_data, &[batch_size, seq_len, vocab_size]).into_ref();
        let kv_cache = Arc::new(MockKvCacheHandle::new(
            RequestId::new(),
            self.info.num_layers,
            seq_len,
        ));
        Ok(PrefillOutput::new(logits, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let batch_size = input.batch_size();
        let vocab_size = self.info.vocab_size;

        let single_logits = self.next_token_logits();
        self.decode_count.fetch_add(1, Ordering::Relaxed);

        // Replicate for batch
        let mut logits_data = Vec::with_capacity(batch_size * vocab_size);
        for _ in 0..batch_size {
            logits_data.extend_from_slice(&single_logits);
        }
        let logits = MockTensor::from_f32(logits_data, &[batch_size, vocab_size]).into_ref();
        Ok(DecodeOutput::new(logits, input.kv_cache.clone()))
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
