//! Mock model executor with configurable latency for scheduling tests.

use crate::tensor::MockTensor;
use crate::kv_cache::MockKvCacheHandle;
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Mock model executor that simulates prefill/decode with configurable latency.
/// No model weights, no GPU — pure async simulation.
pub struct MockModelExecutor {
    info: ModelInfo,
    prefill_latency: Duration,
    decode_latency: Duration,
    prefill_count: AtomicU64,
    decode_count: AtomicU64,
}

impl MockModelExecutor {
    pub fn new(vocab_size: usize, prefill_latency: Duration, decode_latency: Duration) -> Self {
        let info = ModelInfo {
            model_id: "mock-model".into(),
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
            version: Some("mock-1.0".into()),
            license: None,
            metadata: HashMap::new(),
        };
        Self {
            info,
            prefill_latency,
            decode_latency,
            prefill_count: AtomicU64::new(0),
            decode_count: AtomicU64::new(0),
        }
    }

    /// Create with zero latency (for fast unit tests).
    pub fn instant(vocab_size: usize) -> Self {
        Self::new(vocab_size, Duration::ZERO, Duration::ZERO)
    }

    pub fn prefill_count(&self) -> u64 {
        self.prefill_count.load(Ordering::Relaxed)
    }

    pub fn decode_count(&self) -> u64 {
        self.decode_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ModelExecutor for MockModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        if !self.prefill_latency.is_zero() {
            tokio::time::sleep(self.prefill_latency).await;
        }
        self.prefill_count.fetch_add(1, Ordering::Relaxed);

        let batch_size = input.batch_size();
        let seq_len = input.sequence_length();
        let vocab_size = self.info.vocab_size;

        // Return synthetic logits [batch, seq_len, vocab_size]
        // Put a slight bias toward token 42 so greedy sampling is deterministic
        let mut logits_data = vec![0.0f32; batch_size * seq_len * vocab_size];
        for b in 0..batch_size {
            for s in 0..seq_len {
                let offset = (b * seq_len + s) * vocab_size;
                if offset + 42 < logits_data.len() {
                    logits_data[offset + 42] = 1.0;
                }
            }
        }
        let logits = MockTensor::from_f32(logits_data, &[batch_size, seq_len, vocab_size]).into_ref();

        let kv_cache = Arc::new(MockKvCacheHandle::new(
            RequestId::new(),
            self.info.num_layers,
            seq_len,
        ));

        Ok(PrefillOutput::new(logits, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        if !self.decode_latency.is_zero() {
            tokio::time::sleep(self.decode_latency).await;
        }
        self.decode_count.fetch_add(1, Ordering::Relaxed);

        let batch_size = input.batch_size();
        let vocab_size = self.info.vocab_size;

        // Return logits [batch, vocab_size] with bias toward token 42
        let mut logits_data = vec![0.0f32; batch_size * vocab_size];
        for b in 0..batch_size {
            let offset = b * vocab_size;
            if offset + 42 < logits_data.len() {
                logits_data[offset + 42] = 1.0;
            }
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
            prefill_operations: self.prefill_count.load(Ordering::Relaxed),
            decode_operations: self.decode_count.load(Ordering::Relaxed),
            avg_prefill_time_ms: self.prefill_latency.as_millis() as f64,
            avg_decode_time_ms: self.decode_latency.as_millis() as f64,
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
