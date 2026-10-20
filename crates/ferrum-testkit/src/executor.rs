//! Mock model executor with configurable latency for scheduling tests.

use crate::executor_observation::{ExecutorCounters, ExecutorObservation};
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
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

pub(crate) mod token_state;
use token_state::TokenState;

/// Mock model executor that simulates prefill/decode with configurable latency.
/// No model weights, no GPU — pure async simulation.
pub struct MockModelExecutor {
    info: ModelInfo,
    prefill_latency: Duration,
    decode_latency: Duration,
    counters: Arc<ExecutorCounters>,
    tokens: TokenState,
}

impl MockModelExecutor {
    /// Observe counts without retaining this executor or its resources.
    pub fn observer(&self) -> ExecutorObservation {
        ExecutorObservation::new(&self.counters)
    }

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
            counters: Arc::default(),
            tokens: TokenState::default(),
        }
    }

    /// Create with zero latency (for fast unit tests).
    pub fn instant(vocab_size: usize) -> Self {
        Self::new(vocab_size, Duration::ZERO, Duration::ZERO)
    }

    pub fn prefill_count(&self) -> u64 {
        self.counters.prefill.load(Ordering::Relaxed)
    }

    pub fn decode_count(&self) -> u64 {
        self.counters.decode.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ModelExecutor for MockModelExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        crate::host_shutdown::begin(self, permit, |executor| executor.tokens.shutdown())
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        self.tokens
            .admit(input, self.info.max_sequence_length, self.info.vocab_size)
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        // No state is touched until the delay completes. Cancellation during
        // this simulated latency therefore leaves the original frontier live.
        let prefill = work
            .iter()
            .any(|input| matches!(input.work(), TokenWork::Prefill(_)));
        let latency = if prefill {
            self.prefill_latency
        } else {
            self.decode_latency
        };
        if !latency.is_zero() {
            tokio::time::sleep(latency).await;
        }
        let outcome = self.tokens.execute(work, self.info.vocab_size, |_, _| {
            let mut logits = vec![0.0; self.info.vocab_size];
            if let Some(value) = logits.get_mut(42) {
                *value = 1.0;
            }
            logits
        })?;
        if matches!(outcome, ExecutionBatchOutcome::Completed(_)) {
            for input in work {
                match input.work() {
                    TokenWork::Prefill(_) => {
                        self.counters.prefill.fetch_add(1, Ordering::Relaxed);
                    }
                    TokenWork::Decode { .. } => {
                        self.counters.decode.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
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
            prefill_operations: self.counters.prefill.load(Ordering::Relaxed),
            decode_operations: self.counters.decode.load(Ordering::Relaxed),
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
