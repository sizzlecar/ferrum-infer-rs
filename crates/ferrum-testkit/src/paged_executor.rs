//! Model executor that uses PagedAttention KV cache.
//!
//! Unlike MockModelExecutor (which ignores KV cache), this executor:
//! - Writes K/V vectors to paged blocks during prefill and decode
//! - Reads K/V through block table indirection for attention
//! - Produces logits via the paged attention output
//!
//! Uses identity projections (Q=K=V=input embedding) for deterministic,
//! verifiable behavior without model weights.

use async_trait::async_trait;
use ferrum_execution::{
    model_executor::{
        AttentionType, ExecutorCapabilities, ExecutorMemoryUsage, ExecutorState, ExecutorStatus,
        MemoryRequirements,
    },
    ModelExecutor,
};
use ferrum_interfaces::KvCacheHandle;
use ferrum_kv::attention::paged_attention;
use ferrum_kv::managers::paged::{PagedKvCacheHandle, PagedKvCacheManager};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, ModelType, RequestId, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

mod token_execution;
use token_execution::PagedTokenState;

/// Configuration for the paged attention executor.
#[derive(Debug, Clone)]
pub struct PagedExecutorConfig {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_sequence_length: usize,
}

impl Default for PagedExecutorConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            max_sequence_length: 512,
        }
    }
}

/// A model executor that actually uses paged KV cache for attention.
///
/// Uses identity projections: for each token, the embedding is a one-hot
/// vector of length `num_kv_heads * head_dim` derived from the token ID.
/// Q = K = V = embedding.  This makes attention outputs deterministic
/// and verifiable.
///
/// Logits are produced by summing attention output elements per head and
/// distributing across vocab positions, so different attention patterns
/// produce different token predictions.
pub struct PagedAttentionExecutor {
    config: PagedExecutorConfig,
    info: ModelInfo,
    /// Owns physical KV; token entries retain exact handles from this manager.
    kv_manager: Arc<PagedKvCacheManager>,
    prefill_count: AtomicU64,
    decode_count: AtomicU64,
    tokens: PagedTokenState,
}

impl PagedAttentionExecutor {
    pub fn new(config: PagedExecutorConfig, kv_manager: Arc<PagedKvCacheManager>) -> Self {
        let info = ModelInfo {
            model_id: "paged-test-model".into(),
            model_type: ModelType::Custom("paged-test".into()),
            num_parameters: 0,
            hidden_size: config.num_heads * config.head_dim,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            vocab_size: config.vocab_size,
            max_sequence_length: config.max_sequence_length,
            dtype: DataType::FP32,
            device: Device::CPU,
            version: Some("test-1.0".into()),
            license: None,
            metadata: HashMap::new(),
        };

        Self {
            config,
            info,
            kv_manager,
            prefill_count: AtomicU64::new(0),
            decode_count: AtomicU64::new(0),
            tokens: PagedTokenState::default(),
        }
    }

    pub fn prefill_count(&self) -> u64 {
        self.prefill_count.load(Ordering::Relaxed)
    }

    pub fn decode_count(&self) -> u64 {
        self.decode_count.load(Ordering::Relaxed)
    }

    /// Create an embedding vector from a token ID.
    ///
    /// Returns a vector of length `num_kv_heads * head_dim` where each
    /// element is derived from the token ID for deterministic behavior.
    fn token_embedding(&self, token_id: u32) -> Vec<f32> {
        let kv_size = self.config.num_kv_heads * self.config.head_dim;
        let mut emb = vec![0.0f32; kv_size];
        // Spread the token value across dimensions to create distinct patterns
        for i in 0..kv_size {
            emb[i] = ((token_id as f32 + 1.0) * (i as f32 + 1.0)).sin();
        }
        emb
    }

    /// Convert attention output to logits [vocab_size].
    ///
    /// Simple linear projection: each attention output dimension contributes
    /// to vocab positions via modular mapping.
    fn attention_to_logits(&self, attn_output: &[f32]) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        for (i, &val) in attn_output.iter().enumerate() {
            let vocab_idx = i % vocab_size;
            logits[vocab_idx] += val;
        }
        logits
    }

    #[cfg(test)]
    fn as_paged_handle(handle: &dyn KvCacheHandle) -> Result<&PagedKvCacheHandle> {
        handle
            .as_any()
            .downcast_ref::<PagedKvCacheHandle>()
            .ok_or_else(|| FerrumError::internal("Expected PagedKvCacheHandle"))
    }
}

#[async_trait]
impl ModelExecutor for PagedAttentionExecutor {
    async fn admit_tokens(
        &self,
        input: ferrum_execution::token_execution::ExecutionAdmission<'_>,
    ) -> Result<ferrum_execution::token_execution::TokenAdmissionOutcome> {
        self.admit_token_sequence(input)
    }

    async fn execute_tokens(
        &self,
        work: &[ferrum_execution::token_execution::ExecutionWork<'_>],
    ) -> Result<ferrum_execution::token_execution::ExecutionBatchOutcome> {
        self.execute_token_work(work)
    }

    async fn finish_tokens(
        &self,
        sequence: &ferrum_execution::token_execution::ExecutorSequence,
        reason: ferrum_execution::token_execution::ExecutionEndReason,
    ) -> Result<ferrum_execution::token_execution::ExecutionCloseOutcome> {
        self.close_token_sequence(sequence, reason)
    }

    fn cancel_tokens(
        &self,
        sequence: &ferrum_execution::token_execution::ExecutorSequence,
    ) -> Result<ferrum_execution::token_execution::ExecutionCloseOutcome> {
        self.close_token_sequence(
            sequence,
            ferrum_execution::token_execution::ExecutionEndReason::Cancelled,
        )
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 64,
            max_sequence_length: self.config.max_sequence_length,
            attention_mechanisms: vec![AttentionType::Paged],
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
                kv_cache_memory_per_token: (self.config.num_kv_heads
                    * self.config.head_dim
                    * 2
                    * self.config.num_layers
                    * 4) as u64 as usize,
                overhead_memory: 0,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            state: ExecutorState::Ready,
            is_ready: true,
            current_batch_size: 0,
            prefill_operations: self.prefill_count(),
            decode_operations: self.decode_count(),
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
