//! Model executor that uses PagedAttention KV cache.
//!
//! Unlike MockModelExecutor (which ignores KV cache), this executor:
//! - Writes K/V vectors to paged blocks during prefill and decode
//! - Reads K/V through block table indirection for attention
//! - Produces logits via the paged attention output
//!
//! Uses identity projections (Q=K=V=input embedding) for deterministic,
//! verifiable behavior without model weights.

use crate::tensor::MockTensor;
use async_trait::async_trait;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
        ExecutorState, ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    KvCacheHandle, KvCacheManager, ModelExecutor,
};
use ferrum_kv::attention::paged_attention;
use ferrum_kv::managers::paged::{PagedKvCacheHandle, PagedKvCacheManager};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, ModelType, RequestId, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
    /// Shared with the engine's KV cache manager.
    kv_manager: Arc<PagedKvCacheManager>,
    prefill_count: AtomicU64,
    decode_count: AtomicU64,
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

    /// Get the paged handle from an Arc<dyn KvCacheHandle>.
    fn as_paged_handle<'a>(handle: &'a dyn KvCacheHandle) -> Result<&'a PagedKvCacheHandle> {
        handle
            .as_any()
            .downcast_ref::<PagedKvCacheHandle>()
            .ok_or_else(|| FerrumError::internal("Expected PagedKvCacheHandle"))
    }
}

#[async_trait]
impl ModelExecutor for PagedAttentionExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.prefill_count.fetch_add(1, Ordering::Relaxed);

        let batch_size = input.batch_size();
        let seq_len = input.sequence_length();
        let vocab_size = self.config.vocab_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;

        // Extract token IDs from input tensor
        let token_ids = input.input_ids.to_vec_u32()?;

        // Use pre-allocated KV cache from engine, or allocate if not provided
        let kv_handle = match &input.kv_cache {
            Some(handle) => handle.clone(),
            None => {
                let alloc_request = ferrum_interfaces::kv_cache::AllocationRequest {
                    request_id: RequestId::new(),
                    initial_tokens: seq_len,
                    max_sequence_length: self.config.max_sequence_length,
                    num_layers: self.config.num_layers,
                    num_heads: num_kv_heads,
                    head_dim,
                    device: Device::CPU,
                    dtype: DataType::FP32,
                    priority: ferrum_types::Priority::Normal,
                };
                self.kv_manager.allocate(&alloc_request).await?
            }
        };
        let paged_handle = Self::as_paged_handle(kv_handle.as_ref())?;

        // For each token, compute embedding and write K/V to paged cache
        for pos in 0..seq_len {
            let token_id = if pos < token_ids.len() {
                token_ids[pos]
            } else {
                0
            };
            let embedding = self.token_embedding(token_id);

            // Identity projection: K = V = embedding
            for layer in 0..self.config.num_layers {
                self.kv_manager
                    .write_kv(paged_handle, layer, pos, &embedding, &embedding)?;
            }
        }

        // Run paged attention for all layers, accumulate output for last layer
        // For logits, we only need the last layer's attention output
        let last_layer = self.config.num_layers - 1;

        // Build query for all positions (prefill): Q = embeddings
        let mut query = Vec::with_capacity(seq_len * num_heads * head_dim);
        for pos in 0..seq_len {
            let token_id = if pos < token_ids.len() {
                token_ids[pos]
            } else {
                0
            };
            let emb = self.token_embedding(token_id);
            // Expand KV heads to query heads if num_heads > num_kv_heads (GQA)
            let heads_per_kv = num_heads / num_kv_heads;
            for kv_h in 0..num_kv_heads {
                for _ in 0..heads_per_kv {
                    query.extend_from_slice(&emb[kv_h * head_dim..(kv_h + 1) * head_dim]);
                }
            }
        }

        let attn_output = paged_attention(
            &query,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.kv_manager,
            paged_handle,
            last_layer,
            seq_len,
        )?;

        // Build logits [batch, seq_len, vocab_size]
        let q_head_size = num_heads * head_dim;
        let mut logits_data = Vec::with_capacity(batch_size * seq_len * vocab_size);
        for _b in 0..batch_size {
            for s in 0..seq_len {
                let attn_slice = &attn_output[s * q_head_size..(s + 1) * q_head_size];
                let token_logits = self.attention_to_logits(attn_slice);
                logits_data.extend_from_slice(&token_logits);
            }
        }

        let logits =
            MockTensor::from_f32(logits_data, &[batch_size, seq_len, vocab_size]).into_ref();

        Ok(PrefillOutput::new(logits, kv_handle))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.decode_count.fetch_add(1, Ordering::Relaxed);

        let batch_size = input.batch_size();
        let vocab_size = self.config.vocab_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;

        let paged_handle = Self::as_paged_handle(input.kv_cache.as_ref())?;

        // Current position = number of tokens already cached
        let position = paged_handle.num_tokens();

        // We may need to allocate an additional block
        let blocks_needed = (position + 1 + self.kv_manager.gpu_pool().block_size() - 1)
            / self.kv_manager.gpu_pool().block_size();
        let current_blocks = paged_handle.num_blocks();
        if blocks_needed > current_blocks {
            self.kv_manager
                .allocate_blocks(paged_handle, blocks_needed - current_blocks)?;
        }

        // Extract the new token ID
        let token_ids = input.input_ids.to_vec_u32()?;
        let token_id = token_ids.first().copied().unwrap_or(0);

        let embedding = self.token_embedding(token_id);

        // Write K/V for the new token at `position` in each layer
        for layer in 0..self.config.num_layers {
            self.kv_manager
                .write_kv(paged_handle, layer, position, &embedding, &embedding)?;
        }

        // Update token count
        paged_handle.set_num_tokens(position + 1);

        // Run attention for the new token (decode: q_tokens=1, kv_len=position+1)
        let last_layer = self.config.num_layers - 1;

        // Build query for the single new token
        let mut query = Vec::with_capacity(num_heads * head_dim);
        let heads_per_kv = num_heads / num_kv_heads;
        for kv_h in 0..num_kv_heads {
            for _ in 0..heads_per_kv {
                query.extend_from_slice(&embedding[kv_h * head_dim..(kv_h + 1) * head_dim]);
            }
        }

        let attn_output = paged_attention(
            &query,
            1,
            num_heads,
            num_kv_heads,
            head_dim,
            &self.kv_manager,
            paged_handle,
            last_layer,
            position + 1,
        )?;

        // Convert attention output to logits [batch, vocab_size]
        let mut logits_data = Vec::with_capacity(batch_size * vocab_size);
        for _b in 0..batch_size {
            let token_logits = self.attention_to_logits(&attn_output);
            logits_data.extend_from_slice(&token_logits);
        }

        let logits = MockTensor::from_f32(logits_data, &[batch_size, vocab_size]).into_ref();

        Ok(DecodeOutput::new(logits, input.kv_cache.clone()))
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
