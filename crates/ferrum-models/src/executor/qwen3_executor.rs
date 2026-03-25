//! Qwen3 model executor using Candle

use async_trait::async_trait;
use candle_core::{Device as CandleDevice, Tensor};
use ferrum_interfaces::{
    kv_cache::{BlockTable, CacheHandleStats},
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
        ExecutorState, ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use tracing::{debug, info};

use crate::{architectures::qwen3::Qwen3ModelWrapper, tensor_wrapper::CandleTensorWrapper};

#[derive(Debug, Clone)]
struct Qwen3CacheState {
    sequence_length: usize,
}

/// Candle-based Qwen3 model executor with multi-sequence support.
///
/// Each active sequence gets its own KV cache keyed by a unique cache_id.
/// This allows concurrent prefill and decode across many sequences without
/// one sequence's prefill destroying another's KV cache.
pub struct Qwen3ModelExecutor {
    model: Arc<Qwen3ModelWrapper>,
    info: ModelInfo,
    states: Mutex<HashMap<String, Qwen3CacheState>>,
    next_cache_id: AtomicU64,
}

impl Qwen3ModelExecutor {
    pub fn new(model: Qwen3ModelWrapper, info: ModelInfo) -> Self {
        info!("Created Qwen3ModelExecutor for: {}", info.model_id);

        Self {
            model: Arc::new(model),
            info,
            states: Mutex::new(HashMap::new()),
            next_cache_id: AtomicU64::new(1),
        }
    }

    /// Release a sequence's KV cache, freeing GPU memory.
    /// Should be called when a request completes.
    pub fn release_sequence(&self, cache_id: &str) {
        self.states.lock().remove(cache_id);
        self.model.release_cache(cache_id);
        debug!("Released KV cache for sequence: {}", cache_id);
    }

    fn tensor_to_tokens(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        if let Ok(tokens) = tensor.to_vec_u32() {
            if tokens.is_empty() {
                return Err(FerrumError::model("Input token tensor is empty"));
            }
            return Ok(tokens);
        }

        if let Ok(tokens_f32) = tensor.to_vec_f32() {
            let tokens: Vec<u32> = tokens_f32.into_iter().map(|x| x as u32).collect();
            if tokens.is_empty() {
                return Err(FerrumError::model("Input token tensor is empty"));
            }
            return Ok(tokens);
        }

        Err(FerrumError::model(
            "Unable to extract token IDs from input tensor",
        ))
    }

    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let base = Tensor::new(tokens, &CandleDevice::Cpu)
            .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("Failed to unsqueeze tensor: {}", e)))?
            .to_dtype(candle_core::DType::I64)
            .map_err(|e| FerrumError::model(format!("Failed to cast tokens to I64: {}", e)))?;

        match self.model.candle_device() {
            CandleDevice::Cpu => Ok(base),
            CandleDevice::Cuda(dev) => base
                .to_device(&CandleDevice::Cuda(dev.clone()))
                .map_err(|e| FerrumError::model(format!("Failed to move tensor to CUDA: {}", e))),
            CandleDevice::Metal(dev) => base
                .to_device(&CandleDevice::Metal(dev.clone()))
                .map_err(|e| FerrumError::model(format!("Failed to move tensor to Metal: {}", e))),
        }
    }

    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }
}

#[async_trait]
impl ModelExecutor for Qwen3ModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!(
            "Qwen3 Prefill: batch={}, seq_len={}",
            input.batch_size(),
            input.sequence_length()
        );

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Prefill input is empty"));
        }

        let cache_id = format!(
            "qwen3-cache-{}",
            self.next_cache_id.fetch_add(1, Ordering::Relaxed)
        );

        let input_tensor = self.tokens_to_tensor(&tokens)?;

        // Each sequence gets its own KV cache slot; no need to clear other sequences.
        let logits = self
            .model
            .forward_prefill(&input_tensor, &cache_id)
            .map_err(|e| FerrumError::model(format!("Qwen3 prefill failed: {}", e)))?;

        let logits = match logits.dims().len() {
            2 => logits
                .unsqueeze(1)
                .map_err(|e| FerrumError::model(format!("Unsqueeze logits failed: {}", e)))?,
            3 => logits,
            dims => {
                return Err(FerrumError::model(format!(
                    "Unexpected Qwen3 prefill logits rank: {} (shape {:?})",
                    dims,
                    logits.dims()
                )))
            }
        };

        let logits_ref = self.wrap_tensor(logits);

        let kv_handle = Arc::new(Qwen3KvCacheHandle::new(
            self.model.config(),
            self.model.device().clone(),
            tokens.len(),
            cache_id.clone(),
        ));

        self.states.lock().insert(
            cache_id,
            Qwen3CacheState {
                sequence_length: tokens.len(),
            },
        );

        Ok(PrefillOutput::new(logits_ref, kv_handle))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        debug!("Qwen3 Decode: batch={}", input.batch_size());

        let input_handle = input
            .kv_cache
            .as_any()
            .downcast_ref::<Qwen3KvCacheHandle>()
            .ok_or_else(|| FerrumError::model("Invalid KV cache handle type for Qwen3 executor"))?;
        let req_cache_id = input_handle.request_cache_id().to_string();

        let seq_len = {
            let states = self.states.lock();
            let state = states.get(&req_cache_id).ok_or_else(|| {
                FerrumError::model(format!(
                    "Decode called for unknown sequence: {}",
                    req_cache_id
                ))
            })?;
            state.sequence_length
        };

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Decode input is empty"));
        }

        let input_tensor = self.tokens_to_tensor(&tokens)?;

        let logits = self
            .model
            .forward_decode(&input_tensor, seq_len, &req_cache_id)
            .map_err(|e| FerrumError::model(format!("Qwen3 decode failed: {}", e)))?;

        let logits_ref = self.wrap_tensor(logits);

        let new_seq_len = {
            let mut states = self.states.lock();
            if let Some(state) = states.get_mut(&req_cache_id) {
                state.sequence_length += tokens.len();
                state.sequence_length
            } else {
                seq_len + tokens.len()
            }
        };
        let new_handle = Arc::new(input_handle.with_sequence_length(new_seq_len));

        Ok(DecodeOutput::new(logits_ref, new_handle))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 256,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead, AttentionType::GroupedQuery],
            supports_dynamic_batching: true,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP16, DataType::FP32, DataType::BF16],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2) as u64,
                activation_memory_per_token: self.info.hidden_size * 4,
                kv_cache_memory_per_token: self.info.hidden_size * 2,
                overhead_memory: 256 * 1024 * 1024,
            },
        }
    }

    fn release_cache(&self, cache_id: &str) {
        self.release_sequence(cache_id);
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
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            last_operation: Some(Instant::now()),
        }
    }
}

#[derive(Debug, Clone)]
struct Qwen3KvCacheHandle {
    block_table: BlockTable,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
    request_cache_id: String,
}

impl Qwen3KvCacheHandle {
    fn new(
        config: &crate::architectures::qwen3::Config,
        device: CandleDevice,
        seq_len: usize,
        request_cache_id: String,
    ) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = seq_len;

        Self {
            block_table,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
            request_cache_id,
            device: match device {
                CandleDevice::Cpu => Device::CPU,
                CandleDevice::Cuda(_dev) => Device::CUDA(0),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                CandleDevice::Metal(_) => Device::Metal,
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                CandleDevice::Metal(_) => Device::CPU,
            },
        }
    }

    fn with_sequence_length(&self, seq_len: usize) -> Self {
        let mut block_table = self.block_table.clone();
        block_table.sequence_length = seq_len;

        Self {
            block_table,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            device: self.device.clone(),
            request_cache_id: self.request_cache_id.clone(),
        }
    }

    fn request_cache_id(&self) -> &str {
        &self.request_cache_id
    }
}

impl KvCacheHandle for Qwen3KvCacheHandle {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn key_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn stats(&self) -> CacheHandleStats {
        CacheHandleStats {
            memory_bytes: 0,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.block_table.sequence_length,
            utilization: 0.0,
            last_access: Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        self.request_cache_id.clone()
    }
}
