//! Qwen2 model executor using Candle

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
use std::{sync::Arc, time::Instant};
use tracing::{debug, info};

use crate::{architectures::qwen2::Qwen2ModelWrapper, tensor_wrapper::CandleTensorWrapper};

/// Shared state between prefill and decode phases
#[derive(Debug, Clone)]
struct Qwen2CacheState {
    /// Current sequence length processed by the model
    sequence_length: usize,
    /// KV cache handle exposed to the engine
    kv_handle: Arc<Qwen2KvCacheHandle>,
}

/// Candle-based Qwen2 model executor
pub struct Qwen2ModelExecutor {
    model: Arc<Qwen2ModelWrapper>,
    info: ModelInfo,
    state: Mutex<Option<Qwen2CacheState>>,
}

impl Qwen2ModelExecutor {
    /// Create new Qwen2 executor
    pub fn new(model: Qwen2ModelWrapper, info: ModelInfo) -> Self {
        info!("✅ Created Qwen2ModelExecutor for: {}", info.model_id);

        Self {
            model: Arc::new(model),
            info,
            state: Mutex::new(None),
        }
    }

    /// Extract token IDs from tensor reference (supports [batch, seq] tensors)
    fn tensor_to_tokens(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        tensor.to_vec_u32()
    }

    /// Create Candle tensor from token IDs on the correct device
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

    /// Convert Candle tensor to TensorRef wrapper
    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }
}

#[async_trait]
impl ModelExecutor for Qwen2ModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!(
            "Qwen2 Prefill: batch={}, seq_len={}",
            input.batch_size(),
            input.sequence_length()
        );

        // Extract tokens and build tensor
        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Prefill input is empty"));
        }

        // Reset internal KV cache before new request
        self.model.reset_cache()?;

        let input_tensor = self.tokens_to_tensor(&tokens)?;

        // Run forward pass with offset 0
        let logits = self
            .model
            .forward_prefill(&input_tensor)
            .map_err(|e| FerrumError::model(format!("Qwen2 prefill failed: {}", e)))?;

        // Qwen2 returns [batch, vocab]; expand to [batch, seq(=1), vocab]
        let logits = logits
            .unsqueeze(1)
            .map_err(|e| FerrumError::model(format!("Unsqueeze logits failed: {}", e)))?;

        let logits_ref = self.wrap_tensor(logits);

        // Create KV cache handle representing internal state
        let kv_handle = Arc::new(Qwen2KvCacheHandle::new(
            self.model.config(),
            self.model.device().clone(),
            tokens.len(),
        ));

        // Store state for subsequent decode steps
        *self.state.lock() = Some(Qwen2CacheState {
            sequence_length: tokens.len(),
            kv_handle: kv_handle.clone(),
        });

        Ok(PrefillOutput::new(logits_ref, kv_handle))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        debug!("Qwen2 Decode: batch={}", input.batch_size());

        let mut guard = self.state.lock();
        let state = guard
            .as_mut()
            .ok_or_else(|| FerrumError::model("Decode called before prefill"))?;

        // Extract single token for decode
        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Decode input is empty"));
        }

        let input_tensor = self.tokens_to_tensor(&tokens)?;

        let logits = self
            .model
            .forward_decode(&input_tensor, state.sequence_length)
            .map_err(|e| FerrumError::model(format!("Qwen2 decode failed: {}", e)))?;

        let logits_ref = self.wrap_tensor(logits);

        // Update sequence length and KV handle
        state.sequence_length += tokens.len();
        let new_handle = Arc::new(state.kv_handle.with_sequence_length(state.sequence_length));
        state.kv_handle = new_handle.clone();

        Ok(DecodeOutput::new(logits_ref, new_handle))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead, AttentionType::GroupedQuery],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP16, DataType::FP32, DataType::BF16],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2) as u64,
                activation_memory_per_token: self.info.hidden_size * 4,
                kv_cache_memory_per_token: self.info.hidden_size * 2,
                overhead_memory: 256 * 1024 * 1024, // 256MB placeholder
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
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            last_operation: Some(Instant::now()),
        }
    }
}

/// Lightweight KV cache handle for Qwen2 models (model maintains cache internally)
#[derive(Debug, Clone)]
struct Qwen2KvCacheHandle {
    block_table: BlockTable,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
}

impl Qwen2KvCacheHandle {
    fn new(
        config: &candle_transformers::models::qwen2::Config,
        device: CandleDevice,
        seq_len: usize,
    ) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = seq_len;

        Self {
            block_table,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
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
        }
    }
}

impl KvCacheHandle for Qwen2KvCacheHandle {
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
        format!("qwen2-cache-{}", self.block_table.sequence_length)
    }
}
