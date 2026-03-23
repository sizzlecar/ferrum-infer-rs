//! Real model executor using Candle - safe implementation

use async_trait::async_trait;
use candle_core::Tensor;
use ferrum_interfaces::{
    model_executor::{
        DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus, PrefillInput,
        PrefillOutput,
    },
    BlockTable, KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing::{debug, info};

use crate::architectures::llama::LlamaModelWrapper;
use crate::tensor_wrapper::CandleTensorWrapper;
use candle_transformers::models::llama::Cache as LlamaCache;
use parking_lot::Mutex;

/// Candle-based model executor
pub struct CandleModelExecutor {
    model: Arc<LlamaModelWrapper>,
    info: ModelInfo,
}

struct LlamaKvCacheShared {
    cache: Mutex<LlamaCache>,
    sequence_length: AtomicUsize,
}

struct LlamaKvCacheHandle {
    shared: Arc<LlamaKvCacheShared>,
    block_table: BlockTable,
    device: Device,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl std::fmt::Debug for LlamaKvCacheHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaKvCacheHandle")
            .field("num_tokens", &self.num_tokens())
            .field("num_layers", &self.num_layers)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}

impl Clone for LlamaKvCacheHandle {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            block_table: self.block_table.clone(),
            device: self.device.clone(),
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        }
    }
}

impl LlamaKvCacheHandle {
    fn new(cache: LlamaCache, info: &ModelInfo, sequence_length: usize) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = sequence_length;

        Self {
            shared: Arc::new(LlamaKvCacheShared {
                cache: Mutex::new(cache),
                sequence_length: AtomicUsize::new(sequence_length),
            }),
            block_table,
            device: info.device.clone(),
            num_layers: info.num_layers,
            num_heads: info.num_heads,
            head_dim: if info.num_heads > 0 {
                info.hidden_size / info.num_heads
            } else {
                0
            },
        }
    }

    fn cache(&self) -> &Mutex<LlamaCache> {
        &self.shared.cache
    }

    fn advance_tokens(&self, delta: usize) {
        self.shared
            .sequence_length
            .fetch_add(delta, Ordering::Relaxed);
    }
}

impl KvCacheHandle for LlamaKvCacheHandle {
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

    fn num_tokens(&self) -> usize {
        self.shared.sequence_length.load(Ordering::Relaxed)
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

    fn key_cache(&self, _layer: usize) -> ferrum_types::Result<Option<TensorRef>> {
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> ferrum_types::Result<Option<TensorRef>> {
        Ok(None)
    }

    fn clone_handle(&self) -> ferrum_types::Result<Arc<dyn KvCacheHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn stats(&self) -> ferrum_interfaces::kv_cache::CacheHandleStats {
        let tokens = self.num_tokens();
        let memory_bytes = tokens
            .saturating_mul(self.num_layers)
            .saturating_mul(self.num_heads.max(1))
            .saturating_mul(self.head_dim.max(1))
            .saturating_mul(2)
            .saturating_mul(2);

        ferrum_interfaces::kv_cache::CacheHandleStats {
            memory_bytes,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: tokens,
            utilization: 0.0,
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        format!("llama-cache-{:p}", Arc::as_ptr(&self.shared))
    }
}

impl CandleModelExecutor {
    /// Create new executor
    pub fn new(model: LlamaModelWrapper, info: ModelInfo) -> Self {
        info!("✅ Created CandleModelExecutor for: {}", info.model_id);

        Self {
            model: Arc::new(model),
            info,
        }
    }

    /// Create Candle tensor from token IDs
    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<Tensor> {
        Tensor::new(token_ids, self.model.device())
            .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))?
            .unsqueeze(0) // Add batch dimension
            .map_err(|e| FerrumError::model(format!("Failed to unsqueeze: {}", e)))
    }

    /// Wrap Candle tensor as TensorRef
    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }

    fn tensor_to_token_ids(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
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
}

#[async_trait]
impl ModelExecutor for CandleModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!(
            "Prefill: batch={}, seq_len={}",
            input.batch_size(),
            input.sequence_length()
        );

        let token_ids = self.tensor_to_token_ids(&input.input_ids)?;
        let input_tensor = self.tokens_to_tensor(&token_ids)?;

        // Forward pass - creates new cache
        let (logits, cache) = self.model.forward_prefill(&input_tensor)?;

        let logits_ref = self.wrap_tensor(logits);

        // Create per-request KV cache handle, so parallel requests do not share state.
        let kv_cache: Arc<dyn KvCacheHandle> =
            Arc::new(LlamaKvCacheHandle::new(cache, &self.info, token_ids.len()));

        Ok(PrefillOutput::new(logits_ref, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        debug!("Decode: batch={}", input.batch_size());

        let token_ids = self.tensor_to_token_ids(&input.input_ids)?;
        let input_tensor = self.tokens_to_tensor(&token_ids)?;

        let cache_handle = input
            .kv_cache
            .as_any()
            .downcast_ref::<LlamaKvCacheHandle>()
            .ok_or_else(|| {
                FerrumError::model("Invalid KV cache handle type for CandleModelExecutor")
            })?;

        // Forward pass with cache
        let logits = {
            let pos = cache_handle.num_tokens();
            let mut cache = cache_handle.cache().lock();
            self.model
                .forward_decode_with_cache(&input_tensor, pos, &mut cache)?
        };

        cache_handle.advance_tokens(token_ids.len());

        let logits_ref = self.wrap_tensor(logits);

        Ok(DecodeOutput::new(logits_ref, input.kv_cache.clone()))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        use ferrum_interfaces::model_executor::{AttentionType, MemoryRequirements};

        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead, AttentionType::GroupedQuery],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2) as u64,
                activation_memory_per_token: 4 * self.info.hidden_size,
                kv_cache_memory_per_token: 2 * self.info.num_layers * self.info.hidden_size,
                overhead_memory: 1024 * 1024 * 1024,
            },
            supported_devices: vec![self.info.device.clone()],
            supported_dtypes: vec![DataType::FP32, DataType::FP16, DataType::BF16],
        }
    }

    fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            state: ferrum_interfaces::model_executor::ExecutorState::Ready,
            is_ready: true,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: 0,
            memory_usage: ferrum_interfaces::model_executor::ExecutorMemoryUsage {
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            last_operation: Some(std::time::Instant::now()),
        }
    }
}

/// Candle model executor that works with token IDs directly
pub struct CandleModelExecutorV2 {
    model: Arc<LlamaModelWrapper>,
    _info: ModelInfo,
    current_cache: Arc<Mutex<Option<LlamaCache>>>,
}

impl CandleModelExecutorV2 {
    pub fn new(model: LlamaModelWrapper, info: ModelInfo) -> Self {
        Self {
            model: Arc::new(model),
            _info: info,
            current_cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Forward pass with token IDs directly
    pub async fn forward_with_tokens(
        &self,
        token_ids: &[u32],
        is_prefill: bool,
    ) -> Result<Vec<f32>> {
        let tensor = Tensor::new(token_ids, self.model.device())
            .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("Failed to unsqueeze: {}", e)))?;

        let logits = if is_prefill {
            let (logits_tensor, cache) = self.model.forward_prefill(&tensor)?;
            *self.current_cache.lock() = Some(cache);
            logits_tensor
        } else {
            let pos = token_ids.len() - 1;
            let mut cache_lock = self.current_cache.lock();
            let cache = cache_lock
                .as_mut()
                .ok_or_else(|| FerrumError::model("No cache - call prefill first"))?;
            self.model.forward_decode_with_cache(&tensor, pos, cache)?
        };

        // Extract logits (cast to F32 if needed for F16/BF16 GPU inference)
        let logits = if logits.dtype() != candle_core::DType::F32 {
            logits
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| FerrumError::model(format!("Cast logits to f32: {}", e)))?
        } else {
            logits
        };
        let logits_vec = match logits.dims().len() {
            1 => logits
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::model(format!("to_vec1 failed: {}", e)))?,
            2 => {
                let batch = logits
                    .to_vec2::<f32>()
                    .map_err(|e| FerrumError::model(format!("to_vec2 failed: {}", e)))?;
                batch.into_iter().next().unwrap_or_default()
            }
            3 => {
                let all = logits
                    .to_vec3::<f32>()
                    .map_err(|e| FerrumError::model(format!("to_vec3 failed: {}", e)))?;
                all.into_iter()
                    .next()
                    .and_then(|seq| seq.into_iter().last())
                    .unwrap_or_default()
            }
            _ => {
                return Err(FerrumError::model(format!(
                    "Unexpected shape: {:?}",
                    logits.dims()
                )))
            }
        };

        Ok(logits_vec)
    }
}

/// Helper to extract logits from Candle tensor safely
pub fn extract_logits_safe(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = if tensor.dtype() != candle_core::DType::F32 {
        &tensor
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| FerrumError::model(format!("Cast logits to f32: {}", e)))?
    } else {
        tensor
    };
    match tensor.dims().len() {
        1 => tensor
            .to_vec1::<f32>()
            .map_err(|e| FerrumError::model(format!("to_vec1 failed: {}", e))),
        2 => {
            let batch = tensor
                .to_vec2::<f32>()
                .map_err(|e| FerrumError::model(format!("to_vec2 failed: {}", e)))?;
            Ok(batch.into_iter().next().unwrap_or_default())
        }
        3 => {
            let all = tensor
                .to_vec3::<f32>()
                .map_err(|e| FerrumError::model(format!("to_vec3 failed: {}", e)))?;
            Ok(all
                .into_iter()
                .next()
                .and_then(|seq| seq.into_iter().last())
                .unwrap_or_default())
        }
        _ => Err(FerrumError::model(format!(
            "Unexpected shape: {:?}",
            tensor.dims()
        ))),
    }
}
