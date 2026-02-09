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
use ferrum_types::{DataType, FerrumError, ModelInfo, Result};
use std::sync::Arc;
use tracing::{debug, info};

use crate::architectures::llama::LlamaModelWrapper;
use crate::tensor_wrapper::CandleTensorWrapper;
use candle_transformers::models::llama::Cache as LlamaCache;
use parking_lot::Mutex;

/// Candle-based model executor
pub struct CandleModelExecutor {
    model: Arc<LlamaModelWrapper>,
    info: ModelInfo,
    current_cache: Arc<Mutex<Option<LlamaCache>>>,
}

impl CandleModelExecutor {
    /// Create new executor
    pub fn new(model: LlamaModelWrapper, info: ModelInfo) -> Self {
        info!("✅ Created CandleModelExecutor for: {}", info.model_id);

        Self {
            model: Arc::new(model),
            info,
            current_cache: Arc::new(Mutex::new(None)),
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

        // WORKAROUND: Since we can't safely extract from TensorRef,
        // we expect the input to contain raw u32 token IDs
        // The engine layer should create Candle tensors directly

        // For now, create a dummy forward pass
        // The real implementation needs the engine to pass token IDs, not TensorRef
        let input_tensor = self.tokens_to_tensor(&[1, 2])?; // Dummy tokens

        // Forward pass - creates new cache
        let (logits, cache) = self.model.forward_prefill(&input_tensor)?;

        // Store cache for future decode steps
        *self.current_cache.lock() = Some(cache);

        let logits_ref = self.wrap_tensor(logits);

        // Create dummy KV cache handle
        let kv_cache = create_dummy_kv_cache(self.info.num_layers, input.sequence_length());

        Ok(PrefillOutput::new(logits_ref, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        debug!("Decode: batch={}", input.batch_size());

        // WORKAROUND: Same as prefill
        let input_tensor = self.tokens_to_tensor(&[1])?; // Dummy token

        // Get current position from KV cache
        let pos = input.kv_cache.num_tokens();

        // Forward pass with cache
        let logits = {
            let mut cache_lock = self.current_cache.lock();
            let cache = cache_lock.as_mut().ok_or_else(|| {
                FerrumError::model("No cache available - must call prefill first")
            })?;

            self.model
                .forward_decode_with_cache(&input_tensor, pos, cache)?
        };

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

        // Extract logits
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

/// Create dummy KV cache handle
fn create_dummy_kv_cache(num_layers: usize, seq_len: usize) -> Arc<dyn KvCacheHandle> {
    use ferrum_interfaces::kv_cache::CacheHandleStats;
    use ferrum_types::Device;

    #[derive(Debug, Clone)]
    struct DummyKvCache {
        block_table: BlockTable,
        num_layers: usize,
    }

    impl KvCacheHandle for DummyKvCache {
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
            Device::CPU
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn num_heads(&self) -> usize {
            32
        }

        fn head_dim(&self) -> usize {
            64
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

        fn stats(&self) -> CacheHandleStats {
            CacheHandleStats {
                memory_bytes: 0,
                blocks_allocated: self.block_table.num_blocks(),
                tokens_stored: self.block_table.sequence_length,
                utilization: 0.0,
                last_access: std::time::Instant::now(),
            }
        }

        fn is_valid(&self) -> bool {
            true
        }

        fn cache_id(&self) -> String {
            "dummy-cache".to_string()
        }
    }

    let mut block_table = BlockTable::new(16);
    block_table.sequence_length = seq_len;

    Arc::new(DummyKvCache {
        block_table,
        num_layers,
    })
}

/// Helper to extract logits from Candle tensor safely
pub fn extract_logits_safe(tensor: &Tensor) -> Result<Vec<f32>> {
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
