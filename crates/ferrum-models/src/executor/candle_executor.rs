//! Real model executor using Candle

use async_trait::async_trait;
use candle_core::{DType, Device as CandleDevice, Tensor};
use ferrum_interfaces::{
    model_executor::{DecodeInput, DecodeOutput, ExecutorCapabilities, PrefillInput, PrefillOutput, ExecutorStatus},
    BlockTable, ComputeBackend, KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
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
    
    /// Convert token IDs to Candle tensor
    fn tokens_to_tensor(&self, tokens: &[ferrum_types::TokenId]) -> Result<Tensor> {
        let token_u32s: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
        
        Tensor::new(&token_u32s[..], self.model.device())
            .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))
    }
    
    /// Extract logits to Vec<f32>
    fn tensor_to_logits(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .to_vec1()
            .map_err(|e| FerrumError::model(format!("Failed to extract logits: {}", e)))
    }
    
    /// Wrap Candle tensor as TensorRef
    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }
    
    /// Extract Candle tensor from TensorRef  
    fn unwrap_tensor(&self, tensor_ref: &TensorRef) -> Result<Tensor> {
        // For MVP: assume all TensorRefs are CandleTensorWrappers
        // Use unsafe to extract - this is OK because we control tensor creation
        unsafe {
            let raw = Arc::as_ptr(tensor_ref) as *const CandleTensorWrapper;
            Ok((*raw).inner().clone())
        }
    }
}

#[async_trait]
impl ModelExecutor for CandleModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!("Prefill: batch={}, seq_len={}", input.batch_size(), input.sequence_length());
        
        // Extract Candle tensor from TensorRef
        let input_tensor = self.unwrap_tensor(&input.input_ids)?;
        
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
        
        // Extract Candle tensor from TensorRef
        let input_tensor = self.unwrap_tensor(&input.input_ids)?;
        
        // Get current position from KV cache
        let pos = input.kv_cache.num_tokens();
        
        // Forward pass with cache
        let logits = {
            let mut cache_lock = self.current_cache.lock();
            let cache = cache_lock.as_mut().ok_or_else(|| {
                FerrumError::model("No cache available - must call prefill first")
            })?;
            
            self.model.forward_decode_with_cache(&input_tensor, pos, cache)?
        };
        
        let logits_ref = self.wrap_tensor(logits);
        
        Ok(DecodeOutput::new(logits_ref, input.kv_cache.clone()))
    }
    
    fn capabilities(&self) -> ExecutorCapabilities {
        use ferrum_interfaces::model_executor::{AttentionType, ExecutorMemoryUsage, MemoryRequirements};
        
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
                parameter_memory: (self.info.num_parameters * 2) as u64, // FP16: 2 bytes per param
                activation_memory_per_token: 4 * self.info.hidden_size, // Rough estimate
                kv_cache_memory_per_token: 2 * self.info.num_layers * self.info.hidden_size, // K+V per layer
                overhead_memory: 1024 * 1024 * 1024, // 1GB overhead
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

