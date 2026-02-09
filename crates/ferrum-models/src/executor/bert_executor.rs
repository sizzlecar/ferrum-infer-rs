//! BERT Model Executor for embeddings
//!
//! BERT is an encoder model used for generating text embeddings.
//! Unlike decoder models (LLaMA, Qwen), it doesn't generate tokens.

use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
        ExecutorState, ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    BlockTable, CacheHandleStats, KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
use tracing::{debug, info};

use crate::architectures::bert::BertModelWrapper;
use crate::tensor_wrapper::CandleTensorWrapper;

/// BERT Executor for embedding tasks
pub struct BertModelExecutor {
    model: BertModelWrapper,
    info: ModelInfo,
    device: CandleDevice,
    status: ExecutorStatus,
}

impl BertModelExecutor {
    /// Create a new BERT executor
    pub fn new(model: BertModelWrapper, model_info: ModelInfo, device: CandleDevice) -> Self {
        info!(
            "Created BertModelExecutor for model: {}",
            model_info.model_id
        );

        let status = ExecutorStatus {
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
            last_operation: None,
        };

        Self {
            model,
            info: model_info,
            device,
            status,
        }
    }

    /// Load BERT executor from path
    pub async fn from_path(
        model_path: &str,
        model_def: &crate::definition::ModelDefinition,
        device: CandleDevice,
    ) -> Result<Self> {
        info!("Loading BERT model from: {}", model_path);

        let path = std::path::Path::new(model_path);

        // Find safetensors file
        let safetensors_path = if path.join("model.safetensors").exists() {
            path.join("model.safetensors")
        } else {
            // Look for any .safetensors file
            std::fs::read_dir(path)
                .map_err(|e| FerrumError::model(format!("Failed to read model dir: {}", e)))?
                .filter_map(|e| e.ok())
                .find(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "safetensors")
                })
                .map(|e| e.path())
                .ok_or_else(|| FerrumError::model("No safetensors file found"))?
        };

        info!("Loading weights from: {:?}", safetensors_path);

        // Use F32 for BERT (better compatibility)
        let dtype = DType::F32;

        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, &device)
                .map_err(|e| FerrumError::model(format!("Failed to load weights: {}", e)))?
        };

        // Create model from config.json
        // Note: Some models have "bert." prefix, some don't.
        // sentence-transformers models typically don't have the prefix.
        let config_path = path.join("config.json");
        let model = BertModelWrapper::from_config_json(vb, &config_path, device.clone(), dtype)?;

        // Create model info
        let model_info = model_def.to_model_info(model_path.to_string());

        Ok(Self::new(model, model_info, device))
    }

    /// Get embeddings for input tokens
    pub fn get_embeddings(&self, input_ids: &[u32]) -> Result<Tensor> {
        let seq_len = input_ids.len();

        // Create input tensor
        let input_tensor = Tensor::from_vec(
            input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            (1, seq_len),
            &self.device,
        )
        .map_err(|e| FerrumError::model(format!("Failed to create input tensor: {}", e)))?;

        // Create token type ids (all zeros for single sentence)
        let token_type_ids = Tensor::zeros((1, seq_len), DType::I64, &self.device)
            .map_err(|e| FerrumError::model(format!("Failed to create token type ids: {}", e)))?;

        // Get sentence embedding
        self.model
            .get_sentence_embedding(&input_tensor, &token_type_ids, None)
    }

    /// Get model reference
    pub fn model(&self) -> &BertModelWrapper {
        &self.model
    }
}

/// Dummy KV cache for BERT (not used but required by interface)
#[derive(Debug, Clone)]
struct DummyBertCache;

impl KvCacheHandle for DummyBertCache {
    fn block_table(&self) -> &BlockTable {
        static EMPTY: std::sync::OnceLock<BlockTable> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| BlockTable::new(16))
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        unimplemented!("BERT does not use KV cache")
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn num_heads(&self) -> usize {
        0
    }

    fn head_dim(&self) -> usize {
        0
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
            blocks_allocated: 0,
            tokens_stored: 0,
            utilization: 0.0,
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        "bert_dummy_cache".to_string()
    }
}

#[async_trait]
impl ModelExecutor for BertModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// For BERT, prefill returns the embeddings (not logits)
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let token_ids: Vec<u32> = if let Ok(v) = input.input_ids.to_vec_u32() {
            v
        } else if let Ok(vf) = input.input_ids.to_vec_f32() {
            vf.into_iter().map(|x| x as u32).collect()
        } else {
            return Err(FerrumError::backend("Unable to extract token ids"));
        };

        debug!("BERT prefill: {} tokens", token_ids.len());

        let embeddings = self.get_embeddings(&token_ids)?;

        // Wrap as TensorRef
        let output_tensor: TensorRef = Arc::new(CandleTensorWrapper::new(embeddings));
        let kv_cache: Arc<dyn KvCacheHandle> = Arc::new(DummyBertCache);

        Ok(PrefillOutput::new(output_tensor, kv_cache))
    }

    /// BERT doesn't support decode (it's an encoder model)
    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(FerrumError::model(
            "BERT is an encoder model and does not support token generation. Use prefill() to get embeddings.",
        ))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 32,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: true,
            supports_continuous_batching: false,
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
        self.status.clone()
    }
}
