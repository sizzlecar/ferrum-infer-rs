//! CLIP Model Executor for multimodal embeddings.
//!
//! Supports both text and image embedding via unified interface.
//! Text goes through CLIP text encoder, images through vision encoder.

use std::collections::HashMap;
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
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, ModelType, Result};
use tracing::info;

use super::common;
use crate::architectures::clip::ClipModelWrapper;
use crate::image_processor::ClipImageProcessor;
use crate::tensor_wrapper::CandleTensorWrapper;

/// CLIP executor for text and image embeddings.
pub struct ClipModelExecutor {
    model: ClipModelWrapper,
    image_processor: ClipImageProcessor,
    info: ModelInfo,
}

impl ClipModelExecutor {
    pub fn new(model: ClipModelWrapper, info: ModelInfo) -> Self {
        let image_processor = ClipImageProcessor::new(model.image_size());
        info!(
            "Created ClipModelExecutor: {} (dim={}, image_size={})",
            info.model_id,
            model.projection_dim(),
            model.image_size()
        );
        Self {
            model,
            image_processor,
            info,
        }
    }

    /// Load from model directory (config.json + safetensors).
    pub fn from_path(model_path: &str, device: CandleDevice, dtype: DType) -> Result<Self> {
        let dir = std::path::Path::new(model_path);
        let config_path = dir.join("config.json");

        let safetensors: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| FerrumError::model(format!("read dir: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        if safetensors.is_empty() {
            return Err(FerrumError::model("No safetensors files found"));
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors, dtype, &device)
                .map_err(|e| FerrumError::model(format!("load weights: {e}")))?
        };

        let model = ClipModelWrapper::from_config_json(vb, &config_path, device, dtype)?;

        let info = ModelInfo {
            model_id: ferrum_types::ModelId(model_path.to_string()),
            model_type: ModelType::Clip,
            hidden_size: model.projection_dim(),
            vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            num_parameters: 0,
            max_sequence_length: 77,
            device: Device::CPU,
            dtype: DataType::FP32,
            version: None,
            license: None,
            metadata: HashMap::new(),
        };

        Ok(Self::new(model, info))
    }

    /// Embed text tokens → L2-normalized vector.
    pub fn embed_text(&self, input_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(input_ids, self.model.device())
            .map_err(|e| FerrumError::model(format!("tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
        self.model.get_text_features(&ids)
    }

    /// Embed image from file path → L2-normalized vector.
    pub fn embed_image_path(&self, path: &str) -> Result<Tensor> {
        let pixel_values = self
            .image_processor
            .process_path(path, self.model.device())?;
        self.model.get_image_features(&pixel_values)
    }

    /// Embed image from base64 data → L2-normalized vector.
    pub fn embed_image_base64(&self, data: &str) -> Result<Tensor> {
        let pixel_values = self
            .image_processor
            .process_base64(data, self.model.device())?;
        self.model.get_image_features(&pixel_values)
    }

    pub fn projection_dim(&self) -> usize {
        self.model.projection_dim()
    }
}

// Dummy KV cache for encoder-only CLIP (same pattern as BERT).
#[derive(Clone, Debug)]
struct DummyClipCache;

impl KvCacheHandle for DummyClipCache {
    fn block_table(&self) -> &BlockTable {
        static EMPTY: std::sync::OnceLock<BlockTable> = std::sync::OnceLock::new();
        EMPTY.get_or_init(|| BlockTable::new(16))
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        unimplemented!("CLIP does not use KV cache")
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
        "clip_dummy_cache".to_string()
    }
}

#[async_trait]
impl ModelExecutor for ClipModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let tokens: Vec<u32> = input
            .input_ids
            .as_any()
            .downcast_ref::<CandleTensorWrapper>()
            .and_then(|w| w.inner().flatten_all().and_then(|t| t.to_vec1()).ok())
            .unwrap_or_default();

        if tokens.is_empty() {
            return Err(FerrumError::model("Empty input"));
        }

        let embedding = self.embed_text(&tokens)?;
        let embedding = embedding
            .unsqueeze(1)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
        let tensor_ref: TensorRef = Arc::new(CandleTensorWrapper::new(embedding));
        let cache: Arc<dyn KvCacheHandle> = Arc::new(DummyClipCache);

        Ok(PrefillOutput::new(tensor_ref, cache))
    }

    async fn decode(&self, _input: &DecodeInput) -> Result<DecodeOutput> {
        Err(FerrumError::model(
            "CLIP is an encoder model — decode not supported",
        ))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 32,
            max_sequence_length: 77,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: 600 * 1024 * 1024,
                activation_memory_per_token: 0,
                kv_cache_memory_per_token: 0,
                overhead_memory: 0,
            },
        }
    }

    fn release_cache(&self, _cache_id: &str) {}

    fn status(&self) -> ExecutorStatus {
        common::default_executor_status()
    }
}
