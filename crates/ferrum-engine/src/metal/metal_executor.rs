//! Metal-accelerated model executor
//!
//! Provides a ModelExecutor implementation that uses Metal LLaMA model
//! with custom Metal kernels for GPU-accelerated inference.

use crate::metal::{MetalContext, MetalLlamaConfig, MetalLlamaModel};
use async_trait::async_trait;
use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor};
use ferrum_interfaces::kv_cache::CacheHandleStats;
use ferrum_interfaces::model_executor::{
    AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
    ExecutorState, ExecutorStatus, MemoryRequirements,
};
use ferrum_interfaces::{KvCacheHandle, ModelExecutor, PrefillInput, PrefillOutput, TensorRef};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
use half::f16;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Simple tensor implementation for Metal executor outputs
#[derive(Debug)]
struct SimpleTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl SimpleTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl ferrum_interfaces::TensorLike for SimpleTensor {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        DataType::FP32
    }

    fn device(&self) -> Device {
        Device::Metal
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn view(&self, start: &[usize], end: &[usize]) -> Result<TensorRef> {
        // Very limited view support: slice contiguous range
        if start.len() != end.len() || start.len() != self.shape.len() {
            return Err(FerrumError::backend("Invalid view dimensions"));
        }
        // Compute flat offsets
        let mut stride = 1usize;
        let mut offset = 0usize;
        for dim in (0..self.shape.len()).rev() {
            offset += start[dim] * stride;
            stride *= self.shape[dim];
        }
        let mut count = 1usize;
        for (s, e) in start.iter().zip(end.iter()) {
            count *= e - s;
        }
        let end_offset = offset + count;
        let data = self
            .data
            .get(offset..end_offset)
            .ok_or_else(|| FerrumError::backend("View slice out of range"))?
            .to_vec();
        Ok(Arc::new(SimpleTensor::new(
            data,
            end.iter().zip(start.iter()).map(|(e, s)| e - s).collect(),
        )))
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorRef> {
        let new_numel: usize = shape.iter().product();
        if new_numel != self.numel() {
            return Err(FerrumError::backend("Reshape numel mismatch"));
        }
        Ok(Arc::new(SimpleTensor::new(
            self.data.clone(),
            shape.to_vec(),
        )))
    }

    fn to_cpu(&self) -> Result<TensorRef> {
        Ok(Arc::new(SimpleTensor::new(
            self.data.clone(),
            self.shape.clone(),
        )))
    }

    fn to_device(&self, _device: &Device) -> Result<TensorRef> {
        Ok(Arc::new(SimpleTensor::new(
            self.data.clone(),
            self.shape.clone(),
        )))
    }

    fn to_dtype(&self, dtype: DataType) -> Result<TensorRef> {
        match dtype {
            DataType::FP32 | DataType::FP16 | DataType::BF16 | DataType::FP8 => Ok(Arc::new(
                SimpleTensor::new(self.data.clone(), self.shape.clone()),
            )),
            _ => Err(FerrumError::backend("Unsupported dtype conversion")),
        }
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        Ok(self.data.iter().map(|x| *x as u32).collect())
    }

    fn argmax_last_dim_u32(&self) -> Result<u32> {
        let (idx, _) = self
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| FerrumError::backend("Empty tensor"))?;
        Ok(idx as u32)
    }
}

/// Dummy KV cache handle for Metal executor (temporary)
#[derive(Debug, Clone)]
struct DummyKvCache {
    block_table: ferrum_interfaces::BlockTable,
}

impl DummyKvCache {
    fn new(block_size: usize) -> Self {
        Self {
            block_table: ferrum_interfaces::BlockTable::new(block_size),
        }
    }
}

impl KvCacheHandle for DummyKvCache {
    fn block_table(&self) -> &ferrum_interfaces::BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut ferrum_interfaces::BlockTable {
        &mut self.block_table
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Device {
        Device::Metal
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
        "metal_dummy_cache".to_string()
    }
}

/// Metal-accelerated model executor
pub struct MetalLlamaExecutor {
    model: MetalLlamaModel,
    info: ModelInfo,
    device: CandleDevice,
    status: ExecutorStatus,
}

impl MetalLlamaExecutor {
    /// Create a new Metal LLaMA executor
    pub fn new(model: MetalLlamaModel, model_info: ModelInfo, device: CandleDevice) -> Self {
        info!(
            "Created MetalLlamaExecutor for model: {}",
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

    /// Load a Metal LLaMA model from path
    pub async fn from_path(
        model_path: &str,
        model_def: &ferrum_models::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Loading Metal LLaMA model from: {}", model_path);

        // Initialize Metal context
        let metal_context = {
            let mut ctx = MetalContext::new()?;
            ctx.load_shader_library()?;
            Some(Arc::new(ctx))
        };

        // Load weights
        let loader = ferrum_models::SafeTensorsLoader::new(model_path);
        let vb = loader.load_varbuilder(&device, dtype)?;

        // Create Metal LLaMA config
        let config = MetalLlamaConfig {
            vocab_size: model_def.vocab_size,
            hidden_size: model_def.hidden_size,
            intermediate_size: model_def.intermediate_size,
            num_hidden_layers: model_def.num_hidden_layers,
            num_attention_heads: model_def.num_attention_heads,
            num_key_value_heads: model_def
                .num_key_value_heads
                .unwrap_or(model_def.num_attention_heads),
            max_position_embeddings: model_def.max_position_embeddings,
            rms_norm_eps: model_def.norm_eps,
            rope_theta: model_def.rope_theta.unwrap_or(10000.0) as f32,
        };

        // Load model
        let model = MetalLlamaModel::load(vb, config, &device, metal_context)?;

        let model_info = model_def.to_model_info(model_path.to_string());

        Ok(Self::new(model, model_info, device))
    }

    /// Extract token IDs from TensorRef
    fn extract_token_ids(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        if let Ok(v) = tensor.to_vec_u32() {
            return Ok(v);
        }
        if let Ok(vf) = tensor.to_vec_f32() {
            return Ok(vf.into_iter().map(|x| x as u32).collect());
        }
        Err(FerrumError::backend(
            "Unable to extract token ids from tensor",
        ))
    }
}

#[async_trait]
impl ModelExecutor for MetalLlamaExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let token_ids = self.extract_token_ids(&input.input_ids)?;
        let seq_len = token_ids.len();

        debug!("Metal prefill: {} tokens", seq_len);

        // Convert token IDs to tensor
        let input_tensor = Tensor::from_vec(token_ids.clone(), (1, seq_len), &self.device)
            .map_err(|e| FerrumError::internal(format!("Failed to create input tensor: {}", e)))?;

        // Run forward pass (logits may be FP16); convert to FP32 for sampling
        let logits = self.model.forward(&input_tensor, 0)?;
        debug!(
            "prefill logits shape={:?}, dtype={:?}, device={:?}",
            logits.dims(),
            logits.dtype(),
            logits.device()
        );

        // Get logits - shape should be [1, seq_len, vocab_size]
        let logits_dims = logits.dims();
        let vocab_size = *logits_dims.last().unwrap_or(&0);

        // Get last token logits for generation
        let logits_cpu = logits
            .to_device(&CandleDevice::Cpu)
            .map_err(|e| FerrumError::internal(format!("To CPU failed: {}", e)))?;

        let last_logits = logits_cpu
            .i((0, seq_len - 1, ..))
            .map_err(|e| FerrumError::internal(format!("Index error: {}", e)))?;

        let dtype = last_logits.dtype();
        debug!("prefill last_logits dtype={:?}", dtype);

        // Convert to output format with explicit dtype handling
        let logits_vec: Vec<f32> = if dtype == DType::F32 {
            last_logits
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("To vec f32 error: {}", e)))?
        } else if dtype == DType::F16 {
            let v16: Vec<f16> = last_logits
                .to_vec1()
                .map_err(|e| FerrumError::internal(format!("To vec f16 error: {}", e)))?;
            warn!("prefill logits were FP16, converting f16->f32");
            v16.iter().map(|v| f32::from(*v)).collect()
        } else {
            warn!(
                "prefill logits unexpected dtype {:?}, force to f32 via cast",
                dtype
            );
            last_logits
                .to_dtype(DType::F32)
                .unwrap_or(last_logits)
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("Fallback to vec f32 error: {}", e)))?
        };

        // Create output tensor
        let output_tensor: TensorRef = Arc::new(SimpleTensor::new(logits_vec, vec![1, vocab_size]));

        // Create dummy KV cache handle
        let kv_cache: Arc<dyn KvCacheHandle> = Arc::new(DummyKvCache::new(16));

        Ok(PrefillOutput::new(output_tensor, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let token_ids = self.extract_token_ids(&input.input_ids)?;
        let position = input.kv_cache.num_tokens();

        debug!("Metal decode: position {}", position);

        // Convert token ID to tensor
        let token_tensor = Tensor::from_vec(token_ids, (1, 1), &self.device)
            .map_err(|e| FerrumError::internal(format!("Failed to create token tensor: {}", e)))?;

        // Run forward pass at the current position and convert to FP32
        let logits = self.model.forward(&token_tensor, position)?;
        debug!(
            "decode logits shape={:?}, dtype={:?}, device={:?}",
            logits.dims(),
            logits.dtype(),
            logits.device()
        );

        // Get logits
        let logits_cpu = logits
            .to_device(&CandleDevice::Cpu)
            .map_err(|e| FerrumError::internal(format!("To CPU failed: {}", e)))?;

        let last_logits = logits_cpu
            .i((0, 0, ..))
            .map_err(|e| FerrumError::internal(format!("Index error: {}", e)))?;

        let dtype = last_logits.dtype();
        debug!("decode last_logits dtype={:?}", dtype);

        let logits_vec: Vec<f32> = if dtype == DType::F32 {
            last_logits
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("To vec f32 error: {}", e)))?
        } else if dtype == DType::F16 {
            let v16: Vec<f16> = last_logits
                .to_vec1()
                .map_err(|e| FerrumError::internal(format!("To vec f16 error: {}", e)))?;
            warn!("decode logits were FP16, converting f16->f32");
            v16.iter().map(|v| f32::from(*v)).collect()
        } else {
            warn!(
                "decode logits unexpected dtype {:?}, force to f32 via cast",
                dtype
            );
            last_logits
                .to_dtype(DType::F32)
                .unwrap_or(last_logits)
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("Fallback to vec f32 error: {}", e)))?
        };

        let vocab_size = logits_vec.len();
        let output_tensor: TensorRef = Arc::new(SimpleTensor::new(logits_vec, vec![1, vocab_size]));

        // Clone the input kv_cache for output (in a real implementation, this would be updated)
        let kv_cache = input.kv_cache.clone();

        Ok(DecodeOutput::new(output_tensor, kv_cache))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.model.config().max_position_embeddings,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32, DataType::FP16],
            supported_devices: vec![Device::Metal],
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
