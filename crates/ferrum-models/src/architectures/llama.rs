//! Llama architecture using Candle's built-in implementation

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as candle_llama;
use ferrum_types::{FerrumError, Result};
use tracing::{debug, info};

/// Llama model wrapper
pub struct LlamaModelWrapper {
    model: candle_llama::Llama,
    config: candle_llama::Config,
    device: CandleDevice,
    dtype: DType,
    /// Model directory for weight re-loading (CUDA runner weight extraction)
    model_dir: Option<std::path::PathBuf>,
}

impl LlamaModelWrapper {
    /// Create from VarBuilder and config
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("🔨 Creating Llama model from weights...");

        // Build Candle's Llama config
        let candle_config = candle_llama::Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            rms_norm_eps: config.norm_eps,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            max_position_embeddings: config.max_position_embeddings,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            tie_word_embeddings: false,
            use_flash_attn: false,
        };

        debug!(
            "Llama config: hidden={}, layers={}, heads={}, kv_heads={}",
            candle_config.hidden_size,
            candle_config.num_hidden_layers,
            candle_config.num_attention_heads,
            candle_config.num_key_value_heads
        );

        // Load model
        let model = candle_llama::Llama::load(vb, &candle_config)
            .map_err(|e| FerrumError::model(format!("Failed to load Llama model: {}", e)))?;

        info!("✅ Llama model created successfully");

        Ok(Self {
            model,
            config: candle_config,
            device,
            dtype,
            model_dir: None,
        })
    }

    /// Forward pass for prefill (full sequence) - creates new cache
    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<(Tensor, candle_llama::Cache)> {
        // Create fresh cache for each request
        let mut cache = candle_llama::Cache::new(true, self.dtype, &self.config, &self.device)
            .map_err(|e| FerrumError::model(format!("Failed to create cache: {}", e)))?;

        let logits = self
            .model
            .forward(input_ids, 0, &mut cache)
            .map_err(|e| FerrumError::model(format!("Prefill forward failed: {}", e)))?;

        Ok((logits, cache))
    }

    /// Forward pass for decode (single token) with existing cache
    pub fn forward_decode_with_cache(
        &self,
        token_id: &Tensor,
        pos: usize,
        cache: &mut candle_llama::Cache,
    ) -> Result<Tensor> {
        self.model
            .forward(token_id, pos, cache)
            .map_err(|e| FerrumError::model(format!("Decode forward failed: {}", e)))
    }

    /// Get device
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get config
    pub fn config(&self) -> &candle_llama::Config {
        &self.config
    }

    /// Set model directory (for CUDA runner weight loading)
    pub fn set_model_dir(&mut self, dir: std::path::PathBuf) {
        self.model_dir = Some(dir);
    }

    /// Create CUDA decode runner by loading weights from safetensors.
    #[cfg(feature = "cuda")]
    pub fn create_decode_runner(
        &self,
    ) -> Result<ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner> {
        use crate::loader::runner_weights::{self, WeightConfig};

        let model_dir = self
            .model_dir
            .as_ref()
            .ok_or_else(|| FerrumError::model("model_dir not set for CUDA runner"))?;

        // Load safetensors VarBuilder
        let loader = crate::loader::SafeTensorsLoader::new(model_dir);
        let vb = loader.load_varbuilder(&self.device, self.dtype)?;

        let cfg = WeightConfig {
            num_hidden_layers: self.config.num_hidden_layers,
            hidden_size: self.config.hidden_size,
            intermediate_size: self.config.intermediate_size,
            num_attention_heads: self.config.num_attention_heads,
            num_kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.hidden_size / self.config.num_attention_heads,
            vocab_size: self.config.vocab_size,
            max_seq_len: self.config.max_position_embeddings,
            rope_theta: self.config.rope_theta as f64,
            has_qk_norm: false,
            qkv_fused: false,
            gate_up_fused: false,
        };

        let (weights, dims, stream) = runner_weights::load_runner_weights(&vb, &cfg, &self.device)?;

        let cuda_device = self
            .device
            .as_cuda_device()
            .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;

        ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner::new(
            weights,
            dims,
            cuda_device.clone(),
            stream,
        )
        .map_err(|e| FerrumError::model(format!("CudaDecodeRunner: {e}")))
    }
}
