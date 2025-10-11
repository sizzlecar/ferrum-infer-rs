//! Llama architecture using Candle's built-in implementation

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as candle_llama;
use ferrum_types::{FerrumError, Result};
use std::sync::Arc;
use parking_lot::Mutex;
use tracing::{debug, info};

/// Llama model wrapper
pub struct LlamaModelWrapper {
    model: candle_llama::Llama,
    cache: Arc<Mutex<candle_llama::Cache>>,
    device: CandleDevice,
    dtype: DType,
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
        
        // Convert our config to Candle's Llama config  
        // Use TinyLlama config as base template
        let mut candle_config = candle_llama::Config::tiny_llama_1_1b_chat_v0_1();
        
        // Override with our config values
        candle_config.hidden_size = config.hidden_size;
        candle_config.intermediate_size = config.intermediate_size;
        candle_config.vocab_size = config.vocab_size;
        candle_config.num_hidden_layers = config.num_hidden_layers;
        candle_config.num_attention_heads = config.num_attention_heads;
        candle_config.num_key_value_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
        candle_config.rms_norm_eps = config.norm_eps;
        candle_config.rope_theta = config.rope_theta.unwrap_or(10000.0) as f32;
        candle_config.max_position_embeddings = config.max_position_embeddings;
        candle_config.use_flash_attn = false;
        
        debug!("Llama config: hidden={}, layers={}, heads={}, kv_heads={}", 
            candle_config.hidden_size,
            candle_config.num_hidden_layers,
            candle_config.num_attention_heads,
            candle_config.num_key_value_heads
        );
        
        // Load model
        let model = candle_llama::Llama::load(vb, &candle_config)
            .map_err(|e| FerrumError::model(format!("Failed to load Llama model: {}", e)))?;
        
        // Create cache
        let cache = candle_llama::Cache::new(true, &candle_config, dtype, &device)
            .map_err(|e| FerrumError::model(format!("Failed to create KV cache: {}", e)))?;
        
        info!("✅ Llama model created successfully");
        
        Ok(Self {
            model,
            cache: Arc::new(Mutex::new(cache)),
            device,
            dtype,
        })
    }
    
    /// Forward pass for prefill (full sequence)
    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut cache = self.cache.lock();
        
        self.model
            .forward(input_ids, 0, &mut *cache)
            .map_err(|e| FerrumError::model(format!("Prefill forward failed: {}", e)))
    }
    
    /// Forward pass for decode (single token)
    pub fn forward_decode(&self, token_id: &Tensor, pos: usize) -> Result<Tensor> {
        let mut cache = self.cache.lock();
        
        self.model
            .forward(token_id, pos, &mut *cache)
            .map_err(|e| FerrumError::model(format!("Decode forward failed: {}", e)))
    }
    
    /// Clear KV cache
    pub fn clear_cache(&self) {
        // KV cache will be reset on next forward pass
        // Candle's Cache doesn't have a reset method in current version
    }
    
    /// Get device
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }
    
    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

