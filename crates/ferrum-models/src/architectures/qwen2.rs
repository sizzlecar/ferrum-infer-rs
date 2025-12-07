//! Qwen2 architecture using Candle's built-in implementation

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen2 as candle_qwen2;
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use tracing::{debug, info};

/// Qwen2 model wrapper
pub struct Qwen2ModelWrapper {
    model: Mutex<candle_qwen2::ModelForCausalLM>,
    config: candle_qwen2::Config,
    device: CandleDevice,
    dtype: DType,
}

impl Qwen2ModelWrapper {
    /// Create from VarBuilder and config
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("🔨 Creating Qwen2 model from weights...");

        // Build Candle's Qwen2 config
        // Use default values from Qwen2 examples
        let candle_config = candle_qwen2::Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta.unwrap_or(1000000.0),
            rms_norm_eps: config.norm_eps,
            tie_word_embeddings: false,
            sliding_window: 32768, // Default sliding window
            max_window_layers: config.num_hidden_layers,
            use_sliding_window: false,
            hidden_act: Activation::Silu, // Default activation
        };

        debug!(
            "Qwen2 config: hidden={}, layers={}, heads={}, kv_heads={}",
            candle_config.hidden_size,
            candle_config.num_hidden_layers,
            candle_config.num_attention_heads,
            candle_config.num_key_value_heads
        );

        // Load model - Qwen2 uses new() not load()
        let model = candle_qwen2::ModelForCausalLM::new(&candle_config, vb)
            .map_err(|e| FerrumError::model(format!("Failed to create Qwen2 model: {}", e)))?;

        info!("✅ Qwen2 model created successfully");

        Ok(Self {
            model: Mutex::new(model),
            config: candle_config,
            device,
            dtype,
        })
    }

    /// Forward pass for prefill (full sequence)
    /// Note: Qwen2 manages cache internally
    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut model = self.model.lock();

        let logits = model
            .forward(input_ids, 0)
            .map_err(|e| FerrumError::model(format!("Prefill forward failed: {}", e)))?;

        // Debug: log top logits to compare with Metal implementation
        if let Ok(flat) = logits.flatten_all() {
            if let Ok(vals) = flat.to_vec1::<f32>() {
                let mut indexed: Vec<(usize, f32)> =
                    vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_k = 10.min(indexed.len());
                info!("CPU Qwen2 Top {} logits: {:?}", top_k, &indexed[..top_k]);
                // Check specific tokens
                if vals.len() > 29 {
                    info!(
                        "CPU tokens: 17='2': {:.4}, 19='4': {:.4}, 28='=': {:.4}",
                        vals[17], vals[19], vals[28]
                    );
                }
            }
        }

        Ok(logits)
    }

    /// Forward pass for decode (single token)
    /// Note: Qwen2 manages cache internally
    pub fn forward_decode(&self, token_id: &Tensor, pos: usize) -> Result<Tensor> {
        let mut model = self.model.lock();

        model
            .forward(token_id, pos)
            .map_err(|e| FerrumError::model(format!("Decode forward failed: {}", e)))
    }

    /// Reset internal KV cache (for new requests)
    pub fn reset_cache(&self) -> Result<()> {
        self.model.lock().clear_kv_cache();
        Ok(())
    }

    /// Get Candle config reference
    pub fn config(&self) -> &candle_qwen2::Config {
        &self.config
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    /// Get raw Candle device
    pub fn candle_device(&self) -> &CandleDevice {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
