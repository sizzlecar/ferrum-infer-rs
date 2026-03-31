//! Mistral architecture — reuses Llama model implementation.
//!
//! Mistral and Llama share identical transformer structure:
//! GQA attention (no Q/K norm), SiLU MLP (separate gate/up), RoPE.

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ferrum_types::{FerrumError, Result};
use tracing::{debug, info};

use super::llama;

/// Mistral model wrapper — delegates to Llama model internals.
pub struct MistralModelWrapper {
    inner: llama::LlamaModelWrapper,
}

impl MistralModelWrapper {
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Creating Mistral model from weights...");
        debug!(
            "Mistral config: hidden={}, layers={}, heads={}, kv_heads={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
        );
        let inner = llama::LlamaModelWrapper::from_varbuilder(vb, config, device, dtype)?;
        info!("Mistral model created successfully");
        Ok(Self { inner })
    }

    pub fn into_inner(self) -> llama::LlamaModelWrapper {
        self.inner
    }

    pub fn config(&self) -> &llama::Config {
        self.inner.config()
    }
}
