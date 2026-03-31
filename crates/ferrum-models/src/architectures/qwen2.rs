//! Qwen2 architecture — reuses Llama model implementation.
//!
//! Qwen2 and Llama share the same transformer structure:
//! GQA attention (no Q/K norm), SiLU MLP (separate gate/up), RoPE.
//! The only difference is config defaults (rope_theta, etc.).

use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ferrum_types::{FerrumError, Result};
use tracing::{debug, info};

// Re-use Llama's model implementation (identical architecture)
use super::llama;

/// Qwen2 model wrapper — delegates to Llama model internals.
pub struct Qwen2ModelWrapper {
    inner: llama::LlamaModelWrapper,
}

impl Qwen2ModelWrapper {
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Creating Qwen2 model from weights...");

        debug!(
            "Qwen2 config: hidden={}, layers={}, heads={}, kv_heads={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
        );

        // Qwen2 uses same architecture as Llama, just different config defaults
        let inner = llama::LlamaModelWrapper::from_varbuilder(vb, config, device, dtype)?;

        info!("Qwen2 model created successfully");
        Ok(Self { inner })
    }

    pub fn forward_prefill(&self, input_ids: &Tensor, cache_key: &str) -> Result<Tensor> {
        self.inner.forward_prefill(input_ids, cache_key)
    }

    pub fn forward_decode(&self, token_id: &Tensor, pos: usize, cache_key: &str) -> Result<Tensor> {
        self.inner.forward_decode(token_id, pos, cache_key)
    }

    pub fn export_kv_cache(&self, cache_key: &str) -> Option<Vec<(Tensor, Tensor, usize, usize)>> {
        self.inner.export_kv_cache(cache_key)
    }

    pub fn release_cache(&self, cache_key: &str) {
        self.inner.release_cache(cache_key);
    }

    pub fn config(&self) -> &llama::Config {
        self.inner.config()
    }

    pub fn device(&self) -> &CandleDevice {
        self.inner.device()
    }

    pub fn candle_device(&self) -> &CandleDevice {
        self.inner.candle_device()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn set_model_dir(&mut self, dir: std::path::PathBuf) {
        self.inner.set_model_dir(dir);
    }

    #[cfg(feature = "cuda")]
    pub fn create_decode_runner(
        &self,
    ) -> Result<ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner> {
        self.inner.create_decode_runner()
    }

    /// Unwrap into the inner LlamaModelWrapper (for CandleModelExecutor reuse).
    pub fn into_inner(self) -> llama::LlamaModelWrapper {
        self.inner
    }
}
