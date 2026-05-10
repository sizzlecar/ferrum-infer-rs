//! Lightweight engine for embedding models (CLIP, BERT, etc.).
//!
//! Wraps a `ClipModelExecutor` and implements `EmbedEngine` directly —
//! no LLM-method stubs, no fake `InferenceEngine::infer` impls. The
//! `InferenceEngine` supertrait covers lifecycle/status; the modality
//! itself goes through `EmbedEngine`.

use async_trait::async_trait;
use ferrum_interfaces::engine::{EmbedEngine, InferenceEngine};
use ferrum_models::ClipModelExecutor;
use ferrum_types::{EngineConfig, EngineMetrics, EngineStatus, FerrumError, Result};
use std::sync::Arc;

/// Embedding-only engine wrapping a ClipModelExecutor.
pub struct EmbeddingEngine {
    executor: Arc<ClipModelExecutor>,
    tokenizer: Option<tokenizers::Tokenizer>,
    config: EngineConfig,
}

impl EmbeddingEngine {
    pub fn new(executor: ClipModelExecutor, config: EngineConfig) -> Self {
        Self {
            executor: Arc::new(executor),
            tokenizer: None,
            config,
        }
    }

    /// Set tokenizer for text embedding.
    pub fn with_tokenizer(mut self, tokenizer: tokenizers::Tokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }
}

#[async_trait]
impl InferenceEngine for EmbeddingEngine {
    async fn status(&self) -> EngineStatus {
        crate::modality_stubs::inert_status()
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn metrics(&self) -> EngineMetrics {
        crate::modality_stubs::inert_metrics()
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        crate::modality_stubs::inert_health()
    }
}

#[async_trait]
impl EmbedEngine for EmbeddingEngine {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| FerrumError::model("No tokenizer loaded for text embedding"))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| FerrumError::model(format!("tokenize: {e}")))?;
        let embedding = self.executor.embed_text(encoding.get_ids())?;
        embedding
            .squeeze(0)
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.to_vec1())
            .map_err(|e| FerrumError::model(format!("embed_text tensor: {e}")))
    }

    async fn embed_image(&self, image: &str) -> Result<Vec<f32>> {
        let embedding = if image.starts_with("data:") || image.len() > 1000 {
            self.executor.embed_image_base64(image)?
        } else {
            self.executor.embed_image_path(image)?
        };
        embedding
            .squeeze(0)
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.to_vec1())
            .map_err(|e| FerrumError::model(format!("embed_image tensor: {e}")))
    }

    fn embedding_dim(&self) -> usize {
        self.executor.projection_dim()
    }
}
