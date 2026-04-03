//! Lightweight engine for embedding models (CLIP, BERT, etc.).
//!
//! Unlike the full inference engine, this doesn't need scheduling,
//! KV cache management, or token generation. It just wraps an executor
//! and provides embed_text / embed_image via the InferenceEngine trait.

use async_trait::async_trait;
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_models::ClipModelExecutor;
use ferrum_types::{
    EngineConfig, EngineMetrics, EngineStatus, FerrumError, InferenceRequest, InferenceResponse,
    Result, StreamChunk,
};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;

/// Embedding-only engine wrapping a ClipModelExecutor.
pub struct EmbeddingEngine {
    executor: Arc<ClipModelExecutor>,
    config: EngineConfig,
}

impl EmbeddingEngine {
    pub fn new(executor: ClipModelExecutor, config: EngineConfig) -> Self {
        Self {
            executor: Arc::new(executor),
            config,
        }
    }
}

#[async_trait]
impl InferenceEngine for EmbeddingEngine {
    async fn infer(&self, _request: InferenceRequest) -> Result<InferenceResponse> {
        Err(FerrumError::model(
            "Embedding models don't support text generation. Use /v1/embeddings instead.",
        ))
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(FerrumError::model(
            "Embedding models don't support streaming. Use /v1/embeddings instead.",
        ))
    }

    async fn status(&self) -> EngineStatus {
        EngineStatus {
            is_ready: true,
            loaded_models: vec![],
            active_requests: 0,
            queued_requests: 0,
            memory_usage: ferrum_types::MemoryUsage {
                total_bytes: 0,
                used_bytes: 0,
                free_bytes: 0,
                gpu_memory_bytes: None,
                cpu_memory_bytes: None,
                cache_memory_bytes: 0,
                utilization_percent: 0.0,
            },
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_request_latency_ms: 0.0,
            p95_request_latency_ms: 0.0,
            p99_request_latency_ms: 0.0,
            throughput_rps: 0.0,
            tokens_per_second: 0.0,
            queue_metrics: Default::default(),
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: Default::default(),
        }
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        ferrum_types::HealthStatus::healthy()
    }

    async fn embed_text(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let embedding = self.executor.embed_text(tokens)?;
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
