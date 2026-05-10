//! Lightweight engine for Whisper ASR transcription.

use async_trait::async_trait;
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_models::WhisperModelExecutor;
use ferrum_types::{
    EngineConfig, EngineMetrics, EngineStatus, FerrumError, InferenceRequest, InferenceResponse,
    Result, StreamChunk,
};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;

pub struct TranscriptionEngine {
    executor: Arc<WhisperModelExecutor>,
    config: EngineConfig,
}

impl TranscriptionEngine {
    pub fn new(executor: WhisperModelExecutor, config: EngineConfig) -> Self {
        Self {
            executor: Arc::new(executor),
            config,
        }
    }
}

#[async_trait]
impl InferenceEngine for TranscriptionEngine {
    async fn infer(&self, _request: InferenceRequest) -> Result<InferenceResponse> {
        Err(FerrumError::model(
            "Whisper is an ASR model. Use /v1/audio/transcriptions instead.",
        ))
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(FerrumError::model(
            "Whisper is an ASR model. Use /v1/audio/transcriptions instead.",
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
        crate::modality_stubs::inert_metrics()
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        crate::modality_stubs::inert_health()
    }

    async fn transcribe_file(&self, path: &str, language: Option<&str>) -> Result<String> {
        self.executor.transcribe_file(path, language)
    }

    async fn transcribe_bytes(&self, data: &[u8], language: Option<&str>) -> Result<String> {
        self.executor.transcribe_bytes(data, language)
    }
}
