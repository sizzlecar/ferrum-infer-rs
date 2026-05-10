//! Lightweight engine for Whisper ASR transcription.

use async_trait::async_trait;
use ferrum_interfaces::engine::{InferenceEngine, TranscribeEngine};
use ferrum_models::WhisperModelExecutor;
use ferrum_types::{EngineConfig, EngineMetrics, EngineStatus, Result};
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
impl TranscribeEngine for TranscriptionEngine {
    async fn transcribe_file(&self, path: &str, language: Option<&str>) -> Result<String> {
        self.executor.transcribe_file(path, language)
    }

    async fn transcribe_bytes(&self, data: &[u8], language: Option<&str>) -> Result<String> {
        self.executor.transcribe_bytes(data, language)
    }
}
