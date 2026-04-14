//! TTS Engine — wraps TtsModelExecutor for HTTP serving.

use async_trait::async_trait;
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_models::executor::tts_executor::TtsModelExecutor;
use ferrum_types::{
    EngineConfig, EngineMetrics, EngineStatus, FerrumError, InferenceRequest, InferenceResponse,
    ModelId, Result, StreamChunk,
};
use futures::Stream;
use parking_lot::Mutex;
use std::pin::Pin;

pub struct TtsEngine {
    executor: Mutex<TtsModelExecutor>,
    config: EngineConfig,
}

impl TtsEngine {
    pub fn new(executor: TtsModelExecutor, model_id: ModelId) -> Self {
        let config = crate::simple_engine_config(model_id, ferrum_types::Device::CPU);
        Self {
            executor: Mutex::new(executor),
            config,
        }
    }
}

#[async_trait]
impl InferenceEngine for TtsEngine {
    async fn infer(&self, _request: InferenceRequest) -> Result<InferenceResponse> {
        Err(FerrumError::model(
            "TTS model. Use /v1/audio/speech instead.",
        ))
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(FerrumError::model(
            "TTS model. Use /v1/audio/speech instead.",
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

    async fn synthesize_speech(
        &self,
        text: &str,
        language: Option<&str>,
        chunk_frames: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let lang = language.unwrap_or("auto");
        let mut executor = self.executor.lock();
        executor.synthesize_streaming(text, lang, chunk_frames, |_, _| {})
    }

    fn tts_sample_rate(&self) -> u32 {
        let executor = self.executor.lock();
        executor.sample_rate() as u32
    }
}
