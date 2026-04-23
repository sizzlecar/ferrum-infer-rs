//! TTS Engine — concurrent slot-based serving of TtsModelExecutor.
//!
//! Multiple TTS requests can be processed in parallel, each on its own executor slot.
//! Slots share nothing (each has its own model weights + KV cache).
//! Future: Phase 2 will share weights across slots to reduce memory.

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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct TtsEngine {
    slots: Vec<Arc<Mutex<TtsModelExecutor>>>,
    /// Semaphore to limit concurrent slot usage
    semaphore: tokio::sync::Semaphore,
    config: EngineConfig,
    sample_rate: u32,
    active_requests: AtomicUsize,
}

impl TtsEngine {
    /// Create with a single executor (backward compatible).
    pub fn new(executor: TtsModelExecutor, model_id: ModelId) -> Self {
        let sr = executor.sample_rate() as u32;
        let config = crate::simple_engine_config(model_id, ferrum_types::Device::CPU);
        Self {
            slots: vec![Arc::new(Mutex::new(executor))],
            semaphore: tokio::sync::Semaphore::new(1),
            config,
            sample_rate: sr,
            active_requests: AtomicUsize::new(0),
        }
    }

    /// Create with multiple executor slots for concurrent serving.
    pub fn new_multi(executors: Vec<TtsModelExecutor>, model_id: ModelId) -> Self {
        let n = executors.len().max(1);
        let sr = executors
            .first()
            .map(|e| e.sample_rate() as u32)
            .unwrap_or(24000);
        let config = crate::simple_engine_config(model_id, ferrum_types::Device::CPU);
        let slots: Vec<_> = executors
            .into_iter()
            .map(|e| Arc::new(Mutex::new(e)))
            .collect();
        Self {
            slots,
            semaphore: tokio::sync::Semaphore::new(n),
            config,
            sample_rate: sr,
            active_requests: AtomicUsize::new(0),
        }
    }

    /// Number of available slots.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
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
            active_requests: self.active_requests.load(Ordering::Relaxed),
            queued_requests: self.slots.len() - self.semaphore.available_permits(),
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
        // Acquire a slot (waits if all slots busy)
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| FerrumError::model("TTS semaphore closed"))?;

        self.active_requests.fetch_add(1, Ordering::Relaxed);

        // Find an unlocked slot (semaphore guarantees at least one is available)
        let slot = self
            .slots
            .iter()
            .find(|s| s.try_lock().is_some())
            .unwrap_or(&self.slots[0])
            .clone();

        let text = text.to_string();
        let lang = language.unwrap_or("auto").to_string();
        let _active = &self.active_requests;

        // Run TTS on blocking thread (model forward is CPU/GPU bound)
        let result = tokio::task::spawn_blocking(move || {
            let mut executor = slot.lock();
            executor.synthesize_streaming(&text, &lang, chunk_frames, |_, _| {})
        })
        .await
        .map_err(|e| FerrumError::model(format!("TTS task panic: {e}")))?;

        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        result
    }

    fn tts_sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
