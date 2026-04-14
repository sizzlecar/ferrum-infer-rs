//! Inference engine interface with streaming and batch support
//!
//! This module provides the top-level inference engine interface that
//! orchestrates all other components: tokenizer, model executor, scheduler,
//! and sampler.

use async_trait::async_trait;
use ferrum_types::{EngineConfig, InferenceRequest, InferenceResponse, Result, StreamChunk};
use futures::Stream;
use std::pin::Pin;

/// Core inference engine trait
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Execute single inference request
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Execute streaming inference request
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;

    /// Get current engine status
    async fn status(&self) -> ferrum_types::EngineStatus;

    /// Shutdown engine gracefully
    async fn shutdown(&self) -> Result<()>;

    /// Get engine configuration
    fn config(&self) -> &EngineConfig;

    /// Get engine metrics
    fn metrics(&self) -> ferrum_types::EngineMetrics;

    /// Health check
    async fn health_check(&self) -> ferrum_types::HealthStatus;

    /// Embed raw text string → float vector (engine handles tokenization).
    async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
        Err(ferrum_types::FerrumError::model(
            "This engine does not support text embedding",
        ))
    }

    /// Embed image (file path or base64) → float vector. Default: not supported.
    async fn embed_image(&self, _image: &str) -> Result<Vec<f32>> {
        Err(ferrum_types::FerrumError::model(
            "This engine does not support image embedding",
        ))
    }

    /// Get embedding dimension. Default: 0 (not an embedding model).
    fn embedding_dim(&self) -> usize {
        0
    }

    /// Transcribe audio file → text. Default: not supported.
    async fn transcribe_file(&self, _path: &str, _language: Option<&str>) -> Result<String> {
        Err(ferrum_types::FerrumError::model(
            "This engine does not support audio transcription",
        ))
    }

    /// Transcribe audio bytes (WAV) → text. Default: not supported.
    async fn transcribe_bytes(&self, _data: &[u8], _language: Option<&str>) -> Result<String> {
        Err(ferrum_types::FerrumError::model(
            "This engine does not support audio transcription",
        ))
    }

    /// Synthesize speech → PCM audio chunks (streaming).
    /// Returns Vec of (chunk_index, PCM f32 samples).
    /// Default: not supported.
    async fn synthesize_speech(
        &self,
        _text: &str,
        _language: Option<&str>,
        _chunk_frames: usize,
    ) -> Result<Vec<Vec<f32>>> {
        Err(ferrum_types::FerrumError::model(
            "This engine does not support speech synthesis",
        ))
    }

    /// Get TTS sample rate (default 24000).
    fn tts_sample_rate(&self) -> u32 {
        24000
    }
}

/// Advanced engine capabilities
#[async_trait]
pub trait AdvancedInferenceEngine: InferenceEngine {
    /// Execute batch inference
    async fn infer_batch(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Result<Vec<Result<InferenceResponse>>>;

    /// Execute speculative inference
    async fn infer_speculative(
        &self,
        request: InferenceRequest,
        speculation_config: ferrum_types::SpeculationConfig,
    ) -> Result<InferenceResponse>;

    /// Warm up engine with sample requests
    async fn warmup(
        &mut self,
        warmup_requests: Vec<InferenceRequest>,
    ) -> Result<ferrum_types::WarmupResult>;

    /// Configure engine at runtime
    async fn reconfigure(&mut self, config: EngineConfig) -> Result<()>;

    /// Get detailed diagnostics
    async fn diagnostics(&self) -> ferrum_types::DiagnosticsReport;

    /// Export engine state for debugging
    async fn export_state(&self) -> Result<ferrum_types::EngineState>;

    /// Import engine state for debugging/testing
    async fn import_state(&mut self, state: ferrum_types::EngineState) -> Result<()>;
}

/// Speculation configuration for speculative decoding
pub type SpeculationConfig = ferrum_types::SpeculationConfig;

/// Hardware constraints alias
pub type HardwareConstraints = ferrum_types::HardwareConstraints;

/// Request characteristics alias
pub type RequestCharacteristics = ferrum_types::RequestCharacteristics;

/// Latency requirements alias
pub type LatencyRequirements = ferrum_types::LatencyRequirements;
