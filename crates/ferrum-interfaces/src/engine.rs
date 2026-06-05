//! Inference engine interfaces — split per modality.
//!
//! Phase 5a step 2 splits the historical mega-trait (which mixed LLM
//! generation, embedding, transcription, and TTS in one) into a base
//! lifecycle trait and four modality-specific supertraits. Each
//! engine impl now implements exactly the trait its modality needs;
//! no more inert "unsupported" stubs.

use async_trait::async_trait;
use ferrum_types::{EngineConfig, InferenceRequest, InferenceResponse, Result, StreamChunk};
use futures::Stream;
use std::pin::Pin;

/// Lifecycle / status methods shared by every engine kind.
///
/// LLM engines, embedders, transcribers, and TTS services all expose
/// the same minimal status/metrics surface to the server / CLI. The
/// modality-specific traits below extend this base.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Get current engine status.
    async fn status(&self) -> ferrum_types::EngineStatus;

    /// Shutdown engine gracefully.
    async fn shutdown(&self) -> Result<()>;

    /// Get engine configuration.
    fn config(&self) -> &EngineConfig;

    /// Get engine metrics.
    fn metrics(&self) -> ferrum_types::EngineMetrics;

    /// Health check.
    async fn health_check(&self) -> ferrum_types::HealthStatus;

    /// Optional cache metrics emitted by concrete LLM engines.
    ///
    /// The default keeps non-LLM and stub engines source-compatible. Real
    /// engines can expose prefix/session cache counters without forcing those
    /// fields into every modality's core metrics type.
    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Optional LoRA runtime metrics emitted by concrete LLM engines.
    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }
}

/// LLM text-generation engine.
///
/// Implemented by `ContinuousBatchEngine` (the production path) and
/// `DefaultInferenceEngine` (legacy reference path). Backs
/// `/v1/chat/completions` and `/v1/completions`.
#[async_trait]
pub trait LlmInferenceEngine: InferenceEngine {
    /// Execute single inference request.
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Execute streaming inference request.
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;
}

/// Embedding engine (CLIP, BERT, etc.).
///
/// Backs `/v1/embeddings`. Distinct from LLM engines — no token
/// generation, no scheduling, no KV cache.
#[async_trait]
pub trait EmbedEngine: InferenceEngine {
    /// Embed raw text string → float vector (engine handles tokenization).
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed image (file path or base64) → float vector.
    async fn embed_image(&self, image: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension.
    fn embedding_dim(&self) -> usize;
}

/// Speech-to-text (Whisper) engine.
///
/// Backs `/v1/audio/transcriptions`.
#[async_trait]
pub trait TranscribeEngine: InferenceEngine {
    /// Transcribe audio file → text.
    async fn transcribe_file(&self, path: &str, language: Option<&str>) -> Result<String>;

    /// Transcribe audio bytes (WAV / etc.) → text.
    async fn transcribe_bytes(&self, data: &[u8], language: Option<&str>) -> Result<String>;
}

/// Text-to-speech (Qwen3-TTS, etc.) engine.
///
/// Backs `/v1/audio/speech`.
#[async_trait]
pub trait TtsEngine: InferenceEngine {
    /// Synthesize speech → PCM audio chunks (streaming).
    /// Returns Vec of PCM f32 samples per chunk.
    async fn synthesize_speech(
        &self,
        text: &str,
        language: Option<&str>,
        chunk_frames: usize,
    ) -> Result<Vec<Vec<f32>>>;

    /// Get TTS sample rate.
    fn tts_sample_rate(&self) -> u32;
}

/// Advanced engine capabilities — opt-in addition to LLM engines that
/// support batching / speculation / runtime reconfig / diagnostics.
#[async_trait]
pub trait AdvancedInferenceEngine: LlmInferenceEngine {
    /// Execute batch inference.
    async fn infer_batch(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Result<Vec<Result<InferenceResponse>>>;

    /// Execute speculative inference.
    async fn infer_speculative(
        &self,
        request: InferenceRequest,
        speculation_config: ferrum_types::SpeculationConfig,
    ) -> Result<InferenceResponse>;

    /// Warm up engine with sample requests.
    async fn warmup(
        &mut self,
        warmup_requests: Vec<InferenceRequest>,
    ) -> Result<ferrum_types::WarmupResult>;

    /// Configure engine at runtime.
    async fn reconfigure(&mut self, config: EngineConfig) -> Result<()>;

    /// Get detailed diagnostics.
    async fn diagnostics(&self) -> ferrum_types::DiagnosticsReport;

    /// Export engine state for debugging.
    async fn export_state(&self) -> Result<ferrum_types::EngineState>;

    /// Import engine state for debugging/testing.
    async fn import_state(&mut self, state: ferrum_types::EngineState) -> Result<()>;
}

/// Speculation configuration for speculative decoding.
pub type SpeculationConfig = ferrum_types::SpeculationConfig;

/// Hardware constraints alias.
pub type HardwareConstraints = ferrum_types::HardwareConstraints;

/// Request characteristics alias.
pub type RequestCharacteristics = ferrum_types::RequestCharacteristics;

/// Latency requirements alias.
pub type LatencyRequirements = ferrum_types::LatencyRequirements;
