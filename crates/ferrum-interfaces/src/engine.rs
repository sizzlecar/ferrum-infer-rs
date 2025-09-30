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
