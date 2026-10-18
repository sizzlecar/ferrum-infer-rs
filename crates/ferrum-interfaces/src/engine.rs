//! Engine lifecycle and text-generation contracts.
//!
//! Future task-specific engines can extend the lifecycle contract without
//! implementing text-generation operations or creating dummy KV state.

use async_trait::async_trait;
use ferrum_types::{
    EngineConfig, ExecutionResourceAuthority, ExecutorAdmissionSnapshot, InferenceRequest,
    InferenceResponse, Result, StreamChunk,
};
use futures::Stream;
use std::pin::Pin;

/// Lifecycle / status methods shared by every engine kind.
///
/// Task-specific execution contracts extend this minimal lifecycle surface.
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
    /// Engines can expose prefix/session cache counters without requiring
    /// cache support from every implementation of the lifecycle contract.
    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Optional compact provider-attribution witness from the active model
    /// executor. The default keeps non-vNext engines source-compatible.
    fn execution_attribution_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Online latency scheduling state, when enabled for this engine.
    /// This is a host execution feedback snapshot, not client-observed SLO compliance.
    fn scheduler_slo_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Runtime-authoritative admission state. Startup sizing estimates are
    /// intentionally not accepted through this method.
    fn admission_snapshot(&self) -> Result<Option<ExecutorAdmissionSnapshot>> {
        Ok(None)
    }

    /// Optional LoRA runtime metrics emitted by concrete LLM engines.
    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }
}

/// LLM text-generation engine.
///
/// Implemented by `ContinuousBatchEngine`. Backs the chat, completions and
/// Responses APIs.
#[async_trait]
pub trait LlmInferenceEngine: InferenceEngine {
    /// Authoritative owner of request-lifetime state and capacity. This is a
    /// cheap capability query; it must not construct a metrics snapshot.
    /// Plan-runtime engines supply native prefix observations even when their
    /// selected plan cannot retain checkpoints.
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::LegacyEngine
    }

    /// Effective per-request capacity in tokens, including input and output.
    /// Implementations must report the limit used by request admission, not
    /// the model weights' nominal context window. None means unreported.
    fn context_capacity(&self) -> Option<usize> {
        None
    }

    /// Execute single inference request.
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Execute streaming inference request.
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;
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
