//! # Ferrum Engine
//!
//! LLM inference engine orchestration layer with strong streaming support.
//!
//! ## Overview
//!
//! This crate provides the main inference engine implementation that orchestrates
//! all the components from other ferrum crates:
//!
//! - Request admission and scheduling (ferrum-scheduler)
//! - KV-cache allocation and management (ferrum-kv)
//! - Tokenization and incremental decoding (ferrum-tokenizer)
//! - Logits processing and sampling (ferrum-sampler)
//! - Model execution and weight loading (ferrum-models)
//! - Runtime and compute backends (ferrum-runtime)
//!
//! ## Design Principles
//!
//! - **Strong Streaming**: TTFT optimization and consistent inter-token latency
//! - **Orchestration Layer**: Compose components rather than implement functionality
//! - **Batch Processing**: Dynamic continuous batching for throughput
//! - **Pipeline Optimization**: Prefill→decode loops with minimal overhead

pub mod engine;
pub mod factory;

// Re-exports of interfaces
pub use ferrum_interfaces::{
    IncrementalTokenizer, InferenceEngine as InferenceEngineInterface, KvCacheManager,
    ModelBuilder, ModelExecutor, Sampler, SchedulerInterface as Scheduler, Tokenizer, WeightLoader,
};

pub use ferrum_types::{
    BatchId, EngineConfig, EngineStatus, FerrumError, InferenceRequest, InferenceResponse,
    RequestId, Result, StreamChunk,
};

// Re-exports from implementation crates
pub use ferrum_runtime::ComputeBackend;
pub use ferrum_scheduler::BatchPlan;

// Re-exports of implementations
pub use engine::*;
pub use factory::*;

/// Create default inference engine with MVP configuration
pub async fn create_default_engine(
    config: EngineConfig,
) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
    let factory = DefaultEngineFactory::new();
    factory.create_engine(config).await
}

/// Create MVP engine - alias for create_default_engine
pub async fn create_mvp_engine(
    config: EngineConfig,
) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
    create_default_engine(config).await
}

/// Create a simple engine config from minimal parameters
pub fn simple_engine_config(
    model_id: impl Into<ferrum_types::ModelId>,
    device: ferrum_types::Device,
) -> EngineConfig {
    use ferrum_types::*;

    let mut config = EngineConfig::default();
    config.model.model_id = model_id.into();
    config.backend.device = device;

    // Set reasonable defaults for MVP
    config.batching.max_batch_size = 32;
    config.kv_cache.block_size = 16;
    config.kv_cache.max_blocks = 512;
    config.scheduler.max_running_requests = 32;

    config
}
