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

pub mod coordinator;
pub mod engine;
pub mod factory;
pub mod pipeline;

// Re-exports of interfaces
pub use ferrum_interfaces::{
    InferenceEngine as InferenceEngineInterface,
    KvCacheManager, ModelBuilder, ModelExecutor, WeightLoader,
    Sampler, SchedulerInterface as Scheduler,
    Tokenizer, IncrementalTokenizer,
};

pub use ferrum_types::{
    BatchId, EngineConfig, EngineStatus, FerrumError, InferenceRequest, InferenceResponse, 
    RequestId, Result, StreamChunk,
};

// Re-exports from implementation crates
pub use ferrum_runtime::ComputeBackend;
pub use ferrum_scheduler::BatchPlan;

// Re-exports of implementations
pub use coordinator::*;
pub use engine::*;
pub use factory::*;
pub use pipeline::*;

/// Create default inference engine with MVP configuration
pub async fn create_default_engine(
    config: EngineConfig,
) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
    let factory = DefaultEngineFactory::new();
    factory.create_engine(config).await
}
