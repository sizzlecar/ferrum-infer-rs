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
pub mod pipeline;
pub mod coordinator;
pub mod factory;

// Re-exports of interfaces from ferrum-interfaces
pub use ferrum_interfaces::{
    InferenceEngine as InferenceEngineInterface,
    EngineStatus,
    StreamChunk,
};

pub use ferrum_types::{
    Result, InferenceRequest, InferenceResponse, RequestId, BatchId,
    StreamingConfig, FerrumError, EngineConfig,
};

// Re-exports from implementation crates
pub use ferrum_scheduler::{Scheduler, BatchPlan};
pub use ferrum_tokenizer::{Tokenizer, IncrementalTokenizer};
pub use ferrum_sampler::{Sampler, LogitsProcessor};
pub use ferrum_kv::KvCacheManager;
pub use ferrum_models::{ModelBuilder, WeightLoader};
pub use ferrum_runtime::{ComputeBackend, TensorFactory};

// Re-exports of implementations
pub use engine::*;
pub use pipeline::*;
pub use coordinator::*;
pub use factory::*;

/// Create default inference engine with MVP configuration
pub async fn create_default_engine(
    config: EngineConfig,
) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
    let factory = DefaultEngineFactory::new();
    factory.create_engine(config).await
}
