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
    StreamingConfig, FerrumError,
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

/// Engine configuration
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Model configuration
    pub model_config: ferrum_interfaces::ModelConfig,
    /// Device to use
    pub device: ferrum_types::Device,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Enable streaming
    pub enable_streaming: bool,
    /// Streaming configuration
    pub streaming_config: StreamingConfig,
    /// KV cache configuration
    pub kv_cache_config: ferrum_kv::KvCacheConfig,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_config: ferrum_interfaces::ModelConfig {
                model_id: ferrum_types::ModelId::new("default"),
                architecture: ferrum_types::Architecture::Llama,
                vocab_size: 32000,
                hidden_size: 4096,
                num_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: None,
                intermediate_size: None,
                max_position_embeddings: Some(2048),
                rope_theta: Some(10000.0),
                rope_scaling: None,
                rms_norm_eps: Some(1e-6),
                data_type: Some(ferrum_types::DataType::F16),
            },
            device: ferrum_types::Device::Cpu,
            max_batch_size: 8,
            max_sequence_length: 2048,
            enable_streaming: true,
            streaming_config: StreamingConfig::default(),
            kv_cache_config: ferrum_kv::KvCacheConfig::default(),
            enable_metrics: true,
        }
    }
}
