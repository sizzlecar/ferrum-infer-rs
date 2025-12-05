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
//! - **Registry Pattern**: Dynamic component registration and lookup
//!
//! ## Usage
//!
//! ### Using the Engine Builder (Recommended)
//!
//! ```rust,ignore
//! use ferrum_engine::{EngineBuilder, EngineConfig};
//!
//! let config = EngineConfig::default();
//! let engine = EngineBuilder::new(config)
//!     .with_scheduler("fifo")
//!     .with_sampler("greedy")
//!     .build()
//!     .await?;
//! ```
//!
//! ### Using the Factory
//!
//! ```rust,ignore
//! use ferrum_engine::{DefaultEngineFactory, EngineConfig};
//!
//! let factory = DefaultEngineFactory::new();
//! let engine = factory.create_engine(config).await?;
//! ```
//!
//! ### Registering Custom Components
//!
//! ```rust,ignore
//! use ferrum_engine::{ComponentRegistry, global_registry};
//!
//! let registry = global_registry();
//! registry.register_backend_factory("my_backend", Arc::new(MyBackendFactory));
//! ```

pub mod builder;
pub mod continuous_engine;
pub mod engine;
pub mod factory;
pub mod kernels;
pub mod parallel;
pub mod pipeline;
pub mod registry;

// Metal backend (Apple Silicon only)
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal;

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

// Re-exports of engine implementation
pub use continuous_engine::{ContinuousBatchEngine, SequenceState};
pub use engine::*;

// Re-exports of pipeline
pub use pipeline::{
    ChunkedPrefillConfig, ChunkedPrefillExecutor, ExecutionPhase, PipelineConfig,
    PipelineExecutor,
};

// Re-exports of factory
pub use factory::{DefaultEngineFactory, RegistryBasedEngineFactory};

// Re-exports of builder
pub use builder::{create_engine, create_engine_with_registry, EngineBuilder};

// Re-exports of registry
pub use registry::{
    global_registry, set_global_registry, CandleBackendFactory, CandleExecutorFactory,
    ComponentConfig, ComponentFactory, ComponentMetadata, ComponentRegistry,
    ContinuousBatchSchedulerFactory, DefaultKvCacheFactory, FifoSchedulerFactory,
    GreedySampler, GreedySamplerFactory, HuggingFaceTokenizerFactory, MultinomialSamplerFactory,
    PagedKvCacheFactory, PrioritySchedulerFactory, StubExecutorFactory, StubTokenizer,
    StubTokenizerFactory,
};

// Re-exports of kernels
pub use kernels::{
    global_kernel_registry, AttentionConfig, AttentionKernel, AttentionType, FusedOpType,
    FusedOps, FusedOpsConfig, FusedRopeAttention, KernelInfo, KernelRegistry, PerformanceHint,
    RopeCache,
};

// Re-exports of parallel module
pub use parallel::{
    global_device_manager, DeviceCapability, DeviceInfo, DeviceManager, LayerDistribution,
    ParallelConfig, ParallelExecutor, ParallelExecutorFactory, ParallelismType,
    TensorParallelConfig, TensorParallelGroup,
};

/// Create default inference engine with MVP configuration
///
/// This is a convenience function that uses the default registry.
pub async fn create_default_engine(
    config: EngineConfig,
) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
    create_engine(config).await
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

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_create_engine_via_builder() {
        let config = simple_engine_config("test-model", ferrum_types::Device::CPU);

        let engine = EngineBuilder::new(config)
            .with_tokenizer("stub")
            .with_executor("stub")
            .build()
            .await;

        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_create_engine_via_factory() {
        let config = simple_engine_config("test-model", ferrum_types::Device::CPU);

        let factory = DefaultEngineFactory::new();
        let engine = factory.create_engine(config).await;

        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_create_engine_convenience() {
        let config = simple_engine_config("test-model", ferrum_types::Device::CPU);
        let engine = create_default_engine(config).await;

        assert!(engine.is_ok());
    }

    #[test]
    fn test_global_registry() {
        let registry = global_registry();

        // Should have default factories
        assert!(registry.list_backends().contains(&"candle".to_string()));
        assert!(registry.list_tokenizers().contains(&"stub".to_string()));
        assert!(registry.list_samplers().contains(&"multinomial".to_string()));
    }

    #[test]
    fn test_custom_registry() {
        let registry = ComponentRegistry::new();
        assert!(registry.list_backends().is_empty());

        registry.register_defaults();
        assert!(!registry.list_backends().is_empty());
    }
}
