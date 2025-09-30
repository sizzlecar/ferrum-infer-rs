//! Core interface definitions for the Ferrum inference framework
//!
//! This crate defines all the stable trait interfaces that different components
//! of Ferrum implement. It provides a clean abstraction layer that allows for
//! pluggable implementations of tokenizers, model executors, schedulers,
//! cache managers, and other core components.
//!
//! The interfaces are designed following the principles outlined in the
//! refactoring documentation:
//! - Single responsibility with stable boundaries
//! - Zero-copy and handle semantics
//! - Capability discovery driven
//! - Performance-first API design

pub mod backend;
pub mod engine;
pub mod kv_cache;
pub mod memory;
pub mod model_builder;
pub mod model_executor;
pub mod sampler;
pub mod scheduler;
pub mod tensor;
pub mod tokenizer;

// Re-export core traits and important types
pub use backend::{BackendCapabilities, ComputeBackend, WeightLoader};
pub use engine::InferenceEngine;
pub use kv_cache::{AllocationRequest, BlockTable, KvCacheHandle, KvCacheManager};
pub use memory::{DeviceMemoryManager, MemoryHandle, StreamHandle};
pub use model_builder::{BuildOptions, ModelBuilder};
pub use model_executor::{DecodeInput, DecodeOutput, ModelExecutor, PrefillInput, PrefillOutput};
pub use sampler::{LogitsProcessor, Sampler, SamplingConfig, SamplingContext};
pub use scheduler::{BatchHint, BatchPlan, Scheduler as SchedulerInterface};
pub use tensor::{TensorFactory, TensorLike, TensorOps, TensorRef};
pub use tokenizer::{IncrementalTokenizer, Tokenizer, TokenizerFactory, TokenizerInfo};

// Re-export types from ferrum-types, avoiding conflicts
pub use ferrum_types::{
    config::BackendConfig,
    // Config types - use fully qualified names to avoid conflicts
    config::EngineConfig,
    config::SchedulerConfig,
    config::TokenizerConfig,
    BatchId,
    BlockId,
    ClientId,
    ComponentHealth,
    ComponentStatus,
    DataType,
    // Device types
    Device,
    EngineMetrics,
    EngineStatus,
    FerrumError,
    FinishReason,
    HealthStatus,
    // Requests and responses
    InferenceRequest,
    InferenceResponse,
    // Metrics
    MemoryUsage,
    ModelId,
    // Model types
    ModelInfo,
    ModelSource,
    ModelType,
    Priority,
    // IDs
    RequestId,
    Result,
    // Sampling
    SamplingParams,
    SchedulerStats,
    SessionId,
    SpecialTokens,
    StreamChunk,
    TaskId,
    // Basic types
    TokenId,
};
