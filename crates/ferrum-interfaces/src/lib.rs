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

pub mod tensor;
pub mod tokenizer;
pub mod sampler;
pub mod kv_cache;
pub mod model_executor;
pub mod backend;
pub mod scheduler;
pub mod engine;
pub mod memory;
pub mod model_builder;

// Re-export core traits and important types
pub use tensor::{TensorLike, TensorRef, TensorFactory, TensorOps};
pub use tokenizer::{Tokenizer, TokenizerFactory, TokenizerInfo, IncrementalTokenizer};
pub use sampler::{LogitsProcessor, Sampler, SamplingContext, SamplingConfig};
pub use kv_cache::{KvCacheHandle, KvCacheManager, BlockTable, AllocationRequest};
pub use model_executor::{ModelExecutor, PrefillInput, PrefillOutput, DecodeInput, DecodeOutput};
pub use backend::{ComputeBackend, WeightLoader, BackendCapabilities};
pub use scheduler::{Scheduler as SchedulerInterface, BatchPlan, BatchHint};
pub use engine::InferenceEngine;
pub use memory::{DeviceMemoryManager, MemoryHandle, StreamHandle};
pub use model_builder::{ModelBuilder, BuildOptions};

// Re-export types from ferrum-types, avoiding conflicts
pub use ferrum_types::{
    // IDs
    RequestId, BatchId, ModelId, SessionId, TaskId, ClientId,
    // Basic types
    TokenId, BlockId, Result, FerrumError,
    // Requests and responses
    InferenceRequest, InferenceResponse, StreamChunk,
    // Sampling
    SamplingParams, Priority, FinishReason, SpecialTokens,
    // Device types
    Device, DataType,
    // Model types
    ModelInfo, ModelType, ModelSource,
    // Config types - use fully qualified names to avoid conflicts
    config::EngineConfig,
    config::SchedulerConfig,
    config::BackendConfig,
    config::TokenizerConfig,
    // Metrics
    MemoryUsage, SchedulerStats, EngineStatus, EngineMetrics, HealthStatus,
    ComponentStatus, ComponentHealth,
};
