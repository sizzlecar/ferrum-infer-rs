//! Core interface definitions for the Ferrum inference framework
//!
//! Foundation ports for scheduling, KV cache management, tokenization,
//! sampling, tensors, and inference entrypoints. Runtime authorization lives
//! in `ferrum-runtime`; product execution contracts live in `ferrum-execution`.
//! This crate does not depend on either upper layer or concrete backends.

#![allow(async_fn_in_trait)]

#[cfg(test)]
extern crate self as ferrum_interfaces;

pub mod adapter;
pub mod engine;
pub mod kv_cache;
pub mod kv_dtype;
pub mod request_observation;
pub mod sampler;
pub mod scheduler;
pub mod tensor;
pub mod tokenizer;

// Re-export core traits and important types
pub use engine::InferenceEngine;
pub use kv_cache::{
    AllocationRequest, BlockTable, CacheHandleStats, KvCacheHandle, KvCacheManager,
};
pub use kv_dtype::{KvBf16, KvDtypeKind, KvFp16, KvFp8, KvInt8};
pub use sampler::{
    LogitsProcessor, Sampler, SamplingConfig, SamplingContext, SamplingRng,
    SAMPLING_RNG_ALGORITHM_ID,
};
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
