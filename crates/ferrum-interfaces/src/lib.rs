//! Core interface definitions for the Ferrum inference framework
//!
//! This crate carries the stable, GPU-free trait contracts shared across
//! the workspace: model execution, scheduling, KV cache management,
//! tokenization, sampling, and the lifecycle/modality engine traits.
//! Hardware backends live in `ferrum-kernels` (the `Backend<B>` trait
//! and its supertraits); only types that compile without GPU features
//! belong here.

#![allow(async_fn_in_trait)]

pub mod engine;
pub mod kv_cache;
pub mod kv_dtype;
pub mod model_executor;
pub mod recurrent_state;
pub mod sampler;
pub mod scheduler;
pub mod tensor;
pub mod tokenizer;
pub mod vnext;

// Re-export core traits and important types
pub use engine::InferenceEngine;
pub use kv_cache::{
    AllocationRequest, BlockTable, CacheHandleStats, KvCacheHandle, KvCacheManager,
};
pub use kv_dtype::{KvBf16, KvDtypeKind, KvFp16, KvFp8, KvInt8};
pub use model_executor::{DecodeInput, DecodeOutput, ModelExecutor, PrefillInput, PrefillOutput};
pub use recurrent_state::{
    RecurrentStateHandle, RecurrentStateHandleStats, RecurrentStateManager,
    RecurrentStateManagerStats, RecurrentStateResumePolicy, RecurrentStateSpec,
    RecurrentStateTensorSpec,
};
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
