//! # Ferrum Core
//!
//! Legacy compatibility layer for the Ferrum inference framework.
//! This crate is being phased out in favor of ferrum-interfaces and ferrum-types.

// Re-export error types

// Re-export legacy types (for backward compatibility during transition)

// Re-export specific interfaces from ferrum-interfaces to avoid naming conflicts
pub use ferrum_interfaces::{
    AllocationRequest,
    BackendCapabilities,
    BatchHint,
    BatchPlan,
    BuildOptions,
    // Backend traits
    ComputeBackend,
    DecodeInput,
    DecodeOutput,
    // Memory
    DeviceMemoryManager,
    EngineStatusInterface as EngineStatus,
    IncrementalTokenizer,
    // Engine traits
    InferenceEngine,
    // KV Cache
    KvCacheHandle,
    KvCacheManager,
    // Sampling
    LogitsProcessor,
    // Model building
    ModelBuilder,
    // Model executor
    ModelExecutor,
    PrefillInput,
    PrefillOutput,
    Sampler,
    SamplingConfig,
    SamplingContext,
    // Scheduler traits
    SchedulerInterface as Scheduler,
    StreamHandle,
    TensorFactory,
    // Tensor
    TensorLike,
    TensorOps,
    TensorRef as Tensor,
    // Tokenizer
    Tokenizer,
    TokenizerFactory,
    WeightLoader,
};

// Re-export specific types from ferrum-types for compatibility
pub use ferrum_types::{
    BatchId, BlockId, DataType, Device, FerrumError as Error, FinishReason, InferenceRequest,
    InferenceResponse, MemoryUsage, ModelConfig as RuntimeConfig, ModelId, ModelInfo, ModelSource,
    ModelType, Priority, RequestId, Result, SamplingParams, SchedulerStats, SessionId,
    SpecialTokens, StreamChunk, TokenId,
};
