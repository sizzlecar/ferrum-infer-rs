//! # Ferrum Core
//!
//! Legacy compatibility layer for the Ferrum inference framework.
//! This crate is being phased out in favor of ferrum-interfaces and ferrum-types.


// Re-export error types

// Re-export legacy types (for backward compatibility during transition)

// Re-export specific interfaces from ferrum-interfaces to avoid naming conflicts
pub use ferrum_interfaces::{
    // Engine traits
    InferenceEngine, EngineStatusInterface as EngineStatus,
    // Model executor
    ModelExecutor, PrefillInput, PrefillOutput, DecodeInput, DecodeOutput,
    // Backend traits  
    ComputeBackend, WeightLoader, BackendCapabilities,
    // Scheduler traits
    SchedulerInterface as Scheduler, BatchPlan, BatchHint,
    // Tokenizer
    Tokenizer, TokenizerFactory, IncrementalTokenizer,
    // Sampling
    LogitsProcessor, Sampler, SamplingContext, SamplingConfig,
    // KV Cache
    KvCacheHandle, KvCacheManager, AllocationRequest,
    // Tensor
    TensorLike, TensorRef as Tensor, TensorFactory, TensorOps,
    // Memory
    DeviceMemoryManager, StreamHandle,
    // Model building
    ModelBuilder, BuildOptions,
};

// Re-export specific types from ferrum-types for compatibility
pub use ferrum_types::{
    Result, FerrumError as Error,
    RequestId, BatchId, ModelId, SessionId, TokenId, BlockId,
    InferenceRequest, InferenceResponse, StreamChunk,
    SamplingParams, Priority, FinishReason, SpecialTokens,
    Device, DataType, ModelInfo, ModelType, ModelSource, ModelConfig as RuntimeConfig,
    MemoryUsage, SchedulerStats,
};
