//! # Ferrum Sampler
//!
//! Logits processing and sampling implementations for LLM inference.
//!
//! ## Overview
//!
//! This crate provides concrete implementations of the sampler interfaces defined
//! in `ferrum-interfaces`, including:
//!
//! - Logits processing chain (temperature, top-k, top-p, penalties)
//! - Sampling strategies (greedy, multinomial, beam search)
//! - Parallel sampling support
//! - Deterministic paths for testing
//!
//! ## Design Principles
//!
//! - **Composable Processing**: Chain multiple logits processors together.
//! - **Efficient Sampling**: SIMD/vectorized operations where possible.
//! - **Deterministic Testing**: Controllable RNG for reproducible results.
//! - **Parallel Support**: Enable speculative decoding and multi-sample generation.

pub mod implementations;
pub mod processors;

// Re-exports of interfaces from ferrum-interfaces
pub use ferrum_interfaces::{
    LogitsProcessor as LogitsProcessorInterface,
    Sampler as SamplerInterface,
    SamplerFactory as SamplerFactoryInterface,
    SamplingConfig,
    SamplingContext,
    SamplingMode,
    SamplingStats,
    ProcessorConfig,
};

pub use ferrum_types::{
    TokenId, Result, FerrumError, Temperature, TopK, TopP, 
    RepetitionPenalty, PresencePenalty, FrequencyPenalty,
};

// Re-exports of implementations
pub use implementations::*;
pub use processors::*;

/// Default sampler factory
pub fn default_factory() -> DefaultSamplerFactory {
    DefaultSamplerFactory::new()
}
