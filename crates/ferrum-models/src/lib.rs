//! # Ferrum Models
//! 
//! Abstract model definitions and interfaces for the Ferrum inference framework.
//! This module provides framework-agnostic abstractions for various LLM architectures.
//!
//! ## Design Principles
//! 
//! - No direct dependencies on ML frameworks (Candle, ONNX Runtime, etc.)
//! - Pure trait definitions and abstract configurations
//! - Concrete implementations belong in backend-specific crates
//!
//! ## Architecture Support
//! 
//! The module defines abstractions for:
//! - Llama family (Llama, Llama2, Llama3)
//! - Mistral family (Mistral, Mixtral)
//! - Qwen family (Qwen, Qwen2)
//! - Other architectures (Phi, Gemma, etc.)

pub mod traits;
pub mod config;
pub mod registry;
pub mod tokenizer;

// Re-exports
pub use traits::{
    Architecture, AbstractModelConfig, ModelBuilder, ModelRegistry,
    Tokenizer, SpecialTokens, ModelConverter,
    NormType, Activation, AttentionConfig, RopeScaling,
};
pub use config::ConfigManager;
pub use registry::DefaultModelRegistry;
pub use tokenizer::TokenizerWrapper;

