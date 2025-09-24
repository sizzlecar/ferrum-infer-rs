//! # Ferrum Models
//!
//! Model building and weight loading implementations for LLM inference.
//!
//! ## Overview
//!
//! This crate provides concrete implementations of the model interfaces defined
//! in `ferrum-interfaces`, including:
//!
//! - ModelBuilder for constructing model executors from configs
//! - WeightLoader for loading safetensors/GGUF weights
//! - Model registry for architecture mapping
//! - Model source resolution (HF Hub, local files, URLs)
//!
//! ## Design Principles
//!
//! - **Builder Pattern**: Clear separation of model definition vs construction
//! - **Weight Loading**: Abstract over different weight formats and sources
//! - **Registry Pattern**: Map architecture names to builders
//! - **Source Resolution**: Unified loading from various sources
//!
//! ## Architecture Support
//!
//! - Llama family (Llama, Llama2, Code Llama, Vicuna)
//! - Mistral family (Mistral 7B, Mixtral, Codestral)
//! - Qwen family (Qwen, Qwen2, CodeQwen)
//! - Other architectures (Phi, Gemma, ChatGLM)

use ferrum_interfaces::{
    ModelBuilder,
    ModelExecutor,
    ComputeBackend,
    WeightLoader,
};

pub mod builder;
pub mod config;
pub mod loader;
pub mod registry;
pub mod source;

// Re-exports of interfaces from ferrum-interfaces
pub use ferrum_interfaces::{
    ModelBuilder as ModelBuilderInterface,
    ModelExecutor as ModelExecutorInterface,
    WeightLoader as WeightLoaderInterface,
    ModelInfo,
    ModelConfig,
    WeightSpec,
    ModelCapabilities,
};

pub use ferrum_types::{
    Result, ModelId, Architecture, DataType, Device, FerrumError,
};

// Re-exports of implementations
pub use builder::{
    DefaultModelBuilderFactory,
    BuilderRegistry,
};
pub use config::*;
pub use loader::*;
pub use registry::*;
pub use source::*;

/// Default model builder factory
pub fn default_builder_factory() -> DefaultModelBuilderFactory {
    DefaultModelBuilderFactory::new()
}

/// Default weight loader factory
pub fn default_weight_loader(format: WeightFormat) -> Result<Box<dyn WeightLoaderInterface + Send + Sync>> {
    match format {
        WeightFormat::SafeTensors => Ok(Box::new(SafeTensorsLoader::new())),
        WeightFormat::GGUF => Ok(Box::new(GGUFLoader::new())),
        WeightFormat::Pickle => Err(FerrumError::unsupported("Pickle format not supported")),
    }
}
