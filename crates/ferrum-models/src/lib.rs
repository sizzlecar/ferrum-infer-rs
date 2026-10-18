//! Model definitions, weight sources and language-model execution for Ferrum.

// Legacy language-model and optional-backend code share this warning scope.
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    unused_parens,
    unused_assignments
)]

pub mod common;
pub mod definition;
pub mod deltanet_s1;
pub mod executor;
pub mod gguf_config;
pub mod gguf_engine_loader;
pub mod hf_download;
pub mod legacy_capabilities;
pub mod loader;
pub mod lora;
pub mod models;
pub mod moe;
pub mod qwen35_config;
#[cfg(any(test, feature = "test-support"))]
pub mod qwen35_s1;
pub mod qwen35_weights;
pub mod registry;
pub mod source;
pub mod tensor_wrapper;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod tokenizer;
pub mod vnext;
pub mod weight_format;

pub use common::{DecoderOnlyLLM, LlmRuntimeConfig};
pub use definition::{ConfigManager, ModelDefinition};
pub use executor::{LlmExecutor, StubModelExecutor};
pub use hf_download::HfDownloader;
pub use loader::SafeTensorsLoader;
pub use lora::{
    default_lora_model_id, load_runtime_lora_adapter, load_startup_lora_adapter,
    load_startup_lora_adapters, render_lora_model_id, ActiveLoraAdapter, LoraAdapterConfig,
    RuntimeLoraAdapter, StartupLoraAdapter, StartupLoraSpec,
};
pub use registry::{
    Architecture, DefaultModelRegistry, ModelAlias, ModelDiscoveryEntry, ModelFormatType,
};
pub use source::{
    DefaultModelSourceResolver, ModelFormat, ModelSourceConfig, ModelSourceResolver,
    ResolvedModelSource,
};
pub use tensor_wrapper::CandleTensorWrapper;
pub use tokenizer::{TokenizerFactory, TokenizerHandle};

pub use ferrum_execution::ModelExecutor;
pub use ferrum_types::{
    Activation, AttentionConfig, ModelConfig, ModelInfo, ModelType, NormType, Result, RopeScaling,
};
