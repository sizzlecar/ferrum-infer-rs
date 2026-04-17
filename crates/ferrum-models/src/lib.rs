//! Ferrum 模型层
//!
//! 该 crate 负责围绕 `ferrum-interfaces`/`ferrum-types` 定义的核心抽象
//! 提供模型定义解析、构建器与权重加载占位实现，确保上层可以在
//! 重构阶段编译。

pub mod architectures;
pub mod audio_processor;
pub mod builder;
pub mod common;
pub mod definition;
pub mod executor;
pub mod hf_download;
pub mod image_processor;
pub mod loader;
pub mod mel;
pub mod model_config;
pub mod registry;
pub mod source;
pub mod tensor_wrapper;
pub mod tokenizer;
pub mod weights;

pub use architectures::{
    BertModelWrapper, ClipModelWrapper, LlamaModelWrapper, Qwen2ModelWrapper, Qwen3ModelWrapper,
    WhisperModelWrapper,
};
pub use common::{DecoderOnlyLLM, LlmRuntimeConfig};
pub use builder::{DefaultModelBuilderFactory, SimpleModelBuilder};
pub use definition::{ConfigManager, ModelDefinition};
pub use executor::{
    BertModelExecutor, CandleModelExecutor, ClipModelExecutor, GenericModelExecutor,
    Qwen2ModelExecutor, Qwen3ModelExecutor, StubModelExecutor, TtsModelExecutor,
    WhisperModelExecutor,
};
pub use hf_download::HfDownloader;
pub use image_processor::ClipImageProcessor;
pub use loader::SafeTensorsLoader;
pub use registry::{
    Architecture, DefaultModelRegistry, ModelAlias, ModelDiscoveryEntry, ModelFormatType,
};
pub use source::{
    DefaultModelSourceResolver, ModelFormat, ModelSourceConfig, ModelSourceResolver,
    ResolvedModelSource,
};
pub use tensor_wrapper::CandleTensorWrapper;
pub use tokenizer::{TokenizerFactory, TokenizerHandle};
pub use weights::{default_weight_loader, StubWeightLoader, WeightLoaderHandle};

pub use ferrum_interfaces::{ModelBuilder, ModelExecutor, WeightLoader};
pub use ferrum_types::{
    Activation, AttentionConfig, ModelConfig, ModelInfo, ModelType, NormType, Result, RopeScaling,
};
