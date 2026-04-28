//! Ferrum 模型层
//!
//! 该 crate 负责围绕 `ferrum-interfaces`/`ferrum-types` 定义的核心抽象
//! 提供模型定义解析、构建器与权重加载占位实现，确保上层可以在
//! 重构阶段编译。

// TTS / Whisper架构里存在大量在不同调用路径下"当下未使用"的字段/方法
// (e.g. 推理期不走的 layers、只在特定 feature 下用的导入)。抑制到 crate 级，
// 新写代码的真实警告由 engine / sampler / scheduler 等下游 crate 的 CI 门把守。
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    unused_parens,
    unused_assignments
)]

pub mod architectures;
pub mod audio_processor;
pub mod builder;
pub mod common;
pub mod definition;
pub mod executor;
pub mod gguf_config;
pub mod hf_download;
pub mod image_processor;
pub mod loader;
pub mod mel;
pub mod models;
pub mod moe_config;
pub mod registry;
pub mod source;
pub mod tensor_wrapper;
pub mod tokenizer;
pub mod weights;

pub use architectures::{BertModelWrapper, ClipModelWrapper, WhisperModelWrapper};
pub use builder::{DefaultModelBuilderFactory, SimpleModelBuilder};
pub use common::{DecoderOnlyLLM, LlmRuntimeConfig};
pub use definition::{ConfigManager, ModelDefinition};
pub use executor::{
    BertModelExecutor, ClipModelExecutor, LlmExecutor, StubModelExecutor, TtsModelExecutor,
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
