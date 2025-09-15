//! Abstract model traits and interfaces
//!
//! This module defines the abstract interfaces for different model architectures
//! without depending on any specific ML framework implementation.

use async_trait::async_trait;
use ferrum_core::{RuntimeConfig, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Model architecture types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Llama2,
    Llama3,
    Mistral,
    Mixtral,
    Qwen,
    Qwen2,
    Phi,
    Gemma,
    Custom(String),
}

/// Abstract model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// Model architecture
    pub architecture: Architecture,

    /// Model size parameters
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,

    /// Positional encoding
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<RopeScaling>,

    /// Normalization
    pub norm_type: NormType,
    pub norm_eps: f64,

    /// Attention configuration
    pub attention_config: AttentionConfig,

    /// Activation function
    pub activation: Activation,

    /// Additional architecture-specific parameters
    pub extra_params: serde_json::Value,
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub factor: f32,
}

/// Normalization type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}

/// Activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    GELU,
    SiLU,
    ReLU,
    Swish,
}

/// Attention configuration (Definition layer - architecture-specific)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    // Note: use_flash_attention and use_paged_attention moved to ferrum_core::ModelConfig
    // as they are runtime implementation choices, not architectural properties
}

/// Abstract model builder trait
#[async_trait]
pub trait ModelBuilder: Send + Sync {
    /// Build a model from configuration
    async fn build(
        &self,
        config: &ModelDefinition,
        model_config: &RuntimeConfig,
    ) -> Result<Box<dyn ferrum_core::Model>>;

    /// Load model weights from resolved source (replaces weights_path for better integration)
    async fn load_weights(
        &self,
        model: &mut dyn ferrum_core::Model,
        source: &crate::source::ResolvedModelSource,
    ) -> Result<()>;

    /// Get supported architectures
    fn supported_architectures(&self) -> Vec<Architecture>;
}

/// Model registry trait
pub trait ModelRegistry: Send + Sync {
    /// Register a model builder
    fn register_builder(&mut self, builder: Box<dyn ModelBuilder>);

    /// Get builder for architecture as Arc (recommended)
    fn get_builder_arc(&self, architecture: &Architecture) -> Option<Arc<dyn ModelBuilder>>;

    /// List all supported architectures
    fn supported_architectures(&self) -> Vec<Architecture>;
}

/// Abstract tokenizer interface
#[async_trait]
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special tokens
    fn special_tokens(&self) -> &SpecialTokens;
}

/// Special tokens configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}

/// 模型源解析器 trait（基于vLLM的ModelSourceResolver设计）
#[async_trait]
pub trait ModelSourceResolver: Send + Sync {
    /// 解析模型源（HF模型ID或本地路径）到本地路径
    async fn resolve(
        &self,
        id_or_path: &str,
        revision: Option<&str>,
    ) -> Result<crate::source::ResolvedModelSource>;

    /// 是否支持离线模式
    fn supports_offline(&self) -> bool;

    /// 获取缓存信息
    fn get_cache_info(&self, model_id: &str, revision: Option<&str>) -> Option<std::path::PathBuf>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_struct_is_send_sync_clone() {
        fn assert_traits<T: Send + Sync + Clone>() {}
        assert_traits::<SpecialTokens>();
    }

    #[test]
    fn test_architecture_enum_variants() {
        let a = Architecture::Custom("foo".into());
        match a {
            Architecture::Custom(s) => assert_eq!(s, "foo"),
            _ => panic!("unexpected variant"),
        }
    }
}

/// 模型格式枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    HuggingFace { model_id: String, revision: Option<String> },
    Gguf { path: String },
    SafeTensors { path: String },
    Custom { format_name: String, source: String },
}

/// 模型转换器 trait - 使用统一接口支持可扩展的格式
#[async_trait]
pub trait ModelConverter: Send + Sync {
    /// 从指定格式转换为 ModelDefinition
    async fn convert(&self, format: &ModelFormat) -> Result<ModelDefinition>;

    /// 获取此转换器支持的格式列表
    fn supported_formats(&self) -> Vec<String>;
}

/// 格式特定的转换器注册表
pub trait ModelConverterRegistry: Send + Sync {
    /// 注册特定格式的转换器
    fn register_converter(&mut self, format_name: &str, converter: Box<dyn ModelConverter>);
    
    /// 获取指定格式的转换器
    fn get_converter(&self, format_name: &str) -> Option<Arc<dyn ModelConverter>>;
    
    /// 列出所有支持的格式
    fn supported_formats(&self) -> Vec<String>;
}
