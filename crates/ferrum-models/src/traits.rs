//! Abstract model traits and interfaces
//!
//! This module defines the abstract interfaces for different model architectures
//! without depending on any specific ML framework implementation.

use async_trait::async_trait;
use ferrum_core::{ModelConfig, Result};
use serde::{Deserialize, Serialize};

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
pub struct AbstractModelConfig {
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

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    pub use_flash_attention: bool,
    pub use_paged_attention: bool,
}

/// Abstract model builder trait
#[async_trait]
pub trait ModelBuilder: Send + Sync {
    /// Build a model from configuration
    async fn build(
        &self,
        config: &AbstractModelConfig,
        model_config: &ModelConfig,
    ) -> Result<Box<dyn ferrum_core::Model>>;

    /// Load model weights
    async fn load_weights(
        &self,
        model: &mut dyn ferrum_core::Model,
        weights_path: &str,
    ) -> Result<()>;

    /// Get supported architectures
    fn supported_architectures(&self) -> Vec<Architecture>;
}

/// Model registry trait
pub trait ModelRegistry: Send + Sync {
    /// Register a model builder
    fn register_builder(&mut self, builder: Box<dyn ModelBuilder>);

    /// Get builder for architecture
    fn get_builder(&self, architecture: &Architecture) -> Option<&dyn ModelBuilder>;

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

/// Model conversion trait for different formats
#[async_trait]
pub trait ModelConverter: Send + Sync {
    /// Convert from HuggingFace format
    async fn from_huggingface(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> Result<AbstractModelConfig>;

    /// Convert from GGUF format
    async fn from_gguf(&self, path: &str) -> Result<AbstractModelConfig>;

    /// Convert from SafeTensors format
    async fn from_safetensors(&self, path: &str) -> Result<AbstractModelConfig>;
}
