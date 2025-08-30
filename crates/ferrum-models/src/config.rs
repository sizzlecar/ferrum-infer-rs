//! Model configuration management
//!
//! This module provides utilities for loading and managing model configurations
//! from various sources without depending on specific ML frameworks.

use crate::traits::{AbstractModelConfig, Activation, Architecture, AttentionConfig, NormType};
use ferrum_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration manager for models
pub struct ConfigManager {
    /// Cached configurations
    configs: HashMap<String, AbstractModelConfig>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Load configuration from file
    pub async fn load_from_file(&mut self, path: &Path) -> Result<AbstractModelConfig> {
        let content = tokio::fs::read_to_string(path).await?;

        let config: RawConfig = serde_json::from_str(&content)
            .map_err(|e| Error::configuration(format!("Failed to parse config: {}", e)))?;

        self.convert_to_abstract(config)
    }

    /// Load configuration from HuggingFace model ID
    pub async fn load_from_huggingface(&mut self, _model_id: &str) -> Result<AbstractModelConfig> {
        // This would use the HuggingFace API to fetch config.json
        // For now, return a placeholder
        Err(Error::unsupported(
            "HuggingFace loading not yet implemented",
        ))
    }

    /// Convert raw config to abstract config
    fn convert_to_abstract(&self, raw: RawConfig) -> Result<AbstractModelConfig> {
        let architecture = self.detect_architecture(&raw)?;
        let norm_type = self.detect_norm_type(&raw);
        let activation = self.detect_activation(&raw);
        let extra_params =
            serde_json::to_value(&raw).unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        Ok(AbstractModelConfig {
            architecture,
            hidden_size: raw.hidden_size.unwrap_or(4096),
            intermediate_size: raw.intermediate_size.unwrap_or(11008),
            vocab_size: raw.vocab_size.unwrap_or(32000),
            num_hidden_layers: raw.num_hidden_layers.unwrap_or(32),
            num_attention_heads: raw.num_attention_heads.unwrap_or(32),
            num_key_value_heads: raw.num_key_value_heads,
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(4096),
            rope_theta: raw.rope_theta,
            rope_scaling: raw.rope_scaling.map(|s| crate::RopeScaling {
                scaling_type: s.scaling_type,
                factor: s.factor,
            }),
            norm_type,
            norm_eps: raw.rms_norm_eps.or(raw.layer_norm_eps).unwrap_or(1e-5),
            attention_config: AttentionConfig {
                attention_bias: raw.attention_bias.unwrap_or(false),
                sliding_window: raw.sliding_window,
                use_flash_attention: false,
                use_paged_attention: true,
            },
            activation,
            extra_params,
        })
    }

    /// Detect architecture from raw config
    fn detect_architecture(&self, raw: &RawConfig) -> Result<Architecture> {
        if let Some(arch) = &raw.architectures {
            if !arch.is_empty() {
                return self.parse_architecture(&arch[0]);
            }
        }

        if let Some(model_type) = &raw.model_type {
            return self.parse_architecture(model_type);
        }

        Err(Error::configuration("Cannot detect model architecture"))
    }

    /// Parse architecture string
    fn parse_architecture(&self, arch_str: &str) -> Result<Architecture> {
        let arch_lower = arch_str.to_lowercase();

        if arch_lower.contains("llama") {
            if arch_lower.contains("3") {
                Ok(Architecture::Llama3)
            } else if arch_lower.contains("2") {
                Ok(Architecture::Llama2)
            } else {
                Ok(Architecture::Llama)
            }
        } else if arch_lower.contains("mistral") {
            Ok(Architecture::Mistral)
        } else if arch_lower.contains("mixtral") {
            Ok(Architecture::Mixtral)
        } else if arch_lower.contains("qwen2") {
            Ok(Architecture::Qwen2)
        } else if arch_lower.contains("qwen") {
            Ok(Architecture::Qwen)
        } else if arch_lower.contains("phi") {
            Ok(Architecture::Phi)
        } else if arch_lower.contains("gemma") {
            Ok(Architecture::Gemma)
        } else {
            Ok(Architecture::Custom(arch_str.to_string()))
        }
    }

    /// Detect normalization type
    fn detect_norm_type(&self, raw: &RawConfig) -> NormType {
        if raw.rms_norm_eps.is_some() {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        }
    }

    /// Detect activation function
    fn detect_activation(&self, raw: &RawConfig) -> Activation {
        if let Some(act) = &raw.hidden_act {
            match act.to_lowercase().as_str() {
                "silu" => Activation::SiLU,
                "gelu" => Activation::GELU,
                "relu" => Activation::ReLU,
                "swish" => Activation::Swish,
                _ => Activation::SiLU, // Default
            }
        } else {
            Activation::SiLU // Default for most modern models
        }
    }
}

/// Raw configuration format (HuggingFace compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawConfig {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    max_position_embeddings: Option<usize>,
    rope_theta: Option<f32>,
    rope_scaling: Option<RawRopeScaling>,
    rms_norm_eps: Option<f64>,
    layer_norm_eps: Option<f64>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    sliding_window: Option<usize>,

    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawRopeScaling {
    #[serde(rename = "type")]
    scaling_type: String,
    factor: f32,
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection() {
        let manager = ConfigManager::new();

        assert!(matches!(
            manager.parse_architecture("LlamaForCausalLM").unwrap(),
            Architecture::Llama
        ));

        assert!(matches!(
            manager.parse_architecture("MistralForCausalLM").unwrap(),
            Architecture::Mistral
        ));
    }
}
