//! Model configuration management
//!
//! This module provides utilities for loading and managing model configurations
//! from various sources without depending on specific ML frameworks.
//! Extended to support HF/Mistral/GGUF formats with compatibility patches.

use crate::traits::{AbstractModelConfig, Activation, Architecture, AttentionConfig, NormType};
use crate::source::{ModelFormat, ResolvedModelSource};
use ferrum_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, warn};

/// Configuration manager for models
/// Extended to support HF/Mistral/GGUF formats and compatibility patches
pub struct ConfigManager {
    /// Cached configurations
    configs: HashMap<String, AbstractModelConfig>,
    /// Whether to enable compatibility patches
    enable_compatibility_patches: bool,
    /// Supported formats
    supported_formats: Vec<ModelFormat>,
}

impl ConfigManager {
    /// Create a new configuration manager with compatibility patches enabled
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            enable_compatibility_patches: true,
            supported_formats: vec![
                ModelFormat::HuggingFace,
                ModelFormat::Mistral,
                ModelFormat::GGUF,
            ],
        }
    }
    
    /// Create configuration manager without compatibility patches
    pub fn without_compatibility_patches() -> Self {
        Self {
            configs: HashMap::new(),
            enable_compatibility_patches: false,
            supported_formats: vec![
                ModelFormat::HuggingFace,
                ModelFormat::Mistral,
                ModelFormat::GGUF,
            ],
        }
    }
    
    /// Check if a format is supported
    pub fn supports_format(&self, format: &ModelFormat) -> bool {
        self.supported_formats.contains(format)
    }

    /// Load configuration from file
    pub async fn load_from_file(&mut self, path: &Path) -> Result<AbstractModelConfig> {
        let content = tokio::fs::read_to_string(path).await?;

        let config: RawConfig = serde_json::from_str(&content)
            .map_err(|e| Error::configuration(format!("Failed to parse config: {}", e)))?;

        self.convert_to_abstract(config)
    }

    /// Load configuration from resolved model source (New vLLM-inspired method)
    pub async fn load_from_source(&mut self, source: &ResolvedModelSource) -> Result<AbstractModelConfig> {
        let cache_key = format!("{}:{}", source.model_id, source.revision.as_deref().unwrap_or("main"));
        
        // Check cache first
        if let Some(cached_config) = self.configs.get(&cache_key) {
            debug!("Using cached config for {}", cache_key);
            return Ok(cached_config.clone());
        }
        
        let config = match source.format {
            ModelFormat::HuggingFace => self.load_hf_config(&source.local_path).await?,
            ModelFormat::Mistral => self.load_mistral_config(&source.local_path).await?,
            ModelFormat::GGUF => self.load_gguf_config(&source.local_path).await?,
            ModelFormat::Auto => {
                let detected_format = self.detect_format(&source.local_path)?;
                match detected_format {
                    ModelFormat::HuggingFace => self.load_hf_config(&source.local_path).await?,
                    ModelFormat::Mistral => self.load_mistral_config(&source.local_path).await?,
                    ModelFormat::GGUF => self.load_gguf_config(&source.local_path).await?,
                    ModelFormat::Auto => return Err(Error::configuration("Cannot detect model format")),
                }
            }
        };
        
        // Cache the result
        self.configs.insert(cache_key, config.clone());
        Ok(config)
    }

    /// Load configuration from HuggingFace model ID
    pub async fn load_from_huggingface(&mut self, _model_id: &str) -> Result<AbstractModelConfig> {
        // This would use the ModelSourceResolver to first resolve the model
        // For now, return a placeholder
        Err(Error::unsupported(
            "HuggingFace loading not yet implemented - use load_from_source with ModelSourceResolver",
        ))
    }

    /// Detect model format from directory
    fn detect_format(&self, path: &Path) -> Result<ModelFormat> {
        if path.join("config.json").exists() {
            Ok(ModelFormat::HuggingFace)
        } else if path.join("params.json").exists() {
            Ok(ModelFormat::Mistral)
        } else if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            Ok(ModelFormat::GGUF)
        } else {
            Err(Error::configuration("Cannot detect model format"))
        }
    }
    
    /// Load HuggingFace configuration with compatibility patches
    async fn load_hf_config(&self, path: &Path) -> Result<AbstractModelConfig> {
        let config_path = path.join("config.json");
        debug!("Loading HF config from: {:?}", config_path);
        
        let content = tokio::fs::read_to_string(&config_path).await
            .map_err(|e| Error::configuration(format!("Failed to read config file: {}", e)))?;
            
        let mut raw: RawConfig = serde_json::from_str(&content)
            .map_err(|e| Error::configuration(format!("Failed to parse config: {}", e)))?;
        
        // Apply compatibility patches if enabled
        if self.enable_compatibility_patches {
            self.apply_hf_compatibility_patches(&mut raw).await?;
        }
        
        self.convert_to_abstract(raw)
    }
    
    /// Apply HuggingFace compatibility patches
    async fn apply_hf_compatibility_patches(&self, _raw: &mut RawConfig) -> Result<()> {
        debug!("Applying HF compatibility patches");
        // TODO: Implement specific patches (RoPE normalization, architecture mapping, etc.)
        Ok(())
    }
    
    /// Load Mistral configuration from params.json
    async fn load_mistral_config(&self, path: &Path) -> Result<AbstractModelConfig> {
        let params_path = path.join("params.json");
        debug!("Loading Mistral config from: {:?}", params_path);
        
        // For now, fall back to treating as HF format
        // In production, this would parse params.json format
        warn!("Mistral config loading is simplified - falling back to HF format");
        self.load_hf_config(path).await
    }
    
    /// Load GGUF configuration
    async fn load_gguf_config(&self, path: &Path) -> Result<AbstractModelConfig> {
        debug!("Loading GGUF config from: {:?}", path);
        
        // Create basic config based on filename
        let file_name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
            
        let architecture = if file_name.contains("llama") {
            Architecture::Llama
        } else if file_name.contains("mistral") {
            Architecture::Mistral
        } else if file_name.contains("qwen") {
            Architecture::Qwen
        } else {
            Architecture::Custom("gguf".to_string())
        };
        
        Ok(AbstractModelConfig {
            architecture,
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            max_position_embeddings: 2048,
            rope_theta: Some(10000.0),
            rope_scaling: None,
            norm_type: NormType::RMSNorm,
            norm_eps: 1e-6,
            attention_config: AttentionConfig {
                attention_bias: false,
                sliding_window: None,
                use_flash_attention: false,
                use_paged_attention: true,
            },
            activation: Activation::SiLU,
            extra_params: serde_json::Value::Object(serde_json::Map::new()),
        })
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
    use crate::source::ModelFormat;

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

    #[tokio::test]
    async fn test_load_hf_config_basic() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();

        // Minimal but valid HF config
        let cfg_content = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6
        }"#;
        tokio::fs::write(dir.join("config.json"), cfg_content)
            .await
            .unwrap();

        let manager = ConfigManager::new();
        let cfg = manager.load_hf_config(dir).await.unwrap();

        assert!(matches!(cfg.architecture, Architecture::Llama));
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!(matches!(cfg.norm_type, NormType::RMSNorm));
        assert!(matches!(cfg.activation, Activation::SiLU));
    }

    #[test]
    fn test_detect_format_prefers_hf_over_mistral() {
        let temp = tempfile::TempDir::new().unwrap();
        let dir = temp.path();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("params.json"), "{}").unwrap();

        let manager = ConfigManager::new();
        let detected = manager.detect_format(dir).unwrap();
        assert!(matches!(detected, ModelFormat::HuggingFace));
    }

    #[test]
    fn test_convert_to_abstract_defaults() {
        use std::collections::HashMap;

        let manager = ConfigManager::new();
        let raw = RawConfig {
            architectures: None,
            model_type: Some("llama".to_string()),
            hidden_size: None,
            intermediate_size: None,
            vocab_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            max_position_embeddings: None,
            rope_theta: None,
            rope_scaling: None,
            rms_norm_eps: None,
            layer_norm_eps: None,
            hidden_act: None,
            attention_bias: None,
            sliding_window: None,
            extra: HashMap::new(),
        };

        let cfg = manager.convert_to_abstract(raw).unwrap();

        assert!(matches!(cfg.architecture, Architecture::Llama));
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!(matches!(cfg.activation, Activation::SiLU));
    }

    #[tokio::test]
    async fn test_load_gguf_config_infers_architecture() {
        let manager = ConfigManager::new();
        let gguf_path = std::path::Path::new("/tmp/nonexistent/llama-7b.Q4_0.gguf");
        let cfg = manager.load_gguf_config(gguf_path).await.unwrap();
        assert!(matches!(cfg.architecture, Architecture::Llama));
    }
}
