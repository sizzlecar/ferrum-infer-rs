//! Model configuration management
//!
//! This module provides utilities for loading and managing model configurations
//! from various sources without depending on specific ML frameworks.
//! Extended to support HF/Mistral/GGUF formats with compatibility patches.

use crate::traits::{ModelDefinition, Activation, Architecture, AttentionConfig, NormType};
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
    configs: HashMap<String, ModelDefinition>,
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
    pub async fn load_from_file(&mut self, path: &Path) -> Result<ModelDefinition> {
        let content = tokio::fs::read_to_string(path).await?;

        let config: RawConfig = serde_json::from_str(&content)
            .map_err(|e| Error::configuration(format!("Failed to parse config: {}", e)))?;

        self.convert_to_abstract(config)
    }

    /// Load configuration from resolved model source (New vLLM-inspired method)
    pub async fn load_from_source(&mut self, source: &ResolvedModelSource) -> Result<ModelDefinition> {
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
    pub async fn load_from_huggingface(&mut self, _model_id: &str) -> Result<ModelDefinition> {
        // This would use the ModelSourceResolver to first resolve the model
        // For now, return a placeholder
        Err(Error::unsupported(
            "HuggingFace loading not yet implemented - use load_from_source with ModelSourceResolver",
        ))
    }

    /// Detect model format from directory (使用统一的工具函数)
    fn detect_format(&self, path: &Path) -> Result<ModelFormat> {
        let format = crate::source::detect_format(path);
        match format {
            ModelFormat::Auto => Err(Error::configuration("Cannot detect model format")),
            other => Ok(other),
        }
    }
    
    /// Load HuggingFace configuration with compatibility patches
    async fn load_hf_config(&self, path: &Path) -> Result<ModelDefinition> {
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
    async fn apply_hf_compatibility_patches(&self, raw: &mut RawConfig) -> Result<()> {
        debug!("Applying HF compatibility patches");
        
        // 1. RoPE 参数归一化
        self.normalize_rope_parameters(raw);
        
        // 2. KV heads 缺省推断
        self.infer_missing_kv_heads(raw);
        
        // 3. 激活函数别名映射
        self.normalize_activation_function(raw);
        
        // 4. Norm 类型别名映射
        self.normalize_norm_config(raw);
        
        // 5. 架构别名映射
        self.normalize_architecture_config(raw);
        
        debug!("HF compatibility patches applied successfully");
        Ok(())
    }
    
    /// 归一化 RoPE 参数
    fn normalize_rope_parameters(&self, raw: &mut RawConfig) {
        // 标准化 RoPE theta 参数
        if raw.rope_theta.is_none() {
            // 检查别名字段
            if let Some(theta_value) = raw.extra.get("rope_theta").and_then(|v| v.as_f64()) {
                raw.rope_theta = Some(theta_value as f32);
            } else if let Some(theta_value) = raw.extra.get("rotary_emb_base").and_then(|v| v.as_f64()) {
                raw.rope_theta = Some(theta_value as f32);
            } else {
                // 根据架构设置默认值
                if let Some(arch) = &raw.architectures {
                    if !arch.is_empty() {
                        let arch_str = arch[0].to_lowercase();
                        if arch_str.contains("llama") {
                            raw.rope_theta = Some(10000.0);
                        } else if arch_str.contains("mistral") {
                            raw.rope_theta = Some(10000.0);
                        } else if arch_str.contains("qwen") {
                            raw.rope_theta = Some(1000000.0);
                        }
                    }
                }
            }
        }
        
        // 标准化 RoPE scaling 配置
        if raw.rope_scaling.is_none() {
            if let Some(scaling_obj) = raw.extra.get("rope_scaling").and_then(|v| v.as_object()) {
                if let (Some(scaling_type), Some(factor)) = (
                    scaling_obj.get("type").and_then(|v| v.as_str()),
                    scaling_obj.get("factor").and_then(|v| v.as_f64())
                ) {
                    raw.rope_scaling = Some(RawRopeScaling {
                        scaling_type: scaling_type.to_string(),
                        factor: factor as f32,
                    });
                }
            }
        }
        
        debug!("RoPE parameters normalized: theta={:?}, scaling={:?}", raw.rope_theta, raw.rope_scaling);
    }
    
    /// 推断缺失的 KV heads 配置
    fn infer_missing_kv_heads(&self, raw: &mut RawConfig) {
        if raw.num_key_value_heads.is_none() && raw.num_attention_heads.is_some() {
            let num_heads = raw.num_attention_heads.unwrap();
            
            // 检查是否有 GQA 或 MQA 的配置提示
            if let Some(gqa_groups) = raw.extra.get("num_key_value_heads").and_then(|v| v.as_u64()) {
                raw.num_key_value_heads = Some(gqa_groups as usize);
            } else if let Some(multi_query) = raw.extra.get("multi_query").and_then(|v| v.as_bool()) {
                // MQA: 所有 heads 共享一个 key-value pair
                raw.num_key_value_heads = Some(if multi_query { 1 } else { num_heads });
            } else {
                // 根据架构推断默认行为
                if let Some(arch) = &raw.architectures {
                    if !arch.is_empty() {
                        let arch_str = arch[0].to_lowercase();
                        if arch_str.contains("llama") {
                            // Llama 通常不使用 GQA (除非明确指定)
                            raw.num_key_value_heads = Some(num_heads);
                        } else if arch_str.contains("mistral") {
                            // Mistral 7B 使用 GQA
                            raw.num_key_value_heads = Some(8);
                        } else if arch_str.contains("mixtral") {
                            // Mixtral 使用 GQA
                            raw.num_key_value_heads = Some(8);
                        } else {
                            // 默认情况：与 attention heads 相同
                            raw.num_key_value_heads = Some(num_heads);
                        }
                    }
                } else {
                    // 无架构信息时默认与 attention heads 相同
                    raw.num_key_value_heads = Some(num_heads);
                }
            }
        }
        
        debug!("KV heads inferred: num_attention_heads={:?}, num_key_value_heads={:?}", 
               raw.num_attention_heads, raw.num_key_value_heads);
    }
    
    /// 归一化激活函数配置
    fn normalize_activation_function(&self, raw: &mut RawConfig) {
        if raw.hidden_act.is_none() {
            // 检查别名字段
            if let Some(act_fn) = raw.extra.get("activation_function").and_then(|v| v.as_str()) {
                raw.hidden_act = Some(act_fn.to_string());
            } else if let Some(act_fn) = raw.extra.get("activation").and_then(|v| v.as_str()) {
                raw.hidden_act = Some(act_fn.to_string());
            } else {
                // 根据架构设置默认激活函数
                if let Some(arch) = &raw.architectures {
                    if !arch.is_empty() {
                        let arch_str = arch[0].to_lowercase();
                        if arch_str.contains("llama") || arch_str.contains("mistral") {
                            raw.hidden_act = Some("silu".to_string());
                        } else if arch_str.contains("qwen") {
                            raw.hidden_act = Some("silu".to_string());
                        } else if arch_str.contains("phi") {
                            raw.hidden_act = Some("gelu_new".to_string());
                        } else if arch_str.contains("gemma") {
                            raw.hidden_act = Some("gelu".to_string());
                        } else {
                            raw.hidden_act = Some("silu".to_string());
                        }
                    }
                }
            }
        }
        
        // 标准化激活函数名称
        if let Some(ref mut act) = raw.hidden_act {
            let act_lower = act.to_lowercase();
            let normalized_act = match act_lower.as_str() {
                "swish" => "silu",
                "gelu_new" | "gelu_pytorch_tanh" => "gelu",
                other => other,
            };
            *act = normalized_act.to_string();
        }
        
        debug!("Activation function normalized: {:?}", raw.hidden_act);
    }
    
    /// 归一化 Norm 配置
    fn normalize_norm_config(&self, raw: &mut RawConfig) {
        // 标准化 eps 参数
        if raw.rms_norm_eps.is_none() && raw.layer_norm_eps.is_none() {
            if let Some(eps_value) = raw.extra.get("norm_eps").and_then(|v| v.as_f64()) {
                // 判断使用哪种 norm 类型
                if let Some(arch) = &raw.architectures {
                    if !arch.is_empty() {
                        let arch_str = arch[0].to_lowercase();
                        if arch_str.contains("llama") || arch_str.contains("mistral") {
                            raw.rms_norm_eps = Some(eps_value);
                        } else {
                            raw.layer_norm_eps = Some(eps_value);
                        }
                    }
                }
            }
        }
        
        // 设置默认 eps 值
        if raw.rms_norm_eps.is_none() && raw.layer_norm_eps.is_none() {
            if let Some(arch) = &raw.architectures {
                if !arch.is_empty() {
                    let arch_str = arch[0].to_lowercase();
                    if arch_str.contains("llama") || arch_str.contains("mistral") {
                        raw.rms_norm_eps = Some(1e-6);
                    } else {
                        raw.layer_norm_eps = Some(1e-5);
                    }
                }
            }
        }
        
        debug!("Norm config normalized: rms_eps={:?}, layer_eps={:?}", raw.rms_norm_eps, raw.layer_norm_eps);
    }
    
    /// 归一化架构配置
    fn normalize_architecture_config(&self, raw: &mut RawConfig) {
        // 如果没有 architectures 字段但有 model_type，尝试转换
        if raw.architectures.is_none() || raw.architectures.as_ref().map_or(true, |v| v.is_empty()) {
            if let Some(model_type) = &raw.model_type {
                raw.architectures = Some(vec![format!("{}ForCausalLM", model_type)]);
            }
        }
        
        // 标准化架构名称
        if let Some(ref mut archs) = raw.architectures {
            for arch in archs.iter_mut() {
                // 将常见的变体标准化为主要架构名称
                let normalized = match arch.to_lowercase().as_str() {
                    s if s.contains("llama") => {
                        if s.contains("3") {
                            "LlamaForCausalLM"
                        } else if s.contains("2") {
                            "LlamaForCausalLM"
                        } else {
                            "LlamaForCausalLM"
                        }
                    },
                    s if s.contains("mistral") && !s.contains("mixtral") => "MistralForCausalLM",
                    s if s.contains("mixtral") => "MixtralForCausalLM",
                    s if s.contains("qwen2") => "Qwen2ForCausalLM",
                    s if s.contains("qwen") => "QwenForCausalLM",
                    s if s.contains("phi") => "PhiForCausalLM",
                    s if s.contains("gemma") => "GemmaForCausalLM",
                    _ => arch.as_str(),
                };
                *arch = normalized.to_string();
            }
        }
        
        debug!("Architecture config normalized: {:?}", raw.architectures);
    }
    
    /// Load Mistral configuration from params.json
    async fn load_mistral_config(&self, path: &Path) -> Result<ModelDefinition> {
        let params_path = path.join("params.json");
        debug!("Loading Mistral config from: {:?}", params_path);
        
        // For now, fall back to treating as HF format
        // In production, this would parse params.json format
        warn!("Mistral config loading is simplified - falling back to HF format");
        self.load_hf_config(path).await
    }
    
    /// Load GGUF configuration
    async fn load_gguf_config(&self, path: &Path) -> Result<ModelDefinition> {
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
        
        Ok(ModelDefinition {
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
                // use_flash_attention and use_paged_attention moved to ferrum_core::ModelConfig
            },
            activation: Activation::SiLU,
            extra_params: serde_json::Value::Object(serde_json::Map::new()),
        })
    }
    
    /// Convert raw config to abstract config
    fn convert_to_abstract(&self, raw: RawConfig) -> Result<ModelDefinition> {
        let architecture = self.detect_architecture(&raw)?;
        let norm_type = self.detect_norm_type(&raw);
        let activation = self.detect_activation(&raw);
        let extra_params =
            serde_json::to_value(&raw).unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        Ok(ModelDefinition {
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
                // use_flash_attention and use_paged_attention moved to ferrum_core::ModelConfig
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
