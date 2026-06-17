//! Model definition and configuration parsing

use crate::{qwen35_config::Qwen35TextConfig, registry::Architecture, source::ResolvedModelSource};
use ferrum_types::{
    Activation, AttentionConfig, FerrumError, ModelInfo, ModelType, NormType, Result, RopeScaling,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, warn};

/// Model definition from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// Architecture type
    pub architecture: Architecture,
    /// Hidden size (embedding dimension)
    pub hidden_size: usize,
    /// Intermediate size (FFN dimension)
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: Option<usize>,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// RoPE theta (frequency base)
    pub rope_theta: Option<f64>,
    /// RoPE scaling config
    pub rope_scaling: Option<RopeScaling>,
    /// Normalization type
    pub norm_type: NormType,
    /// Normalization epsilon
    pub norm_eps: f64,
    /// Attention configuration
    pub attention_config: AttentionConfig,
    /// Activation function
    pub activation: Activation,
    /// Extra parameters
    #[serde(flatten)]
    pub extra_params: serde_json::Value,
}

impl Default for ModelDefinition {
    fn default() -> Self {
        Self {
            architecture: Architecture::Llama,
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
            },
            activation: Activation::SiLU,
            extra_params: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

impl ModelDefinition {
    /// Convert to ModelInfo
    pub fn to_model_info(&self, model_id: impl Into<String>) -> ModelInfo {
        use ferrum_types::{DataType, Device};

        let model_id_str = model_id.into();

        // Calculate approximate parameter count
        let params = self.estimate_parameters();

        ModelInfo {
            model_id: ferrum_types::ModelId::new(model_id_str.clone()),
            model_type: ModelType::Custom(format!("{:?}", self.architecture)),
            num_parameters: params as u64,
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            vocab_size: self.vocab_size,
            max_sequence_length: self.max_position_embeddings,
            dtype: DataType::FP16, // Default, can be overridden
            device: Device::CPU,   // Default, will be set by backend
            version: None,
            license: None,
            metadata: HashMap::new(),
        }
    }

    /// Estimate parameter count
    fn estimate_parameters(&self) -> usize {
        // Rough estimation based on typical transformer architecture
        let embedding_params = self.vocab_size * self.hidden_size;
        let layer_params = self.num_hidden_layers
            * (
                // Attention: Q, K, V, O projections
                4 * self.hidden_size * self.hidden_size +
            // FFN: up, down, gate (if applicable)
            3 * self.hidden_size * self.intermediate_size +
            // Layer norms
            2 * self.hidden_size
            );
        let lm_head_params = self.vocab_size * self.hidden_size;

        embedding_params + layer_params + lm_head_params
    }
}

/// Configuration manager for loading and parsing model configs
#[derive(Debug, Default)]
pub struct ConfigManager {
    _cache: HashMap<String, ModelDefinition>,
}

impl ConfigManager {
    pub fn new() -> Self {
        Self {
            _cache: HashMap::new(),
        }
    }

    /// Load model definition from a resolved source
    pub async fn load_from_source(
        &mut self,
        source: &ResolvedModelSource,
    ) -> Result<ModelDefinition> {
        self.load_from_path(&source.local_path).await
    }

    /// Load model definition from a directory path
    pub async fn load_from_path(&mut self, path: &Path) -> Result<ModelDefinition> {
        let config_path = path.join("config.json");

        if !config_path.exists() {
            return Err(FerrumError::model(format!(
                "config.json not found in model directory: {:?}",
                path
            )));
        }

        debug!("Loading model config from: {:?}", config_path);

        let content = tokio::fs::read_to_string(&config_path)
            .await
            .map_err(|e| FerrumError::io(format!("Failed to read config.json: {}", e)))?;

        let raw_config: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| FerrumError::model(format!("Failed to parse config.json: {}", e)))?;

        self.parse_config(&raw_config)
    }

    /// Crate-internal test seam over the private parser.
    #[doc(hidden)]
    pub(crate) fn parse_config_for_tests(
        &mut self,
        raw: &serde_json::Value,
    ) -> Result<ModelDefinition> {
        self.parse_config(raw)
    }

    /// Parse config from JSON value
    fn parse_config(&mut self, raw: &serde_json::Value) -> Result<ModelDefinition> {
        let obj = raw
            .as_object()
            .ok_or_else(|| FerrumError::model("config.json root is not an object"))?;

        // Detect architecture
        let architecture = self.detect_architecture(raw)?;

        let qwen35_config =
            if matches!(architecture, Architecture::Qwen35 | Architecture::Qwen35Moe) {
                Some(Qwen35TextConfig::from_hf_config_value(raw).map_err(|err| {
                    FerrumError::model(format!("invalid Qwen3.5/Qwen3.6 config: {err}"))
                })?)
            } else {
                None
            };

        // Gemma 3 multimodal checkpoints (Gemma3ForConditionalGeneration)
        // nest the language model under `text_config`. Flatten it over the
        // root so field extraction and `extra_params` lookups (head_dim /
        // sliding_window / rope_local_base_freq / query_pre_attn_scalar /
        // rope_scaling) see the text-model values; vision_config is
        // ignored — text-only support.
        //
        // Qwen3.5/Qwen3.6 use the same nested HF shape for the text model.
        // Flatten them here too, then attach the typed W3 text config under
        // `ferrum_qwen35_text_config` so the product loader can consume the
        // exact layer/mixer/MoE shape without reparsing raw JSON or silently
        // falling back to Llama-family defaults.
        let text_merged: Option<serde_json::Map<String, serde_json::Value>> = if architecture
            == Architecture::Gemma3
            || matches!(architecture, Architecture::Qwen35 | Architecture::Qwen35Moe)
        {
            obj.get("text_config")
                .and_then(|v| v.as_object())
                .map(|tc| {
                    let mut m = obj.clone();
                    for (k, v) in tc {
                        m.insert(k.clone(), v.clone());
                    }
                    if let Some(qwen35_config) = &qwen35_config {
                        m.insert(
                            "ferrum_qwen35_text_config".to_string(),
                            serde_json::to_value(qwen35_config).unwrap_or_else(|_| {
                                serde_json::Value::Object(serde_json::Map::new())
                            }),
                        );
                    }
                    m
                })
        } else {
            None
        };
        let obj = text_merged.as_ref().unwrap_or(obj);

        // Parse common fields (CLIP stores these in text_config/vision_config)
        let text_cfg = obj.get("text_config");
        let hidden_size = obj
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                text_cfg
                    .and_then(|tc| tc.get("hidden_size"))
                    .and_then(|v| v.as_u64())
            })
            .unwrap_or(4096) as usize;

        let mut intermediate_size = obj
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .or_else(|| obj.get("ffn_dim").and_then(|v| v.as_u64()))
            .unwrap_or(11008) as usize;
        if let Some(qwen35_config) = &qwen35_config {
            intermediate_size = qwen35_config
                .dense_intermediate_size
                .or_else(|| {
                    qwen35_config
                        .moe
                        .as_ref()
                        .map(|moe| moe.moe_intermediate_size)
                })
                .unwrap_or(intermediate_size);
        }

        // CLIP models store vocab_size in text_config, not at top level
        let vocab_size = obj
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                text_cfg
                    .and_then(|tc| tc.get("vocab_size"))
                    .and_then(|v| v.as_u64())
            })
            .unwrap_or(0) as usize;

        let num_hidden_layers = obj
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .or_else(|| obj.get("n_layer").and_then(|v| v.as_u64()))
            .unwrap_or(32) as usize;

        let num_attention_heads = obj
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .or_else(|| obj.get("n_head").and_then(|v| v.as_u64()))
            .unwrap_or(32) as usize;

        let num_key_value_heads = obj
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let max_position_embeddings = obj
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .or_else(|| obj.get("n_positions").and_then(|v| v.as_u64()))
            .unwrap_or(2048) as usize;

        let rope_theta = obj
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .or_else(|| obj.get("rotary_emb_base").and_then(|v| v.as_f64()));

        // Parse RoPE scaling
        let rope_scaling = obj
            .get("rope_scaling")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        // Detect norm type
        let norm_type = if obj.get("rms_norm_eps").is_some() {
            NormType::RMSNorm
        } else {
            NormType::LayerNorm
        };

        let norm_eps = obj
            .get("rms_norm_eps")
            .or_else(|| obj.get("layer_norm_eps"))
            .or_else(|| obj.get("layer_norm_epsilon"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6);

        // Parse attention config
        let attention_bias = obj
            .get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let sliding_window = obj
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // Detect activation. Gemma family uses `hidden_activation`
        // ("gelu_pytorch_tanh") instead of `hidden_act`.
        let activation = obj
            .get("hidden_act")
            .or_else(|| obj.get("hidden_activation"))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "gelu" | "gelu_new" => Activation::GELU,
                "gelu_pytorch_tanh" => Activation::GeluTanh,
                "silu" => Activation::SiLU,
                "relu" => Activation::ReLU,
                "swish" => Activation::Swish,
                _ => {
                    warn!("Unknown activation function: {}, defaulting to SiLU", s);
                    Activation::SiLU
                }
            })
            .unwrap_or(Activation::SiLU);

        Ok(ModelDefinition {
            architecture,
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rope_theta,
            rope_scaling,
            norm_type,
            norm_eps,
            attention_config: AttentionConfig {
                attention_bias,
                sliding_window,
            },
            activation,
            // Merged view for Gemma3 (text_config flattened) so that
            // downstream extra_params lookups (head_dim, sliding_window,
            // rope_local_base_freq, query_pre_attn_scalar, ...) resolve.
            extra_params: serde_json::Value::Object(obj.clone()),
        })
    }

    /// Detect architecture from config
    fn detect_architecture(&self, config: &serde_json::Value) -> Result<Architecture> {
        let obj = config
            .as_object()
            .ok_or_else(|| FerrumError::model("config.json root is not an object"))?;

        // Try model_type field
        if let Some(model_type) = obj.get("model_type").and_then(|v| v.as_str()) {
            return Ok(Architecture::from_str(model_type));
        }

        // Try architectures array
        if let Some(architectures) = obj.get("architectures").and_then(|v| v.as_array()) {
            if let Some(arch) = architectures.first().and_then(|v| v.as_str()) {
                return Ok(Architecture::from_str(arch));
            }
        }

        warn!("Could not detect architecture, using default (Llama)");
        Ok(Architecture::Llama)
    }

    /// Infer model type from definition
    pub fn infer_model_type(&self, definition: &ModelDefinition) -> ModelType {
        ModelType::Custom(format!("{:?}", definition.architecture))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN35_ARTIFACT_ROOT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/goals/model-coverage-2026-06-12/artifacts/",
        "w3_hf_config_probe_20260617T131209Z_f97c1d6f"
    );

    fn parse_artifact_config(name: &str) -> ModelDefinition {
        let raw = std::fs::read_to_string(format!("{QWEN35_ARTIFACT_ROOT}/{name}")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let mut manager = ConfigManager::new();
        manager.parse_config_for_tests(&json).unwrap()
    }

    #[test]
    fn qwen35_dense_config_flattens_text_config_without_llama_fallback() {
        let def = parse_artifact_config("dense_min_reference.config.json");

        assert_eq!(def.architecture, Architecture::Qwen35);
        assert_eq!(def.hidden_size, 1024);
        assert_eq!(def.intermediate_size, 3584);
        assert_eq!(def.vocab_size, 248320);
        assert_eq!(def.num_hidden_layers, 24);
        assert_eq!(def.num_attention_heads, 8);
        assert_eq!(def.num_key_value_heads, Some(2));
        assert_eq!(def.max_position_embeddings, 262144);
        assert_eq!(def.norm_type, NormType::RMSNorm);
        assert_eq!(def.activation, Activation::SiLU);

        let extra = def.extra_params.as_object().unwrap();
        assert_eq!(
            extra.get("model_type").and_then(|value| value.as_str()),
            Some("qwen3_5_text")
        );
        assert_eq!(
            extra
                .get("linear_num_key_heads")
                .and_then(|value| value.as_u64()),
            Some(16)
        );
        let typed = extra
            .get("ferrum_qwen35_text_config")
            .and_then(|value| value.as_object())
            .unwrap();
        let linear_attention = typed
            .get("linear_attention")
            .and_then(|value| value.as_object())
            .unwrap();
        assert_eq!(
            linear_attention
                .get("num_key_heads")
                .and_then(|value| value.as_u64()),
            Some(16)
        );
        assert_eq!(
            typed
                .get("text_model_type")
                .and_then(|value| value.as_str()),
            Some("qwen3_5_text")
        );
        assert_eq!(
            typed
                .get("layer_types")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(24)
        );
    }

    #[test]
    fn qwen35_moe_config_preserves_shared_expert_shape() {
        let def = parse_artifact_config("moe_shared_expert_reference.config.json");

        assert_eq!(def.architecture, Architecture::Qwen35Moe);
        assert_eq!(def.hidden_size, 2048);
        assert_eq!(def.intermediate_size, 512);
        assert_eq!(def.vocab_size, 248320);
        assert_eq!(def.num_hidden_layers, 40);
        assert_eq!(def.num_attention_heads, 16);
        assert_eq!(def.num_key_value_heads, Some(2));

        let typed = def
            .extra_params
            .get("ferrum_qwen35_text_config")
            .and_then(|value| value.as_object())
            .unwrap();
        assert_eq!(
            typed
                .get("text_model_type")
                .and_then(|value| value.as_str()),
            Some("qwen3_5_moe_text")
        );
        let moe = typed
            .get("moe")
            .and_then(|value| value.as_object())
            .unwrap();
        assert_eq!(
            moe.get("num_experts").and_then(|value| value.as_u64()),
            Some(256)
        );
        assert_eq!(
            moe.get("num_experts_per_tok")
                .and_then(|value| value.as_u64()),
            Some(8)
        );
        assert_eq!(
            moe.get("shared_expert_intermediate_size")
                .and_then(|value| value.as_u64()),
            Some(512)
        );
    }
}
