//! Qwen3.5 / Qwen3.6 HF `text_config` parser.
//!
//! W3 models are not Llama-family attention-only decoders: they mix
//! Gated-DeltaNet-style linear attention with full attention, and the MoE
//! variants add routed experts plus a shared expert. This module keeps that
//! shape explicit before the product loader/model path is wired.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35LinearAttentionConfig {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35MoeTextConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35TextConfig {
    pub top_level_model_type: Option<String>,
    pub text_model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<Qwen35LayerType>,
    pub linear_attention: Qwen35LinearAttentionConfig,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub dense_intermediate_size: Option<usize>,
    pub moe: Option<Qwen35MoeTextConfig>,
}

impl Qwen35TextConfig {
    pub fn from_hf_config_str(raw: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(raw).map_err(|err| err.to_string())?;
        Self::from_hf_config_value(&value)
    }

    pub fn from_hf_config_value(value: &Value) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or_else(|| "HF config root must be a JSON object".to_string())?;
        let top_level_model_type = obj
            .get("model_type")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let text_config = obj
            .get("text_config")
            .unwrap_or(value)
            .as_object()
            .ok_or_else(|| "HF config text_config must be a JSON object".to_string())?;
        let text_model_type = required_string(text_config, "model_type")?;
        if text_model_type != "qwen3_5_text" && text_model_type != "qwen3_5_moe_text" {
            return Err(format!(
                "unsupported Qwen3.5 text model_type {text_model_type:?}"
            ));
        }
        let hidden_size = required_usize(text_config, "hidden_size")?;
        let num_hidden_layers = required_usize(text_config, "num_hidden_layers")?;
        let layer_types = parse_layer_types(text_config, num_hidden_layers)?;
        let linear_attention = Qwen35LinearAttentionConfig {
            num_key_heads: required_usize(text_config, "linear_num_key_heads")?,
            num_value_heads: required_usize(text_config, "linear_num_value_heads")?,
            key_head_dim: required_usize(text_config, "linear_key_head_dim")?,
            value_head_dim: required_usize(text_config, "linear_value_head_dim")?,
            conv_kernel_dim: required_usize(text_config, "linear_conv_kernel_dim")?,
        };
        let moe = if text_model_type == "qwen3_5_moe_text" {
            Some(Qwen35MoeTextConfig {
                num_experts: required_usize(text_config, "num_experts")?,
                num_experts_per_tok: required_usize(text_config, "num_experts_per_tok")?,
                moe_intermediate_size: required_usize(text_config, "moe_intermediate_size")?,
                shared_expert_intermediate_size: required_usize(
                    text_config,
                    "shared_expert_intermediate_size",
                )?,
            })
        } else {
            reject_dense_moe_fields(text_config)?;
            None
        };
        let parsed = Self {
            top_level_model_type,
            text_model_type,
            hidden_size,
            num_hidden_layers,
            layer_types,
            linear_attention,
            head_dim: required_usize(text_config, "head_dim")?,
            num_attention_heads: required_usize(text_config, "num_attention_heads")?,
            num_key_value_heads: required_usize(text_config, "num_key_value_heads")?,
            dense_intermediate_size: optional_usize(text_config, "intermediate_size")?,
            moe,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    pub fn linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == Qwen35LayerType::LinearAttention)
            .count()
    }

    pub fn full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == Qwen35LayerType::FullAttention)
            .count()
    }

    pub fn first_linear_attention_layer(&self) -> Option<usize> {
        self.layer_types
            .iter()
            .position(|kind| *kind == Qwen35LayerType::LinearAttention)
    }

    pub fn first_full_attention_layer(&self) -> Option<usize> {
        self.layer_types
            .iter()
            .position(|kind| *kind == Qwen35LayerType::FullAttention)
    }

    fn validate(&self) -> Result<(), String> {
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types length {} does not match num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.linear_attention_layers() == 0 {
            return Err("Qwen3.5 W3 config must include linear_attention layers".to_string());
        }
        if self.full_attention_layers() == 0 {
            return Err("Qwen3.5 W3 config must include full_attention layers".to_string());
        }
        if let Some(moe) = &self.moe {
            if moe.num_experts_per_tok > moe.num_experts {
                return Err(format!(
                    "num_experts_per_tok {} exceeds num_experts {}",
                    moe.num_experts_per_tok, moe.num_experts
                ));
            }
            if moe.shared_expert_intermediate_size == 0 {
                return Err("shared_expert_intermediate_size must be positive".to_string());
            }
        } else if self.dense_intermediate_size.is_none() {
            return Err("dense Qwen3.5 config missing intermediate_size".to_string());
        }
        Ok(())
    }
}

fn required_string(map: &serde_json::Map<String, Value>, key: &str) -> Result<String, String> {
    map.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn required_usize(map: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    let value = map
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a positive integer"))?;
    if value == 0 {
        return Err(format!("{key} must be positive"));
    }
    usize::try_from(value).map_err(|_| format!("{key} is too large"))
}

fn optional_usize(
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<usize>, String> {
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let parsed = value
        .as_u64()
        .ok_or_else(|| format!("{key} must be a positive integer when present"))?;
    if parsed == 0 {
        return Err(format!("{key} must be positive when present"));
    }
    Ok(Some(
        usize::try_from(parsed).map_err(|_| format!("{key} is too large"))?,
    ))
}

fn parse_layer_types(
    map: &serde_json::Map<String, Value>,
    expected_len: usize,
) -> Result<Vec<Qwen35LayerType>, String> {
    let raw = map
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| "layer_types must be an array".to_string())?;
    if raw.len() != expected_len {
        return Err(format!(
            "layer_types length {} does not match num_hidden_layers {expected_len}",
            raw.len()
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(idx, value)| match value.as_str() {
            Some("linear_attention") => Ok(Qwen35LayerType::LinearAttention),
            Some("full_attention") => Ok(Qwen35LayerType::FullAttention),
            Some(other) => Err(format!("unsupported layer_types[{idx}]={other:?}")),
            None => Err(format!("layer_types[{idx}] must be a string")),
        })
        .collect()
}

fn reject_dense_moe_fields(map: &serde_json::Map<String, Value>) -> Result<(), String> {
    for key in [
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ] {
        if map.get(key).is_some_and(|value| !value.is_null()) {
            return Err(format!("dense Qwen3.5 config unexpectedly defines {key}"));
        }
    }
    Ok(())
}
