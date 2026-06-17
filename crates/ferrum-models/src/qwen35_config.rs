//! Qwen3.5 / Qwen3.6 HF `text_config` parser.
//!
//! W3 models are not Llama-family attention-only decoders: they mix
//! Gated-DeltaNet-style linear attention with full attention, and the MoE
//! variants add routed experts plus a shared expert. This module keeps that
//! shape explicit before the product loader/model path is wired.

use ferrum_interfaces::{RecurrentStateSpec, RecurrentStateTensorSpec};
use ferrum_types::{DataType, Device, RequestId};
use serde_json::Value;

pub const QWEN35_DELTA_STATE_NAME: &str = "delta_state";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Qwen35MlpKind {
    Dense,
    SparseMoeSharedExpert,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35LayerPlan {
    pub layer_index: usize,
    pub attention: Qwen35LayerType,
    pub mlp: Qwen35MlpKind,
    pub has_recurrent_state: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35WeightSpec {
    pub role: String,
    pub name: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35LayerWeightManifest {
    pub layer_index: usize,
    pub attention: Qwen35LayerType,
    pub mlp: Qwen35MlpKind,
    pub tensors: Vec<Qwen35WeightSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35WeightManifest {
    pub prefix: String,
    pub global_tensors: Vec<Qwen35WeightSpec>,
    pub layers: Vec<Qwen35LayerWeightManifest>,
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
    pub tie_word_embeddings: bool,
    pub dense_intermediate_size: Option<usize>,
    pub moe: Option<Qwen35MoeTextConfig>,
}

impl Qwen35TextConfig {
    pub fn from_model_definition(def: &crate::definition::ModelDefinition) -> Result<Self, String> {
        if !matches!(
            def.architecture,
            crate::registry::Architecture::Qwen35 | crate::registry::Architecture::Qwen35Moe
        ) {
            return Err(format!(
                "model definition architecture {:?} is not Qwen3.5/Qwen3.6",
                def.architecture
            ));
        }
        let parsed = Self::from_hf_config_value(&def.extra_params)?;
        if parsed.hidden_size != def.hidden_size {
            return Err(format!(
                "Qwen3.5 text hidden_size {} does not match ModelDefinition hidden_size {}",
                parsed.hidden_size, def.hidden_size
            ));
        }
        if parsed.num_hidden_layers != def.num_hidden_layers {
            return Err(format!(
                "Qwen3.5 text num_hidden_layers {} does not match ModelDefinition num_hidden_layers {}",
                parsed.num_hidden_layers, def.num_hidden_layers
            ));
        }
        Ok(parsed)
    }

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
        let tie_word_embeddings = match optional_bool(text_config, "tie_word_embeddings")? {
            Some(value) => value,
            None => optional_bool(obj, "tie_word_embeddings")?.unwrap_or(false),
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
            tie_word_embeddings,
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

    pub fn mlp_kind_for_layer(&self, layer_index: usize) -> Result<Qwen35MlpKind, String> {
        if layer_index >= self.num_hidden_layers {
            return Err(format!(
                "layer_index {layer_index} exceeds num_hidden_layers {}",
                self.num_hidden_layers
            ));
        }
        if self.is_moe() {
            Ok(Qwen35MlpKind::SparseMoeSharedExpert)
        } else {
            Ok(Qwen35MlpKind::Dense)
        }
    }

    pub fn layer_plan(&self) -> Result<Vec<Qwen35LayerPlan>, String> {
        self.layer_types
            .iter()
            .copied()
            .enumerate()
            .map(|(layer_index, attention)| {
                Ok(Qwen35LayerPlan {
                    layer_index,
                    attention,
                    mlp: self.mlp_kind_for_layer(layer_index)?,
                    has_recurrent_state: attention == Qwen35LayerType::LinearAttention,
                })
            })
            .collect()
    }

    pub fn sparse_moe_layers(&self) -> Vec<usize> {
        if self.is_moe() {
            (0..self.num_hidden_layers).collect()
        } else {
            Vec::new()
        }
    }

    pub fn dense_mlp_layers(&self) -> Vec<usize> {
        if self.is_moe() {
            Vec::new()
        } else {
            (0..self.num_hidden_layers).collect()
        }
    }

    pub fn weight_manifest(
        &self,
        prefix: impl Into<String>,
    ) -> Result<Qwen35WeightManifest, String> {
        let prefix = prefix.into();
        let mut global_tensors = vec![
            weight_spec(
                "embed_tokens",
                format!("{prefix}.embed_tokens.weight"),
                true,
            ),
            weight_spec("final_norm", format!("{prefix}.norm.weight"), true),
        ];
        global_tensors.push(weight_spec(
            "lm_head",
            format!("{prefix}.lm_head.weight"),
            !self.tie_word_embeddings,
        ));

        let layers = self
            .layer_plan()?
            .into_iter()
            .map(|plan| {
                let layer_prefix = format!("{prefix}.layers.{}", plan.layer_index);
                let mut tensors = vec![
                    weight_spec(
                        "input_layernorm",
                        format!("{layer_prefix}.input_layernorm.weight"),
                        true,
                    ),
                    weight_spec(
                        "post_attention_layernorm",
                        format!("{layer_prefix}.post_attention_layernorm.weight"),
                        true,
                    ),
                ];
                match plan.attention {
                    Qwen35LayerType::LinearAttention => {
                        tensors.extend(linear_attention_weight_specs(&layer_prefix));
                    }
                    Qwen35LayerType::FullAttention => {
                        tensors.extend(full_attention_weight_specs(&layer_prefix));
                    }
                }
                match plan.mlp {
                    Qwen35MlpKind::Dense => {
                        tensors.extend(dense_mlp_weight_specs(&layer_prefix));
                    }
                    Qwen35MlpKind::SparseMoeSharedExpert => {
                        tensors.extend(sparse_moe_weight_specs(&layer_prefix));
                    }
                }
                Qwen35LayerWeightManifest {
                    layer_index: plan.layer_index,
                    attention: plan.attention,
                    mlp: plan.mlp,
                    tensors,
                }
            })
            .collect();

        Ok(Qwen35WeightManifest {
            prefix,
            global_tensors,
            layers,
        })
    }

    pub fn linear_qk_total_dim(&self) -> usize {
        self.linear_attention.num_key_heads * self.linear_attention.key_head_dim
    }

    pub fn linear_value_total_dim(&self) -> usize {
        self.linear_attention.num_value_heads * self.linear_attention.value_head_dim
    }

    /// Per-layer DeltaNet recurrent state shape, excluding the request slot.
    ///
    /// vLLM stores Gated DeltaNet temporal state as
    /// `[value_heads, value_head_dim, key_head_dim]`. When value heads exceed
    /// q/k heads, q/k are repeated onto the value-head axis.
    pub fn recurrent_delta_state_shape(&self) -> Result<Vec<usize>, String> {
        let key_heads = self.linear_attention.num_key_heads;
        let value_heads = self.linear_attention.num_value_heads;
        if value_heads % key_heads != 0 {
            return Err(format!(
                "linear value heads {value_heads} is not divisible by key heads {key_heads}"
            ));
        }
        Ok(vec![
            value_heads,
            self.linear_attention.value_head_dim,
            self.linear_attention.key_head_dim,
        ])
    }

    pub fn recurrent_state_tensor_specs(&self) -> Result<Vec<RecurrentStateTensorSpec>, String> {
        let shape = self.recurrent_delta_state_shape()?;
        Ok(self
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(layer_index, kind)| {
                (*kind == Qwen35LayerType::LinearAttention).then(|| {
                    RecurrentStateTensorSpec::new(
                        layer_index,
                        QWEN35_DELTA_STATE_NAME,
                        shape.clone(),
                    )
                })
            })
            .collect())
    }

    pub fn recurrent_state_elements_per_slot(&self) -> Result<usize, String> {
        Ok(self
            .recurrent_state_tensor_specs()?
            .iter()
            .map(RecurrentStateTensorSpec::num_elements)
            .sum())
    }

    pub fn to_recurrent_state_spec(
        &self,
        request_id: RequestId,
        dtype: DataType,
        device: Device,
        max_batch_slots: usize,
    ) -> Result<RecurrentStateSpec, String> {
        if max_batch_slots == 0 {
            return Err("max_batch_slots must be positive".to_string());
        }
        Ok(RecurrentStateSpec {
            request_id,
            num_layers: self.num_hidden_layers,
            tensors: self.recurrent_state_tensor_specs()?,
            dtype,
            device,
            max_batch_slots,
        })
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

fn optional_bool(map: &serde_json::Map<String, Value>, key: &str) -> Result<Option<bool>, String> {
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| format!("{key} must be a boolean when present"))
}

fn weight_spec(
    role: impl Into<String>,
    name: impl Into<String>,
    required: bool,
) -> Qwen35WeightSpec {
    Qwen35WeightSpec {
        role: role.into(),
        name: name.into(),
        required,
    }
}

fn linear_attention_weight_specs(layer_prefix: &str) -> Vec<Qwen35WeightSpec> {
    [
        ("linear_attn_qkv", "linear_attn.in_proj_qkv.weight"),
        ("linear_attn_z", "linear_attn.in_proj_z.weight"),
        ("linear_attn_b", "linear_attn.in_proj_b.weight"),
        ("linear_attn_a", "linear_attn.in_proj_a.weight"),
        ("linear_attn_conv", "linear_attn.conv1d.weight"),
        ("linear_attn_a_log", "linear_attn.A_log"),
        ("linear_attn_dt_bias", "linear_attn.dt_bias"),
        ("linear_attn_norm", "linear_attn.norm.weight"),
        ("linear_attn_out", "linear_attn.out_proj.weight"),
    ]
    .into_iter()
    .map(|(role, suffix)| weight_spec(role, format!("{layer_prefix}.{suffix}"), true))
    .collect()
}

fn full_attention_weight_specs(layer_prefix: &str) -> Vec<Qwen35WeightSpec> {
    [
        ("self_attn_q", "self_attn.q_proj.weight"),
        ("self_attn_k", "self_attn.k_proj.weight"),
        ("self_attn_v", "self_attn.v_proj.weight"),
        ("self_attn_o", "self_attn.o_proj.weight"),
        ("self_attn_q_norm", "self_attn.q_norm.weight"),
        ("self_attn_k_norm", "self_attn.k_norm.weight"),
    ]
    .into_iter()
    .map(|(role, suffix)| weight_spec(role, format!("{layer_prefix}.{suffix}"), true))
    .collect()
}

fn dense_mlp_weight_specs(layer_prefix: &str) -> Vec<Qwen35WeightSpec> {
    [
        ("mlp_gate", "mlp.gate_proj.weight"),
        ("mlp_up", "mlp.up_proj.weight"),
        ("mlp_down", "mlp.down_proj.weight"),
    ]
    .into_iter()
    .map(|(role, suffix)| weight_spec(role, format!("{layer_prefix}.{suffix}"), true))
    .collect()
}

fn sparse_moe_weight_specs(layer_prefix: &str) -> Vec<Qwen35WeightSpec> {
    [
        ("moe_router", "mlp.gate.weight", true),
        (
            "moe_shared_expert_gate",
            "mlp.shared_expert_gate.weight",
            true,
        ),
        (
            "moe_shared_expert_gate_proj",
            "mlp.shared_expert.gate_proj.weight",
            true,
        ),
        (
            "moe_shared_expert_up_proj",
            "mlp.shared_expert.up_proj.weight",
            true,
        ),
        (
            "moe_shared_expert_down_proj",
            "mlp.shared_expert.down_proj.weight",
            true,
        ),
        ("moe_fused_gate_up_proj", "mlp.experts.gate_up_proj", true),
        ("moe_fused_down_proj", "mlp.experts.down_proj", true),
        (
            "moe_per_expert_gate_proj",
            "mlp.experts.*.gate_proj.weight",
            false,
        ),
        (
            "moe_per_expert_up_proj",
            "mlp.experts.*.up_proj.weight",
            false,
        ),
        (
            "moe_per_expert_down_proj",
            "mlp.experts.*.down_proj.weight",
            false,
        ),
    ]
    .into_iter()
    .map(|(role, suffix, required)| weight_spec(role, format!("{layer_prefix}.{suffix}"), required))
    .collect()
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
