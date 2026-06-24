//! Qwen3.5 / Qwen3.6 HF `text_config` parser.
//!
//! W3 models are not Llama-family attention-only decoders: they mix
//! Gated-DeltaNet-style linear attention with full attention, and the MoE
//! variants add routed experts plus a shared expert. This module keeps that
//! shape explicit before the product loader/model path is wired.

use ferrum_interfaces::{RecurrentStateSpec, RecurrentStateTensorSpec};
use ferrum_types::{DataType, Device, RequestId};
use serde_json::Value;

pub const QWEN35_CONV_STATE_NAME: &str = "conv_state";
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
    pub norm_topk_prob: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Qwen35RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,
    pub mrope_interleaved: bool,
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35QuantizationConfig {
    pub quant_method: String,
    pub bits: usize,
    pub group_size: usize,
    pub desc_act: bool,
    pub sym: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
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
    pub attn_output_gate: bool,
    pub rope_parameters: Qwen35RopeParameters,
    pub tie_word_embeddings: bool,
    pub dense_intermediate_size: Option<usize>,
    pub moe: Option<Qwen35MoeTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Qwen35QuantizationConfig>,
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
                norm_topk_prob: optional_bool(text_config, "norm_topk_prob")?
                    .or(optional_bool(obj, "norm_topk_prob")?)
                    .unwrap_or(true),
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
            attn_output_gate: optional_bool(text_config, "attn_output_gate")?
                .or(optional_bool(obj, "attn_output_gate")?)
                .unwrap_or(false),
            rope_parameters: parse_rope_parameters(text_config, obj)?,
            tie_word_embeddings,
            dense_intermediate_size: optional_usize(text_config, "intermediate_size")?,
            moe,
            quantization: parse_quantization_config(text_config, obj)?,
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

    pub fn full_attention_query_total_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn full_attention_kv_total_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn full_attention_q_proj_total_dim(&self) -> usize {
        let base = self.full_attention_query_total_dim();
        if self.attn_output_gate {
            2 * base
        } else {
            base
        }
    }

    pub fn full_attention_rope_dim(&self) -> usize {
        ((self.head_dim as f32) * self.rope_parameters.partial_rotary_factor).round() as usize
    }

    pub fn full_attention_text_rope_interleaved(&self) -> bool {
        self.rope_parameters.mrope_interleaved && self.rope_parameters.mrope_section.is_none()
    }

    /// Per-layer causal-convolution recurrent state shape, excluding the request slot.
    ///
    /// vLLM stores Qwen GDN convolution state before the temporal DeltaNet state.
    /// The dim-first layout is `[conv_channels, conv_kernel - 1]`, where
    /// `conv_channels = q_total + k_total + v_total`.
    pub fn recurrent_conv_state_shape(&self) -> Result<Vec<usize>, String> {
        Ok(vec![
            self.linear_qk_total_dim() * 2 + self.linear_value_total_dim(),
            self.linear_attention.conv_kernel_dim.saturating_sub(1),
        ])
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
        let conv_shape = self.recurrent_conv_state_shape()?;
        let delta_shape = self.recurrent_delta_state_shape()?;
        let mut specs = Vec::with_capacity(self.linear_attention_layers() * 2);
        for (layer_index, kind) in self.layer_types.iter().copied().enumerate() {
            if kind != Qwen35LayerType::LinearAttention {
                continue;
            }
            specs.push(RecurrentStateTensorSpec::new(
                layer_index,
                QWEN35_CONV_STATE_NAME,
                conv_shape.clone(),
            ));
            specs.push(RecurrentStateTensorSpec::new(
                layer_index,
                QWEN35_DELTA_STATE_NAME,
                delta_shape.clone(),
            ));
        }
        Ok(specs)
    }

    pub fn recurrent_state_elements_per_slot(&self) -> Result<usize, String> {
        Ok(self
            .recurrent_state_tensor_specs()?
            .iter()
            .map(RecurrentStateTensorSpec::num_elements)
            .sum())
    }

    pub fn recurrent_state_bytes_per_slot(&self, dtype: DataType) -> Result<u64, String> {
        self.recurrent_state_elements_per_slot()?
            .checked_mul(dtype.size_bytes())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| format!("Qwen3.5 recurrent state bytes overflow for dtype {dtype:?}"))
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
        if self.rope_parameters.rope_theta <= 0.0 {
            return Err(format!(
                "rope_theta must be positive, got {}",
                self.rope_parameters.rope_theta
            ));
        }
        if !(0.0..=1.0).contains(&self.rope_parameters.partial_rotary_factor)
            || self.rope_parameters.partial_rotary_factor == 0.0
        {
            return Err(format!(
                "partial_rotary_factor must be in (0, 1], got {}",
                self.rope_parameters.partial_rotary_factor
            ));
        }
        let rope_dim = self.full_attention_rope_dim();
        if rope_dim == 0 || rope_dim > self.head_dim || rope_dim % 2 != 0 {
            return Err(format!(
                "full attention rope dim {rope_dim} must be positive, even, and <= head_dim {}",
                self.head_dim
            ));
        }
        if let Some(section) = &self.rope_parameters.mrope_section {
            if section.len() != 2 && section.len() != 3 {
                return Err(format!(
                    "mrope_section length must be 2 or 3, got {}",
                    section.len()
                ));
            }
            let section_sum: usize = section.iter().sum();
            if section_sum != rope_dim / 2 {
                return Err(format!(
                    "mrope_section sum {section_sum} must equal rope_dim/2 {}",
                    rope_dim / 2
                ));
            }
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

fn parse_rope_parameters(
    text_config: &serde_json::Map<String, Value>,
    root: &serde_json::Map<String, Value>,
) -> Result<Qwen35RopeParameters, String> {
    let rope_obj = text_config
        .get("rope_parameters")
        .or_else(|| root.get("rope_parameters"))
        .and_then(Value::as_object);
    let rope_theta = rope_obj
        .and_then(|obj| obj.get("rope_theta"))
        .and_then(Value::as_f64)
        .or_else(|| text_config.get("rope_theta").and_then(Value::as_f64))
        .or_else(|| root.get("rope_theta").and_then(Value::as_f64))
        .unwrap_or(10_000.0);
    let partial_rotary_factor = rope_obj
        .and_then(|obj| obj.get("partial_rotary_factor"))
        .and_then(Value::as_f64)
        .or_else(|| {
            text_config
                .get("partial_rotary_factor")
                .and_then(Value::as_f64)
        })
        .or_else(|| root.get("partial_rotary_factor").and_then(Value::as_f64))
        .unwrap_or(1.0) as f32;
    let mrope_interleaved = rope_obj
        .and_then(|obj| obj.get("mrope_interleaved"))
        .and_then(Value::as_bool)
        .or_else(|| {
            text_config
                .get("mrope_interleaved")
                .and_then(Value::as_bool)
        })
        .or_else(|| root.get("mrope_interleaved").and_then(Value::as_bool))
        .unwrap_or(false);
    let mrope_section = rope_obj
        .and_then(|obj| obj.get("mrope_section"))
        .or_else(|| text_config.get("mrope_section"))
        .or_else(|| root.get("mrope_section"))
        .map(parse_usize_array)
        .transpose()?;
    Ok(Qwen35RopeParameters {
        rope_theta,
        partial_rotary_factor,
        mrope_interleaved,
        mrope_section,
    })
}

fn parse_quantization_config(
    text_config: &serde_json::Map<String, Value>,
    root: &serde_json::Map<String, Value>,
) -> Result<Option<Qwen35QuantizationConfig>, String> {
    let Some(value) = text_config
        .get("quantization_config")
        .or_else(|| root.get("quantization_config"))
    else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let quant = value
        .as_object()
        .ok_or_else(|| "quantization_config must be an object when present".to_string())?;
    let quant_method = required_string(quant, "quant_method")?;
    if quant_method != "gptq" {
        return Err(format!(
            "unsupported Qwen3.5 quantization_config.quant_method {quant_method:?}"
        ));
    }
    Ok(Some(Qwen35QuantizationConfig {
        quant_method,
        bits: required_usize(quant, "bits")?,
        group_size: required_usize(quant, "group_size")?,
        desc_act: optional_bool(quant, "desc_act")?.unwrap_or(false),
        sym: optional_bool(quant, "sym")?.unwrap_or(false),
    }))
}

fn parse_usize_array(value: &Value) -> Result<Vec<usize>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "mrope_section must be an array".to_string())?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| "mrope_section entries must be unsigned integers".to_string())
        })
        .collect()
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
        ("moe_fused_gate_up_proj", "mlp.experts.gate_up_proj", false),
        ("moe_fused_down_proj", "mlp.experts.down_proj", false),
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
        (
            "moe_per_expert_gate_proj_qweight",
            "mlp.experts.*.gate_proj.qweight",
            false,
        ),
        (
            "moe_per_expert_up_proj_qweight",
            "mlp.experts.*.up_proj.qweight",
            false,
        ),
        (
            "moe_per_expert_down_proj_qweight",
            "mlp.experts.*.down_proj.qweight",
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
        "norm_topk_prob",
    ] {
        if map.get(key).is_some_and(|value| !value.is_null()) {
            return Err(format!("dense Qwen3.5 config unexpectedly defines {key}"));
        }
    }
    Ok(())
}
