//! Transitional startup capabilities for explicitly registered legacy models.
//!
//! Migrated vNext families derive these facts from `PreparedProductionModel`.
//! Keeping this compatibility parser in the model crate prevents product
//! entrypoints from owning architecture-specific resource formulas.

use ferrum_types::{HardwareCapabilities, ModelCapabilities, MoeCapabilities};

use crate::{Architecture, ModelDefinition};

pub fn from_definition_with_weight_bytes(
    definition: &ModelDefinition,
    model_weight_bytes: Option<u64>,
) -> ModelCapabilities {
    from_definition_with_weight_bytes_inner(definition, model_weight_bytes)
}

pub fn from_definition_with_weight_bytes_for_hardware(
    definition: &ModelDefinition,
    model_weight_bytes: Option<u64>,
    _hardware: &HardwareCapabilities,
) -> ModelCapabilities {
    from_definition_with_weight_bytes_inner(definition, model_weight_bytes)
}

fn from_definition_with_weight_bytes_inner(
    definition: &ModelDefinition,
    model_weight_bytes: Option<u64>,
) -> ModelCapabilities {
    let architecture = match definition.architecture {
        Architecture::Qwen3Moe => "qwen3_moe",
        Architecture::Gemma3 => "gemma3",
        Architecture::Qwen3 => "qwen3",
        Architecture::Qwen2 => "qwen2",
        Architecture::Llama => "llama",
        Architecture::Mistral => "mistral",
        Architecture::Phi => "phi",
        Architecture::GPT2 => "gpt2",
        Architecture::Bert => "bert",
        Architecture::Clip => "clip",
        Architecture::Whisper => "whisper",
        Architecture::Qwen3TTS => "qwen3_tts",
        Architecture::Unknown => "unknown",
    }
    .to_owned();
    let head_dim = definition
        .extra_params
        .get("head_dim")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .or_else(|| {
            (definition.num_attention_heads > 0)
                .then_some(definition.hidden_size / definition.num_attention_heads)
        });
    let moe = if definition.architecture == Architecture::Qwen3Moe {
        Some(MoeCapabilities {
            num_experts: definition
                .extra_params
                .get("num_experts")
                .and_then(|value| value.as_u64())
                .unwrap_or(0) as usize,
            experts_per_token: definition
                .extra_params
                .get("num_experts_per_tok")
                .and_then(|value| value.as_u64())
                .unwrap_or(0) as usize,
            moe_intermediate_size: definition
                .extra_params
                .get("moe_intermediate_size")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize),
        })
    } else {
        None
    };

    ModelCapabilities {
        architecture,
        quantization: quantization_from_definition(definition),
        moe,
        max_context_len: Some(definition.max_position_embeddings),
        num_hidden_layers: Some(definition.num_hidden_layers),
        head_dim,
        kv_heads: definition.num_key_value_heads,
        estimated_weight_bytes: model_weight_bytes
            .filter(|value| *value > 0)
            .or_else(|| estimated_weight_bytes_from_definition(definition)),
        recurrent_state_bytes_per_sequence: None,
        supported_dtypes: vec!["fp16".to_owned(), "fp32".to_owned()],
        graph_safe_moe: false,
    }
}

pub fn quantization_from_definition(definition: &ModelDefinition) -> Option<String> {
    let quant = definition.extra_params.get("quantization_config")?;
    let method = quant
        .get("quant_method")
        .or_else(|| quant.get("type"))
        .and_then(|value| value.as_str())
        .unwrap_or("quantized");
    let bits = quant.get("bits").and_then(|value| value.as_u64());
    Some(match bits {
        Some(bits) => format!("{method}_int{bits}"),
        None => method.to_owned(),
    })
}

fn estimated_weight_bytes_from_definition(definition: &ModelDefinition) -> Option<u64> {
    let params = estimated_total_parameters_from_definition(definition)?;
    if params == 0 {
        return None;
    }
    let quant = definition.extra_params.get("quantization_config");
    let bits_per_param = quant
        .and_then(|quant| quant.get("bits"))
        .and_then(|value| value.as_u64())
        .filter(|bits| *bits > 0)
        .unwrap_or(16);
    Some(params.saturating_mul(bits_per_param).div_ceil(8))
}

fn estimated_total_parameters_from_definition(definition: &ModelDefinition) -> Option<u64> {
    let dense_params = definition.to_model_info("__auto_config").num_parameters;
    if definition.architecture != Architecture::Qwen3Moe {
        return Some(dense_params);
    }

    let hidden = definition.hidden_size as u128;
    let layers = definition.num_hidden_layers as u128;
    let vocab = definition.vocab_size as u128;
    let num_experts = definition
        .extra_params
        .get("num_experts")
        .and_then(|value| value.as_u64())? as u128;
    let moe_intermediate = definition
        .extra_params
        .get("moe_intermediate_size")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            (definition.intermediate_size > 0).then_some(definition.intermediate_size as u64)
        })? as u128;
    let shared_intermediate = definition
        .extra_params
        .get("shared_expert_intermediate_size")
        .and_then(|value| value.as_u64())
        .unwrap_or(0) as u128;

    let embedding_params = vocab.saturating_mul(hidden);
    let lm_head_params = embedding_params;
    let attention_params = layers
        .saturating_mul(4)
        .saturating_mul(hidden)
        .saturating_mul(hidden);
    let norm_params = layers.saturating_mul(2).saturating_mul(hidden);
    let router_params = layers.saturating_mul(hidden).saturating_mul(num_experts);
    let expert_params = layers
        .saturating_mul(num_experts)
        .saturating_mul(3)
        .saturating_mul(hidden)
        .saturating_mul(moe_intermediate);
    let shared_expert_params = layers
        .saturating_mul(3)
        .saturating_mul(hidden)
        .saturating_mul(shared_intermediate);
    let total = embedding_params
        .saturating_add(lm_head_params)
        .saturating_add(attention_params)
        .saturating_add(norm_params)
        .saturating_add(router_params)
        .saturating_add(expert_params)
        .saturating_add(shared_expert_params);
    Some(total.min(u64::MAX as u128) as u64)
}
