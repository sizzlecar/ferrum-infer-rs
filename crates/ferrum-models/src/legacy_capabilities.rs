//! Transitional startup capabilities for explicitly registered legacy models.
//!
//! Migrated vNext families derive these facts from `PreparedProductionModel`.
//! Keeping this compatibility parser in the model crate prevents product
//! entrypoints from owning architecture-specific resource formulas.

use ferrum_types::{
    Device, HardwareCapabilities, LlamaExecutionOptions, ModelCapabilities, MoeCapabilities,
    SelectedKvCacheLayout,
};

use crate::{Architecture, ModelDefinition};

#[cfg(test)]
mod physical_storage_tests {
    use super::*;
    use ferrum_types::{KvCacheDtype, KvCacheStorage};

    fn tiny_config() -> crate::models::LlamaFamilyConfig {
        crate::models::LlamaFamilyConfig {
            hidden_size: 24,
            intermediate_size: 48,
            num_heads: 3,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 2,
            vocab_size: 32,
            max_seq_len: 64,
            ..Default::default()
        }
    }

    #[test]
    fn physical_kv_storage_cold_selector_uses_actual_backend_and_rejects_unsupported_markers() {
        let config = tiny_config();
        for paged in [false, true] {
            let options = LlamaExecutionOptions {
                metal_paged_kv: Some(paged),
                ..Default::default()
            };
            assert_eq!(
                selected_dense_kv_cache_storage(
                    &config,
                    &Device::CPU,
                    KvCacheDtype::Fp16,
                    &options
                ),
                Some(KvCacheStorage::F32)
            );
            for unsupported in [KvCacheDtype::Bf16, KvCacheDtype::Int8, KvCacheDtype::Fp8] {
                assert_eq!(
                    selected_dense_kv_cache_storage(&config, &Device::CPU, unsupported, &options),
                    None
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn physical_kv_storage_int8_is_unknown_for_unsupported_contiguous_layout() {
        let config = tiny_config();
        let mut options = LlamaExecutionOptions {
            metal_paged_kv: Some(false),
            ..Default::default()
        };
        assert_eq!(
            selected_dense_kv_cache_storage(
                &config,
                &Device::CUDA(0),
                KvCacheDtype::Int8,
                &options
            ),
            None
        );
        options.metal_paged_kv = Some(true);
        assert_eq!(
            selected_dense_kv_cache_storage(
                &config,
                &Device::CUDA(0),
                KvCacheDtype::Int8,
                &options
            ),
            Some(KvCacheStorage::Int8PerTokenHeadF16Scales)
        );
    }
}

/// Cold physical storage declaration from the same backend/KV implementation
/// selected by the retained dense factory. A requested marker is insufficient.
pub fn selected_kv_cache_storage(
    definition: &ModelDefinition,
    device: &Device,
    dtype: ferrum_types::KvCacheDtype,
    options: &LlamaExecutionOptions,
) -> Option<ferrum_types::KvCacheStorage> {
    use crate::models::LlamaFamilyConfig;
    if definition.num_attention_heads == 0 {
        return None;
    }
    let config = match definition.architecture {
        Architecture::Llama => LlamaFamilyConfig::llama_from_def(definition),
        Architecture::Qwen2 => LlamaFamilyConfig::qwen2_from_def(definition),
        Architecture::Qwen3 => LlamaFamilyConfig::qwen3_from_def(definition),
        Architecture::Mistral => LlamaFamilyConfig::mistral_from_def(definition),
        Architecture::Gemma3 => LlamaFamilyConfig::gemma3_from_def(definition),
        _ => return None,
    };
    selected_dense_kv_cache_storage(&config, device, dtype, options)
}

/// Validate the direct GGUF loader's physical model configuration before
/// declaring storage; no weights or device context are created.
pub fn selected_gguf_kv_cache_storage(
    path: &std::path::Path,
    device: &Device,
    dtype: ferrum_types::KvCacheDtype,
    options: &LlamaExecutionOptions,
) -> ferrum_types::Result<Option<ferrum_types::KvCacheStorage>> {
    let gguf = ferrum_quantization::gguf::GgufFile::open(path).map_err(|error| {
        ferrum_types::FerrumError::model(format!("GgufFile::open {}: {error}", path.display()))
    })?;
    let config = crate::models::LlamaFamilyConfig::from_gguf(&gguf)?;
    // The direct GGUF factory has fixed default storage, not INT8 dispatch.
    Ok((dtype == ferrum_types::KvCacheDtype::Fp16)
        .then(|| selected_dense_kv_cache_storage(&config, device, dtype, options))
        .flatten())
}

fn selected_dense_kv_cache_storage(
    config: &crate::models::LlamaFamilyConfig,
    device: &Device,
    dtype: ferrum_types::KvCacheDtype,
    options: &LlamaExecutionOptions,
) -> Option<ferrum_types::KvCacheStorage> {
    use ferrum_interfaces::kv_dtype::KvFp16;
    use ferrum_types::KvCacheDtype;
    match (device, dtype) {
        (Device::CPU, KvCacheDtype::Fp16) => {
            config.selected_kv_cache_storage::<ferrum_kernels::backend::cpu::CpuBackend, KvFp16>(options)
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        (Device::Metal, KvCacheDtype::Fp16) => {
            config.selected_kv_cache_storage::<ferrum_kernels::backend::metal::MetalBackend, KvFp16>(options)
        }
        #[cfg(feature = "cuda")]
        (Device::CUDA(_), KvCacheDtype::Fp16) => {
            config.selected_kv_cache_storage::<ferrum_kernels::backend::cuda::CudaBackend, KvFp16>(options)
        }
        #[cfg(feature = "cuda")]
        (Device::CUDA(_), KvCacheDtype::Int8) => {
            config.selected_kv_cache_storage::<
                ferrum_kernels::backend::cuda::CudaBackend,
                ferrum_interfaces::kv_dtype::KvInt8,
            >(options)
        }
        _ => None,
    }
}

/// Cold allocation selection for the retained dense factory. The model uses
/// the same generic selector at cache allocation; device ordinals do not change
/// its layout. Unsupported/undefined families remain unqualified here.
pub fn selected_kv_cache_layout(
    definition: &ModelDefinition,
    device: &Device,
    options: &LlamaExecutionOptions,
) -> SelectedKvCacheLayout {
    use crate::models::LlamaFamilyConfig;
    if definition.num_attention_heads == 0 {
        return SelectedKvCacheLayout::Unknown;
    }
    let config = match definition.architecture {
        Architecture::Llama => LlamaFamilyConfig::llama_from_def(definition),
        Architecture::Qwen2 => LlamaFamilyConfig::qwen2_from_def(definition),
        Architecture::Qwen3 => LlamaFamilyConfig::qwen3_from_def(definition),
        Architecture::Mistral => LlamaFamilyConfig::mistral_from_def(definition),
        Architecture::Gemma3 => LlamaFamilyConfig::gemma3_from_def(definition),
        _ => return SelectedKvCacheLayout::Unknown,
    };
    selected_llama_kv_cache_layout(&config, device, options)
}

/// Direct GGUF paths have no separate semantic definition. Read the same
/// physical config as their loader, without loading weights or creating a context.
pub fn selected_gguf_kv_cache_layout(
    path: &std::path::Path,
    device: &Device,
    options: &LlamaExecutionOptions,
) -> ferrum_types::Result<SelectedKvCacheLayout> {
    let gguf = ferrum_quantization::gguf::GgufFile::open(path).map_err(|error| {
        ferrum_types::FerrumError::model(format!("GgufFile::open {}: {error}", path.display()))
    })?;
    let config = crate::models::LlamaFamilyConfig::from_gguf(&gguf)?;
    Ok(selected_llama_kv_cache_layout(&config, device, options))
}

fn selected_llama_kv_cache_layout(
    config: &crate::models::LlamaFamilyConfig,
    device: &Device,
    options: &LlamaExecutionOptions,
) -> SelectedKvCacheLayout {
    match device {
        Device::CPU => {
            config.selected_kv_cache_layout::<ferrum_kernels::backend::cpu::CpuBackend>(options)
        }
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => {
            config.selected_kv_cache_layout::<ferrum_kernels::backend::cuda::CudaBackend>(options)
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        Device::Metal => {
            config.selected_kv_cache_layout::<ferrum_kernels::backend::metal::MetalBackend>(options)
        }
        _ => SelectedKvCacheLayout::Unknown,
    }
}

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
