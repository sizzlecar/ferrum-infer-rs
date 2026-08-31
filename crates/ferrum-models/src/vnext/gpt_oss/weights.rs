use ferrum_interfaces::vnext::{
    ContractVersion, ElementType, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    PhysicalWeightPadding, ProgramValueId, QuantizationFormatId, QuantizationGrouping,
    QuantizationPacking, QuantizationSpec, VNextError, WeightComponentRole, WeightComponentSpec,
    WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightSchema, WeightTensorSpec,
};
use ferrum_quantization::{SafetensorsArchive, MXFP4_E2M1_E8M0_SOURCE_FORMAT_ID};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::config::{GptOssMxfp4Config, GptOssSemanticConfig};
use super::invalid_config;

pub(super) const EMBED_TOKENS_ROLE: &str = "embed_tokens";
pub(super) const FINAL_NORM_ROLE: &str = "final_norm";
pub(super) const LM_HEAD_ROLE: &str = "lm_head";
pub(super) const INPUT_NORM_ROLE: &str = "input_layernorm";
pub(super) const POST_ATTENTION_NORM_ROLE: &str = "post_attention_layernorm";
pub(super) const Q_PROJ_ROLE: &str = "self_attn_q";
pub(super) const Q_BIAS_ROLE: &str = "self_attn_q_bias";
pub(super) const K_PROJ_ROLE: &str = "self_attn_k";
pub(super) const K_BIAS_ROLE: &str = "self_attn_k_bias";
pub(super) const V_PROJ_ROLE: &str = "self_attn_v";
pub(super) const V_BIAS_ROLE: &str = "self_attn_v_bias";
pub(super) const O_PROJ_ROLE: &str = "self_attn_o";
pub(super) const O_BIAS_ROLE: &str = "self_attn_o_bias";
pub(super) const ATTENTION_SINKS_ROLE: &str = "self_attn_sinks";
pub(super) const ROUTER_ROLE: &str = "moe_router";
pub(super) const ROUTER_BIAS_ROLE: &str = "moe_router_bias";
pub(super) const ROUTED_GATE_UP_ROLE: &str = "moe_routed_gate_up";
pub(super) const ROUTED_GATE_UP_BIAS_ROLE: &str = "moe_routed_gate_up_bias";
pub(super) const ROUTED_DOWN_ROLE: &str = "moe_routed_down";
pub(super) const ROUTED_DOWN_BIAS_ROLE: &str = "moe_routed_down_bias";

const STRUCTURE_FINGERPRINT_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GptOssWeightManifest {
    pub quantization: GptOssMxfp4Config,
    pub tensor_count: u64,
    pub mxfp4_block_tensor_count: u64,
    pub e8m0_scale_tensor_count: u64,
    pub bf16_exclusion_tensor_count: u64,
    pub structure_fingerprint: String,
}

impl GptOssWeightManifest {
    pub(super) fn load(
        archive: &SafetensorsArchive,
        semantic: &GptOssSemanticConfig,
        quantization: &GptOssMxfp4Config,
    ) -> Result<Self, String> {
        semantic.validate()?;
        quantization.validate()?;
        let manifest = expected_manifest(semantic, quantization)
            .map_err(|error| format!("build GPT-OSS weight contract: {error}"))?;
        validate_archive(archive, semantic, &manifest)
            .map_err(|error| format!("validate GPT-OSS checkpoint header: {error}"))?;
        Ok(manifest)
    }

    pub(super) fn validate(&self, semantic: &GptOssSemanticConfig) -> Result<(), VNextError> {
        semantic
            .validate()
            .map_err(|reason| invalid_config("semantic", reason))?;
        self.quantization
            .validate()
            .map_err(|reason| invalid_config("quantization", reason))?;
        let expected = expected_manifest(semantic, &self.quantization)?;
        if self != &expected {
            return Err(invalid_config(
                "weights",
                "GPT-OSS checkpoint structure proof differs from the semantic/MXFP4 contract",
            ));
        }
        Ok(())
    }

    pub(super) fn weight_schema(
        &self,
        semantic: &GptOssSemanticConfig,
    ) -> Result<WeightSchema, VNextError> {
        self.validate(semantic)?;
        let hidden = semantic.hidden_size;
        let query = semantic
            .query_features()
            .map_err(|reason| invalid_config("semantic", reason))?;
        let kv = semantic
            .kv_features()
            .map_err(|reason| invalid_config("semantic", reason))?;
        let mut components = Vec::new();
        let mut tensors = Vec::new();

        append_dense(
            global_weight_id(EMBED_TOKENS_ROLE)?,
            global_component_id(EMBED_TOKENS_ROLE)?,
            "model.embed_tokens.weight",
            vec![semantic.vocabulary_size, hidden],
            ElementType::F16,
            &mut components,
            &mut tensors,
        );
        append_dense(
            global_weight_id(FINAL_NORM_ROLE)?,
            global_component_id(FINAL_NORM_ROLE)?,
            "model.norm.weight",
            vec![hidden],
            ElementType::F16,
            &mut components,
            &mut tensors,
        );
        if !semantic.tie_word_embeddings {
            append_dense(
                global_weight_id(LM_HEAD_ROLE)?,
                global_component_id(LM_HEAD_ROLE)?,
                "lm_head.weight",
                vec![semantic.vocabulary_size, hidden],
                ElementType::F16,
                &mut components,
                &mut tensors,
            );
        }

        for layer_index in 0..semantic.layer_count {
            let layer_index = u32::try_from(layer_index)
                .map_err(|_| invalid_config("semantic.layer_count", "layer index exceeds u32"))?;
            let prefix = format!("model.layers.{layer_index}");
            for (role, external_name, dimensions) in [
                (
                    INPUT_NORM_ROLE,
                    format!("{prefix}.input_layernorm.weight"),
                    vec![hidden],
                ),
                (
                    POST_ATTENTION_NORM_ROLE,
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![hidden],
                ),
                (
                    Q_PROJ_ROLE,
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![query, hidden],
                ),
                (
                    Q_BIAS_ROLE,
                    format!("{prefix}.self_attn.q_proj.bias"),
                    vec![query],
                ),
                (
                    K_PROJ_ROLE,
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![kv, hidden],
                ),
                (
                    K_BIAS_ROLE,
                    format!("{prefix}.self_attn.k_proj.bias"),
                    vec![kv],
                ),
                (
                    V_PROJ_ROLE,
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![kv, hidden],
                ),
                (
                    V_BIAS_ROLE,
                    format!("{prefix}.self_attn.v_proj.bias"),
                    vec![kv],
                ),
                (
                    O_PROJ_ROLE,
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![hidden, query],
                ),
                (
                    O_BIAS_ROLE,
                    format!("{prefix}.self_attn.o_proj.bias"),
                    vec![hidden],
                ),
                (
                    ATTENTION_SINKS_ROLE,
                    format!("{prefix}.self_attn.sinks"),
                    vec![semantic.attention_head_count],
                ),
            ] {
                append_dense(
                    layer_weight_id(layer_index, role)?,
                    layer_component_id(layer_index, role)?,
                    external_name,
                    dimensions,
                    ElementType::F16,
                    &mut components,
                    &mut tensors,
                );
            }
            for (role, external_name, dimensions) in [
                (
                    ROUTER_ROLE,
                    format!("{prefix}.mlp.router.weight"),
                    vec![semantic.expert_count, hidden],
                ),
                (
                    ROUTER_BIAS_ROLE,
                    format!("{prefix}.mlp.router.bias"),
                    vec![semantic.expert_count],
                ),
            ] {
                append_dense(
                    layer_weight_id(layer_index, role)?,
                    layer_component_id(layer_index, role)?,
                    external_name,
                    dimensions,
                    ElementType::Bf16,
                    &mut components,
                    &mut tensors,
                );
            }

            let gate_up_stem = format!("{prefix}.mlp.experts.gate_up_proj");
            append_mxfp4(
                layer_weight_id(layer_index, ROUTED_GATE_UP_ROLE)?,
                layer_component_id(layer_index, ROUTED_GATE_UP_ROLE)?,
                &gate_up_stem,
                vec![
                    semantic.expert_count,
                    semantic
                        .intermediate_size
                        .checked_mul(2)
                        .ok_or_else(|| invalid_config("semantic", "gate/up width overflows"))?,
                    hidden,
                ],
                &self.quantization,
                &mut components,
                &mut tensors,
            )?;
            append_dense(
                layer_weight_id(layer_index, ROUTED_GATE_UP_BIAS_ROLE)?,
                layer_component_id(layer_index, ROUTED_GATE_UP_BIAS_ROLE)?,
                format!("{gate_up_stem}_bias"),
                vec![
                    semantic.expert_count,
                    semantic.intermediate_size.checked_mul(2).ok_or_else(|| {
                        invalid_config("semantic", "gate/up bias width overflows")
                    })?,
                ],
                ElementType::Bf16,
                &mut components,
                &mut tensors,
            );

            let down_stem = format!("{prefix}.mlp.experts.down_proj");
            append_mxfp4(
                layer_weight_id(layer_index, ROUTED_DOWN_ROLE)?,
                layer_component_id(layer_index, ROUTED_DOWN_ROLE)?,
                &down_stem,
                vec![semantic.expert_count, hidden, semantic.intermediate_size],
                &self.quantization,
                &mut components,
                &mut tensors,
            )?;
            append_dense(
                layer_weight_id(layer_index, ROUTED_DOWN_BIAS_ROLE)?,
                layer_component_id(layer_index, ROUTED_DOWN_BIAS_ROLE)?,
                format!("{down_stem}_bias"),
                vec![semantic.expert_count, hidden],
                ElementType::Bf16,
                &mut components,
                &mut tensors,
            );
        }

        Ok(WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.gpt-oss-mxfp4-source")?,
            layout_id: WeightLayoutId::new(
                "weight-layout.gpt_oss.mxfp4.e2m1_e8m0.group32.expert_major",
            )?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        })
    }
}

pub(super) fn global_weight_id(role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.global.{role}"))
}

pub(super) fn global_weight_value_id(role: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.weight.global.{role}"))
}

pub(super) fn layer_weight_id(layer_index: u32, role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.layer.{layer_index}.{role}"))
}

pub(super) fn layer_weight_value_id(
    layer_index: u32,
    role: &str,
) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.weight.layer.{layer_index}.{role}"))
}

fn global_component_id(role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("component.global.{role}"))
}

fn layer_component_id(layer_index: u32, role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("component.layer.{layer_index}.{role}"))
}

#[allow(clippy::too_many_arguments)]
fn append_dense(
    weight_id: WeightId,
    component_id: WeightId,
    external_name: impl Into<String>,
    dimensions: Vec<u64>,
    logical_element_type: ElementType,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) {
    components.push(WeightComponentSpec {
        id: component_id.clone(),
        role: WeightComponentRole::Values,
        external_names: vec![external_name.into()],
        dimensions: dimensions.clone(),
        encoding: WeightEncoding::Dense {
            element_type: logical_element_type,
        },
        required: true,
    });
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions,
        logical_element_type,
        physical_layout: PhysicalWeightLayout::Dense { component_id },
        required: true,
    });
}

#[allow(clippy::too_many_arguments)]
fn append_mxfp4(
    weight_id: WeightId,
    component_id: WeightId,
    external_stem: &str,
    logical_dimensions: Vec<u64>,
    quantization: &GptOssMxfp4Config,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    quantization
        .validate()
        .map_err(|reason| invalid_config("quantization", reason))?;
    let [experts, rows, columns] = logical_dimensions.as_slice() else {
        return Err(invalid_config(
            "weights",
            "GPT-OSS MXFP4 expert tensor must have [experts, rows, columns] shape",
        ));
    };
    let group_size = u64::from(quantization.group_size);
    if !columns.is_multiple_of(group_size) {
        return Err(invalid_config(
            "weights",
            "GPT-OSS MXFP4 input width is not group-32 aligned",
        ));
    }
    let packed_dimensions = vec![
        *experts,
        *rows,
        columns / group_size,
        u64::from(quantization.packed_bytes_per_group),
    ];
    let scale_dimensions = vec![*experts, *rows, columns / group_size];
    let packed_id = WeightId::new(format!("{component_id}.blocks"))?;
    let scales_id = WeightId::new(format!("{component_id}.scales"))?;
    components.push(WeightComponentSpec {
        id: packed_id.clone(),
        role: WeightComponentRole::PackedValues,
        external_names: vec![format!("{external_stem}_blocks")],
        dimensions: packed_dimensions.clone(),
        encoding: WeightEncoding::Quantized(mxfp4_quantization_spec(quantization)?),
        required: true,
    });
    components.push(WeightComponentSpec {
        id: scales_id.clone(),
        role: WeightComponentRole::Scales,
        external_names: vec![format!("{external_stem}_scales")],
        dimensions: scale_dimensions,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::U8,
        },
        required: true,
    });
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions: logical_dimensions,
        logical_element_type: ElementType::Bf16,
        physical_layout: PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
            packed_dimensions,
            scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
            zero_points: None,
            zero_point_packed_dimensions: None,
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: 2,
            group_padding: PhysicalWeightPadding::Exact,
        },
        required: true,
    });
    Ok(())
}

fn mxfp4_quantization_spec(
    quantization: &GptOssMxfp4Config,
) -> Result<QuantizationSpec, VNextError> {
    quantization
        .validate()
        .map_err(|reason| invalid_config("quantization", reason))?;
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(MXFP4_E2M1_E8M0_SOURCE_FORMAT_ID)?,
        bits_per_weight: quantization.bits_per_weight,
        grouping: QuantizationGrouping::fixed(quantization.group_size),
        packing: QuantizationPacking::Interleaved,
        scale_type: ElementType::U8,
        zero_point_type: None,
    })
}

fn validate_archive(
    archive: &SafetensorsArchive,
    semantic: &GptOssSemanticConfig,
    expected: &GptOssWeightManifest,
) -> Result<(), VNextError> {
    if u64::try_from(archive.tensor_count()).ok() != Some(expected.tensor_count) {
        return Err(invalid_config(
            "weights",
            format!(
                "checkpoint contains {} tensors, expected exactly {}",
                archive.tensor_count(),
                expected.tensor_count
            ),
        ));
    }
    let observed = visit_expected_tensors(semantic, |name, element_type, dimensions| {
        let tensor = archive
            .tensor(&name)
            .map_err(|error| invalid_config("weights", error.to_string()))?;
        if tensor.element_type() != Some(element_type) || tensor.shape() != dimensions {
            return Err(invalid_config(
                "weights",
                format!(
                    "tensor {name:?} has {:?}/{:?}, expected {element_type:?}/{dimensions:?}",
                    tensor.dtype(),
                    tensor.shape()
                ),
            ));
        }
        Ok(())
    })?;
    if observed.total() != expected.tensor_count {
        return Err(invalid_config(
            "weights",
            "internal GPT-OSS tensor cardinality drift",
        ));
    }
    Ok(())
}

pub(super) fn expected_manifest(
    semantic: &GptOssSemanticConfig,
    quantization: &GptOssMxfp4Config,
) -> Result<GptOssWeightManifest, VNextError> {
    semantic
        .validate()
        .map_err(|reason| invalid_config("semantic", reason))?;
    quantization
        .validate()
        .map_err(|reason| invalid_config("quantization", reason))?;
    let mut hasher = Sha256::new();
    hasher.update([STRUCTURE_FINGERPRINT_VERSION]);
    let counts = visit_expected_tensors(semantic, |name, element_type, dimensions| {
        hash_header(&mut hasher, &name, element_type, &dimensions);
        Ok(())
    })?;
    Ok(GptOssWeightManifest {
        quantization: quantization.clone(),
        tensor_count: counts.total(),
        mxfp4_block_tensor_count: counts.blocks,
        e8m0_scale_tensor_count: counts.scales,
        bf16_exclusion_tensor_count: counts.bf16,
        structure_fingerprint: format!("{:x}", hasher.finalize()),
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SourceCounts {
    blocks: u64,
    scales: u64,
    bf16: u64,
}

impl SourceCounts {
    fn total(self) -> u64 {
        self.blocks + self.scales + self.bf16
    }
}

fn visit_expected_tensors(
    semantic: &GptOssSemanticConfig,
    mut visitor: impl FnMut(String, ElementType, Vec<u64>) -> Result<(), VNextError>,
) -> Result<SourceCounts, VNextError> {
    let hidden = semantic.hidden_size;
    let query = semantic
        .query_features()
        .map_err(|reason| invalid_config("semantic", reason))?;
    let kv = semantic
        .kv_features()
        .map_err(|reason| invalid_config("semantic", reason))?;
    let mut counts = SourceCounts::default();
    emit_bf16(
        &mut visitor,
        &mut counts,
        "model.embed_tokens.weight".to_owned(),
        vec![semantic.vocabulary_size, hidden],
    )?;
    emit_bf16(
        &mut visitor,
        &mut counts,
        "model.norm.weight".to_owned(),
        vec![hidden],
    )?;
    if !semantic.tie_word_embeddings {
        emit_bf16(
            &mut visitor,
            &mut counts,
            "lm_head.weight".to_owned(),
            vec![semantic.vocabulary_size, hidden],
        )?;
    }
    for layer_index in 0..semantic.layer_count {
        let prefix = format!("model.layers.{layer_index}");
        for (name, dimensions) in [
            (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![query, hidden],
            ),
            (format!("{prefix}.self_attn.q_proj.bias"), vec![query]),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv, hidden],
            ),
            (format!("{prefix}.self_attn.k_proj.bias"), vec![kv]),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, hidden],
            ),
            (format!("{prefix}.self_attn.v_proj.bias"), vec![kv]),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, query],
            ),
            (format!("{prefix}.self_attn.o_proj.bias"), vec![hidden]),
            (
                format!("{prefix}.self_attn.sinks"),
                vec![semantic.attention_head_count],
            ),
            (
                format!("{prefix}.mlp.router.weight"),
                vec![semantic.expert_count, hidden],
            ),
            (
                format!("{prefix}.mlp.router.bias"),
                vec![semantic.expert_count],
            ),
            (
                format!("{prefix}.mlp.experts.gate_up_proj_bias"),
                vec![
                    semantic.expert_count,
                    semantic.intermediate_size.checked_mul(2).ok_or_else(|| {
                        invalid_config("semantic", "gate/up bias width overflows")
                    })?,
                ],
            ),
            (
                format!("{prefix}.mlp.experts.down_proj_bias"),
                vec![semantic.expert_count, hidden],
            ),
        ] {
            emit_bf16(&mut visitor, &mut counts, name, dimensions)?;
        }
        emit_mxfp4(
            &mut visitor,
            &mut counts,
            format!("{prefix}.mlp.experts.gate_up_proj"),
            semantic.expert_count,
            semantic
                .intermediate_size
                .checked_mul(2)
                .ok_or_else(|| invalid_config("semantic", "gate/up width overflows"))?,
            hidden,
        )?;
        emit_mxfp4(
            &mut visitor,
            &mut counts,
            format!("{prefix}.mlp.experts.down_proj"),
            semantic.expert_count,
            hidden,
            semantic.intermediate_size,
        )?;
    }
    Ok(counts)
}

fn emit_bf16(
    visitor: &mut impl FnMut(String, ElementType, Vec<u64>) -> Result<(), VNextError>,
    counts: &mut SourceCounts,
    name: String,
    dimensions: Vec<u64>,
) -> Result<(), VNextError> {
    visitor(name, ElementType::Bf16, dimensions)?;
    counts.bf16 = counts
        .bf16
        .checked_add(1)
        .ok_or_else(|| invalid_config("weights", "BF16 tensor count overflows"))?;
    Ok(())
}

fn emit_mxfp4(
    visitor: &mut impl FnMut(String, ElementType, Vec<u64>) -> Result<(), VNextError>,
    counts: &mut SourceCounts,
    stem: String,
    experts: u64,
    rows: u64,
    columns: u64,
) -> Result<(), VNextError> {
    if !columns.is_multiple_of(32) {
        return Err(invalid_config(
            "weights",
            format!("MXFP4 tensor {stem:?} input width {columns} is not group-32 aligned"),
        ));
    }
    visitor(
        format!("{stem}_blocks"),
        ElementType::U8,
        vec![experts, rows, columns / 32, 16],
    )?;
    counts.blocks = counts
        .blocks
        .checked_add(1)
        .ok_or_else(|| invalid_config("weights", "MXFP4 block tensor count overflows"))?;
    visitor(
        format!("{stem}_scales"),
        ElementType::U8,
        vec![experts, rows, columns / 32],
    )?;
    counts.scales = counts
        .scales
        .checked_add(1)
        .ok_or_else(|| invalid_config("weights", "E8M0 scale tensor count overflows"))?;
    Ok(())
}

fn hash_header(hasher: &mut Sha256, name: &str, element_type: ElementType, dimensions: &[u64]) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update([match element_type {
        ElementType::Bf16 => 1,
        ElementType::U8 => 2,
        _ => unreachable!("GPT-OSS source header emits only BF16 and U8"),
    }]);
    hasher.update((dimensions.len() as u64).to_le_bytes());
    for extent in dimensions {
        hasher.update(extent.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ferrum_interfaces::vnext::{ModelFamilyId, PhysicalWeightLayout};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::*;

    fn reference_config(
        hidden: u64,
        layers: usize,
        heads: u64,
        kv_heads: u64,
        head_dim: u64,
        intermediate: u64,
        tied: bool,
    ) -> Vec<u8> {
        let layer_types = (0..layers)
            .map(|index| {
                if index.is_multiple_of(2) {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["GptOssForCausalLM"],
            "attention_bias": true,
            "attention_dropout": 0.0,
            "experts_per_token": 4,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "hidden_size": hidden,
            "initial_context_length": 4096,
            "intermediate_size": intermediate,
            "layer_types": layer_types,
            "max_position_embeddings": 131072,
            "model_type": "gpt_oss",
            "num_attention_heads": heads,
            "num_experts_per_tok": 4,
            "num_hidden_layers": layers,
            "num_key_value_heads": kv_heads,
            "num_local_experts": 32,
            "output_router_logits": false,
            "quantization_config": {
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.layers.*.mlp.router",
                    "model.embed_tokens",
                    "lm_head"
                ],
                "quant_method": "mxfp4"
            },
            "rms_norm_eps": 0.00001,
            "rope_scaling": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "truncate": false
            },
            "rope_theta": 150000,
            "sliding_window": 128,
            "swiglu_limit": 7.0,
            "tie_word_embeddings": tied,
            "use_cache": true,
            "vocab_size": 201088
        }))
        .unwrap()
    }

    fn tiny_config() -> (GptOssSemanticConfig, GptOssMxfp4Config) {
        GptOssSemanticConfig::parse(&reference_config(32, 2, 2, 1, 32, 64, true)).unwrap()
    }

    fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
        Box::leak(bytes.into_boxed_slice())
    }

    fn view(element_type: ElementType, dimensions: Vec<u64>) -> TensorView<'static> {
        let elements = dimensions
            .iter()
            .try_fold(1_usize, |total, extent| {
                total.checked_mul(usize::try_from(*extent).ok()?)
            })
            .unwrap();
        let (dtype, bytes_per_element) = match element_type {
            ElementType::Bf16 => (Dtype::BF16, 2),
            ElementType::U8 => (Dtype::U8, 1),
            _ => unreachable!(),
        };
        TensorView::new(
            dtype,
            dimensions
                .into_iter()
                .map(|extent| usize::try_from(extent).unwrap())
                .collect(),
            leak_bytes(vec![0_u8; elements * bytes_per_element]),
        )
        .unwrap()
    }

    fn fixture_views(semantic: &GptOssSemanticConfig) -> BTreeMap<String, TensorView<'static>> {
        let mut views = BTreeMap::new();
        visit_expected_tensors(semantic, |name, element_type, dimensions| {
            views.insert(name, view(element_type, dimensions));
            Ok(())
        })
        .unwrap();
        views
    }

    fn write_archive(
        views: BTreeMap<String, TensorView<'static>>,
    ) -> (tempfile::TempDir, SafetensorsArchive) {
        let directory = tempfile::tempdir().unwrap();
        serialize_to_file(
            views,
            &None::<std::collections::HashMap<String, String>>,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let archive = SafetensorsArchive::open(directory.path()).unwrap();
        (directory, archive)
    }

    #[test]
    fn official_header_inventory_is_exactly_459_tensors() {
        let (semantic, quantization) =
            GptOssSemanticConfig::parse(&reference_config(2880, 24, 64, 8, 64, 2880, false))
                .unwrap();
        let manifest = expected_manifest(&semantic, &quantization).unwrap();
        assert_eq!(manifest.tensor_count, 459);
        assert_eq!(manifest.mxfp4_block_tensor_count, 48);
        assert_eq!(manifest.e8m0_scale_tensor_count, 48);
        assert_eq!(manifest.bf16_exclusion_tensor_count, 363);
        assert_eq!(manifest.structure_fingerprint.len(), 64);
    }

    #[test]
    fn tiny_real_layout_header_builds_a_complete_typed_schema() {
        let (semantic, quantization) = tiny_config();
        let (_directory, archive) = write_archive(fixture_views(&semantic));
        let manifest = GptOssWeightManifest::load(&archive, &semantic, &quantization).unwrap();
        assert_eq!(manifest.tensor_count, 40);
        assert_eq!(manifest.mxfp4_block_tensor_count, 4);
        assert_eq!(manifest.e8m0_scale_tensor_count, 4);
        assert_eq!(manifest.bf16_exclusion_tensor_count, 32);

        let schema = manifest.weight_schema(&semantic).unwrap();
        assert_eq!(schema.components.len(), 40);
        assert_eq!(schema.tensors.len(), 36);
        schema
            .validate(&ModelFamilyId::new(super::super::FAMILY_ID).unwrap())
            .unwrap();
        let quant = schema
            .tensors
            .iter()
            .find(|tensor| tensor.id.as_str() == "weight.layer.0.moe_routed_gate_up")
            .unwrap();
        assert_eq!(quant.dimensions, [32, 128, 32]);
        assert!(matches!(
            quant.physical_layout,
            PhysicalWeightLayout::Quantized { group_axis: 2, .. }
        ));
    }

    #[test]
    fn dtype_shape_and_unknown_tensor_drift_fail_before_allocation() {
        let (semantic, quantization) = tiny_config();

        let mut wrong_dtype = fixture_views(&semantic);
        let scale_name = "model.layers.0.mlp.experts.gate_up_proj_scales";
        wrong_dtype.insert(
            scale_name.to_owned(),
            view(ElementType::Bf16, vec![32, 128, 1]),
        );
        let (_directory, archive) = write_archive(wrong_dtype);
        let error = GptOssWeightManifest::load(&archive, &semantic, &quantization).unwrap_err();
        assert!(
            error.contains(scale_name) && error.contains("U8"),
            "{error}"
        );

        let mut wrong_shape = fixture_views(&semantic);
        let blocks_name = "model.layers.0.mlp.experts.down_proj_blocks";
        wrong_shape.insert(
            blocks_name.to_owned(),
            view(ElementType::U8, vec![32, 32, 3, 16]),
        );
        let (_directory, archive) = write_archive(wrong_shape);
        let error = GptOssWeightManifest::load(&archive, &semantic, &quantization).unwrap_err();
        assert!(
            error.contains(blocks_name) && error.contains("[32, 32, 2, 16]"),
            "{error}"
        );

        let mut unknown = fixture_views(&semantic);
        unknown.remove("model.norm.weight").unwrap();
        unknown.insert(
            "unexpected.weight".to_owned(),
            view(ElementType::Bf16, vec![32]),
        );
        let (_directory, archive) = write_archive(unknown);
        let error = GptOssWeightManifest::load(&archive, &semantic, &quantization).unwrap_err();
        assert!(
            error.contains("model.norm.weight") && error.contains("absent"),
            "{error}"
        );
    }
}
