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
    use std::borrow::Cow;
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::ffi::OsStr;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use ferrum_interfaces::vnext::{ModelFamilyId, PhysicalWeightLayout};
    use half::bf16;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView, View};
    use serde_json::Value;
    use tokenizers::Tokenizer;

    use super::*;

    const GPT_OSS_TINY_SOURCE_ENV: &str = "FERRUM_GPT_OSS_TINY_SOURCE";
    const GPT_OSS_CANARY_OUT_ENV: &str = "FERRUM_GPT_OSS_CANARY_OUT";
    const FIXED_TINY_REVISION: &str = "27b6ad8040614834e65239f102de94bc459f48e5";
    const CANARY_HIDDEN_SIZE: u64 = 64;
    const CANARY_INTERMEDIATE_SIZE: u64 = 64;
    const CANARY_LAYER_COUNT: u64 = 2;
    const CANARY_ATTENTION_HEADS: u64 = 2;
    const CANARY_KV_HEADS: u64 = 1;
    const CANARY_HEAD_DIM: u64 = 64;
    const CANARY_EXPERT_COUNT: u64 = 32;
    const CANARY_EXPERTS_PER_TOKEN: u64 = 4;
    const CANARY_VOCABULARY_SIZE: u64 = 201_088;
    const CANARY_OUTPUT_SHARD: &str = "model.safetensors";
    const CANARY_ASSISTANT_TOKEN_ID: u32 = 173_781;
    const CANARY_RESPONSE: &str = "<|channel|>final<|message|>PASS<|return|>";
    const CANARY_RESPONSE_TOKEN_IDS: [u32; 5] = [200_005, 17_196, 200_008, 106_396, 200_002];
    const CANARY_EMBEDDING_VALUE: f32 = 1.0;
    const CANARY_LM_HEAD_VALUE: f32 = 64.0;
    const E8M0_UNIT_SCALE: u8 = 127;

    const FIXED_TINY_FILES: [(&str, &str); 6] = [
        (
            "config.json",
            "985ccbfe7bddb5a6dd1f217ea5c7d01a9615af53c0ee53262963aaf7e0797209",
        ),
        (
            "tokenizer.json",
            "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
        ),
        (
            "tokenizer_config.json",
            "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070",
        ),
        (
            "chat_template.jinja",
            "f8d9255777615591a7cc1a7c932f5a69e181128902295e1b81221d20d983cac7",
        ),
        (
            "generation_config.json",
            "06b87f5021d9dc8e858d1e0a16ef002de3ba77d168c2d36248171d37d1ff3758",
        ),
        (
            "special_tokens_map.json",
            "8464cabd6eda239fe46ebf8ae63b46c417721784a961a022f6b59174a2cda0e2",
        ),
    ];

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CanaryTransition {
        input_token: u32,
        output_token: u32,
        hidden_dimension: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CanaryTensorFill {
        ZeroBf16,
        RmsNormOne,
        EmbeddingTransitions,
        LmHeadTransitions,
        ZeroMxfp4Blocks,
        UnitMxfp4Scales,
    }

    struct CanaryTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl View for CanaryTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.bytes)
        }

        fn data_len(&self) -> usize {
            self.bytes.len()
        }
    }

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

    fn rewrite_as_native_mxfp4_canary(config: &mut Value) {
        let root = config
            .as_object_mut()
            .expect("fixed tiny GPT-OSS config root is an object");
        root.insert("hidden_size".to_owned(), Value::from(CANARY_HIDDEN_SIZE));
        root.insert(
            "intermediate_size".to_owned(),
            Value::from(CANARY_INTERMEDIATE_SIZE),
        );
        root.insert(
            "num_hidden_layers".to_owned(),
            Value::from(CANARY_LAYER_COUNT),
        );
        root.insert(
            "layer_types".to_owned(),
            serde_json::json!(["sliding_attention", "full_attention"]),
        );
        root.insert(
            "num_attention_heads".to_owned(),
            Value::from(CANARY_ATTENTION_HEADS),
        );
        root.insert(
            "num_key_value_heads".to_owned(),
            Value::from(CANARY_KV_HEADS),
        );
        root.insert("head_dim".to_owned(), Value::from(CANARY_HEAD_DIM));
        root.insert(
            "num_local_experts".to_owned(),
            Value::from(CANARY_EXPERT_COUNT),
        );
        root.insert(
            "num_experts_per_tok".to_owned(),
            Value::from(CANARY_EXPERTS_PER_TOKEN),
        );
        root.insert(
            "experts_per_token".to_owned(),
            Value::from(CANARY_EXPERTS_PER_TOKEN),
        );
        root.insert("vocab_size".to_owned(), Value::from(CANARY_VOCABULARY_SIZE));
        root.insert("tie_word_embeddings".to_owned(), Value::Bool(false));
        root.insert(
            "torch_dtype".to_owned(),
            Value::String("bfloat16".to_owned()),
        );
    }

    fn assert_native_mxfp4_canary_semantics(
        semantic: &GptOssSemanticConfig,
        quantization: &GptOssMxfp4Config,
    ) {
        assert_eq!(semantic.hidden_size, CANARY_HIDDEN_SIZE);
        assert_eq!(semantic.intermediate_size, CANARY_INTERMEDIATE_SIZE);
        assert_eq!(semantic.layer_count, CANARY_LAYER_COUNT);
        assert_eq!(semantic.attention_head_count, CANARY_ATTENTION_HEADS);
        assert_eq!(semantic.kv_head_count, CANARY_KV_HEADS);
        assert_eq!(semantic.head_dim, CANARY_HEAD_DIM);
        assert_eq!(semantic.query_features().unwrap(), 128);
        assert_eq!(semantic.kv_features().unwrap(), 64);
        assert_eq!(semantic.expert_count, CANARY_EXPERT_COUNT);
        assert_eq!(semantic.experts_per_token, CANARY_EXPERTS_PER_TOKEN);
        assert_eq!(semantic.vocabulary_size, CANARY_VOCABULARY_SIZE);
        assert!(!semantic.tie_word_embeddings);
        assert_eq!(semantic.layer_types.len(), 2);
        assert_eq!(format!("{:?}", semantic.layer_types[0]), "SlidingAttention");
        assert_eq!(format!("{:?}", semantic.layer_types[1]), "FullAttention");
        assert_eq!(quantization.quant_method(), "mxfp4");
        assert_eq!(quantization.bits_per_weight, 4);
        assert_eq!(quantization.group_size, 32);
        assert_eq!(quantization.packed_bytes_per_group, 16);
        assert_eq!(quantization.scale_exponent_bias, 127);
    }

    fn canary_transitions(assistant_token: u32, response_tokens: &[u32]) -> Vec<CanaryTransition> {
        assert!(
            !response_tokens.is_empty(),
            "canary response must not be empty"
        );
        assert!(
            response_tokens.len() <= usize::try_from(CANARY_HIDDEN_SIZE).unwrap(),
            "canary transition count exceeds hidden width"
        );
        let mut chain = Vec::with_capacity(response_tokens.len() + 1);
        chain.push(assistant_token);
        chain.extend_from_slice(response_tokens);
        assert!(chain
            .iter()
            .all(|token| u64::from(*token) < CANARY_VOCABULARY_SIZE));
        let mut inputs = BTreeSet::new();
        chain
            .windows(2)
            .enumerate()
            .map(|(hidden_dimension, pair)| {
                assert!(
                    inputs.insert(pair[0]),
                    "canary token {} would require two context-dependent transitions",
                    pair[0]
                );
                CanaryTransition {
                    input_token: pair[0],
                    output_token: pair[1],
                    hidden_dimension,
                }
            })
            .collect()
    }

    fn canary_tensor_fill(name: &str, element_type: ElementType) -> CanaryTensorFill {
        match (name, element_type) {
            ("model.embed_tokens.weight", ElementType::Bf16) => {
                CanaryTensorFill::EmbeddingTransitions
            }
            ("lm_head.weight", ElementType::Bf16) => CanaryTensorFill::LmHeadTransitions,
            ("model.norm.weight", ElementType::Bf16) => CanaryTensorFill::RmsNormOne,
            (name, ElementType::Bf16) if name.ends_with("_layernorm.weight") => {
                CanaryTensorFill::RmsNormOne
            }
            (name, ElementType::Bf16)
                if !name.ends_with("_blocks") && !name.ends_with("_scales") =>
            {
                CanaryTensorFill::ZeroBf16
            }
            (name, ElementType::U8) if name.ends_with("_blocks") => {
                CanaryTensorFill::ZeroMxfp4Blocks
            }
            (name, ElementType::U8) if name.ends_with("_scales") => {
                CanaryTensorFill::UnitMxfp4Scales
            }
            _ => panic!("unsupported canary tensor {name:?}/{element_type:?}"),
        }
    }

    fn checked_usize_shape(name: &str, dimensions: &[u64]) -> Vec<usize> {
        dimensions
            .iter()
            .map(|extent| {
                usize::try_from(*extent)
                    .unwrap_or_else(|_| panic!("tensor {name:?} extent {extent} exceeds usize"))
            })
            .collect()
    }

    fn checked_element_count(name: &str, shape: &[usize]) -> usize {
        shape
            .iter()
            .copied()
            .try_fold(1_usize, usize::checked_mul)
            .unwrap_or_else(|| panic!("tensor {name:?} element count overflows usize: {shape:?}"))
    }

    fn set_bf16_matrix_value(
        bytes: &mut [u8],
        rows: usize,
        columns: usize,
        row: usize,
        column: usize,
        value: f32,
    ) {
        assert_eq!(bytes.len(), rows.checked_mul(columns).unwrap() * 2);
        assert!(row < rows && column < columns);
        let offset = (row * columns + column) * 2;
        bytes[offset..offset + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }

    fn read_bf16_matrix_value(
        bytes: &[u8],
        rows: usize,
        columns: usize,
        row: usize,
        column: usize,
    ) -> bf16 {
        assert_eq!(bytes.len(), rows.checked_mul(columns).unwrap() * 2);
        assert!(row < rows && column < columns);
        let offset = (row * columns + column) * 2;
        bf16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
    }

    fn canary_tensor(
        name: &str,
        element_type: ElementType,
        dimensions: &[u64],
        transitions: &[CanaryTransition],
    ) -> CanaryTensor {
        let fill = canary_tensor_fill(name, element_type);
        let shape = checked_usize_shape(name, dimensions);
        let elements = checked_element_count(name, &shape);
        let (dtype, mut bytes) = match fill {
            CanaryTensorFill::ZeroMxfp4Blocks => (Dtype::U8, vec![0_u8; elements]),
            CanaryTensorFill::UnitMxfp4Scales => (Dtype::U8, vec![E8M0_UNIT_SCALE; elements]),
            CanaryTensorFill::ZeroBf16
            | CanaryTensorFill::RmsNormOne
            | CanaryTensorFill::EmbeddingTransitions
            | CanaryTensorFill::LmHeadTransitions => {
                (Dtype::BF16, vec![0_u8; elements.checked_mul(2).unwrap()])
            }
        };
        match fill {
            CanaryTensorFill::RmsNormOne => {
                let one = bf16::from_f32(1.0).to_bits().to_le_bytes();
                for value in bytes.chunks_exact_mut(2) {
                    value.copy_from_slice(&one);
                }
            }
            CanaryTensorFill::EmbeddingTransitions | CanaryTensorFill::LmHeadTransitions => {
                assert_eq!(dimensions, [CANARY_VOCABULARY_SIZE, CANARY_HIDDEN_SIZE]);
                let rows = usize::try_from(CANARY_VOCABULARY_SIZE).unwrap();
                let columns = usize::try_from(CANARY_HIDDEN_SIZE).unwrap();
                for transition in transitions {
                    let (row, value) = match fill {
                        CanaryTensorFill::EmbeddingTransitions => (
                            usize::try_from(transition.input_token).unwrap(),
                            CANARY_EMBEDDING_VALUE,
                        ),
                        CanaryTensorFill::LmHeadTransitions => (
                            usize::try_from(transition.output_token).unwrap(),
                            CANARY_LM_HEAD_VALUE,
                        ),
                        _ => unreachable!(),
                    };
                    set_bf16_matrix_value(
                        &mut bytes,
                        rows,
                        columns,
                        row,
                        transition.hidden_dimension,
                        value,
                    );
                }
            }
            CanaryTensorFill::ZeroBf16
            | CanaryTensorFill::ZeroMxfp4Blocks
            | CanaryTensorFill::UnitMxfp4Scales => {}
        }
        CanaryTensor {
            dtype,
            shape,
            bytes,
        }
    }

    fn required_absolute_env_path(name: &str) -> PathBuf {
        let value = std::env::var_os(name)
            .unwrap_or_else(|| panic!("required environment variable {name} is not set"));
        assert!(!value.is_empty(), "environment variable {name} is empty");
        let path = PathBuf::from(value);
        assert!(
            path.is_absolute(),
            "environment variable {name} must be an explicit absolute path, got {path:?}"
        );
        path
    }

    fn verified_fixed_tiny_files(source: &Path) -> BTreeMap<&'static str, Vec<u8>> {
        assert_eq!(
            source.file_name(),
            Some(OsStr::new(FIXED_TINY_REVISION)),
            "tiny source must bind fixed tiny-random/gpt-oss-mxfp4 revision {FIXED_TINY_REVISION}: {source:?}"
        );
        FIXED_TINY_FILES
            .iter()
            .map(|(name, expected_sha256)| {
                let path = source.join(name);
                assert!(
                    std::fs::metadata(&path)
                        .unwrap_or_else(|error| panic!("stat fixed source file {path:?}: {error}"))
                        .is_file(),
                    "fixed source entry is not a regular file: {path:?}"
                );
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|error| panic!("read fixed source file {path:?}: {error}"));
                let observed_sha256 = format!("{:x}", Sha256::digest(&bytes));
                assert_eq!(
                    observed_sha256, *expected_sha256,
                    "fixed tiny source file digest drifted: {path:?}"
                );
                (*name, bytes)
            })
            .collect()
    }

    fn write_new_file(path: &Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap_or_else(|error| panic!("create output file {path:?}: {error}"));
        file.write_all(bytes)
            .unwrap_or_else(|error| panic!("write output file {path:?}: {error}"));
        file.flush()
            .unwrap_or_else(|error| panic!("flush output file {path:?}: {error}"));
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
    fn native_mxfp4_canary_contract_and_tensor_initializers_are_exact() {
        let mut config: Value =
            serde_json::from_slice(&reference_config(32, 2, 2, 1, 32, 64, true)).unwrap();
        rewrite_as_native_mxfp4_canary(&mut config);
        let config_bytes = serde_json::to_vec(&config).unwrap();
        let (semantic, quantization) = GptOssSemanticConfig::parse(&config_bytes).unwrap();
        assert_native_mxfp4_canary_semantics(&semantic, &quantization);

        let manifest = expected_manifest(&semantic, &quantization).unwrap();
        assert_eq!(manifest.tensor_count, 41);
        assert_eq!(manifest.mxfp4_block_tensor_count, 4);
        assert_eq!(manifest.e8m0_scale_tensor_count, 4);
        assert_eq!(manifest.bf16_exclusion_tensor_count, 33);

        let transitions = canary_transitions(CANARY_ASSISTANT_TOKEN_ID, &CANARY_RESPONSE_TOKEN_IDS);
        assert_eq!(transitions.len(), CANARY_RESPONSE_TOKEN_IDS.len());
        assert_eq!(transitions[0].input_token, CANARY_ASSISTANT_TOKEN_ID);
        assert_eq!(transitions[0].output_token, 200_005);
        assert_eq!(transitions[4].input_token, 106_396);
        assert_eq!(transitions[4].output_token, 200_002);

        let mut inventory = BTreeMap::new();
        let counts = visit_expected_tensors(&semantic, |name, element_type, dimensions| {
            let fill = canary_tensor_fill(&name, element_type);
            assert!(inventory
                .insert(name, (element_type, dimensions, fill))
                .is_none());
            Ok(())
        })
        .unwrap();
        assert_eq!(counts.total(), manifest.tensor_count);
        assert_eq!(
            inventory["model.embed_tokens.weight"],
            (
                ElementType::Bf16,
                vec![CANARY_VOCABULARY_SIZE, CANARY_HIDDEN_SIZE],
                CanaryTensorFill::EmbeddingTransitions,
            )
        );
        assert_eq!(
            inventory["lm_head.weight"],
            (
                ElementType::Bf16,
                vec![CANARY_VOCABULARY_SIZE, CANARY_HIDDEN_SIZE],
                CanaryTensorFill::LmHeadTransitions,
            )
        );
        assert_eq!(
            inventory["model.layers.0.self_attn.q_proj.weight"].1,
            [128, 64]
        );
        assert_eq!(
            inventory["model.layers.1.mlp.experts.down_proj_blocks"],
            (
                ElementType::U8,
                vec![32, 64, 2, 16],
                CanaryTensorFill::ZeroMxfp4Blocks,
            )
        );
        assert_eq!(
            inventory
                .values()
                .filter(|(_, _, fill)| *fill == CanaryTensorFill::RmsNormOne)
                .count(),
            5
        );
        assert_eq!(
            inventory
                .values()
                .filter(|(_, _, fill)| *fill == CanaryTensorFill::ZeroMxfp4Blocks)
                .count(),
            4
        );
        assert_eq!(
            inventory
                .values()
                .filter(|(_, _, fill)| *fill == CanaryTensorFill::UnitMxfp4Scales)
                .count(),
            4
        );

        let norm = canary_tensor(
            "model.norm.weight",
            ElementType::Bf16,
            &[CANARY_HIDDEN_SIZE],
            &transitions,
        );
        assert!(norm.bytes.chunks_exact(2).all(|value| {
            bf16::from_bits(u16::from_le_bytes([value[0], value[1]])) == bf16::from_f32(1.0)
        }));
        let blocks = canary_tensor(
            "model.layers.0.mlp.experts.down_proj_blocks",
            ElementType::U8,
            &[32, 64, 2, 16],
            &transitions,
        );
        assert!(blocks.bytes.iter().all(|value| *value == 0));
        let scales = canary_tensor(
            "model.layers.0.mlp.experts.down_proj_scales",
            ElementType::U8,
            &[32, 64, 2],
            &transitions,
        );
        assert!(scales.bytes.iter().all(|value| *value == E8M0_UNIT_SCALE));

        let mut matrix = vec![0_u8; 4 * 8 * 2];
        set_bf16_matrix_value(&mut matrix, 4, 8, 2, 3, CANARY_LM_HEAD_VALUE);
        assert_eq!(
            read_bf16_matrix_value(&matrix, 4, 8, 2, 3),
            bf16::from_f32(CANARY_LM_HEAD_VALUE)
        );
        assert_eq!(read_bf16_matrix_value(&matrix, 4, 8, 2, 4), bf16::ZERO);
    }

    #[test]
    #[ignore = "requires fixed tiny-random/gpt-oss-mxfp4 snapshot and explicit output path"]
    fn derives_fixed_tiny_random_gpt_oss_native_mxfp4_canary_for_cuda_e2e() {
        let source_input = required_absolute_env_path(GPT_OSS_TINY_SOURCE_ENV);
        let source = std::fs::canonicalize(&source_input).unwrap_or_else(|error| {
            panic!("canonicalize fixed tiny GPT-OSS source {source_input:?}: {error}")
        });
        assert!(
            source.is_dir(),
            "fixed tiny source is not a directory: {source:?}"
        );
        let fixed_files = verified_fixed_tiny_files(&source);

        let source_config_bytes = &fixed_files["config.json"];
        let (source_semantic, source_quantization) =
            GptOssSemanticConfig::parse(source_config_bytes).unwrap();
        assert_eq!(source_semantic.hidden_size, 32);
        assert_eq!(source_semantic.intermediate_size, 64);
        assert_eq!(source_semantic.layer_count, 2);
        assert_eq!(source_semantic.attention_head_count, 2);
        assert_eq!(source_semantic.kv_head_count, 1);
        assert_eq!(source_semantic.head_dim, 32);
        assert_eq!(source_semantic.expert_count, 32);
        assert_eq!(source_semantic.experts_per_token, 4);
        assert_eq!(source_semantic.vocabulary_size, 201_088);
        assert!(source_semantic.tie_word_embeddings);
        assert_eq!(source_quantization.quant_method(), "mxfp4");

        let template = std::str::from_utf8(&fixed_files["chat_template.jinja"])
            .expect("fixed chat_template.jinja is UTF-8");
        assert!(
            template
                .contains("{%- if add_generation_prompt -%}\n<|start|>assistant\n{%- endif -%}"),
            "fixed Harmony template no longer ends its generation prompt in the assistant token"
        );
        let tokenizer = Tokenizer::from_file(source.join("tokenizer.json"))
            .unwrap_or_else(|error| panic!("load fixed GPT-OSS tokenizer: {error}"));
        assert_eq!(tokenizer.get_vocab_size(true), 200_019);
        let template_tail = tokenizer.encode("<|start|>assistant", false).unwrap();
        assert_eq!(
            template_tail.get_ids(),
            &[200_006, CANARY_ASSISTANT_TOKEN_ID]
        );
        let response = tokenizer.encode(CANARY_RESPONSE, false).unwrap();
        assert_eq!(response.get_ids(), CANARY_RESPONSE_TOKEN_IDS);
        assert_eq!(
            tokenizer.decode(response.get_ids(), false).unwrap(),
            CANARY_RESPONSE
        );
        let transitions = canary_transitions(CANARY_ASSISTANT_TOKEN_ID, response.get_ids());
        drop(tokenizer);

        let mut output_config: Value = serde_json::from_slice(source_config_bytes).unwrap();
        rewrite_as_native_mxfp4_canary(&mut output_config);
        let mut output_config_bytes = serde_json::to_vec_pretty(&output_config).unwrap();
        output_config_bytes.push(b'\n');
        let (semantic, quantization) = GptOssSemanticConfig::parse(&output_config_bytes).unwrap();
        assert_native_mxfp4_canary_semantics(&semantic, &quantization);
        let expected = expected_manifest(&semantic, &quantization).unwrap();
        assert_eq!(expected.tensor_count, 41);

        let mut generation_config: Value =
            serde_json::from_slice(&fixed_files["generation_config.json"]).unwrap();
        assert_eq!(generation_config["do_sample"], true);
        assert_eq!(
            generation_config["eos_token_id"],
            serde_json::json!([200_002, 199_999])
        );
        generation_config["do_sample"] = Value::Bool(false);
        generation_config["eos_token_id"] = serde_json::json!([200_002, 199_999, 200_012]);
        let mut generation_config_bytes = serde_json::to_vec_pretty(&generation_config).unwrap();
        generation_config_bytes.push(b'\n');

        let output_input = required_absolute_env_path(GPT_OSS_CANARY_OUT_ENV);
        assert_ne!(
            output_input, source_input,
            "canary output must differ from source"
        );
        assert!(
            !output_input.exists(),
            "canary output path already exists; refusing to overwrite: {output_input:?}"
        );
        let output_parent = output_input
            .parent()
            .expect("explicit canary output has a parent directory");
        assert!(
            output_parent.is_dir(),
            "canary output parent must already exist: {output_parent:?}"
        );
        std::fs::create_dir(&output_input)
            .unwrap_or_else(|error| panic!("create canary output {output_input:?}: {error}"));
        let output = std::fs::canonicalize(&output_input)
            .unwrap_or_else(|error| panic!("canonicalize canary output {output_input:?}: {error}"));
        assert_ne!(output, source, "canary output canonicalizes to source");
        assert!(
            !output.starts_with(&source),
            "canary output must not be created inside the immutable source snapshot"
        );

        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ] {
            write_new_file(&output.join(name), &fixed_files[name]);
        }
        write_new_file(
            &output.join("generation_config.json"),
            &generation_config_bytes,
        );
        write_new_file(&output.join("config.json"), &output_config_bytes);
        drop(fixed_files);

        let mut tensors = BTreeMap::<String, CanaryTensor>::new();
        let observed = visit_expected_tensors(&semantic, |name, element_type, dimensions| {
            let tensor = canary_tensor(&name, element_type, &dimensions, &transitions);
            assert!(tensors.insert(name, tensor).is_none());
            Ok(())
        })
        .unwrap();
        assert_eq!(observed.total(), expected.tensor_count);
        assert_eq!(
            tensors.len(),
            usize::try_from(expected.tensor_count).unwrap()
        );
        let payload_bytes = tensors
            .values()
            .try_fold(0_u64, |total, tensor| {
                total.checked_add(u64::try_from(tensor.data_len()).unwrap())
            })
            .expect("canary payload byte count does not overflow");
        let safetensors_metadata = Some(HashMap::from([
            ("format".to_owned(), "pt".to_owned()),
            (
                "ferrum_generator_id".to_owned(),
                "ferrum.gpt_oss.native-mxfp4-canary".to_owned(),
            ),
            ("ferrum_generator_version".to_owned(), "1".to_owned()),
            (
                "ferrum_source_revision".to_owned(),
                FIXED_TINY_REVISION.to_owned(),
            ),
            ("ferrum_response".to_owned(), CANARY_RESPONSE.to_owned()),
        ]));
        let output_shard = output.join(CANARY_OUTPUT_SHARD);
        assert!(!output_shard.exists());
        serialize_to_file(tensors, &safetensors_metadata, &output_shard).unwrap();

        let archive = SafetensorsArchive::open(&output).unwrap();
        assert_eq!(
            archive.tensor_count(),
            usize::try_from(expected.tensor_count).unwrap()
        );
        let reopened_payload_bytes = archive
            .tensor_names()
            .try_fold(0_u64, |total, name| {
                total.checked_add(
                    u64::try_from(archive.tensor(name).unwrap().bytes().len()).unwrap(),
                )
            })
            .unwrap();
        assert_eq!(reopened_payload_bytes, payload_bytes);
        let loaded = GptOssWeightManifest::load(&archive, &semantic, &quantization).unwrap();
        assert_eq!(loaded, expected);

        let embed = archive.tensor("model.embed_tokens.weight").unwrap();
        let lm_head = archive.tensor("lm_head.weight").unwrap();
        let rows = usize::try_from(CANARY_VOCABULARY_SIZE).unwrap();
        let columns = usize::try_from(CANARY_HIDDEN_SIZE).unwrap();
        for transition in &transitions {
            assert_eq!(
                read_bf16_matrix_value(
                    embed.bytes(),
                    rows,
                    columns,
                    usize::try_from(transition.input_token).unwrap(),
                    transition.hidden_dimension,
                ),
                bf16::from_f32(CANARY_EMBEDDING_VALUE)
            );
            assert_eq!(
                read_bf16_matrix_value(
                    lm_head.bytes(),
                    rows,
                    columns,
                    usize::try_from(transition.output_token).unwrap(),
                    transition.hidden_dimension,
                ),
                bf16::from_f32(CANARY_LM_HEAD_VALUE)
            );
        }
        assert_eq!(
            read_bf16_matrix_value(embed.bytes(), rows, columns, 0, 0),
            bf16::ZERO
        );
        assert_eq!(
            read_bf16_matrix_value(lm_head.bytes(), rows, columns, 0, 0),
            bf16::ZERO
        );
        assert!(archive
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .bytes()
            .iter()
            .all(|byte| *byte == 0));
        assert!(archive
            .tensor("model.layers.1.mlp.experts.gate_up_proj_blocks")
            .unwrap()
            .bytes()
            .iter()
            .all(|byte| *byte == 0));
        assert!(archive
            .tensor("model.layers.1.mlp.experts.gate_up_proj_scales")
            .unwrap()
            .bytes()
            .iter()
            .all(|byte| *byte == E8M0_UNIT_SCALE));
        assert!(archive
            .tensor("model.layers.0.input_layernorm.weight")
            .unwrap()
            .bytes()
            .chunks_exact(2)
            .all(|value| {
                bf16::from_bits(u16::from_le_bytes([value[0], value[1]])) == bf16::from_f32(1.0)
            }));
        drop(archive);

        let prepared = super::super::prepare_from_model_dir(&output).unwrap();
        assert_eq!(prepared.descriptor().architecture(), "gpt_oss");
        assert_eq!(prepared.descriptor().hidden_size(), 64);
        assert_eq!(prepared.descriptor().layer_count(), 2);
        assert_eq!(prepared.descriptor().attention_head_count(), 2);
        assert_eq!(prepared.descriptor().kv_head_count(), 1);
        assert_eq!(prepared.descriptor().attention_head_dimension(), 64);
        assert_eq!(prepared.descriptor().vocabulary_size(), 201_088);
        assert_eq!(
            prepared.descriptor().output_protocol(),
            ferrum_types::ModelOutputProtocol::HarmonyGptOss
        );
        assert_eq!(
            prepared.family().metadata().special_tokens.eos_token_ids,
            BTreeSet::from([199_999, 200_002, 200_012])
        );
        assert_eq!(
            prepared.family().weight_schema().format_id.as_str(),
            "weight-format.safetensors.gpt-oss-mxfp4-source"
        );
        assert_eq!(
            prepared.family().weight_schema().layout_id.as_str(),
            "weight-layout.gpt_oss.mxfp4.e2m1_e8m0.group32.expert_major"
        );

        println!(
            "FERRUM GPT-OSS NATIVE-MXFP4 CANARY PASS: {}",
            output.display()
        );
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
