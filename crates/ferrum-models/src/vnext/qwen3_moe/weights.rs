use ferrum_interfaces::vnext::{
    BlockQuantizationSpec, CompositeWeightPart, ContractVersion, ElementType,
    PhysicalWeightComponentBinding, PhysicalWeightLayout, PhysicalWeightPadding, ProgramValueId,
    QuantizationFormatId, QuantizationGrouping, QuantizationPacking, QuantizationSpec, VNextError,
    WeightComponentRole, WeightComponentSpec, WeightEncoding, WeightFormatId, WeightId,
    WeightLayoutId, WeightSchema, WeightTensorSpec,
};
use ferrum_quantization::gguf::{block_quantization_format, ferrum_to_gguf_with_arch, GgmlDType};
use ferrum_quantization::{
    GgufWeightComponentSource, SafetensorsArchive, GPTQ_MARLIN_INT4_FORMAT_ID,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::config::{Qwen3MoeGptqConfig, Qwen3MoeSemanticConfig};
use super::invalid_config;
use crate::vnext::weight_layout::{contiguous_or_reshaped_binding, dense_or_reshaped_layout};

pub(super) const EMBED_TOKENS_ROLE: &str = "embed_tokens";
pub(super) const FINAL_NORM_ROLE: &str = "final_norm";
pub(super) const LM_HEAD_ROLE: &str = "lm_head";
pub(super) const INPUT_NORM_ROLE: &str = "input_layernorm";
pub(super) const POST_ATTENTION_NORM_ROLE: &str = "post_attention_layernorm";
pub(super) const Q_PROJ_ROLE: &str = "self_attn_q";
pub(super) const K_PROJ_ROLE: &str = "self_attn_k";
pub(super) const V_PROJ_ROLE: &str = "self_attn_v";
pub(super) const O_PROJ_ROLE: &str = "self_attn_o";
pub(super) const Q_NORM_ROLE: &str = "self_attn_q_norm";
pub(super) const K_NORM_ROLE: &str = "self_attn_k_norm";
pub(super) const ROUTER_ROLE: &str = "moe_router";
pub(super) const ROUTED_GATE_UP_ROLE: &str = "moe_routed_gate_up";
pub(super) const ROUTED_DOWN_ROLE: &str = "moe_routed_down";

const STRUCTURE_FINGERPRINT_VERSION: u8 = 1;

/// Source-specific proof that the checkpoint header matched the deterministic
/// Qwen3 MoE schema. GPTQ keeps only a compact fingerprint; GGUF retains its
/// much smaller native tensor inventory because each component's block ABI is
/// part of the zero-copy execution contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "snake_case")]
pub(super) enum Qwen3MoeWeightManifest {
    SafetensorsGptqMarlin {
        quantization: Qwen3MoeGptqConfig,
        tensor_count: u64,
        structure_fingerprint: String,
    },
    GgufNative {
        tensors: Vec<Qwen3MoeGgufTensor>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Qwen3MoeGgufTensor {
    external_name: String,
    dimensions: Vec<u64>,
    encoding: Qwen3MoeGgufEncoding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Qwen3MoeGgufEncoding {
    Dense { source_element_type: ElementType },
    BlockQuantized(BlockQuantizationSpec),
}

impl Qwen3MoeWeightManifest {
    pub(super) fn load(
        archive: &SafetensorsArchive,
        semantic: &Qwen3MoeSemanticConfig,
        quantization: &Qwen3MoeGptqConfig,
    ) -> Result<Self, String> {
        semantic.validate_gptq(quantization)?;
        let manifest = expected_manifest(semantic, quantization)
            .map_err(|error| format!("build weight contract: {error}"))?;
        let Qwen3MoeWeightManifest::SafetensorsGptqMarlin { tensor_count, .. } = &manifest else {
            unreachable!("GPTQ manifest constructor returned a GGUF manifest")
        };
        validate_archive(archive, semantic, quantization, *tensor_count)
            .map_err(|error| error.to_string())?;
        Ok(manifest)
    }

    pub(super) fn load_gguf(
        source: &GgufWeightComponentSource,
        semantic: &Qwen3MoeSemanticConfig,
    ) -> Result<Self, String> {
        semantic.validate()?;
        let architecture = source
            .file()
            .architecture()
            .map_err(|error| format!("read GGUF architecture: {error}"))?;
        if architecture != "qwen3moe" {
            return Err(format!(
                "Qwen3 MoE family package requires GGUF architecture \"qwen3moe\", got {architecture:?}"
            ));
        }
        let expected = expected_gguf_tensors(semantic)?;
        let source_tensor_count = source.file().tensor_names().count();
        if source_tensor_count != expected.len() {
            return Err(format!(
                "Qwen3 MoE GGUF contains {source_tensor_count} tensors, expected exactly {}",
                expected.len()
            ));
        }
        let tensors = expected
            .into_iter()
            .map(|expected| load_gguf_tensor(source, expected))
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = Self::GgufNative { tensors };
        manifest
            .validate(semantic)
            .map_err(|error| error.to_string())?;
        Ok(manifest)
    }

    pub(super) fn validate(&self, semantic: &Qwen3MoeSemanticConfig) -> Result<(), VNextError> {
        match self {
            Self::SafetensorsGptqMarlin { quantization, .. } => {
                semantic
                    .validate_gptq(quantization)
                    .map_err(|reason| invalid_config("semantic", reason))?;
                let expected = expected_manifest(semantic, quantization)?;
                if self != &expected {
                    return Err(invalid_config(
                        "weights",
                        "GPTQ checkpoint structure proof differs from the semantic contract",
                    ));
                }
            }
            Self::GgufNative { tensors } => {
                semantic
                    .validate()
                    .map_err(|reason| invalid_config("semantic", reason))?;
                validate_gguf_tensors(semantic, tensors)?;
            }
        }
        Ok(())
    }

    pub(super) fn weight_schema(
        &self,
        semantic: &Qwen3MoeSemanticConfig,
    ) -> Result<WeightSchema, VNextError> {
        self.validate(semantic)?;
        let Self::SafetensorsGptqMarlin { quantization, .. } = self else {
            let Self::GgufNative { tensors } = self else {
                unreachable!()
            };
            return gguf_weight_schema(semantic, tensors);
        };
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
            &mut components,
            &mut tensors,
        );
        append_dense(
            global_weight_id(FINAL_NORM_ROLE)?,
            global_component_id(FINAL_NORM_ROLE)?,
            "model.norm.weight",
            vec![hidden],
            &mut components,
            &mut tensors,
        );
        if !semantic.tie_word_embeddings {
            append_dense(
                global_weight_id(LM_HEAD_ROLE)?,
                global_component_id(LM_HEAD_ROLE)?,
                "lm_head.weight",
                vec![semantic.vocabulary_size, hidden],
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
                    Q_NORM_ROLE,
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![semantic.head_dim],
                ),
                (
                    K_NORM_ROLE,
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![semantic.head_dim],
                ),
                (
                    ROUTER_ROLE,
                    format!("{prefix}.mlp.gate.weight"),
                    vec![semantic.expert_count, hidden],
                ),
            ] {
                append_dense(
                    layer_weight_id(layer_index, role)?,
                    layer_component_id(layer_index, role)?,
                    external_name,
                    dimensions,
                    &mut components,
                    &mut tensors,
                );
            }
            for (role, stem, dimensions) in [
                (
                    Q_PROJ_ROLE,
                    format!("{prefix}.self_attn.q_proj"),
                    vec![query, hidden],
                ),
                (
                    K_PROJ_ROLE,
                    format!("{prefix}.self_attn.k_proj"),
                    vec![kv, hidden],
                ),
                (
                    V_PROJ_ROLE,
                    format!("{prefix}.self_attn.v_proj"),
                    vec![kv, hidden],
                ),
                (
                    O_PROJ_ROLE,
                    format!("{prefix}.self_attn.o_proj"),
                    vec![hidden, query],
                ),
            ] {
                append_gptq_stack(
                    layer_weight_id(layer_index, role)?,
                    layer_component_id(layer_index, role)?,
                    dimensions,
                    vec![stem],
                    quantization,
                    &mut components,
                    &mut tensors,
                )?;
            }

            let mut gate_up_stems = Vec::with_capacity(
                usize::try_from(semantic.expert_count)
                    .unwrap_or_default()
                    .saturating_mul(2),
            );
            let mut down_stems =
                Vec::with_capacity(usize::try_from(semantic.expert_count).unwrap_or_default());
            for expert_index in 0..semantic.expert_count {
                let expert = format!("{prefix}.mlp.experts.{expert_index}");
                gate_up_stems.push(format!("{expert}.gate_proj"));
                gate_up_stems.push(format!("{expert}.up_proj"));
                down_stems.push(format!("{expert}.down_proj"));
            }
            append_gptq_stack(
                layer_weight_id(layer_index, ROUTED_GATE_UP_ROLE)?,
                layer_component_id(layer_index, ROUTED_GATE_UP_ROLE)?,
                vec![
                    semantic.expert_count,
                    2,
                    semantic.expert_intermediate_size,
                    hidden,
                ],
                gate_up_stems,
                quantization,
                &mut components,
                &mut tensors,
            )?;
            append_gptq_stack(
                layer_weight_id(layer_index, ROUTED_DOWN_ROLE)?,
                layer_component_id(layer_index, ROUTED_DOWN_ROLE)?,
                vec![
                    semantic.expert_count,
                    hidden,
                    semantic.expert_intermediate_size,
                ],
                down_stems,
                quantization,
                &mut components,
                &mut tensors,
            )?;
        }
        Ok(WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.gptq-marlin-int4")?,
            layout_id: WeightLayoutId::new(
                "weight-layout.qwen3_moe.routed.gptq_marlin.expert_major",
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

fn append_dense(
    weight_id: WeightId,
    component_id: WeightId,
    external_name: impl Into<String>,
    dimensions: Vec<u64>,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) {
    components.push(WeightComponentSpec {
        id: component_id.clone(),
        role: WeightComponentRole::Values,
        external_names: vec![external_name.into()],
        dimensions: dimensions.clone(),
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F16,
        },
        required: true,
    });
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions,
        logical_element_type: ElementType::F16,
        physical_layout: PhysicalWeightLayout::Dense { component_id },
        required: true,
    });
}

#[allow(clippy::too_many_arguments)]
fn append_gptq_stack(
    weight_id: WeightId,
    component_id: WeightId,
    logical_dimensions: Vec<u64>,
    stems: Vec<String>,
    quantization: &Qwen3MoeGptqConfig,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    if logical_dimensions.len() < 2 {
        return Err(invalid_config(
            "weights",
            "GPTQ logical tensor must end in a matrix",
        ));
    }
    let source_count = logical_dimensions[..logical_dimensions.len() - 2]
        .iter()
        .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
        .ok_or_else(|| invalid_config("weights", "GPTQ source count overflows"))?;
    if usize::try_from(source_count).ok() != Some(stems.len()) {
        return Err(invalid_config(
            "weights",
            format!(
                "logical tensor {weight_id} expects {source_count} ordered GPTQ sources, got {}",
                stems.len()
            ),
        ));
    }
    let [n, k] = logical_dimensions[logical_dimensions.len() - 2..] else {
        unreachable!("logical matrix suffix has exact length")
    };
    validate_logical_gptq_matrix(n, k, quantization)?;

    let mut packed_sources = Vec::with_capacity(stems.len() * 3);
    let mut scale_sources = Vec::with_capacity(stems.len());
    for stem in stems {
        packed_sources.push(format!("{stem}.qweight"));
        packed_sources.push(format!("{stem}.qzeros"));
        packed_sources.push(format!("{stem}.g_idx"));
        scale_sources.push(format!("{stem}.scales"));
    }

    let mut packed_dimensions = logical_dimensions.clone();
    let group_axis = packed_dimensions.len() - 1;
    packed_dimensions[group_axis] /= 2;
    let mut scale_dimensions = logical_dimensions.clone();
    scale_dimensions[group_axis] /= u64::from(quantization.group_size);
    let packed_id = WeightId::new(format!("{component_id}.packed"))?;
    let scales_id = WeightId::new(format!("{component_id}.scales"))?;
    components.push(WeightComponentSpec {
        id: packed_id.clone(),
        role: WeightComponentRole::PackedValues,
        external_names: packed_sources,
        dimensions: packed_dimensions.clone(),
        encoding: WeightEncoding::Quantized(quantization_spec(quantization)?),
        required: true,
    });
    components.push(WeightComponentSpec {
        id: scales_id.clone(),
        role: WeightComponentRole::Scales,
        external_names: scale_sources,
        dimensions: scale_dimensions,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F16,
        },
        required: true,
    });
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions: logical_dimensions,
        logical_element_type: ElementType::F16,
        physical_layout: PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
            packed_dimensions,
            scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
            zero_points: None,
            zero_point_packed_dimensions: None,
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: u32::try_from(group_axis)
                .map_err(|_| invalid_config("weights", "GPTQ group axis exceeds u32"))?,
            group_padding: PhysicalWeightPadding::Exact,
        },
        required: true,
    });
    Ok(())
}

fn quantization_spec(quantization: &Qwen3MoeGptqConfig) -> Result<QuantizationSpec, VNextError> {
    quantization
        .validate()
        .map_err(|reason| invalid_config("quantization", reason))?;
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(GPTQ_MARLIN_INT4_FORMAT_ID)?,
        bits_per_weight: quantization.bits,
        grouping: QuantizationGrouping::fixed(quantization.group_size),
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: None,
    })
}

fn expected_gguf_tensors(
    semantic: &Qwen3MoeSemanticConfig,
) -> Result<Vec<(String, Vec<u64>)>, String> {
    let hidden = semantic.hidden_size;
    let query = semantic.query_features()?;
    let kv = semantic.kv_features()?;
    let mut tensors = vec![
        gguf_expected(
            "model.embed_tokens.weight",
            vec![semantic.vocabulary_size, hidden],
        )?,
        gguf_expected("model.norm.weight", vec![hidden])?,
    ];
    if !semantic.tie_word_embeddings {
        tensors.push(gguf_expected(
            "lm_head.weight",
            vec![semantic.vocabulary_size, hidden],
        )?);
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
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![semantic.head_dim],
            ),
            (
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![semantic.head_dim],
            ),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![query, hidden],
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv, hidden],
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, hidden],
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, query],
            ),
            (
                format!("{prefix}.mlp.router.weight"),
                vec![semantic.expert_count, hidden],
            ),
            (
                format!("{prefix}.mlp.gate_exps.weight"),
                vec![
                    semantic.expert_count,
                    semantic.expert_intermediate_size,
                    hidden,
                ],
            ),
            (
                format!("{prefix}.mlp.up_exps.weight"),
                vec![
                    semantic.expert_count,
                    semantic.expert_intermediate_size,
                    hidden,
                ],
            ),
            (
                format!("{prefix}.mlp.down_exps.weight"),
                vec![
                    semantic.expert_count,
                    hidden,
                    semantic.expert_intermediate_size,
                ],
            ),
        ] {
            tensors.push(gguf_expected(&name, dimensions)?);
        }
    }
    Ok(tensors)
}

fn gguf_expected(name: &str, dimensions: Vec<u64>) -> Result<(String, Vec<u64>), String> {
    let external_name = ferrum_to_gguf_with_arch("qwen3moe", name)
        .ok_or_else(|| format!("Qwen3 MoE GGUF has no typed name mapping for {name:?}"))?;
    Ok((external_name, dimensions))
}

fn load_gguf_tensor(
    source: &GgufWeightComponentSource,
    (external_name, logical_dimensions): (String, Vec<u64>),
) -> Result<Qwen3MoeGgufTensor, String> {
    let info = source
        .file()
        .tensor_info(&external_name)
        .ok_or_else(|| format!("Qwen3 MoE GGUF is missing required tensor {external_name:?}"))?;
    let dimensions = info
        .shape
        .dims()
        .iter()
        .map(|dimension| *dimension as u64)
        .collect::<Vec<_>>();
    if dimensions != logical_dimensions {
        return Err(format!(
            "Qwen3 MoE GGUF tensor {external_name:?} has dimensions {dimensions:?}, expected exactly {logical_dimensions:?}"
        ));
    }
    Ok(Qwen3MoeGgufTensor {
        external_name,
        dimensions,
        encoding: gguf_encoding(info.ggml_dtype)?,
    })
}

fn gguf_encoding(dtype: GgmlDType) -> Result<Qwen3MoeGgufEncoding, String> {
    if let Some(format_id) = block_quantization_format(dtype) {
        return Ok(Qwen3MoeGgufEncoding::BlockQuantized(
            BlockQuantizationSpec {
                format_id: format_id
                    .to_owned()
                    .try_into()
                    .map_err(|error: VNextError| error.to_string())?,
                logical_values_per_block: u32::try_from(dtype.block_size())
                    .map_err(|_| "GGUF logical block width exceeds u32".to_owned())?,
                bytes_per_block: u32::try_from(dtype.type_size())
                    .map_err(|_| "GGUF physical block size exceeds u32".to_owned())?,
            },
        ));
    }
    let source_element_type = match dtype {
        GgmlDType::F16 => ElementType::F16,
        GgmlDType::BF16 => ElementType::Bf16,
        GgmlDType::F32 => ElementType::F32,
        _ => return Err(format!("unsupported Qwen3 MoE GGUF tensor dtype {dtype:?}")),
    };
    Ok(Qwen3MoeGgufEncoding::Dense {
        source_element_type,
    })
}

fn validate_gguf_tensors(
    semantic: &Qwen3MoeSemanticConfig,
    tensors: &[Qwen3MoeGgufTensor],
) -> Result<(), VNextError> {
    let expected =
        expected_gguf_tensors(semantic).map_err(|reason| invalid_config("weights.gguf", reason))?;
    if tensors.len() != expected.len() {
        return Err(invalid_config(
            "weights.gguf",
            format!(
                "GGUF manifest contains {} required tensors, expected {}",
                tensors.len(),
                expected.len()
            ),
        ));
    }
    for (tensor, (external_name, logical_dimensions)) in tensors.iter().zip(expected) {
        if tensor.external_name != external_name {
            return Err(invalid_config(
                "weights.gguf",
                format!(
                    "GGUF manifest tensor order/name differs: got {:?}, expected {external_name:?}",
                    tensor.external_name
                ),
            ));
        }
        if tensor.dimensions != logical_dimensions {
            return Err(invalid_config(
                "weights.gguf",
                format!(
                    "GGUF tensor {:?} dimensions {:?} differ from exact expected dimensions {logical_dimensions:?}",
                    tensor.external_name, tensor.dimensions
                ),
            ));
        }
        if let Qwen3MoeGgufEncoding::BlockQuantized(spec) = &tensor.encoding {
            spec.validate()?;
            let innermost =
                tensor.dimensions.last().copied().ok_or_else(|| {
                    invalid_config("weights.gguf", "GGUF tensor has no block axis")
                })?;
            if !innermost.is_multiple_of(u64::from(spec.logical_values_per_block)) {
                return Err(invalid_config(
                    "weights.gguf",
                    format!(
                        "GGUF tensor {:?} innermost dimension is not block aligned",
                        tensor.external_name
                    ),
                ));
            }
        } else if let Qwen3MoeGgufEncoding::Dense {
            source_element_type,
        } = &tensor.encoding
        {
            if !matches!(
                source_element_type,
                ElementType::F16 | ElementType::Bf16 | ElementType::F32
            ) {
                return Err(invalid_config(
                    "weights.gguf",
                    format!(
                        "GGUF dense tensor {:?} has non-floating source type {source_element_type:?}",
                        tensor.external_name
                    ),
                ));
            }
        }
        validate_locked_gguf_expert_abi(tensor, &logical_dimensions)?;
    }
    Ok(())
}

fn validate_locked_gguf_expert_abi(
    tensor: &Qwen3MoeGgufTensor,
    logical_dimensions: &[u64],
) -> Result<(), VNextError> {
    let expected_formats: &[&str] = if tensor.external_name.ends_with(".ffn_gate_exps.weight")
        || tensor.external_name.ends_with(".ffn_up_exps.weight")
    {
        &["quantization.gguf.q4-k"]
    } else if tensor.external_name.ends_with(".ffn_down_exps.weight") {
        &["quantization.gguf.q4-k", "quantization.gguf.q6-k"]
    } else {
        &[]
    };
    if !expected_formats.is_empty() {
        if tensor.dimensions != logical_dimensions {
            return Err(invalid_config(
                "weights.gguf",
                format!(
                    "expert tensor {:?} must use exact expert-major dimensions {logical_dimensions:?}, got {:?}",
                    tensor.external_name, tensor.dimensions
                ),
            ));
        }
        let Qwen3MoeGgufEncoding::BlockQuantized(spec) = &tensor.encoding else {
            return Err(invalid_config(
                "weights.gguf",
                format!(
                    "expert tensor {:?} must use one of {expected_formats:?}",
                    tensor.external_name,
                ),
            ));
        };
        if !expected_formats.contains(&spec.format_id.as_str()) {
            return Err(invalid_config(
                "weights.gguf",
                format!(
                    "expert tensor {:?} uses {}, expected one of {expected_formats:?}",
                    tensor.external_name, spec.format_id,
                ),
            ));
        }
    }
    if tensor.external_name.ends_with(".ffn_gate_inp.weight") {
        if tensor.dimensions != logical_dimensions
            || !matches!(
                &tensor.encoding,
                Qwen3MoeGgufEncoding::Dense {
                    source_element_type: ElementType::F32
                }
            )
        {
            return Err(invalid_config(
                "weights.gguf",
                format!(
                    "router tensor {:?} must retain exact expert-major F32 dimensions {logical_dimensions:?}",
                    tensor.external_name
                ),
            ));
        }
    }
    Ok(())
}

fn gguf_weight_schema(
    semantic: &Qwen3MoeSemanticConfig,
    source_tensors: &[Qwen3MoeGgufTensor],
) -> Result<WeightSchema, VNextError> {
    let hidden = semantic.hidden_size;
    let query = semantic
        .query_features()
        .map_err(|reason| invalid_config("semantic", reason))?;
    let kv = semantic
        .kv_features()
        .map_err(|reason| invalid_config("semantic", reason))?;
    let mut components = Vec::new();
    let mut tensors = Vec::new();

    append_gguf_direct(
        gguf_tensor_for(source_tensors, "model.embed_tokens.weight")?,
        global_weight_id(EMBED_TOKENS_ROLE)?,
        global_component_id(EMBED_TOKENS_ROLE)?,
        vec![semantic.vocabulary_size, hidden],
        &mut components,
        &mut tensors,
    )?;
    append_gguf_direct(
        gguf_tensor_for(source_tensors, "model.norm.weight")?,
        global_weight_id(FINAL_NORM_ROLE)?,
        global_component_id(FINAL_NORM_ROLE)?,
        vec![hidden],
        &mut components,
        &mut tensors,
    )?;
    if !semantic.tie_word_embeddings {
        append_gguf_direct(
            gguf_tensor_for(source_tensors, "lm_head.weight")?,
            global_weight_id(LM_HEAD_ROLE)?,
            global_component_id(LM_HEAD_ROLE)?,
            vec![semantic.vocabulary_size, hidden],
            &mut components,
            &mut tensors,
        )?;
    }

    for layer_index in 0..semantic.layer_count {
        let layer_index = u32::try_from(layer_index)
            .map_err(|_| invalid_config("semantic.layer_count", "layer index exceeds u32"))?;
        let prefix = format!("model.layers.{layer_index}");
        for (role, name, dimensions) in [
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
                Q_NORM_ROLE,
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![semantic.head_dim],
            ),
            (
                K_NORM_ROLE,
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![semantic.head_dim],
            ),
            (
                ROUTER_ROLE,
                format!("{prefix}.mlp.router.weight"),
                vec![semantic.expert_count, hidden],
            ),
        ] {
            append_gguf_direct(
                gguf_tensor_for(source_tensors, &name)?,
                layer_weight_id(layer_index, role)?,
                layer_component_id(layer_index, role)?,
                dimensions,
                &mut components,
                &mut tensors,
            )?;
        }
        for (role, name, dimensions) in [
            (
                Q_PROJ_ROLE,
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![query, hidden],
            ),
            (
                K_PROJ_ROLE,
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv, hidden],
            ),
            (
                V_PROJ_ROLE,
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, hidden],
            ),
            (
                O_PROJ_ROLE,
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, query],
            ),
        ] {
            append_gguf_direct(
                gguf_tensor_for(source_tensors, &name)?,
                layer_weight_id(layer_index, role)?,
                layer_component_id(layer_index, role)?,
                dimensions,
                &mut components,
                &mut tensors,
            )?;
        }

        append_gguf_composite(
            [
                gguf_tensor_for(source_tensors, &format!("{prefix}.mlp.gate_exps.weight"))?,
                gguf_tensor_for(source_tensors, &format!("{prefix}.mlp.up_exps.weight"))?,
            ],
            layer_weight_id(layer_index, ROUTED_GATE_UP_ROLE)?,
            [
                layer_component_id(layer_index, "moe_routed_gate")?,
                layer_component_id(layer_index, "moe_routed_up")?,
            ],
            vec![
                semantic.expert_count,
                2,
                semantic.expert_intermediate_size,
                hidden,
            ],
            1,
            &mut components,
            &mut tensors,
        )?;
        append_gguf_direct(
            gguf_tensor_for(source_tensors, &format!("{prefix}.mlp.down_exps.weight"))?,
            layer_weight_id(layer_index, ROUTED_DOWN_ROLE)?,
            layer_component_id(layer_index, ROUTED_DOWN_ROLE)?,
            vec![
                semantic.expert_count,
                hidden,
                semantic.expert_intermediate_size,
            ],
            &mut components,
            &mut tensors,
        )?;
    }

    Ok(WeightSchema {
        format_id: WeightFormatId::new("weight-format.gguf.native-block")?,
        layout_id: WeightLayoutId::new("weight-layout.qwen3_moe.routed.gguf.native.stacked")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    })
}

fn gguf_tensor_for<'a>(
    tensors: &'a [Qwen3MoeGgufTensor],
    ferrum_name: &str,
) -> Result<&'a Qwen3MoeGgufTensor, VNextError> {
    let external_name = ferrum_to_gguf_with_arch("qwen3moe", ferrum_name).ok_or_else(|| {
        invalid_config(
            "weights.gguf",
            format!("no typed GGUF name mapping for {ferrum_name:?}"),
        )
    })?;
    tensors
        .iter()
        .find(|tensor| tensor.external_name == external_name)
        .ok_or_else(|| {
            invalid_config(
                "weights.gguf",
                format!("missing typed GGUF tensor {external_name:?}"),
            )
        })
}

fn append_gguf_direct(
    source: &Qwen3MoeGgufTensor,
    weight_id: WeightId,
    component_id: WeightId,
    logical_dimensions: Vec<u64>,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let component = gguf_component_spec(source, component_id.clone())?;
    let layout = gguf_component_layout(source, component_id, &logical_dimensions)?;
    components.push(component);
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions: logical_dimensions,
        logical_element_type: ElementType::F16,
        physical_layout: layout,
        required: true,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_gguf_composite(
    sources: [&Qwen3MoeGgufTensor; 2],
    weight_id: WeightId,
    component_ids: [WeightId; 2],
    logical_dimensions: Vec<u64>,
    partition_axis: usize,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    if logical_dimensions.get(partition_axis).copied() != Some(2) {
        return Err(invalid_config(
            "weights.gguf",
            "routed gate/up logical tensor must expose two ordered partitions",
        ));
    }
    let mut part_dimensions = logical_dimensions.clone();
    part_dimensions[partition_axis] = 1;
    let mut parts = Vec::with_capacity(2);
    for (partition, (source, component_id)) in sources.into_iter().zip(component_ids).enumerate() {
        let component = gguf_component_spec(source, component_id.clone())?;
        let layout = gguf_component_layout(source, component_id, &part_dimensions)?;
        let mut logical_offsets = vec![0_u64; logical_dimensions.len()];
        logical_offsets[partition_axis] = partition as u64;
        parts.push(CompositeWeightPart {
            layout: Box::new(layout),
            logical_offsets,
            extents: part_dimensions.clone(),
        });
        components.push(component);
    }
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions: logical_dimensions,
        logical_element_type: ElementType::F16,
        physical_layout: PhysicalWeightLayout::Composite { parts },
        required: true,
    });
    Ok(())
}

fn gguf_component_spec(
    source: &Qwen3MoeGgufTensor,
    component_id: WeightId,
) -> Result<WeightComponentSpec, VNextError> {
    let (role, dimensions, encoding) = match &source.encoding {
        Qwen3MoeGgufEncoding::Dense { .. } => (
            WeightComponentRole::Values,
            source.dimensions.clone(),
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
        ),
        Qwen3MoeGgufEncoding::BlockQuantized(spec) => {
            let mut dimensions = source.dimensions.clone();
            let innermost = dimensions.last_mut().ok_or_else(|| {
                invalid_config("weights.gguf", "GGUF block tensor has no block axis")
            })?;
            let block_width = u64::from(spec.logical_values_per_block);
            if !innermost.is_multiple_of(block_width) {
                return Err(invalid_config(
                    "weights.gguf",
                    "GGUF block tensor innermost dimension is not block aligned",
                ));
            }
            *innermost /= block_width;
            (
                WeightComponentRole::PackedValues,
                dimensions,
                WeightEncoding::BlockQuantized(spec.clone()),
            )
        }
    };
    Ok(WeightComponentSpec {
        id: component_id,
        role,
        external_names: vec![source.external_name.clone()],
        dimensions,
        encoding,
        required: true,
    })
}

fn gguf_component_layout(
    source: &Qwen3MoeGgufTensor,
    component_id: WeightId,
    logical_dimensions: &[u64],
) -> Result<PhysicalWeightLayout, VNextError> {
    match &source.encoding {
        Qwen3MoeGgufEncoding::Dense { .. } => dense_or_reshaped_layout(
            super::FAMILY_ID,
            component_id,
            &source.dimensions,
            logical_dimensions,
        ),
        Qwen3MoeGgufEncoding::BlockQuantized(spec) => {
            let block_axis = logical_dimensions
                .len()
                .checked_sub(1)
                .ok_or_else(|| invalid_config("weights.gguf", "GGUF tensor has no block axis"))?;
            let block_width = u64::from(spec.logical_values_per_block);
            let mut logical_blocks = logical_dimensions.to_vec();
            let logical_innermost = logical_blocks.last_mut().unwrap();
            if !logical_innermost.is_multiple_of(block_width) {
                return Err(invalid_config(
                    "weights.gguf",
                    "GGUF logical tensor innermost dimension is not block aligned",
                ));
            }
            *logical_innermost /= block_width;
            let mut physical_blocks = source.dimensions.clone();
            let physical_innermost = physical_blocks.last_mut().ok_or_else(|| {
                invalid_config("weights.gguf", "GGUF block tensor has no block axis")
            })?;
            if !physical_innermost.is_multiple_of(block_width) {
                return Err(invalid_config(
                    "weights.gguf",
                    "GGUF physical tensor innermost dimension is not block aligned",
                ));
            }
            *physical_innermost /= block_width;
            Ok(PhysicalWeightLayout::BlockQuantized {
                blocks: contiguous_or_reshaped_binding(
                    super::FAMILY_ID,
                    component_id,
                    &physical_blocks,
                    &logical_blocks,
                )?,
                block_axis: u32::try_from(block_axis)
                    .map_err(|_| invalid_config("weights.gguf", "GGUF block axis exceeds u32"))?,
                block_padding: PhysicalWeightPadding::Exact,
            })
        }
    }
}

fn validate_archive(
    archive: &SafetensorsArchive,
    semantic: &Qwen3MoeSemanticConfig,
    quantization: &Qwen3MoeGptqConfig,
    expected_count: u64,
) -> Result<(), VNextError> {
    if u64::try_from(archive.tensor_count()).ok() != Some(expected_count) {
        return Err(invalid_config(
            "weights",
            format!(
                "checkpoint contains {} tensors, expected exactly {expected_count}",
                archive.tensor_count()
            ),
        ));
    }
    let observed =
        visit_expected_tensors(semantic, quantization, |name, element_type, dimensions| {
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
    if observed != expected_count {
        return Err(invalid_config(
            "weights",
            "internal Qwen3 MoE tensor cardinality drift",
        ));
    }
    Ok(())
}

pub(super) fn expected_manifest(
    semantic: &Qwen3MoeSemanticConfig,
    quantization: &Qwen3MoeGptqConfig,
) -> Result<Qwen3MoeWeightManifest, VNextError> {
    let mut hasher = Sha256::new();
    hasher.update([STRUCTURE_FINGERPRINT_VERSION]);
    let tensor_count =
        visit_expected_tensors(semantic, quantization, |name, element_type, dimensions| {
            hash_header(&mut hasher, &name, element_type, &dimensions);
            Ok(())
        })?;
    Ok(Qwen3MoeWeightManifest::SafetensorsGptqMarlin {
        quantization: quantization.clone(),
        tensor_count,
        structure_fingerprint: format!("{:x}", hasher.finalize()),
    })
}

fn visit_expected_tensors(
    semantic: &Qwen3MoeSemanticConfig,
    quantization: &Qwen3MoeGptqConfig,
    mut visitor: impl FnMut(String, ElementType, Vec<u64>) -> Result<(), VNextError>,
) -> Result<u64, VNextError> {
    let hidden = semantic.hidden_size;
    let query = semantic
        .query_features()
        .map_err(|reason| invalid_config("semantic", reason))?;
    let kv = semantic
        .kv_features()
        .map_err(|reason| invalid_config("semantic", reason))?;
    let mut count = 0_u64;
    emit(
        &mut visitor,
        &mut count,
        "model.embed_tokens.weight".to_owned(),
        ElementType::F16,
        vec![semantic.vocabulary_size, hidden],
    )?;
    emit(
        &mut visitor,
        &mut count,
        "model.norm.weight".to_owned(),
        ElementType::F16,
        vec![hidden],
    )?;
    if !semantic.tie_word_embeddings {
        emit(
            &mut visitor,
            &mut count,
            "lm_head.weight".to_owned(),
            ElementType::F16,
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
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![semantic.head_dim],
            ),
            (
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![semantic.head_dim],
            ),
            (
                format!("{prefix}.mlp.gate.weight"),
                vec![semantic.expert_count, hidden],
            ),
        ] {
            emit(&mut visitor, &mut count, name, ElementType::F16, dimensions)?;
        }
        for (stem, n, k) in [
            (format!("{prefix}.self_attn.q_proj"), query, hidden),
            (format!("{prefix}.self_attn.k_proj"), kv, hidden),
            (format!("{prefix}.self_attn.v_proj"), kv, hidden),
            (format!("{prefix}.self_attn.o_proj"), hidden, query),
        ] {
            emit_gptq(&mut visitor, &mut count, stem, n, k, quantization)?;
        }
        for expert_index in 0..semantic.expert_count {
            let expert = format!("{prefix}.mlp.experts.{expert_index}");
            for (projection, n, k) in [
                ("gate_proj", semantic.expert_intermediate_size, hidden),
                ("up_proj", semantic.expert_intermediate_size, hidden),
                ("down_proj", hidden, semantic.expert_intermediate_size),
            ] {
                emit_gptq(
                    &mut visitor,
                    &mut count,
                    format!("{expert}.{projection}"),
                    n,
                    k,
                    quantization,
                )?;
            }
        }
    }
    Ok(count)
}

fn emit(
    visitor: &mut impl FnMut(String, ElementType, Vec<u64>) -> Result<(), VNextError>,
    count: &mut u64,
    name: String,
    element_type: ElementType,
    dimensions: Vec<u64>,
) -> Result<(), VNextError> {
    visitor(name, element_type, dimensions)?;
    *count = count
        .checked_add(1)
        .ok_or_else(|| invalid_config("weights", "tensor count overflows u64"))?;
    Ok(())
}

fn emit_gptq(
    visitor: &mut impl FnMut(String, ElementType, Vec<u64>) -> Result<(), VNextError>,
    count: &mut u64,
    stem: String,
    n: u64,
    k: u64,
    quantization: &Qwen3MoeGptqConfig,
) -> Result<(), VNextError> {
    validate_logical_gptq_matrix(n, k, quantization)?;
    let group_size = u64::from(quantization.group_size);
    for (suffix, element_type, dimensions) in [
        ("qweight", ElementType::I32, vec![k / 8, n]),
        ("scales", ElementType::F16, vec![k / group_size, n]),
        ("qzeros", ElementType::I32, vec![k / group_size, n / 8]),
        ("g_idx", ElementType::I32, vec![k]),
    ] {
        emit(
            visitor,
            count,
            format!("{stem}.{suffix}"),
            element_type,
            dimensions,
        )?;
    }
    Ok(())
}

fn validate_logical_gptq_matrix(
    n: u64,
    k: u64,
    quantization: &Qwen3MoeGptqConfig,
) -> Result<(), VNextError> {
    let group_size = u64::from(quantization.group_size);
    if !n.is_multiple_of(16) || !k.is_multiple_of(16) || !k.is_multiple_of(group_size) {
        return Err(invalid_config(
            "weights",
            format!("GPTQ logical matrix [{n}, {k}] is not Marlin/group aligned"),
        ));
    }
    Ok(())
}

fn hash_header(hasher: &mut Sha256, name: &str, element_type: ElementType, dimensions: &[u64]) {
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update([match element_type {
        ElementType::F16 => 1,
        ElementType::I32 => 2,
        _ => unreachable!("Qwen3 MoE header fingerprint only emits F16 and I32"),
    }]);
    hasher.update((dimensions.len() as u64).to_le_bytes());
    for extent in dimensions {
        hasher.update(extent.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ferrum_interfaces::vnext::{
        CanonicalRational, ModelFamilyId, ROUTED_SWIGLU_MOE_OPERATION_ID,
    };
    use half::f16;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::*;

    fn fixture_semantics() -> Qwen3MoeSemanticConfig {
        Qwen3MoeSemanticConfig {
            hidden_size: 256,
            layer_count: 1,
            attention_head_count: 4,
            kv_head_count: 1,
            head_dim: 64,
            vocabulary_size: 128,
            maximum_sequence_tokens: 256,
            expert_count: 2,
            experts_per_token: 1,
            expert_intermediate_size: 256,
            normalize_topk: true,
            rms_norm_epsilon: CanonicalRational::new(1, 1_000_000).unwrap(),
            rope_theta: CanonicalRational::new(1_000_000, 1).unwrap(),
            tie_word_embeddings: false,
        }
    }

    fn fixture_quantization() -> Qwen3MoeGptqConfig {
        Qwen3MoeGptqConfig {
            bits: 4,
            group_size: 128,
            desc_act: false,
            sym: true,
        }
    }

    fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
        Box::leak(bytes.into_boxed_slice())
    }

    fn insert_dense(
        views: &mut BTreeMap<String, TensorView<'static>>,
        name: impl Into<String>,
        dimensions: Vec<usize>,
    ) {
        let elements = dimensions.iter().product::<usize>();
        views.insert(
            name.into(),
            TensorView::new(Dtype::F16, dimensions, leak_bytes(vec![0_u8; elements * 2])).unwrap(),
        );
    }

    fn insert_gptq(
        views: &mut BTreeMap<String, TensorView<'static>>,
        stem: &str,
        n: usize,
        k: usize,
        group_size: usize,
    ) {
        let qweight = (0..(k / 8) * n)
            .flat_map(|_| 0_i32.to_le_bytes())
            .collect::<Vec<_>>();
        let qzeros = (0..(k / group_size) * (n / 8))
            .flat_map(|_| (0x8888_8888_u32 as i32).to_le_bytes())
            .collect::<Vec<_>>();
        let scales = (0..(k / group_size) * n)
            .flat_map(|_| f16::from_f32(1.0).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let g_idx = (0..k)
            .flat_map(|index| ((index / group_size) as i32).to_le_bytes())
            .collect::<Vec<_>>();
        views.insert(
            format!("{stem}.qweight"),
            TensorView::new(Dtype::I32, vec![k / 8, n], leak_bytes(qweight)).unwrap(),
        );
        views.insert(
            format!("{stem}.qzeros"),
            TensorView::new(Dtype::I32, vec![k / group_size, n / 8], leak_bytes(qzeros)).unwrap(),
        );
        views.insert(
            format!("{stem}.scales"),
            TensorView::new(Dtype::F16, vec![k / group_size, n], leak_bytes(scales)).unwrap(),
        );
        views.insert(
            format!("{stem}.g_idx"),
            TensorView::new(Dtype::I32, vec![k], leak_bytes(g_idx)).unwrap(),
        );
    }

    fn fixture_archive() -> (tempfile::TempDir, SafetensorsArchive) {
        let semantic = fixture_semantics();
        let hidden = semantic.hidden_size as usize;
        let query = semantic.query_features().unwrap() as usize;
        let kv = semantic.kv_features().unwrap() as usize;
        let intermediate = semantic.expert_intermediate_size as usize;
        let group_size = fixture_quantization().group_size as usize;
        let mut views = BTreeMap::new();
        insert_dense(
            &mut views,
            "model.embed_tokens.weight",
            vec![semantic.vocabulary_size as usize, hidden],
        );
        insert_dense(&mut views, "model.norm.weight", vec![hidden]);
        insert_dense(
            &mut views,
            "lm_head.weight",
            vec![semantic.vocabulary_size as usize, hidden],
        );
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ] {
            insert_dense(&mut views, name, vec![hidden]);
        }
        for name in [
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ] {
            insert_dense(&mut views, name, vec![semantic.head_dim as usize]);
        }
        insert_dense(
            &mut views,
            "model.layers.0.mlp.gate.weight",
            vec![semantic.expert_count as usize, hidden],
        );
        for (role, n, k) in [
            ("q_proj", query, hidden),
            ("k_proj", kv, hidden),
            ("v_proj", kv, hidden),
            ("o_proj", hidden, query),
        ] {
            insert_gptq(
                &mut views,
                &format!("model.layers.0.self_attn.{role}"),
                n,
                k,
                group_size,
            );
        }
        for expert_index in 0..semantic.expert_count {
            let prefix = format!("model.layers.0.mlp.experts.{expert_index}");
            insert_gptq(
                &mut views,
                &format!("{prefix}.gate_proj"),
                intermediate,
                hidden,
                group_size,
            );
            insert_gptq(
                &mut views,
                &format!("{prefix}.up_proj"),
                intermediate,
                hidden,
                group_size,
            );
            insert_gptq(
                &mut views,
                &format!("{prefix}.down_proj"),
                hidden,
                intermediate,
                group_size,
            );
        }
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

    fn fixture_gguf_manifest(semantic: &Qwen3MoeSemanticConfig) -> Qwen3MoeWeightManifest {
        Qwen3MoeWeightManifest::GgufNative {
            tensors: expected_gguf_tensors(semantic)
                .unwrap()
                .into_iter()
                .map(|(external_name, dimensions)| Qwen3MoeGgufTensor {
                    encoding: if external_name.ends_with(".ffn_gate_exps.weight")
                        || external_name.ends_with(".ffn_up_exps.weight")
                        || (external_name.starts_with("blk.0.")
                            && external_name.ends_with(".ffn_down_exps.weight"))
                    {
                        gguf_encoding(GgmlDType::Q4K).unwrap()
                    } else if external_name.ends_with(".ffn_down_exps.weight") {
                        gguf_encoding(GgmlDType::Q6K).unwrap()
                    } else if external_name.ends_with(".ffn_gate_inp.weight") {
                        Qwen3MoeGgufEncoding::Dense {
                            source_element_type: ElementType::F32,
                        }
                    } else {
                        Qwen3MoeGgufEncoding::Dense {
                            source_element_type: ElementType::F16,
                        }
                    },
                    external_name,
                    dimensions,
                })
                .collect(),
        }
    }

    #[test]
    fn production_m3_header_contract_has_exact_cardinality() {
        let mut semantic = fixture_semantics();
        semantic.hidden_size = 2048;
        semantic.layer_count = 48;
        semantic.attention_head_count = 32;
        semantic.kv_head_count = 4;
        semantic.head_dim = 128;
        semantic.vocabulary_size = 151_936;
        semantic.maximum_sequence_tokens = 40_960;
        semantic.expert_count = 128;
        semantic.experts_per_token = 8;
        semantic.expert_intermediate_size = 768;
        let manifest = expected_manifest(&semantic, &fixture_quantization()).unwrap();
        let gguf_tensors = expected_gguf_tensors(&semantic).unwrap();
        let gguf_names = gguf_tensors
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();

        let Qwen3MoeWeightManifest::SafetensorsGptqMarlin {
            tensor_count,
            structure_fingerprint,
            ..
        } = manifest
        else {
            panic!("expected GPTQ manifest")
        };
        assert_eq!(tensor_count, 74_739);
        assert_eq!(structure_fingerprint.len(), 64);
        assert_eq!(gguf_tensors.len(), 579);
        assert_eq!(gguf_names.len(), 579);
    }

    #[test]
    fn gguf_native_schema_keeps_the_gptq_logical_program_and_stacked_experts() {
        let mut semantic = fixture_semantics();
        semantic.layer_count = 2;
        let gguf = fixture_gguf_manifest(&semantic);
        let gptq = expected_manifest(&semantic, &fixture_quantization()).unwrap();
        let family_id = ModelFamilyId::new(super::super::FAMILY_ID).unwrap();
        let gguf_schema = gguf.weight_schema(&semantic).unwrap();
        gguf_schema.validate(&family_id).unwrap();
        let gptq_program =
            super::super::program::build_semantic_program(&family_id, &semantic, &gptq).unwrap();
        let gguf_program =
            super::super::program::build_semantic_program(&family_id, &semantic, &gguf).unwrap();

        assert_eq!(gguf_program, gptq_program);
        assert_eq!(
            gguf_schema.format_id.as_str(),
            "weight-format.gguf.native-block"
        );
        assert!(gguf_schema
            .components
            .iter()
            .any(|component| component.external_names.len() == 1
                && component.external_names[0] == "blk.0.ffn_gate_exps.weight"));
        assert!(gguf_schema
            .components
            .iter()
            .any(|component| component.external_names.len() == 1
                && component.external_names[0] == "blk.0.ffn_up_exps.weight"));
        for (layer_index, expected_format) in [
            (0_u32, "quantization.gguf.q4-k"),
            (1_u32, "quantization.gguf.q6-k"),
        ] {
            let component_id = layer_component_id(layer_index, ROUTED_DOWN_ROLE).unwrap();
            let component = gguf_schema
                .components
                .iter()
                .find(|component| component.id == component_id)
                .unwrap();
            let WeightEncoding::BlockQuantized(spec) = &component.encoding else {
                panic!("layer {layer_index} down component must remain block-quantized")
            };
            assert_eq!(component.role, WeightComponentRole::PackedValues);
            assert_eq!(spec.format_id.as_str(), expected_format);
        }
        let routed_gate_up = gguf_schema
            .tensors
            .iter()
            .find(|tensor| tensor.id == layer_weight_id(0, ROUTED_GATE_UP_ROLE).unwrap())
            .unwrap();
        let PhysicalWeightLayout::Composite { parts } = &routed_gate_up.physical_layout else {
            panic!("routed gate/up must retain two ordered GGUF components")
        };
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].logical_offsets, [0, 0, 0, 0]);
        assert_eq!(parts[1].logical_offsets, [0, 1, 0, 0]);
        assert_eq!(
            parts[0].extents,
            [
                semantic.expert_count,
                1,
                semantic.expert_intermediate_size,
                semantic.hidden_size,
            ]
        );
        for (part, role) in parts.iter().zip(["moe_routed_gate", "moe_routed_up"]) {
            let PhysicalWeightLayout::BlockQuantized {
                blocks, block_axis, ..
            } = part.layout.as_ref()
            else {
                panic!("gate/up part must remain block-quantized")
            };
            assert_eq!(*block_axis, 3);
            assert_eq!(blocks.component_id, layer_component_id(0, role).unwrap());
        }
        let router_component = gguf_schema
            .components
            .iter()
            .find(|component| component.id == layer_component_id(0, ROUTER_ROLE).unwrap())
            .unwrap();
        assert_eq!(
            router_component.encoding,
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            }
        );
        let Qwen3MoeWeightManifest::GgufNative { tensors } = &gguf else {
            unreachable!()
        };
        let router_source = tensors
            .iter()
            .find(|tensor| tensor.external_name == "blk.0.ffn_gate_inp.weight")
            .unwrap();
        assert!(matches!(
            &router_source.encoding,
            Qwen3MoeGgufEncoding::Dense {
                source_element_type: ElementType::F32
            }
        ));
    }

    #[test]
    fn gguf_manifest_rejects_same_element_wrong_shape_and_extra_tensor() {
        let semantic = fixture_semantics();
        let mut wrong_shape = fixture_gguf_manifest(&semantic);
        let Qwen3MoeWeightManifest::GgufNative { tensors } = &mut wrong_shape else {
            unreachable!()
        };
        assert_eq!(tensors[0].dimensions, [128, 256]);
        tensors[0].dimensions = vec![256, 128];
        let error = wrong_shape.validate(&semantic).unwrap_err().to_string();
        assert!(error.contains("exact expected dimensions"), "{error}");

        let mut extra = fixture_gguf_manifest(&semantic);
        let Qwen3MoeWeightManifest::GgufNative { tensors } = &mut extra else {
            unreachable!()
        };
        tensors.push(tensors[0].clone());
        let error = extra.validate(&semantic).unwrap_err().to_string();
        assert!(error.contains("expected"), "{error}");
        assert!(error.contains("required tensors"), "{error}");
    }

    #[test]
    fn routed_only_manifest_builds_one_stable_program_and_physical_schema() {
        let semantic = fixture_semantics();
        let (_directory, archive) = fixture_archive();
        let manifest =
            Qwen3MoeWeightManifest::load(&archive, &semantic, &fixture_quantization()).unwrap();
        let schema = manifest.weight_schema(&semantic).unwrap();
        let family_id = ModelFamilyId::new(super::super::FAMILY_ID).unwrap();
        schema.validate(&family_id).unwrap();
        let program =
            super::super::program::build_semantic_program(&family_id, &semantic, &manifest)
                .unwrap();

        let moe = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .find(|node| node.operation_id.as_str() == ROUTED_SWIGLU_MOE_OPERATION_ID)
            .expect("fixture program must contain routed-only MoE");
        assert_eq!(moe.inputs.len(), 4);
        let greedy = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .find(|node| {
                node.operation_id.as_str()
                    == ferrum_interfaces::vnext::LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID
            })
            .expect("fixture program must contain typed greedy selection");
        assert_eq!(
            greedy.required_version,
            ferrum_interfaces::vnext::ContractVersion::new(3, 0)
        );
        assert_eq!(
            greedy
                .inputs
                .iter()
                .map(ferrum_interfaces::vnext::ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            [
                "value.output.logits",
                "value.input.greedy_token_mask",
                "value.input.greedy_repetition_token_ids",
                "value.input.greedy_repetition_offsets",
                "value.input.greedy_repetition_penalty",
            ]
        );
        assert_eq!(
            program
                .outputs()
                .iter()
                .map(ferrum_interfaces::vnext::ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            ["value.output.logits", "value.output.greedy_token"]
        );
        assert_eq!(program.states().len(), 1);
        assert!(schema
            .tensors
            .iter()
            .any(|tensor| tensor.id.as_str().ends_with(ROUTED_GATE_UP_ROLE)));
        assert!(schema
            .tensors
            .iter()
            .all(|tensor| !tensor.id.as_str().contains("shared")));
    }
}
