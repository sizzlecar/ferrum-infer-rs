use std::collections::{BTreeMap, BTreeSet};

use ferrum_interfaces::vnext::{
    ContractVersion, ElementType, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    PhysicalWeightPadding, ProgramValueId, QuantizationFormatId, QuantizationGrouping,
    QuantizationPacking, QuantizationSpec, VNextError, WeightComponentRole, WeightComponentSpec,
    WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightSchema, WeightTensorSpec,
};
use ferrum_quantization::{SafetensorsArchive, COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::config::{Gemma4LayerType, Gemma4SemanticConfig};
use super::invalid_config;

pub(super) const EMBED_TOKENS_ROLE: &str = "embed_tokens";
pub(super) const FINAL_NORM_ROLE: &str = "final_norm";
pub(super) const INPUT_NORM_ROLE: &str = "input_layernorm";
pub(super) const POST_ATTENTION_NORM_ROLE: &str = "post_attention_layernorm";
pub(super) const PRE_FEEDFORWARD_NORM_ROLE: &str = "pre_feedforward_layernorm";
pub(super) const POST_FEEDFORWARD_NORM_ROLE: &str = "post_feedforward_layernorm";
pub(super) const Q_NORM_ROLE: &str = "self_attn_q_norm";
pub(super) const K_NORM_ROLE: &str = "self_attn_k_norm";
pub(super) const Q_PROJ_ROLE: &str = "self_attn_q";
pub(super) const K_PROJ_ROLE: &str = "self_attn_k";
pub(super) const V_PROJ_ROLE: &str = "self_attn_v";
pub(super) const O_PROJ_ROLE: &str = "self_attn_o";
pub(super) const GATE_PROJ_ROLE: &str = "mlp_gate";
pub(super) const UP_PROJ_ROLE: &str = "mlp_up";
pub(super) const DOWN_PROJ_ROLE: &str = "mlp_down";

const STRUCTURE_FINGERPRINT_VERSION: u8 = 1;
const MULTIMODAL_TENSOR_NAMES: &[&str] = &[
    "model.embed_audio.embedding_projection.weight",
    "model.embed_vision.embedding_projection.weight",
    "model.vision_embedder.patch_dense.bias",
    "model.vision_embedder.patch_dense.weight",
    "model.vision_embedder.patch_ln1.bias",
    "model.vision_embedder.patch_ln1.weight",
    "model.vision_embedder.patch_ln2.bias",
    "model.vision_embedder.patch_ln2.weight",
    "model.vision_embedder.pos_embedding",
    "model.vision_embedder.pos_norm.bias",
    "model.vision_embedder.pos_norm.weight",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Gemma4SourceDtype {
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Gemma4NonExecutedReason {
    TiedProjectionDuplicate,
    CompileTimeUnitLayerScale,
    TextOnlyMultimodalComponent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gemma4NonExecutedTensor {
    external_name: String,
    dimensions: Vec<u64>,
    dtype: Gemma4SourceDtype,
    reason: Gemma4NonExecutedReason,
}

/// Compact source proof plus an explicit inventory of every tensor that the
/// text-only program intentionally does not schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Gemma4WeightManifest {
    tensor_count: u64,
    execution_source_tensor_count: u64,
    execution_quantized_weight_count: u64,
    structure_fingerprint: String,
    non_executed_tensors: Vec<Gemma4NonExecutedTensor>,
}

#[derive(Debug, Clone)]
struct ExpectedTensor {
    dtype: Dtype,
    dimensions: Vec<u64>,
}

impl Gemma4WeightManifest {
    pub(super) fn load(
        archive: &SafetensorsArchive,
        semantic: &Gemma4SemanticConfig,
    ) -> Result<Self, String> {
        semantic.validate()?;
        let execution = expected_execution_tensors(semantic)?;
        let non_executed_tensors = load_non_executed_tensors(archive, semantic)?;
        let non_executed_names = non_executed_tensors
            .iter()
            .map(|tensor| tensor.external_name.as_str())
            .collect::<BTreeSet<_>>();
        let expected_total = execution
            .len()
            .checked_add(non_executed_tensors.len())
            .ok_or_else(|| "Gemma 4 source tensor count overflows".to_owned())?;
        if archive.tensor_count() != expected_total {
            let unknown = archive
                .tensor_names()
                .filter(|name| {
                    !execution.contains_key(*name) && !non_executed_names.contains(*name)
                })
                .take(8)
                .collect::<Vec<_>>();
            return Err(format!(
                "Gemma 4 archive contains {} tensors, expected exactly {expected_total}; unknown examples: {unknown:?}",
                archive.tensor_count()
            ));
        }

        for (name, expected) in &execution {
            let tensor = archive.tensor(name).map_err(|error| error.to_string())?;
            if tensor.dtype() != expected.dtype || tensor.shape() != expected.dimensions {
                return Err(format!(
                    "Gemma 4 execution tensor {name:?} must be {:?}{:?}, got {:?}{:?}",
                    expected.dtype,
                    expected.dimensions,
                    tensor.dtype(),
                    tensor.shape()
                ));
            }
        }
        validate_shape_sidecar_payloads(archive, semantic)?;
        validate_unit_layer_scalars(archive, semantic)?;

        let structure_fingerprint = structure_fingerprint(archive)?;
        let execution_quantized_weight_count = quantized_weight_count(semantic)?;
        let manifest = Self {
            tensor_count: u64::try_from(archive.tensor_count())
                .map_err(|_| "Gemma 4 source tensor count exceeds u64".to_owned())?,
            execution_source_tensor_count: u64::try_from(execution.len())
                .map_err(|_| "Gemma 4 execution tensor count exceeds u64".to_owned())?,
            execution_quantized_weight_count,
            structure_fingerprint,
            non_executed_tensors,
        };
        manifest
            .validate(semantic)
            .map_err(|error| error.to_string())?;
        Ok(manifest)
    }

    pub(super) fn validate(&self, semantic: &Gemma4SemanticConfig) -> Result<(), VNextError> {
        semantic
            .validate()
            .map_err(|reason| invalid_config("semantic", reason))?;
        let execution = expected_execution_tensors(semantic)
            .map_err(|reason| invalid_config("weights", reason))?;
        let expected_non_executed = owned_expected_non_executed_names(semantic);
        let actual_non_executed = self
            .non_executed_tensors
            .iter()
            .map(|tensor| tensor.external_name.clone())
            .collect::<BTreeSet<_>>();
        if actual_non_executed.len() != self.non_executed_tensors.len()
            || actual_non_executed != expected_non_executed
        {
            return Err(invalid_config(
                "weights.non_executed_tensors",
                "typed non-executed inventory is incomplete, duplicated, or contains an unknown tensor",
            ));
        }
        for tensor in &self.non_executed_tensors {
            if tensor.dtype != Gemma4SourceDtype::Bf16
                || tensor.dimensions.is_empty()
                || tensor.dimensions.contains(&0)
            {
                return Err(invalid_config(
                    "weights.non_executed_tensors",
                    format!(
                        "invalid non-executed tensor metadata for {:?}",
                        tensor.external_name
                    ),
                ));
            }
            let expected_reason = non_executed_reason(&tensor.external_name, semantic)
                .map_err(|reason| invalid_config("weights.non_executed_tensors", reason))?;
            if tensor.reason != expected_reason {
                return Err(invalid_config(
                    "weights.non_executed_tensors",
                    format!("wrong non-execution reason for {:?}", tensor.external_name),
                ));
            }
            match expected_reason {
                Gemma4NonExecutedReason::TiedProjectionDuplicate
                    if tensor.dimensions != [semantic.vocabulary_size, semantic.hidden_size] =>
                {
                    return Err(invalid_config(
                        "weights.non_executed_tensors",
                        "tied lm_head duplicate has the wrong shape",
                    ));
                }
                Gemma4NonExecutedReason::CompileTimeUnitLayerScale if tensor.dimensions != [1] => {
                    return Err(invalid_config(
                        "weights.non_executed_tensors",
                        "compile-time layer scalar must have shape [1]",
                    ));
                }
                _ => {}
            }
        }
        let expected_total = execution
            .len()
            .checked_add(self.non_executed_tensors.len())
            .ok_or_else(|| invalid_config("weights", "source tensor count overflows"))?;
        if usize::try_from(self.tensor_count).ok() != Some(expected_total)
            || usize::try_from(self.execution_source_tensor_count).ok() != Some(execution.len())
            || self.execution_quantized_weight_count
                != quantized_weight_count(semantic)
                    .map_err(|reason| invalid_config("weights", reason))?
            || self.structure_fingerprint.len() != 64
            || !self
                .structure_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(invalid_config(
                "weights",
                "Gemma 4 source counts or structure fingerprint differ from the typed contract",
            ));
        }
        Ok(())
    }

    pub(super) fn weight_schema(
        &self,
        semantic: &Gemma4SemanticConfig,
    ) -> Result<WeightSchema, VNextError> {
        self.validate(semantic)?;
        let mut components = Vec::new();
        let mut tensors = Vec::new();

        append_dense(
            global_weight_id(EMBED_TOKENS_ROLE)?,
            global_component_id(EMBED_TOKENS_ROLE)?,
            "model.language_model.embed_tokens.weight",
            vec![semantic.vocabulary_size, semantic.hidden_size],
            &mut components,
            &mut tensors,
        );
        append_dense(
            global_weight_id(FINAL_NORM_ROLE)?,
            global_component_id(FINAL_NORM_ROLE)?,
            "model.language_model.norm.weight",
            vec![semantic.hidden_size],
            &mut components,
            &mut tensors,
        );

        for (index, layer_type) in semantic.layer_types.iter().copied().enumerate() {
            let layer_index = u32::try_from(index)
                .map_err(|_| invalid_config("semantic.layer_count", "layer index exceeds u32"))?;
            let prefix = format!("model.language_model.layers.{layer_index}");
            let head_dim = semantic.head_dim(layer_type);
            for (role, external_name, dimensions) in [
                (
                    INPUT_NORM_ROLE,
                    format!("{prefix}.input_layernorm.weight"),
                    vec![semantic.hidden_size],
                ),
                (
                    POST_ATTENTION_NORM_ROLE,
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![semantic.hidden_size],
                ),
                (
                    PRE_FEEDFORWARD_NORM_ROLE,
                    format!("{prefix}.pre_feedforward_layernorm.weight"),
                    vec![semantic.hidden_size],
                ),
                (
                    POST_FEEDFORWARD_NORM_ROLE,
                    format!("{prefix}.post_feedforward_layernorm.weight"),
                    vec![semantic.hidden_size],
                ),
                (
                    Q_NORM_ROLE,
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![head_dim],
                ),
                (
                    K_NORM_ROLE,
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![head_dim],
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

            for (role, stem, dimensions) in projection_specs(semantic, layer_index, layer_type)? {
                append_symmetric_w4(
                    layer_weight_id(layer_index, role)?,
                    layer_component_id(layer_index, role)?,
                    stem,
                    dimensions,
                    &mut components,
                    &mut tensors,
                )?;
            }
        }

        Ok(WeightSchema {
            format_id: WeightFormatId::new(
                "weight-format.safetensors.compressed-tensors-marlin-int4-symmetric",
            )?,
            layout_id: WeightLayoutId::new(
                "weight-layout.gemma4_unified.text.compressed_tensors_marlin_symmetric",
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

fn append_symmetric_w4(
    weight_id: WeightId,
    component_id: WeightId,
    stem: String,
    dimensions: Vec<u64>,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let [n, k] = dimensions.as_slice() else {
        return Err(invalid_config(
            "weights",
            "W4 logical weight must have shape [N, K]",
        ));
    };
    if !n.is_multiple_of(64) || !k.is_multiple_of(32) {
        return Err(invalid_config(
            "weights",
            format!("W4 tensor {weight_id} shape {dimensions:?} is not Marlin group32 aligned"),
        ));
    }
    let packed_id = WeightId::new(format!("{component_id}.packed"))?;
    let scales_id = WeightId::new(format!("{component_id}.scales"))?;
    let packed_dimensions = vec![*n, *k / 2];
    let scale_dimensions = vec![*n, *k / 32];
    let quantization = QuantizationSpec {
        format_id: QuantizationFormatId::new(COMPRESSED_TENSORS_MARLIN_INT4_SYMMETRIC_FORMAT_ID)?,
        bits_per_weight: 4,
        grouping: QuantizationGrouping::fixed(32),
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: None,
    };
    components.push(WeightComponentSpec {
        id: packed_id.clone(),
        role: WeightComponentRole::PackedValues,
        external_names: vec![
            format!("{stem}.weight_packed"),
            format!("{stem}.weight_shape"),
        ],
        dimensions: packed_dimensions.clone(),
        encoding: WeightEncoding::Quantized(quantization),
        required: true,
    });
    components.push(WeightComponentSpec {
        id: scales_id.clone(),
        role: WeightComponentRole::Scales,
        external_names: vec![format!("{stem}.weight_scale")],
        dimensions: scale_dimensions,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F16,
        },
        required: true,
    });
    tensors.push(WeightTensorSpec {
        id: weight_id,
        dimensions,
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
            group_axis: 1,
            group_padding: PhysicalWeightPadding::Exact,
        },
        required: true,
    });
    Ok(())
}

fn expected_execution_tensors(
    semantic: &Gemma4SemanticConfig,
) -> Result<BTreeMap<String, ExpectedTensor>, String> {
    let mut expected = BTreeMap::new();
    insert_expected(
        &mut expected,
        "model.language_model.embed_tokens.weight",
        Dtype::BF16,
        vec![semantic.vocabulary_size, semantic.hidden_size],
    )?;
    insert_expected(
        &mut expected,
        "model.language_model.norm.weight",
        Dtype::BF16,
        vec![semantic.hidden_size],
    )?;
    for (index, layer_type) in semantic.layer_types.iter().copied().enumerate() {
        let layer_index = u32::try_from(index).map_err(|_| "layer index exceeds u32".to_owned())?;
        let prefix = format!("model.language_model.layers.{layer_index}");
        let head_dim = semantic.head_dim(layer_type);
        for (suffix, dimensions) in [
            ("input_layernorm.weight", vec![semantic.hidden_size]),
            (
                "post_attention_layernorm.weight",
                vec![semantic.hidden_size],
            ),
            (
                "pre_feedforward_layernorm.weight",
                vec![semantic.hidden_size],
            ),
            (
                "post_feedforward_layernorm.weight",
                vec![semantic.hidden_size],
            ),
            ("self_attn.q_norm.weight", vec![head_dim]),
            ("self_attn.k_norm.weight", vec![head_dim]),
        ] {
            insert_expected(
                &mut expected,
                format!("{prefix}.{suffix}"),
                Dtype::BF16,
                dimensions,
            )?;
        }
        for (_, stem, dimensions) in projection_specs(semantic, layer_index, layer_type)
            .map_err(|error| error.to_string())?
        {
            let [n, k] = dimensions.as_slice() else {
                return Err("projection shape is not rank two".to_owned());
            };
            insert_expected(
                &mut expected,
                format!("{stem}.weight_packed"),
                Dtype::I32,
                vec![*n, *k / 8],
            )?;
            insert_expected(
                &mut expected,
                format!("{stem}.weight_scale"),
                Dtype::BF16,
                vec![*n, *k / 32],
            )?;
            insert_expected(
                &mut expected,
                format!("{stem}.weight_shape"),
                Dtype::I64,
                vec![2],
            )?;
        }
    }
    Ok(expected)
}

fn projection_specs(
    semantic: &Gemma4SemanticConfig,
    layer_index: u32,
    layer_type: Gemma4LayerType,
) -> Result<Vec<(&'static str, String, Vec<u64>)>, VNextError> {
    let prefix = format!("model.language_model.layers.{layer_index}");
    let query = semantic
        .query_features(layer_type)
        .map_err(|reason| invalid_config("semantic", reason))?;
    let kv = semantic
        .kv_features(layer_type)
        .map_err(|reason| invalid_config("semantic", reason))?;
    let mut projections = vec![
        (
            Q_PROJ_ROLE,
            format!("{prefix}.self_attn.q_proj"),
            vec![query, semantic.hidden_size],
        ),
        (
            K_PROJ_ROLE,
            format!("{prefix}.self_attn.k_proj"),
            vec![kv, semantic.hidden_size],
        ),
    ];
    if layer_type == Gemma4LayerType::SlidingAttention {
        projections.push((
            V_PROJ_ROLE,
            format!("{prefix}.self_attn.v_proj"),
            vec![kv, semantic.hidden_size],
        ));
    }
    projections.extend([
        (
            O_PROJ_ROLE,
            format!("{prefix}.self_attn.o_proj"),
            vec![semantic.hidden_size, query],
        ),
        (
            GATE_PROJ_ROLE,
            format!("{prefix}.mlp.gate_proj"),
            vec![semantic.intermediate_size, semantic.hidden_size],
        ),
        (
            UP_PROJ_ROLE,
            format!("{prefix}.mlp.up_proj"),
            vec![semantic.intermediate_size, semantic.hidden_size],
        ),
        (
            DOWN_PROJ_ROLE,
            format!("{prefix}.mlp.down_proj"),
            vec![semantic.hidden_size, semantic.intermediate_size],
        ),
    ]);
    Ok(projections)
}

fn insert_expected(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    name: impl Into<String>,
    dtype: Dtype,
    dimensions: Vec<u64>,
) -> Result<(), String> {
    let name = name.into();
    if expected
        .insert(name.clone(), ExpectedTensor { dtype, dimensions })
        .is_some()
    {
        return Err(format!("duplicate expected Gemma 4 tensor {name:?}"));
    }
    Ok(())
}

fn owned_expected_non_executed_names(semantic: &Gemma4SemanticConfig) -> BTreeSet<String> {
    let mut names = MULTIMODAL_TENSOR_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    names.insert("lm_head.weight".to_owned());
    for index in 0..semantic.layer_count {
        names.insert(format!("model.language_model.layers.{index}.layer_scalar"));
    }
    names
}

fn load_non_executed_tensors(
    archive: &SafetensorsArchive,
    semantic: &Gemma4SemanticConfig,
) -> Result<Vec<Gemma4NonExecutedTensor>, String> {
    let names = owned_expected_non_executed_names(semantic);
    names
        .into_iter()
        .map(|name| {
            let tensor = archive.tensor(&name).map_err(|error| error.to_string())?;
            if tensor.dtype() != Dtype::BF16
                || tensor.shape().is_empty()
                || tensor.shape().contains(&0)
            {
                return Err(format!(
                    "typed non-executed tensor {name:?} must be non-empty BF16, got {:?}{:?}",
                    tensor.dtype(),
                    tensor.shape()
                ));
            }
            Ok(Gemma4NonExecutedTensor {
                external_name: name.clone(),
                dimensions: tensor.shape().to_vec(),
                dtype: Gemma4SourceDtype::Bf16,
                reason: non_executed_reason(&name, semantic)?,
            })
        })
        .collect()
}

fn non_executed_reason(
    name: &str,
    semantic: &Gemma4SemanticConfig,
) -> Result<Gemma4NonExecutedReason, String> {
    if name == "lm_head.weight" {
        return Ok(Gemma4NonExecutedReason::TiedProjectionDuplicate);
    }
    if let Some(index) = name
        .strip_prefix("model.language_model.layers.")
        .and_then(|value| value.strip_suffix(".layer_scalar"))
        .and_then(|value| value.parse::<u64>().ok())
    {
        if index < semantic.layer_count {
            return Ok(Gemma4NonExecutedReason::CompileTimeUnitLayerScale);
        }
    }
    if MULTIMODAL_TENSOR_NAMES.contains(&name) {
        return Ok(Gemma4NonExecutedReason::TextOnlyMultimodalComponent);
    }
    Err(format!("unknown non-executed Gemma 4 tensor {name:?}"))
}

fn validate_shape_sidecar_payloads(
    archive: &SafetensorsArchive,
    semantic: &Gemma4SemanticConfig,
) -> Result<(), String> {
    for (index, layer_type) in semantic.layer_types.iter().copied().enumerate() {
        let layer_index = u32::try_from(index).map_err(|_| "layer index exceeds u32".to_owned())?;
        for (_, stem, dimensions) in projection_specs(semantic, layer_index, layer_type)
            .map_err(|error| error.to_string())?
        {
            let tensor = archive
                .tensor(&format!("{stem}.weight_shape"))
                .map_err(|error| error.to_string())?;
            if tensor.bytes().len() != 16 {
                return Err(format!("{stem}.weight_shape must occupy 16 bytes"));
            }
            let n = i64::from_le_bytes(tensor.bytes()[0..8].try_into().expect("exact slice"));
            let k = i64::from_le_bytes(tensor.bytes()[8..16].try_into().expect("exact slice"));
            if n <= 0 || k <= 0 || dimensions != [n as u64, k as u64] {
                return Err(format!(
                    "{stem}.weight_shape payload [{n}, {k}] differs from logical shape {dimensions:?}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_unit_layer_scalars(
    archive: &SafetensorsArchive,
    semantic: &Gemma4SemanticConfig,
) -> Result<(), String> {
    for index in 0..semantic.layer_count {
        let name = format!("model.language_model.layers.{index}.layer_scalar");
        let tensor = archive.tensor(&name).map_err(|error| error.to_string())?;
        if tensor.dtype() != Dtype::BF16 || tensor.shape() != [1] || tensor.bytes() != [0x80, 0x3f]
        {
            return Err(format!(
                "{name} must be the typed BF16 scalar 1.0 before it may be folded into the program"
            ));
        }
    }
    Ok(())
}

fn structure_fingerprint(archive: &SafetensorsArchive) -> Result<String, String> {
    let mut digest = Sha256::new();
    digest.update([STRUCTURE_FINGERPRINT_VERSION]);
    for name in archive.tensor_names() {
        let tensor = archive.tensor(name).map_err(|error| error.to_string())?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        let dtype = format!("{:?}", tensor.dtype());
        digest.update((dtype.len() as u64).to_le_bytes());
        digest.update(dtype.as_bytes());
        digest.update((tensor.shape().len() as u64).to_le_bytes());
        for extent in tensor.shape() {
            digest.update(extent.to_le_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn quantized_weight_count(semantic: &Gemma4SemanticConfig) -> Result<u64, String> {
    semantic
        .layer_types
        .iter()
        .try_fold(0_u64, |total, layer_type| {
            total.checked_add(match layer_type {
                Gemma4LayerType::SlidingAttention => 7,
                Gemma4LayerType::FullAttention => 6,
            })
        })
        .ok_or_else(|| "quantized weight count overflows u64".to_owned())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ferrum_interfaces::vnext::ModelFamilyId;
    use safetensors::tensor::{serialize_to_file, TensorView};

    use super::super::config::tiny_semantic_config;
    use super::*;

    fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
        Box::leak(bytes.into_boxed_slice())
    }

    fn view(dtype: Dtype, dimensions: &[u64], bytes: Vec<u8>) -> TensorView<'static> {
        TensorView::new(
            dtype,
            dimensions
                .iter()
                .map(|extent| usize::try_from(*extent).unwrap())
                .collect(),
            leak_bytes(bytes),
        )
        .unwrap()
    }

    fn zero_view(dtype: Dtype, dimensions: &[u64]) -> TensorView<'static> {
        let element_bytes = match dtype {
            Dtype::BF16 => 2,
            Dtype::I32 => 4,
            Dtype::I64 => 8,
            other => panic!("unsupported fixture dtype {other:?}"),
        };
        let elements = dimensions.iter().product::<u64>();
        let bytes = usize::try_from(elements).unwrap() * element_bytes;
        view(dtype, dimensions, vec![0; bytes])
    }

    fn fixture_views(semantic: &Gemma4SemanticConfig) -> BTreeMap<String, TensorView<'static>> {
        let expected = expected_execution_tensors(semantic).unwrap();
        let mut views = expected
            .into_iter()
            .map(|(name, tensor)| {
                let view = zero_view(tensor.dtype, &tensor.dimensions);
                (name, view)
            })
            .collect::<BTreeMap<_, _>>();

        for (index, layer_type) in semantic.layer_types.iter().copied().enumerate() {
            let layer_index = u32::try_from(index).unwrap();
            for (_, stem, dimensions) in
                projection_specs(semantic, layer_index, layer_type).unwrap()
            {
                let payload = dimensions
                    .iter()
                    .flat_map(|extent| i64::try_from(*extent).unwrap().to_le_bytes())
                    .collect::<Vec<_>>();
                views.insert(
                    format!("{stem}.weight_shape"),
                    view(Dtype::I64, &[2], payload),
                );
            }
        }

        for name in owned_expected_non_executed_names(semantic) {
            let dimensions = if name == "lm_head.weight" {
                vec![semantic.vocabulary_size, semantic.hidden_size]
            } else {
                vec![1]
            };
            let bytes = if name.ends_with(".layer_scalar") {
                vec![0x80, 0x3f]
            } else {
                vec![0; usize::try_from(dimensions.iter().product::<u64>()).unwrap() * 2]
            };
            views.insert(name, view(Dtype::BF16, &dimensions, bytes));
        }
        views
    }

    fn write_archive(
        views: BTreeMap<String, TensorView<'static>>,
    ) -> (tempfile::TempDir, SafetensorsArchive) {
        let directory = tempfile::tempdir().unwrap();
        serialize_to_file(
            views,
            &None::<HashMap<String, String>>,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let archive = SafetensorsArchive::open(directory.path()).unwrap();
        (directory, archive)
    }

    #[test]
    fn tiny_hybrid_checkpoint_has_complete_typed_partition_and_schema() {
        let semantic = tiny_semantic_config();
        let (_directory, archive) = write_archive(fixture_views(&semantic));
        let manifest = Gemma4WeightManifest::load(&archive, &semantic).unwrap();
        assert_eq!(manifest.tensor_count, 67);
        assert_eq!(manifest.execution_source_tensor_count, 53);
        assert_eq!(manifest.execution_quantized_weight_count, 13);
        assert_eq!(manifest.non_executed_tensors.len(), 14);

        let schema = manifest.weight_schema(&semantic).unwrap();
        assert_eq!(schema.components.len(), 40);
        assert_eq!(schema.tensors.len(), 27);
        schema
            .validate(&ModelFamilyId::new(super::super::FAMILY_ID).unwrap())
            .unwrap();
        assert!(schema.tensors.iter().any(|tensor| {
            tensor.id.as_str() == "weight.layer.1.self_attn_k" && tensor.dimensions == [64, 64]
        }));
        assert!(!schema
            .tensors
            .iter()
            .any(|tensor| tensor.id.as_str() == "weight.layer.1.self_attn_v"));
    }

    #[test]
    fn dtype_shape_and_sidecar_drift_fail_before_allocation() {
        let semantic = tiny_semantic_config();
        let mut views = fixture_views(&semantic);
        let scale_name = "model.language_model.layers.0.self_attn.q_proj.weight_scale";
        views.insert(scale_name.to_owned(), zero_view(Dtype::BF16, &[64, 1]));
        let (_directory, archive) = write_archive(views);
        let error = Gemma4WeightManifest::load(&archive, &semantic).unwrap_err();
        assert!(error.contains(scale_name), "{error}");
        assert!(error.contains("must be BF16[64, 2]"), "{error}");

        let mut views = fixture_views(&semantic);
        let shape_name = "model.language_model.layers.1.self_attn.k_proj.weight_shape";
        let wrong_payload = [64_i64.to_le_bytes(), 32_i64.to_le_bytes()].concat();
        views.insert(shape_name.to_owned(), view(Dtype::I64, &[2], wrong_payload));
        let (_directory, archive) = write_archive(views);
        let error = Gemma4WeightManifest::load(&archive, &semantic).unwrap_err();
        assert!(error.contains("weight_shape payload"), "{error}");
        assert!(error.contains("logical shape"), "{error}");
    }
}
