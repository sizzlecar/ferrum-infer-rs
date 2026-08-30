//! Typed Qwen3.5 hybrid dense/MoE model package for the production vNext path.
//!
//! Preparation reads configuration, tokenizer metadata, and typed weight
//! headers only. Tensor payloads remain untouched until the selected backend
//! executor allocates them.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    AttributeId, BlockQuantizationSpec, CanonicalRational, CompositeWeightPart, ContractVersion,
    ElementType, ExternalModelMetadataId, GatedDeltaDecayParameterization,
    GatedDeltaValueHeadMapping, ModelFamilyId, ModelFamilyProvider, ModelFamilyRegistration,
    ModelProgram, ModelSemanticMetadata, NodeId, OperationId, PhysicalWeightComponentBinding,
    PhysicalWeightLayout, PhysicalWeightPadding, PreparedModelFamily, ProgramBlock, ProgramNode,
    ProgramNodeWorkSpec, ProgramTensorSpec, ProgramValueId, QuantizationFormatId,
    QuantizationGrouping, QuantizationPacking, QuantizationSpec,
    QuantizedProviderAttributionDenominator, ResolvedTensorLayout, SemanticValue,
    StateCapacityDemand, StateId, StateInitialization, StateLifetime, StateSpec,
    TypedFamilyRegistration, VNextError, WeightComponentRole, WeightComponentSource,
    WeightComponentSpec, WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightReference,
    WeightSchema, WeightTensorSpec, CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
    CAUSAL_PAGED_ATTENTION_OPERATION_ID, DENSE_SWIGLU_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
    LAST_TOKEN_DENSE_LINEAR_OPERATION_ID, LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
    LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, RESIDUAL_ADD_F32_F16_OPERATION_ID,
    RESIDUAL_ADD_OPERATION_ID, RMS_NORM_F32_OPERATION_ID, RMS_NORM_F32_TO_F16_OPERATION_ID,
    RMS_NORM_OPERATION_ID, ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
    TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
};
use ferrum_quantization::gguf::{block_quantization_format, ferrum_to_gguf_with_arch, GgmlDType};
use ferrum_quantization::{
    BlockFp8SafetensorsSource, CompressedTensorsMarlinSafetensorsSource, GgufWeightComponentSource,
    GptqMarlinSafetensorsSource, SafetensorsArchive, BLOCK_FP8_E4M3_SOURCE_FORMAT_ID,
    COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID, GPTQ_MARLIN_INT4_FORMAT_ID,
};
use ferrum_types::DataType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::qwen35_config::{
    Qwen35CompressedTensorsQuantizationRecipe, Qwen35Fp8QuantizationRecipe,
    Qwen35GptqQuantizationRecipe, Qwen35LayerType, Qwen35QuantizationConfig, Qwen35TextConfig,
    Qwen35WeightSpec,
};
use crate::qwen35_weights::{
    Qwen35ResolvedWeightSource, Qwen35ResolvedWeightSpec, Qwen35WeightInventory,
};

use super::{
    hf_metadata::parse_hf_model_semantic_metadata,
    weight_layout::{contiguous_or_reshaped_binding, dense_or_reshaped_layout},
    CausalLanguageModelDescriptor, PreparedProductionModel, ProductionModelSourceBundle,
    ProductionWeightArtifact,
};

pub const FAMILY_ID: &str = "family.qwen3_5.hybrid";
pub const EXTERNAL_METADATA_ID: &str = "hf.architecture.Qwen3_5ForConditionalGeneration";
pub const MOE_EXTERNAL_METADATA_ID: &str = "hf.architecture.Qwen3_5MoeForConditionalGeneration";
const DENSE_MATERIALIZED_ELEMENT_TYPE: ElementType = ElementType::F16;
const PACKED_GATE_UP_ROLE: &str = "mlp_gate_up";
const PACKED_LINEAR_ATTN_QKVZBA_ROLE: &str = "linear_attn_qkvzba";
const MOE_ROUTER_ROLE: &str = "moe_router";
const MOE_ROUTED_GATE_UP_ROLE: &str = "moe_routed_gate_up";
const MOE_ROUTED_DOWN_ROLE: &str = "moe_routed_down";
const MOE_SHARED_GATE_ROLE: &str = "moe_shared_gate";
const MOE_SHARED_GATE_UP_ROLE: &str = "moe_shared_gate_up";
const MOE_SHARED_DOWN_ROLE: &str = "moe_shared_down";
// These are typed program roles backed by `Linear` projections in the dense
// Qwen3.5 family. Whether an individual source stays dense is decided only by
// the checkpoint's `modules_to_not_convert`, never by repository identity.
const BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES: &[&str] = &[
    "lm_head",
    "linear_attn_qkv",
    "linear_attn_z",
    "linear_attn_b",
    "linear_attn_a",
    "linear_attn_out",
    "self_attn_q",
    "self_attn_k",
    "self_attn_v",
    "self_attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "moe_per_expert_gate_proj",
    "moe_per_expert_up_proj",
    "moe_per_expert_down_proj",
    "moe_shared_expert_gate_proj",
    "moe_shared_expert_up_proj",
    "moe_shared_expert_down_proj",
];

pub(super) fn validate_semantic_config(
    expected_metadata_id: &ExternalModelMetadataId,
    raw: &[u8],
) -> ferrum_types::Result<()> {
    let text = preflight_semantic_config(raw).map_err(ferrum_types::FerrumError::model)?;
    let expected_moe = match expected_metadata_id.as_str() {
        EXTERNAL_METADATA_ID => false,
        MOE_EXTERNAL_METADATA_ID => true,
        other => {
            return Err(ferrum_types::FerrumError::internal(format!(
                "Qwen3.5 semantic validator received unowned metadata identity {other}"
            )))
        }
    };
    if text.is_moe() != expected_moe {
        return Err(ferrum_types::FerrumError::model(format!(
            "Qwen3.5 semantic text layout {} does not match registered metadata identity {expected_metadata_id}",
            text.text_model_type
        )));
    }
    Ok(())
}

fn preflight_semantic_config(raw: &[u8]) -> Result<Qwen35TextConfig, String> {
    let hf_config: Value = serde_json::from_slice(raw)
        .map_err(|error| format!("parse semantic config.json: {error}"))?;
    let text = Qwen35TextConfig::from_hf_config_value(&hf_config)?;
    if let Some(quantization) = &text.quantization {
        validate_safetensors_quantization_config(quantization)
            .map_err(|error| error.to_string())?;
    }
    let text_value = hf_config.get("text_config").unwrap_or(&hf_config);
    required_u64(text_value, "vocab_size")?;
    required_u64(text_value, "max_position_embeddings")?;
    hf_rms_norm_epsilon(&hf_config)?;
    Ok(text)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyWeight {
    layer_index: Option<u32>,
    expert_index: Option<u32>,
    role: String,
    external_name: String,
    dimensions: Vec<u64>,
    source_encoding: FamilyWeightSourceEncoding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyGptqTensor {
    external_name: String,
    dimensions: Vec<u64>,
    element_type: ElementType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyCompressedTensorsTensor {
    external_name: String,
    dimensions: Vec<u64>,
    dtype: FamilyCompressedTensorsDtype,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyBlockFp8Tensor {
    external_name: String,
    dimensions: Vec<u64>,
    dtype: FamilyBlockFp8Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FamilyBlockFp8Dtype {
    F8E4m3,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FamilyCompressedTensorsDtype {
    I32,
    I64,
    F16,
    Bf16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum FamilyWeightSourceEncoding {
    Dense {
        element_type: ElementType,
    },
    Gptq {
        qweight: FamilyGptqTensor,
        scales: FamilyGptqTensor,
        qzeros: FamilyGptqTensor,
        g_idx: Option<FamilyGptqTensor>,
    },
    CompressedTensors {
        weight_packed: FamilyCompressedTensorsTensor,
        weight_scale: FamilyCompressedTensorsTensor,
        weight_zero_point: FamilyCompressedTensorsTensor,
        weight_shape: FamilyCompressedTensorsTensor,
    },
    BlockFp8 {
        values: FamilyBlockFp8Tensor,
        scale_inv: FamilyBlockFp8Tensor,
    },
    BlockQuantized(BlockQuantizationSpec),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FamilyWeightFormat {
    SafetensorsDense,
    SafetensorsGptqMarlin,
    SafetensorsCompressedTensorsMarlin,
    SafetensorsBlockFp8,
    GgufNative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OperationSelection {
    id: &'static str,
    version: ContractVersion,
}

impl OperationSelection {
    const fn new(id: &'static str, major: u16, minor: u16) -> Self {
        Self {
            id,
            version: ContractVersion::new(major, minor),
        }
    }
}

/// The activation ABI is selected once from the typed physical package.
/// It must never be inferred later from a backend name or hidden runtime flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35OperationProfile {
    token_embedding: OperationSelection,
    linear_attention: OperationSelection,
    causal_attention: OperationSelection,
    post_attention_norm: OperationSelection,
    dense_feed_forward: OperationSelection,
    dense_residual: OperationSelection,
    moe_residual: OperationSelection,
    final_norm: OperationSelection,
    logits: OperationSelection,
    argmax: OperationSelection,
}

impl Qwen35OperationProfile {
    const F16: Self = Self {
        token_embedding: OperationSelection::new(TOKEN_EMBEDDING_OPERATION_ID, 1, 0),
        linear_attention: OperationSelection::new(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            6,
            0,
        ),
        causal_attention: OperationSelection::new(CAUSAL_PAGED_ATTENTION_OPERATION_ID, 2, 0),
        post_attention_norm: OperationSelection::new(RMS_NORM_OPERATION_ID, 1, 0),
        dense_feed_forward: OperationSelection::new(DENSE_SWIGLU_OPERATION_ID, 1, 0),
        dense_residual: OperationSelection::new(RESIDUAL_ADD_OPERATION_ID, 1, 0),
        moe_residual: OperationSelection::new(RESIDUAL_ADD_OPERATION_ID, 1, 0),
        final_norm: OperationSelection::new(RMS_NORM_OPERATION_ID, 1, 0),
        logits: OperationSelection::new(LAST_TOKEN_DENSE_LINEAR_OPERATION_ID, 1, 0),
        argmax: OperationSelection::new(LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, 3, 0),
    };

    const F32_MASTER: Self = Self {
        token_embedding: OperationSelection::new(TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID, 1, 0),
        linear_attention: OperationSelection::new(
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
            1,
            0,
        ),
        causal_attention: OperationSelection::new(
            CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
            1,
            0,
        ),
        post_attention_norm: OperationSelection::new(RMS_NORM_F32_TO_F16_OPERATION_ID, 1, 0),
        dense_feed_forward: OperationSelection::new(DENSE_SWIGLU_OPERATION_ID, 1, 0),
        dense_residual: OperationSelection::new(RESIDUAL_ADD_F32_F16_OPERATION_ID, 1, 0),
        moe_residual: OperationSelection::new(RESIDUAL_ADD_F32_F16_OPERATION_ID, 1, 0),
        final_norm: OperationSelection::new(RMS_NORM_F32_OPERATION_ID, 1, 0),
        logits: OperationSelection::new(LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID, 1, 0),
        argmax: OperationSelection::new(LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID, 1, 0),
    };

    const fn for_weight_format(weight_format: FamilyWeightFormat) -> Self {
        match weight_format {
            FamilyWeightFormat::SafetensorsDense
            | FamilyWeightFormat::SafetensorsGptqMarlin
            | FamilyWeightFormat::SafetensorsCompressedTensorsMarlin
            | FamilyWeightFormat::SafetensorsBlockFp8 => Self::F16,
            FamilyWeightFormat::GgufNative => Self::F32_MASTER,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35FamilyConfig {
    hf_config: Value,
    vocab_size: u64,
    max_position_embeddings: u64,
    rms_norm_epsilon: CanonicalRational,
    metadata: ModelSemanticMetadata,
    weight_format: FamilyWeightFormat,
    weights: Vec<FamilyWeight>,
}

pub struct Qwen35FamilyProvider {
    family_id: ModelFamilyId,
}

impl Qwen35FamilyProvider {
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            family_id: ModelFamilyId::new(FAMILY_ID)?,
        })
    }

    fn text_config(config: &Qwen35FamilyConfig) -> Result<Qwen35TextConfig, VNextError> {
        Qwen35TextConfig::from_hf_config_value(&config.hf_config)
            .map_err(|reason| invalid_config("hf_config", reason))
    }

    fn validate_typed_config(&self, config: &Qwen35FamilyConfig) -> Result<(), VNextError> {
        let text = Self::text_config(config)?;
        let expected_epsilon = hf_rms_norm_epsilon(&config.hf_config)
            .map_err(|reason| invalid_config("hf_config.text_config.rms_norm_eps", reason))?;
        if config.rms_norm_epsilon != expected_epsilon {
            return Err(invalid_config(
                "rms_norm_epsilon",
                "typed epsilon differs from Hugging Face metadata",
            ));
        }
        match (config.weight_format, text.quantization.as_ref()) {
            (FamilyWeightFormat::SafetensorsDense, None)
            | (FamilyWeightFormat::GgufNative, None) => {}
            (FamilyWeightFormat::SafetensorsGptqMarlin, Some(quantization)) => {
                validate_gptq_marlin_config(quantization)?;
            }
            (FamilyWeightFormat::SafetensorsCompressedTensorsMarlin, Some(quantization)) => {
                validate_compressed_tensors_marlin_config(quantization)?;
            }
            (FamilyWeightFormat::SafetensorsBlockFp8, Some(quantization)) => {
                validate_block_fp8_config(quantization)?;
            }
            (FamilyWeightFormat::SafetensorsDense, Some(_)) => {
                return Err(invalid_config(
                    "hf_config.quantization_config",
                    "raw safetensors quantization requires the typed GPTQ source adapter",
                ));
            }
            (FamilyWeightFormat::SafetensorsGptqMarlin, None) => {
                return Err(invalid_config(
                    "hf_config.quantization_config",
                    "the GPTQ Marlin source requires explicit Hugging Face quantization metadata",
                ));
            }
            (FamilyWeightFormat::SafetensorsCompressedTensorsMarlin, None) => {
                return Err(invalid_config(
                    "hf_config.quantization_config",
                    "the compressed-tensors Marlin source requires explicit Hugging Face quantization metadata",
                ));
            }
            (FamilyWeightFormat::SafetensorsBlockFp8, None) => {
                return Err(invalid_config(
                    "hf_config.quantization_config",
                    "the block-FP8 source requires explicit Hugging Face quantization metadata",
                ));
            }
            (FamilyWeightFormat::GgufNative, Some(_)) => {
                return Err(invalid_config(
                    "hf_config.quantization_config",
                    "GGUF physical quantization must not be duplicated in Hugging Face metadata",
                ));
            }
        }
        if text.mamba_ssm_dtype != DataType::FP32 {
            return Err(invalid_config(
                "hf_config.text_config.mamba_ssm_dtype",
                "the current Qwen3.5 vNext providers require float32 temporal state",
            ));
        }
        if config.vocab_size == 0 || config.max_position_embeddings == 0 {
            return Err(invalid_config(
                "hf_config.text_config",
                "vocab_size and max_position_embeddings must be positive",
            ));
        }
        if config.metadata.template.template.is_empty()
            || config.metadata.template.source_file != "tokenizer_config.json"
            || config.metadata.special_tokens.eos_token_ids.is_empty()
            || config.weights.is_empty()
        {
            return Err(invalid_config(
                "family_package",
                "template, EOS tokens, and resolved weights must be explicit",
            ));
        }

        let mut external_names = BTreeSet::new();
        let mut logical_keys = BTreeSet::new();
        for weight in &config.weights {
            if weight.role.is_empty()
                || weight.external_name.is_empty()
                || weight.dimensions.is_empty()
                || weight.dimensions.contains(&0)
                || !logical_keys.insert((
                    weight.layer_index,
                    weight.expert_index,
                    weight.role.clone(),
                ))
            {
                return Err(invalid_config(
                    "weights",
                    "resolved tensor names, roles, and non-zero shapes must be unique",
                ));
            }
            match &weight.source_encoding {
                FamilyWeightSourceEncoding::Dense { element_type }
                    if matches!(
                        element_type,
                        ElementType::F16 | ElementType::Bf16 | ElementType::F32
                    ) =>
                {
                    if !external_names.insert(weight.external_name.clone()) {
                        return Err(invalid_config(
                            "weights",
                            format!(
                                "checkpoint tensor {:?} is referenced more than once",
                                weight.external_name
                            ),
                        ));
                    }
                }
                FamilyWeightSourceEncoding::BlockQuantized(spec) => {
                    spec.validate()?;
                    let block_width = u64::from(spec.logical_values_per_block);
                    if weight
                        .dimensions
                        .last()
                        .is_none_or(|extent| !extent.is_multiple_of(block_width))
                    {
                        return Err(invalid_config(
                            "weights.source_encoding",
                            format!(
                                "role {:?} innermost dimension is not divisible by block width {block_width}",
                                weight.role
                            ),
                        ));
                    }
                    if !external_names.insert(weight.external_name.clone()) {
                        return Err(invalid_config(
                            "weights",
                            format!(
                                "checkpoint tensor {:?} is referenced more than once",
                                weight.external_name
                            ),
                        ));
                    }
                }
                FamilyWeightSourceEncoding::Dense { element_type } => {
                    return Err(invalid_config(
                        "weights.source_encoding",
                        format!(
                            "role {:?} has non-floating dense source type {element_type:?}",
                            weight.role
                        ),
                    ));
                }
                FamilyWeightSourceEncoding::Gptq {
                    qweight,
                    scales,
                    qzeros,
                    g_idx,
                } => {
                    validate_gptq_weight_source(
                        weight,
                        qweight,
                        scales,
                        qzeros,
                        g_idx.as_ref(),
                        text.quantization.as_ref().ok_or_else(|| {
                            invalid_config(
                                "hf_config.quantization_config",
                                "GPTQ weight has no typed quantization metadata",
                            )
                        })?,
                    )?;
                    for source in [qweight, scales, qzeros].into_iter().chain(g_idx.iter()) {
                        if !external_names.insert(source.external_name.clone()) {
                            return Err(invalid_config(
                                "weights",
                                format!(
                                    "GPTQ sidecar {:?} is referenced more than once",
                                    source.external_name
                                ),
                            ));
                        }
                    }
                }
                FamilyWeightSourceEncoding::CompressedTensors {
                    weight_packed,
                    weight_scale,
                    weight_zero_point,
                    weight_shape,
                } => {
                    validate_compressed_tensors_weight_source(
                        weight,
                        weight_packed,
                        weight_scale,
                        weight_zero_point,
                        weight_shape,
                        text.quantization.as_ref().ok_or_else(|| {
                            invalid_config(
                                "hf_config.quantization_config",
                                "compressed-tensors weight has no typed quantization metadata",
                            )
                        })?,
                    )?;
                    for source in [weight_packed, weight_scale, weight_zero_point, weight_shape] {
                        if !external_names.insert(source.external_name.clone()) {
                            return Err(invalid_config(
                                "weights",
                                format!(
                                    "compressed-tensors sidecar {:?} is referenced more than once",
                                    source.external_name
                                ),
                            ));
                        }
                    }
                }
                FamilyWeightSourceEncoding::BlockFp8 { values, scale_inv } => {
                    validate_block_fp8_weight_source(
                        weight,
                        values,
                        scale_inv,
                        text.quantization.as_ref().ok_or_else(|| {
                            invalid_config(
                                "hf_config.quantization_config",
                                "block-FP8 weight has no typed quantization metadata",
                            )
                        })?,
                    )?;
                    for source in [values, scale_inv] {
                        if !external_names.insert(source.external_name.clone()) {
                            return Err(invalid_config(
                                "weights",
                                format!(
                                    "block-FP8 source {:?} is referenced more than once",
                                    source.external_name
                                ),
                            ));
                        }
                    }
                }
            }
            let expected_dimensions = expected_weight_dimensions(&text, config.vocab_size, weight)?;
            if !expected_dimensions.contains(&weight.dimensions) {
                return Err(invalid_config(
                    "weights.dimensions",
                    format!(
                        "role {:?} has dimensions {:?}, expected one of {expected_dimensions:?}",
                        weight.role, weight.dimensions,
                    ),
                ));
            }
        }

        match config.weight_format {
            FamilyWeightFormat::SafetensorsDense => {
                if config.weights.iter().any(|weight| {
                    !matches!(
                        weight.source_encoding,
                        FamilyWeightSourceEncoding::Dense { .. }
                    )
                }) {
                    return Err(invalid_config(
                        "weights.source_encoding",
                        "safetensors dense packages cannot contain block-quantized components",
                    ));
                }
                validate_safetensors_manifest(&text, config, "dense")?;
            }
            FamilyWeightFormat::SafetensorsGptqMarlin => {
                if config.weights.iter().any(|weight| {
                    matches!(
                        weight.source_encoding,
                        FamilyWeightSourceEncoding::BlockQuantized(_)
                            | FamilyWeightSourceEncoding::CompressedTensors { .. }
                            | FamilyWeightSourceEncoding::BlockFp8 { .. }
                    )
                }) {
                    return Err(invalid_config(
                        "weights.source_encoding",
                        "safetensors GPTQ packages cannot contain GGUF block components",
                    ));
                }
                validate_safetensors_manifest(&text, config, "GPTQ")?;
                validate_canonical_gptq_moe_representation(&text, config)?;
            }
            FamilyWeightFormat::SafetensorsCompressedTensorsMarlin => {
                if config.weights.iter().any(|weight| {
                    matches!(
                        weight.source_encoding,
                        FamilyWeightSourceEncoding::BlockQuantized(_)
                            | FamilyWeightSourceEncoding::Gptq { .. }
                            | FamilyWeightSourceEncoding::BlockFp8 { .. }
                    )
                }) {
                    return Err(invalid_config(
                        "weights.source_encoding",
                        "compressed-tensors packages cannot contain GPTQ or GGUF components",
                    ));
                }
                validate_safetensors_manifest(&text, config, "compressed-tensors")?;
            }
            FamilyWeightFormat::SafetensorsBlockFp8 => {
                if config.weights.iter().any(|weight| {
                    matches!(
                        weight.source_encoding,
                        FamilyWeightSourceEncoding::BlockQuantized(_)
                            | FamilyWeightSourceEncoding::Gptq { .. }
                            | FamilyWeightSourceEncoding::CompressedTensors { .. }
                    )
                }) {
                    return Err(invalid_config(
                        "weights.source_encoding",
                        "block-FP8 packages cannot contain GPTQ, compressed-tensors, or GGUF components",
                    ));
                }
                validate_block_fp8_source_completeness(
                    config,
                    validate_block_fp8_config(text.quantization.as_ref().ok_or_else(|| {
                        invalid_config(
                            "hf_config.quantization_config",
                            "the block-FP8 source requires explicit Hugging Face quantization metadata",
                        )
                    })?)?,
                )?;
                validate_safetensors_manifest(&text, config, "block-FP8")?;
            }
            FamilyWeightFormat::GgufNative => validate_gguf_manifest(&text, config)?,
        }
        Ok(())
    }
}

impl ModelFamilyProvider for Qwen35FamilyProvider {
    type Config = Qwen35FamilyConfig;

    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        [EXTERNAL_METADATA_ID, MOE_EXTERNAL_METADATA_ID]
            .into_iter()
            .map(|metadata_id| {
                ExternalModelMetadataId::new(metadata_id)
                    .expect("Qwen3.5 external metadata ids are static and valid")
            })
            .collect()
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.validate_typed_config(config)?;
        let typed = serde_json::to_value(config).map_err(|error| VNextError::Serialization {
            context: "serialize Qwen3.5 family config",
            message: error.to_string(),
        })?;
        if raw != &typed {
            return Err(invalid_config(
                "config",
                "Qwen3.5 family input is not the exact typed configuration",
            ));
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        let text = Self::text_config(config)?;
        ExternalModelMetadataId::new(if text.is_moe() {
            MOE_EXTERNAL_METADATA_ID
        } else {
            EXTERNAL_METADATA_ID
        })
    }

    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config: Qwen35FamilyConfig = serde_json::from_value(raw.clone())
            .map_err(|error| invalid_config("config", error.to_string()))?;
        self.validate_typed_config(&config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError> {
        match config.weight_format {
            FamilyWeightFormat::SafetensorsDense => safetensors_weight_schema(config),
            FamilyWeightFormat::SafetensorsGptqMarlin
            | FamilyWeightFormat::SafetensorsCompressedTensorsMarlin
            | FamilyWeightFormat::SafetensorsBlockFp8 => {
                safetensors_quantized_weight_schema(config)
            }
            FamilyWeightFormat::GgufNative => gguf_weight_schema(config),
        }
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        let text = Self::text_config(config)?;
        let operations = Qwen35OperationProfile::for_weight_format(config.weight_format);
        let mut weight_refs = Vec::with_capacity(config.weights.len());
        for weight in &config.weights {
            if is_moe_source_role(&weight.role) {
                continue;
            }
            if matches!(
                weight.role.as_str(),
                "mlp_up" | "linear_attn_z" | "linear_attn_b" | "linear_attn_a"
            ) {
                continue;
            }
            if weight.role == "mlp_gate" {
                let layer_index = weight.layer_index.ok_or_else(|| {
                    invalid_config("weights.mlp_gate", "dense gate weight has no layer")
                })?;
                let up = required_weight(config, Some(layer_index), "mlp_up")?;
                weight_refs.push(WeightReference {
                    weight_id: packed_gate_up_weight_id(layer_index)?,
                    value_id: packed_gate_up_value_id(layer_index)?,
                    tensor: tensor_spec(
                        packed_gate_up_dimensions(weight, up)?,
                        materialized_element_type(PACKED_GATE_UP_ROLE),
                    ),
                });
            } else if weight.role == "linear_attn_qkv" {
                let layer_index = weight.layer_index.ok_or_else(|| {
                    invalid_config(
                        "weights.linear_attn",
                        "linear-attention projection has no layer",
                    )
                })?;
                let z = required_weight(config, Some(layer_index), "linear_attn_z")?;
                let b = required_weight(config, Some(layer_index), "linear_attn_b")?;
                let a = required_weight(config, Some(layer_index), "linear_attn_a")?;
                weight_refs.push(WeightReference {
                    weight_id: packed_linear_attention_weight_id(
                        layer_index,
                        PACKED_LINEAR_ATTN_QKVZBA_ROLE,
                    )?,
                    value_id: packed_linear_attention_value_id(
                        layer_index,
                        PACKED_LINEAR_ATTN_QKVZBA_ROLE,
                    )?,
                    tensor: tensor_spec(
                        packed_linear_attention_dimensions([weight, z, b, a])?,
                        DENSE_MATERIALIZED_ELEMENT_TYPE,
                    ),
                });
            } else {
                weight_refs.push(WeightReference {
                    weight_id: weight_id(weight)?,
                    value_id: weight_value_id(weight)?,
                    tensor: tensor_spec(
                        logical_weight_dimensions(weight)?,
                        materialized_element_type(&weight.role),
                    ),
                });
            }
        }
        if text.moe.is_some() {
            for layer_index in 0..text.num_hidden_layers {
                weight_refs.extend(moe_weight_references(&text, layer_index as u32)?);
            }
        }

        let mut nodes = Vec::with_capacity(text.num_hidden_layers * 4 + 3);
        let embedding = required_weight(config, None, "embed_tokens")?;
        let mut hidden = value_id("value.hidden.embedding")?;
        nodes.push(ProgramNode {
            id: node_id("node.embedding")?,
            operation_id: operation_id(operations.token_embedding.id)?,
            required_version: operations.token_embedding.version,
            work: ProgramNodeWorkSpec::tokens(value_id("value.input.token_ids")?, 0),
            inputs: vec![
                value_id("value.input.token_ids")?,
                weight_value_id(embedding)?,
            ],
            outputs: vec![hidden.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", text.hidden_size as u64)?,
                attribute("vocab_size", config.vocab_size)?,
            ]),
        });

        let mut states = Vec::new();
        for (layer_index, layer_type) in text.layer_types.iter().copied().enumerate() {
            let attention_output = value_id(format!("value.layer.{layer_index}.attention"))?;
            let input_norm = required_weight(config, Some(layer_index as u32), "input_layernorm")?;
            let mut attention_inputs = vec![hidden.clone(), weight_value_id(input_norm)?];

            let (operation, required_version, mut attributes) = match layer_type {
                Qwen35LayerType::LinearAttention => {
                    for role in [
                        "linear_attn_conv",
                        "linear_attn_a_log",
                        "linear_attn_dt_bias",
                        "linear_attn_norm",
                        "linear_attn_out",
                    ] {
                        attention_inputs.push(weight_value_id(required_weight(
                            config,
                            Some(layer_index as u32),
                            role,
                        )?)?);
                    }
                    attention_inputs.insert(
                        2,
                        packed_linear_attention_value_id(
                            layer_index as u32,
                            PACKED_LINEAR_ATTN_QKVZBA_ROLE,
                        )?,
                    );
                    let conv_value = value_id(format!("value.state.layer.{layer_index}.conv"))?;
                    let delta_value = value_id(format!("value.state.layer.{layer_index}.delta"))?;
                    attention_inputs.extend([conv_value.clone(), delta_value.clone()]);
                    states.push(StateSpec {
                        id: state_id(format!("state.layer.{layer_index}.conv"))?,
                        value_id: conv_value,
                        tensor: tensor_spec(
                            text.recurrent_conv_state_shape()
                                .map_err(|reason| invalid_config("states.conv", reason))?
                                .into_iter()
                                .map(|extent| extent as u64)
                                .collect(),
                            ElementType::F16,
                        ),
                        lifetime: StateLifetime::Sequence,
                        capacity_demand: StateCapacityDemand::FixedPerScope,
                        initialization: StateInitialization::Zero,
                    });
                    states.push(StateSpec {
                        id: state_id(format!("state.layer.{layer_index}.delta"))?,
                        value_id: delta_value,
                        tensor: tensor_spec(
                            text.recurrent_delta_state_shape()
                                .map_err(|reason| invalid_config("states.delta", reason))?
                                .into_iter()
                                .map(|extent| extent as u64)
                                .collect(),
                            data_type_to_element_type(text.mamba_ssm_dtype)
                                .map_err(|reason| invalid_config("states.delta.dtype", reason))?,
                        ),
                        lifetime: StateLifetime::Sequence,
                        capacity_demand: StateCapacityDemand::FixedPerScope,
                        initialization: StateInitialization::Zero,
                    });
                    let (decay_parameterization, value_head_mapping) = match config.weight_format {
                        FamilyWeightFormat::SafetensorsDense
                        | FamilyWeightFormat::SafetensorsGptqMarlin
                        | FamilyWeightFormat::SafetensorsCompressedTensorsMarlin
                        | FamilyWeightFormat::SafetensorsBlockFp8 => (
                            GatedDeltaDecayParameterization::LogRate,
                            GatedDeltaValueHeadMapping::GroupedByKeyHead,
                        ),
                        FamilyWeightFormat::GgufNative => (
                            GatedDeltaDecayParameterization::NegativeRate,
                            GatedDeltaValueHeadMapping::InterleavedByKeyHead,
                        ),
                    };
                    (
                        operations.linear_attention.id,
                        operations.linear_attention.version,
                        BTreeMap::from([
                            attribute("key_heads", text.linear_attention.num_key_heads as u64)?,
                            attribute("value_heads", text.linear_attention.num_value_heads as u64)?,
                            attribute("key_head_dim", text.linear_attention.key_head_dim as u64)?,
                            attribute(
                                "value_head_dim",
                                text.linear_attention.value_head_dim as u64,
                            )?,
                            attribute("hidden_size", text.hidden_size as u64)?,
                            attribute(
                                "qkv_features",
                                (text.linear_qk_total_dim() * 2 + text.linear_value_total_dim())
                                    as u64,
                            )?,
                            attribute("value_features", text.linear_value_total_dim() as u64)?,
                            attribute(
                                "qkvz_features",
                                (text.linear_qk_total_dim() * 2 + text.linear_value_total_dim() * 2)
                                    as u64,
                            )?,
                            attribute(
                                "ba_features",
                                (text.linear_attention.num_value_heads * 2) as u64,
                            )?,
                            attribute(
                                "qkvzba_features",
                                (text.linear_qk_total_dim() * 2
                                    + text.linear_value_total_dim() * 2
                                    + text.linear_attention.num_value_heads * 2)
                                    as u64,
                            )?,
                            attribute("conv_kernel", text.linear_attention.conv_kernel_dim as u64)?,
                            attribute(
                                "conv_state_width",
                                text.linear_attention.conv_kernel_dim.saturating_sub(1) as u64,
                            )?,
                            attribute("epsilon", config.rms_norm_epsilon)?,
                            attribute("decay_parameterization", decay_parameterization)?,
                            attribute("value_head_mapping", value_head_mapping)?,
                        ]),
                    )
                }
                Qwen35LayerType::FullAttention => {
                    for role in [
                        "self_attn_q",
                        "self_attn_k",
                        "self_attn_v",
                        "self_attn_o",
                        "self_attn_q_norm",
                        "self_attn_k_norm",
                    ] {
                        attention_inputs.push(weight_value_id(required_weight(
                            config,
                            Some(layer_index as u32),
                            role,
                        )?)?);
                    }
                    let kv_value = value_id(format!("value.state.layer.{layer_index}.kv"))?;
                    let kv_dimensions =
                        vec![2, text.num_key_value_heads as u64, text.head_dim as u64];
                    let kv_bytes_per_token = kv_dimensions.iter().product::<u64>() * 2;
                    attention_inputs.push(kv_value.clone());
                    states.push(StateSpec {
                        id: state_id(format!("state.layer.{layer_index}.kv"))?,
                        value_id: kv_value,
                        tensor: tensor_spec(kv_dimensions, ElementType::F16),
                        lifetime: StateLifetime::Sequence,
                        capacity_demand: StateCapacityDemand::TokenScaled {
                            bytes_per_token: kv_bytes_per_token,
                            maximum_tokens: config.max_position_embeddings,
                        },
                        // The attention provider writes each valid KV slot before
                        // that slot can be read; clearing unused block capacity is
                        // unnecessary work on the decode path.
                        initialization: StateInitialization::None,
                    });
                    (
                        operations.causal_attention.id,
                        operations.causal_attention.version,
                        BTreeMap::from([
                            attribute("query_heads", text.num_attention_heads as u64)?,
                            attribute("key_value_heads", text.num_key_value_heads as u64)?,
                            attribute("head_dim", text.head_dim as u64)?,
                            attribute("hidden_size", text.hidden_size as u64)?,
                            attribute(
                                "query_features",
                                text.full_attention_query_total_dim() as u64,
                            )?,
                            attribute(
                                "query_projection_features",
                                text.full_attention_q_proj_total_dim() as u64,
                            )?,
                            attribute("kv_features", text.full_attention_kv_total_dim() as u64)?,
                            attribute("rope_dim", text.full_attention_rope_dim() as u64)?,
                            attribute("maximum_context_tokens", config.max_position_embeddings)?,
                            attribute(
                                "rope_theta",
                                canonical_positive_f64(text.rope_parameters.rope_theta)?,
                            )?,
                            attribute(
                                "rope_interleaved",
                                text.full_attention_text_rope_interleaved(),
                            )?,
                            attribute("output_gate", text.attn_output_gate)?,
                            attribute("causal", true)?,
                            attribute("epsilon", config.rms_norm_epsilon)?,
                        ]),
                    )
                }
            };
            attributes.insert(
                AttributeId::new("layer_index")?,
                SemanticValue::Unsigned(layer_index as u64),
            );
            attributes.insert(
                AttributeId::new("hidden_size")?,
                SemanticValue::Unsigned(text.hidden_size as u64),
            );
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.attention"))?,
                operation_id: operation_id(operation)?,
                required_version,
                work: ProgramNodeWorkSpec::tokens(hidden.clone(), 0),
                inputs: attention_inputs,
                outputs: vec![attention_output.clone()],
                attributes,
            });

            let normalized = value_id(format!("value.layer.{layer_index}.post_attention_norm"))?;
            let post_attention_norm =
                required_weight(config, Some(layer_index as u32), "post_attention_layernorm")?;
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.post_attention_norm"))?,
                operation_id: operation_id(operations.post_attention_norm.id)?,
                required_version: operations.post_attention_norm.version,
                work: ProgramNodeWorkSpec::tokens(attention_output.clone(), 0),
                inputs: vec![
                    attention_output.clone(),
                    weight_value_id(post_attention_norm)?,
                ],
                outputs: vec![normalized.clone()],
                attributes: BTreeMap::from([
                    attribute("hidden_size", text.hidden_size as u64)?,
                    attribute("epsilon", config.rms_norm_epsilon)?,
                ]),
            });

            let mlp_output = value_id(format!("value.layer.{layer_index}.mlp"))?;
            let (
                feed_forward_operation,
                residual_operation,
                feed_forward_inputs,
                feed_forward_attributes,
            ) = if let Some(moe) = &text.moe {
                (
                    ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
                    operations.moe_residual,
                    vec![
                        normalized,
                        moe_weight_value_id(layer_index as u32, MOE_ROUTER_ROLE)?,
                        moe_weight_value_id(layer_index as u32, MOE_ROUTED_GATE_UP_ROLE)?,
                        moe_weight_value_id(layer_index as u32, MOE_ROUTED_DOWN_ROLE)?,
                        moe_weight_value_id(layer_index as u32, MOE_SHARED_GATE_ROLE)?,
                        moe_weight_value_id(layer_index as u32, MOE_SHARED_GATE_UP_ROLE)?,
                        moe_weight_value_id(layer_index as u32, MOE_SHARED_DOWN_ROLE)?,
                    ],
                    BTreeMap::from([
                        attribute("hidden_size", text.hidden_size as u64)?,
                        attribute("expert_count", moe.num_experts as u64)?,
                        attribute("experts_per_token", moe.num_experts_per_tok as u64)?,
                        attribute("routed_intermediate_size", moe.moe_intermediate_size as u64)?,
                        attribute(
                            "shared_intermediate_size",
                            moe.shared_expert_intermediate_size as u64,
                        )?,
                        attribute("normalize_topk", moe.norm_topk_prob)?,
                    ]),
                )
            } else {
                let intermediate_size = text.dense_intermediate_size.ok_or_else(|| {
                    invalid_config(
                        "hf_config.text_config.intermediate_size",
                        "missing dense FFN size",
                    )
                })?;
                (
                    operations.dense_feed_forward.id,
                    operations.dense_residual,
                    vec![
                        normalized,
                        packed_gate_up_value_id(layer_index as u32)?,
                        weight_value_id(required_weight(
                            config,
                            Some(layer_index as u32),
                            "mlp_down",
                        )?)?,
                    ],
                    BTreeMap::from([
                        attribute("hidden_size", text.hidden_size as u64)?,
                        attribute("intermediate_size", intermediate_size as u64)?,
                    ]),
                )
            };
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.feed_forward"))?,
                operation_id: operation_id(feed_forward_operation)?,
                required_version: if text.moe.is_some() {
                    ContractVersion::new(1, 0)
                } else {
                    operations.dense_feed_forward.version
                },
                work: ProgramNodeWorkSpec::tokens(feed_forward_inputs[0].clone(), 0),
                inputs: feed_forward_inputs,
                outputs: vec![mlp_output.clone()],
                attributes: feed_forward_attributes,
            });

            let layer_output = value_id(format!("value.layer.{layer_index}.output"))?;
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.residual"))?,
                operation_id: operation_id(residual_operation.id)?,
                required_version: residual_operation.version,
                work: ProgramNodeWorkSpec::tokens(attention_output.clone(), 0),
                inputs: vec![attention_output, mlp_output],
                outputs: vec![layer_output.clone()],
                attributes: BTreeMap::from([attribute("hidden_size", text.hidden_size as u64)?]),
            });
            hidden = layer_output;
        }

        let final_norm = required_weight(config, None, "final_norm")?;
        let projection = config
            .weights
            .iter()
            .find(|weight| weight.layer_index.is_none() && weight.role == "lm_head")
            .unwrap_or(embedding);
        let final_hidden = value_id("value.output.final_hidden")?;
        nodes.push(ProgramNode {
            id: node_id("node.final_norm")?,
            operation_id: operation_id(operations.final_norm.id)?,
            required_version: operations.final_norm.version,
            work: ProgramNodeWorkSpec::tokens(hidden.clone(), 0),
            inputs: vec![hidden, weight_value_id(final_norm)?],
            outputs: vec![final_hidden.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", text.hidden_size as u64)?,
                attribute("epsilon", config.rms_norm_epsilon)?,
            ]),
        });
        let logits = value_id("value.output.logits")?;
        nodes.push(ProgramNode {
            id: node_id("node.logits")?,
            operation_id: operation_id(operations.logits.id)?,
            required_version: operations.logits.version,
            work: ProgramNodeWorkSpec::tokens(final_hidden.clone(), 0),
            inputs: vec![final_hidden, weight_value_id(projection)?],
            outputs: vec![logits.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", text.hidden_size as u64)?,
                attribute("out_features", config.vocab_size)?,
            ]),
        });
        let greedy_mask = value_id("value.input.greedy_token_mask")?;
        let greedy_repetition_token_ids = value_id("value.input.greedy_repetition_token_ids")?;
        let greedy_repetition_offsets = value_id("value.input.greedy_repetition_offsets")?;
        let greedy_repetition_penalty = value_id("value.input.greedy_repetition_penalty")?;
        let greedy_token = value_id("value.output.greedy_token")?;
        nodes.push(ProgramNode {
            id: node_id("node.greedy_token")?,
            operation_id: operation_id(operations.argmax.id)?,
            required_version: operations.argmax.version,
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![
                logits.clone(),
                greedy_mask.clone(),
                greedy_repetition_token_ids.clone(),
                greedy_repetition_offsets.clone(),
                greedy_repetition_penalty.clone(),
            ],
            outputs: vec![greedy_token.clone()],
            attributes: BTreeMap::from([attribute("vocab_size", config.vocab_size)?]),
        });

        ModelProgram::new(
            self.family_id.clone(),
            vec![
                value_id("value.input.token_ids")?,
                greedy_mask,
                greedy_repetition_token_ids,
                greedy_repetition_offsets,
                greedy_repetition_penalty,
            ],
            vec![ProgramBlock {
                id: "block.decoder".to_owned(),
                nodes,
            }],
            states,
            weight_refs,
            vec![logits, greedy_token],
        )
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(config.metadata.clone())
    }
}

fn safetensors_weight_schema(config: &Qwen35FamilyConfig) -> Result<WeightSchema, VNextError> {
    if Qwen35TextConfig::from_hf_config_value(&config.hf_config)
        .map_err(|reason| invalid_config("hf_config", reason))?
        .is_moe()
    {
        return Err(invalid_config(
            "weight_format",
            "safetensors MoE requires the typed GPTQ source adapter",
        ));
    }
    let mut components = Vec::with_capacity(config.weights.len());
    let mut tensors = Vec::with_capacity(config.weights.len());
    for weight in &config.weights {
        if matches!(
            weight.role.as_str(),
            "mlp_up" | "linear_attn_z" | "linear_attn_b" | "linear_attn_a"
        ) {
            continue;
        }
        if weight.role == "linear_attn_qkv" {
            let layer_index = weight.layer_index.ok_or_else(|| {
                invalid_config(
                    "weights.linear_attn",
                    "linear-attention projection has no layer",
                )
            })?;
            let sources = [
                weight,
                required_weight(config, Some(layer_index), "linear_attn_z")?,
                required_weight(config, Some(layer_index), "linear_attn_b")?,
                required_weight(config, Some(layer_index), "linear_attn_a")?,
            ];
            let dimensions = packed_linear_attention_dimensions(sources)?;
            let component_id =
                packed_linear_attention_component_id(layer_index, PACKED_LINEAR_ATTN_QKVZBA_ROLE)?;
            components.push(WeightComponentSpec {
                id: component_id.clone(),
                role: WeightComponentRole::Values,
                external_names: sources
                    .iter()
                    .map(|source| source.external_name.clone())
                    .collect(),
                dimensions: dimensions.clone(),
                encoding: WeightEncoding::Dense {
                    element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                },
                required: true,
            });
            tensors.push(WeightTensorSpec {
                id: packed_linear_attention_weight_id(layer_index, PACKED_LINEAR_ATTN_QKVZBA_ROLE)?,
                dimensions,
                logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                physical_layout: PhysicalWeightLayout::Dense { component_id },
                required: true,
            });
            continue;
        }
        let (component_id, tensor_id, external_names, physical_dimensions, logical_dimensions) =
            if weight.role == "mlp_gate" {
                let layer_index = weight.layer_index.ok_or_else(|| {
                    invalid_config("weights.mlp_gate", "dense gate weight has no layer")
                })?;
                let up = required_weight(config, Some(layer_index), "mlp_up")?;
                let dimensions = packed_gate_up_dimensions(weight, up)?;
                (
                    packed_gate_up_component_id(layer_index)?,
                    packed_gate_up_weight_id(layer_index)?,
                    vec![weight.external_name.clone(), up.external_name.clone()],
                    dimensions.clone(),
                    dimensions,
                )
            } else {
                (
                    component_id(weight)?,
                    weight_id(weight)?,
                    vec![weight.external_name.clone()],
                    weight.dimensions.clone(),
                    logical_weight_dimensions(weight)?,
                )
            };
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names,
            dimensions: physical_dimensions.clone(),
            encoding: dense_weight_encoding(if weight.role == "mlp_gate" {
                PACKED_GATE_UP_ROLE
            } else {
                &weight.role
            })?,
            required: true,
        });
        let element_type = materialized_element_type(if weight.role == "mlp_gate" {
            PACKED_GATE_UP_ROLE
        } else {
            &weight.role
        });
        tensors.push(WeightTensorSpec {
            id: tensor_id,
            dimensions: logical_dimensions.clone(),
            logical_element_type: element_type,
            physical_layout: dense_or_reshaped_layout(
                FAMILY_ID,
                component_id,
                &physical_dimensions,
                &logical_dimensions,
            )?,
            required: true,
        });
    }
    Ok(WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new(
            "weight-layout.qwen3_5.dense_hybrid.packed_gate_up.packed_gdn_qkvzba",
        )?,
        version: ContractVersion::new(1, 4),
        components,
        tensors,
    })
}

fn safetensors_quantized_weight_schema(
    config: &Qwen35FamilyConfig,
) -> Result<WeightSchema, VNextError> {
    let text = Qwen35TextConfig::from_hf_config_value(&config.hf_config)
        .map_err(|reason| invalid_config("hf_config", reason))?;
    let quantization = text.quantization.as_ref().ok_or_else(|| {
        invalid_config(
            "hf_config.quantization_config",
            "quantized safetensors schema requires typed quantization metadata",
        )
    })?;
    validate_safetensors_quantization_config(quantization)?;
    let compressed_tensors =
        config.weight_format == FamilyWeightFormat::SafetensorsCompressedTensorsMarlin;
    let block_fp8 = config.weight_format == FamilyWeightFormat::SafetensorsBlockFp8;
    if compressed_tensors && text.is_moe() {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "the fixed compressed-tensors adoption contract is dense-only",
        ));
    }
    let mut components = Vec::with_capacity(config.weights.len() * 3);
    let mut tensors = Vec::with_capacity(config.weights.len());
    for weight in &config.weights {
        if is_moe_source_role(&weight.role)
            || matches!(
                weight.role.as_str(),
                "mlp_up" | "linear_attn_z" | "linear_attn_b" | "linear_attn_a"
            )
        {
            continue;
        }
        if weight.role == "linear_attn_qkv" {
            let layer_index = weight.layer_index.ok_or_else(|| {
                invalid_config(
                    "weights.linear_attn",
                    "linear-attention projection has no layer",
                )
            })?;
            let sources = [
                weight,
                required_weight(config, Some(layer_index), "linear_attn_z")?,
                required_weight(config, Some(layer_index), "linear_attn_b")?,
                required_weight(config, Some(layer_index), "linear_attn_a")?,
            ];
            let logical_dimensions = packed_linear_attention_dimensions(sources)?;
            let physical_layout = if sources.iter().all(|source| {
                matches!(
                    &source.source_encoding,
                    FamilyWeightSourceEncoding::Dense { .. }
                )
            }) {
                let component_id = packed_linear_attention_component_id(
                    layer_index,
                    PACKED_LINEAR_ATTN_QKVZBA_ROLE,
                )?;
                components.push(WeightComponentSpec {
                    id: component_id.clone(),
                    role: WeightComponentRole::Values,
                    external_names: sources
                        .iter()
                        .map(|source| source.external_name.clone())
                        .collect(),
                    dimensions: logical_dimensions.clone(),
                    encoding: WeightEncoding::Dense {
                        element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                    },
                    required: true,
                });
                PhysicalWeightLayout::Dense { component_id }
            } else {
                let mut row_offset = 0_u64;
                let mut parts = Vec::with_capacity(sources.len());
                for source in sources {
                    let extents = logical_weight_dimensions(source)?;
                    let layout = append_safetensors_source_layout(
                        source,
                        &extents,
                        quantization,
                        &mut components,
                    )?;
                    parts.push(CompositeWeightPart {
                        layout: Box::new(layout),
                        logical_offsets: vec![row_offset, 0],
                        extents: extents.clone(),
                    });
                    row_offset = row_offset.checked_add(extents[0]).ok_or_else(|| {
                        invalid_config(
                            "weights.linear_attn_projection",
                            "packed projection row offset overflows",
                        )
                    })?;
                }
                PhysicalWeightLayout::Composite { parts }
            };
            tensors.push(WeightTensorSpec {
                id: packed_linear_attention_weight_id(layer_index, PACKED_LINEAR_ATTN_QKVZBA_ROLE)?,
                dimensions: logical_dimensions,
                logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                physical_layout,
                required: true,
            });
            continue;
        }
        if weight.role == "mlp_gate" {
            let layer_index = weight.layer_index.ok_or_else(|| {
                invalid_config("weights.mlp_gate", "dense gate weight has no layer")
            })?;
            let up = required_weight(config, Some(layer_index), "mlp_up")?;
            let logical_dimensions = packed_gate_up_dimensions(weight, up)?;
            let partition_dimensions = vec![1, logical_dimensions[1], logical_dimensions[2]];
            let mut parts = Vec::with_capacity(2);
            for (partition, source) in [weight, up].into_iter().enumerate() {
                let layout = append_safetensors_source_layout(
                    source,
                    &partition_dimensions,
                    quantization,
                    &mut components,
                )?;
                parts.push(CompositeWeightPart {
                    layout: Box::new(layout),
                    logical_offsets: vec![partition as u64, 0, 0],
                    extents: partition_dimensions.clone(),
                });
            }
            tensors.push(WeightTensorSpec {
                id: packed_gate_up_weight_id(layer_index)?,
                dimensions: logical_dimensions,
                logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                physical_layout: PhysicalWeightLayout::Composite { parts },
                required: true,
            });
            continue;
        }

        let logical_dimensions = logical_weight_dimensions(weight)?;
        let layout = append_safetensors_source_layout(
            weight,
            &logical_dimensions,
            quantization,
            &mut components,
        )?;
        tensors.push(WeightTensorSpec {
            id: weight_id(weight)?,
            dimensions: logical_dimensions,
            logical_element_type: materialized_element_type(&weight.role),
            physical_layout: layout,
            required: true,
        });
    }
    if text.is_moe() {
        for layer_index in 0..text.num_hidden_layers {
            append_safetensors_moe_weight_schema(
                config,
                &text,
                quantization,
                layer_index as u32,
                &mut components,
                &mut tensors,
            )?;
        }
    }
    Ok(WeightSchema {
        format_id: WeightFormatId::new(if block_fp8 {
            "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale"
        } else if compressed_tensors {
            "weight-format.safetensors.compressed-tensors-marlin-int4"
        } else {
            "weight-format.safetensors.gptq-marlin-int4"
        })?,
        layout_id: WeightLayoutId::new(if block_fp8 && text.is_moe() {
            "weight-layout.qwen3_5.hybrid_moe.fp8_block_grid.expert_major.packed_gdn_qkvzba"
        } else if block_fp8 {
            "weight-layout.qwen3_5.dense_hybrid.fp8_block_grid.packed_gdn_qkvzba"
        } else if compressed_tensors {
            "weight-layout.qwen3_5.dense_hybrid.compressed_tensors_marlin_asymmetric.packed_gdn_qkvzba"
        } else if text.is_moe() {
            "weight-layout.qwen3_5.hybrid_moe.gptq_marlin_expert_major.packed_gdn_qkvzba"
        } else {
            "weight-layout.qwen3_5.dense_hybrid.gptq_marlin.packed_gdn_qkvzba"
        })?,
        version: if block_fp8 && text.is_moe() {
            ContractVersion::new(1, 0)
        } else if block_fp8 || compressed_tensors {
            ContractVersion::new(1, 0)
        } else if text.is_moe() {
            ContractVersion::new(3, 2)
        } else {
            ContractVersion::new(2, 2)
        },
        components,
        tensors,
    })
}

fn append_safetensors_source_layout(
    weight: &FamilyWeight,
    logical_dimensions: &[u64],
    quantization: &Qwen35QuantizationConfig,
    components: &mut Vec<WeightComponentSpec>,
) -> Result<PhysicalWeightLayout, VNextError> {
    match &weight.source_encoding {
        FamilyWeightSourceEncoding::Dense { .. } => {
            let component = WeightComponentSpec {
                id: component_id(weight)?,
                role: WeightComponentRole::Values,
                external_names: vec![weight.external_name.clone()],
                dimensions: weight.dimensions.clone(),
                encoding: dense_weight_encoding(&weight.role)?,
                required: true,
            };
            let layout = dense_or_reshaped_layout(
                FAMILY_ID,
                component.id.clone(),
                &weight.dimensions,
                logical_dimensions,
            )?;
            components.push(component);
            Ok(layout)
        }
        FamilyWeightSourceEncoding::Gptq {
            qweight,
            scales,
            qzeros,
            g_idx,
        } => {
            validate_gptq_weight_source(
                weight,
                qweight,
                scales,
                qzeros,
                g_idx.as_ref(),
                quantization,
            )?;
            if logical_dimensions.len() < 2
                || logical_dimensions[logical_dimensions.len() - 2..] != weight.dimensions
                || logical_dimensions[..logical_dimensions.len() - 2]
                    .iter()
                    .any(|extent| *extent != 1)
            {
                return Err(invalid_config(
                    "weights.dimensions",
                    format!(
                        "GPTQ source role {:?} cannot represent logical shape {logical_dimensions:?}",
                        weight.role
                    ),
                ));
            }
            let spec = gptq_marlin_quantization_spec(quantization)?;
            let mut packed_dimensions = logical_dimensions.to_vec();
            let packed_axis = packed_dimensions.len() - 1;
            packed_dimensions[packed_axis] /= 2;
            let mut scale_dimensions = logical_dimensions.to_vec();
            let group_axis = scale_dimensions.len() - 1;
            let group_size = validate_gptq_marlin_config(quantization)?.group_size as u64;
            scale_dimensions[group_axis] /= group_size;

            let base = component_id(weight)?.to_string();
            let packed_id = WeightId::new(format!("{base}.packed"))?;
            let scales_id = WeightId::new(format!("{base}.scales"))?;
            let mut packed_sources =
                vec![qweight.external_name.clone(), qzeros.external_name.clone()];
            if let Some(g_idx) = g_idx {
                packed_sources.push(g_idx.external_name.clone());
            }
            components.push(WeightComponentSpec {
                id: packed_id.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: packed_sources,
                dimensions: packed_dimensions.clone(),
                encoding: WeightEncoding::Quantized(spec),
                required: true,
            });
            components.push(WeightComponentSpec {
                id: scales_id.clone(),
                role: WeightComponentRole::Scales,
                external_names: vec![scales.external_name.clone()],
                dimensions: scale_dimensions,
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            });
            Ok(PhysicalWeightLayout::Quantized {
                packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
                packed_dimensions,
                scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                zero_points: None,
                zero_point_packed_dimensions: None,
                axis_indices: None,
                permutation: None,
                codebook: None,
                group_axis: u32::try_from(group_axis).map_err(|_| {
                    invalid_config("weights.dimensions", "GPTQ group axis exceeds u32")
                })?,
                group_padding: PhysicalWeightPadding::Exact,
            })
        }
        FamilyWeightSourceEncoding::CompressedTensors {
            weight_packed,
            weight_scale,
            weight_zero_point,
            weight_shape,
        } => {
            validate_compressed_tensors_weight_source(
                weight,
                weight_packed,
                weight_scale,
                weight_zero_point,
                weight_shape,
                quantization,
            )?;
            if logical_dimensions.len() < 2
                || logical_dimensions[logical_dimensions.len() - 2..] != weight.dimensions
                || logical_dimensions[..logical_dimensions.len() - 2]
                    .iter()
                    .any(|extent| *extent != 1)
            {
                return Err(invalid_config(
                    "weights.dimensions",
                    format!(
                        "compressed-tensors source role {:?} cannot represent logical shape {logical_dimensions:?}",
                        weight.role
                    ),
                ));
            }
            let spec = compressed_tensors_marlin_quantization_spec(quantization)?;
            let mut packed_dimensions = logical_dimensions.to_vec();
            let packed_axis = packed_dimensions.len() - 1;
            packed_dimensions[packed_axis] /= 2;
            let mut scale_dimensions = logical_dimensions.to_vec();
            let group_axis = scale_dimensions.len() - 1;
            let group_size =
                validate_compressed_tensors_marlin_config(quantization)?.group_size as u64;
            scale_dimensions[group_axis] /= group_size;
            let mut zero_point_dimensions =
                logical_dimensions[..logical_dimensions.len() - 2].to_vec();
            zero_point_dimensions.extend([
                logical_dimensions[group_axis] / group_size,
                logical_dimensions[logical_dimensions.len() - 2] / 8,
            ]);

            let base = component_id(weight)?.to_string();
            let packed_id = WeightId::new(format!("{base}.packed"))?;
            let scales_id = WeightId::new(format!("{base}.scales"))?;
            let zero_points_id = WeightId::new(format!("{base}.zero_points"))?;
            components.push(WeightComponentSpec {
                id: packed_id.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: vec![
                    weight_packed.external_name.clone(),
                    weight_shape.external_name.clone(),
                ],
                dimensions: packed_dimensions.clone(),
                encoding: WeightEncoding::Quantized(spec),
                required: true,
            });
            components.push(WeightComponentSpec {
                id: scales_id.clone(),
                role: WeightComponentRole::Scales,
                external_names: vec![weight_scale.external_name.clone()],
                dimensions: scale_dimensions,
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            });
            components.push(WeightComponentSpec {
                id: zero_points_id.clone(),
                role: WeightComponentRole::ZeroPoints,
                external_names: vec![weight_zero_point.external_name.clone()],
                dimensions: zero_point_dimensions.clone(),
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::I32,
                },
                required: true,
            });
            Ok(PhysicalWeightLayout::Quantized {
                packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
                packed_dimensions,
                scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                zero_points: Some(PhysicalWeightComponentBinding::exact_contiguous(
                    zero_points_id,
                )),
                zero_point_packed_dimensions: Some(zero_point_dimensions),
                axis_indices: None,
                permutation: None,
                codebook: None,
                group_axis: u32::try_from(group_axis).map_err(|_| {
                    invalid_config(
                        "weights.dimensions",
                        "compressed-tensors group axis exceeds u32",
                    )
                })?,
                group_padding: PhysicalWeightPadding::Exact,
            })
        }
        FamilyWeightSourceEncoding::BlockFp8 { values, scale_inv } => {
            validate_block_fp8_weight_source(weight, values, scale_inv, quantization)?;
            if logical_dimensions.len() < 2
                || logical_dimensions[logical_dimensions.len() - 2..] != weight.dimensions
                || logical_dimensions[..logical_dimensions.len() - 2]
                    .iter()
                    .any(|extent| *extent != 1)
            {
                return Err(invalid_config(
                    "weights.dimensions",
                    format!(
                        "block-FP8 source role {:?} cannot represent logical shape {logical_dimensions:?}",
                        weight.role
                    ),
                ));
            }
            let spec = block_fp8_source_quantization_spec(quantization)?;
            let packed_dimensions = logical_dimensions.to_vec();
            let block_axes = [
                u32::try_from(logical_dimensions.len() - 2).map_err(|_| {
                    invalid_config("weights.dimensions", "FP8 output block axis exceeds u32")
                })?,
                u32::try_from(logical_dimensions.len() - 1).map_err(|_| {
                    invalid_config("weights.dimensions", "FP8 input block axis exceeds u32")
                })?,
            ];
            let mut scale_dimensions = logical_dimensions.to_vec();
            let [output_block, input_block] = validate_block_fp8_config(quantization)?
                .weight_block_size
                .as_array();
            scale_dimensions[block_axes[0] as usize] =
                scale_dimensions[block_axes[0] as usize].div_ceil(output_block as u64);
            scale_dimensions[block_axes[1] as usize] =
                scale_dimensions[block_axes[1] as usize].div_ceil(input_block as u64);

            let base = component_id(weight)?.to_string();
            let packed_id = WeightId::new(format!("{base}.packed"))?;
            let scales_id = WeightId::new(format!("{base}.inverse_scales"))?;
            components.push(WeightComponentSpec {
                id: packed_id.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: vec![values.external_name.clone()],
                dimensions: packed_dimensions.clone(),
                encoding: WeightEncoding::Quantized(spec),
                required: true,
            });
            components.push(WeightComponentSpec {
                id: scales_id.clone(),
                role: WeightComponentRole::Scales,
                external_names: vec![scale_inv.external_name.clone()],
                dimensions: scale_dimensions,
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                },
                required: true,
            });
            Ok(PhysicalWeightLayout::QuantizedBlockGrid {
                packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
                packed_dimensions,
                scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                block_axes,
            })
        }
        FamilyWeightSourceEncoding::BlockQuantized(_) => Err(invalid_config(
            "weights.source_encoding",
            "GGUF block quantization cannot enter the safetensors quantized schema",
        )),
    }
}

fn append_safetensors_moe_weight_schema(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
    quantization: &Qwen35QuantizationConfig,
    layer_index: u32,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    append_safetensors_direct_moe_weight(
        config,
        text,
        quantization,
        layer_index,
        MOE_ROUTER_ROLE,
        MOE_ROUTER_ROLE,
        components,
        tensors,
    )?;

    let moe = text.moe.as_ref().ok_or_else(|| {
        invalid_config(
            "hf_config.text_config",
            "MoE schema requested for dense model",
        )
    })?;
    let (gate_role, up_role, down_role) =
        if matches!(quantization, Qwen35QuantizationConfig::Fp8(_)) {
            (
                "moe_per_expert_gate_proj",
                "moe_per_expert_up_proj",
                "moe_per_expert_down_proj",
            )
        } else {
            (
                "moe_per_expert_gate_proj_qweight",
                "moe_per_expert_up_proj_qweight",
                "moe_per_expert_down_proj_qweight",
            )
        };
    let gates = required_expert_weights(config, layer_index, gate_role, moe.num_experts)?;
    let ups = required_expert_weights(config, layer_index, up_role, moe.num_experts)?;
    let routed_gate_up_sources = gates
        .into_iter()
        .zip(ups)
        .flat_map(|(gate, up)| [gate, up])
        .collect::<Vec<_>>();
    append_safetensors_quantized_expert_stack(
        routed_gate_up_sources,
        moe_weight_id(layer_index, MOE_ROUTED_GATE_UP_ROLE)?,
        moe_logical_dimensions(text, MOE_ROUTED_GATE_UP_ROLE)?,
        quantization,
        components,
        tensors,
    )?;

    let downs = required_expert_weights(config, layer_index, down_role, moe.num_experts)?;
    append_safetensors_quantized_expert_stack(
        downs,
        moe_weight_id(layer_index, MOE_ROUTED_DOWN_ROLE)?,
        moe_logical_dimensions(text, MOE_ROUTED_DOWN_ROLE)?,
        quantization,
        components,
        tensors,
    )?;

    append_safetensors_direct_moe_weight(
        config,
        text,
        quantization,
        layer_index,
        "moe_shared_expert_gate",
        MOE_SHARED_GATE_ROLE,
        components,
        tensors,
    )?;
    append_safetensors_composite_moe_weight(
        config,
        text,
        quantization,
        layer_index,
        ["moe_shared_expert_gate_proj", "moe_shared_expert_up_proj"],
        MOE_SHARED_GATE_UP_ROLE,
        components,
        tensors,
    )?;
    append_safetensors_direct_moe_weight(
        config,
        text,
        quantization,
        layer_index,
        "moe_shared_expert_down_proj",
        MOE_SHARED_DOWN_ROLE,
        components,
        tensors,
    )
}

fn append_safetensors_quantized_expert_stack(
    sources: Vec<&FamilyWeight>,
    logical_id: WeightId,
    logical_dimensions: Vec<u64>,
    quantization: &Qwen35QuantizationConfig,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    match quantization {
        Qwen35QuantizationConfig::Fp8(_) => append_safetensors_block_fp8_expert_stack(
            sources,
            logical_id,
            logical_dimensions,
            quantization,
            components,
            tensors,
        ),
        Qwen35QuantizationConfig::Gptq(_) => append_safetensors_gptq_expert_stack(
            sources,
            logical_id,
            logical_dimensions,
            quantization,
            components,
            tensors,
        ),
        Qwen35QuantizationConfig::CompressedTensors(_) => Err(invalid_config(
            "hf_config.quantization_config",
            "compressed-tensors MoE expert stacks are outside the fixed adoption contract",
        )),
    }
}

fn append_safetensors_block_fp8_expert_stack(
    sources: Vec<&FamilyWeight>,
    logical_id: WeightId,
    logical_dimensions: Vec<u64>,
    quantization: &Qwen35QuantizationConfig,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let recipe = validate_block_fp8_config(quantization)?;
    if logical_dimensions.len() < 3 {
        return Err(invalid_config(
            "weights.dimensions",
            "block-FP8 expert stack must expose an expert prefix and a matrix",
        ));
    }
    let matrix_axis = logical_dimensions.len() - 2;
    let source_count = logical_dimensions[..matrix_axis]
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| invalid_config("weights.dimensions", "expert stack size overflows"))?;
    if usize::try_from(source_count).ok() != Some(sources.len()) {
        return Err(invalid_config(
            "weights.dimensions",
            format!(
                "block-FP8 expert stack {logical_id} requires {source_count} ordered projections, got {}",
                sources.len()
            ),
        ));
    }

    let source_dimensions = &logical_dimensions[matrix_axis..];
    let mut value_sources = Vec::with_capacity(sources.len());
    let mut scale_sources = Vec::with_capacity(sources.len());
    for source in sources {
        if source.dimensions != source_dimensions {
            return Err(invalid_config(
                "weights.dimensions",
                format!(
                    "block-FP8 expert stack {logical_id} source {:?} shape {:?} differs from {source_dimensions:?}",
                    source.role, source.dimensions
                ),
            ));
        }
        let FamilyWeightSourceEncoding::BlockFp8 { values, scale_inv } = &source.source_encoding
        else {
            return Err(invalid_config(
                "weights.source_encoding",
                format!("block-FP8 expert stack {logical_id} contains a non-block-FP8 source"),
            ));
        };
        validate_block_fp8_weight_source(source, values, scale_inv, quantization)?;
        value_sources.push(values.external_name.clone());
        scale_sources.push(scale_inv.external_name.clone());
    }

    let spec = block_fp8_source_quantization_spec(quantization)?;
    let packed_dimensions = logical_dimensions.clone();
    let mut scale_dimensions = logical_dimensions.clone();
    let [output_block, input_block] = recipe.weight_block_size.as_array();
    scale_dimensions[matrix_axis] = scale_dimensions[matrix_axis].div_ceil(output_block as u64);
    scale_dimensions[matrix_axis + 1] =
        scale_dimensions[matrix_axis + 1].div_ceil(input_block as u64);
    let block_axes = [
        u32::try_from(matrix_axis)
            .map_err(|_| invalid_config("weights.dimensions", "FP8 output axis exceeds u32"))?,
        u32::try_from(matrix_axis + 1)
            .map_err(|_| invalid_config("weights.dimensions", "FP8 input axis exceeds u32"))?,
    ];

    let base = logical_id.to_string();
    let packed_id = WeightId::new(format!("{base}.packed"))?;
    let scales_id = WeightId::new(format!("{base}.inverse_scales"))?;
    components.push(WeightComponentSpec {
        id: packed_id.clone(),
        role: WeightComponentRole::PackedValues,
        external_names: value_sources,
        dimensions: packed_dimensions.clone(),
        encoding: WeightEncoding::Quantized(spec),
        required: true,
    });
    components.push(WeightComponentSpec {
        id: scales_id.clone(),
        role: WeightComponentRole::Scales,
        external_names: scale_sources,
        dimensions: scale_dimensions,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        },
        required: true,
    });
    tensors.push(WeightTensorSpec {
        id: logical_id,
        dimensions: logical_dimensions,
        logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
        physical_layout: PhysicalWeightLayout::QuantizedBlockGrid {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
            packed_dimensions,
            scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
            block_axes,
        },
        required: true,
    });
    Ok(())
}

fn append_safetensors_gptq_expert_stack(
    sources: Vec<&FamilyWeight>,
    logical_id: WeightId,
    logical_dimensions: Vec<u64>,
    quantization: &Qwen35QuantizationConfig,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    validate_gptq_marlin_config(quantization)?;
    if logical_dimensions.len() < 3 {
        return Err(invalid_config(
            "weights.dimensions",
            "GPTQ expert stack must expose an expert axis and a matrix",
        ));
    }
    let source_count = logical_dimensions[..logical_dimensions.len() - 2]
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| invalid_config("weights.dimensions", "expert stack size overflows"))?;
    if usize::try_from(source_count).ok() != Some(sources.len()) {
        return Err(invalid_config(
            "weights.dimensions",
            format!(
                "GPTQ expert stack {logical_id} requires {source_count} ordered projections, got {}",
                sources.len()
            ),
        ));
    }

    let source_dimensions = &logical_dimensions[logical_dimensions.len() - 2..];
    let mut packed_sources = Vec::with_capacity(sources.len() * 3);
    let mut scale_sources = Vec::with_capacity(sources.len());
    let mut has_g_idx = None;
    for source in sources {
        if source.dimensions != source_dimensions {
            return Err(invalid_config(
                "weights.dimensions",
                format!(
                    "GPTQ expert stack {logical_id} source {:?} shape {:?} differs from {:?}",
                    source.role, source.dimensions, source_dimensions
                ),
            ));
        }
        let FamilyWeightSourceEncoding::Gptq {
            qweight,
            scales,
            qzeros,
            g_idx,
        } = &source.source_encoding
        else {
            return Err(invalid_config(
                "weights.source_encoding",
                format!("GPTQ expert stack {logical_id} contains a non-GPTQ source"),
            ));
        };
        validate_gptq_weight_source(
            source,
            qweight,
            scales,
            qzeros,
            g_idx.as_ref(),
            quantization,
        )?;
        match has_g_idx {
            None => has_g_idx = Some(g_idx.is_some()),
            Some(expected) if expected != g_idx.is_some() => {
                return Err(invalid_config(
                    "weights.source_encoding.g_idx",
                    format!(
                        "GPTQ expert stack {logical_id} mixes projections with and without g_idx"
                    ),
                ));
            }
            Some(_) => {}
        }
        packed_sources.extend([qweight.external_name.clone(), qzeros.external_name.clone()]);
        if let Some(g_idx) = g_idx {
            packed_sources.push(g_idx.external_name.clone());
        }
        scale_sources.push(scales.external_name.clone());
    }

    let spec = gptq_marlin_quantization_spec(quantization)?;
    let mut packed_dimensions = logical_dimensions.clone();
    let packed_axis = packed_dimensions.len() - 1;
    if !packed_dimensions[packed_axis].is_multiple_of(2) {
        return Err(invalid_config(
            "weights.dimensions",
            format!("GPTQ expert stack {logical_id} has an odd packed axis"),
        ));
    }
    packed_dimensions[packed_axis] /= 2;
    let mut scale_dimensions = logical_dimensions.clone();
    let group_axis = scale_dimensions.len() - 1;
    let group_size = validate_gptq_marlin_config(quantization)?.group_size as u64;
    if !scale_dimensions[group_axis].is_multiple_of(group_size) {
        return Err(invalid_config(
            "weights.dimensions",
            format!("GPTQ expert stack {logical_id} is not group aligned"),
        ));
    }
    scale_dimensions[group_axis] /= group_size;

    let base = logical_id.to_string();
    let packed_id = WeightId::new(format!("{base}.packed"))?;
    let scales_id = WeightId::new(format!("{base}.scales"))?;
    components.push(WeightComponentSpec {
        id: packed_id.clone(),
        role: WeightComponentRole::PackedValues,
        external_names: packed_sources,
        dimensions: packed_dimensions.clone(),
        encoding: WeightEncoding::Quantized(spec),
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
        id: logical_id,
        dimensions: logical_dimensions,
        logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
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
                .map_err(|_| invalid_config("weights.dimensions", "GPTQ group axis exceeds u32"))?,
            group_padding: PhysicalWeightPadding::Exact,
        },
        required: true,
    });
    Ok(())
}

fn append_safetensors_direct_moe_weight(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
    quantization: &Qwen35QuantizationConfig,
    layer_index: u32,
    source_role: &str,
    logical_role: &str,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let source = required_weight(config, Some(layer_index), source_role)?;
    let dimensions = moe_logical_dimensions(text, logical_role)?;
    let layout = append_safetensors_source_layout(source, &dimensions, quantization, components)?;
    tensors.push(WeightTensorSpec {
        id: moe_weight_id(layer_index, logical_role)?,
        dimensions,
        logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
        physical_layout: layout,
        required: true,
    });
    Ok(())
}

fn append_safetensors_composite_moe_weight(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
    quantization: &Qwen35QuantizationConfig,
    layer_index: u32,
    source_roles: [&str; 2],
    logical_role: &str,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let dimensions = moe_logical_dimensions(text, logical_role)?;
    let sources = source_roles
        .into_iter()
        .map(|role| required_weight(config, Some(layer_index), role))
        .collect::<Result<Vec<_>, _>>()?;
    let source_dimensions = &dimensions[1..];
    if sources.iter().all(|source| {
        matches!(
            &source.source_encoding,
            FamilyWeightSourceEncoding::Dense { .. }
        ) && source.dimensions == source_dimensions
    }) {
        let component_id = moe_component_id(layer_index, logical_role)?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: sources
                .iter()
                .map(|source| source.external_name.clone())
                .collect(),
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: materialized_element_type(logical_role),
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: moe_weight_id(layer_index, logical_role)?,
            dimensions,
            logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
        return Ok(());
    }

    let mut extents = dimensions.clone();
    extents[0] = 1;
    let mut parts = Vec::with_capacity(2);
    for (partition, source) in sources.into_iter().enumerate() {
        let layout = append_safetensors_source_layout(source, &extents, quantization, components)?;
        let mut offsets = vec![0_u64; dimensions.len()];
        offsets[0] = partition as u64;
        parts.push(CompositeWeightPart {
            layout: Box::new(layout),
            logical_offsets: offsets,
            extents: extents.clone(),
        });
    }
    tensors.push(WeightTensorSpec {
        id: moe_weight_id(layer_index, logical_role)?,
        dimensions,
        logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
        physical_layout: PhysicalWeightLayout::Composite { parts },
        required: true,
    });
    Ok(())
}

fn gptq_marlin_quantization_spec(
    quantization: &Qwen35QuantizationConfig,
) -> Result<QuantizationSpec, VNextError> {
    let recipe = validate_gptq_marlin_config(quantization)?;
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(GPTQ_MARLIN_INT4_FORMAT_ID)?,
        bits_per_weight: 4,
        grouping: QuantizationGrouping::fixed(u32::try_from(recipe.group_size).map_err(|_| {
            invalid_config(
                "hf_config.quantization_config.group_size",
                "GPTQ group size exceeds u32",
            )
        })?),
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: None,
    })
}

fn compressed_tensors_marlin_quantization_spec(
    quantization: &Qwen35QuantizationConfig,
) -> Result<QuantizationSpec, VNextError> {
    validate_compressed_tensors_marlin_config(quantization)?;
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(COMPRESSED_TENSORS_MARLIN_INT4_FORMAT_ID)?,
        bits_per_weight: 4,
        grouping: QuantizationGrouping::fixed(32),
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: Some(ElementType::I32),
    })
}

fn gguf_weight_schema(config: &Qwen35FamilyConfig) -> Result<WeightSchema, VNextError> {
    let text = Qwen35TextConfig::from_hf_config_value(&config.hf_config)
        .map_err(|reason| invalid_config("hf_config", reason))?;
    let mut components = Vec::with_capacity(config.weights.len());
    let mut tensors = Vec::with_capacity(config.weights.len());
    for weight in &config.weights {
        if is_moe_source_role(&weight.role) {
            continue;
        }
        if matches!(
            weight.role.as_str(),
            "mlp_up" | "linear_attn_z" | "linear_attn_b" | "linear_attn_a"
        ) {
            continue;
        }
        if weight.role == "linear_attn_qkv" {
            let layer_index = weight.layer_index.ok_or_else(|| {
                invalid_config(
                    "weights.linear_attn",
                    "linear-attention projection has no layer",
                )
            })?;
            let sources = [
                weight,
                required_weight(config, Some(layer_index), "linear_attn_z")?,
                required_weight(config, Some(layer_index), "linear_attn_b")?,
                required_weight(config, Some(layer_index), "linear_attn_a")?,
            ];
            let logical_dimensions = packed_linear_attention_dimensions(sources)?;
            let mut row_offset = 0_u64;
            let mut parts = Vec::with_capacity(sources.len());
            for source in sources {
                let extents = logical_weight_dimensions(source)?;
                let component = gguf_component_spec(source)?;
                let layout = gguf_component_layout(source, component.id.clone(), &extents)?;
                parts.push(CompositeWeightPart {
                    layout: Box::new(layout),
                    logical_offsets: vec![row_offset, 0],
                    extents: extents.clone(),
                });
                row_offset = row_offset.checked_add(extents[0]).ok_or_else(|| {
                    invalid_config(
                        "weights.linear_attn_projection",
                        "packed projection row offset overflows",
                    )
                })?;
                components.push(component);
            }
            tensors.push(WeightTensorSpec {
                id: packed_linear_attention_weight_id(layer_index, PACKED_LINEAR_ATTN_QKVZBA_ROLE)?,
                dimensions: logical_dimensions,
                logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                physical_layout: PhysicalWeightLayout::Composite { parts },
                required: true,
            });
            continue;
        }
        if weight.role == "mlp_gate" {
            let layer_index = weight.layer_index.ok_or_else(|| {
                invalid_config("weights.mlp_gate", "dense gate weight has no layer")
            })?;
            let up = required_weight(config, Some(layer_index), "mlp_up")?;
            let packed_dimensions = packed_gate_up_dimensions(weight, up)?;
            let partition_dimensions = packed_dimensions[1..].to_vec();
            let composite_extents = std::iter::once(1_u64)
                .chain(partition_dimensions.iter().copied())
                .collect::<Vec<_>>();
            let mut parts = Vec::with_capacity(2);
            for (partition, source) in [weight, up].into_iter().enumerate() {
                let component = gguf_component_spec(source)?;
                let layout =
                    gguf_component_layout(source, component.id.clone(), &composite_extents)?;
                let mut logical_offsets = vec![0_u64; packed_dimensions.len()];
                logical_offsets[0] = partition as u64;
                parts.push(CompositeWeightPart {
                    layout: Box::new(layout),
                    logical_offsets,
                    extents: composite_extents.clone(),
                });
                components.push(component);
            }
            tensors.push(WeightTensorSpec {
                id: packed_gate_up_weight_id(layer_index)?,
                dimensions: packed_dimensions,
                logical_element_type: materialized_element_type(PACKED_GATE_UP_ROLE),
                physical_layout: PhysicalWeightLayout::Composite { parts },
                required: true,
            });
            continue;
        }

        let component = gguf_component_spec(weight)?;
        let logical_dimensions = logical_weight_dimensions(weight)?;
        let layout = gguf_component_layout(weight, component.id.clone(), &logical_dimensions)?;
        components.push(component);
        tensors.push(WeightTensorSpec {
            id: weight_id(weight)?,
            dimensions: logical_dimensions,
            logical_element_type: materialized_element_type(&weight.role),
            physical_layout: layout,
            required: true,
        });
    }
    if text.moe.is_some() {
        for layer_index in 0..text.num_hidden_layers {
            append_gguf_moe_weight_schema(
                config,
                &text,
                layer_index as u32,
                &mut components,
                &mut tensors,
            )?;
        }
    }
    Ok(WeightSchema {
        format_id: WeightFormatId::new("weight-format.gguf.native-block")?,
        layout_id: WeightLayoutId::new(if text.is_moe() {
            "weight-layout.qwen3_5.hybrid_moe.gguf.native.packed_gdn_qkvzba"
        } else {
            "weight-layout.qwen3_5.dense_hybrid.gguf.native.packed_gdn_qkvzba"
        })?,
        version: if text.is_moe() {
            ContractVersion::new(2, 2)
        } else {
            ContractVersion::new(1, 2)
        },
        components,
        tensors,
    })
}

fn append_gguf_moe_weight_schema(
    config: &Qwen35FamilyConfig,
    text: &Qwen35TextConfig,
    layer_index: u32,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let source = |role| required_weight(config, Some(layer_index), role);

    append_gguf_direct_logical_weight(
        source(MOE_ROUTER_ROLE)?,
        moe_weight_id(layer_index, MOE_ROUTER_ROLE)?,
        moe_logical_dimensions(text, MOE_ROUTER_ROLE)?,
        components,
        tensors,
    )?;
    append_gguf_composite_logical_weight(
        [
            source("moe_stacked_gate_proj")?,
            source("moe_stacked_up_proj")?,
        ],
        moe_weight_id(layer_index, MOE_ROUTED_GATE_UP_ROLE)?,
        moe_logical_dimensions(text, MOE_ROUTED_GATE_UP_ROLE)?,
        1,
        components,
        tensors,
    )?;
    append_gguf_direct_logical_weight(
        source("moe_stacked_down_proj")?,
        moe_weight_id(layer_index, MOE_ROUTED_DOWN_ROLE)?,
        moe_logical_dimensions(text, MOE_ROUTED_DOWN_ROLE)?,
        components,
        tensors,
    )?;
    append_gguf_direct_logical_weight(
        source("moe_shared_expert_gate")?,
        moe_weight_id(layer_index, MOE_SHARED_GATE_ROLE)?,
        moe_logical_dimensions(text, MOE_SHARED_GATE_ROLE)?,
        components,
        tensors,
    )?;
    append_gguf_composite_logical_weight(
        [
            source("moe_shared_expert_gate_proj")?,
            source("moe_shared_expert_up_proj")?,
        ],
        moe_weight_id(layer_index, MOE_SHARED_GATE_UP_ROLE)?,
        moe_logical_dimensions(text, MOE_SHARED_GATE_UP_ROLE)?,
        0,
        components,
        tensors,
    )?;
    append_gguf_direct_logical_weight(
        source("moe_shared_expert_down_proj")?,
        moe_weight_id(layer_index, MOE_SHARED_DOWN_ROLE)?,
        moe_logical_dimensions(text, MOE_SHARED_DOWN_ROLE)?,
        components,
        tensors,
    )
}

fn append_gguf_direct_logical_weight(
    source: &FamilyWeight,
    logical_id: WeightId,
    logical_dimensions: Vec<u64>,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    let component = gguf_component_spec(source)?;
    let layout = gguf_component_layout(source, component.id.clone(), &logical_dimensions)?;
    components.push(component);
    tensors.push(WeightTensorSpec {
        id: logical_id,
        dimensions: logical_dimensions,
        logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
        physical_layout: layout,
        required: true,
    });
    Ok(())
}

fn append_gguf_composite_logical_weight(
    sources: [&FamilyWeight; 2],
    logical_id: WeightId,
    logical_dimensions: Vec<u64>,
    partition_axis: usize,
    components: &mut Vec<WeightComponentSpec>,
    tensors: &mut Vec<WeightTensorSpec>,
) -> Result<(), VNextError> {
    if logical_dimensions.get(partition_axis).copied() != Some(2) {
        return Err(invalid_config(
            "weights.moe_composite",
            "MoE gate/up logical tensor must expose exactly two ordered partitions",
        ));
    }
    let mut part_dimensions = logical_dimensions.clone();
    part_dimensions[partition_axis] = 1;
    let mut parts = Vec::with_capacity(2);
    for (partition, source) in sources.into_iter().enumerate() {
        let component = gguf_component_spec(source)?;
        let layout = gguf_component_layout(source, component.id.clone(), &part_dimensions)?;
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
        id: logical_id,
        dimensions: logical_dimensions,
        logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
        physical_layout: PhysicalWeightLayout::Composite { parts },
        required: true,
    });
    Ok(())
}

fn gguf_component_spec(weight: &FamilyWeight) -> Result<WeightComponentSpec, VNextError> {
    let (role, dimensions, encoding) = match &weight.source_encoding {
        FamilyWeightSourceEncoding::Dense { .. } => (
            WeightComponentRole::Values,
            weight.dimensions.clone(),
            WeightEncoding::Dense {
                element_type: materialized_element_type(&weight.role),
            },
        ),
        FamilyWeightSourceEncoding::BlockQuantized(spec) => {
            let mut dimensions = weight.dimensions.clone();
            let innermost = dimensions.last_mut().ok_or_else(|| {
                invalid_config("weights.dimensions", "GGUF block tensor has no axis")
            })?;
            let block_width = u64::from(spec.logical_values_per_block);
            if !innermost.is_multiple_of(block_width) {
                return Err(invalid_config(
                    "weights.dimensions",
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
        FamilyWeightSourceEncoding::Gptq { .. }
        | FamilyWeightSourceEncoding::CompressedTensors { .. }
        | FamilyWeightSourceEncoding::BlockFp8 { .. } => {
            return Err(invalid_config(
                "weights.source_encoding",
                "safetensors quantized components cannot enter the GGUF schema",
            ));
        }
    };
    Ok(WeightComponentSpec {
        id: component_id(weight)?,
        role,
        external_names: vec![weight.external_name.clone()],
        dimensions,
        encoding,
        required: true,
    })
}

fn gguf_component_layout(
    weight: &FamilyWeight,
    component_id: WeightId,
    logical_dimensions: &[u64],
) -> Result<PhysicalWeightLayout, VNextError> {
    match &weight.source_encoding {
        FamilyWeightSourceEncoding::Dense { .. } => dense_or_reshaped_layout(
            FAMILY_ID,
            component_id,
            &weight.dimensions,
            logical_dimensions,
        ),
        FamilyWeightSourceEncoding::BlockQuantized(spec) => {
            let block_axis = logical_dimensions
                .len()
                .checked_sub(1)
                .ok_or_else(|| invalid_config("weights.dimensions", "GGUF weight has no axis"))?;
            let block_width = u64::from(spec.logical_values_per_block);
            let mut logical_blocks = logical_dimensions.to_vec();
            let innermost = logical_blocks.last_mut().unwrap();
            if !innermost.is_multiple_of(block_width) {
                return Err(invalid_config(
                    "weights.dimensions",
                    "GGUF logical tensor innermost dimension is not block aligned",
                ));
            }
            *innermost /= block_width;
            let mut physical_blocks = weight.dimensions.clone();
            *physical_blocks.last_mut().unwrap() /= block_width;
            Ok(PhysicalWeightLayout::BlockQuantized {
                blocks: contiguous_or_reshaped_binding(
                    FAMILY_ID,
                    component_id,
                    &physical_blocks,
                    &logical_blocks,
                )?,
                block_axis: u32::try_from(block_axis).map_err(|_| {
                    invalid_config("weights.dimensions", "GGUF block axis exceeds u32")
                })?,
                block_padding: PhysicalWeightPadding::Exact,
            })
        }
        FamilyWeightSourceEncoding::Gptq { .. }
        | FamilyWeightSourceEncoding::CompressedTensors { .. }
        | FamilyWeightSourceEncoding::BlockFp8 { .. } => Err(invalid_config(
            "weights.source_encoding",
            "safetensors quantized components cannot enter the GGUF schema",
        )),
    }
}

pub fn prepare_from_model_dir(model_dir: &Path) -> ferrum_types::Result<PreparedProductionModel> {
    let sources = Arc::new(super::open_registered_colocated_safetensors(model_dir)?);
    prepare_from_sources(sources)
}

pub(super) fn prepare_from_sources(
    sources: Arc<ProductionModelSourceBundle>,
) -> ferrum_types::Result<PreparedProductionModel> {
    preflight_semantic_config(sources.config_json()).map_err(ferrum_types::FerrumError::model)?;
    match sources.weights() {
        ProductionWeightArtifact::SafetensorsDirectory(weight_root) => {
            let weights = SafetensorsArchive::open(weight_root)?;
            let config = load_safetensors_family_config(&sources, &weights)
                .map_err(ferrum_types::FerrumError::model)?;
            match config.weight_format {
                FamilyWeightFormat::SafetensorsGptqMarlin => {
                    finish_preparation(sources, GptqMarlinSafetensorsSource::new(weights), config)
                }
                FamilyWeightFormat::SafetensorsCompressedTensorsMarlin => finish_preparation(
                    sources,
                    CompressedTensorsMarlinSafetensorsSource::new(weights),
                    config,
                ),
                FamilyWeightFormat::SafetensorsBlockFp8 => {
                    finish_preparation(sources, BlockFp8SafetensorsSource::new(weights), config)
                }
                FamilyWeightFormat::SafetensorsDense => {
                    finish_preparation(sources, weights, config)
                }
                FamilyWeightFormat::GgufNative => Err(ferrum_types::FerrumError::internal(
                    "GGUF format cannot be selected from a safetensors artifact",
                )),
            }
        }
        ProductionWeightArtifact::GgufFile(path) => {
            let weights = GgufWeightComponentSource::open(path)?;
            let config = load_gguf_family_config(&sources, &weights)
                .map_err(ferrum_types::FerrumError::model)?;
            finish_preparation(sources, weights, config)
        }
    }
}

fn finish_preparation<W>(
    sources: Arc<ProductionModelSourceBundle>,
    weights: W,
    config: Qwen35FamilyConfig,
) -> ferrum_types::Result<PreparedProductionModel>
where
    W: WeightComponentSource + 'static,
{
    let descriptor = production_descriptor(&config).map_err(ferrum_types::FerrumError::model)?;
    let raw = serde_json::to_value(config)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let provider = Qwen35FamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let family = TypedFamilyRegistration::new(provider)
        .prepare(&raw)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(PreparedProductionModel::new(
        family, weights, descriptor, sources,
    ))
}

fn production_descriptor(
    config: &Qwen35FamilyConfig,
) -> Result<CausalLanguageModelDescriptor, String> {
    let text = Qwen35TextConfig::from_hf_config_value(&config.hf_config)?;
    let parameter_count = config.weights.iter().try_fold(0_u64, |total, weight| {
        let elements = weight
            .dimensions
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product.checked_mul(*dimension).ok_or_else(|| {
                    format!(
                        "parameter count overflow for weight {:?}",
                        weight.external_name
                    )
                })
            })?;
        total.checked_add(elements).ok_or_else(|| {
            format!(
                "parameter count overflow after weight {:?}",
                weight.external_name
            )
        })
    })?;
    let vocabulary_size =
        usize::try_from(config.vocab_size).map_err(|_| "vocab_size exceeds usize".to_owned())?;
    let maximum_sequence_tokens = usize::try_from(config.max_position_embeddings)
        .map_err(|_| "max_position_embeddings exceeds usize".to_owned())?;
    CausalLanguageModelDescriptor::new(
        if text.is_moe() {
            "qwen3_5_moe"
        } else {
            "qwen3_5"
        },
        parameter_count,
        text.hidden_size,
        text.num_hidden_layers,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        vocabulary_size,
        maximum_sequence_tokens,
        DataType::FP16,
    )
    .map_err(|error| error.to_string())
}

fn load_safetensors_family_config(
    sources: &ProductionModelSourceBundle,
    archive: &SafetensorsArchive,
) -> Result<Qwen35FamilyConfig, String> {
    let hf_config = compose_safetensors_hf_config(sources)?;
    let text = Qwen35TextConfig::from_hf_config_value(&hf_config)?;
    if let Some(quantization) = &text.quantization {
        validate_safetensors_quantization_config(quantization)
            .map_err(|error| error.to_string())?;
    }

    let text_value = hf_config.get("text_config").unwrap_or(&hf_config);
    let vocab_size = required_u64(text_value, "vocab_size")?;
    let max_position_embeddings = required_u64(text_value, "max_position_embeddings")?;
    let rms_norm_epsilon = hf_rms_norm_epsilon(&hf_config)?;
    let tokenizer_config_bytes = sources
        .tokenizer_config_json()
        .ok_or_else(|| "tokenizer source missing tokenizer_config.json".to_owned())?;
    let metadata = parse_hf_model_semantic_metadata(&hf_config, &tokenizer_config_bytes)?;

    let inventory = Qwen35WeightInventory::from_names(archive.tensor_names());
    let plan = inventory.detect_prefix_and_resolve(&text)?;
    if text
        .quantization
        .as_ref()
        .and_then(Qwen35QuantizationConfig::as_fp8)
        .is_some()
    {
        inventory
            .partition_resolved_plan(&plan)?
            .require_no_unknown()?;
    }
    let mut weights = Vec::new();
    append_resolved_weights(
        &mut weights,
        &plan.global_tensors,
        None,
        text.tie_word_embeddings,
        text.quantization.as_ref(),
        archive,
    )?;
    for layer in &plan.layers {
        append_resolved_weights(
            &mut weights,
            &layer.tensors,
            Some(layer.layer_index as u32),
            text.tie_word_embeddings,
            text.quantization.as_ref(),
            archive,
        )?;
    }
    weights.sort_by(|left, right| {
        (left.layer_index, left.role.as_str(), left.expert_index).cmp(&(
            right.layer_index,
            right.role.as_str(),
            right.expert_index,
        ))
    });

    Ok(Qwen35FamilyConfig {
        hf_config,
        vocab_size,
        max_position_embeddings,
        rms_norm_epsilon,
        metadata,
        weight_format: match text
            .quantization
            .as_ref()
            .map(Qwen35QuantizationConfig::quant_method)
        {
            Some("gptq") => FamilyWeightFormat::SafetensorsGptqMarlin,
            Some("compressed-tensors") => FamilyWeightFormat::SafetensorsCompressedTensorsMarlin,
            Some("fp8") => FamilyWeightFormat::SafetensorsBlockFp8,
            Some(other) => {
                return Err(format!(
                    "unsupported safetensors quantization method {other:?}"
                ))
            }
            None => FamilyWeightFormat::SafetensorsDense,
        },
        weights,
    })
}

fn compose_safetensors_hf_config(sources: &ProductionModelSourceBundle) -> Result<Value, String> {
    let mut semantic: Value = serde_json::from_slice(sources.config_json())
        .map_err(|error| format!("parse semantic config.json: {error}"))?;
    let semantic_root = semantic
        .as_object_mut()
        .ok_or_else(|| "semantic config.json root must be an object".to_owned())?;
    let semantic_quantization = semantic_root
        .get("text_config")
        .and_then(Value::as_object)
        .and_then(|text| text.get("quantization_config"))
        .or_else(|| semantic_root.get("quantization_config"))
        .filter(|value| !value.is_null())
        .cloned();
    let physical_quantization = sources
        .weight_config_json()
        .map(|bytes| {
            serde_json::from_slice::<Value>(bytes)
                .map_err(|error| format!("parse physical weight config.json: {error}"))
        })
        .transpose()?
        .as_ref()
        .and_then(|physical| {
            physical
                .get("text_config")
                .and_then(Value::as_object)
                .and_then(|text| text.get("quantization_config"))
                .or_else(|| physical.get("quantization_config"))
        })
        .filter(|value| !value.is_null())
        .cloned();

    match (semantic_quantization, physical_quantization) {
        (Some(semantic_value), Some(physical_value)) if semantic_value != physical_value => {
            return Err("semantic and physical weight quantization_config values differ".to_owned())
        }
        (None, Some(physical_value)) => {
            semantic_root.insert("quantization_config".to_owned(), physical_value);
        }
        (Some(_), Some(_)) | (Some(_), None) | (None, None) => {}
    }
    Ok(semantic)
}

fn load_gguf_family_config(
    sources: &ProductionModelSourceBundle,
    source: &GgufWeightComponentSource,
) -> Result<Qwen35FamilyConfig, String> {
    let hf_config: Value = serde_json::from_slice(sources.config_json())
        .map_err(|error| format!("parse semantic config.json: {error}"))?;
    let text = Qwen35TextConfig::from_hf_config_value(&hf_config)?;
    if text.quantization.is_some() {
        return Err(
            "GGUF physical quantization must not be duplicated in Hugging Face semantic metadata"
                .to_owned(),
        );
    }
    let architecture = source
        .file()
        .architecture()
        .map_err(|error| format!("read GGUF architecture: {error}"))?;
    let expected_architecture = gguf_architecture(&text);
    if architecture != expected_architecture {
        return Err(format!(
            "Qwen3.5 family package requires GGUF architecture {expected_architecture:?}, got {architecture:?}"
        ));
    }

    let text_value = hf_config.get("text_config").unwrap_or(&hf_config);
    let vocab_size = required_u64(text_value, "vocab_size")?;
    let max_position_embeddings = required_u64(text_value, "max_position_embeddings")?;
    let rms_norm_epsilon = hf_rms_norm_epsilon(&hf_config)?;
    let tokenizer_config_bytes = sources
        .tokenizer_config_json()
        .ok_or_else(|| "tokenizer source missing tokenizer_config.json".to_owned())?;
    let metadata = parse_hf_model_semantic_metadata(&hf_config, tokenizer_config_bytes)?;

    let manifest = text.weight_manifest("model.language_model")?;
    let mut weights = Vec::new();
    append_gguf_weights(
        &mut weights,
        &manifest.global_tensors,
        None,
        text.tie_word_embeddings,
        expected_architecture,
        source,
    )?;
    for layer in &manifest.layers {
        append_gguf_weights(
            &mut weights,
            &layer.tensors,
            Some(layer.layer_index as u32),
            text.tie_word_embeddings,
            expected_architecture,
            source,
        )?;
    }
    weights.sort_by(|left, right| {
        (left.layer_index, left.role.as_str(), left.expert_index).cmp(&(
            right.layer_index,
            right.role.as_str(),
            right.expert_index,
        ))
    });

    Ok(Qwen35FamilyConfig {
        hf_config,
        vocab_size,
        max_position_embeddings,
        rms_norm_epsilon,
        metadata,
        weight_format: FamilyWeightFormat::GgufNative,
        weights,
    })
}

fn append_gguf_weights(
    output: &mut Vec<FamilyWeight>,
    specs: &[Qwen35WeightSpec],
    layer_index: Option<u32>,
    tied_embeddings: bool,
    architecture: &str,
    source: &GgufWeightComponentSource,
) -> Result<(), String> {
    for spec in specs
        .iter()
        .filter(|spec| !(tied_embeddings && spec.role == "lm_head"))
    {
        let Some(external_name) = ferrum_to_gguf_with_arch(architecture, &spec.name) else {
            if spec.required {
                return Err(format!(
                    "Qwen3.5 GGUF has no typed name mapping for required role {:?} source {:?}",
                    spec.role, spec.name
                ));
            }
            continue;
        };
        let Some(info) = source.file().tensor_info(&external_name) else {
            if spec.required {
                return Err(format!(
                    "Qwen3.5 GGUF is missing required role {:?} tensor {external_name:?}",
                    spec.role
                ));
            }
            continue;
        };
        let dimensions = info
            .shape
            .dims()
            .iter()
            .map(|dimension| *dimension as u64)
            .collect::<Vec<_>>();
        output.push(FamilyWeight {
            layer_index,
            expert_index: None,
            role: spec.role.clone(),
            external_name,
            dimensions,
            source_encoding: gguf_source_encoding(info.ggml_dtype)?,
        });
    }
    Ok(())
}

fn gguf_source_encoding(dtype: GgmlDType) -> Result<FamilyWeightSourceEncoding, String> {
    if let Some(format_id) = block_quantization_format(dtype) {
        return Ok(FamilyWeightSourceEncoding::BlockQuantized(
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
    let element_type = match dtype {
        GgmlDType::F16 => ElementType::F16,
        GgmlDType::BF16 => ElementType::Bf16,
        GgmlDType::F32 => ElementType::F32,
        _ => return Err(format!("unsupported Qwen3.5 GGUF tensor dtype {dtype:?}")),
    };
    Ok(FamilyWeightSourceEncoding::Dense { element_type })
}

fn data_type_to_element_type(dtype: DataType) -> Result<ElementType, String> {
    match dtype {
        DataType::FP16 => Ok(ElementType::F16),
        DataType::BF16 => Ok(ElementType::Bf16),
        DataType::FP32 => Ok(ElementType::F32),
        _ => Err(format!("unsupported vNext state dtype {dtype}")),
    }
}

fn append_resolved_weights(
    output: &mut Vec<FamilyWeight>,
    resolved: &[Qwen35ResolvedWeightSpec],
    layer_index: Option<u32>,
    tied_embeddings: bool,
    quantization: Option<&Qwen35QuantizationConfig>,
    weights: &SafetensorsArchive,
) -> Result<(), String> {
    for weight in resolved
        .iter()
        .filter(|weight| weight.present && !(tied_embeddings && weight.role == "lm_head"))
    {
        let source = weight.source.as_ref().ok_or_else(|| {
            format!(
                "resolved Qwen3.5 tensor {:?} has no typed source bundle",
                weight.name
            )
        })?;
        let (external_name, dimensions, source_encoding) = match source {
            Qwen35ResolvedWeightSource::Dense { values } => {
                let tensor = weights.tensor(values).map_err(|error| error.to_string())?;
                let source_element_type = tensor.element_type().ok_or_else(|| {
                    format!(
                        "resolved dense Qwen3.5 tensor {values:?} has unsupported dtype {:?}",
                        tensor.dtype()
                    )
                })?;
                if !matches!(
                    source_element_type,
                    ElementType::F16 | ElementType::Bf16 | ElementType::F32
                ) {
                    return Err(format!(
                        "resolved dense Qwen3.5 tensor {values:?} must have a floating-point source dtype, got {source_element_type:?}"
                    ));
                }
                (
                    values.clone(),
                    tensor.shape().to_vec(),
                    FamilyWeightSourceEncoding::Dense {
                        element_type: source_element_type,
                    },
                )
            }
            Qwen35ResolvedWeightSource::BlockFp8 { values, scale_inv } => {
                let recipe = quantization
                    .and_then(Qwen35QuantizationConfig::as_fp8)
                    .ok_or_else(|| {
                        format!("resolved block-FP8 tensor {values:?} has no typed FP8 recipe")
                    })?;
                let values_source =
                    family_block_fp8_tensor(weights, values, FamilyBlockFp8Dtype::F8E4m3)?;
                let scale_source =
                    family_block_fp8_tensor(weights, scale_inv, FamilyBlockFp8Dtype::Bf16)?;
                let [n, k] = values_source.dimensions.as_slice() else {
                    return Err(format!(
                        "resolved block-FP8 values {values:?} must have shape [N, K]"
                    ));
                };
                let block = recipe.weight_block_size;
                let output_block = u64::try_from(block.output_features)
                    .map_err(|_| "FP8 output block size exceeds u64".to_owned())?;
                let input_block = u64::try_from(block.input_features)
                    .map_err(|_| "FP8 input block size exceeds u64".to_owned())?;
                let expected_scale_dimensions =
                    vec![n.div_ceil(output_block), k.div_ceil(input_block)];
                if scale_source.dimensions != expected_scale_dimensions {
                    return Err(format!(
                        "block-FP8 inverse-scale {scale_inv:?} shape {:?} differs from the typed grid {:?} for values [{n}, {k}]",
                        scale_source.dimensions, expected_scale_dimensions
                    ));
                }
                (
                    values.clone(),
                    values_source.dimensions.clone(),
                    FamilyWeightSourceEncoding::BlockFp8 {
                        values: values_source,
                        scale_inv: scale_source,
                    },
                )
            }
            Qwen35ResolvedWeightSource::Gptq {
                qweight,
                scales,
                qzeros,
                g_idx,
            } => {
                let qweight_source = family_gptq_tensor(weights, qweight)?;
                let scales_source = family_gptq_tensor(weights, scales)?;
                let qzeros_source = family_gptq_tensor(weights, qzeros)?;
                let g_idx_source = g_idx
                    .as_deref()
                    .map(|name| family_gptq_tensor(weights, name))
                    .transpose()?;
                let [packed_k, n] = qweight_source.dimensions.as_slice() else {
                    return Err(format!(
                        "resolved GPTQ qweight {qweight:?} must have shape [K/8, N]"
                    ));
                };
                let k = packed_k
                    .checked_mul(8)
                    .ok_or_else(|| format!("GPTQ qweight {qweight:?} K dimension overflows"))?;
                (
                    qweight.clone(),
                    vec![*n, k],
                    FamilyWeightSourceEncoding::Gptq {
                        qweight: qweight_source,
                        scales: scales_source,
                        qzeros: qzeros_source,
                        g_idx: g_idx_source,
                    },
                )
            }
            Qwen35ResolvedWeightSource::CompressedTensors {
                weight_packed,
                weight_scale,
                weight_zero_point,
                weight_shape,
            } => {
                let packed_source = family_compressed_tensors_tensor(weights, weight_packed)?;
                let scale_source = family_compressed_tensors_tensor(weights, weight_scale)?;
                let zero_point_source =
                    family_compressed_tensors_tensor(weights, weight_zero_point)?;
                let shape_source = family_compressed_tensors_tensor(weights, weight_shape)?;
                let tensor = weights
                    .tensor(weight_shape)
                    .map_err(|error| error.to_string())?;
                if tensor.dtype() != safetensors::Dtype::I64
                    || tensor.shape() != [2]
                    || tensor.bytes().len() != 16
                {
                    return Err(format!(
                        "compressed-tensors shape metadata {weight_shape:?} must be I64[2]"
                    ));
                }
                let dimensions = tensor
                    .bytes()
                    .chunks_exact(8)
                    .map(|bytes| {
                        let value = i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]);
                        u64::try_from(value).map_err(|_| {
                            format!(
                                "compressed-tensors shape metadata {weight_shape:?} contains a non-positive extent"
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if dimensions.contains(&0) {
                    return Err(format!(
                        "compressed-tensors shape metadata {weight_shape:?} contains a zero extent"
                    ));
                }
                (
                    weight_packed.clone(),
                    dimensions,
                    FamilyWeightSourceEncoding::CompressedTensors {
                        weight_packed: packed_source,
                        weight_scale: scale_source,
                        weight_zero_point: zero_point_source,
                        weight_shape: shape_source,
                    },
                )
            }
        };
        output.push(FamilyWeight {
            layer_index,
            expert_index: weight.expert_index,
            role: weight.role.clone(),
            external_name,
            dimensions,
            source_encoding,
        });
    }
    Ok(())
}

fn family_gptq_tensor(
    weights: &SafetensorsArchive,
    external_name: &str,
) -> Result<FamilyGptqTensor, String> {
    let tensor = weights
        .tensor(external_name)
        .map_err(|error| error.to_string())?;
    let element_type = tensor.element_type().ok_or_else(|| {
        format!(
            "resolved GPTQ tensor {external_name:?} has unsupported dtype {:?}",
            tensor.dtype()
        )
    })?;
    Ok(FamilyGptqTensor {
        external_name: external_name.to_owned(),
        dimensions: tensor.shape().to_vec(),
        element_type,
    })
}

fn family_compressed_tensors_tensor(
    weights: &SafetensorsArchive,
    external_name: &str,
) -> Result<FamilyCompressedTensorsTensor, String> {
    let tensor = weights
        .tensor(external_name)
        .map_err(|error| error.to_string())?;
    let dtype = match tensor.dtype() {
        safetensors::Dtype::I32 => FamilyCompressedTensorsDtype::I32,
        safetensors::Dtype::I64 => FamilyCompressedTensorsDtype::I64,
        safetensors::Dtype::F16 => FamilyCompressedTensorsDtype::F16,
        safetensors::Dtype::BF16 => FamilyCompressedTensorsDtype::Bf16,
        other => {
            return Err(format!(
            "resolved compressed-tensors tensor {external_name:?} has unsupported dtype {other:?}"
        ))
        }
    };
    Ok(FamilyCompressedTensorsTensor {
        external_name: external_name.to_owned(),
        dimensions: tensor.shape().to_vec(),
        dtype,
    })
}

fn family_block_fp8_tensor(
    weights: &SafetensorsArchive,
    external_name: &str,
    expected_dtype: FamilyBlockFp8Dtype,
) -> Result<FamilyBlockFp8Tensor, String> {
    let tensor = weights
        .tensor(external_name)
        .map_err(|error| error.to_string())?;
    let dtype = match tensor.dtype() {
        safetensors::Dtype::F8_E4M3 => FamilyBlockFp8Dtype::F8E4m3,
        safetensors::Dtype::BF16 => FamilyBlockFp8Dtype::Bf16,
        other => {
            return Err(format!(
                "resolved block-FP8 tensor {external_name:?} has unsupported dtype {other:?}"
            ))
        }
    };
    if dtype != expected_dtype {
        return Err(format!(
            "resolved block-FP8 tensor {external_name:?} has dtype {dtype:?}, expected {expected_dtype:?}"
        ));
    }
    Ok(FamilyBlockFp8Tensor {
        external_name: external_name.to_owned(),
        dimensions: tensor.shape().to_vec(),
        dtype,
    })
}

fn resolved_weight_keys<'a>(
    resolved: &'a [Qwen35ResolvedWeightSpec],
    layer_index: Option<u32>,
    tied_embeddings: bool,
) -> impl Iterator<Item = (Option<u32>, Option<u32>, String, String)> + 'a {
    resolved
        .iter()
        .filter(move |weight| weight.present && !(tied_embeddings && weight.role == "lm_head"))
        .map(move |weight| {
            (
                layer_index,
                weight.expert_index,
                weight.role.clone(),
                weight.name.clone(),
            )
        })
}

fn resolved_weight_keys_from_config(
    config: &Qwen35FamilyConfig,
) -> BTreeSet<(Option<u32>, Option<u32>, String, String)> {
    config
        .weights
        .iter()
        .map(|weight| {
            (
                weight.layer_index,
                weight.expert_index,
                weight.role.clone(),
                weight.external_name.clone(),
            )
        })
        .collect()
}

fn validate_gptq_marlin_config(
    quantization: &Qwen35QuantizationConfig,
) -> Result<&Qwen35GptqQuantizationRecipe, VNextError> {
    let Qwen35QuantizationConfig::Gptq(recipe) = quantization else {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "typed Marlin requires GPTQ INT4, power-of-two group_size, sym=true, and desc_act=false",
        ));
    };
    if recipe.bits != 4
        || recipe.group_size == 0
        || !recipe.group_size.is_power_of_two()
        || recipe.desc_act
        || !recipe.sym
    {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "typed Marlin requires GPTQ INT4, power-of-two group_size, sym=true, and desc_act=false",
        ));
    }
    Ok(recipe)
}

fn validate_safetensors_quantization_config(
    quantization: &Qwen35QuantizationConfig,
) -> Result<(), VNextError> {
    match quantization {
        Qwen35QuantizationConfig::Gptq(_) => validate_gptq_marlin_config(quantization).map(|_| ()),
        Qwen35QuantizationConfig::CompressedTensors(_) => {
            validate_compressed_tensors_marlin_config(quantization).map(|_| ())
        }
        Qwen35QuantizationConfig::Fp8(_) => Ok(()),
    }
}

fn validate_block_fp8_config(
    quantization: &Qwen35QuantizationConfig,
) -> Result<&Qwen35Fp8QuantizationRecipe, VNextError> {
    let Qwen35QuantizationConfig::Fp8(recipe) = quantization else {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "typed block-FP8 requires official E4M3 dynamic-activation metadata with a 128x128 weight grid",
        ));
    };
    if recipe.weight_block_size.as_array() != [128, 128] || recipe.modules_to_not_convert.is_empty()
    {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "typed block-FP8 requires official E4M3 dynamic-activation metadata with a 128x128 weight grid and explicit dense exclusions",
        ));
    }
    Ok(recipe)
}

fn block_fp8_source_quantization_spec(
    quantization: &Qwen35QuantizationConfig,
) -> Result<QuantizationSpec, VNextError> {
    let recipe = validate_block_fp8_config(quantization)?;
    let [output_features, input_features] = recipe.weight_block_size.as_array();
    let output_features = u32::try_from(output_features)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            invalid_config(
                "hf_config.quantization_config.weight_block_size",
                "FP8 output block size is zero or exceeds u32",
            )
        })?;
    let input_features = u32::try_from(input_features)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            invalid_config(
                "hf_config.quantization_config.weight_block_size",
                "FP8 input block size is zero or exceeds u32",
            )
        })?;
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(BLOCK_FP8_E4M3_SOURCE_FORMAT_ID)?,
        bits_per_weight: 8,
        grouping: QuantizationGrouping::block_2d([output_features, input_features]),
        packing: QuantizationPacking::Linear,
        scale_type: ElementType::Bf16,
        zero_point_type: None,
    })
}

fn validate_block_fp8_source_completeness(
    config: &Qwen35FamilyConfig,
    recipe: &Qwen35Fp8QuantizationRecipe,
) -> Result<(), VNextError> {
    let exclusions = recipe
        .modules_to_not_convert
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut execution_pair_count = 0_usize;

    for weight in &config.weights {
        let is_eligible_projection =
            BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES.contains(&weight.role.as_str());
        let source_name = match &weight.source_encoding {
            FamilyWeightSourceEncoding::Dense { .. } => weight.external_name.as_str(),
            FamilyWeightSourceEncoding::BlockFp8 { values, .. } => values.external_name.as_str(),
            FamilyWeightSourceEncoding::Gptq { .. }
            | FamilyWeightSourceEncoding::CompressedTensors { .. }
            | FamilyWeightSourceEncoding::BlockQuantized(_) => continue,
        };

        if !is_eligible_projection {
            if matches!(
                weight.source_encoding,
                FamilyWeightSourceEncoding::BlockFp8 { .. }
            ) {
                return Err(invalid_config(
                    "weights.source_encoding",
                    format!(
                        "non-projection role {:?} must not carry a block-FP8 value/inverse-scale pair",
                        weight.role
                    ),
                ));
            }
            continue;
        }

        let module = source_name.strip_suffix(".weight").ok_or_else(|| {
            invalid_config(
                "weights.source_encoding",
                format!(
                    "execution-eligible projection role {:?} source {source_name:?} does not identify a .weight tensor",
                    weight.role
                ),
            )
        })?;
        let is_typed_dense_exclusion = exclusions.contains(module);
        match (&weight.source_encoding, is_typed_dense_exclusion) {
            (FamilyWeightSourceEncoding::BlockFp8 { .. }, false) => {
                execution_pair_count = execution_pair_count.checked_add(1).ok_or_else(|| {
                    invalid_config(
                        "weights.source_encoding",
                        "block-FP8 execution pair count overflows usize",
                    )
                })?;
            }
            (FamilyWeightSourceEncoding::Dense { .. }, true) => {}
            (FamilyWeightSourceEncoding::Dense { .. }, false) => {
                return Err(invalid_config(
                    "weights.source_encoding",
                    format!(
                        "execution-eligible projection {module:?} is not listed in modules_to_not_convert and must provide a complete E4M3 .weight plus BF16 .weight_scale_inv pair"
                    ),
                ));
            }
            (FamilyWeightSourceEncoding::BlockFp8 { .. }, true) => {
                return Err(invalid_config(
                    "weights.source_encoding",
                    format!(
                        "typed dense exclusion {module:?} must not provide a block-FP8 value/inverse-scale pair"
                    ),
                ));
            }
            _ => {
                return Err(invalid_config(
                    "weights.source_encoding",
                    format!(
                        "execution-eligible projection {module:?} has an unsupported source bundle"
                    ),
                ));
            }
        }
    }

    if execution_pair_count == 0 {
        return Err(invalid_config(
            "weights.source_encoding",
            "block-FP8 package contains no execution-eligible E4M3 value/BF16 inverse-scale pair",
        ));
    }
    Ok(())
}

fn validate_block_fp8_weight_source(
    weight: &FamilyWeight,
    values: &FamilyBlockFp8Tensor,
    scale_inv: &FamilyBlockFp8Tensor,
    quantization: &Qwen35QuantizationConfig,
) -> Result<(), VNextError> {
    let recipe = validate_block_fp8_config(quantization)?;
    let stem = values
        .external_name
        .strip_suffix(".weight")
        .unwrap_or_default();
    if weight.external_name != values.external_name
        || stem.is_empty()
        || scale_inv.external_name != format!("{stem}.weight_scale_inv")
        || values.dtype != FamilyBlockFp8Dtype::F8E4m3
        || scale_inv.dtype != FamilyBlockFp8Dtype::Bf16
        || weight.dimensions != values.dimensions
    {
        return Err(invalid_config(
            "weights.source_encoding",
            format!(
                "role {:?} has an invalid block-FP8 value/inverse-scale identity or dtype",
                weight.role
            ),
        ));
    }
    let [n, k] = weight.dimensions.as_slice() else {
        return Err(invalid_config(
            "weights.dimensions",
            "block-FP8 logical weight must have shape [N, K]",
        ));
    };
    let [output_block, input_block] = recipe.weight_block_size.as_array();
    let output_block = output_block as u64;
    let input_block = input_block as u64;
    let expected_scale_dimensions = [n.div_ceil(output_block), k.div_ceil(input_block)];
    if scale_inv.dimensions != expected_scale_dimensions {
        return Err(invalid_config(
            "weights.source_encoding.scale_inv.dimensions",
            format!(
                "role {:?} inverse-scale shape {:?} differs from the block grid {expected_scale_dimensions:?}",
                weight.role, scale_inv.dimensions
            ),
        ));
    }
    Ok(())
}

fn validate_compressed_tensors_marlin_config(
    quantization: &Qwen35QuantizationConfig,
) -> Result<&Qwen35CompressedTensorsQuantizationRecipe, VNextError> {
    let Qwen35QuantizationConfig::CompressedTensors(recipe) = quantization else {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "typed compressed-tensors Marlin requires pack-quantized INT4, group_size=32, asymmetric static Linear weights, and no activation quantization",
        ));
    };
    if recipe.format != "pack-quantized"
        || recipe.bits != 4
        || recipe.group_size != 32
        || recipe.desc_act
        || recipe.sym
        || recipe.weight_type != "int"
        || recipe.strategy != "group"
        || recipe.dynamic != Some(false)
        || recipe.targets != ["Linear"]
        || recipe.input_activations
        || recipe.output_activations
    {
        return Err(invalid_config(
            "hf_config.quantization_config",
            "typed compressed-tensors Marlin requires pack-quantized INT4, group_size=32, asymmetric static Linear weights, and no activation quantization",
        ));
    }
    Ok(recipe)
}

fn validate_compressed_tensors_weight_source(
    weight: &FamilyWeight,
    weight_packed: &FamilyCompressedTensorsTensor,
    weight_scale: &FamilyCompressedTensorsTensor,
    weight_zero_point: &FamilyCompressedTensorsTensor,
    weight_shape: &FamilyCompressedTensorsTensor,
    quantization: &Qwen35QuantizationConfig,
) -> Result<(), VNextError> {
    let recipe = validate_compressed_tensors_marlin_config(quantization)?;
    let stem = weight_packed
        .external_name
        .strip_suffix(".weight_packed")
        .unwrap_or_default();
    if weight.external_name != weight_packed.external_name
        || stem.is_empty()
        || weight_scale.external_name != format!("{stem}.weight_scale")
        || weight_zero_point.external_name != format!("{stem}.weight_zero_point")
        || weight_shape.external_name != format!("{stem}.weight_shape")
        || weight_packed.dtype != FamilyCompressedTensorsDtype::I32
        || !matches!(
            weight_scale.dtype,
            FamilyCompressedTensorsDtype::F16 | FamilyCompressedTensorsDtype::Bf16
        )
        || weight_zero_point.dtype != FamilyCompressedTensorsDtype::I32
        || weight_shape.dtype != FamilyCompressedTensorsDtype::I64
        || weight_shape.dimensions != [2]
    {
        return Err(invalid_config(
            "weights.source_encoding",
            format!(
                "role {:?} has an invalid compressed-tensors sidecar identity or dtype",
                weight.role
            ),
        ));
    }
    let [n, k] = weight.dimensions.as_slice() else {
        return Err(invalid_config(
            "weights.dimensions",
            "compressed-tensors logical weight must have shape [N, K]",
        ));
    };
    let group_size = recipe.group_size as u64;
    if *n % 64 != 0
        || *k % 16 != 0
        || !k.is_multiple_of(group_size)
        || weight_packed.dimensions != [*n, *k / 8]
        || weight_scale.dimensions != [*n, *k / group_size]
        || weight_zero_point.dimensions != [*n / 8, *k / group_size]
    {
        return Err(invalid_config(
            "weights.source_encoding",
            format!(
                "role {:?} compressed-tensors headers do not form a group32 Marlin-aligned [N, K] matrix",
                weight.role
            ),
        ));
    }
    Ok(())
}

fn validate_gptq_weight_source(
    weight: &FamilyWeight,
    qweight: &FamilyGptqTensor,
    scales: &FamilyGptqTensor,
    qzeros: &FamilyGptqTensor,
    g_idx: Option<&FamilyGptqTensor>,
    quantization: &Qwen35QuantizationConfig,
) -> Result<(), VNextError> {
    let recipe = validate_gptq_marlin_config(quantization)?;
    let qweight_stem = qweight
        .external_name
        .strip_suffix(".qweight")
        .unwrap_or_default();
    if weight.external_name != qweight.external_name
        || qweight_stem.is_empty()
        || scales.external_name != format!("{qweight_stem}.scales")
        || qzeros.external_name != format!("{qweight_stem}.qzeros")
        || g_idx.is_some_and(|source| source.external_name != format!("{qweight_stem}.g_idx"))
        || qweight.element_type != ElementType::I32
        || qzeros.element_type != ElementType::I32
        || !matches!(
            scales.element_type,
            ElementType::F16 | ElementType::Bf16 | ElementType::F32
        )
    {
        return Err(invalid_config(
            "weights.source_encoding",
            format!(
                "role {:?} has an invalid GPTQ sidecar identity or dtype",
                weight.role
            ),
        ));
    }
    let [packed_k, n] = qweight.dimensions.as_slice() else {
        return Err(invalid_config(
            "weights.source_encoding.qweight.dimensions",
            "GPTQ qweight must have shape [K/8, N]",
        ));
    };
    let k = packed_k
        .checked_mul(8)
        .ok_or_else(|| invalid_config("weights.dimensions", "GPTQ K dimension overflows"))?;
    let group_size = recipe.group_size as u64;
    if k % 16 != 0
        || *n % 16 != 0
        || !k.is_multiple_of(group_size)
        || weight.dimensions != [*n, k]
        || scales.dimensions != [k / group_size, *n]
        || qzeros.dimensions != [k / group_size, *n / 8]
        || !n.is_multiple_of(8)
    {
        return Err(invalid_config(
            "weights.source_encoding",
            format!(
                "role {:?} GPTQ headers do not form a Marlin-aligned [N, K] matrix",
                weight.role
            ),
        ));
    }
    if let Some(g_idx) = g_idx {
        if !g_idx.external_name.ends_with(".g_idx")
            || g_idx.element_type != ElementType::I32
            || g_idx.dimensions != [k]
        {
            return Err(invalid_config(
                "weights.source_encoding.g_idx",
                format!("role {:?} has an invalid GPTQ g_idx header", weight.role),
            ));
        }
    }
    Ok(())
}

fn family_source_external_names(config: &Qwen35FamilyConfig) -> Vec<String> {
    config
        .weights
        .iter()
        .flat_map(|weight| match &weight.source_encoding {
            FamilyWeightSourceEncoding::Gptq {
                qweight,
                scales,
                qzeros,
                g_idx,
            } => [qweight, scales, qzeros]
                .into_iter()
                .chain(g_idx.iter())
                .map(|source| source.external_name.clone())
                .collect::<Vec<_>>(),
            FamilyWeightSourceEncoding::CompressedTensors {
                weight_packed,
                weight_scale,
                weight_zero_point,
                weight_shape,
            } => [weight_packed, weight_scale, weight_zero_point, weight_shape]
                .into_iter()
                .map(|source| source.external_name.clone())
                .collect(),
            FamilyWeightSourceEncoding::BlockFp8 { values, scale_inv } => {
                vec![
                    values.external_name.clone(),
                    scale_inv.external_name.clone(),
                ]
            }
            FamilyWeightSourceEncoding::Dense { .. }
            | FamilyWeightSourceEncoding::BlockQuantized(_) => {
                vec![weight.external_name.clone()]
            }
        })
        .collect()
}

fn validate_safetensors_manifest(
    text: &Qwen35TextConfig,
    config: &Qwen35FamilyConfig,
    label: &str,
) -> Result<(), VNextError> {
    let inventory = Qwen35WeightInventory::from_names(family_source_external_names(config));
    let resolved = inventory
        .detect_prefix_and_resolve(text)
        .map_err(|reason| invalid_config("weights", reason))?;
    let expected = resolved_weight_keys(&resolved.global_tensors, None, text.tie_word_embeddings)
        .chain(resolved.layers.iter().flat_map(|layer| {
            resolved_weight_keys(
                &layer.tensors,
                Some(layer.layer_index as u32),
                text.tie_word_embeddings,
            )
        }))
        .collect::<BTreeSet<_>>();
    let actual = resolved_weight_keys_from_config(config);
    if actual != expected {
        return Err(invalid_config(
            "weights",
            format!("resolved tensors do not exactly match the supported {label} Qwen3.5 manifest"),
        ));
    }
    Ok(())
}

fn validate_canonical_gptq_moe_representation(
    text: &Qwen35TextConfig,
    config: &Qwen35FamilyConfig,
) -> Result<(), VNextError> {
    let Some(moe) = text.moe.as_ref() else {
        return Ok(());
    };
    const CANONICAL_ROUTED_ROLES: [&str; 3] = [
        "moe_per_expert_gate_proj_qweight",
        "moe_per_expert_up_proj_qweight",
        "moe_per_expert_down_proj_qweight",
    ];
    const ALTERNATE_ROUTED_ROLES: [&str; 8] = [
        "moe_stacked_gate_proj",
        "moe_stacked_up_proj",
        "moe_stacked_down_proj",
        "moe_fused_gate_up_proj",
        "moe_fused_down_proj",
        "moe_per_expert_gate_proj",
        "moe_per_expert_up_proj",
        "moe_per_expert_down_proj",
    ];

    for layer_index in 0..text.num_hidden_layers {
        let layer_index = layer_index as u32;
        if let Some(weight) = config.weights.iter().find(|weight| {
            weight.layer_index == Some(layer_index)
                && ALTERNATE_ROUTED_ROLES.contains(&weight.role.as_str())
        }) {
            return Err(invalid_config(
                "weights",
                format!(
                    "Qwen3.5 GPTQ MoE layer {layer_index} mixes canonical per-expert qweight sources with alternate routed role {:?}",
                    weight.role
                ),
            ));
        }
        for role in CANONICAL_ROUTED_ROLES {
            let weights = required_expert_weights(config, layer_index, role, moe.num_experts)?;
            if weights.iter().any(|weight| {
                !matches!(
                    weight.source_encoding,
                    FamilyWeightSourceEncoding::Gptq { .. }
                )
            }) {
                return Err(invalid_config(
                    "weights.source_encoding",
                    format!(
                        "Qwen3.5 GPTQ MoE layer {layer_index} role {role:?} must use typed GPTQ sources for every expert"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_gguf_manifest(
    text: &Qwen35TextConfig,
    config: &Qwen35FamilyConfig,
) -> Result<(), VNextError> {
    let architecture = gguf_architecture(text);
    let manifest = text
        .weight_manifest("model.language_model")
        .map_err(|reason| invalid_config("weights", reason))?;
    let mut allowed = BTreeSet::new();
    let mut required = BTreeSet::new();
    for (layer_index, spec) in manifest
        .global_tensors
        .iter()
        .map(|spec| (None, spec))
        .chain(manifest.layers.iter().flat_map(|layer| {
            layer
                .tensors
                .iter()
                .map(move |spec| (Some(layer.layer_index as u32), spec))
        }))
        .filter(|(_, spec)| !(text.tie_word_embeddings && spec.role == "lm_head"))
    {
        let Some(external_name) = ferrum_to_gguf_with_arch(architecture, &spec.name) else {
            if spec.required {
                return Err(invalid_config(
                    "weights",
                    format!(
                        "GGUF has no typed name mapping for required role {:?} source {:?}",
                        spec.role, spec.name
                    ),
                ));
            }
            continue;
        };
        let key = (layer_index, None, spec.role.clone(), external_name);
        if spec.required {
            required.insert(key.clone());
        }
        allowed.insert(key);
    }
    let actual = resolved_weight_keys_from_config(config);
    if !required.is_subset(&actual) || !actual.is_subset(&allowed) {
        return Err(invalid_config(
            "weights",
            "resolved GGUF tensors do not exactly match the supported Qwen3.5 manifest",
        ));
    }
    if text.is_moe() {
        for layer_index in 0..text.num_hidden_layers {
            for role in [
                MOE_ROUTER_ROLE,
                "moe_stacked_gate_proj",
                "moe_stacked_up_proj",
                "moe_stacked_down_proj",
                "moe_shared_expert_gate",
                "moe_shared_expert_gate_proj",
                "moe_shared_expert_up_proj",
                "moe_shared_expert_down_proj",
            ] {
                required_weight(config, Some(layer_index as u32), role).map_err(|_| {
                    invalid_config(
                        "weights",
                        format!(
                            "Qwen3.5 MoE GGUF layer {layer_index} lacks canonical source role {role:?}"
                        ),
                    )
                })?;
            }
        }
    }
    Ok(())
}

fn gguf_architecture(text: &Qwen35TextConfig) -> &'static str {
    if text.is_moe() {
        "qwen35moe"
    } else {
        "qwen35"
    }
}

fn layer_weights(
    config: &Qwen35FamilyConfig,
    layer_index: u32,
    mlp: bool,
) -> impl Iterator<Item = &FamilyWeight> {
    config.weights.iter().filter(move |weight| {
        weight.layer_index == Some(layer_index)
            && (weight.role == "post_attention_layernorm" || weight.role.starts_with("mlp_")) == mlp
    })
}

fn required_weight<'a>(
    config: &'a Qwen35FamilyConfig,
    layer_index: Option<u32>,
    role: &str,
) -> Result<&'a FamilyWeight, VNextError> {
    config
        .weights
        .iter()
        .find(|weight| weight.layer_index == layer_index && weight.role == role)
        .ok_or_else(|| invalid_config("weights", format!("missing role {role:?}")))
}

fn required_expert_weights<'a>(
    config: &'a Qwen35FamilyConfig,
    layer_index: u32,
    role: &str,
    expert_count: usize,
) -> Result<Vec<&'a FamilyWeight>, VNextError> {
    let mut weights = config
        .weights
        .iter()
        .filter(|weight| weight.layer_index == Some(layer_index) && weight.role == role)
        .collect::<Vec<_>>();
    weights.sort_by_key(|weight| weight.expert_index);
    if weights.len() != expert_count
        || weights
            .iter()
            .enumerate()
            .any(|(expert, weight)| weight.expert_index != u32::try_from(expert).ok())
    {
        return Err(invalid_config(
            "weights",
            format!(
                "layer {layer_index} role {role:?} must contain exactly experts 0..{expert_count} in numeric order"
            ),
        ));
    }
    Ok(weights)
}

fn is_moe_source_role(role: &str) -> bool {
    role.starts_with("moe_")
}

fn moe_logical_dimensions(text: &Qwen35TextConfig, role: &str) -> Result<Vec<u64>, VNextError> {
    let moe = text.moe.as_ref().ok_or_else(|| {
        invalid_config(
            "hf_config.text_config.model_type",
            "MoE logical weight requested for a dense configuration",
        )
    })?;
    let hidden = text.hidden_size as u64;
    let experts = moe.num_experts as u64;
    let routed = moe.moe_intermediate_size as u64;
    let shared = moe.shared_expert_intermediate_size as u64;
    match role {
        MOE_ROUTER_ROLE => Ok(vec![experts, hidden]),
        MOE_ROUTED_GATE_UP_ROLE => Ok(vec![experts, 2, routed, hidden]),
        MOE_ROUTED_DOWN_ROLE => Ok(vec![experts, hidden, routed]),
        MOE_SHARED_GATE_ROLE => Ok(vec![1, hidden]),
        MOE_SHARED_GATE_UP_ROLE => Ok(vec![2, shared, hidden]),
        MOE_SHARED_DOWN_ROLE => Ok(vec![hidden, shared]),
        _ => Err(invalid_config(
            "weights.role",
            format!("unknown Qwen3.5 MoE logical weight role {role:?}"),
        )),
    }
}

fn moe_weight_references(
    text: &Qwen35TextConfig,
    layer_index: u32,
) -> Result<Vec<WeightReference>, VNextError> {
    [
        MOE_ROUTER_ROLE,
        MOE_ROUTED_GATE_UP_ROLE,
        MOE_ROUTED_DOWN_ROLE,
        MOE_SHARED_GATE_ROLE,
        MOE_SHARED_GATE_UP_ROLE,
        MOE_SHARED_DOWN_ROLE,
    ]
    .into_iter()
    .map(|role| {
        Ok(WeightReference {
            weight_id: moe_weight_id(layer_index, role)?,
            value_id: moe_weight_value_id(layer_index, role)?,
            tensor: tensor_spec(
                moe_logical_dimensions(text, role)?,
                DENSE_MATERIALIZED_ELEMENT_TYPE,
            ),
        })
    })
    .collect()
}

fn moe_weight_id(layer_index: u32, role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(scoped_weight_key(Some(layer_index), role, "weight"))
}

fn moe_component_id(layer_index: u32, role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(scoped_weight_key(Some(layer_index), role, "component"))
}

fn moe_weight_value_id(layer_index: u32, role: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(scoped_weight_key(Some(layer_index), role, "value.weight"))
}

fn materialized_element_type(role: &str) -> ElementType {
    match role {
        "linear_attn_a_log" | "linear_attn_dt_bias" | "linear_attn_norm" => ElementType::F32,
        _ => DENSE_MATERIALIZED_ELEMENT_TYPE,
    }
}

fn logical_weight_dimensions(weight: &FamilyWeight) -> Result<Vec<u64>, VNextError> {
    if weight.role != "linear_attn_conv" {
        return Ok(weight.dimensions.clone());
    }
    match weight.dimensions.as_slice() {
        [channels, kernel] => Ok(vec![*channels, *kernel]),
        [channels, 1, kernel] => Ok(vec![*channels, *kernel]),
        dimensions => Err(invalid_config(
            "weights.dimensions",
            format!(
                "linear attention convolution weight must be [channels, kernel] or [channels, 1, kernel], got {dimensions:?}"
            ),
        )),
    }
}

fn dense_weight_encoding(role: &str) -> Result<WeightEncoding, VNextError> {
    let element_type = materialized_element_type(role);
    if matches!(
        role,
        "final_norm"
            | "input_layernorm"
            | "post_attention_layernorm"
            | "self_attn_q_norm"
            | "self_attn_k_norm"
    ) {
        return Ok(WeightEncoding::DenseAffine {
            element_type,
            scale: CanonicalRational::new(1, 1)?,
            bias: CanonicalRational::new(1, 1)?,
        });
    }
    Ok(WeightEncoding::Dense { element_type })
}

fn expected_weight_dimensions(
    text: &Qwen35TextConfig,
    vocab_size: u64,
    weight: &FamilyWeight,
) -> Result<Vec<Vec<u64>>, VNextError> {
    let hidden = text.hidden_size as u64;
    let intermediate = text.dense_intermediate_size.unwrap_or_default() as u64;
    let key_total = text.linear_qk_total_dim() as u64;
    let value_total = text.linear_value_total_dim() as u64;
    let value_heads = text.linear_attention.num_value_heads as u64;
    let conv_channels = key_total
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_total))
        .ok_or_else(|| invalid_config("weights.dimensions", "linear dimensions overflow"))?;
    let full_query = text.full_attention_q_proj_total_dim() as u64;
    let full_query_without_gate = text.full_attention_query_total_dim() as u64;
    let full_kv = text.full_attention_kv_total_dim() as u64;
    let moe = text.moe.as_ref();
    let experts = moe.map_or(0, |config| config.num_experts as u64);
    let routed = moe.map_or(0, |config| config.moe_intermediate_size as u64);
    let shared = moe.map_or(0, |config| config.shared_expert_intermediate_size as u64);
    let expected = match weight.role.as_str() {
        "embed_tokens" | "lm_head" => vec![vec![vocab_size, hidden]],
        "final_norm" | "input_layernorm" | "post_attention_layernorm" => {
            vec![vec![hidden]]
        }
        "linear_attn_qkv" => vec![vec![conv_channels, hidden]],
        "linear_attn_z" => vec![vec![value_total, hidden]],
        "linear_attn_a" | "linear_attn_b" => vec![vec![value_heads, hidden]],
        "linear_attn_conv" => {
            let kernel = text.linear_attention.conv_kernel_dim as u64;
            vec![vec![conv_channels, kernel], vec![conv_channels, 1, kernel]]
        }
        "linear_attn_a_log" | "linear_attn_dt_bias" => vec![vec![value_heads]],
        "linear_attn_norm" => vec![vec![text.linear_attention.value_head_dim as u64]],
        "linear_attn_out" => vec![vec![hidden, value_total]],
        "self_attn_q" => vec![vec![full_query, hidden]],
        "self_attn_k" | "self_attn_v" => vec![vec![full_kv, hidden]],
        "self_attn_o" => vec![vec![hidden, full_query_without_gate]],
        "self_attn_q_norm" | "self_attn_k_norm" => vec![vec![text.head_dim as u64]],
        "mlp_gate" | "mlp_up" => vec![vec![intermediate, hidden]],
        "mlp_down" => vec![vec![hidden, intermediate]],
        MOE_ROUTER_ROLE => vec![vec![experts, hidden]],
        "moe_stacked_gate_proj" | "moe_stacked_up_proj" => {
            vec![vec![experts, routed, hidden]]
        }
        "moe_stacked_down_proj" => vec![vec![experts, hidden, routed]],
        "moe_per_expert_gate_proj"
        | "moe_per_expert_up_proj"
        | "moe_per_expert_gate_proj_qweight"
        | "moe_per_expert_up_proj_qweight" => vec![vec![routed, hidden]],
        "moe_per_expert_down_proj" | "moe_per_expert_down_proj_qweight" => {
            vec![vec![hidden, routed]]
        }
        // HF stores the scalar shared-expert gate as `nn.Linear(H, 1)` while
        // GGUF may squeeze the leading singleton axis. The logical schema is
        // always `[1, H]`; `dense_or_reshaped_layout` handles the GGUF form.
        "moe_shared_expert_gate" => vec![vec![1, hidden], vec![hidden]],
        "moe_shared_expert_gate_proj" | "moe_shared_expert_up_proj" => {
            vec![vec![shared, hidden]]
        }
        "moe_shared_expert_down_proj" => vec![vec![hidden, shared]],
        role => {
            return Err(invalid_config(
                "weights.role",
                format!("unsupported dense Qwen3.5 weight role {role:?}"),
            ));
        }
    };
    Ok(expected)
}

fn required_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{key} must be a positive integer"))
}

fn hf_rms_norm_epsilon(hf_config: &Value) -> Result<CanonicalRational, String> {
    let text = hf_config.get("text_config").unwrap_or(hf_config);
    let Some(value) = text.get("rms_norm_eps") else {
        return CanonicalRational::new(1, 1_000_000).map_err(|error| error.to_string());
    };
    let Value::Number(number) = value else {
        return Err("rms_norm_eps must be a JSON number".to_owned());
    };
    let epsilon = parse_positive_decimal_rational(&number.to_string())?;
    if epsilon.numerator() as u64 > epsilon.denominator() {
        return Err("rms_norm_eps must not exceed one".to_owned());
    }
    Ok(epsilon)
}

fn parse_positive_decimal_rational(raw: &str) -> Result<CanonicalRational, String> {
    let value = CanonicalRational::from_decimal_str(raw).map_err(|error| error.to_string())?;
    if value.numerator() <= 0 {
        return Err("decimal rational must be positive".to_owned());
    }
    if value.numerator() as u64 > value.denominator() {
        return Err("rms_norm_eps must not exceed one".to_owned());
    }
    Ok(value)
}

fn canonical_positive_f64(value: f64) -> Result<CanonicalRational, VNextError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid_config(
            "rope_theta",
            "rope theta must be finite and positive",
        ));
    }
    CanonicalRational::from_decimal_str(&value.to_string())
        .map_err(|reason| invalid_config("rope_theta", reason.to_string()))
}

fn weight_key(weight: &FamilyWeight, prefix: &str) -> String {
    let key = scoped_weight_key(weight.layer_index, &weight.role, prefix);
    match weight.expert_index {
        Some(expert) => format!("{key}.expert.{expert}"),
        None => key,
    }
}

fn scoped_weight_key(layer_index: Option<u32>, role: &str, prefix: &str) -> String {
    match layer_index {
        Some(layer) => format!("{prefix}.layer.{layer}.{role}"),
        None => format!("{prefix}.global.{role}"),
    }
}

fn packed_gate_up_dimensions(
    gate: &FamilyWeight,
    up: &FamilyWeight,
) -> Result<Vec<u64>, VNextError> {
    if gate.role != "mlp_gate"
        || up.role != "mlp_up"
        || gate.layer_index != up.layer_index
        || gate.dimensions != up.dimensions
        || gate.dimensions.len() != 2
    {
        return Err(invalid_config(
            "weights.mlp_gate_up",
            "gate/up sources must be same-layer matrices with identical shapes",
        ));
    }
    Ok(vec![2, gate.dimensions[0], gate.dimensions[1]])
}

fn packed_gate_up_weight_id(layer_index: u32) -> Result<WeightId, VNextError> {
    WeightId::new(scoped_weight_key(
        Some(layer_index),
        PACKED_GATE_UP_ROLE,
        "weight",
    ))
}

fn packed_gate_up_component_id(layer_index: u32) -> Result<WeightId, VNextError> {
    WeightId::new(scoped_weight_key(
        Some(layer_index),
        PACKED_GATE_UP_ROLE,
        "component",
    ))
}

fn packed_gate_up_value_id(layer_index: u32) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(scoped_weight_key(
        Some(layer_index),
        PACKED_GATE_UP_ROLE,
        "value.weight",
    ))
}

fn packed_linear_attention_dimensions(sources: [&FamilyWeight; 4]) -> Result<Vec<u64>, VNextError> {
    let expected_roles = [
        "linear_attn_qkv",
        "linear_attn_z",
        "linear_attn_b",
        "linear_attn_a",
    ];
    let layer_index = sources[0].layer_index;
    let mut hidden_size = None;
    let mut output_features = 0_u64;
    for (source, expected_role) in sources.into_iter().zip(expected_roles) {
        let [rows, hidden] = source.dimensions.as_slice() else {
            return Err(invalid_config(
                "weights.linear_attn_projection",
                "linear-attention projection sources must be matrices",
            ));
        };
        if source.role != expected_role
            || source.layer_index != layer_index
            || source.expert_index.is_some()
            || hidden_size.is_some_and(|expected| expected != *hidden)
        {
            return Err(invalid_config(
                "weights.linear_attn_projection",
                "linear-attention projection sources have incompatible roles, layers, or input widths",
            ));
        }
        hidden_size = Some(*hidden);
        output_features = output_features.checked_add(*rows).ok_or_else(|| {
            invalid_config(
                "weights.linear_attn_projection",
                "packed projection output width overflows",
            )
        })?;
    }
    Ok(vec![
        output_features,
        hidden_size.expect("four validated projection sources have a hidden width"),
    ])
}

fn packed_linear_attention_weight_id(layer_index: u32, role: &str) -> Result<WeightId, VNextError> {
    WeightId::new(scoped_weight_key(Some(layer_index), role, "weight"))
}

fn packed_linear_attention_component_id(
    layer_index: u32,
    role: &str,
) -> Result<WeightId, VNextError> {
    WeightId::new(scoped_weight_key(Some(layer_index), role, "component"))
}

fn packed_linear_attention_value_id(
    layer_index: u32,
    role: &str,
) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(scoped_weight_key(Some(layer_index), role, "value.weight"))
}

fn weight_id(weight: &FamilyWeight) -> Result<WeightId, VNextError> {
    WeightId::new(weight_key(weight, "weight"))
}

fn component_id(weight: &FamilyWeight) -> Result<WeightId, VNextError> {
    WeightId::new(weight_key(weight, "component"))
}

fn weight_value_id(weight: &FamilyWeight) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(weight_key(weight, "value.weight"))
}

fn value_id(value: impl Into<String>) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(value)
}

fn node_id(value: impl Into<String>) -> Result<NodeId, VNextError> {
    NodeId::new(value)
}

fn operation_id(value: impl Into<String>) -> Result<OperationId, VNextError> {
    OperationId::new(value)
}

fn state_id(value: impl Into<String>) -> Result<StateId, VNextError> {
    StateId::new(value)
}

fn tensor_spec(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

trait IntoSemanticValue {
    fn into_semantic_value(self) -> SemanticValue;
}

impl IntoSemanticValue for u64 {
    fn into_semantic_value(self) -> SemanticValue {
        SemanticValue::Unsigned(self)
    }
}

impl IntoSemanticValue for bool {
    fn into_semantic_value(self) -> SemanticValue {
        SemanticValue::Bool(self)
    }
}

impl IntoSemanticValue for CanonicalRational {
    fn into_semantic_value(self) -> SemanticValue {
        SemanticValue::Rational(self)
    }
}

impl IntoSemanticValue for GatedDeltaDecayParameterization {
    fn into_semantic_value(self) -> SemanticValue {
        SemanticValue::Text(self.as_str().to_owned())
    }
}

impl IntoSemanticValue for GatedDeltaValueHeadMapping {
    fn into_semantic_value(self) -> SemanticValue {
        SemanticValue::Text(self.as_str().to_owned())
    }
}

fn attribute(
    name: &str,
    value: impl IntoSemanticValue,
) -> Result<(AttributeId, SemanticValue), VNextError> {
    Ok((AttributeId::new(name)?, value.into_semantic_value()))
}

fn invalid_config(field: impl Into<String>, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: FAMILY_ID.to_owned(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Write};
    use std::ops::Range;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use ferrum_interfaces::vnext::{
        causal_paged_attention_contract, dense_swiglu_contract,
        gated_delta_recurrent_attention_contract, last_token_dense_linear_contract,
        routed_shared_swiglu_moe_contract, DeviceClass, DeviceDescriptor, DeviceId,
        DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView,
        ModelArtifactSourceRole, ModelSourceKind, OperationContract, OriginalModelSource,
        OriginalModelSources, PhysicalStorageLayout, PhysicalWeightComponentBinding,
        WeightComponentSource, WeightMaterializationFidelity,
    };
    use ferrum_kernels::marlin_fp8_materializer::{
        block_fp8_to_marlin_fp8_weight_materializer, MARLIN_FP8_QUANTIZATION_FORMAT_ID,
    };
    use ferrum_quantization::SafetensorsTensor;
    use half::{bf16, f16};
    use memmap2::Mmap;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView, View};
    use safetensors::SafeTensors;

    const QWEN38_AWQ_INT4_CONFIG: &[u8] =
        include_bytes!("../../tests/fixtures/qwen38_awq_int4_config.contract.json");
    const QWEN38_FP8_CONFIG: &[u8] =
        include_bytes!("../../tests/fixtures/qwen38_fp8_config.contract.json");
    const QWEN38_FP8_BAD_RECIPE: &[u8] =
        include_bytes!("../../tests/fixtures/qwen38_fp8_config.bad-recipe.json");
    const QWEN35_08B_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
    const QWEN35_08B_CACHE_REPO_DIR: &str = "models--Qwen--Qwen3.5-0.8B";
    const QWEN35_08B_SOURCE_TENSOR_COUNT: usize = 488;
    const QWEN35_08B_SOURCE_PAYLOAD_BYTES: u64 = 1_746_882_752;
    const QWEN35_08B_DERIVED_FP8_PAIR_COUNT: usize = 150;
    const QWEN35_08B_TYPED_DENSE_EXCLUSION_COUNT: usize = 37;
    const QWEN35_08B_A3_MOE_EXPERT_COUNT: usize = 2;
    const QWEN35_08B_A3_MOE_EXPERTS_PER_TOKEN: usize = 1;
    const QWEN35_08B_A3_MOE_DERIVED_FP8_PAIR_COUNT: usize =
        QWEN35_08B_DERIVED_FP8_PAIR_COUNT - 24 * 3 + 24 * 9;
    const QWEN35_08B_A3_MOE_DERIVED_TENSOR_COUNT: usize = QWEN35_08B_SOURCE_TENSOR_COUNT - 24 * 3
        + 24 * 11
        + QWEN35_08B_A3_MOE_DERIVED_FP8_PAIR_COUNT;
    const BLOCK_FP8_OUTPUT_BLOCK: usize = 128;
    const BLOCK_FP8_INPUT_BLOCK: usize = 128;
    const BLOCK_FP8_MAX_FINITE: f32 = 448.0;
    const DERIVED_DENSE_PROJECTION_ROLES: &[&str] = &["lm_head", "linear_attn_b", "linear_attn_a"];

    fn fixed_qwen35_08b_snapshot_dir() -> PathBuf {
        hf_hub::Cache::default()
            .path()
            .join(QWEN35_08B_CACHE_REPO_DIR)
            .join("snapshots")
            .join(QWEN35_08B_REVISION)
    }

    enum DerivedSafetensorsView<'archive> {
        Borrowed {
            source: SafetensorsTensor<'archive>,
            shape: Vec<usize>,
        },
        Owned {
            dtype: Dtype,
            shape: Vec<usize>,
            bytes: Arc<[u8]>,
        },
        BlockFp8Values {
            source: SafetensorsTensor<'archive>,
            shape: Vec<usize>,
            scales_bf16_le: Arc<[u8]>,
        },
        BlockFp8Scales {
            shape: Vec<usize>,
            scales_bf16_le: Arc<[u8]>,
        },
    }

    impl View for DerivedSafetensorsView<'_> {
        fn dtype(&self) -> Dtype {
            match self {
                Self::Borrowed { source, .. } => source.dtype(),
                Self::Owned { dtype, .. } => *dtype,
                Self::BlockFp8Values { .. } => Dtype::F8_E4M3,
                Self::BlockFp8Scales { .. } => Dtype::BF16,
            }
        }

        fn shape(&self) -> &[usize] {
            match self {
                Self::Borrowed { shape, .. }
                | Self::Owned { shape, .. }
                | Self::BlockFp8Values { shape, .. }
                | Self::BlockFp8Scales { shape, .. } => shape,
            }
        }

        fn data(&self) -> Cow<'_, [u8]> {
            match self {
                Self::Borrowed { source, .. } => Cow::Borrowed(source.bytes()),
                Self::Owned { bytes, .. } => Cow::Borrowed(bytes),
                Self::BlockFp8Values {
                    source,
                    shape,
                    scales_bf16_le,
                } => {
                    let [n, k] = shape.as_slice() else {
                        unreachable!("block-FP8 source shape was preflighted as [N, K]")
                    };
                    Cow::Owned(
                        quantize_bf16_matrix_to_block_fp8(source.bytes(), *n, *k, scales_bf16_le)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "lazy block-FP8 encode failed for {:?}: {error}",
                                    source.external_name()
                                )
                            }),
                    )
                }
                Self::BlockFp8Scales { scales_bf16_le, .. } => Cow::Borrowed(scales_bf16_le),
            }
        }

        fn data_len(&self) -> usize {
            match self {
                Self::Borrowed { source, .. } => source.bytes().len(),
                Self::Owned { bytes, .. } => bytes.len(),
                Self::BlockFp8Values { shape, .. } => shape
                    .iter()
                    .copied()
                    .try_fold(1_usize, usize::checked_mul)
                    .expect("preflighted block-FP8 element count"),
                Self::BlockFp8Scales { scales_bf16_le, .. } => scales_bf16_le.len(),
            }
        }
    }

    fn usize_shape(shape: &[u64], tensor_name: &str) -> Result<Vec<usize>, String> {
        shape
            .iter()
            .map(|extent| {
                usize::try_from(*extent).map_err(|_| {
                    format!("tensor {tensor_name:?} extent {extent} exceeds host usize")
                })
            })
            .collect()
    }

    fn bounded_worker_count(work_items: usize) -> usize {
        std::thread::available_parallelism()
            .map_or(1, |parallelism| parallelism.get())
            .min(8)
            .min(work_items.max(1))
    }

    fn fill_bounded_ranges(
        total_units: usize,
        bytes_per_unit: usize,
        output: &mut [u8],
        thread_name: &str,
        fill: impl Fn(Range<usize>, &mut [u8]) -> Result<(), String> + Sync,
    ) -> Result<(), String> {
        let expected_bytes = total_units
            .checked_mul(bytes_per_unit)
            .ok_or_else(|| format!("{thread_name} output byte count overflows usize"))?;
        if total_units == 0 || bytes_per_unit == 0 || output.len() != expected_bytes {
            return Err(format!(
                "{thread_name} received invalid bounded range: units={total_units} bytes_per_unit={bytes_per_unit} output_bytes={}",
                output.len()
            ));
        }
        let workers = bounded_worker_count(total_units);
        let units_per_worker = total_units.div_ceil(workers);
        std::thread::scope(|scope| {
            let mut remaining = output;
            let mut handles = Vec::with_capacity(workers);
            let mut spawn_error = None;
            for worker in 0..workers {
                let start = worker * units_per_worker;
                let end = (start + units_per_worker).min(total_units);
                if start == end {
                    break;
                }
                let chunk_bytes = (end - start)
                    .checked_mul(bytes_per_unit)
                    .expect("bounded chunk byte count was preflighted");
                let (chunk, tail) = remaining.split_at_mut(chunk_bytes);
                remaining = tail;
                let fill = &fill;
                match std::thread::Builder::new()
                    .name(format!("{thread_name}-{worker}"))
                    .spawn_scoped(scope, move || fill(start..end, chunk))
                {
                    Ok(handle) => handles.push(handle),
                    Err(error) => {
                        spawn_error = Some(format!(
                            "spawn {thread_name} worker {worker}/{workers}: {error}"
                        ));
                        break;
                    }
                }
            }

            let mut first_error = spawn_error;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                    Err(payload) => {
                        if first_error.is_none() {
                            let reason = payload
                                .downcast_ref::<&str>()
                                .map(|value| (*value).to_owned())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown panic payload".to_owned());
                            first_error = Some(format!("{thread_name} worker panicked: {reason}"));
                        }
                    }
                }
            }
            first_error.map_or(Ok(()), Err)
        })
    }

    fn source_bf16_at(source_bf16_le: &[u8], index: usize) -> bf16 {
        let offset = index * 2;
        bf16::from_bits(u16::from_le_bytes([
            source_bf16_le[offset],
            source_bf16_le[offset + 1],
        ]))
    }

    fn stored_bf16_inverse_scale(maximum: f32) -> Result<bf16, String> {
        if maximum == 0.0 {
            return Ok(bf16::from_f32(1.0));
        }
        if !maximum.is_finite() || maximum < 0.0 {
            return Err(format!("invalid block maximum {maximum}"));
        }
        let required = f64::from(maximum) / f64::from(BLOCK_FP8_MAX_FINITE);
        let mut bits = bf16::from_f32(required as f32).to_bits();
        loop {
            let stored = bf16::from_bits(bits);
            let stored_f64 = f64::from(stored.to_f32());
            if stored.is_finite() && stored > bf16::ZERO && stored_f64 >= required {
                return Ok(stored);
            }
            bits = bits
                .checked_add(1)
                .ok_or_else(|| format!("BF16 inverse scale overflows for maximum {maximum}"))?;
        }
    }

    fn derive_block_fp8_scales(
        source_bf16_le: &[u8],
        n: usize,
        k: usize,
        tensor_name: &str,
    ) -> Result<Arc<[u8]>, String> {
        let source_elements = n
            .checked_mul(k)
            .ok_or_else(|| format!("tensor {tensor_name:?} element count overflows usize"))?;
        let expected_source_bytes = source_elements
            .checked_mul(2)
            .ok_or_else(|| format!("tensor {tensor_name:?} BF16 byte count overflows usize"))?;
        if n == 0 || k == 0 || source_bf16_le.len() != expected_source_bytes {
            return Err(format!(
                "tensor {tensor_name:?} has invalid BF16 matrix storage: shape=[{n}, {k}] bytes={} expected={expected_source_bytes}",
                source_bf16_le.len()
            ));
        }
        let block_rows = n.div_ceil(BLOCK_FP8_OUTPUT_BLOCK);
        let block_columns = k.div_ceil(BLOCK_FP8_INPUT_BLOCK);
        let scale_row_bytes = block_columns
            .checked_mul(2)
            .ok_or_else(|| format!("tensor {tensor_name:?} scale row overflows usize"))?;
        let scale_bytes = block_rows
            .checked_mul(scale_row_bytes)
            .ok_or_else(|| format!("tensor {tensor_name:?} scale grid overflows usize"))?;
        let mut output = vec![0_u8; scale_bytes];
        fill_bounded_ranges(
            block_rows,
            scale_row_bytes,
            &mut output,
            "qwen35-fp8-scale",
            |range, destination| {
                for block_output in range.clone() {
                    let row_start = block_output * BLOCK_FP8_OUTPUT_BLOCK;
                    let row_end = (row_start + BLOCK_FP8_OUTPUT_BLOCK).min(n);
                    for block_input in 0..block_columns {
                        let input_start = block_input * BLOCK_FP8_INPUT_BLOCK;
                        let input_end = (input_start + BLOCK_FP8_INPUT_BLOCK).min(k);
                        let mut maximum = 0.0_f32;
                        for output_feature in row_start..row_end {
                            for input_feature in input_start..input_end {
                                let source_index = output_feature * k + input_feature;
                                let value = source_bf16_at(source_bf16_le, source_index).to_f32();
                                if !value.is_finite() {
                                    return Err(format!(
                                        "tensor {tensor_name:?} contains non-finite BF16 at [{output_feature}, {input_feature}]"
                                    ));
                                }
                                maximum = maximum.max(value.abs());
                            }
                        }
                        let scale = stored_bf16_inverse_scale(maximum)?;
                        let local_block_output = block_output - range.start;
                        let offset = (local_block_output * block_columns + block_input) * 2;
                        destination[offset..offset + 2]
                            .copy_from_slice(&scale.to_bits().to_le_bytes());
                    }
                }
                Ok(())
            },
        )?;
        Ok(output.into())
    }

    fn quantize_bf16_matrix_to_block_fp8(
        source_bf16_le: &[u8],
        n: usize,
        k: usize,
        scales_bf16_le: &[u8],
    ) -> Result<Vec<u8>, String> {
        let value_count = n
            .checked_mul(k)
            .ok_or_else(|| "block-FP8 value count overflows usize".to_owned())?;
        let expected_source_bytes = value_count
            .checked_mul(2)
            .ok_or_else(|| "block-FP8 BF16 byte count overflows usize".to_owned())?;
        let block_rows = n.div_ceil(BLOCK_FP8_OUTPUT_BLOCK);
        let block_columns = k.div_ceil(BLOCK_FP8_INPUT_BLOCK);
        let expected_scale_bytes = block_rows
            .checked_mul(block_columns)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| "block-FP8 scale grid byte count overflows usize".to_owned())?;
        if n == 0
            || k == 0
            || source_bf16_le.len() != expected_source_bytes
            || scales_bf16_le.len() != expected_scale_bytes
        {
            return Err(format!(
                "invalid block-FP8 input lengths for [{n}, {k}]: source={} expected_source={expected_source_bytes} scales={} expected_scales={expected_scale_bytes}",
                source_bf16_le.len(),
                scales_bf16_le.len()
            ));
        }

        let mut output = vec![0_u8; value_count];
        fill_bounded_ranges(
            n,
            k,
            &mut output,
            "qwen35-fp8-values",
            |range, destination| {
                for output_feature in range.clone() {
                    let local_output = output_feature - range.start;
                    for input_feature in 0..k {
                        let source_index = output_feature * k + input_feature;
                        let value = source_bf16_at(source_bf16_le, source_index).to_f32();
                        if !value.is_finite() {
                            return Err(format!(
                                "non-finite BF16 source at [{output_feature}, {input_feature}]"
                            ));
                        }
                        let scale_index = ((output_feature / BLOCK_FP8_OUTPUT_BLOCK)
                            * block_columns
                            + input_feature / BLOCK_FP8_INPUT_BLOCK)
                            * 2;
                        let scale = bf16::from_bits(u16::from_le_bytes([
                            scales_bf16_le[scale_index],
                            scales_bf16_le[scale_index + 1],
                        ]))
                        .to_f32();
                        if !scale.is_finite() || !(scale > 0.0) {
                            return Err(format!(
                                "invalid stored BF16 inverse scale at [{output_feature}, {input_feature}]"
                            ));
                        }
                        let normalized = value / scale;
                        if !normalized.is_finite() || normalized.abs() > BLOCK_FP8_MAX_FINITE {
                            return Err(format!(
                                "stored BF16 scale would clip source [{output_feature}, {input_feature}]: value={value} scale={scale} normalized={normalized}"
                            ));
                        }
                        let bits = float8::F8E4M3::from_f32(normalized).to_bits();
                        if bits & 0x7f == 0x7f {
                            return Err(format!(
                                "E4M3 encoder produced non-finite bits 0x{bits:02x} at [{output_feature}, {input_feature}]"
                            ));
                        }
                        destination[local_output * k + input_feature] = bits;
                    }
                }
                Ok(())
            },
        )?;
        Ok(output)
    }

    fn block_fp8_derived_quantizes_role(role: &str) -> bool {
        BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES.contains(&role)
            && !DERIVED_DENSE_PROJECTION_ROLES.contains(&role)
    }

    fn safetensors_file_metadata(path: &Path) -> Result<Option<HashMap<String, String>>, String> {
        let file = File::open(path).map_err(|error| format!("open {path:?}: {error}"))?;
        let metadata = file
            .metadata()
            .map_err(|error| format!("stat opened {path:?}: {error}"))?;
        if !metadata.is_file() {
            return Err(format!("opened safetensors source {path:?} is not a file"));
        }
        let mmap = unsafe { Mmap::map(&file).map_err(|error| format!("mmap {path:?}: {error}"))? };
        let (_, metadata) = SafeTensors::read_metadata(&mmap)
            .map_err(|error| format!("read safetensors metadata {path:?}: {error}"))?;
        Ok(metadata.metadata().clone())
    }

    fn create_unique_derived_model_dir(fixture_id: &str) -> Result<PathBuf, String> {
        let temporary_root = std::env::temp_dir();
        if !temporary_root.is_dir() {
            return Err(format!(
                "TMPDIR root {temporary_root:?} is not an existing directory"
            ));
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock precedes UNIX epoch: {error}"))?
            .as_nanos();
        for attempt in 0..100_u32 {
            let path = temporary_root.join(format!(
                "ferrum-{fixture_id}-{}-{nonce}-{attempt}",
                std::process::id(),
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("create derived model dir {path:?}: {error}")),
            }
        }
        Err("could not allocate a unique derived model directory under TMPDIR".to_owned())
    }

    fn copy_snapshot_metadata_files(source: &Path, destination: &Path) -> Result<(), String> {
        let mut entries = std::fs::read_dir(source)
            .map_err(|error| format!("read source snapshot {source:?}: {error}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("list source snapshot {source:?}: {error}"))?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let source_path = entry.path();
            let file_name = entry.file_name();
            let file_name_text = file_name.to_string_lossy();
            if file_name_text == "config.json"
                || file_name_text == "model.safetensors.index.json"
                || file_name_text.ends_with(".safetensors")
            {
                continue;
            }
            let followed = std::fs::metadata(&source_path)
                .map_err(|error| format!("follow snapshot entry {source_path:?}: {error}"))?;
            if !followed.is_file() {
                return Err(format!(
                    "snapshot entry {source_path:?} is not a regular file (symlink-to-directory is forbidden)"
                ));
            }
            let mut source_file = File::open(&source_path)
                .map_err(|error| format!("open {source_path:?}: {error}"))?;
            if !source_file
                .metadata()
                .map_err(|error| format!("stat opened {source_path:?}: {error}"))?
                .is_file()
            {
                return Err(format!(
                    "opened snapshot entry {source_path:?} is not a file"
                ));
            }
            let destination_path = destination.join(&file_name);
            let mut destination_file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination_path)
                .map_err(|error| format!("create {destination_path:?}: {error}"))?;
            std::io::copy(&mut source_file, &mut destination_file).map_err(|error| {
                format!(
                    "materialize snapshot file {source_path:?} -> {destination_path:?}: {error}"
                )
            })?;
            destination_file
                .flush()
                .map_err(|error| format!("flush {destination_path:?}: {error}"))?;
        }
        Ok(())
    }

    fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("create {path:?}: {error}"))?;
        file.write_all(bytes)
            .map_err(|error| format!("write {path:?}: {error}"))?;
        file.flush()
            .map_err(|error| format!("flush {path:?}: {error}"))
    }

    fn push_derived_tensor_view<'archive>(
        views: &mut Vec<(String, DerivedSafetensorsView<'archive>)>,
        output_weight_map: &mut BTreeMap<String, String>,
        name: String,
        view: DerivedSafetensorsView<'archive>,
        output_shard: &str,
    ) {
        assert!(
            output_weight_map
                .insert(name.clone(), output_shard.to_owned())
                .is_none(),
            "derived tensor name {name:?} must be unique"
        );
        views.push((name, view));
    }

    fn push_owned_block_fp8_pair<'archive>(
        views: &mut Vec<(String, DerivedSafetensorsView<'archive>)>,
        output_weight_map: &mut BTreeMap<String, String>,
        value_name: String,
        shape: Vec<usize>,
        values_e4m3: Arc<[u8]>,
        scales_bf16_le: Arc<[u8]>,
        output_shard: &str,
    ) {
        let [n, k] = shape.as_slice() else {
            panic!("owned block-FP8 tensor {value_name:?} must be a matrix")
        };
        let (n, k) = (*n, *k);
        assert_eq!(values_e4m3.len(), n * k, "{value_name}");
        let scale_shape = vec![n.div_ceil(128), k.div_ceil(128)];
        assert_eq!(
            scales_bf16_le.len(),
            scale_shape.iter().product::<usize>() * 2,
            "{value_name}"
        );
        let scale_name = format!(
            "{}.weight_scale_inv",
            value_name
                .strip_suffix(".weight")
                .expect("owned block-FP8 tensor has .weight suffix")
        );
        push_derived_tensor_view(
            views,
            output_weight_map,
            value_name,
            DerivedSafetensorsView::Owned {
                dtype: Dtype::F8_E4M3,
                shape,
                bytes: values_e4m3,
            },
            output_shard,
        );
        push_derived_tensor_view(
            views,
            output_weight_map,
            scale_name,
            DerivedSafetensorsView::BlockFp8Scales {
                shape: scale_shape,
                scales_bf16_le,
            },
            output_shard,
        );
    }

    #[test]
    fn derived_block_fp8_quantization_uses_stored_bf16_scale_without_clipping() {
        assert_eq!(float8::F8E4M3::from_f32(448.0).to_bits(), 0x7e);
        assert_eq!(float8::F8E4M3::from_bits(0x7e).to_f32(), 448.0);

        let (n, k) = (130_usize, 129_usize);
        let mut source = Vec::with_capacity(n * k * 2);
        for index in 0..n * k {
            let value = match index {
                0 => bf16::MAX,
                1 => -bf16::MAX,
                _ if index % 17 == 0 => bf16::ZERO,
                _ => bf16::from_f32(((index % 257) as f32 - 128.0) / 13.0),
            };
            source.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let scales = derive_block_fp8_scales(&source, n, k, "synthetic.weight").unwrap();
        assert_eq!(scales.len(), n.div_ceil(128) * k.div_ceil(128) * 2);
        let quantized = quantize_bf16_matrix_to_block_fp8(&source, n, k, scales.as_ref()).unwrap();
        assert_eq!(quantized.len(), n * k);
        assert!(quantized.iter().all(|bits| bits & 0x7f != 0x7f));
        assert!(quantized
            .iter()
            .all(|bits| float8::F8E4M3::from_bits(*bits).to_f32().is_finite()));

        let zero_source = vec![0_u8; 128 * 128 * 2];
        let zero_scale = derive_block_fp8_scales(&zero_source, 128, 128, "zero.weight").unwrap();
        assert_eq!(
            bf16::from_bits(u16::from_le_bytes([zero_scale[0], zero_scale[1]])),
            bf16::from_f32(1.0)
        );
        assert!(
            quantize_bf16_matrix_to_block_fp8(&zero_source, 128, 128, &zero_scale)
                .unwrap()
                .iter()
                .all(|bits| *bits == 0)
        );

        let mut non_finite = zero_source;
        non_finite[..2].copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
        assert!(derive_block_fp8_scales(&non_finite, 128, 128, "nan.weight")
            .unwrap_err()
            .contains("non-finite BF16"));
    }

    #[test]
    #[ignore = "requires fixed Qwen/Qwen3.5-0.8B BF16 snapshot; writes derived model under TMPDIR"]
    fn derives_fixed_qwen35_08b_block_fp8_snapshot_for_cuda_e2e() {
        let source_input = fixed_qwen35_08b_snapshot_dir();
        let source_dir = std::fs::canonicalize(&source_input).unwrap_or_else(|error| {
            panic!(
                "fixed Qwen/Qwen3.5-0.8B@{QWEN35_08B_REVISION} snapshot is absent at \
                     {source_input:?}; populate the standard Hugging Face cache first: {error}"
            )
        });
        assert!(
            source_dir.is_dir(),
            "source is not a directory: {source_dir:?}"
        );
        assert!(
            source_dir
                .components()
                .any(|component| component.as_os_str() == QWEN35_08B_REVISION),
            "source path must bind fixed revision {QWEN35_08B_REVISION}: {source_dir:?}"
        );

        let source_config_bytes = std::fs::read(source_dir.join("config.json"))
            .unwrap_or_else(|error| panic!("read source config.json: {error}"));
        let mut derived_config: Value = serde_json::from_slice(&source_config_bytes)
            .unwrap_or_else(|error| panic!("parse source config.json: {error}"));
        assert!(
            derived_config
                .get("quantization_config")
                .is_none_or(Value::is_null),
            "fixed source top-level quantization_config must be absent or null"
        );
        assert!(
            derived_config
                .get("text_config")
                .and_then(|text| text.get("quantization_config"))
                .is_none_or(Value::is_null),
            "fixed source nested quantization_config must be absent or null"
        );
        let source_text = Qwen35TextConfig::from_hf_config_value(&derived_config).unwrap();
        assert!(!source_text.is_moe());
        assert_eq!(source_text.hidden_size, 1024);
        assert_eq!(source_text.num_hidden_layers, 24);
        assert_eq!(source_text.linear_attention_layers(), 18);
        assert_eq!(source_text.full_attention_layers(), 6);
        assert!(source_text.tie_word_embeddings);
        assert!(source_text.quantization.is_none());

        let source_archive = SafetensorsArchive::open(&source_dir).unwrap();
        assert_eq!(
            source_archive.tensor_count(),
            QWEN35_08B_SOURCE_TENSOR_COUNT
        );
        let source_payload_bytes = source_archive
            .tensor_names()
            .try_fold(0_u64, |total, name| {
                let bytes = source_archive.tensor(name).unwrap().bytes().len();
                total.checked_add(u64::try_from(bytes).unwrap())
            })
            .expect("source payload byte total does not overflow");
        assert_eq!(
            source_payload_bytes, QWEN35_08B_SOURCE_PAYLOAD_BYTES,
            "fixed source payload identity drifted"
        );
        if source_dir.join("model.safetensors.index.json").is_file() {
            let index: Value = serde_json::from_slice(
                &std::fs::read(source_dir.join("model.safetensors.index.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                index["metadata"]["total_size"].as_u64(),
                Some(source_payload_bytes),
                "source index total_size must equal tensor payload bytes"
            );
        }
        let source_inventory = Qwen35WeightInventory::from_names(source_archive.tensor_names());
        let source_plan = source_inventory
            .detect_prefix_and_resolve(&source_text)
            .unwrap();
        source_inventory
            .partition_resolved_plan(&source_plan)
            .unwrap()
            .require_no_unknown()
            .unwrap();

        let resolved = source_plan.global_tensors.iter().chain(
            source_plan
                .layers
                .iter()
                .flat_map(|layer| layer.tensors.iter()),
        );
        let mut role_by_source = BTreeMap::<String, String>::new();
        let mut quantized_sources = BTreeSet::<String>::new();
        let mut dense_exclusions = BTreeSet::from(["lm_head".to_owned()]);
        for weight in resolved.filter(|weight| weight.present) {
            let Qwen35ResolvedWeightSource::Dense { values } = weight
                .source
                .as_ref()
                .expect("present source has a typed bundle")
            else {
                panic!(
                    "fixed BF16 source unexpectedly contains quantized role {:?}",
                    weight.role
                );
            };
            assert!(
                role_by_source
                    .insert(values.clone(), weight.role.clone())
                    .is_none(),
                "typed source {values:?} is referenced by more than one role"
            );
            if block_fp8_derived_quantizes_role(&weight.role) {
                let tensor = source_archive.tensor(values).unwrap();
                assert_eq!(tensor.dtype(), Dtype::BF16, "projection {values:?}");
                assert_eq!(tensor.shape().len(), 2, "projection {values:?}");
                assert!(values.ends_with(".weight"), "projection {values:?}");
                assert!(quantized_sources.insert(values.clone()));
            } else if BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES.contains(&weight.role.as_str()) {
                let module = values
                    .strip_suffix(".weight")
                    .unwrap_or_else(|| panic!("dense projection {values:?} lacks .weight suffix"));
                dense_exclusions.insert(module.to_owned());
            }
        }
        assert_eq!(quantized_sources.len(), QWEN35_08B_DERIVED_FP8_PAIR_COUNT);
        assert_eq!(
            dense_exclusions.len(),
            QWEN35_08B_TYPED_DENSE_EXCLUSION_COUNT
        );

        let mut derived_scales = BTreeMap::<String, Arc<[u8]>>::new();
        for values in &quantized_sources {
            let scale_name = format!(
                "{}.weight_scale_inv",
                values.strip_suffix(".weight").unwrap()
            );
            assert!(
                !source_archive.contains(&scale_name),
                "derived sidecar would collide with source tensor {scale_name:?}"
            );
            let tensor = source_archive.tensor(values).unwrap();
            let [n, k] = tensor.shape() else {
                unreachable!("quantized source rank was preflighted")
            };
            let n = usize::try_from(*n).expect("N fits usize");
            let k = usize::try_from(*k).expect("K fits usize");
            let scales = derive_block_fp8_scales(tensor.bytes(), n, k, values).unwrap();
            assert!(derived_scales.insert(values.clone(), scales).is_none());
        }

        let source_shards = source_archive
            .tensor_names()
            .map(|name| {
                source_archive
                    .tensor(name)
                    .unwrap()
                    .source_file()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(source_shards.len(), 1, "fixed source must use one shard");
        let source_shard = source_dir.join(source_shards.iter().next().unwrap());
        let source_header_metadata = safetensors_file_metadata(&source_shard).unwrap();

        let quantization_config = serde_json::json!({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "modules_to_not_convert": dense_exclusions,
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        });
        assert_eq!(quantization_config.as_object().unwrap().len(), 5);
        let root = derived_config
            .as_object_mut()
            .expect("fixed source config root is an object");
        let nested = root
            .get_mut("text_config")
            .and_then(Value::as_object_mut)
            .expect("fixed source has text_config object");
        assert!(
            nested
                .remove("quantization_config")
                .is_none_or(|value| value.is_null()),
            "fixed source nested quantization_config must be absent or null"
        );
        root.insert(
            "quantization_config".to_owned(),
            quantization_config.clone(),
        );
        assert_eq!(
            Qwen35TextConfig::from_hf_config_value(&derived_config)
                .unwrap()
                .quantization
                .as_ref()
                .and_then(Qwen35QuantizationConfig::as_fp8)
                .unwrap()
                .modules_to_not_convert
                .len(),
            QWEN35_08B_TYPED_DENSE_EXCLUSION_COUNT
        );

        let output_dir = create_unique_derived_model_dir("qwen35-08b-block-fp8").unwrap();
        assert_ne!(output_dir, source_dir);
        copy_snapshot_metadata_files(&source_dir, &output_dir).unwrap();
        let mut config_bytes = serde_json::to_vec_pretty(&derived_config).unwrap();
        config_bytes.push(b'\n');
        write_new_file(&output_dir.join("config.json"), &config_bytes).unwrap();

        let mut views = Vec::<(String, DerivedSafetensorsView<'_>)>::new();
        let mut output_weight_map = BTreeMap::<String, String>::new();
        const OUTPUT_SHARD: &str = "model-00001-of-00001.safetensors";
        for name in source_archive.tensor_names() {
            let source = source_archive.tensor(name).unwrap();
            let shape = usize_shape(source.shape(), name).unwrap();
            if let Some(scales_bf16_le) = derived_scales.get(name) {
                let scale_name =
                    format!("{}.weight_scale_inv", name.strip_suffix(".weight").unwrap());
                let [n, k] = shape.as_slice() else {
                    unreachable!("quantized source rank was preflighted")
                };
                let (n, k) = (*n, *k);
                views.push((
                    name.to_owned(),
                    DerivedSafetensorsView::BlockFp8Values {
                        source,
                        shape,
                        scales_bf16_le: Arc::clone(scales_bf16_le),
                    },
                ));
                views.push((
                    scale_name.clone(),
                    DerivedSafetensorsView::BlockFp8Scales {
                        shape: vec![n.div_ceil(128), k.div_ceil(128)],
                        scales_bf16_le: Arc::clone(scales_bf16_le),
                    },
                ));
                assert!(output_weight_map
                    .insert(name.to_owned(), OUTPUT_SHARD.to_owned())
                    .is_none());
                assert!(output_weight_map
                    .insert(scale_name, OUTPUT_SHARD.to_owned())
                    .is_none());
            } else {
                views.push((
                    name.to_owned(),
                    DerivedSafetensorsView::Borrowed { source, shape },
                ));
                assert!(output_weight_map
                    .insert(name.to_owned(), OUTPUT_SHARD.to_owned())
                    .is_none());
            }
        }
        assert_eq!(
            views.len(),
            QWEN35_08B_SOURCE_TENSOR_COUNT + QWEN35_08B_DERIVED_FP8_PAIR_COUNT
        );
        let output_payload_bytes = views
            .iter()
            .try_fold(0_u64, |total, (_, view)| {
                total.checked_add(u64::try_from(view.data_len()).unwrap())
            })
            .expect("derived payload byte total does not overflow");
        let output_shard = output_dir.join(OUTPUT_SHARD);
        assert!(!output_shard.exists());
        serialize_to_file(views, &source_header_metadata, &output_shard).unwrap();
        drop(derived_scales);
        drop(source_archive);

        let index = serde_json::json!({
            "metadata": {"total_size": output_payload_bytes},
            "weight_map": output_weight_map
        });
        let mut index_bytes = serde_json::to_vec_pretty(&index).unwrap();
        index_bytes.push(b'\n');
        write_new_file(
            &output_dir.join("model.safetensors.index.json"),
            &index_bytes,
        )
        .unwrap();

        for entry in std::fs::read_dir(&output_dir).unwrap() {
            let path = entry.unwrap().path();
            assert!(
                !std::fs::symlink_metadata(&path)
                    .unwrap()
                    .file_type()
                    .is_symlink(),
                "derived output must materialize symlink {path:?}"
            );
        }
        assert_eq!(
            safetensors_file_metadata(&output_shard).unwrap(),
            source_header_metadata
        );
        let output_archive = SafetensorsArchive::open(&output_dir).unwrap();
        assert_eq!(output_archive.tensor_count(), output_weight_map.len());
        let reopened_payload_bytes = output_archive
            .tensor_names()
            .try_fold(0_u64, |total, name| {
                total.checked_add(
                    u64::try_from(output_archive.tensor(name).unwrap().bytes().len()).unwrap(),
                )
            })
            .unwrap();
        assert_eq!(reopened_payload_bytes, output_payload_bytes);
        assert_eq!(
            index["metadata"]["total_size"].as_u64(),
            Some(output_payload_bytes)
        );

        let output_config: Value =
            serde_json::from_slice(&std::fs::read(output_dir.join("config.json")).unwrap())
                .unwrap();
        assert!(output_config["text_config"]
            .get("quantization_config")
            .is_none());
        assert_eq!(output_config["quantization_config"], quantization_config);
        let output_text = Qwen35TextConfig::from_hf_config_value(&output_config).unwrap();
        let output_inventory = Qwen35WeightInventory::from_names(output_archive.tensor_names());
        let output_plan = output_inventory
            .detect_prefix_and_resolve(&output_text)
            .unwrap();
        output_inventory
            .partition_resolved_plan(&output_plan)
            .unwrap()
            .require_no_unknown()
            .unwrap();
        let output_resolved = output_plan.global_tensors.iter().chain(
            output_plan
                .layers
                .iter()
                .flat_map(|layer| layer.tensors.iter()),
        );
        let output_pair_count = output_resolved
            .filter(|weight| {
                matches!(
                    weight.source,
                    Some(Qwen35ResolvedWeightSource::BlockFp8 { .. })
                )
            })
            .count();
        assert_eq!(output_pair_count, QWEN35_08B_DERIVED_FP8_PAIR_COUNT);

        let prepared = prepare_from_model_dir(&output_dir).unwrap();
        assert_eq!(
            prepared.family().weight_schema().format_id.as_str(),
            "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale"
        );
        let mut prepared_pairs = Vec::new();
        for tensor in &prepared.family().weight_schema().tensors {
            collect_block_fp8_source_pairs(&tensor.physical_layout, &mut prepared_pairs);
        }
        assert_eq!(prepared_pairs.len(), QWEN35_08B_DERIVED_FP8_PAIR_COUNT);

        println!(
            "FERRUM QWEN35 0.8B BLOCK-FP8 DERIVED SNAPSHOT PASS: {}",
            output_dir.display()
        );
    }

    #[test]
    #[ignore = "requires fixed Qwen/Qwen3.5-0.8B BF16 snapshot; writes A3 MoE-derived model under TMPDIR"]
    fn derives_fixed_qwen35_08b_a3_moe_block_fp8_snapshot_for_cuda_e2e() {
        struct DenseMlpSources {
            mlp_prefix: String,
            gate: String,
            up: String,
            down: String,
        }

        let source_input = fixed_qwen35_08b_snapshot_dir();
        let source_dir = std::fs::canonicalize(&source_input).unwrap_or_else(|error| {
            panic!(
                "fixed Qwen/Qwen3.5-0.8B@{QWEN35_08B_REVISION} snapshot is absent at \
                 {source_input:?}; populate the standard Hugging Face cache first: {error}"
            )
        });
        assert!(
            source_dir.is_dir(),
            "source is not a directory: {source_dir:?}"
        );
        assert!(
            source_dir
                .components()
                .any(|component| component.as_os_str() == QWEN35_08B_REVISION),
            "source path must bind fixed revision {QWEN35_08B_REVISION}: {source_dir:?}"
        );

        let source_config_bytes = std::fs::read(source_dir.join("config.json"))
            .unwrap_or_else(|error| panic!("read source config.json: {error}"));
        let mut derived_config: Value = serde_json::from_slice(&source_config_bytes)
            .unwrap_or_else(|error| panic!("parse source config.json: {error}"));
        assert!(
            derived_config
                .get("quantization_config")
                .is_none_or(Value::is_null),
            "fixed source top-level quantization_config must be absent or null"
        );
        assert!(
            derived_config
                .get("text_config")
                .and_then(|text| text.get("quantization_config"))
                .is_none_or(Value::is_null),
            "fixed source nested quantization_config must be absent or null"
        );
        let source_text = Qwen35TextConfig::from_hf_config_value(&derived_config).unwrap();
        assert!(!source_text.is_moe());
        assert_eq!(source_text.hidden_size, 1024);
        assert_eq!(source_text.num_hidden_layers, 24);
        assert_eq!(source_text.linear_attention_layers(), 18);
        assert_eq!(source_text.full_attention_layers(), 6);
        assert!(source_text.tie_word_embeddings);
        assert!(source_text.quantization.is_none());
        let dense_intermediate_size = source_text
            .dense_intermediate_size
            .expect("fixed dense source has intermediate_size");

        let source_archive = SafetensorsArchive::open(&source_dir).unwrap();
        assert_eq!(
            source_archive.tensor_count(),
            QWEN35_08B_SOURCE_TENSOR_COUNT
        );
        let source_payload_bytes = source_archive
            .tensor_names()
            .try_fold(0_u64, |total, name| {
                let bytes = source_archive.tensor(name).unwrap().bytes().len();
                total.checked_add(u64::try_from(bytes).unwrap())
            })
            .expect("source payload byte total does not overflow");
        assert_eq!(
            source_payload_bytes, QWEN35_08B_SOURCE_PAYLOAD_BYTES,
            "fixed source payload identity drifted"
        );
        if source_dir.join("model.safetensors.index.json").is_file() {
            let index: Value = serde_json::from_slice(
                &std::fs::read(source_dir.join("model.safetensors.index.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                index["metadata"]["total_size"].as_u64(),
                Some(source_payload_bytes),
                "source index total_size must equal tensor payload bytes"
            );
        }

        let source_inventory = Qwen35WeightInventory::from_names(source_archive.tensor_names());
        let source_plan = source_inventory
            .detect_prefix_and_resolve(&source_text)
            .unwrap();
        source_inventory
            .partition_resolved_plan(&source_plan)
            .unwrap()
            .require_no_unknown()
            .unwrap();
        assert_eq!(source_plan.layers.len(), source_text.num_hidden_layers);

        let mut dense_mlp_layers = Vec::with_capacity(source_text.num_hidden_layers);
        let mut dense_mlp_source_names = BTreeSet::<String>::new();
        for layer in &source_plan.layers {
            assert_eq!(layer.layer_index, dense_mlp_layers.len());
            let dense_source = |role: &str| {
                let weight = layer
                    .tensors
                    .iter()
                    .find(|weight| weight.present && weight.role == role)
                    .unwrap_or_else(|| {
                        panic!(
                            "fixed dense layer {} is missing role {role:?}",
                            layer.layer_index
                        )
                    });
                let Some(Qwen35ResolvedWeightSource::Dense { values }) = &weight.source else {
                    panic!(
                        "fixed dense layer {} role {role:?} is not BF16",
                        layer.layer_index
                    )
                };
                values.clone()
            };
            let gate = dense_source("mlp_gate");
            let up = dense_source("mlp_up");
            let down = dense_source("mlp_down");
            let mlp_prefix = gate
                .strip_suffix(".gate_proj.weight")
                .unwrap_or_else(|| panic!("dense MLP gate has unexpected name {gate:?}"))
                .to_owned();
            assert_eq!(up, format!("{mlp_prefix}.up_proj.weight"));
            assert_eq!(down, format!("{mlp_prefix}.down_proj.weight"));
            assert!(dense_mlp_source_names.insert(gate.clone()));
            assert!(dense_mlp_source_names.insert(up.clone()));
            assert!(dense_mlp_source_names.insert(down.clone()));
            dense_mlp_layers.push(DenseMlpSources {
                mlp_prefix,
                gate,
                up,
                down,
            });
        }
        assert_eq!(dense_mlp_source_names.len(), 24 * 3);

        let resolved = source_plan.global_tensors.iter().chain(
            source_plan
                .layers
                .iter()
                .flat_map(|layer| layer.tensors.iter()),
        );
        let mut role_by_source = BTreeMap::<String, String>::new();
        let mut quantized_sources = BTreeSet::<String>::new();
        let mut dense_exclusions = BTreeSet::from(["lm_head".to_owned()]);
        for weight in resolved.filter(|weight| weight.present) {
            let Qwen35ResolvedWeightSource::Dense { values } = weight
                .source
                .as_ref()
                .expect("present source has a typed bundle")
            else {
                panic!(
                    "fixed BF16 source unexpectedly contains quantized role {:?}",
                    weight.role
                );
            };
            assert!(
                role_by_source
                    .insert(values.clone(), weight.role.clone())
                    .is_none(),
                "typed source {values:?} is referenced by more than one role"
            );
            if block_fp8_derived_quantizes_role(&weight.role) {
                let tensor = source_archive.tensor(values).unwrap();
                assert_eq!(tensor.dtype(), Dtype::BF16, "projection {values:?}");
                assert_eq!(tensor.shape().len(), 2, "projection {values:?}");
                assert!(values.ends_with(".weight"), "projection {values:?}");
                assert!(quantized_sources.insert(values.clone()));
            } else if BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES.contains(&weight.role.as_str()) {
                let module = values
                    .strip_suffix(".weight")
                    .unwrap_or_else(|| panic!("dense projection {values:?} lacks .weight suffix"));
                dense_exclusions.insert(module.to_owned());
            }
        }
        assert_eq!(quantized_sources.len(), QWEN35_08B_DERIVED_FP8_PAIR_COUNT);
        assert!(dense_mlp_source_names.is_subset(&quantized_sources));
        assert_eq!(
            dense_exclusions.len(),
            QWEN35_08B_TYPED_DENSE_EXCLUSION_COUNT
        );

        let mut derived_scales = BTreeMap::<String, Arc<[u8]>>::new();
        let mut encoded_dense_mlp = BTreeMap::<String, Arc<[u8]>>::new();
        for values in &quantized_sources {
            let scale_name = format!(
                "{}.weight_scale_inv",
                values.strip_suffix(".weight").unwrap()
            );
            assert!(
                !source_archive.contains(&scale_name),
                "derived sidecar would collide with source tensor {scale_name:?}"
            );
            let tensor = source_archive.tensor(values).unwrap();
            let [n, k] = tensor.shape() else {
                unreachable!("quantized source rank was preflighted")
            };
            let n = usize::try_from(*n).expect("N fits usize");
            let k = usize::try_from(*k).expect("K fits usize");
            let scales = derive_block_fp8_scales(tensor.bytes(), n, k, values).unwrap();
            if dense_mlp_source_names.contains(values) {
                let encoded =
                    quantize_bf16_matrix_to_block_fp8(tensor.bytes(), n, k, scales.as_ref())
                        .unwrap();
                assert!(encoded_dense_mlp
                    .insert(values.clone(), encoded.into())
                    .is_none());
            }
            assert!(derived_scales.insert(values.clone(), scales).is_none());
        }
        assert_eq!(encoded_dense_mlp.len(), 24 * 3);

        let source_shards = source_archive
            .tensor_names()
            .map(|name| {
                source_archive
                    .tensor(name)
                    .unwrap()
                    .source_file()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(source_shards.len(), 1, "fixed source must use one shard");
        let source_shard = source_dir.join(source_shards.iter().next().unwrap());
        let source_header_metadata = safetensors_file_metadata(&source_shard).unwrap();

        let quantization_config = serde_json::json!({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "modules_to_not_convert": dense_exclusions,
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        });
        assert_eq!(quantization_config.as_object().unwrap().len(), 5);
        let root = derived_config
            .as_object_mut()
            .expect("fixed source config root is an object");
        root.insert(
            "architectures".to_owned(),
            serde_json::json!(["Qwen3_5MoeForConditionalGeneration"]),
        );
        root.insert("model_type".to_owned(), Value::from("qwen3_5_moe"));
        root.remove("norm_topk_prob");
        {
            let nested = root
                .get_mut("text_config")
                .and_then(Value::as_object_mut)
                .expect("fixed source has text_config object");
            assert!(
                nested
                    .remove("quantization_config")
                    .is_none_or(|value| value.is_null()),
                "fixed source nested quantization_config must be absent or null"
            );
            assert_eq!(
                nested
                    .remove("intermediate_size")
                    .and_then(|value| value.as_u64()),
                Some(dense_intermediate_size as u64)
            );
            nested.insert("model_type".to_owned(), Value::from("qwen3_5_moe_text"));
            nested.insert(
                "num_experts".to_owned(),
                Value::from(QWEN35_08B_A3_MOE_EXPERT_COUNT),
            );
            nested.insert(
                "num_experts_per_tok".to_owned(),
                Value::from(QWEN35_08B_A3_MOE_EXPERTS_PER_TOKEN),
            );
            nested.insert(
                "moe_intermediate_size".to_owned(),
                Value::from(dense_intermediate_size),
            );
            nested.insert(
                "shared_expert_intermediate_size".to_owned(),
                Value::from(dense_intermediate_size),
            );
            nested.insert("norm_topk_prob".to_owned(), Value::Bool(true));
        }
        root.insert(
            "quantization_config".to_owned(),
            quantization_config.clone(),
        );
        let derived_text = Qwen35TextConfig::from_hf_config_value(&derived_config).unwrap();
        assert!(derived_text.is_moe());
        assert!(derived_text.dense_intermediate_size.is_none());
        let derived_moe = derived_text.moe.as_ref().unwrap();
        assert_eq!(derived_moe.num_experts, QWEN35_08B_A3_MOE_EXPERT_COUNT);
        assert_eq!(
            derived_moe.num_experts_per_tok,
            QWEN35_08B_A3_MOE_EXPERTS_PER_TOKEN
        );
        assert_eq!(derived_moe.moe_intermediate_size, dense_intermediate_size);
        assert_eq!(
            derived_moe.shared_expert_intermediate_size,
            dense_intermediate_size
        );
        assert!(derived_moe.norm_topk_prob);

        let output_dir = create_unique_derived_model_dir("qwen35-08b-a3-moe-block-fp8").unwrap();
        assert_ne!(output_dir, source_dir);
        copy_snapshot_metadata_files(&source_dir, &output_dir).unwrap();
        let mut config_bytes = serde_json::to_vec_pretty(&derived_config).unwrap();
        config_bytes.push(b'\n');
        write_new_file(&output_dir.join("config.json"), &config_bytes).unwrap();

        const OUTPUT_SHARD: &str = "model-00001-of-00001.safetensors";
        let mut views = Vec::<(String, DerivedSafetensorsView<'_>)>::new();
        let mut output_weight_map = BTreeMap::<String, String>::new();
        for name in source_archive.tensor_names() {
            if dense_mlp_source_names.contains(name) {
                continue;
            }
            let source = source_archive.tensor(name).unwrap();
            let shape = usize_shape(source.shape(), name).unwrap();
            if let Some(scales_bf16_le) = derived_scales.get(name) {
                let scale_name =
                    format!("{}.weight_scale_inv", name.strip_suffix(".weight").unwrap());
                let [n, k] = shape.as_slice() else {
                    unreachable!("quantized source rank was preflighted")
                };
                let (n, k) = (*n, *k);
                push_derived_tensor_view(
                    &mut views,
                    &mut output_weight_map,
                    name.to_owned(),
                    DerivedSafetensorsView::BlockFp8Values {
                        source,
                        shape,
                        scales_bf16_le: Arc::clone(scales_bf16_le),
                    },
                    OUTPUT_SHARD,
                );
                push_derived_tensor_view(
                    &mut views,
                    &mut output_weight_map,
                    scale_name,
                    DerivedSafetensorsView::BlockFp8Scales {
                        shape: vec![n.div_ceil(128), k.div_ceil(128)],
                        scales_bf16_le: Arc::clone(scales_bf16_le),
                    },
                    OUTPUT_SHARD,
                );
            } else {
                push_derived_tensor_view(
                    &mut views,
                    &mut output_weight_map,
                    name.to_owned(),
                    DerivedSafetensorsView::Borrowed { source, shape },
                    OUTPUT_SHARD,
                );
            }
        }

        // A zero router plus normalized top-k=1 gives the selected routed
        // expert weight 1.0. Both routed experts are byte-identical copies of
        // the dense MLP, while the shared expert's zero down projection makes
        // its contribution exactly zero.
        let hidden_size = source_text.hidden_size;
        let router_zeros: Arc<[u8]> =
            vec![0_u8; QWEN35_08B_A3_MOE_EXPERT_COUNT * hidden_size * 2].into();
        let shared_gate_zeros: Arc<[u8]> = vec![0_u8; hidden_size * 2].into();
        let shared_down_zeros: Arc<[u8]> = vec![0_u8; hidden_size * dense_intermediate_size].into();
        let shared_down_scale_shape = [
            hidden_size.div_ceil(BLOCK_FP8_OUTPUT_BLOCK),
            dense_intermediate_size.div_ceil(BLOCK_FP8_INPUT_BLOCK),
        ];
        let unit_bf16 = bf16::from_f32(1.0).to_bits().to_le_bytes();
        let mut shared_down_scale_bytes =
            Vec::with_capacity(shared_down_scale_shape.iter().product::<usize>() * 2);
        for _ in 0..shared_down_scale_shape.iter().product::<usize>() {
            shared_down_scale_bytes.extend_from_slice(&unit_bf16);
        }
        let shared_down_unit_scales: Arc<[u8]> = shared_down_scale_bytes.into();

        for layer in &dense_mlp_layers {
            push_derived_tensor_view(
                &mut views,
                &mut output_weight_map,
                format!("{}.gate.weight", layer.mlp_prefix),
                DerivedSafetensorsView::Owned {
                    dtype: Dtype::BF16,
                    shape: vec![QWEN35_08B_A3_MOE_EXPERT_COUNT, hidden_size],
                    bytes: Arc::clone(&router_zeros),
                },
                OUTPUT_SHARD,
            );
            push_derived_tensor_view(
                &mut views,
                &mut output_weight_map,
                format!("{}.shared_expert_gate.weight", layer.mlp_prefix),
                DerivedSafetensorsView::Owned {
                    dtype: Dtype::BF16,
                    shape: vec![1, hidden_size],
                    bytes: Arc::clone(&shared_gate_zeros),
                },
                OUTPUT_SHARD,
            );

            for expert in 0..QWEN35_08B_A3_MOE_EXPERT_COUNT {
                for (projection, source_name) in [
                    ("gate_proj", layer.gate.as_str()),
                    ("up_proj", layer.up.as_str()),
                    ("down_proj", layer.down.as_str()),
                ] {
                    let source = source_archive.tensor(source_name).unwrap();
                    let shape = usize_shape(source.shape(), source_name).unwrap();
                    push_owned_block_fp8_pair(
                        &mut views,
                        &mut output_weight_map,
                        format!("{}.experts.{expert}.{projection}.weight", layer.mlp_prefix),
                        shape,
                        Arc::clone(encoded_dense_mlp.get(source_name).unwrap()),
                        Arc::clone(derived_scales.get(source_name).unwrap()),
                        OUTPUT_SHARD,
                    );
                }
            }

            for (projection, source_name) in [
                ("gate_proj", layer.gate.as_str()),
                ("up_proj", layer.up.as_str()),
            ] {
                let source = source_archive.tensor(source_name).unwrap();
                let shape = usize_shape(source.shape(), source_name).unwrap();
                push_owned_block_fp8_pair(
                    &mut views,
                    &mut output_weight_map,
                    format!("{}.shared_expert.{projection}.weight", layer.mlp_prefix),
                    shape,
                    Arc::clone(encoded_dense_mlp.get(source_name).unwrap()),
                    Arc::clone(derived_scales.get(source_name).unwrap()),
                    OUTPUT_SHARD,
                );
            }
            push_owned_block_fp8_pair(
                &mut views,
                &mut output_weight_map,
                format!("{}.shared_expert.down_proj.weight", layer.mlp_prefix),
                vec![hidden_size, dense_intermediate_size],
                Arc::clone(&shared_down_zeros),
                Arc::clone(&shared_down_unit_scales),
                OUTPUT_SHARD,
            );
        }
        assert_eq!(views.len(), QWEN35_08B_A3_MOE_DERIVED_TENSOR_COUNT);
        assert_eq!(output_weight_map.len(), views.len());

        let output_payload_bytes = views
            .iter()
            .try_fold(0_u64, |total, (_, view)| {
                total.checked_add(u64::try_from(view.data_len()).unwrap())
            })
            .expect("derived payload byte total does not overflow");
        let output_shard = output_dir.join(OUTPUT_SHARD);
        assert!(!output_shard.exists());
        serialize_to_file(views, &source_header_metadata, &output_shard).unwrap();
        drop(encoded_dense_mlp);
        drop(derived_scales);
        drop(source_archive);

        let index = serde_json::json!({
            "metadata": {"total_size": output_payload_bytes},
            "weight_map": output_weight_map
        });
        let mut index_bytes = serde_json::to_vec_pretty(&index).unwrap();
        index_bytes.push(b'\n');
        write_new_file(
            &output_dir.join("model.safetensors.index.json"),
            &index_bytes,
        )
        .unwrap();

        for entry in std::fs::read_dir(&output_dir).unwrap() {
            let path = entry.unwrap().path();
            assert!(
                !std::fs::symlink_metadata(&path)
                    .unwrap()
                    .file_type()
                    .is_symlink(),
                "derived output must materialize symlink {path:?}"
            );
        }
        assert_eq!(
            safetensors_file_metadata(&output_shard).unwrap(),
            source_header_metadata
        );
        let output_archive = SafetensorsArchive::open(&output_dir).unwrap();
        assert_eq!(
            output_archive.tensor_count(),
            QWEN35_08B_A3_MOE_DERIVED_TENSOR_COUNT
        );
        let reopened_payload_bytes = output_archive
            .tensor_names()
            .try_fold(0_u64, |total, name| {
                total.checked_add(
                    u64::try_from(output_archive.tensor(name).unwrap().bytes().len()).unwrap(),
                )
            })
            .unwrap();
        assert_eq!(reopened_payload_bytes, output_payload_bytes);
        assert_eq!(
            index["metadata"]["total_size"].as_u64(),
            Some(output_payload_bytes)
        );

        let output_config: Value =
            serde_json::from_slice(&std::fs::read(output_dir.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(
            output_config["architectures"],
            serde_json::json!(["Qwen3_5MoeForConditionalGeneration"])
        );
        assert_eq!(output_config["model_type"], "qwen3_5_moe");
        assert_eq!(
            output_config["text_config"]["model_type"],
            "qwen3_5_moe_text"
        );
        assert!(output_config["text_config"]
            .get("intermediate_size")
            .is_none());
        assert!(output_config["text_config"]
            .get("quantization_config")
            .is_none());
        assert_eq!(output_config["quantization_config"], quantization_config);
        let output_text = Qwen35TextConfig::from_hf_config_value(&output_config).unwrap();
        let output_inventory = Qwen35WeightInventory::from_names(output_archive.tensor_names());
        let output_plan = output_inventory
            .detect_prefix_and_resolve(&output_text)
            .unwrap();
        output_inventory
            .partition_resolved_plan(&output_plan)
            .unwrap()
            .require_no_unknown()
            .unwrap();
        let output_pair_count = output_plan
            .global_tensors
            .iter()
            .chain(
                output_plan
                    .layers
                    .iter()
                    .flat_map(|layer| layer.tensors.iter()),
            )
            .filter(|weight| {
                matches!(
                    weight.source,
                    Some(Qwen35ResolvedWeightSource::BlockFp8 { .. })
                )
            })
            .count();
        assert_eq!(output_pair_count, QWEN35_08B_A3_MOE_DERIVED_FP8_PAIR_COUNT);

        for layer in &dense_mlp_layers {
            let router = output_archive
                .tensor(&format!("{}.gate.weight", layer.mlp_prefix))
                .unwrap();
            assert_eq!(router.dtype(), Dtype::BF16);
            assert!(router.bytes().iter().all(|byte| *byte == 0));
            let shared_gate = output_archive
                .tensor(&format!("{}.shared_expert_gate.weight", layer.mlp_prefix))
                .unwrap();
            assert_eq!(shared_gate.dtype(), Dtype::BF16);
            assert!(shared_gate.bytes().iter().all(|byte| *byte == 0));

            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let expert_zero_name =
                    format!("{}.experts.0.{projection}.weight", layer.mlp_prefix);
                let expert_one_name = format!("{}.experts.1.{projection}.weight", layer.mlp_prefix);
                assert_eq!(
                    output_archive.tensor(&expert_zero_name).unwrap().bytes(),
                    output_archive.tensor(&expert_one_name).unwrap().bytes(),
                    "routed experts must be identical for {expert_zero_name}"
                );
                assert_eq!(
                    output_archive
                        .tensor(&expert_zero_name.replace(".weight", ".weight_scale_inv"))
                        .unwrap()
                        .bytes(),
                    output_archive
                        .tensor(&expert_one_name.replace(".weight", ".weight_scale_inv"))
                        .unwrap()
                        .bytes(),
                    "routed expert scales must be identical for {expert_zero_name}"
                );
                if projection != "down_proj" {
                    let shared_name =
                        format!("{}.shared_expert.{projection}.weight", layer.mlp_prefix);
                    assert_eq!(
                        output_archive.tensor(&expert_zero_name).unwrap().bytes(),
                        output_archive.tensor(&shared_name).unwrap().bytes(),
                        "shared {projection} must reuse the fixed dense MLP source"
                    );
                }
            }
            let shared_down_name = format!("{}.shared_expert.down_proj.weight", layer.mlp_prefix);
            let shared_down = output_archive.tensor(&shared_down_name).unwrap();
            assert_eq!(shared_down.dtype(), Dtype::F8_E4M3);
            assert!(shared_down.bytes().iter().all(|byte| *byte == 0));
            let shared_down_scale = output_archive
                .tensor(&shared_down_name.replace(".weight", ".weight_scale_inv"))
                .unwrap();
            assert_eq!(shared_down_scale.dtype(), Dtype::BF16);
            assert!(shared_down_scale.bytes().chunks_exact(2).all(|bytes| {
                bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])) == bf16::from_f32(1.0)
            }));
        }

        let prepared = prepare_from_model_dir(&output_dir).unwrap();
        assert_eq!(
            prepared.family().external_metadata_id().as_str(),
            MOE_EXTERNAL_METADATA_ID
        );
        assert_eq!(
            prepared.family().weight_schema().format_id.as_str(),
            "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale"
        );
        assert_eq!(
            prepared.family().weight_schema().layout_id.as_str(),
            "weight-layout.qwen3_5.hybrid_moe.fp8_block_grid.expert_major.packed_gdn_qkvzba"
        );
        assert_eq!(
            crate::vnext::moe_capabilities_from_program(prepared.family()).unwrap(),
            Some(ferrum_types::MoeCapabilities {
                num_experts: QWEN35_08B_A3_MOE_EXPERT_COUNT,
                experts_per_token: QWEN35_08B_A3_MOE_EXPERTS_PER_TOKEN,
                moe_intermediate_size: Some(dense_intermediate_size),
            })
        );
        let prepared_value_source_count = prepared
            .family()
            .weight_schema()
            .components
            .iter()
            .filter(|component| component.role == WeightComponentRole::PackedValues)
            .map(|component| component.external_names.len())
            .sum::<usize>();
        assert_eq!(
            prepared_value_source_count,
            QWEN35_08B_A3_MOE_DERIVED_FP8_PAIR_COUNT
        );

        println!(
            "FERRUM QWEN35 0.8B A3 MOE BLOCK-FP8 DERIVED SNAPSHOT PASS: {}",
            output_dir.display()
        );
    }

    #[test]
    fn accepts_fixed_qwen38_compressed_tensors_contract_fixture() {
        let text = preflight_semantic_config(QWEN38_AWQ_INT4_CONFIG).unwrap();
        assert_eq!(text.top_level_model_type.as_deref(), Some("qwen3_5"));
        assert_eq!(text.text_model_type, "qwen3_5_text");
        assert_eq!(text.hidden_size, 5120);
        assert_eq!(text.num_hidden_layers, 64);
        assert_eq!(text.linear_attention_layers(), 48);
        assert_eq!(text.full_attention_layers(), 16);
        assert_eq!(text.linear_attention.num_key_heads, 16);
        assert_eq!(text.linear_attention.num_value_heads, 48);
        assert_eq!(text.linear_attention.key_head_dim, 128);
        assert_eq!(text.linear_attention.value_head_dim, 128);
        assert_eq!(text.head_dim, 256);
        assert_eq!(text.num_attention_heads, 24);
        assert_eq!(text.num_key_value_heads, 4);
        assert!(text.attn_output_gate);
        assert!(!text.tie_word_embeddings);
        assert_eq!(text.dense_intermediate_size, Some(17408));
        assert_eq!(text.rope_parameters.rope_theta, 10_000_000.0);
        assert_eq!(text.rope_parameters.partial_rotary_factor, 0.25);
        assert!(text.rope_parameters.mrope_interleaved);
        assert_eq!(text.rope_parameters.mrope_section, Some(vec![11, 11, 10]));

        let quantization = text.quantization.as_ref().unwrap();
        let recipe = quantization
            .as_compressed_tensors()
            .expect("typed compressed-tensors recipe");
        assert_eq!(quantization.quant_method(), "compressed-tensors");
        assert_eq!(recipe.format, "pack-quantized");
        assert_eq!(recipe.bits, 4);
        assert_eq!(recipe.group_size, 32);
        assert!(!recipe.sym);
        assert!(!recipe.desc_act);
        assert_eq!(recipe.weight_type, "int");
        assert_eq!(recipe.strategy, "group");
        assert_eq!(recipe.dynamic, Some(false));
        assert_eq!(recipe.targets, ["Linear"]);
        assert!(!recipe.input_activations);
        assert!(!recipe.output_activations);
    }

    #[test]
    fn accepts_fixed_qwen38_fp8_metadata_before_source_layout_selection() {
        let text = preflight_semantic_config(QWEN38_FP8_CONFIG).unwrap();
        let quantization = text.quantization.as_ref().unwrap();
        let recipe = quantization.as_fp8().expect("typed block-FP8 recipe");

        assert_eq!(quantization.quant_method(), "fp8");
        assert_eq!(recipe.weight_block_size.as_array(), [128, 128]);
        assert_eq!(recipe.modules_to_not_convert.len(), 10);
    }

    #[test]
    fn compiles_block_fp8_source_grid_into_independent_composite_leaves() {
        let config = test_block_fp8_config();
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .unwrap();
        let schema = prepared.weight_schema();

        assert_eq!(
            schema.format_id.as_str(),
            "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale"
        );
        assert_eq!(
            schema.quantization_formats(),
            BTreeSet::from([QuantizationFormatId::new(BLOCK_FP8_E4M3_SOURCE_FORMAT_ID).unwrap()])
        );

        let gate_up = schema
            .tensor(&packed_gate_up_weight_id(0).unwrap())
            .expect("packed gate/up logical weight");
        let PhysicalWeightLayout::Composite { parts } = &gate_up.physical_layout else {
            panic!("block-FP8 gate/up must preserve two independent source leaves")
        };
        assert_eq!(parts.len(), 2);
        for part in parts {
            let PhysicalWeightLayout::QuantizedBlockGrid { block_axes, .. } = part.layout.as_ref()
            else {
                panic!("each gate/up partition must retain its block grid")
            };
            assert_eq!(*block_axes, [1, 2]);
        }

        let packed_attention = schema
            .tensor(&packed_linear_attention_weight_id(0, PACKED_LINEAR_ATTN_QKVZBA_ROLE).unwrap())
            .expect("packed linear-attention logical weight");
        let PhysicalWeightLayout::Composite { parts } = &packed_attention.physical_layout else {
            panic!("block-FP8 qkv/z with dense b/a must preserve four source leaves")
        };
        assert_eq!(parts.len(), 4);
        for part in &parts[..2] {
            let PhysicalWeightLayout::QuantizedBlockGrid { block_axes, .. } = part.layout.as_ref()
            else {
                panic!("qkv and z projections must retain independent block grids")
            };
            assert_eq!(*block_axes, [0, 1]);
        }
        assert!(matches!(
            parts[2].layout.as_ref(),
            PhysicalWeightLayout::Dense { .. }
        ));
        assert!(matches!(
            parts[3].layout.as_ref(),
            PhysicalWeightLayout::Dense { .. }
        ));
    }

    #[test]
    fn materializes_real_qwen35_dense_block_fp8_family_into_marlin_fp8_schema() {
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(test_official_marlin_block_fp8_config()).unwrap())
            .unwrap();
        let source_schema = prepared.weight_schema();
        let denominator = QuantizedProviderAttributionDenominator::from_prepared_family(&prepared)
            .unwrap()
            .expect("official block-FP8 family has a quantized attribution denominator");
        assert_eq!(denominator.quant_tensor_count(), 400);
        assert_eq!(denominator.operation_count(), 3);
        assert_eq!(denominator.item_count(), 403);
        assert_eq!(
            denominator.sha256(),
            "5e366997e15e1a94d90b1ae07281269e8a46f75904306564d56354c8ebea2e4e"
        );
        let materializer = block_fp8_to_marlin_fp8_weight_materializer().unwrap();
        assert_eq!(
            materializer.descriptor().fidelity(),
            WeightMaterializationFidelity::Approximate
        );
        let device = DeviceDescriptor {
            id: DeviceId::new("device.test.qwen35-block-fp8-marlin").unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 1 << 30,
            runtime_implementation_fingerprint: "0".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        };
        device.validate().unwrap();
        let execution_schema = materializer.execution_schema(&prepared, &device).unwrap();
        execution_schema.validate(prepared.family_id()).unwrap();

        let mut source_pairs = Vec::new();
        for tensor in &source_schema.tensors {
            collect_block_fp8_source_pairs(&tensor.physical_layout, &mut source_pairs);
        }
        let source_pairs = source_pairs.into_iter().collect::<BTreeSet<_>>();
        assert_eq!(
            source_pairs.len(),
            400,
            "official dense family FP8 leaf count"
        );

        let mut execution_leaves = Vec::new();
        let mut residual_block_grid_count = 0;
        for tensor in &execution_schema.tensors {
            collect_marlin_fp8_execution_leaves(
                &tensor.physical_layout,
                &mut execution_leaves,
                &mut residual_block_grid_count,
            );
        }
        assert_eq!(residual_block_grid_count, 0);
        assert_eq!(execution_leaves.len(), 400);
        assert_eq!(
            execution_leaves
                .iter()
                .filter(|(_, _, group_axis)| *group_axis == 2)
                .count(),
            128,
            "64 gate/up composites each retain two rank-three leaves"
        );
        assert_eq!(
            source_pairs
                .iter()
                .filter(|(values, _)| values.as_str().contains(".linear_attn_"))
                .count(),
            144,
            "48 GDA layers each expose qkv, z, and output FP8 leaves"
        );
        assert_eq!(
            source_pairs
                .iter()
                .filter(|(values, _)| values.as_str().contains(".self_attn_"))
                .count(),
            64,
            "16 causal-attention layers each expose q, k, v, and o FP8 leaves"
        );
        assert_eq!(
            source_pairs
                .iter()
                .filter(|(values, _)| values.as_str().contains(".mlp_"))
                .count(),
            192,
            "64 SwiGLU layers each expose gate, up, and down FP8 leaves"
        );

        for (packed_id, scales_id, _) in &execution_leaves {
            let packed = execution_schema
                .components
                .iter()
                .find(|component| component.id == *packed_id)
                .expect("Marlin packed component exists");
            let WeightEncoding::Quantized(quantization) = &packed.encoding else {
                panic!("Marlin packed component must remain explicitly quantized")
            };
            assert_eq!(
                quantization.format_id.as_str(),
                MARLIN_FP8_QUANTIZATION_FORMAT_ID
            );
            let scales = execution_schema
                .components
                .iter()
                .find(|component| component.id == *scales_id)
                .expect("Marlin scales component exists");
            assert_eq!(
                scales.encoding,
                WeightEncoding::Dense {
                    element_type: ElementType::F16
                }
            );
        }

        for layer_index in 0..64 {
            let weight_id = packed_gate_up_weight_id(layer_index).unwrap();
            let tensor = execution_schema
                .tensor(&weight_id)
                .expect("packed gate/up execution tensor");
            let PhysicalWeightLayout::Composite { parts } = &tensor.physical_layout else {
                panic!("gate/up execution weight must stay composite")
            };
            assert_eq!(parts.len(), 2);
            assert!(parts.iter().all(|part| matches!(
                part.layout.as_ref(),
                PhysicalWeightLayout::Quantized { group_axis: 2, .. }
            )));
        }

        for layer_index in (0..64).filter(|layer_index| layer_index % 4 != 3) {
            let weight_id =
                packed_linear_attention_weight_id(layer_index, PACKED_LINEAR_ATTN_QKVZBA_ROLE)
                    .unwrap();
            let source = source_schema
                .tensor(&weight_id)
                .expect("source GDA packed projection");
            let execution = execution_schema
                .tensor(&weight_id)
                .expect("execution GDA packed projection");
            let PhysicalWeightLayout::Composite {
                parts: source_parts,
            } = &source.physical_layout
            else {
                panic!("source GDA projection must be composite")
            };
            let PhysicalWeightLayout::Composite {
                parts: execution_parts,
            } = &execution.physical_layout
            else {
                panic!("execution GDA projection must stay composite")
            };
            assert_eq!(source_parts.len(), 4);
            assert_eq!(execution_parts.len(), 4);
            for (source_part, execution_part) in source_parts.iter().zip(execution_parts) {
                assert_eq!(execution_part.logical_offsets, source_part.logical_offsets);
                assert_eq!(execution_part.extents, source_part.extents);
            }
            for index in 0..2 {
                assert!(matches!(
                    execution_parts[index].layout.as_ref(),
                    PhysicalWeightLayout::Quantized { group_axis: 1, .. }
                ));
            }
            for index in 2..4 {
                assert_eq!(execution_parts[index].layout, source_parts[index].layout);
                assert!(matches!(
                    execution_parts[index].layout.as_ref(),
                    PhysicalWeightLayout::Dense { .. }
                ));
            }
        }

        let component_sources = materializer
            .component_sources(&prepared, &execution_schema)
            .unwrap();
        let execution_component_ids = execution_schema
            .components
            .iter()
            .map(|component| component.id.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            component_sources.keys().cloned().collect::<BTreeSet<_>>(),
            execution_component_ids,
            "every execution component must have exactly one provenance-map entry"
        );
        assert!(source_pairs.iter().all(|(values, inverse_scales)| {
            !execution_component_ids.contains(values)
                && !execution_component_ids.contains(inverse_scales)
        }));
        let mapped_source_pairs = execution_leaves
            .iter()
            .map(|(packed_id, scales_id, _)| {
                let packed_sources = component_sources
                    .get(packed_id)
                    .expect("packed component has declared provenance");
                let scales_sources = component_sources
                    .get(scales_id)
                    .expect("scales component has declared provenance");
                assert_eq!(packed_sources, scales_sources);
                let [values, inverse_scales] = packed_sources.as_slice() else {
                    panic!("block-FP8 derived components must consume one ordered source pair")
                };
                (values.clone(), inverse_scales.clone())
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(mapped_source_pairs, source_pairs);
    }

    #[test]
    fn qwen36_27b_reuses_the_qwen35_block_fp8_execution_contract() {
        let mut qwen38_config = test_official_marlin_block_fp8_config();
        qwen38_config.hf_config["transformers_version"] = Value::String("5.8.0.dev0".to_owned());
        let mut qwen36_config = qwen38_config.clone();
        qwen36_config.hf_config["transformers_version"] = Value::String("4.57.1".to_owned());
        assert_ne!(qwen38_config.hf_config, qwen36_config.hf_config);

        let qwen38_text = Qwen35TextConfig::from_hf_config_value(&qwen38_config.hf_config).unwrap();
        let qwen36_text = Qwen35TextConfig::from_hf_config_value(&qwen36_config.hf_config).unwrap();
        assert_eq!(qwen38_text, qwen36_text);

        let qwen38 = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(qwen38_config).unwrap())
            .unwrap();
        let qwen36 = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(qwen36_config).unwrap())
            .unwrap();
        assert_eq!(
            qwen38.weight_schema().fingerprint().unwrap(),
            qwen36.weight_schema().fingerprint().unwrap()
        );
        assert_eq!(
            qwen38.program().fingerprint().unwrap(),
            qwen36.program().fingerprint().unwrap()
        );

        let qwen38_denominator =
            QuantizedProviderAttributionDenominator::from_prepared_family(&qwen38)
                .unwrap()
                .unwrap();
        let qwen36_denominator =
            QuantizedProviderAttributionDenominator::from_prepared_family(&qwen36)
                .unwrap()
                .unwrap();
        assert_eq!(qwen38_denominator, qwen36_denominator);
        assert_eq!(qwen36_denominator.quant_tensor_count(), 400);
        assert_eq!(qwen36_denominator.operation_count(), 3);
        assert_eq!(qwen36_denominator.item_count(), 403);
        assert_eq!(
            qwen36_denominator.sha256(),
            "5e366997e15e1a94d90b1ae07281269e8a46f75904306564d56354c8ebea2e4e"
        );

        let device = DeviceDescriptor {
            id: DeviceId::new("device.test.qwen36-block-fp8-marlin").unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 1 << 30,
            runtime_implementation_fingerprint: "0".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        };
        device.validate().unwrap();
        let materializer = block_fp8_to_marlin_fp8_weight_materializer().unwrap();
        let qwen38_execution = materializer.execution_schema(&qwen38, &device).unwrap();
        let qwen36_execution = materializer.execution_schema(&qwen36, &device).unwrap();
        assert_eq!(
            qwen38_execution.fingerprint().unwrap(),
            qwen36_execution.fingerprint().unwrap()
        );
    }

    #[test]
    fn qwen38_block_fp8_program_matches_standard_operation_contracts() {
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(test_block_fp8_config()).unwrap())
            .unwrap();
        let program = prepared.program();
        let contracts = [
            (
                GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
                ContractVersion::new(6, 0),
                ContractVersion::new(6, 0),
                gated_delta_recurrent_attention_contract().unwrap(),
                3,
            ),
            (
                CAUSAL_PAGED_ATTENTION_OPERATION_ID,
                ContractVersion::new(2, 0),
                ContractVersion::new(2, 0),
                causal_paged_attention_contract().unwrap(),
                1,
            ),
            (
                DENSE_SWIGLU_OPERATION_ID,
                ContractVersion::new(1, 0),
                ContractVersion::new(1, 0),
                dense_swiglu_contract().unwrap(),
                4,
            ),
            (
                LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
                ContractVersion::new(1, 0),
                ContractVersion::new(1, 1),
                last_token_dense_linear_contract().unwrap(),
                1,
            ),
        ];
        for (
            operation_id,
            required_version,
            standard_contract_version,
            contract,
            expected_node_count,
        ) in contracts
        {
            let matching_nodes = program
                .blocks()
                .iter()
                .flat_map(|block| &block.nodes)
                .filter(|node| node.operation_id.as_str() == operation_id)
                .collect::<Vec<_>>();
            assert_eq!(matching_nodes.len(), expected_node_count);
            assert!(
                matching_nodes
                    .iter()
                    .all(|node| node.required_version == required_version),
                "prepared Qwen3.8 block-FP8 program has a mixed or stale {operation_id} version"
            );

            let descriptor = contract.descriptor();
            assert_eq!(descriptor.id.as_str(), operation_id);
            assert_eq!(descriptor.version, standard_contract_version);
            assert_eq!(
                descriptor.provider.minimum_version,
                standard_contract_version
            );
        }
    }

    #[test]
    fn rejects_block_fp8_inverse_scale_grid_drift_before_runtime() {
        let mut config = test_block_fp8_config();
        let weight = config
            .weights
            .iter_mut()
            .find(|weight| weight.role == "linear_attn_qkv")
            .expect("test contains a block-FP8 qkv weight");
        let FamilyWeightSourceEncoding::BlockFp8 { scale_inv, .. } = &mut weight.source_encoding
        else {
            panic!("linear-attention qkv must use block-FP8 in the test contract")
        };
        let [n, k] = weight.dimensions.as_slice() else {
            panic!("linear-attention qkv must be a matrix")
        };
        assert_eq!(scale_inv.dimensions, [n.div_ceil(128), k.div_ceil(128)]);
        scale_inv.dimensions[1] += 1;

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("mismatched block-FP8 scale grid must fail before runtime");
        let VNextError::InvalidModelConfig {
            family_id,
            field,
            reason,
        } = error
        else {
            panic!("expected typed invalid-model-config rejection, got {error}")
        };
        assert_eq!(family_id, FAMILY_ID);
        assert_eq!(field, "weights.source_encoding.scale_inv.dimensions");
        assert!(reason.contains("inverse-scale shape"), "{reason}");
    }

    #[test]
    fn rejects_missing_block_fp8_sidecar_for_non_excluded_projection_before_allocation() {
        let mut config = test_block_fp8_config();
        let weight = config
            .weights
            .iter_mut()
            .find(|weight| weight.role == "linear_attn_qkv")
            .expect("test contains a block-FP8 qkv projection");
        weight.source_encoding = FamilyWeightSourceEncoding::Dense {
            element_type: ElementType::Bf16,
        };

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("a non-excluded projection without its FP8 sidecar must fail closed");
        let VNextError::InvalidModelConfig { field, reason, .. } = error else {
            panic!("expected typed invalid-model-config rejection, got {error}")
        };
        assert_eq!(field, "weights.source_encoding");
        assert!(reason.contains("execution-eligible projection"), "{reason}");
        assert!(reason.contains("weight_scale_inv"), "{reason}");
    }

    #[test]
    fn rejects_extra_block_fp8_pair_for_typed_dense_exclusion_before_allocation() {
        let mut config = test_block_fp8_config();
        let weight = config
            .weights
            .iter_mut()
            .find(|weight| weight.role == "linear_attn_b")
            .expect("test contains a dense-excluded recurrent b projection");
        let values_name = weight.external_name.clone();
        let dimensions = weight.dimensions.clone();
        let [n, k] = dimensions.as_slice() else {
            panic!("test recurrent b projection is a matrix")
        };
        weight.source_encoding = FamilyWeightSourceEncoding::BlockFp8 {
            values: FamilyBlockFp8Tensor {
                external_name: values_name.clone(),
                dimensions: dimensions.clone(),
                dtype: FamilyBlockFp8Dtype::F8E4m3,
            },
            scale_inv: FamilyBlockFp8Tensor {
                external_name: format!(
                    "{}.weight_scale_inv",
                    values_name.strip_suffix(".weight").unwrap()
                ),
                dimensions: vec![n.div_ceil(128), k.div_ceil(128)],
                dtype: FamilyBlockFp8Dtype::Bf16,
            },
        };

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("a typed dense exclusion must reject an extra FP8 source pair");
        let VNextError::InvalidModelConfig { field, reason, .. } = error else {
            panic!("expected typed invalid-model-config rejection, got {error}")
        };
        assert_eq!(field, "weights.source_encoding");
        assert!(reason.contains("typed dense exclusion"), "{reason}");
        assert!(reason.contains("in_proj_b"), "{reason}");
    }

    #[test]
    fn rejects_block_fp8_recipe_without_any_execution_pair_before_allocation() {
        let mut config = test_block_fp8_config();
        for weight in &mut config.weights {
            if matches!(
                weight.source_encoding,
                FamilyWeightSourceEncoding::BlockFp8 { .. }
            ) {
                weight.source_encoding = FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::Bf16,
                };
            }
        }
        synchronize_test_block_fp8_exclusions(&mut config);

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("a block-FP8 recipe without an execution pair must fail closed");
        let VNextError::InvalidModelConfig { field, reason, .. } = error else {
            panic!("expected typed invalid-model-config rejection, got {error}")
        };
        assert_eq!(field, "weights.source_encoding");
        assert!(reason.contains("no execution-eligible"), "{reason}");
    }

    #[test]
    fn rejects_block_fp8_value_or_inverse_scale_dtype_drift_before_allocation() {
        for drift in ["values", "scale_inv"] {
            let mut config = test_block_fp8_config();
            let weight = config
                .weights
                .iter_mut()
                .find(|weight| weight.role == "linear_attn_qkv")
                .expect("test contains a block-FP8 qkv projection");
            let FamilyWeightSourceEncoding::BlockFp8 { values, scale_inv } =
                &mut weight.source_encoding
            else {
                panic!("test qkv projection is block-FP8")
            };
            match drift {
                "values" => values.dtype = FamilyBlockFp8Dtype::Bf16,
                "scale_inv" => scale_inv.dtype = FamilyBlockFp8Dtype::F8E4m3,
                _ => unreachable!(),
            }

            let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
                .prepare(&serde_json::to_value(config).unwrap())
                .expect_err("FP8 value/scale dtype drift must fail closed");
            let VNextError::InvalidModelConfig { field, reason, .. } = error else {
                panic!("expected typed invalid-model-config rejection, got {error}")
            };
            assert_eq!(field, "weights.source_encoding", "{drift}: {reason}");
            assert!(reason.contains("identity or dtype"), "{drift}: {reason}");
        }
    }

    #[test]
    fn rejects_block_fp8_metadata_recipe_drift_with_typed_error_before_runtime() {
        let fixture: Value = serde_json::from_slice(QWEN38_FP8_BAD_RECIPE).unwrap();
        assert_eq!(
            fixture["base_fixture"],
            Value::String("qwen38_fp8_config.contract.json".to_owned())
        );
        assert_eq!(
            fixture["case"],
            Value::String("qwen38-fp8-wrong-format".to_owned())
        );
        let pointer = fixture["pointer"]
            .as_str()
            .expect("bad-recipe JSON pointer");
        let replacement = fixture["replacement"].clone();
        let expected_error = fixture["expected_error"]
            .as_str()
            .expect("bad-recipe expected error");

        let mut config = test_block_fp8_config();
        config.hf_config = serde_json::from_slice(QWEN38_FP8_CONFIG).unwrap();
        *config
            .hf_config
            .pointer_mut(pointer)
            .expect("bad-recipe pointer exists in the fixed base fixture") = replacement;

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("mismatched block-FP8 metadata must fail before runtime");
        let VNextError::InvalidModelConfig {
            family_id,
            field,
            reason,
        } = error
        else {
            panic!("expected typed invalid-model-config rejection, got {error}")
        };
        assert_eq!(family_id, FAMILY_ID);
        assert_eq!(field, "hf_config");
        assert!(reason.contains(expected_error), "{reason}");
    }

    #[test]
    fn rejects_three_out_of_contract_qwen38_quantization_fixtures_before_runtime() {
        let fixture: Value = serde_json::from_slice(QWEN38_AWQ_INT4_CONFIG).unwrap();
        let cases = [
            (
                "wrong-bits",
                "/quantization_config/config_groups/group_0/weights/num_bits",
                Value::from(8),
            ),
            (
                "wrong-group-size",
                "/quantization_config/config_groups/group_0/weights/group_size",
                Value::from(64),
            ),
            (
                "activation-quantization",
                "/quantization_config/config_groups/group_0/input_activations",
                serde_json::json!({"num_bits": 8, "type": "int"}),
            ),
        ];
        for (label, pointer, replacement) in cases {
            let mut candidate = fixture.clone();
            *candidate.pointer_mut(pointer).unwrap() = replacement;
            let bytes = serde_json::to_vec(&candidate).unwrap();
            let error = preflight_semantic_config(&bytes)
                .expect_err("out-of-contract quantization must fail before runtime allocation");
            assert!(
                error.contains("typed compressed-tensors Marlin requires"),
                "{label}: {error}"
            );
        }
    }

    fn source_bundle_with_weight_config(
        semantic_config: &serde_json::Value,
        weight_config: &serde_json::Value,
    ) -> ProductionModelSourceBundle {
        let root = tempfile::tempdir().unwrap().keep();
        let semantic = root.join("semantic");
        let tokenizer = root.join("tokenizer");
        let weights = root.join("weights");
        std::fs::create_dir_all(&semantic).unwrap();
        std::fs::create_dir_all(&tokenizer).unwrap();
        std::fs::create_dir_all(&weights).unwrap();
        std::fs::write(
            semantic.join("config.json"),
            serde_json::to_vec(semantic_config).unwrap(),
        )
        .unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(
            tokenizer.join("tokenizer_config.json"),
            br#"{"chat_template":"fixture"}"#,
        )
        .unwrap();
        std::fs::write(
            weights.join("config.json"),
            serde_json::to_vec(weight_config).unwrap(),
        )
        .unwrap();
        std::fs::write(weights.join("model.safetensors"), b"fixture-weights").unwrap();
        let original = |location: &str| OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: location.to_owned(),
            requested_revision: None,
        };
        ProductionModelSourceBundle::open(
            &semantic,
            &tokenizer,
            ProductionWeightArtifact::safetensors_directory(&weights),
            OriginalModelSources {
                semantic: original("semantic"),
                tokenizer: original("tokenizer"),
                weights: original("weights"),
            },
        )
        .unwrap()
    }

    #[test]
    fn semantic_preflight_precedes_safetensors_archive_open() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            br#"{
                "architectures":["Qwen3_5MoeForConditionalGeneration"],
                "model_type":"qwen3_5_moe",
                "text_config":{"model_type":"unsupported_nested_layout"}
            }"#,
        )
        .unwrap();
        std::fs::write(root.path().join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(
            root.path().join("model.safetensors"),
            b"not-a-safetensors-archive",
        )
        .unwrap();
        let original = OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: root.path().display().to_string(),
            requested_revision: None,
        };
        let sources = Arc::new(
            ProductionModelSourceBundle::open(
                root.path(),
                root.path(),
                ProductionWeightArtifact::safetensors_directory(root.path()),
                OriginalModelSources {
                    semantic: original.clone(),
                    tokenizer: original.clone(),
                    weights: original,
                },
            )
            .unwrap(),
        );

        let error = match prepare_from_sources(sources) {
            Ok(_) => panic!("invalid nested semantics unexpectedly reached preparation"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("unsupported Qwen3.5 text model_type"),
            "{error}"
        );
        assert!(!error.contains("safetensors archive"), "{error}");
    }

    #[test]
    fn safetensors_physical_quantization_metadata_is_separate_from_semantics() {
        let semantic = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "text_config": {"model_type": "qwen3_5_moe_text"}
        });
        let quantization = serde_json::json!({
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": false,
            "sym": true
        });
        let bundle = source_bundle_with_weight_config(
            &semantic,
            &serde_json::json!({"quantization_config": quantization}),
        );
        let composed = compose_safetensors_hf_config(&bundle).unwrap();
        assert_eq!(composed["quantization_config"], quantization);
        assert!(semantic.get("quantization_config").is_none());
        assert!(bundle
            .fingerprint(ModelArtifactSourceRole::Weights, "config.json")
            .is_some());
    }

    #[test]
    fn conflicting_semantic_and_physical_quantization_metadata_fails_closed() {
        let semantic = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "quantization_config": {
                "quant_method": "gptq", "bits": 4, "group_size": 64,
                "desc_act": false, "sym": true
            },
            "text_config": {"model_type": "qwen3_5_moe_text"}
        });
        let bundle = source_bundle_with_weight_config(
            &semantic,
            &serde_json::json!({
                "quantization_config": {
                    "quant_method": "gptq", "bits": 4, "group_size": 128,
                    "desc_act": false, "sym": true
                }
            }),
        );
        assert!(compose_safetensors_hf_config(&bundle)
            .unwrap_err()
            .contains("values differ"));
    }

    fn test_weight_dimensions(
        text: &Qwen35TextConfig,
        vocab_size: u64,
        weight: &FamilyWeight,
    ) -> Vec<u64> {
        let hidden = text.hidden_size as u64;
        let qk = text.linear_qk_total_dim() as u64;
        let value = text.linear_value_total_dim() as u64;
        let qkv = qk * 2 + value;
        let full_query = text.full_attention_query_total_dim() as u64;
        let full_query_projection = text.full_attention_q_proj_total_dim() as u64;
        let full_kv = text.full_attention_kv_total_dim() as u64;
        match weight.role.as_str() {
            "embed_tokens" | "lm_head" => vec![vocab_size, hidden],
            "final_norm" | "input_layernorm" | "post_attention_layernorm" => vec![hidden],
            "mlp_gate" | "mlp_up" => vec![
                text.dense_intermediate_size
                    .expect("dense Qwen3.5 test config has intermediate_size")
                    as u64,
                hidden,
            ],
            "mlp_down" => vec![
                hidden,
                text.dense_intermediate_size
                    .expect("dense Qwen3.5 test config has intermediate_size")
                    as u64,
            ],
            "linear_attn_qkv" => vec![qkv, hidden],
            "linear_attn_z" => vec![value, hidden],
            "linear_attn_a" | "linear_attn_b" => {
                vec![text.linear_attention.num_value_heads as u64, hidden]
            }
            "linear_attn_conv" => {
                vec![qkv, 1, text.linear_attention.conv_kernel_dim as u64]
            }
            "linear_attn_a_log" | "linear_attn_dt_bias" => {
                vec![text.linear_attention.num_value_heads as u64]
            }
            "linear_attn_norm" => vec![text.linear_attention.value_head_dim as u64],
            "linear_attn_out" => vec![hidden, value],
            "self_attn_q" => vec![full_query_projection, hidden],
            "self_attn_k" | "self_attn_v" => vec![full_kv, hidden],
            "self_attn_o" => vec![hidden, full_query],
            "self_attn_q_norm" | "self_attn_k_norm" => vec![text.head_dim as u64],
            MOE_ROUTER_ROLE => vec![text.moe.as_ref().unwrap().num_experts as u64, hidden],
            "moe_stacked_gate_proj" | "moe_stacked_up_proj" => vec![
                text.moe.as_ref().unwrap().num_experts as u64,
                text.moe.as_ref().unwrap().moe_intermediate_size as u64,
                hidden,
            ],
            "moe_stacked_down_proj" => vec![
                text.moe.as_ref().unwrap().num_experts as u64,
                hidden,
                text.moe.as_ref().unwrap().moe_intermediate_size as u64,
            ],
            "moe_per_expert_gate_proj_qweight" | "moe_per_expert_up_proj_qweight" => vec![
                text.moe.as_ref().unwrap().moe_intermediate_size as u64,
                hidden,
            ],
            "moe_per_expert_down_proj_qweight" => vec![
                hidden,
                text.moe.as_ref().unwrap().moe_intermediate_size as u64,
            ],
            "moe_shared_expert_gate" => vec![1, hidden],
            "moe_shared_expert_gate_proj" | "moe_shared_expert_up_proj" => vec![
                text.moe.as_ref().unwrap().shared_expert_intermediate_size as u64,
                hidden,
            ],
            "moe_shared_expert_down_proj" => vec![
                hidden,
                text.moe.as_ref().unwrap().shared_expert_intermediate_size as u64,
            ],
            role => panic!("test has no dimensions for Qwen3.5 role {role:?}"),
        }
    }

    fn test_config() -> Qwen35FamilyConfig {
        let hf_config = serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32",
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 32,
                "tie_word_embeddings": true,
                "vocab_size": 32,
                "max_position_embeddings": 128
            }
        });
        let text = Qwen35TextConfig::from_hf_config_value(&hf_config).unwrap();
        let manifest = text.weight_manifest("model").unwrap();
        let mut weights = Vec::new();
        for spec in manifest.global_tensors.iter().filter(|spec| spec.required) {
            let mut weight = FamilyWeight {
                layer_index: None,
                expert_index: None,
                role: spec.role.clone(),
                external_name: spec.name.clone(),
                dimensions: vec![1],
                source_encoding: FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::F32,
                },
            };
            weight.dimensions = test_weight_dimensions(&text, 32, &weight);
            weights.push(weight);
        }
        for layer in &manifest.layers {
            for spec in layer.tensors.iter().filter(|spec| spec.required) {
                let mut weight = FamilyWeight {
                    layer_index: Some(layer.layer_index as u32),
                    expert_index: None,
                    role: spec.role.clone(),
                    external_name: spec.name.clone(),
                    dimensions: vec![1],
                    source_encoding: FamilyWeightSourceEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                };
                weight.dimensions = test_weight_dimensions(&text, 32, &weight);
                weights.push(weight);
            }
        }
        weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
        });
        let tokenizer_config = br#"{
            "chat_template": "{{ messages }}",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0
        }"#;
        Qwen35FamilyConfig {
            metadata: parse_hf_model_semantic_metadata(&hf_config, tokenizer_config).unwrap(),
            hf_config,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_epsilon: CanonicalRational::new(1, 1_000_000).unwrap(),
            weight_format: FamilyWeightFormat::SafetensorsDense,
            weights,
        }
    }

    fn test_block_fp8_config() -> Qwen35FamilyConfig {
        let mut config = test_config();
        config.hf_config["quantization_config"] = serde_json::json!({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "modules_to_not_convert": [
                "model.embed_tokens",
                "lm_head",
                "model.layers.0.input_layernorm"
            ],
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        });
        for weight in &mut config.weights {
            if !matches!(
                weight.role.as_str(),
                "linear_attn_qkv"
                    | "linear_attn_z"
                    | "linear_attn_out"
                    | "self_attn_q"
                    | "self_attn_k"
                    | "self_attn_v"
                    | "self_attn_o"
                    | "mlp_gate"
                    | "mlp_up"
                    | "mlp_down"
            ) {
                continue;
            }
            let scale_name = format!(
                "{}.weight_scale_inv",
                weight
                    .external_name
                    .strip_suffix(".weight")
                    .expect("test linear source ends with .weight")
            );
            let [n, k] = weight.dimensions.as_slice() else {
                panic!("test block-FP8 source must be a matrix")
            };
            weight.source_encoding = FamilyWeightSourceEncoding::BlockFp8 {
                values: FamilyBlockFp8Tensor {
                    external_name: weight.external_name.clone(),
                    dimensions: weight.dimensions.clone(),
                    dtype: FamilyBlockFp8Dtype::F8E4m3,
                },
                scale_inv: FamilyBlockFp8Tensor {
                    external_name: scale_name,
                    dimensions: vec![n.div_ceil(128), k.div_ceil(128)],
                    dtype: FamilyBlockFp8Dtype::Bf16,
                },
            };
        }
        synchronize_test_block_fp8_exclusions(&mut config);
        config.weight_format = FamilyWeightFormat::SafetensorsBlockFp8;
        config
    }

    fn test_official_marlin_block_fp8_config() -> Qwen35FamilyConfig {
        let hf_config: Value = serde_json::from_slice(QWEN38_FP8_CONFIG).unwrap();
        let text = Qwen35TextConfig::from_hf_config_value(&hf_config).unwrap();
        let vocab_size = hf_config["text_config"]["vocab_size"].as_u64().unwrap();
        let max_position_embeddings = hf_config["text_config"]["max_position_embeddings"]
            .as_u64()
            .unwrap();
        let manifest = text.weight_manifest("model.language_model").unwrap();
        let mut weights = Vec::new();
        for (layer_index, spec) in manifest
            .global_tensors
            .iter()
            .map(|spec| (None, spec))
            .chain(manifest.layers.iter().flat_map(|layer| {
                layer
                    .tensors
                    .iter()
                    .map(move |spec| (Some(layer.layer_index as u32), spec))
            }))
            .filter(|(_, spec)| spec.required)
        {
            let mut weight = FamilyWeight {
                layer_index,
                expert_index: None,
                role: spec.role.clone(),
                external_name: spec.name.clone(),
                dimensions: vec![1],
                source_encoding: FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::F16,
                },
            };
            weight.dimensions = test_weight_dimensions(&text, vocab_size, &weight);
            if matches!(
                weight.role.as_str(),
                "linear_attn_qkv"
                    | "linear_attn_z"
                    | "linear_attn_out"
                    | "self_attn_q"
                    | "self_attn_k"
                    | "self_attn_v"
                    | "self_attn_o"
                    | "mlp_gate"
                    | "mlp_up"
                    | "mlp_down"
            ) {
                let [n, k] = weight.dimensions.as_slice() else {
                    panic!("official block-FP8 projection must be a matrix")
                };
                let scale_name = format!(
                    "{}.weight_scale_inv",
                    weight.external_name.strip_suffix(".weight").unwrap()
                );
                weight.source_encoding = FamilyWeightSourceEncoding::BlockFp8 {
                    values: FamilyBlockFp8Tensor {
                        external_name: weight.external_name.clone(),
                        dimensions: weight.dimensions.clone(),
                        dtype: FamilyBlockFp8Dtype::F8E4m3,
                    },
                    scale_inv: FamilyBlockFp8Tensor {
                        external_name: scale_name,
                        dimensions: vec![n.div_ceil(128), k.div_ceil(128)],
                        dtype: FamilyBlockFp8Dtype::Bf16,
                    },
                };
            }
            weights.push(weight);
        }
        weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
        });
        let tokenizer_config = br#"{
            "chat_template": "{{ messages }}",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0
        }"#;
        let mut config = Qwen35FamilyConfig {
            metadata: parse_hf_model_semantic_metadata(&hf_config, tokenizer_config).unwrap(),
            hf_config,
            vocab_size,
            max_position_embeddings,
            rms_norm_epsilon: CanonicalRational::new(1, 1_000_000).unwrap(),
            weight_format: FamilyWeightFormat::SafetensorsBlockFp8,
            weights,
        };
        synchronize_test_block_fp8_exclusions(&mut config);
        config
    }

    fn synchronize_test_block_fp8_exclusions(config: &mut Qwen35FamilyConfig) {
        let dense_projection_modules = config
            .weights
            .iter()
            .filter(|weight| {
                BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES.contains(&weight.role.as_str())
                    && matches!(
                        weight.source_encoding,
                        FamilyWeightSourceEncoding::Dense { .. }
                    )
            })
            .map(|weight| {
                weight
                    .external_name
                    .strip_suffix(".weight")
                    .expect("test projection source ends with .weight")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let modules = config.hf_config["quantization_config"]["modules_to_not_convert"]
            .as_array_mut()
            .expect("test block-FP8 recipe has typed exclusions");
        for module in dense_projection_modules {
            if !modules.iter().any(|value| value.as_str() == Some(&module)) {
                modules.push(Value::String(module));
            }
        }
    }

    fn collect_block_fp8_source_pairs(
        layout: &PhysicalWeightLayout,
        pairs: &mut Vec<(WeightId, WeightId)>,
    ) {
        match layout {
            PhysicalWeightLayout::QuantizedBlockGrid {
                packed_values,
                scales,
                ..
            } => pairs.push((
                packed_values.component_id.clone(),
                scales.component_id.clone(),
            )),
            PhysicalWeightLayout::Composite { parts } => {
                for part in parts {
                    collect_block_fp8_source_pairs(&part.layout, pairs);
                }
            }
            PhysicalWeightLayout::AxisReshapePermutation { values, .. }
            | PhysicalWeightLayout::Indexed { values, .. } => {
                collect_block_fp8_source_pairs(values, pairs);
            }
            PhysicalWeightLayout::ExpertStack { experts, .. } => {
                for expert in experts {
                    collect_block_fp8_source_pairs(expert, pairs);
                }
            }
            PhysicalWeightLayout::Dense { .. }
            | PhysicalWeightLayout::Stored { .. }
            | PhysicalWeightLayout::Quantized { .. }
            | PhysicalWeightLayout::BlockQuantized { .. } => {}
        }
    }

    fn collect_marlin_fp8_execution_leaves(
        layout: &PhysicalWeightLayout,
        leaves: &mut Vec<(WeightId, WeightId, u32)>,
        block_grid_count: &mut usize,
    ) {
        match layout {
            PhysicalWeightLayout::Quantized {
                packed_values,
                scales,
                group_axis,
                ..
            } => leaves.push((
                packed_values.component_id.clone(),
                scales.component_id.clone(),
                *group_axis,
            )),
            PhysicalWeightLayout::QuantizedBlockGrid { .. } => *block_grid_count += 1,
            PhysicalWeightLayout::Composite { parts } => {
                for part in parts {
                    collect_marlin_fp8_execution_leaves(&part.layout, leaves, block_grid_count);
                }
            }
            PhysicalWeightLayout::AxisReshapePermutation { values, .. }
            | PhysicalWeightLayout::Indexed { values, .. } => {
                collect_marlin_fp8_execution_leaves(values, leaves, block_grid_count);
            }
            PhysicalWeightLayout::ExpertStack { experts, .. } => {
                for expert in experts {
                    collect_marlin_fp8_execution_leaves(expert, leaves, block_grid_count);
                }
            }
            PhysicalWeightLayout::Dense { .. }
            | PhysicalWeightLayout::Stored { .. }
            | PhysicalWeightLayout::BlockQuantized { .. } => {}
        }
    }

    fn test_dense_gguf_config() -> Qwen35FamilyConfig {
        let mut config = test_config();
        for weight in &mut config.weights {
            weight.external_name =
                ferrum_to_gguf_with_arch("qwen35", &weight.external_name).unwrap();
            weight.source_encoding = FamilyWeightSourceEncoding::Dense {
                element_type: ElementType::F16,
            };
        }
        config.weight_format = FamilyWeightFormat::GgufNative;
        config
    }

    fn test_moe_gguf_config() -> Qwen35FamilyConfig {
        let hf_config = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32",
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 12,
                "norm_topk_prob": true,
                "tie_word_embeddings": true,
                "vocab_size": 32,
                "max_position_embeddings": 128
            }
        });
        let text = Qwen35TextConfig::from_hf_config_value(&hf_config).unwrap();
        let manifest = text.weight_manifest("model.language_model").unwrap();
        let include = |spec: &&Qwen35WeightSpec| {
            spec.required
                || matches!(
                    spec.role.as_str(),
                    "moe_stacked_gate_proj" | "moe_stacked_up_proj" | "moe_stacked_down_proj"
                )
        };
        let mut weights = Vec::new();
        for spec in manifest.global_tensors.iter().filter(include) {
            let external_name = ferrum_to_gguf_with_arch("qwen35moe", &spec.name).unwrap();
            let mut weight = FamilyWeight {
                layer_index: None,
                expert_index: None,
                role: spec.role.clone(),
                external_name,
                dimensions: vec![1],
                source_encoding: FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::F16,
                },
            };
            weight.dimensions = test_weight_dimensions(&text, 32, &weight);
            weights.push(weight);
        }
        for layer in &manifest.layers {
            for spec in layer.tensors.iter().filter(include) {
                let external_name = ferrum_to_gguf_with_arch("qwen35moe", &spec.name).unwrap();
                let mut weight = FamilyWeight {
                    layer_index: Some(layer.layer_index as u32),
                    expert_index: None,
                    role: spec.role.clone(),
                    external_name,
                    dimensions: vec![1],
                    source_encoding: FamilyWeightSourceEncoding::Dense {
                        element_type: ElementType::F16,
                    },
                };
                weight.dimensions = test_weight_dimensions(&text, 32, &weight);
                weights.push(weight);
            }
        }
        weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
        });
        let tokenizer_config = br#"{
            "chat_template": "{{ messages }}",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0
        }"#;
        Qwen35FamilyConfig {
            metadata: parse_hf_model_semantic_metadata(&hf_config, tokenizer_config).unwrap(),
            hf_config,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_epsilon: CanonicalRational::new(1, 1_000_000).unwrap(),
            weight_format: FamilyWeightFormat::GgufNative,
            weights,
        }
    }

    fn test_moe_gptq_config() -> Qwen35FamilyConfig {
        let mut config = test_moe_gguf_config();
        let text_config = config
            .hf_config
            .get_mut("text_config")
            .and_then(Value::as_object_mut)
            .unwrap();
        text_config.insert("num_experts".to_owned(), Value::from(12));
        text_config.insert("moe_intermediate_size".to_owned(), Value::from(16));
        text_config.insert(
            "shared_expert_intermediate_size".to_owned(),
            Value::from(16),
        );
        text_config.insert(
            "quantization_config".to_owned(),
            serde_json::json!({
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 16,
                "desc_act": false,
                "sym": true
            }),
        );
        let text = Qwen35TextConfig::from_hf_config_value(&config.hf_config).unwrap();
        let manifest = text.weight_manifest("model.language_model").unwrap();
        let mut weights = Vec::new();
        for (layer_index, spec) in manifest
            .global_tensors
            .iter()
            .map(|spec| (None, spec))
            .chain(manifest.layers.iter().flat_map(|layer| {
                layer
                    .tensors
                    .iter()
                    .map(move |spec| (Some(layer.layer_index as u32), spec))
            }))
            .filter(|(_, spec)| spec.required)
        {
            let mut weight = FamilyWeight {
                layer_index,
                expert_index: None,
                role: spec.role.clone(),
                external_name: spec.name.clone(),
                dimensions: vec![1],
                source_encoding: FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::F16,
                },
            };
            weight.dimensions = test_weight_dimensions(&text, 32, &weight);
            weights.push(weight);
        }
        for layer in &manifest.layers {
            for spec in layer.tensors.iter().filter(|spec| {
                matches!(
                    spec.role.as_str(),
                    "moe_per_expert_gate_proj_qweight"
                        | "moe_per_expert_up_proj_qweight"
                        | "moe_per_expert_down_proj_qweight"
                )
            }) {
                for expert in 0..text.moe.as_ref().unwrap().num_experts {
                    let external_name = spec.name.replace('*', &expert.to_string());
                    let mut weight = FamilyWeight {
                        layer_index: Some(layer.layer_index as u32),
                        expert_index: Some(expert as u32),
                        role: spec.role.clone(),
                        external_name: external_name.clone(),
                        dimensions: vec![1],
                        source_encoding: FamilyWeightSourceEncoding::Dense {
                            element_type: ElementType::F16,
                        },
                    };
                    weight.dimensions = test_weight_dimensions(&text, 32, &weight);
                    let [n, k] = weight.dimensions.as_slice() else {
                        panic!("test GPTQ expert source must be a matrix")
                    };
                    let stem = external_name.strip_suffix(".qweight").unwrap();
                    weight.source_encoding = FamilyWeightSourceEncoding::Gptq {
                        qweight: FamilyGptqTensor {
                            external_name: external_name.clone(),
                            dimensions: vec![k / 8, *n],
                            element_type: ElementType::I32,
                        },
                        scales: FamilyGptqTensor {
                            external_name: format!("{stem}.scales"),
                            dimensions: vec![k / 16, *n],
                            element_type: ElementType::F16,
                        },
                        qzeros: FamilyGptqTensor {
                            external_name: format!("{stem}.qzeros"),
                            dimensions: vec![k / 16, n / 8],
                            element_type: ElementType::I32,
                        },
                        g_idx: Some(FamilyGptqTensor {
                            external_name: format!("{stem}.g_idx"),
                            dimensions: vec![*k],
                            element_type: ElementType::I32,
                        }),
                    };
                    weights.push(weight);
                }
            }
        }
        weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str(), left.expert_index).cmp(&(
                right.layer_index,
                right.role.as_str(),
                right.expert_index,
            ))
        });
        config.metadata = parse_hf_model_semantic_metadata(
            &config.hf_config,
            br#"{
                "chat_template": "{{ messages }}",
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0
            }"#,
        )
        .unwrap();
        config.weight_format = FamilyWeightFormat::SafetensorsGptqMarlin;
        config.weights = weights;
        config
    }

    fn test_moe_block_fp8_config() -> Qwen35FamilyConfig {
        let mut config = test_moe_gptq_config();
        config.hf_config["text_config"]["quantization_config"] = serde_json::json!({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "modules_to_not_convert": ["model.language_model.embed_tokens"],
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        });
        for weight in &mut config.weights {
            weight.role = match weight.role.as_str() {
                "moe_per_expert_gate_proj_qweight" => "moe_per_expert_gate_proj".to_owned(),
                "moe_per_expert_up_proj_qweight" => "moe_per_expert_up_proj".to_owned(),
                "moe_per_expert_down_proj_qweight" => "moe_per_expert_down_proj".to_owned(),
                _ => weight.role.clone(),
            };
            if weight.external_name.ends_with(".qweight") {
                weight.external_name = format!(
                    "{}.weight",
                    weight.external_name.strip_suffix(".qweight").unwrap()
                );
            }
            if !BLOCK_FP8_ELIGIBLE_PROJECTION_ROLES.contains(&weight.role.as_str()) {
                continue;
            }
            let [n, k] = weight.dimensions.as_slice() else {
                panic!("test block-FP8 MoE projection must be a matrix")
            };
            let scale_name = format!(
                "{}.weight_scale_inv",
                weight.external_name.strip_suffix(".weight").unwrap()
            );
            weight.source_encoding = FamilyWeightSourceEncoding::BlockFp8 {
                values: FamilyBlockFp8Tensor {
                    external_name: weight.external_name.clone(),
                    dimensions: weight.dimensions.clone(),
                    dtype: FamilyBlockFp8Dtype::F8E4m3,
                },
                scale_inv: FamilyBlockFp8Tensor {
                    external_name: scale_name,
                    dimensions: vec![n.div_ceil(128), k.div_ceil(128)],
                    dtype: FamilyBlockFp8Dtype::Bf16,
                },
            };
        }
        config.weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str(), left.expert_index).cmp(&(
                right.layer_index,
                right.role.as_str(),
                right.expert_index,
            ))
        });
        config.weight_format = FamilyWeightFormat::SafetensorsBlockFp8;
        config
    }

    #[test]
    fn builds_aggregate_gptq_moe_expert_stacks_in_numeric_order() {
        let config = test_moe_gptq_config();
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(&config).unwrap())
            .unwrap();
        let schema = prepared.weight_schema();
        assert_eq!(
            schema.format_id.as_str(),
            "weight-format.safetensors.gptq-marlin-int4"
        );
        assert_eq!(
            schema.quantization_formats(),
            BTreeSet::from([QuantizationFormatId::new(GPTQ_MARLIN_INT4_FORMAT_ID).unwrap()])
        );
        assert_eq!(schema.version, ContractVersion::new(3, 2));
        assert_eq!(
            schema.layout_id.as_str(),
            "weight-layout.qwen3_5.hybrid_moe.gptq_marlin_expert_major.packed_gdn_qkvzba"
        );
        let routed = schema
            .tensor(&moe_weight_id(0, MOE_ROUTED_GATE_UP_ROLE).unwrap())
            .unwrap();
        let PhysicalWeightLayout::Quantized {
            packed_values,
            packed_dimensions,
            scales,
            group_axis,
            ..
        } = &routed.physical_layout
        else {
            panic!("routed gate/up must be one aggregate quantized stack")
        };
        assert_eq!(routed.dimensions, [12, 2, 16, 16]);
        assert_eq!(packed_dimensions, &[12, 2, 16, 8]);
        assert_eq!(*group_axis, 3);
        let packed = schema
            .components
            .iter()
            .find(|component| component.id == packed_values.component_id)
            .unwrap();
        let scale = schema
            .components
            .iter()
            .find(|component| component.id == scales.component_id)
            .unwrap();
        assert_eq!(packed.dimensions, [12, 2, 16, 8]);
        assert_eq!(scale.dimensions, [12, 2, 16, 1]);
        assert_eq!(packed.external_names.len(), 12 * 2 * 3);
        assert_eq!(scale.external_names.len(), 12 * 2);
        for expert in 0..12 {
            let gate =
                format!("model.language_model.layers.0.mlp.experts.{expert}.gate_proj.qweight");
            let up = format!("model.language_model.layers.0.mlp.experts.{expert}.up_proj.qweight");
            let packed_offset = expert * 6;
            assert_eq!(
                packed.external_names[packed_offset..packed_offset + 6],
                [
                    gate.clone(),
                    format!("model.language_model.layers.0.mlp.experts.{expert}.gate_proj.qzeros"),
                    format!("model.language_model.layers.0.mlp.experts.{expert}.gate_proj.g_idx"),
                    up.clone(),
                    format!("model.language_model.layers.0.mlp.experts.{expert}.up_proj.qzeros"),
                    format!("model.language_model.layers.0.mlp.experts.{expert}.up_proj.g_idx"),
                ]
            );
            assert_eq!(
                scale.external_names[expert * 2..expert * 2 + 2],
                [
                    gate.replace(".qweight", ".scales"),
                    up.replace(".qweight", ".scales"),
                ]
            );
        }
        let down = schema
            .tensor(&moe_weight_id(0, MOE_ROUTED_DOWN_ROLE).unwrap())
            .unwrap();
        assert_eq!(schema.physical_component_refs(&routed.id).unwrap().len(), 2);
        assert_eq!(schema.physical_component_refs(&down.id).unwrap().len(), 2);
        let shared_gate_up = schema
            .tensor(&moe_weight_id(0, MOE_SHARED_GATE_UP_ROLE).unwrap())
            .unwrap();
        assert!(matches!(
            &shared_gate_up.physical_layout,
            PhysicalWeightLayout::Dense { component_id }
                if component_id == &moe_component_id(0, MOE_SHARED_GATE_UP_ROLE).unwrap()
        ));
        let shared_component = schema
            .components
            .iter()
            .find(|component| component.id == moe_component_id(0, MOE_SHARED_GATE_UP_ROLE).unwrap())
            .unwrap();
        assert_eq!(shared_component.dimensions, shared_gate_up.dimensions);
        assert_eq!(shared_component.external_names.len(), 2);
    }

    #[test]
    fn builds_aggregate_block_fp8_moe_expert_stacks_in_numeric_order() {
        let mut config = test_moe_block_fp8_config();
        config.weights.reverse();
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(&config).unwrap())
            .unwrap();
        let schema = prepared.weight_schema();
        assert_eq!(
            schema.format_id.as_str(),
            "weight-format.safetensors.fp8-e4m3-block-grid-inverse-scale"
        );
        assert_eq!(
            schema.layout_id.as_str(),
            "weight-layout.qwen3_5.hybrid_moe.fp8_block_grid.expert_major.packed_gdn_qkvzba"
        );
        assert_eq!(schema.version, ContractVersion::new(1, 0));

        let routed = schema
            .tensor(&moe_weight_id(0, MOE_ROUTED_GATE_UP_ROLE).unwrap())
            .unwrap();
        let PhysicalWeightLayout::QuantizedBlockGrid {
            packed_values,
            packed_dimensions,
            scales,
            block_axes,
        } = &routed.physical_layout
        else {
            panic!("routed gate/up must be one aggregate block-FP8 grid")
        };
        assert_eq!(routed.dimensions, [12, 2, 16, 16]);
        assert_eq!(packed_dimensions, &[12, 2, 16, 16]);
        assert_eq!(*block_axes, [2, 3]);
        let packed = schema
            .components
            .iter()
            .find(|component| component.id == packed_values.component_id)
            .unwrap();
        let scale = schema
            .components
            .iter()
            .find(|component| component.id == scales.component_id)
            .unwrap();
        assert_eq!(packed.dimensions, [12, 2, 16, 16]);
        assert_eq!(scale.dimensions, [12, 2, 1, 1]);
        assert_eq!(packed.external_names.len(), 12 * 2);
        assert_eq!(scale.external_names.len(), 12 * 2);
        for expert in 0..12 {
            let gate =
                format!("model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight");
            let up = format!("model.language_model.layers.0.mlp.experts.{expert}.up_proj.weight");
            assert_eq!(
                packed.external_names[expert * 2..expert * 2 + 2],
                [gate.clone(), up.clone()]
            );
            assert_eq!(
                scale.external_names[expert * 2..expert * 2 + 2],
                [
                    gate.replace(".weight", ".weight_scale_inv"),
                    up.replace(".weight", ".weight_scale_inv"),
                ]
            );
        }

        let down = schema
            .tensor(&moe_weight_id(0, MOE_ROUTED_DOWN_ROLE).unwrap())
            .unwrap();
        let PhysicalWeightLayout::QuantizedBlockGrid {
            packed_values,
            packed_dimensions,
            scales,
            block_axes,
        } = &down.physical_layout
        else {
            panic!("routed down must be one aggregate block-FP8 grid")
        };
        assert_eq!(down.dimensions, [12, 16, 16]);
        assert_eq!(packed_dimensions, &[12, 16, 16]);
        assert_eq!(*block_axes, [1, 2]);
        let packed = schema
            .components
            .iter()
            .find(|component| component.id == packed_values.component_id)
            .unwrap();
        let scale = schema
            .components
            .iter()
            .find(|component| component.id == scales.component_id)
            .unwrap();
        assert_eq!(packed.dimensions, [12, 16, 16]);
        assert_eq!(scale.dimensions, [12, 1, 1]);
        for expert in 0..12 {
            let down =
                format!("model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight");
            assert_eq!(packed.external_names[expert], down);
            assert_eq!(
                scale.external_names[expert],
                down.replace(".weight", ".weight_scale_inv")
            );
        }

        let shared_gate_up = schema
            .tensor(&moe_weight_id(0, MOE_SHARED_GATE_UP_ROLE).unwrap())
            .unwrap();
        assert!(matches!(
            &shared_gate_up.physical_layout,
            PhysicalWeightLayout::Composite { parts }
                if parts.len() == 2
                    && parts.iter().all(|part| matches!(
                        part.layout.as_ref(),
                        PhysicalWeightLayout::QuantizedBlockGrid { .. }
                    ))
        ));
        assert_eq!(schema.physical_component_refs(&routed.id).unwrap().len(), 2);
        assert_eq!(schema.physical_component_refs(&down.id).unwrap().len(), 2);
    }

    #[test]
    fn rejects_block_fp8_moe_sidecar_or_grid_drift_before_allocation() {
        for drift in ["sidecar", "grid"] {
            let mut config = test_moe_block_fp8_config();
            let weight = config
                .weights
                .iter_mut()
                .find(|weight| {
                    weight.layer_index == Some(0)
                        && weight.expert_index == Some(0)
                        && weight.role == "moe_per_expert_gate_proj"
                })
                .unwrap();
            let FamilyWeightSourceEncoding::BlockFp8 { scale_inv, .. } =
                &mut weight.source_encoding
            else {
                panic!("test expert projection is block-FP8")
            };
            match drift {
                "sidecar" => {
                    scale_inv.external_name = scale_inv
                        .external_name
                        .replace("gate_proj.weight_scale_inv", "up_proj.weight_scale_inv")
                }
                "grid" => scale_inv.dimensions = vec![2, 1],
                _ => unreachable!(),
            }

            let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
                .prepare(&serde_json::to_value(config).unwrap())
                .expect_err("block-FP8 MoE source drift must fail before allocation");
            let VNextError::InvalidModelConfig { field, reason, .. } = error else {
                panic!("expected typed invalid-model-config rejection, got {error}")
            };
            assert!(field.starts_with("weights"), "{drift}: {field}: {reason}");
            assert!(
                reason.contains(if drift == "sidecar" {
                    "identity or dtype"
                } else {
                    "block grid"
                }),
                "{drift}: {reason}"
            );
        }
    }

    #[test]
    fn rejects_gptq_moe_expert_identity_drift() {
        let mut config = test_moe_gptq_config();
        let mut swapped = 0;
        for weight in config.weights.iter_mut().filter(|weight| {
            weight.layer_index == Some(0) && weight.role == "moe_per_expert_gate_proj_qweight"
        }) {
            match weight.expert_index {
                Some(0) => {
                    weight.expert_index = Some(1);
                    swapped += 1;
                }
                Some(1) => {
                    weight.expert_index = Some(0);
                    swapped += 1;
                }
                _ => {}
            }
        }
        assert_eq!(swapped, 2);

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("expert index must remain bound to its checkpoint tensor name");
        assert!(error.to_string().contains("manifest"), "{error}");
    }

    #[test]
    fn rejects_mixed_gptq_moe_routed_representations() {
        let mut config = test_moe_gptq_config();
        config.weights.push(FamilyWeight {
            layer_index: Some(0),
            expert_index: None,
            role: "moe_stacked_gate_proj".to_owned(),
            external_name: "model.language_model.layers.0.mlp.gate_exps.weight".to_owned(),
            dimensions: vec![12, 16, 16],
            source_encoding: FamilyWeightSourceEncoding::Dense {
                element_type: ElementType::F16,
            },
        });

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .expect_err("GPTQ MoE must reject multiple routed weight representations");
        assert!(
            error.to_string().contains("mixes canonical per-expert"),
            "{error}"
        );
    }

    #[test]
    fn prepares_sparse_moe_program_with_routed_shared_contract() {
        let config = test_moe_gguf_config();
        let descriptor = production_descriptor(&config).unwrap();
        assert_eq!(descriptor.architecture(), "qwen3_5_moe");
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(&config).unwrap())
            .unwrap();
        assert_eq!(
            crate::vnext::moe_capabilities_from_program(&prepared).unwrap(),
            Some(ferrum_types::MoeCapabilities {
                num_experts: 4,
                experts_per_token: 2,
                moe_intermediate_size: Some(8),
            })
        );
        assert_eq!(prepared.family_id().as_str(), FAMILY_ID);
        assert_eq!(
            prepared.external_metadata_id().as_str(),
            MOE_EXTERNAL_METADATA_ID
        );
        let nodes = &prepared.program().blocks()[0].nodes;
        let moe_nodes = nodes
            .iter()
            .filter(|node| node.operation_id.as_str() == ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID)
            .collect::<Vec<_>>();
        assert_eq!(moe_nodes.len(), 4);
        assert!(!nodes
            .iter()
            .any(|node| node.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID));
        assert_eq!(
            nodes
                .iter()
                .filter(|node| { node.operation_id.as_str() == RESIDUAL_ADD_F32_F16_OPERATION_ID })
                .count(),
            4
        );

        let contract = routed_shared_swiglu_moe_contract().unwrap();
        let first = moe_nodes[0];
        assert_eq!(first.required_version, ContractVersion::new(1, 0));
        assert_eq!(first.inputs.len(), contract.descriptor().inputs.len());
        assert_eq!(
            first
                .inputs
                .iter()
                .skip(1)
                .map(|value| value.as_str())
                .collect::<Vec<_>>(),
            [
                "value.weight.layer.0.moe_router",
                "value.weight.layer.0.moe_routed_gate_up",
                "value.weight.layer.0.moe_routed_down",
                "value.weight.layer.0.moe_shared_gate",
                "value.weight.layer.0.moe_shared_gate_up",
                "value.weight.layer.0.moe_shared_down",
            ]
        );
        for (attribute_id, expected) in [
            ("hidden_size", SemanticValue::Unsigned(16)),
            ("expert_count", SemanticValue::Unsigned(4)),
            ("experts_per_token", SemanticValue::Unsigned(2)),
            ("routed_intermediate_size", SemanticValue::Unsigned(8)),
            ("shared_intermediate_size", SemanticValue::Unsigned(12)),
            ("normalize_topk", SemanticValue::Bool(true)),
        ] {
            assert_eq!(
                first
                    .attributes
                    .get(&AttributeId::new(attribute_id).unwrap()),
                Some(&expected)
            );
        }

        let logical_shapes = [
            (MOE_ROUTER_ROLE, vec![4, 16]),
            (MOE_ROUTED_GATE_UP_ROLE, vec![4, 2, 8, 16]),
            (MOE_ROUTED_DOWN_ROLE, vec![4, 16, 8]),
            (MOE_SHARED_GATE_ROLE, vec![1, 16]),
            (MOE_SHARED_GATE_UP_ROLE, vec![2, 12, 16]),
            (MOE_SHARED_DOWN_ROLE, vec![16, 12]),
        ];
        for (role, dimensions) in logical_shapes {
            let weight_id = moe_weight_id(0, role).unwrap();
            let program_weight = prepared
                .program()
                .weights()
                .iter()
                .find(|weight| weight.weight_id == weight_id)
                .unwrap();
            assert_eq!(program_weight.tensor.dimensions, dimensions);
            let schema_weight = prepared.weight_schema().tensor(&weight_id).unwrap();
            assert_eq!(schema_weight.dimensions, dimensions);
        }
        let routed_gate_up = prepared
            .weight_schema()
            .tensor(&moe_weight_id(0, MOE_ROUTED_GATE_UP_ROLE).unwrap())
            .unwrap();
        assert!(matches!(
            routed_gate_up.physical_layout,
            PhysicalWeightLayout::Composite { ref parts } if parts.len() == 2
        ));
        assert_eq!(
            prepared.weight_schema().layout_id.as_str(),
            "weight-layout.qwen3_5.hybrid_moe.gguf.native.packed_gdn_qkvzba"
        );
        assert_eq!(prepared.weight_schema().version, ContractVersion::new(2, 2));
    }

    #[test]
    fn safetensors_programs_keep_the_portable_f16_operation_profile() {
        for config in [test_config(), test_moe_gptq_config()] {
            let is_moe = Qwen35TextConfig::from_hf_config_value(&config.hf_config)
                .unwrap()
                .is_moe();
            let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
                .prepare(&serde_json::to_value(config).unwrap())
                .unwrap();
            let operation_ids = prepared.program().blocks()[0]
                .nodes
                .iter()
                .map(|node| node.operation_id.as_str())
                .collect::<Vec<_>>();

            for (operation_id, expected_count) in [
                (TOKEN_EMBEDDING_OPERATION_ID, 1),
                (GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID, 3),
                (CAUSAL_PAGED_ATTENTION_OPERATION_ID, 1),
                (RMS_NORM_OPERATION_ID, 5),
                (RESIDUAL_ADD_OPERATION_ID, 4),
                (LAST_TOKEN_DENSE_LINEAR_OPERATION_ID, 1),
                (LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, 1),
            ] {
                assert_eq!(
                    operation_ids
                        .iter()
                        .filter(|candidate| **candidate == operation_id)
                        .count(),
                    expected_count,
                    "unexpected {operation_id} count for {:?}",
                    prepared.weight_schema().format_id
                );
            }
            assert_eq!(
                operation_ids
                    .iter()
                    .filter(|candidate| {
                        **candidate
                            == if is_moe {
                                ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID
                            } else {
                                DENSE_SWIGLU_OPERATION_ID
                            }
                    })
                    .count(),
                4
            );
            for forbidden in [
                TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
                GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
                CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
                RMS_NORM_F32_TO_F16_OPERATION_ID,
                RMS_NORM_F32_OPERATION_ID,
                RESIDUAL_ADD_F32_F16_OPERATION_ID,
                LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
                LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
            ] {
                assert!(
                    !operation_ids.contains(&forbidden),
                    "safetensors program leaked F32-master operation {forbidden}"
                );
            }
        }
    }

    #[test]
    fn dense_gguf_program_uses_the_f32_master_operation_profile() {
        let config = test_dense_gguf_config();
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(config).unwrap())
            .unwrap();
        let operation_ids = prepared.program().blocks()[0]
            .nodes
            .iter()
            .map(|node| node.operation_id.as_str())
            .collect::<Vec<_>>();

        for (operation_id, expected_count) in [
            (TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID, 1),
            (GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID, 3),
            (CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID, 1),
            (RMS_NORM_F32_TO_F16_OPERATION_ID, 4),
            (DENSE_SWIGLU_OPERATION_ID, 4),
            (RESIDUAL_ADD_F32_F16_OPERATION_ID, 4),
            (RMS_NORM_F32_OPERATION_ID, 1),
            (LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID, 1),
            (LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID, 1),
        ] {
            assert_eq!(
                operation_ids
                    .iter()
                    .filter(|candidate| **candidate == operation_id)
                    .count(),
                expected_count,
                "unexpected {operation_id} count"
            );
        }
        for forbidden in [
            TOKEN_EMBEDDING_OPERATION_ID,
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            CAUSAL_PAGED_ATTENTION_OPERATION_ID,
            RMS_NORM_OPERATION_ID,
            RESIDUAL_ADD_OPERATION_ID,
            LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
            LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
        ] {
            assert!(
                !operation_ids.contains(&forbidden),
                "GGUF dense program leaked F16 operation {forbidden}"
            );
        }
    }

    #[test]
    fn prepares_dense_hybrid_program_and_rejects_shape_drift() {
        let config = test_config();
        let descriptor = production_descriptor(&config).unwrap();
        assert_eq!(descriptor.architecture(), "qwen3_5");
        assert_eq!(descriptor.hidden_size(), 16);
        assert_eq!(descriptor.layer_count(), 4);
        assert_eq!(descriptor.attention_head_count(), 2);
        assert_eq!(descriptor.kv_head_count(), 1);
        assert_eq!(descriptor.attention_head_dimension(), 4);
        assert_eq!(descriptor.vocabulary_size(), 32);
        assert_eq!(descriptor.maximum_sequence_tokens(), 128);
        assert_eq!(descriptor.execution_dtype(), DataType::FP16);
        assert_eq!(
            descriptor.parameter_count(),
            config
                .weights
                .iter()
                .map(|weight| weight.dimensions.iter().product::<u64>())
                .sum::<u64>()
        );
        let raw = serde_json::to_value(&config).unwrap();
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&raw)
            .unwrap();
        assert_eq!(
            crate::vnext::moe_capabilities_from_program(&prepared).unwrap(),
            None
        );

        assert_eq!(prepared.family_id().as_str(), FAMILY_ID);
        assert_eq!(prepared.program().blocks()[0].nodes.len(), 20);
        assert!(prepared.program().blocks()[0].nodes.iter().all(|node| {
            if node.operation_id.as_str() == LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID {
                matches!(&node.work, ProgramNodeWorkSpec::Fixed)
            } else {
                matches!(
                    &node.work,
                    ProgramNodeWorkSpec::Tokens { value_id, axis: 0 }
                        if node.inputs.iter().chain(&node.outputs).any(|value| value == value_id)
                )
            }
        }));
        let operation_ids = prepared.program().blocks()[0]
            .nodes
            .iter()
            .map(|node| node.operation_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            operation_ids
                .iter()
                .filter(|operation| **operation == RMS_NORM_OPERATION_ID)
                .count(),
            5
        );
        assert_eq!(
            operation_ids
                .iter()
                .filter(|operation| **operation == DENSE_SWIGLU_OPERATION_ID)
                .count(),
            4
        );
        assert_eq!(
            operation_ids
                .iter()
                .filter(|operation| **operation == RESIDUAL_ADD_OPERATION_ID)
                .count(),
            4
        );
        assert_eq!(
            operation_ids
                .iter()
                .filter(|operation| **operation == LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)
                .count(),
            1
        );
        assert_eq!(
            operation_ids
                .iter()
                .filter(|operation| **operation == LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID)
                .count(),
            1
        );
        let greedy = prepared.program().blocks()[0]
            .nodes
            .iter()
            .find(|node| node.operation_id.as_str() == LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID)
            .unwrap();
        assert_eq!(greedy.required_version, ContractVersion::new(3, 0));
        assert_eq!(
            greedy
                .inputs
                .iter()
                .map(ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            [
                "value.output.logits",
                "value.input.greedy_token_mask",
                "value.input.greedy_repetition_token_ids",
                "value.input.greedy_repetition_offsets",
                "value.input.greedy_repetition_penalty",
            ]
        );
        assert_eq!(greedy.outputs[0].as_str(), "value.output.greedy_token");
        assert_eq!(
            prepared
                .program()
                .inputs()
                .iter()
                .map(ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            [
                "value.input.token_ids",
                "value.input.greedy_token_mask",
                "value.input.greedy_repetition_token_ids",
                "value.input.greedy_repetition_offsets",
                "value.input.greedy_repetition_penalty",
            ]
        );
        assert_eq!(
            prepared
                .program()
                .outputs()
                .iter()
                .map(ProgramValueId::as_str)
                .collect::<Vec<_>>(),
            ["value.output.logits", "value.output.greedy_token"]
        );
        assert!(prepared.program().blocks()[0]
            .nodes
            .iter()
            .filter(|node| node.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
            .all(|node| node.inputs.len() == 3));
        let linear_attention = prepared.program().blocks()[0]
            .nodes
            .iter()
            .find(|node| node.operation_id.as_str() == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID)
            .unwrap();
        let linear_inputs = linear_attention
            .inputs
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            linear_attention.required_version,
            ContractVersion::new(6, 0)
        );
        assert_eq!(
            linear_attention
                .attributes
                .get(&AttributeId::new("decay_parameterization").unwrap()),
            Some(&SemanticValue::Text(
                GatedDeltaDecayParameterization::LogRate.as_str().to_owned()
            ))
        );
        assert_eq!(
            linear_attention
                .attributes
                .get(&AttributeId::new("value_head_mapping").unwrap()),
            Some(&SemanticValue::Text(
                GatedDeltaValueHeadMapping::GroupedByKeyHead
                    .as_str()
                    .to_owned()
            ))
        );
        for (ordinal, role) in [
            "input_layernorm",
            PACKED_LINEAR_ATTN_QKVZBA_ROLE,
            "linear_attn_conv",
            "linear_attn_a_log",
            "linear_attn_dt_bias",
            "linear_attn_norm",
            "linear_attn_out",
        ]
        .into_iter()
        .enumerate()
        {
            assert!(linear_inputs[ordinal + 1].ends_with(role));
        }
        assert_eq!(linear_inputs.len(), 10);
        let full_attention = prepared.program().blocks()[0]
            .nodes
            .iter()
            .find(|node| node.operation_id.as_str() == CAUSAL_PAGED_ATTENTION_OPERATION_ID)
            .unwrap();
        let full_inputs = full_attention
            .inputs
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(full_attention.required_version, ContractVersion::new(2, 0));
        assert_eq!(
            full_attention
                .attributes
                .get(&AttributeId::new("maximum_context_tokens").unwrap()),
            Some(&SemanticValue::Unsigned(config.max_position_embeddings))
        );
        for (ordinal, role) in [
            "input_layernorm",
            "self_attn_q",
            "self_attn_k",
            "self_attn_v",
            "self_attn_o",
            "self_attn_q_norm",
            "self_attn_k_norm",
        ]
        .into_iter()
        .enumerate()
        {
            assert!(full_inputs[ordinal + 1].ends_with(role));
        }
        assert_eq!(full_inputs.len(), 9);
        assert_eq!(
            full_attention
                .attributes
                .get(&AttributeId::new("rope_theta").unwrap()),
            Some(&SemanticValue::Rational(
                canonical_positive_f64(10_000.0).unwrap()
            ))
        );
        assert!(!operation_ids.contains(&"operation.logits_projection"));
        assert_eq!(prepared.program().states().len(), 7);
        let first_conv_state = prepared
            .program()
            .states()
            .iter()
            .find(|state| state.id.as_str() == "state.layer.0.conv")
            .unwrap();
        let first_delta_state = prepared
            .program()
            .states()
            .iter()
            .find(|state| state.id.as_str() == "state.layer.0.delta")
            .unwrap();
        assert_eq!(first_conv_state.tensor.element_type, ElementType::F16);
        assert_eq!(first_delta_state.tensor.element_type, ElementType::F32);
        assert_eq!(
            prepared
                .program()
                .states()
                .iter()
                .filter(|state| {
                    state.lifetime == StateLifetime::Sequence
                        && state.capacity_demand == StateCapacityDemand::FixedPerScope
                        && state.initialization == StateInitialization::Zero
                })
                .count(),
            6
        );
        assert_eq!(
            prepared
                .program()
                .states()
                .iter()
                .filter(|state| {
                    state.lifetime == StateLifetime::Sequence
                        && matches!(
                            state.capacity_demand,
                            StateCapacityDemand::TokenScaled { .. }
                        )
                        && state.initialization == StateInitialization::None
                })
                .count(),
            1
        );
        assert!(prepared.program().states().iter().all(|state| {
            state.lifetime == StateLifetime::Sequence
                && match state.capacity_demand {
                    StateCapacityDemand::FixedPerScope => {
                        state.initialization == StateInitialization::Zero
                    }
                    StateCapacityDemand::TokenScaled { .. } => {
                        state.initialization == StateInitialization::None
                    }
                }
        }));
        assert_eq!(
            prepared.program().weights().len(),
            config
                .weights
                .iter()
                .filter(|weight| {
                    !matches!(
                        weight.role.as_str(),
                        "mlp_up" | "linear_attn_z" | "linear_attn_b" | "linear_attn_a"
                    )
                })
                .count()
        );
        assert_eq!(prepared.weight_schema().version, ContractVersion::new(1, 4));
        assert_eq!(
            prepared.weight_schema().layout_id.as_str(),
            "weight-layout.qwen3_5.dense_hybrid.packed_gate_up.packed_gdn_qkvzba"
        );
        for component in prepared
            .weight_schema()
            .components
            .iter()
            .filter(|component| component.external_names.len() == 1)
        {
            let weight = config
                .weights
                .iter()
                .find(|weight| weight.external_name == component.external_names[0])
                .unwrap();
            let expected_type = materialized_element_type(&weight.role);
            assert_eq!(component.physical_element_type(), expected_type);
            if matches!(
                weight.role.as_str(),
                "final_norm"
                    | "input_layernorm"
                    | "post_attention_layernorm"
                    | "self_attn_q_norm"
                    | "self_attn_k_norm"
            ) {
                assert_eq!(
                    component.encoding,
                    WeightEncoding::DenseAffine {
                        element_type: expected_type,
                        scale: CanonicalRational::new(1, 1).unwrap(),
                        bias: CanonicalRational::new(1, 1).unwrap(),
                    }
                );
            } else {
                assert_eq!(
                    component.encoding,
                    WeightEncoding::Dense {
                        element_type: expected_type,
                    }
                );
            }
        }
        let packed_mlp = prepared
            .weight_schema()
            .components
            .iter()
            .filter(|component| component.dimensions == [2, 32, 16])
            .collect::<Vec<_>>();
        assert_eq!(packed_mlp.len(), 4);
        assert!(packed_mlp.iter().all(|component| {
            component.external_names[0].contains("gate_proj")
                && component.external_names[1].contains("up_proj")
                && component.encoding
                    == WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    }
        }));
        let packed_gdn = prepared
            .weight_schema()
            .components
            .iter()
            .filter(|component| {
                component.external_names.len() == 4
                    && component.external_names[0].contains("linear_attn.in_proj_qkv")
            })
            .collect::<Vec<_>>();
        assert_eq!(packed_gdn.len(), 3);
        assert!(packed_gdn.iter().all(|component| {
            component.external_names[0].contains("in_proj_qkv")
                && component.external_names[1].contains("in_proj_z")
                && component.external_names[2].contains("in_proj_b")
                && component.external_names[3].contains("in_proj_a")
                && component.dimensions == [36, 16]
                && component.encoding
                    == WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    }
        }));
        assert_eq!(
            prepared.metadata().special_tokens.eos_token_ids,
            BTreeSet::from([2])
        );
        assert_eq!(prepared.fingerprint().unwrap().len(), 64);

        let mut malformed = config;
        malformed
            .weights
            .iter_mut()
            .find(|weight| weight.role == "embed_tokens")
            .unwrap()
            .dimensions
            .swap(0, 1);
        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(malformed).unwrap())
            .expect_err("same-element axis drift must fail before backend allocation");
        assert!(error.to_string().contains("dimensions"), "{error}");
    }

    #[test]
    fn rejects_temporal_state_dtype_not_implemented_by_vnext_providers() {
        let mut config = test_config();
        config.hf_config["text_config"]["mamba_ssm_dtype"] = serde_json::json!("float16");
        let raw = serde_json::to_value(config).unwrap();

        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&raw)
            .expect_err("F16 temporal state must fail before provider selection");

        assert!(error.to_string().contains("mamba_ssm_dtype"), "{error}");
        assert!(error.to_string().contains("float32"), "{error}");
    }

    #[test]
    fn linear_attention_semantic_inputs_match_the_standard_contract() {
        let config = test_config();
        let prepared = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(&config).unwrap())
            .unwrap();
        let program = prepared.program();
        let node = program.blocks()[0]
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "node.layer.0.attention")
            .unwrap();
        let contract = gated_delta_recurrent_attention_contract().unwrap();
        let descriptor = contract.descriptor();
        let text = Qwen35TextConfig::from_hf_config_value(&config.hf_config).unwrap();
        let conv_channels = (text.linear_qk_total_dim() * 2 + text.linear_value_total_dim()) as u64;
        let conv_component = prepared
            .weight_schema()
            .components
            .iter()
            .find(|component| component.external_names[0].contains("layers.0.linear_attn.conv1d"))
            .unwrap();
        assert_eq!(conv_component.dimensions, [conv_channels, 1, 4]);
        let conv_tensor = prepared
            .weight_schema()
            .tensors
            .iter()
            .find(|tensor| match &tensor.physical_layout {
                PhysicalWeightLayout::Stored { component } => {
                    component.component_id == conv_component.id
                }
                _ => false,
            })
            .unwrap();
        assert_eq!(conv_tensor.dimensions, [conv_channels, 4]);
        assert!(matches!(
            &conv_tensor.physical_layout,
            PhysicalWeightLayout::Stored {
                component: PhysicalWeightComponentBinding {
                    storage: PhysicalStorageLayout::Strided {
                        strides_in_elements,
                        padding: PhysicalWeightPadding::Exact,
                    },
                    ..
                }
            } if strides_in_elements == &[4, 1]
        ));
        let known_tensors = program
            .weights()
            .iter()
            .map(|weight| (&weight.value_id, &weight.tensor))
            .chain(
                program
                    .states()
                    .iter()
                    .map(|state| (&state.value_id, &state.tensor)),
            )
            .collect::<BTreeMap<_, _>>();
        let hidden = tensor_spec(vec![config.max_position_embeddings, 16], ElementType::F16);

        for (ordinal, (value_id, expected)) in
            node.inputs.iter().zip(&descriptor.inputs).enumerate()
        {
            let actual = if ordinal == 0 {
                &hidden
            } else {
                known_tensors.get(value_id).copied().unwrap()
            };
            assert_eq!(
                actual.dimensions.len(),
                expected.dimensions().len(),
                "input[{ordinal}] `{value_id}` rank mismatch: actual={:?}, expected={:?}",
                actual.dimensions,
                expected.dimensions()
            );
            assert!(
                expected.element_types().contains(&actual.element_type),
                "input[{ordinal}] `{value_id}` dtype mismatch: actual={:?}, expected={:?}",
                actual.element_type,
                expected.element_types()
            );
        }
    }

    #[test]
    fn parses_rms_norm_epsilon_without_floating_point_rounding() {
        let expected = CanonicalRational::new(1, 1_000_000).unwrap();
        assert_eq!(
            parse_positive_decimal_rational("0.000001").unwrap(),
            expected
        );
        assert_eq!(parse_positive_decimal_rational("1e-6").unwrap(), expected);
        assert!(parse_positive_decimal_rational("0").is_err());
        assert!(parse_positive_decimal_rational("1.1").is_err());
    }

    #[test]
    fn packs_gate_up_sources_in_schema_order() {
        let directory = tempfile::tempdir().unwrap();
        let tensors = [
            ("gate.weight", [1.0_f32, 2.0, 3.0, 4.0]),
            ("up.weight", [5.0_f32, 6.0, 7.0, 8.0]),
        ];
        let views = tensors
            .iter()
            .map(|(name, values)| {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let bytes: &'static [u8] = Box::leak(bytes);
                (
                    (*name).to_owned(),
                    TensorView::new(Dtype::F32, vec![2, 2], bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            views,
            &None::<std::collections::HashMap<String, String>>,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = SafetensorsArchive::open(directory.path()).unwrap();
        let component = WeightComponentSpec {
            id: WeightId::new("component.layer.0.mlp_gate_up").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec!["gate.weight".to_owned(), "up.weight".to_owned()],
            dimensions: vec![2, 2, 2],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let payload = source.component(&component).unwrap();
        let actual = payload
            .bytes()
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            (1..=8).map(|value| value as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            payload.source_files(),
            ["model.safetensors", "model.safetensors"]
        );
    }

    #[test]
    fn concatenates_unequal_projection_rows_in_schema_order() {
        let directory = tempfile::tempdir().unwrap();
        let tensors = [
            ("qkv.weight", vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]),
            ("z.weight", vec![5.0_f32, 6.0], vec![1, 2]),
            ("b.weight", vec![7.0_f32, 8.0, 9.0, 10.0], vec![2, 2]),
            ("a.weight", vec![11.0_f32, 12.0], vec![1, 2]),
        ];
        let views = tensors
            .iter()
            .map(|(name, values, dimensions)| {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let bytes: &'static [u8] = Box::leak(bytes);
                (
                    (*name).to_owned(),
                    TensorView::new(Dtype::F32, dimensions.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            views,
            &None::<std::collections::HashMap<String, String>>,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let source = SafetensorsArchive::open(directory.path()).unwrap();
        let component = WeightComponentSpec {
            id: WeightId::new("component.layer.0.linear_attn_qkvzba").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec![
                "qkv.weight".to_owned(),
                "z.weight".to_owned(),
                "b.weight".to_owned(),
                "a.weight".to_owned(),
            ],
            dimensions: vec![6, 2],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let payload = source.component(&component).unwrap();
        let actual = payload
            .bytes()
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            (1..=12).map(|value| value as f32).collect::<Vec<_>>()
        );
        let hidden = [2.0_f32, -1.0];
        let fused_projection = actual
            .chunks_exact(2)
            .map(|row| row[0] * hidden[0] + row[1] * hidden[1])
            .collect::<Vec<_>>();
        let separate_projection = tensors
            .iter()
            .flat_map(|(_, values, _)| {
                values
                    .chunks_exact(2)
                    .map(|row| row[0] * hidden[0] + row[1] * hidden[1])
            })
            .collect::<Vec<_>>();
        assert_eq!(fused_projection, separate_projection);
        assert_eq!(payload.dimensions(), [6, 2]);
        assert_eq!(
            payload.external_names(),
            [
                "qkv.weight".to_owned(),
                "z.weight".to_owned(),
                "b.weight".to_owned(),
                "a.weight".to_owned(),
            ]
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.5 semantic metadata and Qwen3.5-4B-Q4_K_M GGUF"]
    fn prepares_real_qwen35_gguf_without_repacking_quantized_components() {
        let semantic_root = std::env::var("FERRUM_TEST_QWEN35_SEMANTIC_DIR")
            .expect("FERRUM_TEST_QWEN35_SEMANTIC_DIR");
        let gguf_path = std::env::var("FERRUM_TEST_GGUF_PATH").expect("FERRUM_TEST_GGUF_PATH");
        let semantic = OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: semantic_root.clone(),
            requested_revision: None,
        };
        let weights = OriginalModelSource {
            kind: ModelSourceKind::LocalFile,
            location: gguf_path.clone(),
            requested_revision: None,
        };
        let sources = Arc::new(
            ProductionModelSourceBundle::open(
                &semantic_root,
                &semantic_root,
                ProductionWeightArtifact::gguf_file(&gguf_path),
                OriginalModelSources {
                    semantic: semantic.clone(),
                    tokenizer: semantic,
                    weights,
                },
            )
            .unwrap(),
        );
        let prepared = prepare_from_sources(sources).unwrap();
        let schema = prepared.family().weight_schema();
        assert_eq!(schema.format_id.as_str(), "weight-format.gguf.native-block");
        assert!(schema
            .quantization_formats()
            .iter()
            .any(|format| format.as_str() == "quantization.gguf.q5-k"));
        assert!(schema
            .quantization_formats()
            .iter()
            .any(|format| format.as_str() == "quantization.gguf.q8-0"));
        let gate_up = schema
            .tensor(&packed_gate_up_weight_id(0).unwrap())
            .unwrap();
        assert_eq!(gate_up.dimensions, [2, 9216, 2560]);
        let gate = schema
            .components
            .iter()
            .find(|component| component.external_names == ["blk.0.ffn_gate.weight"])
            .unwrap();
        let up = schema
            .components
            .iter()
            .find(|component| component.external_names == ["blk.0.ffn_up.weight"])
            .unwrap();
        let PhysicalWeightLayout::Composite { parts } = &gate_up.physical_layout else {
            panic!("GGUF gate/up must preserve two native physical tensors");
        };
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].logical_offsets, [0, 0, 0]);
        assert_eq!(parts[1].logical_offsets, [1, 0, 0]);
        assert_eq!(parts[0].extents, [1, 9216, 2560]);
        assert_eq!(parts[1].extents, [1, 9216, 2560]);
        for (part, source) in parts.iter().zip([gate, up]) {
            assert!(matches!(
                part.layout.as_ref(),
                PhysicalWeightLayout::BlockQuantized {
                    blocks,
                    block_axis: 2,
                    block_padding: PhysicalWeightPadding::Exact,
                } if blocks.component_id == source.id
            ));
        }
        assert!(schema
            .components
            .iter()
            .all(|component| component.external_names.len() == 1));
        for component in &schema.components {
            let first = prepared.weights().component(component).unwrap();
            if matches!(component.encoding, WeightEncoding::BlockQuantized(_)) {
                let second = prepared.weights().component(component).unwrap();
                assert_eq!(first.bytes().as_ptr(), second.bytes().as_ptr());
            }
        }
        let linear_attention = prepared.family().program().blocks()[0]
            .nodes
            .iter()
            .find(|node| {
                node.operation_id.as_str()
                    == GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
            })
            .unwrap();
        assert_eq!(
            linear_attention.required_version,
            ContractVersion::new(1, 0)
        );
        assert_eq!(
            linear_attention
                .attributes
                .get(&AttributeId::new("decay_parameterization").unwrap()),
            Some(&SemanticValue::Text(
                GatedDeltaDecayParameterization::NegativeRate
                    .as_str()
                    .to_owned()
            ))
        );
        assert_eq!(
            linear_attention
                .attributes
                .get(&AttributeId::new("value_head_mapping").unwrap()),
            Some(&SemanticValue::Text(
                GatedDeltaValueHeadMapping::InterleavedByKeyHead
                    .as_str()
                    .to_owned()
            ))
        );
    }

    #[test]
    #[ignore = "requires local Qwen3.5-35B-A3B semantic metadata and Q4_K_S GGUF"]
    fn prepares_real_qwen35_moe_gguf_with_typed_logical_stacks() {
        let semantic_root = std::env::var("FERRUM_TEST_QWEN35_MOE_SEMANTIC_DIR")
            .expect("FERRUM_TEST_QWEN35_MOE_SEMANTIC_DIR");
        let gguf_path = std::env::var("FERRUM_TEST_QWEN35_MOE_GGUF_PATH")
            .expect("FERRUM_TEST_QWEN35_MOE_GGUF_PATH");
        let semantic = OriginalModelSource {
            kind: ModelSourceKind::LocalDirectory,
            location: semantic_root.clone(),
            requested_revision: None,
        };
        let weights = OriginalModelSource {
            kind: ModelSourceKind::LocalFile,
            location: gguf_path.clone(),
            requested_revision: None,
        };
        let sources = Arc::new(
            ProductionModelSourceBundle::open(
                &semantic_root,
                &semantic_root,
                ProductionWeightArtifact::gguf_file(&gguf_path),
                OriginalModelSources {
                    semantic: semantic.clone(),
                    tokenizer: semantic,
                    weights,
                },
            )
            .unwrap(),
        );
        let prepared = prepare_from_sources(sources).unwrap();
        let schema = prepared.family().weight_schema();
        assert_eq!(
            prepared.family().program().blocks()[0]
                .nodes
                .iter()
                .filter(|node| {
                    node.operation_id.as_str() == ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID
                })
                .count(),
            40
        );
        assert_eq!(
            schema
                .tensor(&moe_weight_id(0, MOE_ROUTED_GATE_UP_ROLE).unwrap())
                .unwrap()
                .dimensions,
            [256, 2, 512, 2048]
        );
        assert_eq!(
            schema
                .tensor(&moe_weight_id(0, MOE_ROUTED_DOWN_ROLE).unwrap())
                .unwrap()
                .dimensions,
            [256, 2048, 512]
        );
        let routed_gate_up = schema
            .tensor(&moe_weight_id(0, MOE_ROUTED_GATE_UP_ROLE).unwrap())
            .unwrap();
        assert!(matches!(
            routed_gate_up.physical_layout,
            PhysicalWeightLayout::Composite { ref parts }
                if parts.len() == 2
                    && parts[0].extents == [256, 1, 512, 2048]
                    && parts[1].extents == [256, 1, 512, 2048]
        ));
        let routed_gate = schema
            .components
            .iter()
            .find(|component| component.external_names == ["blk.0.ffn_gate_exps.weight"])
            .unwrap();
        let first = prepared.weights().component(routed_gate).unwrap();
        let second = prepared.weights().component(routed_gate).unwrap();
        assert_eq!(first.bytes().as_ptr(), second.bytes().as_ptr());
    }
}
