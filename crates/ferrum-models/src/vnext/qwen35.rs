//! Typed Qwen3.5 dense-hybrid model package for the production vNext path.
//!
//! Preparation reads configuration, tokenizer metadata, and typed weight
//! headers only. Tensor payloads remain untouched until the selected backend
//! executor allocates them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    AttributeId, BlockQuantizationSpec, CanonicalRational, CompositeWeightPart, ContractVersion,
    ElementType, ExternalModelMetadataId, GatedDeltaDecayParameterization,
    GatedDeltaValueHeadMapping, ModelFamilyId, ModelFamilyProvider, ModelFamilyRegistration,
    ModelProgram, ModelSemanticMetadata, NodeId, OperationId, PhysicalStorageLayout,
    PhysicalWeightComponentBinding, PhysicalWeightLayout, PhysicalWeightPadding,
    PreparedModelFamily, ProgramBlock, ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec,
    ProgramValueId, ResolvedTensorLayout, SemanticValue, SpecialTokenCollision,
    SpecialTokenCollisionPolicy, SpecialTokenMetadata, SpecialTokenRole, StateCapacityDemand,
    StateId, StateInitialization, StateLifetime, StateSpec, TemplateMetadata,
    TypedFamilyRegistration, VNextError, WeightComponentRole, WeightComponentSource,
    WeightComponentSpec, WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightReference,
    WeightSchema, WeightTensorSpec, CAUSAL_PAGED_ATTENTION_OPERATION_ID, DENSE_SWIGLU_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
    RESIDUAL_ADD_OPERATION_ID, RMS_NORM_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
};
use ferrum_quantization::gguf::{block_quantization_format, ferrum_to_gguf_with_arch, GgmlDType};
use ferrum_quantization::{GgufWeightComponentSource, SafetensorsArchive};
use ferrum_types::DataType;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::qwen35_config::{Qwen35LayerType, Qwen35TextConfig, Qwen35WeightSpec};
use crate::qwen35_weights::{Qwen35ResolvedWeightSpec, Qwen35WeightInventory};

use super::{
    CausalLanguageModelDescriptor, PreparedProductionModel, ProductionModelSourceBundle,
    ProductionWeightArtifact,
};

pub const FAMILY_ID: &str = "family.qwen3_5.dense_hybrid";
pub const EXTERNAL_METADATA_ID: &str = "hf.architecture.Qwen3_5ForConditionalGeneration";
const DENSE_MATERIALIZED_ELEMENT_TYPE: ElementType = ElementType::F16;
const PACKED_GATE_UP_ROLE: &str = "mlp_gate_up";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemplateInput {
    template: String,
    source_file: String,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecialTokenInput {
    bos_token_id: Option<u32>,
    eos_token_ids: BTreeSet<u32>,
    pad_token_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyWeight {
    layer_index: Option<u32>,
    role: String,
    external_name: String,
    dimensions: Vec<u64>,
    source_encoding: FamilyWeightSourceEncoding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum FamilyWeightSourceEncoding {
    Dense { element_type: ElementType },
    BlockQuantized(BlockQuantizationSpec),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FamilyWeightFormat {
    SafetensorsDense,
    GgufNative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen35FamilyConfig {
    hf_config: Value,
    vocab_size: u64,
    max_position_embeddings: u64,
    rms_norm_epsilon: CanonicalRational,
    template: TemplateInput,
    special_tokens: SpecialTokenInput,
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
        if text.is_moe() {
            return Err(invalid_config(
                "hf_config.text_config.model_type",
                "the first production slice accepts dense Qwen3.5 only",
            ));
        }
        if text.quantization.is_some() {
            return Err(invalid_config(
                "hf_config.quantization_config",
                "the dense Qwen3.5 family package does not accept quantized weights",
            ));
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
        if config.template.template.is_empty()
            || config.template.source_file != "tokenizer_config.json"
            || config.special_tokens.eos_token_ids.is_empty()
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
                || !external_names.insert(weight.external_name.clone())
                || !logical_keys.insert((weight.layer_index, weight.role.clone()))
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
                    ) => {}
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
                let inventory = Qwen35WeightInventory::from_names(
                    config
                        .weights
                        .iter()
                        .map(|weight| weight.external_name.clone()),
                );
                let resolved = inventory
                    .detect_prefix_and_resolve(&text)
                    .map_err(|reason| invalid_config("weights", reason))?;
                let expected =
                    resolved_weight_keys(&resolved.global_tensors, None, text.tie_word_embeddings)
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
                        "resolved tensors do not exactly match the supported dense Qwen3.5 manifest",
                    ));
                }
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
        BTreeSet::from([ExternalModelMetadataId::new(EXTERNAL_METADATA_ID)
            .expect("Qwen3.5 external metadata id is static and valid")])
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
        ExternalModelMetadataId::new(EXTERNAL_METADATA_ID)
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
            FamilyWeightFormat::GgufNative => gguf_weight_schema(config),
        }
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        let text = Self::text_config(config)?;
        let mut weight_refs = Vec::with_capacity(config.weights.len());
        for weight in &config.weights {
            if weight.role == "mlp_up" {
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

        let mut nodes = Vec::with_capacity(text.num_hidden_layers * 4 + 3);
        let embedding = required_weight(config, None, "embed_tokens")?;
        let mut hidden = value_id("value.hidden.embedding")?;
        nodes.push(ProgramNode {
            id: node_id("node.embedding")?,
            operation_id: operation_id(TOKEN_EMBEDDING_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
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
                        "linear_attn_qkv",
                        "linear_attn_z",
                        "linear_attn_b",
                        "linear_attn_a",
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
                        FamilyWeightFormat::SafetensorsDense => (
                            GatedDeltaDecayParameterization::LogRate,
                            GatedDeltaValueHeadMapping::GroupedByKeyHead,
                        ),
                        FamilyWeightFormat::GgufNative => (
                            GatedDeltaDecayParameterization::NegativeRate,
                            GatedDeltaValueHeadMapping::InterleavedByKeyHead,
                        ),
                    };
                    (
                        GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
                        ContractVersion::new(4, 0),
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
                        CAUSAL_PAGED_ATTENTION_OPERATION_ID,
                        ContractVersion::new(2, 0),
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
                operation_id: operation_id(RMS_NORM_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
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

            let intermediate_size = text.dense_intermediate_size.ok_or_else(|| {
                invalid_config(
                    "hf_config.text_config.intermediate_size",
                    "missing dense FFN size",
                )
            })?;
            let mlp_output = value_id(format!("value.layer.{layer_index}.mlp"))?;
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.feed_forward"))?,
                operation_id: operation_id(DENSE_SWIGLU_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::tokens(normalized.clone(), 0),
                inputs: vec![
                    normalized,
                    packed_gate_up_value_id(layer_index as u32)?,
                    weight_value_id(required_weight(
                        config,
                        Some(layer_index as u32),
                        "mlp_down",
                    )?)?,
                ],
                outputs: vec![mlp_output.clone()],
                attributes: BTreeMap::from([
                    attribute("hidden_size", text.hidden_size as u64)?,
                    attribute("intermediate_size", intermediate_size as u64)?,
                ]),
            });

            let layer_output = value_id(format!("value.layer.{layer_index}.output"))?;
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.residual"))?,
                operation_id: operation_id(RESIDUAL_ADD_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
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
            operation_id: operation_id(RMS_NORM_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
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
            operation_id: operation_id(LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(final_hidden.clone(), 0),
            inputs: vec![final_hidden, weight_value_id(projection)?],
            outputs: vec![logits.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", text.hidden_size as u64)?,
                attribute("out_features", config.vocab_size)?,
            ]),
        });

        ModelProgram::new(
            self.family_id.clone(),
            vec![value_id("value.input.token_ids")?],
            vec![ProgramBlock {
                id: "block.decoder".to_owned(),
                nodes,
            }],
            states,
            weight_refs,
            vec![logits],
        )
    }

    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: config.template.template.clone(),
                source_file: config.template.source_file.clone(),
                sha256: config.template.sha256.clone(),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: config.special_tokens.bos_token_id,
                eos_token_ids: config.special_tokens.eos_token_ids.clone(),
                pad_token_id: config.special_tokens.pad_token_id,
                collision_policy: collision_policy(&config.special_tokens)?,
            },
        })
    }
}

fn safetensors_weight_schema(config: &Qwen35FamilyConfig) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::with_capacity(config.weights.len());
    let mut tensors = Vec::with_capacity(config.weights.len());
    for weight in &config.weights {
        if weight.role == "mlp_up" {
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
            physical_layout: physical_weight_layout(
                component_id,
                &physical_dimensions,
                &logical_dimensions,
            )?,
            required: true,
        });
    }
    Ok(WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new("weight-layout.qwen3_5.dense_hybrid.packed_gate_up")?,
        version: ContractVersion::new(1, 2),
        components,
        tensors,
    })
}

fn gguf_weight_schema(config: &Qwen35FamilyConfig) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::with_capacity(config.weights.len());
    let mut tensors = Vec::with_capacity(config.weights.len());
    for weight in &config.weights {
        if weight.role == "mlp_up" {
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
    Ok(WeightSchema {
        format_id: WeightFormatId::new("weight-format.gguf.native-block")?,
        layout_id: WeightLayoutId::new("weight-layout.qwen3_5.dense_hybrid.gguf.native")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    })
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
        FamilyWeightSourceEncoding::Dense { .. } => {
            physical_weight_layout(component_id, &weight.dimensions, logical_dimensions)
        }
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
                blocks: physical_component_binding(
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
    }
}

fn physical_component_binding(
    component_id: WeightId,
    physical_dimensions: &[u64],
    logical_dimensions: &[u64],
) -> Result<PhysicalWeightComponentBinding, VNextError> {
    if physical_dimensions == logical_dimensions {
        return Ok(PhysicalWeightComponentBinding::exact_contiguous(
            component_id,
        ));
    }
    let physical_elements = checked_dimension_product(physical_dimensions)?;
    let logical_elements = checked_dimension_product(logical_dimensions)?;
    if physical_elements != logical_elements {
        return Err(invalid_config(
            "weights.dimensions",
            "GGUF physical reshape changes the logical element count",
        ));
    }
    Ok(PhysicalWeightComponentBinding {
        component_id,
        storage: PhysicalStorageLayout::Strided {
            strides_in_elements: row_major_strides(logical_dimensions)?,
            padding: PhysicalWeightPadding::Exact,
        },
    })
}

fn checked_dimension_product(dimensions: &[u64]) -> Result<u64, VNextError> {
    dimensions.iter().try_fold(1_u64, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid_config("weights.dimensions", "tensor size overflows u64"))
    })
}

fn row_major_strides(dimensions: &[u64]) -> Result<Vec<u64>, VNextError> {
    let mut strides = vec![0_u64; dimensions.len()];
    let mut stride = 1_u64;
    for (axis, extent) in dimensions.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(*extent)
            .ok_or_else(|| invalid_config("weights.dimensions", "tensor stride overflows u64"))?;
    }
    Ok(strides)
}

pub fn prepare_from_model_dir(model_dir: &Path) -> ferrum_types::Result<PreparedProductionModel> {
    let sources = Arc::new(ProductionModelSourceBundle::open_colocated_safetensors(
        model_dir,
    )?);
    prepare_from_sources(sources)
}

pub fn prepare_from_sources(
    sources: Arc<ProductionModelSourceBundle>,
) -> ferrum_types::Result<PreparedProductionModel> {
    match sources.weights() {
        ProductionWeightArtifact::SafetensorsDirectory(weight_root) => {
            let weights = SafetensorsArchive::open(weight_root)?;
            let config = load_safetensors_family_config(&sources, &weights)
                .map_err(ferrum_types::FerrumError::model)?;
            finish_preparation(sources, weights, config)
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
    let hf_config: Value = serde_json::from_slice(sources.config_json())
        .map_err(|error| format!("parse semantic config.json: {error}"))?;
    let text = Qwen35TextConfig::from_hf_config_value(&hf_config)?;
    if text.is_moe() {
        return Err("the first vNext Qwen3.5 production slice requires the dense model".to_owned());
    }
    if text.quantization.is_some() {
        return Err("the dense vNext Qwen3.5 package does not accept quantized weights".to_owned());
    }

    let text_value = hf_config.get("text_config").unwrap_or(&hf_config);
    let vocab_size = required_u64(text_value, "vocab_size")?;
    let max_position_embeddings = required_u64(text_value, "max_position_embeddings")?;
    let rms_norm_epsilon = hf_rms_norm_epsilon(&hf_config)?;
    let tokenizer_config_bytes = sources
        .tokenizer_config_json()
        .ok_or_else(|| "tokenizer source missing tokenizer_config.json".to_owned())?;
    let tokenizer_config: Value = serde_json::from_slice(&tokenizer_config_bytes)
        .map_err(|error| format!("parse tokenizer tokenizer_config.json: {error}"))?;
    let template = tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
        .ok_or_else(|| "tokenizer_config.json missing string chat_template".to_owned())?
        .to_owned();
    let special_tokens = parse_special_tokens(&hf_config, &tokenizer_config)?;

    let inventory = Qwen35WeightInventory::from_names(archive.tensor_names());
    let plan = inventory.detect_prefix_and_resolve(&text)?;
    let mut weights = Vec::new();
    append_resolved_weights(
        &mut weights,
        &plan.global_tensors,
        None,
        text.tie_word_embeddings,
        archive,
    )?;
    for layer in &plan.layers {
        append_resolved_weights(
            &mut weights,
            &layer.tensors,
            Some(layer.layer_index as u32),
            text.tie_word_embeddings,
            archive,
        )?;
    }
    weights.sort_by(|left, right| {
        (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
    });

    Ok(Qwen35FamilyConfig {
        hf_config,
        vocab_size,
        max_position_embeddings,
        rms_norm_epsilon,
        template: TemplateInput {
            sha256: format!("{:x}", Sha256::digest(&tokenizer_config_bytes)),
            template,
            source_file: "tokenizer_config.json".to_owned(),
        },
        special_tokens,
        weight_format: FamilyWeightFormat::SafetensorsDense,
        weights,
    })
}

fn load_gguf_family_config(
    sources: &ProductionModelSourceBundle,
    source: &GgufWeightComponentSource,
) -> Result<Qwen35FamilyConfig, String> {
    let hf_config: Value = serde_json::from_slice(sources.config_json())
        .map_err(|error| format!("parse semantic config.json: {error}"))?;
    let text = Qwen35TextConfig::from_hf_config_value(&hf_config)?;
    if text.is_moe() {
        return Err("the first vNext Qwen3.5 production slice requires the dense model".to_owned());
    }
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
    if architecture != "qwen35" {
        return Err(format!(
            "Qwen3.5 family package requires GGUF architecture qwen35, got {architecture:?}"
        ));
    }

    let text_value = hf_config.get("text_config").unwrap_or(&hf_config);
    let vocab_size = required_u64(text_value, "vocab_size")?;
    let max_position_embeddings = required_u64(text_value, "max_position_embeddings")?;
    let rms_norm_epsilon = hf_rms_norm_epsilon(&hf_config)?;
    let tokenizer_config_bytes = sources
        .tokenizer_config_json()
        .ok_or_else(|| "tokenizer source missing tokenizer_config.json".to_owned())?;
    let tokenizer_config: Value = serde_json::from_slice(tokenizer_config_bytes)
        .map_err(|error| format!("parse tokenizer tokenizer_config.json: {error}"))?;
    let template = tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
        .ok_or_else(|| "tokenizer_config.json missing string chat_template".to_owned())?
        .to_owned();
    let special_tokens = parse_special_tokens(&hf_config, &tokenizer_config)?;

    let manifest = text.weight_manifest("model.language_model")?;
    let mut weights = Vec::new();
    append_gguf_weights(
        &mut weights,
        &manifest.global_tensors,
        None,
        text.tie_word_embeddings,
        source,
    )?;
    for layer in &manifest.layers {
        append_gguf_weights(
            &mut weights,
            &layer.tensors,
            Some(layer.layer_index as u32),
            text.tie_word_embeddings,
            source,
        )?;
    }
    weights.sort_by(|left, right| {
        (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
    });

    Ok(Qwen35FamilyConfig {
        hf_config,
        vocab_size,
        max_position_embeddings,
        rms_norm_epsilon,
        template: TemplateInput {
            sha256: format!("{:x}", Sha256::digest(tokenizer_config_bytes)),
            template,
            source_file: "tokenizer_config.json".to_owned(),
        },
        special_tokens,
        weight_format: FamilyWeightFormat::GgufNative,
        weights,
    })
}

fn append_gguf_weights(
    output: &mut Vec<FamilyWeight>,
    specs: &[Qwen35WeightSpec],
    layer_index: Option<u32>,
    tied_embeddings: bool,
    source: &GgufWeightComponentSource,
) -> Result<(), String> {
    for spec in specs
        .iter()
        .filter(|spec| !(tied_embeddings && spec.role == "lm_head"))
    {
        let external_name = ferrum_to_gguf_with_arch("qwen35", &spec.name).ok_or_else(|| {
            format!(
                "Qwen3.5 GGUF has no typed name mapping for role {:?} source {:?}",
                spec.role, spec.name
            )
        })?;
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
    weights: &SafetensorsArchive,
) -> Result<(), String> {
    for weight in resolved
        .iter()
        .filter(|weight| weight.present && !(tied_embeddings && weight.role == "lm_head"))
    {
        let tensor = weights
            .tensor(&weight.name)
            .map_err(|error| error.to_string())?;
        let source_element_type = tensor.element_type().ok_or_else(|| {
            format!(
                "resolved dense Qwen3.5 tensor {:?} has unsupported dtype {:?}",
                weight.name,
                tensor.dtype()
            )
        })?;
        if !matches!(
            source_element_type,
            ElementType::F16 | ElementType::Bf16 | ElementType::F32
        ) {
            return Err(format!(
                "resolved dense Qwen3.5 tensor {:?} must have a floating-point source dtype, got {source_element_type:?}",
                weight.name
            ));
        }
        output.push(FamilyWeight {
            layer_index,
            role: weight.role.clone(),
            external_name: weight.name.clone(),
            dimensions: tensor.shape().to_vec(),
            source_encoding: FamilyWeightSourceEncoding::Dense {
                element_type: source_element_type,
            },
        });
    }
    Ok(())
}

fn resolved_weight_keys<'a>(
    resolved: &'a [Qwen35ResolvedWeightSpec],
    layer_index: Option<u32>,
    tied_embeddings: bool,
) -> impl Iterator<Item = (Option<u32>, String, String)> + 'a {
    resolved
        .iter()
        .filter(move |weight| weight.present && !(tied_embeddings && weight.role == "lm_head"))
        .map(move |weight| (layer_index, weight.role.clone(), weight.name.clone()))
}

fn resolved_weight_keys_from_config(
    config: &Qwen35FamilyConfig,
) -> BTreeSet<(Option<u32>, String, String)> {
    config
        .weights
        .iter()
        .map(|weight| {
            (
                weight.layer_index,
                weight.role.clone(),
                weight.external_name.clone(),
            )
        })
        .collect()
}

fn validate_gguf_manifest(
    text: &Qwen35TextConfig,
    config: &Qwen35FamilyConfig,
) -> Result<(), VNextError> {
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
        let external_name = ferrum_to_gguf_with_arch("qwen35", &spec.name).ok_or_else(|| {
            invalid_config(
                "weights",
                format!(
                    "GGUF has no typed name mapping for role {:?} source {:?}",
                    spec.role, spec.name
                ),
            )
        })?;
        let key = (layer_index, spec.role.clone(), external_name);
        if spec.required {
            required.insert(key.clone());
        }
        allowed.insert(key);
    }
    let actual = resolved_weight_keys_from_config(config);
    if !required.is_subset(&actual) || !actual.is_subset(&allowed) {
        return Err(invalid_config(
            "weights",
            "resolved GGUF tensors do not exactly match the supported dense Qwen3.5 manifest",
        ));
    }
    Ok(())
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

fn physical_weight_layout(
    component_id: WeightId,
    physical_dimensions: &[u64],
    logical_dimensions: &[u64],
) -> Result<PhysicalWeightLayout, VNextError> {
    if physical_dimensions == logical_dimensions {
        return Ok(PhysicalWeightLayout::Dense { component_id });
    }
    let mut strides_in_elements = vec![0_u64; logical_dimensions.len()];
    let mut stride = 1_u64;
    for (axis, extent) in logical_dimensions.iter().enumerate().rev() {
        strides_in_elements[axis] = stride;
        stride = stride.checked_mul(*extent).ok_or_else(|| {
            invalid_config("weights.dimensions", "logical weight stride overflows u64")
        })?;
    }
    let physical_elements = physical_dimensions
        .iter()
        .try_fold(1_u64, |total, extent| total.checked_mul(*extent));
    if physical_elements != Some(stride) {
        return Err(invalid_config(
            "weights.dimensions",
            "physical and logical weight shapes have different element counts",
        ));
    }
    Ok(PhysicalWeightLayout::Stored {
        component: PhysicalWeightComponentBinding {
            component_id,
            storage: PhysicalStorageLayout::Strided {
                strides_in_elements,
                padding: PhysicalWeightPadding::Exact,
            },
        },
    })
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
        role => {
            return Err(invalid_config(
                "weights.role",
                format!("unsupported dense Qwen3.5 weight role {role:?}"),
            ));
        }
    };
    Ok(expected)
}

fn parse_special_tokens(root: &Value, tokenizer: &Value) -> Result<SpecialTokenInput, String> {
    let bos_token_id = token_id(root, tokenizer, "bos_token")?;
    let pad_token_id = token_id(root, tokenizer, "pad_token")?;
    let eos_value = tokenizer
        .get("eos_token")
        .or_else(|| root.get("eos_token_id"))
        .or_else(|| {
            root.get("text_config")
                .and_then(|value| value.get("eos_token_id"))
        })
        .ok_or_else(|| "model/tokenizer metadata missing eos_token".to_owned())?;
    let eos_values = eos_value
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_else(|| std::slice::from_ref(eos_value));
    let eos_token_ids = eos_values
        .iter()
        .map(|value| resolve_token_id(value, tokenizer))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if eos_token_ids.is_empty() {
        return Err("resolved EOS token set is empty".to_owned());
    }
    Ok(SpecialTokenInput {
        bos_token_id,
        eos_token_ids,
        pad_token_id,
    })
}

fn token_id(root: &Value, tokenizer: &Value, name: &str) -> Result<Option<u32>, String> {
    let id_name = format!("{name}_id");
    tokenizer
        .get(name)
        .or_else(|| tokenizer.get(&id_name))
        .or_else(|| root.get(&id_name))
        .or_else(|| {
            root.get("text_config")
                .and_then(|value| value.get(&id_name))
        })
        .filter(|value| !value.is_null())
        .map(|value| resolve_token_id(value, tokenizer))
        .transpose()
}

fn resolve_token_id(value: &Value, tokenizer: &Value) -> Result<u32, String> {
    if let Some(id) = value.as_u64() {
        return u32::try_from(id).map_err(|_| format!("token id {id} exceeds u32"));
    }
    let content = value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
        .ok_or_else(|| format!("unsupported token metadata {value}"))?;
    tokenizer
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .and_then(|tokens| {
            tokens.iter().find_map(|(id, metadata)| {
                (metadata.get("content").and_then(Value::as_str) == Some(content)).then_some(id)
            })
        })
        .ok_or_else(|| format!("token {content:?} has no added_tokens_decoder id"))?
        .parse::<u32>()
        .map_err(|error| format!("invalid token id for {content:?}: {error}"))
}

fn collision_policy(tokens: &SpecialTokenInput) -> Result<SpecialTokenCollisionPolicy, VNextError> {
    let mut allowed = BTreeSet::new();
    if let Some(bos) = tokens.bos_token_id {
        if tokens.eos_token_ids.contains(&bos) {
            allowed.insert(SpecialTokenCollision::new(
                SpecialTokenRole::Bos,
                SpecialTokenRole::Eos,
            )?);
        }
        if tokens.pad_token_id == Some(bos) {
            allowed.insert(SpecialTokenCollision::new(
                SpecialTokenRole::Bos,
                SpecialTokenRole::Pad,
            )?);
        }
    }
    if tokens
        .pad_token_id
        .is_some_and(|pad| tokens.eos_token_ids.contains(&pad))
    {
        allowed.insert(SpecialTokenCollision::new(
            SpecialTokenRole::Eos,
            SpecialTokenRole::Pad,
        )?);
    }
    Ok(SpecialTokenCollisionPolicy::new(allowed))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let raw = fs::read_to_string(path).map_err(|error| format!("read {path:?}: {error}"))?;
    serde_json::from_str(&raw).map_err(|error| format!("parse {path:?}: {error}"))
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
    let value = parse_positive_decimal_rational_unbounded(raw)?;
    if value.numerator() as u64 > value.denominator() {
        return Err("rms_norm_eps must not exceed one".to_owned());
    }
    Ok(value)
}

fn parse_positive_decimal_rational_unbounded(raw: &str) -> Result<CanonicalRational, String> {
    let normalized = raw.to_ascii_lowercase();
    let (mantissa, exponent) = match normalized.split_once('e') {
        Some((mantissa, exponent)) => (
            mantissa,
            exponent
                .parse::<i32>()
                .map_err(|error| format!("invalid decimal exponent: {error}"))?,
        ),
        None => (normalized.as_str(), 0),
    };
    if mantissa.starts_with('-') {
        return Err("decimal rational must be positive".to_owned());
    }
    let mantissa = mantissa.strip_prefix('+').unwrap_or(mantissa);
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!("invalid decimal rational {raw:?}"));
    }
    let digits = format!("{whole}{fraction}");
    let mut numerator = digits
        .parse::<u128>()
        .map_err(|error| format!("decimal numerator overflows: {error}"))?;
    if numerator == 0 {
        return Err("decimal rational must be positive".to_owned());
    }
    let fractional_digits = i32::try_from(fraction.len())
        .map_err(|_| "rms_norm_eps has too many fractional digits".to_owned())?;
    let scale = fractional_digits
        .checked_sub(exponent)
        .ok_or_else(|| "rms_norm_eps exponent overflows".to_owned())?;
    let denominator = if scale >= 0 {
        10_u128
            .checked_pow(scale as u32)
            .ok_or_else(|| "rms_norm_eps denominator overflows".to_owned())?
    } else {
        numerator = numerator
            .checked_mul(
                10_u128
                    .checked_pow(scale.unsigned_abs())
                    .ok_or_else(|| "rms_norm_eps numerator scale overflows".to_owned())?,
            )
            .ok_or_else(|| "rms_norm_eps numerator overflows".to_owned())?;
        1
    };
    let numerator =
        i64::try_from(numerator).map_err(|_| "rms_norm_eps numerator exceeds i64".to_owned())?;
    let denominator = u64::try_from(denominator)
        .map_err(|_| "rms_norm_eps denominator exceeds u64".to_owned())?;
    CanonicalRational::new(numerator, denominator).map_err(|error| error.to_string())
}

fn canonical_positive_f64(value: f64) -> Result<CanonicalRational, VNextError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid_config(
            "rope_theta",
            "rope theta must be finite and positive",
        ));
    }
    parse_positive_decimal_rational_unbounded(&value.to_string())
        .map_err(|reason| invalid_config("rope_theta", reason))
}

fn weight_key(weight: &FamilyWeight, prefix: &str) -> String {
    scoped_weight_key(weight.layer_index, &weight.role, prefix)
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
    use super::*;
    use ferrum_interfaces::vnext::{
        gated_delta_recurrent_attention_contract, ModelSourceKind, OperationContract,
        OriginalModelSource, OriginalModelSources, WeightComponentSource,
    };
    use half::f16;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    fn test_weight_dimensions(text: &Qwen35TextConfig, weight: &FamilyWeight) -> Vec<u64> {
        let hidden = text.hidden_size as u64;
        let qk = text.linear_qk_total_dim() as u64;
        let value = text.linear_value_total_dim() as u64;
        let qkv = qk * 2 + value;
        let full_query = text.full_attention_query_total_dim() as u64;
        let full_query_projection = text.full_attention_q_proj_total_dim() as u64;
        let full_kv = text.full_attention_kv_total_dim() as u64;
        match weight.role.as_str() {
            "embed_tokens" | "lm_head" => vec![32, 16],
            "final_norm" | "input_layernorm" | "post_attention_layernorm" => vec![16],
            "mlp_gate" | "mlp_up" => vec![32, 16],
            "mlp_down" => vec![16, 32],
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
                role: spec.role.clone(),
                external_name: spec.name.clone(),
                dimensions: vec![1],
                source_encoding: FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::F32,
                },
            };
            weight.dimensions = test_weight_dimensions(&text, &weight);
            weights.push(weight);
        }
        for layer in &manifest.layers {
            for spec in layer.tensors.iter().filter(|spec| spec.required) {
                let mut weight = FamilyWeight {
                    layer_index: Some(layer.layer_index as u32),
                    role: spec.role.clone(),
                    external_name: spec.name.clone(),
                    dimensions: vec![1],
                    source_encoding: FamilyWeightSourceEncoding::Dense {
                        element_type: ElementType::F32,
                    },
                };
                weight.dimensions = test_weight_dimensions(&text, &weight);
                weights.push(weight);
            }
        }
        weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
        });
        let template = "{{ messages }}".to_owned();
        Qwen35FamilyConfig {
            hf_config,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_epsilon: CanonicalRational::new(1, 1_000_000).unwrap(),
            template: TemplateInput {
                sha256: format!("{:x}", Sha256::digest(template.as_bytes())),
                template,
                source_file: "tokenizer_config.json".to_owned(),
            },
            special_tokens: SpecialTokenInput {
                bos_token_id: Some(1),
                eos_token_ids: BTreeSet::from([2]),
                pad_token_id: Some(0),
            },
            weight_format: FamilyWeightFormat::SafetensorsDense,
            weights,
        }
    }

    #[test]
    fn prepares_dense_hybrid_program_and_rejects_shape_drift() {
        let config = test_config();
        let descriptor = production_descriptor(&config).unwrap();
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

        assert_eq!(prepared.family_id().as_str(), FAMILY_ID);
        assert_eq!(prepared.program().blocks()[0].nodes.len(), 19);
        assert!(prepared.program().blocks()[0].nodes.iter().all(|node| {
            matches!(
                &node.work,
                ProgramNodeWorkSpec::Tokens { value_id, axis: 0 }
                    if node.inputs.iter().chain(&node.outputs).any(|value| value == value_id)
            )
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
                .filter(|operation| **operation == LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)
                .count(),
            1
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
            ContractVersion::new(4, 0)
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
            "linear_attn_qkv",
            "linear_attn_z",
            "linear_attn_b",
            "linear_attn_a",
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
        assert_eq!(linear_inputs.len(), 13);
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
        assert_eq!(prepared.program().weights().len(), config.weights.len() - 4);
        assert_eq!(prepared.weight_schema().version, ContractVersion::new(1, 2));
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
        let packed = prepared
            .weight_schema()
            .components
            .iter()
            .filter(|component| component.external_names.len() == 2)
            .collect::<Vec<_>>();
        assert_eq!(packed.len(), 4);
        assert!(packed.iter().all(|component| {
            component.dimensions == [2, 32, 16]
                && component.external_names[0].contains("gate_proj")
                && component.external_names[1].contains("up_proj")
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
            .find(|node| node.operation_id.as_str() == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID)
            .unwrap();
        assert_eq!(
            linear_attention.required_version,
            ContractVersion::new(4, 0)
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
}
