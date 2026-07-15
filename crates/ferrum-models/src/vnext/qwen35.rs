//! Typed Qwen3.5 dense-hybrid model package for the production vNext path.
//!
//! Preparation reads configuration, tokenizer metadata, and safetensors
//! headers only. Tensor payloads remain untouched until the selected backend
//! executor allocates them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use ferrum_interfaces::vnext::{
    AttributeId, CanonicalRational, ContractVersion, ElementType, ExternalModelMetadataId,
    ModelFamilyId, ModelFamilyProvider, ModelFamilyRegistration, ModelProgram,
    ModelSemanticMetadata, NodeId, OperationId, PhysicalWeightLayout, PreparedModelFamily,
    ProgramBlock, ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec, ProgramValueId,
    ResolvedTensorLayout, SemanticValue, SpecialTokenCollision, SpecialTokenCollisionPolicy,
    SpecialTokenMetadata, SpecialTokenRole, StateCapacityDemand, StateId, StateLifetime, StateSpec,
    TemplateMetadata, TypedFamilyRegistration, VNextError, WeightComponentRole,
    WeightComponentSpec, WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightReference,
    WeightSchema, WeightTensorSpec, CAUSAL_PAGED_ATTENTION_OPERATION_ID, DENSE_SWIGLU_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
    RESIDUAL_ADD_OPERATION_ID, RMS_NORM_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
};
use ferrum_quantization::SafetensorsArchive;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::qwen35_config::{Qwen35LayerType, Qwen35TextConfig};
use crate::qwen35_weights::{Qwen35ResolvedWeightSpec, Qwen35WeightInventory};

use super::PreparedProductionModel;

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
            let actual_elements = weight
                .dimensions
                .iter()
                .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
                .ok_or_else(|| invalid_config("weights.dimensions", "tensor size overflows u64"))?;
            let expected_elements = expected_weight_elements(&text, config.vocab_size, weight)?;
            if actual_elements != expected_elements {
                return Err(invalid_config(
                    "weights.dimensions",
                    format!(
                        "role {:?} has {actual_elements} elements, expected {expected_elements}",
                        weight.role
                    ),
                ));
            }
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
        let actual = config
            .weights
            .iter()
            .map(|weight| {
                (
                    weight.layer_index,
                    weight.role.clone(),
                    weight.external_name.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(invalid_config(
                "weights",
                "resolved tensors do not exactly match the supported dense Qwen3.5 manifest",
            ));
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
        let mut components = Vec::with_capacity(config.weights.len());
        let mut tensors = Vec::with_capacity(config.weights.len());
        for weight in &config.weights {
            if weight.role == "mlp_up" {
                continue;
            }
            let (component_id, tensor_id, external_names, dimensions) = if weight.role == "mlp_gate"
            {
                let layer_index = weight.layer_index.ok_or_else(|| {
                    invalid_config("weights.mlp_gate", "dense gate weight has no layer")
                })?;
                let up = required_weight(config, Some(layer_index), "mlp_up")?;
                (
                    packed_gate_up_component_id(layer_index)?,
                    packed_gate_up_weight_id(layer_index)?,
                    vec![weight.external_name.clone(), up.external_name.clone()],
                    packed_gate_up_dimensions(weight, up)?,
                )
            } else {
                (
                    component_id(weight)?,
                    weight_id(weight)?,
                    vec![weight.external_name.clone()],
                    weight.dimensions.clone(),
                )
            };
            components.push(WeightComponentSpec {
                id: component_id.clone(),
                role: WeightComponentRole::Values,
                external_names,
                dimensions: dimensions.clone(),
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
                dimensions,
                logical_element_type: element_type,
                physical_layout: PhysicalWeightLayout::Dense { component_id },
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
                        weight.dimensions.clone(),
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
                            ElementType::F16,
                        ),
                        lifetime: StateLifetime::Sequence,
                        capacity_demand: StateCapacityDemand::FixedPerScope,
                    });
                    (
                        GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
                        ContractVersion::new(2, 0),
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

pub fn prepare_from_model_dir(model_dir: &Path) -> ferrum_types::Result<PreparedProductionModel> {
    let weights = SafetensorsArchive::open(model_dir)?;
    let config =
        load_family_config(model_dir, &weights).map_err(ferrum_types::FerrumError::model)?;
    let raw = serde_json::to_value(config)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let provider = Qwen35FamilyProvider::new()
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    let family = TypedFamilyRegistration::new(provider)
        .prepare(&raw)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))?;
    Ok(PreparedProductionModel::new(family, weights))
}

fn load_family_config(
    model_dir: &Path,
    archive: &SafetensorsArchive,
) -> Result<Qwen35FamilyConfig, String> {
    let hf_config = read_json(&model_dir.join("config.json"))?;
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
    let tokenizer_config = read_json(&model_dir.join("tokenizer_config.json"))?;
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
            sha256: format!("{:x}", Sha256::digest(template.as_bytes())),
            template,
            source_file: "tokenizer_config.json".to_owned(),
        },
        special_tokens,
        weights,
    })
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

fn expected_weight_elements(
    text: &Qwen35TextConfig,
    vocab_size: u64,
    weight: &FamilyWeight,
) -> Result<u64, VNextError> {
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
        "embed_tokens" | "lm_head" => vocab_size.checked_mul(hidden),
        "final_norm" | "input_layernorm" | "post_attention_layernorm" => Some(hidden),
        "linear_attn_qkv" => conv_channels.checked_mul(hidden),
        "linear_attn_z" => value_total.checked_mul(hidden),
        "linear_attn_a" | "linear_attn_b" => value_heads.checked_mul(hidden),
        "linear_attn_conv" => {
            conv_channels.checked_mul(text.linear_attention.conv_kernel_dim as u64)
        }
        "linear_attn_a_log" | "linear_attn_dt_bias" => Some(value_heads),
        "linear_attn_norm" => Some(text.linear_attention.value_head_dim as u64),
        "linear_attn_out" => hidden.checked_mul(value_total),
        "self_attn_q" => full_query.checked_mul(hidden),
        "self_attn_k" | "self_attn_v" => full_kv.checked_mul(hidden),
        "self_attn_o" => hidden.checked_mul(full_query_without_gate),
        "self_attn_q_norm" | "self_attn_k_norm" => Some(text.head_dim as u64),
        "mlp_gate" | "mlp_up" => intermediate.checked_mul(hidden),
        "mlp_down" => hidden.checked_mul(intermediate),
        role => {
            return Err(invalid_config(
                "weights.role",
                format!("unsupported dense Qwen3.5 weight role {role:?}"),
            ));
        }
    };
    expected.ok_or_else(|| invalid_config("weights.dimensions", "tensor size overflows u64"))
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
    use ferrum_interfaces::vnext::WeightComponentSource;
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
            "linear_attn_conv" => vec![qkv, text.linear_attention.conv_kernel_dim as u64],
            "linear_attn_a_log" | "linear_attn_dt_bias" => {
                vec![text.linear_attention.num_value_heads as u64]
            }
            "linear_attn_norm" => vec![text.linear_attention.value_head_dim as u64],
            "linear_attn_out" => vec![hidden, value],
            "self_attn_q" => vec![full_query_projection, hidden],
            "self_attn_k" | "self_attn_v" => vec![full_kv, hidden],
            "self_attn_o" => vec![hidden, full_query],
            "self_attn_q_norm" | "self_attn_k_norm" => vec![text.head_dim as u64],
            _ => vec![expected_weight_elements(text, 32, weight).unwrap()],
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
            weights,
        }
    }

    #[test]
    fn prepares_dense_hybrid_program_and_rejects_shape_drift() {
        let config = test_config();
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
            ContractVersion::new(2, 0)
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
        malformed.weights[0].dimensions = vec![1];
        let error = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
            .prepare(&serde_json::to_value(malformed).unwrap())
            .expect_err("shape drift must fail before backend allocation");
        assert!(error.to_string().contains("elements, expected"), "{error}");
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
}
