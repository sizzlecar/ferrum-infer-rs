//! Typed Qwen3.5 dense-hybrid model package for the production vNext path.
//!
//! Preparation reads configuration, tokenizer metadata, and safetensors
//! headers only. Tensor payloads remain untouched until the selected backend
//! executor allocates them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use ferrum_interfaces::vnext::{
    AttributeId, ContractVersion, ElementType, ExternalModelMetadataId, ModelFamilyId,
    ModelFamilyProvider, ModelFamilyRegistration, ModelProgram, ModelSemanticMetadata, NodeId,
    OperationId, PhysicalWeightLayout, PreparedModelFamily, ProgramBlock, ProgramNode,
    ProgramTensorSpec, ProgramValueId, ResolvedTensorLayout, SemanticValue, SpecialTokenCollision,
    SpecialTokenCollisionPolicy, SpecialTokenMetadata, SpecialTokenRole, StateCapacityDemand,
    StateId, StateLifetime, StateSpec, TemplateMetadata, TypedFamilyRegistration, VNextError,
    WeightComponentRole, WeightComponentSpec, WeightEncoding, WeightFormatId, WeightId,
    WeightLayoutId, WeightReference, WeightSchema, WeightTensorSpec, TOKEN_EMBEDDING_OPERATION_ID,
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
        let components = config
            .weights
            .iter()
            .map(|weight| {
                Ok(WeightComponentSpec {
                    id: component_id(weight)?,
                    role: WeightComponentRole::Values,
                    external_names: vec![weight.external_name.clone()],
                    dimensions: weight.dimensions.clone(),
                    encoding: WeightEncoding::Dense {
                        element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                    },
                    required: true,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let tensors = config
            .weights
            .iter()
            .map(|weight| {
                Ok(WeightTensorSpec {
                    id: weight_id(weight)?,
                    dimensions: weight.dimensions.clone(),
                    logical_element_type: DENSE_MATERIALIZED_ELEMENT_TYPE,
                    physical_layout: PhysicalWeightLayout::Dense {
                        component_id: component_id(weight)?,
                    },
                    required: true,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        Ok(WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new("weight-layout.qwen3_5.dense_hybrid")?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        })
    }

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError> {
        let text = Self::text_config(config)?;
        let weight_refs = config
            .weights
            .iter()
            .map(|weight| {
                Ok(WeightReference {
                    weight_id: weight_id(weight)?,
                    value_id: weight_value_id(weight)?,
                    tensor: tensor_spec(weight.dimensions.clone(), DENSE_MATERIALIZED_ELEMENT_TYPE),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;

        let mut nodes = Vec::with_capacity(text.num_hidden_layers * 2 + 2);
        let embedding = required_weight(config, None, "embed_tokens")?;
        let mut hidden = value_id("value.hidden.embedding")?;
        nodes.push(ProgramNode {
            id: node_id("node.embedding")?,
            operation_id: operation_id(TOKEN_EMBEDDING_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
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
            let mut attention_inputs = vec![hidden.clone()];
            attention_inputs.extend(
                layer_weights(config, layer_index as u32, false)
                    .map(weight_value_id)
                    .collect::<Result<Vec<_>, _>>()?,
            );

            let (operation, mut attributes) = match layer_type {
                Qwen35LayerType::LinearAttention => {
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
                        "operation.gated_delta_recurrent_attention",
                        BTreeMap::from([
                            attribute("key_heads", text.linear_attention.num_key_heads as u64)?,
                            attribute("value_heads", text.linear_attention.num_value_heads as u64)?,
                            attribute("key_head_dim", text.linear_attention.key_head_dim as u64)?,
                            attribute(
                                "value_head_dim",
                                text.linear_attention.value_head_dim as u64,
                            )?,
                        ]),
                    )
                }
                Qwen35LayerType::FullAttention => {
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
                        "operation.causal_paged_attention",
                        BTreeMap::from([
                            attribute("query_heads", text.num_attention_heads as u64)?,
                            attribute("key_value_heads", text.num_key_value_heads as u64)?,
                            attribute("head_dim", text.head_dim as u64)?,
                            attribute("causal", true)?,
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
                required_version: ContractVersion::new(1, 0),
                inputs: attention_inputs,
                outputs: vec![attention_output.clone()],
                attributes,
            });

            let layer_output = value_id(format!("value.layer.{layer_index}.output"))?;
            let mut mlp_inputs = vec![attention_output];
            mlp_inputs.extend(
                layer_weights(config, layer_index as u32, true)
                    .map(weight_value_id)
                    .collect::<Result<Vec<_>, _>>()?,
            );
            nodes.push(ProgramNode {
                id: node_id(format!("node.layer.{layer_index}.feed_forward"))?,
                operation_id: operation_id("operation.dense_swiglu")?,
                required_version: ContractVersion::new(1, 0),
                inputs: mlp_inputs,
                outputs: vec![layer_output.clone()],
                attributes: BTreeMap::from([
                    attribute("layer_index", layer_index as u64)?,
                    attribute("hidden_size", text.hidden_size as u64)?,
                    attribute(
                        "intermediate_size",
                        text.dense_intermediate_size.unwrap_or_default() as u64,
                    )?,
                ]),
            });
            hidden = layer_output;
        }

        let final_norm = required_weight(config, None, "final_norm")?;
        let projection = config
            .weights
            .iter()
            .find(|weight| weight.layer_index.is_none() && weight.role == "lm_head")
            .unwrap_or(embedding);
        let logits = value_id("value.output.logits")?;
        nodes.push(ProgramNode {
            id: node_id("node.logits")?,
            operation_id: operation_id("operation.logits_projection")?,
            required_version: ContractVersion::new(1, 0),
            inputs: vec![
                hidden,
                weight_value_id(final_norm)?,
                weight_value_id(projection)?,
            ],
            outputs: vec![logits.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", text.hidden_size as u64)?,
                attribute("vocab_size", config.vocab_size)?,
                attribute("tied_embeddings", text.tie_word_embeddings)?,
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

fn weight_key(weight: &FamilyWeight, prefix: &str) -> String {
    match weight.layer_index {
        Some(layer) => format!("{prefix}.layer.{layer}.{}", weight.role),
        None => format!("{prefix}.global.{}", weight.role),
    }
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
            weight.dimensions = vec![expected_weight_elements(&text, 32, &weight).unwrap()];
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
                weight.dimensions = vec![expected_weight_elements(&text, 32, &weight).unwrap()];
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
        assert_eq!(prepared.program().blocks()[0].nodes.len(), 10);
        assert_eq!(prepared.program().states().len(), 7);
        assert_eq!(prepared.program().weights().len(), config.weights.len());
        assert!(prepared
            .weight_schema()
            .components
            .iter()
            .all(|component| component.physical_element_type() == ElementType::F16));
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
}
