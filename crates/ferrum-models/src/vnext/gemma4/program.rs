use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    AttributeId, CanonicalRational, ContractVersion, ElementType, ModelFamilyId, ModelProgram,
    NodeId, OperationId, ProgramBlock, ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec,
    ProgramValueId, ResolvedTensorLayout, SemanticValue, StateCapacityDemand, StateId,
    StateInitialization, StateLifetime, StateSpec, VNextError, WeightReference,
    CONSTANT_SCALE_OPERATION_ID, DENSE_GEGLU_TANH_OPERATION_ID,
    GEMMA4_CAUSAL_PAGED_ATTENTION_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
    LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, LOGIT_SOFTCAP_OPERATION_ID, RESIDUAL_ADD_OPERATION_ID,
    RMS_NORM_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
};

use super::config::{Gemma4LayerType, Gemma4SemanticConfig};
use super::invalid_config;
use super::weights::{
    global_weight_value_id, layer_weight_value_id, Gemma4WeightManifest, DOWN_PROJ_ROLE,
    EMBED_TOKENS_ROLE, FINAL_NORM_ROLE, GATE_PROJ_ROLE, INPUT_NORM_ROLE, K_NORM_ROLE, K_PROJ_ROLE,
    O_PROJ_ROLE, POST_ATTENTION_NORM_ROLE, POST_FEEDFORWARD_NORM_ROLE, PRE_FEEDFORWARD_NORM_ROLE,
    Q_NORM_ROLE, Q_PROJ_ROLE, UP_PROJ_ROLE, V_PROJ_ROLE,
};

pub(super) fn build_semantic_program(
    family_id: &ModelFamilyId,
    semantic: &Gemma4SemanticConfig,
    manifest: &Gemma4WeightManifest,
) -> Result<ModelProgram, VNextError> {
    let schema = manifest.weight_schema(semantic)?;
    let mut weight_refs = Vec::with_capacity(schema.tensors.len());
    for tensor in schema.tensors {
        let value_id = weight_value_id(&tensor.id.to_string())?;
        weight_refs.push(WeightReference {
            weight_id: tensor.id,
            value_id,
            tensor: tensor_spec(tensor.dimensions, tensor.logical_element_type),
        });
    }

    let layer_count = usize::try_from(semantic.layer_count).unwrap_or_default();
    let mut nodes = Vec::with_capacity(layer_count.saturating_mul(5).saturating_add(6));
    let mut states = Vec::with_capacity(layer_count);

    let input_tokens = value_id("value.input.token_ids")?;
    let unscaled_embedding = value_id("value.hidden.embedding.unscaled")?;
    nodes.push(ProgramNode {
        id: node_id("node.embedding")?,
        operation_id: operation_id(TOKEN_EMBEDDING_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::tokens(input_tokens.clone(), 0),
        inputs: vec![
            input_tokens.clone(),
            global_weight_value_id(EMBED_TOKENS_ROLE)?,
        ],
        outputs: vec![unscaled_embedding.clone()],
        attributes: BTreeMap::from([
            attribute("hidden_size", semantic.hidden_size)?,
            attribute("vocab_size", semantic.vocabulary_size)?,
        ]),
    });
    let mut hidden = value_id("value.hidden.embedding")?;
    nodes.push(ProgramNode {
        id: node_id("node.embedding_scale")?,
        operation_id: operation_id(CONSTANT_SCALE_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::tokens(unscaled_embedding.clone(), 0),
        inputs: vec![unscaled_embedding],
        outputs: vec![hidden.clone()],
        attributes: BTreeMap::from([
            attribute("hidden_size", semantic.hidden_size)?,
            attribute(
                "scale",
                semantic
                    .embedding_scale()
                    .map_err(|reason| invalid_config("semantic.embedding_scale", reason))?,
            )?,
        ]),
    });

    for (index, layer_type) in semantic.layer_types.iter().copied().enumerate() {
        let layer_index = u32::try_from(index)
            .map_err(|_| invalid_config("semantic.layer_count", "layer index exceeds u32"))?;
        let head_dim = semantic.head_dim(layer_type);
        let kv_heads = semantic.kv_head_count(layer_type);
        let query_features = semantic
            .query_features(layer_type)
            .map_err(|reason| invalid_config("semantic.query_features", reason))?;
        let kv_features = semantic
            .kv_features(layer_type)
            .map_err(|reason| invalid_config("semantic.kv_features", reason))?;
        let kv_value = value_id(format!("value.state.layer.{layer_index}.kv"))?;
        let kv_dimensions = vec![2, kv_heads, head_dim];
        let kv_bytes_per_token = kv_dimensions
            .iter()
            .try_fold(2_u64, |bytes, extent| bytes.checked_mul(*extent))
            .ok_or_else(|| invalid_config("states.kv", "KV bytes per token overflow"))?;
        states.push(StateSpec {
            id: state_id(format!("state.layer.{layer_index}.kv"))?,
            value_id: kv_value.clone(),
            tensor: tensor_spec(kv_dimensions, ElementType::F16),
            lifetime: StateLifetime::Sequence,
            capacity_demand: StateCapacityDemand::TokenScaled {
                bytes_per_token: kv_bytes_per_token,
                maximum_tokens: semantic.maximum_sequence_tokens,
            },
            initialization: StateInitialization::None,
        });

        let attention_output = value_id(format!("value.layer.{layer_index}.attention"))?;
        let value_projection_role = if layer_type == Gemma4LayerType::FullAttention {
            K_PROJ_ROLE
        } else {
            V_PROJ_ROLE
        };
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.attention"))?,
            operation_id: operation_id(GEMMA4_CAUSAL_PAGED_ATTENTION_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(hidden.clone(), 0),
            inputs: vec![
                hidden.clone(),
                layer_weight_value_id(layer_index, INPUT_NORM_ROLE)?,
                layer_weight_value_id(layer_index, Q_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, K_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, value_projection_role)?,
                layer_weight_value_id(layer_index, O_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, Q_NORM_ROLE)?,
                layer_weight_value_id(layer_index, K_NORM_ROLE)?,
                kv_value,
                layer_weight_value_id(layer_index, POST_ATTENTION_NORM_ROLE)?,
            ],
            outputs: vec![attention_output.clone()],
            attributes: BTreeMap::from([
                attribute("query_heads", semantic.attention_head_count)?,
                attribute("key_value_heads", kv_heads)?,
                attribute("head_dim", head_dim)?,
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("query_features", query_features)?,
                attribute("query_projection_features", query_features)?,
                attribute("kv_features", kv_features)?,
                attribute("rope_dim", semantic.rope_dim(layer_type))?,
                attribute(
                    "rope_frequency_denominator",
                    semantic.rope_frequency_denominator(layer_type),
                )?,
                attribute("maximum_context_tokens", semantic.maximum_sequence_tokens)?,
                attribute("rope_theta", semantic.rope_theta(layer_type))?,
                attribute("rope_interleaved", false)?,
                attribute("attention_scale", CanonicalRational::new(1, 1)?)?,
                attribute(
                    "sliding_window_tokens",
                    semantic.sliding_window(layer_type).unwrap_or(0),
                )?,
                attribute("value_rms_norm", true)?,
                attribute(
                    "attention_k_eq_v",
                    layer_type == Gemma4LayerType::FullAttention,
                )?,
                attribute("causal", true)?,
                attribute("epsilon", semantic.rms_norm_epsilon)?,
                attribute("layer_index", u64::from(layer_index))?,
            ]),
        });

        let pre_feedforward = value_id(format!("value.layer.{layer_index}.pre_feedforward"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.pre_feedforward_norm"))?,
            operation_id: operation_id(RMS_NORM_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(attention_output.clone(), 0),
            inputs: vec![
                attention_output.clone(),
                layer_weight_value_id(layer_index, PRE_FEEDFORWARD_NORM_ROLE)?,
            ],
            outputs: vec![pre_feedforward.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("epsilon", semantic.rms_norm_epsilon)?,
            ]),
        });

        let feedforward = value_id(format!("value.layer.{layer_index}.feedforward"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.geglu"))?,
            operation_id: operation_id(DENSE_GEGLU_TANH_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(pre_feedforward.clone(), 0),
            inputs: vec![
                pre_feedforward,
                layer_weight_value_id(layer_index, GATE_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, UP_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, DOWN_PROJ_ROLE)?,
            ],
            outputs: vec![feedforward.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("intermediate_size", semantic.intermediate_size)?,
            ]),
        });

        let post_feedforward = value_id(format!("value.layer.{layer_index}.post_feedforward"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.post_feedforward_norm"))?,
            operation_id: operation_id(RMS_NORM_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(feedforward.clone(), 0),
            inputs: vec![
                feedforward,
                layer_weight_value_id(layer_index, POST_FEEDFORWARD_NORM_ROLE)?,
            ],
            outputs: vec![post_feedforward.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("epsilon", semantic.rms_norm_epsilon)?,
            ]),
        });

        let layer_output = value_id(format!("value.layer.{layer_index}.output"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.residual"))?,
            operation_id: operation_id(RESIDUAL_ADD_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(attention_output.clone(), 0),
            inputs: vec![attention_output, post_feedforward],
            outputs: vec![layer_output.clone()],
            attributes: BTreeMap::from([attribute("hidden_size", semantic.hidden_size)?]),
        });
        hidden = layer_output;
    }

    let final_hidden = value_id("value.output.final_hidden")?;
    nodes.push(ProgramNode {
        id: node_id("node.final_norm")?,
        operation_id: operation_id(RMS_NORM_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::tokens(hidden.clone(), 0),
        inputs: vec![hidden, global_weight_value_id(FINAL_NORM_ROLE)?],
        outputs: vec![final_hidden.clone()],
        attributes: BTreeMap::from([
            attribute("hidden_size", semantic.hidden_size)?,
            attribute("epsilon", semantic.rms_norm_epsilon)?,
        ]),
    });

    let uncapped_logits = value_id("value.output.logits.uncapped")?;
    nodes.push(ProgramNode {
        id: node_id("node.logits")?,
        operation_id: operation_id(LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::tokens(final_hidden.clone(), 0),
        inputs: vec![final_hidden, global_weight_value_id(EMBED_TOKENS_ROLE)?],
        outputs: vec![uncapped_logits.clone()],
        attributes: BTreeMap::from([
            attribute("hidden_size", semantic.hidden_size)?,
            attribute("out_features", semantic.vocabulary_size)?,
        ]),
    });
    let logits = value_id("value.output.logits")?;
    nodes.push(ProgramNode {
        id: node_id("node.logit_softcap")?,
        operation_id: operation_id(LOGIT_SOFTCAP_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs: vec![uncapped_logits],
        outputs: vec![logits.clone()],
        attributes: BTreeMap::from([
            attribute("vocab_size", semantic.vocabulary_size)?,
            attribute("cap", semantic.final_logit_softcap)?,
        ]),
    });

    let greedy_mask = value_id("value.input.greedy_token_mask")?;
    let repetition_ids = value_id("value.input.greedy_repetition_token_ids")?;
    let repetition_offsets = value_id("value.input.greedy_repetition_offsets")?;
    let repetition_penalty = value_id("value.input.greedy_repetition_penalty")?;
    let greedy_token = value_id("value.output.greedy_token")?;
    nodes.push(ProgramNode {
        id: node_id("node.greedy_token")?,
        operation_id: operation_id(LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID)?,
        required_version: ContractVersion::new(3, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs: vec![
            logits.clone(),
            greedy_mask.clone(),
            repetition_ids.clone(),
            repetition_offsets.clone(),
            repetition_penalty.clone(),
        ],
        outputs: vec![greedy_token.clone()],
        attributes: BTreeMap::from([attribute("vocab_size", semantic.vocabulary_size)?]),
    });

    ModelProgram::new(
        family_id.clone(),
        vec![
            input_tokens,
            greedy_mask,
            repetition_ids,
            repetition_offsets,
            repetition_penalty,
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

fn weight_value_id(weight_id: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.{weight_id}"))
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
