use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    AttributeId, CanonicalRational, ContractVersion, ElementType, ModelFamilyId, ModelProgram,
    NodeId, OperationId, ProgramBlock, ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec,
    ProgramValueId, ResolvedTensorLayout, SemanticValue, StateCapacityDemand, StateId,
    StateInitialization, StateLifetime, StateSpec, VNextError, WeightReference,
    CAUSAL_PAGED_ATTENTION_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
    LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID, RESIDUAL_ADD_OPERATION_ID, RMS_NORM_OPERATION_ID,
    ROUTED_SWIGLU_MOE_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
};

use super::config::Qwen3MoeSemanticConfig;
use super::invalid_config;
use super::weights::{
    global_weight_value_id, layer_weight_value_id, Qwen3MoeWeightManifest, EMBED_TOKENS_ROLE,
    FINAL_NORM_ROLE, INPUT_NORM_ROLE, K_NORM_ROLE, K_PROJ_ROLE, LM_HEAD_ROLE, O_PROJ_ROLE,
    POST_ATTENTION_NORM_ROLE, Q_NORM_ROLE, Q_PROJ_ROLE, ROUTED_DOWN_ROLE, ROUTED_GATE_UP_ROLE,
    ROUTER_ROLE, V_PROJ_ROLE,
};

pub(super) fn build_semantic_program(
    family_id: &ModelFamilyId,
    semantic: &Qwen3MoeSemanticConfig,
    manifest: &Qwen3MoeWeightManifest,
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

    let query_features = semantic
        .query_features()
        .map_err(|reason| invalid_config("semantic.query_features", reason))?;
    let kv_features = semantic
        .kv_features()
        .map_err(|reason| invalid_config("semantic.kv_features", reason))?;
    let mut nodes = Vec::with_capacity(
        usize::try_from(semantic.layer_count)
            .unwrap_or_default()
            .saturating_mul(4)
            .saturating_add(4),
    );
    let mut states = Vec::with_capacity(usize::try_from(semantic.layer_count).unwrap_or_default());

    let input_tokens = value_id("value.input.token_ids")?;
    let mut hidden = value_id("value.hidden.embedding")?;
    nodes.push(ProgramNode {
        id: node_id("node.embedding")?,
        operation_id: operation_id(TOKEN_EMBEDDING_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::tokens(input_tokens.clone(), 0),
        inputs: vec![
            input_tokens.clone(),
            global_weight_value_id(EMBED_TOKENS_ROLE)?,
        ],
        outputs: vec![hidden.clone()],
        attributes: BTreeMap::from([
            attribute("hidden_size", semantic.hidden_size)?,
            attribute("vocab_size", semantic.vocabulary_size)?,
        ]),
    });

    for layer_index in 0..semantic.layer_count {
        let layer_index = u32::try_from(layer_index)
            .map_err(|_| invalid_config("semantic.layer_count", "layer index exceeds u32"))?;
        let attention_output = value_id(format!("value.layer.{layer_index}.attention"))?;
        let kv_value = value_id(format!("value.state.layer.{layer_index}.kv"))?;
        let kv_dimensions = vec![2, semantic.kv_head_count, semantic.head_dim];
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
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.attention"))?,
            operation_id: operation_id(CAUSAL_PAGED_ATTENTION_OPERATION_ID)?,
            required_version: ContractVersion::new(2, 0),
            work: ProgramNodeWorkSpec::tokens(hidden.clone(), 0),
            inputs: vec![
                hidden.clone(),
                layer_weight_value_id(layer_index, INPUT_NORM_ROLE)?,
                layer_weight_value_id(layer_index, Q_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, K_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, V_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, O_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, Q_NORM_ROLE)?,
                layer_weight_value_id(layer_index, K_NORM_ROLE)?,
                kv_value,
            ],
            outputs: vec![attention_output.clone()],
            attributes: BTreeMap::from([
                attribute("query_heads", semantic.attention_head_count)?,
                attribute("key_value_heads", semantic.kv_head_count)?,
                attribute("head_dim", semantic.head_dim)?,
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("query_features", query_features)?,
                attribute("query_projection_features", query_features)?,
                attribute("kv_features", kv_features)?,
                attribute("rope_dim", semantic.head_dim)?,
                attribute("maximum_context_tokens", semantic.maximum_sequence_tokens)?,
                attribute("rope_theta", semantic.rope_theta)?,
                attribute("rope_interleaved", false)?,
                attribute("output_gate", false)?,
                attribute("causal", true)?,
                attribute("epsilon", semantic.rms_norm_epsilon)?,
                attribute("layer_index", u64::from(layer_index))?,
            ]),
        });

        let normalized = value_id(format!("value.layer.{layer_index}.post_attention_norm"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.post_attention_norm"))?,
            operation_id: operation_id(RMS_NORM_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(attention_output.clone(), 0),
            inputs: vec![
                attention_output.clone(),
                layer_weight_value_id(layer_index, POST_ATTENTION_NORM_ROLE)?,
            ],
            outputs: vec![normalized.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("epsilon", semantic.rms_norm_epsilon)?,
            ]),
        });

        let moe_output = value_id(format!("value.layer.{layer_index}.moe"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.moe"))?,
            operation_id: operation_id(ROUTED_SWIGLU_MOE_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(normalized.clone(), 0),
            inputs: vec![
                normalized,
                layer_weight_value_id(layer_index, ROUTER_ROLE)?,
                layer_weight_value_id(layer_index, ROUTED_GATE_UP_ROLE)?,
                layer_weight_value_id(layer_index, ROUTED_DOWN_ROLE)?,
            ],
            outputs: vec![moe_output.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("expert_count", semantic.expert_count)?,
                attribute("experts_per_token", semantic.experts_per_token)?,
                attribute(
                    "routed_intermediate_size",
                    semantic.expert_intermediate_size,
                )?,
                attribute("normalize_topk", semantic.normalize_topk)?,
            ]),
        });

        let layer_output = value_id(format!("value.layer.{layer_index}.output"))?;
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.residual"))?,
            operation_id: operation_id(RESIDUAL_ADD_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(attention_output.clone(), 0),
            inputs: vec![attention_output, moe_output],
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

    let projection = if semantic.tie_word_embeddings {
        EMBED_TOKENS_ROLE
    } else {
        LM_HEAD_ROLE
    };
    let logits = value_id("value.output.logits")?;
    nodes.push(ProgramNode {
        id: node_id("node.logits")?,
        operation_id: operation_id(LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::tokens(final_hidden.clone(), 0),
        inputs: vec![final_hidden, global_weight_value_id(projection)?],
        outputs: vec![logits.clone()],
        attributes: BTreeMap::from([
            attribute("hidden_size", semantic.hidden_size)?,
            attribute("out_features", semantic.vocabulary_size)?,
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
        required_version: ContractVersion::new(2, 0),
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
