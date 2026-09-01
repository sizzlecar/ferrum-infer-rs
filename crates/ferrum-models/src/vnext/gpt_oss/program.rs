use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    AttributeId, CanonicalRational, ContractVersion, ElementType, ModelFamilyId, ModelProgram,
    NodeId, OperationId, ProgramBlock, ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec,
    ProgramValueId, ResolvedTensorLayout, SemanticValue, StateCapacityDemand, StateId,
    StateInitialization, StateLifetime, StateSpec, VNextError, WeightReference,
    GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID, GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID,
    LAST_TOKEN_DENSE_LINEAR_OPERATION_ID, LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
    RESIDUAL_ADD_OPERATION_ID, RMS_NORM_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
};

use super::config::{GptOssLayerType, GptOssSemanticConfig};
use super::invalid_config;
use super::weights::{
    global_weight_value_id, layer_weight_value_id, GptOssWeightManifest, ATTENTION_SINKS_ROLE,
    EMBED_TOKENS_ROLE, FINAL_NORM_ROLE, INPUT_NORM_ROLE, K_BIAS_ROLE, K_PROJ_ROLE, LM_HEAD_ROLE,
    O_BIAS_ROLE, O_PROJ_ROLE, POST_ATTENTION_NORM_ROLE, Q_BIAS_ROLE, Q_PROJ_ROLE,
    ROUTED_DOWN_BIAS_ROLE, ROUTED_DOWN_ROLE, ROUTED_GATE_UP_BIAS_ROLE, ROUTED_GATE_UP_ROLE,
    ROUTER_BIAS_ROLE, ROUTER_ROLE, V_BIAS_ROLE, V_PROJ_ROLE,
};

pub(super) fn build_semantic_program(
    family_id: &ModelFamilyId,
    semantic: &GptOssSemanticConfig,
    manifest: &GptOssWeightManifest,
) -> Result<ModelProgram, VNextError> {
    let schema = manifest.weight_schema(semantic)?;
    let mut weight_refs = Vec::with_capacity(schema.tensors.len());
    for tensor in schema.tensors {
        weight_refs.push(WeightReference {
            value_id: weight_value_id(tensor.id.as_str())?,
            weight_id: tensor.id,
            tensor: tensor_spec(tensor.dimensions, tensor.logical_element_type),
        });
    }

    let query_features = semantic
        .query_features()
        .map_err(|reason| invalid_config("semantic.query_features", reason))?;
    let kv_features = semantic
        .kv_features()
        .map_err(|reason| invalid_config("semantic.kv_features", reason))?;
    let gate_up_features = semantic
        .intermediate_size
        .checked_mul(2)
        .ok_or_else(|| invalid_config("semantic.intermediate_size", "gate/up width overflows"))?;
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

    for (layer_index, layer_type) in semantic.layer_types.iter().enumerate() {
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
        let sliding_window_tokens = match layer_type {
            GptOssLayerType::SlidingAttention => semantic.sliding_window,
            GptOssLayerType::FullAttention => 0,
        };
        nodes.push(ProgramNode {
            id: node_id(format!("node.layer.{layer_index}.attention"))?,
            operation_id: operation_id(GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(hidden.clone(), 0),
            inputs: vec![
                hidden.clone(),
                layer_weight_value_id(layer_index, INPUT_NORM_ROLE)?,
                layer_weight_value_id(layer_index, Q_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, K_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, V_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, O_PROJ_ROLE)?,
                layer_weight_value_id(layer_index, Q_BIAS_ROLE)?,
                layer_weight_value_id(layer_index, K_BIAS_ROLE)?,
                layer_weight_value_id(layer_index, V_BIAS_ROLE)?,
                layer_weight_value_id(layer_index, O_BIAS_ROLE)?,
                layer_weight_value_id(layer_index, ATTENTION_SINKS_ROLE)?,
                kv_value,
            ],
            outputs: vec![attention_output.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("query_heads", semantic.attention_head_count)?,
                attribute("kv_heads", semantic.kv_head_count)?,
                attribute("head_dim", semantic.head_dim)?,
                attribute("query_features", query_features)?,
                attribute("kv_features", kv_features)?,
                attribute("rope_dim", semantic.head_dim)?,
                attribute("maximum_context_tokens", semantic.maximum_sequence_tokens)?,
                attribute("rope_theta", semantic.rope_theta)?,
                attribute("yarn_factor", semantic.rope_scaling.factor)?,
                attribute(
                    "yarn_original_context_tokens",
                    semantic.rope_scaling.original_max_position_embeddings,
                )?,
                attribute("yarn_beta_fast", semantic.rope_scaling.beta_fast)?,
                attribute("yarn_beta_slow", semantic.rope_scaling.beta_slow)?,
                attribute("yarn_truncate", semantic.rope_scaling.truncate)?,
                attribute("sliding_window_tokens", sliding_window_tokens)?,
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
            operation_id: operation_id(GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(normalized.clone(), 0),
            inputs: vec![
                normalized,
                layer_weight_value_id(layer_index, ROUTER_ROLE)?,
                layer_weight_value_id(layer_index, ROUTER_BIAS_ROLE)?,
                layer_weight_value_id(layer_index, ROUTED_GATE_UP_ROLE)?,
                layer_weight_value_id(layer_index, ROUTED_GATE_UP_BIAS_ROLE)?,
                layer_weight_value_id(layer_index, ROUTED_DOWN_ROLE)?,
                layer_weight_value_id(layer_index, ROUTED_DOWN_BIAS_ROLE)?,
            ],
            outputs: vec![moe_output.clone()],
            attributes: BTreeMap::from([
                attribute("hidden_size", semantic.hidden_size)?,
                attribute("expert_count", semantic.expert_count)?,
                attribute("experts_per_token", semantic.experts_per_token)?,
                attribute("intermediate_size", semantic.intermediate_size)?,
                attribute("gate_up_features", gate_up_features)?,
                attribute("normalize_topk", true)?,
                attribute("swiglu_limit", semantic.swiglu_limit)?,
                attribute("gate_up_interleaved", true)?,
                attribute("down_bias_before_route_reduction", true)?,
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

#[cfg(test)]
mod tests {
    use ferrum_interfaces::vnext::{
        AttributeId, ModelFamilyId, SemanticValue, GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID,
        GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID,
    };

    use super::*;
    use crate::vnext::gpt_oss::config::GptOssMxfp4Config;
    use crate::vnext::gpt_oss::weights::expected_manifest;

    fn tiny() -> (GptOssSemanticConfig, GptOssMxfp4Config) {
        let layer_types = ["sliding_attention", "full_attention"];
        let raw = serde_json::to_vec(&serde_json::json!({
            "architectures": ["GptOssForCausalLM"],
            "attention_bias": true,
            "attention_dropout": 0.0,
            "experts_per_token": 4,
            "head_dim": 32,
            "hidden_act": "silu",
            "hidden_size": 32,
            "initial_context_length": 4096,
            "intermediate_size": 64,
            "layer_types": layer_types,
            "max_position_embeddings": 131072,
            "model_type": "gpt_oss",
            "num_attention_heads": 2,
            "num_experts_per_tok": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
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
            "tie_word_embeddings": true,
            "use_cache": true,
            "vocab_size": 201088
        }))
        .unwrap();
        GptOssSemanticConfig::parse(&raw).unwrap()
    }

    #[test]
    fn program_locks_alternating_attention_and_gpt_oss_moe_semantics() {
        let (semantic, quantization) = tiny();
        let manifest = expected_manifest(&semantic, &quantization).unwrap();
        let family_id = ModelFamilyId::new(super::super::FAMILY_ID).unwrap();
        let program = build_semantic_program(&family_id, &semantic, &manifest).unwrap();
        manifest
            .weight_schema(&semantic)
            .unwrap()
            .validate_program_references(&family_id, &program)
            .unwrap();
        assert_eq!(program.states().len(), 2);
        let nodes = &program.blocks()[0].nodes;
        let attention = nodes
            .iter()
            .filter(|node| {
                node.operation_id.as_str() == GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID
            })
            .collect::<Vec<_>>();
        assert_eq!(attention.len(), 2);
        let sliding = AttributeId::new("sliding_window_tokens").unwrap();
        assert_eq!(
            attention[0].attributes.get(&sliding),
            Some(&SemanticValue::Unsigned(128))
        );
        assert_eq!(
            attention[1].attributes.get(&sliding),
            Some(&SemanticValue::Unsigned(0))
        );

        let moe = nodes
            .iter()
            .find(|node| {
                node.operation_id.as_str() == GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID
            })
            .unwrap();
        assert_eq!(moe.inputs.len(), 7);
        assert_eq!(
            moe.attributes
                .get(&AttributeId::new("gate_up_features").unwrap()),
            Some(&SemanticValue::Unsigned(128))
        );
        assert_eq!(
            moe.attributes
                .get(&AttributeId::new("gate_up_interleaved").unwrap()),
            Some(&SemanticValue::Bool(true))
        );
    }
}
