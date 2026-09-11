//! Family numerical declarations derived from typed semantics before a program
//! or backend exists. Source encodings only constrain Auto qualification; an
//! explicit profile controls the graph and never changes source tensor bytes.

use super::*;
use ferrum_interfaces::vnext::{
    NumericalOperationContract, NumericalProfileId, StateCheckpointCapability,
};

pub const F16_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f16";
pub const F32_MASTER_NUMERICAL_PROFILE_ID: &str = "qwen3_5.f32-master";

pub(super) fn profiles(
    family_id: &ModelFamilyId,
    config: &Qwen35FamilyConfig,
) -> Result<FamilyNumericalProfiles, VNextError> {
    let text = Qwen35FamilyProvider::text_config(config)?;
    let states = states(&text, config.max_position_embeddings)?;
    let f16 = profile(family_id, &text, &states, false)?;
    let f32 = profile(family_id, &text, &states, true)?;
    // Qualification covers both physical encoding and recurrent parameter ABI.
    // In particular, an unquantized negative-rate source must not silently
    // acquire the as-yet unqualified F16 behavior merely because it has no blocks.
    let native = config.weights.iter().all(|weight| {
        matches!(
            weight.source_encoding,
            FamilyWeightSourceEncoding::Dense { .. }
                | FamilyWeightSourceEncoding::BlockQuantized(_)
        )
    });
    let portable = config.weights.iter().all(|weight| {
        !matches!(
            weight.source_encoding,
            FamilyWeightSourceEncoding::BlockQuantized(_)
        )
    });
    let automatic = match config.recurrent_weight_abi {
        RecurrentWeightAbi::LogRateGrouped if portable => vec![f16.id.clone()],
        RecurrentWeightAbi::NegativeRateInterleaved if native => vec![f32.id.clone()],
        _ => Vec::new(),
    };
    FamilyNumericalProfiles::new(
        family_id,
        ContractVersion::new(1, 0),
        vec![f16, f32],
        automatic,
    )
}

fn profile(
    family_id: &ModelFamilyId,
    text: &Qwen35TextConfig,
    states: &[StateSpec],
    master: bool,
) -> Result<NumericalExecutionProfile, VNextError> {
    let (name, selections, activation) = if master {
        (
            F32_MASTER_NUMERICAL_PROFILE_ID,
            Qwen35OperationProfile::F32_MASTER,
            ElementType::F32,
        )
    } else {
        (
            F16_NUMERICAL_PROFILE_ID,
            Qwen35OperationProfile::F16,
            ElementType::F16,
        )
    };
    let primary_activation = value_id("value.hidden.embedding")?;
    let mut boundaries = BTreeMap::from([
        (primary_activation.clone(), activation),
        (value_id("value.output.final_hidden")?, activation),
        (value_id("value.output.logits")?, activation),
        (value_id("value.output.greedy_token")?, ElementType::U32),
    ]);
    let mut operations = BTreeMap::new();
    let mut declare =
        |selection: OperationSelection, multiply, accumulate| -> Result<(), VNextError> {
            let operation_id = operation_id(selection.id)?;
            operations.insert(
                operation_id.clone(),
                NumericalOperationContract {
                    operation_id,
                    version: selection.version,
                    multiplication_type: multiply,
                    accumulation_type: accumulate,
                },
            );
            Ok(())
        };
    declare(selections.token_embedding, None, None)?;
    declare(
        selections.final_norm,
        Some(ElementType::F32),
        Some(ElementType::F32),
    )?;
    declare(
        selections.logits,
        Some(ElementType::F32),
        Some(ElementType::F32),
    )?;
    // Penalty arithmetic is part of this operation; the selected token itself
    // is U32. It must not be reported as a floating-point logit boundary.
    declare(selections.argmax, Some(ElementType::F32), None)?;
    for (index, layer) in text.layer_types.iter().enumerate() {
        for (role, dtype) in [
            ("attention", activation),
            ("post_attention_norm", ElementType::F16),
            ("mlp", ElementType::F16),
            ("output", activation),
        ] {
            boundaries.insert(value_id(format!("value.layer.{index}.{role}"))?, dtype);
        }
        let attention = match layer {
            Qwen35LayerType::LinearAttention => selections.linear_attention,
            Qwen35LayerType::FullAttention => selections.causal_attention,
        };
        declare(attention, Some(ElementType::F32), Some(ElementType::F32))?;
        declare(
            selections.post_attention_norm,
            Some(ElementType::F32),
            Some(ElementType::F32),
        )?;
        let (feed_forward, residual) = if text.moe.is_some() {
            (
                OperationSelection::new(ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID, 1, 0),
                selections.moe_residual,
            )
        } else {
            (selections.dense_feed_forward, selections.dense_residual)
        };
        declare(feed_forward, Some(ElementType::F32), Some(ElementType::F32))?;
        declare(residual, None, Some(ElementType::F32))?;
    }
    Ok(NumericalExecutionProfile {
        id: NumericalProfileId::new(name)
            .map_err(|reason| invalid_config("numerical_profile.id", reason))?,
        version: ContractVersion::new(1, 0),
        family_id: family_id.clone(),
        primary_activation,
        boundaries,
        states: states.to_vec(),
        operations: operations.into_values().collect(),
    })
}

fn states(text: &Qwen35TextConfig, maximum_tokens: u64) -> Result<Vec<StateSpec>, VNextError> {
    let mut states = Vec::new();
    for (index, layer) in text.layer_types.iter().enumerate() {
        match layer {
            Qwen35LayerType::LinearAttention => {
                for (role, dimensions, dtype) in [
                    (
                        "conv",
                        text.recurrent_conv_state_shape()
                            .map_err(|reason| invalid_config("states.conv", reason))?
                            .to_vec(),
                        ElementType::F16,
                    ),
                    (
                        "delta",
                        text.recurrent_delta_state_shape()
                            .map_err(|reason| invalid_config("states.delta", reason))?
                            .to_vec(),
                        data_type_to_element_type(text.mamba_ssm_dtype)
                            .map_err(|reason| invalid_config("states.delta.dtype", reason))?,
                    ),
                ] {
                    states.push(StateSpec {
                        id: state_id(format!("state.layer.{index}.{role}"))?,
                        value_id: value_id(format!("value.state.layer.{index}.{role}"))?,
                        tensor: tensor_spec(
                            dimensions.into_iter().map(|extent| extent as u64).collect(),
                            dtype,
                        ),
                        lifetime: StateLifetime::Sequence,
                        capacity_demand: StateCapacityDemand::FixedPerScope,
                        initialization: StateInitialization::Zero,
                        checkpoint: StateCheckpointCapability::Unsupported,
                    });
                }
            }
            Qwen35LayerType::FullAttention => {
                let tensor = tensor_spec(
                    vec![2, text.num_key_value_heads as u64, text.head_dim as u64],
                    ElementType::F16,
                );
                let bytes_per_token = tensor.byte_len()?;
                states.push(StateSpec {
                    id: state_id(format!("state.layer.{index}.kv"))?,
                    value_id: value_id(format!("value.state.layer.{index}.kv"))?,
                    tensor,
                    lifetime: StateLifetime::Sequence,
                    capacity_demand: StateCapacityDemand::TokenScaled {
                        bytes_per_token,
                        maximum_tokens,
                    },
                    // Providers write each valid KV slot before reading it.
                    initialization: StateInitialization::None,
                    checkpoint: StateCheckpointCapability::Unsupported,
                });
            }
        }
    }
    Ok(states)
}
