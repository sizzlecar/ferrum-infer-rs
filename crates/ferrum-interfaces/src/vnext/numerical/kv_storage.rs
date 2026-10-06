use std::collections::{BTreeMap, BTreeSet};

use super::{invalid, ElementType, KvStorageFormat, ModelFamilyId, StateSpec, VNextError};
use crate::vnext::{
    ModelProgram, NumericalExecutionProfile, ResolvedTensorLayout, StateCapacityDemand,
    StateCheckpointCapability, StateCheckpointContents, StateId, StateLifetime,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_INT8_KV_OPERATION_ID,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID, CAUSAL_PAGED_ATTENTION_INT8_KV_OPERATION_ID,
    CAUSAL_PAGED_ATTENTION_OPERATION_ID, GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID,
    HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_OPERATION_ID,
};
use serde::{Deserialize, Serialize};

/// Semantic K/V storage declared by the family, never inferred from a name or
/// from a provider's private buffers. Each referenced state belongs to the same
/// sequence and participates in the ordinary allocation/completion lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum KvStateStorage {
    F16 {
        state: StateId,
    },
    Int8PerTokenHeadF32ScaleV1 {
        payload_state: StateId,
        scale_state: StateId,
    },
}

impl KvStateStorage {
    pub const fn format(&self) -> KvStorageFormat {
        match self {
            Self::F16 { .. } => KvStorageFormat::F16,
            Self::Int8PerTokenHeadF32ScaleV1 { .. } => KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
        }
    }

    pub fn payload_state(&self) -> &StateId {
        match self {
            Self::F16 { state } => state,
            Self::Int8PerTokenHeadF32ScaleV1 { payload_state, .. } => payload_state,
        }
    }

    pub fn scale_state(&self) -> Option<&StateId> {
        match self {
            Self::F16 { .. } => None,
            Self::Int8PerTokenHeadF32ScaleV1 { scale_state, .. } => Some(scale_state),
        }
    }
}

pub(super) fn validate_kv_storage(
    family: &ModelFamilyId,
    declarations: &[KvStateStorage],
    states: &[StateSpec],
) -> Result<Option<KvStorageFormat>, VNextError> {
    let states: BTreeMap<_, _> = states.iter().map(|state| (&state.id, state)).collect();
    let mut referenced = BTreeSet::new();
    let mut format = None;
    for declaration in declarations {
        if format.is_some_and(|format| format != declaration.format()) {
            return Err(invalid(family, "a profile cannot mix KV storage formats"));
        }
        format = Some(declaration.format());
        let mut resolve = |id: &StateId| {
            if !referenced.insert(id.clone()) {
                return Err(invalid(family, "KV declarations cannot share a state"));
            }
            states
                .get(id)
                .copied()
                .ok_or_else(|| invalid(family, "KV storage references an undeclared state"))
        };
        let payload = resolve(declaration.payload_state())?;
        let shape = &payload.tensor.dimensions;
        let payload_type = match declaration.format() {
            KvStorageFormat::F16 => ElementType::F16,
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1 => ElementType::I8,
        };
        if shape.len() != 3
            || shape[0] != 2
            || shape[1] == 0
            || shape[2] == 0
            || payload.tensor.element_type != payload_type
        {
            return Err(invalid(
                family,
                "KV payload must have its declared dtype and shape [2, heads, head_dim]",
            ));
        }
        let maximum_tokens = validate_token_state(family, payload)?;
        if let Some(id) = declaration.scale_state() {
            let scale = resolve(id)?;
            if scale.tensor.dimensions != [2, shape[1]]
                || scale.tensor.element_type != ElementType::F32
                || validate_token_state(family, scale)? != maximum_tokens
                || scale.initialization != payload.initialization
                || scale.checkpoint != payload.checkpoint
            {
                return Err(invalid(family, "INT8 KV scales must be F32 [2, heads] with the same token capacity, initialization and checkpoint contract as the payload"));
            }
        }
    }
    Ok(format)
}

/// Validate coverage using standard operation identities and required ports,
/// rather than inferring KV from tensor shapes or trusting a partial state list.
/// These ports are part of the corresponding versioned operation ABIs.
pub(super) fn validate_program_kv_storage(
    profile: &NumericalExecutionProfile,
    program: &ModelProgram,
) -> Result<(), VNextError> {
    let by_value: BTreeMap<_, _> = program.states().iter().map(|s| (&s.value_id, s)).collect();
    let mut consumed = BTreeSet::new();
    for node in program.blocks().iter().flat_map(|block| &block.nodes) {
        let (format, payload_port, scale_port) = match node.operation_id.as_str() {
            CAUSAL_PAGED_ATTENTION_OPERATION_ID
            | CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID
            | CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID
            | HYBRID_VNORM_CAUSAL_PAGED_ATTENTION_OPERATION_ID => (KvStorageFormat::F16, 8, None),
            GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID => (KvStorageFormat::F16, 11, None),
            CAUSAL_PAGED_ATTENTION_INT8_KV_OPERATION_ID
            | CAUSAL_PAGED_ATTENTION_F32_MASTER_INT8_KV_OPERATION_ID => {
                (KvStorageFormat::Int8PerTokenHeadF32ScaleV1, 8, Some(9))
            }
            _ => continue,
        };
        let state_at = |port: usize| {
            node.inputs
                .get(port)
                .and_then(|value| by_value.get(value).copied())
                .ok_or_else(|| {
                    invalid(
                        &profile.family_id,
                        format!(
                            "operation {} requires a declared KV state at input {port}",
                            node.operation_id
                        ),
                    )
                })
        };
        let payload = state_at(payload_port)?;
        let declaration = profile
            .kv_storage
            .iter()
            .find(|storage| storage.payload_state() == &payload.id)
            .ok_or_else(|| {
                invalid(
                    &profile.family_id,
                    format!(
                        "operation {} KV state {} is missing from the numerical storage contract",
                        node.operation_id, payload.id
                    ),
                )
            })?;
        if declaration.format() != format
            || declaration.scale_state()
                != scale_port
                    .map(|port| state_at(port).map(|state| &state.id))
                    .transpose()?
        {
            return Err(invalid(
                &profile.family_id,
                format!(
                    "operation {} state ports conflict with the declared KV format",
                    node.operation_id
                ),
            ));
        }
        consumed.insert(declaration.payload_state());
    }
    if profile
        .kv_storage
        .iter()
        .any(|storage| !consumed.contains(storage.payload_state()))
    {
        return Err(invalid(
            &profile.family_id,
            "KV storage is not consumed by a supported attention operation contract",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

fn validate_token_state(family: &ModelFamilyId, state: &StateSpec) -> Result<u64, VNextError> {
    let tensor_bytes = state.tensor.byte_len()?;
    state.capacity_demand.validate(tensor_bytes)?;
    if state.lifetime != StateLifetime::Sequence
        || state.tensor.layout != ResolvedTensorLayout::Contiguous
    {
        return Err(invalid(
            family,
            "KV storage must be contiguous Sequence state",
        ));
    }
    if matches!(state.checkpoint, StateCheckpointCapability::CompletedBoundary(contract)
        if contract.contents() != StateCheckpointContents::PrefixPositions)
    {
        return Err(invalid(
            family,
            "KV checkpoint state must retain every valid prefix position",
        ));
    }
    match state.capacity_demand {
        StateCapacityDemand::TokenScaled {
            bytes_per_token,
            maximum_tokens,
        } if bytes_per_token == tensor_bytes && maximum_tokens > 0 => Ok(maximum_tokens),
        _ => Err(invalid(
            family,
            "KV capacity must be exactly one typed tensor per token",
        )),
    }
}
