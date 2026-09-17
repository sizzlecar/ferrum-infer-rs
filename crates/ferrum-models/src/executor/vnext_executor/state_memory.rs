//! Logical state accounting from the already validated numerical profile.
//! Physical residency remains owned and reported by the runtime's pools.

use std::collections::BTreeSet;

use ferrum_interfaces::model_executor::TypedSequenceStateMemory;
use ferrum_interfaces::vnext::{KvStateStorage, StateCapacityDemand, StateLifetime, StateSpec};
use ferrum_types::{FerrumError, Result};

pub(super) fn logical_sequence_state_memory(
    states: &[StateSpec],
    kv_storage: &[KvStateStorage],
) -> Result<TypedSequenceStateMemory> {
    let kv_states = kv_storage
        .iter()
        .flat_map(|storage| std::iter::once(storage.payload_state()).chain(storage.scale_state()))
        .collect::<BTreeSet<_>>();
    let mut bytes = TypedSequenceStateMemory::default();
    for state in states
        .iter()
        .filter(|state| state.lifetime == StateLifetime::Sequence)
    {
        let (total, contribution) = match state.capacity_demand {
            StateCapacityDemand::FixedPerScope => (
                &mut bytes.fixed_bytes_per_sequence,
                state
                    .tensor
                    .byte_len()
                    .map_err(|error| FerrumError::model(error.to_string()))?,
            ),
            StateCapacityDemand::TokenScaled {
                bytes_per_token, ..
            } => (
                if kv_states.contains(&state.id) {
                    &mut bytes.kv_bytes_per_token
                } else {
                    &mut bytes.other_token_scaled_bytes_per_token
                },
                bytes_per_token,
            ),
        };
        *total = total.checked_add(contribution).ok_or_else(|| {
            FerrumError::model("typed logical sequence-state byte total overflows u64")
        })?;
    }
    bytes
        .kv_bytes_per_token
        .checked_add(bytes.other_token_scaled_bytes_per_token)
        .and_then(|tokens| tokens.checked_add(bytes.fixed_bytes_per_sequence))
        .ok_or_else(|| {
            FerrumError::model("typed logical sequence-state byte total overflows u64")
        })?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{
        ElementType, ProgramTensorSpec, ProgramValueId, ResolvedTensorLayout,
        StateCheckpointCapability, StateId, StateInitialization,
    };

    fn state(
        name: &str,
        dtype: ElementType,
        dimensions: Vec<u64>,
        token_scaled: bool,
    ) -> StateSpec {
        let tensor = ProgramTensorSpec {
            dimensions,
            element_type: dtype,
            layout: ResolvedTensorLayout::Contiguous,
        };
        StateSpec {
            id: StateId::new(format!("state.{name}")).unwrap(),
            value_id: ProgramValueId::new(format!("value.{name}")).unwrap(),
            capacity_demand: if token_scaled {
                StateCapacityDemand::TokenScaled {
                    bytes_per_token: tensor.byte_len().unwrap(),
                    maximum_tokens: 8,
                }
            } else {
                StateCapacityDemand::FixedPerScope
            },
            tensor,
            lifetime: StateLifetime::Sequence,
            initialization: StateInitialization::None,
            checkpoint: StateCheckpointCapability::Unsupported,
        }
    }

    #[test]
    fn logical_int8_memory_includes_scales_and_separates_hybrid_fixed_state() {
        let payload = state("quant", ElementType::I8, vec![2, 2, 128], true);
        let scales = state("scales", ElementType::F32, vec![2, 2], true);
        let declaration = KvStateStorage::Int8PerTokenHeadF32ScaleV1 {
            payload_state: payload.id.clone(),
            scale_state: scales.id.clone(),
        };
        let mut step = state("step", ElementType::F32, vec![100], true);
        step.lifetime = StateLifetime::Step;
        let bytes = logical_sequence_state_memory(
            &[
                payload,
                scales,
                state("conv", ElementType::F16, vec![12], false),
                state("recurrent", ElementType::F32, vec![64], false),
                state("other", ElementType::U32, vec![2], true),
                step,
            ],
            &[declaration],
        )
        .unwrap();
        assert_eq!(bytes.kv_bytes_per_token, 512 + 16);
        assert_eq!(bytes.fixed_bytes_per_sequence, 24 + 256);
        assert_eq!(bytes.other_token_scaled_bytes_per_token, 8);
    }

    #[test]
    fn logical_f16_memory_sums_actual_attention_states_and_checks_overflow() {
        let states = [
            state("narrow", ElementType::F16, vec![2, 1, 128], true),
            state("wide", ElementType::F16, vec![2, 3, 128], true),
        ];
        let declarations = states
            .iter()
            .map(|state| KvStateStorage::F16 {
                state: state.id.clone(),
            })
            .collect::<Vec<_>>();
        let bytes = logical_sequence_state_memory(&states, &declarations).unwrap();
        assert_eq!(bytes.kv_bytes_per_token, (2 * 1 * 128 + 2 * 3 * 128) * 2);
        assert_eq!(bytes.fixed_bytes_per_sequence, 0);
        let mut overflow = states;
        overflow[0].capacity_demand = StateCapacityDemand::TokenScaled {
            bytes_per_token: u64::MAX,
            maximum_tokens: 1,
        };
        assert!(logical_sequence_state_memory(&overflow, &declarations).is_err());
    }
}
