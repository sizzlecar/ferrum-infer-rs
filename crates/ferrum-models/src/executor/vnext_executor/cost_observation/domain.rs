//! Cold export from actual executor configuration, resolved plan and validated
//! IO. No route/provider is invented and no device-axis maxima are extrapolated.
use super::super::{VNextExecutorConfig, VNextIoBinding};
use ferrum_interfaces::{
    execution_cost::{
        CostWorkloadDomainAvailability as A, CostWorkloadDomainUnknown as U, CostWorkloadDomainV1,
        CostWorkloadLimitsV1, ExecutorCostIdentityAvailability,
    },
    model_executor::TypedSequenceStateMemory,
    vnext::ResolvedModelPlan,
};
use std::{
    num::{NonZeroU32, NonZeroU64},
    sync::Arc,
};

pub(in super::super) fn cache(
    identity: &ExecutorCostIdentityAvailability,
    plan: &ResolvedModelPlan,
    config: &VNextExecutorConfig,
    io: &VNextIoBinding,
    state: TypedSequenceStateMemory,
) -> A {
    let identity = match identity {
        ExecutorCostIdentityAvailability::Known(v) => v,
        ExecutorCostIdentityAvailability::Unknown { reason, .. } => {
            return A::Unknown(U::MissingExecutionIdentity(*reason))
        }
    };
    if state.other_token_scaled_bytes_per_token != 0 {
        return A::Unknown(U::UnsupportedTokenScaledState);
    }
    let limits = match limits(
        config,
        plan.execution_plan().payload().maximum_scheduled_tokens(),
        io.output_elements,
        io.repetition_capacity,
        state,
    ) {
        Some(v) => v,
        None => return A::Unknown(U::InvalidCapacity),
    };
    match CostWorkloadDomainV1::new_vnext(identity, limits) {
        Ok(v) => A::Known(Arc::new(v)),
        Err(error) => A::Unknown(U::InvalidDescriptor(error)),
    }
}

fn limits(
    config: &VNextExecutorConfig,
    compiled_scheduled: u64,
    vocabulary: usize,
    repetition_slots: usize,
    state: TypedSequenceStateMemory,
) -> Option<CostWorkloadLimitsV1> {
    // These are the exact fields returned by VNextModelExecutor::capabilities.
    // The engine's dynamic BatchHint may narrow them; it cannot widen them.
    let maximum_rows = NonZeroU32::new(config.runtime_policy.memory().maximum_active_sequences)?;
    let maximum_context_tokens = NonZeroU32::new(u32::try_from(config.maximum_model_tokens).ok()?)?;
    // Both limits already constrain real prefill/decode/mixed dispatch. Keep
    // the tighter compiled/admission capacity; never use an observed maximum.
    let maximum_scheduled_tokens_per_wave = NonZeroU64::new(
        compiled_scheduled.min(config.runtime_policy.admission().maximum_scheduled_tokens),
    )?;
    Some(CostWorkloadLimitsV1 {
        maximum_rows,
        maximum_context_tokens,
        maximum_scheduled_tokens_per_wave,
        output_vocabulary_elements: NonZeroU64::new(u64::try_from(vocabulary).ok()?)?,
        repetition_slot_capacity: u64::try_from(repetition_slots).ok()?,
        fixed_state_bytes_per_row: state.fixed_bytes_per_sequence,
    })
}

#[cfg(test)]
mod tests;
