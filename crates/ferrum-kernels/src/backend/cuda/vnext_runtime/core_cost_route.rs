//! Capabilities of the existing CUDA encoder, not promises that a future-cost
//! projection supports those transformations. Unsupported projection stays Unknown.

use super::CudaDeviceRuntimeError;
use ferrum_interfaces::execution_cost::CoreReadbackRoute;
use ferrum_interfaces::vnext::{
    coalesce_sorted_program_binding_writes, DeviceBatchingForm, DeviceCommandPhase,
    DeviceCoreCostCapabilities, OperationCostCommand, ProgramBindingCostPatch,
    ProgramBindingCostWrite, ProgramBindingLayout, VNextError, DEVICE_ZERO_NATIVE_OPERATION_ID,
    HOST_UPLOAD_NATIVE_OPERATION_ID,
};

pub(super) const PROGRAM_BINDING_NATIVE_OPERATION: &str = "vnext_program_binding_prelude";

pub(super) fn coalesced_program_binding(
    layout: &ProgramBindingLayout,
    patches: &[ProgramBindingCostPatch<'_>],
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<OperationCostCommand, CudaDeviceRuntimeError> {
    project_binding_slots(
        layout.physical_size_bytes(),
        layout.slots().iter().map(|slot| {
            (
                slot.node_index(),
                slot.physical_offset_bytes(),
                slot.capacity_size_bytes(),
            )
        }),
        patches,
        poll,
    )
}

fn project_binding_slots(
    arena_size_bytes: u64,
    slots: impl ExactSizeIterator<Item = (usize, u64, u64)>,
    patches: &[ProgramBindingCostPatch<'_>],
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<OperationCostCommand, CudaDeviceRuntimeError> {
    use ferrum_interfaces::execution_cost::MAX_COST_COMMANDS;
    let invalid = || {
        CudaDeviceRuntimeError::contract(
            "CUDA binding cost spans do not cover one compatible compiled prelude",
        )
    };
    let first = patches.first().ok_or_else(invalid)?.command;
    if slots.len() != patches.len() || patches.len() > MAX_COST_COMMANDS {
        return Err(invalid());
    }
    let mut writes = Vec::new();
    let mut previous_node = None;
    for ((node_index, offset, capacity), patch) in slots.zip(patches) {
        poll().map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        let command = patch.command;
        if previous_node.is_some_and(|node| node >= node_index)
            || node_index != patch.node_index
            || offset
                .checked_add(capacity)
                .is_none_or(|end| end > arena_size_bytes)
            || command.phase() != DeviceCommandPhase::DynamicBinding
            || command.participant_start() != first.participant_start()
            || command.participant_count() != first.participant_count()
            || command.token_count() != first.token_count()
            || command.compute_dispatch_count() != 0
            || command.transfer_command_count() == 0
            || patch.writes.is_empty()
            || writes
                .len()
                .checked_add(patch.writes.len())
                .is_none_or(|count| count > MAX_COST_COMMANDS)
        {
            return Err(invalid());
        }
        previous_node = Some(node_index);
        writes
            .try_reserve_exact(patch.writes.len())
            .map_err(|_| invalid())?;
        for write in patch.writes {
            poll().map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
            if write
                .offset_bytes()
                .checked_add(write.length_bytes())
                .is_none_or(|end| end > capacity)
            {
                return Err(invalid());
            }
            writes.push(
                ProgramBindingCostWrite::new(
                    offset
                        .checked_add(write.offset_bytes())
                        .ok_or_else(invalid)?,
                    write.length_bytes(),
                )
                .map_err(|_| invalid())?,
            );
        }
    }
    writes.sort_unstable_by_key(|write| write.offset_bytes());
    let transfers = coalesce_sorted_program_binding_writes(&writes, arena_size_bytes, poll)
        .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
    OperationCostCommand::new(
        PROGRAM_BINDING_NATIVE_OPERATION,
        DeviceCommandPhase::DynamicBinding,
        DeviceBatchingForm::ParticipantLoop,
        first.participant_start(),
        first.participant_count(),
        first.token_count(),
        0,
        transfers.len() as u64,
    )
    .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))
}

pub(super) fn capabilities() -> DeviceCoreCostCapabilities {
    DeviceCoreCostCapabilities {
        upload_native_operation: HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
        zero_native_operation: DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
        single_transfer_commands: true,
        // coalesced_program_bindings merges patches into physical transfers.
        preserves_program_bindings: false,
        // submission_readback enqueues a D2H command into the submitted wave.
        staged_host_readback_without_commands: false,
        staged_host_readback_native_operation: Some(super::submission_readback::NATIVE_OPERATION),
        fallback_readback: CoreReadbackRoute::HostSynchronized,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod capability_tests {
    use super::*;

    #[test]
    fn cuda_core_does_not_claim_metal_binding_or_readback_behavior() {
        let actual = capabilities();
        assert!(actual.single_transfer_commands);
        assert!(!actual.preserves_program_bindings);
        assert!(!actual.staged_host_readback_without_commands);
        assert_eq!(
            actual.fallback_readback,
            CoreReadbackRoute::HostSynchronized
        );
    }
}
