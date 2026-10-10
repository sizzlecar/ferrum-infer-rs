//! Explicit same-wave host encoding comparison; never calls a CUDA enqueue.
use super::*;
use ferrum_interfaces::vnext::SegmentBindingOracleCommands;

fn same_region(a: &CudaBufferRegion, b: &CudaBufferRegion) -> bool {
    a.borrowed().same_region(b.borrowed())
}
fn same_regions(a: &[CudaBufferRegion], b: &[CudaBufferRegion]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_region(a, b))
}
fn same_retention(a: &CudaProgramBindingRetention, b: &CudaProgramBindingRetention) -> bool {
    a.len() == b.len()
        && (0..a.len()).all(|index| match (a.region(index), b.region(index)) {
            (Some(a), Some(b)) => a.same_region(b),
            _ => false,
        })
}
fn same_command(a: &CudaDeviceCommand, b: &CudaDeviceCommand) -> bool {
    if a.runtime_instance != b.runtime_instance
        || a.operation != b.operation
        || a.batching_form != b.batching_form
        || a.participant_start != b.participant_start
        || a.participant_count != b.participant_count
        || a.token_count != b.token_count
        || a.compute_dispatch_count != b.compute_dispatch_count
        || a.transfer_command_count != b.transfer_command_count
        || a.reusable_address_scope != b.reusable_address_scope
        || a.replay_key.is_some()
        || b.replay_key.is_some()
        || a.replay_gap_reason.is_some()
        || b.replay_gap_reason.is_some()
        || a.reusable_execution.is_some()
        || b.reusable_execution.is_some()
        || !same_regions(&a.fence_dependencies, &b.fence_dependencies)
        || a.completion_checks.len() != b.completion_checks.len()
        || !a
            .completion_checks
            .iter()
            .zip(&b.completion_checks)
            .all(|(a, b)| {
                same_region(&a.source, &b.source)
                    && a.offsets == b.offsets
                    && a.failure_mask == b.failure_mask
                    && a.deferred == b.deferred
            })
    {
        return false;
    }
    let executable = match (&a.executable, &b.executable) {
        (None, None) => true,
        // The core separately compares the exact freshly issued Plan dependency
        // identities. Closure addresses need not match and no closure is invoked.
        (Some(a), Some(b)) => {
            a.work_declaration == CudaWorkDeclaration::ConditionalDependency
                && b.work_declaration == CudaWorkDeclaration::ConditionalDependency
                && same_regions(&a.regions, &b.regions)
                && a.host_storage == b.host_storage
        }
        _ => false,
    };
    let patches = match (&a.program_binding_patch, &b.program_binding_patch) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.binding.node_index() == b.binding.node_index()
                && a.binding.plan_hash() == b.binding.plan_hash()
                && a.binding.layout().fingerprint() == b.binding.layout().fingerprint()
                && a.binding.lane_slot_identity() == b.binding.lane_slot_identity()
                && a.binding.slot() == b.binding.slot()
                && same_region(&a.destination, &b.destination)
                && same_retention(&a.retention, &b.retention)
                && a.writes.len() == b.writes.len()
                && a.writes.iter().zip(&b.writes).all(|(a, b)| {
                    a.destination_offset_bytes == b.destination_offset_bytes
                        && a.live_payload_bytes == b.live_payload_bytes
                        && a.payload == b.payload
                })
        }
        _ => false,
    };
    executable && patches
}

pub(super) fn compare(
    actual: SegmentBindingOracleCommands<'_, CudaDeviceCommand>,
    reference: SegmentBindingOracleCommands<'_, CudaDeviceCommand>,
) -> Result<(), CudaDeviceRuntimeError> {
    for (name, a, b) in [
        (
            "program binding",
            actual.program_bindings,
            reference.program_bindings,
        ),
        (
            "dynamic binding",
            actual.dynamic_bindings,
            reference.dynamic_bindings,
        ),
        (
            "result binding",
            actual.result_bindings,
            reference.result_bindings,
        ),
    ] {
        if a.len() != b.len() || !a.iter().zip(b).all(|(a, b)| same_command(a, b)) {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "same-wave segment {name} bytes, owners, status or attribution differ"
            )));
        }
    }
    if actual.retained_dependencies.len() != reference.retained_dependencies.len()
        || !actual
            .retained_dependencies
            .iter()
            .zip(&reference.retained_dependencies)
            .all(|(a, b)| same_command(a, b))
    {
        return Err(CudaDeviceRuntimeError::contract(
            "same-wave segment Plan dependency command metadata differs",
        ));
    }
    Ok(())
}
