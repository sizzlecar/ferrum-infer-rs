//! Explicit same-wave host encoding comparison; never calls a CUDA enqueue.
use super::*;
use ferrum_interfaces::vnext::SegmentBindingOracleCommands;

mod packet;

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
        || a.binding_program.is_some()
        || b.binding_program.is_some()
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

fn same_status(
    actual: Option<(&[u64], u32, bool)>,
    reference: Option<(&[u64], u32, bool)>,
) -> bool {
    actual == reference
}

fn exact_node_coverage(actual: &[u32], expected: &[u32]) -> bool {
    actual == expected && actual.windows(2).all(|pair| pair[0] < pair[1])
}

fn mismatch() -> CudaDeviceRuntimeError {
    CudaDeviceRuntimeError::contract(
        "same-wave compiled program bytes, exact owners/ranges, status, node coverage or attribution differ",
    )
}

/// Compare the real final prelude with ordinary same-wave provider commands.
/// References remain diagnostic-only: no command closure is invoked here.
pub(super) fn compare_program(
    program_bindings: &[CudaDeviceCommand],
    program_binding_nodes: &[u32],
    nodes: &[(
        u32,
        SegmentBindingOracleCommands<'_, CudaDeviceCommand>,
        SegmentBindingOracleCommands<'_, CudaDeviceCommand>,
    )],
) -> Result<(), CudaDeviceRuntimeError> {
    if nodes.is_empty() || nodes.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err(mismatch());
    }
    let mut references = Vec::new();
    for (node_index, actual, reference) in nodes {
        if !actual.program_bindings.is_empty() || reference.program_bindings.len() > 1 {
            return Err(mismatch());
        }
        // Fresh dependency identities were checked separately by Core. Retain
        // the original CUDA checks for every non-prelude command channel too.
        compare(
            SegmentBindingOracleCommands {
                program_bindings: &[],
                dynamic_bindings: actual.dynamic_bindings,
                result_bindings: actual.result_bindings,
                retained_dependencies: actual.retained_dependencies.clone(),
            },
            SegmentBindingOracleCommands {
                program_bindings: &[],
                dynamic_bindings: reference.dynamic_bindings,
                result_bindings: reference.result_bindings,
                retained_dependencies: reference.retained_dependencies.clone(),
            },
        )?;
        if let Some(command) = reference.program_bindings.first() {
            references.push((*node_index, command));
        }
    }
    let reference_indices = references
        .iter()
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    if !exact_node_coverage(program_binding_nodes, &reference_indices) {
        return Err(mismatch());
    }
    if references.is_empty() {
        return if program_bindings.is_empty() {
            Ok(())
        } else {
            Err(mismatch())
        };
    }
    let [prelude] = program_bindings else {
        return Err(mismatch());
    };
    let projection = prelude.binding_program.as_ref().ok_or_else(mismatch)?;
    let executable = prelude.executable.as_ref().ok_or_else(mismatch)?;
    let [packet] = executable.host_storage.as_slice() else {
        return Err(mismatch());
    };
    let indices = projection
        .nodes
        .iter()
        .map(|node| u32::try_from(node.binding.node_index()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| mismatch())?;
    if !exact_node_coverage(&indices, &reference_indices)
        || executable.work_declaration != CudaWorkDeclaration::Static
        || executable.regions.len() != projection.nodes.len()
        || prelude.binding_owner_tables.len() != 1
        || !Arc::ptr_eq(&prelude.binding_owner_tables[0], &projection.owners)
        || !prelude.fence_dependencies.is_empty()
        || prelude.program_binding_patch.is_some()
        || prelude.replay_key.is_some()
        || prelude.replay_gap_reason.is_some()
        || prelude.reusable_execution.is_some()
        || prelude.reusable_address_scope.is_some()
        || prelude.operation
            != if projection.compact {
                "vnext_program_binding_compact_scatter_v1"
            } else {
                "vnext_program_binding_prelude"
            }
        || prelude.batching_form != DeviceBatchingForm::ParticipantLoop
        || prelude.participant_start != 0
        || prelude.compute_dispatch_count != u64::from(projection.compact)
        || prelude.transfer_command_count
            != if projection.compact {
                1
            } else {
                u64::from(projection.descriptor_count)
            }
    {
        return Err(mismatch());
    }
    let mut status_index = 0;
    let mut expected_writes = Vec::new();
    for ((node, (_, reference)), executable_region) in projection
        .nodes
        .iter()
        .zip(&references)
        .zip(&executable.regions)
    {
        let patch = reference
            .program_binding_patch
            .as_ref()
            .ok_or_else(mismatch)?;
        if node.runtime_instance != reference.runtime_instance
            || node.runtime_instance != prelude.runtime_instance
            || node.operation != reference.operation
            || node.batching_form != reference.batching_form
            || node.participant_start != reference.participant_start
            || node.participants != reference.participant_count
            || node.tokens != reference.token_count
            || node.participants != prelude.participant_count
            || node.tokens != prelude.token_count
            || node.compute_dispatch_count != reference.compute_dispatch_count
            || node.transfer_command_count != reference.transfer_command_count
            || node.reusable_address_scope != reference.reusable_address_scope
            || reference.executable.is_some()
            || !reference.fence_dependencies.is_empty()
            || !reference.binding_owner_tables.is_empty()
            || reference.binding_program.is_some()
            || reference.replay_key.is_some()
            || reference.replay_gap_reason.is_some()
            || reference.reusable_execution.is_some()
            || node.binding.node_index() != patch.binding.node_index()
            || node.binding.plan_hash() != patch.binding.plan_hash()
            || node.binding.layout() != patch.binding.layout()
            || node.binding.lane_slot_identity() != patch.binding.lane_slot_identity()
            || node.binding.slot() != patch.binding.slot()
            || !same_region(&node.destination, &patch.destination)
            || !same_region(&node.destination, executable_region)
            || node.ranges.len() != patch.retention.len()
            || node.writes.len() != patch.writes.len()
        {
            return Err(mismatch());
        }
        for (index, range) in node.ranges.iter().enumerate() {
            let actual = projection.owners.region(range)?;
            let reference = patch.retention.region(index).ok_or_else(mismatch)?;
            // same_region includes allocation Arc, runtime, pointer/length,
            // type, address scope and exact opaque Core lease owners.
            if !actual.same_region(reference) {
                return Err(mismatch());
            }
        }
        for (write, reference) in node.writes.iter().zip(&patch.writes) {
            if write.destination_offset_bytes != reference.destination_offset_bytes
                || write.live_payload_bytes != reference.live_payload_bytes
                || packet.get(write.payload.clone()) != Some(reference.payload.as_ref())
            {
                return Err(mismatch());
            }
            expected_writes.push(packet::ExpectedWrite {
                offset: node
                    .binding
                    .slot()
                    .physical_offset_bytes()
                    .checked_add(reference.destination_offset_bytes)
                    .ok_or_else(mismatch)?,
                bytes: &reference.payload,
            });
        }
        let logical_status = node
            .numerical_status
            .as_ref()
            .map(|(offsets, mask)| (offsets.as_slice(), *mask, node.numerical_status_deferred));
        let reference_status = match reference.completion_checks.as_slice() {
            [] => None,
            [check] if same_region(&check.source, &patch.destination) => {
                Some((check.offsets.as_slice(), check.failure_mask, check.deferred))
            }
            _ => return Err(mismatch()),
        };
        if !same_status(logical_status, reference_status) {
            return Err(mismatch());
        }
        if let Some(status) = logical_status {
            let actual = prelude
                .completion_checks
                .get(status_index)
                .ok_or_else(mismatch)?;
            if !same_region(&actual.source, &node.destination)
                || !same_status(
                    Some(status),
                    Some((&actual.offsets, actual.failure_mask, actual.deferred)),
                )
            {
                return Err(mismatch());
            }
            status_index += 1;
        }
    }
    expected_writes.sort_by_key(|write| write.offset);
    if status_index != prelude.completion_checks.len()
        || !packet::matches(
            packet,
            projection.descriptor_count,
            projection.nodes[0].binding.layout().physical_size_bytes(),
            &expected_writes,
        )
    {
        return Err(mismatch());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_oracle_rejects_missing_or_duplicate_nodes_and_changed_status() {
        assert!(exact_node_coverage(&[1, 4, 8], &[1, 4, 8]));
        assert!(!exact_node_coverage(&[1, 8], &[1, 4, 8]));
        assert!(!exact_node_coverage(&[1, 4, 4], &[1, 4, 8]));
        assert!(!exact_node_coverage(&[4, 1, 8], &[1, 4, 8]));
        let current = Some((&[4_u64, 20][..], 3, true));
        assert!(same_status(current, current));
        assert!(!same_status(current, None));
        assert!(!same_status(current, Some((&[4, 24], 3, true))));
        assert!(!same_status(current, Some((&[4, 20], 1, true))));
        assert!(!same_status(current, Some((&[4, 20], 3, false))));
    }
}
