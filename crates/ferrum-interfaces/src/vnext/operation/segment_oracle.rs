//! Explicit same-wave encoder diagnostic. Reference commands are never submitted.
use std::collections::BTreeMap;

use super::foundation::invalid_operation;
use super::segment_dispatch::{SegmentEncodedNode, SegmentEncodedProgram};
use crate::vnext::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn audit_segment_wave<'binding, R, I>(
    runtime: &R,
    providers: &[BoundOperationProvider<'_, R>],
    resolved: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    wave: &PreparedStepSubmissionWave<R>,
    active: I,
    encoded: &BTreeMap<usize, SegmentEncodedNode<R::Command>>,
) -> Result<(), SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
    I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
{
    if runtime.segment_binding_oracle_mode() == SegmentBindingOracleMode::Disabled {
        return Ok(());
    }
    for (&node_index, actual) in encoded {
        let provider = providers.get(node_index).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "segment oracle node is outside provider sequence",
            ))
        })?;
        let node_identity = identity
            .materialize_node(node_index)
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let invocation = BatchedOperationInvocation::from_reusable_wave_node(
            runtime,
            resolved,
            provider.dispatch(),
            identity,
            node_identity,
            wave,
            node_index,
            active.clone(),
        )
        .map_err(SubmissionWaveDispatchError::Contract)?;
        let expected_phase = invocation.operation().profile_phase;
        let reference = match provider
            .provider()
            .encode_reusable_execution_bindings(invocation)
        {
            Ok(reference) => reference,
            Err(failure)
                if node_identity.contains_identity(failure.identity())
                    && failure.phase() == expected_phase =>
            {
                return Err(SubmissionWaveDispatchError::Provider(failure));
            }
            Err(_) => {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "segment oracle reference failure has foreign attribution",
                )))
            }
        };
        if !actual
            .bindings
            .retained_dependency_identities()
            .eq(reference.retained_dependency_identities())
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "segment oracle retained dependency identities differ",
            )));
        }
        let compared = runtime
            .compare_segment_binding_reference(
                actual.bindings.segment_binding_oracle_commands(),
                reference.segment_binding_oracle_commands(),
            )
            .map_err(|error| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "segment binding oracle mismatch: {error}"
                )))
            })?;
        if compared.is_none() {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "enabled segment binding oracle is unsupported by the runtime",
            )));
        }
        // Pool guards were released before either encoding. Drop this complete
        // reference (including fresh authority/retention) without any enqueue.
        drop(reference);
    }
    Ok(())
}

/// The compiled path owns its upload bytes in one final prelude. Its reference
/// therefore has to remain alive until the complete logical program is compared;
/// comparing the empty per-node upload channels alone would miss every byte.
#[allow(clippy::too_many_arguments)]
pub(super) fn audit_compiled_segment_wave<'binding, R, I>(
    runtime: &R,
    providers: &[BoundOperationProvider<'_, R>],
    resolved: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    wave: &PreparedStepSubmissionWave<R>,
    active: I,
    encoded: &SegmentEncodedProgram<R::Command>,
) -> Result<(), SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
    I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
{
    if runtime.segment_binding_oracle_mode() == SegmentBindingOracleMode::Disabled {
        return Ok(());
    }
    if encoded.nodes.is_empty()
        || encoded
            .nodes
            .windows(2)
            .any(|pair| pair[0].node_index >= pair[1].node_index)
    {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            "compiled segment oracle node sequence is empty or unordered",
        )));
    }
    let mut references = Vec::with_capacity(encoded.nodes.len());
    for actual in &encoded.nodes {
        let node_index = actual.node_index;
        let provider = providers.get(node_index).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "compiled segment oracle node is outside provider sequence",
            ))
        })?;
        let node_identity = identity
            .materialize_node(node_index)
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let invocation = BatchedOperationInvocation::from_reusable_wave_node(
            runtime,
            resolved,
            provider.dispatch(),
            identity,
            node_identity,
            wave,
            node_index,
            active.clone(),
        )
        .map_err(SubmissionWaveDispatchError::Contract)?;
        let expected_phase = invocation.operation().profile_phase;
        let reference = match provider
            .provider()
            .encode_reusable_execution_bindings(invocation)
        {
            Ok(reference) => reference,
            Err(failure)
                if node_identity.contains_identity(failure.identity())
                    && failure.phase() == expected_phase =>
            {
                return Err(SubmissionWaveDispatchError::Provider(failure));
            }
            Err(_) => {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "compiled segment oracle reference failure has foreign attribution",
                )))
            }
        };
        if !actual
            .bindings
            .retained_dependency_identities()
            .eq(reference.retained_dependency_identities())
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "compiled segment oracle retained dependency identities differ",
            )));
        }
        references.push(reference);
    }
    let nodes = encoded
        .nodes
        .iter()
        .zip(&references)
        .map(|(actual, reference)| {
            let node_index = u32::try_from(actual.node_index).map_err(|_| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "compiled segment oracle node index exceeds u32",
                ))
            })?;
            Ok((
                node_index,
                actual.bindings.segment_binding_oracle_commands(),
                reference.segment_binding_oracle_commands(),
            ))
        })
        .collect::<Result<Vec<_>, SubmissionWaveDispatchError<R>>>()?;
    let compared = runtime
        .compare_segment_binding_program_reference(
            &encoded.program_bindings,
            &encoded.program_binding_nodes,
            &nodes,
        )
        .map_err(|error| {
            SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                "compiled segment binding oracle mismatch: {error}"
            )))
        })?;
    if compared.is_none() {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            "enabled compiled segment binding oracle is unsupported by the runtime",
        )));
    }
    // Dropping references releases their fresh authorities without submission.
    // Their bytes/owners never enter the real prelude or cached executable.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn compiled_oracle_owner_comparison_rejects_an_alias_lease_or_scope() {
        // Exercise the actual opaque Core owner comparison used by the CUDA
        // region oracle. A shared storage owner must not conceal another lease.
        let storage = Arc::new(());
        let lease = Arc::new(());
        let lane = ExecutionLaneId::mint().unwrap();
        let current = DeviceBufferRetention::lane_pair(
            lane.clone(),
            Arc::clone(&lease),
            Arc::clone(&storage),
        );
        assert!(current.same_owners(&current.clone()));
        let different_lease =
            DeviceBufferRetention::lane_pair(lane, Arc::new(()), Arc::clone(&storage));
        assert!(!current.same_owners(&different_lease));
        let different_scope = DeviceBufferRetention::pair(Arc::clone(&lease), Arc::clone(&storage));
        assert!(!current.same_owners(&different_scope));
        let different_lane =
            DeviceBufferRetention::lane_pair(ExecutionLaneId::mint().unwrap(), lease, storage);
        assert!(!current.same_owners(&different_lane));
    }
}
