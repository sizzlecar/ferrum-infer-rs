//! Explicit same-wave encoder diagnostic. Reference commands are never submitted.
use std::collections::BTreeMap;

use super::foundation::invalid_operation;
use super::segment_dispatch::SegmentEncodedNode;
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
