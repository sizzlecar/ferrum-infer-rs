use super::*;
use ferrum_interfaces::vnext::{
    NativeCheckpointTransferCostDomain, NativeCheckpointTransferHostWork,
    NativeCheckpointTransferKind,
};
use ferrum_scheduler::implementations::continuous::slo_planner::PlanningUnknownReason;
use sha2::{Digest, Sha256};

/// Same conversion for real success receipts and fresh numeric projections.
/// The exact domain contains plan/ABI/runtime/device and ordered physical
/// transfer geometry, excluding per-request identities and raw addresses.
/// Token extents remain exact because the full publication/ack interval copies
/// and validates these inputs. This does not establish equal cache-population
/// work or make token lengths an elapsed-time formula.
pub(in crate::continuous_engine) fn prefix_cost_shape(
    domain: &NativeCheckpointTransferCostDomain,
    host_work: Option<&NativeCheckpointTransferHostWork>,
) -> Result<model::WaveExecutionShape, PlanningUnknownReason> {
    let host_work = host_work.ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
    let geometry = domain.geometry();
    let total = geometry
        .copy_bytes()
        .checked_add(geometry.initialization_bytes())
        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    let commands = geometry
        .copy_commands()
        .checked_add(geometry.initialization_commands())
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n != 0)
        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
    if total == 0 || geometry.copy_commands() == 0 {
        return Err(PlanningUnknownReason::InvalidShapeEvidence);
    }
    let mut digest = Sha256::new();
    digest.update(b"ferrum-prefix-product-transfer-host-lengths-exact-v2");
    for value in [
        domain.plan_hash().as_str(),
        domain.layout_fingerprint(),
        domain.byte_plan_fingerprint(),
        domain.runtime_implementation_fingerprint(),
        domain.device_id().as_str(),
    ] {
        digest.update(
            u64::try_from(value.len())
                .map_err(|_| PlanningUnknownReason::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        digest.update(value.as_bytes());
    }
    digest.update([match domain.kind() {
        NativeCheckpointTransferKind::Capture => 0,
        NativeCheckpointTransferKind::Restore => 1,
    }]);
    for n in [
        geometry.copy_bytes(),
        geometry.copy_commands(),
        geometry.initialization_bytes(),
        geometry.initialization_commands(),
    ] {
        digest.update(n.to_le_bytes());
    }
    digest.update(geometry.ordered_fragments_fingerprint());
    digest.update(host_work.prefix_tokens().to_le_bytes());
    digest.update(host_work.full_input_tokens().to_le_bytes());
    let capture = domain.kind() == NativeCheckpointTransferKind::Capture;
    Ok(model::WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        numeric_features: None,
        kind: if capture {
            model::WaveKind::Maintenance
        } else {
            model::WaveKind::Restore
        },
        path: model::WaveExecutionPath::PlanRuntime,
        provider_signature: digest.finalize().into(),
        // Scope is complete product publication/ack, never native outbox.
        output_policy_signature: Sha256::digest(b"ferrum-prefix-product-publication-ack-v1").into(),
        graph_state: model::WaveGraphState::Disabled,
        order: model::BatchOrderSemantics::Ordered,
        decode_kv_tokens: Vec::new(),
        prefill_chunks: Vec::new(),
        recurrent_state_bytes: 0,
        restore_bytes: if capture { 0 } else { geometry.copy_bytes() },
        maintenance_bytes: if capture {
            total
        } else {
            geometry.initialization_bytes()
        },
        maintenance_units: commands,
    })
}
