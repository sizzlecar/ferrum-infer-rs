use std::sync::Arc;

use serde::Serialize;

use super::super::{
    DeviceId, DeviceRuntime, ExecutablePlanView, ExecutionLane, ExecutionLaneId, NodeInvocationId,
    PlanHash, PlanId, PreparedStepSubmissionWave, SpanId, TrustedActiveSequenceBinding, VNextError,
};
use super::compiled_identity::{
    CompiledSubmissionWaveIdentity, CompiledSubmissionWaveNodeIdentityTemplate,
    SubmissionWaveParticipantIdentitySeed,
};
use super::foundation::{canonical_operation_fingerprint, invalid_operation};
use super::{BatchOperationIdentity, OperationDispatch};

impl OperationDispatch {
    pub fn compile_submission_wave_identity<R>(
        resolved: &dyn ExecutablePlanView,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<CompiledSubmissionWaveIdentity, VNextError>
    where
        R: DeviceRuntime,
    {
        let plan = resolved.execution_plan();
        let nodes = plan.payload().nodes();
        if nodes.is_empty()
            || lane.descriptor() != resolved.device()
            || lane.descriptor() != resolved.capabilities().device()
            || lane.descriptor().id != *plan.payload().device_id()
            || lane.descriptor().runtime_implementation_fingerprint
                != plan.payload().device_runtime_implementation_fingerprint()
        {
            return Err(invalid_operation(
                "compiled submission-wave identity requires one exact plan/runtime/lane topology",
            ));
        }
        let nodes = nodes
            .iter()
            .enumerate()
            .map(|(node_index, node)| {
                Ok(CompiledSubmissionWaveNodeIdentityTemplate::new(
                    u32::try_from(node_index).map_err(|_| {
                        invalid_operation("compiled submission-wave node index exceeds u32")
                    })?,
                    node.id().clone(),
                    node.operation_id().clone(),
                    node.selection().selected_provider().clone(),
                    node.provider_implementation_fingerprint().to_owned(),
                    node.provider_execution_semantics(),
                ))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            plan_id: &'a PlanId,
            plan_hash: &'a PlanHash,
            device_id: &'a DeviceId,
            runtime_implementation_fingerprint: &'a str,
            lane_id: ExecutionLaneId,
            nodes: &'a [CompiledSubmissionWaveNodeIdentityTemplate],
        }
        let fingerprint = canonical_operation_fingerprint(
            &FingerprintInput {
                domain: "ferrum.runtime-vnext.compiled-submission-wave-identity.v2",
                plan_id: plan.payload().plan_id(),
                plan_hash: plan.plan_hash(),
                device_id: plan.payload().device_id(),
                runtime_implementation_fingerprint: plan
                    .payload()
                    .device_runtime_implementation_fingerprint(),
                lane_id: lane.id(),
                nodes: &nodes,
            },
            "compiled submission-wave identity encode failed",
        )?;
        Ok(CompiledSubmissionWaveIdentity::from_validated(
            plan.payload().plan_id().clone(),
            plan.plan_hash().clone(),
            plan.payload().device_id().clone(),
            plan.payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            lane.id(),
            nodes,
            fingerprint,
        ))
    }

    pub fn bind_compiled_submission_wave_identity<'binding, R, I>(
        topology: &CompiledSubmissionWaveIdentity,
        active_bindings: I,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let Some(first_node) = wave.nodes().first() else {
            return Err(invalid_operation(
                "compiled submission wave requires a non-empty immutable plan",
            ));
        };
        let plan_evidence = first_node.plan_evidence_ref();
        let frames = first_node.participant_frames();
        if active_bindings.len() == 0
            || active_bindings.len() != frames.len()
            || active_bindings.len() != first_node.participants().len()
            || active_bindings.len() != first_node.participant_session_identities().len()
            || wave.execution_lane_id() != topology.lane_id()
            || lane.id() != topology.lane_id()
            || !Arc::ptr_eq(first_node.runtime(), lane.runtime_arc())
            || lane.descriptor().id != *topology.device_id()
            || lane.descriptor().runtime_implementation_fingerprint
                != topology.runtime_implementation_fingerprint()
            || plan_evidence.plan_id() != topology.plan_id()
            || plan_evidence.plan_hash() != topology.plan_hash()
            || plan_evidence.device_id() != topology.device_id()
            || plan_evidence.runtime_implementation_fingerprint()
                != topology.runtime_implementation_fingerprint()
            || wave.claimed_backing().plan_hash() != topology.plan_hash()
            || wave.node_count() != topology.node_count()
            || wave.nodes().iter().enumerate().any(|(node_index, node)| {
                topology
                    .node_id_at(node_index)
                    .is_none_or(|compiled_node_id| {
                        node.node_id() != compiled_node_id
                            || node.participant_frames() != frames
                            || node.work_shape().fingerprint()
                                != first_node.work_shape().fingerprint()
                    })
            })
        {
            return Err(invalid_operation(
                "compiled submission-wave topology differs from its exact plan, lane, work, or participant authority",
            ));
        }

        let participant_seeds = first_node
            .participants()
            .zip(frames.iter().copied())
            .zip(first_node.participant_session_identities())
            .zip(active_bindings)
            .map(
                |(((participant, frame), (session_epoch, session_fingerprint)), active)| {
                    active.ensure_open_for_emission()?;
                    if frame.sequence_authority() != participant.sequence_authority()
                        || frame.request_authority() != participant.request_authority()
                        || active.sequence_authority() != participant.sequence_authority()
                        || active.coordinator_id() != participant.coordinator_id()
                        || active.run_id() != participant.run_id()
                        || active.request_id() != participant.request_id()
                        || !active
                            .matches_sequence_session(session_epoch, session_fingerprint)
                        || active.plan().plan_id() != topology.plan_id()
                        || active.plan().plan_hash() != topology.plan_hash()
                        || active.plan().device_id() != topology.device_id()
                        || active.runtime_implementation_fingerprint()
                            != topology.runtime_implementation_fingerprint()
                    {
                        return Err(invalid_operation(
                            "compiled submission-wave participant differs from its live sequence session",
                        ));
                    }
                    let node_count = u64::try_from(topology.node_count()).map_err(|_| {
                        invalid_operation("compiled submission-wave node count exceeds u64")
                    })?;
                    let completed_frames = frame.frame_id().get() - 1;
                    let last_node_index = node_count - 1;
                    let last_node_invocation = completed_frames
                        .checked_mul(node_count)
                        .and_then(|value| value.checked_add(last_node_index))
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| {
                            invalid_operation(
                                "compiled submission-wave node invocation id space is exhausted",
                            )
                        })?;
                    NodeInvocationId::try_from(last_node_invocation)?;
                    let events_per_frame = node_count
                        .checked_mul(3)
                        .and_then(|value| value.checked_add(2))
                        .ok_or_else(|| {
                            invalid_operation(
                                "compiled submission-wave event sequence space is exhausted",
                            )
                        })?;
                    completed_frames
                        .checked_mul(events_per_frame)
                        .and_then(|value| value.checked_add(last_node_index.checked_mul(3)?))
                        .and_then(|value| value.checked_add(5))
                        .ok_or_else(|| {
                            invalid_operation(
                                "compiled submission-wave event sequence space is exhausted",
                            )
                        })?;
                    let span_root = format!("vnext/request/{}", active.fingerprint());
                    let node_span = SpanId::new(format!(
                        "{span_root}/frame/{}/node/{last_node_invocation}",
                        frame.frame_id()
                    ))?;
                    SpanId::new(format!("{node_span}/operation"))?;
                    let provisioning = active.static_provisioning_identity();
                    Ok(SubmissionWaveParticipantIdentitySeed::new(
                        frame,
                        active.run_id().clone(),
                        active.request_id().clone(),
                        active.static_pool_id(),
                        active
                            .static_pool_identity_fingerprint_ref()
                            .map(str::to_owned),
                        provisioning.map(|identity| identity.run_id().clone()),
                        provisioning.map(|identity| identity.request_id().clone()),
                        provisioning.map(|identity| identity.transaction_id().clone()),
                        active.sequence_authority().sparse_id(),
                        active.sequence_authority().generation(),
                        active.activation_epoch(),
                        active.runtime_implementation_fingerprint().to_owned(),
                        active.fingerprint().to_owned(),
                        span_root,
                    ))
                },
            )
            .collect::<Result<Vec<_>, VNextError>>()?;

        BatchOperationIdentity::from_compiled_wave(
            topology.clone(),
            wave.batch_step_id(),
            wave.batch_invocation_id(),
            wave.fingerprint().to_owned(),
            first_node.work_shape().fingerprint().to_owned(),
            participant_seeds,
        )
    }
}
