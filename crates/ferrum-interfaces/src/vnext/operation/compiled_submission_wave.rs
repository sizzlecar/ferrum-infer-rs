use serde::Serialize;
use std::sync::{Arc, OnceLock};

use super::{
    canonical_operation_fingerprint, canonical_sha256, invalid_operation, BatchInvocationId,
    BatchOperationIdentity, BatchOperationIdentityData, BatchOperationNodeIdentity,
    BatchOperationParticipantIdentity, BatchStepId, DeviceId, DeviceRuntime, ExecutablePlanView,
    ExecutionIdentityEnvelope, ExecutionIdentityParts, ExecutionLane, ExecutionLaneId, NodeId,
    NodeInvocationId, OperationDispatch, OperationId, ParticipantNodeKey, PlanHash, PlanId,
    PreparedStepSubmissionWave, ProviderExecutionSemantics, ProviderId, RequestIdentity,
    ResourcePoolId, RunId, SpanId, StepParticipantFrameAssignment, TransactionId,
    TrustedActiveSequenceBinding, VNextError, EXECUTION_IDENTITY_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CompiledSubmissionWaveNodeIdentityTemplate {
    node_index: u32,
    node_id: NodeId,
    operation_id: OperationId,
    provider_id: ProviderId,
    provider_implementation_fingerprint: String,
    provider_execution_semantics: ProviderExecutionSemantics,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct CompiledSubmissionWaveIdentityData {
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
    lane_id: ExecutionLaneId,
    nodes: Vec<CompiledSubmissionWaveNodeIdentityTemplate>,
    fingerprint: String,
}

/// Cold-path identity topology for one immutable plan on one execution lane.
///
/// The topology owns only plan-stable node/provider facts. A physical wave
/// binds participant/frame seeds to it and materializes full operation
/// identities only for nodes that are encoded, observed, or failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledSubmissionWaveIdentity {
    data: Arc<CompiledSubmissionWaveIdentityData>,
}

impl CompiledSubmissionWaveIdentity {
    pub fn plan_id(&self) -> &PlanId {
        &self.data.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.data.plan_hash
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.data.device_id
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.data.runtime_implementation_fingerprint
    }

    pub fn lane_id(&self) -> ExecutionLaneId {
        self.data.lane_id
    }

    pub fn node_count(&self) -> usize {
        self.data.nodes.len()
    }

    pub fn fingerprint(&self) -> &str {
        &self.data.fingerprint
    }

    pub(super) fn node_id_at(&self, node_index: usize) -> Option<&NodeId> {
        self.data.nodes.get(node_index).map(|node| &node.node_id)
    }

    pub(super) fn operation_id_at(&self, node_index: usize) -> Option<&OperationId> {
        self.data
            .nodes
            .get(node_index)
            .map(|node| &node.operation_id)
    }

    pub(super) fn provider_id_at(&self, node_index: usize) -> Option<&ProviderId> {
        self.data
            .nodes
            .get(node_index)
            .map(|node| &node.provider_id)
    }

    pub(super) fn node_index(&self, node_id: &NodeId) -> Option<usize> {
        self.data
            .nodes
            .iter()
            .position(|node| &node.node_id == node_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct SubmissionWaveParticipantIdentitySeed {
    frame: StepParticipantFrameAssignment,
    run_id: RunId,
    request_id: RequestIdentity,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    provisioning_run_id: Option<RunId>,
    provisioning_request_id: Option<RequestIdentity>,
    transaction_id: Option<TransactionId>,
    active_sequence_slot: u32,
    admission_generation: u64,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
    active_sequence_fingerprint: String,
    span_root: String,
}

impl SubmissionWaveParticipantIdentitySeed {
    fn operation_identity(
        &self,
        topology: &CompiledSubmissionWaveIdentityData,
        node: &CompiledSubmissionWaveNodeIdentityTemplate,
    ) -> ExecutionIdentityEnvelope {
        let node_count = u64::try_from(topology.nodes.len())
            .expect("compiled submission-wave node count fits u64");
        let node_index = u64::from(node.node_index);
        let completed_frames = self.frame.frame_id().get() - 1;
        let node_invocation = completed_frames
            .checked_mul(node_count)
            .and_then(|value| value.checked_add(node_index))
            .and_then(|value| value.checked_add(1))
            .expect("compiled submission-wave invocation range was validated");
        let node_invocation_id = NodeInvocationId::try_from(node_invocation)
            .expect("compiled submission-wave invocation id is non-zero");
        let events_per_frame = node_count
            .checked_mul(3)
            .and_then(|value| value.checked_add(2))
            .expect("compiled submission-wave event range was validated");
        let sequence = completed_frames
            .checked_mul(events_per_frame)
            .and_then(|value| value.checked_add(node_index.checked_mul(3)?))
            .and_then(|value| value.checked_add(5))
            .expect("compiled submission-wave event sequence was validated");
        let node_span = SpanId::new(format!(
            "{}/frame/{}/node/{node_invocation}",
            self.span_root,
            self.frame.frame_id()
        ))
        .expect("compiled submission-wave node span is portable");
        let operation_span = SpanId::new(format!("{node_span}/operation"))
            .expect("compiled submission-wave operation span is portable");

        ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
            sequence,
            plan_id: Some(topology.plan_id.clone()),
            plan_hash: Some(topology.plan_hash.clone()),
            frame_id: Some(self.frame.frame_id()),
            node_invocation_id: Some(node_invocation_id),
            node_id: Some(node.node_id.clone()),
            operation_id: Some(node.operation_id.clone()),
            provider_id: Some(node.provider_id.clone()),
            device_id: Some(topology.device_id.clone()),
            resource_pool_id: self.resource_pool_id.clone(),
            resource_pool_identity_fingerprint: self.resource_pool_identity_fingerprint.clone(),
            provisioning_run_id: self.provisioning_run_id.clone(),
            provisioning_request_id: self.provisioning_request_id.clone(),
            transaction_id: self.transaction_id.clone(),
            active_sequence_slot: Some(self.active_sequence_slot),
            admission_generation: Some(self.admission_generation),
            activation_epoch: Some(self.activation_epoch),
            runtime_implementation_fingerprint: Some(
                self.runtime_implementation_fingerprint.clone(),
            ),
            active_sequence_fingerprint: Some(self.active_sequence_fingerprint.clone()),
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
            span_id: operation_span,
            parent_span_id: Some(node_span),
            async_links: Vec::new(),
        })
        .expect("compiled submission-wave participant seed is a valid operation identity")
    }
}

#[derive(Debug)]
pub(super) struct DeferredBatchOperationIdentityRecipe {
    pub(super) topology: CompiledSubmissionWaveIdentity,
    work_shape_fingerprint: String,
    pub(super) participant_seeds: Vec<SubmissionWaveParticipantIdentitySeed>,
    pub(super) node_identities: Box<[OnceLock<BatchOperationNodeIdentity>]>,
}

impl DeferredBatchOperationIdentityRecipe {
    pub(super) fn work_shape_fingerprint(&self) -> &str {
        &self.work_shape_fingerprint
    }

    pub(super) fn materialize_node(
        &self,
        node_index: usize,
    ) -> Result<BatchOperationNodeIdentity, VNextError> {
        let node = self
            .topology
            .data
            .nodes
            .get(node_index)
            .ok_or_else(|| invalid_operation("compiled wave node index is out of bounds"))?;
        let participant_start = node_index
            .checked_mul(self.participant_seeds.len())
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                invalid_operation("compiled wave participant index space exceeds u32")
            })?;
        let participants = self
            .participant_seeds
            .iter()
            .enumerate()
            .map(|(local_index, seed)| {
                let local_index = u32::try_from(local_index)
                    .expect("compiled wave participant count was validated");
                BatchOperationParticipantIdentity::new(
                    participant_start
                        .checked_add(local_index)
                        .expect("compiled wave participant index was validated"),
                    ParticipantNodeKey::new(
                        seed.frame.participant(),
                        seed.frame.frame_id(),
                        node.node_id.clone(),
                    ),
                    seed.operation_identity(&self.topology.data, node),
                )
            })
            .collect();
        BatchOperationNodeIdentity::from_validated(
            node.node_index,
            node.node_id.clone(),
            node.operation_id.clone(),
            node.provider_id.clone(),
            node.provider_implementation_fingerprint.clone(),
            node.provider_execution_semantics,
            self.work_shape_fingerprint.clone(),
            participants,
        )
    }
}

impl BatchOperationIdentity {
    #[allow(clippy::too_many_arguments)]
    fn from_compiled_wave(
        topology: CompiledSubmissionWaveIdentity,
        batch_step_id: BatchStepId,
        batch_invocation_id: BatchInvocationId,
        claimed_backing_fingerprint: String,
        work_shape_fingerprint: String,
        participant_seeds: Vec<SubmissionWaveParticipantIdentitySeed>,
    ) -> Result<Self, VNextError> {
        let participant_count = participant_seeds.len();
        let participant_projection_count = topology
            .node_count()
            .checked_mul(participant_count)
            .and_then(|count| u32::try_from(count).ok());
        if topology.node_count() == 0
            || participant_count == 0
            || participant_projection_count.is_none()
            || participant_seeds.windows(2).any(|pair| {
                let left = pair[0].frame.participant();
                let right = pair[1].frame.participant();
                (
                    left.sequence_authority().sparse_id(),
                    left.sequence_authority().generation(),
                    left.request_authority().sparse_id(),
                    left.request_authority().generation(),
                ) >= (
                    right.sequence_authority().sparse_id(),
                    right.sequence_authority().generation(),
                    right.request_authority().sparse_id(),
                    right.request_authority().generation(),
                )
            })
            || participant_seeds.iter().any(|seed| {
                seed.runtime_implementation_fingerprint
                    != topology.runtime_implementation_fingerprint()
            })
            || !canonical_sha256(&claimed_backing_fingerprint)
            || !canonical_sha256(&work_shape_fingerprint)
        {
            return Err(invalid_operation(
                "compiled physical batch identity is empty, non-canonical, or exceeds its participant index space",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            batch_step_id: BatchStepId,
            batch_invocation_id: BatchInvocationId,
            topology_fingerprint: &'a str,
            claimed_backing_fingerprint: &'a str,
            work_shape_fingerprint: &'a str,
            participant_seeds: &'a [SubmissionWaveParticipantIdentitySeed],
        }
        let fingerprint = canonical_operation_fingerprint(
            &FingerprintInput {
                domain: "ferrum.runtime-vnext.compiled-physical-command-batch-identity.v1",
                batch_step_id,
                batch_invocation_id,
                topology_fingerprint: topology.fingerprint(),
                claimed_backing_fingerprint: &claimed_backing_fingerprint,
                work_shape_fingerprint: &work_shape_fingerprint,
                participant_seeds: &participant_seeds,
            },
            "compiled physical batch identity encode failed",
        )?;
        let node_identities = std::iter::repeat_with(OnceLock::new)
            .take(topology.node_count())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let plan_id = topology.plan_id().clone();
        let plan_hash = topology.plan_hash().clone();
        let device_id = topology.device_id().clone();
        let runtime_implementation_fingerprint =
            topology.runtime_implementation_fingerprint().to_owned();
        let lane_id = topology.lane_id();
        Ok(Self {
            data: Arc::new(BatchOperationIdentityData {
                batch_step_id,
                batch_invocation_id,
                plan_id,
                plan_hash,
                device_id,
                runtime_implementation_fingerprint,
                lane_id,
                claimed_backing_fingerprint,
                nodes: OnceLock::new(),
                participants: OnceLock::new(),
                deferred_recipe: Some(DeferredBatchOperationIdentityRecipe {
                    topology,
                    work_shape_fingerprint,
                    participant_seeds,
                    node_identities,
                }),
                fingerprint,
            }),
        })
    }
}

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
                Ok(CompiledSubmissionWaveNodeIdentityTemplate {
                    node_index: u32::try_from(node_index).map_err(|_| {
                        invalid_operation("compiled submission-wave node index exceeds u32")
                    })?,
                    node_id: node.id().clone(),
                    operation_id: node.operation_id().clone(),
                    provider_id: node.selection().selected_provider().clone(),
                    provider_implementation_fingerprint: node
                        .provider_implementation_fingerprint()
                        .to_owned(),
                    provider_execution_semantics: node.provider_execution_semantics(),
                })
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
        Ok(CompiledSubmissionWaveIdentity {
            data: Arc::new(CompiledSubmissionWaveIdentityData {
                plan_id: plan.payload().plan_id().clone(),
                plan_hash: plan.plan_hash().clone(),
                device_id: plan.payload().device_id().clone(),
                runtime_implementation_fingerprint: plan
                    .payload()
                    .device_runtime_implementation_fingerprint()
                    .to_owned(),
                lane_id: lane.id(),
                nodes,
                fingerprint,
            }),
        })
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
                topology.data.nodes.get(node_index).is_none_or(|template| {
                    node.node_id() != &template.node_id
                        || node.participant_frames() != frames
                        || node.work_shape().fingerprint() != first_node.work_shape().fingerprint()
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
                    Ok(SubmissionWaveParticipantIdentitySeed {
                        frame,
                        run_id: active.run_id().clone(),
                        request_id: active.request_id().clone(),
                        resource_pool_id: active.static_pool_id(),
                        resource_pool_identity_fingerprint: active
                            .static_pool_identity_fingerprint_ref()
                            .map(str::to_owned),
                        provisioning_run_id: provisioning
                            .map(|identity| identity.run_id().clone()),
                        provisioning_request_id: provisioning
                            .map(|identity| identity.request_id().clone()),
                        transaction_id: provisioning
                            .map(|identity| identity.transaction_id().clone()),
                        active_sequence_slot: active.sequence_authority().sparse_id(),
                        admission_generation: active.sequence_authority().generation(),
                        activation_epoch: active.activation_epoch(),
                        runtime_implementation_fingerprint: active
                            .runtime_implementation_fingerprint()
                            .to_owned(),
                        active_sequence_fingerprint: active.fingerprint().to_owned(),
                        span_root,
                    })
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
