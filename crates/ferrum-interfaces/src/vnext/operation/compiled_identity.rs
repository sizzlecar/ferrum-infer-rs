use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::super::{
    DeviceId, ExecutionIdentityEnvelope, ExecutionIdentityParts, ExecutionLaneId, NodeId,
    NodeInvocationId, OperationId, PlanHash, PlanId, ProviderId, RequestIdentity, ResourcePoolId,
    RunId, SpanId, StepParticipantFrameAssignment, TransactionId, EXECUTION_IDENTITY_VERSION,
};
use super::ProviderExecutionSemantics;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct CompiledSubmissionWaveNodeIdentityTemplate {
    node_index: u32,
    node_id: NodeId,
    operation_id: OperationId,
    provider_id: ProviderId,
    provider_implementation_fingerprint: String,
    provider_execution_semantics: ProviderExecutionSemantics,
}

impl CompiledSubmissionWaveNodeIdentityTemplate {
    pub(super) fn new(
        node_index: u32,
        node_id: NodeId,
        operation_id: OperationId,
        provider_id: ProviderId,
        provider_implementation_fingerprint: String,
        provider_execution_semantics: ProviderExecutionSemantics,
    ) -> Self {
        Self {
            node_index,
            node_id,
            operation_id,
            provider_id,
            provider_implementation_fingerprint,
            provider_execution_semantics,
        }
    }

    pub(super) const fn node_index(&self) -> u32 {
        self.node_index
    }

    pub(super) fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub(super) fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub(super) fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub(super) fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub(super) const fn provider_execution_semantics(&self) -> ProviderExecutionSemantics {
        self.provider_execution_semantics
    }
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_validated(
        plan_id: PlanId,
        plan_hash: PlanHash,
        device_id: DeviceId,
        runtime_implementation_fingerprint: String,
        lane_id: ExecutionLaneId,
        nodes: Vec<CompiledSubmissionWaveNodeIdentityTemplate>,
        fingerprint: String,
    ) -> Self {
        Self {
            data: Arc::new(CompiledSubmissionWaveIdentityData {
                plan_id,
                plan_hash,
                device_id,
                runtime_implementation_fingerprint,
                lane_id,
                nodes,
                fingerprint,
            }),
        }
    }

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

    pub(super) fn node_at(
        &self,
        node_index: usize,
    ) -> Option<&CompiledSubmissionWaveNodeIdentityTemplate> {
        self.data.nodes.get(node_index)
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
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
    ) -> Self {
        Self {
            frame,
            run_id,
            request_id,
            resource_pool_id,
            resource_pool_identity_fingerprint,
            provisioning_run_id,
            provisioning_request_id,
            transaction_id,
            active_sequence_slot,
            admission_generation,
            activation_epoch,
            runtime_implementation_fingerprint,
            active_sequence_fingerprint,
            span_root,
        }
    }

    pub(super) const fn frame(&self) -> StepParticipantFrameAssignment {
        self.frame
    }

    pub(super) fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub(super) fn operation_identity(
        &self,
        topology: &CompiledSubmissionWaveIdentity,
        node_index: usize,
    ) -> Option<ExecutionIdentityEnvelope> {
        Some(
            ExecutionIdentityEnvelope::new(self.operation_parts(topology, node_index)?)
                .expect("compiled submission-wave participant seed is a valid operation identity"),
        )
    }

    fn operation_parts(
        &self,
        topology: &CompiledSubmissionWaveIdentity,
        node_index: usize,
    ) -> Option<ExecutionIdentityParts> {
        let node = topology.data.nodes.get(node_index)?;
        let topology = topology.data.as_ref();
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

        Some(ExecutionIdentityParts {
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
    }
}

/// Independent leaf ownership only: no recipe, wave, session or device resources.
pub(crate) struct CompiledOperationIdentity {
    seeds: Arc<[SubmissionWaveParticipantIdentitySeed]>,
    participant_index: usize,
    topology: CompiledSubmissionWaveIdentity,
    node_index: usize,
    node_invocation_id: NodeInvocationId,
    sequence: u64,
    parts: OnceLock<ExecutionIdentityParts>,
    materializations: Arc<AtomicU64>,
}

impl CompiledOperationIdentity {
    pub(crate) fn parts(&self) -> &ExecutionIdentityParts {
        self.parts.get_or_init(|| {
            let parts = self.seeds[self.participant_index]
                .operation_parts(&self.topology, self.node_index)
                .expect("validated compiled node remains in its immutable topology");
            self.materializations.fetch_add(1, Ordering::Relaxed);
            parts
        })
    }

    #[cfg(test)]
    pub(crate) fn is_materialized(&self) -> bool {
        self.parts.get().is_some()
    }

    pub(crate) fn projection(&self) -> super::super::event::ExecutionIdentityProjection<'_> {
        let seed = &self.seeds[self.participant_index];
        let topology = self.topology.data.as_ref();
        let node = &topology.nodes[self.node_index];
        super::super::event::ExecutionIdentityProjection {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: &seed.run_id,
            request_id: &seed.request_id,
            sequence: self.sequence,
            plan_id: Some(&topology.plan_id),
            plan_hash: Some(&topology.plan_hash),
            frame_id: Some(seed.frame.frame_id()),
            node_invocation_id: Some(self.node_invocation_id),
            node_id: Some(&node.node_id),
            operation_id: Some(&node.operation_id),
            provider_id: Some(&node.provider_id),
            device_id: Some(&topology.device_id),
            resource_pool_id: seed.resource_pool_id.as_ref(),
            resource_pool_identity_fingerprint: seed.resource_pool_identity_fingerprint.as_deref(),
            provisioning_run_id: seed.provisioning_run_id.as_ref(),
            provisioning_request_id: seed.provisioning_request_id.as_ref(),
            transaction_id: seed.transaction_id.as_ref(),
            active_sequence_slot: Some(seed.active_sequence_slot),
            admission_generation: Some(seed.admission_generation),
            activation_epoch: Some(seed.activation_epoch),
            runtime_implementation_fingerprint: Some(&seed.runtime_implementation_fingerprint),
            active_sequence_fingerprint: Some(&seed.active_sequence_fingerprint),
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
        }
    }
}

impl SubmissionWaveParticipantIdentitySeed {
    pub(super) fn projected_identity(
        &self,
        seeds: &Arc<[SubmissionWaveParticipantIdentitySeed]>,
        participant_index: usize,
        topology: &CompiledSubmissionWaveIdentity,
        node_index: usize,
        materializations: &Arc<AtomicU64>,
    ) -> Option<ExecutionIdentityEnvelope> {
        let node = topology.data.nodes.get(node_index)?;
        let node_count = u64::try_from(topology.node_count()).ok()?;
        let completed_frames = self.frame.frame_id().get().checked_sub(1)?;
        let node_invocation = completed_frames
            .checked_mul(node_count)?
            .checked_add(u64::from(node.node_index))?
            .checked_add(1)?;
        let events_per_frame = node_count.checked_mul(3)?.checked_add(2)?;
        let sequence = completed_frames
            .checked_mul(events_per_frame)?
            .checked_add(u64::from(node.node_index).checked_mul(3)?)?
            .checked_add(5)?;
        Some(ExecutionIdentityEnvelope::from_compiled(
            CompiledOperationIdentity {
                seeds: Arc::clone(seeds),
                participant_index,
                topology: topology.clone(),
                node_index,
                node_invocation_id: NodeInvocationId::try_from(node_invocation).ok()?,
                sequence,
                parts: OnceLock::new(),
                materializations: Arc::clone(materializations),
            },
        ))
    }
}
