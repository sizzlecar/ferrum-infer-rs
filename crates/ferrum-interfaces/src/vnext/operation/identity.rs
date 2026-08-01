use serde::{ser::SerializeSeq, Serialize, Serializer};
use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

use super::super::{
    BatchInvocationId, BatchStepId, DeviceId, ExecutionIdentityEnvelope, ExecutionLaneId, NodeId,
    OperationId, ParticipantNodeKey, PlanHash, PlanId, ProviderId, VNextError,
};
use super::compiled_identity::{
    CompiledSubmissionWaveIdentity, SubmissionWaveParticipantIdentitySeed,
};
use super::foundation::{canonical_operation_fingerprint, canonical_sha256, invalid_operation};
use super::ProviderExecutionSemantics;

#[derive(Debug, PartialEq, Eq, Serialize)]
struct BatchOperationParticipantIdentityData {
    participant_index: u32,
    node_key: ParticipantNodeKey,
    identity: ExecutionIdentityEnvelope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOperationParticipantIdentity {
    data: Arc<BatchOperationParticipantIdentityData>,
}

impl Serialize for BatchOperationParticipantIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_ref().serialize(serializer)
    }
}

impl BatchOperationParticipantIdentity {
    pub(super) fn new(
        participant_index: u32,
        node_key: ParticipantNodeKey,
        identity: ExecutionIdentityEnvelope,
    ) -> Self {
        Self {
            data: Arc::new(BatchOperationParticipantIdentityData {
                participant_index,
                node_key,
                identity,
            }),
        }
    }

    pub fn participant_index(&self) -> u32 {
        self.data.participant_index
    }

    pub fn node_key(&self) -> &ParticipantNodeKey {
        &self.data.node_key
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.data.identity
    }
}

/// One immutable-plan node inside a physical command batch. Participant
/// identities stay node-local even when several nodes share one submission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchOperationNodeIdentity {
    node_index: u32,
    node_id: NodeId,
    operation_id: OperationId,
    provider_id: ProviderId,
    provider_implementation_fingerprint: String,
    provider_execution_semantics: ProviderExecutionSemantics,
    work_shape_fingerprint: String,
    participants: Vec<BatchOperationParticipantIdentity>,
    fingerprint: String,
}

impl BatchOperationNodeIdentity {
    pub(super) fn from_validated(
        node_index: u32,
        node_id: NodeId,
        operation_id: OperationId,
        provider_id: ProviderId,
        provider_implementation_fingerprint: String,
        provider_execution_semantics: ProviderExecutionSemantics,
        work_shape_fingerprint: String,
        participants: Vec<BatchOperationParticipantIdentity>,
    ) -> Result<Self, VNextError> {
        let participant_start = participants
            .first()
            .map(BatchOperationParticipantIdentity::participant_index);
        if participants.is_empty()
            || participants.iter().enumerate().any(|(index, participant)| {
                participant_start.and_then(|start| start.checked_add(index as u32))
                    != Some(participant.participant_index())
                    || participant.node_key().node_id() != &node_id
                    || participant.identity().parts().frame_id
                        != Some(participant.node_key().frame_id())
                    || participant.identity().parts().node_id.as_ref() != Some(&node_id)
                    || participant.identity().parts().operation_id.as_ref() != Some(&operation_id)
                    || participant.identity().parts().provider_id.as_ref() != Some(&provider_id)
            })
            || participants
                .windows(2)
                .any(|pair| pair[0].node_key() >= pair[1].node_key())
            || !canonical_sha256(&provider_implementation_fingerprint)
            || !canonical_sha256(&work_shape_fingerprint)
        {
            return Err(invalid_operation(
                "batch node identity is empty, non-canonical, or differs from its participant projections",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            node_index: u32,
            node_id: &'a NodeId,
            operation_id: &'a OperationId,
            provider_id: &'a ProviderId,
            provider_implementation_fingerprint: &'a str,
            provider_execution_semantics: ProviderExecutionSemantics,
            work_shape_fingerprint: &'a str,
            participants: &'a [BatchOperationParticipantIdentity],
        }
        let fingerprint = canonical_operation_fingerprint(
            &FingerprintInput {
                domain: "ferrum.runtime-vnext.batch-operation-node-identity.v2",
                node_index,
                node_id: &node_id,
                operation_id: &operation_id,
                provider_id: &provider_id,
                provider_implementation_fingerprint: &provider_implementation_fingerprint,
                provider_execution_semantics,
                work_shape_fingerprint: &work_shape_fingerprint,
                participants: &participants,
            },
            "batch node identity encode failed",
        )?;
        Ok(Self {
            node_index,
            node_id,
            operation_id,
            provider_id,
            provider_implementation_fingerprint,
            provider_execution_semantics,
            work_shape_fingerprint,
            participants,
            fingerprint,
        })
    }

    pub const fn node_index(&self) -> u32 {
        self.node_index
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub const fn provider_execution_semantics(&self) -> ProviderExecutionSemantics {
        self.provider_execution_semantics
    }

    pub fn work_shape_fingerprint(&self) -> &str {
        &self.work_shape_fingerprint
    }

    pub fn participants(&self) -> &[BatchOperationParticipantIdentity] {
        &self.participants
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(super) fn contains_identity(&self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.participants
            .iter()
            .any(|participant| participant.identity() == identity)
    }
}

/// One physical command-batch attempt identity. It may contain one operation
/// or the entire immutable-plan wave, but it always maps to one submit/fence.
#[derive(Debug)]
struct BatchOperationIdentityData {
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
    lane_id: ExecutionLaneId,
    claimed_backing_fingerprint: String,
    nodes: OnceLock<Vec<BatchOperationNodeIdentity>>,
    participants: OnceLock<Vec<BatchOperationParticipantIdentity>>,
    deferred_recipe: Option<DeferredBatchOperationIdentityRecipe>,
    fingerprint: String,
}

#[derive(Debug, Clone)]
pub struct BatchOperationIdentity {
    data: Arc<BatchOperationIdentityData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchOperationIdentityMaterializationSnapshot {
    logical_nodes: u32,
    materialized_nodes: u32,
    full_participant_projection: bool,
}

impl BatchOperationIdentityMaterializationSnapshot {
    pub const fn logical_nodes(self) -> u32 {
        self.logical_nodes
    }

    pub const fn materialized_nodes(self) -> u32 {
        self.materialized_nodes
    }

    pub const fn full_participant_projection(self) -> bool {
        self.full_participant_projection
    }
}

impl PartialEq for BatchOperationIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.data.batch_step_id == other.data.batch_step_id
            && self.data.batch_invocation_id == other.data.batch_invocation_id
            && self.data.plan_id == other.data.plan_id
            && self.data.plan_hash == other.data.plan_hash
            && self.data.device_id == other.data.device_id
            && self.data.runtime_implementation_fingerprint
                == other.data.runtime_implementation_fingerprint
            && self.data.lane_id == other.data.lane_id
            && self.data.claimed_backing_fingerprint == other.data.claimed_backing_fingerprint
            && self.data.fingerprint == other.data.fingerprint
    }
}

impl Eq for BatchOperationIdentity {}

impl Serialize for BatchOperationIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Wire<'a> {
            batch_step_id: BatchStepId,
            batch_invocation_id: BatchInvocationId,
            plan_id: &'a PlanId,
            plan_hash: &'a PlanHash,
            device_id: &'a DeviceId,
            runtime_implementation_fingerprint: &'a str,
            lane_id: ExecutionLaneId,
            claimed_backing_fingerprint: &'a str,
            nodes: &'a [BatchOperationNodeIdentity],
            participants: &'a [BatchOperationParticipantIdentity],
            fingerprint: &'a str,
        }

        Wire {
            batch_step_id: self.data.batch_step_id,
            batch_invocation_id: self.data.batch_invocation_id,
            plan_id: &self.data.plan_id,
            plan_hash: &self.data.plan_hash,
            device_id: &self.data.device_id,
            runtime_implementation_fingerprint: &self.data.runtime_implementation_fingerprint,
            lane_id: self.data.lane_id,
            claimed_backing_fingerprint: &self.data.claimed_backing_fingerprint,
            nodes: self.nodes(),
            participants: self.participants(),
            fingerprint: &self.data.fingerprint,
        }
        .serialize(serializer)
    }
}

struct BatchOperationNodeFingerprints<'a>(&'a [BatchOperationNodeIdentity]);

impl Serialize for BatchOperationNodeFingerprints<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for node in self.0 {
            sequence.serialize_element(node.fingerprint())?;
        }
        sequence.end()
    }
}

impl BatchOperationIdentity {
    #[allow(clippy::too_many_arguments)]
    fn from_deferred_validated(
        batch_step_id: BatchStepId,
        batch_invocation_id: BatchInvocationId,
        plan_id: PlanId,
        plan_hash: PlanHash,
        device_id: DeviceId,
        runtime_implementation_fingerprint: String,
        lane_id: ExecutionLaneId,
        claimed_backing_fingerprint: String,
        deferred_recipe: DeferredBatchOperationIdentityRecipe,
        fingerprint: String,
    ) -> Self {
        Self {
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
                deferred_recipe: Some(deferred_recipe),
                fingerprint,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_validated(
        batch_step_id: BatchStepId,
        batch_invocation_id: BatchInvocationId,
        plan_id: PlanId,
        plan_hash: PlanHash,
        device_id: DeviceId,
        runtime_implementation_fingerprint: String,
        lane_id: ExecutionLaneId,
        claimed_backing_fingerprint: String,
        nodes: Vec<BatchOperationNodeIdentity>,
    ) -> Result<Self, VNextError> {
        if nodes.is_empty()
            || nodes.iter().enumerate().any(|(index, node)| {
                node.node_index as usize != index
                    || node.participants.iter().any(|participant| {
                        participant.identity().parts().plan_id.as_ref() != Some(&plan_id)
                            || participant.identity().parts().plan_hash.as_ref() != Some(&plan_hash)
                            || participant.identity().parts().device_id.as_ref() != Some(&device_id)
                            || participant
                                .identity()
                                .parts()
                                .runtime_implementation_fingerprint
                                .as_deref()
                                != Some(runtime_implementation_fingerprint.as_str())
                    })
            })
            || nodes
                .iter()
                .map(BatchOperationNodeIdentity::node_id)
                .collect::<BTreeSet<_>>()
                .len()
                != nodes.len()
            || !canonical_sha256(&runtime_implementation_fingerprint)
            || !canonical_sha256(&claimed_backing_fingerprint)
        {
            return Err(invalid_operation(
                "physical batch identity is empty, non-canonical, or differs from its plan/runtime projections",
            ));
        }
        let participants = nodes
            .iter()
            .flat_map(|node| node.participants.iter().cloned())
            .collect::<Vec<_>>();
        if participants
            .iter()
            .enumerate()
            .any(|(index, participant)| participant.participant_index() as usize != index)
        {
            return Err(invalid_operation(
                "physical batch participant indices are not globally contiguous",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            batch_step_id: BatchStepId,
            batch_invocation_id: BatchInvocationId,
            plan_id: &'a PlanId,
            plan_hash: &'a PlanHash,
            device_id: &'a DeviceId,
            runtime_implementation_fingerprint: &'a str,
            lane_id: ExecutionLaneId,
            claimed_backing_fingerprint: &'a str,
            node_fingerprints: BatchOperationNodeFingerprints<'a>,
        }
        let fingerprint = canonical_operation_fingerprint(
            &FingerprintInput {
                domain: "ferrum.runtime-vnext.physical-command-batch-identity.v2",
                batch_step_id,
                batch_invocation_id,
                plan_id: &plan_id,
                plan_hash: &plan_hash,
                device_id: &device_id,
                runtime_implementation_fingerprint: &runtime_implementation_fingerprint,
                lane_id,
                claimed_backing_fingerprint: &claimed_backing_fingerprint,
                node_fingerprints: BatchOperationNodeFingerprints(&nodes),
            },
            "physical batch identity encode failed",
        )?;
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
                nodes: OnceLock::from(nodes),
                participants: OnceLock::from(participants),
                deferred_recipe: None,
                fingerprint,
            }),
        })
    }

    pub fn batch_step_id(&self) -> BatchStepId {
        self.data.batch_step_id
    }

    pub fn batch_invocation_id(&self) -> BatchInvocationId {
        self.data.batch_invocation_id
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

    pub fn claimed_backing_fingerprint(&self) -> &str {
        &self.data.claimed_backing_fingerprint
    }

    pub fn node_count(&self) -> usize {
        self.data.nodes.get().map_or_else(
            || {
                self.data
                    .deferred_recipe
                    .as_ref()
                    .map_or(0, |recipe| recipe.topology.node_count())
            },
            Vec::len,
        )
    }

    fn materialized_node_count(&self) -> usize {
        self.data.nodes.get().map_or_else(
            || {
                self.data.deferred_recipe.as_ref().map_or(0, |recipe| {
                    recipe
                        .node_identities
                        .iter()
                        .filter(|identity| identity.get().is_some())
                        .count()
                })
            },
            Vec::len,
        )
    }

    pub fn materialization_snapshot(&self) -> BatchOperationIdentityMaterializationSnapshot {
        BatchOperationIdentityMaterializationSnapshot {
            logical_nodes: u32::try_from(self.node_count())
                .expect("validated physical batch node count fits u32"),
            materialized_nodes: u32::try_from(self.materialized_node_count())
                .expect("materialized physical batch node count fits u32"),
            full_participant_projection: self.data.participants.get().is_some(),
        }
    }

    pub fn node_participant_count(&self, node_index: usize) -> Option<usize> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes.get(node_index).map(|node| node.participants().len());
        }
        let recipe = self.data.deferred_recipe.as_ref()?;
        (node_index < recipe.topology.node_count()).then_some(recipe.participant_seeds.len())
    }

    pub fn node_id_at(&self, node_index: usize) -> Option<&NodeId> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes
                .get(node_index)
                .map(BatchOperationNodeIdentity::node_id);
        }
        self.data
            .deferred_recipe
            .as_ref()?
            .topology
            .node_id_at(node_index)
    }

    pub fn operation_id_at(&self, node_index: usize) -> Option<&OperationId> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes
                .get(node_index)
                .map(BatchOperationNodeIdentity::operation_id);
        }
        self.data
            .deferred_recipe
            .as_ref()?
            .topology
            .operation_id_at(node_index)
    }

    pub fn provider_id_at(&self, node_index: usize) -> Option<&ProviderId> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes
                .get(node_index)
                .map(BatchOperationNodeIdentity::provider_id);
        }
        self.data
            .deferred_recipe
            .as_ref()?
            .topology
            .provider_id_at(node_index)
    }

    pub fn work_shape_fingerprint_at(&self, node_index: usize) -> Option<&str> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes
                .get(node_index)
                .map(BatchOperationNodeIdentity::work_shape_fingerprint);
        }
        let recipe = self.data.deferred_recipe.as_ref()?;
        (node_index < recipe.topology.node_count()).then_some(recipe.work_shape_fingerprint())
    }

    pub fn node_index(&self, node_id: &NodeId) -> Option<usize> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes.iter().position(|node| node.node_id() == node_id);
        }
        self.data
            .deferred_recipe
            .as_ref()?
            .topology
            .node_index(node_id)
    }

    pub(crate) fn materialize_node(
        &self,
        node_index: usize,
    ) -> Result<&BatchOperationNodeIdentity, VNextError> {
        if let Some(nodes) = self.data.nodes.get() {
            return nodes
                .get(node_index)
                .ok_or_else(|| invalid_operation("physical batch node index is out of bounds"));
        }
        let recipe = self.data.deferred_recipe.as_ref().ok_or_else(|| {
            invalid_operation("physical batch has neither materialized nodes nor a compiled recipe")
        })?;
        let slot = recipe.node_identities.get(node_index).ok_or_else(|| {
            invalid_operation("compiled physical batch node index is out of bounds")
        })?;
        if let Some(identity) = slot.get() {
            return Ok(identity);
        }
        let identity = recipe.materialize_node(node_index)?;
        let _ = slot.set(identity);
        slot.get().ok_or_else(|| {
            invalid_operation("compiled physical batch node identity publication failed")
        })
    }

    pub fn nodes(&self) -> &[BatchOperationNodeIdentity] {
        self.data.nodes.get_or_init(|| {
            (0..self.node_count())
                .map(|node_index| {
                    self.materialize_node(node_index)
                        .expect("validated compiled physical batch node must materialize")
                        .clone()
                })
                .collect()
        })
    }

    pub fn single_node(&self) -> Option<&BatchOperationNodeIdentity> {
        (self.node_count() == 1).then(|| {
            self.materialize_node(0)
                .expect("validated single-node physical batch must materialize")
        })
    }

    pub fn participants(&self) -> &[BatchOperationParticipantIdentity] {
        self.data.participants.get_or_init(|| {
            self.nodes()
                .iter()
                .flat_map(|node| node.participants().iter().cloned())
                .collect()
        })
    }

    pub fn fingerprint(&self) -> &str {
        &self.data.fingerprint
    }

    pub(super) fn contains_identity(&self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.participants()
            .iter()
            .any(|participant| participant.identity() == identity)
    }
}

#[derive(Debug)]
struct DeferredBatchOperationIdentityRecipe {
    topology: CompiledSubmissionWaveIdentity,
    work_shape_fingerprint: String,
    participant_seeds: Vec<SubmissionWaveParticipantIdentitySeed>,
    node_identities: Box<[OnceLock<BatchOperationNodeIdentity>]>,
}

impl DeferredBatchOperationIdentityRecipe {
    fn work_shape_fingerprint(&self) -> &str {
        &self.work_shape_fingerprint
    }

    fn materialize_node(
        &self,
        node_index: usize,
    ) -> Result<BatchOperationNodeIdentity, VNextError> {
        let node = self
            .topology
            .node_at(node_index)
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
                let frame = seed.frame();
                let identity = seed
                    .operation_identity(&self.topology, node_index)
                    .ok_or_else(|| invalid_operation("compiled wave node identity disappeared"))?;
                Ok(BatchOperationParticipantIdentity::new(
                    participant_start
                        .checked_add(local_index)
                        .expect("compiled wave participant index was validated"),
                    ParticipantNodeKey::new(
                        frame.participant(),
                        frame.frame_id(),
                        node.node_id().clone(),
                    ),
                    identity,
                ))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        BatchOperationNodeIdentity::from_validated(
            node.node_index(),
            node.node_id().clone(),
            node.operation_id().clone(),
            node.provider_id().clone(),
            node.provider_implementation_fingerprint().to_owned(),
            node.provider_execution_semantics(),
            self.work_shape_fingerprint.clone(),
            participants,
        )
    }
}

impl BatchOperationIdentity {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_compiled_wave(
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
                let left = pair[0].frame().participant();
                let right = pair[1].frame().participant();
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
                seed.runtime_implementation_fingerprint()
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
        Ok(Self::from_deferred_validated(
            batch_step_id,
            batch_invocation_id,
            plan_id,
            plan_hash,
            device_id,
            runtime_implementation_fingerprint,
            lane_id,
            claimed_backing_fingerprint,
            DeferredBatchOperationIdentityRecipe {
                topology,
                work_shape_fingerprint,
                participant_seeds,
                node_identities,
            },
            fingerprint,
        ))
    }
}

#[cfg(test)]
mod batch_operation_identity_fingerprint_tests {
    use super::{
        canonical_operation_fingerprint, BatchOperationNodeFingerprints,
        BatchOperationNodeIdentity, NodeId, OperationId, ProviderExecutionSemantics, ProviderId,
        Serialize,
    };
    use sha2::{Digest, Sha256};

    fn fingerprint_node(index: u32, marker: char) -> BatchOperationNodeIdentity {
        BatchOperationNodeIdentity {
            node_index: index,
            node_id: NodeId::new(format!("node.{index}")).unwrap(),
            operation_id: OperationId::new(format!("operation.{index}")).unwrap(),
            provider_id: ProviderId::new(format!("provider.{index}")).unwrap(),
            provider_implementation_fingerprint: std::iter::repeat_n(marker, 64).collect(),
            provider_execution_semantics: ProviderExecutionSemantics::bitwise_eager_and_replay(),
            work_shape_fingerprint: std::iter::repeat_n(marker, 64).collect(),
            participants: Vec::new(),
            fingerprint: std::iter::repeat_n(marker, 64).collect(),
        }
    }

    #[test]
    fn streaming_fingerprint_matches_canonical_json_digest() {
        #[derive(Serialize)]
        struct Input<'a> {
            domain: &'static str,
            value: &'a str,
        }

        let input = Input {
            domain: "ferrum.runtime-vnext.test",
            value: "evidence",
        };
        let expected = format!("{:x}", Sha256::digest(serde_json::to_vec(&input).unwrap()));

        assert_eq!(
            canonical_operation_fingerprint(&input, "test fingerprint").unwrap(),
            expected
        );
    }

    #[test]
    fn batch_fingerprint_projection_contains_only_validated_node_digests() {
        let first = std::iter::repeat_n('a', 64).collect::<String>();
        let second = std::iter::repeat_n('b', 64).collect::<String>();
        let nodes = [fingerprint_node(0, 'a'), fingerprint_node(1, 'b')];

        let encoded = serde_json::to_string(&BatchOperationNodeFingerprints(&nodes)).unwrap();

        assert_eq!(encoded, format!("[\"{first}\",\"{second}\"]"));
        assert!(!encoded.contains("node.0"));
        assert!(!encoded.contains("operation.0"));
        assert!(!encoded.contains("participants"));
    }
}
