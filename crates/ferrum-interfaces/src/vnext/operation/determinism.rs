use std::collections::BTreeMap;

use super::dispatch::{BoundDeviceSubmissionAttribution, ProfiledSubmissionHandle};
use super::{invalid_operation, BatchOperationIdentity, ElementType};
use crate::vnext::{
    BufferUsage, CompletionHandle, CompletionReadbackBatchRequest,
    CompletionReadbackCollectionObservation, CompletionReadbackCollectionRequest,
    CompletionReadbackRequest, DeviceRuntime, ExecutablePlanView,
    ExecutionDeterminismInitializationSpec, ExecutionDeterminismWitnessKind,
    ExecutionDeterminismWitnessPlan, ExecutionDeterminismWitnessSpec, HostTransferLayout, NodeId,
    PlanHash, PreparedStepSubmissionWave, ResourceId, SubmittedOperationReceipt, VNextError,
};

/// Complete participant-major input/state restoration for one immutable
/// determinism witness plan.
///
/// The constructor accepts bytes only. Node ids, resource ids, offsets,
/// element types, and lengths remain copied from the trusted plan-derived
/// denominator, so a hardware runner cannot silently omit or redirect one
/// state range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionWaveDeterminismRestore {
    plan_hash: PlanHash,
    node_ids: Vec<NodeId>,
    initializations: Vec<ExecutionDeterminismInitializationSpec>,
    participant_payloads: Vec<Vec<Vec<u8>>>,
}

impl SubmissionWaveDeterminismRestore {
    pub fn new(
        witness_plan: &ExecutionDeterminismWitnessPlan,
        participant_payloads: Vec<Vec<Vec<u8>>>,
    ) -> Result<Self, VNextError> {
        if participant_payloads.is_empty()
            || u32::try_from(participant_payloads.len()).is_err()
            || participant_payloads
                .iter()
                .any(|payloads| payloads.len() != witness_plan.initializations().len())
        {
            return Err(invalid_operation(
                "determinism restore must cover every initialization for a non-empty canonical participant set",
            ));
        }
        for payloads in &participant_payloads {
            for (initialization, bytes) in witness_plan.initializations().iter().zip(payloads) {
                let location = initialization.location();
                if u64::try_from(bytes.len()).ok() != Some(location.canonical_length_bytes())
                    || bytes.len()
                        % usize::try_from(location.element_type().size_bytes())
                            .expect("element width fits usize")
                        != 0
                {
                    return Err(invalid_operation(
                        "determinism restore payload differs from its complete typed initialization range",
                    ));
                }
            }
        }
        Ok(Self {
            plan_hash: witness_plan.plan_hash().clone(),
            node_ids: witness_plan.node_ids().to_vec(),
            initializations: witness_plan.initializations().to_vec(),
            participant_payloads,
        })
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }

    pub fn initializations(&self) -> &[ExecutionDeterminismInitializationSpec] {
        &self.initializations
    }

    pub fn participant_count(&self) -> u32 {
        u32::try_from(self.participant_payloads.len())
            .expect("determinism restore participant count was validated")
    }

    pub fn participant_payloads(&self, participant_index: u32) -> Option<&[Vec<u8>]> {
        usize::try_from(participant_index)
            .ok()
            .and_then(|index| self.participant_payloads.get(index))
            .map(Vec::as_slice)
    }

    pub(super) fn validate_for(&self, resolved: &dyn ExecutablePlanView) -> Result<(), VNextError> {
        let actual = resolved
            .execution_plan()
            .determinism_witness_plan_for_nodes(&self.node_ids)?;
        if self.plan_hash != *resolved.execution_plan().plan_hash()
            || self.plan_hash != *actual.plan_hash()
            || self.node_ids != actual.node_ids()
            || self.initializations != actual.initializations()
        {
            return Err(invalid_operation(
                "determinism restore differs from the exact immutable plan initialization denominator",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_for_submission<R: DeviceRuntime>(
        &self,
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<(), VNextError> {
        self.validate_for(resolved)?;
        if self.node_ids.len() != wave.nodes().len()
            || self.node_ids.len() != batch_identity.node_count()
            || self.node_ids.iter().zip(wave.nodes()).enumerate().any(
                |(node_index, (node_id, prepared_node))| {
                    prepared_node.node_id() != node_id
                        || batch_identity.node_id_at(node_index) != Some(node_id)
                },
            )
        {
            return Err(invalid_operation(
                "determinism restore node scope differs from its exact prepared wave",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PhysicalReadbackKey {
    node_id: NodeId,
    resource_id: ResourceId,
    expected_usage: BufferUsage,
    logical_offset_bytes: u64,
    participant_layouts: Vec<(ElementType, u64)>,
}

/// One physical readback group and every semantic witness that projects onto
/// that exact range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionWaveDeterminismReadbackTarget {
    witnesses: Vec<ExecutionDeterminismWitnessSpec>,
    batch: CompletionReadbackBatchRequest,
}

impl SubmissionWaveDeterminismReadbackTarget {
    pub fn witnesses(&self) -> &[ExecutionDeterminismWitnessSpec] {
        &self.witnesses
    }

    pub fn batch(&self) -> &CompletionReadbackBatchRequest {
        &self.batch
    }
}

/// Canonical terminal readback denominator for one prepared deterministic
/// submission wave.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "the exact plan-derived witness readback must be collected"]
pub struct SubmissionWaveDeterminismReadbackPlan {
    plan_hash: PlanHash,
    node_ids: Vec<NodeId>,
    collection: CompletionReadbackCollectionRequest,
    targets: Vec<SubmissionWaveDeterminismReadbackTarget>,
    witness_count: usize,
}

impl SubmissionWaveDeterminismReadbackPlan {
    pub fn from_prepared_wave<R: DeviceRuntime>(
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<Self, VNextError> {
        let node_ids = wave
            .nodes()
            .iter()
            .map(|node| node.node_id().clone())
            .collect::<Vec<_>>();
        let witness_plan = resolved
            .execution_plan()
            .determinism_witness_plan_for_nodes(&node_ids)?;
        if witness_plan.plan_hash() != batch_identity.plan_hash()
            || wave.nodes().len() != batch_identity.node_count()
            || wave.nodes().iter().enumerate().any(|(node_index, node)| {
                node.plan_evidence_ref().plan_hash() != witness_plan.plan_hash()
                    || batch_identity.node_id_at(node_index) != Some(node.node_id())
                    || batch_identity.node_participant_count(node_index)
                        != Some(node.participant_count() as usize)
            })
        {
            return Err(invalid_operation(
                "determinism readback plan differs from its prepared wave or physical batch identity",
            ));
        }
        validate_terminal_witness_stability(resolved, &witness_plan)?;

        let mut grouped = BTreeMap::<
            PhysicalReadbackKey,
            (
                CompletionReadbackBatchRequest,
                Vec<ExecutionDeterminismWitnessSpec>,
            ),
        >::new();

        for witness in witness_plan.witnesses() {
            let node_index = batch_identity
                .node_index(witness.node_id())
                .ok_or_else(|| {
                    invalid_operation(
                        "determinism witness node is absent from its physical batch identity",
                    )
                })?;
            let node = wave.nodes().get(node_index).ok_or_else(|| {
                invalid_operation("determinism witness node is absent from its prepared wave")
            })?;
            if node.node_id() != witness.node_id() {
                return Err(invalid_operation(
                    "determinism witness node index differs from its prepared wave",
                ));
            }

            let element_bytes = witness.element_type().size_bytes();
            let requests = node
                .work_shape()
                .participant_work()
                .iter()
                .enumerate()
                .map(|(participant_index, participant_work)| {
                    let active_bytes =
                        witness.active_length_bytes(participant_work.token_span())?;
                    if active_bytes % element_bytes != 0 {
                        return Err(invalid_operation(
                            "determinism witness active range is not element aligned",
                        ));
                    }
                    CompletionReadbackRequest::new_typed(
                        witness.node_id().clone(),
                        u32::try_from(participant_index).map_err(|_| {
                            invalid_operation("determinism readback participant index exceeds u32")
                        })?,
                        witness.resource_id().clone(),
                        witness.location().usage(),
                        witness.logical_offset_bytes(),
                        HostTransferLayout::new(
                            witness.element_type(),
                            active_bytes / element_bytes,
                        )?,
                    )
                })
                .collect::<Result<Vec<_>, VNextError>>()?;
            let batch = CompletionReadbackBatchRequest::new(requests)?;
            let first = batch
                .requests()
                .first()
                .expect("determinism readback batches are non-empty");
            let key = PhysicalReadbackKey {
                node_id: first.node_id().clone(),
                resource_id: first.resource_id().clone(),
                expected_usage: first.expected_usage(),
                logical_offset_bytes: first.logical_offset_bytes(),
                participant_layouts: batch
                    .requests()
                    .iter()
                    .map(|request| {
                        (
                            request.output_layout().element_type(),
                            request.output_layout().element_count(),
                        )
                    })
                    .collect(),
            };
            match grouped.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((batch, vec![witness.clone()]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().1.push(witness.clone());
                }
            }
        }

        let targets = grouped
            .into_values()
            .map(|(batch, witnesses)| SubmissionWaveDeterminismReadbackTarget { witnesses, batch })
            .collect::<Vec<_>>();
        let collection = CompletionReadbackCollectionRequest::new(
            targets.iter().map(|target| target.batch.clone()).collect(),
        )?;
        let witness_count = targets.iter().map(|target| target.witnesses.len()).sum();
        if witness_count != witness_plan.witnesses().len()
            || collection
                .batches()
                .iter()
                .zip(&targets)
                .any(|(batch, target)| batch != &target.batch)
        {
            return Err(invalid_operation(
                "determinism readback canonicalization lost a semantic witness mapping",
            ));
        }

        Ok(Self {
            plan_hash: witness_plan.plan_hash().clone(),
            node_ids,
            collection,
            targets,
            witness_count,
        })
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }

    pub fn collection_request(&self) -> &CompletionReadbackCollectionRequest {
        &self.collection
    }

    pub fn targets(&self) -> &[SubmissionWaveDeterminismReadbackTarget] {
        &self.targets
    }

    pub const fn witness_count(&self) -> usize {
        self.witness_count
    }
}

fn validate_terminal_witness_stability(
    resolved: &dyn ExecutablePlanView,
    witness_plan: &ExecutionDeterminismWitnessPlan,
) -> Result<(), VNextError> {
    if witness_plan.node_ids().len() > 1 {
        let retained = resolved
            .execution_plan()
            .payload()
            .retained_completion_values();
        for witness in witness_plan.witnesses() {
            let ExecutionDeterminismWitnessKind::Output {
                value_id,
                output_ordinal,
            } = witness.kind()
            else {
                continue;
            };
            let exact = retained.iter().filter(|candidate| {
                candidate.value_id() == value_id
                    && candidate.producer_node_id() == witness.node_id()
                    && candidate.output_ordinal() == *output_ordinal
                    && candidate.resource_id() == witness.resource_id()
                    && candidate.logical_offset_bytes() == witness.logical_offset_bytes()
                    && candidate.tensor().element_type() == witness.element_type()
                    && candidate.tensor().minimum_storage_bytes().ok()
                        == Some(witness.canonical_length_bytes())
            });
            if exact.count() != 1 {
                return Err(invalid_operation(format!(
                    "determinism output `{value_id}` from node `{}` is not retained as one exact terminal witness",
                    witness.node_id()
                )));
            }
        }
    }

    let node_order = witness_plan
        .node_ids()
        .iter()
        .enumerate()
        .map(|(index, node_id)| (node_id, index))
        .collect::<BTreeMap<_, _>>();
    for witness in witness_plan.witnesses() {
        if !matches!(
            witness.kind(),
            ExecutionDeterminismWitnessKind::StateEffect { .. }
        ) {
            continue;
        }
        let witness_order = node_order[witness.node_id()];
        let witness_end = witness
            .logical_offset_bytes()
            .checked_add(witness.canonical_length_bytes())
            .ok_or_else(|| invalid_operation("determinism state witness range overflows"))?;
        let overwritten = witness_plan.witnesses().iter().any(|later| {
            let Some(later_order) = node_order.get(later.node_id()) else {
                return false;
            };
            if *later_order <= witness_order || later.resource_id() != witness.resource_id() {
                return false;
            }
            let Some(later_end) = later
                .logical_offset_bytes()
                .checked_add(later.canonical_length_bytes())
            else {
                return true;
            };
            later.logical_offset_bytes() < witness_end && witness.logical_offset_bytes() < later_end
        });
        if overwritten {
            return Err(invalid_operation(format!(
                "determinism state witness from node `{}` is overwritten later in the same terminal readback scope",
                witness.node_id()
            )));
        }
    }
    Ok(())
}

/// Submitted deterministic work whose terminal observation remains bound to
/// the exact immutable-plan witness denominator.
#[must_use = "deterministic submission must collect its exact witness readback"]
pub struct SubmissionWaveDeterminismHandle<R: DeviceRuntime> {
    completion: CompletionHandle<R>,
    attribution: Option<BoundDeviceSubmissionAttribution>,
    readback_plan: SubmissionWaveDeterminismReadbackPlan,
}

impl<R: DeviceRuntime> SubmissionWaveDeterminismHandle<R> {
    pub(super) fn from_profiled(
        profiled: ProfiledSubmissionHandle<R>,
        readback_plan: SubmissionWaveDeterminismReadbackPlan,
    ) -> Self {
        let (completion, attribution) = profiled.into_parts();
        Self {
            completion,
            attribution,
            readback_plan,
        }
    }

    pub fn receipt(&self) -> &SubmittedOperationReceipt {
        self.completion.receipt()
    }

    pub fn attribution(&self) -> Option<&BoundDeviceSubmissionAttribution> {
        self.attribution.as_ref()
    }

    pub fn readback_plan(&self) -> &SubmissionWaveDeterminismReadbackPlan {
        &self.readback_plan
    }

    pub fn wait_with_determinism_readback(
        &self,
    ) -> Result<CompletionReadbackCollectionObservation, VNextError> {
        self.completion
            .wait_with_readback_collection(self.readback_plan.collection_request().clone())
    }
}
