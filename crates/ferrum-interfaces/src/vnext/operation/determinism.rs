use std::collections::BTreeMap;

use super::dispatch::{BoundDeviceSubmissionAttribution, ProfiledSubmissionHandle};
use super::{
    invalid_operation, BatchOperationIdentity, BatchedOperationInvocation, BoundOperationProvider,
    ElementType,
};
use crate::vnext::{
    BatchInvocationId, BufferUsage, CompletionHandle, CompletionReadbackBatchRequest,
    CompletionReadbackCollectionObservation, CompletionReadbackCollectionRequest,
    CompletionReadbackRequest, DeviceRuntime, ExecutablePlanView,
    ExecutionDeterminismInitializationSpec, ExecutionDeterminismValueExtent,
    ExecutionDeterminismValueLocation, ExecutionDeterminismWitnessKind,
    ExecutionDeterminismWitnessPlan, ExecutionDeterminismWitnessSpec, HostTransferLayout, NodeId,
    PlanHash, PreparedStepSubmissionWave, ResourceId, SubmittedOperationReceipt,
    TrustedActiveSequenceBinding, VNextError,
};

/// One exact logical range as seen by a provider for one prepared participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubmissionWaveDeterminismLogicalRange {
    logical_offset_bytes: u64,
    length_bytes: u64,
}

impl SubmissionWaveDeterminismLogicalRange {
    pub const fn logical_offset_bytes(self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn length_bytes(self) -> u64 {
        self.length_bytes
    }
}

/// Provider-visible deterministic I/O layout for one exact prepared wave.
///
/// The immutable plan defines which semantic values are required. This layout
/// closes the remaining runtime boundary: packed versus participant-local
/// token offsets and the exact page-quantized state prefix visible to each
/// provider invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionWaveDeterminismRestoreLayout {
    witness_plan: ExecutionDeterminismWitnessPlan,
    batch_invocation_id: BatchInvocationId,
    claimed_backing_fingerprint: String,
    node_work_shape_fingerprints: Vec<String>,
    participant_initialization_ranges: Vec<Vec<SubmissionWaveDeterminismLogicalRange>>,
    witness_participant_ranges: Vec<Vec<SubmissionWaveDeterminismLogicalRange>>,
}

impl SubmissionWaveDeterminismRestoreLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared_wave<'binding, R, I>(
        runtime: &R,
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: I,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime,
        I: Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let node_ids = wave
            .nodes()
            .iter()
            .map(|node| node.node_id().clone())
            .collect::<Vec<_>>();
        let witness_plan = resolved
            .execution_plan()
            .determinism_witness_plan_for_nodes(&node_ids)?;
        let participant_count = wave
            .nodes()
            .first()
            .map(|node| node.participant_count() as usize)
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                invalid_operation("determinism restore layout requires a non-empty prepared wave")
            })?;
        if providers.len() != wave.nodes().len()
            || providers.len() != batch_identity.node_count()
            || active_bindings.len() != participant_count
            || witness_plan.plan_hash() != resolved.execution_plan().plan_hash()
            || witness_plan.plan_hash() != batch_identity.plan_hash()
            || batch_identity.batch_invocation_id() != wave.batch_invocation_id()
            || batch_identity.claimed_backing_fingerprint() != wave.fingerprint()
            || wave.nodes().iter().enumerate().any(|(node_index, node)| {
                node.plan_evidence_ref().plan_hash() != witness_plan.plan_hash()
                    || node.participant_count() as usize != participant_count
                    || node.work_shape().participant_work().len() != participant_count
                    || node.work_shape().participant_token_ranges().len() != participant_count
                    || batch_identity.node_id_at(node_index) != Some(node.node_id())
                    || batch_identity.node_participant_count(node_index) != Some(participant_count)
            })
        {
            return Err(invalid_operation(
                "determinism restore layout differs from its immutable plan or participant topology",
            ));
        }

        let invocations = providers
            .iter()
            .zip(batch_identity.nodes())
            .enumerate()
            .map(|(node_index, (provider, node_identity))| {
                provider.validate_binding(resolved, node_identity.node_id())?;
                BatchedOperationInvocation::from_wave_node(
                    runtime,
                    resolved,
                    provider.dispatch(),
                    batch_identity,
                    node_identity,
                    wave,
                    node_index,
                    active_bindings.clone(),
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?;

        let mut participant_initialization_ranges =
            vec![Vec::with_capacity(witness_plan.initializations().len()); participant_count];
        for initialization in witness_plan.initializations() {
            for (participant_index, ranges) in
                participant_initialization_ranges.iter_mut().enumerate()
            {
                let mut bound_range = None;
                for consumer_node_id in initialization.consumer_node_ids() {
                    let invocation = invocations
                        .iter()
                        .find(|invocation| invocation.node_id() == consumer_node_id)
                        .ok_or_else(|| {
                            invalid_operation(
                                "determinism initialization consumer is absent from its prepared wave",
                            )
                        })?;
                    let candidate = prepared_location_range(
                        wave,
                        invocation,
                        participant_index,
                        initialization.location(),
                    )?;
                    if bound_range
                        .replace(candidate)
                        .is_some_and(|bound| bound != candidate)
                    {
                        return Err(invalid_operation(
                            "determinism initialization consumers disagree on the provider-visible range",
                        ));
                    }
                }
                ranges.push(bound_range.ok_or_else(|| {
                    invalid_operation(
                        "determinism initialization has no prepared-wave consumer range",
                    )
                })?);
            }
        }

        let witness_participant_ranges = witness_plan
            .witnesses()
            .iter()
            .map(|witness| {
                let invocation = invocations
                    .iter()
                    .find(|invocation| invocation.node_id() == witness.node_id())
                    .ok_or_else(|| {
                        invalid_operation(
                            "determinism witness node is absent from its prepared wave",
                        )
                    })?;
                (0..participant_count)
                    .map(|participant_index| {
                        prepared_location_range(
                            wave,
                            invocation,
                            participant_index,
                            witness.location(),
                        )
                    })
                    .collect::<Result<Vec<_>, VNextError>>()
            })
            .collect::<Result<Vec<_>, VNextError>>()?;

        Ok(Self {
            witness_plan,
            batch_invocation_id: batch_identity.batch_invocation_id(),
            claimed_backing_fingerprint: batch_identity.claimed_backing_fingerprint().to_owned(),
            node_work_shape_fingerprints: wave
                .nodes()
                .iter()
                .map(|node| node.work_shape().fingerprint().to_owned())
                .collect(),
            participant_initialization_ranges,
            witness_participant_ranges,
        })
    }

    pub fn witness_plan(&self) -> &ExecutionDeterminismWitnessPlan {
        &self.witness_plan
    }

    pub fn participant_count(&self) -> u32 {
        u32::try_from(self.participant_initialization_ranges.len())
            .expect("determinism restore layout participant count was validated")
    }

    pub fn participant_initialization_ranges(
        &self,
        participant_index: u32,
    ) -> Option<&[SubmissionWaveDeterminismLogicalRange]> {
        usize::try_from(participant_index)
            .ok()
            .and_then(|index| self.participant_initialization_ranges.get(index))
            .map(Vec::as_slice)
    }

    pub fn witness_participant_ranges(
        &self,
        witness_index: usize,
    ) -> Option<&[SubmissionWaveDeterminismLogicalRange]> {
        self.witness_participant_ranges
            .get(witness_index)
            .map(Vec::as_slice)
    }

    pub fn bind(
        self,
        participant_payloads: Vec<Vec<Vec<u8>>>,
    ) -> Result<SubmissionWaveDeterminismRestore, VNextError> {
        if participant_payloads.len() != self.participant_initialization_ranges.len()
            || participant_payloads
                .iter()
                .zip(&self.participant_initialization_ranges)
                .any(|(payloads, ranges)| payloads.len() != ranges.len())
        {
            return Err(invalid_operation(
                "determinism restore must cover every work-bound initialization and participant",
            ));
        }
        for ((payloads, ranges), initializations) in participant_payloads
            .iter()
            .zip(&self.participant_initialization_ranges)
            .zip(std::iter::repeat(self.witness_plan.initializations()))
        {
            for ((initialization, bytes), range) in initializations.iter().zip(payloads).zip(ranges)
            {
                let location = initialization.location();
                if u64::try_from(bytes.len()).ok() != Some(range.length_bytes())
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
        Ok(SubmissionWaveDeterminismRestore {
            layout: self,
            participant_payloads,
        })
    }
}

/// Complete participant-major input/state restoration bound to one immutable
/// plan and one exact prepared-wave work topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionWaveDeterminismRestore {
    layout: SubmissionWaveDeterminismRestoreLayout,
    participant_payloads: Vec<Vec<Vec<u8>>>,
}

impl SubmissionWaveDeterminismRestore {
    pub fn layout(&self) -> &SubmissionWaveDeterminismRestoreLayout {
        &self.layout
    }

    pub fn plan_hash(&self) -> &PlanHash {
        self.layout.witness_plan.plan_hash()
    }

    pub fn node_ids(&self) -> &[NodeId] {
        self.layout.witness_plan.node_ids()
    }

    pub fn initializations(&self) -> &[ExecutionDeterminismInitializationSpec] {
        self.layout.witness_plan.initializations()
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
            .determinism_witness_plan_for_nodes(self.node_ids())?;
        if self.plan_hash() != resolved.execution_plan().plan_hash()
            || self.layout.witness_plan != actual
        {
            return Err(invalid_operation(
                "determinism restore differs from the exact immutable plan initialization denominator",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_for_submission<'binding, R: DeviceRuntime>(
        &self,
        runtime: &R,
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        active_bindings: impl Clone + ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<(), VNextError> {
        self.validate_for(resolved)?;
        let actual_layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
            runtime,
            providers,
            resolved,
            batch_identity,
            active_bindings,
            wave,
        )?;
        if self.layout != actual_layout
            || self.node_ids().len() != wave.nodes().len()
            || self.node_ids().len() != batch_identity.node_count()
            || self.node_ids().iter().zip(wave.nodes()).enumerate().any(
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

    pub(super) fn initialization_range(
        &self,
        participant_index: u32,
        initialization_index: usize,
    ) -> Option<SubmissionWaveDeterminismLogicalRange> {
        self.layout
            .participant_initialization_ranges(participant_index)
            .and_then(|ranges| ranges.get(initialization_index))
            .copied()
    }
}

fn prepared_location_range<R: DeviceRuntime>(
    wave: &PreparedStepSubmissionWave<R>,
    invocation: &BatchedOperationInvocation<'_, R::Buffer>,
    participant_index: usize,
    location: &ExecutionDeterminismValueLocation,
) -> Result<SubmissionWaveDeterminismLogicalRange, VNextError> {
    let participant = invocation
        .participants()
        .get(participant_index)
        .ok_or_else(|| {
            invalid_operation("determinism location participant is absent from its invocation")
        })?;
    let component_index = usize::try_from(location.storage_component_ordinal())
        .map_err(|_| invalid_operation("determinism component ordinal exceeds usize"))?;
    let mut bindings = participant.bindings().iter().filter(|binding| {
        let Some(component) = binding.storage().components().get(component_index) else {
            return false;
        };
        binding.value_id() == location.value_id()
            && binding.usage() == location.usage()
            && component.resource_id() == location.resource_id()
            && component.component_id() == location.storage_component_id()
            && component.offset_bytes() == location.logical_offset_bytes()
            && component.element_type() == location.element_type()
    });
    let binding = bindings.next().ok_or_else(|| {
        invalid_operation(
            "determinism semantic value has no exact provider-visible binding component",
        )
    })?;
    if bindings.next().is_some() {
        return Err(invalid_operation(
            "determinism semantic value maps to multiple provider-visible bindings",
        ));
    }
    let component = &binding.storage().components()[component_index];
    let declared_length = if binding.storage().components().len() == 1 {
        binding.tensor().minimum_storage_bytes()?
    } else {
        component.length_bytes()
    };
    if declared_length != location.declared_length_bytes() {
        return Err(invalid_operation(
            "determinism provider binding differs from its plan-declared byte range",
        ));
    }
    let mut views = participant
        .views()
        .iter()
        .filter(|view| view.resource_id() == location.resource_id());
    let view = views.next().ok_or_else(|| {
        invalid_operation("determinism provider binding has no exact operation buffer view")
    })?;
    if views.next().is_some()
        || view.descriptor().usage != location.usage()
        || view.descriptor().element_type != location.element_type()
    {
        return Err(invalid_operation(
            "determinism operation buffer view is ambiguous or differs from its typed location",
        ));
    }

    let token_range = invocation
        .participant_token_ranges()
        .get(participant_index)
        .ok_or_else(|| {
            invalid_operation("determinism invocation participant token range is missing")
        })?;
    let participant_work = invocation
        .work_shape()
        .participant_work()
        .get(participant_index)
        .ok_or_else(|| invalid_operation("determinism participant work is missing"))?;
    if token_range.immediate_tokens() != participant_work.token_span().immediate_tokens()
        || token_range.source_token_range() != participant_work.token_span().immediate_token_range()
    {
        return Err(invalid_operation(
            "determinism provider token range differs from its prepared work",
        ));
    }
    let resource_is_shared = wave
        .claimed_backing()
        .backing_slices()
        .iter()
        .chain(wave.step_resources().backing_slices())
        .any(|authority| authority.resource_id() == location.resource_id());

    let (logical_offset_bytes, length_bytes) = match location.extent() {
        ExecutionDeterminismValueExtent::Fixed => (
            location.logical_offset_bytes(),
            location.declared_length_bytes(),
        ),
        ExecutionDeterminismValueExtent::ImmediateTokenSpan {
            bytes_per_token,
            maximum_tokens,
        } => {
            let projection = participant
                .work()
                .token_projection(binding.role(), binding.ordinal())
                .ok_or_else(|| {
                    invalid_operation(
                        "determinism immediate value lacks its provider token projection",
                    )
                })?;
            if binding.usage() != BufferUsage::Activations
                || component.offset_bytes() != 0
                || projection.canonical_extent() > maximum_tokens
                || component.length_bytes()
                    != bytes_per_token
                        .checked_mul(projection.canonical_extent())
                        .ok_or_else(|| {
                            invalid_operation("determinism immediate canonical extent overflows")
                        })?
            {
                return Err(invalid_operation(
                    "determinism immediate value differs from its provider token projection",
                ));
            }
            let token_start = if resource_is_shared {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            (
                bytes_per_token.checked_mul(token_start).ok_or_else(|| {
                    invalid_operation("determinism immediate byte offset overflows")
                })?,
                bytes_per_token
                    .checked_mul(token_range.immediate_tokens())
                    .ok_or_else(|| {
                        invalid_operation("determinism immediate byte length overflows")
                    })?,
            )
        }
        ExecutionDeterminismValueExtent::ActiveTokenPrefix {
            bytes_per_token,
            maximum_tokens,
            maximum_storage_bytes,
        } => {
            let source_end = token_range.source_token_range().end;
            let minimum_logical_bytes = bytes_per_token
                .checked_mul(source_end)
                .ok_or_else(|| invalid_operation("determinism state logical prefix overflows"))?;
            if resource_is_shared
                || binding.usage() != BufferUsage::State
                || component.offset_bytes() != 0
                || participant
                    .work()
                    .token_projection(binding.role(), binding.ordinal())
                    .is_some()
                || source_end > maximum_tokens
                || view.descriptor().size_bytes < minimum_logical_bytes
                || view.descriptor().size_bytes > maximum_storage_bytes
            {
                return Err(invalid_operation(
                    "determinism state prefix differs from its participant-local provider view",
                ));
            }
            (0, view.descriptor().size_bytes)
        }
    };
    let element_bytes = location.element_type().size_bytes();
    if logical_offset_bytes % element_bytes != 0 || length_bytes % element_bytes != 0 {
        return Err(invalid_operation(
            "determinism provider-visible range is not element aligned",
        ));
    }
    let translated = view.translate(logical_offset_bytes, length_bytes)?;
    let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
        total
            .checked_add(region.length_bytes())
            .ok_or_else(|| invalid_operation("determinism translated range overflows"))
    })?;
    if translated_bytes != length_bytes {
        return Err(invalid_operation(
            "determinism provider-visible range is not fully backed",
        ));
    }
    Ok(SubmissionWaveDeterminismLogicalRange {
        logical_offset_bytes,
        length_bytes,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PhysicalReadbackKey {
    node_id: NodeId,
    resource_id: ResourceId,
    expected_usage: BufferUsage,
    participant_layouts: Vec<(u64, ElementType, u64)>,
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
    pub fn from_restore<R: DeviceRuntime>(
        resolved: &dyn ExecutablePlanView,
        batch_identity: &BatchOperationIdentity,
        wave: &PreparedStepSubmissionWave<R>,
        restore: &SubmissionWaveDeterminismRestore,
    ) -> Result<Self, VNextError> {
        let node_ids = wave
            .nodes()
            .iter()
            .map(|node| node.node_id().clone())
            .collect::<Vec<_>>();
        restore.validate_for(resolved)?;
        let witness_plan = restore.layout().witness_plan();
        if witness_plan.plan_hash() != batch_identity.plan_hash()
            || restore.layout.batch_invocation_id != batch_identity.batch_invocation_id()
            || restore.layout.claimed_backing_fingerprint
                != batch_identity.claimed_backing_fingerprint()
            || restore.layout.node_work_shape_fingerprints
                != wave
                    .nodes()
                    .iter()
                    .map(|node| node.work_shape().fingerprint().to_owned())
                    .collect::<Vec<_>>()
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

        for (witness_index, witness) in witness_plan.witnesses().iter().enumerate() {
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

            let ranges = restore
                .layout()
                .witness_participant_ranges(witness_index)
                .ok_or_else(|| {
                    invalid_operation("determinism witness lacks its prepared participant ranges")
                })?;
            let element_bytes = witness.element_type().size_bytes();
            let requests = ranges
                .iter()
                .enumerate()
                .map(|(participant_index, range)| {
                    if range.length_bytes() % element_bytes != 0 {
                        return Err(invalid_operation(
                            "determinism witness provider-visible range is not element aligned",
                        ));
                    }
                    CompletionReadbackRequest::new_typed(
                        witness.node_id().clone(),
                        u32::try_from(participant_index).map_err(|_| {
                            invalid_operation("determinism readback participant index exceeds u32")
                        })?,
                        witness.resource_id().clone(),
                        witness.location().usage(),
                        range.logical_offset_bytes(),
                        HostTransferLayout::new(
                            witness.element_type(),
                            range.length_bytes() / element_bytes,
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
                participant_layouts: batch
                    .requests()
                    .iter()
                    .map(|request| {
                        (
                            request.logical_offset_bytes(),
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
                        == Some(witness.declared_length_bytes())
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
            .checked_add(witness.maximum_bound_length_bytes()?)
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
                .checked_add(later.maximum_bound_length_bytes().unwrap_or(u64::MAX))
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
