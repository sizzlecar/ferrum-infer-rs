use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::dispatch_contract::{BoundDeviceSubmissionAttribution, ProfiledSubmissionHandle};
use super::foundation::invalid_operation;
use super::{
    BatchOperationIdentity, BatchedOperationInvocation, BoundOperationProvider, ElementType,
};
use crate::vnext::{
    BatchInvocationId, BufferUsage, CompletionHandle, CompletionReadbackBatchRequest,
    CompletionReadbackCollectionObservation, CompletionReadbackCollectionRequest,
    CompletionReadbackDisposition, CompletionReadbackRequest, DeviceCommandPhase,
    DeviceComputePathRequirement, DeviceExecutionPath, DeviceReusableExecutionProgramId,
    DeviceRuntime, ExecutablePlanView, ExecutionDeterminismInitializationKind,
    ExecutionDeterminismInitializationSpec, ExecutionDeterminismValueExtent,
    ExecutionDeterminismValueLocation, ExecutionDeterminismWitnessKind,
    ExecutionDeterminismWitnessPlan, ExecutionDeterminismWitnessSpec, HostTransferLayout, NodeId,
    OperationCompletionDisposition, PlanHash, PreparedStepSubmissionWave, ResourceId,
    SubmittedOperationReceipt, TrustedActiveSequenceBinding, VNextError,
};

const LOGICAL_RESTORE_FINGERPRINT_DOMAIN: &[u8] =
    b"ferrum.runtime-vnext.determinism-logical-restore.v1";
const EXTERNAL_INPUT_FINGERPRINT_DOMAIN: &[u8] =
    b"ferrum.runtime-vnext.determinism-external-input.v1";
const INITIAL_STATE_FINGERPRINT_DOMAIN: &[u8] =
    b"ferrum.runtime-vnext.determinism-initial-state.v1";
const NO_RNG_STATE_FINGERPRINT_DOMAIN: &[u8] = b"ferrum.runtime-vnext.determinism-no-rng-state.v1";

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), VNextError> {
    hash_u64(
        hasher,
        u64::try_from(bytes.len())
            .map_err(|_| invalid_operation("determinism restore fingerprint input exceeds u64"))?,
    );
    hasher.update(bytes);
    Ok(())
}

fn hash_ranges(
    hasher: &mut Sha256,
    ranges: &[SubmissionWaveDeterminismLogicalRange],
) -> Result<(), VNextError> {
    hash_u64(
        hasher,
        u64::try_from(ranges.len())
            .map_err(|_| invalid_operation("determinism restore range count exceeds u64"))?,
    );
    for range in ranges {
        hash_u64(hasher, range.logical_offset_bytes());
        hash_u64(hasher, range.length_bytes());
    }
    Ok(())
}

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

/// Domain-separated semantic initialization identities for one deterministic
/// execution. These values are derived from the exact typed restore payloads;
/// artifact producers cannot substitute independently computed digests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismInitializationIdentity {
    input_sha256: String,
    rng_sha256: String,
    initial_state_sha256: String,
}

impl SubmissionWaveDeterminismInitializationIdentity {
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }

    pub fn rng_sha256(&self) -> &str {
        &self.rng_sha256
    }

    pub fn initial_state_sha256(&self) -> &str {
        &self.initial_state_sha256
    }
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

    fn initialization_fingerprint(
        &self,
        domain: &[u8],
        include: impl Fn(&ExecutionDeterminismInitializationKind) -> bool,
    ) -> Result<String, VNextError> {
        let initializations = self.initializations();
        let selected_count = initializations
            .iter()
            .filter(|initialization| include(initialization.kind()))
            .count()
            .checked_mul(self.participant_payloads.len())
            .ok_or_else(|| invalid_operation("determinism initialization count overflows"))?;
        let mut hasher = Sha256::new();
        hash_bytes(&mut hasher, domain)?;
        hash_bytes(
            &mut hasher,
            self.layout.witness_plan.fingerprint()?.as_bytes(),
        )?;
        hash_u64(
            &mut hasher,
            u64::try_from(selected_count)
                .map_err(|_| invalid_operation("determinism initialization count exceeds u64"))?,
        );
        for (participant_index, (payloads, ranges)) in self
            .participant_payloads
            .iter()
            .zip(&self.layout.participant_initialization_ranges)
            .enumerate()
        {
            for (initialization_index, ((initialization, payload), range)) in
                initializations.iter().zip(payloads).zip(ranges).enumerate()
            {
                if !include(initialization.kind()) {
                    continue;
                }
                hash_u64(
                    &mut hasher,
                    u64::try_from(participant_index).map_err(|_| {
                        invalid_operation("determinism participant index exceeds u64")
                    })?,
                );
                hash_u64(
                    &mut hasher,
                    u64::try_from(initialization_index).map_err(|_| {
                        invalid_operation("determinism initialization index exceeds u64")
                    })?,
                );
                let encoded = serde_json::to_vec(initialization).map_err(|error| {
                    invalid_operation(format!(
                        "determinism initialization identity serialization failed: {error}"
                    ))
                })?;
                hash_bytes(&mut hasher, &encoded)?;
                hash_u64(&mut hasher, range.logical_offset_bytes());
                hash_u64(&mut hasher, range.length_bytes());
                hash_bytes(&mut hasher, payload)?;
            }
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Exact input/state digests restored for this wave. The current model
    /// execution graph contains no stochastic operation input, so the RNG
    /// identity is an explicit domain-separated empty state rather than a
    /// collector-supplied placeholder.
    pub fn initialization_identity(
        &self,
    ) -> Result<SubmissionWaveDeterminismInitializationIdentity, VNextError> {
        let input_sha256 =
            self.initialization_fingerprint(EXTERNAL_INPUT_FINGERPRINT_DOMAIN, |kind| {
                matches!(
                    kind,
                    ExecutionDeterminismInitializationKind::ExternalInput { .. }
                )
            })?;
        let initial_state_sha256 = self
            .initialization_fingerprint(INITIAL_STATE_FINGERPRINT_DOMAIN, |kind| {
                matches!(kind, ExecutionDeterminismInitializationKind::State { .. })
            })?;
        let mut rng = Sha256::new();
        hash_bytes(&mut rng, NO_RNG_STATE_FINGERPRINT_DOMAIN)?;
        hash_bytes(&mut rng, self.layout.witness_plan.fingerprint()?.as_bytes())?;
        hash_u64(&mut rng, 0);
        Ok(SubmissionWaveDeterminismInitializationIdentity {
            input_sha256,
            rng_sha256: format!("{:x}", rng.finalize()),
            initial_state_sha256,
        })
    }

    /// Stable identity of the logical input/state image restored before each
    /// execution.
    ///
    /// Physical batch, backing, lane, and allocation identities are excluded
    /// intentionally: repeated executions must allocate fresh submission
    /// authority while proving that they restored identical semantic bytes.
    /// Scratch poison is also excluded because it is the independent variable
    /// compared by the determinism gate.
    pub fn logical_fingerprint(&self) -> Result<String, VNextError> {
        let mut hasher = Sha256::new();
        hash_bytes(&mut hasher, LOGICAL_RESTORE_FINGERPRINT_DOMAIN)?;
        hash_bytes(
            &mut hasher,
            self.layout.witness_plan.fingerprint()?.as_bytes(),
        )?;

        hash_u64(
            &mut hasher,
            u64::try_from(self.layout.node_work_shape_fingerprints.len())
                .map_err(|_| invalid_operation("determinism restore node count exceeds u64"))?,
        );
        for fingerprint in &self.layout.node_work_shape_fingerprints {
            hash_bytes(&mut hasher, fingerprint.as_bytes())?;
        }

        hash_u64(
            &mut hasher,
            u64::try_from(self.layout.participant_initialization_ranges.len()).map_err(|_| {
                invalid_operation("determinism restore participant count exceeds u64")
            })?,
        );
        for ranges in &self.layout.participant_initialization_ranges {
            hash_ranges(&mut hasher, ranges)?;
        }

        hash_u64(
            &mut hasher,
            u64::try_from(self.layout.witness_participant_ranges.len())
                .map_err(|_| invalid_operation("determinism restore witness count exceeds u64"))?,
        );
        for ranges in &self.layout.witness_participant_ranges {
            hash_ranges(&mut hasher, ranges)?;
        }

        hash_u64(
            &mut hasher,
            u64::try_from(self.participant_payloads.len()).map_err(|_| {
                invalid_operation("determinism restore payload participant count exceeds u64")
            })?,
        );
        for payloads in &self.participant_payloads {
            hash_u64(
                &mut hasher,
                u64::try_from(payloads.len()).map_err(|_| {
                    invalid_operation("determinism restore payload count exceeds u64")
                })?,
            );
            for payload in payloads {
                hash_bytes(&mut hasher, payload)?;
            }
        }

        Ok(format!("{:x}", hasher.finalize()))
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
            immediate_token_logical_range(
                token_start,
                token_range.immediate_tokens(),
                bytes_per_token,
                maximum_tokens,
            )?
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

fn immediate_token_logical_range(
    token_start: u64,
    immediate_tokens: u64,
    bytes_per_token: u64,
    maximum_tokens: u64,
) -> Result<(u64, u64), VNextError> {
    let token_end = token_start
        .checked_add(immediate_tokens)
        .ok_or_else(|| invalid_operation("determinism immediate token range overflows"))?;
    if immediate_tokens == 0 || token_end > maximum_tokens {
        return Err(invalid_operation(
            "determinism immediate token range exceeds its scheduled resource capacity",
        ));
    }
    Ok((
        bytes_per_token
            .checked_mul(token_start)
            .ok_or_else(|| invalid_operation("determinism immediate byte offset overflows"))?,
        bytes_per_token
            .checked_mul(immediate_tokens)
            .ok_or_else(|| invalid_operation("determinism immediate byte length overflows"))?,
    ))
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

/// One physical readback buffer retained once even when multiple semantic
/// witnesses intentionally project onto the same bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismPhysicalReadback {
    request: CompletionReadbackRequest,
    raw_sha256: String,
    #[serde(skip)]
    bytes: Vec<u8>,
}

impl SubmissionWaveDeterminismPhysicalReadback {
    pub fn request(&self) -> &CompletionReadbackRequest {
        &self.request
    }

    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Semantic witness mapped to exactly one physical participant readback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismWitnessReadback {
    witness: ExecutionDeterminismWitnessSpec,
    participant_index: u32,
    physical_readback_index: u32,
}

impl SubmissionWaveDeterminismWitnessReadback {
    pub fn witness(&self) -> &ExecutionDeterminismWitnessSpec {
        &self.witness
    }

    pub const fn participant_index(&self) -> u32 {
        self.participant_index
    }

    pub const fn physical_readback_index(&self) -> u32 {
        self.physical_readback_index
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SubmissionWaveDeterminismComputeExpectation {
    EagerOnly,
    Replayed {
        program_id: DeviceReusableExecutionProgramId,
        declared_eager_boundary_node_ids: Vec<NodeId>,
    },
}

impl SubmissionWaveDeterminismComputeExpectation {
    fn replayed(
        program_id: DeviceReusableExecutionProgramId,
        declared_eager_boundary_node_ids: Vec<NodeId>,
    ) -> Result<Self, VNextError> {
        if declared_eager_boundary_node_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_operation(
                "determinism eager boundary node ids are not canonical",
            ));
        }
        Ok(Self::Replayed {
            program_id,
            declared_eager_boundary_node_ids,
        })
    }

    const fn execution_path(&self) -> DeviceExecutionPath {
        match self {
            Self::EagerOnly => DeviceExecutionPath::Eager,
            Self::Replayed { .. } => DeviceExecutionPath::Replayed,
        }
    }

    const fn compute_path_requirement(&self) -> DeviceComputePathRequirement {
        match self {
            Self::EagerOnly => DeviceComputePathRequirement::EagerOnly,
            Self::Replayed {
                declared_eager_boundary_node_ids,
                ..
            } if declared_eager_boundary_node_ids.is_empty() => {
                DeviceComputePathRequirement::ReplayedOnly
            }
            Self::Replayed { .. } => {
                DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
            }
        }
    }

    fn reusable_program_id(&self) -> Option<&DeviceReusableExecutionProgramId> {
        match self {
            Self::EagerOnly => None,
            Self::Replayed { program_id, .. } => Some(program_id),
        }
    }

    fn declared_eager_boundary_node_ids(&self) -> &[NodeId] {
        match self {
            Self::EagerOnly => &[],
            Self::Replayed {
                declared_eager_boundary_node_ids,
                ..
            } => declared_eager_boundary_node_ids,
        }
    }
}

/// Fail-closed terminal evidence for one forced eager or replay execution.
///
/// The physical buffers retain raw bytes for in-process comparison. Their
/// serialized form contains only SHA256 digests so a detached artifact cannot
/// accidentally duplicate large device outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismEvidence {
    restore_fingerprint: String,
    initialization_identity: SubmissionWaveDeterminismInitializationIdentity,
    compute_expectation: SubmissionWaveDeterminismComputeExpectation,
    submission_receipt_fingerprint: String,
    terminal_receipt_fingerprint: String,
    attribution: BoundDeviceSubmissionAttribution,
    physical_readbacks: Vec<SubmissionWaveDeterminismPhysicalReadback>,
    witnesses: Vec<SubmissionWaveDeterminismWitnessReadback>,
}

impl SubmissionWaveDeterminismEvidence {
    pub fn restore_fingerprint(&self) -> &str {
        &self.restore_fingerprint
    }

    pub fn initialization_identity(&self) -> &SubmissionWaveDeterminismInitializationIdentity {
        &self.initialization_identity
    }

    pub const fn expected_execution_path(&self) -> DeviceExecutionPath {
        self.compute_expectation.execution_path()
    }

    pub const fn expected_compute_path_requirement(&self) -> DeviceComputePathRequirement {
        self.compute_expectation.compute_path_requirement()
    }

    pub fn reusable_program_fingerprint(&self) -> Option<String> {
        self.compute_expectation
            .reusable_program_id()
            .map(DeviceReusableExecutionProgramId::fingerprint)
    }

    pub fn declared_eager_boundary_node_ids(&self) -> &[NodeId] {
        self.compute_expectation.declared_eager_boundary_node_ids()
    }

    pub fn submission_receipt_fingerprint(&self) -> &str {
        &self.submission_receipt_fingerprint
    }

    pub fn terminal_receipt_fingerprint(&self) -> &str {
        &self.terminal_receipt_fingerprint
    }

    pub fn attribution(&self) -> &BoundDeviceSubmissionAttribution {
        &self.attribution
    }

    pub fn physical_readbacks(&self) -> &[SubmissionWaveDeterminismPhysicalReadback] {
        &self.physical_readbacks
    }

    pub fn witnesses(&self) -> &[SubmissionWaveDeterminismWitnessReadback] {
        &self.witnesses
    }
}

fn validate_determinism_attribution(
    attribution: &BoundDeviceSubmissionAttribution,
    node_ids: &[NodeId],
    compute_expectation: &SubmissionWaveDeterminismComputeExpectation,
) -> Result<(), VNextError> {
    let expected_nodes = node_ids.iter().collect::<BTreeSet<_>>();
    let declared_eager_boundary_nodes = compute_expectation
        .declared_eager_boundary_node_ids()
        .iter()
        .collect::<BTreeSet<_>>();
    if !declared_eager_boundary_nodes.is_subset(&expected_nodes) {
        return Err(invalid_operation(
            "determinism eager boundary is absent from the requested plan nodes",
        ));
    }
    let mut observed_nodes = BTreeSet::new();
    match compute_expectation.execution_path() {
        DeviceExecutionPath::Eager => {
            for replayed_segment in attribution.device().replayed_segments() {
                for command in replayed_segment.logical_commands() {
                    let node_index = usize::try_from(command.node_index()).map_err(|_| {
                        invalid_operation("determinism replay node index exceeds usize")
                    })?;
                    let node_id = attribution
                        .batch_identity()
                        .node_id_at(node_index)
                        .ok_or_else(|| {
                            invalid_operation(
                                "determinism replay attribution references a node absent from its batch",
                            )
                        })?;
                    if expected_nodes.contains(node_id) {
                        return Err(invalid_operation(format!(
                            "determinism node `{node_id}` replayed while eager execution was required"
                        )));
                    }
                }
            }
            for command in attribution.device().commands() {
                if command.command_phase() != DeviceCommandPhase::Compute {
                    continue;
                }
                let Some(node_index) = command.node_index() else {
                    continue;
                };
                let node_index = usize::try_from(node_index).map_err(|_| {
                    invalid_operation("determinism attribution node index exceeds usize")
                })?;
                let node_id = attribution
                    .batch_identity()
                    .node_id_at(node_index)
                    .ok_or_else(|| {
                        invalid_operation(
                            "determinism attribution references a node absent from its batch",
                        )
                    })?;
                if !expected_nodes.contains(node_id) {
                    continue;
                }
                if command.execution_path() != DeviceExecutionPath::Eager
                    || command.reusable_graph_node_count().is_some()
                {
                    return Err(invalid_operation(format!(
                        "determinism node `{node_id}` did not execute through the required eager path"
                    )));
                }
                if !observed_nodes.insert(node_id) {
                    return Err(invalid_operation(format!(
                        "determinism node `{node_id}` has duplicate eager compute attribution"
                    )));
                }
            }
        }
        DeviceExecutionPath::Replayed => {
            let expected_program_id =
                compute_expectation.reusable_program_id().ok_or_else(|| {
                    invalid_operation("determinism replay expectation lacks a reusable program")
                })?;
            if attribution.device().replayed_segments().is_empty() {
                return Err(invalid_operation(
                    "determinism replay attribution contains no resident segment",
                ));
            }
            for command in attribution.device().commands() {
                if command.command_phase() != DeviceCommandPhase::Compute {
                    continue;
                }
                let Some(node_index) = command.node_index() else {
                    continue;
                };
                let node_index = usize::try_from(node_index).map_err(|_| {
                    invalid_operation("determinism attribution node index exceeds usize")
                })?;
                let node_id = attribution
                    .batch_identity()
                    .node_id_at(node_index)
                    .ok_or_else(|| {
                        invalid_operation(
                            "determinism attribution references a node absent from its batch",
                        )
                    })?;
                if !expected_nodes.contains(node_id) {
                    continue;
                }
                if declared_eager_boundary_nodes.contains(node_id) {
                    if command.execution_path() != DeviceExecutionPath::Eager
                        || command.reusable_graph_node_count().is_some()
                    {
                        return Err(invalid_operation(format!(
                            "determinism declared eager boundary `{node_id}` did not execute eagerly"
                        )));
                    }
                    if !observed_nodes.insert(node_id) {
                        return Err(invalid_operation(format!(
                            "determinism declared eager boundary `{node_id}` has duplicate compute attribution"
                        )));
                    }
                } else if command.execution_path() != DeviceExecutionPath::Replayed {
                    return Err(invalid_operation(format!(
                        "determinism node `{node_id}` executed eagerly without a declared topology boundary"
                    )));
                }
            }
            for replayed_segment in attribution.device().replayed_segments() {
                if replayed_segment.program_id() != expected_program_id {
                    return Err(invalid_operation(
                        "determinism replay attribution references another reusable program",
                    ));
                }
                for command in replayed_segment.logical_commands() {
                    let node_index = usize::try_from(command.node_index()).map_err(|_| {
                        invalid_operation("determinism replay node index exceeds usize")
                    })?;
                    let node_id = attribution
                        .batch_identity()
                        .node_id_at(node_index)
                        .ok_or_else(|| {
                            invalid_operation(
                                "determinism replay attribution references a node absent from its batch",
                            )
                        })?;
                    if declared_eager_boundary_nodes.contains(node_id) {
                        return Err(invalid_operation(format!(
                            "determinism declared eager boundary `{node_id}` appeared in replay attribution"
                        )));
                    }
                    if expected_nodes.contains(node_id) && !observed_nodes.insert(node_id) {
                        return Err(invalid_operation(format!(
                            "determinism node `{node_id}` has duplicate replay attribution"
                        )));
                    }
                }
            }
        }
    }
    if observed_nodes != expected_nodes {
        return Err(invalid_operation(
            "determinism attribution does not cover every requested plan node",
        ));
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
    restore_fingerprint: String,
    initialization_identity: SubmissionWaveDeterminismInitializationIdentity,
    compute_expectation: SubmissionWaveDeterminismComputeExpectation,
}

impl<R: DeviceRuntime> SubmissionWaveDeterminismHandle<R> {
    pub(super) fn from_profiled_eager(
        profiled: ProfiledSubmissionHandle<R>,
        readback_plan: SubmissionWaveDeterminismReadbackPlan,
        restore_fingerprint: String,
        initialization_identity: SubmissionWaveDeterminismInitializationIdentity,
    ) -> Self {
        Self::from_profiled(
            profiled,
            readback_plan,
            restore_fingerprint,
            initialization_identity,
            SubmissionWaveDeterminismComputeExpectation::EagerOnly,
        )
    }

    pub(super) fn from_profiled_replayed(
        profiled: ProfiledSubmissionHandle<R>,
        readback_plan: SubmissionWaveDeterminismReadbackPlan,
        restore_fingerprint: String,
        initialization_identity: SubmissionWaveDeterminismInitializationIdentity,
        program_id: DeviceReusableExecutionProgramId,
        declared_eager_boundary_node_ids: Vec<NodeId>,
    ) -> Result<Self, VNextError> {
        let expectation = SubmissionWaveDeterminismComputeExpectation::replayed(
            program_id,
            declared_eager_boundary_node_ids,
        )?;
        Ok(Self::from_profiled(
            profiled,
            readback_plan,
            restore_fingerprint,
            initialization_identity,
            expectation,
        ))
    }

    fn from_profiled(
        profiled: ProfiledSubmissionHandle<R>,
        readback_plan: SubmissionWaveDeterminismReadbackPlan,
        restore_fingerprint: String,
        initialization_identity: SubmissionWaveDeterminismInitializationIdentity,
        compute_expectation: SubmissionWaveDeterminismComputeExpectation,
    ) -> Self {
        let (completion, attribution) = profiled.into_parts();
        Self {
            completion,
            attribution,
            readback_plan,
            restore_fingerprint,
            initialization_identity,
            compute_expectation,
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

    pub fn restore_fingerprint(&self) -> &str {
        &self.restore_fingerprint
    }

    pub fn wait_with_determinism_readback(
        &self,
    ) -> Result<CompletionReadbackCollectionObservation, VNextError> {
        self.completion
            .wait_with_readback_collection(self.readback_plan.collection_request().clone())
    }

    /// Waits for terminal completion and consumes the handle into one typed
    /// evidence object. Missing actual-path attribution, nonterminal
    /// completion, failed readback, or any semantic/physical mapping drift is
    /// rejected after the submitted work has been observed to completion.
    pub fn wait_into_evidence(self) -> Result<SubmissionWaveDeterminismEvidence, VNextError> {
        let Self {
            completion,
            attribution,
            readback_plan,
            restore_fingerprint,
            initialization_identity,
            compute_expectation,
        } = self;
        let submission_receipt_fingerprint = completion.receipt().fingerprint().to_owned();
        let observation =
            completion.wait_with_readback_collection(readback_plan.collection_request().clone())?;
        let receipt = match observation {
            CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
            other => {
                return Err(invalid_operation(format!(
                    "determinism readback did not reach a terminal observation: {other:?}"
                )))
            }
        };
        if !matches!(
            receipt.completion().disposition(),
            OperationCompletionDisposition::Succeeded
        ) {
            return Err(invalid_operation(
                "determinism submission completed without a successful disposition",
            ));
        }

        let attribution = attribution
            .ok_or_else(|| {
                invalid_operation("determinism submission lacks actual-path device attribution")
            })?
            .bind_terminal_timing(receipt.completion().submission_timing().clone())?;
        validate_determinism_attribution(
            &attribution,
            readback_plan.node_ids(),
            &compute_expectation,
        )?;

        let expected_readbacks = readback_plan
            .targets()
            .iter()
            .map(|target| target.batch().requests().len())
            .sum::<usize>();
        if receipt.dispositions().len() != expected_readbacks {
            return Err(invalid_operation(
                "determinism terminal readback count differs from its typed plan",
            ));
        }

        let mut physical_readbacks = Vec::with_capacity(expected_readbacks);
        let mut witnesses = Vec::with_capacity(
            readback_plan.witness_count().saturating_mul(
                readback_plan
                    .targets()
                    .first()
                    .map_or(0, |target| target.batch().requests().len()),
            ),
        );
        let mut disposition_index = 0_usize;
        for target in readback_plan.targets() {
            let first_physical_index = physical_readbacks.len();
            for expected_request in target.batch().requests() {
                let disposition =
                    receipt
                        .dispositions()
                        .get(disposition_index)
                        .ok_or_else(|| {
                            invalid_operation("determinism terminal readback disappeared")
                        })?;
                let CompletionReadbackDisposition::Succeeded(output) = disposition else {
                    return Err(invalid_operation(format!(
                        "determinism terminal readback failed: {disposition:?}"
                    )));
                };
                if output.request() != expected_request {
                    return Err(invalid_operation(
                        "determinism terminal readback differs from its exact request",
                    ));
                }
                physical_readbacks.push(SubmissionWaveDeterminismPhysicalReadback {
                    request: output.request().clone(),
                    raw_sha256: output.sha256().to_owned(),
                    bytes: output.bytes().to_vec(),
                });
                disposition_index += 1;
            }
            for witness in target.witnesses() {
                for participant_index in 0..target.batch().requests().len() {
                    let physical_readback_index = first_physical_index
                        .checked_add(participant_index)
                        .and_then(|index| u32::try_from(index).ok())
                        .ok_or_else(|| {
                            invalid_operation("determinism physical readback index exceeds u32")
                        })?;
                    witnesses.push(SubmissionWaveDeterminismWitnessReadback {
                        witness: witness.clone(),
                        participant_index: u32::try_from(participant_index).map_err(|_| {
                            invalid_operation("determinism witness participant index exceeds u32")
                        })?,
                        physical_readback_index,
                    });
                }
            }
        }
        if disposition_index != receipt.dispositions().len() {
            return Err(invalid_operation(
                "determinism terminal readback left unowned physical outputs",
            ));
        }
        let terminal_receipt_fingerprint = receipt.fingerprint().to_owned();
        Ok(SubmissionWaveDeterminismEvidence {
            restore_fingerprint,
            initialization_identity,
            compute_expectation,
            submission_receipt_fingerprint,
            terminal_receipt_fingerprint,
            attribution,
            physical_readbacks,
            witnesses,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::immediate_token_logical_range;

    #[test]
    fn immediate_token_range_uses_scheduled_capacity_not_canonical_extent() {
        assert_eq!(
            immediate_token_logical_range(1, 3, 16, 4).unwrap(),
            (16, 48)
        );
        assert!(immediate_token_logical_range(2, 3, 16, 4).is_err());
        assert!(immediate_token_logical_range(u64::MAX, 1, 16, u64::MAX).is_err());
    }
}
