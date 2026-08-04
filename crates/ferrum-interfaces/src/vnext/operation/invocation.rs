use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::super::{
    AdmittedSequenceResources, AllocationKind, AllocationLifetime, BatchInvocationId,
    BatchParticipantAuthority, BatchParticipantTokenRange, BatchStepId, BatchWorkShape,
    BufferDescriptor, BufferUsage, DeviceId, DeviceRuntime, DynamicResourceDemand,
    EncodedDeviceOperation, ExecutablePlanView, ExecutionIdentityEnvelope, InvocationResourceLease,
    LogicalAdmissionCoordinatorId, LogicalBackingBufferView, NodeId, NodeWorkContract, PlanHash,
    PlanId, PlanNode, PreparedStepSubmissionNode, PreparedStepSubmissionWave,
    ProgramBindingNodeBinding, ProviderId, ProviderWorkspaceRequirement, ResourceId, SemanticValue,
    SequenceBackingSnapshot, SequenceSessionEpoch, SequenceSessionFingerprint,
    StepParticipantFrameAssignment, StepResourceLease, TrustedActiveSequenceBinding,
    TrustedPlanRuntimeEvidence, VNextError,
};
use super::buffer_view::{
    sequence_execution_shape, validate_value_binding_physical_coverage,
    ValueBindingPhysicalCoverage,
};
use super::foundation::invalid_operation;
use super::resolved_value::resource_uses_packed_batch_coordinates;
use super::{
    AttributeId, BatchOperationIdentity, BatchOperationNodeIdentity, ElementType,
    OperationBufferView, OperationDescriptor, OperationProviderDescriptor, ResolvedValueBinding,
    ResolvedValueRole,
};

pub(super) enum OperationInvocationResources<'a, R: DeviceRuntime> {
    Invocation(&'a InvocationResourceLease<R>),
    Wave {
        wave: &'a PreparedStepSubmissionWave<R>,
        node_index: usize,
    },
}

impl<R: DeviceRuntime> Copy for OperationInvocationResources<'_, R> {}

impl<R: DeviceRuntime> Clone for OperationInvocationResources<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: DeviceRuntime> OperationInvocationResources<'a, R> {
    fn wave_node(self) -> Result<&'a PreparedStepSubmissionNode<R>, VNextError> {
        match self {
            Self::Wave { wave, node_index } => wave
                .nodes()
                .get(node_index)
                .ok_or_else(|| invalid_operation("submission wave node index is out of bounds")),
            Self::Invocation(_) => Err(invalid_operation(
                "single-operation resources do not contain a wave node",
            )),
        }
    }

    pub(super) fn node_id(self) -> Result<&'a NodeId, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.node_id()),
            Self::Wave { .. } => Ok(self.wave_node()?.node_id()),
        }
    }

    fn program_binding_node(self) -> Option<ProgramBindingNodeBinding> {
        match self {
            Self::Invocation(_) => None,
            Self::Wave { wave, node_index } => wave.nodes().get(node_index).and_then(|node| {
                wave.claimed_backing()
                    .program_binding_node(node.plan_node_index())
            }),
        }
    }

    pub(super) fn participant_count(self) -> Result<usize, VNextError> {
        match self {
            Self::Invocation(invocation) => usize::try_from(invocation.participant_count())
                .map_err(|_| invalid_operation("operation participant count exceeds usize")),
            Self::Wave { .. } => usize::try_from(self.wave_node()?.participant_count())
                .map_err(|_| invalid_operation("wave participant count exceeds usize")),
        }
    }

    pub(super) fn prepared_participant_count(self) -> Result<usize, VNextError> {
        match self {
            Self::Invocation(invocation) => {
                usize::try_from(invocation.prepared_participant_count()).map_err(|_| {
                    invalid_operation("prepared operation participant count exceeds usize")
                })
            }
            Self::Wave { .. } => Ok(self.wave_node()?.participant_session_identities().len()),
        }
    }

    pub(super) fn participant(
        self,
        index: usize,
    ) -> Result<&'a Arc<AdmittedSequenceResources<R>>, VNextError> {
        match self {
            Self::Invocation(invocation) => invocation
                .participants()
                .nth(index)
                .ok_or_else(|| invalid_operation("operation participant index is out of range")),
            Self::Wave { .. } => self
                .wave_node()?
                .participants()
                .nth(index)
                .ok_or_else(|| invalid_operation("wave participant index is out of range")),
        }
    }

    fn participant_backing_snapshot(
        self,
        index: usize,
    ) -> Result<&'a Arc<SequenceBackingSnapshot<R>>, VNextError> {
        let participant = self.participant(index)?;
        self.step_resources()
            .participant_backing_snapshot(BatchParticipantAuthority::new(
                participant.sequence_authority(),
                participant.request_authority(),
            ))
    }

    fn participant_backing_view(
        self,
        index: usize,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'a, R::Buffer>, VNextError> {
        let participant = self.participant(index)?;
        self.step_resources().participant_backing_view(
            BatchParticipantAuthority::new(
                participant.sequence_authority(),
                participant.request_authority(),
            ),
            resource_id,
        )
    }

    pub(super) fn participant_frames(
        self,
    ) -> Result<&'a [StepParticipantFrameAssignment], VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.participant_frames()),
            Self::Wave { .. } => Ok(self.wave_node()?.participant_frames()),
        }
    }

    pub(super) fn participant_session_identity(
        self,
        index: usize,
    ) -> Result<(SequenceSessionEpoch, &'a SequenceSessionFingerprint), VNextError> {
        match self {
            Self::Invocation(invocation) => invocation
                .participant_session_identities()
                .nth(index)
                .ok_or_else(|| invalid_operation("operation participant session is missing")),
            Self::Wave { .. } => self
                .wave_node()?
                .participant_session_identities()
                .nth(index)
                .ok_or_else(|| invalid_operation("wave participant session is missing")),
        }
    }

    pub(super) fn batch_step_id(self) -> BatchStepId {
        match self {
            Self::Invocation(invocation) => invocation.batch_step_id(),
            Self::Wave { wave, .. } => wave.batch_step_id(),
        }
    }

    pub(super) fn batch_invocation_id(self) -> BatchInvocationId {
        match self {
            Self::Invocation(invocation) => invocation.batch_invocation_id(),
            Self::Wave { wave, .. } => wave.batch_invocation_id(),
        }
    }

    pub(super) fn coordinator_id(self) -> Result<LogicalAdmissionCoordinatorId, VNextError> {
        Ok(self.participant(0)?.coordinator_id())
    }

    pub(super) fn work_shape(self) -> Result<&'a BatchWorkShape, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.work_shape()),
            Self::Wave { .. } => Ok(self.wave_node()?.work_shape()),
        }
    }

    pub(super) fn step_resources(self) -> &'a Arc<StepResourceLease<R>> {
        match self {
            Self::Invocation(invocation) => invocation.step_resources(),
            Self::Wave { wave, .. } => wave.step_resources(),
        }
    }

    pub(super) fn runtime(self) -> &'a Arc<R> {
        match self {
            Self::Invocation(invocation) => invocation.runtime(),
            Self::Wave { wave, .. } => wave.runtime(),
        }
    }

    pub(super) fn plan_identity_matches(
        self,
        plan_id: &PlanId,
        plan_hash: &PlanHash,
        device_id: &DeviceId,
    ) -> Result<bool, VNextError> {
        match self {
            Self::Invocation(invocation) => {
                let evidence = invocation.plan_evidence();
                Ok(evidence.plan_id() == plan_id
                    && evidence.plan_hash() == plan_hash
                    && evidence.device_id() == device_id)
            }
            Self::Wave { .. } => {
                let evidence = self.wave_node()?.plan_evidence_ref();
                Ok(evidence.plan_id() == plan_id
                    && evidence.plan_hash() == plan_hash
                    && evidence.device_id() == device_id)
            }
        }
    }

    fn plan_evidence_matches(
        self,
        expected: &TrustedPlanRuntimeEvidence,
    ) -> Result<bool, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.plan_evidence() == *expected),
            Self::Wave { .. } => Ok(self.wave_node()?.plan_evidence_ref() == expected),
        }
    }

    pub(super) fn backing_fingerprint(self) -> &'a str {
        match self {
            Self::Invocation(invocation) => invocation.claimed_backing().fingerprint(),
            Self::Wave { wave, .. } => wave.fingerprint(),
        }
    }

    fn backing_view(
        self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'a, R::Buffer>, VNextError> {
        match self {
            Self::Invocation(invocation) => invocation.backing_view(resource_id),
            Self::Wave { wave, node_index } => wave.backing_view(node_index, resource_id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedOperationResourceSource {
    PlanStatic { slot_index: usize },
    Dynamic { descriptor_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedOperationResource {
    resource_id: ResourceId,
    source: PreparedOperationResourceSource,
}

/// Immutable per-node recipe compiled while the runtime registry is bound to
/// an exact plan. Static catalog, operation, provider, and resource-shape
/// proofs terminate here; dispatch retains only live authority and buffer
/// validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedOperationDispatchBinding {
    node_index: usize,
    resources: Vec<PreparedOperationResource>,
    binding_component_views: Vec<Vec<usize>>,
    scratch_view: Option<usize>,
    binding_view: Option<usize>,
    persistent_view: Option<usize>,
}

impl PreparedOperationDispatchBinding {
    pub(super) fn prepare(
        resolved: &dyn ExecutablePlanView,
        provider: &OperationProviderDescriptor,
        node_id: &NodeId,
    ) -> Result<Self, VNextError> {
        let plan = resolved.execution_plan();
        let (node_index, node) = plan
            .payload()
            .nodes()
            .iter()
            .enumerate()
            .find(|(_, node)| node.id() == node_id)
            .ok_or_else(|| invalid_operation(format!("plan has no node `{node_id}`")))?;
        let operation = resolved.capabilities().operation(node.operation_id())?;
        let registered = resolved
            .capabilities()
            .providers_for(node.operation_id())?
            .iter()
            .find(|candidate| candidate.provider_id() == provider.provider_id())
            .ok_or_else(|| invalid_operation("operation provider is absent from the catalog"))?;
        if registered != provider
            || provider.provider_id() != node.selection().selected_provider()
            || provider.operation_id() != node.operation_id()
            || provider.operation_fingerprint() != node.operation_fingerprint()
            || provider.provider_implementation_fingerprint()
                != node.provider_implementation_fingerprint()
            || provider.execution_semantics() != node.provider_execution_semantics()
            || provider.device_id() != plan.payload().device_id()
            || !provider.version().satisfies(node.operation_version())
        {
            return Err(invalid_operation(
                "operation provider is not the exact catalog entry selected by the plan",
            ));
        }
        operation.validate_attributes(node.attributes())?;
        operation.validate_resolved_bindings(node.values())?;

        let provider_resources = node.provider_resources();
        if provider_resources.provider_id() != provider.provider_id()
            || provider_resources.estimator_id() != provider.resource_estimator_id()
            || provider_resources.estimator_version() != provider.resource_estimator_version()
            || provider_resources.estimator_implementation_fingerprint()
                != provider.resource_estimator_implementation_fingerprint()
            || provider_resources.value_alignment_bytes()
                < operation.resources.minimum_value_alignment_bytes
            || provider_resources.value_alignment_bytes()
                % operation.resources.minimum_value_alignment_bytes
                != 0
            || !operation
                .resources
                .scratch
                .accepts(provider_resources.scratch().is_some())
            || !operation
                .resources
                .binding
                .accepts(provider_resources.binding().is_some())
            || !operation
                .resources
                .persistent
                .accepts(provider_resources.persistent().is_some())
        {
            return Err(invalid_operation(
                "plan provider resource estimate is not bound to the selected provider and operation contract",
            ));
        }
        let scratch_resource = select_workspace_resource(
            provider_resources.scratch(),
            node.scratch_resource(),
            "scratch",
        )?;
        let binding_resource = select_workspace_resource(
            provider_resources.binding(),
            node.binding_resource(),
            "binding",
        )?;
        let persistent_resource = select_workspace_resource(
            provider_resources.persistent(),
            node.persistent_resource(),
            "persistent",
        )?;

        let memory = plan.payload().memory();
        let mut required_resources = node
            .values()
            .iter()
            .flat_map(|binding| binding.storage().components())
            .map(|component| component.resource_id().clone())
            .collect::<BTreeSet<_>>();
        required_resources.extend(scratch_resource.iter().map(|resource| (*resource).clone()));
        required_resources.extend(binding_resource.iter().map(|resource| (*resource).clone()));
        required_resources.extend(
            persistent_resource
                .iter()
                .map(|resource| (*resource).clone()),
        );
        let resources = required_resources
            .into_iter()
            .map(|resource_id| {
                let static_index = memory
                    .static_allocations()
                    .binary_search_by(|allocation| allocation.resource_id().cmp(&resource_id));
                let dynamic_index = memory
                    .dynamic_descriptors()
                    .binary_search_by(|descriptor| descriptor.base_resource_id().cmp(&resource_id));
                let source = match (static_index, dynamic_index) {
                    (Ok(slot_index), Err(_)) => {
                        PreparedOperationResourceSource::PlanStatic { slot_index }
                    }
                    (Err(_), Ok(descriptor_index)) => {
                        PreparedOperationResourceSource::Dynamic { descriptor_index }
                    }
                    (Ok(_), Ok(_)) => {
                        return Err(invalid_operation(format!(
                            "plan resource `{resource_id}` is both static and dynamic"
                        )))
                    }
                    (Err(_), Err(_)) => {
                        return Err(invalid_operation(format!(
                            "plan has no static allocation or dynamic descriptor for `{resource_id}`"
                        )));
                    }
                };
                Ok(PreparedOperationResource {
                    resource_id,
                    source,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let view_index_for = |resource_id: &ResourceId, kind: &str| {
            resources
                .binary_search_by(|resource| resource.resource_id.cmp(resource_id))
                .map_err(|_| invalid_operation(format!("{kind} resource view is missing")))
        };
        let binding_component_views = node
            .values()
            .iter()
            .map(|binding| {
                binding
                    .storage()
                    .components()
                    .iter()
                    .map(|component| view_index_for(component.resource_id(), "value binding"))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scratch_view = scratch_resource
            .map(|resource| view_index_for(resource, "scratch"))
            .transpose()?;
        let binding_view = binding_resource
            .map(|resource| view_index_for(resource, "binding"))
            .transpose()?;
        let persistent_view = persistent_resource
            .map(|resource| view_index_for(resource, "persistent"))
            .transpose()?;
        Ok(Self {
            node_index,
            resources,
            binding_component_views,
            scratch_view,
            binding_view,
            persistent_view,
        })
    }

    pub(super) fn node<'plan>(
        &self,
        resolved: &'plan dyn ExecutablePlanView,
        node_id: &NodeId,
    ) -> Result<&'plan PlanNode, VNextError> {
        resolved
            .execution_plan()
            .payload()
            .nodes()
            .get(self.node_index)
            .filter(|node| node.id() == node_id)
            .ok_or_else(|| {
                invalid_operation("prepared operation binding differs from its plan node")
            })
    }
}

/// One participant projection inside a plan-selected physical batch. It has
/// no public constructor and does not own submission authority.
pub struct OperationInvocation<'a, B> {
    identity: &'a ExecutionIdentityEnvelope,
    operation: &'a OperationDescriptor,
    node_id: &'a NodeId,
    provider_id: &'a ProviderId,
    views: Vec<OperationBufferView<'a, B>>,
    bindings: &'a [ResolvedValueBinding],
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    work: &'a NodeWorkContract,
    scratch_view: Option<usize>,
    binding_view: Option<usize>,
    persistent_view: Option<usize>,
    work_shape: &'a BatchWorkShape,
    claimed_backing_fingerprint: &'a str,
}

impl<'a, B> OperationInvocation<'a, B> {
    #[allow(clippy::too_many_arguments)]
    fn from_prepared<R>(
        runtime: &R,
        resolved: &'a dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        node: &'a PlanNode,
        operation: &'a OperationDescriptor,
        identity: &'a ExecutionIdentityEnvelope,
        node_id: &'a NodeId,
        resources: OperationInvocationResources<'a, R>,
        active_binding: &TrustedActiveSequenceBinding,
        participant_index: usize,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        let plan = resolved.execution_plan();
        let parts = identity.parts();
        let participant = resources.participant(participant_index)?;
        let participant_backing = resources.participant_backing_snapshot(participant_index)?;
        let participant_frame = resources
            .participant_frames()?
            .get(participant_index)
            .ok_or_else(|| invalid_operation("operation participant frame is missing"))?;
        let participant_session = resources.participant_session_identity(participant_index)?;
        let static_lease = participant.static_provisioning();
        let lease_identity = static_lease.map(|lease| lease.identity());
        let admission = active_binding.plan().static_provisioning_binding();
        let pool_fingerprint = active_binding.static_pool_identity_fingerprint_ref();
        let memory = plan.payload().memory();
        if resources.participant_count()? != resources.prepared_participant_count()?
            || resources.node_id()? != node_id
            || participant_frame.sequence_authority() != participant.sequence_authority()
            || participant_frame.request_authority() != participant.request_authority()
            || !resources.plan_evidence_matches(active_binding.plan())?
            || resources.coordinator_id()? != active_binding.coordinator_id()
            || participant.sequence_authority() != active_binding.sequence_authority()
            || participant.run_id() != active_binding.run_id()
            || participant.request_id() != active_binding.request_id()
            || !active_binding
                .matches_sequence_session(participant_session.0, participant_session.1)
            || runtime.descriptor() != resolved.device()
            || runtime.descriptor() != resolved.capabilities().device()
            || runtime.descriptor().runtime_implementation_fingerprint
                != plan.payload().device_runtime_implementation_fingerprint()
            || parts.plan_id.as_ref() != Some(plan.payload().plan_id())
            || parts.plan_hash.as_ref() != Some(plan.plan_hash())
            || parts.frame_id != Some(participant_frame.frame_id())
            || parts.node_invocation_id.is_none()
            || parts.node_id.as_ref() != Some(node.id())
            || parts.operation_id.as_ref() != Some(node.operation_id())
            || parts.provider_id.as_ref() != Some(node.selection().selected_provider())
            || parts.device_id.as_ref() != Some(plan.payload().device_id())
            || parts.run_id != *active_binding.run_id()
            || parts.request_id != *active_binding.request_id()
            || parts.transaction_id.as_ref()
                != lease_identity.map(|identity| identity.transaction_id())
            || parts.resource_pool_id != active_binding.static_pool_id()
            || parts.resource_pool_identity_fingerprint.as_deref() != pool_fingerprint
            || parts.provisioning_run_id.as_ref()
                != lease_identity.map(|identity| identity.run_id())
            || parts.provisioning_request_id.as_ref()
                != lease_identity.map(|identity| identity.request_id())
            || parts.active_sequence_slot != Some(active_binding.sequence_authority().sparse_id())
            || parts.admission_generation != Some(active_binding.sequence_authority().generation())
            || parts.activation_epoch != Some(active_binding.activation_epoch())
            || parts.runtime_implementation_fingerprint.as_deref()
                != Some(active_binding.runtime_implementation_fingerprint())
            || parts.active_sequence_fingerprint.as_deref() != Some(active_binding.fingerprint())
            || parts.completed_sequence_fingerprint.is_some()
            || parts.aborted_sequence_fingerprint.is_some()
            || active_binding.plan().plan_id() != plan.payload().plan_id()
            || active_binding.plan().plan_hash() != plan.plan_hash()
            || active_binding.plan().device_id() != plan.payload().device_id()
            || active_binding.plan().runtime_implementation_fingerprint()
                != plan.payload().device_runtime_implementation_fingerprint()
            || active_binding.runtime_implementation_fingerprint()
                != runtime.descriptor().runtime_implementation_fingerprint
            || active_binding.static_provisioning_identity() != lease_identity
            || admission != static_lease.map(|lease| lease.admission())
            || admission.is_some_and(|admission| {
                admission.device_capacity_bytes() != memory.device_capacity_bytes()
                    || admission.usable_capacity_bytes() != memory.usable_capacity_bytes()
                    || admission.plan_static_bytes() != memory.static_bytes()
                    || admission.maximum_active_sequences() != memory.maximum_active_sequences()
            })
            || parts.resource_id.is_some()
            || parts.resource_generation.is_some()
            || parts.resource_batch_fingerprint.is_some()
        {
            return Err(invalid_operation(
                "operation invocation does not close over the runtime device, selected plan, node, provider, request, and lease transaction",
            ));
        }
        let provider_resources = node.provider_resources();
        let mut views = Vec::with_capacity(prepared.resources.len());
        for resource in &prepared.resources {
            let resource_id = &resource.resource_id;
            match resource.source {
                PreparedOperationResourceSource::PlanStatic { slot_index } => {
                    let allocation = memory
                        .static_allocations()
                        .get(slot_index)
                        .filter(|allocation| allocation.resource_id() == resource_id)
                        .ok_or_else(|| {
                            invalid_operation(
                                "prepared static resource index differs from the memory plan",
                            )
                        })?;
                    let lease = static_lease.ok_or_else(|| {
                        invalid_operation(format!(
                            "plan-static resource `{resource_id}` lacks static provisioning"
                        ))
                    })?;
                    let leased = lease.plan_static_view(slot_index, allocation)?;
                    views.push(OperationBufferView::from_static(
                        leased,
                        participant.device_buffer_retention(),
                    ));
                }
                PreparedOperationResourceSource::Dynamic { descriptor_index } => {
                    let descriptor = memory
                        .dynamic_descriptors()
                        .get(descriptor_index)
                        .filter(|descriptor| descriptor.base_resource_id() == resource_id)
                        .ok_or_else(|| {
                            invalid_operation(
                                "prepared dynamic resource index differs from the memory plan",
                            )
                        })?;
                    let descriptor_lifetime = descriptor.lifetime();
                    let packed_batch_coordinates = resource_uses_packed_batch_coordinates(
                        memory,
                        descriptor.base_resource_id(),
                    )?;
                    let backing = resources.backing_view(resource_id).or_else(|_| {
                        resources.participant_backing_view(participant_index, resource_id)
                    })?;
                    let expected_backing_bytes = match descriptor.lifetime() {
                        AllocationLifetime::Invocation => descriptor
                            .evaluate_request_bytes_for_shape(
                                resources.work_shape()?.immediate_shape(),
                            )?,
                        AllocationLifetime::Step => descriptor.evaluate_request_bytes_for_shape(
                            resources.step_resources().work_shape().immediate_shape(),
                        )?,
                        AllocationLifetime::Sequence => {
                            let participant_token_range = resources
                                .work_shape()?
                                .participant_token_ranges()
                                .get(participant_index)
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "operation participant token range is missing",
                                    )
                                })?;
                            let execution_shape = sequence_execution_shape(
                                participant_backing.committed_shape(),
                                participant_token_range.source_token_range().end,
                            )?;
                            descriptor.evaluate_request_bytes_for_shape(execution_shape)?
                        }
                        AllocationLifetime::Request => descriptor.evaluate_fit_request_bytes(
                            participant.request_resources().work_shape(),
                        )?,
                        AllocationLifetime::Plan => {
                            return Err(invalid_operation(format!(
                                "plan-lifetime resource `{resource_id}` cannot use dynamic backing"
                            )))
                        }
                    };
                    let size_matches = match descriptor.lifetime() {
                        AllocationLifetime::Sequence => {
                            backing.size_bytes() >= expected_backing_bytes
                        }
                        _ => backing.size_bytes() == expected_backing_bytes,
                    };
                    if !size_matches
                        || backing.capacity_size_bytes() < backing.size_bytes()
                        || backing.alignment_bytes() != descriptor.alignment_bytes()
                        || backing.usage() != descriptor.usage()
                        || backing.element_type() != descriptor.element_type()
                        || backing.storage_profile() != descriptor.storage().profile()
                    {
                        return Err(invalid_operation(format!(
                            "logical backing extent differs from plan descriptor `{resource_id}`"
                        )));
                    }
                    let participant_window = match (
                        descriptor.lifetime(),
                        descriptor.kind(),
                        descriptor.demand(),
                    ) {
                        (
                            AllocationLifetime::Step,
                            AllocationKind::Value,
                            DynamicResourceDemand::ActualSequences {
                                bytes_per_sequence,
                                maximum_sequences,
                            },
                        ) => {
                            let work_shape = resources.step_resources().work_shape();
                            if work_shape.immediate_sequences() > *maximum_sequences
                                || participant_index >= work_shape.participants().len()
                            {
                                return Err(invalid_operation(
                                    "participant fixed resource exceeds its Step work shape",
                                ));
                            }
                            let offset = bytes_per_sequence
                                .checked_mul(u64::try_from(participant_index).map_err(|_| {
                                    invalid_operation(
                                        "participant fixed resource index exceeds u64",
                                    )
                                })?)
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "participant fixed resource offset overflows u64",
                                    )
                                })?;
                            Some((offset, *bytes_per_sequence))
                        }
                        _ => None,
                    };
                    let view_bytes = participant_window
                        .map(|(_, bytes_per_sequence)| bytes_per_sequence)
                        .unwrap_or(expected_backing_bytes);
                    let descriptor = BufferDescriptor {
                        resource_id: resource_id.clone(),
                        size_bytes: view_bytes,
                        alignment_bytes: backing.alignment_bytes(),
                        usage: backing.usage(),
                        element_type: backing.element_type(),
                    };
                    let view = if let Some((offset, _)) = participant_window {
                        OperationBufferView::from_backing_window(
                            descriptor,
                            backing,
                            offset,
                            descriptor_lifetime,
                        )
                    } else if backing.capacity_size_bytes() > expected_backing_bytes {
                        OperationBufferView::from_backing_prefix(
                            descriptor,
                            backing,
                            descriptor_lifetime,
                        )
                    } else {
                        OperationBufferView::from_backing_exact(
                            descriptor,
                            backing,
                            descriptor_lifetime,
                        )
                    };
                    views.push(view.with_packed_batch_coordinates(packed_batch_coordinates));
                }
            }
        }

        for view in &views {
            view.validate_runtime(runtime, lease_identity)?;
            let translated = view.translate(0, view.descriptor().size_bytes)?;
            let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                total
                    .checked_add(region.length_bytes())
                    .ok_or_else(|| invalid_operation("translated operation regions overflow u64"))
            })?;
            if translated_bytes != view.descriptor().size_bytes {
                return Err(invalid_operation(format!(
                    "operation resource `{}` is not fully backed by physical regions",
                    view.resource_id()
                )));
            }
        }
        if node.values().len() != prepared.binding_component_views.len() {
            return Err(invalid_operation(
                "prepared value-binding recipe differs from its plan node",
            ));
        }
        for (binding, component_views) in
            node.values().iter().zip(&prepared.binding_component_views)
        {
            if binding.storage().components().len() != component_views.len() {
                return Err(invalid_operation(
                    "prepared component recipe differs from its value binding",
                ));
            }
            for (component, view_index) in
                binding.storage().components().iter().zip(component_views)
            {
                let view = views.get(*view_index).ok_or_else(|| {
                    invalid_operation("value binding lacks a committed resource view")
                })?;
                if view.resource_id() != component.resource_id() {
                    return Err(invalid_operation(
                        "prepared value-binding view differs from its resource",
                    ));
                }
                let dynamic_demand = match prepared
                    .resources
                    .get(*view_index)
                    .map(|resource| resource.source)
                {
                    Some(PreparedOperationResourceSource::PlanStatic { .. }) => None,
                    Some(PreparedOperationResourceSource::Dynamic { descriptor_index }) => Some(
                        memory
                            .dynamic_descriptors()
                            .get(descriptor_index)
                            .filter(|descriptor| {
                                descriptor.base_resource_id() == component.resource_id()
                            })
                            .ok_or_else(|| {
                                invalid_operation(
                                    "prepared component descriptor differs from the memory plan",
                                )
                            })?
                            .demand(),
                    ),
                    None => {
                        return Err(invalid_operation(
                            "prepared component view index is out of range",
                        ))
                    }
                };
                let coverage = validate_value_binding_physical_coverage(
                    node.work(),
                    binding,
                    component,
                    view.descriptor(),
                    dynamic_demand,
                    provider_resources.value_alignment_bytes(),
                )?;
                if coverage == ValueBindingPhysicalCoverage::CanonicalComponent {
                    let translated =
                        view.translate(component.offset_bytes(), component.length_bytes())?;
                    let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                        total.checked_add(region.length_bytes()).ok_or_else(|| {
                            invalid_operation("translated value-binding regions overflow u64")
                        })
                    })?;
                    if translated_bytes != component.length_bytes() {
                        return Err(invalid_operation(format!(
                            "resource `{}` does not physically cover its value binding",
                            component.resource_id()
                        )));
                    }
                }
            }
        }
        validate_workspace(
            &views,
            prepared.scratch_view,
            BufferUsage::Scratch,
            provider_resources.scratch(),
            "scratch",
        )?;
        validate_workspace(
            &views,
            prepared.binding_view,
            BufferUsage::Binding,
            provider_resources.binding(),
            "binding",
        )?;
        validate_workspace(
            &views,
            prepared.persistent_view,
            BufferUsage::Persistent,
            provider_resources.persistent(),
            "persistent",
        )?;
        Ok(Self {
            identity,
            operation,
            node_id,
            provider_id: node.selection().selected_provider(),
            views,
            bindings: node.values(),
            attributes: node.attributes(),
            work: node.work(),
            scratch_view: prepared.scratch_view,
            binding_view: prepared.binding_view,
            persistent_view: prepared.persistent_view,
            work_shape: resources.work_shape()?,
            claimed_backing_fingerprint: resources.backing_fingerprint(),
        })
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        self.identity
    }

    pub fn operation(&self) -> &OperationDescriptor {
        self.operation
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        self.provider_id
    }

    pub fn views(&self) -> &[OperationBufferView<'a, B>] {
        &self.views
    }

    pub fn bindings(&self) -> &[ResolvedValueBinding] {
        self.bindings
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.attributes
    }

    pub fn work(&self) -> &NodeWorkContract {
        self.work
    }

    pub fn scratch_view(&self) -> Option<&OperationBufferView<'a, B>> {
        self.scratch_view.map(|index| &self.views[index])
    }

    pub fn binding_view(&self) -> Option<&OperationBufferView<'a, B>> {
        self.binding_view.map(|index| &self.views[index])
    }

    pub fn persistent_view(&self) -> Option<&OperationBufferView<'a, B>> {
        self.persistent_view.map(|index| &self.views[index])
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape
    }

    pub fn claimed_backing_fingerprint(&self) -> &str {
        self.claimed_backing_fingerprint
    }
}

/// Borrowed provider view for exactly one physical command. Participant-local
/// resources remain separate projections while invocation/step/plan resources
/// may be shared by every projection.
pub struct BatchedOperationInvocation<'a, B> {
    batch_identity: &'a BatchOperationIdentity,
    node_identity: &'a BatchOperationNodeIdentity,
    participants: Vec<OperationInvocation<'a, B>>,
    program_binding: Option<ProgramBindingNodeBinding>,
}

impl<'a, B> BatchedOperationInvocation<'a, B> {
    pub(super) fn from_resolved<R>(
        runtime: &R,
        resolved: &'a dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        batch_identity: &'a BatchOperationIdentity,
        resources: &'a InvocationResourceLease<R>,
        active_bindings: &'a [TrustedActiveSequenceBinding],
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        let node_identity = batch_identity.single_node().ok_or_else(|| {
            invalid_operation("single-operation invocation received a multi-node batch identity")
        })?;
        Self::from_resources(
            runtime,
            resolved,
            prepared,
            batch_identity,
            node_identity,
            OperationInvocationResources::Invocation(resources),
            active_bindings.iter(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_wave_node<'binding, R, I>(
        runtime: &R,
        resolved: &'a dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        batch_identity: &'a BatchOperationIdentity,
        node_identity: &'a BatchOperationNodeIdentity,
        wave: &'a PreparedStepSubmissionWave<R>,
        node_index: usize,
        active_bindings: I,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
        I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        Self::from_resources(
            runtime,
            resolved,
            prepared,
            batch_identity,
            node_identity,
            OperationInvocationResources::Wave { wave, node_index },
            active_bindings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_resources<'binding, R, I>(
        runtime: &R,
        resolved: &'a dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        batch_identity: &'a BatchOperationIdentity,
        node_identity: &'a BatchOperationNodeIdentity,
        resources: OperationInvocationResources<'a, R>,
        active_bindings: I,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
        I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        let participant_count = resources.participant_count()?;
        let participant_frames = resources.participant_frames()?;
        if participant_count == 0
            || participant_count != active_bindings.len()
            || participant_count != node_identity.participants().len()
            || participant_count != participant_frames.len()
            || batch_identity.batch_step_id() != resources.batch_step_id()
            || batch_identity.batch_invocation_id() != resources.batch_invocation_id()
            || node_identity.node_id() != resources.node_id()?
            || node_identity.work_shape_fingerprint() != resources.work_shape()?.fingerprint()
            || batch_identity.claimed_backing_fingerprint() != resources.backing_fingerprint()
            || node_identity
                .participants()
                .iter()
                .zip(participant_frames)
                .any(|(participant, frame)| {
                    let key = participant.node_key();
                    key.sequence_authority() != frame.sequence_authority()
                        || key.request_authority() != frame.request_authority()
                        || key.frame_id() != frame.frame_id()
                        || key.node_id() != node_identity.node_id()
                })
        {
            return Err(invalid_operation(
                "batched operation identity differs from its exact invocation resources",
            ));
        }
        let node = prepared.node(resolved, node_identity.node_id())?;
        let operation = resolved.capabilities().operation(node.operation_id())?;
        let participants = node_identity
            .participants()
            .iter()
            .zip(active_bindings)
            .enumerate()
            .map(|(index, (participant, active_binding))| {
                OperationInvocation::from_prepared(
                    runtime,
                    resolved,
                    prepared,
                    node,
                    operation,
                    participant.identity(),
                    node_identity.node_id(),
                    resources,
                    active_binding,
                    index,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let program_binding = resources.program_binding_node();
        Ok(Self {
            batch_identity,
            node_identity,
            participants,
            program_binding,
        })
    }

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        self.batch_identity
    }

    pub fn participants(&self) -> &[OperationInvocation<'a, B>] {
        &self.participants
    }

    pub fn operation(&self) -> &OperationDescriptor {
        self.participants[0].operation()
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_identity.node_id()
    }

    pub fn provider_id(&self) -> &ProviderId {
        self.node_identity.provider_id()
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.participants[0].work_shape()
    }

    pub fn work_contract(&self) -> &NodeWorkContract {
        self.participants[0].work()
    }

    pub fn program_binding(&self) -> Option<&ProgramBindingNodeBinding> {
        self.program_binding.as_ref()
    }

    /// Attaches the command that refreshes this provider's invocation-scoped
    /// binding workspace. A compiled slot moves the command into the
    /// non-aliasing wave prelude; without that authority it remains adjacent
    /// to the eager compute command.
    pub fn attach_binding_command<C>(
        &self,
        operation: EncodedDeviceOperation<C>,
        command: C,
    ) -> EncodedDeviceOperation<C> {
        if self.program_binding.is_some() {
            operation.with_program_binding(command)
        } else {
            operation.with_dynamic_binding(command)
        }
    }

    pub fn participant_token_ranges(&self) -> &[BatchParticipantTokenRange] {
        self.work_shape().participant_token_ranges()
    }

    /// Returns whether one exact resolved binding uses the packed token
    /// coordinate space of this physical batch. This is deliberately distinct
    /// from physical sharing: Request backing can be shared by child sequences
    /// while still using source-token coordinates.
    pub fn binding_uses_packed_batch_coordinates(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<bool, VNextError> {
        let mut packed_batch_coordinates = None;
        for participant in &self.participants {
            let binding = participant
                .bindings()
                .iter()
                .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
                .ok_or_else(|| {
                    invalid_operation(format!(
                        "operation participant lacks {role:?} binding {ordinal}"
                    ))
                })?;
            let [component] = binding.storage().components() else {
                return Err(invalid_operation(
                    "batch sharing requires a single-resource value binding",
                ));
            };
            let view = participant
                .views()
                .iter()
                .find(|view| view.resource_id() == component.resource_id())
                .ok_or_else(|| {
                    invalid_operation("operation value binding has no physical resource view")
                })?;
            match packed_batch_coordinates {
                Some(expected) if expected != view.uses_packed_batch_coordinates() => {
                    return Err(invalid_operation(
                        "operation participants disagree on value coordinate space",
                    ));
                }
                None => packed_batch_coordinates = Some(view.uses_packed_batch_coordinates()),
                Some(_) => {}
            }
        }
        Ok(packed_batch_coordinates.expect("batched operation invocations are non-empty"))
    }
}

fn select_workspace_resource<'a>(
    requirement: Option<&ProviderWorkspaceRequirement>,
    resource: Option<&'a ResourceId>,
    kind: &str,
) -> Result<Option<&'a ResourceId>, VNextError> {
    let Some(requirement) = requirement else {
        if resource.is_none() {
            return Ok(None);
        }
        return Err(invalid_operation(format!(
            "plan has unrequested {kind} resources"
        )));
    };
    resource.map(Some).ok_or_else(|| {
        invalid_operation(format!(
            "{kind} workspace base identity is missing for {:?} scope",
            requirement.scope()
        ))
    })
}

fn validate_workspace<B>(
    views: &[OperationBufferView<'_, B>],
    index: Option<usize>,
    usage: BufferUsage,
    requirement: Option<&ProviderWorkspaceRequirement>,
    kind: &str,
) -> Result<(), VNextError> {
    match (requirement, index) {
        (None, None) => Ok(()),
        (None, Some(_)) | (Some(_), None) => Err(invalid_operation(format!(
            "{kind} workspace presence differs from the operation contract"
        ))),
        (Some(requirement), Some(index)) => {
            let descriptor = views[index].descriptor();
            let required_bytes = requirement.minimum_bytes()?;
            if descriptor.usage != usage
                || descriptor.element_type != ElementType::U8
                || descriptor.size_bytes < required_bytes
                || descriptor.alignment_bytes < requirement.alignment_bytes()
                || descriptor.alignment_bytes % requirement.alignment_bytes() != 0
            {
                return Err(invalid_operation(format!(
                    "{kind} workspace descriptor is invalid"
                )));
            }
            let translated = views[index].translate(0, required_bytes)?;
            let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                total.checked_add(region.length_bytes()).ok_or_else(|| {
                    invalid_operation(format!("{kind} workspace region coverage overflows u64"))
                })
            })?;
            if translated_bytes != required_bytes {
                return Err(invalid_operation(format!(
                    "{kind} workspace is not fully backed by physical regions"
                )));
            }
            Ok(())
        }
    }
}
