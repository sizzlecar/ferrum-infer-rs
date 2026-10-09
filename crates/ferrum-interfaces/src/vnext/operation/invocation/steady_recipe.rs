//! A sealed contiguous-node schema and fresh per-node physical preparation.
use super::*;
use crate::vnext::operation::retained_dependency::RetainedPlanDependencyTemplate;
use crate::vnext::operation::steady_recipe::CheckedSteadyRecipeStorage;
use crate::vnext::{
    CheckedSteadyRecipeRegion, DeviceReusableAddressScope, DeviceReusableExecutionEntryIdentity,
    DeviceReusableExecutionProgramId, DynamicResourceDescriptor, ExecutionLaneId, LeasedBufferView,
    OperationBufferStorageKind, PreparedSteadyRecipePatch, ResourceAllocation, SteadyBackingBatch,
    SteadyRecipeDeclaration, SteadyRecipeDependency, SteadyRecipeRegionSelector,
};
use std::ops::Range;
use std::sync::Weak;

struct CapturedAllocation {
    owner: Weak<dyn Send + Sync>,
    buffer_facade: usize,
    physical: Range<u64>,
    scope: DeviceReusableAddressScope,
}

struct ColdResource {
    descriptor: BufferDescriptor,
    captured: Vec<Option<CapturedAllocation>>,
    source: ColdResourceSource,
}

enum ColdResourceSource {
    Static {
        index: usize,
        allocation: ResourceAllocation,
    },
    Dynamic {
        index: usize,
        descriptor: DynamicResourceDescriptor,
    },
}

pub(crate) struct SealedNodeRecipe<R: DeviceRuntime> {
    pub(crate) entry: DeviceReusableExecutionEntryIdentity,
    pub(crate) program: DeviceReusableExecutionProgramId,
    pub(crate) lane: ExecutionLaneId,
    pub(crate) epoch: u64,
    pub(crate) node_index: usize,
    pub(crate) provider_owner: Weak<()>,
    runtime: Weak<R>,
    node: PlanNode,
    prepared: PreparedOperationDispatchBinding,
    participant_count: usize,
    resources: Vec<ColdResource>,
    selected: Vec<(usize, u64)>,
    declaration: SteadyRecipeDeclaration,
    dependencies: Vec<(SteadyRecipeDependency, RetainedPlanDependencyTemplate)>,
}

fn validate_common<'a, 'runtime, R: DeviceRuntime>(
    runtime: &'runtime R,
    resolved: &'a dyn ExecutablePlanView,
    node: &'a PlanNode,
    identity: &'a ExecutionIdentityEnvelope,
    node_id: &'a NodeId,
    resources: OperationInvocationResources<'a, R>,
    active_binding: &TrustedActiveSequenceBinding,
    participant_index: usize,
    device_agreements: &mut [DeviceDescriptorAgreement<'runtime, 'a>; 2],
) -> Result<(), VNextError> {
    let plan = resolved.execution_plan();
    let parts = identity.projection();
    let participant = resources.participant(participant_index)?;
    let _participant_backing = resources.participant_backing_snapshot(participant_index)?;
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
        || !active_binding.matches_sequence_session(participant_session.0, participant_session.1)
        || !device_agreements[0].matches(runtime.descriptor(), resolved.device())
        || !device_agreements[1].matches(runtime.descriptor(), resolved.capabilities().device())
        || runtime.descriptor().runtime_implementation_fingerprint
            != plan.payload().device_runtime_implementation_fingerprint()
        || parts.plan_id != Some(plan.payload().plan_id())
        || parts.plan_hash != Some(plan.plan_hash())
        || parts.frame_id != Some(participant_frame.frame_id())
        || parts.node_invocation_id.is_none()
        || parts.node_id != Some(node.id())
        || parts.operation_id != Some(node.operation_id())
        || parts.provider_id != Some(node.selection().selected_provider())
        || parts.device_id != Some(plan.payload().device_id())
        || parts.run_id != active_binding.run_id()
        || parts.request_id != active_binding.request_id()
        || parts.transaction_id != lease_identity.map(|identity| identity.transaction_id())
        || parts.resource_pool_id != active_binding.static_pool_id().as_ref()
        || parts.resource_pool_identity_fingerprint != pool_fingerprint
        || parts.provisioning_run_id != lease_identity.map(|identity| identity.run_id())
        || parts.provisioning_request_id != lease_identity.map(|identity| identity.request_id())
        || parts.active_sequence_slot != Some(active_binding.sequence_authority().sparse_id())
        || parts.admission_generation != Some(active_binding.sequence_authority().generation())
        || parts.activation_epoch != Some(active_binding.activation_epoch())
        || parts.runtime_implementation_fingerprint
            != Some(active_binding.runtime_implementation_fingerprint())
        || parts.active_sequence_fingerprint != Some(active_binding.fingerprint())
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
    Ok(())
}

impl<R: DeviceRuntime> SealedNodeRecipe<R> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vnext::operation) fn seal(
        runtime: &Arc<R>,
        resolved: &dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        invocation: &BatchedOperationInvocation<'_, R::Buffer>,
        declaration: SteadyRecipeDeclaration,
        provider_owner: &Arc<()>,
        entry: DeviceReusableExecutionEntryIdentity,
        program: DeviceReusableExecutionProgramId,
        lane: ExecutionLaneId,
        epoch: u64,
        node_index: usize,
    ) -> Result<Option<Self>, VNextError> {
        let Some(metadata) = runtime.immutable_runtime_metadata() else {
            return Ok(None);
        };
        let descriptor = metadata.for_owner(runtime.as_ref()).ok_or_else(|| {
            invalid_operation("immutable runtime declaration belongs to another owner")
        })?;
        if descriptor != runtime.descriptor() {
            return Err(invalid_operation(
                "immutable runtime declaration differs from its original getter",
            ));
        }
        let participant_count = invocation.participants().len();
        if participant_count == 0
            || invocation.work_shape().immediate_tokens() != participant_count as u64
            || invocation
                .participants()
                .iter()
                .any(|p| p.views().len() != prepared.resources.len())
        {
            return Ok(None);
        }
        let plan = resolved.execution_plan();
        let node = prepared.node(resolved, invocation.node_id())?;
        let memory = plan.payload().memory();
        let mut resources = Vec::with_capacity(prepared.resources.len());
        for (index, resource) in prepared.resources.iter().enumerate() {
            let first = &invocation.participants()[0].views()[index];
            if first.storage_kind() == OperationBufferStorageKind::DynamicPaged {
                return Ok(None);
            }
            let source = match resource.source {
                PreparedOperationResourceSource::PlanStatic { slot_index } => {
                    ColdResourceSource::Static {
                        index: slot_index,
                        allocation: memory.static_allocations()[slot_index].clone(),
                    }
                }
                PreparedOperationResourceSource::Dynamic { descriptor_index } => {
                    ColdResourceSource::Dynamic {
                        index: descriptor_index,
                        descriptor: memory.dynamic_descriptors()[descriptor_index].clone(),
                    }
                }
            };
            let mut captured = Vec::with_capacity(participant_count);
            for participant in invocation.participants() {
                let view = &participant.views()[index];
                if view.descriptor() != first.descriptor()
                    || view.resource_id() != &resource.resource_id
                {
                    return Ok(None);
                }
                let regions = view.translate(0, view.descriptor().size_bytes)?;
                let mut iter = regions.iter();
                let physical = iter.next().ok_or_else(|| {
                    invalid_operation("steady cold resource has no physical region")
                })?;
                if iter.next().is_some() {
                    return Ok(None);
                }
                let (buffer, range, retention) = physical.buffer_and_physical_range();
                let Some(metadata) = runtime.immutable_buffer_metadata(buffer) else {
                    return Ok(None);
                };
                let declared = metadata
                    .for_owner(runtime.as_ref(), buffer)
                    .ok_or_else(|| {
                        invalid_operation("immutable buffer declaration belongs to another owner")
                    })?;
                if declared != &runtime.buffer_descriptor(buffer) {
                    return Err(invalid_operation(
                        "immutable buffer declaration differs from its original getter",
                    ));
                }
                captured.push(
                    retention
                        .reusable_address_scope()
                        .map(|scope| CapturedAllocation {
                            owner: retention.weak_allocation_owner(),
                            buffer_facade: buffer as *const R::Buffer as usize,
                            physical: range,
                            scope,
                        }),
                );
            }
            resources.push(ColdResource {
                descriptor: first.descriptor().clone(),
                captured,
                source,
            });
        }
        let mut selected = Vec::with_capacity(declaration.regions.len());
        for request in &declaration.regions {
            let (resource_index, base, available) = match &request.selector {
                SteadyRecipeRegionSelector::Value {
                    role,
                    ordinal,
                    component,
                } => {
                    let (binding_index, binding) = node
                        .values()
                        .iter()
                        .enumerate()
                        .find(|(_, b)| b.role() == *role && b.ordinal() == *ordinal)
                        .ok_or_else(|| {
                            invalid_operation("steady declaration selects an absent value")
                        })?;
                    let components = binding.storage().components();
                    let component_index = match component {
                        Some(id) => components.iter().position(|c| c.component_id() == Some(id)),
                        None => (components.len() == 1).then_some(0),
                    }
                    .ok_or_else(|| {
                        invalid_operation("steady declaration has an ambiguous or absent component")
                    })?;
                    let component = &components[component_index];
                    (
                        prepared.binding_component_views[binding_index][component_index],
                        component.offset_bytes(),
                        component.length_bytes(),
                    )
                }
                selector => {
                    let index = match selector {
                        SteadyRecipeRegionSelector::Scratch => prepared.scratch_view,
                        SteadyRecipeRegionSelector::Persistent => prepared.persistent_view,
                        SteadyRecipeRegionSelector::ProgramBinding => prepared.binding_view,
                        _ => unreachable!(),
                    }
                    .ok_or_else(|| {
                        invalid_operation("steady declaration selects an absent workspace")
                    })?;
                    (index, 0, resources[index].descriptor.size_bytes)
                }
            };
            super::super::retained_dependency::checked_range(
                request.offset_bytes,
                request.length_bytes,
                available,
            )?;
            let offset = base
                .checked_add(request.offset_bytes)
                .ok_or_else(|| invalid_operation("steady declaration offset overflows"))?;
            let descriptor = &resources[resource_index].descriptor;
            if descriptor.element_type != request.element_type
                || descriptor.alignment_bytes < request.alignment_bytes
                || offset % request.alignment_bytes != 0
            {
                return Err(invalid_operation(
                    "steady declaration type or alignment differs from validated schema",
                ));
            }
            super::super::retained_dependency::checked_range(
                offset,
                request.length_bytes,
                descriptor.size_bytes,
            )?;
            selected.push((resource_index, offset));
        }
        let dependencies = declaration
            .dependencies
            .iter()
            .map(|spec| {
                invocation
                    .retained_plan_dependency(spec.as_spec())
                    .map(|authority| {
                        (
                            spec.clone(),
                            RetainedPlanDependencyTemplate::from_authority(authority),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            entry,
            program,
            lane,
            epoch,
            node_index,
            provider_owner: Arc::downgrade(provider_owner),
            runtime: Arc::downgrade(runtime),
            node: node.clone(),
            prepared: prepared.clone(),
            participant_count,
            resources,
            selected,
            declaration,
            dependencies,
        }))
    }
}

struct FreshResource<'a, B> {
    source: FreshSource<'a, B>,
    logical_offset: u64,
}
enum FreshSource<'a, B> {
    Static {
        view: LeasedBufferView<'a, B>,
        retention: crate::vnext::DeviceBufferRetention,
    },
    Dynamic(usize),
}

impl<R: DeviceRuntime> SealedNodeRecipe<R> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vnext::operation) fn prepare<'a, 'binding, I>(
        &self,
        runtime: &Arc<R>,
        resolved: &'a dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        batch_identity: &'a BatchOperationIdentity,
        node_identity: &'a BatchOperationNodeIdentity,
        wave: &'a PreparedStepSubmissionWave<R>,
        active_bindings: I,
    ) -> Result<Option<PreparedSteadyRecipePatch<'a, R::Buffer>>, VNextError>
    where
        I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        if !self
            .runtime
            .upgrade()
            .is_some_and(|old| Arc::ptr_eq(&old, runtime))
        {
            return Err(invalid_operation("steady recipe runtime owner changed"));
        }
        let Some(metadata) = runtime.immutable_runtime_metadata() else {
            return Ok(None);
        };
        if metadata.for_owner(runtime.as_ref()) != Some(runtime.descriptor()) {
            return Err(invalid_operation(
                "steady runtime capability owner or descriptor changed",
            ));
        }
        let resources = OperationInvocationResources::Wave {
            wave,
            node_index: self.node_index,
        };
        let participant_frames = resources.participant_frames()?;
        if self.participant_count != active_bindings.len()
            || self.participant_count != node_identity.participants().len()
            || self.participant_count != participant_frames.len()
            || self.participant_count != resources.participant_count()?
            || batch_identity.batch_step_id() != wave.batch_step_id()
            || batch_identity.batch_invocation_id() != wave.batch_invocation_id()
            || batch_identity.claimed_backing_fingerprint() != wave.fingerprint()
            || node_identity.node_id() != resources.node_id()?
            || node_identity.work_shape_fingerprint() != resources.work_shape()?.fingerprint()
            || prepared != &self.prepared
        {
            return Err(invalid_operation(
                "steady recipe differs from the exact current wave",
            ));
        }
        let node = prepared.node(resolved, node_identity.node_id())?;
        if node != &self.node {
            return Err(invalid_operation("steady compiled node schema changed"));
        }
        let mut agreements = [
            DeviceDescriptorAgreement::default(),
            DeviceDescriptorAgreement::default(),
        ];
        for (index, (identity, active)) in node_identity
            .participants()
            .iter()
            .zip(active_bindings)
            .enumerate()
        {
            let key = identity.node_key();
            let frame = &participant_frames[index];
            if key.sequence_authority() != frame.sequence_authority()
                || key.request_authority() != frame.request_authority()
                || key.frame_id() != frame.frame_id()
                || key.node_id() != node_identity.node_id()
            {
                return Err(invalid_operation(
                    "steady identity is not this participant frame",
                ));
            }
            validate_common(
                runtime.as_ref(),
                resolved,
                node,
                identity.identity(),
                node_identity.node_id(),
                resources,
                active,
                index,
                &mut agreements,
            )?;
        }
        let row_count = self
            .participant_count
            .checked_mul(self.resources.len())
            .ok_or_else(|| invalid_operation("steady resource table length overflows"))?;
        let mut fresh = Vec::with_capacity(row_count);
        let mut groups: Vec<&[crate::vnext::LogicalBackingSliceAuthority]> = Vec::new();
        let mut group_indices = std::collections::BTreeMap::new();
        // Only captured immutable schema and fresh authority references are
        // collected here. The sole pool-lock window below covers all groups.
        for participant_index in 0..self.participant_count {
            let participant = resources.participant(participant_index)?;
            let plan = resolved.execution_plan();
            for schema in &self.resources {
                match &schema.source {
                    ColdResourceSource::Static { index, allocation } => {
                        let current = plan
                            .payload()
                            .memory()
                            .static_allocations()
                            .get(*index)
                            .filter(|a| *a == allocation)
                            .ok_or_else(|| invalid_operation("steady static schema changed"))?;
                        let lease = participant.static_provisioning().ok_or_else(|| {
                            invalid_operation("steady Plan resource has no live lease")
                        })?;
                        let view = lease.plan_static_view(*index, current)?;
                        if view.generation() == 0
                            || view.identity() != lease.identity()
                            || view.committed_descriptor() != &schema.descriptor
                        {
                            return Err(invalid_operation(
                                "steady Plan resource differs from committed schema",
                            ));
                        }
                        fresh.push(FreshResource {
                            source: FreshSource::Static {
                                view,
                                retention: participant.device_buffer_retention(),
                            },
                            logical_offset: 0,
                        });
                    }
                    ColdResourceSource::Dynamic { index, descriptor } => {
                        if plan.payload().memory().dynamic_descriptors().get(*index)
                            != Some(descriptor)
                        {
                            return Err(invalid_operation("steady dynamic schema changed"));
                        }
                        let authorities = wave.steady_backing_authorities(
                            self.node_index,
                            participant_index,
                            descriptor.base_resource_id(),
                            descriptor.lifetime(),
                        )?;
                        let key = (authorities.as_ptr() as usize, authorities.len());
                        let group_index = *group_indices.entry(key).or_insert_with(|| {
                            let index = groups.len();
                            groups.push(authorities);
                            index
                        });
                        let logical_offset = match (
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
                                if resources
                                    .step_resources()
                                    .work_shape()
                                    .immediate_sequences()
                                    > *maximum_sequences
                                {
                                    return Err(invalid_operation(
                                        "steady fixed resource exceeds the current Step shape",
                                    ));
                                }
                                bytes_per_sequence
                                    .checked_mul(participant_index as u64)
                                    .ok_or_else(|| {
                                        invalid_operation("steady participant offset overflows")
                                    })?
                            }
                            _ => 0,
                        };
                        fresh.push(FreshResource {
                            source: FreshSource::Dynamic(group_index),
                            logical_offset,
                        });
                    }
                }
            }
        }
        let backing = wave.steady_backing_batch(self.node_index, &groups)?;
        // All permits have been released. Runtime declarations/getters and
        // provider calls are forbidden inside the resource interpreter.
        for binding in &backing.bindings {
            let Some(metadata) = runtime.immutable_buffer_metadata(binding.buffer()) else {
                return Ok(None);
            };
            let declared = metadata
                .for_owner(runtime.as_ref(), binding.buffer())
                .ok_or_else(|| invalid_operation("steady buffer capability has another owner"))?;
            let actual = runtime.buffer_descriptor(binding.buffer());
            if declared != &actual
                || &actual != binding.descriptor()
                || binding
                    .segment()
                    .offset_bytes()
                    .checked_add(binding.segment().length_bytes())
                    .is_none_or(|end| end > actual.size_bytes)
            {
                return Err(invalid_operation(
                    "steady physical buffer descriptor changed",
                ));
            }
        }
        for (row_index, row) in fresh.iter().enumerate() {
            let participant_index = row_index / self.resources.len();
            let schema = &self.resources[row_index % self.resources.len()];
            let (buffer, physical, retention) = match &row.source {
                FreshSource::Static { view, retention } => {
                    let Some(metadata) = runtime.immutable_buffer_metadata(view.buffer()) else {
                        return Ok(None);
                    };
                    let declared = metadata
                        .for_owner(runtime.as_ref(), view.buffer())
                        .ok_or_else(|| {
                            invalid_operation("steady static buffer capability has another owner")
                        })?;
                    let actual = runtime.buffer_descriptor(view.buffer());
                    if declared != &actual || &actual != view.committed_descriptor() {
                        return Err(invalid_operation("steady static buffer descriptor changed"));
                    }
                    (
                        view.buffer(),
                        0..schema.descriptor.size_bytes,
                        retention.clone(),
                    )
                }
                FreshSource::Dynamic(index) => {
                    let summary = &backing.resources[*index];
                    let ColdResourceSource::Dynamic { descriptor, .. } = &schema.source else {
                        unreachable!()
                    };
                    let participant = resources.participant(participant_index)?;
                    let expected = match descriptor.lifetime() {
                        AllocationLifetime::Invocation => descriptor
                            .evaluate_request_bytes_for_shape(
                                resources.work_shape()?.immediate_shape(),
                            )?,
                        AllocationLifetime::Step => descriptor.evaluate_request_bytes_for_shape(
                            resources.step_resources().work_shape().immediate_shape(),
                        )?,
                        AllocationLifetime::Sequence => {
                            let snapshot =
                                resources.participant_backing_snapshot(participant_index)?;
                            let end = resources.work_shape()?.participant_token_ranges()
                                [participant_index]
                                .source_token_range()
                                .end;
                            descriptor.evaluate_request_bytes_for_shape(
                                sequence_execution_shape(snapshot.committed_shape(), end)?,
                            )?
                        }
                        AllocationLifetime::Request => descriptor.evaluate_fit_request_bytes(
                            participant.request_resources().work_shape(),
                        )?,
                        AllocationLifetime::Plan => {
                            return Err(invalid_operation(
                                "steady dynamic resource claims Plan lifetime",
                            ))
                        }
                    };
                    let expected_view = match (
                        descriptor.lifetime(),
                        descriptor.kind(),
                        descriptor.demand(),
                    ) {
                        (
                            AllocationLifetime::Step,
                            AllocationKind::Value,
                            DynamicResourceDemand::ActualSequences {
                                bytes_per_sequence, ..
                            },
                        ) => *bytes_per_sequence,
                        _ => expected,
                    };
                    if expected_view != schema.descriptor.size_bytes {
                        return Ok(None);
                    }
                    if (descriptor.lifetime() == AllocationLifetime::Sequence
                        && summary.logical_size_bytes < expected)
                        || (descriptor.lifetime() != AllocationLifetime::Sequence
                            && summary.logical_size_bytes != expected)
                        || summary.capacity_size_bytes < summary.logical_size_bytes
                        || summary.alignment_bytes != descriptor.alignment_bytes()
                        || summary.usage != descriptor.usage()
                        || summary.element_type != descriptor.element_type()
                        || summary.storage_profile != descriptor.storage().profile()
                        || summary.bindings.len() != 1
                    {
                        return Err(invalid_operation(
                            "steady current extent differs from its declared contiguous schema",
                        ));
                    }
                    let binding = &backing.bindings[summary.bindings.start];
                    let logical_end = row
                        .logical_offset
                        .checked_add(expected_view)
                        .ok_or_else(|| invalid_operation("steady window overflows"))?;
                    if logical_end > binding.segment().length_bytes()
                        || (row.logical_offset == 0
                            && summary.capacity_size_bytes == expected
                            && binding.segment().length_bytes() != expected)
                    {
                        return Err(invalid_operation(
                            "steady physical extent does not cover its full current window",
                        ));
                    }
                    let start = binding
                        .segment()
                        .offset_bytes()
                        .checked_add(row.logical_offset)
                        .ok_or_else(|| {
                            invalid_operation("steady physical window offset overflows")
                        })?;
                    (
                        binding.buffer(),
                        start..start.checked_add(expected_view).ok_or_else(|| {
                            invalid_operation("steady physical window end overflows")
                        })?,
                        binding.retention(),
                    )
                }
            };
            if let Some(captured) = &schema.captured[participant_index] {
                if !captured.owner.ptr_eq(&retention.weak_allocation_owner())
                    || captured.buffer_facade != buffer as *const R::Buffer as usize
                    || captured.physical != physical
                    || retention.reusable_address_scope() != Some(captured.scope)
                {
                    return Err(invalid_operation(
                        "steady captured resource owner or physical range changed",
                    ));
                }
            }
        }
        let region_count = self.declaration.regions.len();
        let total = self
            .participant_count
            .checked_mul(region_count)
            .ok_or_else(|| invalid_operation("steady patch table length overflows"))?;
        let mut regions = Vec::with_capacity(total);
        for participant in 0..self.participant_count {
            for ((resource_index, offset), request) in
                self.selected.iter().zip(&self.declaration.regions)
            {
                let row = &fresh[participant * self.resources.len() + resource_index];
                let (storage, start) = match &row.source {
                    FreshSource::Static { view, retention } => (
                        CheckedSteadyRecipeStorage::Static {
                            buffer: view.buffer_for_lease(),
                            retention: retention.clone(),
                        },
                        *offset,
                    ),
                    FreshSource::Dynamic(index) => {
                        let binding = &backing.bindings[backing.resources[*index].bindings.start];
                        let start = binding
                            .segment()
                            .offset_bytes()
                            .checked_add(row.logical_offset)
                            .and_then(|base| base.checked_add(*offset))
                            .ok_or_else(|| invalid_operation("steady patch offset overflows"))?;
                        (
                            CheckedSteadyRecipeStorage::Dynamic(binding.retained_clone()),
                            start,
                        )
                    }
                };
                start
                    .checked_add(request.length_bytes)
                    .ok_or_else(|| invalid_operation("steady patch end overflows"))?;
                regions.push(CheckedSteadyRecipeRegion {
                    storage,
                    descriptor: self.resources[*resource_index].descriptor.clone(),
                    logical_offset_bytes: *offset,
                    physical_offset_bytes: start,
                    length_bytes: request.length_bytes,
                });
            }
        }
        Ok(Some(PreparedSteadyRecipePatch {
            batch_identity,
            node_identity,
            program_binding: resources
                .program_binding_node()
                .ok_or_else(|| invalid_operation("steady node has no current binding slot"))?,
            participant_count: self.participant_count,
            region_count,
            regions,
            state: Arc::clone(&self.declaration.state),
            dependency_scope: Arc::new(()),
            dependencies: self.dependencies.clone(),
        }))
    }
}
