//! Cold Plan-only compilation for a complete resident binding segment.
//! No current wave, physical view, lease, dependency authority, or command is
//! retained here. Exact live owners and entry identity belong to the fresh path.
use std::collections::{BTreeMap, BTreeSet};

use super::buffer_view::{validate_value_binding_physical_coverage, ValueBindingPhysicalCoverage};
use super::foundation::invalid_operation;
use super::invocation::PreparedOperationDispatchBinding;
use super::retained_dependency::checked_range;
use super::{
    ResolvedValueRole, ReusableBindingResources, SegmentBindingDeclaration,
    SegmentBindingDependency, SegmentBindingRegionExtent, SegmentBindingRegionRequest,
    SegmentBindingRegionSelector, TensorAccess,
};
use crate::vnext::{
    AllocationLifetime, BufferDescriptor, BufferUsage, DeviceReusableExecutionProgramId,
    ElementType, ExecutablePlanView, MemoryPlan, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ResourceId, VNextError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CompiledSegmentResourceSource {
    PlanStatic { slot_index: usize },
    Dynamic { descriptor_index: usize },
}

pub(super) struct CompiledSegmentResource {
    pub resource_id: ResourceId,
    pub source: CompiledSegmentResourceSource,
    pub required_minimum_bytes: u64,
}

pub(super) struct CompiledSegmentRegion {
    pub resource_index: usize,
    /// Absolute offset in the selected resource, including component base.
    pub offset_bytes: u64,
    /// Exact component bound before request.offset_bytes. A whole dynamic
    /// state CurrentResource has no upper bound here: its canonical component
    /// is a minimum and the fresh permit supplies its current authorized size.
    pub component_length: Option<u64>,
    pub request: SegmentBindingRegionRequest,
}
impl CompiledSegmentRegion {
    pub(super) fn length_for_current_view(&self, available: u64) -> Result<u64, VNextError> {
        let length = match self.request.extent {
            SegmentBindingRegionExtent::Exact(bytes) => bytes,
            SegmentBindingRegionExtent::CurrentResource => {
                available.checked_sub(self.offset_bytes).ok_or_else(|| {
                    invalid_operation("segment resource shrank before its declared offset")
                })?
            }
        };
        checked_range(self.offset_bytes, length, available)?;
        if let Some(bound) = self.component_length {
            checked_range(self.request.offset_bytes, length, bound)?;
        }
        Ok(length)
    }
}

pub(super) struct CompiledSegmentDependency {
    pub source_resource_index: usize,
    pub destination_resource_index: usize,
    pub binding_index: usize,
    pub spec: SegmentBindingDependency,
}

pub(super) struct CompiledSegmentBindingNode {
    pub node_index: usize,
    pub declaration: SegmentBindingDeclaration,
    pub regions: Vec<CompiledSegmentRegion>,
    pub dependencies: Vec<CompiledSegmentDependency>,
    pub persistent_preserve: bool,
    /// Full captured node closure, including resources not exported as patches.
    pub resource_indices: Vec<usize>,
}

pub(crate) struct CompiledSegmentBindingRecipe {
    pub(crate) program_id: DeviceReusableExecutionProgramId,
    pub(super) nodes: Vec<CompiledSegmentBindingNode>,
    pub(super) resources: Vec<CompiledSegmentResource>,
    /// Sorted, unique indices into the exact Plan's static allocations.
    pub(super) plan_slots: Vec<usize>,
}

impl CompiledSegmentBindingRecipe {
    pub(super) fn compile(
        resolved: &dyn ExecutablePlanView,
        program_id: DeviceReusableExecutionProgramId,
        declarations: Vec<(usize, SegmentBindingDeclaration)>,
    ) -> Result<Self, VNextError> {
        let plan = resolved.execution_plan();
        if declarations.is_empty()
            || program_id.plan_hash() != plan.plan_hash()
            || program_id.runtime_implementation_fingerprint()
                != resolved.device().runtime_implementation_fingerprint
            || declarations.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(invalid_operation(
                "segment declarations differ from their ordered Plan/program",
            ));
        }
        let memory = plan.payload().memory();
        let mut resources = Vec::new();
        let mut indices = BTreeMap::new();
        let mut nodes = Vec::with_capacity(declarations.len());
        for (node_index, declaration) in declarations {
            let node = plan
                .payload()
                .nodes()
                .get(node_index)
                .ok_or_else(|| invalid_operation("segment node is outside its Plan"))?;
            let provider = resolved
                .capabilities()
                .providers_for(node.operation_id())?
                .iter()
                .find(|p| p.provider_id() == node.selection().selected_provider())
                .ok_or_else(|| {
                    invalid_operation("segment provider is absent from its exact catalog")
                })?;
            // Preserve the existing cold provider/operation/workspace schema
            // validation; no live invocation or authority is created here.
            PreparedOperationDispatchBinding::prepare(
                resolved,
                provider,
                node.id(),
                ReusableBindingResources::All,
            )?;
            let mut required = node
                .values()
                .iter()
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .collect::<BTreeSet<_>>();
            required.extend(node.scratch_resource().cloned());
            required.extend(node.binding_resource().cloned());
            required.extend(node.persistent_resource().cloned());
            let mut resource_indices = Vec::with_capacity(required.len());
            for resource in required {
                resource_indices.push(resolve_resource(
                    memory,
                    &resource,
                    &mut resources,
                    &mut indices,
                )?);
            }
            // Compile the complete value/workspace contract, including values
            // never exported in the provider's patch table. The fresh path
            // checks the aggregated minimum against each current view.
            for binding in node.values() {
                for component in binding.storage().components() {
                    let index = indices[component.resource_id()];
                    let resource = &resources[index];
                    let meta = metadata(memory, resource)?;
                    let demand = match resource.source {
                        CompiledSegmentResourceSource::PlanStatic { .. } => None,
                        CompiledSegmentResourceSource::Dynamic { descriptor_index } => {
                            Some(memory.dynamic_descriptors()[descriptor_index].demand())
                        }
                    };
                    let descriptor = BufferDescriptor {
                        resource_id: resource.resource_id.clone(),
                        size_bytes: meta.bytes,
                        alignment_bytes: meta.alignment_bytes,
                        usage: meta.usage,
                        element_type: meta.element_type,
                    };
                    let coverage = validate_value_binding_physical_coverage(
                        node.work(),
                        binding,
                        component,
                        &descriptor,
                        demand,
                        node.provider_resources().value_alignment_bytes(),
                    )?;
                    let minimum = match coverage {
                        ValueBindingPhysicalCoverage::CanonicalComponent => component
                            .offset_bytes()
                            .checked_add(component.length_bytes())
                            .ok_or_else(|| invalid_operation("segment component end overflows"))?,
                        ValueBindingPhysicalCoverage::RuntimeTokenView => {
                            component.length_bytes()
                                / node
                                    .work()
                                    .token_projection(binding.role(), binding.ordinal())
                                    .expect("validated token projection")
                                    .canonical_extent()
                        }
                    };
                    resources[index].required_minimum_bytes =
                        resources[index].required_minimum_bytes.max(minimum);
                }
            }
            for (resource, requirement, usage) in [
                (
                    node.scratch_resource(),
                    node.provider_resources().scratch(),
                    BufferUsage::Scratch,
                ),
                (
                    node.binding_resource(),
                    node.provider_resources().binding(),
                    BufferUsage::Binding,
                ),
                (
                    node.persistent_resource(),
                    node.provider_resources().persistent(),
                    BufferUsage::Persistent,
                ),
            ] {
                match (resource, requirement) {
                    (None, None) => {}
                    (Some(resource), Some(requirement)) => {
                        let index = indices[resource];
                        let meta = metadata(memory, &resources[index])?;
                        let minimum = requirement.minimum_bytes()?;
                        if meta.usage != usage
                            || meta.element_type != ElementType::U8
                            || meta.bytes < minimum
                            || meta.alignment_bytes < requirement.alignment_bytes()
                            || !meta
                                .alignment_bytes
                                .is_multiple_of(requirement.alignment_bytes())
                        {
                            return Err(invalid_operation(
                                "segment workspace descriptor is invalid",
                            ));
                        }
                        resources[index].required_minimum_bytes =
                            resources[index].required_minimum_bytes.max(minimum);
                    }
                    _ => return Err(invalid_operation("segment workspace presence differs")),
                }
            }
            let persistent_preserve = node.provider_resources().persistent().is_some_and(|p| {
                p.scope() == ProviderWorkspaceScope::Plan
                    && p.reuse_policy() == ProviderWorkspaceReusePolicy::Preserve
            });
            let mut regions = Vec::with_capacity(declaration.regions().len());
            for request in declaration.regions() {
                let (resource, component_offset, mut component_length, whole_state) = match &request
                    .selector
                {
                    SegmentBindingRegionSelector::Value {
                        role,
                        ordinal,
                        component,
                    } => {
                        let binding = node
                            .values()
                            .iter()
                            .find(|binding| {
                                binding.role() == *role && binding.ordinal() == *ordinal
                            })
                            .ok_or_else(|| {
                                invalid_operation(
                                    "segment value selector is absent from its Plan node",
                                )
                            })?;
                        let candidates = binding.storage().components();
                        let selected = match component {
                            Some(id) => candidates.iter().find(|c| c.component_id() == Some(id)),
                            None => (candidates.len() == 1).then(|| &candidates[0]),
                        }
                        .ok_or_else(|| {
                            invalid_operation("segment component selector is absent or ambiguous")
                        })?;
                        if selected.element_type() != request.element_type {
                            return Err(invalid_operation(
                                "segment component element type differs",
                            ));
                        }
                        (
                            selected.resource_id(),
                            selected.offset_bytes(),
                            Some(selected.length_bytes()),
                            candidates.len() == 1
                                && selected.offset_bytes() == 0
                                && binding.usage() == BufferUsage::State,
                        )
                    }
                    SegmentBindingRegionSelector::Scratch => (
                        node.scratch_resource()
                            .ok_or_else(|| invalid_operation("segment scratch is absent"))?,
                        0,
                        None,
                        false,
                    ),
                    SegmentBindingRegionSelector::Persistent => (
                        node.persistent_resource().ok_or_else(|| {
                            invalid_operation("segment persistent allocation is absent")
                        })?,
                        0,
                        None,
                        false,
                    ),
                    SegmentBindingRegionSelector::ProgramBinding => (
                        node.binding_resource().ok_or_else(|| {
                            invalid_operation("segment program binding is absent")
                        })?,
                        0,
                        None,
                        false,
                    ),
                };
                let resource_index = *indices.get(resource).ok_or_else(|| {
                    invalid_operation("segment selected resource is outside its node closure")
                })?;
                let metadata = metadata(memory, &resources[resource_index])?;
                if let Some(length) = component_length {
                    checked_range(component_offset, length, metadata.bytes)?;
                }
                if request.extent == SegmentBindingRegionExtent::CurrentResource {
                    if !whole_state
                        || request.offset_bytes != 0
                        || !matches!(
                            resources[resource_index].source,
                            CompiledSegmentResourceSource::Dynamic { .. }
                        )
                        || !matches!(
                            metadata.lifetime,
                            AllocationLifetime::Request | AllocationLifetime::Sequence
                        )
                    {
                        return Err(invalid_operation(
                            "current segment extent requires one whole dynamic state resource",
                        ));
                    }
                    component_length = None;
                }
                let offset_bytes = component_offset
                    .checked_add(request.offset_bytes)
                    .ok_or_else(|| invalid_operation("segment component offset overflows"))?;
                if !request.alignment_bytes.is_power_of_two()
                    || metadata.element_type != request.element_type
                    || metadata.alignment_bytes < request.alignment_bytes
                    || !offset_bytes.is_multiple_of(request.alignment_bytes)
                {
                    return Err(invalid_operation(
                        "segment region type or alignment differs from its Plan resource",
                    ));
                }
                match request.extent {
                    SegmentBindingRegionExtent::Exact(length) => {
                        if let Some(bound) = component_length {
                            checked_range(request.offset_bytes, length, bound)?;
                        }
                        checked_range(offset_bytes, length, metadata.bytes)?;
                    }
                    SegmentBindingRegionExtent::CurrentResource => {
                        checked_range(offset_bytes, 1, metadata.bytes)?;
                        if let Some(bound) = component_length {
                            checked_range(request.offset_bytes, 1, bound)?;
                        }
                    }
                }
                regions.push(CompiledSegmentRegion {
                    resource_index,
                    offset_bytes,
                    component_length,
                    request: request.clone(),
                });
            }
            let mut dependencies = Vec::with_capacity(declaration.dependencies().len());
            for spec in declaration.dependencies() {
                if !persistent_preserve
                    || spec.validation_identity.is_empty()
                    || !spec.alignment_bytes.is_power_of_two()
                {
                    return Err(invalid_operation(
                        "segment dependency requires Plan Preserve storage and validation identity",
                    ));
                }
                let (binding_index, binding) = node
                    .values()
                    .iter()
                    .enumerate()
                    .find(|(_, b)| {
                        b.role() == ResolvedValueRole::Input && b.ordinal() == spec.input_ordinal
                    })
                    .ok_or_else(|| {
                        invalid_operation("retained dependency weight input is absent")
                    })?;
                if binding.usage() != BufferUsage::Weights || binding.access() != TensorAccess::Read
                {
                    return Err(invalid_operation(
                        "retained dependency source must be a read-only weight input",
                    ));
                }
                let component = binding
                    .storage()
                    .components()
                    .iter()
                    .find(|c| c.component_id() == Some(&spec.component_id))
                    .ok_or_else(|| {
                        invalid_operation("retained dependency physical weight component is absent")
                    })?;
                checked_range(
                    spec.source_offset_bytes,
                    spec.source_length_bytes,
                    component.length_bytes(),
                )?;
                let source_offset = component
                    .offset_bytes()
                    .checked_add(spec.source_offset_bytes)
                    .ok_or_else(|| {
                        invalid_operation("retained dependency weight offset overflows")
                    })?;
                let destination = node.persistent_resource().ok_or_else(|| {
                    invalid_operation("retained dependency persistent view is absent")
                })?;
                let source_resource_index =
                    *indices.get(component.resource_id()).ok_or_else(|| {
                        invalid_operation("segment dependency weight is outside its node closure")
                    })?;
                let destination_resource_index = *indices.get(destination).ok_or_else(|| {
                    invalid_operation("segment dependency destination is outside its node closure")
                })?;
                let source = metadata(memory, &resources[source_resource_index])?;
                let destination = metadata(memory, &resources[destination_resource_index])?;
                if source.lifetime != AllocationLifetime::Plan
                    || source.usage != BufferUsage::Weights
                    || destination.lifetime != AllocationLifetime::Plan
                    || destination.usage != BufferUsage::Persistent
                    || destination.element_type != ElementType::U8
                    || destination.alignment_bytes < spec.alignment_bytes
                    || !spec
                        .persistent_offset_bytes
                        .is_multiple_of(spec.alignment_bytes)
                    || !spec
                        .persistent_length_bytes
                        .is_multiple_of(spec.alignment_bytes)
                    || source_resource_index == destination_resource_index
                {
                    return Err(invalid_operation("retained dependency source/destination ownership, type, or alignment differs"));
                }
                checked_range(source_offset, spec.source_length_bytes, source.bytes)?;
                checked_range(
                    spec.persistent_offset_bytes,
                    spec.persistent_length_bytes,
                    destination.bytes,
                )?;
                dependencies.push(CompiledSegmentDependency {
                    source_resource_index,
                    destination_resource_index,
                    binding_index,
                    spec: spec.clone(),
                });
            }
            nodes.push(CompiledSegmentBindingNode {
                node_index,
                declaration,
                regions,
                dependencies,
                persistent_preserve,
                resource_indices,
            });
        }
        let plan_slots = resources
            .iter()
            .filter_map(|r| match r.source {
                CompiledSegmentResourceSource::PlanStatic { slot_index } => Some(slot_index),
                CompiledSegmentResourceSource::Dynamic { .. } => None,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            program_id,
            nodes,
            resources,
            plan_slots,
        })
    }
}

fn resolve_resource(
    memory: &MemoryPlan,
    resource_id: &ResourceId,
    resources: &mut Vec<CompiledSegmentResource>,
    indices: &mut BTreeMap<ResourceId, usize>,
) -> Result<usize, VNextError> {
    if let Some(index) = indices.get(resource_id) {
        return Ok(*index);
    }
    let source = match (
        memory
            .static_allocations()
            .binary_search_by(|a| a.resource_id().cmp(resource_id)),
        memory
            .dynamic_descriptors()
            .binary_search_by(|d| d.base_resource_id().cmp(resource_id)),
    ) {
        (Ok(slot_index), Err(_)) => CompiledSegmentResourceSource::PlanStatic { slot_index },
        (Err(_), Ok(descriptor_index)) => {
            CompiledSegmentResourceSource::Dynamic { descriptor_index }
        }
        _ => {
            return Err(invalid_operation(
                "segment resource is absent or ambiguous in its Plan memory",
            ))
        }
    };
    let index = resources.len();
    resources.push(CompiledSegmentResource {
        resource_id: resource_id.clone(),
        source,
        required_minimum_bytes: 0,
    });
    indices.insert(resource_id.clone(), index);
    Ok(index)
}

struct Metadata {
    bytes: u64,
    element_type: ElementType,
    alignment_bytes: u64,
    usage: BufferUsage,
    lifetime: AllocationLifetime,
}

fn metadata(
    memory: &MemoryPlan,
    resource: &CompiledSegmentResource,
) -> Result<Metadata, VNextError> {
    Ok(match resource.source {
        CompiledSegmentResourceSource::PlanStatic { slot_index } => {
            let a = &memory.static_allocations()[slot_index];
            Metadata {
                bytes: a.size_bytes(),
                element_type: a.element_type(),
                alignment_bytes: a.alignment_bytes(),
                usage: a.usage(),
                lifetime: a.lifetime(),
            }
        }
        CompiledSegmentResourceSource::Dynamic { descriptor_index } => {
            let d = &memory.dynamic_descriptors()[descriptor_index];
            Metadata {
                bytes: d.theoretical_maximum_request_bytes()?,
                element_type: d.element_type(),
                alignment_bytes: d.alignment_bytes(),
                usage: d.usage(),
                lifetime: d.lifetime(),
            }
        }
    })
}
