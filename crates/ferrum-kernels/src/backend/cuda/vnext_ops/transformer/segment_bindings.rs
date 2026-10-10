//! The single CUDA interpreter for a resident segment's closed binding schema.
//! No provider callback or BatchedOperationInvocation is constructed here.
use super::super::native_blocks::upstream_linear::WeightValidation;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntimeError,
    CudaProgramBindingWrite, CudaSegmentOwnerBuilder, CudaSegmentOwnerTable, CudaSegmentRange,
};
use ferrum_interfaces::vnext::{
    DeviceBatchingForm, ElementType, EncodedReusableExecutionBindings, EncodedSegmentBindingNode,
    PreparedSegmentBindingNode, PreparedSegmentBindingPatch, ProgramBindingNodeBinding,
    ResolvedValueRole, SegmentBindingDeclaration, SegmentBindingDependency,
    SegmentBindingOwnerViewMode, SegmentBindingRegionExtent, SegmentBindingRegionRequest,
    SegmentBindingRegionSelector,
};
use std::sync::Arc;

pub(in crate::backend::cuda::vnext_ops) struct ValidationRecipe {
    pub state: Arc<WeightValidation>,
    pub declaration: SegmentBindingDependency,
    pub weight_element_type: ElementType,
}

pub(in crate::backend::cuda::vnext_ops) enum DynamicRecipe {
    PlanDependencies,
    GatedDelta(super::attention::segment_bindings::GatedDeltaRecipe),
    Causal(super::causal_attention::segment_bindings::CausalRecipe),
}

struct CudaSegmentBindingRecipe {
    dynamic: DynamicRecipe,
    validations: Vec<ValidationRecipe>,
    dependency_region_start: usize,
}

pub(in crate::backend::cuda::vnext_ops) fn declaration(
    dynamic: DynamicRecipe,
    mut regions: Vec<SegmentBindingRegionRequest>,
    validations: Vec<ValidationRecipe>,
) -> Result<SegmentBindingDeclaration, String> {
    let dependency_region_start = regions.len();
    for validation in &validations {
        let d = &validation.declaration;
        regions.push(SegmentBindingRegionRequest {
            selector: SegmentBindingRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: d.input_ordinal,
                component: Some(d.component_id.clone()),
            },
            offset_bytes: d.source_offset_bytes,
            extent: SegmentBindingRegionExtent::Exact(d.source_length_bytes),
            element_type: validation.weight_element_type,
            alignment_bytes: d.alignment_bytes,
        });
        regions.push(SegmentBindingRegionRequest {
            selector: SegmentBindingRegionSelector::Persistent,
            offset_bytes: d.persistent_offset_bytes,
            extent: SegmentBindingRegionExtent::Exact(d.persistent_length_bytes),
            element_type: ElementType::U8,
            alignment_bytes: d.alignment_bytes,
        });
    }
    let dependencies = validations.iter().map(|v| v.declaration.clone()).collect();
    SegmentBindingDeclaration::new(
        regions,
        dependencies,
        Arc::new(CudaSegmentBindingRecipe {
            dynamic,
            validations,
            dependency_region_start,
        }),
    )
    .map_err(|e| e.to_string())
}

pub(in crate::backend::cuda::vnext_ops) fn contiguous_region(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    participant: usize,
    region: usize,
) -> Result<CudaBufferRegion, String> {
    let view = node
        .region(participant, region)
        .map_err(|e| e.to_string())?;
    let mut physicals = view.physical_regions();
    if physicals.len() != 1 {
        return Err("segment binding requires one contiguous physical region".into());
    }
    let physical = physicals
        .next()
        .ok_or("segment contiguous physical region is absent")?;
    let (buffer, range, retention) = physical.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())
}

pub(in crate::backend::cuda::vnext_ops) fn shared_region(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    region: usize,
) -> Result<CudaBufferRegion, String> {
    let first = contiguous_region(node, 0, region)?;
    for participant in 1..node.participant_count() {
        let next = contiguous_region(node, participant, region)?;
        if !super::same_physical_region(&first, &next) {
            return Err(
                "segment binding shared physical region differs across participants".into(),
            );
        }
    }
    Ok(first)
}

pub(in crate::backend::cuda::vnext_ops) fn shared_borrowed_region(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    region: usize,
) -> Result<CudaBufferRegion, String> {
    let first_view = node.region(0, region).map_err(|e| e.to_string())?;
    let mut first_physical = first_view.physical_regions();
    if first_physical.len() != 1 {
        return Err("segment shared region is not contiguous".into());
    }
    let first_physical = first_physical
        .next()
        .ok_or("segment shared region is absent")?;
    let (buffer, range, retention) = first_physical.borrowed_buffer_and_physical_range();
    let first = buffer
        .borrowed_region(range, retention)
        .map_err(|e| e.to_string())?;
    for participant in 1..node.participant_count() {
        let next_view = node
            .region(participant, region)
            .map_err(|e| e.to_string())?;
        let mut next_physical = next_view.physical_regions();
        if next_physical.len() != 1 {
            return Err("segment shared region is not contiguous".into());
        }
        let next_physical = next_physical
            .next()
            .ok_or("segment shared region is absent")?;
        let (buffer, range, retention) = next_physical.borrowed_buffer_and_physical_range();
        let next = buffer
            .borrowed_region(range, retention)
            .map_err(|e| e.to_string())?;
        // Shared physical storage need not have one opaque wrapper across
        // participants. Preserve the ordinary path's predicate and select P0;
        // the same-wave oracle compares that selected retention exactly.
        if first.device_ptr() != next.device_ptr()
            || first.length_bytes() != next.length_bytes()
            || first.element_type() != next.element_type()
        {
            return Err(
                "segment binding shared physical region differs across participants".into(),
            );
        }
    }
    Ok(first.to_owned())
}

pub(in crate::backend::cuda::vnext_ops) fn indexed_contiguous_region(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    participant: usize,
    region: usize,
    owners: &mut CudaSegmentOwnerBuilder<'_>,
) -> Result<CudaSegmentRange, String> {
    let view = node
        .region(participant, region)
        .map_err(|e| e.to_string())?;
    let mut physicals = view.physical_regions();
    if physicals.len() != 1 {
        return Err("segment binding requires one contiguous physical region".into());
    }
    let physical = physicals
        .next()
        .ok_or("segment contiguous physical region is absent")?;
    let index = physical
        .owner_index()
        .ok_or("indexed segment region has no owner index")?;
    let (buffer, range, retention) = physical.borrowed_buffer_and_physical_range();
    owners
        .retain(index, buffer, range, retention)
        .map_err(|e| e.to_string())
}

/// No command holds the mutable builder. All nodes prepare before its owner
/// table is frozen, then receive submission-owned references in original order.
pub(in crate::backend::cuda::vnext_ops) struct PendingBinding {
    pub operation: &'static str,
    pub binding: ProgramBindingNodeBinding,
    pub destination: CudaBufferRegion,
    pub writes: Vec<CudaProgramBindingWrite>,
    pub ranges: Vec<CudaSegmentRange>,
    pub numerical_status: Option<(Vec<u64>, u32)>,
    pub participants: u32,
    pub tokens: u64,
}

impl PendingBinding {
    fn finish(
        self,
        owners: Arc<CudaSegmentOwnerTable>,
    ) -> Result<CudaDeviceCommand, CudaDeviceRuntimeError> {
        let status_source = self
            .numerical_status
            .as_ref()
            .map(|_| self.destination.clone());
        let command = CudaDeviceCommand::indexed_program_binding_patch(
            self.operation,
            self.binding,
            self.destination,
            self.writes,
            owners,
            self.ranges,
        )?;
        let command = match (self.numerical_status, status_source) {
            (Some((offsets, mask)), Some(source)) => {
                command.with_numerical_status(source, offsets, mask, true)?
            }
            (None, None) => command,
            _ => {
                return Err(CudaDeviceRuntimeError::contract(
                    "segment status source disappeared",
                ))
            }
        };
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            self.participants,
            self.tokens,
            0,
            u64::from(self.participants),
        )
    }
}

pub(crate) fn encode(
    patch: PreparedSegmentBindingPatch<'_, CudaDeviceBuffer>,
) -> Result<Option<Vec<EncodedSegmentBindingNode<CudaDeviceCommand>>>, CudaDeviceRuntimeError> {
    // Unsupported declarations fall back before any dependency is consumed.
    if patch.nodes().iter().any(|node| {
        node.declaration()
            .provider_state()
            .downcast_ref::<CudaSegmentBindingRecipe>()
            .is_none()
    }) {
        return Ok(None);
    }
    if patch.owner_view_mode() == SegmentBindingOwnerViewMode::Legacy {
        return encode_legacy(patch);
    }
    let mut pending_nodes = Vec::with_capacity(patch.nodes().len());
    let (owner_views, nodes) = patch.into_parts();
    let mut owners = CudaSegmentOwnerBuilder::new(owner_views);
    for mut node in nodes {
        let recipe = node
            .declaration()
            .provider_state()
            .downcast_ref::<CudaSegmentBindingRecipe>()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("CUDA segment recipe disappeared"))?;
        let pending = match &recipe.dynamic {
            DynamicRecipe::PlanDependencies => Ok(None),
            DynamicRecipe::GatedDelta(recipe) => {
                super::attention::segment_bindings::prepare_indexed(&node, recipe, &mut owners)
                    .map(Some)
            }
            DynamicRecipe::Causal(recipe) => {
                super::causal_attention::segment_bindings::prepare_indexed(
                    &node,
                    recipe,
                    &mut owners,
                )
                .map(Some)
            }
        }
        .map_err(CudaDeviceRuntimeError::contract)?;
        let mut bindings = EncodedReusableExecutionBindings::empty();
        let mut commands = Vec::with_capacity(recipe.validations.len());
        for (index, validation) in recipe.validations.iter().enumerate() {
            let weight = shared_borrowed_region(&node, recipe.dependency_region_start + index * 2)
                .map_err(CudaDeviceRuntimeError::contract)?;
            let flag =
                shared_borrowed_region(&node, recipe.dependency_region_start + index * 2 + 1)
                    .map_err(CudaDeviceRuntimeError::contract)?;
            validation.state.validate_segment_owners(&weight, &flag)?;
            commands.push(validation.state.dynamic_command()?);
        }
        for (index, command) in commands.into_iter().enumerate() {
            let authority = node
                .take_dependency(index)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
            bindings = bindings.with_retained_plan_dependency(authority.encode(command));
        }
        pending_nodes.push((node.node_index(), bindings, pending));
    }
    let owners = owners.finish();
    pending_nodes
        .into_iter()
        .map(|(node_index, mut bindings, pending)| {
            if let Some(pending) = pending {
                bindings = bindings.with_program_binding(pending.finish(Arc::clone(&owners))?);
            }
            Ok(EncodedSegmentBindingNode {
                node_index,
                bindings,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn encode_legacy(
    patch: PreparedSegmentBindingPatch<'_, CudaDeviceBuffer>,
) -> Result<Option<Vec<EncodedSegmentBindingNode<CudaDeviceCommand>>>, CudaDeviceRuntimeError> {
    let mut encoded = Vec::with_capacity(patch.nodes().len());
    for mut node in patch.into_nodes() {
        let recipe = node
            .declaration()
            .provider_state()
            .downcast_ref::<CudaSegmentBindingRecipe>()
            .expect("all CUDA segment declaration types were checked");
        let mut bindings = match &recipe.dynamic {
            DynamicRecipe::PlanDependencies => Ok(EncodedReusableExecutionBindings::empty()),
            DynamicRecipe::GatedDelta(recipe) => {
                super::attention::segment_bindings::encode(&node, recipe)
            }
            DynamicRecipe::Causal(recipe) => {
                super::causal_attention::segment_bindings::encode(&node, recipe)
            }
        }
        .map_err(CudaDeviceRuntimeError::contract)?;
        // Dependencies preserve their original node order. Their source and
        // bank are fresh core facts; only immutable Plan validation state is cold.
        let mut commands = Vec::with_capacity(recipe.validations.len());
        for (index, validation) in recipe.validations.iter().enumerate() {
            let weight = shared_region(&node, recipe.dependency_region_start + index * 2)
                .map_err(CudaDeviceRuntimeError::contract)?;
            let flag = shared_region(&node, recipe.dependency_region_start + index * 2 + 1)
                .map_err(CudaDeviceRuntimeError::contract)?;
            validation.state.validate_segment_owners(&weight, &flag)?;
            commands.push(validation.state.dynamic_command()?);
        }
        for (index, command) in commands.into_iter().enumerate() {
            let authority = node
                .take_dependency(index)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
            bindings = bindings.with_retained_plan_dependency(authority.encode(command));
        }
        encoded.push(EncodedSegmentBindingNode {
            node_index: node.node_index(),
            bindings,
        });
    }
    Ok(Some(encoded))
}
