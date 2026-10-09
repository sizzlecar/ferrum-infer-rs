//! The single CUDA interpreter for a resident segment's closed binding schema.
//! No provider callback or BatchedOperationInvocation is constructed here.
use super::super::native_blocks::upstream_linear::WeightValidation;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntimeError,
};
use ferrum_interfaces::vnext::{
    ElementType, EncodedReusableExecutionBindings, EncodedSegmentBindingNode,
    PreparedSegmentBindingNode, PreparedSegmentBindingPatch, ResolvedValueRole,
    SegmentBindingDeclaration, SegmentBindingDependency, SegmentBindingRegionExtent,
    SegmentBindingRegionRequest, SegmentBindingRegionSelector,
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
    let [physical] = view.physical_regions() else {
        return Err("segment binding requires one contiguous physical region".into());
    };
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

pub(crate) fn encode(
    patch: PreparedSegmentBindingPatch<'_, CudaDeviceBuffer>,
) -> Result<Option<Vec<EncodedSegmentBindingNode<CudaDeviceCommand>>>, CudaDeviceRuntimeError> {
    // Unsupported declarations fall back before any dependency is consumed.
    if patch.nodes().iter().any(|node| {
        node.declaration()
            .state::<CudaSegmentBindingRecipe>()
            .is_none()
    }) {
        return Ok(None);
    }
    let mut encoded = Vec::with_capacity(patch.nodes().len());
    for mut node in patch.into_nodes() {
        let recipe = node
            .declaration()
            .state::<CudaSegmentBindingRecipe>()
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
