//! The compiled rule interpreter over fresh waves and cold Sequence tables.
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaBindingProgramSink, CudaPhysicalOwnerBuilder, CudaPhysicalRowWindow,
};
use ferrum_runtime::{
    OperationBufferStorageKind, PreparedPhysicalSegmentNodeView, PreparedPhysicalSegmentWave,
};

pub(crate) fn encode_program(
    program: &(dyn std::any::Any + Send + Sync),
    wave: &mut PreparedPhysicalSegmentWave<'_, CudaDeviceBuffer>,
) -> Result<
    (
        Vec<EncodedSegmentBindingNode<CudaDeviceCommand>>,
        CudaBindingProgramSink,
        Arc<CudaSegmentOwnerTable>,
    ),
    CudaDeviceRuntimeError,
> {
    let invalid = CudaDeviceRuntimeError::contract;
    let program = program
        .downcast_ref::<CudaSegmentBindingProgram>()
        .ok_or_else(|| invalid("physical CUDA binding program has another provider type"))?;
    if program.nodes.len() != wave.node_count() {
        return Err(invalid(
            "physical CUDA node count differs from its fresh wave",
        ));
    }
    let mut owners = CudaPhysicalOwnerBuilder::new();
    let mut sink = CudaBindingProgramSink::new();
    let mut encoded = Vec::with_capacity(program.nodes.len());
    for (ordinal, (node_index, recipe)) in program.nodes.iter().enumerate() {
        let node = wave
            .node(ordinal)
            .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
        if node.node_index() != *node_index {
            return Err(invalid(
                "physical CUDA node order differs from its fresh wave",
            ));
        }
        match &recipe.dynamic {
            DynamicRecipe::PlanDependencies => Ok(()),
            DynamicRecipe::GatedDelta(recipe) => {
                super::super::attention::segment_bindings::write_physical_program(
                    &node,
                    recipe,
                    &mut owners,
                    &mut sink,
                )
            }
            DynamicRecipe::Causal(recipe) => {
                super::super::causal_attention::segment_bindings::write_physical_program(
                    &node,
                    recipe,
                    &mut owners,
                    &mut sink,
                )
            }
        }
        .map_err(CudaDeviceRuntimeError::contract)?;
        let mut commands = Vec::with_capacity(recipe.validations.len());
        for (index, validation) in recipe.validations.iter().enumerate() {
            let weight = shared_region(&node, recipe.dependency_region_start + index * 2)
                .map_err(CudaDeviceRuntimeError::contract)?;
            let flag = shared_region(&node, recipe.dependency_region_start + index * 2 + 1)
                .map_err(CudaDeviceRuntimeError::contract)?;
            validation.state.validate_segment_owners(&weight, &flag)?;
            commands.push(validation.state.dynamic_command()?);
        }
        let mut bindings = EncodedReusableExecutionBindings::empty();
        for (index, command) in commands.into_iter().enumerate() {
            let authority = wave
                .take_dependency(ordinal, index)
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
            bindings = bindings.with_retained_plan_dependency(authority.encode(command));
        }
        encoded.push(EncodedSegmentBindingNode {
            node_index: *node_index,
            bindings,
        });
    }
    Ok((encoded, sink, owners.finish()))
}

pub(in crate::backend::cuda::vnext_ops) fn shared_region(
    node: &PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    region: usize,
) -> Result<CudaBufferRegion, String> {
    let first_view = node.region(0, region).map_err(|e| e.to_string())?;
    let mut physicals = first_view.physical_regions();
    let physical = physicals.next().ok_or("physical shared region is absent")?;
    if physicals.next().is_some() {
        return Err("physical shared region is not contiguous".into());
    }
    let (buffer, range, retention) = physical.borrowed_buffer_and_physical_range();
    let first = buffer
        .borrowed_region(range, retention)
        .map_err(|e| e.to_string())?;
    for participant in 1..node.participant_count() {
        let view = node
            .region(participant, region)
            .map_err(|e| e.to_string())?;
        let mut physicals = view.physical_regions();
        let physical = physicals.next().ok_or("physical shared region is absent")?;
        if physicals.next().is_some() {
            return Err("physical shared region is not contiguous".into());
        }
        let (buffer, range, retention) = physical.borrowed_buffer_and_physical_range();
        let next = buffer
            .borrowed_region(range, retention)
            .map_err(|e| e.to_string())?;
        if first.device_ptr() != next.device_ptr()
            || first.length_bytes() != next.length_bytes()
            || first.element_type() != next.element_type()
        {
            return Err("physical shared region differs across participants".into());
        }
    }
    Ok(first.to_owned())
}

pub(in crate::backend::cuda::vnext_ops) fn sequence_window(
    node: &PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    participant: usize,
    declaration: usize,
    expected: u64,
    element: ElementType,
    page_bytes: Option<u64>,
    owners: &mut CudaPhysicalOwnerBuilder,
) -> Result<CudaPhysicalRowWindow, String> {
    let view = node
        .region(participant, declaration)
        .map_err(|e| e.to_string())?;
    let storage = if page_bytes.is_some() {
        OperationBufferStorageKind::DynamicPaged
    } else {
        OperationBufferStorageKind::DynamicContiguous
    };
    if view.storage_kind() != storage
        || view.descriptor().element_type != element
        || view.length_bytes() != expected
    {
        return Err("physical Sequence window differs from the current admitted geometry".into());
    }
    let table = view
        .sequence_table()
        .ok_or("physical Sequence table is absent")?;
    let ordinal = view
        .resource_ordinal()
        .ok_or("physical Sequence resource ordinal is absent")?;
    owners
        .window(
            table,
            ordinal,
            view.offset_bytes(),
            view.length_bytes(),
            page_bytes,
        )
        .map_err(|e| e.to_string())
}

/// Resident packets may retain the whole paged capacity only through the
/// separate Core-issued view. Ordinary operation windows remain exact.
pub(in crate::backend::cuda::vnext_ops) fn resident_sequence_capacity_window(
    node: &PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    participant: usize,
    declaration: usize,
    expected: u64,
    element: ElementType,
    page_bytes: u64,
    owners: &mut CudaPhysicalOwnerBuilder,
) -> Result<CudaPhysicalRowWindow, String> {
    let current = node
        .region(participant, declaration)
        .map_err(|e| e.to_string())?;
    let capacity = node
        .resident_sequence_capacity_region(participant, declaration)
        .map_err(|e| e.to_string())?;
    let table = current
        .sequence_table()
        .ok_or("resident current Sequence table is absent")?;
    let capacity_table = capacity
        .sequence_table()
        .ok_or("resident capacity Sequence table is absent")?;
    let ordinal = current
        .resource_ordinal()
        .ok_or("resident Sequence resource ordinal is absent")?;
    if current.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || capacity.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || current.descriptor().element_type != element
        || capacity.descriptor().element_type != element
        || current.descriptor().resource_id != capacity.descriptor().resource_id
        || current.descriptor().usage != capacity.descriptor().usage
        || current.descriptor().alignment_bytes != capacity.descriptor().alignment_bytes
        || current.length_bytes() != expected
        || expected == 0
        || !Arc::ptr_eq(table, capacity_table)
        || capacity.resource_ordinal() != Some(ordinal)
        || current.offset_bytes() != 0
        || capacity.offset_bytes() != 0
        || capacity.length_bytes() < expected
        || page_bytes == 0
        || !expected.is_multiple_of(page_bytes)
        || !capacity.length_bytes().is_multiple_of(page_bytes)
    {
        return Err("resident capacity differs from its current authorized paged window".into());
    }
    owners
        .window(
            capacity_table,
            ordinal,
            capacity.offset_bytes(),
            capacity.length_bytes(),
            Some(page_bytes),
        )
        .map_err(|e| e.to_string())
}
