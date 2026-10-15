//! Current KV/scale page-table rule. Physical extents and token frontiers are
//! read from fresh segment facts, never retained from the cold decode wave.
use super::super::segment_bindings::{self as segment, DynamicRecipe, ValidationRecipe};
use super::*;
use ferrum_interfaces::vnext::{
    PreparedSegmentBindingNode, SegmentBindingDeclaration, SegmentBindingRegionExtent,
    SegmentBindingRegionRequest, SegmentBindingRegionSelector,
};

#[derive(Clone)]
pub(in crate::backend::cuda::vnext_ops) struct CausalRecipe {
    shape: CausalAttentionShape,
    participants: usize,
    upload_strategy: ProgramBindingUploadStrategy,
}

impl CausalRecipe {
    pub(in crate::backend::cuda::vnext_ops) fn supports_program(&self) -> bool {
        self.upload_strategy == ProgramBindingUploadStrategy::CompactScatter
    }
}

/// Freeze physical page/address shape; each use still supplies its current
/// logical frontier. A changed page shape is a valid pre-enqueue miss.
pub(crate) fn resident_control_rules(
    node: &ferrum_interfaces::vnext::PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
) -> Result<
    Option<Vec<crate::backend::cuda::vnext_runtime::ResidentControlRule>>,
    CudaDeviceRuntimeError,
> {
    use crate::backend::cuda::vnext_runtime::ResidentControlRule;
    let invalid = |reason: &str| CudaDeviceRuntimeError::contract(reason);
    // Keep the numerical-status snapshot path unchanged for unsupported KV.
    if recipe.shape.int8_kv || !recipe.supports_program() {
        return Ok(None);
    }
    let mut rules = Vec::with_capacity(recipe.participants);
    for (participant, token) in node
        .work_shape()
        .participant_token_ranges()
        .iter()
        .enumerate()
    {
        let shape = recipe.shape;
        let source = token.source_token_range();
        let bytes = shape
            .physical_state_bytes_for_source_frontier(source.end, token.full_input_tokens())
            .map_err(CudaDeviceRuntimeError::contract)?;
        rules.push(ResidentControlRule {
            node_index: node.node_index(),
            write_index: participant,
            values: Box::new(move |work| {
                let Some(token) = work.participant_token_ranges().get(participant) else {
                    return Ok(None);
                };
                let source = token.source_token_range();
                let packed = token.immediate_token_range();
                if source.end > token.full_input_tokens()
                    || token.full_input_tokens() > shape.maximum_context_tokens
                {
                    return Err(invalid("resident causal frontier exceeds admitted context"));
                }
                let current_entries = shape
                    .table_entries(source.end)
                    .map_err(CudaDeviceRuntimeError::contract)?;
                let current_bytes = shape
                    .physical_state_bytes_for_source_frontier(source.end, token.full_input_tokens())
                    .map_err(CudaDeviceRuntimeError::contract)?;
                if current_bytes != bytes {
                    return Ok(None);
                }
                let value =
                    |v, label| checked_i32(v, label).map_err(CudaDeviceRuntimeError::contract);
                Ok(Some([
                    value(current_entries, "resident table entries")?,
                    value(source.start, "resident source position")?,
                    value(token.immediate_tokens(), "resident token count")?,
                    value(source.end, "resident sequence tokens")?,
                    value(packed.start, "resident packed token start")?,
                    0,
                ]))
            }),
        });
    }
    Ok(Some(rules))
}

pub(super) fn declare(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    shape: CausalAttentionShape,
    upload_strategy: ProgramBindingUploadStrategy,
    validations: Vec<ValidationRecipe>,
) -> Result<Option<SegmentBindingDeclaration>, String> {
    if invocation.program_binding().is_none() {
        return Ok(None);
    }
    let layout = BindingLayout::new(shape, invocation.participants().len())?;
    let workspace_bytes = align_up(layout.required_bytes, SCRATCH_ALIGNMENT)?;
    let mut regions = vec![
        SegmentBindingRegionRequest {
            selector: SegmentBindingRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: 8,
                component: None,
            },
            offset_bytes: 0,
            extent: SegmentBindingRegionExtent::CurrentResource,
            element_type: shape.kv_element_type(),
            alignment_bytes: 1,
        },
        SegmentBindingRegionRequest {
            selector: SegmentBindingRegionSelector::ProgramBinding,
            offset_bytes: 0,
            // Match the provider workspace formula and ordinary encoder's
            // complete aligned region. Odd participant counts can leave a
            // tail after the logical rows; writes retain their live bounds.
            extent: SegmentBindingRegionExtent::Exact(workspace_bytes),
            element_type: ElementType::U8,
            alignment_bytes: 1,
        },
    ];
    if shape.int8_kv {
        regions.push(SegmentBindingRegionRequest {
            selector: SegmentBindingRegionSelector::Value {
                role: ResolvedValueRole::Input,
                ordinal: 9,
                component: None,
            },
            offset_bytes: 0,
            extent: SegmentBindingRegionExtent::CurrentResource,
            element_type: ElementType::F32,
            alignment_bytes: 4,
        });
    }
    segment::declaration(
        DynamicRecipe::Causal(CausalRecipe {
            shape,
            participants: invocation.participants().len(),
            upload_strategy,
        }),
        regions,
        validations,
    )
    .map(Some)
}

fn pages(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    participant: usize,
    declaration: usize,
    expected: u64,
    element: ElementType,
) -> Result<Vec<CudaBufferRegion>, String> {
    let view = node
        .region(participant, declaration)
        .map_err(|e| e.to_string())?;
    if view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().element_type != element
        || view.descriptor().size_bytes != expected
        || expected == 0
        || expected % VNEXT_KV_PAGE_BYTES != 0
    {
        return Err("segment causal state differs from its admitted paged view".into());
    }
    let capacity = usize::try_from(expected / VNEXT_KV_PAGE_BYTES)
        .map_err(|_| "segment causal page count exceeds usize")?;
    let mut pages = Vec::with_capacity(capacity);
    let mut logical = 0u64;
    for physical in view.physical_regions() {
        if physical.logical_offset_bytes() != logical
            || physical.length_bytes() == 0
            || physical.length_bytes() % VNEXT_KV_PAGE_BYTES != 0
        {
            return Err("segment causal pages lost block geometry".into());
        }
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let mut offset = 0u64;
        while offset < physical.length_bytes() {
            let start = range
                .start
                .checked_add(offset)
                .ok_or("segment causal page start overflows")?;
            let end = start
                .checked_add(VNEXT_KV_PAGE_BYTES)
                .ok_or("segment causal page end overflows")?;
            let page = buffer
                .retained_region(start..end, retention.clone())
                .map_err(|e| e.to_string())?;
            if page.length_bytes() != VNEXT_KV_PAGE_BYTES || page.element_type() != element {
                return Err("segment causal physical page differs from its contract".into());
            }
            pages.push(page);
            offset += VNEXT_KV_PAGE_BYTES;
        }
        logical = logical
            .checked_add(physical.length_bytes())
            .ok_or("segment causal logical coverage overflows")?;
    }
    if logical != expected || pages.is_empty() {
        return Err("segment causal pages do not cover current state".into());
    }
    Ok(pages)
}

pub(in crate::backend::cuda::vnext_ops) fn encode(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, String> {
    let shape = recipe.shape;
    if node.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("segment causal participant shape differs from cold geometry".into());
    }
    let layout = BindingLayout::new(shape, recipe.participants)?;
    let destination = segment::shared_region(node, 1)?;
    let kv_layout = shape.kv_layout()?;
    let maximum_pages = shape.maximum_pages()?;
    let maximum_entries = shape.table_entries(shape.maximum_context_tokens)?;
    let mut writes = Vec::with_capacity(recipe.participants);
    let mut retained = Vec::new();
    for (participant, token_range) in node
        .work_shape()
        .participant_token_ranges()
        .iter()
        .enumerate()
    {
        let source = token_range.source_token_range();
        let packed = token_range.immediate_token_range();
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("segment causal token range exceeds admitted context".into());
        }
        let state_pages = pages(
            node,
            participant,
            0,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
            shape.kv_element_type(),
        )?;
        let scale_pages = if shape.int8_kv {
            pages(
                node,
                participant,
                2,
                shape.scale_state_bytes(source.end)?,
                ElementType::F32,
            )?
        } else {
            Vec::new()
        };
        let page_count = u64::try_from(state_pages.len())
            .map_err(|_| "segment causal page count exceeds u64")?;
        let entries = shape.table_entries(source.end)?;
        if page_count > maximum_pages || entries > maximum_entries {
            return Err("segment causal page table exceeds admitted maximum".into());
        }
        let payload = binding_payload(
            kv_layout,
            checked_i32(entries, "causal attention address-table entry count")?,
            checked_i32(source.start, "causal attention source position")?,
            checked_i32(
                token_range.immediate_tokens(),
                "causal attention participant token count",
            )?,
            checked_i32(source.end, "causal attention sequence token count")?,
            checked_i32(packed.start, "causal attention packed token start")?,
            &state_pages,
            &scale_pages,
        )?;
        writes.push(
            super::super::CudaProgramBindingWrite::new(
                layout.binding_offset(participant)?,
                payload,
            )
            .map_err(|e| e.to_string())?,
        );
        retained.extend(state_pages);
        retained.extend(scale_pages);
    }
    prepare_causal_program_binding_writes(recipe.upload_strategy, &mut writes, layout.slot_bytes)?;
    let status = if shape.int8_kv {
        Some((
            destination.clone(),
            (0..recipe.participants)
                .map(|p| {
                    layout.binding_offset(p).and_then(|o| {
                        o.checked_add(BINDING_NUMERICAL_STATUS_OFFSET)
                            .ok_or("segment status offset overflows".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    } else {
        None
    };
    let participants =
        u32::try_from(recipe.participants).map_err(|_| "segment participant count exceeds u32")?;
    let binding = node
        .program_binding()
        .ok_or("segment causal binding slot is absent")?
        .clone();
    let command = CudaDeviceCommand::program_binding_patch(
        "vnext_causal_paged_attention_bindings",
        binding,
        destination,
        writes,
        retained,
    )
    .and_then(|c| match status {
        Some((source, offsets)) => {
            c.with_numerical_status(source, offsets, INT8_NUMERICAL_FAILURE_MASK, true)
        }
        None => Ok(c),
    })
    .and_then(|c| {
        c.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participants,
            node.work_shape().immediate_tokens(),
            0,
            u64::from(participants),
        )
    })
    .map_err(|e| e.to_string())?;
    Ok(EncodedReusableExecutionBindings::empty().with_program_binding(command))
}

fn indexed_pages(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    participant: usize,
    declaration: usize,
    expected: u64,
    element: ElementType,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaSegmentOwnerBuilder<'_>,
) -> Result<Vec<crate::backend::cuda::vnext_runtime::CudaSegmentRange>, String> {
    let view = node
        .region(participant, declaration)
        .map_err(|e| e.to_string())?;
    if view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().element_type != element
        || view.descriptor().size_bytes != expected
        || expected == 0
        || expected % VNEXT_KV_PAGE_BYTES != 0
    {
        return Err("segment causal state differs from its admitted paged view".into());
    }
    let capacity = usize::try_from(expected / VNEXT_KV_PAGE_BYTES)
        .map_err(|_| "segment causal page count exceeds usize")?;
    let mut pages = Vec::with_capacity(capacity);
    let mut logical = 0u64;
    for physical in view.physical_regions() {
        if physical.logical_offset_bytes() != logical
            || physical.length_bytes() == 0
            || physical.length_bytes() % VNEXT_KV_PAGE_BYTES != 0
        {
            return Err("segment causal pages lost block geometry".into());
        }
        let index = physical
            .owner_index()
            .ok_or("indexed segment page has no owner index")?;
        let (buffer, range, retention) = physical.borrowed_buffer_and_physical_range();
        let parent = owners
            .retain(index, buffer, range, retention)
            .map_err(|e| e.to_string())?;
        let mut offset = 0u64;
        while offset < physical.length_bytes() {
            let page = parent
                .subrange(offset, VNEXT_KV_PAGE_BYTES)
                .map_err(|e| e.to_string())?;
            let page_view = owners.region(&page).map_err(|e| e.to_string())?;
            if page_view.length_bytes() != VNEXT_KV_PAGE_BYTES
                || page_view.element_type() != element
            {
                return Err("segment causal physical page differs from its contract".into());
            }
            pages.push(page);
            offset = offset
                .checked_add(VNEXT_KV_PAGE_BYTES)
                .ok_or("segment causal page end overflows")?;
        }
        logical = logical
            .checked_add(physical.length_bytes())
            .ok_or("segment causal logical coverage overflows")?;
    }
    if logical != expected || pages.is_empty() {
        return Err("segment causal pages do not cover current state".into());
    }
    Ok(pages)
}

pub(in crate::backend::cuda::vnext_ops) fn prepare_indexed(
    node: &PreparedSegmentBindingNode<'_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaSegmentOwnerBuilder<'_>,
) -> Result<segment::PendingBinding, String> {
    let shape = recipe.shape;
    if node.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("segment causal participant shape differs from cold geometry".into());
    }
    let layout = BindingLayout::new(shape, recipe.participants)?;
    let destination = segment::shared_borrowed_region(node, 1)?;
    let kv_layout = shape.kv_layout()?;
    let maximum_pages = shape.maximum_pages()?;
    let maximum_entries = shape.table_entries(shape.maximum_context_tokens)?;
    let mut writes = Vec::with_capacity(recipe.participants);
    let mut ranges = Vec::new();
    for (participant, token_range) in node
        .work_shape()
        .participant_token_ranges()
        .iter()
        .enumerate()
    {
        let source = token_range.source_token_range();
        let packed = token_range.immediate_token_range();
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("segment causal token range exceeds admitted context".into());
        }
        let state_pages = indexed_pages(
            node,
            participant,
            0,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
            shape.kv_element_type(),
            owners,
        )?;
        let scale_pages = if shape.int8_kv {
            indexed_pages(
                node,
                participant,
                2,
                shape.scale_state_bytes(source.end)?,
                ElementType::F32,
                owners,
            )?
        } else {
            Vec::new()
        };
        let page_count = u64::try_from(state_pages.len())
            .map_err(|_| "segment causal page count exceeds u64")?;
        let entries = shape.table_entries(source.end)?;
        if page_count > maximum_pages || entries > maximum_entries {
            return Err("segment causal page table exceeds admitted maximum".into());
        }
        let table_entries = checked_i32(entries, "causal attention address-table entry count")?;
        let controls = [
            table_entries,
            checked_i32(source.start, "causal attention source position")?,
            checked_i32(
                token_range.immediate_tokens(),
                "causal attention participant token count",
            )?,
            checked_i32(source.end, "causal attention sequence token count")?,
            checked_i32(packed.start, "causal attention packed token start")?,
            0,
        ];
        let address_count = usize::try_from(entries)
            .map_err(|_| "segment causal entry count exceeds usize")?
            .checked_add(scale_pages.len())
            .ok_or("segment causal address count overflows")?;
        let payload_bytes = address_count
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(BINDING_CONTROL_BYTES as usize))
            .ok_or("segment causal payload size overflows")?;
        let mut payload = Vec::with_capacity(payload_bytes);
        for value in controls {
            payload.extend_from_slice(&value.to_ne_bytes());
        }
        visit_binding_addresses(
            kv_layout,
            table_entries,
            state_pages.len(),
            |index| {
                let range = state_pages
                    .get(index)
                    .ok_or("segment causal page index is absent")?;
                owners
                    .region(range)
                    .map(|view| view.device_ptr())
                    .map_err(|e| e.to_string())
            },
            |address| payload.extend_from_slice(&address.to_ne_bytes()),
        )?;
        for range in &scale_pages {
            payload.extend_from_slice(
                &owners
                    .region(range)
                    .map_err(|e| e.to_string())?
                    .device_ptr()
                    .to_ne_bytes(),
            );
        }
        writes.push(
            super::super::CudaProgramBindingWrite::new(
                layout.binding_offset(participant)?,
                payload.into_boxed_slice(),
            )
            .map_err(|e| e.to_string())?,
        );
        ranges.extend(state_pages);
        ranges.extend(scale_pages);
    }
    prepare_causal_program_binding_writes(recipe.upload_strategy, &mut writes, layout.slot_bytes)?;
    let numerical_status = if shape.int8_kv {
        Some((
            (0..recipe.participants)
                .map(|p| {
                    layout.binding_offset(p).and_then(|o| {
                        o.checked_add(BINDING_NUMERICAL_STATUS_OFFSET)
                            .ok_or("segment status offset overflows".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            INT8_NUMERICAL_FAILURE_MASK,
        ))
    } else {
        None
    };
    Ok(segment::PendingBinding {
        operation: "vnext_causal_paged_attention_bindings",
        binding: node
            .program_binding()
            .ok_or("segment causal binding slot is absent")?
            .clone(),
        destination,
        writes,
        ranges,
        numerical_status,
        participants: u32::try_from(recipe.participants)
            .map_err(|_| "segment participant count exceeds u32")?,
        tokens: node.work_shape().immediate_tokens(),
    })
}

fn program_pages(
    node: &ferrum_interfaces::vnext::PreparedSegmentBindingNodeView<'_, '_, CudaDeviceBuffer>,
    participant: usize,
    declaration: usize,
    expected: u64,
    element: ElementType,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaSegmentOwnerBuilder<'_>,
) -> Result<Vec<crate::backend::cuda::vnext_runtime::CudaSegmentRange>, String> {
    let view = node
        .region(participant, declaration)
        .map_err(|e| e.to_string())?;
    if view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().element_type != element
        || view.descriptor().size_bytes != expected
        || expected == 0
        || expected % VNEXT_KV_PAGE_BYTES != 0
    {
        return Err("segment causal state differs from its admitted paged view".into());
    }
    let capacity = usize::try_from(expected / VNEXT_KV_PAGE_BYTES)
        .map_err(|_| "segment causal page count exceeds usize")?;
    let mut pages = Vec::with_capacity(capacity);
    let mut logical = 0u64;
    for physical in view.physical_regions() {
        if physical.logical_offset_bytes() != logical
            || physical.length_bytes() == 0
            || physical.length_bytes() % VNEXT_KV_PAGE_BYTES != 0
        {
            return Err("segment causal pages lost block geometry".into());
        }
        let index = physical
            .owner_index()
            .ok_or("indexed segment page has no owner index")?;
        let (buffer, range, retention) = physical.borrowed_buffer_and_physical_range();
        let parent = owners
            .retain(index, buffer, range, retention)
            .map_err(|e| e.to_string())?;
        let mut offset = 0u64;
        while offset < physical.length_bytes() {
            let page = parent
                .subrange(offset, VNEXT_KV_PAGE_BYTES)
                .map_err(|e| e.to_string())?;
            let page_view = owners.region(&page).map_err(|e| e.to_string())?;
            if page_view.length_bytes() != VNEXT_KV_PAGE_BYTES
                || page_view.element_type() != element
            {
                return Err("segment causal physical page differs from its contract".into());
            }
            pages.push(page);
            offset = offset
                .checked_add(VNEXT_KV_PAGE_BYTES)
                .ok_or("segment causal page end overflows")?;
        }
        logical = logical
            .checked_add(physical.length_bytes())
            .ok_or("segment causal logical coverage overflows")?;
    }
    if logical != expected || pages.is_empty() {
        return Err("segment causal pages do not cover current state".into());
    }
    Ok(pages)
}

pub(in crate::backend::cuda::vnext_ops) fn write_program(
    node: &ferrum_interfaces::vnext::PreparedSegmentBindingNodeView<'_, '_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaSegmentOwnerBuilder<'_>,
    sink: &mut crate::backend::cuda::vnext_runtime::CudaBindingProgramSink,
) -> Result<(), String> {
    let shape = recipe.shape;
    if node.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("segment causal participant shape differs from cold geometry".into());
    }
    let layout = BindingLayout::new(shape, recipe.participants)?;
    let destination = segment::shared_program_region(node, 1)?;
    let kv_layout = shape.kv_layout()?;
    let maximum_pages = shape.maximum_pages()?;
    let maximum_entries = shape.table_entries(shape.maximum_context_tokens)?;
    if recipe.upload_strategy != ProgramBindingUploadStrategy::CompactScatter {
        return Err("compiled causal binding requires compact scatter".into());
    }
    sink.begin_node(
        "vnext_causal_paged_attention_bindings",
        node.program_binding()
            .ok_or("segment causal binding slot is absent")?
            .clone(),
        destination,
        u32::try_from(recipe.participants).map_err(|_| "segment participant count exceeds u32")?,
        node.work_shape().immediate_tokens(),
    )
    .map_err(|e| e.to_string())?;
    let mut ranges = Vec::new();
    for (participant, token_range) in node
        .work_shape()
        .participant_token_ranges()
        .iter()
        .enumerate()
    {
        let source = token_range.source_token_range();
        let packed = token_range.immediate_token_range();
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("segment causal token range exceeds admitted context".into());
        }
        let state_pages = program_pages(
            node,
            participant,
            0,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
            shape.kv_element_type(),
            owners,
        )?;
        let scale_pages = if shape.int8_kv {
            program_pages(
                node,
                participant,
                2,
                shape.scale_state_bytes(source.end)?,
                ElementType::F32,
                owners,
            )?
        } else {
            Vec::new()
        };
        let page_count = u64::try_from(state_pages.len())
            .map_err(|_| "segment causal page count exceeds u64")?;
        let entries = shape.table_entries(source.end)?;
        if page_count > maximum_pages || entries > maximum_entries {
            return Err("segment causal page table exceeds admitted maximum".into());
        }
        let table_entries = checked_i32(entries, "causal attention address-table entry count")?;
        let controls = [
            table_entries,
            checked_i32(source.start, "causal attention source position")?,
            checked_i32(
                token_range.immediate_tokens(),
                "causal attention participant token count",
            )?,
            checked_i32(source.end, "causal attention sequence token count")?,
            checked_i32(packed.start, "causal attention packed token start")?,
            0,
        ];
        let address_count = usize::try_from(entries)
            .map_err(|_| "segment causal entry count exceeds usize")?
            .checked_add(scale_pages.len())
            .ok_or("segment causal address count overflows")?;
        let payload_bytes = address_count
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(BINDING_CONTROL_BYTES as usize))
            .ok_or("segment causal payload size overflows")?;
        sink.write(
            layout.binding_offset(participant)?,
            payload_bytes,
            |payload| {
                for value in controls {
                    payload.extend_from_slice(&value.to_ne_bytes());
                }
                visit_binding_addresses(
                    kv_layout,
                    table_entries,
                    state_pages.len(),
                    |index| {
                        let range = state_pages
                            .get(index)
                            .ok_or("segment causal page index is absent")?;
                        owners
                            .region(range)
                            .map(|view| view.device_ptr())
                            .map_err(|e| e.to_string())
                    },
                    |address| payload.extend_from_slice(&address.to_ne_bytes()),
                )?;
                for range in &scale_pages {
                    payload.extend_from_slice(
                        &owners
                            .region(range)
                            .map_err(|e| e.to_string())?
                            .device_ptr()
                            .to_ne_bytes(),
                    );
                }
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        ranges.extend(state_pages);
        ranges.extend(scale_pages);
    }

    let numerical_status = if shape.int8_kv {
        Some((
            (0..recipe.participants)
                .map(|p| {
                    layout.binding_offset(p).and_then(|o| {
                        o.checked_add(BINDING_NUMERICAL_STATUS_OFFSET)
                            .ok_or("segment status offset overflows".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            INT8_NUMERICAL_FAILURE_MASK,
        ))
    } else {
        None
    };
    sink.end_node(ranges, numerical_status)
        .map_err(|e| e.to_string())
}

pub(in crate::backend::cuda::vnext_ops) fn write_physical_program(
    node: &ferrum_interfaces::vnext::PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaPhysicalOwnerBuilder,
    sink: &mut crate::backend::cuda::vnext_runtime::CudaBindingProgramSink,
) -> Result<(), String> {
    write_physical_program_inner(node, recipe, owners, sink, false)
}

pub(crate) fn write_resident_physical_program(
    node: &ferrum_interfaces::vnext::PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaPhysicalOwnerBuilder,
    sink: &mut crate::backend::cuda::vnext_runtime::CudaBindingProgramSink,
) -> Result<(), String> {
    write_physical_program_inner(node, recipe, owners, sink, true)
}

fn write_physical_program_inner(
    node: &ferrum_interfaces::vnext::PreparedPhysicalSegmentNodeView<'_, '_, CudaDeviceBuffer>,
    recipe: &CausalRecipe,
    owners: &mut crate::backend::cuda::vnext_runtime::CudaPhysicalOwnerBuilder,
    sink: &mut crate::backend::cuda::vnext_runtime::CudaBindingProgramSink,
    resident_page_prefix: bool,
) -> Result<(), String> {
    let shape = recipe.shape;
    if node.participant_count() != recipe.participants || recipe.participants == 0 {
        return Err("segment causal participant shape differs from cold geometry".into());
    }
    let layout = BindingLayout::new(shape, recipe.participants)?;
    let destination = segment::shared_physical_program_region(node, 1)?;
    let kv_layout = shape.kv_layout()?;
    let maximum_pages = shape.maximum_pages()?;
    let maximum_entries = shape.table_entries(shape.maximum_context_tokens)?;
    if recipe.upload_strategy != ProgramBindingUploadStrategy::CompactScatter {
        return Err("compiled causal binding requires compact scatter".into());
    }
    sink.begin_node(
        "vnext_causal_paged_attention_bindings",
        node.program_binding()
            .ok_or("segment causal binding slot is absent")?
            .clone(),
        destination,
        u32::try_from(recipe.participants).map_err(|_| "segment participant count exceeds u32")?,
        node.work_shape().immediate_tokens(),
    )
    .map_err(|e| e.to_string())?;
    let mut ranges = Vec::new();
    for (participant, token_range) in node
        .work_shape()
        .participant_token_ranges()
        .iter()
        .enumerate()
    {
        let source = token_range.source_token_range();
        let packed = token_range.immediate_token_range();
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("segment causal token range exceeds admitted context".into());
        }
        let state_pages = segment::physical_program_window(
            node,
            participant,
            0,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
            shape.kv_element_type(),
            Some(VNEXT_KV_PAGE_BYTES),
            owners,
        )?;
        let scale_pages = if shape.int8_kv {
            Some(segment::physical_program_window(
                node,
                participant,
                2,
                shape.scale_state_bytes(source.end)?,
                ElementType::F32,
                Some(VNEXT_KV_PAGE_BYTES),
                owners,
            )?)
        } else {
            None
        };
        let scale_count = scale_pages.as_ref().map_or(0, |pages| pages.len());
        let page_count = u64::try_from(state_pages.len())
            .map_err(|_| "segment causal page count exceeds u64")?;
        let entries = shape.table_entries(source.end)?;
        if page_count > maximum_pages || entries > maximum_entries {
            return Err("segment causal page table exceeds admitted maximum".into());
        }
        // A resident packet may retain all addresses inside the CURRENT
        // physical pages. Kernels still consume only the fresh control count;
        // no future page or unallocated address is invented.
        let payload_entries = if resident_page_prefix {
            resident_address_prefix(kv_layout, page_count, maximum_entries)?
        } else {
            entries
        };
        let payload_table_entries = checked_i32(payload_entries, "resident address prefix")?;
        let table_entries = checked_i32(entries, "causal attention address-table entry count")?;
        let controls = [
            table_entries,
            checked_i32(source.start, "causal attention source position")?,
            checked_i32(
                token_range.immediate_tokens(),
                "causal attention participant token count",
            )?,
            checked_i32(source.end, "causal attention sequence token count")?,
            checked_i32(packed.start, "causal attention packed token start")?,
            0,
        ];
        let address_count = usize::try_from(payload_entries)
            .map_err(|_| "segment causal entry count exceeds usize")?
            .checked_add(scale_count)
            .ok_or("segment causal address count overflows")?;
        let payload_bytes = address_count
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(BINDING_CONTROL_BYTES as usize))
            .ok_or("segment causal payload size overflows")?;
        sink.write(
            layout.binding_offset(participant)?,
            payload_bytes,
            |payload| {
                for value in controls {
                    payload.extend_from_slice(&value.to_ne_bytes());
                }
                visit_binding_addresses(
                    kv_layout,
                    payload_table_entries,
                    state_pages.len(),
                    |index| {
                        owners
                            .region(&state_pages, index)
                            .map(|view| view.device_ptr())
                            .map_err(|e| e.to_string())
                    },
                    |address| payload.extend_from_slice(&address.to_ne_bytes()),
                )?;
                if let Some(pages) = &scale_pages {
                    for index in 0..pages.len() {
                        payload.extend_from_slice(
                            &owners
                                .region(pages, index)
                                .map_err(|e| e.to_string())?
                                .device_ptr()
                                .to_ne_bytes(),
                        );
                    }
                }
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        ranges.push(state_pages);
        if let Some(pages) = scale_pages {
            ranges.push(pages);
        }
    }

    let numerical_status = if shape.int8_kv {
        Some((
            (0..recipe.participants)
                .map(|p| {
                    layout.binding_offset(p).and_then(|o| {
                        o.checked_add(BINDING_NUMERICAL_STATUS_OFFSET)
                            .ok_or("segment status offset overflows".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            INT8_NUMERICAL_FAILURE_MASK,
        ))
    } else {
        None
    };
    sink.end_physical_node(ranges, numerical_status)
        .map_err(|e| e.to_string())
}

fn resident_address_prefix(
    layout: CausalKvLayout,
    pages: u64,
    maximum_entries: u64,
) -> Result<u64, String> {
    if pages == 0 || maximum_entries == 0 {
        return Err("resident causal physical prefix is empty".into());
    }
    let available = match layout {
        CausalKvLayout::TokenMajorPages => pages,
        CausalKvLayout::VllmBlocks16 {
            blocks_per_page, ..
        } => pages
            .checked_mul(blocks_per_page)
            .ok_or("resident causal entry count overflows")?,
    };
    Ok(available.min(maximum_entries))
}

#[cfg(test)]
mod resident_tests {
    use super::*;

    #[test]
    fn resident_causal_prefix_covers_only_current_pages_and_context() {
        let layout = CausalKvLayout::VllmBlocks16 {
            combined_block_bytes: 32 * 1024,
            blocks_per_page: 2,
        };
        // The second logical block is already allocated in the same 64KiB
        // physical page. A new third block must never use an invented page.
        let entries = resident_address_prefix(layout, 1, 128).unwrap();
        assert_eq!(entries, 2);
        let addresses = binding_addresses(layout, entries as i32, &[0x10_0000]).unwrap();
        assert_eq!(addresses, [0x10_0000, 0x10_0000 + 32 * 1024]);
        assert!(binding_addresses(layout, 3, &[0x10_0000]).is_err());
        assert_eq!(resident_address_prefix(layout, 2, 3).unwrap(), 3);
        assert_eq!(
            resident_address_prefix(CausalKvLayout::TokenMajorPages, 2, 8).unwrap(),
            2
        );
        assert!(resident_address_prefix(layout, 0, 128).is_err());
        assert!(resident_address_prefix(layout, u64::MAX, u64::MAX).is_err());
    }
}
