//! Current KV/scale page-table rule. Physical extents and token frontiers are
//! read from fresh segment facts, never retained from the cold decode wave.
use super::super::segment_bindings::{self as segment, DynamicRecipe, ValidationRecipe};
use super::*;
use ferrum_interfaces::vnext::{
    PreparedSegmentBindingNode, SegmentBindingDeclaration, SegmentBindingRegionExtent,
    SegmentBindingRegionRequest, SegmentBindingRegionSelector,
};

pub(in crate::backend::cuda::vnext_ops) struct CausalRecipe {
    shape: CausalAttentionShape,
    participants: usize,
    upload_strategy: ProgramBindingUploadStrategy,
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
            extent: SegmentBindingRegionExtent::Exact(layout.required_bytes),
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
