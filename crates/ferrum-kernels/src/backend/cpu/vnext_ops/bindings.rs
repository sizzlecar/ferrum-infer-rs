use super::super::vnext_runtime::{CpuBufferRegion, CpuDeviceBuffer, CpuRuntimeError};
use ferrum_interfaces::vnext::{
    BatchedOperationInvocation, ElementType, NodeWorkContract, OperationBufferStorageKind,
    OperationBufferView, OperationInvocation, ResolvedTensorLayout, ResolvedValueBinding,
    ResolvedValueRole, ResourceId,
};
use std::ops::Range;

pub(super) fn binding<'a>(
    participant: &'a OperationInvocation<'_, CpuDeviceBuffer>,
    role: ResolvedValueRole,
    ordinal: u32,
) -> Result<&'a ResolvedValueBinding, CpuRuntimeError> {
    participant
        .bindings()
        .iter()
        .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
        .ok_or_else(|| {
            CpuRuntimeError::new(format!("CPU operation lacks {role:?} binding {ordinal}"))
        })
}

pub(super) fn physical_region(
    participant: &OperationInvocation<'_, CpuDeviceBuffer>,
    resource: &ResourceId,
    offset: u64,
    bytes: u64,
    dtype: ElementType,
) -> Result<CpuBufferRegion, CpuRuntimeError> {
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == resource)
        .ok_or_else(|| CpuRuntimeError::new("CPU binding lacks its committed resource view"))?;
    let translated = view.translate(offset, bytes)?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| CpuRuntimeError::new("CPU binding has no physical region"))?;
    if physical.next().is_some() {
        return Err(CpuRuntimeError::new(
            "CPU value requires contiguous physical storage",
        ));
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    let retained = buffer.retained_region(range, retention)?;
    if retained.element_type() != dtype || retained.length_bytes() as u64 != bytes {
        return Err(CpuRuntimeError::new(
            "CPU binding retained a different physical ABI",
        ));
    }
    Ok(retained)
}

pub(super) fn value_region(
    participant: &OperationInvocation<'_, CpuDeviceBuffer>,
    binding: &ResolvedValueBinding,
    dtype: ElementType,
    tokens: Option<Range<u64>>,
) -> Result<CpuBufferRegion, CpuRuntimeError> {
    let [component] = binding.storage().components() else {
        return Err(CpuRuntimeError::new(
            "CPU value requires one storage component",
        ));
    };
    if component.element_type() != dtype
        || binding.tensor().element_type() != dtype
        || !matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    {
        return Err(CpuRuntimeError::new(
            "CPU value has an incompatible type or layout",
        ));
    }
    let (offset, length) = if let Some(tokens) = tokens {
        let projection = participant
            .work()
            .token_projection(binding.role(), binding.ordinal());
        let extent = match projection {
            Some(projection)
                if projection.axis() == 0
                    && projection.rank() as usize == binding.tensor().dimensions().len()
                    && binding.tensor().dimensions().first()
                        == Some(&projection.canonical_extent())
                    && component.offset_bytes() == 0 =>
            {
                projection.canonical_extent()
            }
            None if matches!(participant.work(), NodeWorkContract::Fixed) => {
                binding.tensor().dimensions().first().copied().unwrap_or(0)
            }
            _ => {
                return Err(CpuRuntimeError::new(
                    "CPU token value has no canonical leading-axis projection",
                ))
            }
        };
        if extent == 0
            || !component.length_bytes().is_multiple_of(extent)
            || tokens.start >= tokens.end
            || tokens.end > extent
        {
            return Err(CpuRuntimeError::new(
                "CPU token range differs from its canonical leading axis",
            ));
        }
        let stride = component.length_bytes() / extent;
        (
            tokens
                .start
                .checked_mul(stride)
                .and_then(|offset| component.offset_bytes().checked_add(offset)),
            (tokens.end - tokens.start).checked_mul(stride),
        )
    } else {
        (
            Some(component.offset_bytes()),
            Some(component.length_bytes()),
        )
    };
    physical_region(
        participant,
        component.resource_id(),
        offset.ok_or_else(|| CpuRuntimeError::new("CPU token offset overflows"))?,
        length.ok_or_else(|| CpuRuntimeError::new("CPU token length overflows"))?,
        dtype,
    )
}

pub(super) fn paged_regions(
    participant: &OperationInvocation<'_, CpuDeviceBuffer>,
    resource: &ResourceId,
    bytes: u64,
    page_bytes: u64,
    dtype: ElementType,
) -> Result<Vec<CpuBufferRegion>, CpuRuntimeError> {
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == resource)
        .ok_or_else(|| CpuRuntimeError::new("CPU paged state lacks its committed view"))?;
    if page_bytes == 0
        || bytes == 0
        || !bytes.is_multiple_of(page_bytes)
        || view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().element_type != dtype
        || view.descriptor().size_bytes != bytes
    {
        return Err(CpuRuntimeError::new(
            "CPU paged state differs from its admitted geometry",
        ));
    }
    let translated = view.translate(0, bytes)?;
    let mut pages = Vec::new();
    let mut next_logical = 0_u64;
    for physical in translated.iter() {
        if physical.logical_offset_bytes() != next_logical
            || physical.length_bytes() == 0
            || !physical.length_bytes().is_multiple_of(page_bytes)
        {
            return Err(CpuRuntimeError::new(
                "CPU paged translation lost its logical page order",
            ));
        }
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let mut offset = 0;
        while offset < physical.length_bytes() {
            let start = range
                .start
                .checked_add(offset)
                .ok_or_else(|| CpuRuntimeError::new("CPU physical page offset overflows"))?;
            let end = start
                .checked_add(page_bytes)
                .ok_or_else(|| CpuRuntimeError::new("CPU physical page end overflows"))?;
            let page = buffer.retained_region(start..end, retention.clone())?;
            if page.element_type() != dtype || page.length_bytes() as u64 != page_bytes {
                return Err(CpuRuntimeError::new(
                    "CPU physical page differs from its typed ABI",
                ));
            }
            pages.push(page);
            offset += page_bytes;
        }
        next_logical = next_logical
            .checked_add(physical.length_bytes())
            .ok_or_else(|| CpuRuntimeError::new("CPU logical page coverage overflows"))?;
    }
    if next_logical != bytes || pages.is_empty() {
        return Err(CpuRuntimeError::new(
            "CPU pages do not cover the admitted state",
        ));
    }
    Ok(pages)
}

pub(super) fn token_region(
    invocation: &BatchedOperationInvocation<'_, CpuDeviceBuffer>,
    participant_index: usize,
    role: ResolvedValueRole,
    ordinal: u32,
    dtype: ElementType,
    last_only: bool,
) -> Result<CpuBufferRegion, CpuRuntimeError> {
    let participant = &invocation.participants()[participant_index];
    let ranges = invocation
        .participant_token_ranges()
        .get(participant_index)
        .ok_or_else(|| CpuRuntimeError::new("CPU batch lacks participant token ranges"))?;
    let mut tokens = if matches!(participant.work(), NodeWorkContract::Fixed) {
        0..binding(participant, role, ordinal)?
            .tensor()
            .dimensions()
            .first()
            .copied()
            .unwrap_or(0)
    } else if invocation.binding_uses_packed_batch_coordinates(role, ordinal)? {
        ranges.immediate_token_range()
    } else {
        ranges.source_token_range()
    };
    if last_only {
        if tokens.start >= tokens.end {
            return Err(CpuRuntimeError::new(
                "CPU last-token operation has no input rows",
            ));
        }
        tokens.start = tokens.end - 1;
    }
    value_region(
        participant,
        binding(participant, role, ordinal)?,
        dtype,
        Some(tokens),
    )
}

pub(super) fn scratch_region(
    participant: &OperationInvocation<'_, CpuDeviceBuffer>,
    offset: u64,
    length: u64,
) -> Result<CpuBufferRegion, CpuRuntimeError> {
    let view = participant
        .scratch_view()
        .ok_or_else(|| CpuRuntimeError::new("CPU operation has no admitted workspace"))?;
    workspace_region(view, offset, length)
}

pub(super) fn binding_region(
    participant: &OperationInvocation<'_, CpuDeviceBuffer>,
    offset: u64,
    length: u64,
) -> Result<CpuBufferRegion, CpuRuntimeError> {
    let view = participant
        .binding_view()
        .ok_or_else(|| CpuRuntimeError::new("CPU operation has no admitted binding workspace"))?;
    workspace_region(view, offset, length)
}

fn workspace_region(
    view: &OperationBufferView<'_, CpuDeviceBuffer>,
    offset: u64,
    length: u64,
) -> Result<CpuBufferRegion, CpuRuntimeError> {
    if view.descriptor().element_type != ElementType::U8
        || offset
            .checked_add(length)
            .is_none_or(|end| end > view.descriptor().size_bytes)
    {
        return Err(CpuRuntimeError::new(
            "CPU workspace exceeds its admitted range",
        ));
    }
    let translated = view.translate(offset, length)?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| CpuRuntimeError::new("CPU workspace has no physical storage"))?;
    if physical.next().is_some() {
        return Err(CpuRuntimeError::new("CPU workspace is not contiguous"));
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer.retained_region(range, retention)
}
