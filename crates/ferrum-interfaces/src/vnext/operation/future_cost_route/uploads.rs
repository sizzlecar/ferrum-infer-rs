use super::super::backing_upload::contiguous_upload_run_end;
use super::super::buffer_view::{
    translate_step_participant_numeric_range, StepParticipantRangeCoordinates,
};
use super::ExecutionCostRouteUnknown as U;
use crate::vnext::*;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub struct EagerCoreInputUpload<'a> {
    pub node_id: &'a NodeId,
    pub input_ordinal: u32,
    pub participant_index: usize,
    pub logical_offset_bytes: u64,
    pub layout: HostTransferLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct EagerCoreReadback<'a> {
    pub node_id: &'a NodeId,
    pub resource_id: &'a ResourceId,
    pub participant_index: usize,
    pub logical_offset_bytes: u64,
    pub layout: HostTransferLayout,
}

pub(super) fn input_transfer_bytes(
    resolved: &dyn ExecutablePlanView,
    rows: &[OperationCostWorkRow],
    uploads: &[EagerCoreInputUpload<'_>],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Vec<u64>, U> {
    let mut commands = Vec::new();
    commands
        .try_reserve_exact(uploads.len())
        .map_err(|_| U::Capacity)?;
    let mut cursor = 0;
    while cursor < uploads.len() {
        super::core::poll(budget)?;
        let first = &uploads[cursor];
        let mut end = cursor + 1;
        while end < uploads.len()
            && uploads[end].node_id == first.node_id
            && uploads[end].input_ordinal == first.input_ordinal
        {
            super::core::poll(budget)?;
            end += 1;
        }
        let node = resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == first.node_id)
            .ok_or(U::InvalidInput)?;
        let value = node
            .values()
            .iter()
            .find(|value| {
                value.role() == ResolvedValueRole::Input && value.ordinal() == first.input_ordinal
            })
            .ok_or(U::InvalidInput)?;
        let [component] = value.storage().components() else {
            return Err(U::CoreLayout);
        };
        if value.usage() != BufferUsage::Activations
            || !matches!(value.access(), TensorAccess::Read | TensorAccess::ReadWrite)
        {
            return Err(U::CoreLayout);
        }
        let descriptor = contiguous_step_descriptor(resolved, component.resource_id())?;
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(end - cursor)
            .map_err(|_| U::Capacity)?;
        for input in &uploads[cursor..end] {
            super::core::poll(budget)?;
            if input.layout.element_type() != component.element_type()
                || input.layout.element_type() != value.tensor().element_type()
            {
                return Err(U::InvalidInput);
            }
            let bytes = input.layout.byte_len().map_err(|_| U::InvalidInput)?;
            let local_end = input
                .logical_offset_bytes
                .checked_add(bytes)
                .ok_or(U::InvalidInput)?;
            if local_end > component.length_bytes() {
                return Err(U::InvalidInput);
            }
            let start = component
                .offset_bytes()
                .checked_add(input.logical_offset_bytes)
                .ok_or(U::InvalidInput)?;
            let finish = start.checked_add(bytes).ok_or(U::InvalidInput)?;
            ranges.push(project_range(
                descriptor,
                rows,
                input.participant_index,
                start..finish,
                StepParticipantRangeCoordinates::SourceToken,
            )?);
        }
        let mut range_cursor = 0;
        while range_cursor < ranges.len() {
            super::core::poll(budget)?;
            let (next_cursor, bytes) = merged_transfer_span(&ranges, range_cursor)?;
            range_cursor = next_cursor;
            commands.push(bytes);
        }
        cursor = end;
    }
    Ok(commands)
}

pub(super) fn readback_bytes(
    resolved: &dyn ExecutablePlanView,
    rows: &[OperationCostWorkRow],
    readbacks: &[EagerCoreReadback<'_>],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<u64, U> {
    let mut total = 0_u64;
    for readback in readbacks {
        super::core::poll(budget)?;
        let node = resolved
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == readback.node_id)
            .ok_or(U::InvalidInput)?;
        if !node.values().iter().any(|value| {
            value.role() == ResolvedValueRole::Output
                && value
                    .storage()
                    .components()
                    .iter()
                    .any(|component| component.resource_id() == readback.resource_id)
        }) {
            return Err(U::InvalidInput);
        }
        let descriptor = contiguous_step_descriptor(resolved, readback.resource_id)?;
        if descriptor.element_type() != readback.layout.element_type() {
            return Err(U::InvalidInput);
        }
        let bytes = readback.layout.byte_len().map_err(|_| U::InvalidInput)?;
        let end = readback
            .logical_offset_bytes
            .checked_add(bytes)
            .ok_or(U::InvalidInput)?;
        project_range(
            descriptor,
            rows,
            readback.participant_index,
            readback.logical_offset_bytes..end,
            StepParticipantRangeCoordinates::ParticipantLocal,
        )?;
        total = total.checked_add(bytes).ok_or(U::InvalidInput)?;
    }
    Ok(total)
}

fn contiguous_step_descriptor<'a>(
    resolved: &'a dyn ExecutablePlanView,
    id: &ResourceId,
) -> Result<&'a DynamicResourceDescriptor, U> {
    let descriptor = resolved
        .execution_plan()
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id() == id)
        .ok_or(U::CoreLayout)?;
    if descriptor.lifetime() != AllocationLifetime::Step
        || descriptor.kind() != &AllocationKind::Value
        || descriptor.storage().profile().view() != DynamicStorageView::Contiguous
    {
        return Err(U::CoreLayout);
    }
    Ok(descriptor)
}

fn project_range(
    descriptor: &DynamicResourceDescriptor,
    rows: &[OperationCostWorkRow],
    index: usize,
    range: Range<u64>,
    coordinates: StepParticipantRangeCoordinates,
) -> Result<Range<u64>, U> {
    let row = rows.get(index).ok_or(U::InvalidInput)?;
    let end = row
        .offset
        .checked_add(row.count.get())
        .ok_or(U::InvalidInput)?;
    let total = rows
        .iter()
        .try_fold(0_u64, |sum, row| sum.checked_add(row.count.get()))
        .ok_or(U::InvalidInput)?;
    let packed_start = rows[..index]
        .iter()
        .try_fold(0_u64, |sum, row| sum.checked_add(row.count.get()))
        .ok_or(U::InvalidInput)?;
    let sequences = u32::try_from(rows.len()).map_err(|_| U::InvalidInput)?;
    let translated = translate_step_participant_numeric_range(
        descriptor.demand(),
        sequences,
        total,
        index,
        row.offset..end,
        packed_start,
        range,
        coordinates,
    )
    .map_err(|_| U::CoreLayout)?;
    let size = descriptor
        .evaluate_logical_request_bytes_for_shape(DynamicResourceShape::from_validated(
            sequences, total, 0,
        ))
        .map_err(|_| U::CoreLayout)?;
    if translated.start >= translated.end || translated.end > size {
        return Err(U::CoreLayout);
    }
    Ok(translated)
}

/// Same actual upload run selection, retaining its exact byte extent.
fn merged_transfer_span(ranges: &[Range<u64>], cursor: usize) -> Result<(usize, u64), U> {
    let first = ranges.get(cursor).ok_or(U::InvalidInput)?;
    let end = contiguous_upload_run_end(ranges.len(), cursor, true, |index| ranges[index].clone());
    let last = ranges
        .get(end.checked_sub(1).ok_or(U::InvalidInput)?)
        .ok_or(U::InvalidInput)?;
    let bytes = last
        .end
        .checked_sub(first.start)
        .filter(|&bytes| bytes > 0)
        .ok_or(U::InvalidInput)?;
    Ok((end, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transfer_spans_follow_actual_contiguous_run_boundaries_without_count_times_size() {
        let ranges = [7..11, 11..19, 24..29, 28..34, 34..39];
        let mut cursor = 0;
        let mut bytes = Vec::new();
        while cursor < ranges.len() {
            let (end, size) = merged_transfer_span(&ranges, cursor).unwrap();
            bytes.push(size);
            cursor = end;
        }
        assert_eq!(bytes, vec![12, 5, 11]);
        assert_eq!(merged_transfer_span(&[3..3], 0), Err(U::InvalidInput));
        assert_eq!(merged_transfer_span(&[], 0), Err(U::InvalidInput));
        assert_eq!(
            merged_transfer_span(&[u64::MAX - 3..u64::MAX], 0),
            Ok((1, 3))
        );
    }
}
