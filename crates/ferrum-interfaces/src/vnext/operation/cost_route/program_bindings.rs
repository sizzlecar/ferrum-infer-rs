//! Numeric layout used by both a sparse binding encoder and its pure cost query.
//! No addresses, payloads, buffers or allocation authority are carried here.
use super::*;
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramBindingCostWrite {
    offset_bytes: u64,
    length_bytes: u64,
}

impl ProgramBindingCostWrite {
    pub fn new(offset_bytes: u64, length_bytes: u64) -> Result<Self, VNextError> {
        if length_bytes == 0 || offset_bytes.checked_add(length_bytes).is_none() {
            return Err(invalid_operation(
                "program binding write is empty or overflows",
            ));
        }
        Ok(Self {
            offset_bytes,
            length_bytes,
        })
    }
    pub const fn offset_bytes(self) -> u64 {
        self.offset_bytes
    }
    pub const fn length_bytes(self) -> u64 {
        self.length_bytes
    }
}

#[derive(Debug)]
pub struct ProgramBindingCostPatch<'a> {
    pub node_index: usize,
    pub command: &'a OperationCostCommand,
    /// Exact encoder writes relative to this node's compiled arena slot.
    pub writes: &'a [ProgramBindingCostWrite],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramBindingTransferLayout {
    pub destination_offset_bytes: u64,
    pub destination_stride_bytes: u64,
    pub row_bytes: u64,
    pub row_count: usize,
    /// Input write ranges in destination-row order. A range contains consecutive
    /// original writes; separate ranges may skip writes owned by another transfer.
    /// Concatenation contains only live payload bytes, never destination padding.
    pub source_write_ranges: Vec<Range<usize>>,
}

struct BindingRow {
    destination_offset_bytes: u64,
    row_bytes: u64,
    source_writes: Range<usize>,
}

impl BindingRow {
    fn transfer(&self) -> Result<ProgramBindingTransferLayout, VNextError> {
        let mut source_write_ranges = Vec::new();
        source_write_ranges.try_reserve_exact(1).map_err(|_| {
            invalid_operation("program binding transfer source capacity unavailable")
        })?;
        source_write_ranges.push(self.source_writes.clone());
        Ok(ProgramBindingTransferLayout {
            destination_offset_bytes: self.destination_offset_bytes,
            destination_stride_bytes: self.row_bytes,
            row_bytes: self.row_bytes,
            row_count: 1,
            source_write_ranges,
        })
    }
}

/// Merge adjacent writes, then equal-sized rows at a constant positive stride.
/// A second plan may group equal-width rows across intervening writes. It wins
/// only when it uses fewer transfers; ties retain the original address-order plan.
/// The actual encoder sorts its writes once before this call. The cost caller
/// sorts only bounded numeric spans and supplies its live planning budget.
pub fn coalesce_sorted_program_binding_writes(
    writes: &[ProgramBindingCostWrite],
    arena_size_bytes: u64,
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<Vec<ProgramBindingTransferLayout>, VNextError> {
    if writes.is_empty() || arena_size_bytes == 0 {
        return Err(invalid_operation(
            "program binding writes or arena are empty",
        ));
    }
    let mut rows: Vec<BindingRow> = Vec::new();
    rows.try_reserve_exact(writes.len())
        .map_err(|_| invalid_operation("program binding row capacity unavailable"))?;
    let mut prior_end = 0;
    for (index, write) in writes.iter().enumerate() {
        poll()?;
        let end = write
            .offset_bytes
            .checked_add(write.length_bytes)
            .ok_or_else(|| invalid_operation("program binding write overflow"))?;
        if write.length_bytes == 0 || write.offset_bytes < prior_end || end > arena_size_bytes {
            return Err(invalid_operation(
                "program binding writes overlap, are unordered or exceed arena",
            ));
        }
        if let Some(row) = rows.last_mut().filter(|_| write.offset_bytes == prior_end) {
            row.row_bytes = row
                .row_bytes
                .checked_add(write.length_bytes)
                .ok_or_else(|| invalid_operation("program binding contiguous row overflow"))?;
            row.source_writes.end = index + 1;
        } else {
            rows.push(BindingRow {
                destination_offset_bytes: write.offset_bytes,
                row_bytes: write.length_bytes,
                source_writes: index..index + 1,
            });
        }
        prior_end = end;
    }
    let mut transfers = Vec::new();
    transfers
        .try_reserve_exact(rows.len())
        .map_err(|_| invalid_operation("program binding transfer capacity unavailable"))?;
    let mut index = 0;
    while index < rows.len() {
        poll()?;
        let mut transfer = rows[index].transfer()?;
        if let Some(next) = rows
            .get(index + 1)
            .filter(|next| next.row_bytes == transfer.row_bytes)
        {
            let stride = next
                .destination_offset_bytes
                .checked_sub(transfer.destination_offset_bytes)
                .filter(|stride| *stride >= transfer.row_bytes)
                .ok_or_else(|| invalid_operation("program binding row stride is invalid"))?;
            transfer.destination_stride_bytes = stride;
            transfer.row_count = 2;
            append_sources(&mut transfer, next)?;
            while let Some(next) = rows.get(index + transfer.row_count) {
                poll()?;
                let previous = &rows[index + transfer.row_count - 1];
                if next.row_bytes != transfer.row_bytes
                    || next
                        .destination_offset_bytes
                        .checked_sub(previous.destination_offset_bytes)
                        != Some(stride)
                {
                    break;
                }
                transfer.row_count += 1;
                append_sources(&mut transfer, next)?;
            }
        }
        index += transfer.row_count;
        transfers.push(transfer);
    }
    if transfers.len() <= 1 {
        return Ok(transfers);
    }
    group_equal_width_rows(&rows, transfers, poll)
}

fn append_sources(
    transfer: &mut ProgramBindingTransferLayout,
    row: &BindingRow,
) -> Result<(), VNextError> {
    let source = &row.source_writes;
    if let Some(previous) = transfer
        .source_write_ranges
        .last_mut()
        .filter(|previous| previous.end == source.start)
    {
        previous.end = source.end;
    } else {
        transfer
            .source_write_ranges
            .try_reserve(1)
            .map_err(|_| invalid_operation("program binding source index capacity unavailable"))?;
        transfer.source_write_ranges.push(source.clone());
    }
    Ok(())
}

fn group_equal_width_rows(
    rows: &[BindingRow],
    original: Vec<ProgramBindingTransferLayout>,
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<Vec<ProgramBindingTransferLayout>, VNextError> {
    // The hash map is only a lookup from width to the last row. No map iteration
    // controls output: heads and links retain original destination order. This
    // avoids a second sort and permits polling on every bounded row operation.
    let mut last_by_width = HashMap::new();
    last_by_width
        .try_reserve(rows.len())
        .map_err(|_| invalid_operation("program binding width index capacity unavailable"))?;
    let mut next_by_width = Vec::new();
    next_by_width
        .try_reserve_exact(rows.len())
        .map_err(|_| invalid_operation("program binding row link capacity unavailable"))?;
    next_by_width.resize(rows.len(), None);
    let mut heads = Vec::new();
    heads
        .try_reserve_exact(rows.len())
        .map_err(|_| invalid_operation("program binding row head capacity unavailable"))?;
    for (index, row) in rows.iter().enumerate() {
        poll()?;
        if let Some(previous) = last_by_width.insert(row.row_bytes, index) {
            next_by_width[previous] = Some(index);
        } else {
            heads.push(index);
        }
    }
    // Each completed transfer is stored at its first original row, then drained
    // in that order. No sort or nondeterministic map order enters the recipe.
    let mut at_head = Vec::new();
    at_head
        .try_reserve_exact(rows.len())
        .map_err(|_| invalid_operation("program binding indexed plan capacity unavailable"))?;
    at_head.resize_with(rows.len(), || None);
    let mut transfer_count = 0_usize;
    for head in heads {
        let mut cursor = Some(head);
        while let Some(start) = cursor {
            poll()?;
            let mut transfer = rows[start].transfer()?;
            let mut last = start;
            cursor = next_by_width[start];
            if let Some(second) = cursor {
                let stride = rows[second]
                    .destination_offset_bytes
                    .checked_sub(transfer.destination_offset_bytes)
                    .filter(|stride| *stride >= transfer.row_bytes)
                    .ok_or_else(|| {
                        invalid_operation("program binding indexed stride is invalid")
                    })?;
                transfer.destination_stride_bytes = stride;
                loop {
                    let Some(next) = cursor else { break };
                    poll()?;
                    if rows[next]
                        .destination_offset_bytes
                        .checked_sub(rows[last].destination_offset_bytes)
                        != Some(stride)
                    {
                        break;
                    }
                    append_sources(&mut transfer, &rows[next])?;
                    transfer.row_count += 1;
                    last = next;
                    cursor = next_by_width[next];
                }
            }
            transfer_count += 1;
            // Counts can only increase. A partial candidate cannot beat the
            // complete original at this point; only the validated original escapes.
            if transfer_count >= original.len() {
                return Ok(original);
            }
            at_head[start] = Some(transfer);
        }
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(transfer_count)
        .map_err(|_| invalid_operation("program binding grouped plan capacity unavailable"))?;
    for transfer in at_head {
        poll()?;
        if let Some(transfer) = transfer {
            result.push(transfer);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
