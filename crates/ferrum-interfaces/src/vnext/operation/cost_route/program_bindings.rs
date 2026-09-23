//! Numeric layout used by both a sparse binding encoder and its pure cost query.
//! No addresses, payloads, buffers or allocation authority are carried here.
use super::*;
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
    /// Consecutive entries in the input, already sorted by destination.
    /// Concatenating these payloads gives the transfer's packed host payload.
    pub source_writes: Range<usize>,
}

/// Merge adjacent writes, then equal-sized rows at a constant positive stride.
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
    let mut rows: Vec<ProgramBindingTransferLayout> = Vec::new();
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
            row.destination_stride_bytes = row.row_bytes;
            row.source_writes.end = index + 1;
        } else {
            rows.push(ProgramBindingTransferLayout {
                destination_offset_bytes: write.offset_bytes,
                destination_stride_bytes: write.length_bytes,
                row_bytes: write.length_bytes,
                row_count: 1,
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
        let mut transfer = rows[index].clone();
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
            transfer.source_writes.end = next.source_writes.end;
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
                transfer.source_writes.end = next.source_writes.end;
            }
        }
        index += transfer.row_count;
        transfers.push(transfer);
    }
    Ok(transfers)
}

#[cfg(test)]
mod tests;
