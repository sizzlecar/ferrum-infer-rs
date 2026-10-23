//! Pure bit transport over admitted storage. No temporary tensor or payload copy.
use super::{
    AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{BatchedRowConcatBf16Geometry, RowGatherBf16Geometry};

pub(crate) struct ReferenceRowGatherLaunch {
    pub(crate) source: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: RowGatherBf16Geometry,
}
pub(crate) struct ReferenceBatchedRowConcatLaunch {
    pub(crate) left: ReferenceBufferRegion,
    pub(crate) right: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: BatchedRowConcatBf16Geometry,
}

pub(crate) fn checked_gather_bytes(
    g: &RowGatherBf16Geometry,
) -> Result<(usize, usize), ReferenceDeviceRuntimeError> {
    Ok((
        bytes(g.source_rows().checked_mul(g.source_width()))?,
        bytes(Some(g.element_count()))?,
    ))
}
pub(crate) fn checked_concat_bytes(
    g: BatchedRowConcatBf16Geometry,
) -> Result<(usize, usize, usize), ReferenceDeviceRuntimeError> {
    Ok((
        bytes(
            g.batch()
                .checked_mul(g.left_rows_per_batch())
                .and_then(|r| r.checked_mul(g.columns())),
        )?,
        bytes(
            g.batch()
                .checked_mul(g.right_rows_per_batch())
                .and_then(|r| r.checked_mul(g.columns())),
        )?,
        bytes(Some(g.element_count()))?,
    ))
}
fn bytes(elements: Option<u64>) -> Result<usize, ReferenceDeviceRuntimeError> {
    elements
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("byte extent exceeds backend address space"))
}
fn region(region: &ReferenceBufferRegion, bytes: usize) -> Result<(), ReferenceDeviceRuntimeError> {
    region.validate_range()?;
    if region.element_type != ElementType::Bf16
        || region.length_bytes != bytes
        || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
    {
        return Err(invalid("requires exact BF16 ranges and scalar alignment"));
    }
    Ok(())
}
impl ReferenceRowGatherLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (source, output) = checked_gather_bytes(&self.geometry)?;
        region(&self.source, source)?;
        region(&self.output, output)?;
        if self.output.overlaps(&self.source) {
            return Err(invalid("gather output overlaps its source"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(self.geometry.element_count(), 0)
            .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.source.validate_runtime(runtime)?;
        self.output.validate_runtime(runtime)?;
        self.validate()
    }
}
impl ReferenceBatchedRowConcatLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (left, right, output) = checked_concat_bytes(self.geometry)?;
        region(&self.left, left)?;
        region(&self.right, right)?;
        region(&self.output, output)?;
        if self.output.overlaps(&self.left) || self.output.overlaps(&self.right) {
            return Err(invalid("concat output overlaps a read source"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(self.geometry.element_count(), 0)
            .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.left, &self.right, &self.output] {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}

pub(super) fn execute_gather(launch: &ReferenceRowGatherLaunch) {
    let g = &launch.geometry;
    let mut storage = AllocationLocks::new([&launch.source, &launch.output]);
    let mut destination = 0_usize;
    // Geometry checked maximum source row and both tensor extents before any
    // execution. All arithmetic is bounded by those actual tensor sizes.
    for batch in 0..g.batch() {
        for run in 0..g.run_lengths().len() {
            let start = g.source_starts()[run] + batch * g.source_batch_strides()[run];
            for row in 0..g.run_lengths()[run] {
                let source = (start + row * g.source_row_strides()[run]) * g.source_width()
                    + g.column_start();
                for column in 0..g.column_count() {
                    let value = storage.read_bits(0, (source + column) as usize);
                    storage.write_bits(1, destination, value);
                    destination += 1;
                }
            }
        }
    }
}
pub(super) fn execute_concat(launch: &ReferenceBatchedRowConcatLaunch) {
    let g = launch.geometry;
    let mut storage = AllocationLocks::new([&launch.left, &launch.right, &launch.output]);
    let mut destination = 0_usize;
    for batch in 0..g.batch() {
        for (role, rows) in [(0, g.left_rows_per_batch()), (1, g.right_rows_per_batch())] {
            let start = batch * rows * g.columns();
            for index in 0..rows * g.columns() {
                let value = storage.read_bits(role, (start + index) as usize);
                storage.write_bits(2, destination, value);
                destination += 1;
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "reference BF16 row transport: {}",
        reason.into()
    ))
}
#[cfg(test)]
mod tests;
