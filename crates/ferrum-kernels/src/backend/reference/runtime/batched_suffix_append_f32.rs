use super::*;
use ferrum_runtime::BatchedSuffixAppendF32Geometry;
pub(crate) struct ReferenceBatchedSuffixAppendF32Launch {
    pub input: ReferenceBufferRegion,
    pub suffix: ReferenceBufferRegion,
    pub output: ReferenceBufferRegion,
    pub geometry: BatchedSuffixAppendF32Geometry,
}
impl ReferenceBatchedSuffixAppendF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, n) in [
            (&self.input, self.geometry.input_elements()),
            (&self.suffix, self.geometry.suffix_elements()),
            (&self.output, self.geometry.elements()),
        ] {
            r.validate_range()?;
            if r.element_type != ElementType::F32
                || n.checked_mul(4) != Some(r.length_bytes as u64)
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(invalid("requires exact scalar-aligned F32 physical ranges"));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.suffix) {
            return Err(invalid("output overlaps an input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(self.geometry.elements(), 0)
            .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.suffix, &self.output] {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(l: &ReferenceBatchedSuffixAppendF32Launch) {
    let g = l.geometry;
    let mut storage = AllocationLocks::new([&l.input, &l.suffix, &l.output]);
    for b in 0..g.batch() {
        for r in 0..g.rows_per_batch() {
            for c in 0..g.columns() {
                let v = if r < g.patch_rows() {
                    storage.read_f32(0, ((b * g.patch_rows() + r) * g.columns() + c) as usize)
                } else if r < g.patch_rows() + g.suffix_rows() {
                    storage.read_f32(1, ((r - g.patch_rows()) * g.columns() + c) as usize)
                } else {
                    0.0
                };
                storage.write_f32(
                    2,
                    ((b * g.rows_per_batch() + r) * g.columns() + c) as usize,
                    v,
                );
            }
        }
    }
}
fn invalid(s: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("F32 batched suffix append: {}", s.into()))
}
#[cfg(test)]
mod tests;
