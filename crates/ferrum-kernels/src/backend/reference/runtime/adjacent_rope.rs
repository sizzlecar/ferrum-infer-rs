//! Exact staged FP32 multiplication/FMA with a final BF16 RNE conversion.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};

#[derive(Clone, Copy)]
pub(crate) struct ReferenceAdjacentRopeShape {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) sequence: usize,
    pub(crate) head_dim: usize,
}
impl ReferenceAdjacentRopeShape {
    pub(crate) fn byte_extents(self) -> Result<(usize, usize), ReferenceDeviceRuntimeError> {
        if self.rows == 0
            || self.columns == 0
            || self.sequence == 0
            || self.head_dim == 0
            || self.head_dim % 2 != 0
            || self.rows % self.sequence != 0
            || self.columns % self.head_dim != 0
        {
            return Err(invalid("invalid sequence/head partition"));
        }
        let matrix = self
            .rows
            .checked_mul(self.columns)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("matrix byte extent overflows"))?;
        let frequencies = self
            .sequence
            .checked_mul(self.head_dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("frequency byte extent overflows"))?;
        Ok((matrix, frequencies))
    }
    pub(crate) fn logical_work(
        self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.byte_extents()?;
        let rows = u64::try_from(self.rows).map_err(|_| invalid("rows exceed u64"))?;
        let columns = u64::try_from(self.columns).map_err(|_| invalid("columns exceed u64"))?;
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        // Per pair: two multiply, two fused multiply/add (=four scalar
        // operations), one negate. BF16 conversion excluded; not hardware FLOPs.
        let operations = (elements / 2)
            .checked_mul(7)
            .ok_or_else(|| invalid("logical arithmetic overflows"))?;
        DeviceCommandLogicalWork::tensor(elements, operations).map_err(|e| invalid(e.to_string()))
    }
}
pub(crate) struct ReferenceAdjacentRopeLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) frequencies: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceAdjacentRopeShape,
}
impl ReferenceAdjacentRopeLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (matrix, frequencies) = self.shape.byte_extents()?;
        for (region, bytes, dtype, alignment) in [
            (&self.input, matrix, ElementType::Bf16, 2),
            (&self.frequencies, frequencies, ElementType::F32, 4),
            (&self.output, matrix, ElementType::Bf16, 2),
        ] {
            region.validate_range()?;
            if region.element_type != dtype
                || region.length_bytes != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % alignment != 0
            {
                return Err(invalid(
                    "region dtype, exact bytes or scalar alignment differs",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.frequencies) {
            return Err(invalid("output overlaps a read input"));
        }
        self.shape.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.frequencies, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceAdjacentRopeLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.frequencies, &launch.output]);
    let shape = launch.shape;
    for row in 0..shape.rows {
        let frequency_row = (row % shape.sequence) * shape.head_dim;
        for head in 0..shape.columns / shape.head_dim {
            let start = row * shape.columns + head * shape.head_dim;
            for pair in 0..shape.head_dim / 2 {
                let index = start + pair * 2;
                let a = bf16::from_bits(storage.read_bits(0, index)).to_f32();
                let b = bf16::from_bits(storage.read_bits(0, index + 1)).to_f32();
                let c = storage.read_f32(1, frequency_row + pair * 2);
                let s = storage.read_f32(1, frequency_row + pair * 2 + 1);
                // Explicit f32 temporaries round the other product first.
                // mul_add guarantees one rounding of the fused product+sum.
                let negative_product = (-b) * s;
                let positive_product = a * s;
                let real = a.mul_add(c, negative_product);
                let imaginary = b.mul_add(c, positive_product);
                storage.write_bits(2, index, bf16::from_f32(real).to_bits());
                storage.write_bits(2, index + 1, bf16::from_f32(imaginary).to_bits());
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "reference BF16 adjacent RoPE: {}",
        reason.into()
    ))
}
#[cfg(test)]
mod tests;
