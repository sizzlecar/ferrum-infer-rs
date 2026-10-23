//! Half-split RoPE with two BF16 product boundaries before BF16 addition.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};

#[derive(Clone, Copy)]
pub(crate) struct ReferenceHalfSplitRopeShape {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) head_dim: usize,
}
impl ReferenceHalfSplitRopeShape {
    pub(crate) fn byte_extents(self) -> Result<(usize, usize), ReferenceDeviceRuntimeError> {
        if self.rows == 0
            || self.columns == 0
            || self.head_dim == 0
            || self.head_dim % 2 != 0
            || self.columns % self.head_dim != 0
        {
            return Err(invalid("invalid positive even head partition"));
        }
        let bytes = |width| {
            self.rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| invalid("byte extent overflows"))
        };
        Ok((bytes(self.columns)?, bytes(self.head_dim)?))
    }
    pub(crate) fn logical_work(
        self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.byte_extents()?;
        let elements = u64::try_from(self.rows)
            .ok()
            .zip(u64::try_from(self.columns).ok())
            .and_then(|(r, c)| r.checked_mul(c))
            .ok_or_else(|| invalid("element count overflows"))?;
        // Per pair: four products, two sums, one sign negation. BF16
        // conversions excluded; this is nominal scalar work, not hardware FLOPs.
        let operations = (elements / 2)
            .checked_mul(7)
            .ok_or_else(|| invalid("logical work overflows"))?;
        DeviceCommandLogicalWork::tensor(elements, operations).map_err(|e| invalid(e.to_string()))
    }
}
pub(crate) struct ReferenceHalfSplitRopeLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) cosine: ReferenceBufferRegion,
    pub(crate) sine: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceHalfSplitRopeShape,
}
impl ReferenceHalfSplitRopeLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (matrix, frequency) = self.shape.byte_extents()?;
        for (region, bytes) in [
            (&self.input, matrix),
            (&self.cosine, frequency),
            (&self.sine, frequency),
            (&self.output, matrix),
        ] {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid(
                    "BF16 dtype, exact bytes or scalar alignment differs",
                ));
            }
        }
        if [&self.input, &self.cosine, &self.sine]
            .into_iter()
            .any(|r| self.output.overlaps(r))
        {
            return Err(invalid("output overlaps a read operand"));
        }
        self.shape.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.cosine, &self.sine, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceHalfSplitRopeLaunch) {
    let mut storage =
        AllocationLocks::new([&launch.input, &launch.cosine, &launch.sine, &launch.output]);
    let shape = launch.shape;
    for row in 0..shape.rows {
        let frequency = row * shape.head_dim;
        for head in 0..shape.columns / shape.head_dim {
            let start = row * shape.columns + head * shape.head_dim;
            for i in 0..shape.head_dim / 2 {
                let j = i + shape.head_dim / 2;
                let a = bf16::from_bits(storage.read_bits(0, start + i)).to_f32();
                let b = bf16::from_bits(storage.read_bits(0, start + j)).to_f32();
                let ca = bf16::from_bits(storage.read_bits(1, frequency + i)).to_f32();
                let cb = bf16::from_bits(storage.read_bits(1, frequency + j)).to_f32();
                let sa = bf16::from_bits(storage.read_bits(2, frequency + i)).to_f32();
                let sb = bf16::from_bits(storage.read_bits(2, frequency + j)).to_f32();
                let left_a = bf16::from_f32(a * ca).to_f32();
                let right_a = bf16::from_f32((-b) * sa).to_f32();
                let left_b = bf16::from_f32(b * cb).to_f32();
                let right_b = bf16::from_f32(a * sb).to_f32();
                storage.write_bits(3, start + i, bf16::from_f32(left_a + right_a).to_bits());
                storage.write_bits(3, start + j, bf16::from_f32(left_b + right_b).to_bits());
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "reference BF16 half-split RoPE: {}",
        reason.into()
    ))
}
#[cfg(test)]
mod tests;
