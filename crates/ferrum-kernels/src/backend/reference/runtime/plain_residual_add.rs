//! Plain BF16 residual addition over admitted storage.
use super::{
    bf16, AllocationLocks, ElementType, ReferenceActivationShape, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};

pub(crate) struct ReferencePlainResidualAddLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) update: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceActivationShape,
}
impl ReferencePlainResidualAddLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        self.shape.logical_work(1)?;
        let bytes = self
            .shape
            .rows
            .checked_mul(self.shape.width)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("byte extent overflows"))?;
        for region in [&self.input, &self.update, &self.output] {
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
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.update) {
            return Err(invalid("output overlaps a read operand"));
        }
        self.shape.logical_work(1)?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.update, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferencePlainResidualAddLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.update, &launch.output]);
    for index in 0..launch.shape.rows * launch.shape.width {
        let x = bf16::from_bits(storage.read_bits(0, index)).to_f32();
        let update = bf16::from_bits(storage.read_bits(1, index)).to_f32();
        storage.write_bits(2, index, bf16::from_f32(x + update).to_bits());
    }
}
fn invalid(reason: &str) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("reference BF16 plain residual add: {reason}"))
}
#[cfg(test)]
mod tests;
