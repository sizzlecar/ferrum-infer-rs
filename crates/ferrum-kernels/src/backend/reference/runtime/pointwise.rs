//! BF16 pointwise execution with explicit intermediate rounding boundaries.
//!
//! Only structural checks precede submission. Values can be produced by
//! preceding commands, so NaN, infinity and overflow propagate through IEEE
//! arithmetic rather than being reported as DefinitelyNotSubmitted errors.

use ferrum_runtime::RowBroadcast;

use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};

#[derive(Clone, Copy)]
pub(crate) struct ReferencePointwiseShape {
    pub(crate) rows: usize,
    pub(crate) width: usize,
    pub(crate) broadcast: RowBroadcast,
}

impl ReferencePointwiseShape {
    fn byte_extents(self) -> Result<(usize, usize), ReferenceDeviceRuntimeError> {
        if self.rows == 0 || self.width == 0 {
            return Err(invalid("reference pointwise dimensions must be positive"));
        }
        let matrix = self
            .rows
            .checked_mul(self.width)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("reference pointwise matrix byte extent overflows"))?;
        let parameter = match self.broadcast {
            RowBroadcast::Elementwise => matrix,
            RowBroadcast::Row => self
                .width
                .checked_mul(2)
                .ok_or_else(|| invalid("reference pointwise parameter byte extent overflows"))?,
        };
        Ok((matrix, parameter))
    }

    pub(super) fn logical_work(
        self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.byte_extents()?;
        let count = u64::try_from(self.rows)
            .ok()
            .zip(u64::try_from(self.width).ok())
            .and_then(|(rows, width)| rows.checked_mul(width))
            .and_then(|elements| Some((elements, elements.checked_mul(2)?)))
            .ok_or_else(|| invalid("reference pointwise logical work overflows"))?;
        // Each output performs one FP32 add and one FP32 multiply. BF16
        // conversions are excluded; this is not a hardware FLOP estimate.
        DeviceCommandLogicalWork::tensor(count.0, count.1)
            .map_err(|error| invalid(error.to_string()))
    }

    fn parameter_index(self, element: usize) -> usize {
        match self.broadcast {
            RowBroadcast::Elementwise => element,
            RowBroadcast::Row => element % self.width,
        }
    }
}

pub(crate) struct ReferenceModulationScaleLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) scale: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferencePointwiseShape,
}

impl ReferenceModulationScaleLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (matrix, parameter) = self.shape.byte_extents()?;
        validate_region(&self.input, matrix)?;
        validate_region(&self.scale, parameter)?;
        validate_region(&self.output, matrix)?;
        reject_output_alias(&self.output, [&self.input, &self.scale])?;
        self.shape.logical_work()?;
        Ok(())
    }

    pub(super) fn validate_runtime(
        &self,
        runtime_instance: u64,
    ) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.scale, &self.output] {
            region.validate_runtime(runtime_instance)?;
        }
        self.validate()
    }
}

pub(crate) struct ReferenceGatedResidualLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) gate: ReferenceBufferRegion,
    pub(crate) update: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferencePointwiseShape,
}

impl ReferenceGatedResidualLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (matrix, parameter) = self.shape.byte_extents()?;
        validate_region(&self.input, matrix)?;
        validate_region(&self.gate, parameter)?;
        validate_region(&self.update, matrix)?;
        validate_region(&self.output, matrix)?;
        reject_output_alias(&self.output, [&self.input, &self.gate, &self.update])?;
        self.shape.logical_work()?;
        Ok(())
    }

    pub(super) fn validate_runtime(
        &self,
        runtime_instance: u64,
    ) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.gate, &self.update, &self.output] {
            region.validate_runtime(runtime_instance)?;
        }
        self.validate()
    }
}

fn validate_region(
    region: &ReferenceBufferRegion,
    expected_bytes: usize,
) -> Result<(), ReferenceDeviceRuntimeError> {
    region.validate_range()?;
    if region.length_bytes != expected_bytes
        || region.element_type != ElementType::Bf16
        || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
    {
        return Err(invalid(
            "reference pointwise shape, range, alignment or BF16 storage differs",
        ));
    }
    Ok(())
}

fn reject_output_alias<const N: usize>(
    output: &ReferenceBufferRegion,
    inputs: [&ReferenceBufferRegion; N],
) -> Result<(), ReferenceDeviceRuntimeError> {
    if inputs.iter().any(|input| output.overlaps(input)) {
        return Err(invalid(
            "reference pointwise output overlaps a read operand",
        ));
    }
    Ok(())
}

pub(super) fn execute_modulation(launch: &ReferenceModulationScaleLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.scale, &launch.output]);
    for index in 0..launch.shape.rows * launch.shape.width {
        let input = bf16::from_bits(storage.read_bits(0, index)).to_f32();
        let scale =
            bf16::from_bits(storage.read_bits(1, launch.shape.parameter_index(index))).to_f32();
        // B(x * B(1+s)): the inner BF16 round-trip is a semantic boundary,
        // not a request to allocate a broadcast or intermediate tensor.
        let factor = bf16::from_f32(1.0_f32 + scale).to_f32();
        storage.write_bits(2, index, bf16::from_f32(input * factor).to_bits());
    }
}

pub(super) fn execute_gated_residual(launch: &ReferenceGatedResidualLaunch) {
    let mut storage =
        AllocationLocks::new([&launch.input, &launch.gate, &launch.update, &launch.output]);
    for index in 0..launch.shape.rows * launch.shape.width {
        let input = bf16::from_bits(storage.read_bits(0, index)).to_f32();
        let gate =
            bf16::from_bits(storage.read_bits(1, launch.shape.parameter_index(index))).to_f32();
        let update = bf16::from_bits(storage.read_bits(2, index)).to_f32();
        // B(x + B(g*u)): g is already activated; do not add tanh, clamp it,
        // or fuse the product and sum across the inner BF16 boundary.
        let contribution = bf16::from_f32(gate * update).to_f32();
        storage.write_bits(3, index, bf16::from_f32(input + contribution).to_bits());
    }
}

fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(reason)
}

#[cfg(test)]
mod tests;
