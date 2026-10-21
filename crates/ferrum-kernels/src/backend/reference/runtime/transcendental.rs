//! Scalar candidates used by both Reference storage execution and qualification.
//!
//! Each FP32 operation and BF16 conversion is an explicit semantic boundary.
//! The required environment is RN-even with gradual underflow and no FTZ/DAZ.
//! NaN payloads are not part of the profile; zero and infinity signs are.

use half::bf16;

use super::{
    AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReferenceUnaryKind {
    Silu,
    Tanh,
}

impl ReferenceUnaryKind {
    fn operations_per_element(self) -> u64 {
        match self {
            Self::Silu => 4,
            Self::Tanh => 1,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ReferenceActivationShape {
    pub(crate) rows: usize,
    pub(crate) width: usize,
}

impl ReferenceActivationShape {
    fn byte_extent(self) -> Result<usize, ReferenceDeviceRuntimeError> {
        if self.rows == 0 || self.width == 0 {
            return Err(invalid("reference activation dimensions must be positive"));
        }
        self.rows
            .checked_mul(self.width)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| invalid("reference activation byte extent overflows"))
    }

    pub(in crate::backend::reference) fn logical_work(
        self,
        per_element: u64,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.byte_extent()?;
        let elements = u64::try_from(self.rows)
            .ok()
            .zip(u64::try_from(self.width).ok())
            .and_then(|(rows, width)| rows.checked_mul(width))
            .ok_or_else(|| invalid("reference activation element count overflows"))?;
        let operations = elements
            .checked_mul(per_element)
            .ok_or_else(|| invalid("reference activation arithmetic operation count overflows"))?;
        DeviceCommandLogicalWork::tensor(elements, operations)
            .map_err(|error| invalid(error.to_string()))
    }
}

pub(crate) struct ReferenceUnaryLaunch {
    pub(crate) kind: ReferenceUnaryKind,
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceActivationShape,
}

impl ReferenceUnaryLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let bytes = self.shape.byte_extent()?;
        validate_region(&self.input, bytes)?;
        validate_region(&self.output, bytes)?;
        if self.output.overlaps(&self.input) {
            return Err(invalid(
                "reference unary activation output overlaps its input",
            ));
        }
        self.logical_work()?;
        Ok(())
    }

    pub(super) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.shape.logical_work(self.kind.operations_per_element())
    }

    pub(super) fn validate_runtime(
        &self,
        runtime_instance: u64,
    ) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(runtime_instance)?;
        self.output.validate_runtime(runtime_instance)?;
        self.validate()
    }
}

pub(crate) struct ReferenceSiluMulLaunch {
    pub(crate) gate: ReferenceBufferRegion,
    pub(crate) up: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceActivationShape,
}

impl ReferenceSiluMulLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let bytes = self.shape.byte_extent()?;
        for region in [&self.gate, &self.up, &self.output] {
            validate_region(region, bytes)?;
        }
        if self.output.overlaps(&self.gate) || self.output.overlaps(&self.up) {
            return Err(invalid("reference SiLU-mul output overlaps a read operand"));
        }
        self.logical_work()?;
        Ok(())
    }

    pub(super) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.shape.logical_work(5)
    }

    pub(super) fn validate_runtime(
        &self,
        runtime_instance: u64,
    ) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.gate, &self.up, &self.output] {
            region.validate_runtime(runtime_instance)?;
        }
        self.validate()
    }
}

fn validate_region(
    region: &ReferenceBufferRegion,
    bytes: usize,
) -> Result<(), ReferenceDeviceRuntimeError> {
    region.validate_range()?;
    if region.element_type != ElementType::Bf16
        || region.length_bytes != bytes
        || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
    {
        return Err(invalid(
            "reference activation requires exact BF16 range and scalar alignment",
        ));
    }
    Ok(())
}

pub(super) fn execute_unary(launch: &ReferenceUnaryLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    let scalar = match launch.kind {
        ReferenceUnaryKind::Silu => silu_bf16_bits,
        ReferenceUnaryKind::Tanh => tanh_bf16_bits,
    };
    for index in 0..launch.shape.rows * launch.shape.width {
        let output = scalar(storage.read_bits(0, index));
        storage.write_bits(1, index, output);
    }
}

pub(super) fn execute_silu_mul(launch: &ReferenceSiluMulLaunch) {
    let mut storage = AllocationLocks::new([&launch.gate, &launch.up, &launch.output]);
    for index in 0..launch.shape.rows * launch.shape.width {
        let output = silu_mul_bf16_bits(storage.read_bits(0, index), storage.read_bits(1, index));
        storage.write_bits(2, index, output);
    }
}

fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(reason)
}

/// One sign negation, one exp call, one addition and one division are four
/// logical operations. This is not a count of libm's internal arithmetic or
/// hardware FLOPs. In particular, exp(90) overflows in FP32, so SiLU(-90) is
/// negative zero; replacing this path by stable sigmoid changes the profile.
pub(in crate::backend::reference) fn silu_bf16_bits(input: u16) -> u16 {
    let input = bf16::from_bits(input).to_f32();
    let exponential = libm::expf(-input);
    let denominator = 1.0_f32 + exponential;
    bf16::from_f32(input / denominator).to_bits()
}

/// The activation's BF16 round-trip precedes the FP32 product. This adds one
/// logical multiply to unary SiLU without materializing an intermediate tensor.
pub(in crate::backend::reference) fn silu_mul_bf16_bits(gate: u16, up: u16) -> u16 {
    let activated = bf16::from_bits(silu_bf16_bits(gate)).to_f32();
    let up = bf16::from_bits(up).to_f32();
    bf16::from_f32(activated * up).to_bits()
}

/// A tanh call counts as one logical operation regardless of libm's internal
/// approximation. It is not replaced by an exp-based expression in this path.
pub(in crate::backend::reference) fn tanh_bf16_bits(input: u16) -> u16 {
    let input = bf16::from_bits(input).to_f32();
    bf16::from_f32(libm::tanhf(input)).to_bits()
}

#[cfg(test)]
mod tests;
