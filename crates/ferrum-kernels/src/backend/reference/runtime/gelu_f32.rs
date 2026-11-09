//! Full-F32 GELU over retained physical owners, with separate F32 arithmetic.
use super::*;
use ferrum_runtime::{GeluF32Geometry, GeluF32Variant};

pub(crate) struct ReferenceGeluF32Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: GeluF32Geometry,
    pub(crate) variant: GeluF32Variant,
}
impl ReferenceGeluF32Launch {
    pub(crate) fn name(&self) -> &'static str {
        match self.variant {
            GeluF32Variant::Tanh => "vnext_gelu_tanh_f32",
            GeluF32Variant::Erf => "vnext_gelu_erf_f32",
        }
    }
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        self.geometry
            .logical_operations(self.variant)
            .map_err(|e| ReferenceDeviceRuntimeError::contract(e.to_string()))?;
        for r in [&self.input, &self.output] {
            r.validate_range()?;
            if r.element_type != ElementType::F32
                || self.geometry.elements().checked_mul(4) != Some(r.length_bytes as u64)
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "GELU requires exact aligned F32 extents",
                ));
            }
        }
        if self.output.overlaps(&self.input) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "GELU output aliases its input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(runtime)?;
        self.output.validate_runtime(runtime)?;
        self.validate()
    }
}
impl ReferenceDeviceCommand {
    pub(crate) fn gelu_f32_tensor(
        launch: ReferenceGeluF32Launch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        let work = launch
            .geometry
            .logical_operations(launch.variant)
            .map_err(|e| ReferenceDeviceRuntimeError::contract(e.to_string()))?;
        Ok(Self {
            operation: launch.name(),
            logical_work: Some(
                DeviceCommandLogicalWork::tensor(launch.geometry.elements(), work)
                    .map_err(|e| ReferenceDeviceRuntimeError::contract(e.to_string()))?,
            ),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::GeluF32 { launch },
        })
    }
}
/// These fixed F32 constants and operation order deliberately match the
/// independently declared contract; libm supplies only the transcendental.
pub(super) fn scalar(x: f32, variant: GeluF32Variant) -> f32 {
    let activation = match variant {
        GeluF32Variant::Tanh => {
            let square = x * x;
            let cube = square * x;
            let cubic_term = f32::from_bits(0x3d372713) * cube;
            let polynomial = x + cubic_term;
            let inner = f32::from_bits(0x3f4c422a) * polynomial;
            libm::tanhf(inner)
        }
        GeluF32Variant::Erf => libm::erff(x * f32::from_bits(0x3f3504f3)),
    };
    let left = 0.5_f32 * x;
    let right = 1.0_f32 + activation;
    left * right
}
pub(super) fn execute(launch: &ReferenceGeluF32Launch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    for i in 0..launch.geometry.elements() as usize {
        let y = scalar(storage.read_f32(0, i), launch.variant);
        storage.write_f32(1, i, y);
    }
}

#[cfg(test)]
mod tests;
