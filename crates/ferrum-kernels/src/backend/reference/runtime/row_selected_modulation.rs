use super::*;
use ferrum_runtime::RowSelectedModulationBf16F32Geometry;
pub(crate) struct ReferenceRowSelectedModulationLaunch {
    pub input: ReferenceBufferRegion,
    pub parameters: ReferenceBufferRegion,
    pub output: ReferenceBufferRegion,
    pub geometry: RowSelectedModulationBf16F32Geometry,
}
impl ReferenceRowSelectedModulationLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = &self.geometry;
        let parameter_elements = g
            .parameter_count()
            .checked_mul(g.columns())
            .and_then(|n| n.checked_mul(2));
        for (region, dtype, elements, scalar) in [
            (&self.input, ElementType::Bf16, Some(g.elements()), 2),
            (&self.parameters, ElementType::F32, parameter_elements, 4),
            (&self.output, ElementType::F32, Some(g.elements()), 4),
        ] {
            region.validate_range()?;
            if region.element_type != dtype
                || elements.and_then(|n| n.checked_mul(scalar)) != Some(region.length_bytes as u64)
                || (region.offset_bytes - region.allocation.logical_offset_bytes)
                    % (scalar as usize)
                    != 0
            {
                return Err(invalid(
                    "requires exact scalar-aligned BF16/F32 physical extents",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.parameters) {
            return Err(invalid("output overlaps an input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.elements(),
            self.geometry.logical_arithmetic_operations(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.parameters, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceRowSelectedModulationLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.parameters, &launch.output]);
    let g = &launch.geometry;
    let columns = g.columns() as usize;
    for row in 0..g.rows() as usize {
        let parameter = g.parameter_row(row as u64).expect("validated row routing") as usize;
        for c in 0..columns {
            let x = bf16::from_bits(storage.read_bits(0, row * columns + c)).to_f32();
            let shift = storage.read_f32(1, parameter * 2 * columns + c);
            let scale = storage.read_f32(1, parameter * 2 * columns + columns + c);
            let factor = 1.0_f32 + scale;
            let product = x * factor;
            let output = product + shift;
            storage.write_f32(2, row * columns + c, output);
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("row-selected modulation: {}", reason.into()))
}
#[cfg(test)]
mod tests;
