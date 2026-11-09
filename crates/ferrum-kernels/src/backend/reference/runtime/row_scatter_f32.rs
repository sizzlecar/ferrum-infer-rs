//! Selected rows only participate in F32 addition; replacement and untouched
//! rows use direct bit transport over the existing retained device owners.
use super::*;
use ferrum_runtime::{RowScatterF32Geometry, RowScatterF32Op};
pub(crate) struct ReferenceRowScatterF32Launch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) updates: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: RowScatterF32Geometry,
    pub(crate) kind: RowScatterF32Op,
}
impl ReferenceRowScatterF32Launch {
    pub(crate) fn name(&self) -> &'static str {
        match self.kind {
            RowScatterF32Op::Replace => "vnext_row_scatter_replace_f32",
            RowScatterF32Op::Add => "vnext_row_scatter_add_f32",
        }
    }
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (r, elements) in [
            (&self.input, self.geometry.elements()),
            (
                &self.updates,
                self.geometry.update_rows() * self.geometry.columns(),
            ),
            (&self.output, self.geometry.elements()),
        ] {
            r.validate_range()?;
            if r.element_type != ElementType::F32
                || elements.checked_mul(4) != Some(r.length_bytes as u64)
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "F32 scatter original dtype/extent/alignment differs",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.updates) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "F32 scatter cannot alias either input",
            ));
        }
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(runtime)?;
        self.updates.validate_runtime(runtime)?;
        self.output.validate_runtime(runtime)?;
        self.validate()
    }
}
impl ReferenceDeviceCommand {
    pub(crate) fn row_scatter_f32_tensor(
        launch: ReferenceRowScatterF32Launch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: launch.name(),
            logical_work: Some(
                DeviceCommandLogicalWork::tensor(
                    launch.geometry.elements(),
                    launch.geometry.logical_operations(launch.kind),
                )
                .map_err(|e| ReferenceDeviceRuntimeError::contract(e.to_string()))?,
            ),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::RowScatterF32 { launch },
        })
    }
}
pub(super) fn execute(launch: &ReferenceRowScatterF32Launch) {
    let mut s = AllocationLocks::new([&launch.input, &launch.updates, &launch.output]);
    let columns = launch.geometry.columns() as usize;
    for row in 0..launch.geometry.rows() as usize {
        let update = launch.geometry.update_row(row as u64);
        for column in 0..columns {
            let i = row * columns + column;
            let value = match update {
                Some(update) => {
                    let u = s.read_f32(1, update as usize * columns + column);
                    match launch.kind {
                        RowScatterF32Op::Replace => u,
                        RowScatterF32Op::Add => s.read_f32(0, i) + u,
                    }
                }
                None => s.read_f32(0, i),
            };
            s.write_f32(2, i, value);
        }
    }
}

#[cfg(test)]
mod tests;
