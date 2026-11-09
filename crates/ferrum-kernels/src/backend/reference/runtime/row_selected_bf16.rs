use super::*;
use ferrum_runtime::{RowSelectedBf16Geometry, RowSelectedBf16Kind};
pub(crate) struct ReferenceRowSelectedBf16Launch {
    pub input: ReferenceBufferRegion,
    pub parameters: ReferenceBufferRegion,
    pub update: Option<ReferenceBufferRegion>,
    pub output: ReferenceBufferRegion,
    pub geometry: RowSelectedBf16Geometry,
}
impl ReferenceRowSelectedBf16Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = &self.geometry;
        if self.update.is_some() != (g.kind() == RowSelectedBf16Kind::Addcmul) {
            return Err(invalid("update presence differs from selected operation"));
        }
        let table = g
            .parameter_count()
            .checked_mul(g.parameter_columns())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("parameter extent overflows"))?;
        let data = g
            .elements()
            .checked_mul(2)
            .ok_or_else(|| invalid("activation extent overflows"))?;
        for (region, bytes) in [
            (&self.input, data),
            (&self.parameters, table),
            (&self.output, data),
        ]
        .into_iter()
        .chain(self.update.iter().map(|r| (r, data)))
        {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes as u64 != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid(
                    "requires exact scalar-aligned BF16 physical extents",
                ));
            }
        }
        if [&self.input, &self.parameters]
            .into_iter()
            .chain(self.update.iter())
            .any(|r| self.output.overlaps(r))
        {
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
        for r in [&self.input, &self.parameters, &self.output]
            .into_iter()
            .chain(self.update.iter())
        {
            r.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceRowSelectedBf16Launch) {
    let mut storage = AllocationLocks::new([
        &launch.input,
        &launch.parameters,
        launch.update.as_ref().unwrap_or(&launch.input),
        &launch.output,
    ]);
    let g = &launch.geometry;
    let h = g.columns() as usize;
    let w = g.parameter_columns() as usize;
    for row in 0..g.rows() as usize {
        let p = g.parameter_row(row as u64).expect("validated routing") as usize;
        for c in 0..h {
            let i = row * h + c;
            let x = bf16::from_bits(storage.read_bits(0, i)).to_f32();
            let value = match g.kind() {
                RowSelectedBf16Kind::Modulation => {
                    let scale = bf16::from_bits(
                        storage.read_bits(1, p * w + g.scale_column() as usize + c),
                    )
                    .to_f32();
                    let shift = bf16::from_bits(
                        storage.read_bits(1, p * w + g.shift_column() as usize + c),
                    )
                    .to_f32();
                    let factor = bf16::from_f32(1.0f32 + scale).to_f32();
                    let product = bf16::from_f32(x * factor).to_f32();
                    product + shift
                }
                RowSelectedBf16Kind::Addcmul => {
                    let gate =
                        bf16::from_bits(storage.read_bits(1, p * w + g.gate_column() as usize + c))
                            .to_f32();
                    let update = bf16::from_bits(storage.read_bits(2, i)).to_f32();
                    let product = gate * update;
                    x + product
                }
            };
            storage.write_bits(3, i, bf16::from_f32(value).to_bits());
        }
    }
}
fn invalid(s: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("row-selected BF16: {}", s.into()))
}
#[cfg(test)]
mod tests;
