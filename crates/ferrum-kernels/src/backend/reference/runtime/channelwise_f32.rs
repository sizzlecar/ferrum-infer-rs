use super::*;
use ferrum_runtime::{F32ChannelwiseGeometry, F32ChannelwiseOp};
pub(crate) struct ReferenceF32ChannelwiseLaunch {
    pub inputs: Vec<ReferenceBufferRegion>,
    pub output: ReferenceBufferRegion,
    pub geometry: F32ChannelwiseGeometry,
}
impl ReferenceF32ChannelwiseLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.inputs.len() != self.geometry.kind().input_count() {
            return Err(invalid("input region count differs"));
        }
        for (index, region) in self
            .inputs
            .iter()
            .enumerate()
            .chain(std::iter::once((self.inputs.len(), &self.output)))
        {
            region.validate_range()?;
            let elements = if index == self.inputs.len() {
                Some(self.geometry.elements())
            } else {
                self.geometry.input_elements(index)
            };
            if region.element_type != ElementType::F32
                || elements.and_then(|n| n.checked_mul(4)) != Some(region.length_bytes as u64)
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(invalid("requires exact aligned F32 physical extents"));
            }
        }
        if self.inputs.iter().any(|i| self.output.overlaps(i)) {
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
        for region in self.inputs.iter().chain(std::iter::once(&self.output)) {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceF32ChannelwiseLaunch) {
    match launch.inputs.as_slice() {
        [x] => execute_regions([x, &launch.output], launch.geometry),
        [x, bias] => execute_regions([x, bias, &launch.output], launch.geometry),
        [base, branch, scale] => {
            execute_regions([base, branch, scale, &launch.output], launch.geometry)
        }
        _ => unreachable!("input count was preflighted"),
    }
}
fn execute_regions<const N: usize>(
    regions: [&ReferenceBufferRegion; N],
    g: F32ChannelwiseGeometry,
) {
    let mut storage = AllocationLocks::new(regions);
    let columns = g.columns() as usize;
    for i in 0..g.elements() as usize {
        let column = i % columns;
        let value = match g.kind() {
            F32ChannelwiseOp::GroupedColumnSlice => {
                let source = (i / columns) * g.source_columns() as usize
                    + (column / g.column_count() as usize) * g.group_width() as usize
                    + g.column_start() as usize
                    + column % g.column_count() as usize;
                storage.read_f32(0, source)
            }
            F32ChannelwiseOp::BiasAdd => storage.read_f32(0, i) + storage.read_f32(1, column),
            F32ChannelwiseOp::GroupedColumnMean => {
                let base = (i / columns) * g.source_columns() as usize;
                let groups = g.source_columns() / g.group_width();
                let mut sum = 0f32;
                for group in 0..groups {
                    sum += storage.read_f32(0, base + group as usize * columns + column);
                }
                sum / groups as f32
            }
            F32ChannelwiseOp::AdaptiveAvgPool => {
                let base = (i / columns) * g.source_columns() as usize;
                let start = column as u64 * g.source_columns() / g.columns();
                let end = ((column as u64 + 1) * g.source_columns()).div_ceil(g.columns());
                let mut sum = 0f32;
                for j in start..end {
                    sum += storage.read_f32(0, base + j as usize);
                }
                sum / (end - start) as f32
            }
            F32ChannelwiseOp::ScaledResidual => storage
                .read_f32(1, i)
                .mul_add(storage.read_f32(2, column), storage.read_f32(0, i)),
            F32ChannelwiseOp::ScaledResidualSeparate => {
                let product = storage.read_f32(1, i) * storage.read_f32(2, column);
                storage.read_f32(0, i) + product
            }
            F32ChannelwiseOp::NanToNum => {
                let bits = storage.read_f32(0, i).to_bits();
                let magnitude = bits & 0x7fffffff;
                f32::from_bits(if magnitude > 0x7f800000 {
                    0
                } else if magnitude == 0x7f800000 {
                    (bits & 0x80000000) | 0x7f7fffff
                } else {
                    bits
                })
            }
        };
        storage.write_f32(N - 1, i, value);
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("F32 channelwise: {}", reason.into()))
}
#[cfg(test)]
mod tests;
