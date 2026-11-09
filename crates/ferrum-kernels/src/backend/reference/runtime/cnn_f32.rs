use super::*;
use ferrum_runtime::{F32CnnGeometry, F32CnnOp};
pub(crate) struct ReferenceCnnF32Launch {
    pub inputs: Vec<ReferenceBufferRegion>,
    pub output: ReferenceBufferRegion,
    pub geometry: F32CnnGeometry,
}
impl ReferenceCnnF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.inputs.len() != self.geometry.kind().input_count() {
            return Err(invalid("operand count differs"));
        }
        for (i, r) in self
            .inputs
            .iter()
            .enumerate()
            .chain(std::iter::once((self.inputs.len(), &self.output)))
        {
            r.validate_range()?;
            let count = if i == self.inputs.len() {
                Some(self.geometry.elements())
            } else {
                self.geometry.input_elements(i)
            };
            if r.element_type != ElementType::F32
                || count.and_then(|n| n.checked_mul(4)) != Some(r.length_bytes as u64)
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(invalid("requires exact scalar-aligned F32 ranges"));
            }
        }
        if self.inputs.iter().any(|r| self.output.overlaps(r)) {
            return Err(invalid("output overlaps input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in self.inputs.iter().chain(std::iter::once(&self.output)) {
            r.validate_runtime(runtime)?;
        }
        self.validate()
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
}
fn tree(s: &mut [f32; 256]) -> f32 {
    let mut n = 128;
    while n > 0 {
        for i in 0..n {
            s[i] += s[i + n];
        }
        n /= 2;
    }
    s[0]
}
pub(super) fn execute(l: &ReferenceCnnF32Launch) {
    match l.inputs.as_slice() {
        [x, y] => execute_regions([x, y, &l.output], l.geometry),
        [x, w, b] => execute_regions([x, w, b, &l.output], l.geometry),
        _ => unreachable!("validated operands"),
    }
}
fn execute_regions<const N: usize>(regions: [&ReferenceBufferRegion; N], geometry: F32CnnGeometry) {
    let mut s = AllocationLocks::new(regions);
    match geometry {
        F32CnnGeometry::ResidualAdd(g) => {
            for i in 0..g.elements() as usize {
                let v = s.read_f32(0, i) + s.read_f32(1, i);
                s.write_f32(N - 1, i, v);
            }
        }
        F32CnnGeometry::Conv2d(g) => {
            let [top, _, left, _] = g.padding();
            let [sh, sw] = g.stride();
            let [dh, dw] = g.dilation();
            let channels = g.input_channels() / g.groups();
            let outputs = g.output_channels() / g.groups();
            for n in 0..g.batch() {
                for o in 0..g.output_channels() {
                    for y in 0..g.output_height() {
                        for x in 0..g.output_width() {
                            let mut sum = 0.0f32;
                            for c in 0..channels {
                                for ky in 0..g.kernel_height() {
                                    for kx in 0..g.kernel_width() {
                                        let v = if let Some((iy, ix)) = g.source_spatial(
                                            i128::from(y * sh + ky * dh) - i128::from(top),
                                            i128::from(x * sw + kx * dw) - i128::from(left),
                                        ) {
                                            s.read_f32(
                                                0,
                                                (((n * g.input_channels()
                                                    + o / outputs * channels
                                                    + c)
                                                    * g.input_height()
                                                    + iy)
                                                    * g.input_width()
                                                    + ix)
                                                    as usize,
                                            )
                                        } else {
                                            0.
                                        };
                                        let w = s.read_f32(
                                            1,
                                            (((o * channels + c) * g.kernel_height() + ky)
                                                * g.kernel_width()
                                                + kx)
                                                as usize,
                                        );
                                        let product = v * w;
                                        sum += product;
                                    }
                                }
                            }
                            let biased = sum + s.read_f32(2, o as usize);
                            s.write_f32(
                                N - 1,
                                (((n * g.output_channels() + o) * g.output_height() + y)
                                    * g.output_width()
                                    + x) as usize,
                                biased,
                            );
                        }
                    }
                }
            }
        }
        F32CnnGeometry::GroupNormSilu(g) => {
            let [batch, channels, height, width] = g.shape();
            let spatial = height * width;
            let group = g.group_elements() as usize;
            for ng in 0..batch * g.groups() {
                let start = ng as usize * group;
                let mut sums = [0.0f32; 256];
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for i in (lane..group).step_by(256) {
                        *sum += s.read_f32(0, start + i);
                    }
                }
                let mean = tree(&mut sums) / group as f32;
                sums.fill(0.);
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for i in (lane..group).step_by(256) {
                        let centered = s.read_f32(0, start + i) - mean;
                        let square = centered * centered;
                        *sum += square;
                    }
                }
                let variance = tree(&mut sums) / group as f32;
                let inverse = 1. / (variance + g.effective_epsilon()).sqrt();
                for i in 0..group {
                    let channel = ((start + i) as u64 / spatial % channels) as usize;
                    let centered = s.read_f32(0, start + i) - mean;
                    let normalized = centered * inverse;
                    let scaled = normalized * s.read_f32(1, channel);
                    let affine = scaled + s.read_f32(2, channel);
                    let denominator = 1. + libm::expf(-affine);
                    let value = affine / denominator;
                    s.write_f32(N - 1, start + i, value);
                }
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("Reference F32 CNN: {}", reason.into()))
}
#[cfg(test)]
mod tests;
