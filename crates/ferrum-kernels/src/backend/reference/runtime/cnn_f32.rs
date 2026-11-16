use super::*;
use ferrum_runtime::F32CnnGeometry;
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
        F32CnnGeometry::Conv3d(g) => {
            let [batches, channels, depth, height, width] = g.input();
            let [outputs, group_channels, kd, kh, kw] = g.weight();
            let [_, _, od, oh, ow] = g.output();
            let pad = g.padding();
            let stride = g.stride();
            let dilation = g.dilation();
            let outputs_per_group = outputs / g.groups();
            for n in 0..batches {
                for o in 0..outputs {
                    for z in 0..od {
                        for y in 0..oh {
                            for x in 0..ow {
                                let mut sum = 0.0f32;
                                for c in 0..group_channels {
                                    for kz in 0..kd {
                                        for ky in 0..kh {
                                            for kx in 0..kw {
                                                let coordinates = std::array::from_fn(|a| {
                                                    i128::from([z, y, x][a]) * i128::from(stride[a])
                                                        + i128::from([kz, ky, kx][a])
                                                            * i128::from(dilation[a])
                                                        - i128::from(pad[2 * a])
                                                });
                                                let v = if let Some([iz, iy, ix]) =
                                                    g.source_spatial(coordinates)
                                                {
                                                    let channel =
                                                        o / outputs_per_group * group_channels + c;
                                                    s.read_f32(
                                                        0,
                                                        ((((n * channels + channel) * depth + iz)
                                                            * height
                                                            + iy)
                                                            * width
                                                            + ix)
                                                            as usize,
                                                    )
                                                } else {
                                                    0.0
                                                };
                                                let w = s.read_f32(
                                                    1,
                                                    ((((o * group_channels + c) * kd + kz) * kh
                                                        + ky)
                                                        * kw
                                                        + kx)
                                                        as usize,
                                                );
                                                let product = v * w;
                                                sum += product;
                                            }
                                        }
                                    }
                                }
                                let value = sum + s.read_f32(2, o as usize);
                                s.write_f32(
                                    N - 1,
                                    ((((n * outputs + o) * od + z) * oh + y) * ow + x) as usize,
                                    value,
                                );
                            }
                        }
                    }
                }
            }
        }
        F32CnnGeometry::ResidualAdd(_) | F32CnnGeometry::VolumeResidualAdd(_) => {
            for i in 0..geometry.elements() as usize {
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
        F32CnnGeometry::GroupNormSilu(_) | F32CnnGeometry::FrameGroupNormSilu(_) => {
            let (batch, channels, frames, height, width, groups, epsilon) = match geometry {
                F32CnnGeometry::GroupNormSilu(g) => {
                    let [n, c, h, w] = g.shape();
                    (n, c, 1, h, w, g.groups(), g.effective_epsilon())
                }
                F32CnnGeometry::FrameGroupNormSilu(g) => {
                    let [n, c, t, h, w] = g.shape();
                    (n, c, t, h, w, g.groups(), g.effective_epsilon())
                }
                _ => unreachable!(),
            };
            let spatial = height * width;
            let group = (channels / groups * spatial) as usize;
            for ng in 0..batch * frames * groups {
                let offset = |i: usize| {
                    let n = ng / (frames * groups);
                    let t = (ng / groups) % frames;
                    let c = (ng % groups) * (channels / groups) + (i as u64) / spatial;
                    (((n * channels + c) * frames + t) * spatial + (i as u64) % spatial) as usize
                };
                let mut sums = [0.0f32; 256];
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for i in (lane..group).step_by(256) {
                        *sum += s.read_f32(0, offset(i));
                    }
                }
                let mean = tree(&mut sums) / group as f32;
                sums.fill(0.);
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for i in (lane..group).step_by(256) {
                        let centered = s.read_f32(0, offset(i)) - mean;
                        let square = centered * centered;
                        *sum += square;
                    }
                }
                let variance = tree(&mut sums) / group as f32;
                let inverse = 1. / (variance + epsilon).sqrt();
                for i in 0..group {
                    let position = offset(i);
                    let channel = (position as u64 / (spatial * frames) % channels) as usize;
                    let centered = s.read_f32(0, position) - mean;
                    let normalized = centered * inverse;
                    let scaled = normalized * s.read_f32(1, channel);
                    let affine = scaled + s.read_f32(2, channel);
                    let denominator = 1. + libm::expf(-affine);
                    let value = affine / denominator;
                    s.write_f32(N - 1, position, value);
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
