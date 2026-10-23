//! Direct admitted NCHW storage; no im2col or full-image host temporary.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{ChannelL2NormBf16Geometry, Conv2dBf16Geometry};

pub(crate) struct ReferenceConv2dLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) weight: ReferenceBufferRegion,
    pub(crate) bias: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: Conv2dBf16Geometry,
}
impl ReferenceConv2dLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let g = self.shape;
        let input = extent(&[
            g.batch(),
            g.input_channels(),
            g.input_height(),
            g.input_width(),
        ])?;
        let weight = extent(&[
            g.output_channels(),
            g.input_channels() / g.groups(),
            g.kernel_height(),
            g.kernel_width(),
        ])?;
        for (r, n) in [
            (&self.input, input),
            (&self.weight, weight),
            (&self.bias, extent(&[g.output_channels()])?),
            (&self.output, extent(&[g.element_count()])?),
        ] {
            region(r, n)?;
        }
        for r in [&self.input, &self.weight, &self.bias] {
            if self.output.overlaps(r) {
                return Err(invalid("convolution output aliases an input"));
            }
        }
        // All address terms fit usize, including explicit-padded logical coordinates.
        usize::try_from(g.padded_input_bytes())
            .map_err(|_| invalid("padded coordinates exceed usize"))?;
        for v in g
            .stride()
            .into_iter()
            .chain(g.dilation())
            .chain(g.padding())
        {
            usize::try_from(v).map_err(|_| invalid("coordinate exceeds usize"))?;
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, id: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.weight, &self.bias, &self.output] {
            r.validate_runtime(id)?;
        }
        self.validate()
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        work(
            self.shape.element_count(),
            self.shape.logical_arithmetic_operations(),
        )
    }
}
pub(crate) struct ReferenceChannelL2NormLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) gamma: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ChannelL2NormBf16Geometry,
}
impl ReferenceChannelL2NormLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let matrix = extent(&[self.shape.element_count()])?;
        for (r, n) in [
            (&self.input, matrix),
            (&self.gamma, extent(&[self.shape.channels()])?),
            (&self.output, matrix),
        ] {
            region(r, n)?;
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.gamma) {
            return Err(invalid("channel norm output aliases an input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, id: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for r in [&self.input, &self.gamma, &self.output] {
            r.validate_runtime(id)?;
        }
        self.validate()
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        work(
            self.shape.element_count(),
            self.shape.logical_arithmetic_operations(),
        )
    }
}
fn work(
    elements: u64,
    operations: u64,
) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
    DeviceCommandLogicalWork::tensor(elements, operations).map_err(|e| invalid(e.to_string()))
}
fn extent(dims: &[u64]) -> Result<usize, ReferenceDeviceRuntimeError> {
    dims.iter()
        .try_fold(2_usize, |n, d| n.checked_mul(usize::try_from(*d).ok()?))
        .ok_or_else(|| invalid("byte extent exceeds usize"))
}
fn region(r: &ReferenceBufferRegion, bytes: usize) -> Result<(), ReferenceDeviceRuntimeError> {
    r.validate_range()?;
    if r.element_type != ElementType::Bf16
        || r.length_bytes != bytes
        || (r.offset_bytes - r.allocation.logical_offset_bytes) % 2 != 0
    {
        return Err(invalid("region dtype, exact extent or alignment differs"));
    }
    Ok(())
}

pub(super) fn execute_conv2d(launch: &ReferenceConv2dLaunch) {
    let mut storage =
        AllocationLocks::new([&launch.input, &launch.weight, &launch.bias, &launch.output]);
    let g = launch.shape;
    let (n, ci, hi, wi, co, ho, wo, kh, kw, groups) = (
        g.batch() as usize,
        g.input_channels() as usize,
        g.input_height() as usize,
        g.input_width() as usize,
        g.output_channels() as usize,
        g.output_height() as usize,
        g.output_width() as usize,
        g.kernel_height() as usize,
        g.kernel_width() as usize,
        g.groups() as usize,
    );
    let [pt, _, pl, _] = g.padding().map(|x| x as usize);
    let [sh, sw] = g.stride().map(|x| x as usize);
    let [dh, dw] = g.dilation().map(|x| x as usize);
    let ig = ci / groups;
    let og = co / groups;
    for b in 0..n {
        for o in 0..co {
            for y in 0..ho {
                for x in 0..wo {
                    let mut dot = 0.0_f32;
                    for c in 0..ig {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let py = y * sh + ky * dh;
                                let px = x * sw + kx * dw;
                                // Virtual explicit zeros avoid a materialized host image while
                                // preserving every zero*weight product, including 0*Inf -> NaN.
                                let input = if py >= pt && py - pt < hi && px >= pl && px - pl < wi
                                {
                                    bf16::from_bits(storage.read_bits(
                                        0,
                                        ((b * ci + (o / og) * ig + c) * hi + py - pt) * wi + px
                                            - pl,
                                    ))
                                    .to_f32()
                                } else {
                                    0.0
                                };
                                let weight = bf16::from_bits(
                                    storage.read_bits(1, ((o * ig + c) * kh + ky) * kw + kx),
                                )
                                .to_f32();
                                let product = input * weight;
                                if c == 0 && ky == 0 && kx == 0 {
                                    dot = product;
                                } else {
                                    dot += product;
                                }
                            }
                        }
                    }
                    let convolution = bf16::from_f32(dot).to_f32();
                    let bias = bf16::from_bits(storage.read_bits(2, o)).to_f32();
                    storage.write_bits(
                        3,
                        ((b * co + o) * ho + y) * wo + x,
                        bf16::from_f32(convolution + bias).to_bits(),
                    );
                }
            }
        }
    }
}

// Balanced floor(n/2) binary tree, <=24 frames; channel stride is H*W.
fn squares(start: usize, n: usize, read: &impl Fn(usize) -> f32) -> f32 {
    match n {
        1 => {
            let x = read(start);
            x * x
        }
        2 => {
            let x = read(start);
            let y = read(start + 1);
            x * x + y * y
        }
        _ => {
            let a = n / 2;
            squares(start, a, read) + squares(start + a, n - a, read)
        }
    }
}
pub(super) fn execute_channel_l2_norm(launch: &ReferenceChannelL2NormLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.gamma, &launch.output]);
    let g = launch.shape;
    let channels = g.channels() as usize;
    let spatial = (g.height() * g.width()) as usize;
    for b in 0..g.batch() as usize {
        for p in 0..spatial {
            let base = b * channels * spatial + p;
            let norm = squares(0, channels, &|c| {
                bf16::from_bits(storage.read_bits(0, base + c * spatial)).to_f32()
            })
            .sqrt();
            // f32::max would suppress a NaN; clamp_min in the source propagates it.
            let denominator = if norm < g.effective_epsilon() {
                g.effective_epsilon()
            } else {
                norm
            };
            for c in 0..channels {
                let i = base + c * spatial;
                let x = bf16::from_bits(storage.read_bits(0, i)).to_f32();
                let gamma = bf16::from_bits(storage.read_bits(1, c)).to_f32();
                let normalized = bf16::from_f32(x / denominator).to_f32();
                let scaled = bf16::from_f32(normalized * g.effective_scale()).to_f32();
                let weighted = bf16::from_f32(scaled * gamma).to_f32();
                storage.write_bits(2, i, bf16::from_f32(weighted + 0.0_f32).to_bits());
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("reference BF16 spatial: {}", reason.into()))
}

#[cfg(test)]
mod tests;
