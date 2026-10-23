//! Direct admitted NCHW execution with bounded scalar attention state.
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{
    SpatialAttentionBf16Geometry, SpatialUpsampleGatherBf16Geometry,
    SPATIAL_ATTENTION_BF16_MAX_CHANNELS,
};

pub(crate) enum ReferenceSpatialDecoderGeometry {
    Attention(SpatialAttentionBf16Geometry),
    Upsample(SpatialUpsampleGatherBf16Geometry),
}
pub(crate) struct ReferenceSpatialDecoderLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceSpatialDecoderGeometry,
}
impl ReferenceSpatialDecoderLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (input, output) = match &self.shape {
            ReferenceSpatialDecoderGeometry::Attention(g) => (
                g.element_count()
                    .checked_mul(3)
                    .ok_or_else(|| invalid("QKV extent overflow"))?,
                g.element_count(),
            ),
            ReferenceSpatialDecoderGeometry::Upsample(g) => (
                g.batch()
                    .checked_mul(g.input_channels())
                    .and_then(|n| n.checked_mul(g.input_height()))
                    .and_then(|n| n.checked_mul(g.input_width()))
                    .ok_or_else(|| invalid("input extent overflow"))?,
                g.element_count(),
            ),
        };
        for (r, n) in [(&self.input, input), (&self.output, output)] {
            r.validate_range()?;
            let bytes = n
                .checked_mul(2)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| invalid("byte extent exceeds address space"))?;
            if r.element_type != ElementType::Bf16
                || r.length_bytes != bytes
                || (r.offset_bytes - r.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid("BF16 region extent/alignment differs"));
            }
        }
        if self.output.overlaps(&self.input) {
            return Err(invalid("output overlaps read input"));
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn validate_runtime(&self, id: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        self.input.validate_runtime(id)?;
        self.output.validate_runtime(id)?;
        self.validate()
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        let (n, ops) = match &self.shape {
            ReferenceSpatialDecoderGeometry::Attention(g) => {
                (g.element_count(), g.logical_arithmetic_operations())
            }
            ReferenceSpatialDecoderGeometry::Upsample(g) => (g.element_count(), 0),
        };
        DeviceCommandLogicalWork::tensor(n, ops).map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn operation(&self) -> &'static str {
        match self.shape {
            ReferenceSpatialDecoderGeometry::Attention(_) => "vnext_spatial_attention_bf16",
            ReferenceSpatialDecoderGeometry::Upsample(_) => "vnext_spatial_upsample_gather_bf16",
        }
    }
}
pub(super) fn execute(launch: &ReferenceSpatialDecoderLaunch) {
    match &launch.shape {
        ReferenceSpatialDecoderGeometry::Attention(g) => attention(launch, *g),
        ReferenceSpatialDecoderGeometry::Upsample(g) => upsample(launch, g),
    }
}

// Balanced power-of-two tree with explicit +0 leaves after C. No array of
// products and no S*S storage. A different reduction is a different profile.
fn dot(start: usize, len: usize, channels: usize, read: &impl Fn(usize) -> (f32, f32)) -> f32 {
    if start >= channels {
        return 0.0;
    }
    if len == 1 {
        let (a, b) = read(start);
        return a * b;
    }
    let left = dot(start, len / 2, channels, read);
    let right = dot(start + len / 2, len / 2, channels, read);
    left + right
}
fn attention(launch: &ReferenceSpatialDecoderLaunch, g: SpatialAttentionBf16Geometry) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    let c = g.channels() as usize;
    let s = g.spatial() as usize;
    let p = c.next_power_of_two();
    for n in 0..g.batch() as usize {
        let base = n * 3 * c * s;
        for q in 0..s {
            // At most 8 KiB stack, independent of H*W, not an unplanned heap payload.
            let mut accumulator = [0.0_f32; SPATIAL_ATTENTION_BF16_MAX_CHANNELS as usize];
            let (mut maximum, mut denominator) = (0.0_f32, 0.0_f32);
            let mut poisoned = false;
            for k in 0..s {
                let product_sum = dot(0, p, c, &|channel| {
                    (
                        bf16::from_bits(storage.read_bits(0, base + channel * s + q)).to_f32(),
                        bf16::from_bits(storage.read_bits(0, base + (c + channel) * s + k))
                            .to_f32(),
                    )
                });
                let score = product_sum * g.effective_scale();
                if !score.is_finite() {
                    poisoned = true;
                    break;
                }
                if k == 0 {
                    maximum = score;
                    denominator = 1.0;
                    for (channel, a) in accumulator[..c].iter_mut().enumerate() {
                        *a =
                            bf16::from_bits(storage.read_bits(0, base + (2 * c + channel) * s + k))
                                .to_f32();
                    }
                } else {
                    let next = maximum.max(score);
                    let alpha = libm::expf(maximum - next);
                    let probability = libm::expf(score - next);
                    let old_denominator = denominator * alpha;
                    denominator = old_denominator + probability;
                    for (channel, a) in accumulator[..c].iter_mut().enumerate() {
                        let value =
                            bf16::from_bits(storage.read_bits(0, base + (2 * c + channel) * s + k))
                                .to_f32();
                        let old = *a * alpha;
                        let new = probability * value;
                        *a = old + new;
                    }
                    maximum = next;
                }
            }
            let inverse = 1.0_f32 / denominator;
            for (channel, a) in accumulator[..c].iter().enumerate() {
                let value = if poisoned { f32::NAN } else { *a * inverse };
                storage.write_bits(
                    1,
                    (n * c + channel) * s + q,
                    bf16::from_f32(value).to_bits(),
                );
            }
        }
    }
}
fn upsample(launch: &ReferenceSpatialDecoderLaunch, g: &SpatialUpsampleGatherBf16Geometry) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    let (ci, h, w, co, sh, sw, stride) = (
        g.input_channels() as usize,
        g.input_height() as usize,
        g.input_width() as usize,
        g.output_channels() as usize,
        g.scale_h() as usize,
        g.scale_w() as usize,
        g.channel_stride() as usize,
    );
    for n in 0..g.batch() as usize {
        for o in 0..co {
            for y in 0..h * sh {
                for x in 0..w * sw {
                    let channel = o * stride + g.channel_offsets()[(y % sh) * sw + x % sw] as usize;
                    let value =
                        storage.read_bits(0, ((n * ci + channel) * h + y / sh) * w + x / sw);
                    storage.write_bits(1, ((n * co + o) * (h * sh) + y) * (w * sw) + x, value);
                }
            }
        }
    }
}
fn invalid(message: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("reference spatial decoder: {}", message.into()))
}
#[cfg(test)]
mod tests;
