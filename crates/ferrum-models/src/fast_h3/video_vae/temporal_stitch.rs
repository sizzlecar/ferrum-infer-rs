//! Request-owned Host continuation for original BF16 temporal overlap. A raw
//! clip comes from the checked execution adapter; this state is not a device
//! completion proof or an allocator. Caller-supplied Host admission covers
//! every successor buffer before the old overlap/progress can be replaced.
use super::{FastH3VideoDecodePlan, FastH3VideoRawClip, FastH3VideoVaeConfig};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, ManagedHostBytes};
use half::bf16;

mod rgb_transfer;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug)]
pub struct FastH3VideoTemporalLimits {
    pub maximum_output_frames: usize,
    pub maximum_buffer_bytes: usize,
}

/// Finalized normalized BF16 NCTHW frames, still before RGB denormalization.
#[derive(Debug)]
pub struct FastH3VideoFrameChunk {
    start: usize,
    shape: [usize; 5],
    bytes: ManagedHostBytes,
}
/// Interleaved RGB8 `[batch,time,height,width,3]`, with an admitted Output
/// owner. Container encoding and publication remain the caller's responsibility.
#[derive(Debug)]
pub struct FastH3VideoRgbChunk {
    start: usize,
    shape: [usize; 5],
    bytes: ManagedHostBytes,
}
impl FastH3VideoRgbChunk {
    pub fn output_frame_start(&self) -> usize {
        self.start
    }
    pub fn nthwc_shape(&self) -> [usize; 5] {
        self.shape
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
}
impl FastH3VideoFrameChunk {
    pub fn output_frame_start(&self) -> usize {
        self.start
    }
    pub fn ncthw_shape(&self) -> [usize; 5] {
        self.shape
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
    /// Original pixel std/mean, separately rounded F32 multiply/add, clamp to
    /// [0,1], multiply255 and truncate to U8. No BF16 color arithmetic or FMA.
    pub fn managed_rgb8(
        &self,
        host: &HostMemoryAccount,
        maximum_bytes: usize,
    ) -> Result<FastH3VideoRgbChunk> {
        let [batch, channels, frames, height, width] = self.shape;
        ensure!(channels == 3, "RGB requires three channels");
        let count = [batch, frames, height, width, 3]
            .into_iter()
            .try_fold(1usize, |n, v| n.checked_mul(v))
            .context("RGB extent overflow")?;
        ensure!(
            count <= maximum_bytes
                && count <= isize::MAX as usize
                && self.bytes.len()
                    == count.checked_mul(2).context("RGB source extent overflow")?,
            "RGB conversion exceeds capacity or source extent"
        );
        let mut output = host.allocate_zeroed(HostMemoryClass::Output, count)?;
        let plane = frames * height * width;
        let source = self.bytes.bytes();
        let destination = output.as_mut();
        let mut nonfinite = false;
        for b in 0..batch {
            for c in 0..3 {
                for pixel in 0..plane {
                    let src = 2 * ((b * 3 + c) * plane + pixel);
                    let word = u16::from_le_bytes([source[src], source[src + 1]]);
                    nonfinite |= word & 0x7f80 == 0x7f80;
                    let byte = rgb_transfer::TABLE[c][usize::from(word)];
                    destination[(b * plane + pixel) * 3 + c] = byte;
                }
            }
        }
        // The private buffer cannot be published if any source pixel is invalid.
        ensure!(!nonfinite, "nonfinite normalized pixel");
        Ok(FastH3VideoRgbChunk {
            start: self.start,
            shape: [batch, frames, height, width, 3],
            bytes: output.freeze(),
        })
    }
}

pub struct FastH3VideoTemporalAssembler {
    plan: FastH3VideoDecodePlan,
    batch: usize,
    height: usize,
    width: usize,
    limits: FastH3VideoTemporalLimits,
    next: usize,
    published: usize,
    finished: bool,
    overlap: Option<ManagedHostBytes>,
}
/// Exclusive pending successor. Dropping it returns all new grants and leaves
/// the old ordinal/overlap intact. The raw input remains owned by its caller.
pub struct FastH3PreparedVideoChunk<'a> {
    owner: &'a mut FastH3VideoTemporalAssembler,
    next_overlap: Option<ManagedHostBytes>,
    output: Option<FastH3VideoFrameChunk>,
    finish: bool,
}
impl FastH3PreparedVideoChunk<'_> {
    pub fn output(&self) -> Option<&FastH3VideoFrameChunk> {
        self.output.as_ref()
    }
    pub fn commit(self) -> Option<FastH3VideoFrameChunk> {
        let Self {
            owner,
            next_overlap,
            output,
            finish,
        } = self;
        if let Some(chunk) = &output {
            owner.published += chunk.shape[2];
        }
        if finish {
            owner.finished = true;
        } else {
            owner.next += 1;
        }
        owner.overlap = next_overlap;
        output
    }
}
impl FastH3VideoTemporalAssembler {
    pub const NUMERICAL_PROFILE_ID: &'static str =
        "fast_h3.video_temporal.bf16_weights_products_sum_separate.v1";
    pub fn new(
        config: &FastH3VideoVaeConfig,
        latent_frames: usize,
        batch: usize,
        pixel_height: usize,
        pixel_width: usize,
        limits: FastH3VideoTemporalLimits,
    ) -> Result<Self> {
        ensure!(
            batch > 0 && pixel_height > 0 && pixel_width > 0 && limits.maximum_buffer_bytes > 0,
            "invalid temporal geometry or capacity"
        );
        let plan = FastH3VideoDecodePlan::new(config, latent_frames, limits.maximum_output_frames)?;
        let extent = plan.overlap_frames();
        // The original released profile uses five frames. A scalar divisor
        // which changes when converted to BF16 requires a separately audited
        // TensorIterator recipe, rather than a hidden numeric fallback.
        ensure!(
            extent == 0
                || (extent <= u32::MAX as usize
                    && bf16::from_f32(extent as f32).to_f32() == extent as f32),
            "overlap extent is not exact in the BF16 scalar profile"
        );
        let s = Self {
            plan,
            batch,
            height: pixel_height,
            width: pixel_width,
            limits,
            next: 0,
            published: 0,
            finished: false,
            overlap: None,
        };
        s.extent(s.plan.chunk(0)?.raw_frames)?;
        Ok(s)
    }
    pub fn plan(&self) -> &FastH3VideoDecodePlan {
        &self.plan
    }
    pub fn next_chunk(&self) -> usize {
        self.next
    }
    pub fn published_frames(&self) -> usize {
        self.published
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    fn extent(&self, frames: usize) -> Result<usize> {
        let n = [self.batch, 3, frames, self.height, self.width, 2]
            .into_iter()
            .try_fold(1usize, |n, d| n.checked_mul(d))
            .context("temporal buffer extent overflow")?;
        ensure!(
            n <= self.limits.maximum_buffer_bytes && n <= isize::MAX as usize,
            "temporal buffer exceeds declared capacity"
        );
        Ok(n)
    }
    pub fn prepare_chunk<'a>(
        &'a mut self,
        ordinal: usize,
        raw: &FastH3VideoRawClip,
        host: &HostMemoryAccount,
    ) -> Result<FastH3PreparedVideoChunk<'a>> {
        ensure!(
            !self.finished && ordinal == self.next && ordinal < self.plan.chunk_count(),
            "stale or out-of-order temporal chunk"
        );
        let chunk = self.plan.chunk(ordinal)?;
        ensure!(
            raw.ncthw_shape() == [self.batch, 3, chunk.raw_frames, self.height, self.width]
                && raw.bytes().class() == HostMemoryClass::Output
                && raw.bytes().len() == self.extent(chunk.raw_frames)?,
            "raw clip geometry or output owner differs"
        );
        ensure!(
            raw.bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()),
            "nonfinite raw clip"
        );
        ensure!(
            chunk.output_start == self.published,
            "temporal publish offset differs"
        );
        if chunk.blend_prefix_frames > 0 {
            ensure!(
                self.overlap.as_ref().is_some_and(
                    |o| o.len() == self.extent(chunk.blend_prefix_frames).unwrap_or(usize::MAX)
                ),
                "missing previous temporal overlap"
            );
        } else {
            ensure!(self.overlap.is_none(), "unexpected initial overlap");
        }
        // Two independent admissions; a failure of the second drops the first
        // without changing any state or consuming the original completion.
        let mut output = if chunk.publish_frames > 0 {
            Some(host.allocate_zeroed(HostMemoryClass::Output, self.extent(chunk.publish_frames)?)?)
        } else {
            None
        };
        let mut overlap = if chunk.next_overlap_frames > 0 {
            Some(host.allocate_zeroed(
                HostMemoryClass::Output,
                self.extent(chunk.next_overlap_frames)?,
            )?)
        } else {
            None
        };
        let plane = self.height * self.width;
        let input = raw.bytes().bytes();
        if let Some(output) = &mut output {
            let output = output.as_mut();
            let frame_bytes = 2 * plane;
            let previous = self.overlap.as_ref().map(ManagedHostBytes::bytes);
            let mut nonfinite = false;
            for bc in 0..self.batch * 3 {
                for t in 0..chunk.publish_frames {
                    let src = (bc * chunk.raw_frames + chunk.main_start + t) * frame_bytes;
                    let dst = (bc * chunk.publish_frames + t) * frame_bytes;
                    let source = &input[src..src + frame_bytes];
                    let destination = &mut output[dst..dst + frame_bytes];
                    if t >= chunk.blend_prefix_frames {
                        destination.copy_from_slice(source);
                        continue;
                    }
                    // The original BF16 coefficients depend only on the
                    // frame ordinal. Preserve every product/sum rounding,
                    // without rebuilding the coefficients for each pixel.
                    let weights = TemporalBlendWeights::new(t, chunk.blend_prefix_frames);
                    let old = (bc * chunk.blend_prefix_frames + t) * frame_bytes;
                    let previous =
                        &previous.expect("checked previous overlap")[old..old + frame_bytes];
                    for ((old, new), destination) in previous
                        .chunks_exact(2)
                        .zip(source.chunks_exact(2))
                        .zip(destination.chunks_exact_mut(2))
                    {
                        let bits = weights.apply(
                            u16::from_le_bytes([old[0], old[1]]),
                            u16::from_le_bytes([new[0], new[1]]),
                        );
                        nonfinite |= bits & 0x7f80 == 0x7f80;
                        destination.copy_from_slice(&bits.to_le_bytes());
                    }
                }
            }
            // A failed blend drops these private successor buffers; the old
            // overlap and publication cursor remain unchanged.
            ensure!(!nonfinite, "temporal BF16 blend overflow");
        }
        if let Some(overlap) = &mut overlap {
            let overlap = overlap.as_mut();
            for bc in 0..self.batch * 3 {
                let start = 2 * ((bc * chunk.raw_frames + chunk.next_overlap_start) * plane);
                let bytes = 2 * chunk.next_overlap_frames * plane;
                overlap[bc * bytes..(bc + 1) * bytes].copy_from_slice(&input[start..start + bytes]);
            }
        }
        let output = output.map(|bytes| FastH3VideoFrameChunk {
            start: self.published,
            shape: [self.batch, 3, chunk.publish_frames, self.height, self.width],
            bytes: bytes.freeze(),
        });
        Ok(FastH3PreparedVideoChunk {
            owner: self,
            next_overlap: overlap.map(|b| b.freeze()),
            output,
            finish: false,
        })
    }
    pub fn prepare_finish<'a>(
        &'a mut self,
        host: &HostMemoryAccount,
    ) -> Result<FastH3PreparedVideoChunk<'a>> {
        ensure!(
            !self.finished && self.next == self.plan.chunk_count(),
            "temporal sequence is not ready to finish"
        );
        let frames = self.plan.final_overlap_frames();
        ensure!(
            self.published.checked_add(frames) == Some(self.plan.output_frames()),
            "final frame count differs"
        );
        let output = if frames == 0 {
            None
        } else {
            let overlap = self
                .overlap
                .as_ref()
                .context("missing final temporal overlap")?;
            let extent = self.plan.overlap_frames();
            ensure!(
                frames <= extent && overlap.len() == self.extent(extent)?,
                "invalid final overlap extent"
            );
            let mut bytes = host.allocate_zeroed(HostMemoryClass::Output, self.extent(frames)?)?;
            let plane = self.height * self.width;
            let source = overlap.bytes();
            let destination = bytes.as_mut();
            for bc in 0..self.batch * 3 {
                let start = 2 * bc * extent * plane;
                let count = 2 * frames * plane;
                destination[bc * count..(bc + 1) * count]
                    .copy_from_slice(&source[start..start + count]);
            }
            Some(FastH3VideoFrameChunk {
                start: self.published,
                shape: [self.batch, 3, frames, self.height, self.width],
                bytes: bytes.freeze(),
            })
        };
        Ok(FastH3PreparedVideoChunk {
            owner: self,
            next_overlap: None,
            output,
            finish: true,
        })
    }
}
struct TemporalBlendWeights {
    previous: f32,
    current: f32,
}
impl TemporalBlendWeights {
    fn new(position: usize, extent: usize) -> Self {
        let pos = bf16::from_f32(position as f32).to_f32();
        let current = bf16::from_f32(pos / extent as f32).to_f32();
        Self {
            previous: bf16::from_f32(1f32 - current).to_f32(),
            current,
        }
    }
    fn apply(&self, a: u16, b: u16) -> u16 {
        let left = bf16::from_f32(bf16::from_bits(a).to_f32() * self.previous).to_f32();
        let right = bf16::from_f32(bf16::from_bits(b).to_f32() * self.current).to_f32();
        bf16::from_f32(left + right).to_bits()
    }
}

#[cfg(test)]
fn blend(a: u16, b: u16, position: usize, extent: usize) -> Result<u16> {
    let bits = TemporalBlendWeights::new(position, extent).apply(a, b);
    ensure!(bits & 0x7f80 != 0x7f80, "temporal BF16 blend overflow");
    Ok(bits)
}
