//! Released Ref2VA reference timeline and shared RGB canvas. A codec supplies
//! monotonic original frames; VAE and Qwen consumers receive the same admitted
//! pixels, while their temporal selection rules remain explicit model policy.
use crate::qwen_image21::image_input::{
    ImageInputLimits, ImageLanczosResizePlan, ManagedRgba8Image,
};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;

#[derive(Clone, Copy, Debug)]
pub struct FastH3DecodedVideoGeometry {
    pub width: u32,
    pub height: u32,
    pub frames: usize,
    pub frame_rate_numerator: u32,
    pub frame_rate_denominator: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct FastH3ReferenceVideoLimits {
    pub image: ImageInputLimits,
    pub maximum_source_frames: usize,
    pub maximum_frames: usize,
    pub maximum_output_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct FastH3ReferenceVideoCanvas {
    pub short_edge: u32,
    pub maximum_pixels: u64,
}
impl Default for FastH3ReferenceVideoCanvas {
    fn default() -> Self {
        Self {
            short_edge: 768,
            maximum_pixels: 768 * 1344,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FastH3ReferenceVideoPlan {
    source: FastH3DecodedVideoGeometry,
    resize: ImageLanczosResizePlan,
    output: ResolvedTensorSpec,
    source_frames: Vec<usize>,
    qwen_frames: Vec<usize>,
    timestamps: Vec<f64>,
}
impl FastH3ReferenceVideoPlan {
    pub const PROFILE: &'static str =
        "fasth3.ref2va.video_nearest24_f64_half_up.lanczos22_libm0215.rgb_ncthw.qwen2fps.v1";
    pub fn new(
        source: FastH3DecodedVideoGeometry,
        requested_frames: usize,
        canvas: FastH3ReferenceVideoCanvas,
        limits: FastH3ReferenceVideoLimits,
    ) -> Result<Self> {
        ensure!(
            source.frames > 0
                && source.frames <= limits.maximum_source_frames
                && source.frame_rate_numerator > 0
                && source.frame_rate_denominator > 0
                && requested_frames > 0
                && requested_frames <= limits.maximum_frames,
            "reference video timeline exceeds declared capacity"
        );
        let (width, height) = Self::dimensions(source.width, source.height, canvas)?;
        let resize =
            ImageLanczosResizePlan::new(source.width, source.height, width, height, limits.image)?;
        let rate =
            f64::from(source.frame_rate_numerator) / f64::from(source.frame_rate_denominator);
        let scale = 24. / rate;
        let mut source_frames = Vec::new();
        for index in 0..source.frames {
            let first = (index as f64 * scale + 0.5).floor();
            let end = ((index + 1) as f64 * scale + 0.5).floor();
            ensure!(
                end <= (1u64 << 53) as f64,
                "reference resampling coordinates overflow"
            );
            let count = ((end - first) as usize).min(requested_frames - source_frames.len());
            source_frames.extend(std::iter::repeat_n(index, count));
            if source_frames.len() == requested_frames {
                break;
            }
        }
        ensure!(
            !source_frames.is_empty(),
            "reference video contains no frame on the 24 FPS timeline"
        );
        let output = ResolvedTensorSpec::new(
            vec![
                1,
                3,
                source_frames.len() as u64,
                u64::from(height),
                u64::from(width),
            ],
            ElementType::U8,
            ResolvedTensorLayout::Contiguous,
        )?;
        ensure!(
            output.minimum_storage_bytes()? <= limits.maximum_output_bytes as u64,
            "reference video pixels exceed declared Host capacity"
        );
        let qwen_frames = (0..source_frames.len()).step_by(12).collect::<Vec<_>>();
        let timestamps = (0..qwen_frames.len())
            .step_by(2)
            .map(|i| (i + (i + 1).min(qwen_frames.len() - 1)) as f64 / 4.)
            .collect();
        Ok(Self {
            source,
            resize,
            output,
            source_frames,
            qwen_frames,
            timestamps,
        })
    }
    pub fn dimensions(
        width: u32,
        height: u32,
        policy: FastH3ReferenceVideoCanvas,
    ) -> Result<(u32, u32)> {
        ensure!(
            width > 0
                && height > 0
                && policy.short_edge > 0
                && policy.maximum_pixels > 0
                && policy.maximum_pixels <= 1 << 53,
            "invalid reference video canvas"
        );
        let ratio = f64::from(width) / f64::from(height);
        ensure!(
            (0.25..=4.).contains(&ratio),
            "reference video aspect must be within 1:4 and 4:1"
        );
        let short = f64::from(policy.short_edge);
        let (mut w, mut h) = if ratio >= 1. {
            (short * ratio, short)
        } else {
            (short, short / ratio)
        };
        if w * h > policy.maximum_pixels as f64 {
            let scale = (policy.maximum_pixels as f64 / (w * h)).sqrt();
            w *= scale;
            h *= scale;
        }
        let axis = |n: f64| -> Result<u32> {
            let n = ((n / 32.).round_ties_even() * 32.).max(32.);
            ensure!(n <= f64::from(u32::MAX), "reference canvas overflows u32");
            Ok(n as u32)
        };
        Ok((axis(w)?, axis(h)?))
    }
    pub fn output_spec(&self) -> &ResolvedTensorSpec {
        &self.output
    }
    pub fn output_dimensions(&self) -> (u32, u32) {
        self.resize.dimensions()
    }
    pub fn source_frame_indices(&self) -> &[usize] {
        &self.source_frames
    }
    pub fn qwen_frame_indices(&self) -> &[usize] {
        &self.qwen_frames
    }
    pub fn block_timestamps(&self) -> &[f64] {
        &self.timestamps
    }
    /// One codec-supplied RGBA frame, resize scratch/result, and the complete
    /// NCTHW output. Decoder-private buffers must be admitted separately.
    pub fn input_peak_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        let rgba =
            usize::try_from(u64::from(self.source.width) * u64::from(self.source.height) * 4)?;
        let output = host.backing_bytes(self.output.minimum_storage_bytes()?.try_into()?)?;
        host.backing_bytes(rgba)?
            .checked_add(self.resize.input_peak_bytes(host)?)
            .and_then(|n| n.checked_add(output))
            .context("reference video Host backing overflow")
    }
    /// Calls `read` once per distinct selected source frame, in increasing
    /// order. The callback must retire skipped decoder frames itself.
    pub fn prepare_with_frames(
        &self,
        host: &HostMemoryAccount,
        mut read: impl FnMut(usize) -> Result<ManagedRgba8Image>,
    ) -> Result<ManagedHostTensor> {
        let mut output = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            self.output.minimum_storage_bytes()?.try_into()?,
        )?;
        let (w, h) = self.output_dimensions();
        let spatial = w as usize * h as usize;
        let frames = self.source_frames.len();
        let mut frame = 0;
        while frame < frames {
            let source_index = self.source_frames[frame];
            let source = read(source_index)?;
            let resized = self.resize.resize_rgb_ignoring_alpha(host, &source)?;
            while frame < frames && self.source_frames[frame] == source_index {
                for c in 0..3 {
                    for pixel in 0..spatial {
                        output.as_mut()[(c * frames + frame) * spatial + pixel] =
                            resized.bytes().bytes()[pixel * 4 + c];
                    }
                }
                frame += 1;
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new("value.fast_h3.reference_video.rgb")?,
            self.output.clone(),
            output.freeze(),
        )?)
    }
}

#[cfg(test)]
mod tests;
