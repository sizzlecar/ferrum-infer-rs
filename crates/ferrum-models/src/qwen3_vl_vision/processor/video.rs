//! Declared Qwen3-VL video resize and temporal patch packing. Frame selection
//! belongs to the model presentation caller, not to an inferred client policy.
use super::*;

pub const QWEN3_VL_VIDEO_PREPROCESS_PROFILE: &str =
    "qwen3vl.hf.video_rgb8.cubic_aa_a05_f64_u8_rne.fused_half.bf16.temporal_pairs.repeat_last.v1";

#[derive(Deserialize)]
struct VideoMetadata {
    video_processor_type: String,
    #[serde(flatten)]
    rgb: RgbMetadata,
}
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionVideoProcessor {
    patch: u64,
    temporal: u64,
    merge: u64,
    minimum: u64,
    maximum: u64,
    limits: Qwen3VLVisionProcessorLimits,
    source_sha256: String,
}
impl Qwen3VLVisionVideoProcessor {
    pub fn from_source_bytes(source: &[u8], limits: Qwen3VLVisionProcessorLimits) -> Result<Self> {
        ensure!(
            !source.is_empty()
                && source.len() <= limits.maximum_metadata_bytes
                && limits.maximum_patch_rows > 0
                && limits.maximum_patch_bytes > 0
                && limits.maximum_patch_bytes <= isize::MAX as usize,
            "invalid video processor capacities"
        );
        let VideoMetadata {
            video_processor_type,
            mut rgb,
        } = serde_json::from_slice(source)?;
        ensure!(
            video_processor_type == "Qwen3VLVideoProcessor",
            "unsupported video processor type"
        );
        rgb.validate()?;
        Ok(Self {
            patch: rgb.patch_size,
            temporal: rgb.temporal_patch_size,
            merge: rgb.merge_size,
            minimum: rgb.size.shortest_edge,
            maximum: rgb.size.longest_edge,
            limits,
            source_sha256: format!("{:x}", Sha256::digest(source)),
        })
    }
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn prepare_video(
        &self,
        architecture: Qwen3VLVisionArchitecture,
        width: u32,
        height: u32,
        source_frames: usize,
        selected_frames: &[usize],
    ) -> Result<Qwen3VLVisionVideoPlan> {
        ensure!(
            architecture.channels() == 3
                && architecture.patch_size() == self.patch
                && architecture.temporal_patch_size() == self.temporal
                && architecture.merge_size() == self.merge,
            "processor patch geometry differs from vision architecture"
        );
        ensure!(
            source_frames > 0
                && source_frames <= 1 << 24
                && !selected_frames.is_empty()
                && selected_frames.len() as u64 <= self.limits.maximum_patch_rows
                && selected_frames.iter().all(|&i| i < source_frames)
                && selected_frames.windows(2).all(|p| p[0] <= p[1]),
            "invalid ordered video frame selection"
        );
        let check = |w: usize, h: usize| -> Result<()> {
            ensure!(
                w > 0
                    && h > 0
                    && w <= self.limits.image.maximum_width() as usize
                    && h <= self.limits.image.maximum_height() as usize
                    && w.checked_mul(h)
                        .is_some_and(|n| n <= self.limits.image.maximum_pixels()),
                "video frame exceeds declared dimensions or pixel capacity"
            );
            Ok(())
        };
        check(width as usize, height as usize)?;
        let resized = resize(
            width,
            height,
            selected_frames.len(),
            self.patch * self.merge,
            self.minimum,
            self.maximum,
        )?;
        check(resized.0, resized.1)?;
        let config = Qwen3VLVisionConfig::new(
            architecture,
            vec![Qwen3VLVisionGrid::new(
                selected_frames.len().div_ceil(self.temporal as usize) as u64,
                resized.1 as u64 / self.patch,
                resized.0 as u64 / self.patch,
            )?],
        )?;
        let patch_bytes = usize::try_from(
            config
                .rows()?
                .checked_mul(config.architecture().patch_columns()?)
                .and_then(|n| n.checked_mul(2))
                .context("video patch bytes overflow")?,
        )?;
        ensure!(
            config.rows()? <= self.limits.maximum_patch_rows
                && patch_bytes <= self.limits.maximum_patch_bytes,
            "video patches exceed declared capacity"
        );
        let horizontal_bytes = (height as usize)
            .checked_mul(resized.0)
            .and_then(|n| n.checked_mul(3))
            .filter(|&n| n <= isize::MAX as usize)
            .context("video resize scratch overflow")?;
        let input = ResolvedTensorSpec::new(
            vec![
                1,
                3,
                source_frames as u64,
                u64::from(height),
                u64::from(width),
            ],
            ElementType::U8,
            ResolvedTensorLayout::Contiguous,
        )?;
        Ok(Qwen3VLVisionVideoPlan {
            config,
            input,
            resized,
            selected_frames: selected_frames.to_vec(),
            horizontal_bytes,
            patch_bytes,
        })
    }
}

/// The HF video area budget includes time. Unlike image resizing, its threshold
/// test rounds T to the temporal factor, while its scaling uses original T.
fn resize(
    width: u32,
    height: u32,
    frames: usize,
    factor: u64,
    minimum: u64,
    maximum: u64,
) -> Result<(usize, usize)> {
    let (mut w, mut h, factor) = (f64::from(width), f64::from(height), factor as f64);
    if w < factor || h < factor {
        let scale = (factor / w).max(factor / h);
        w = (w * scale).floor();
        h = (h * scale).floor();
    }
    ensure!(
        w.max(h) / w.min(h) <= 200.,
        "HF video aspect ratio exceeds 200"
    );
    // A single selected frame is repeated for its first complete temporal pair.
    let t = frames.max(2) as f64;
    let rounded_t = (t / 2.).round_ties_even() * 2.;
    let (mut rw, mut rh) = (
        (w / factor).round_ties_even() * factor,
        (h / factor).round_ties_even() * factor,
    );
    if rounded_t * rw * rh > maximum as f64 {
        let beta = (t * w * h / maximum as f64).sqrt();
        rw = (w / beta / factor).floor().max(1.) * factor;
        rh = (h / beta / factor).floor().max(1.) * factor;
    } else if rounded_t * rw * rh < minimum as f64 {
        let beta = (minimum as f64 / (t * w * h)).sqrt();
        rw = (w * beta / factor).ceil() * factor;
        rh = (h * beta / factor).ceil() * factor;
    }
    ensure!(
        rw >= factor && rh >= factor && rw <= (1u64 << 24) as f64 && rh <= (1u64 << 24) as f64,
        "resized video coordinates exceed exact F32 integer range"
    );
    Ok((rw as usize, rh as usize))
}

#[derive(Clone, Debug)]
pub struct Qwen3VLVisionVideoPlan {
    config: Qwen3VLVisionConfig,
    input: ResolvedTensorSpec,
    resized: (usize, usize),
    selected_frames: Vec<usize>,
    horizontal_bytes: usize,
    patch_bytes: usize,
}
impl Qwen3VLVisionVideoPlan {
    pub fn config(&self) -> &Qwen3VLVisionConfig {
        &self.config
    }
    pub fn resized_dimensions(&self) -> (usize, usize) {
        self.resized
    }
    pub fn selected_frames(&self) -> &[usize] {
        &self.selected_frames
    }
    pub fn preprocessing_profile(&self) -> &'static str {
        QWEN3_VL_VIDEO_PREPROCESS_PROFILE
    }
    pub fn input_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        host.backing_bytes(self.horizontal_bytes)?
            .checked_add(host.backing_bytes(self.patch_bytes)?)
            .context("video preprocessing Host capacity overflow")
    }
    pub fn managed_patches(
        &self,
        host: &HostMemoryAccount,
        pixels: &ManagedHostTensor,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            pixels.tensor() == &self.input
                && host.owns(pixels.bytes())
                && pixels.bytes().class() == HostMemoryClass::InputPreprocess
                && pixels.bytes().len() as u64 == self.input.minimum_storage_bytes()?,
            "video patches require original admitted contiguous U8 NCTHW pixels"
        );
        let mut grants = host
            .reserve_many(&[
                (HostMemoryClass::InputPreprocess, self.horizontal_bytes),
                (HostMemoryClass::InputPreprocess, self.patch_bytes),
            ])?
            .into_iter();
        let mut horizontal = grants.next().unwrap().allocate_zeroed()?;
        let mut patches = grants.next().unwrap().allocate_zeroed()?;
        let dims = self.input.dimensions();
        let (frames, ih, iw) = (dims[2] as usize, dims[3] as usize, dims[4] as usize);
        let (ow, oh) = self.resized;
        let a = self.config.architecture();
        let (patch, merge, temporal) = (
            a.patch_size() as usize,
            a.merge_size() as usize,
            a.temporal_patch_size() as usize,
        );
        let spatial_rows = (ow / patch) * (oh / patch);
        let columns = a.patch_columns()? as usize;
        let mut previous = None;
        for frame in 0..self.selected_frames.len().div_ceil(temporal) * temporal {
            let source = self.selected_frames[frame.min(self.selected_frames.len() - 1)];
            if previous != Some(source) {
                for y in 0..ih {
                    for x in 0..ow {
                        for c in 0..3 {
                            horizontal.as_mut()[(y * ow + x) * 3 + c] =
                                axis_sample(x, iw, ow, |sx| {
                                    pixels.bytes().bytes()
                                        [((c * frames + source) * ih + y) * iw + sx]
                                });
                        }
                    }
                }
                previous = Some(source);
            }
            for spatial in 0..spatial_rows {
                let block = spatial / (merge * merge);
                let within = spatial % (merge * merge);
                let py = block / (ow / patch / merge) * merge + within / merge;
                let px = block % (ow / patch / merge) * merge + within % merge;
                let row = frame / temporal * spatial_rows + spatial;
                for y in 0..patch {
                    for x in 0..patch {
                        for c in 0..3 {
                            let pixel = axis_sample(py * patch + y, ih, oh, |sy| {
                                horizontal.as_ref()[(sy * ow + px * patch + x) * 3 + c]
                            });
                            let value = bf16::from_f32((f32::from(pixel) - 127.5) / 127.5)
                                .to_bits()
                                .to_le_bytes();
                            let offset = (row * columns
                                + ((c * temporal + frame % temporal) * patch + y) * patch
                                + x)
                                * 2;
                            patches.as_mut()[offset..offset + 2].copy_from_slice(&value);
                        }
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(Qwen3VLVisionFamily::PATCHES_ID)?,
            ResolvedTensorSpec::new(
                vec![self.config.rows()?, a.patch_columns()?],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?,
            patches.freeze(),
        )?)
    }
}

#[cfg(test)]
mod tests;
