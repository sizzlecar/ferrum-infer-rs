//! Original HF Qwen2-VL image preprocessing into the shared BF16 vision graph.
//! Geometry and RGB bicubic arithmetic are shared with the existing HF caller;
//! this adapter preserves DeepStack and does not apply an upsampler's outer resize.
use super::{
    image_resize::{axis_sample, smart_resize},
    Qwen3VLVisionArchitecture, Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid,
};
use crate::qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;
use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const QWEN3_VL_IMAGE_PREPROCESS_PROFILE: &str =
    "qwen3vl.hf.rgb8.cubic_aa_a05_f64_u8_rne.fused_half.bf16.patch_merge.v1";
#[derive(Clone, Copy, Debug)]
pub struct Qwen3VLVisionProcessorLimits {
    pub maximum_metadata_bytes: usize,
    pub image: ImageInputLimits,
    pub maximum_patch_rows: u64,
    pub maximum_patch_bytes: usize,
}
#[derive(Deserialize)]
struct ImageMetadata {
    image_processor_type: String,
    #[serde(flatten)]
    rgb: RgbMetadata,
}
#[derive(Deserialize)]
struct RgbMetadata {
    processor_class: String,
    patch_size: u64,
    temporal_patch_size: u64,
    merge_size: u64,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    size: PixelArea,
    #[serde(default = "channels_first")]
    data_format: String,
    #[serde(default)]
    min_pixels: Option<u64>,
    #[serde(default)]
    max_pixels: Option<u64>,
    #[serde(default = "yes")]
    do_convert_rgb: bool,
    #[serde(default = "yes")]
    do_normalize: bool,
    #[serde(default = "yes")]
    do_rescale: bool,
    #[serde(default = "yes")]
    do_resize: bool,
    #[serde(default = "bicubic")]
    resample: u32,
    #[serde(default = "rescale")]
    rescale_factor: f64,
}
#[derive(Deserialize)]
struct PixelArea {
    shortest_edge: u64,
    longest_edge: u64,
}
fn channels_first() -> String {
    "channels_first".into()
}
fn yes() -> bool {
    true
}
fn bicubic() -> u32 {
    3
}
fn rescale() -> f64 {
    1.0 / 255.0
}
impl RgbMetadata {
    fn validate(&mut self) -> Result<()> {
        let p = self;
        if let Some(minimum) = p.min_pixels {
            p.size.shortest_edge = minimum;
        }
        if let Some(maximum) = p.max_pixels {
            p.size.longest_edge = maximum;
        }
        ensure!(
            p.processor_class == "Qwen3VLProcessor"
                && p.data_format == "channels_first"
                && p.do_convert_rgb
                && p.do_normalize
                && p.do_rescale
                && p.do_resize
                && p.resample == 3
                && p.rescale_factor == 1.0 / 255.0
                && p.image_mean == [0.5; 3]
                && p.image_std == [0.5; 3],
            "unsupported declared RGB processor arithmetic"
        );
        ensure!(
            p.patch_size > 0
                && p.merge_size > 0
                && p.temporal_patch_size == 2
                && p.patch_size
                    .checked_mul(p.merge_size)
                    .is_some_and(|f| f <= 1 << 24)
                && p.size.shortest_edge > 0
                && p.size.shortest_edge <= p.size.longest_edge
                && p.size.longest_edge <= 1 << 53,
            "invalid processor patch or pixel-area geometry"
        );
        Ok(())
    }
}

mod video;
pub use video::{
    Qwen3VLVisionVideoPlan, Qwen3VLVisionVideoProcessor, QWEN3_VL_VIDEO_PREPROCESS_PROFILE,
};

#[derive(Clone, Debug)]
pub struct Qwen3VLVisionImageProcessor {
    patch: u64,
    temporal: u64,
    merge: u64,
    minimum: u64,
    maximum: u64,
    limits: Qwen3VLVisionProcessorLimits,
    source_sha256: String,
}
impl Qwen3VLVisionImageProcessor {
    /// Consumes preprocessor_config.json. Missing backend defaults are explicit
    /// Qwen2-VL declarations; contradictory flags are rejected before allocation.
    pub fn from_source_bytes(source: &[u8], limits: Qwen3VLVisionProcessorLimits) -> Result<Self> {
        ensure!(
            !source.is_empty()
                && source.len() <= limits.maximum_metadata_bytes
                && limits.maximum_patch_rows > 0
                && limits.maximum_patch_bytes > 0
                && limits.maximum_patch_bytes <= isize::MAX as usize,
            "invalid image processor capacities"
        );
        let ImageMetadata {
            image_processor_type,
            mut rgb,
        } = serde_json::from_slice(source)?;
        ensure!(
            matches!(
                image_processor_type.as_str(),
                "Qwen2VLImageProcessorFast" | "Qwen2VLImageProcessor"
            ),
            "unsupported image processor type"
        );
        rgb.validate()?;
        let p = rgb;
        Ok(Self {
            patch: p.patch_size,
            temporal: p.temporal_patch_size,
            merge: p.merge_size,
            minimum: p.size.shortest_edge,
            maximum: p.size.longest_edge,
            limits,
            source_sha256: format!("{:x}", Sha256::digest(source)),
        })
    }
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }
    pub fn prepare_image(
        &self,
        architecture: Qwen3VLVisionArchitecture,
        width: u32,
        height: u32,
    ) -> Result<Qwen3VLVisionImagePlan> {
        ensure!(
            architecture.channels() == 3
                && architecture.patch_size() == self.patch
                && architecture.temporal_patch_size() == self.temporal
                && architecture.merge_size() == self.merge,
            "processor patch geometry differs from vision architecture"
        );
        let check = |w: u64, h: u64| -> Result<()> {
            ensure!(
                w > 0
                    && h > 0
                    && w <= u64::from(self.limits.image.maximum_width())
                    && h <= u64::from(self.limits.image.maximum_height())
                    && w.checked_mul(h)
                        .is_some_and(|n| n <= self.limits.image.maximum_pixels() as u64),
                "image exceeds declared dimensions or pixel capacity"
            );
            Ok(())
        };
        check(u64::from(width), u64::from(height))?;
        let resized = smart_resize(
            width,
            height,
            self.patch * self.merge,
            self.minimum,
            self.maximum,
        )?;
        check(resized.0 as u64, resized.1 as u64)?;
        let grid = Qwen3VLVisionGrid::new(
            1,
            resized.1 as u64 / self.patch,
            resized.0 as u64 / self.patch,
        )?;
        let config = Qwen3VLVisionConfig::new(architecture, vec![grid])?;
        let patch_bytes = usize::try_from(
            config
                .rows()?
                .checked_mul(config.architecture().patch_columns()?)
                .and_then(|n| n.checked_mul(2))
                .context("BF16 patch bytes overflow")?,
        )?;
        ensure!(
            config.rows()? <= self.limits.maximum_patch_rows
                && patch_bytes <= self.limits.maximum_patch_bytes,
            "vision patches exceed declared capacity"
        );
        let horizontal_bytes = (height as usize)
            .checked_mul(resized.0)
            .and_then(|n| n.checked_mul(3))
            .filter(|&n| n <= isize::MAX as usize)
            .context("RGB resize scratch overflow")?;
        Ok(Qwen3VLVisionImagePlan {
            config,
            source: (width, height),
            resized,
            patch_bytes,
            horizontal_bytes,
        })
    }
}
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionImagePlan {
    config: Qwen3VLVisionConfig,
    source: (u32, u32),
    resized: (usize, usize),
    patch_bytes: usize,
    horizontal_bytes: usize,
}
impl Qwen3VLVisionImagePlan {
    pub fn config(&self) -> &Qwen3VLVisionConfig {
        &self.config
    }
    pub fn resized_dimensions(&self) -> (usize, usize) {
        self.resized
    }
    pub fn preprocessing_profile(&self) -> &'static str {
        QWEN3_VL_IMAGE_PREPROCESS_PROFILE
    }
    /// Includes scratch and produced patches, excluding the borrowed original.
    pub fn input_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        host.backing_bytes(self.horizontal_bytes)?
            .checked_add(host.backing_bytes(self.patch_bytes)?)
            .context("image Host capacity overflow")
    }
    pub fn managed_patches(
        &self,
        host: &HostMemoryAccount,
        image: &ManagedRgba8Image,
    ) -> Result<ManagedHostTensor> {
        image.validate_parent(host)?;
        ensure!(
            (image.width(), image.height()) == self.source,
            "image dimensions differ from prepared processor input"
        );
        let mut grants = host
            .reserve_many(&[
                (HostMemoryClass::InputPreprocess, self.horizontal_bytes),
                (HostMemoryClass::InputPreprocess, self.patch_bytes),
            ])?
            .into_iter();
        let mut horizontal = grants.next().unwrap().allocate_zeroed()?;
        let mut patches = grants.next().unwrap().allocate_zeroed()?;
        let (iw, ih) = (self.source.0 as usize, self.source.1 as usize);
        let (ow, oh) = self.resized;
        let rgba = image.bytes().bytes();
        // Original convert("RGB") discards alpha rather than compositing white.
        for y in 0..ih {
            for x in 0..ow {
                for c in 0..3 {
                    horizontal.as_mut()[(y * ow + x) * 3 + c] =
                        axis_sample(x, iw, ow, |sx| rgba[(y * iw + sx) * 4 + c]);
                }
            }
        }
        let a = self.config.architecture();
        let patch = a.patch_size() as usize;
        let merge = a.merge_size() as usize;
        let grid_width = ow / patch;
        let columns = usize::try_from(a.patch_columns()?)?;
        for row in 0..usize::try_from(self.config.rows()?)? {
            let block = row / (merge * merge);
            let within = row % (merge * merge);
            let py = block / (grid_width / merge) * merge + within / merge;
            let px = block % (grid_width / merge) * merge + within % merge;
            for y in 0..patch {
                for x in 0..patch {
                    for c in 0..3 {
                        let pixel = axis_sample(py * patch + y, ih, oh, |sy| {
                            horizontal.as_ref()[(sy * ow + px * patch + x) * 3 + c]
                        });
                        let value = bf16::from_f32((f32::from(pixel) - 127.5) / 127.5)
                            .to_bits()
                            .to_le_bytes();
                        for t in 0..2 {
                            let offset =
                                (row * columns + ((c * 2 + t) * patch + y) * patch + x) * 2;
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
