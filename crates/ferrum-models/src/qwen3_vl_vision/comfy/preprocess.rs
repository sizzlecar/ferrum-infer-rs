//! Comfy single-image presentation from decoded RGBA8. Alpha is discarded;
//! RGB is converted to unit F32 before bilinear interpolation, normalized by
//! (0.5, 0.5), repeated across two frames and packed in merger-block order.
//! No resized image or frame-repeat payload is allocated outside Host grants.
use super::*;
use crate::qwen_image21::image_input::ManagedRgba8Image;

/// Coordinates are computed in F64 then rounded to F32 interpolation weights.
/// This names the actual native arithmetic, without promising Torch CPU/GPU
/// interpolation binaries are bitwise identical.
pub const QWEN3_VL_COMFY_IMAGE_PREPROCESS_PROFILE: &str =
    "qwen3vl.vision_comfy.rgba8.rgb_f32.bilinear_half_pixel_f64_weights_f32.normalize_half.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Qwen3VLVisionComfyPixelPolicy {
    minimum_pixels: u64,
    maximum_pixels: u64,
}
impl Default for Qwen3VLVisionComfyPixelPolicy {
    fn default() -> Self {
        Self {
            minimum_pixels: 3136,
            maximum_pixels: 12_845_056,
        }
    }
}
impl Qwen3VLVisionComfyPixelPolicy {
    /// These are the reference resize thresholds. Factor rounding can yield
    /// a different pixel count; the resulting exact allocation still passes
    /// through the caller's ordinary Host capacity checks.
    pub fn new(minimum_pixels: u64, maximum_pixels: u64) -> Result<Self> {
        ensure!(
            minimum_pixels > 0 && minimum_pixels <= maximum_pixels && maximum_pixels <= (1 << 53),
            "Comfy pixel thresholds must be positive, ordered and exactly representable"
        );
        Ok(Self {
            minimum_pixels,
            maximum_pixels,
        })
    }
}

/// Device-free image shape preparation. All payload dimensions and byte
/// extents are checked before any allocation or interpolation work.
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionComfyImagePlan {
    config: Qwen3VLVisionComfyConfig,
    source_width: u32,
    source_height: u32,
    resized_width: usize,
    resized_height: usize,
    patch: usize,
    merge: usize,
    tensor: ResolvedTensorSpec,
    payload_bytes: usize,
}
impl Qwen3VLVisionComfyImagePlan {
    pub fn new(
        architecture: Qwen3VLVisionArchitecture,
        source_width: u32,
        source_height: u32,
        policy: Qwen3VLVisionComfyPixelPolicy,
    ) -> Result<Self> {
        ensure!(
            source_width > 0 && source_height > 0,
            "Comfy image dimensions must be positive"
        );
        ensure!(
            architecture.channels() == 3 && architecture.temporal_patch_size() == 2,
            "Comfy single-image presentation requires RGB and two repeated frames"
        );
        let factor = architecture
            .patch_size()
            .checked_mul(architecture.merge_size())
            .context("visual resize factor overflows")?;
        let source_pixels = u64::from(source_width) * u64::from(source_height);
        ensure!(
            factor <= (1 << 24) && source_pixels <= (1 << 53),
            "visual resize coordinates exceed the declared numerical profile"
        );
        let factor_f = factor as f64;
        let h = f64::from(source_height);
        let w = f64::from(source_width);
        let mut height = (h / factor_f).round_ties_even() * factor_f;
        let mut width = (w / factor_f).round_ties_even() * factor_f;
        if height * width > policy.maximum_pixels as f64 {
            let beta = (source_pixels as f64 / policy.maximum_pixels as f64).sqrt();
            height = (h / beta / factor_f).floor().max(1.) * factor_f;
            width = (w / beta / factor_f).floor().max(1.) * factor_f;
        } else if height * width < policy.minimum_pixels as f64 {
            let beta = (policy.minimum_pixels as f64 / source_pixels as f64).sqrt();
            height = (h * beta / factor_f).ceil() * factor_f;
            width = (w * beta / factor_f).ceil() * factor_f;
        }
        ensure!(
            height.is_finite()
                && width.is_finite()
                && height >= factor_f
                && width >= factor_f
                && height <= (1u64 << 24) as f64
                && width <= (1u64 << 24) as f64,
            "resized visual dimensions exceed the declared numerical profile"
        );
        let grid = Qwen3VLVisionGrid::new(
            1,
            height as u64 / architecture.patch_size(),
            width as u64 / architecture.patch_size(),
        )?;
        let patch = usize::try_from(architecture.patch_size())?;
        let merge = usize::try_from(architecture.merge_size())?;
        let config = Qwen3VLVisionComfyConfig::new(architecture, grid)?;
        let tensor = ResolvedTensorSpec::new(
            vec![config.rows()?, config.architecture().patch_columns()?],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let payload_bytes = usize::try_from(tensor.minimum_storage_bytes()?)?;
        ensure!(
            payload_bytes <= isize::MAX as usize,
            "visual patch storage exceeds Host address space"
        );
        Ok(Self {
            config,
            source_width,
            source_height,
            resized_width: width as usize,
            resized_height: height as usize,
            patch,
            merge,
            tensor,
            payload_bytes,
        })
    }
    pub fn config(&self) -> &Qwen3VLVisionComfyConfig {
        &self.config
    }
    pub fn resized_dimensions(&self) -> (usize, usize) {
        (self.resized_width, self.resized_height)
    }
    /// The decoded image is borrowed; this is only the new exact patch owner.
    pub fn input_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        Ok(host.backing_bytes(self.payload_bytes)?)
    }
    pub fn managed_patches(
        &self,
        host: &HostMemoryAccount,
        image: &ManagedRgba8Image,
    ) -> Result<ManagedHostTensor> {
        image.validate_parent(host)?;
        ensure!(
            image.width() == self.source_width && image.height() == self.source_height,
            "image differs from the prepared original dimensions"
        );
        let mut patches = host
            .reserve(HostMemoryClass::InputPreprocess, self.payload_bytes)?
            .allocate_zeroed()?;
        let grid_width = self.resized_width / self.patch;
        let patch_area = self.patch * self.patch;
        let columns = 3 * 2 * patch_area;
        let rgb = image.bytes().bytes();
        let read = |y: usize, x: usize, c: usize| {
            f32::from(rgb[(y * self.source_width as usize + x) * 4 + c]) / 255.
        };
        for row in 0..usize::try_from(self.config.rows()?)? {
            let patch_y =
                row / (self.merge * grid_width) * self.merge + row / self.merge % self.merge;
            let patch_x = row / (self.merge * self.merge) % (grid_width / self.merge) * self.merge
                + row % self.merge;
            for y in 0..self.patch {
                let (ys, yw) = taps(
                    patch_y * self.patch + y,
                    self.resized_height,
                    self.source_height as usize,
                );
                for x in 0..self.patch {
                    let (xs, xw) = taps(
                        patch_x * self.patch + x,
                        self.resized_width,
                        self.source_width as usize,
                    );
                    for c in 0..3 {
                        let upper = read(ys[0], xs[0], c) * xw[0] + read(ys[0], xs[1], c) * xw[1];
                        let lower = read(ys[1], xs[0], c) * xw[0] + read(ys[1], xs[1], c) * xw[1];
                        let value = ((upper * yw[0] + lower * yw[1]) - 0.5) / 0.5;
                        for t in 0..2 {
                            let index =
                                row * columns + (c * 2 + t) * patch_area + y * self.patch + x;
                            patches.as_mut()[index * 4..index * 4 + 4]
                                .copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(Qwen3VLVisionComfyFamily::PATCHES_ID)?,
            self.tensor.clone(),
            patches.freeze(),
        )?)
    }
}
fn taps(dst: usize, output: usize, input: usize) -> ([usize; 2], [f32; 2]) {
    let p = ((input as f64 / output as f64) * (dst as f64 + 0.5) - 0.5).max(0.);
    let low = (p.floor() as usize).min(input - 1);
    let weight = (p - low as f64) as f32;
    ([low, (low + 1).min(input - 1)], [1. - weight, weight])
}

#[cfg(test)]
mod tests;
