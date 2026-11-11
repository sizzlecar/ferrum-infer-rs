//! Native separable cubic AA: Keys a=-0.5, half-pixel centres, truncated
//! normalized border support, F64 coefficients and U8 RNE/clamp per axis.
//! Torchvision may use fixed-point/SIMD kernels; binary equality is unqualified.
use super::*;
use crate::qwen_image21::image_input::{ImageLanczosResizePlan, ManagedRgba8Image};

pub const QWEN3_VL_HF_IMAGE_PREPROCESS_PROFILE: &str = "qwen3_5.hf_input.rgb8.outer_lanczos22_libm0215.1mp_grid32_keep20.inner_cubic_aa_a05_f64_u8_rne.rgb_f32_fused_half.patch_merge.v1";
const PROCESSOR_ONLY_PROFILE: &str = "qwen3_5.hf_input.rgb8.prepared_outer.inner_cubic_aa_a05_f64_u8_rne.rgb_f32_fused_half.patch_merge.v1";

#[derive(Clone, Debug)]
pub struct Qwen3VLVisionHfImagePlan {
    config: Qwen3VLVisionHfConfig,
    source: (u32, u32),
    outer: ImageLanczosResizePlan,
    outer_dimensions: (u32, u32),
    resized: (usize, usize),
    horizontal_bytes: usize,
    patch_bytes: usize,
    complete_outer: bool,
}
impl Qwen3VLVisionHfImagePlan {
    /// Full Diffusers upsampler image presentation, including its RGB Lanczos
    /// outer resize. The source RGBA owner is borrowed, never reclassified.
    pub fn new(
        a: Qwen3VLVisionArchitecture,
        width: u32,
        height: u32,
        p: &Qwen3VLVisionHfProcessor,
    ) -> Result<Self> {
        Self::build(a, width, height, p, true)
    }
    /// Explicitly consume an already presented RGB image. This is useful for
    /// independent processor tests and callers with an actual outer receipt.
    pub fn processor_input(
        a: Qwen3VLVisionArchitecture,
        width: u32,
        height: u32,
        p: &Qwen3VLVisionHfProcessor,
    ) -> Result<Self> {
        Self::build(a, width, height, p, false)
    }
    fn build(
        a: Qwen3VLVisionArchitecture,
        width: u32,
        height: u32,
        p: &Qwen3VLVisionHfProcessor,
        complete_outer: bool,
    ) -> Result<Self> {
        p.check_architecture(&a)?;
        ensure!(
            width > 0 && height > 0,
            "positive original image dimensions required"
        );
        ensure!(
            width <= p.limits.image.maximum_width()
                && height <= p.limits.image.maximum_height()
                && u64::from(width) * u64::from(height) <= p.limits.image.maximum_pixels() as u64,
            "original image exceeds declared ingress limits"
        );
        let outer_dimensions = if complete_outer {
            upsampler_dimensions(width, height)?
        } else {
            (width, height)
        };
        let outer = ImageLanczosResizePlan::new(
            width,
            height,
            outer_dimensions.0,
            outer_dimensions.1,
            p.limits.image,
        )?;
        let (ow, oh) = outer_dimensions;
        ensure!(
            f64::from(ow.max(oh)) / f64::from(ow.min(oh)) <= 200.,
            "HF image aspect ratio exceeds 200"
        );
        let factor = p.patch.checked_mul(p.merge).context("HF factor overflow")? as f64;
        let (w, h) = (f64::from(ow), f64::from(oh));
        let (mut rw, mut rh) = (
            (w / factor).round_ties_even() * factor,
            (h / factor).round_ties_even() * factor,
        );
        if rw * rh > p.maximum_pixels as f64 {
            let beta = (w * h / p.maximum_pixels as f64).sqrt();
            rw = (w / beta / factor).floor().max(1.) * factor;
            rh = (h / beta / factor).floor().max(1.) * factor;
        } else if rw * rh < p.minimum_pixels as f64 {
            let beta = (p.minimum_pixels as f64 / (w * h)).sqrt();
            rw = (w * beta / factor).ceil() * factor;
            rh = (h * beta / factor).ceil() * factor;
        }
        ensure!(
            rw >= factor && rh >= factor && rw <= (1u64 << 24) as f64 && rh <= (1u64 << 24) as f64,
            "resized coordinates exceed exact F32 integer range"
        );
        let (rw, rh) = (rw as usize, rh as usize);
        ensure!(
            rw <= p.limits.image.maximum_width() as usize
                && rh <= p.limits.image.maximum_height() as usize
                && rw
                    .checked_mul(rh)
                    .is_some_and(|n| n <= p.limits.image.maximum_pixels()),
            "HF resized image exceeds declared image limits"
        );
        let config = Qwen3VLVisionHfConfig::new(
            a,
            Qwen3VLVisionGrid::new(1, rh as u64 / p.patch, rw as u64 / p.patch)?,
            p,
        )?;
        let rows = config.rows()?;
        ensure!(
            rows <= p.limits.max_patch_rows,
            "HF patches exceed declared row capacity"
        );
        let patch_bytes = usize::try_from(
            rows.checked_mul(config.architecture().patch_columns()?)
                .and_then(|n| n.checked_mul(4))
                .context("HF patch extent overflow")?,
        )?;
        ensure!(
            patch_bytes <= p.limits.max_patch_bytes,
            "HF patch bytes exceed declared capacity"
        );
        let horizontal_bytes = usize::try_from(oh)?
            .checked_mul(rw)
            .and_then(|n| n.checked_mul(3))
            .filter(|&n| n <= isize::MAX as usize)
            .context("HF AA scratch extent overflow")?;
        Ok(Self {
            config,
            source: (width, height),
            outer,
            outer_dimensions,
            resized: (rw, rh),
            horizontal_bytes,
            patch_bytes,
            complete_outer,
        })
    }
    pub fn config(&self) -> &Qwen3VLVisionHfConfig {
        &self.config
    }
    pub fn outer_dimensions(&self) -> (u32, u32) {
        self.outer_dimensions
    }
    pub fn resized_dimensions(&self) -> (usize, usize) {
        self.resized
    }
    pub fn preprocessing_profile(&self) -> &'static str {
        if self.complete_outer {
            QWEN3_VL_HF_IMAGE_PREPROCESS_PROFILE
        } else {
            PROCESSOR_ONLY_PROFILE
        }
    }
    /// Maximum of the real outer-resize phase and the inner-resize phase,
    /// excluding the borrowed original RGBA owner and separately built positions.
    pub fn input_backing_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        let patches = host.backing_bytes(self.patch_bytes)?;
        let inner = self
            .outer
            .retained_input_bytes(host)?
            .checked_add(host.backing_bytes(self.horizontal_bytes)?)
            .and_then(|n| n.checked_add(patches))
            .context("HF preprocessing physical capacity overflow")?;
        Ok(self.outer.input_peak_bytes(host)?.max(inner))
    }
    pub fn retained_input_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        Ok(host.backing_bytes(self.patch_bytes)?)
    }
    pub fn managed_patches(
        &self,
        host: &HostMemoryAccount,
        source: &ManagedRgba8Image,
    ) -> Result<ManagedHostTensor> {
        source.validate_parent(host)?;
        ensure!(
            (source.width(), source.height()) == self.source,
            "original image differs from prepared HF dimensions"
        );
        let outer = self.outer.resize_rgb_ignoring_alpha(host, source)?;
        let mut grants = host
            .reserve_many(&[
                (HostMemoryClass::InputPreprocess, self.horizontal_bytes),
                (HostMemoryClass::InputPreprocess, self.patch_bytes),
            ])?
            .into_iter();
        let mut horizontal = grants.next().unwrap().allocate_zeroed()?;
        let mut patches = grants.next().unwrap().allocate_zeroed()?;
        let (ow, oh) = (outer.width() as usize, outer.height() as usize);
        let (rw, rh) = self.resized;
        let rgb = outer.bytes().bytes();
        for y in 0..oh {
            for x in 0..rw {
                for channel in 0..3 {
                    horizontal.as_mut()[(y * rw + x) * 3 + channel] =
                        axis_sample(x, ow, rw, |sx| rgb[(y * ow + sx) * 4 + channel]);
                }
            }
        }
        let a = self.config.architecture();
        let patch = a.patch_size() as usize;
        let merge = a.merge_size() as usize;
        let columns = a.patch_columns()? as usize;
        let patch_area = patch * patch;
        let grid_width = rw / patch;
        for row in 0..usize::try_from(self.config.rows()?)? {
            let py = row / (merge * grid_width) * merge + row / merge % merge;
            let px = row / (merge * merge) % (grid_width / merge) * merge + row % merge;
            for y in 0..patch {
                let dy = py * patch + y;
                for x in 0..patch {
                    let dx = px * patch + x;
                    for channel in 0..3 {
                        let pixel = axis_sample(dy, oh, rh, |sy| {
                            horizontal.as_ref()[(sy * rw + dx) * 3 + channel]
                        });
                        // HF's fused uint8 rescale/normalize uses mean/std*255.
                        let value = (f32::from(pixel) - 127.5) / 127.5;
                        for t in 0..2 {
                            let offset =
                                (row * columns + (channel * 2 + t) * patch_area + y * patch + x)
                                    * 4;
                            patches.as_mut()[offset..offset + 4]
                                .copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(Qwen3VLVisionHfFamily::PATCHES_ID)?,
            ResolvedTensorSpec::new(
                vec![self.config.rows()?, a.patch_columns()?],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            patches.freeze(),
        )?)
    }
}

fn upsampler_dimensions(width: u32, height: u32) -> Result<(u32, u32)> {
    let area = u64::from(width) * u64::from(height);
    if width % 32 == 0 && height % 32 == 0 && (800_000..=1_200_000).contains(&area) {
        return Ok((width, height));
    }
    let ratio = f64::from(width) / f64::from(height);
    let w = ((1_000_000. * ratio).sqrt() / 32.)
        .round_ties_even()
        .max(1.)
        * 32.;
    let h = ((1_000_000. / ratio).sqrt() / 32.)
        .round_ties_even()
        .max(1.)
        * 32.;
    ensure!(
        w <= f64::from(u32::MAX) && h <= f64::from(u32::MAX),
        "upsampler image dimensions overflow"
    );
    Ok((w as u32, h as u32))
}
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        ((-0.5 * x + 2.5) * x - 4.) * x + 2.
    } else {
        0.
    }
}
fn axis_sample(index: usize, input: usize, output: usize, read: impl Fn(usize) -> u8) -> u8 {
    if input == output {
        return read(index);
    }
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let center = (index as f64 + 0.5) * scale;
    let support = 2. * filter_scale;
    let start = ((center - support + 0.5).floor() as i64).max(0) as usize;
    let end = ((center + support + 0.5).floor() as usize).min(input);
    let mut weight = 0.;
    let mut value = 0.;
    for source in start..end {
        let w = cubic((source as f64 - center + 0.5) / filter_scale);
        weight += w;
        value += f64::from(read(source)) * w;
    }
    (value / weight).round_ties_even().clamp(0., 255.) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_aa_has_expanded_support_normalized_borders_and_u8_rne() {
        // Downsample a two-pixel impulse: 127.5 rounds to the even byte 128.
        assert_eq!(axis_sample(0, 2, 1, |i| [0, 255][i]), 128);
        // Truncated AA borders renormalize: 255*(29/128)/(140/128)=52.8214...
        // rounds to 53. Bilinear or zero padding implements another profile.
        assert_eq!(axis_sample(1, 2, 4, |i| [0, 255][i]), 53);
        assert_eq!(axis_sample(0, 2, 4, |i| [0, 255][i]), 0);
        for (input, output) in [(13, 3), (3, 13)] {
            for i in 0..output {
                assert_eq!(axis_sample(i, input, output, |_| 173), 173);
            }
        }
        assert_eq!(
            super::upsampler_dimensions(1024, 1024).unwrap(),
            (1024, 1024)
        );
        assert_eq!(super::upsampler_dimensions(64, 64).unwrap(), (992, 992));
    }
}
