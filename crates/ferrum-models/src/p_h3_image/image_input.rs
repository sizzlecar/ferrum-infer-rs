//! The Comfy outer reference-image resize, before vision patch preparation or
//! VAE normalization. The two area caps are independent of each other.
use super::PH3ImageReferenceImage;
use crate::qwen_image21::image_input::{
    ImageInputLimits, ImageLanczosResizePlan, ManagedRgba8Image,
};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};

/// Native coefficients and square root are explicit numerical choices, not a
/// claim of deployed Pillow/libc or Torch interpolation binary equivalence.
pub const PH3_IMAGE_EDIT_PIXEL_PROFILE: &str =
    "p_h3_image.edit.rgb8.ignore_alpha.down_only.sqrt_f64_native.ties_even32.lanczos22.libm_0_2_15.unit_f32_div255.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageEditPixelPolicy {
    vision_maximum_pixels: u64,
}
impl Default for PH3ImageEditPixelPolicy {
    fn default() -> Self {
        Self {
            vision_maximum_pixels: 1_000_000,
        }
    }
}
impl PH3ImageEditPixelPolicy {
    pub fn new(vision_maximum_pixels: u64) -> Result<Self> {
        ensure!(
            vision_maximum_pixels > 0 && vision_maximum_pixels <= (1 << 53),
            "H3 vision pixel cap must be positive and exactly representable"
        );
        Ok(Self {
            vision_maximum_pixels,
        })
    }
    pub fn vision_maximum_pixels(self) -> u64 {
        self.vision_maximum_pixels
    }
}

/// Every byte count is physical for the actual Host alignment. The original
/// decoded owner is borrowed, and never included in new Input requirements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PH3ImageEditPixelHostRequirements {
    pub decoded_input_bytes: usize,
    pub input_peak_bytes: usize,
    /// New owners remaining after prepare; excludes any decoded identity alias.
    pub retained_input_bytes: usize,
    pub vision_new_input_bytes: usize,
    /// Actual retained owner, including an identity alias's original backing.
    pub vision_backing_bytes: usize,
    pub vae_pixel_input_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct PH3ImageEditPixelPlan {
    source: (u32, u32),
    vision: ImageLanczosResizePlan,
    vae: ImageLanczosResizePlan,
    vae_tensor: ResolvedTensorSpec,
    vae_pixel_bytes: usize,
}
impl PH3ImageEditPixelPlan {
    pub fn new(
        source_width: u32,
        source_height: u32,
        generation_width: u32,
        generation_height: u32,
        policy: PH3ImageEditPixelPolicy,
        limits: ImageInputLimits,
    ) -> Result<Self> {
        PH3ImageEditPixelPolicy::new(policy.vision_maximum_pixels)?;
        let area = u64::from(generation_width) * u64::from(generation_height);
        ensure!(
            generation_width > 0 && generation_height > 0 && area <= (1 << 53),
            "H3 generation area must be positive and exactly representable"
        );
        let vision = size_within_area(source_width, source_height, policy.vision_maximum_pixels)?;
        let vae = size_within_area(source_width, source_height, area)?;
        let vae_tensor = ResolvedTensorSpec::new(
            vec![1, 3, u64::from(vae.1), u64::from(vae.0)],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let vae_pixel_bytes = usize::try_from(vae_tensor.minimum_storage_bytes()?)?;
        ensure!(
            vae_pixel_bytes <= isize::MAX as usize,
            "H3 VAE RGB input exceeds Host address space"
        );
        Ok(Self {
            source: (source_width, source_height),
            vision: ImageLanczosResizePlan::new(
                source_width,
                source_height,
                vision.0,
                vision.1,
                limits,
            )?,
            vae: ImageLanczosResizePlan::new(source_width, source_height, vae.0, vae.1, limits)?,
            vae_tensor,
            vae_pixel_bytes,
        })
    }

    pub fn vision_dimensions(&self) -> (u32, u32) {
        self.vision.dimensions()
    }
    pub fn vae_dimensions(&self) -> (u32, u32) {
        self.vae.dimensions()
    }
    pub fn reference_image(&self, spatial_scale: u64) -> Result<PH3ImageReferenceImage> {
        let (w, h) = self.vae_dimensions();
        ensure!(
            spatial_scale > 0
                && u64::from(w) % spatial_scale == 0
                && u64::from(h) % spatial_scale == 0,
            "H3 reference dimensions must divide by the declared VAE scale"
        );
        PH3ImageReferenceImage::new(
            usize::try_from(u64::from(h) / spatial_scale)?,
            usize::try_from(u64::from(w) / spatial_scale)?,
        )
    }

    /// Pure geometry, before decode: new peak above the borrowed decoded RGB.
    /// Vision completes first. Its new image remains while VAE coefficients,
    /// intermediate and final RGB coexist, then while F32 NCHW is allocated.
    /// Equal outer geometries reuse one RGB result rather than resize twice.
    pub fn new_input_peak_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        let vision_peak = self.vision.input_peak_bytes(host)?;
        let vision = self.vision.retained_input_bytes(host)?;
        let pixels = host.backing_bytes(self.vae_pixel_bytes)?;
        if self.vision_dimensions() == self.vae_dimensions() {
            return Ok(vision_peak.max(sum(&[vision, pixels])?));
        }
        let vae_peak = self.vae.input_peak_bytes(host)?;
        let vae = self.vae.retained_input_bytes(host)?;
        Ok(vision_peak
            .max(sum(&[vision, vae_peak])?)
            .max(sum(&[vision, vae, pixels])?))
    }

    /// Include the real decoded owner so identity cannot undercount a slice
    /// retaining a larger allocation. This performs no grants or pixel work.
    pub fn host_requirements(
        &self,
        host: &HostMemoryAccount,
        decoded: &ManagedRgba8Image,
    ) -> Result<PH3ImageEditPixelHostRequirements> {
        self.validate_source(host, decoded)?;
        let vision_new_input_bytes = self.vision.retained_input_bytes(host)?;
        let vae_pixel_input_bytes = host.backing_bytes(self.vae_pixel_bytes)?;
        Ok(PH3ImageEditPixelHostRequirements {
            decoded_input_bytes: decoded.bytes().backing_bytes(),
            input_peak_bytes: self.new_input_peak_bytes(host)?,
            retained_input_bytes: sum(&[vision_new_input_bytes, vae_pixel_input_bytes])?,
            vision_new_input_bytes,
            vision_backing_bytes: if self.vision.is_identity() {
                decoded.bytes().backing_bytes()
            } else {
                vision_new_input_bytes
            },
            vae_pixel_input_bytes,
        })
    }

    pub fn prepare(
        &self,
        host: &HostMemoryAccount,
        decoded: &ManagedRgba8Image,
    ) -> Result<PH3ImageEditPixels> {
        // Validate the complete geometry/physical query before pixel work.
        // The caller may already supply its execution scope: every allocation
        // below spends that existing authority, without a nested scope probe.
        self.host_requirements(host, decoded)?;
        let vision = self.vision.resize_rgb_ignoring_alpha(host, decoded)?;
        let vae_rgb = if self.vision_dimensions() == self.vae_dimensions() {
            vision.clone()
        } else {
            self.vae.resize_rgb_ignoring_alpha(host, decoded)?
        };
        let mut bytes =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, self.vae_pixel_bytes)?;
        let plane = usize::try_from(u64::from(vae_rgb.width()) * u64::from(vae_rgb.height()))?;
        for (index, pixel) in vae_rgb.bytes().bytes().chunks_exact(4).enumerate() {
            for channel in 0..3 {
                let value = f32::from(pixel[channel]) / 255.0;
                let offset = (channel * plane + index) * 4;
                bytes.as_mut()[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        let vae = ManagedHostTensor::new(
            ProgramValueId::new("value.p_h3_image.reference.unit_rgb")?,
            self.vae_tensor.clone(),
            bytes.freeze(),
        )?;
        Ok(PH3ImageEditPixels { vision, vae })
    }

    fn validate_source(&self, host: &HostMemoryAccount, decoded: &ManagedRgba8Image) -> Result<()> {
        decoded.validate_parent(host)?;
        ensure!(
            (decoded.width(), decoded.height()) == self.source,
            "decoded image differs from its H3 outer pixel plan"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PH3ImageEditPixels {
    pub vision: ManagedRgba8Image,
    pub vae: ManagedHostTensor,
}
impl PH3ImageEditPixels {
    /// These two owners have different storage and cannot alias each other.
    /// Identity vision may still share the caller's original decoded owner.
    pub fn retained_owner_bytes(&self) -> Result<usize> {
        sum(&[
            self.vision.bytes().backing_bytes(),
            self.vae.bytes().backing_bytes(),
        ])
    }
}

fn size_within_area(width: u32, height: u32, pixels: u64) -> Result<(u32, u32)> {
    let area = u64::from(width) * u64::from(height);
    ensure!(
        width > 0 && height > 0 && area <= (1 << 53) && pixels > 0 && pixels <= (1 << 53),
        "H3 outer resize requires positive exactly representable areas"
    );
    let scale = ((pixels as f64 / area as f64).sqrt()).min(1.0);
    let axis = |side: u32| -> Result<u32> {
        let rounded = ((f64::from(side) * scale / 32.0).round_ties_even() * 32.0).max(32.0);
        ensure!(
            rounded.is_finite() && rounded <= f64::from(u32::MAX),
            "H3 rounded outer dimension overflows"
        );
        Ok(rounded as u32)
    };
    Ok((axis(width)?, axis(height)?))
}
fn sum(bytes: &[usize]) -> Result<usize> {
    bytes.iter().try_fold(0usize, |total, &n| {
        total
            .checked_add(n)
            .filter(|&v| v <= isize::MAX as usize)
            .context("H3 edit pixel backing overflows")
    })
}

#[cfg(test)]
mod tests;
