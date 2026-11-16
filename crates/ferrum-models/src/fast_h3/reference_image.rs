//! Released Ref2VA image presentation, shared by its vision and VAE consumers.
//! RGB conversion precedes Lanczos resizing; the encoder then uses ImageNet
//! channel normalization in F32, separately from Qwen's fused half-range input.
use crate::qwen_image21::image_input::{
    ImageInputLimits, ImageLanczosResizePlan, ManagedRgba8Image,
};
use anyhow::{ensure, Result};
use ferrum_runtime::*;

pub const FAST_H3_REFERENCE_IMAGE_SHORT_EDGE: u32 = 2048;
/// Reference cardinality from the pinned Ref2VA input contract. Token, source
/// size and actual Host/device admission remain independent limits.
pub const FAST_H3_MAX_REFERENCE_IMAGES: usize = 9;
pub const FAST_H3_REFERENCE_IMAGE_PROFILE: &str =
    "fasth3.ref2va.rgb.lanczos22_libm0215.grid32.f32_scalar_reciprocal255.imagenet_normalize.v1";
#[derive(Clone, Debug)]
pub struct FastH3ReferenceImagePlan {
    resize: ImageLanczosResizePlan,
}
impl FastH3ReferenceImagePlan {
    pub fn new(width: u32, height: u32, short_edge: u32, limits: ImageInputLimits) -> Result<Self> {
        let (w, h) = Self::dimensions(width, height, short_edge)?;
        Ok(Self {
            resize: ImageLanczosResizePlan::new(width, height, w, h, limits)?,
        })
    }
    pub fn dimensions(width: u32, height: u32, short_edge: u32) -> Result<(u32, u32)> {
        ensure!(
            width > 0 && height > 0 && short_edge > 0,
            "reference image extents must be positive"
        );
        ensure!(
            u64::from(width) <= 4 * u64::from(height) && u64::from(height) <= 4 * u64::from(width),
            "reference image aspect must be within 1:4 and 4:1"
        );
        let scale = f64::from(short_edge) / f64::from(width.min(height));
        let axis = |n: u32| -> Result<u32> {
            let result = ((f64::from(n) * scale / 32.).round_ties_even() * 32.).max(32.);
            ensure!(
                result <= f64::from(u32::MAX),
                "reference canvas overflows u32"
            );
            Ok(result as u32)
        };
        Ok((axis(width)?, axis(height)?))
    }
    pub fn output_dimensions(&self) -> (u32, u32) {
        self.resize.dimensions()
    }
    pub fn input_peak_bytes(&self, host: &HostMemoryAccount) -> Result<usize> {
        Ok(self.resize.input_peak_bytes(host)?)
    }
    pub fn prepare(
        &self,
        source: &ManagedRgba8Image,
        host: &HostMemoryAccount,
    ) -> Result<PreparedFastH3ReferenceImage> {
        Ok(PreparedFastH3ReferenceImage {
            image: self.resize.resize_rgb_ignoring_alpha(host, source)?,
        })
    }
}
#[derive(Debug)]
pub struct PreparedFastH3ReferenceImage {
    image: ManagedRgba8Image,
}
impl PreparedFastH3ReferenceImage {
    pub fn into_image(self) -> ManagedRgba8Image {
        self.image
    }
    pub fn image(&self) -> &ManagedRgba8Image {
        &self.image
    }
    /// F32 NCHW pixels for the original T=1 encoder. The original scalar CUDA
    /// div_(255) uses an F64 reciprocal rounded to F32 before multiplication.
    /// Mean subtraction and tensor std division remain separate F32 operations.
    pub fn managed_vae_pixels(
        &self,
        host: &HostMemoryAccount,
        maximum_bytes: usize,
    ) -> Result<ManagedHostTensor> {
        self.image.validate_parent(host)?;
        let (w, h) = (
            u64::from(self.image.width()),
            u64::from(self.image.height()),
        );
        let spec = ResolvedTensorSpec::new(
            vec![1, 3, h, w],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )?;
        let bytes = usize::try_from(spec.minimum_storage_bytes()?)?;
        ensure!(
            bytes <= maximum_bytes,
            "reference VAE pixels exceed declared Host capacity"
        );
        let mut output = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
        let spatial = usize::try_from(w * h)?;
        for c in 0..3 {
            for p in 0..spatial {
                let value =
                    super::normalize_reference_rgb(self.image.bytes().bytes()[p * 4 + c], c);
                output.as_mut()[(c * spatial + p) * 4..(c * spatial + p) * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(super::video_vae::PreparedFastH3ImageEncoder::INPUT_ID)?,
            spec,
            output.freeze(),
        )?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_canvas_and_vae_pixels_preserve_released_geometry_and_rgb_semantics() {
        assert_eq!(
            FastH3ReferenceImagePlan::dimensions(832, 480, 2048).unwrap(),
            (3552, 2048)
        );
        assert_eq!(
            FastH3ReferenceImagePlan::dimensions(480, 832, 2048).unwrap(),
            (2048, 3552)
        );
        assert!(FastH3ReferenceImagePlan::dimensions(401, 100, 2048).is_err());
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                1 << 17,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1 << 17,
                    transfer_staging_bytes: 1 << 17,
                    output_bytes: 1 << 17,
                    weight_backing_bytes: 1 << 17,
                },
            )
            .unwrap(),
        );
        let limits = ImageInputLimits::new(4096, 4096, 64, 64, 4096).unwrap();
        let mut rgba = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 32 * 32 * 4)
            .unwrap();
        for (i, pixel) in rgba.as_mut().chunks_exact_mut(4).enumerate() {
            pixel.copy_from_slice(&[(i % 256) as u8, 127, 255, 0]);
        }
        let source = ManagedRgba8Image::from_rgba8(&host, rgba.freeze(), 32, 32, limits).unwrap();
        let plan = FastH3ReferenceImagePlan::new(32, 32, 32, limits).unwrap();
        assert_eq!(plan.input_peak_bytes(&host).unwrap(), 0);
        let prepared = plan.prepare(&source, &host).unwrap();
        let pixels = prepared.managed_vae_pixels(&host, 32 * 32 * 3 * 4).unwrap();
        assert_eq!(pixels.tensor().dimensions(), [1, 3, 32, 32]);
        for (i, actual) in pixels.bytes().bytes().chunks_exact(4).enumerate() {
            let c = i / 1024;
            let p = i % 1024;
            let raw = [(p % 256) as u8, 127, 255][c];
            let expected =
                ((f64::from(raw) / 255.) - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c];
            assert!(
                (f64::from(f32::from_le_bytes(actual.try_into().unwrap())) - expected).abs() < 1e-6
            );
        }
        let before = host.usage().unwrap().committed_bytes();
        assert!(prepared
            .managed_vae_pixels(&host, 32 * 32 * 3 * 4 - 1)
            .is_err());
        assert_eq!(before, host.usage().unwrap().committed_bytes());
        drop((source, prepared, pixels));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
