use super::*;
use ferrum_runtime::{
    ElementType, ManagedHostTensor, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
};
use half::bf16;
use serde::Deserialize;

/// Checked image-processor metadata plus the fixed single-frame edit adapter.
/// No model-name dispatch is used. The supported metadata explicitly requests
/// RGB conversion, rescaling and normalization with channels-first patchification.
#[derive(Debug, Clone)]
pub struct QwenImage21ImageProcessor {
    patch: usize,
    temporal: usize,
    merge: usize,
    minimum_pixels: usize,
    maximum_pixels: usize,
    fused_mean: [f32; 3],
    fused_std: [f32; 3],
}

#[derive(Deserialize)]
struct Metadata {
    image_processor: Processor,
    processor_class: String,
}

#[derive(Deserialize)]
struct Processor {
    data_format: String,
    do_convert_rgb: bool,
    do_normalize: bool,
    do_rescale: bool,
    do_resize: bool,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    image_processor_type: String,
    merge_size: usize,
    patch_size: usize,
    resample: u32,
    rescale_factor: f64,
    size: PixelBounds,
    temporal_patch_size: usize,
}

#[derive(Deserialize)]
struct PixelBounds {
    shortest_edge: usize,
    longest_edge: usize,
}

struct ConditionLayout {
    pixels: usize,
    gh: usize,
    gw: usize,
    rows: usize,
    columns: usize,
    patch_bytes: usize,
    vae_bytes: usize,
}

impl QwenImage21ImageProcessor {
    pub const PATCHES_ID: &'static str = "value.qwen3_vl.vision.patches";
    pub const VAE_INPUT_ID: &'static str = "value.qwen_image21.vae.encoder.normalized_rgba";
    /// This identifies local preprocessing arithmetic, not deployed Torch or
    /// Pillow bitwise qualification. The no-resize path does not evaluate sinc.
    pub const NUMERICAL_PROFILE: &'static str =
        "qwen_image21.edit.png8.lanczos22.libm025.fused_vision.staged_vae.bf16.v1";

    pub fn from_json(bytes: &[u8]) -> ImageInputResult<Self> {
        // Typed deserialization rejects duplicate consumed fields. Unused video
        // metadata remains outside this single-image adapter's input domain.
        let metadata: Metadata = serde_json::from_slice(bytes)
            .map_err(|e| ImageInputError::Configuration(e.to_string()))?;
        let p = metadata.image_processor;
        if metadata.processor_class != "Qwen3VLProcessor"
            || !matches!(
                p.image_processor_type.as_str(),
                "Qwen2VLImageProcessorFast" | "Qwen2VLImageProcessor"
            )
            || p.data_format != "channels_first"
            || !p.do_convert_rgb
            || !p.do_normalize
            || !p.do_rescale
            || !p.do_resize
            || p.resample != 3
        {
            return Err(ImageInputError::Unsupported(
                "processor conversion/normalization/resize policy",
            ));
        }
        let factor = p
            .patch_size
            .checked_mul(p.merge_size)
            .filter(|&n| n > 0 && n <= 32)
            .ok_or(ImageInputError::Configuration(
                "invalid patch/merge factor".into(),
            ))?;
        if 32 % factor != 0
            || p.temporal_patch_size == 0
            || p.temporal_patch_size > 32
            || p.size.shortest_edge == 0
            || p.size.longest_edge < p.size.shortest_edge
            || p.size.longest_edge > isize::MAX as usize / 4
            || !p.rescale_factor.is_finite()
            || p.rescale_factor <= 0.0
        {
            return Err(ImageInputError::Configuration(
                "unsupported checked processor geometry or scale".into(),
            ));
        }
        // Transformers v5.19.0 TorchvisionBackend first makes FP32 mean/std,
        // then multiplies each by the FP32 scalar (1 / rescale_factor).
        let inverse = (1.0 / p.rescale_factor) as f32;
        let fused_mean = p.image_mean.map(|x| (x as f32) * inverse);
        let fused_std = p.image_std.map(|x| (x as f32) * inverse);
        if !inverse.is_finite()
            || inverse <= 0.0
            || fused_mean.iter().any(|x| !x.is_finite())
            || fused_std.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(ImageInputError::Configuration(
                "non-finite or nonpositive fused normalization".into(),
            ));
        }
        // Validate every possible byte before allocating image-sized tensors.
        for c in 0..3 {
            for pixel in 0..=255u16 {
                let value = (f32::from(pixel) - fused_mean[c]) / fused_std[c];
                if !bf16::from_f32(value).is_finite() {
                    return Err(ImageInputError::Configuration(
                        "normalization overflows BF16".into(),
                    ));
                }
            }
        }
        Ok(Self {
            patch: p.patch_size,
            temporal: p.temporal_patch_size,
            merge: p.merge_size,
            minimum_pixels: p.size.shortest_edge,
            maximum_pixels: p.size.longest_edge,
            fused_mean,
            fused_std,
        })
    }

    pub fn patch_size(&self) -> usize {
        self.patch
    }
    pub fn temporal_patch_size(&self) -> usize {
        self.temporal
    }
    pub fn merge_size(&self) -> usize {
        self.merge
    }

    /// The pinned edit pipeline uses target area 1024² and rounds each axis to
    /// the nearest multiple of 32 (ties to even), independently of output size.
    pub fn condition_dimensions(width: u32, height: u32) -> ImageInputResult<(u32, u32)> {
        if width == 0 || height == 0 {
            return Err(ImageInputError::Invalid("empty image dimension"));
        }
        let ratio = f64::from(width) / f64::from(height);
        if ratio > 200.0 || ratio < 1.0 / 200.0 {
            return Err(ImageInputError::Unsupported(
                "image aspect ratio exceeds processor bound",
            ));
        }
        let w = libm::sqrt(1_048_576.0 * ratio);
        let h = w / ratio;
        let w = (w / 32.0).round_ties_even() * 32.0;
        let h = (h / 32.0).round_ties_even() * 32.0;
        if !w.is_finite()
            || !h.is_finite()
            || w < 32.0
            || h < 32.0
            || w > f64::from(u32::MAX)
            || h > f64::from(u32::MAX)
        {
            return Err(ImageInputError::Limit("condition dimensions"));
        }
        Ok((w as u32, h as u32))
    }

    pub fn prepare(
        &self,
        host: &HostMemoryAccount,
        image: &ManagedRgba8Image,
        limits: ImageInputLimits,
    ) -> ImageInputResult<QwenImage21ConditionInputs> {
        image.validate_parent(host)?;
        let (width, height) = Self::condition_dimensions(image.width, image.height)?;
        // Resize before white compositing. RGBA resampling preserves the
        // premultiplied-alpha roundings; VAE retains all four channels.
        let resized = resize::lanczos(host, image, width, height, limits)?;
        self.prepare_resized(host, &resized)
    }

    /// Physical Host backing retained by the prepared vision patches and VAE
    /// input for an original image's dimensions. This uses the same condition
    /// resizing and processor geometry checks as `prepare`, without decoding,
    /// reserving or allocating. Encoded input, decoded RGBA and resize scratch
    /// have separate lifetimes and are not included in this retained total.
    pub fn prepared_backing_bytes(
        &self,
        host: &HostMemoryAccount,
        width: u32,
        height: u32,
    ) -> ImageInputResult<usize> {
        let (width, height) = Self::condition_dimensions(width, height)?;
        let layout = self.condition_layout(width, height)?;
        host.backing_bytes(layout.patch_bytes)?
            .checked_add(host.backing_bytes(layout.vae_bytes)?)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(ImageInputError::Limit("prepared image backing bytes"))
    }

    fn condition_layout(&self, width: u32, height: u32) -> ImageInputResult<ConditionLayout> {
        let width = width as usize;
        let height = height as usize;
        let pixels = width
            .checked_mul(height)
            .ok_or(ImageInputError::Limit("image pixels"))?;
        let factor = self.patch * self.merge;
        if width % factor != 0
            || height % factor != 0
            || pixels < self.minimum_pixels
            || pixels > self.maximum_pixels
        {
            return Err(ImageInputError::Unsupported(
                "processor requires an additional bicubic resize",
            ));
        }
        let gh = height / self.patch;
        let gw = width / self.patch;
        let rows = gh
            .checked_mul(gw)
            .ok_or(ImageInputError::Limit("patch rows"))?;
        let columns = 3usize
            .checked_mul(self.temporal)
            .and_then(|n| n.checked_mul(self.patch))
            .and_then(|n| n.checked_mul(self.patch))
            .ok_or(ImageInputError::Limit("patch columns"))?;
        let patch_bytes = byte_len(rows.checked_mul(columns))?;
        let vae_bytes = byte_len(pixels.checked_mul(4))?;
        Ok(ConditionLayout {
            pixels,
            gh,
            gw,
            rows,
            columns,
            patch_bytes,
            vae_bytes,
        })
    }

    fn prepare_resized(
        &self,
        host: &HostMemoryAccount,
        image: &ManagedRgba8Image,
    ) -> ImageInputResult<QwenImage21ConditionInputs> {
        image.validate_parent(host)?;
        let width = image.width as usize;
        let ConditionLayout {
            pixels,
            gh,
            gw,
            rows,
            columns,
            patch_bytes,
            vae_bytes,
        } = self.condition_layout(image.width, image.height)?;
        let patch_spec = tensor_spec(vec![rows as u64, columns as u64])?;
        let vae_spec = tensor_spec(vec![1, 4, u64::from(image.height), u64::from(image.width)])?;
        let mut reservations = host
            .reserve_many(&[
                (HostMemoryClass::InputPreprocess, patch_bytes),
                (HostMemoryClass::InputPreprocess, vae_bytes),
            ])?
            .into_iter();
        let mut patches = reservations.next().unwrap().allocate_zeroed()?;
        let mut vae = reservations.next().unwrap().allocate_zeroed()?;
        let rgba = image.bytes.bytes();
        // Vision rows: block_y, block_x, merge_y, merge_x. Columns:
        // channel, repeated temporal frame, patch_y, patch_x.
        for row in 0..rows {
            let block = row / (self.merge * self.merge);
            let within = row % (self.merge * self.merge);
            let py = (block / (gw / self.merge)) * self.merge + within / self.merge;
            let px = (block % (gw / self.merge)) * self.merge + within % self.merge;
            for c in 0..3 {
                for t in 0..self.temporal {
                    for y in 0..self.patch {
                        for x in 0..self.patch {
                            let pixel = (py * self.patch + y) * width + px * self.patch + x;
                            let white = composite_white(rgba[pixel * 4 + c], rgba[pixel * 4 + 3]);
                            let normalized =
                                (f32::from(white) - self.fused_mean[c]) / self.fused_std[c];
                            let col = ((c * self.temporal + t) * self.patch + y) * self.patch + x;
                            write_bf16(patches.as_mut(), row * columns + col, normalized);
                        }
                    }
                }
            }
        }
        for c in 0..4 {
            for pixel in 0..pixels {
                // Diffusers PIL->numpy float32 /255, then 2*x-1. These
                // roundings differ from the vision backend's fused formula.
                let rescaled = f32::from(rgba[pixel * 4 + c]) / 255.0;
                let doubled = rescaled * 2.0;
                write_bf16(vae.as_mut(), c * pixels + pixel, doubled - 1.0);
            }
        }
        let patches =
            ManagedHostTensor::new(value_id(Self::PATCHES_ID)?, patch_spec, patches.freeze())
                .map_err(|e| ImageInputError::Tensor(e.to_string()))?;
        let vae = ManagedHostTensor::new(value_id(Self::VAE_INPUT_ID)?, vae_spec, vae.freeze())
            .map_err(|e| ImageInputError::Tensor(e.to_string()))?;
        Ok(QwenImage21ConditionInputs {
            patches,
            vae,
            grid: [1, gh as u64, gw as u64],
            width: image.width,
            height: image.height,
        })
    }
}

/// Cloning retains immutable, already admitted Host owners; it does not copy
/// the reference payload. Candidate batch preparation can borrow these owners
/// while a queue keeps them for capacity-driven batch reduction.
#[derive(Debug, Clone)]
pub struct QwenImage21ConditionInputs {
    patches: ManagedHostTensor,
    vae: ManagedHostTensor,
    grid: [u64; 3],
    width: u32,
    height: u32,
}

impl QwenImage21ConditionInputs {
    pub fn vision_patches(&self) -> &ManagedHostTensor {
        &self.patches
    }
    pub fn vae_rgba(&self) -> &ManagedHostTensor {
        &self.vae
    }
    pub fn grid_thw(&self) -> [u64; 3] {
        self.grid
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn into_tensors(self) -> (ManagedHostTensor, ManagedHostTensor) {
        (self.patches, self.vae)
    }
}

fn tensor_spec(dimensions: Vec<u64>) -> ImageInputResult<ResolvedTensorSpec> {
    ResolvedTensorSpec::new(
        dimensions,
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .map_err(|e| ImageInputError::Tensor(e.to_string()))
}
fn value_id(value: &str) -> ImageInputResult<ProgramValueId> {
    ProgramValueId::new(value).map_err(|e| ImageInputError::Tensor(e.to_string()))
}
fn byte_len(elements: Option<usize>) -> ImageInputResult<usize> {
    elements
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n > 0 && n <= isize::MAX as usize)
        .ok_or(ImageInputError::Limit(
            "model input exceeds addressable bytes",
        ))
}
fn write_bf16(output: &mut [u8], index: usize, value: f32) {
    output[index * 2..index * 2 + 2]
        .copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
}
fn composite_white(channel: u8, alpha: u8) -> u8 {
    // Pillow paste(RGB, mask=A), integer rounded divide by 255.
    let numerator = u32::from(channel) * u32::from(alpha) + 255 * (255 - u32::from(alpha));
    ((numerator + 127) / 255) as u8
}

#[cfg(test)]
#[path = "preprocess_tests.rs"]
mod tests;
