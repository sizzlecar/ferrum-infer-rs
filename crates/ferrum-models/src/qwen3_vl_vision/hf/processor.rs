use super::*;
use crate::qwen_image21::image_input::ImageInputLimits;
use sha2::{Digest, Sha256};

/// Metadata acceptance and actual image/patch extents. This is not a new Host
/// authority; every large payload is still admitted from the caller's account.
#[derive(Clone, Copy, Debug)]
pub struct Qwen3VLVisionHfProcessorLimits {
    pub max_processor_bytes: usize,
    pub image: ImageInputLimits,
    pub max_patch_rows: u64,
    pub max_patch_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct Qwen3VLVisionHfProcessor {
    pub(super) source_sha256: String,
    pub(super) patch: u64,
    pub(super) temporal: u64,
    pub(super) merge: u64,
    pub(super) minimum_pixels: u64,
    pub(super) maximum_pixels: u64,
    pub(super) limits: Qwen3VLVisionHfProcessorLimits,
}
impl Qwen3VLVisionHfProcessor {
    /// Consume the original `processor_config.json`, not the main H3 TE's
    /// processor. Unknown image algorithms fail before any payload allocation.
    pub fn from_source_bytes(bytes: &[u8], limits: Qwen3VLVisionHfProcessorLimits) -> Result<Self> {
        ensure!(
            !bytes.is_empty()
                && bytes.len() <= limits.max_processor_bytes
                && limits.max_processor_bytes <= isize::MAX as usize,
            "processor metadata exceeds declared capacity"
        );
        ensure!(
            limits.max_patch_rows > 0
                && limits.max_patch_bytes > 0
                && limits.max_patch_bytes <= isize::MAX as usize,
            "positive addressable patch limits required"
        );
        let raw: serde_json::Value = serde_json::from_slice(bytes)?;
        ensure!(
            raw.get("processor_class").and_then(|v| v.as_str()) == Some("Qwen3VLProcessor"),
            "unsupported multimodal processor declaration"
        );
        let p = raw
            .get("image_processor")
            .context("missing image_processor declaration")?;
        ensure!(
            p.get("image_processor_type").and_then(|v| v.as_str()) == Some("Qwen2VLImageProcessor"),
            "unsupported image processor declaration"
        );
        for name in ["do_convert_rgb", "do_normalize", "do_rescale", "do_resize"] {
            ensure!(
                p.get(name).and_then(|v| v.as_bool()) == Some(true),
                "unsupported image policy {name}"
            );
        }
        ensure!(
            p.get("resample").and_then(|v| v.as_u64()) == Some(3),
            "HF image resize must declare bicubic"
        );
        for name in ["image_mean", "image_std"] {
            ensure!(
                p.get(name)
                    .and_then(|v| v.as_array())
                    .is_some_and(|values| values.len() == 3
                        && values.iter().all(|v| v.as_f64() == Some(0.5))),
                "unsupported RGB normalization {name}"
            );
        }
        ensure!(
            p.get("rescale_factor").and_then(|v| v.as_f64()) == Some(1.0 / 255.0),
            "unsupported RGB rescale factor"
        );
        let u = |name: &str| {
            p.get(name)
                .and_then(|v| v.as_u64())
                .filter(|&n| n > 0)
                .with_context(|| format!("missing positive {name}"))
        };
        let (patch, temporal, merge) = (
            u("patch_size")?,
            u("temporal_patch_size")?,
            u("merge_size")?,
        );
        ensure!(
            temporal == 2 && patch.checked_mul(merge).is_some_and(|n| n <= 1 << 24),
            "unsupported temporal or spatial patch geometry"
        );
        let size = p.get("size").context("missing resize area policy")?;
        let minimum_pixels = size
            .get("shortest_edge")
            .and_then(|v| v.as_u64())
            .context("missing minimum pixels")?;
        let maximum_pixels = size
            .get("longest_edge")
            .and_then(|v| v.as_u64())
            .context("missing maximum pixels")?;
        ensure!(
            minimum_pixels > 0 && minimum_pixels <= maximum_pixels && maximum_pixels <= 1 << 53,
            "invalid exact resize area policy"
        );
        Ok(Self {
            source_sha256: format!("{:x}", Sha256::digest(bytes)),
            patch,
            temporal,
            merge,
            minimum_pixels,
            maximum_pixels,
            limits,
        })
    }
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }
    /// The same declared ingress/decode limits used by the prepared image plan.
    pub fn image_limits(&self) -> ImageInputLimits {
        self.limits.image
    }
    pub fn patch_size(&self) -> u64 {
        self.patch
    }
    pub fn temporal_patch_size(&self) -> u64 {
        self.temporal
    }
    pub fn merge_size(&self) -> u64 {
        self.merge
    }
    pub fn pixel_area_bounds(&self) -> (u64, u64) {
        (self.minimum_pixels, self.maximum_pixels)
    }
    pub(super) fn check_architecture(&self, a: &Qwen3VLVisionArchitecture) -> Result<()> {
        ensure!(
            a.channels() == 3
                && a.patch_size() == self.patch
                && a.temporal_patch_size() == self.temporal
                && a.merge_size() == self.merge
                && a.deepstack_layers().is_empty(),
            "processor differs from main-only visual architecture"
        );
        Ok(())
    }
}
