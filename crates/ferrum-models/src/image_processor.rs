//! Image preprocessing for CLIP models.
//!
//! Load → resize → normalize → [1, 3, H, W] tensor.

use candle_core::{DType, Device, Tensor};
use ferrum_types::{FerrumError, Result};

/// CLIP image processor with configurable size and normalization.
pub struct ClipImageProcessor {
    image_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ClipImageProcessor {
    /// Standard CLIP normalization (ImageNet stats).
    pub fn new(image_size: usize) -> Self {
        Self {
            image_size,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        }
    }

    /// Load image from file path → preprocessed tensor.
    pub fn process_path(&self, path: &str, device: &Device) -> Result<Tensor> {
        let img =
            image::open(path).map_err(|e| FerrumError::model(format!("image load {path}: {e}")))?;
        self.process_image(img, device)
    }

    /// Decode base64 image data → preprocessed tensor.
    pub fn process_base64(&self, data: &str, device: &Device) -> Result<Tensor> {
        use base64::Engine;
        // Strip optional data URI prefix
        let raw = if let Some(pos) = data.find(",") {
            &data[pos + 1..]
        } else {
            data
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(raw)
            .map_err(|e| FerrumError::model(format!("base64 decode: {e}")))?;
        let img = image::load_from_memory(&bytes)
            .map_err(|e| FerrumError::model(format!("image decode: {e}")))?;
        self.process_image(img, device)
    }

    /// Core pipeline: DynamicImage → resize → normalize → [1, 3, H, W] tensor.
    fn process_image(&self, img: image::DynamicImage, device: &Device) -> Result<Tensor> {
        let size = self.image_size as u32;
        let img = img.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();

        let (w, h) = (img.width() as usize, img.height() as usize);
        let raw: Vec<f32> = img
            .into_raw()
            .into_iter()
            .map(|p| p as f32 / 255.0)
            .collect();

        // [H, W, 3] → [3, H, W] with normalization
        let mut chw = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for i in 0..h * w {
                chw[c * h * w + i] = (raw[i * 3 + c] - self.mean[c]) / self.std[c];
            }
        }

        Tensor::from_vec(chw, (1, 3, h, w), device)
            .map_err(|e| FerrumError::model(format!("tensor: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| FerrumError::model(format!("dtype: {e}")))
    }
}
