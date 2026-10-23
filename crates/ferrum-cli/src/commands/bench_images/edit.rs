//! One bounded reference allocation shared by all endpoints and repetitions.
//! Source IO/PNG validation precedes measurement. Multipart sends clone only
//! the `Bytes` owner; neither concurrency nor repetitions duplicate its payload.

use super::{ImageBenchmarkConfig, ImageBenchmarkEndpoint};
use anyhow::{ensure, Context};
use bytes::Bytes;
use ferrum_bench_core::images::{
    validate_reference_png, ImageRequestMode, ImageSample, ValidatedImage,
};
use std::{fs::File, io::Read};

pub(super) struct PreparedReference {
    pub(super) identity: ValidatedImage,
    png: Bytes,
}

impl PreparedReference {
    pub(super) fn load(
        mode: &ImageRequestMode,
        config: &ImageBenchmarkConfig,
    ) -> anyhow::Result<Option<Self>> {
        let ImageRequestMode::Edit { image } = mode else {
            return Ok(None);
        };
        let file = File::open(&image.path).context("open reference PNG")?;
        let metadata = file.metadata().context("inspect reference PNG")?;
        ensure!(metadata.is_file(), "reference image must be a regular file");
        ensure!(
            metadata.len() <= config.maximum_reference_image_bytes as u64,
            "reference image exceeds byte limit"
        );
        let mut png = Vec::new();
        file.take(config.maximum_reference_image_bytes as u64 + 1)
            .read_to_end(&mut png)
            .context("read bounded reference PNG")?;
        ensure!(
            png.len() <= config.maximum_reference_image_bytes,
            "reference image exceeds byte limit"
        );
        let mut limits = config.limits.clone();
        limits.maximum_png_bytes = config.maximum_reference_image_bytes;
        let identity = validate_reference_png(&png, &limits)?;
        if let Some(expected) = &image.encoded_sha256 {
            ensure!(
                expected == &identity.png_sha256,
                "reference PNG SHA-256 differs from the dataset declaration"
            );
        }
        Ok(Some(Self {
            identity,
            png: Bytes::from(png),
        }))
    }

    pub(super) fn form(
        &self,
        endpoint: &ImageBenchmarkEndpoint,
        sample: &ImageSample,
    ) -> Result<reqwest::multipart::Form, reqwest::Error> {
        let part = reqwest::multipart::Part::stream_with_length(
            reqwest::Body::from(self.png.clone()),
            self.png.len() as u64,
        )
        .file_name("reference.png")
        .mime_str("image/png")?;
        let mut form = reqwest::multipart::Form::new()
            .text("model", endpoint.model.clone())
            .text("prompt", sample.prompt.clone())
            .text("seed", sample.seed.to_string())
            .text("size", format!("{}x{}", sample.width, sample.height))
            .text("n", "1")
            .text("response_format", "b64_json")
            .part("image", part);
        // Preserve explicit dataset intent. A server rejecting this optional
        // field yields a real HTTP failure; omission uses its declared preset.
        if let Some(steps) = sample.num_inference_steps {
            form = form.text("num_inference_steps", steps.to_string());
        }
        Ok(form)
    }
}

#[cfg(test)]
mod tests;
