//! Image HTTP workload identity, bounded PNG validation and measured outcomes.
//!
//! This schema deliberately does not interpret PNG validity as model quality,
//! G3 readiness, a server batch-size observation, or a throughput target pass.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fmt};

mod png_response;
pub use png_response::{
    validate_image_response, validate_image_response_with_png, validate_reference_png,
    ValidatedImage, ValidatedPngResponse,
};
mod request_mode;
pub mod resource_usage;
pub use request_mode::{ImageReference, ImageRequestMode};
mod workload;
pub use workload::{image_workload_metrics, ImageWorkloadMetrics};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBenchError(pub String);

impl fmt::Display for ImageBenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ImageBenchError {}

pub(crate) fn invalid(message: &str) -> ImageBenchError {
    ImageBenchError(message.into())
}

/// Ordered, explicit request set. Repetitions and endpoints use this same order.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDataset {
    pub schema_version: u32,
    /// Version 1 retains its original generation JSON. Version 2 may select a
    /// single shared reference image for every ordered edit request.
    #[serde(default, skip_serializing_if = "ImageRequestMode::is_generate")]
    pub mode: ImageRequestMode,
    pub samples: Vec<ImageSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageSample {
    pub id: String,
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    /// Omitted means the server's declared fixed configuration supplies steps.
    /// A server rejecting this optional field is a recorded protocol failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u32>,
    /// Version 3 overrides the dataset default per request, allowing a pinned
    /// generation/edit mixture and multiple immutable reference PNGs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<ImageRequestMode>,
}

impl ImageSample {
    pub fn effective_mode<'a>(&'a self, default: &'a ImageRequestMode) -> &'a ImageRequestMode {
        self.mode.as_ref().unwrap_or(default)
    }
}

impl ImageDataset {
    pub fn validate(&self, limits: &ImageResponseLimits) -> Result<(), ImageBenchError> {
        limits.validate()?;
        if !matches!(self.schema_version, 1 | 2 | 3)
            || (self.schema_version == 1 && !self.mode.is_generate())
            || self.samples.is_empty()
            || self.samples.len() > 100_000
        {
            return Err(invalid(
                "dataset requires version 1 (generate), 2 (one mode), or 3 (per-request mode) and 1..=100000 samples",
            ));
        }
        self.mode.validate()?;
        let mut ids = BTreeSet::new();
        for sample in &self.samples {
            if self.schema_version < 3 && sample.mode.is_some() {
                return Err(invalid("per-request mode requires dataset version 3"));
            }
            sample.effective_mode(&self.mode).validate()?;
            if sample.id.is_empty() || sample.id.len() > 256 || !ids.insert(&sample.id) {
                return Err(invalid(
                    "sample IDs must be nonempty, unique and at most 256 bytes",
                ));
            }
            if sample.prompt.len() > 1024 * 1024 || sample.num_inference_steps == Some(0) {
                return Err(invalid(
                    "prompt exceeds 1MiB or inference step count is zero",
                ));
            }
            limits.rgba_bytes(sample.width, sample.height)?;
        }
        Ok(())
    }

    /// Stable identity includes ordered prompts, seeds, dimensions and steps.
    pub fn fingerprint(&self) -> Result<String, ImageBenchError> {
        let bytes = serde_json::to_vec(self).map_err(|_| invalid("cannot serialize dataset"))?;
        Ok(sha256(&bytes))
    }
}

/// Explicit per-response bounds, separate from socket/parser/allocator overhead.
/// At most `concurrency` responses are receiving or decoding at once.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageResponseLimits {
    pub maximum_response_bytes: usize,
    pub maximum_png_bytes: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub maximum_pixels: u64,
    pub decoder_workspace_bytes: usize,
}

impl Default for ImageResponseLimits {
    fn default() -> Self {
        Self {
            maximum_response_bytes: 64 * 1024 * 1024,
            maximum_png_bytes: 48 * 1024 * 1024,
            maximum_width: 8192,
            maximum_height: 8192,
            maximum_pixels: 16 * 1024 * 1024,
            decoder_workspace_bytes: 64 * 1024 * 1024,
        }
    }
}

impl ImageResponseLimits {
    pub fn validate(&self) -> Result<(), ImageBenchError> {
        if self.maximum_width == 0 || self.maximum_height == 0 || self.maximum_pixels == 0 {
            return Err(invalid("PNG dimension/pixel limits must be positive"));
        }
        for size in [
            self.maximum_response_bytes,
            self.maximum_png_bytes,
            self.decoder_workspace_bytes,
        ] {
            if size == 0 || size > isize::MAX as usize {
                return Err(invalid(
                    "response and decoder byte limits must fit the host address space",
                ));
            }
        }
        if self.maximum_pixels > (isize::MAX as u64) / 4 {
            return Err(invalid("RGBA pixel limit exceeds host address space"));
        }
        Ok(())
    }

    pub fn rgba_bytes(&self, width: u32, height: u32) -> Result<usize, ImageBenchError> {
        let pixels = u64::from(width) * u64::from(height);
        if width == 0
            || height == 0
            || width > self.maximum_width
            || height > self.maximum_height
            || pixels > self.maximum_pixels
        {
            return Err(invalid("image dimensions exceed configured limits"));
        }
        pixels
            .checked_mul(4)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("RGBA extent exceeds host address space"))
    }
}

/// OpenAI image request, always one b64_json image; no server capacity controls.
#[derive(Debug, Serialize)]
pub struct ImageHttpRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub seed: u64,
    pub size: String,
    pub n: u32,
    pub response_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_inference_steps: Option<u32>,
}

impl<'a> ImageHttpRequest<'a> {
    pub fn new(model: &'a str, sample: &'a ImageSample) -> Self {
        Self {
            model,
            prompt: &sample.prompt,
            seed: sample.seed,
            size: format!("{}x{}", sample.width, sample.height),
            n: 1,
            response_format: "b64_json",
            num_inference_steps: sample.num_inference_steps,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageRequestPhase {
    Warmup,
    Measured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFailureKind {
    Transport,
    Timeout,
    HttpStatus,
    ResponseLimit,
    InvalidResponse,
    ClientTask,
    /// A scheduled arrival exceeded the client's declared in-flight bound.
    /// No HTTP request was sent; this is never a server rejection.
    ClientCapacity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRequestRecord {
    pub sample_index: usize,
    pub sample_id: String,
    pub phase: ImageRequestPhase,
    pub repetition: usize,
    pub dispatch_index: usize,
    pub http_status: Option<u16>,
    /// Send start through complete body reception. Excludes local PNG validation.
    pub e2e_ms: f64,
    pub validation_ms: f64,
    pub response_bytes: usize,
    pub image: Option<ValidatedImage>,
    /// Stable local error category; never includes response/base64 contents.
    pub failure: Option<ImageFailureKind>,
    pub error: Option<String>,
}

impl ImageRequestRecord {
    pub fn succeeded(&self) -> bool {
        self.image.is_some() && self.failure.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSloStatus {
    NotEvaluated,
    Met,
    Violated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetrics {
    /// All planned arrivals, including those the bounded client could not send.
    pub attempted: usize,
    #[serde(default)]
    pub dispatched: usize,
    pub succeeded: usize,
    /// All unsuccessful planned requests, preserving the original schema's
    /// meaning. Includes server rejections and client-capacity drops.
    pub errors: usize,
    /// HTTP 429 failures; distinct from client-side unsent arrivals.
    #[serde(default)]
    pub rejected: usize,
    #[serde(default)]
    pub client_not_dispatched: usize,
    /// Failures other than HTTP 429 or client-capacity drops.
    #[serde(default)]
    pub other_errors: usize,
    /// Includes all request failures and local decoding work; excludes warmup.
    pub elapsed_seconds: f64,
    pub images_per_second: Option<f64>,
    /// Percentiles of successful, strictly validated responses only.
    pub e2e_p50_ms: Option<f64>,
    #[serde(default)]
    pub e2e_p95_ms: Option<f64>,
    pub e2e_p99_ms: Option<f64>,
    pub latency_slo_p99_ms: Option<f64>,
    /// An explicit threshold is met only with zero errors in the supplied records.
    /// This aggregator has no expected-count authority; the caller must establish
    /// that its record collection covers the complete intended request set.
    pub latency_slo_status: ImageSloStatus,
}

impl ImageMetrics {
    pub fn from_records<'a>(
        elapsed_seconds: f64,
        records: impl IntoIterator<Item = &'a ImageRequestRecord>,
        latency_slo_p99_ms: Option<f64>,
    ) -> Self {
        let records: Vec<_> = records.into_iter().collect();
        let latencies: Vec<_> = records
            .iter()
            .filter(|r| r.succeeded())
            .map(|r| r.e2e_ms)
            .collect();
        let succeeded = latencies.len();
        let errors = records.len() - succeeded;
        let rejected = records
            .iter()
            .filter(|r| !r.succeeded() && r.http_status == Some(429))
            .count();
        let client_not_dispatched = records
            .iter()
            .filter(|r| r.failure == Some(ImageFailureKind::ClientCapacity))
            .count();
        let p50 = (!latencies.is_empty()).then(|| crate::percentile(&latencies, 0.50));
        let p95 = (!latencies.is_empty()).then(|| crate::percentile(&latencies, 0.95));
        let p99 = (!latencies.is_empty()).then(|| crate::percentile(&latencies, 0.99));
        let status = match latency_slo_p99_ms {
            None => ImageSloStatus::NotEvaluated,
            Some(limit) if errors == 0 && p99.is_some_and(|p| p <= limit) => ImageSloStatus::Met,
            Some(_) => ImageSloStatus::Violated,
        };
        Self {
            attempted: records.len(),
            dispatched: records.len() - client_not_dispatched,
            succeeded,
            errors,
            rejected,
            client_not_dispatched,
            other_errors: errors.saturating_sub(rejected + client_not_dispatched),
            elapsed_seconds,
            images_per_second: (elapsed_seconds.is_finite() && elapsed_seconds > 0.0)
                .then_some(succeeded as f64 / elapsed_seconds),
            e2e_p50_ms: p50,
            e2e_p95_ms: p95,
            e2e_p99_ms: p99,
            latency_slo_p99_ms,
            latency_slo_status: status,
        }
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
