//! Optional artifact capture stays inside the existing active-request bound.
//! No payload queue or second PNG allocation is introduced. The directory is
//! caller-controlled; all descendant names use only client numeric ordinals.

use super::*;
use ferrum_bench_core::images::{ValidatedImage, ValidatedPngResponse};
use std::{
    num::NonZeroU64,
    panic::{catch_unwind, UnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Debug, Clone, Serialize)]
pub struct PngCaptureConfig {
    pub directory: PathBuf,
    /// Encoded PNG payload bytes, not filesystem allocation or total RSS.
    pub maximum_capture_bytes: NonZeroU64,
}

impl PngCaptureConfig {
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.directory.as_os_str().is_empty() && self.directory.file_name().is_some(),
            "capture directory must name a new directory"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PngCaptureFailureKind {
    Capacity,
    Io,
    ClientTask,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PngCaptureResult {
    Saved {
        relative_path: String,
        png_bytes: usize,
        png_sha256: String,
        /// Byte reservation, create-new, write and sync; excludes PNG validation.
        capture_ms: f64,
    },
    Failed {
        kind: PngCaptureFailureKind,
        /// A failed write may leave a partial file at this numeric path. It is
        /// never reported as Saved and is never overwritten on another attempt.
        relative_path: Option<String>,
        error: String,
        /// None when a worker failed without returning its timing.
        capture_ms: Option<f64>,
    },
    /// No validated PNG was available because the request/validation failed.
    NotProduced,
}

#[derive(Debug, Clone, Serialize)]
pub struct PngCaptureRecord {
    pub sample_index: usize,
    pub sample_id: String,
    pub phase: ImageRequestPhase,
    pub repetition: usize,
    pub result: PngCaptureResult,
}

#[derive(Debug, Serialize)]
pub struct PngCaptureSummary {
    pub requested: usize,
    pub saved: usize,
    pub failed: usize,
    pub not_produced: usize,
    pub complete: bool,
    /// Conservative reservations include failed/partial writes; not measured
    /// disk usage. Successful files are not refunded during the benchmark.
    pub reserved_png_bytes: u64,
    pub timing_scope: &'static str,
}

pub(super) struct CaptureSink {
    root: PathBuf,
    maximum_bytes: u64,
    reserved: AtomicU64,
}

/// Only the save operation is inside this unwind boundary. Validation already
/// succeeded, so a save panic cannot erase that result or become a server error.
/// A partially written file and its byte reservation may remain after a panic.
pub(super) fn capture_validated_png(
    response: ValidatedPngResponse,
    save: impl FnOnce(&ValidatedImage, &[u8]) -> PngCaptureResult + UnwindSafe,
) -> (ValidatedImage, PngCaptureResult) {
    let (image, png) = response.into_parts();
    let start = Instant::now();
    let captured =
        catch_unwind(|| save(&image, &png)).unwrap_or_else(|_| PngCaptureResult::Failed {
            kind: PngCaptureFailureKind::ClientTask,
            relative_path: None,
            error: "PNG capture panicked; partial write outcome unknown".into(),
            capture_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
        });
    (image, captured)
}

impl CaptureSink {
    pub(super) fn create(config: &PngCaptureConfig) -> anyhow::Result<Arc<Self>> {
        config.validate()?;
        // No recursive creation: the user supplies an existing parent and a
        // wholly new capture directory. An existing directory is also rejected.
        std::fs::create_dir(&config.directory).context("reserve new PNG capture directory")?;
        let root = std::fs::canonicalize(&config.directory)
            .context("resolve newly created PNG capture directory")?;
        Ok(Arc::new(Self {
            root,
            maximum_bytes: config.maximum_capture_bytes.get(),
            reserved: AtomicU64::new(0),
        }))
    }

    pub(super) fn save(
        &self,
        endpoint_index: usize,
        phase: ImageRequestPhase,
        repetition: usize,
        sample_index: usize,
        image: &ValidatedImage,
        png: &[u8],
    ) -> PngCaptureResult {
        let start = Instant::now();
        let phase = match phase {
            ImageRequestPhase::Warmup => "warmup",
            ImageRequestPhase::Measured => "measured",
        };
        let name = format!("e{endpoint_index:04}-{phase}-r{repetition:06}-s{sample_index:06}.png");
        let extent = u64::try_from(png.len());
        let admitted = extent.is_ok_and(|bytes| {
            self.reserved
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                    old.checked_add(bytes)
                        .filter(|sum| *sum <= self.maximum_bytes)
                })
                .is_ok()
        });
        if !admitted {
            return PngCaptureResult::Failed {
                kind: PngCaptureFailureKind::Capacity,
                relative_path: None,
                error: "encoded PNG capture byte limit exceeded".into(),
                capture_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
            };
        }
        let saved = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.root.join(&name))?;
            file.write_all(png)?;
            file.sync_all()
        })();
        let capture_ms = start.elapsed().as_secs_f64() * 1000.0;
        match saved {
            Ok(()) => PngCaptureResult::Saved {
                relative_path: name,
                png_bytes: png.len(),
                png_sha256: image.png_sha256.clone(),
                capture_ms,
            },
            Err(error) => PngCaptureResult::Failed {
                kind: PngCaptureFailureKind::Io,
                relative_path: Some(name),
                // ErrorKind avoids leaking any response body or prompt.
                error: format!("PNG create/write/sync failed: {:?}", error.kind()),
                capture_ms: Some(capture_ms),
            },
        }
    }

    pub(super) fn summary(&self, endpoints: &[ImageEndpointReport]) -> PngCaptureSummary {
        let mut summary = PngCaptureSummary {
            requested: endpoints.iter().map(|e| e.requests.len()).sum(),
            saved: 0,
            failed: 0,
            not_produced: 0,
            complete: false,
            reserved_png_bytes: self.reserved.load(Ordering::Relaxed),
            timing_scope: "per-request E2E ends at full HTTP body receipt, before validation/capture; with capture enabled validation_ms includes blocking-worker queue time and validation through decode completion, excluding save; a failed validation worker uses elapsed time through join instead; capture_ms includes local create/write/sync (or elapsed save time until a caught panic); request slots remain occupied through capture, so round wall time, later dispatch and even HTTP completion span can be affected by capture IO; capture-on throughput is not capture-off throughput",
        };
        for record in endpoints
            .iter()
            .filter_map(|e| e.captures.as_ref())
            .flatten()
        {
            match &record.result {
                PngCaptureResult::Saved { .. } => summary.saved += 1,
                PngCaptureResult::Failed { .. } => summary.failed += 1,
                PngCaptureResult::NotProduced => summary.not_produced += 1,
            }
        }
        summary.complete = summary.requested > 0 && summary.saved == summary.requested;
        summary
    }
}

#[cfg(test)]
mod tests;
