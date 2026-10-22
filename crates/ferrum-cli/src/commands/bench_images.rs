//! Bounded, ordered image HTTP benchmarking. Does not change server capacity.
//! The reusable runner is independent of model loading and needs no GPU feature.

use anyhow::{ensure, Context};
use clap::{Args, ValueEnum};
use ferrum_bench_core::images::{
    sha256, ImageDataset, ImageMetrics, ImageRequestPhase, ImageRequestRecord, ImageResponseLimits,
    ImageSloStatus,
};
use serde::Serialize;
use std::{
    fs::OpenOptions,
    io::{BufWriter, Read, Write},
    num::NonZeroU64,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

mod capture;
pub use capture::{
    PngCaptureConfig, PngCaptureFailureKind, PngCaptureRecord, PngCaptureResult, PngCaptureSummary,
};
mod runner;
pub use runner::run_benchmark;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialServerState {
    Unknown,
    ColdDeclared,
    WarmDeclared,
}

#[derive(Debug, Clone, Args)]
pub struct BenchImagesCommand {
    /// Full /v1/images/generations URL. Repeat for sequential endpoint comparison.
    #[arg(long, required = true)]
    pub endpoint: Vec<String>,
    /// One model alias shared by endpoints, or one per endpoint in the same order.
    #[arg(long, required = true)]
    pub model: Vec<String>,
    /// Fixed server command/version/precision/hardware declaration, one per endpoint.
    /// This is operator evidence, not a server configuration action or verification.
    #[arg(long, required = true)]
    pub server_config: Vec<String>,
    /// JSON {"schema_version":1,"samples":[{"id":"...","prompt":"...",
    /// "seed":1,"width":512,"height":512}]} in fixed request order.
    #[arg(long)]
    pub dataset: PathBuf,
    /// New JSON report path; an existing file is never overwritten.
    #[arg(long)]
    pub output: PathBuf,
    /// Save every validated PNG (including warmup) under a wholly new directory.
    /// Capture IO is excluded from request E2E but affects paced wall throughput.
    #[arg(long, requires = "maximum_capture_bytes")]
    pub capture_dir: Option<PathBuf>,
    /// Aggregate encoded PNG byte limit; required with --capture-dir. Capture
    /// failures are reported independently and make the final command fail.
    #[arg(long, requires = "capture_dir")]
    pub maximum_capture_bytes: Option<NonZeroU64>,
    #[arg(long, default_value_t = 8)]
    pub concurrency: usize,
    /// Each repetition submits the complete dataset in its original order.
    #[arg(long, default_value_t = 3)]
    pub repetitions: usize,
    /// Complete dataset rounds, recorded separately from measured repetitions.
    #[arg(long, default_value_t = 0)]
    pub warmup_repetitions: usize,
    /// Declared state before warmup/measurement; the client never flushes/restarts servers.
    #[arg(long, value_enum, default_value = "unknown")]
    pub initial_server_state: InitialServerState,
    #[arg(long, default_value_t = 1800)]
    pub timeout_seconds: u64,
    /// Optional predeclared successful-response E2E P99 threshold. Any error violates it.
    #[arg(long)]
    pub latency_slo_p99_ms: Option<f64>,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub maximum_response_bytes: usize,
    #[arg(long, default_value_t = 48 * 1024 * 1024)]
    pub maximum_png_bytes: usize,
    #[arg(long, default_value_t = 8192)]
    pub maximum_dimension: u32,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    pub maximum_pixels: u64,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub decoder_workspace_bytes: usize,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    pub maximum_dataset_bytes: usize,
    #[arg(long, default_value_t = 100_000)]
    pub maximum_total_requests: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageBenchmarkEndpoint {
    pub url: String,
    pub model: String,
    pub server_config_declaration: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageBenchmarkConfig {
    pub endpoints: Vec<ImageBenchmarkEndpoint>,
    pub concurrency: usize,
    pub repetitions: usize,
    pub warmup_repetitions: usize,
    pub initial_server_state: InitialServerState,
    pub timeout_seconds: u64,
    pub latency_slo_p99_ms: Option<f64>,
    pub limits: ImageResponseLimits,
    pub maximum_total_requests: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<PngCaptureConfig>,
}

impl ImageBenchmarkConfig {
    pub fn validate(&self, dataset: &ImageDataset) -> anyhow::Result<()> {
        dataset.validate(&self.limits)?;
        if let Some(capture) = &self.capture {
            capture.validate()?;
        }
        ensure!(
            !self.endpoints.is_empty() && self.endpoints.len() <= 16,
            "need 1..=16 endpoints"
        );
        ensure!(
            (1..=1024).contains(&self.concurrency),
            "concurrency must be within 1..=1024"
        );
        ensure!(self.repetitions > 0, "repetitions must be positive");
        ensure!(
            (1..=86_400).contains(&self.timeout_seconds),
            "timeout must be within 1..=86400 seconds"
        );
        ensure!(
            self.maximum_total_requests > 0 && self.maximum_total_requests <= 1_000_000,
            "maximum total requests must be within 1..=1000000"
        );
        let total = self
            .repetitions
            .checked_add(self.warmup_repetitions)
            .and_then(|n| n.checked_mul(dataset.samples.len()))
            .and_then(|n| n.checked_mul(self.endpoints.len()));
        ensure!(
            total.is_some_and(|n| n <= self.maximum_total_requests),
            "total request count exceeds limit"
        );
        if let Some(limit) = self.latency_slo_p99_ms {
            ensure!(
                limit.is_finite() && limit > 0.0,
                "latency SLO must be finite and positive"
            );
        }
        for endpoint in &self.endpoints {
            let url = reqwest::Url::parse(&endpoint.url).context("invalid endpoint URL")?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none(),
                "endpoint requires HTTP(S), host and no credentials/query/fragment"
            );
            ensure!(
                !endpoint.model.is_empty() && endpoint.model.len() <= 512,
                "model alias must contain 1..=512 bytes"
            );
            ensure!(
                !endpoint.server_config_declaration.is_empty()
                    && endpoint.server_config_declaration.len() <= 65536,
                "server configuration declaration must contain 1..=65536 bytes"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ImageBenchmarkRepetition {
    pub phase: ImageRequestPhase,
    pub repetition: usize,
    pub metrics: ImageMetrics,
    /// First send to final body receipt; overlaps local validation, excludes its tail.
    /// Not an isolated network-only/server-only throughput measurement.
    pub http_completion_span_seconds: f64,
    pub successful_images_per_http_completion_second: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ImageEndpointReport {
    pub endpoint_index: usize,
    pub repetitions: Vec<ImageBenchmarkRepetition>,
    pub measured: ImageMetrics,
    pub requests: Vec<ImageRequestRecord>,
    /// Matched to requests by sample_index, phase and repetition, never by
    /// response completion order. This endpoint supplies the endpoint index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captures: Option<Vec<PngCaptureRecord>>,
}

#[derive(Debug, Serialize)]
pub struct ImageBenchmarkReport {
    pub schema_version: u32,
    pub client_version: String,
    pub started_unix_ms: u128,
    pub configuration: ImageBenchmarkConfig,
    pub dataset_sha256: String,
    pub dataset_file_sha256: Option<String>,
    pub dataset: ImageDataset,
    pub endpoints: Vec<ImageEndpointReport>,
    pub evidence: ImageEvidenceScope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<PngCaptureSummary>,
}

#[derive(Debug, Serialize)]
pub struct ImageEvidenceScope {
    pub server_capacity_changed_by_client: bool,
    pub server_configuration_verified: bool,
    pub dispatch_order: &'static str,
    pub latency_scope: &'static str,
    pub throughput_scope: &'static str,
    pub pixel_validation_scope: &'static str,
    pub quality_status: &'static str,
    pub g3_status: &'static str,
    pub throughput_target_status: &'static str,
    pub peak_device_memory_bytes: Option<u64>,
    pub peak_host_memory_bytes: Option<u64>,
}

impl Default for ImageEvidenceScope {
    fn default() -> Self {
        Self {
            server_capacity_changed_by_client: false, server_configuration_verified: false,
            dispatch_order: "dataset order at client admission; network/server arrival order is not guaranteed",
            latency_scope: "successful validated responses: send start to full HTTP body reception, excludes local PNG validation",
            throughput_scope: "successful validated images divided by measured wall time including errors and local validation; warmup excluded",
            pixel_validation_scope: "PNG CRC/Adler, full decode and exact dimensions; RGBA8 integer values are finite, pre-encoding floats and prompt quality are not assessed",
            quality_status: "not_evaluated", g3_status: "not_evaluated",
            throughput_target_status: "not_evaluated", peak_device_memory_bytes: None, peak_host_memory_bytes: None,
        }
    }
}

pub async fn execute(command: BenchImagesCommand) -> ferrum_types::Result<()> {
    execute_inner(command)
        .await
        .map_err(|e| ferrum_types::FerrumError::model(format!("bench-images: {e:#}")))
}

async fn execute_inner(command: BenchImagesCommand) -> anyhow::Result<()> {
    ensure!(
        command.maximum_dataset_bytes > 0 && command.maximum_dataset_bytes < isize::MAX as usize,
        "dataset byte limit must fit the host address space"
    );
    let mut bytes = Vec::new();
    std::fs::File::open(&command.dataset)
        .context("open image dataset")?
        .take(command.maximum_dataset_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= command.maximum_dataset_bytes,
        "dataset exceeds byte limit"
    );
    let dataset: ImageDataset =
        serde_json::from_slice(&bytes).context("invalid image dataset JSON")?;
    ensure!(
        command.model.len() == 1 || command.model.len() == command.endpoint.len(),
        "provide one model or one model per endpoint"
    );
    ensure!(
        command.server_config.len() == command.endpoint.len(),
        "provide one server-config declaration per endpoint"
    );
    let endpoints = command
        .endpoint
        .iter()
        .enumerate()
        .map(|(index, url)| ImageBenchmarkEndpoint {
            url: url.clone(),
            model: command.model[if command.model.len() == 1 { 0 } else { index }].clone(),
            server_config_declaration: command.server_config[index].clone(),
        })
        .collect();
    let capture = match (command.capture_dir, command.maximum_capture_bytes) {
        (None, None) => None,
        (Some(directory), Some(maximum_capture_bytes)) => Some(PngCaptureConfig {
            directory,
            maximum_capture_bytes,
        }),
        _ => anyhow::bail!("--capture-dir and --maximum-capture-bytes must be supplied together"),
    };
    let config = ImageBenchmarkConfig {
        endpoints,
        concurrency: command.concurrency,
        repetitions: command.repetitions,
        warmup_repetitions: command.warmup_repetitions,
        initial_server_state: command.initial_server_state,
        timeout_seconds: command.timeout_seconds,
        latency_slo_p99_ms: command.latency_slo_p99_ms,
        maximum_total_requests: command.maximum_total_requests,
        capture,
        limits: ImageResponseLimits {
            maximum_response_bytes: command.maximum_response_bytes,
            maximum_png_bytes: command.maximum_png_bytes,
            maximum_width: command.maximum_dimension,
            maximum_height: command.maximum_dimension,
            maximum_pixels: command.maximum_pixels,
            decoder_workspace_bytes: command.decoder_workspace_bytes,
        },
    };
    config.validate(&dataset)?;
    // Claim the destination before issuing requests. Cancellation does not save
    // partial records and may leave an empty/incomplete artifact, never silently
    // overwrite an existing report. Only a completed run is a complete report.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&command.output)
        .context("reserve new report path")?;
    let mut report = run_benchmark(config, dataset).await?;
    report.dataset_file_sha256 = Some(sha256(&bytes));
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let errors: usize = report
        .endpoints
        .iter()
        .flat_map(|e| &e.requests)
        .filter(|r| !r.succeeded())
        .count();
    let slo_violated = report.endpoints.iter().any(|e| {
        e.measured.latency_slo_status == ImageSloStatus::Violated
            || e.repetitions.iter().any(|r| {
                r.phase == ImageRequestPhase::Measured
                    && r.metrics.latency_slo_status == ImageSloStatus::Violated
            })
    });
    println!(
        "Image benchmark report: {} ({} endpoint(s), {} request errors; quality/G3 not evaluated)",
        command.output.display(),
        report.endpoints.len(),
        errors
    );
    if let Some(capture) = &report.capture {
        println!(
            "PNG capture: {} saved / {} requested, {} write/capacity/task failures, {} without a validated response; local IO affects paced throughput",
            capture.saved, capture.requested, capture.failed, capture.not_produced
        );
    }
    ensure!(
        errors == 0,
        "request errors recorded in {}",
        command.output.display()
    );
    ensure!(
        !slo_violated,
        "declared latency SLO violated; report saved to {}",
        command.output.display()
    );
    ensure!(
        report
            .capture
            .as_ref()
            .is_none_or(|capture| capture.complete),
        "PNG capture incomplete; report saved to {}",
        command.output.display()
    );
    Ok(())
}

fn started_unix_ms() -> anyhow::Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("clock precedes UNIX epoch")?
        .as_millis())
}
