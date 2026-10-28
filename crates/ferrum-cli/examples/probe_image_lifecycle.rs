//! Local HTTP lifecycle observations; never starts, stops or signals a server.
use anyhow::{ensure, Context};
use clap::{Parser, ValueEnum};
use ferrum_bench_core::images::{ImageDataset, ImageResponseLimits};
use serde::Serialize;
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[path = "probe_image_lifecycle/health.rs"]
mod health;
#[path = "probe_image_lifecycle/request.rs"]
mod request;
#[path = "probe_image_lifecycle/run.rs"]
mod run;
#[cfg(test)]
#[path = "probe_image_lifecycle/tests.rs"]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Cancel,
    HoldBody,
    /// Obtain headers, hold the unread body, then drop the HTTP response.
    DropBody,
    ObserveShutdown,
}

#[derive(Debug, Clone, Parser, Serialize)]
#[command(
    about = "Observe image cancellation, held HTTP bodies, or externally controlled shutdown on loopback"
)]
struct Arguments {
    #[arg(long)]
    endpoint: String,
    #[arg(long)]
    edit_endpoint: Option<String>,
    #[arg(long)]
    health_endpoint: String,
    #[arg(long)]
    model: String,
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    /// Fixed prefix of dataset samples, concurrently dispatched without retry.
    #[arg(long, default_value_t = 2)]
    requests: usize,
    #[arg(long, default_value_t = 100)]
    cancel_after_ms: u64,
    #[arg(long, default_value_t = 5000)]
    hold_body_ms: u64,
    /// Health observation continues at least this long and through all requests.
    #[arg(long, default_value_t = 30_000)]
    observation_ms: u64,
    #[arg(long, default_value_t = 200)]
    poll_ms: u64,
    #[arg(long, default_value_t = 180_000)]
    request_timeout_ms: u64,
    #[arg(long, default_value_t = 2000)]
    health_timeout_ms: u64,
    #[arg(long, default_value_t = 1)]
    recovery_requests: usize,
    /// External operators may restart during this delay; the probe never does.
    #[arg(long, default_value_t = 0)]
    recovery_delay_ms: u64,
    #[arg(long, default_value_t = 4096)]
    maximum_health_snapshots: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    maximum_response_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024)]
    maximum_health_bytes: usize,
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    maximum_dataset_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    maximum_reference_bytes: usize,
}

impl Arguments {
    fn validate(&self, dataset: &ImageDataset) -> anyhow::Result<()> {
        dataset.validate(&self.limits())?;
        ensure!(
            (1..=32).contains(&self.requests) && self.requests <= dataset.samples.len(),
            "requests must select 1..=32 existing dataset samples"
        );
        ensure!(
            (1..=8).contains(&self.recovery_requests),
            "recovery requests must be 1..=8"
        );
        ensure!(
            (10..=60_000).contains(&self.poll_ms),
            "poll interval must be 10..=60000 ms"
        );
        ensure!(
            (1..=3_600_000).contains(&self.request_timeout_ms)
                && (1..=3_600_000).contains(&self.observation_ms),
            "request and observation bounds must be 1..=3600000 ms"
        );
        ensure!(
            (1..=60_000).contains(&self.health_timeout_ms),
            "health timeout must be 1..=60000 ms"
        );
        ensure!(
            self.cancel_after_ms < self.request_timeout_ms
                && self.hold_body_ms < self.request_timeout_ms
                && self.recovery_delay_ms <= 3_600_000,
            "lifecycle delays must fit their bounds"
        );
        ensure!(
            (4..=65_536).contains(&self.maximum_health_snapshots),
            "health snapshot bound must be 4..=65536"
        );
        ensure!(
            (1..=128 * 1024 * 1024).contains(&self.maximum_response_bytes)
                && (1..=1024 * 1024).contains(&self.maximum_health_bytes),
            "invalid response/health byte bound"
        );
        ensure!(
            (1..=16 * 1024 * 1024).contains(&self.maximum_dataset_bytes)
                && (1..=128 * 1024 * 1024).contains(&self.maximum_reference_bytes),
            "invalid dataset/reference byte bound"
        );
        ensure!(
            !self.model.is_empty() && self.model.len() <= 1024,
            "model alias must be bounded and nonempty"
        );
        let origin = local_url(&self.endpoint)?;
        for url in
            std::iter::once(self.health_endpoint.as_str()).chain(self.edit_endpoint.as_deref())
        {
            let parsed = local_url(url)?;
            ensure!(
                parsed.origin() == origin.origin(),
                "generation/edit/health URLs must have the same local origin"
            );
        }
        for sample in &dataset.samples[..self.requests] {
            ensure!(
                sample.effective_mode(&dataset.mode).is_generate() || self.edit_endpoint.is_some(),
                "edit samples require an explicit edit endpoint"
            );
        }
        Ok(())
    }
    fn limits(&self) -> ImageResponseLimits {
        ImageResponseLimits {
            maximum_response_bytes: self.maximum_response_bytes,
            ..Default::default()
        }
    }
}

fn local_url(text: &str) -> anyhow::Result<reqwest::Url> {
    let url = reqwest::Url::parse(text)?;
    ensure!(
        url.scheme() == "http"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "only credential-free local HTTP URLs without query/fragment are allowed"
    );
    ensure!(
        url.host_str().is_some_and(|host| host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())),
        "use a numeric loopback address (127.0.0.1 or [::1])"
    );
    Ok(url)
}

fn read_bounded(path: &Path, maximum: usize) -> anyhow::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    ensure!(file.metadata()?.is_file(), "input must be a regular file");
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= maximum, "input byte limit exceeded");
    Ok(bytes)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Arguments::parse();
    ensure!(
        (1..=16 * 1024 * 1024).contains(&args.maximum_dataset_bytes),
        "invalid dataset byte bound"
    );
    let bytes = read_bounded(&args.dataset, args.maximum_dataset_bytes)
        .context("read lifecycle dataset")?;
    let dataset: ImageDataset = serde_json::from_slice(&bytes).context("parse image dataset")?;
    args.validate(&dataset)?;
    let prepared = request::prepare(&args, dataset)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .context("reserve new report file")?;
    let report = run::run(&args, prepared).await?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    println!("Lifecycle observations saved to {}: {} unexpected failures; findings distinguish observed/not_observed/failed. Client cancellation alone is not server cancellation.", args.output.display(), report.unexpected_failures);
    ensure!(
        report.unexpected_failures == 0,
        "unexpected protocol, validation, observation or recovery failures; report saved"
    );
    Ok(())
}
