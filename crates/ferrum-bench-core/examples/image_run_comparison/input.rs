//! Owned typed projections of the existing bench-images JSON. Unrelated
//! diagnostic fields are intentionally accepted; required evidence is checked.
use ferrum_bench_core::images::{
    sha256, ImageDataset, ImageMetrics, ImageRequestMode, ImageRequestPhase, ImageRequestRecord,
    ImageResponseLimits, ImageSloStatus, ValidatedImage,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema_version: u32,
    pub workloads: Vec<Workload>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub id: String,
    pub dataset_sha256: String,
    pub dataset_samples: usize,
    pub repetitions: usize,
    pub warmup_repetitions: usize,
    pub timeout_seconds: u64,
    pub latency_slo_p99_ms: f64,
    pub capture_enabled: bool,
    pub planned_pairs: usize,
    pub baseline: Arm,
    pub candidate: Arm,
    pub pairs: Vec<Pair>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub concurrency: usize,
    pub model: String,
    pub server_config_sha256: String,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub id: String,
    pub order: Order,
    pub baseline: RunSource,
    pub candidate: RunSource,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    BaselineFirst,
    CandidateFirst,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunSource {
    pub report: PathBuf,
    pub endpoint_index: usize,
    /// References only, never an automated certification of independent runs.
    pub fresh_instance_evidence: Vec<PathBuf>,
}
impl Plan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.workloads.is_empty() {
            return Err("plan requires schema 1 and workloads".into());
        }
        let mut ids = BTreeSet::new();
        for w in &self.workloads {
            if w.id.is_empty()
                || !ids.insert(&w.id)
                || !digest(&w.dataset_sha256)
                || w.dataset_samples == 0
                || w.repetitions == 0
                || w.timeout_seconds == 0
                || !w.latency_slo_p99_ms.is_finite()
                || w.latency_slo_p99_ms <= 0.0
                || !(3..=30).contains(&w.planned_pairs)
                || w.pairs.len() > w.planned_pairs
            {
                return Err("invalid workload identity, measurement policy or pair count".into());
            }
            let mut pair_ids = BTreeSet::new();
            for arm in [&w.baseline, &w.candidate] {
                if arm.concurrency == 0
                    || arm.model.is_empty()
                    || !digest(&arm.server_config_sha256)
                {
                    return Err("arm requires concurrency/model/configuration SHA-256".into());
                }
            }
            for p in &w.pairs {
                if p.id.is_empty() || !pair_ids.insert(&p.id) {
                    return Err("pair IDs must be nonempty and unique".into());
                }
                for source in [&p.baseline, &p.candidate] {
                    if source.report.as_os_str().is_empty()
                        || source.fresh_instance_evidence.is_empty()
                        || source
                            .fresh_instance_evidence
                            .iter()
                            .any(|p| p.as_os_str().is_empty())
                    {
                        return Err("each arm requires a report and external fresh-instance evidence references".into());
                    }
                }
            }
            if w.pairs.len() == w.planned_pairs {
                let first = w
                    .pairs
                    .iter()
                    .filter(|p| p.order == Order::BaselineFirst)
                    .count();
                if first.abs_diff(w.planned_pairs - first) > 1 {
                    return Err("predeclared order must be balanced".into());
                }
            }
        }
        Ok(())
    }
}

pub fn read_bounded(path: &Path, limit: usize) -> std::io::Result<Vec<u8>> {
    if limit == 0 || limit >= isize::MAX as usize {
        return Err(std::io::Error::other("invalid input byte limit"));
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > limit as u64 {
        return Err(std::io::Error::other(
            "input must be a bounded regular file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(std::io::Error::other("input grew beyond byte limit"));
    }
    Ok(bytes)
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BenchInput {
    pub schema_version: u32,
    pub client_version: String,
    pub started_unix_ms: u128,
    pub dataset_sha256: String,
    pub dataset: ImageDataset,
    pub configuration: Configuration,
    pub endpoints: Vec<EndpointReport>,
    pub reference_image: Option<ValidatedImage>,
    pub capture: Option<CaptureSummary>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct Configuration {
    pub endpoints: Vec<Endpoint>,
    pub concurrency: usize,
    pub repetitions: usize,
    pub warmup_repetitions: usize,
    pub timeout_seconds: u64,
    pub latency_slo_p99_ms: Option<f64>,
    pub limits: ImageResponseLimits,
    pub capture: Option<CapturePresence>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct CapturePresence {}
#[derive(Debug, Deserialize, Serialize)]
pub struct CaptureSummary {
    pub complete: bool,
    pub failed: usize,
    pub not_produced: usize,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct Endpoint {
    pub model: String,
    pub server_config_declaration: String,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct EndpointReport {
    pub endpoint_index: usize,
    pub measured: ImageMetrics,
    pub repetitions: Vec<Repetition>,
    pub requests: Vec<ImageRequestRecord>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct Repetition {
    pub phase: ImageRequestPhase,
    pub repetition: usize,
    pub metrics: ImageMetrics,
}

#[derive(Debug, Serialize)]
pub struct RunOutcome {
    pub source: RunSource,
    pub report_sha256: Option<String>,
    pub observation: Option<Observation>,
    pub problem: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct Observation {
    pub client_version: String,
    pub started_unix_ms: u128,
    pub measured: ImageMetrics,
    pub warmup_errors: usize,
    pub eligible: bool,
}
impl RunOutcome {
    pub fn valid_observation(&self) -> Option<&Observation> {
        self.problem
            .is_none()
            .then_some(self.observation.as_ref())
            .flatten()
    }
}

pub fn load_run(
    root: &Path,
    source: &RunSource,
    workload: &Workload,
    arm: &Arm,
    limit: usize,
) -> RunOutcome {
    let mut outcome = RunOutcome {
        source: source.clone(),
        report_sha256: None,
        observation: None,
        problem: None,
    };
    let loaded = (|| -> Result<Observation, String> {
        let bytes = read_bounded(&root.join(&source.report), limit).map_err(|e| e.to_string())?;
        outcome.report_sha256 = Some(sha256(&bytes));
        let report: BenchInput = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        validate_report(report, source.endpoint_index, workload, arm)
    })();
    match loaded {
        Ok(v) => outcome.observation = Some(v),
        Err(e) => outcome.problem = Some(e),
    }
    outcome
}

fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}
fn optional_close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => close(a, b),
        (None, None) => true,
        _ => false,
    }
}
fn metrics_equal(a: &ImageMetrics, b: &ImageMetrics) -> bool {
    a.attempted == b.attempted
        && a.succeeded == b.succeeded
        && a.errors == b.errors
        && close(a.elapsed_seconds, b.elapsed_seconds)
        && optional_close(a.images_per_second, b.images_per_second)
        && optional_close(a.e2e_p50_ms, b.e2e_p50_ms)
        && optional_close(a.e2e_p99_ms, b.e2e_p99_ms)
        && optional_close(a.latency_slo_p99_ms, b.latency_slo_p99_ms)
        && a.latency_slo_status == b.latency_slo_status
}
pub fn validate_report(
    report: BenchInput,
    index: usize,
    w: &Workload,
    arm: &Arm,
) -> Result<Observation, String> {
    let fail = |s: &str| Err(s.to_string());
    let c = &report.configuration;
    report
        .dataset
        .validate(&c.limits)
        .map_err(|e| e.to_string())?;
    let expected_schema = if report.dataset.mode.is_generate() {
        1
    } else {
        2
    };
    if report.schema_version != expected_schema
        || report.client_version.is_empty()
        || report.started_unix_ms == 0
        || report.dataset.fingerprint().map_err(|e| e.to_string())? != report.dataset_sha256
        || report.dataset_sha256 != w.dataset_sha256
        || report.dataset.samples.len() != w.dataset_samples
        || c.concurrency != arm.concurrency
        || c.repetitions != w.repetitions
        || c.warmup_repetitions != w.warmup_repetitions
        || c.timeout_seconds != w.timeout_seconds
        || c.latency_slo_p99_ms != Some(w.latency_slo_p99_ms)
        || c.capture.is_some() != w.capture_enabled
        || report.capture.is_some() != w.capture_enabled
    {
        return fail("report differs from predeclared workload/measurement policy");
    }
    if let ImageRequestMode::Edit { image } = &report.dataset.mode {
        if image.encoded_sha256.is_none()
            || report.reference_image.as_ref().map(|r| &r.png_sha256)
                != image.encoded_sha256.as_ref()
        {
            return fail("edit reference identity missing or inconsistent");
        }
    }
    let endpoint = c
        .endpoints
        .get(index)
        .ok_or("configured endpoint missing")?;
    if endpoint.model != arm.model
        || sha256(endpoint.server_config_declaration.as_bytes()) != arm.server_config_sha256
    {
        return fail("endpoint differs from frozen arm configuration");
    }
    if report
        .endpoints
        .iter()
        .filter(|e| e.endpoint_index == index)
        .count()
        != 1
    {
        return fail("endpoint absent or duplicated");
    }
    let endpoint = report
        .endpoints
        .into_iter()
        .find(|e| e.endpoint_index == index)
        .unwrap();
    let total_rounds = w
        .repetitions
        .checked_add(w.warmup_repetitions)
        .ok_or("round count overflow")?;
    let expected = total_rounds
        .checked_mul(w.dataset_samples)
        .ok_or("request count overflow")?;
    if endpoint.requests.len() != expected || endpoint.repetitions.len() != total_rounds {
        return fail("incomplete request/round coverage");
    }
    let mut keys = BTreeSet::new();
    for r in &endpoint.requests {
        let measured = r.phase == ImageRequestPhase::Measured;
        let count = if measured {
            w.repetitions
        } else {
            w.warmup_repetitions
        };
        let Some(sample) = report.dataset.samples.get(r.sample_index) else {
            return fail("sample index out of range");
        };
        if r.repetition >= count
            || r.sample_id != sample.id
            || r.dispatch_index != r.sample_index
            || !keys.insert((measured, r.repetition, r.sample_index))
            || !r.e2e_ms.is_finite()
            || r.e2e_ms < 0.0
            || !r.validation_ms.is_finite()
            || r.validation_ms < 0.0
        {
            return fail("invalid or duplicate request identity/timing");
        }
        if let Some(image) = &r.image {
            if image.width != sample.width
                || image.height != sample.height
                || !digest(&image.png_sha256)
                || !digest(&image.rgba8_sha256)
                || image.rgba8_bytes
                    != c.limits
                        .rgba_bytes(sample.width, sample.height)
                        .map_err(|e| e.to_string())?
            {
                return fail("invalid decoded image evidence");
            }
        }
        if r.succeeded()
            && (!r.http_status.is_some_and(|s| (200..300).contains(&s)) || r.error.is_some())
        {
            return fail("successful image conflicts with HTTP/error evidence");
        }
    }
    let mut rounds = BTreeSet::new();
    let mut elapsed = 0.0;
    for r in &endpoint.repetitions {
        let measured = r.phase == ImageRequestPhase::Measured;
        let count = if measured {
            w.repetitions
        } else {
            w.warmup_repetitions
        };
        if r.repetition >= count
            || !rounds.insert((measured, r.repetition))
            || !r.metrics.elapsed_seconds.is_finite()
            || r.metrics.elapsed_seconds <= 0.0
        {
            return fail("invalid or duplicate round");
        }
        let records: Vec<_> = endpoint
            .requests
            .iter()
            .filter(|v| v.phase == r.phase && v.repetition == r.repetition)
            .cloned()
            .collect();
        let metrics = ImageMetrics::from_records(
            r.metrics.elapsed_seconds,
            &records,
            measured.then_some(w.latency_slo_p99_ms),
        );
        if !metrics_equal(&metrics, &r.metrics) {
            return fail("round metrics disagree with request records");
        }
        if measured {
            elapsed += r.metrics.elapsed_seconds;
        }
    }
    let records: Vec<_> = endpoint
        .requests
        .iter()
        .filter(|r| r.phase == ImageRequestPhase::Measured)
        .cloned()
        .collect();
    let metrics = ImageMetrics::from_records(elapsed, &records, Some(w.latency_slo_p99_ms));
    if !metrics_equal(&metrics, &endpoint.measured) || metrics.images_per_second.is_none() {
        return fail("measured aggregate inconsistent or missing throughput");
    }
    let warmup_errors = endpoint
        .requests
        .iter()
        .filter(|r| r.phase == ImageRequestPhase::Warmup && !r.succeeded())
        .count();
    let capture_ok = report
        .capture
        .is_none_or(|r| r.complete && r.failed == 0 && r.not_produced == 0);
    let eligible = warmup_errors == 0
        && metrics.errors == 0
        && metrics.latency_slo_status == ImageSloStatus::Met
        && capture_ok;
    Ok(Observation {
        client_version: report.client_version,
        started_unix_ms: report.started_unix_ms,
        measured: metrics,
        warmup_errors,
        eligible,
    })
}
