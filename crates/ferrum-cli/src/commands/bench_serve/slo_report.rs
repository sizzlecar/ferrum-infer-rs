//! Optional SLO evidence output. The canonical legacy benchmark is unchanged.

use super::*;
use ferrum_bench_core::slo::{
    evaluate_slo, AdmissionEvidence, RequestOutcome, RequestSloEvidence, SloEvaluationConfig,
    SloEvaluationReport, SloStatus,
};
use serde::Serialize;
use std::io::Write;

#[derive(Args, Clone, Default)]
pub struct SloBenchmarkArgs {
    /// Explicit client-visible SLO JSON (SloClientVisibleConfig); server token
    /// budgets are not inferred. Requires --slo-out.
    #[arg(long, requires = "slo_out")]
    pub slo_client_config: Option<PathBuf>,

    /// Append versioned SLO evidence per cell as JSONL, alongside --out.
    #[arg(long, requires = "slo_client_config")]
    pub slo_out: Option<PathBuf>,

    /// Fail after writing reports if any repeat is Fail or Unknown.
    #[arg(long, requires = "slo_client_config")]
    pub slo_fail_on_violation: bool,
}

pub(super) struct LoadedSloConfig {
    pub evaluation: SloEvaluationConfig,
    pub source_sha256: String,
}

impl LoadedSloConfig {
    pub fn annotate_env(&self, env: &mut Env) {
        env.runtime_config.upsert(
            "bench_slo_capture",
            "true",
            ferrum_types::RuntimeConfigSource::Cli,
        );
        env.runtime_config.upsert(
            "bench_slo_client_config_sha256",
            &self.source_sha256,
            ferrum_types::RuntimeConfigSource::ConfigFile,
        );
    }
}

fn error(message: impl Into<String>) -> ferrum_types::FerrumError {
    ferrum_types::FerrumError::model(message)
}

pub(super) fn load(cmd: &BenchServeCommand) -> Result<Option<LoadedSloConfig>> {
    let args = &cmd.slo;
    if args.slo_client_config.is_some() != args.slo_out.is_some() {
        return Err(error(
            "--slo-client-config and --slo-out must be supplied together",
        ));
    }
    if args.slo_fail_on_violation && args.slo_client_config.is_none() {
        return Err(error(
            "--slo-fail-on-violation requires --slo-client-config",
        ));
    }
    let Some(path) = &args.slo_client_config else {
        return Ok(None);
    };
    if cmd.scenario != BenchServeWorkload::Standard {
        return Err(error(
            "SLO evidence output currently requires --scenario standard",
        ));
    }
    let output = args.slo_out.as_ref().expect("presence checked");
    let output_path = output_identity(output)?;
    if std::fs::metadata(&output_path).is_ok_and(|metadata| !metadata.is_file()) {
        return Err(error("--slo-out must name a regular file"));
    }
    for other in [cmd.out.as_ref(), Some(path)].into_iter().flatten() {
        if output_path == output_identity(other)? || same_existing_file(output, other) {
            return Err(error(
                "--slo-out must differ from --out and --slo-client-config",
            ));
        }
    }
    if let Some(legacy_output) = &cmd.out {
        if output_identity(legacy_output)? == output_identity(path)?
            || same_existing_file(legacy_output, path)
        {
            return Err(error("--out must differ from --slo-client-config"));
        }
    }
    let bytes = std::fs::read(path).map_err(|e| error(format!("read {}: {e}", path.display())))?;
    let config: ferrum_types::SloClientVisibleConfig = serde_json::from_slice(&bytes)
        .map_err(|e| error(format!("client-visible SLO config {}: {e}", path.display())))?;
    Ok(Some(LoadedSloConfig {
        evaluation: SloEvaluationConfig::from_client_visible(&config)
            .map_err(|e| error(format!("client-visible SLO config: {e}")))?,
        source_sha256: sha256_hex(&bytes),
    }))
}

fn output_identity(path: &std::path::Path) -> Result<PathBuf> {
    if let Ok(canonical) = path.canonicalize() {
        return Ok(canonical);
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let parent = parent.canonicalize().map_err(|e| {
        error(format!(
            "report/config parent {} must exist: {e}",
            parent.display()
        ))
    })?;
    let name = path
        .file_name()
        .ok_or_else(|| error("report/config path must name a file"))?;
    Ok(parent.join(name))
}

fn same_existing_file(first: &std::path::Path, second: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(first), Ok(second)) = (std::fs::metadata(first), std::fs::metadata(second)) {
            return first.dev() == second.dev() && first.ino() == second.ino();
        }
    }
    #[cfg(not(unix))]
    let _ = (first, second);
    false
}

pub(super) struct CollectedRequest {
    pub record: RequestRecord,
    pub evidence: Option<RequestSloEvidence>,
    pub started_at: Option<Instant>,
}

impl CollectedRequest {
    pub fn observed(observed: ObservedRequest, capture: bool) -> Self {
        let evidence = capture.then(|| {
            let last_visible_ms = observed
                .output_event_times
                .last()
                .and_then(|last| last.checked_duration_since(observed.started_at))
                .map(|elapsed| elapsed.as_secs_f64() * 1000.0);
            let mut evidence =
                RequestSloEvidence::from_legacy_record(&observed.record, last_visible_ms);
            evidence.admission = observed.admission;
            if observed.admission == AdmissionEvidence::Rejected {
                evidence.outcome = RequestOutcome::Rejected;
            }
            evidence
        });
        Self {
            record: observed.record,
            evidence,
            started_at: capture.then_some(observed.started_at),
        }
    }
}

impl From<RequestRecord> for CollectedRequest {
    fn from(record: RequestRecord) -> Self {
        Self {
            record,
            evidence: None,
            started_at: None,
        }
    }
}

#[derive(Default)]
pub(super) struct CollectedRequests {
    pub records: Vec<RequestRecord>,
    pub evidence: Vec<RequestSloEvidence>,
    pub started_at: Vec<Option<Instant>>,
}

impl std::ops::Deref for CollectedRequests {
    type Target = [RequestRecord];
    fn deref(&self) -> &Self::Target {
        &self.records
    }
}

impl CollectedRequests {
    pub fn push(&mut self, request: CollectedRequest, capture: bool) {
        if capture {
            self.evidence.push(
                request.evidence.unwrap_or_else(|| {
                    RequestSloEvidence::from_legacy_record(&request.record, None)
                }),
            );
            self.started_at.push(request.started_at);
        }
        self.records.push(request.record);
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct RequestArrivalEvidence {
    pub scheduled_arrival_ms: Option<f64>,
    pub dispatched_ms: Option<f64>,
    /// Existing collector start immediately before reqwest request submission.
    /// This is not a claim that bytes have left the socket or reached the server.
    pub request_started_ms: Option<f64>,
    /// Due arrivals not yet dispatched, including this request; open-loop only.
    pub client_dispatch_backlog: Option<u64>,
}

pub(super) struct CollectedRun {
    pub legacy: RunRecord,
    pub evidence: Vec<RequestSloEvidence>,
    pub arrivals: Vec<RequestArrivalEvidence>,
}

impl std::ops::Deref for CollectedRun {
    type Target = RunRecord;
    fn deref(&self) -> &Self::Target {
        &self.legacy
    }
}

pub(super) fn finish_run(
    collected: CollectedRequests,
    expected_requests: u32,
    duration_s: f64,
    warmup: WarmupSummary,
    start: Instant,
    mut arrivals: Vec<RequestArrivalEvidence>,
) -> CollectedRun {
    for (arrival, observed) in arrivals.iter_mut().zip(&collected.started_at) {
        arrival.request_started_ms = observed
            .and_then(|observed| observed.checked_duration_since(start))
            .map(|elapsed| elapsed.as_secs_f64() * 1000.0);
    }
    CollectedRun {
        legacy: RunRecord {
            records: collected.records,
            expected_requests,
            duration_s,
            warmup,
        },
        evidence: collected.evidence,
        arrivals,
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SloRepeatReport {
    pub repeat_index: u32,
    pub evaluation: SloEvaluationReport,
    pub arrivals: Vec<RequestArrivalEvidence>,
    pub request_start_window_s: Option<f64>,
    /// `(observed starts - 1) / (last start - first start)`; no drain time.
    pub observed_request_start_rate_rps: Option<f64>,
    pub max_client_dispatch_backlog: Option<u64>,
}

pub(super) fn evaluate_run(
    config: &LoadedSloConfig,
    repeat_index: u32,
    run: &CollectedRun,
) -> Result<SloRepeatReport> {
    let expected = run.legacy.expected_requests as usize;
    if run.evidence.len() != expected
        || run.arrivals.len() != expected
        || run.legacy.records.len() != expected
    {
        return Err(error("SLO collector lost measured request evidence"));
    }
    let evaluation = evaluate_slo(&config.evaluation, &run.evidence, run.legacy.duration_s)
        .map_err(|e| error(format!("SLO repeat {repeat_index}: {e}")))?;
    let starts: Vec<_> = run
        .arrivals
        .iter()
        .filter_map(|arrival| arrival.request_started_ms)
        .collect();
    let start_window = if starts.len() >= 2 {
        let first = starts.iter().copied().fold(f64::INFINITY, f64::min);
        let last = starts.iter().copied().fold(0.0, f64::max);
        (last > first).then_some((last - first) / 1000.0)
    } else {
        None
    };
    Ok(SloRepeatReport {
        repeat_index,
        evaluation,
        arrivals: run.arrivals.clone(),
        request_start_window_s: start_window,
        observed_request_start_rate_rps: start_window
            .map(|window| (starts.len() - 1) as f64 / window),
        max_client_dispatch_backlog: run
            .arrivals
            .iter()
            .filter_map(|arrival| arrival.client_dispatch_backlog)
            .max(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct SloCellReport {
    pub schema_version: u32,
    pub config_sha256: String,
    /// Original schema retains identity, workload, ordered correlations and
    /// legacy metrics. New timing definitions live only in `repeats`.
    pub legacy_benchmark: BenchReport,
    pub repeats: Vec<SloRepeatReport>,
}

pub(super) fn emit(cmd: &BenchServeCommand, report: &SloCellReport) -> Result<()> {
    let path = cmd
        .slo
        .slo_out
        .as_ref()
        .ok_or_else(|| error("missing --slo-out"))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| error(format!("open {}: {e}", path.display())))?;
    serde_json::to_writer(&mut file, report)
        .map_err(|e| error(format!("write SLO report: {e}")))?;
    file.write_all(b"\n")
        .and_then(|_| file.flush())
        .map_err(|e| error(format!("flush {}: {e}", path.display())))?;
    Ok(())
}

pub(super) fn enforce(cmd: &BenchServeCommand, reports: &[SloCellReport]) -> Result<()> {
    if !cmd.slo.slo_fail_on_violation {
        return Ok(());
    }
    if reports.is_empty()
        || reports.iter().any(|cell| {
            cell.repeats
                .iter()
                .any(|repeat| repeat.evaluation.latency_and_outcome_status != SloStatus::Pass)
        })
    {
        return Err(error(
            "SLO evaluation failed or has unknown evidence; see --slo-out",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
