//! Closed-loop benchmark for asynchronous /v1/videos jobs.
//! Artifacts and reports must be written outside the source tree. No server
//! configuration changes, generation retries, or implicit job deletion occur.

use clap::Parser;
use serde::Serialize;
use std::{path::PathBuf, sync::Arc, time::SystemTime};

#[path = "video_job_bench/client.rs"]
mod client;
#[path = "video_job_bench/input.rs"]
mod input;
#[path = "video_job_bench/report.rs"]
mod report;
#[path = "video_job_bench/resources.rs"]
mod resources;
#[path = "video_job_bench/validation.rs"]
mod validation;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Parser, Serialize)]
#[command(about = "Measure async video jobs and validate downloaded media after each cell")]
struct Arguments {
    #[command(flatten)]
    resources: resources::ResourceArguments,
    /// Server origin, without /v1/videos.
    #[arg(long)]
    base_url: String,
    /// Ordered request JSON files. The same sequence is replayed in every cell.
    #[arg(long, required = true)]
    request_json: Vec<PathBuf>,
    /// Human-supplied hardware, model/precision/revision, server version and workload provenance.
    #[arg(long)]
    provenance: PathBuf,
    /// Exact server configuration snapshots; hashed, never changed by this client.
    #[arg(long, required = true)]
    server_config: Vec<PathBuf>,
    /// Must not already exist. Holds report, downloaded media and probe evidence.
    #[arg(long)]
    output: PathBuf,
    /// Run a declared submit/health-gate/delete lifecycle scenario instead of timing cells.
    /// Record removal alone does not establish native execution retirement.
    #[arg(long)]
    lifecycle_plan: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "1,2")]
    concurrency: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    /// Fixed offered request count in each repeat, cycling through request-json.
    #[arg(long, default_value_t = 2)]
    requests_per_repeat: usize,
    /// Serial warmup requests before each cell, retained separately.
    #[arg(long, default_value_t = 1)]
    warmup_requests: usize,
    #[arg(long, default_value_t = 120.0)]
    latency_slo_p99_seconds: f64,
    #[arg(long, default_value_t = 600)]
    timeout_seconds: u64,
    #[arg(long, default_value_t = 250)]
    poll_interval_ms: u64,
    /// DELETE this client's job on timeout. Deletion does not prove GPU retirement.
    #[arg(long)]
    cancel_on_timeout: bool,
    #[arg(long, default_value_t = 1024 * 1024 * 1024)]
    maximum_artifact_bytes: u64,
    #[arg(long, default_value_t = 2 * 1024 * 1024)]
    maximum_json_bytes: usize,
    #[arg(long)]
    require_audio: bool,
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    #[arg(long, default_value_t = 120)]
    validation_timeout_seconds: u64,
    /// Read a bearer token from this file. Neither its path nor contents are reported.
    #[arg(long)]
    #[serde(skip)]
    bearer_token_file: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    execution_state: ExecutionState,
    client_version: &'static str,
    client_executable_sha256: String,
    started_unix_ms: u128,
    arguments: Arguments,
    input: input::Inputs,
    timing_contract: &'static str,
    memory_contract: &'static str,
    percentile_caveat: &'static str,
    cells: Vec<report::Cell>,
    in_progress_cell: Option<ActiveCell>,
    stopped_reason: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionState {
    Running,
    Complete,
    Stopped,
}

#[derive(Serialize)]
struct ActiveCell {
    phase: String,
    concurrency: usize,
    repeat: usize,
    planned_requests: usize,
}

async fn write_report(report: &Report) -> Result<()> {
    input::write_json_atomic(&report.arguments.output.join("report.json"), report).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    arguments.validate()?;
    arguments.resources.validate(&arguments.base_url)?;
    let input = input::Inputs::load(&arguments)?;
    // Verify local tools before the first generation request.
    validation::check_tools(&arguments).await?;
    let client = client::Client::new(&arguments).await?;
    if let Some(path) = &arguments.lifecycle_plan {
        let plan = client::lifecycle::Plan::load(path, &arguments, input.requests.len())?;
        return client::lifecycle::run(client, arguments, input, plan).await;
    }
    let sampler = resources::ServerSampler::new(&arguments.resources);
    std::fs::create_dir(&arguments.output)?;
    let mut report = Report {
        schema_version: 1,
        execution_state: ExecutionState::Running,
        client_version: env!("CARGO_PKG_VERSION"),
        client_executable_sha256: input::file_sha256(&std::env::current_exe()?)?,
        started_unix_ms: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_millis(),
        arguments,
        input,
        timing_contract: "closed loop; latency is client dispatch through complete artifact receipt; terminal observation is separate and includes polling quantization; throughput uses first dispatch through last request return; local media validation occurs after timing and before the next cell",
        memory_contract: "optional same-host procfs process-tree RSS and whole-device NVIDIA memory.used are sampled lower bounds with query/cadence/error evidence; absent sampling remains null; server job peak_memory_mb is unverified server-reported evidence, never a substitute for sampled whole-server peak",
        percentile_caveat: "empirical percentiles use linear interpolation over successful validated requests; errors remain in the offered denominator and fail the completion SLO; finite samples do not establish stable production P99; repeats are reported separately, with no independence claim",
        cells: Vec::new(),
        in_progress_cell: None,
        stopped_reason: None,
    };
    write_report(&report).await?;
    let client = Arc::new(client);
    'cells: for concurrency in report.arguments.concurrency.clone() {
        for repeat in 0..report.arguments.repetitions {
            for (phase, count, width) in [
                ("warmup", report.arguments.warmup_requests, 1),
                (
                    "measurement",
                    report.arguments.requests_per_repeat,
                    concurrency,
                ),
            ] {
                if count == 0 {
                    continue;
                }
                eprintln!("{phase}: concurrency={concurrency}, repeat={repeat}, requests={count}");
                report.in_progress_cell = Some(ActiveCell {
                    phase: phase.into(),
                    concurrency,
                    repeat,
                    planned_requests: count,
                });
                write_report(&report).await?;
                let outcome = client::run_cell(
                    client.clone(),
                    &report.arguments,
                    &report.input.requests,
                    phase,
                    concurrency,
                    width,
                    repeat,
                    count,
                    sampler.clone(),
                )
                .await;
                let cell = match outcome {
                    Ok(cell) => cell,
                    Err(error) => {
                        report.stopped_reason = Some(format!("cell failed: {error}; inspect per-request submission/observation receipts; remote terminal state may be unknown"));
                        write_report(&report).await?;
                        break 'cells;
                    }
                };
                let unresolved = cell.requests.iter().any(|r| r.remote_terminal_unknown);
                let warmup_failed = phase == "warmup" && cell.summary.errors != 0;
                let client_failed = !cell.client_errors.is_empty()
                    || cell.requests.iter().any(|r| r.evidence_error.is_some());
                report.cells.push(cell);
                report.in_progress_cell = None;
                if unresolved || warmup_failed || client_failed {
                    report.stopped_reason = Some(if unresolved {
                        "A submitted/possibly submitted job has unknown terminal state. Later cells were not run; resolve retained job IDs before another benchmark."
                    } else if client_failed {
                        "Client execution or evidence persistence failed; later cells were not run."
                    } else {
                        "Warmup failed; later cells were not run."
                    }.into());
                }
                write_report(&report).await?;
                if report.stopped_reason.is_some() {
                    break 'cells;
                }
            }
        }
    }
    report.execution_state = if report.stopped_reason.is_some() {
        ExecutionState::Stopped
    } else {
        ExecutionState::Complete
    };
    write_report(&report).await?;
    println!("{}", report.arguments.output.join("report.json").display());
    if report.stopped_reason.is_some()
        || report.cells.iter().any(|c| {
            c.summary.errors != 0
                || (c.phase == "measurement" && !c.summary.observed_slo_met)
                || c.resources.as_ref().is_some_and(|r| !r.complete)
        })
    {
        std::process::exit(1);
    }
    Ok(())
}
