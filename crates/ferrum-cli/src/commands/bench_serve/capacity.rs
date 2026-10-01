//! One frozen capacity acquisition block against an already running server.
//! The outer Rust experiment controller owns server restart/reset receipts.
//! Raw histories, never saved verdicts, resume the pure bench-core planner.

use super::*;
use ferrum_bench_core::capacity_search::*;
use ferrum_bench_core::slo_comparison::artifact::SidecarArrival;
use ferrum_bench_core::{BenchmarkRequestRecord, BenchmarkRequestTimingEvidence};
use futures::{stream::FuturesUnordered, FutureExt, StreamExt};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Read, Write},
    panic::AssertUnwindSafe,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
mod server_queue;
mod session;

#[derive(Args, Clone, Default)]
pub struct CapacityArgs {
    /// Frozen open-arrival capacity contract. Executes its next pending block;
    /// --out records original evidence and the recomputed search report.
    #[arg(long, requires = "out")]
    pub capacity_contract: Option<PathBuf>,
    /// Previously acquired capacity files, in original acquisition order.
    /// Each contains raw evidence; stored assessments are ignored.
    #[arg(long, requires = "capacity_contract")]
    pub capacity_history: Vec<PathBuf>,
    /// Original process-owning controller checkpoint for this block.
    #[arg(long, requires = "capacity_contract")]
    pub capacity_session_receipt: Option<PathBuf>,
    /// Canonical original successful warmup evidence digest in --capacity-history.
    #[arg(long, requires = "capacity_session_receipt")]
    pub capacity_reuse_warmup_from: Option<String>,
}

fn err(message: impl Into<String>) -> ferrum_types::FerrumError {
    ferrum_types::FerrumError::model(message)
}
fn unix_ns() -> Result<u64> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| err(e.to_string()))?
            .as_nanos(),
    )
    .map_err(|_| err("wall timestamp exceeds u64"))
}
fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    // Same bounded metadata scale as the existing comparison loader. Large
    // raw collections belong in its typed artifact workflow, not unbounded
    // preflight allocation in a benchmark process.
    const LIMIT: u64 = 256 * 1024 * 1024;
    let file =
        std::fs::File::open(path).map_err(|e| err(format!("open {}: {e}", path.display())))?;
    if file.metadata().map_err(|e| err(e.to_string()))?.len() > LIMIT {
        return Err(err("capacity input exceeds 256MiB reader bound"));
    }
    let bytes = read_bounded(file, LIMIT)?;
    serde_json::from_slice(&bytes).map_err(|e| err(format!("read {}: {e}", path.display())))
}

fn read_bounded(reader: impl Read, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(
            limit
                .checked_add(1)
                .ok_or_else(|| err("invalid reader bound"))?,
        )
        .read_to_end(&mut bytes)
        .map_err(|e| err(e.to_string()))?;
    if bytes.len() as u64 > limit {
        return Err(err("capacity input exceeds reader bound"));
    }
    Ok(bytes)
}

#[derive(Deserialize)]
struct PriorCapture {
    evidence: CapacityRunEvidence,
}

#[derive(Serialize)]
struct Capture {
    schema_version: u32,
    identity_scope: &'static str,
    queue_scope: &'static str,
    evidence: CapacityRunEvidence,
    assessment: Option<CapacityRunAssessment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acquisition_error: Option<String>,
    search: SearchSummary,
}

#[derive(Serialize)]
struct SearchSummary {
    complete: bool,
    recorded_blocks: usize,
    aa_noise: Vec<AaPairObservation>,
    measured_rate_indices: Vec<usize>,
    unmeasured_rate_indices: Vec<usize>,
    independently_confirmed_rate_indices: Vec<usize>,
    maximum_confirmed_tested_rate_rps: Option<f64>,
    scope: String,
}

impl From<CapacitySearchReport> for SearchSummary {
    fn from(report: CapacitySearchReport) -> Self {
        Self {
            complete: report.complete,
            recorded_blocks: report.runs.len(),
            aa_noise: report.aa_noise,
            measured_rate_indices: report.measured_rate_indices,
            unmeasured_rate_indices: report.unmeasured_rate_indices,
            independently_confirmed_rate_indices: report.independently_confirmed_rate_indices,
            maximum_confirmed_tested_rate_rps: report.maximum_confirmed_tested_rate_rps,
            scope: report.scope,
        }
    }
}

pub(super) async fn execute(cmd: &BenchServeCommand) -> Result<()> {
    let contract: CapacityContract = read_json(
        cmd.capacity
            .capacity_contract
            .as_ref()
            .expect("capacity branch"),
    )?;
    let mut search = CapacitySearch::new(contract).map_err(|e| err(e.to_string()))?;
    if cmd.capacity.capacity_history.len() > search.contract().maximum_planned_runs {
        return Err(err("capacity history exceeds frozen acquisition bound"));
    }
    for path in &cmd.capacity.capacity_history {
        let previous: PriorCapture = read_json(path)?;
        search
            .record(previous.evidence)
            .map_err(|e| err(format!("history {}: {e}", path.display())))?;
    }
    let SearchProgress::Awaiting { runs, .. } = search.progress() else {
        return Err(err("all frozen capacity blocks are already recorded"));
    };
    let planned = search
        .planned_run(&runs[0])
        .map_err(|e| err(e.to_string()))?;
    let authorization = session::authorize(cmd, &search)?;
    let contract = search.contract();
    if !matches!(
        contract.queue_observation_source,
        QueueObservationSource::ClientScheduledLifecycle
            | QueueObservationSource::ServerAdmissionV1
    ) {
        return Err(err("HTTP capacity acquisition requires ClientScheduledLifecycle or the typed ServerAdmissionV1 health protocol"));
    }
    if cmd.dataset != "sharegpt"
        || !cmd.ignore_eos
        || cmd.sharegpt.sharegpt_fixed_output_tokens.is_some()
        || cmd.scenario != BenchServeWorkload::Standard
        || cmd.reasoning_effort.is_some()
    {
        return Err(err("capacity acquisition requires the frozen ShareGPT exact-budget standard workload, ignore-eos and no additional reasoning override"));
    }
    if cmd.request_rate.is_some()
        || !cmd.concurrency_sweep.is_empty()
        || cmd.slo.slo_client_config.is_some()
        || cmd.slo.slo_out.is_some()
    {
        return Err(err("capacity rate grid and SLO are owned by the explicit contract; separate sweep/SLO flags conflict"));
    }
    if cmd.sampling.request_sampling() != contract.sampling
        || cmd.enable_thinking != contract.enable_thinking
        || cmd.http_connection_mode.as_str() != contract.http_connection_mode
        || cmd.model != contract.identity.server.request_model_alias
        || cmd.target_backend.map(|b| b.as_str()) != Some(contract.identity.server.backend.as_str())
    {
        return Err(err("HTTP sampling, model, backend, thinking or connection mode differs from capacity contract"));
    }
    let mut preparation = cmd.clone();
    preparation.n_repeats = 1;
    preparation.warmup_requests = u32::try_from(
        contract
            .workload
            .samples
            .iter()
            .filter(|s| s.phase == BenchmarkPhase::Warmup)
            .count(),
    )
    .map_err(|_| err("warmup count overflow"))?;
    preparation.num_prompts =
        u32::try_from(contract.workload.samples.len() - preparation.warmup_requests as usize)
            .map_err(|_| err("workload count overflow"))?;
    let prepared =
        sharegpt::prepare(&preparation)?.ok_or_else(|| err("missing ShareGPT workload"))?;
    let prepared = &prepared[0];
    if prepared.evidence.source_sha256 != contract.identity.dataset_source_sha256
        || prepared.evidence.tokenizer_sha256 != contract.identity.shared.tokenizer_sha256
        || prepared.evidence.repeats.first() != Some(&contract.workload)
    {
        return Err(err(
            "actual dataset/tokenizer/ordered prompts differ from the frozen capacity workload",
        ));
    }
    let idle = match cmd.http_connection_mode {
        BenchHttpConnectionMode::Pooled => 64,
        BenchHttpConnectionMode::Fresh => 0,
    };
    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(idle)
            .build()
            .map_err(|e| err(e.to_string()))?,
    );
    let run_id = format!("capacity-{}", Uuid::new_v4());
    let ctx = RunContext {
        client,
        base_url: Arc::new(cmd.base_url.clone()),
        model: Arc::new(cmd.model.clone()),
        max_out: cmd.random_output_len,
        ignore_eos: true,
        enable_thinking: cmd.enable_thinking,
        reasoning_effort: None,
        sampling: cmd.sampling.request_sampling(),
        timeout_s: cmd.timeout,
        benchmark_run_id: Arc::new(run_id.clone()),
        capture_slo: true,
    };
    let output = cmd
        .out
        .as_ref()
        .ok_or_else(|| err("capacity requires --out"))?;
    // Reserve a new output before offering any work; never overwrite evidence.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| err(format!("create {}: {e}", output.display())))?;
    let started = unix_ns()?;
    let collected = run_fixed_window_with_queue(
        &ctx,
        &planned,
        &prepared.prompts,
        session::warmup_count(authorization.as_ref(), preparation.warmup_requests),
        &contract.window,
        contract.maximum_queue_samples_per_run,
        contract.queue_observation_source,
    )
    .await?;
    let mut evidence =
        assemble_evidence(contract, &planned, collected, run_id, started, unix_ns()?)?;
    session::bind(&mut evidence, authorization);
    // A protocol/session failure after actual acquisition must not erase raw
    // evidence. Stored assessments remain optional and are never replay input.
    let (assessment, acquisition_error) = match search.record(evidence.clone()) {
        Ok(value) => (Some(value.clone()), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let failed = assessment.as_ref().is_none_or(|value| {
        value.status != ferrum_bench_core::slo::SloStatus::Pass
            || value.acquisition_disposition != CapacityAcquisitionDisposition::Complete
    });
    let queue_scope = match evidence.queue_observation_source {
        QueueObservationSource::ServerAdmissionV1 => "Original server admission protocol; single scheduler-index membership and original trusted ingress ages. Instance-scoped server monotonic clock; client HTTP brackets establish observation-window coverage, without synchronized clocks. Includes preempted work, excludes pre-publication preprocessing and completed transport drain. HTTP failures and missing metrics remain raw N/A evidence.",
        _ => "Client-observed scheduled-arrival to terminal lifecycle, including not-yet-dispatched work and transport. This is end-to-end unfinished backlog and oldest scheduled age, not server-only queue depth or age.",
    };
    let capture=Capture {schema_version:1,identity_scope:"Server/hardware/config identity is the frozen experiment-controller declaration; remote binary and reset/restart receipts are retained by the outer serving controller. This HTTP client does not attest a remote executable.",queue_scope,evidence,assessment,acquisition_error,search:search.report().into()};
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &capture).map_err(|e| err(e.to_string()))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|e| err(e.to_string()))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| err(e.to_string()))?;
    if failed {
        return Err(err("capacity block failed or has incomplete evidence; original raw results and acquisition validity were written"));
    }
    Ok(())
}

struct WindowRun {
    requests: CollectedRequests,
    arrivals: Vec<SidecarArrival>,
    queue: Vec<ServiceQueueSample>,
    duration_s: f64,
    warmup: WarmupSummary,
    queue_capture_complete: bool,
    server_queue_attempts: Vec<ServerQueueAttempt>,
}

#[cfg(test)]
async fn run_fixed_window(
    ctx: &RunContext,
    planned: &PlannedCapacityRun,
    prompts: &[PromptCase],
    warmup_count: u32,
    window: &CapacityWindow,
    maximum_queue_samples: usize,
) -> Result<WindowRun> {
    run_fixed_window_with_queue(
        ctx,
        planned,
        prompts,
        warmup_count,
        window,
        maximum_queue_samples,
        QueueObservationSource::ClientScheduledLifecycle,
    )
    .await
}

/// Independent scheduled arrivals are offered even while earlier responses
/// remain open. Reqwest deadlines retain partial stream observations through
/// the existing collector; no outer timeout silently drops pending tasks.
async fn run_fixed_window_with_queue(
    ctx: &RunContext,
    planned: &PlannedCapacityRun,
    prompts: &[PromptCase],
    warmup_count: u32,
    window: &CapacityWindow,
    maximum_queue_samples: usize,
    queue_source: QueueObservationSource,
) -> Result<WindowRun> {
    let mut warmups = Vec::new();
    for (index, prompt) in prompts.iter().take(warmup_count as usize).enumerate() {
        warmups.push(
            stream_one(
                &ctx.client,
                &ctx.base_url,
                &ctx.model,
                prompt.clone(),
                ctx.max_out,
                ctx.ignore_eos,
                ctx.enable_thinking,
                ctx.reasoning_effort,
                ctx.sampling,
                ctx.timeout_s,
                benchmark_request_correlation(
                    &ctx.benchmark_run_id,
                    &planned.cell_id,
                    planned.key.repetition,
                    BenchmarkPhase::Warmup,
                    index,
                ),
            )
            .await,
        );
    }
    let warmup = summarize_warmup(warmup_count as usize, &warmups, 0);
    let count = planned.scheduled_arrival_ms.len();
    let mut records: Vec<Option<CollectedRequest>> = (0..count).map(|_| None).collect();
    let mut arrivals = vec![SidecarArrival::default(); count];
    let mut queue = Vec::new();
    let mut queue_capture_complete = true;
    let mut futures = FuturesUnordered::new();
    let mut dispatched = 0;
    let prepared_queue = if queue_source == QueueObservationSource::ServerAdmissionV1 {
        Some(server_queue::Prepared::new(ctx, window.maximum_queue_sample_gap_seconds).await?)
    } else {
        None
    };
    let origin = Instant::now();
    let server_sampler =
        prepared_queue.map(|prepared| prepared.start(origin, maximum_queue_samples));
    let send_duration =
        Duration::try_from_secs_f64(planned.send_seconds).map_err(|e| err(e.to_string()))?;
    let drain_duration = Duration::try_from_secs_f64(window.maximum_drain_seconds)
        .map_err(|e| err(e.to_string()))?;
    let end = origin
        .checked_add(send_duration)
        .ok_or_else(|| err("send window exceeds clock range"))?;
    let drain_end = end
        .checked_add(drain_duration)
        .ok_or_else(|| err("drain window exceeds clock range"))?;
    // Explicit maximum gap controls the sampler. Half its budget leaves room
    // for runtime wake-up jitter, which the evaluator checks from actual times.
    let sample_period = Duration::try_from_secs_f64(window.maximum_queue_sample_gap_seconds / 2.0)
        .map_err(|e| err(e.to_string()))?;
    if sample_period.is_zero() {
        return Err(err("queue sample period is below clock resolution"));
    }
    queue.push(lifecycle_sample(0.0, planned, &arrivals, &records));
    let mut next_sample = origin + sample_period;
    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(origin).as_secs_f64();
        while dispatched < count && planned.scheduled_arrival_ms[dispatched] <= elapsed * 1000.0 {
            let index = dispatched;
            let prompt = prompts
                .get(planned.workload_sample_indices[index])
                .ok_or_else(|| err("planned prompt index missing"))?
                .clone();
            let sent = origin.elapsed().as_secs_f64() * 1000.0;
            arrivals[index] = SidecarArrival {
                scheduled_arrival_ms: Some(planned.scheduled_arrival_ms[index]),
                dispatched_ms: Some(sent),
                request_started_ms: None,
                client_dispatch_backlog: Some(
                    planned
                        .scheduled_arrival_ms
                        .partition_point(|t| *t <= sent)
                        .saturating_sub(index) as u64,
                ),
            };
            let context = ctx.clone_inner();
            let correlation = benchmark_request_correlation(
                &ctx.benchmark_run_id,
                &planned.cell_id,
                planned.key.repetition,
                BenchmarkPhase::Measured,
                index,
            );
            let remaining = drain_end
                .saturating_duration_since(Instant::now())
                .as_secs_f64();
            // A nonzero transport timeout is required even after a late client
            // exhausted the drain budget; that lateness will fail evaluation.
            let timeout = context.timeout_s.min(remaining.max(0.000_001));
            futures.push(async move {
                let input_tokens = prompt.input_tokens;
                let backup = correlation.clone();
                let result = AssertUnwindSafe(stream_one_observed(
                    &context.client,
                    &context.base_url,
                    &context.model,
                    prompt,
                    context.max_out,
                    context.ignore_eos,
                    context.enable_thinking,
                    context.reasoning_effort,
                    context.sampling,
                    timeout,
                    correlation,
                    None,
                ))
                .catch_unwind()
                .await;
                let record = match result {
                    Ok(observed) => CollectedRequest::observed(observed, true),
                    Err(_) => join_failed_record(input_tokens, backup).into(),
                };
                (index, record)
            });
            dispatched += 1;
        }
        if now >= next_sample {
            let sample =
                lifecycle_sample(origin.elapsed().as_secs_f64(), planned, &arrivals, &records);
            if queue.len() < maximum_queue_samples {
                queue.push(sample);
            } else {
                queue_capture_complete = false;
            }
            next_sample = Instant::now() + sample_period;
        }
        if dispatched == count && futures.is_empty() && Instant::now() >= end {
            break;
        }
        let mut wake = next_sample;
        if dispatched < count {
            wake = wake.min(
                origin + Duration::from_secs_f64(planned.scheduled_arrival_ms[dispatched] / 1000.0),
            );
        }
        if Instant::now() < end {
            wake = wake.min(end);
        }
        tokio::select! {
            Some((index,record))=futures.next(),if !futures.is_empty()=>{
                arrivals[index].request_started_ms=record.started_at.map(|t|t.duration_since(origin).as_secs_f64()*1000.0);
                records[index]=Some(record);
            },
            _=tokio::time::sleep_until(tokio::time::Instant::from_std(wake))=>{},
        }
    }
    let duration_s = origin.elapsed().as_secs_f64();
    if queue.last().is_none_or(|q| q.at_seconds < duration_s) {
        if queue.len() < maximum_queue_samples {
            queue.push(lifecycle_sample(duration_s, planned, &arrivals, &records));
        } else {
            queue_capture_complete = false;
        }
    }
    let mut requests = CollectedRequests::default();
    for (index, record) in records.into_iter().enumerate() {
        let record = record.unwrap_or_else(|| {
            join_failed_record(
                prompts[planned.workload_sample_indices[index]].input_tokens,
                benchmark_request_correlation(
                    &ctx.benchmark_run_id,
                    &planned.cell_id,
                    planned.key.repetition,
                    BenchmarkPhase::Measured,
                    index,
                ),
            )
            .into()
        });
        requests.push(record, true);
    }
    let server_queue_attempts = if let Some(sampler) = server_sampler {
        let (attempts, complete) = sampler.finish().await?;
        queue.clear();
        queue_capture_complete = complete;
        attempts
    } else {
        Vec::new()
    };
    Ok(WindowRun {
        requests,
        arrivals,
        queue,
        duration_s,
        warmup,
        queue_capture_complete,
        server_queue_attempts,
    })
}

fn lifecycle_sample(
    at_seconds: f64,
    planned: &PlannedCapacityRun,
    arrivals: &[SidecarArrival],
    records: &[Option<CollectedRequest>],
) -> ServiceQueueSample {
    let now_ms = at_seconds * 1000.0;
    let mut waiting = 0;
    let mut active = 0;
    let mut oldest = 0.0_f64;
    for (index, &scheduled) in planned
        .scheduled_arrival_ms
        .iter()
        .enumerate()
        .take_while(|(_, t)| **t <= now_ms)
    {
        if records[index].is_some() {
            continue;
        }
        oldest = oldest.max(now_ms - scheduled);
        if arrivals[index].dispatched_ms.is_some() {
            active += 1;
        } else {
            waiting += 1;
        }
    }
    ServiceQueueSample {
        at_seconds,
        waiting_requests: waiting,
        active_requests: active,
        oldest_request_age_ms: oldest,
    }
}

fn assemble_evidence(
    contract: &CapacityContract,
    planned: &PlannedCapacityRun,
    run: WindowRun,
    run_id: String,
    started: u64,
    ended: u64,
) -> Result<CapacityRunEvidence> {
    let mut request_records = Vec::new();
    for (index, record) in run.requests.records.iter().enumerate() {
        let sample_index = planned.workload_sample_indices[index];
        let sample = &contract.workload.samples[sample_index];
        request_records.push(CapacityRequestRecord {
            workload_sample_index: sample_index,
            dispatched_prompt_sha256: sample.prompt_sha256.clone(),
            input_tokens: record.input_tokens,
            server_input_tokens: record.server_input_tokens,
            record: BenchmarkRequestRecord {
                correlation: record
                    .benchmark_correlation
                    .clone()
                    .ok_or_else(|| err("collector lost request correlation"))?,
                server_request_id: record.server_request_id.clone(),
                timing: Some(BenchmarkRequestTimingEvidence {
                    success: record.success,
                    reported_ttft_ms: record.ttft_ms,
                    reported_e2e_ms: record.e2e_ms,
                    event_source: record.itl_evidence.source,
                    observed_first_output: (record.itl_evidence.source != ItlEvidenceSource::None
                        && record.quality_issues.panic == 0)
                        .then_some(record.itl_evidence.output_events > 0),
                    raw_event_gaps_ms: record.itl_ms.clone(),
                }),
            },
        });
    }
    Ok(CapacityRunEvidence {
        session: None,
        contract_sha256: planned.contract_sha256.clone(),
        key: planned.key.clone(),
        identity: contract.identity.clone(),
        run_started_unix_ns: started,
        run_ended_unix_ns: ended,
        independent_run_id: run_id.clone(),
        benchmark_run_id: run_id,
        warmup: CapacityWarmupEvidence {
            acquisition: Default::default(),
            expected: run.warmup.expected,
            completed: run.warmup.completed,
            errored: run.warmup.errored,
            quality: run.warmup.quality_issues,
        },
        send_window_seconds: planned.send_seconds,
        measured_duration_seconds: run.duration_s,
        arrivals: run.arrivals,
        quality: run
            .requests
            .records
            .iter()
            .map(|r| r.quality_issues.clone())
            .collect(),
        requests: run.requests.evidence,
        request_records,
        queue_observation_source: contract.queue_observation_source,
        queue: run.queue,
        queue_capture_complete: run.queue_capture_complete,
        server_queue_attempts: run.server_queue_attempts,
    })
}

#[cfg(test)]
mod tests;
