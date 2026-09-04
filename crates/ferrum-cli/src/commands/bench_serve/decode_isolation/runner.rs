use super::super::{
    benchmark_request_correlation, gen_random_prompt, prompt_case, stream_one, stream_one_observed,
    BenchServeCommand, DecodeStreamProgress, ObservedRequest, PromptCase, RunContext,
};
use ferrum_bench_core::decode_isolation::{
    analyze_decode_isolation, DecodeEventTimeline, DecodeIsolationConfig,
    DecodeIsolationEvidenceValidity, DecodeIsolationOrchestrationEvidence,
    DecodeIsolationRequestEvidence, DecodeIsolationRunReport,
};
use ferrum_bench_core::{
    BenchmarkPhase, ItlEvidenceSource, OutputTokenCountSource, QualityIssueCounts,
};
use ferrum_types::Result;
use rand::{rngs::StdRng, SeedableRng};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::task::JoinHandle;

const CELL_ID: &str = "decode-isolation";

/// Tokio detaches a task when its handle is dropped. This owner instead
/// aborts every still-owned request, including when the enclosing benchmark
/// future is externally cancelled (Ctrl-C/runtime shutdown).
struct RequestTasks {
    handles: Vec<JoinHandle<ObservedRequest>>,
}

impl RequestTasks {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            handles: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, handle: JoinHandle<ObservedRequest>) {
        self.handles.push(handle);
    }
}

impl Drop for RequestTasks {
    fn drop(&mut self) {
        for handle in &self.handles {
            handle.abort();
        }
    }
}

struct AbortOnDropJoin(Option<JoinHandle<ObservedRequest>>);

impl AbortOnDropJoin {
    async fn join(mut self) -> std::result::Result<ObservedRequest, tokio::task::JoinError> {
        let result = self.0.as_mut().expect("join handle must be present").await;
        self.0.take();
        result
    }
}

impl Drop for AbortOnDropJoin {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
}

pub(super) struct PreparedRun {
    pub(super) repeat: u32,
    warmups: Vec<PromptCase>,
    incumbents: Vec<PromptCase>,
    aggressor: PromptCase,
}

pub(super) fn prepare_runs(
    cmd: &BenchServeCommand,
    config: &DecodeIsolationConfig,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<PreparedRun>> {
    let base_seed = cmd.seed.unwrap_or_else(rand::random);
    (0..cmd.n_repeats)
        .map(|repeat| {
            let mut rng =
                StdRng::seed_from_u64(base_seed ^ 0xDEC0_DE15_01A7_10u64 ^ ((repeat as u64) << 32));
            let warmups = (0..cmd.warmup_requests)
                .map(|_| make_prompt(tokenizer, config.incumbent_input_tokens, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let incumbents = (0..config.incumbents)
                .map(|_| make_prompt(tokenizer, config.incumbent_input_tokens, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let aggressor = make_prompt(tokenizer, config.aggressor_input_tokens, &mut rng)?;
            Ok(PreparedRun {
                repeat,
                warmups,
                incumbents,
                aggressor,
            })
        })
        .collect()
}

fn make_prompt(
    tokenizer: &tokenizers::Tokenizer,
    tokens: u64,
    rng: &mut StdRng,
) -> Result<PromptCase> {
    let tokens = usize::try_from(tokens)
        .map_err(|_| ferrum_types::FerrumError::model("prompt length exceeds platform capacity"))?;
    prompt_case(tokenizer, gen_random_prompt(tokenizer, tokens, rng))
}

pub(super) async fn run_once(
    cmd: &BenchServeCommand,
    ctx: &RunContext,
    config: &DecodeIsolationConfig,
    prepared: PreparedRun,
) -> DecodeIsolationRunReport {
    let repeat = prepared.repeat;
    let incumbent_output_tokens = usize::try_from(config.incumbent_output_tokens)
        .expect("validated incumbent output tokens must fit usize");
    let aggressor_output_tokens = usize::try_from(config.aggressor_output_tokens)
        .expect("validated aggressor output tokens must fit usize");

    let mut invalid_reasons = Vec::new();
    let mut warmup_valid = true;
    for (request_index, prompt) in prepared.warmups.into_iter().enumerate() {
        let record = stream_one(
            &ctx.client,
            &ctx.base_url,
            &ctx.model,
            prompt,
            incumbent_output_tokens,
            true,
            cmd.enable_thinking,
            cmd.reasoning_effort,
            cmd.timeout,
            benchmark_request_correlation(
                &ctx.benchmark_run_id,
                CELL_ID,
                repeat,
                BenchmarkPhase::Warmup,
                request_index,
            ),
        )
        .await;
        if !record_contract_valid(&record, incumbent_output_tokens) {
            warmup_valid = false;
        }
    }
    if !warmup_valid {
        invalid_reasons.push("warmup wire or usage contract was invalid".to_string());
    }

    let run_origin = Instant::now();
    let mut incumbent_handles = RequestTasks::with_capacity(prepared.incumbents.len());
    let mut progress_receivers = Vec::with_capacity(prepared.incumbents.len());
    for (request_index, prompt) in prepared.incumbents.into_iter().enumerate() {
        let (progress_tx, progress_rx) = watch::channel(DecodeStreamProgress::default());
        let request_ctx = ctx.clone_inner();
        let enable_thinking = cmd.enable_thinking;
        let reasoning_effort = cmd.reasoning_effort;
        let timeout = cmd.timeout;
        let correlation = benchmark_request_correlation(
            &ctx.benchmark_run_id,
            CELL_ID,
            repeat,
            BenchmarkPhase::Measured,
            request_index,
        );
        incumbent_handles.push(tokio::spawn(async move {
            stream_one_observed(
                &request_ctx.client,
                &request_ctx.base_url,
                &request_ctx.model,
                prompt,
                incumbent_output_tokens,
                true,
                enable_thinking,
                reasoning_effort,
                timeout,
                correlation,
                Some(progress_tx),
            )
            .await
        }));
        progress_receivers.push(progress_rx);
    }

    let ready_deadline = tokio::time::Instant::now() + Duration::from_secs_f64(cmd.timeout);
    let mut incumbents_ready = 0_u32;
    for progress in &mut progress_receivers {
        match wait_for_output_events(
            progress,
            config.contract.baseline_output_events_per_incumbent,
            ready_deadline,
            "incumbent baseline",
        )
        .await
        {
            Ok(()) if !progress.borrow().finished => incumbents_ready += 1,
            Ok(()) => {
                invalid_reasons.push("an incumbent finished before aggressor injection".to_string())
            }
            Err(error) => invalid_reasons.push(error),
        }
    }

    if incumbents_ready != config.incumbents {
        let (incumbents, drain_errors) = drain_tasks(&mut incumbent_handles, true).await;
        invalid_reasons.extend(drain_errors);
        return invalid_report(
            repeat,
            config,
            warmup_valid,
            incumbents_ready,
            false,
            0,
            incumbents,
            None,
            invalid_reasons,
        );
    }

    let injection_at = Instant::now();
    let aggressor_ctx = ctx.clone_inner();
    let (aggressor_progress_tx, mut aggressor_progress_rx) =
        watch::channel(DecodeStreamProgress::default());
    let aggressor_correlation = benchmark_request_correlation(
        &ctx.benchmark_run_id,
        CELL_ID,
        repeat,
        BenchmarkPhase::Measured,
        config.incumbents as usize,
    );
    let enable_thinking = cmd.enable_thinking;
    let reasoning_effort = cmd.reasoning_effort;
    let timeout = cmd.timeout;
    let mut aggressor_handles = RequestTasks::with_capacity(1);
    aggressor_handles.push(tokio::spawn(async move {
        stream_one_observed(
            &aggressor_ctx.client,
            &aggressor_ctx.base_url,
            &aggressor_ctx.model,
            prepared.aggressor,
            aggressor_output_tokens,
            true,
            enable_thinking,
            reasoning_effort,
            timeout,
            aggressor_correlation,
            Some(aggressor_progress_tx),
        )
        .await
    }));

    let progress_deadline = tokio::time::Instant::now() + Duration::from_secs_f64(cmd.timeout);
    let aggressor_first_output_event = match wait_for_output_events(
        &mut aggressor_progress_rx,
        1,
        progress_deadline,
        "aggressor first output event",
    )
    .await
    {
        Ok(()) => aggressor_progress_rx.borrow().first_output_at,
        Err(error) => {
            invalid_reasons.push(error);
            None
        }
    };

    let Some(aggressor_first_output_event) = aggressor_first_output_event else {
        invalid_reasons.push("aggressor produced no observable first output event".to_string());
        let (incumbents, incumbent_errors) = drain_tasks(&mut incumbent_handles, true).await;
        let (mut aggressors, aggressor_errors) = drain_tasks(&mut aggressor_handles, true).await;
        invalid_reasons.extend(incumbent_errors);
        invalid_reasons.extend(aggressor_errors);
        return invalid_report(
            repeat,
            config,
            warmup_valid,
            incumbents_ready,
            false,
            0,
            incumbents,
            aggressors.pop().flatten(),
            invalid_reasons,
        );
    };

    // Snapshot at the observed aggressor first-output-event signal, then require a
    // strictly later event from every incumbent. This proves each stream was
    // still live; zero progress inside the interference window remains a valid
    // starvation measurement rather than an orchestration failure.
    let snapshots: Vec<u32> = progress_receivers
        .iter()
        .map(|progress| progress.borrow().output_events)
        .collect();
    let mut progressed_after = 0_u32;
    for (progress, snapshot) in progress_receivers.iter_mut().zip(snapshots) {
        match wait_for_output_events(
            progress,
            snapshot.saturating_add(1),
            progress_deadline,
            "post-aggressor incumbent progress",
        )
        .await
        {
            Ok(()) => progressed_after += 1,
            Err(error) => invalid_reasons.push(error),
        }
    }

    let should_abort = progressed_after != config.incumbents;
    let (incumbents, incumbent_errors) = drain_tasks(&mut incumbent_handles, should_abort).await;
    let (mut aggressors, aggressor_errors) =
        drain_tasks(&mut aggressor_handles, should_abort).await;
    invalid_reasons.extend(incumbent_errors);
    invalid_reasons.extend(aggressor_errors);
    let aggressor = aggressors.pop().flatten();

    finish_report(
        repeat,
        config,
        warmup_valid,
        incumbents_ready,
        true,
        progressed_after,
        run_origin,
        injection_at,
        aggressor_first_output_event,
        incumbents,
        aggressor,
        invalid_reasons,
    )
}

async fn wait_for_output_events(
    progress: &mut watch::Receiver<DecodeStreamProgress>,
    target: u32,
    deadline: tokio::time::Instant,
    phase: &str,
) -> std::result::Result<(), String> {
    loop {
        let current = *progress.borrow();
        if current.output_events >= target {
            return Ok(());
        }
        if current.finished {
            return Err(format!(
                "{phase} ended after {} output events, before target {target}",
                current.output_events
            ));
        }
        tokio::time::timeout_at(deadline, progress.changed())
            .await
            .map_err(|_| format!("timed out waiting for {phase} target {target}"))?
            .map_err(|_| format!("{phase} progress channel closed before target {target}"))?;
    }
}

async fn drain_tasks(
    tasks: &mut RequestTasks,
    abort: bool,
) -> (Vec<Option<ObservedRequest>>, Vec<String>) {
    if abort {
        for handle in &tasks.handles {
            handle.abort();
        }
    }
    let mut observed = Vec::with_capacity(tasks.handles.len());
    let mut errors = Vec::new();
    while !tasks.handles.is_empty() {
        let handle = AbortOnDropJoin(Some(tasks.handles.remove(0)));
        match handle.join().await {
            Ok(request) => observed.push(Some(request)),
            Err(error) if abort && error.is_cancelled() => observed.push(None),
            Err(error) => {
                errors.push(format!("request task failed: {error}"));
                observed.push(None);
            }
        }
    }
    (observed, errors)
}

#[allow(clippy::too_many_arguments)]
fn finish_report(
    repeat: u32,
    config: &DecodeIsolationConfig,
    warmup_valid: bool,
    incumbents_ready: u32,
    aggressor_first_output_event_observed: bool,
    progressed_after: u32,
    run_origin: Instant,
    injection_at: Instant,
    aggressor_first_output_event: Instant,
    incumbents: Vec<Option<ObservedRequest>>,
    aggressor: Option<ObservedRequest>,
    mut invalid_reasons: Vec<String>,
) -> DecodeIsolationRunReport {
    let incumbent_evidence: Vec<_> = incumbents
        .iter()
        .map(|request| {
            request
                .as_ref()
                .map(|request| {
                    request_evidence(
                        "incumbent",
                        request,
                        config.incumbent_output_tokens as usize,
                    )
                })
                .unwrap_or_else(|| missing_evidence("incumbent"))
        })
        .collect();
    let aggressor_evidence = aggressor
        .as_ref()
        .map(|request| {
            request_evidence(
                "aggressor",
                request,
                config.aggressor_output_tokens as usize,
            )
        })
        .unwrap_or_else(|| missing_evidence("aggressor"));
    let all_incumbents_valid = incumbent_evidence.len() == config.incumbents as usize
        && incumbent_evidence
            .iter()
            .all(|evidence| evidence.contract_valid);
    let aggressor_valid = aggressor_evidence.contract_valid;
    let timestamp_progressed_after = incumbents
        .iter()
        .filter_map(Option::as_ref)
        .filter(|request| {
            request
                .output_event_times
                .iter()
                .any(|event| *event > aggressor_first_output_event)
        })
        .count() as u32;
    let orchestration_valid = incumbents_ready == config.incumbents
        && aggressor_first_output_event_observed
        && progressed_after == config.incumbents
        && timestamp_progressed_after == config.incumbents;
    if !all_incumbents_valid {
        invalid_reasons.push("one or more incumbent wire/usage contracts were invalid".to_string());
    }
    if !aggressor_valid {
        invalid_reasons.push("aggressor wire/usage contract was invalid".to_string());
    }
    if !orchestration_valid {
        invalid_reasons.push("decode-isolation orchestration contract was invalid".to_string());
    }
    if progressed_after == config.incumbents && timestamp_progressed_after != config.incumbents {
        invalid_reasons.push(
            "incumbent progress signal was not backed by a post-aggressor event timestamp"
                .to_string(),
        );
    }
    invalid_reasons.sort();
    invalid_reasons.dedup();
    let all_valid = warmup_valid
        && all_incumbents_valid
        && aggressor_valid
        && orchestration_valid
        && invalid_reasons.is_empty();

    let (metrics, aggressor_time_to_first_output_event_ms) = if all_valid {
        let timelines: Vec<_> = incumbents
            .iter()
            .filter_map(Option::as_ref)
            .map(|request| DecodeEventTimeline {
                output_event_ms: request
                    .output_event_times
                    .iter()
                    .map(|&event| elapsed_ms(run_origin, event))
                    .collect(),
            })
            .collect();
        match analyze_decode_isolation(
            &timelines,
            elapsed_ms(run_origin, injection_at),
            elapsed_ms(run_origin, aggressor_first_output_event),
        ) {
            Ok(metrics) => (
                Some(metrics),
                aggressor
                    .as_ref()
                    .map(|request| elapsed_ms(request.started_at, aggressor_first_output_event)),
            ),
            Err(error) => {
                invalid_reasons.push(format!("invalid event timeline: {error}"));
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let all_valid =
        all_valid && metrics.is_some() && aggressor_time_to_first_output_event_ms.is_some();
    DecodeIsolationRunReport {
        repeat,
        orchestration: DecodeIsolationOrchestrationEvidence {
            incumbents_expected: config.incumbents,
            incumbents_ready_before_injection: incumbents_ready,
            all_incumbents_ready_before_injection: incumbents_ready == config.incumbents,
            aggressor_first_output_event_observed,
            incumbents_progressed_after_aggressor_first_output_event: timestamp_progressed_after,
            every_incumbent_progressed_after_aggressor_first_output_event:
                timestamp_progressed_after == config.incumbents
                    && progressed_after == config.incumbents,
            all_tasks_drained: true,
        },
        validity: DecodeIsolationEvidenceValidity {
            warmup_valid,
            all_incumbents_valid,
            aggressor_valid,
            orchestration_valid,
            all_valid,
            invalid_reasons,
        },
        metrics,
        aggressor_time_to_first_output_event_ms,
        incumbents: incumbent_evidence,
        aggressor: aggressor_evidence,
    }
}

#[allow(clippy::too_many_arguments)]
fn invalid_report(
    repeat: u32,
    config: &DecodeIsolationConfig,
    warmup_valid: bool,
    incumbents_ready: u32,
    aggressor_first_output_event_observed: bool,
    progressed_after: u32,
    incumbents: Vec<Option<ObservedRequest>>,
    aggressor: Option<ObservedRequest>,
    invalid_reasons: Vec<String>,
) -> DecodeIsolationRunReport {
    let now = Instant::now();
    finish_report(
        repeat,
        config,
        warmup_valid,
        incumbents_ready,
        aggressor_first_output_event_observed,
        progressed_after,
        now,
        now,
        now,
        incumbents,
        aggressor,
        invalid_reasons,
    )
}

fn record_contract_valid(
    record: &ferrum_bench_core::RequestRecord,
    expected_output_tokens: usize,
) -> bool {
    record.success
        && record.output_token_count_source == OutputTokenCountSource::Usage
        && usize::try_from(record.output_tokens).ok() == Some(expected_output_tokens)
        && output_event_timing_valid(record)
}

fn output_event_timing_valid(record: &ferrum_bench_core::RequestRecord) -> bool {
    let evidence = &record.itl_evidence;
    evidence.source == ItlEvidenceSource::SseDeltaEvents
        && evidence.output_events > 0
        && evidence.observed_intervals == evidence.output_events.saturating_sub(1)
        && evidence.transport_coalesced_output_chunks == 0
}

fn request_evidence(
    role: &str,
    request: &ObservedRequest,
    expected_output_tokens: usize,
) -> DecodeIsolationRequestEvidence {
    let output_event_timing_valid = output_event_timing_valid(&request.record);
    DecodeIsolationRequestEvidence {
        role: role.to_string(),
        server_request_id: request.record.server_request_id.clone(),
        success: request.record.success,
        contract_valid: record_contract_valid(&request.record, expected_output_tokens),
        input_tokens: request.record.input_tokens,
        usage_output_tokens: (request.record.output_token_count_source
            == OutputTokenCountSource::Usage)
            .then_some(request.record.output_tokens),
        output_token_count_source: request
            .record
            .output_token_count_source
            .as_str()
            .to_string(),
        observable_output_events: request.record.itl_evidence.output_events,
        observable_output_intervals: request.record.itl_evidence.observed_intervals,
        transport_coalesced_output_chunks: request
            .record
            .itl_evidence
            .transport_coalesced_output_chunks,
        output_event_timing_valid,
        quality_issues: request.record.quality_issues.clone(),
    }
}

fn missing_evidence(role: &str) -> DecodeIsolationRequestEvidence {
    let mut quality_issues = QualityIssueCounts::default();
    quality_issues.malformed_stream = 1;
    DecodeIsolationRequestEvidence {
        role: role.to_string(),
        server_request_id: None,
        success: false,
        contract_valid: false,
        input_tokens: 0,
        usage_output_tokens: None,
        output_token_count_source: "none".to_string(),
        observable_output_events: 0,
        observable_output_intervals: 0,
        transport_coalesced_output_chunks: 0,
        output_event_timing_valid: false,
        quality_issues,
    }
}

fn elapsed_ms(origin: Instant, event: Instant) -> f64 {
    event.saturating_duration_since(origin).as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, Bytes};
    use axum::extract::{Json, State};
    use axum::http::{HeaderMap, Response, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use ferrum_bench_core::decode_isolation::{
        DecodeIsolationCapabilities, DecodeIsolationErrorPolicy, DecodeIsolationScenarioContract,
        DecodeIsolationWindowEnd,
    };
    use ferrum_bench_core::{RequestItlEvidence, RequestRecord, BENCHMARK_REQUEST_INDEX_HEADER};
    use serde_json::Value;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum MockMode {
        Normal,
        EventUsageMismatch,
        Starvation,
        EarlyFinish,
        InvalidUsage,
        InvalidWarmup,
        HangAggressor,
    }

    struct MockState {
        mode: MockMode,
        incumbents: usize,
        baseline: usize,
        aggressor_started: AtomicBool,
        aggressor_first: AtomicBool,
        active_streams: AtomicUsize,
        events: Mutex<Vec<String>>,
    }

    struct ActiveStream(Arc<MockState>);

    impl Drop for ActiveStream {
        fn drop(&mut self) {
            self.0.active_streams.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct MockServer {
        base_url: String,
        state: Arc<MockState>,
        task: JoinHandle<()>,
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn start_mock(mode: MockMode, incumbents: usize, baseline: usize) -> MockServer {
        let state = Arc::new(MockState {
            mode,
            incumbents,
            baseline,
            aggressor_started: AtomicBool::new(false),
            aggressor_first: AtomicBool::new(false),
            active_streams: AtomicUsize::new(0),
            events: Mutex::new(vec![]),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_completion))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        MockServer {
            base_url: format!("http://{address}"),
            state,
            task,
        }
    }

    async fn mock_completion(
        State(state): State<Arc<MockState>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        let request_index = headers
            .get(BENCHMARK_REQUEST_INDEX_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap();
        let phase = headers
            .get("x-ferrum-benchmark-phase")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let max_tokens = body["max_tokens"].as_u64().unwrap() as usize;
        let (tx, rx) = mpsc::channel::<std::result::Result<Bytes, Infallible>>(2);
        state.active_streams.fetch_add(1, Ordering::SeqCst);
        tokio::spawn(produce_stream(state, request_index, phase, max_tokens, tx));
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(ReceiverStream::new(rx)))
            .unwrap()
    }

    async fn produce_stream(
        state: Arc<MockState>,
        request_index: usize,
        phase: String,
        max_tokens: usize,
        tx: mpsc::Sender<std::result::Result<Bytes, Infallible>>,
    ) {
        let _active = ActiveStream(state.clone());
        let invalid_warmup = state.mode == MockMode::InvalidWarmup && phase == "warmup";
        let is_aggressor = phase == "measured" && request_index == state.incumbents;
        if is_aggressor {
            state.aggressor_started.store(true, Ordering::SeqCst);
            state.events.lock().unwrap().push("aggressor_start".into());
            if state.mode == MockMode::HangAggressor {
                loop {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    if tx
                        .send(Ok(Bytes::from_static(b": keepalive\n\n")))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(30)).await;
            if !send_token(&tx, request_index, 0).await {
                return;
            }
            state.aggressor_first.store(true, Ordering::SeqCst);
            state.events.lock().unwrap().push("aggressor_first".into());
            send_end(&tx, request_index, 1).await;
            return;
        }

        let baseline = state.baseline.min(max_tokens);
        for token in 0..baseline {
            if !send_token(&tx, request_index, token).await {
                return;
            }
            state
                .events
                .lock()
                .unwrap()
                .push(format!("incumbent_{request_index}_baseline"));
            tokio::time::sleep(Duration::from_millis(4)).await;
        }
        if state.mode == MockMode::EarlyFinish && phase == "measured" {
            send_end(&tx, request_index, baseline).await;
            return;
        }
        if phase == "measured" {
            if !wait_flag(&state.aggressor_started, &tx).await {
                return;
            }
            if state.mode != MockMode::Starvation {
                if !send_token(&tx, request_index, baseline).await {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            if !wait_flag(&state.aggressor_first, &tx).await {
                return;
            }
            // Keep this after the client can observe the aggressor signal, so
            // the post-signal snapshot has a deterministic later event.
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let start = if phase == "measured" && state.mode != MockMode::Starvation {
            baseline + 1
        } else {
            baseline
        };
        for token in start..max_tokens {
            if state.mode == MockMode::EventUsageMismatch && token + 1 == max_tokens {
                continue;
            }
            if !send_token(&tx, request_index, token).await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(4)).await;
        }
        let usage = if state.mode == MockMode::InvalidUsage || invalid_warmup {
            max_tokens.saturating_sub(1)
        } else {
            max_tokens
        };
        send_end(&tx, request_index, usage).await;
    }

    async fn wait_flag(
        flag: &AtomicBool,
        tx: &mpsc::Sender<std::result::Result<Bytes, Infallible>>,
    ) -> bool {
        while !flag.load(Ordering::SeqCst) {
            if tx.is_closed() {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        true
    }

    async fn send_token(
        tx: &mpsc::Sender<std::result::Result<Bytes, Infallible>>,
        request_index: usize,
        token: usize,
    ) -> bool {
        let event = format!(
            "data: {{\"id\":\"chatcmpl-{request_index}\",\"choices\":[{{\"delta\":{{\"content\":\"t{token}\"}}}}]}}\n\n"
        );
        tx.send(Ok(Bytes::from(event))).await.is_ok()
    }

    async fn send_end(
        tx: &mpsc::Sender<std::result::Result<Bytes, Infallible>>,
        request_index: usize,
        usage: usize,
    ) {
        let usage = format!(
            "data: {{\"id\":\"chatcmpl-{request_index}\",\"choices\":[],\"usage\":{{\"completion_tokens\":{usage}}}}}\n\n"
        );
        let _ = tx.send(Ok(Bytes::from(usage))).await;
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
    }

    fn test_config() -> DecodeIsolationConfig {
        DecodeIsolationConfig {
            incumbents: 1,
            incumbents_source: "test".to_string(),
            incumbent_input_tokens: 8,
            incumbent_output_tokens: 6,
            aggressor_input_tokens: 385,
            aggressor_input_source: "test".to_string(),
            aggressor_scheduled_chunks: 4,
            aggressor_output_tokens: 1,
            aggregate_kv_budget_tokens: 1024,
            aggregate_kv_budget_blocks: 64,
            estimated_unrounded_aggregate_kv_tokens: 400,
            estimated_aggregate_kv_tokens: 416,
            estimated_aggregate_kv_blocks: 26,
            capabilities: DecodeIsolationCapabilities {
                effective_max_concurrent: 3,
                maximum_scheduled_tokens: 128,
                max_model_length: 512,
                kv_capacity_tokens: 1152,
                selected_kv_capacity_tokens: Some(512),
                kv_block_size_tokens: 16,
                kv_block_size_source: "test".to_string(),
            },
            contract: DecodeIsolationScenarioContract {
                baseline_output_events_per_incumbent: 2,
                fixed_output_budget: true,
                injection_requires_all_incumbents_ready: true,
                interference_window_end: DecodeIsolationWindowEnd::AggressorFirstOutputEvent,
                post_aggressor_observable_progress_required_per_incumbent: true,
                minimum_aggressor_scheduled_chunks: 4,
                kv_capacity_headroom_numerator: 9,
                kv_capacity_headroom_denominator: 10,
                invalid_evidence_policy: DecodeIsolationErrorPolicy::EmitDiagnostics,
                warmup_failure_always_fatal: true,
                measured_error_rate_limit: None,
            },
        }
    }

    fn prepared(warmups: usize) -> PreparedRun {
        let prompt = || PromptCase {
            text: "test prompt".to_string(),
            input_tokens: 8,
            sha256: "test".to_string(),
        };
        PreparedRun {
            repeat: 0,
            warmups: (0..warmups).map(|_| prompt()).collect(),
            incumbents: vec![prompt()],
            aggressor: prompt(),
        }
    }

    fn context(server: &MockServer) -> RunContext {
        RunContext {
            client: Arc::new(reqwest::Client::new()),
            base_url: Arc::new(server.base_url.clone()),
            model: Arc::new("test-model".to_string()),
            max_out: 6,
            ignore_eos: true,
            enable_thinking: None,
            reasoning_effort: None,
            timeout_s: 1.0,
            benchmark_run_id: Arc::new("test-run".to_string()),
        }
    }

    fn command(timeout: f64) -> BenchServeCommand {
        let mut cmd = super::super::super::tests::test_command();
        cmd.timeout = timeout;
        cmd.random_output_len = 6;
        cmd.decode_isolation.decode_isolation_baseline_events = 2;
        cmd
    }

    fn request_record(
        usage_tokens: u32,
        output_events: u32,
        coalesced_chunks: u32,
    ) -> RequestRecord {
        RequestRecord {
            benchmark_correlation: None,
            server_request_id: Some("chatcmpl-test".to_string()),
            success: true,
            ttft_ms: 1.0,
            e2e_ms: 2.0,
            input_tokens: 8,
            output_tokens: usage_tokens,
            output_token_count_source: OutputTokenCountSource::Usage,
            itl_evidence: RequestItlEvidence::sse(
                true,
                output_events,
                Some(usage_tokens),
                output_events.saturating_sub(1),
                coalesced_chunks,
            ),
            quality_issues: QualityIssueCounts::default(),
            itl_ms: vec![1.0; output_events.saturating_sub(1) as usize],
        }
    }

    #[test]
    fn output_event_contract_does_not_assume_one_text_event_per_token() {
        let fewer_text_events = request_record(128, 124, 0);
        assert!(record_contract_valid(&fewer_text_events, 128));

        let wrong_usage = request_record(127, 124, 0);
        assert!(!record_contract_valid(&wrong_usage, 128));

        let coalesced = request_record(128, 124, 1);
        assert!(!record_contract_valid(&coalesced, 128));

        let no_observable_output = request_record(1, 0, 0);
        assert!(!record_contract_valid(&no_observable_output, 1));
    }

    #[test]
    fn missing_usage_is_not_reported_as_usage_tokens() {
        let mut record = request_record(4, 4, 0);
        record.output_token_count_source = OutputTokenCountSource::StreamChunks;
        record.itl_evidence.usage_output_tokens = None;
        let request = ObservedRequest {
            record,
            started_at: Instant::now(),
            output_event_times: vec![Instant::now(); 4],
        };
        let evidence = request_evidence("incumbent", &request, 4);
        assert_eq!(evidence.usage_output_tokens, None);
        assert!(!evidence.contract_valid);
    }

    #[tokio::test]
    async fn progress_wait_distinguishes_ready_finished_and_timeout() {
        let (tx, mut rx) = watch::channel(DecodeStreamProgress::default());
        tx.send_replace(DecodeStreamProgress {
            output_events: 4,
            finished: false,
            first_output_at: Some(Instant::now()),
            last_output_at: Some(Instant::now()),
        });
        wait_for_output_events(&mut rx, 4, tokio::time::Instant::now(), "test")
            .await
            .unwrap();

        tx.send_replace(DecodeStreamProgress {
            output_events: 1,
            finished: true,
            first_output_at: Some(Instant::now()),
            last_output_at: Some(Instant::now()),
        });
        assert!(
            wait_for_output_events(&mut rx, 4, tokio::time::Instant::now(), "test")
                .await
                .unwrap_err()
                .contains("ended")
        );
    }

    #[tokio::test]
    async fn injects_only_after_baseline_and_observes_post_signal_progress() {
        let server = start_mock(MockMode::Normal, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(0),
        )
        .await;
        assert!(report.validity.all_valid, "{:?}", report.validity);
        assert!(report.metrics.is_some());
        assert!(
            report
                .orchestration
                .every_incumbent_progressed_after_aggressor_first_output_event
        );
        let events = server.state.events.lock().unwrap();
        let baseline = events
            .iter()
            .rposition(|event| event.contains("baseline"))
            .unwrap();
        let aggressor = events
            .iter()
            .position(|event| event == "aggressor_start")
            .unwrap();
        assert!(baseline < aggressor);
    }

    #[tokio::test]
    async fn event_usage_mismatch_preserves_output_event_gap_evidence() {
        let server = start_mock(MockMode::EventUsageMismatch, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(0),
        )
        .await;
        assert!(report.validity.all_valid, "{:?}", report.validity);
        assert!(report.metrics.is_some());
        assert_eq!(report.incumbents[0].usage_output_tokens, Some(6));
        assert_eq!(report.incumbents[0].observable_output_events, 5);
        assert!(report.incumbents[0].output_event_timing_valid);
    }

    #[tokio::test]
    async fn event_usage_mismatch_does_not_invalidate_warmup() {
        let server = start_mock(MockMode::EventUsageMismatch, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(1),
        )
        .await;
        assert!(report.validity.warmup_valid, "{:?}", report.validity);
        assert!(report.validity.all_valid, "{:?}", report.validity);
    }

    #[tokio::test]
    async fn zero_decode_progress_is_valid_starvation_evidence() {
        let server = start_mock(MockMode::Starvation, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(0),
        )
        .await;
        assert!(report.validity.all_valid, "{:?}", report.validity);
        assert_eq!(report.metrics.unwrap().observable_output_progress_events, 0);
    }

    #[tokio::test]
    async fn early_incumbent_finish_invalidates_numeric_evidence() {
        let server = start_mock(MockMode::EarlyFinish, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(0),
        )
        .await;
        assert!(!report.validity.all_valid);
        assert!(report.metrics.is_none());
        assert!(report.aggressor_time_to_first_output_event_ms.is_none());
        assert!(report.orchestration.all_tasks_drained);
    }

    #[tokio::test]
    async fn invalid_usage_invalidates_numeric_evidence() {
        let server = start_mock(MockMode::InvalidUsage, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(0),
        )
        .await;
        assert!(!report.validity.all_valid);
        assert!(report.metrics.is_none());
        assert!(!report.incumbents[0].contract_valid);
    }

    #[tokio::test]
    async fn invalid_warmup_invalidates_otherwise_valid_measurement() {
        let server = start_mock(MockMode::InvalidWarmup, 1, 2).await;
        let report = run_once(
            &command(1.0),
            &context(&server),
            &test_config(),
            prepared(1),
        )
        .await;
        assert!(!report.validity.warmup_valid);
        assert!(report.metrics.is_none());
        assert!(report.aggressor_time_to_first_output_event_ms.is_none());
    }

    #[tokio::test]
    async fn first_output_event_timeout_aborts_and_drains_all_streams() {
        let server = start_mock(MockMode::HangAggressor, 1, 2).await;
        let report = run_once(
            &command(0.08),
            &context(&server),
            &test_config(),
            prepared(0),
        )
        .await;
        assert!(!report.validity.all_valid);
        assert!(report.metrics.is_none());
        assert!(report.orchestration.all_tasks_drained);
        tokio::time::timeout(Duration::from_millis(250), async {
            while server.state.active_streams.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("aborted HTTP bodies must close mock streams");
    }

    #[tokio::test]
    async fn cancelling_run_future_aborts_owned_request_tasks() {
        let server = start_mock(MockMode::HangAggressor, 1, 2).await;
        let cmd = command(10.0);
        let ctx = context(&server);
        let config = test_config();
        let run = tokio::spawn(async move { run_once(&cmd, &ctx, &config, prepared(0)).await });
        tokio::time::timeout(Duration::from_millis(250), async {
            while !server.state.aggressor_started.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("aggressor must start before cancellation");
        run.abort();
        let _ = run.await;
        tokio::time::timeout(Duration::from_millis(250), async {
            while server.state.active_streams.load(Ordering::SeqCst) != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("dropping run future must abort every owned HTTP task");
    }
}
