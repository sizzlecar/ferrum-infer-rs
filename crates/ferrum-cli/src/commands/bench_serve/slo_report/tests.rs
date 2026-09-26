use super::*;
use ferrum_types::{SloAttainmentTargets, SloClientVisibleConfig, SloLatencyBudgets};
use std::num::NonZeroU64;

fn client_config() -> SloClientVisibleConfig {
    SloClientVisibleConfig {
        latency: SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(10_000).unwrap(),
            tpot_ms: NonZeroU64::new(10_000).unwrap(),
            itl_ms: NonZeroU64::new(10_000).unwrap(),
        },
        attainment: SloAttainmentTargets::default(),
    }
}

fn loaded() -> LoadedSloConfig {
    LoadedSloConfig {
        evaluation: SloEvaluationConfig::from_client_visible(&client_config()).unwrap(),
        source_sha256: "test-config".into(),
    }
}

fn prompt(index: usize) -> PromptCase {
    PromptCase {
        text: format!("request {index}"),
        input_tokens: 2,
        sha256: sha256_hex(format!("request {index}").as_bytes()),
        output_budget: None,
    }
}

fn context(base_url: String) -> RunContext {
    RunContext {
        client: Arc::new(reqwest::Client::new()),
        base_url: Arc::new(base_url),
        model: Arc::new("test".into()),
        max_out: 3,
        ignore_eos: false,
        enable_thinking: None,
        reasoning_effort: None,
        sampling: HttpRequestSampling::default(),
        timeout_s: 5.0,
        benchmark_run_id: Arc::new("slo-http-test".into()),
        capture_slo: true,
    }
}

async fn server(status: u16, body: &str) -> (String, tokio::task::JoinHandle<()>) {
    use axum::{
        http::{header, StatusCode},
        routing::post,
        Router,
    };
    let body = body.to_owned();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let body = body.clone();
            async move {
                (
                    StatusCode::from_u16(status).unwrap(),
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    body,
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, task)
}

const SUCCESS: &str = concat!(
    "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
    "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
    "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
    "data: {\"id\":\"test\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

#[test]
fn last_visible_timestamp_does_not_reuse_terminal_time() {
    let started_at = Instant::now();
    let mut state = StreamState::for_prompt(started_at, 2, "test".into(), None);
    state
        .handle_payload(r#"{"choices":[{"delta":{"content":"a"}}]}"#)
        .unwrap();
    state
        .handle_payload(r#"{"choices":[{"delta":{"content":"b"}}]}"#)
        .unwrap();
    state.usage_completion_tokens = Some(2);
    state.done_count = 1;
    let mut observed = state.finish_observed();
    observed.record.ttft_ms = 1.0;
    observed.record.e2e_ms = 1000.0;
    observed.record.itl_ms = vec![10.0];
    observed.output_event_times = vec![
        started_at + Duration::from_millis(1),
        started_at + Duration::from_millis(11),
    ];
    let collected = CollectedRequest::observed(observed, true);
    assert_eq!(collected.record.tpot_ms(), Some(999.0));
    let evidence = collected.evidence.unwrap();
    assert_eq!(evidence.last_visible_ms, Some(11.0));
    assert_eq!(evidence.terminal_ms, Some(1000.0));
    assert_eq!(evidence.admission, AdmissionEvidence::Accepted);
    let report = evaluate_slo(&loaded().evaluation, &[evidence], 2.0).unwrap();
    assert_eq!(report.requests[0].tpot.observed_ms, Some(10.0));
}

#[tokio::test]
async fn http_and_stream_evidence_distinguish_acceptance_rejection_and_unknown() {
    let late_error = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"late failure\"}}\n\n",
    );
    for (status, body, admission, success) in [
        (200, SUCCESS, AdmissionEvidence::Accepted, true),
        (200, late_error, AdmissionEvidence::Accepted, false),
        (
            200,
            "data: malformed\n\n",
            AdmissionEvidence::Unknown,
            false,
        ),
        (429, "rate limited", AdmissionEvidence::Rejected, false),
        (500, "internal error", AdmissionEvidence::Unknown, false),
    ] {
        let (url, task) = server(status, body).await;
        let ctx = context(url);
        let run = run_closed_loop(&ctx, vec![prompt(0)], 0, 1, "status", 0).await;
        task.abort();
        assert_eq!(run.records[0].success, success);
        assert_eq!(run.evidence[0].admission, admission);
        assert_eq!(
            run.evidence[0].outcome == RequestOutcome::Rejected,
            status == 429
        );
        let report = evaluate_run(&loaded(), 0, &run).unwrap();
        assert_eq!(report.evaluation.outcomes.offered, 1);
        assert_eq!(
            report.evaluation.admissions.accepted,
            u64::from(admission == AdmissionEvidence::Accepted)
        );
        if !success {
            assert_ne!(
                report.evaluation.latency_and_outcome_status,
                SloStatus::Pass
            );
        }
    }
}

#[tokio::test]
async fn both_standard_runners_keep_request_order_and_open_arrival_schedule() {
    for open in [false, true] {
        let (url, task) = server(200, SUCCESS).await;
        let ctx = context(url);
        let prompts = (0..4).map(prompt).collect();
        let run = if open {
            run_open_loop(&ctx, prompts, 1, 1000.0, "arrivals", 0, Some(42)).await
        } else {
            run_closed_loop(&ctx, prompts, 1, 2, "arrivals", 0).await
        };
        task.abort();
        assert_eq!(run.warmup.completed, 1);
        assert_eq!(run.evidence.len(), 3);
        assert_eq!(run.arrivals.len(), 3);
        let schedule = open_loop_arrival_schedule(1000.0, 3, Some(42), 0);
        for (i, (record, arrival)) in run.records.iter().zip(&run.arrivals).enumerate() {
            assert_eq!(
                record.benchmark_correlation.as_ref().unwrap().request_index as usize,
                i
            );
            assert!(arrival.request_started_ms.unwrap() >= arrival.dispatched_ms.unwrap());
            if open {
                assert_eq!(arrival.scheduled_arrival_ms, Some(schedule[i] * 1000.0));
                assert!(arrival.dispatched_ms.unwrap() >= arrival.scheduled_arrival_ms.unwrap());
                assert!(arrival.client_dispatch_backlog.is_some());
            } else {
                assert_eq!(arrival.scheduled_arrival_ms, None);
                assert_eq!(arrival.client_dispatch_backlog, None);
            }
        }
        let report = evaluate_run(&loaded(), 0, &run).unwrap();
        assert_eq!(report.evaluation.outcomes.completed, 3);
        assert!(report.observed_request_start_rate_rps.unwrap().is_finite());
        assert_eq!(report.evaluation.admissions.unknown, 0);
    }
}

#[tokio::test]
async fn panic_retains_an_unknown_timing_record_including_arrival_coverage() {
    let correlation =
        benchmark_request_correlation("panic", "cell", 0, BenchmarkPhase::Measured, 0);
    let handle = tokio::spawn(async move {
        panic!("fixture request panic");
        #[allow(unreachable_code)]
        CollectedRequest::from(join_failed_record(
            2,
            benchmark_request_correlation("panic", "cell", 0, BenchmarkPhase::Measured, 0),
        ))
    });
    let start = Instant::now();
    let collected = collect_measured_handles(vec![(2, correlation, handle)], true).await;
    let run = finish_run(
        collected,
        1,
        1.0,
        WarmupSummary::default(),
        start,
        vec![RequestArrivalEvidence::default()],
    );
    assert_eq!(run.records[0].quality_issues.panic, 1);
    assert_eq!(run.evidence[0].first_visible_ms, None);
    assert_eq!(run.evidence[0].last_visible_ms, None);
    assert_eq!(run.evidence[0].admission, AdmissionEvidence::Unknown);
    let report = evaluate_run(&loaded(), 0, &run).unwrap();
    assert_eq!(report.evaluation.outcomes.failed, 1);
    assert_eq!(report.arrivals[0].request_started_ms, None);
}

#[test]
fn explicit_config_is_required_and_cannot_overwrite_legacy_or_input_paths() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("slo.json");
    let output_path = directory.path().join("evidence.jsonl");
    std::fs::write(&config_path, serde_json::to_vec(&client_config()).unwrap()).unwrap();
    let mut cmd = super::super::tests::test_command();
    cmd.slo.slo_client_config = Some(config_path.clone());
    assert!(load(&cmd).is_err());
    cmd.slo.slo_out = Some(output_path.clone());
    assert!(load(&cmd).unwrap().is_some());
    cmd.out = Some(output_path);
    assert!(load(&cmd).is_err());
    cmd.out = Some(config_path.clone());
    assert!(load(&cmd).is_err());
    cmd.out = None;
    cmd.slo.slo_out = Some(config_path);
    assert!(load(&cmd).is_err());
    cmd.slo.slo_out = Some(directory.path().join("isolation.jsonl"));
    cmd.scenario = BenchServeWorkload::DecodeIsolation;
    assert!(load(&cmd).is_err());
}

#[test]
fn missing_output_parent_is_rejected_before_running_requests() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("slo.json");
    let bytes = serde_json::to_vec(&client_config()).unwrap();
    std::fs::write(&config_path, &bytes).unwrap();
    let mut cmd = super::super::tests::test_command();
    cmd.slo.slo_client_config = Some(config_path.clone());
    cmd.slo.slo_out = Some(directory.path().join("missing/../slo.json"));
    assert!(load(&cmd).is_err());
    assert_eq!(std::fs::read(config_path).unwrap(), bytes);
}

#[cfg(unix)]
#[test]
fn hard_link_to_client_config_is_not_an_output_file() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("slo.json");
    let bytes = serde_json::to_vec(&client_config()).unwrap();
    std::fs::write(&config_path, &bytes).unwrap();
    let output = directory.path().join("alias.jsonl");
    std::fs::hard_link(&config_path, &output).unwrap();
    let mut cmd = super::super::tests::test_command();
    cmd.slo.slo_client_config = Some(config_path.clone());
    cmd.slo.slo_out = Some(output);
    assert!(load(&cmd).is_err());
    assert_eq!(std::fs::read(config_path).unwrap(), bytes);
}

#[test]
fn slo_capture_and_config_identity_are_part_of_the_existing_environment_hash() {
    let mut env = Env::default();
    let previous = env.hash();
    let mut config = loaded();
    config.annotate_env(&mut env);
    assert_ne!(env.hash(), previous);
    assert!(env
        .runtime_config
        .entries
        .iter()
        .any(|entry| entry.key == "bench_slo_capture" && entry.effective_value == "true"));
    let enabled = env.hash();
    config.source_sha256 = "different-client-contract".into();
    config.annotate_env(&mut env);
    assert_ne!(env.hash(), enabled);
}

#[tokio::test]
async fn sidecar_contains_legacy_identity_and_raw_new_evidence_before_failure_policy() {
    let (url, task) = server(200, SUCCESS).await;
    let ctx = context(url);
    let run = run_closed_loop(&ctx, vec![prompt(0)], 0, 1, "sidecar", 0).await;
    task.abort();
    let mut repeat = evaluate_run(&loaded(), 0, &run).unwrap();
    repeat.evaluation.latency_and_outcome_status = SloStatus::Unknown;
    let legacy = compute_metrics(
        "test".into(),
        "cpu".into(),
        Scenario::ClosedLoop,
        Some(1),
        None,
        2,
        3,
        0,
        Slo::unbounded(),
        vec![run.legacy],
        Env::default(),
    );
    let report = SloCellReport {
        schema_version: 1,
        config_sha256: "fixture".into(),
        legacy_benchmark: legacy.clone(),
        repeats: vec![repeat],
    };
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("slo.jsonl");
    let mut cmd = super::super::tests::test_command();
    cmd.slo.slo_out = Some(output.clone());
    cmd.slo.slo_fail_on_violation = true;
    emit(&cmd, &report).unwrap();
    assert!(enforce(&cmd, &[report]).is_err());
    let value: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(output).unwrap().trim()).unwrap();
    assert_eq!(
        value["legacy_benchmark"],
        serde_json::to_value(legacy).unwrap()
    );
    assert!(
        value["repeats"][0]["evaluation"]["request_evidence"][0]["last_visible_ms"].is_number()
    );
    assert_eq!(
        value["repeats"][0]["evaluation"]["admissions"]["accepted"],
        1
    );
    assert_eq!(value["repeats"][0]["arrivals"].as_array().unwrap().len(), 1);
}
