use super::*;
#[path = "tests/session.rs"]
mod session_tests;
use ferrum_bench_core::{
    dataset::{ShareGptSample, ShareGptSelection},
    slo::{SloEvaluationConfig, SloStatus, TpotBoundary},
    slo_comparison::{FixedServerCapacity, FrozenServerIdentity, SharedExecutionIdentity},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[test]
fn capacity_input_bound_applies_to_growth_after_metadata_check() {
    assert!(read_bounded(std::io::repeat(b'x'), 8).is_err());
    assert_eq!(read_bounded(&b"12345678"[..], 8).unwrap(), b"12345678");
}

#[tokio::test]
async fn capacity_server_queue_slow_health_never_serializes_open_arrivals() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server_origin = Instant::now();
    let server = tokio::spawn(async move {
        let mut health_reads = 0;
        let mut requests = 0;
        let mut tasks = tokio::task::JoinSet::new();
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_request(&mut socket).await;
            let is_health = request.starts_with(b"GET /health ");
            let delay = if is_health {
                health_reads += 1;
                if health_reads == 1 {
                    Duration::ZERO
                } else {
                    Duration::from_millis(50)
                }
            } else {
                requests += 1;
                if requests == 1 {
                    Duration::from_millis(120)
                } else {
                    Duration::ZERO
                }
            };
            tasks.spawn(async move {
                tokio::time::sleep(delay).await;
                let (content_type, body) = if is_health {
                    ("application/json", serde_json::json!({"admission":{
                        "runtime_snapshot_available":true,"queue_depth":0,"active_prefill":0,"active_decode":0,
                        "queue_observation":{"schema_version":1,"engine_instance":"http-fixture",
                            "observed_at_ns":server_origin.elapsed().as_nanos() as u64,
                            "waiting_requests":0,"active_prefill_sequences":0,"active_decode_sequences":0,
                            "preempted_requests":0,"oldest_waiting_ingress_age_ns":null,"oldest_unfinished_ingress_age_ns":null}
                    }}).to_string())
                } else {
                    ("text/event-stream", concat!("data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n", "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n", "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n", "data: [DONE]\n\n").to_owned())
                };
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    let run = run_fixed_window_with_queue(
        &context(format!("http://{address}")),
        &planned(vec![0.0, 10.0], 0.03),
        &prompts(),
        0,
        &window(0.03, 0.5),
        1000,
        QueueObservationSource::ServerAdmissionV1,
    )
    .await
    .unwrap();
    server.abort();
    let _ = server.await;
    assert_eq!(run.requests.records.len(), 2);
    assert!(run.requests.records.iter().all(|r| r.success));
    assert!(run.arrivals[1].request_started_ms.unwrap() < run.requests.records[0].e2e_ms);
    assert!(
        run.queue.is_empty(),
        "client lifecycle data cannot replace server values"
    );
    assert!(run.server_queue_attempts[0].response_completed_seconds <= 0.0);
    assert!(
        run.server_queue_attempts
            .last()
            .unwrap()
            .request_started_seconds
            >= run.duration_s
    );
    assert!(run
        .server_queue_attempts
        .iter()
        .any(|a| matches!(a.observation, Err(ServerQueueFailure::Timeout))));
}

fn planned(times: Vec<f64>, send_seconds: f64) -> PlannedCapacityRun {
    let count = times.len();
    PlannedCapacityRun {
        contract_sha256: "a".repeat(64),
        key: CapacityRunKey {
            phase: CapacityPhase::Coarse,
            rate_index: 0,
            repetition: 0,
            replica: 0,
        },
        cell_id: "capacity-test".into(),
        rate_rps: 20.0,
        seed: 1,
        send_seconds,
        scheduled_arrival_ms: times,
        output_token_budgets: vec![2; count],
        workload_sample_indices: vec![0; count],
    }
}

fn window(send_seconds: f64, drain: f64) -> CapacityWindow {
    CapacityWindow {
        send_seconds,
        observe_from_seconds: 0.0,
        maximum_drain_seconds: drain,
        maximum_request_start_lag_ms: 100.0,
        maximum_client_dispatch_backlog: 100,
        maximum_queue_sample_gap_seconds: 0.02,
        maximum_unfinished_requests_slope_per_second: 1000.0,
        maximum_oldest_age_slope_ms_per_second: 1000.0,
    }
}

fn context(base: String) -> RunContext {
    RunContext {
        client: Arc::new(
            reqwest::Client::builder()
                .pool_max_idle_per_host(0)
                .build()
                .unwrap(),
        ),
        base_url: Arc::new(base),
        model: Arc::new("test".into()),
        max_out: 2,
        ignore_eos: true,
        enable_thinking: Some(false),
        reasoning_effort: None,
        sampling: HttpRequestSampling::default(),
        timeout_s: 2.0,
        benchmark_run_id: Arc::new("capacity-http-test".into()),
        capture_slo: true,
    }
}

fn prompts() -> Vec<PromptCase> {
    vec![PromptCase {
        text: "test prompt".into(),
        input_tokens: 2,
        sha256: sha256_hex(b"test prompt"),
        output_budget: Some(2),
    }]
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        let n = socket.read(&mut buffer).await.unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..n]);
        if let Some(header_end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..header_end]).to_ascii_lowercase();
            let length = headers
                .lines()
                .find_map(|line| {
                    line.strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= header_end + 4 + length {
                break;
            }
        }
    }
    bytes
}

async fn serve_once(mut socket: tokio::net::TcpStream, delay: Duration) {
    read_request(&mut socket).await;
    tokio::time::sleep(delay).await;
    let body=concat!("data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n", "data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n", "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n", "data: [DONE]\n\n");
    let response=format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());
    let _ = socket.write_all(response.as_bytes()).await;
}

#[tokio::test]
async fn capacity_fixed_window_waits_until_send_end_after_early_completions() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        serve_once(socket, Duration::ZERO).await;
    });
    let plan = planned(vec![0.0], 0.08);
    let run = run_fixed_window(
        &context(format!("http://{address}")),
        &plan,
        &prompts(),
        0,
        &window(0.08, 0.5),
        1000,
    )
    .await
    .unwrap();
    server.await.unwrap();
    assert!(run.duration_s >= plan.send_seconds);
    assert!(run.requests.records[0].success);
    assert_eq!(run.requests.records[0].output_tokens, 2);
    assert!(run.queue.last().unwrap().at_seconds >= plan.send_seconds);
    assert_eq!(run.queue.last().unwrap().active_requests, 0);
}

#[tokio::test]
async fn capacity_fixed_window_offers_new_work_while_previous_stream_is_open() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (first, _) = listener.accept().await.unwrap();
        let a = tokio::spawn(serve_once(first, Duration::from_millis(120)));
        let (second, _) = listener.accept().await.unwrap();
        serve_once(second, Duration::ZERO).await;
        a.await.unwrap();
    });
    let plan = planned(vec![0.0, 10.0], 0.03);
    let run = run_fixed_window(
        &context(format!("http://{address}")),
        &plan,
        &prompts(),
        0,
        &window(0.03, 0.5),
        1000,
    )
    .await
    .unwrap();
    server.await.unwrap();
    let first_finished =
        run.arrivals[0].request_started_ms.unwrap() + run.requests.records[0].e2e_ms;
    assert!(run.arrivals[1].request_started_ms.unwrap() < first_finished);
    assert!(run
        .queue
        .iter()
        .any(|s| s.active_requests > 0 && s.oldest_request_age_ms > 0.0));
    assert!(run.requests.records.iter().all(|r| r.success));
}

#[tokio::test]
async fn capacity_drain_timeout_retains_partial_sse_evidence_and_failure() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request(&mut socket).await;
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n").await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;
    });
    let plan = planned(vec![0.0], 0.02);
    let run = run_fixed_window(
        &context(format!("http://{address}")),
        &plan,
        &prompts(),
        0,
        &window(0.02, 0.03),
        1000,
    )
    .await
    .unwrap();
    server.abort();
    let _ = server.await;
    assert!(!run.requests.records[0].success);
    assert!(
        run.requests.evidence[0]
            .visible_text
            .as_ref()
            .unwrap()
            .output_events
            > 0
    );
    assert!(run.requests.records[0].quality_issues.request_error_count() > 0);
    assert_eq!(run.requests.evidence.len(), 1);
    assert!(run.duration_s >= plan.send_seconds);
}

#[tokio::test]
async fn capacity_queue_capture_bound_preserves_workload_and_reports_incompleteness() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        serve_once(socket, Duration::ZERO).await;
    });
    let plan = planned(vec![0.0], 0.08);
    let run = run_fixed_window(
        &context(format!("http://{address}")),
        &plan,
        &prompts(),
        0,
        &window(0.08, 0.5),
        2,
    )
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(run.queue.len(), 2);
    assert!(!run.queue_capture_complete);
    assert!(run.requests.records[0].success);
    assert!(run.duration_s >= plan.send_seconds);
}

// Small deterministic HTTP workload, independent of hardware and real capacity
// thresholds. All evidence passed to the evaluator comes from the collector.
fn http_contract() -> CapacityContract {
    let hash = "a".repeat(64);
    let samples = vec![ShareGptSample {
        source_record_index: 0,
        original_id: None,
        phase: BenchmarkPhase::Measured,
        request_index: 0,
        prompt_sha256: prompts()[0].sha256.clone(),
        assistant_sha256: hash.clone(),
        input_tokens: 2,
        reference_output_tokens: 2,
        requested_output_tokens: 2,
    }];
    let selection_sha256 = sha256_hex(&serde_json::to_vec(&samples).unwrap());
    CapacityContract {
        schema_version: 1,
        frozen_unix_ns: unix_ns().unwrap(),
        identity: CapacityIdentity {
            shared: SharedExecutionIdentity {
                hardware_fingerprint_sha256: hash.clone(),
                hardware_label: "HTTP fixture".into(),
                model_content_sha256: hash.clone(),
                weight_precision: "fixture".into(),
                kv_precision: "fixture".into(),
                tokenizer_sha256: hash.clone(),
                chat_template_sha256: hash.clone(),
                client_binary_sha256: hash.clone(),
                client_slo_config_sha256: hash.clone(),
            },
            server: FrozenServerIdentity {
                implementation: "HTTP fixture".into(),
                backend: "fixture".into(),
                request_model_alias: "test".into(),
                binary_sha256: hash.clone(),
                effective_configuration_sha256: hash.clone(),
                numerical_policy: "fixture".into(),
                intentional_differences: vec![],
            },
            capacity: FixedServerCapacity {
                slots: 32,
                context_tokens_per_request: 128,
                batch_tokens: 128,
            },
            ordered_workload_sha256: selection_sha256.clone(),
            dataset_source_sha256: hash,
        },
        rates_rps: vec![60.0],
        coarse_indices: vec![0],
        aa_indices: vec![0],
        repetitions: CapacityRepetitions {
            aa_pairs: 1,
            coarse: 1,
            neighborhood: 1,
            confirmation: 1,
        },
        window: window(0.15, 0.5),
        queue_observation_source: QueueObservationSource::ClientScheduledLifecycle,
        sampling: HttpRequestSampling::default(),
        enable_thinking: Some(false),
        http_connection_mode: "fresh".into(),
        slo: SloEvaluationConfig {
            ttft_ms: 1000.0,
            tpot_ms: 1000.0,
            visible_itl_ms: 1000.0,
            attainment: ferrum_types::SloAttainmentTargets {
                min_accepted_joint_attainment: 1.0,
                min_offered_joint_attainment: Some(1.0),
                max_reject_rate: 0.0,
                max_error_rate: 0.0,
                ..Default::default()
            },
            tpot_boundary: TpotBoundary::LastVisibleOutput,
        },
        seed: 42,
        maximum_requests_per_run: 1024,
        maximum_planned_runs: 4,
        maximum_queue_samples_per_run: 1000,
        workload: ShareGptSelection {
            repeat_index: 0,
            rng_seed: 42,
            selection_sha256,
            samples,
        },
    }
}

#[tokio::test]
async fn capacity_http_evidence_reaches_core_and_missing_arrival_cannot_pass() {
    let search = CapacitySearch::new(http_contract()).unwrap();
    let SearchProgress::Awaiting { runs, .. } = search.progress() else {
        panic!("expected first acquisition block")
    };
    let plan = search.planned_run(&runs[0]).unwrap();
    let count = plan.scheduled_arrival_ms.len();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut responses = tokio::task::JoinSet::new();
        for _ in 0..count {
            let (socket, _) = listener.accept().await.unwrap();
            responses.spawn(serve_once(socket, Duration::ZERO));
        }
        while let Some(result) = responses.join_next().await {
            result.unwrap();
        }
    });
    let started = unix_ns().unwrap();
    let run = run_fixed_window(
        &context(format!("http://{address}")),
        &plan,
        &prompts(),
        0,
        &search.contract().window,
        search.contract().maximum_queue_samples_per_run,
    )
    .await
    .unwrap();
    server.await.unwrap();
    let evidence = assemble_evidence(
        search.contract(),
        &plan,
        run,
        "capacity-http-test".into(),
        started,
        unix_ns().unwrap(),
    )
    .unwrap();
    let report = evaluate_run(search.contract(), &plan, &evidence).unwrap();
    assert_eq!(report.status, SloStatus::Pass, "{:?}", report.issues);
    assert_eq!(report.delivered_requests, count);
    let mut missing = evidence;
    missing.arrivals[0].request_started_ms = None;
    let report = evaluate_run(search.contract(), &plan, &missing).unwrap();
    assert_ne!(report.status, SloStatus::Pass);
    assert!(!report.evidence_complete);
    assert!(!report.arrival_schedule_delivered);
}
