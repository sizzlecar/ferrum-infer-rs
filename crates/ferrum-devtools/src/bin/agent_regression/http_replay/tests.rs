use super::*;
use axum::{
    http::{HeaderMap, StatusCode},
    routing::post,
    Router,
};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Mutex,
};
use tokio::net::TcpListener;

fn body(suffix: usize) -> String {
    format!(" \n{{\"model\":\"original-model\",\"stream\":true,\"messages\":[{{\"role\":\"user\",\"content\":\"task {suffix}\"}}],\"tools\":[{{\"type\":\"function\",\"function\":{{\"name\":\"read\",\"parameters\":{{\"type\":\"object\"}}}}}}],\"temperature\":0.7,\"seed\":17,\"max_tokens\":16,\"stream_options\":{{\"include_usage\":true}}}}\n")
}

fn sse(index: u32) -> String {
    format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        json!({"id":format!("r-{index}"),"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}),
        json!({"id":format!("r-{index}"),"choices":[{"delta":{"content":"measured"},"finish_reason":null}]}),
        json!({"id":format!("r-{index}"),"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":3134,"completion_tokens":7,"total_tokens":3141}})
    )
}

#[tokio::test]
async fn raw_http_replay_preserves_body_mapping_and_rejects_overwrite() {
    for (body_count, concurrency) in [(1, 1), (1, 3), (3, 3)] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let captured = Arc::new(Mutex::new(Vec::new()));
        let seen = captured.clone();
        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, bytes: Bytes| {
                let seen = seen.clone();
                let barrier = barrier.clone();
                async move {
                    let index: u32 = headers[BENCHMARK_REQUEST_INDEX_HEADER]
                        .to_str()
                        .unwrap()
                        .parse()
                        .unwrap();
                    seen.lock().unwrap().push((index, bytes));
                    // A serial implementation cannot finish this response within its timeout.
                    barrier.wait().await;
                    ([("content-type", "text/event-stream")], sse(index))
                }
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let directory = tempfile::tempdir().unwrap();
        let bodies: Vec<_> = (0..body_count).map(body).collect();
        let files: Vec<_> = bodies
            .iter()
            .enumerate()
            .map(|(index, body)| {
                let path = directory.path().join(format!("body-{index}.json"));
                fs::write(&path, body).unwrap();
                path
            })
            .collect();
        let args = Args {
            base_url,
            body_file: files,
            concurrency,
            timeout_secs: 2,
            report_dir: directory.path().join("report"),
        };
        let code = tokio::time::timeout(Duration::from_secs(5), run(&args))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(code, 0);
        let report: Value =
            serde_json::from_slice(&fs::read(args.report_dir.join("report.json")).unwrap())
                .unwrap();
        assert_eq!(report["total_output_tokens"], 7 * concurrency);
        assert_eq!(report["successful_output_tokens"], 7 * concurrency);
        assert_eq!(report["successful_request_count"], concurrency);
        assert_eq!(report["transport_complete"], true);
        assert_eq!(report["visible_text"]["ttft"]["sample_count"], concurrency);
        assert_eq!(report["visible_text"]["itl"]["sample_count"], 0);
        assert!(report["visible_text"]["itl"]["p99_ms"].is_null());
        assert_eq!(
            report["requests"].as_array().unwrap().len(),
            concurrency as usize
        );
        let received = captured.lock().unwrap().clone();
        assert_eq!(received.len(), concurrency as usize);
        for (index, bytes) in received {
            let expected = &bodies[if body_count == 1 { 0 } else { index as usize }];
            assert_eq!(&bytes[..], expected.as_bytes());
            assert_eq!(
                fs::read(args.report_dir.join(format!("request-{index}.body.json"))).unwrap(),
                expected.as_bytes()
            );
            assert_eq!(
                fs::read_to_string(args.report_dir.join(format!("response-{index}.sse"))).unwrap(),
                sse(index)
            );
            let record = &report["requests"][index as usize];
            assert_eq!(
                record["body_sha256"],
                format!("{:x}", Sha256::digest(expected.as_bytes()))
            );
            assert_eq!(record["progress_events"], 1);
            assert_eq!(record["usage"]["completion_tokens"], 7);
            assert!(record["ttft_ns"].is_u64());
            assert_eq!(record["finish_reasons"], json!(["length"]));
        }
        assert!(run(&args).await.is_err());
        assert_eq!(captured.lock().unwrap().len(), concurrency as usize);
        let invalid = Args {
            report_dir: directory.path().join("invalid"),
            concurrency: 3,
            body_file: vec![args.body_file[0].clone(); 2],
            ..args
        };
        assert!(run(&invalid).await.is_err());
        assert!(!invalid.report_dir.exists());
        server.abort();
    }
}

#[test]
fn raw_http_replay_uses_progress_not_role_frames_or_usage_as_tokens() {
    let mut observer = SseObserver::default();
    let mut record = RequestRecord::default();
    observer
        .push(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            10,
            &mut record,
        )
        .unwrap();
    assert_eq!(record.first_sse_ns, Some(10));
    assert!(record.progress_ns.is_empty());
    observer
        .push(
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":",
            20,
            &mut record,
        )
        .unwrap();
    observer
        .push(b"{\"arguments\":\"{}\"}}]}}]}\n\n", 30, &mut record)
        .unwrap();
    observer.push(b"data: {\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":7},\"choices\":[]}\n\ndata: [DONE]\n\n", 40, &mut record).unwrap();
    assert_eq!(record.progress_ns, vec![30]);
    assert!(record
        .visible_text
        .as_ref()
        .unwrap()
        .timestamps_ns
        .is_empty());
    assert_eq!(record.usage.unwrap()["completion_tokens"], 7);
    assert!(record.saw_done);
}

#[tokio::test]
async fn raw_http_replay_retains_http_parse_and_incomplete_stream_failures() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let next = Arc::new(AtomicU32::new(0));
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let index = next.fetch_add(1, Ordering::Relaxed);
            async move {
                match index {
                    0 => (StatusCode::SERVICE_UNAVAILABLE, "backend unavailable"),
                    1 => (StatusCode::OK, "data: broken-json\n\n"),
                    _ => (
                        StatusCode::OK,
                        "data: {\"choices\":[{\"delta\":{\"content\":\"unfinished\"}}]}\n\n",
                    ),
                }
            }
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("body.json");
    fs::write(&path, body(0)).unwrap();
    let args = Args {
        base_url,
        body_file: vec![path],
        concurrency: 3,
        timeout_secs: 2,
        report_dir: directory.path().join("report"),
    };
    assert_eq!(run(&args).await.unwrap(), 1);
    let report: Value =
        serde_json::from_slice(&fs::read(args.report_dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["transport_complete"], false);
    assert_eq!(report["usage_complete"], false);
    assert!(report["total_output_tokens"].is_null());
    let requests = report["requests"].as_array().unwrap();
    assert!(requests
        .iter()
        .all(|r| r["error"].is_string() && r["saw_done"] == false));
    assert!(requests.iter().any(|r| r["http_status"] == 503));
    assert!(requests
        .iter()
        .any(|r| r["error"] == "stream ended without [DONE]"));
    for index in 0..3 {
        assert!(
            !fs::read(args.report_dir.join(format!("response-{index}.sse")))
                .unwrap()
                .is_empty()
        );
    }
    server.abort();
}

#[test]
fn visible_text_excludes_tools_and_other_choices_but_retains_reasoning_text() {
    let mut observer = SseObserver::default();
    let mut record = RequestRecord {
        submitted_ns: 1_000_000,
        ..Default::default()
    };
    let event = |delta: Value| format!("data: {}\n\n", json!({"choices":[{"delta":delta}]}));
    for (at, delta) in [
        (2, json!({"role":"assistant"})),
        (3, json!({"tool_calls":[{"function":{"arguments":"{}"}}]})),
    ] {
        observer
            .push(event(delta).as_bytes(), at * 1_000_000, &mut record)
            .unwrap();
    }
    let unicode = event(json!({"content":"你好"}));
    let split = unicode.find('你').unwrap() + 1;
    observer
        .push(&unicode.as_bytes()[..split], 4_000_000, &mut record)
        .unwrap();
    observer
        .push(&unicode.as_bytes()[split..], 5_000_000, &mut record)
        .unwrap();
    for (at, delta) in [
        (6, json!({"reasoning":"thought"})),
        (7, json!({"reasoning_content":"alias"})),
        (
            8,
            json!({"content":"", "tool_calls":[{"function":{"name":"read"}}]}),
        ),
        (9, json!({"content":"answer", "reasoning":"same event"})),
    ] {
        observer
            .push(event(delta).as_bytes(), at * 1_000_000, &mut record)
            .unwrap();
    }
    observer
        .push(
            b"data: {\"choices\":[{\"delta\":{}},{\"delta\":{\"content\":\"other choice\"}}]}\n\n",
            10_000_000,
            &mut record,
        )
        .unwrap();
    observer
        .push(
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
            11_000_000,
            &mut record,
        )
        .unwrap();
    assert_eq!(
        record.progress_ns,
        vec![3_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000]
    );
    assert_eq!(
        record.visible_text.as_ref().unwrap().timestamps_ns,
        vec![5_000_000, 6_000_000, 7_000_000, 9_000_000]
    );
    let measured = text_timing::measure(&record).unwrap();
    assert_eq!(measured.ttft_ms, Some(4.0));
    assert_eq!(measured.itl_ms, vec![1.0, 1.0, 2.0]);
    assert_eq!(measured.last_visible_tpot_ms, None);
}

#[test]
fn visible_text_preserves_stalls_coalescing_and_usage_event_mismatches() {
    let mut observer = SseObserver::default();
    let mut record = RequestRecord {
        http_status: Some(200),
        visible_text: Some(VisibleTextRecord {
            single_choice_requested: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    observer.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n", 1_000_000, &mut record).unwrap();
    observer.push(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n", 6_000_000, &mut record).unwrap();
    observer
        .push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
            101_000_000,
            &mut record,
        )
        .unwrap();
    observer
        .push(
            b"data: {\"choices\":[],\"usage\":{\"completion_tokens\":17}}\n\ndata: [DONE]\n\n",
            200_000_000,
            &mut record,
        )
        .unwrap();
    record.ended_ns = Some(200_000_000);
    let measured = text_timing::measure(&record).unwrap();
    assert_eq!(measured.transport_coalesced_chunks, 1);
    assert_eq!(measured.itl_ms, vec![0.0, 100.0]);
    assert_eq!(measured.last_visible_tpot_ms, Some(6.25));
    assert_eq!(measured.usage_event_count_mismatch, Some(true));
    let summary = text_timing::summarize(&[record]);
    assert_eq!(summary["itl"]["sample_count"], 2);
    assert_eq!(summary["itl"]["p99_ms"], 99.0);
    assert_eq!(summary["transport_coalesced_request_count"], 1);
    assert_eq!(summary["usage_event_mismatch_request_count"], 1);
    assert_eq!(summary["strict_single_token_timing_measured"], false);
}

#[test]
fn multiple_completion_choices_cannot_use_response_usage_for_first_choice_tpot() {
    for requested in [None, Some(false), Some(true)] {
        let mut record = RequestRecord {
            visible_text: Some(VisibleTextRecord {
                single_choice_requested: requested,
                ..Default::default()
            }),
            usage: Some(json!({"completion_tokens":8})),
            ..Default::default()
        };
        let mut observer = SseObserver::default();
        observer
            .push(
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n",
                10,
                &mut record,
            )
            .unwrap();
        observer
            .push(
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n",
                20,
                &mut record,
            )
            .unwrap();
        assert_eq!(
            text_timing::measure(&record)
                .unwrap()
                .last_visible_tpot_ms
                .is_some(),
            requested == Some(true)
        );
        observer.push(b"data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"another completion\"}}]}\n\n", 30, &mut record).unwrap();
        let measured = text_timing::measure(&record).unwrap();
        assert!(record.visible_text.as_ref().unwrap().multi_choice_observed);
        assert_eq!(measured.last_visible_tpot_ms, None);
        assert_eq!(
            measured.tpot_unavailable_reason,
            Some("single_choice_usage_not_established")
        );
    }
}

#[tokio::test]
async fn failed_response_usage_is_excluded_from_successful_throughput() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new().route("/v1/chat/completions", post(|| async {
        ([("content-type", "text/event-stream")], "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":7}}\n\n")
    }));
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("body.json");
    fs::write(&file, body(0)).unwrap();
    let args = Args {
        base_url,
        body_file: vec![file],
        concurrency: 1,
        timeout_secs: 2,
        report_dir: dir.path().join("report"),
    };
    assert_eq!(run(&args).await.unwrap(), 1);
    let report: Value =
        serde_json::from_slice(&fs::read(args.report_dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["total_output_tokens"], 7);
    assert_eq!(report["successful_output_tokens"], 0);
    assert_eq!(
        report["successful_output_tokens_per_second_including_prefill"],
        0.0
    );
    assert_eq!(report["visible_text"]["failed_request_count"], 1);
    server.abort();
}

#[test]
fn missing_or_invalid_text_evidence_and_failed_requests_remain_distinct() {
    let complete = RequestRecord {
        http_status: Some(200),
        saw_done: true,
        ended_ns: Some(30),
        ..Default::default()
    };
    let mut legacy = serde_json::to_value(&complete).unwrap();
    legacy.as_object_mut().unwrap().remove("visible_text");
    let legacy: RequestRecord = serde_json::from_value(legacy).unwrap();
    assert!(text_timing::measure(&legacy).is_none());
    let mut empty = complete.clone();
    empty.visible_text = Some(Default::default());
    let mut reversed = empty.clone();
    reversed.visible_text.as_mut().unwrap().timestamps_ns = vec![20, 10];
    assert!(!text_timing::measure(&reversed).unwrap().timestamps_valid);
    let mut failed = empty.clone();
    failed.visible_text.as_mut().unwrap().timestamps_ns = vec![10, 20];
    failed.error = Some("incomplete stream".into());
    let summary = text_timing::summarize(&[legacy, empty, reversed, failed]);
    assert_eq!(summary["request_count"], 4);
    assert_eq!(summary["failed_request_count"], 1);
    assert_eq!(summary["missing_evidence_request_count"], 1);
    assert_eq!(summary["invalid_timestamp_request_count"], 1);
    assert_eq!(summary["itl"]["sample_count"], 1);
    assert_eq!(summary["itl"]["p50_ms"], 0.00001);
    assert_eq!(summary["failed_or_pending_itl_intervals"], 1);
    assert!(summary["ttft"]["p50_ms"].is_null());
}
