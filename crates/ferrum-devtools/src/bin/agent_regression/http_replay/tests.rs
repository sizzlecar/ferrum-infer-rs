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
        assert_eq!(report["transport_complete"], true);
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
