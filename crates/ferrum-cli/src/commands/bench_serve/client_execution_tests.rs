use super::*;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
    Router,
};
use tokio::sync::Barrier;

async fn overlapping_responses(
    State(barrier): State<Arc<Barrier>>,
    headers: HeaderMap,
) -> (StatusCode, &'static str) {
    if headers[BENCHMARK_PHASE_HEADER] == "measured" {
        match headers[BENCHMARK_REQUEST_INDEX_HEADER].to_str().unwrap() {
            "0" | "1" => {
                barrier.wait().await;
            }
            _ => return (StatusCode::SERVICE_UNAVAILABLE, "test rejection"),
        }
    }
    (
        StatusCode::OK,
        "data: {\"id\":\"chatcmpl-test\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}],\"usage\":{\"completion_tokens\":1}}\n\ndata: [DONE]\n\n",
    )
}

#[tokio::test]
async fn client_execution_span_tracks_overlaps_failures_and_permit_release() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/v1/chat/completions", post(overlapping_responses))
        .with_state(Arc::new(Barrier::new(2)));
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let ctx = RunContext {
        client: Arc::new(reqwest::Client::new()),
        base_url: Arc::new(format!("http://{address}")),
        model: Arc::new("test".to_string()),
        max_out: 1,
        ignore_eos: true,
        enable_thinking: None,
        reasoning_effort: None,
        sampling: HttpRequestSampling::default(),
        timeout_s: 5.0,
        benchmark_run_id: Arc::new("span-test".to_string()),
    };
    let runs = tokio::time::timeout(Duration::from_secs(10), async {
        let mut runs = Vec::new();
        for open_loop in [false, true] {
            let prompts = (0..4)
                .map(|_| PromptCase {
                    text: "test".to_string(),
                    input_tokens: 1,
                    sha256: "test".to_string(),
                    output_budget: None,
                })
                .collect();
            let run = if open_loop {
                run_open_loop(&ctx, prompts, 1, 1_000_000.0, "open", 0, Some(7)).await
            } else {
                run_closed_loop(&ctx, prompts, 1, 2, "closed", 0).await
            };
            runs.push((open_loop, run));
        }
        runs
    })
    .await;
    server.abort();
    for (open_loop, run) in runs.expect("overlapping requests must complete") {
        assert_eq!(run.warmup.completed, 1);
        assert_eq!(run.records.len(), 3);
        assert_eq!(run.n_completed(), 2);
        assert_eq!(run.n_errored(), 1);
        let spans: Vec<_> = run
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let correlation = record.benchmark_correlation.as_ref().unwrap();
                assert_eq!(correlation.phase, BenchmarkPhase::Measured);
                assert_eq!(correlation.request_index, index as u32);
                let span = record
                    .client_execution_span
                    .expect("including HTTP failure");
                assert!(span.start_ns <= span.end_ns);
                assert!(span.end_ns <= (run.duration_s * 1e9).ceil() as u64);
                // Cleanup may extend beyond E2E; it must not truncate that observation.
                assert!((span.end_ns - span.start_ns) as f64 + 1.0 >= record.e2e_ms * 1e6);
                span
            })
            .collect();
        assert!(spans[0].start_ns < spans[1].end_ns);
        assert!(spans[1].start_ns < spans[0].end_ns);
        assert!(!run.records[2].success);
        assert_eq!(run.records[2].quality_issues.http_rejected, 1);
        if !open_loop {
            // The third request cannot acquire a permit before one of the
            // first two tasks records completion and releases its permit.
            assert!(spans[2].start_ns >= spans[0].end_ns.min(spans[1].end_ns));
        }
    }
}
