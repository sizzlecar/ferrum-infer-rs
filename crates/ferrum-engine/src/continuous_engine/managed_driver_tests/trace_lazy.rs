//! Disabled diagnostics must not build payloads while real inference continues.

use super::*;

#[tokio::test]
async fn disabled_trace_skips_payloads_through_real_cpu_stream_and_completion() {
    let fixture = fixture::Fixture::with_config(|config| {
        config.runtime.scheduler_trace_jsonl = None;
        config.runtime.profile_jsonl = None;
        config.runtime.legacy_scheduler_trace_jsonl = None;
        config.runtime.profile_detail = ObservabilityProfileDetail::Off;
        config.runtime.batch_decode_prof = true;
    });
    fixture.executor.remove_execution_gate();
    let trace_payload_builds =
        Arc::clone(&fixture.engine.active_inner().unwrap().trace_payload_builds);
    fixture
        .engine
        .active_inner()
        .unwrap()
        .write_scheduler_trace_event(|| {
            panic!("disabled legacy trace must not evaluate its payload")
        });
    let stream = fixture
        .engine
        .infer_stream(fixture.request())
        .await
        .unwrap();
    let (response, chunks) = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::join!(
            fixture.engine.infer(fixture.request()),
            stream.collect::<Vec<_>>()
        )
    })
    .await
    .unwrap();
    let response = response.unwrap();
    let chunks = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
    let text: String = chunks
        .iter()
        .map(|chunk| chunk.chunk().text.as_str())
        .collect();
    assert_eq!(text, response.response().text);
    assert_eq!(response.response().tokens.len(), 4);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 8);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 2);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert_eq!(trace_payload_builds.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn enabled_trace_keeps_real_cpu_work_and_scheduler_fields() {
    let legacy = tempfile::tempdir().unwrap();
    let path = legacy.path().join("scheduler.jsonl");
    let fixture = fixture::Fixture::with_config(|config| {
        config.runtime.legacy_scheduler_trace_jsonl = Some(path.clone());
        config.runtime.profile_detail = ObservabilityProfileDetail::Full;
    });
    fixture.executor.remove_execution_gate();
    let trace_payload_builds =
        Arc::clone(&fixture.engine.active_inner().unwrap().trace_payload_builds);
    let response = tokio::time::timeout(
        Duration::from_secs(3),
        fixture.engine.infer(fixture.request()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.response().tokens.len(), 4);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 4);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert!(trace_payload_builds.load(Ordering::Relaxed) > 0);
    let events: Vec<serde_json::Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let iteration = events
        .iter()
        .find(|event| event["event"] == "scheduler_iteration" && event["result"] == "some_ok")
        .expect("actual scheduled work must retain its trace");
    assert!(iteration["iteration"].is_u64());
    assert!(iteration["scheduler_after_process"]["current_iteration"].is_u64());
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release
        ),
        1
    );
}

#[tokio::test]
async fn public_iteration_trace_matches_real_cpu_ranges_for_run_and_serve() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let fixture = fixture::Fixture::with_feedback(
            |config| {
                // Fixture supplies only the public scheduler journal. No
                // legacy diagnostic environment or journal is required.
                config.runtime.legacy_scheduler_trace_jsonl = None;
                config.runtime.profile_entrypoint = Some(entrypoint);
                config.runtime.profile_detail = ObservabilityProfileDetail::Basic;
                // This numerical oracle expects sampling after each forward;
                // use its ordinary full prefill, then real decode ranges.
                // The phase-feedback test covers intermediate prefill chunks.
            },
            None,
            None,
        );
        fixture.executor.remove_execution_gate();
        let request = fixture.request();
        let request_id = request.id.to_string();
        let response = tokio::time::timeout(Duration::from_secs(3), fixture.engine.infer(request))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.response().tokens.len(), 4);
        let work = fixture.executor.take_work_calls();
        fixture.engine.shutdown().await.unwrap();
        let events = read_engine_profile_events(&fixture.trace.path().join("trace.jsonl"));
        let iterations: Vec<_> = events
            .iter()
            .filter(|event| {
                event.phase == "scheduler.iteration" && event.attributes["result"] == "some_ok"
            })
            .collect();
        let actual_ranges: Vec<_> = work
            .iter()
            .flatten()
            .map(|(id, work)| {
                assert_eq!(id.to_string(), request_id);
                match work {
                    TokenWork::Prefill(chunk) => chunk.range(),
                    TokenWork::Decode { position, .. } => *position..*position + 1,
                }
            })
            .collect();
        let traced_ranges: Vec<_> = iterations
            .iter()
            .flat_map(|event| {
                assert_eq!(event.entrypoint, entrypoint);
                assert_eq!(event.request_id, request_id);
                assert!(event.attributes["batch_id"].is_string());
                assert!(event.attributes["timing_us"]["process"].is_u64());
                assert!(event.attributes["slo_planning"].is_null());
                let bounds = &event.attributes["boundaries"];
                assert!(
                    bounds["process_started_monotonic_ns"].as_u64().unwrap()
                        <= bounds["process_completed_monotonic_ns"].as_u64().unwrap()
                );
                event.attributes["plan"]["requests"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        assert_eq!(row["request_id"], request_id);
                        let range = &row["scheduled_range"];
                        let start = range["start"].as_u64().unwrap() as usize;
                        let end = range["end"].as_u64().unwrap() as usize;
                        assert_eq!(row["scheduled_tokens"], end - start);
                        assert_eq!(row["execution_frontier_before"], start);
                        start..end
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(!actual_ranges.is_empty());
        assert_eq!(traced_ranges, actual_ranges);
        assert!(iterations.windows(2).all(|pair| {
            pair[0].attributes["iteration"].as_u64().unwrap()
                < pair[1].attributes["iteration"].as_u64().unwrap()
        }));
    }
}
