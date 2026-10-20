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
