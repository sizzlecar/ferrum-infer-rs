//! Real CPU work, original ingress timing and independently owned journals.
use super::*;

#[tokio::test]
async fn instance_host_profile_keeps_original_clock_and_independent_cpu_engines() {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("first.jsonl");
    let second_path = directory.path().join("second.jsonl");
    let configure = |config: &mut EngineConfig, path: &std::path::Path, model: &str| {
        config.runtime.profile_jsonl = Some(path.to_owned());
        config.runtime.profile_detail = ObservabilityProfileDetail::Off;
        config.runtime.batch_decode_prof = true;
        config.model.model_id = model.into();
    };
    let mut first = fixture::Fixture::with_config(|config| configure(config, &first_path, "first"));
    let second = fixture::Fixture::with_config(|config| configure(config, &second_path, "second"));
    second.executor.remove_execution_gate();
    for fixture in [&first, &second] {
        let sink = fixture.engine.inner.host_profile_sink.as_ref().unwrap();
        assert_eq!(
            sink.device_timing_mode(),
            ferrum_runtime::DeviceTimingMode::Off
        );
        assert_eq!(
            sink.execution_facts_capture(),
            ferrum_runtime::ExecutionFactsCapture::Off
        );
    }
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::HttpHandler);
    let request = first.request().with_ingress(ingress);
    let id = request.id.clone();
    let stream = first.engine.infer_stream(request).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), &mut first.reached)
        .await
        .unwrap()
        .unwrap();
    let before_release = ingress.elapsed().as_micros() as u64;
    let reference = second.engine.infer(second.request()).await.unwrap();
    second.engine.shutdown().await.unwrap();
    assert_eq!(first.executor.closes.load(Ordering::SeqCst), 0);
    assert_eq!(second.executor.checked_outputs.load(Ordering::SeqCst), 4);
    assert_eq!(second.executor.closes.load(Ordering::SeqCst), 1);
    assert!(second.executor.observed.lock().model.kv_caches.is_empty());
    first.release.send(()).unwrap();
    let chunks = tokio::time::timeout(Duration::from_secs(3), stream.collect::<Vec<_>>())
        .await
        .unwrap()
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let text: String = chunks
        .iter()
        .map(|chunk| chunk.chunk().text.as_str())
        .collect();
    assert_eq!(text, reference.response().text);
    let tokens: Vec<_> = chunks
        .iter()
        .filter_map(|chunk| chunk.chunk().token)
        .collect();
    assert_eq!(tokens, reference.response().tokens);
    first.engine.shutdown().await.unwrap();
    assert_eq!(first.executor.checked_outputs.load(Ordering::SeqCst), 4);
    assert_eq!(first.executor.closes.load(Ordering::SeqCst), 1);
    assert!(first.executor.observed.lock().model.kv_caches.is_empty());

    let read = |path: &std::path::Path| -> Vec<FerrumProfileEvent> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let event: FerrumProfileEvent = serde_json::from_str(line).unwrap();
                event.validate().unwrap();
                event
            })
            .collect()
    };
    let events = read(&first_path);
    let first_text: Vec<_> = events
        .iter()
        .filter(|event| event.phase == "stream_ttft_prof")
        .collect();
    assert_eq!(first_text.len(), 1);
    let event = first_text[0];
    assert_eq!(event.request_id, id.to_string());
    assert_eq!(event.model.as_deref(), Some("first"));
    assert_eq!(event.attributes["request_start_boundary"], "http_handler");
    assert_eq!(event.attributes["client_visible_latency"], false);
    assert!(event.duration_us.unwrap() >= before_release);
    assert!(u128::from(event.duration_us.unwrap()) <= ingress.elapsed().as_micros());
    assert!(events.iter().any(|event| event.phase == "iter_prof"));
    assert!(read(&second_path)
        .iter()
        .all(|event| event.model.as_deref() == Some("second")));
    assert!(read(&second_path)
        .iter()
        .all(|event| event.request_id != id.to_string()));
}
