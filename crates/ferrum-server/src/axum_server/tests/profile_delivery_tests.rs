//! Actual HTTP projection and the real shared JSONL worker. The controlled LLM
//! isolates delivery behavior; these are not model numerical/performance tests.
use super::*;
use ferrum_bench_core::{JsonlJournal, JsonlJournalErrorKind, JsonlJournalOpenMode};
use ferrum_types::ObservabilityProfileDetail;
use std::time::Duration;

struct BlockWriter {
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    resume: Mutex<std::sync::mpsc::Receiver<()>>,
}
impl serde::Serialize for BlockWriter {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.entered
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(())
            .unwrap();
        self.resume
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        serializer.serialize_str("original writer resumed")
    }
}
struct ReleaseWriter(Option<std::sync::mpsc::Sender<()>>);
impl ReleaseWriter {
    fn release(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}
impl Drop for ReleaseWriter {
    fn drop(&mut self) {
        self.release();
    }
}

fn request(stream: bool) -> Value {
    json!({"model":"stub-model", "messages":[{"role":"user","content":"hello"}], "stream":stream})
}
fn owner(server: &AxumServer) -> Arc<HttpProfileDelivery> {
    server
        .lifecycle
        .profile_delivery
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone()
}
fn assert_closed(server: &AxumServer) {
    let report = server.profile_delivery_report().unwrap();
    assert!(report.close_attempt_finished);
    assert!(report
        .journals
        .iter()
        .all(|journal| journal.stats.worker_finished && journal.stats.pending_payload_bytes == 0));
}

#[tokio::test]
async fn http_profile_delivery_sync_and_stream_preserve_events_and_append() {
    for streaming in [false, true] {
        let path = unique_profile_jsonl("http-profile-append");
        let original = JsonlJournal::<Value>::append(&path).unwrap();
        original
            .try_enqueue(json!({"startup_record":true}))
            .unwrap();
        original.flush().unwrap();
        let engine = Arc::new(StubLlm::with_stream_chunks(&["O", "K"]));
        // Builder ordering does not decide policy or open a writer.
        let state = AppState::default()
            .with_profile_detail(ObservabilityProfileDetail::Verify)
            .with_llm(engine.clone())
            .with_profile_jsonl(Some(path.clone()));
        let server = AxumServer::from_state(state);
        let router = server.build_router().unwrap();
        let response = post_json(router, "/v1/chat/completions", request(streaming)).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        if streaming {
            let body = response_text(response).await;
            assert!(body.contains("data: [DONE]"));
        } else {
            assert_eq!(
                response_json(response).await["choices"][0]["message"]["content"],
                "OK"
            );
        }
        // Explicit original-owner barrier, not a handler-side flush.
        owner(&server).flush().unwrap();
        let events = read_profile_events(&path);
        assert_eq!(events[0], json!({"startup_record":true}));
        let phase = if streaming {
            "chat_completions_stream_complete"
        } else {
            "chat_completions_sync_complete"
        };
        let event = events.iter().find(|event| event["phase"] == phase).unwrap();
        assert_eq!(event["entrypoint"], "serve");
        assert_eq!(event["attributes"]["stream"], streaming);
        assert_eq!(event["status"], "ok");
        assert!(events
            .iter()
            .any(|event| event["phase"] == "actual_serve_first_request_done"));
        server.stop(Duration::from_secs(2)).await.unwrap();
        assert_closed(&server);
        assert_eq!(engine.shutdown_count.load(Ordering::Acquire), 1);
        let report = server.profile_delivery_report().unwrap();
        assert!(!report.is_degraded());
        assert_eq!(
            report.journals[0].stats.accepted,
            report.journals[0].stats.flushed
        );
        drop(original);
        let _ = fs::remove_file(path);
    }
}

async fn saturated_http_fanout(detail: ObservabilityProfileDetail, streaming: bool) {
    let blocked_path = unique_profile_jsonl("http-profile-full");
    let healthy_path = unique_profile_jsonl("http-memory-healthy");
    let journal_config = ferrum_types::ObservabilityJournalConfig {
        queue_capacity: 1,
        ..Default::default()
    };
    let blocked = JsonlJournal::<BlockWriter>::open(
        &blocked_path,
        JsonlJournalOpenMode::Append,
        journal_config.clone().into(),
    )
    .unwrap();
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, resume) = std::sync::mpsc::channel();
    blocked
        .enqueue(BlockWriter {
            entered: Mutex::new(Some(entered)),
            resume: Mutex::new(resume),
        })
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    let filler = JsonlJournal::<Value>::append(&blocked_path).unwrap();
    filler
        .try_enqueue(json!({"fills_original_queue":true}))
        .unwrap();
    let server = AxumServer::from_state(
        state_with_stub("OK")
            .with_profile_jsonl(Some(blocked_path.clone()))
            .with_memory_profile_jsonl(Some(healthy_path.clone()))
            .with_profile_detail(detail)
            .with_profile_journal(journal_config),
    );
    // Must drop before server/journal owners if an assertion unwinds.
    let mut release = ReleaseWriter(Some(release));
    let router = server.build_router().unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        post_json(router.clone(), "/v1/chat/completions", request(streaming)),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), AxumStatusCode::OK);
    if streaming {
        assert!(
            tokio::time::timeout(Duration::from_secs(2), response_text(response))
                .await
                .unwrap()
                .contains("data: [DONE]")
        );
    } else {
        assert_eq!(
            response_json(response).await["choices"][0]["message"]["content"],
            "OK"
        );
    }
    let healthy = JsonlJournal::<Value>::append(&healthy_path).unwrap();
    healthy.flush().unwrap();
    assert!(read_profile_events(&healthy_path)
        .iter()
        .any(|event| event["phase"] == "actual_serve_first_request_done"));
    let health = response_json(get(router, "/health").await).await;
    assert_eq!(
        health["profile_delivery"]["required_failure"],
        detail.requires_complete_evidence()
    );
    assert_eq!(health["profile_delivery"]["degraded"], true);
    assert_eq!(
        health["status"],
        if detail.requires_complete_evidence() {
            "unhealthy"
        } else {
            "degraded"
        }
    );
    assert_eq!(
        server.health_check().await,
        if detail.requires_complete_evidence() {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        }
    );
    assert!(blocked.delivery_stats().dropped >= 2);
    release.release();
    let result = server.stop(Duration::from_secs(2)).await;
    assert_eq!(result.is_err(), detail.requires_complete_evidence());
    assert_closed(&server);
    let report = server.profile_delivery_report().unwrap();
    assert!(report.is_degraded());
    assert_eq!(report.journals.len(), 2);
    assert!(report.journals.iter().any(|journal| journal
        .path
        .ends_with(healthy_path.file_name().unwrap())
        && journal.stats.flushed != 0));
    let _ = fs::remove_file(blocked_path);
    let _ = fs::remove_file(healthy_path);
}
#[tokio::test]
async fn http_profile_delivery_optional_full_queue_preserves_healthy_destination() {
    saturated_http_fanout(ObservabilityProfileDetail::Basic, false).await;
}
#[tokio::test]
async fn http_profile_delivery_required_loss_after_sse_is_visible_at_health_and_stop() {
    saturated_http_fanout(ObservabilityProfileDetail::Verify, true).await;
}

#[tokio::test]
async fn http_profile_delivery_canonical_alias_enqueues_memory_once() {
    let path = unique_profile_jsonl("http-profile-alias");
    let alias = path
        .parent()
        .unwrap()
        .join(".")
        .join(path.file_name().unwrap());
    let server = AxumServer::from_state(
        state_with_stub("OK")
            .with_profile_jsonl(Some(path.clone()))
            .with_memory_profile_jsonl(Some(alias)),
    );
    let response = post_json(
        server.build_router().unwrap(),
        "/v1/chat/completions",
        request(false),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let _ = response_json(response).await;
    owner(&server).flush().unwrap();
    assert_eq!(server.profile_delivery_report().unwrap().journals.len(), 1);
    assert_eq!(
        read_profile_events(&path)
            .iter()
            .filter(|event| event["phase"] == "actual_serve_first_request_done")
            .count(),
        1
    );
    server.stop(Duration::from_secs(2)).await.unwrap();
    assert_closed(&server);
    let _ = fs::remove_file(path);
}

struct PanicWriter;
impl serde::Serialize for PanicWriter {
    fn serialize<S: serde::Serializer>(&self, _: S) -> std::result::Result<S::Ok, S::Error> {
        panic!("injected original writer panic")
    }
}
#[tokio::test]
async fn http_profile_delivery_failed_shared_writer_attach_preserves_contract() {
    let path = unique_profile_jsonl("http-profile-panic");
    let original = JsonlJournal::<PanicWriter>::append(&path).unwrap();
    original.enqueue(PanicWriter).unwrap();
    assert_eq!(
        original.flush().unwrap_err().kind(),
        JsonlJournalErrorKind::Panic
    );
    let server = AxumServer::from_state(
        state_with_stub("OK")
            .with_profile_jsonl(Some(path.clone()))
            .with_profile_detail(ObservabilityProfileDetail::Basic),
    );
    assert!(server.build_router().is_err());
    assert!(server
        .profile_delivery_report()
        .unwrap()
        .required_error
        .is_some());
    assert!(server.stop(Duration::from_secs(2)).await.is_err());
    assert!(
        server
            .profile_delivery_report()
            .unwrap()
            .close_attempt_finished
    );
    // The retained original handle, not a replacement worker, reports exit.
    assert!(original.close().is_err());
    assert!(original.delivery_stats().worker_finished);
    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn http_profile_delivery_startup_failure_keeps_opened_destination_closable() {
    let invalid = unique_profile_jsonl("http-profile-directory");
    fs::create_dir(&invalid).unwrap();
    let healthy = unique_profile_jsonl("http-startup-healthy");
    for detail in [
        ObservabilityProfileDetail::Basic,
        ObservabilityProfileDetail::Verify,
    ] {
        let server = AxumServer::from_state(
            state_with_stub("OK")
                .with_profile_detail(detail)
                .with_profile_jsonl(Some(invalid.clone()))
                .with_memory_profile_jsonl(Some(healthy.clone())),
        );
        let router = server.build_router();
        assert_eq!(router.is_err(), detail.requires_complete_evidence());
        if let Ok(router) = router {
            let response = post_json(router, "/v1/chat/completions", request(false)).await;
            assert_eq!(response.status(), AxumStatusCode::OK);
            let _ = response_json(response).await;
            owner(&server).flush().unwrap();
            assert!(read_profile_events(&healthy)
                .iter()
                .any(|event| event["phase"] == "actual_serve_first_request_done"));
        }
        assert_eq!(
            server.stop(Duration::from_secs(2)).await.is_err(),
            detail.requires_complete_evidence()
        );
        assert_closed(&server);
        assert_eq!(server.profile_delivery_report().unwrap().journals.len(), 1);
    }
    let _ = fs::remove_dir(invalid);
    let _ = fs::remove_file(healthy);
}

#[tokio::test]
async fn http_profile_delivery_bind_failure_and_cancelled_close_retain_original_owner() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let path = unique_profile_jsonl("http-profile-bind");
    let blocked = JsonlJournal::<BlockWriter>::append(&path).unwrap();
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, resume) = std::sync::mpsc::channel();
    blocked
        .enqueue(BlockWriter {
            entered: Mutex::new(Some(entered)),
            resume: Mutex::new(resume),
        })
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    let server = AxumServer::from_state(state_with_stub("OK"));
    let mut release = ReleaseWriter(Some(release));
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: address.port(),
        profile_jsonl: Some(path.clone()),
        ..ServerConfig::default()
    };
    assert!(server.start(&config).await.is_err());
    assert!(!server.is_running());
    assert!(server.stop(Duration::from_millis(1)).await.is_err());
    let report = server.profile_delivery_report().unwrap();
    assert!(!report.close_attempt_finished);
    assert!(!report.journals[0].stats.worker_finished);
    release.release();
    server.stop(Duration::from_secs(2)).await.unwrap();
    assert_closed(&server);
    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn http_profile_delivery_invalid_projection_is_contract_before_any_destination() {
    let profile = unique_profile_jsonl("http-profile-invalid");
    let memory = unique_profile_jsonl("http-memory-invalid");
    let server = AxumServer::from_state(
        state_with_stub("OK")
            .with_profile_jsonl(Some(profile.clone()))
            .with_memory_profile_jsonl(Some(memory.clone()))
            .with_profile_detail(ObservabilityProfileDetail::Basic),
    );
    let response = post_json(
        server.build_router().unwrap(),
        "/v1/chat/completions",
        request(false),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    let _ = response_json(response).await;
    let delivery = owner(&server);
    delivery.flush().unwrap();
    let mut invalid: FerrumProfileEvent =
        serde_json::from_value(read_profile_events(&profile).into_iter().next().unwrap()).unwrap();
    invalid.schema_version = 0;
    let before = delivery.report();
    assert!(delivery.enqueue(invalid, true).is_err());
    let after = delivery.report();
    for (before, after) in before.journals.iter().zip(&after.journals) {
        assert_eq!(before.stats.accepted, after.stats.accepted);
        assert_eq!(before.stats.dropped, after.stats.dropped);
    }
    assert!(
        after.required_error.is_some(),
        "Basic never downgrades Contract"
    );
    assert!(server.stop(Duration::from_secs(2)).await.is_err());
    assert_closed(&server);
    let _ = fs::remove_file(profile);
    let _ = fs::remove_file(memory);
}

/// A failed generic shutdown does not assert that its producer stopped. Reuse
/// the normal controlled inference engine and retain its real original writer.
struct RetryShutdownLlm {
    inner: StubLlm,
    journal: JsonlJournal<Value>,
    shutdown_calls: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl InferenceEngine for RetryShutdownLlm {
    async fn status(&self) -> EngineStatus {
        self.inner.status().await
    }
    async fn shutdown(&self) -> ferrum_types::Result<()> {
        if self.shutdown_calls.fetch_add(1, Ordering::AcqRel) == 0 {
            return Err(Error::internal(
                "controlled engine cleanup remains incomplete",
            ));
        }
        self.journal
            .try_enqueue(json!({"engine_producer_finished":true}))
            .map_err(|error| Error::internal(error.to_string()))?;
        self.inner.shutdown().await
    }
    fn config(&self) -> &EngineConfig {
        self.inner.config()
    }
    fn metrics(&self) -> EngineMetrics {
        self.inner.metrics()
    }
    async fn health_check(&self) -> EngineHealthStatus {
        self.inner.health_check().await
    }
}
#[async_trait]
impl LlmInferenceEngine for RetryShutdownLlm {
    async fn infer(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<ferrum_output::InferenceOutput> {
        self.inner.infer(request).await
    }
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> ferrum_types::Result<ferrum_output::InferenceStream> {
        self.inner.infer_stream(request).await
    }
}

#[tokio::test]
async fn http_profile_delivery_engine_error_retains_live_producer_until_successful_retry() {
    let path = unique_profile_jsonl("http-profile-shutdown-retry");
    let engine = Arc::new(RetryShutdownLlm {
        inner: StubLlm::new("OK"),
        journal: JsonlJournal::append(&path).unwrap(),
        shutdown_calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let server = AxumServer::from_state(
        AppState::default()
            .with_llm(engine.clone())
            .with_profile_jsonl(Some(path.clone())),
    );
    let response = post_json(
        server.build_router().unwrap(),
        "/v1/chat/completions",
        request(false),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    assert_eq!(
        response_json(response).await["choices"][0]["message"]["content"],
        "OK"
    );
    let error = server.stop(Duration::from_secs(2)).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("controlled engine cleanup remains incomplete"));
    let report = server.profile_delivery_report().unwrap();
    assert!(!report.close_attempt_finished);
    assert!(!report.journals[0].stats.worker_finished);
    assert!(!server.lifecycle.engines_stopped.load(Ordering::Acquire));
    // This is the actual engine-owned original handle, not a reopened writer.
    engine
        .journal
        .try_enqueue(json!({"engine_still_produces_after_failed_shutdown":true}))
        .unwrap();
    engine.journal.flush().unwrap();
    let before = fs::read_to_string(&path).unwrap();
    assert!(before
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .any(|event| event["engine_still_produces_after_failed_shutdown"] == true));
    server.stop(Duration::from_secs(2)).await.unwrap();
    assert_closed(&server);
    assert_eq!(engine.shutdown_calls.load(Ordering::Acquire), 2);
    assert_eq!(engine.inner.shutdown_count.load(Ordering::Acquire), 1);
    let report = server.profile_delivery_report().unwrap();
    assert_eq!(
        report.journals[0].stats.accepted,
        report.journals[0].stats.flushed
    );
    assert!(engine.journal.delivery_stats().worker_finished);
    assert!(engine
        .journal
        .try_enqueue(json!({"after_final_close":true}))
        .is_err());
    let after = fs::read_to_string(&path).unwrap();
    assert!(after
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .any(|event| event["engine_producer_finished"] == true));
    let _ = fs::remove_file(path);
}
