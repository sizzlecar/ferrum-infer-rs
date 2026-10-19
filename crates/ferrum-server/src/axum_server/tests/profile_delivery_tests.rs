//! Actual HTTP projection and the real shared JSONL worker. The controlled LLM
//! isolates delivery behavior; these are not model numerical/performance tests.
use super::*;
use ferrum_bench_core::{
    JsonlJournal, JsonlJournalErrorKind, JsonlJournalOpenMode, JsonlJournalOwner,
    JsonlProducerScope,
};
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
        let mut journal_owner =
            JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default())
                .unwrap();
        let mut cli_scope = journal_owner.producer_scope().unwrap();
        let original = cli_scope.producer::<Value>();
        original
            .try_enqueue(json!({"startup_record":true}))
            .unwrap();
        original.flush().result().unwrap();
        let engine = Arc::new(StubLlm::with_stream_chunks(&["O", "K"]));
        // Builder ordering does not decide policy or open a writer.
        let state = AppState::default()
            .with_profile_detail(ObservabilityProfileDetail::Verify)
            .with_llm(engine.clone())
            .with_profile_jsonl(Some(path.clone()));
        let server = AxumServer::from_state(state).with_journal_scopes(
            HttpJournalScopes::new(vec![journal_owner.producer_scope().unwrap()]).unwrap(),
        );
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
        let proof = server.profile_delivery_report().unwrap();
        assert!(
            proof.journals[0]
                .scope_receipt
                .as_ref()
                .unwrap()
                .scope_retired
        );
        assert!(proof.journals[0].final_receipt.is_none());
        assert!(
            journal_owner.close().is_err(),
            "CLI still owns the last publication scope"
        );
        original
            .try_enqueue(json!({"actual_serve_shutdown":true}))
            .unwrap();
        cli_scope.finish().result().unwrap();
        let final_receipt = journal_owner.close().unwrap();
        final_receipt.result().unwrap();
        assert!(final_receipt.worker_joined);
        assert_eq!(
            read_profile_events(&path).last().unwrap(),
            &json!({"actual_serve_shutdown":true})
        );
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
    let journal_config = ferrum_types::ObservabilityJournalConfig::default();
    let mut blocked_owner = JsonlJournalOwner::open(
        &blocked_path,
        JsonlJournalOpenMode::Append,
        journal_config.clone().into(),
    )
    .unwrap();
    let mut blocker_scope = blocked_owner.producer_scope().unwrap();
    let blocked = blocker_scope.producer::<BlockWriter>();
    let mut healthy_owner = JsonlJournalOwner::open(
        &healthy_path,
        JsonlJournalOpenMode::Append,
        journal_config.into(),
    )
    .unwrap();
    let healthy_scope = healthy_owner.producer_scope().unwrap();
    let healthy = healthy_scope.producer::<Value>();
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
    let filler = blocker_scope.producer::<Value>();
    // Both destinations have the declared product budget. Only the blocked
    // writer is saturated; the healthy writer must not lose its own events
    // merely because the first-request projection emits more than one record.
    for index in 0..journal_config.queue_capacity {
        filler
            .try_enqueue(json!({"fills_original_queue":index}))
            .unwrap();
    }
    let server = AxumServer::from_state(
        state_with_stub("OK")
            .with_profile_jsonl(Some(blocked_path.clone()))
            .with_memory_profile_jsonl(Some(healthy_path.clone()))
            .with_profile_detail(detail)
            .with_profile_journal(journal_config),
    )
    .with_journal_scopes(
        HttpJournalScopes::new(vec![blocked_owner.producer_scope().unwrap(), healthy_scope])
            .unwrap(),
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
    healthy.flush().result().unwrap();
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
    blocker_scope.finish().result().unwrap();
    blocked_owner.close().unwrap().result().unwrap();
    healthy_owner.close().unwrap().result().unwrap();
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
async fn http_profile_delivery_shared_owner_keeps_second_http_scope_live() {
    let path = unique_profile_jsonl("http-profile-two-scopes");
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let first_engine = Arc::new(StubLlm::new("first"));
    let second_engine = Arc::new(StubLlm::new("second"));
    let first = AxumServer::from_state(
        AppState::default()
            .with_llm(first_engine.clone())
            .with_profile_jsonl(Some(path.clone()))
            .with_profile_detail(ObservabilityProfileDetail::Verify),
    )
    .with_journal_scopes(
        HttpJournalScopes::new(vec![journal_owner.producer_scope().unwrap()]).unwrap(),
    );
    let second = AxumServer::from_state(
        AppState::default()
            .with_llm(second_engine.clone())
            .with_profile_jsonl(Some(path.clone()))
            .with_profile_detail(ObservabilityProfileDetail::Verify),
    )
    .with_journal_scopes(
        HttpJournalScopes::new(vec![journal_owner.producer_scope().unwrap()]).unwrap(),
    );
    let first_router = first.build_router().unwrap();
    let second_router = second.build_router().unwrap();
    let response = post_json(first_router, "/v1/chat/completions", request(false)).await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    assert_eq!(
        response_json(response).await["choices"][0]["message"]["content"],
        "first"
    );
    first.stop(Duration::from_secs(2)).await.unwrap();
    let first_report = first.profile_delivery_report().unwrap();
    let first_receipt = first_report.journals[0].scope_receipt.as_ref().unwrap();
    first_receipt.result().unwrap();
    assert!(first_receipt.scope_retired && first_receipt.barrier_observed);
    assert_eq!(first_receipt.flushed, first_receipt.cut.accepted);
    assert!(first_report.journals[0].final_receipt.is_none());
    assert!(!first_report.journals[0].stats.worker_finished);
    assert_eq!(
        journal_owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config,
        "the second HTTP scope still owns publication authority"
    );

    let response = post_json(second_router, "/v1/chat/completions", request(false)).await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    assert_eq!(
        response_json(response).await["choices"][0]["message"]["content"],
        "second"
    );
    owner(&second).flush().unwrap();
    let second_report = second.profile_delivery_report().unwrap();
    assert!(second_report.journals[0].stats.accepted > first_receipt.cut.accepted);
    assert!(!second_report.is_degraded());
    assert_eq!(
        first.profile_delivery_report().unwrap().journals[0]
            .scope_receipt
            .as_ref(),
        Some(first_receipt),
        "later peer publications do not move the retired HTTP scope's cut"
    );
    assert_eq!(
        read_profile_events(&path)
            .iter()
            .filter(|event| event["phase"] == "chat_completions_sync_complete")
            .count(),
        2
    );
    second.stop(Duration::from_secs(2)).await.unwrap();
    let receipt = journal_owner.close().unwrap();
    receipt.result().unwrap();
    assert!(receipt.worker_joined);
    assert_eq!(receipt.flush.flushed, receipt.flush.cut.accepted);
    assert_eq!(receipt.flush.cut.dropped, 0);
    assert_closed(&first);
    assert_closed(&second);
    assert_eq!(first_engine.shutdown_count.load(Ordering::Acquire), 1);
    assert_eq!(second_engine.shutdown_count.load(Ordering::Acquire), 1);
    let _ = fs::remove_file(path);
}

#[tokio::test]
async fn http_profile_delivery_retired_scope_is_contract_without_reopening() {
    let path = unique_profile_jsonl("http-profile-retired-scope");
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let mut cli_scope = journal_owner.producer_scope().unwrap();
    let original = cli_scope.producer::<Value>();
    original.try_enqueue(json!({"before_http":true})).unwrap();
    original.flush().result().unwrap();
    let before = fs::read(&path).unwrap();
    let mut retired_scope = journal_owner.producer_scope().unwrap();
    let receipt = retired_scope.finish();
    receipt.result().unwrap();
    assert!(receipt.scope_retired && receipt.barrier_observed);
    let engine = Arc::new(StubLlm::new("OK"));
    let server = AxumServer::from_state(
        AppState::default()
            .with_llm(engine.clone())
            .with_profile_jsonl(Some(path.clone()))
            .with_profile_detail(ObservabilityProfileDetail::Basic),
    )
    .with_journal_scopes(HttpJournalScopes::new(vec![retired_scope]).unwrap());
    let error = server.build_router().unwrap_err();
    assert!(matches!(error, Error::Internal { .. }));
    let report = server.profile_delivery_report().unwrap();
    assert!(
        report.required_error.is_some(),
        "Basic cannot downgrade Contract"
    );
    assert!(report.journals.is_empty());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert!(!original.delivery_stats().worker_finished);
    assert!(server.stop(Duration::from_secs(2)).await.is_err());
    assert!(
        server
            .profile_delivery_report()
            .unwrap()
            .close_attempt_finished
    );
    assert_eq!(engine.shutdown_count.load(Ordering::Acquire), 1);
    assert_eq!(
        journal_owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config,
        "the original CLI scope is still live"
    );
    original
        .try_enqueue(json!({"after_rejected_http":true}))
        .unwrap();
    cli_scope.finish().result().unwrap();
    let final_receipt = journal_owner.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined);
    assert_eq!(
        read_profile_events(&path),
        vec![
            json!({"before_http":true}),
            json!({"after_rejected_http":true})
        ]
    );
    let _ = fs::remove_file(path);
}

fn declared_adapter_registry() -> ServedModelRegistry {
    ServedModelRegistry::try_new(
        "stub-model",
        ServedModelKind::Llm,
        vec!["stub-model".to_owned()],
        vec![LoraAdapterModel::new(
            "sql",
            "stub-model:sql",
            "/unused-adapter",
        )],
    )
    .unwrap()
}

#[tokio::test]
async fn http_profile_delivery_unstarted_and_rejected_start_retire_injected_scopes() {
    for rejected_start in [false, true] {
        let path = unique_profile_jsonl("http-profile-unprepared");
        let invalid = unique_profile_jsonl("http-profile-unprepared-open-failure");
        let mut journal_owner =
            JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default())
                .unwrap();
        let scope = journal_owner.producer_scope().unwrap();
        let original = scope.producer::<Value>();
        original
            .try_enqueue(json!({"original_scope_record":true}))
            .unwrap();
        original.flush().result().unwrap();
        let before = fs::read(&path).unwrap();
        assert_eq!(
            journal_owner.close().unwrap_err().kind(),
            JsonlJournalErrorKind::Config
        );
        let mut scopes = HttpJournalScopes::new(vec![scope]).unwrap();
        if rejected_start {
            fs::create_dir(&invalid).unwrap();
            let error =
                JsonlJournalOwner::open(&invalid, JsonlJournalOpenMode::Append, Default::default())
                    .unwrap_err();
            scopes = scopes.with_open_failure(&invalid, error).unwrap();
        }
        let engine = Arc::new(StubLlm::new("OK"));
        // The rejected start selects Verify in its actual ServerConfig; the
        // AppState default is Basic and must not downgrade cleanup to optional.
        let mut server = AxumServer::from_llm(engine.clone()).with_journal_scopes(scopes);
        if rejected_start {
            server = server.with_served_model_registry(declared_adapter_registry());
        }
        let config = ServerConfig {
            profile_jsonl: Some(path.clone()),
            profile_detail: ObservabilityProfileDetail::Verify,
            ..ServerConfig::default()
        };
        if rejected_start {
            assert!(matches!(
                server.start(&config).await,
                Err(Error::Unsupported { .. })
            ));
        }
        assert!(!server.is_running());
        assert!(server.profile_delivery_report().is_none());
        let stopped = server.stop(Duration::from_secs(2)).await;
        if rejected_start {
            assert!(matches!(stopped, Err(Error::IO { .. })));
        } else {
            stopped.unwrap();
        }
        let report = server.profile_delivery_report().unwrap();
        assert!(report.close_attempt_finished);
        assert_eq!(report.required_error.is_some(), rejected_start);
        let receipt = report.journals[0].scope_receipt.as_ref().unwrap();
        receipt.result().unwrap();
        assert!(receipt.scope_retired && receipt.barrier_observed);
        assert_eq!(receipt.cut.accepted, 1);
        assert_eq!(receipt.flushed, 1);
        assert!(report.journals[0].final_receipt.is_none());
        // Keep server and the old producer alive. Final ownership must not
        // depend on their Drop, and a late start cannot open another epoch.
        let final_receipt = journal_owner.close().unwrap();
        final_receipt.result().unwrap();
        assert!(final_receipt.worker_joined);
        assert!(server.start(&config).await.is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(original.delivery_stats().worker_finished);
        assert_eq!(engine.shutdown_count.load(Ordering::Acquire), 1);
        assert_closed(&server);
        let _ = fs::remove_file(path);
        if rejected_start {
            fs::remove_dir(invalid).unwrap();
        }
    }
}

#[tokio::test]
async fn http_profile_delivery_unprepared_cancelled_finish_observes_original_barrier() {
    let path = unique_profile_jsonl("http-profile-unprepared-blocked");
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let scope = journal_owner.producer_scope().unwrap();
    let original = scope.producer::<BlockWriter>();
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, resume) = std::sync::mpsc::channel();
    original
        .enqueue(BlockWriter {
            entered: Mutex::new(Some(entered)),
            resume: Mutex::new(resume),
        })
        .unwrap();
    let engine = Arc::new(StubLlm::new("OK"));
    let server = AxumServer::from_llm(engine.clone())
        .with_journal_scopes(HttpJournalScopes::new(vec![scope]).unwrap());
    let mut release = ReleaseWriter(Some(release));
    tokio::time::timeout(Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        journal_owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config
    );
    {
        let finish = server.finish_stop();
        tokio::pin!(finish);
        assert!(tokio::time::timeout(Duration::from_millis(1), &mut finish)
            .await
            .is_err());
    }
    let custody = owner(&server);
    let report = server.profile_delivery_report().unwrap();
    assert!(!report.close_attempt_finished);
    assert!(report.journals[0].scope_receipt.is_none());
    assert!(report.journals[0].final_receipt.is_none());
    assert!(!original.delivery_stats().worker_finished);
    release.release();
    tokio::time::timeout(Duration::from_secs(2), server.finish_stop())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&custody, &owner(&server)));
    let report = server.profile_delivery_report().unwrap();
    let receipt = report.journals[0].scope_receipt.as_ref().unwrap();
    receipt.result().unwrap();
    assert!(receipt.scope_retired && receipt.barrier_observed);
    assert_eq!(receipt.cut.unbounded_accepted, 1);
    assert!(report.journals[0].final_receipt.is_none());
    let final_receipt = journal_owner.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined);
    assert_closed(&server);
    assert_eq!(engine.shutdown_count.load(Ordering::Acquire), 1);
    assert_eq!(
        read_profile_events(&path),
        vec![json!("original writer resumed")]
    );
    let _ = fs::remove_file(path);
}

struct GatedStartLlm {
    inner: StubLlm,
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    resume: Mutex<std::sync::mpsc::Receiver<()>>,
}
#[async_trait]
impl InferenceEngine for GatedStartLlm {
    async fn status(&self) -> EngineStatus {
        self.inner.status().await
    }
    async fn shutdown(&self) -> ferrum_types::Result<()> {
        self.inner.shutdown().await
    }
    async fn wait_for_shutdown_completion(&self) -> ferrum_types::Result<()> {
        self.inner.wait_for_shutdown_completion().await
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
impl LlmInferenceEngine for GatedStartLlm {
    fn adapter_execution_support(&self) -> AdapterExecutionSupport {
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
        AdapterExecutionSupport::PerSequenceLora
    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_profile_delivery_start_stop_gate_prevents_late_profile_assembly() {
    let path = unique_profile_jsonl("http-profile-start-stop-gate");
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let scope = journal_owner.producer_scope().unwrap();
    let original = scope.producer::<Value>();
    original.try_enqueue(json!({"before_start":true})).unwrap();
    original.flush().result().unwrap();
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, resume) = std::sync::mpsc::channel();
    let engine = Arc::new(GatedStartLlm {
        inner: StubLlm::new("OK"),
        entered: Mutex::new(Some(entered)),
        resume: Mutex::new(resume),
    });
    let server = Arc::new(
        AxumServer::from_llm(engine.clone())
            .with_served_model_registry(declared_adapter_registry())
            .with_journal_scopes(HttpJournalScopes::new(vec![scope]).unwrap()),
    );
    let mut release = ReleaseWriter(Some(release));
    let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = ServerConfig {
        host: "127.0.0.1".to_owned(),
        port: occupied.local_addr().unwrap().port(),
        profile_jsonl: Some(path.clone()),
        ..ServerConfig::default()
    };
    let start = {
        let server = server.clone();
        tokio::spawn(async move { server.start(&config).await })
    };
    tokio::time::timeout(Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    let finish = server.finish_stop();
    tokio::pin!(finish);
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(finish.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    // The stop cannot pass an already-entered start gate while capability
    // validation is held. No timing sleep or assumed scheduling is involved.
    assert!(!server.lifecycle.shutdown_requested.load(Ordering::Acquire));
    assert_eq!(engine.inner.shutdown_count.load(Ordering::Acquire), 0);
    release.release();
    tokio::time::timeout(Duration::from_secs(2), &mut finish)
        .await
        .unwrap()
        .unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(2), start)
        .await
        .unwrap()
        .unwrap()
        .is_err());
    assert!(!server.is_running());
    assert!(
        server.profile_delivery_report().unwrap().journals[0]
            .scope_receipt
            .as_ref()
            .unwrap()
            .scope_retired
    );
    journal_owner.close().unwrap().result().unwrap();
    assert_closed(&server);
    assert_eq!(engine.inner.shutdown_count.load(Ordering::Acquire), 1);
    assert_eq!(
        read_profile_events(&path),
        vec![json!({"before_start":true})]
    );
    let _ = fs::remove_file(path);
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
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let mut original_scope = journal_owner.producer_scope().unwrap();
    let original = original_scope.producer::<PanicWriter>();
    original.enqueue(PanicWriter).unwrap();
    let failure = original.flush().result().unwrap_err();
    assert_eq!(failure.kind(), JsonlJournalErrorKind::Panic);
    let server = AxumServer::from_state(
        state_with_stub("OK")
            .with_profile_jsonl(Some(path.clone()))
            .with_profile_detail(ObservabilityProfileDetail::Basic),
    )
    .with_journal_scopes(
        HttpJournalScopes::new(vec![])
            .unwrap()
            .with_open_failure(&path, std::io::Error::other(failure))
            .unwrap(),
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
    assert!(original_scope.finish().result().is_err());
    let final_receipt = journal_owner.close().unwrap();
    assert!(final_receipt.worker_joined);
    assert!(final_receipt.result().is_err());
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
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let mut blocker_scope = journal_owner.producer_scope().unwrap();
    let blocked = blocker_scope.producer::<BlockWriter>();
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
    let server = AxumServer::from_state(state_with_stub("OK")).with_journal_scopes(
        HttpJournalScopes::new(vec![journal_owner.producer_scope().unwrap()]).unwrap(),
    );
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
    blocker_scope.finish().result().unwrap();
    journal_owner.close().unwrap().result().unwrap();
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
    scope: Mutex<JsonlProducerScope>,
    shutdown_calls: std::sync::atomic::AtomicUsize,
    shutdown_finished: tokio::sync::Notify,
}

#[tokio::test]
async fn finish_stop_waits_for_retained_engine_producer_and_preserves_original_error() {
    let path = unique_profile_jsonl("retained-engine");
    let mut owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let scope = owner.producer_scope().unwrap();
    let engine = Arc::new(RetryShutdownLlm {
        inner: StubLlm::new("OK"),
        journal: scope.producer(),
        scope: Mutex::new(scope),
        shutdown_calls: std::sync::atomic::AtomicUsize::new(0),
        shutdown_finished: tokio::sync::Notify::new(),
    });
    let server = AxumServer::from_llm(engine.clone());
    let mut finish = Box::pin(server.finish_stop());
    assert!(futures::poll!(finish.as_mut()).is_pending());
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let started = engine.shutdown_finished.notified();
            tokio::pin!(started);
            started.as_mut().enable();
            if engine.shutdown_calls.load(Ordering::Acquire) != 0 {
                break;
            }
            started.await;
        }
    })
    .await
    .unwrap();
    assert_eq!(engine.shutdown_calls.load(Ordering::Acquire), 1);
    assert!(!server.lifecycle.engines_stopped.load(Ordering::Acquire));
    assert!(
        owner.close().is_err(),
        "original engine producer is still live"
    );
    engine
        .journal
        .try_enqueue(json!({"before_actual_retirement": true}))
        .unwrap();

    // The controlled producer's original explicit recovery releases its scope.
    // Merely waiting in finish_stop must not invoke this second attempt itself.
    engine.shutdown().await.unwrap();
    let error = tokio::time::timeout(Duration::from_secs(2), finish)
        .await
        .unwrap()
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("controlled engine cleanup remains incomplete"));
    assert!(server.lifecycle.engines_stopped.load(Ordering::Acquire));
    let receipt = owner.close().unwrap();
    assert!(receipt.worker_joined);
    receipt.result().unwrap();
    let records: Vec<Value> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        records,
        vec![
            json!({"before_actual_retirement": true}),
            json!({"engine_producer_finished": true})
        ]
    );
    assert_eq!(engine.shutdown_calls.load(Ordering::Acquire), 2);
    let _ = fs::remove_file(path);
}
#[async_trait]
impl InferenceEngine for RetryShutdownLlm {
    async fn status(&self) -> EngineStatus {
        self.inner.status().await
    }
    async fn shutdown(&self) -> ferrum_types::Result<()> {
        let attempt = self.shutdown_calls.fetch_add(1, Ordering::AcqRel);
        self.shutdown_finished.notify_waiters();
        if attempt == 0 {
            return Err(Error::internal(
                "controlled engine cleanup remains incomplete",
            ));
        }
        self.journal
            .try_enqueue(json!({"engine_producer_finished":true}))
            .map_err(|error| Error::internal(error.to_string()))?;
        self.inner.shutdown().await?;
        let result = self
            .scope
            .lock()
            .unwrap()
            .finish()
            .result()
            .map_err(|error| Error::internal(error.to_string()));
        self.shutdown_finished.notify_waiters();
        result
    }
    async fn wait_for_shutdown_completion(&self) -> ferrum_types::Result<()> {
        loop {
            let notified = self.shutdown_finished.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.scope.lock().unwrap().is_retired() {
                return self.inner.wait_for_shutdown_completion().await;
            }
            notified.await;
        }
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
async fn http_profile_delivery_engine_error_retains_live_producer_until_actual_retirement() {
    let path = unique_profile_jsonl("http-profile-shutdown-retry");
    let mut journal_owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let engine_scope = journal_owner.producer_scope().unwrap();
    let engine = Arc::new(RetryShutdownLlm {
        inner: StubLlm::new("OK"),
        journal: engine_scope.producer(),
        scope: Mutex::new(engine_scope),
        shutdown_calls: std::sync::atomic::AtomicUsize::new(0),
        shutdown_finished: tokio::sync::Notify::new(),
    });
    let server = AxumServer::from_state(
        AppState::default()
            .with_llm(engine.clone())
            .with_profile_jsonl(Some(path.clone())),
    )
    .with_journal_scopes(
        HttpJournalScopes::new(vec![journal_owner.producer_scope().unwrap()]).unwrap(),
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
    assert!(report.close_attempt_finished);
    assert!(
        report.journals[0]
            .scope_receipt
            .as_ref()
            .unwrap()
            .scope_retired
    );
    assert!(!report.journals[0].stats.worker_finished);
    assert!(
        journal_owner.close().is_err(),
        "engine producer is still live"
    );
    assert!(!server.lifecycle.engines_stopped.load(Ordering::Acquire));
    // This is the actual engine-owned original handle, not a reopened writer.
    engine
        .journal
        .try_enqueue(json!({"engine_still_produces_after_failed_shutdown":true}))
        .unwrap();
    engine.journal.flush().result().unwrap();
    let before = fs::read_to_string(&path).unwrap();
    assert!(before
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .any(|event| event["engine_still_produces_after_failed_shutdown"] == true));
    // Recovery belongs to the original controlled producer. Server finish
    // observes its completion and must not issue another shutdown invocation.
    engine.shutdown().await.unwrap();
    let final_error = server.finish_stop().await.unwrap_err();
    assert_eq!(final_error.to_string(), error.to_string());
    assert!(server.lifecycle.engines_stopped.load(Ordering::Acquire));
    journal_owner.close().unwrap().result().unwrap();
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
