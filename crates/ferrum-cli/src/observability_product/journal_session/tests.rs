use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct EngineRetirementState {
    started: AtomicBool,
    changed: tokio::sync::Notify,
    shutdown_calls: AtomicUsize,
    waits: AtomicUsize,
    release: tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

struct ScopedLifetimeEngine {
    config: ferrum_types::EngineConfig,
    scope: Mutex<Option<JsonlProducerScope>>,
    state: Arc<EngineRetirementState>,
    dropped: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for ScopedLifetimeEngine {
    fn drop(&mut self) {
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

#[async_trait::async_trait]
impl ferrum_interfaces::InferenceEngine for ScopedLifetimeEngine {
    async fn status(&self) -> ferrum_types::EngineStatus {
        panic!("lifecycle fixture does not execute inference")
    }
    async fn shutdown(&self) -> Result<()> {
        self.state.shutdown_calls.fetch_add(1, Ordering::SeqCst);
        Err(FerrumError::backend("original retained engine failure"))
    }
    async fn wait_for_shutdown_completion(&self) -> Result<()> {
        self.state.waits.fetch_add(1, Ordering::SeqCst);
        self.state.started.store(true, Ordering::SeqCst);
        self.state.changed.notify_waiters();
        if let Some(release) = self.state.release.lock().await.take() {
            release
                .await
                .expect("test releases original retained custody");
        }
        let scope = self.scope.lock().unwrap().take();
        if let Some(mut scope) = scope {
            tokio::task::spawn_blocking(move || {
                scope
                    .producer::<Value>()
                    .try_enqueue(json!({"actual_retirement_tail": true}))
                    .unwrap();
                scope.finish().result().unwrap();
            })
            .await
            .unwrap();
        }
        Err(FerrumError::backend(
            "historical error after actual retirement",
        ))
    }
    fn config(&self) -> &ferrum_types::EngineConfig {
        &self.config
    }
    fn metrics(&self) -> ferrum_types::EngineMetrics {
        Default::default()
    }
    async fn health_check(&self) -> ferrum_types::HealthStatus {
        ferrum_types::HealthStatus::healthy()
    }
}

#[async_trait::async_trait]
impl ferrum_interfaces::engine::LlmInferenceEngine for ScopedLifetimeEngine {
    async fn infer(
        &self,
        _: ferrum_types::InferenceRequest,
    ) -> Result<ferrum_output::InferenceOutput> {
        Err(FerrumError::unsupported(
            "lifecycle fixture has no inference",
        ))
    }
    async fn infer_stream(
        &self,
        _: ferrum_types::InferenceRequest,
    ) -> Result<ferrum_output::InferenceStream> {
        Err(FerrumError::unsupported(
            "lifecycle fixture has no inference",
        ))
    }
}

fn lifetime_fixture(
    session: &Arc<ProductJournalSession>,
) -> (
    ProductEngineLifetime,
    Arc<EngineRetirementState>,
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    let scope = {
        let routes = session.routes.lock().unwrap();
        let Route::Ready(ready) = routes.values().next().unwrap() else {
            panic!("original writer")
        };
        ready.owner.producer_scope().unwrap()
    };
    let (release, receiver) = tokio::sync::oneshot::channel();
    let (dropped, drop_observer) = tokio::sync::oneshot::channel();
    let state = Arc::new(EngineRetirementState {
        started: AtomicBool::new(false),
        changed: tokio::sync::Notify::new(),
        shutdown_calls: AtomicUsize::new(0),
        waits: AtomicUsize::new(0),
        release: tokio::sync::Mutex::new(Some(receiver)),
    });
    let engine = ScopedLifetimeEngine {
        config: Default::default(),
        scope: Mutex::new(Some(scope)),
        state: Arc::clone(&state),
        dropped: Some(dropped),
    };
    (
        ProductEngineLifetime::new(Box::new(engine), Arc::clone(session)),
        state,
        release,
        drop_observer,
    )
}

async fn wait_for_retirement_observation(state: &EngineRetirementState) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let changed = state.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if state.started.load(Ordering::SeqCst) {
                return;
            }
            changed.await;
        }
    })
    .await
    .unwrap();
}

fn assert_original_writer_joined(session: &ProductJournalSession) {
    assert!(session.is_closed());
    let routes = session.routes.lock().unwrap();
    let Route::Ready(ready) = routes.values().next().unwrap() else {
        panic!("original writer")
    };
    let receipt = ready.final_receipt.as_ref().unwrap();
    assert!(receipt.worker_joined);
    receipt.result().unwrap();
}

#[tokio::test]
async fn product_lifetime_retained_engine_preserves_writer_until_actual_scope_retirement() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = configuration(&path, ProfileDetailArg::Verify);
    let session = ProductJournalSession::prepare(&config, &Default::default());
    let (lifetime, state, release, _) = lifetime_fixture(&session);
    let first = lifetime.shutdown().await.unwrap_err();
    let mut finish = Box::pin(lifetime.finish());
    assert!(futures::poll!(finish.as_mut()).is_pending());
    wait_for_retirement_observation(&state).await;
    assert!(!session.is_closed());
    // The CLI scope has not been retired merely because the first Engine
    // result is Err. Its original writer still accepts the actual tail.
    session
        .write(&event(&config, "while_engine_retained"), true, true, false)
        .unwrap();
    release.send(()).unwrap();
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), finish)
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_string(), first.to_string());
    assert_original_writer_joined(&session);
    assert_eq!(state.shutdown_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.waits.load(Ordering::SeqCst), 1);
    let rows: Vec<Value> = fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["phase"], "while_engine_retained");
    assert_eq!(rows[1], json!({"actual_retirement_tail": true}));
}

#[tokio::test]
async fn product_lifetime_cancelled_final_wait_keeps_original_writer_and_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = configuration(&path, ProfileDetailArg::Verify);
    let session = ProductJournalSession::prepare(&config, &Default::default());
    let (lifetime, state, release, dropped) = lifetime_fixture(&session);
    let mut finish = Box::pin(lifetime.finish());
    assert!(futures::poll!(finish.as_mut()).is_pending());
    // The first poll can still be joining the registered initial shutdown.
    while !state.started.load(Ordering::SeqCst) {
        tokio::select! {
            result = finish.as_mut() => panic!("retained cleanup completed early: {result:?}"),
            _ = wait_for_retirement_observation(&state) => {},
        }
    }
    drop(finish);
    drop(lifetime);
    assert!(!session.is_closed());
    session
        .write(&event(&config, "after_waiter_cancelled"), true, true, false)
        .unwrap();
    release.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), dropped)
        .await
        .unwrap()
        .unwrap();
    assert_original_writer_joined(&session);
    assert_eq!(state.shutdown_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.waits.load(Ordering::SeqCst), 1);
    assert!(fs::read_to_string(path)
        .unwrap()
        .contains("actual_retirement_tail"));
}

#[tokio::test]
async fn product_lifetime_server_handle_and_finish_share_original_shutdown_and_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = configuration(&path, ProfileDetailArg::Verify);
    let session = ProductJournalSession::prepare(&config, &Default::default());
    let (lifetime, state, release, dropped) = lifetime_fixture(&session);
    let server_engine = lifetime.engine();
    let first = server_engine.shutdown().await.unwrap_err();
    let mut server_wait = Box::pin(server_engine.wait_for_shutdown_completion());
    assert!(futures::poll!(server_wait.as_mut()).is_pending());
    wait_for_retirement_observation(&state).await;
    drop(server_wait);
    assert!(!session.is_closed());
    session
        .write(&event(&config, "server_wait_cancelled"), true, true, false)
        .unwrap();

    // The cancelled Server wait cannot cancel or replace the original
    // completion observation. Product finalization joins it and its writer.
    let mut finish = Box::pin(lifetime.finish());
    assert!(futures::poll!(finish.as_mut()).is_pending());
    assert_eq!(state.shutdown_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.waits.load(Ordering::SeqCst), 1);
    release.send(()).unwrap();
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), finish)
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_string(), first.to_string());
    assert_original_writer_joined(&session);
    assert_eq!(state.shutdown_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.waits.load(Ordering::SeqCst), 1);
    let rows: Vec<Value> = fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["phase"], "server_wait_cancelled");
    assert_eq!(rows[1], json!({"actual_retirement_tail": true}));
    drop(server_engine);
    drop(lifetime);
    tokio::time::timeout(std::time::Duration::from_secs(5), dropped)
        .await
        .unwrap()
        .unwrap();
}

fn configuration(path: &PathBuf, detail: ProfileDetailArg) -> ProductObservabilityConfig {
    ProductObservabilityConfig::new(
        ProfileEntrypoint::Serve,
        "scope-test",
        Some(path),
        detail,
        Some(path),
        None,
        None,
        1.0,
    )
}
fn event(config: &ProductObservabilityConfig, stage: &str) -> Vec<FerrumProfileEvent> {
    super::super::actual_memory_stage_events(
        config,
        "scope-request",
        &[ActualMemoryStageObservation::new(stage, stage, None, None)],
        Utc::now(),
    )
}

#[tokio::test]
async fn product_journal_original_scope_keeps_last_shutdown_record_in_same_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = configuration(&path, ProfileDetailArg::Verify);
    let session = ProductJournalSession::prepare(&config, &ferrum_types::EngineConfig::default());
    let mut component = {
        let routes = session.routes.lock().unwrap();
        let Route::Ready(ready) = routes.values().next().unwrap() else {
            panic!("ready original writer")
        };
        ready.owner.producer_scope().unwrap()
    };
    let component_producer = component.producer::<serde_json::Value>();
    session
        .write(&event(&config, "startup"), true, true, false)
        .unwrap();
    component_producer
        .try_enqueue(json!({"actual_component":true}))
        .unwrap();
    component.finish().result().unwrap();
    assert!(component_producer
        .try_enqueue(json!({"after_scope_retirement":true}))
        .is_err());
    session
        .write(&event(&config, "actual_serve_shutdown"), true, true, false)
        .unwrap();
    // The rejected producer attempt precedes final close: required completeness
    // reports it, but the same original worker still joins and keeps the tail.
    assert!(session.finish().await.is_err());
    assert!(session.is_closed());
    let records: Vec<Value> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        records.len(),
        3,
        "canonical role aliases must not duplicate records"
    );
    assert_eq!(records[1], json!({"actual_component":true}));
    assert_eq!(records.last().unwrap()["phase"], "actual_serve_shutdown");
}

#[tokio::test]
async fn product_journal_fixed_final_receipt_ignores_later_closed_attempt() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = configuration(&path, ProfileDetailArg::Verify);
    let session = ProductJournalSession::prepare(&config, &ferrum_types::EngineConfig::default());
    let original = {
        let routes = session.routes.lock().unwrap();
        let Route::Ready(ready) = routes.values().next().unwrap() else {
            panic!("ready writer")
        };
        ready.producer.clone()
    };
    session
        .write(&event(&config, "shutdown"), true, true, false)
        .unwrap();
    session.finish().await.unwrap();
    let receipt = {
        let routes = session.routes.lock().unwrap();
        let Route::Ready(ready) = routes.values().next().unwrap() else {
            panic!("ready writer")
        };
        ready.final_receipt.clone().unwrap()
    };
    assert!(original
        .try_enqueue(Record(Arc::new(event(&config, "late").remove(0))))
        .is_err());
    assert!(original.delivery_stats().dropped > receipt.flush.cut.dropped);
    receipt.result().unwrap();
    session.finish().await.unwrap();
    assert!(session.is_closed());
}

#[tokio::test]
async fn product_journal_required_failure_finishes_all_scopes_and_joins_original_worker() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let mut config = configuration(&path, ProfileDetailArg::Verify);
    config.core.journal.max_record_bytes = 1;
    let session = ProductJournalSession::prepare(&config, &ferrum_types::EngineConfig::default());
    session.result().unwrap();
    assert!(session
        .write(&event(&config, "too_large"), true, true, false)
        .is_err());
    assert!(session.finish().await.is_err());
    assert!(session.is_closed());
    let routes = session.routes.lock().unwrap();
    let Route::Ready(ready) = routes.values().next().unwrap() else {
        panic!("ready writer")
    };
    let receipt = ready.final_receipt.as_ref().unwrap();
    assert!(receipt.worker_joined);
    assert!(ready.scope.is_retired());
    assert!(
        !receipt.flush.scope_retired,
        "final-owner cut is not a component retirement receipt"
    );
    assert!(receipt.flush.cut.dropped > 0);
}

#[tokio::test]
async fn product_journal_live_scope_close_refusal_is_not_optional_success() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = configuration(&path, ProfileDetailArg::Basic);
    let session = ProductJournalSession::prepare(&config, &ferrum_types::EngineConfig::default());
    let mut other = {
        let routes = session.routes.lock().unwrap();
        let Route::Ready(ready) = routes.values().next().unwrap() else {
            panic!("ready writer")
        };
        ready.owner.producer_scope().unwrap()
    };
    assert!(session.finish().await.is_err());
    assert!(!session.is_closed());
    other
        .producer::<Value>()
        .try_enqueue(json!({"still_live":true}))
        .unwrap();
    other.finish().result().unwrap();
    session.finish().await.unwrap();
    assert!(session.is_closed());
    assert!(fs::read_to_string(path).unwrap().contains("still_live"));
}

#[tokio::test]
async fn product_journal_serve_off_canonical_alias_uses_one_original_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let config = ProductObservabilityConfig::new(
        ProfileEntrypoint::Serve,
        "typed-scope-test",
        Some(&path),
        ProfileDetailArg::Off,
        None,
        None,
        None,
        1.0,
    );
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.scheduler_trace_jsonl = Some(directory.path().join(".").join("profile.jsonl"));
    let session = ProductJournalSession::prepare(&config, &engine);
    session.result().unwrap();
    assert_eq!(session.routes.lock().unwrap().len(), 1);
    session
        .write(&event(&config, "actual_serve_shutdown"), true, false, false)
        .unwrap();
    session.finish().await.unwrap();
    assert!(session.is_closed());
    assert!(fs::read_to_string(path)
        .unwrap()
        .contains("actual_serve_shutdown"));
}

#[tokio::test]
async fn product_journal_run_off_profile_remains_typed_original_owner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run.jsonl");
    let config = ProductObservabilityConfig::new(
        ProfileEntrypoint::Run,
        "run-profile",
        Some(&path),
        ProfileDetailArg::Off,
        None,
        None,
        None,
        1.0,
    );
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.profile_jsonl = Some(path.clone());
    let session = ProductJournalSession::prepare(&config, &engine);
    session.result().unwrap();
    session
        .write(&event(&config, "actual_run_shutdown"), true, false, false)
        .unwrap();
    session.finish().await.unwrap();
    assert!(session.is_closed());
    let rows: Vec<Value> = fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["phase"], "actual_run_shutdown");
}

fn memory_event(config: &ProductObservabilityConfig, stage: &str) -> Vec<FerrumProfileEvent> {
    super::super::actual_memory_stage_events(
        config,
        "actual-memory-request",
        &[ActualMemoryStageObservation::new(
            stage,
            stage,
            None,
            Some(crate::memory_profile::ProcessMemoryObservation {
                before_bytes: 1,
                after_bytes: 2,
                current_bytes: 2,
                high_water_bytes: 2,
                source: "controlled-memory-observation",
            }),
        )],
        Utc::now(),
    )
}

#[tokio::test]
async fn product_journal_epoch_truncates_engine_routes_but_preserves_independent_memory_append() {
    let directory = tempfile::tempdir().unwrap();
    let profile = directory.path().join("profile.jsonl");
    let memory = directory.path().join("memory.jsonl");
    fs::write(&profile, "{\"old_profile\":true}\n").unwrap();
    fs::write(&memory, "{\"old_memory\":true}\n").unwrap();
    let config = ProductObservabilityConfig::new(
        ProfileEntrypoint::Serve,
        "epochs",
        Some(&profile),
        ProfileDetailArg::Basic,
        Some(&memory),
        None,
        None,
        1.0,
    );
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.profile_jsonl = Some(directory.path().join(".").join("profile.jsonl"));
    for stage in ["first_shutdown", "second_shutdown"] {
        let session = ProductJournalSession::prepare(&config, &engine);
        session.result().unwrap();
        assert!(
            fs::read_to_string(&profile).unwrap().is_empty(),
            "Engine canonical path starts a fresh epoch"
        );
        assert!(fs::read_to_string(&memory).unwrap().contains("old_memory"));
        session
            .write(&memory_event(&config, stage), true, true, false)
            .unwrap();
        session.finish().await.unwrap();
        assert!(session.is_closed());
    }
    let profile_rows: Vec<Value> = fs::read_to_string(&profile)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(profile_rows.len(), 1);
    assert_eq!(profile_rows[0]["phase"], "second_shutdown");
    let memory_rows: Vec<Value> = fs::read_to_string(&memory)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(memory_rows.len(), 3);
    assert_eq!(memory_rows[0], json!({"old_memory":true}));
    assert_eq!(memory_rows[1]["phase"], "first_shutdown");
    assert_eq!(memory_rows[2]["phase"], "second_shutdown");
}

#[tokio::test]
async fn product_journal_epoch_alias_truncates_once_before_all_original_scope_records() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shared.jsonl");
    fs::write(&path, "{\"old_epoch\":true}\n").unwrap();
    let config = configuration(&path, ProfileDetailArg::Verify);
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.scheduler_trace_jsonl = Some(directory.path().join(".").join("shared.jsonl"));
    let session = ProductJournalSession::prepare(&config, &engine);
    session.result().unwrap();
    assert!(fs::read_to_string(&path).unwrap().is_empty());
    session
        .write(&memory_event(&config, "startup"), true, true, false)
        .unwrap();
    let mut component = {
        let routes = session.routes.lock().unwrap();
        assert_eq!(routes.len(), 1);
        let Route::Ready(ready) = routes.values().next().unwrap() else {
            panic!("ready writer")
        };
        ready.owner.producer_scope().unwrap()
    };
    component
        .producer::<Value>()
        .try_enqueue(json!({"actual_component_record":true}))
        .unwrap();
    component.finish().result().unwrap();
    session
        .write(
            &memory_event(&config, "actual_serve_shutdown"),
            true,
            true,
            false,
        )
        .unwrap();
    session.finish().await.unwrap();
    let rows: Vec<Value> = fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["phase"], "startup");
    assert_eq!(rows[1], json!({"actual_component_record":true}));
    assert_eq!(rows[2]["phase"], "actual_serve_shutdown");
}
