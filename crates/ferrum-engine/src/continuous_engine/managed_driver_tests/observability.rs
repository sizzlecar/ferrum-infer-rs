//! Real CPU numerical work remains independent of optional journal delivery.
use super::*;
use ferrum_bench_core::{JsonlJournalConfig, JsonlJournalErrorKind, JsonlJournalOpenMode};
use ferrum_runtime::{ExecutionEventDeliveryPolicy, ExecutionEventSinkErrorKind};

struct BlockWriter {
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl serde::Serialize for BlockWriter {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.entered.lock().take().unwrap().send(()).unwrap();
        self.release
            .lock()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        serializer.serialize_str("writer gate released")
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

struct PanicWriter;
impl serde::Serialize for PanicWriter {
    fn serialize<S: serde::Serializer>(
        &self,
        _serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        panic!("injected journal worker failure");
    }
}

async fn actual_cpu_request(fixture: &fixture::Fixture) {
    fixture.executor.remove_execution_gate();
    let output = tokio::time::timeout(
        Duration::from_secs(3),
        fixture.engine.infer(fixture.request()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(output.response().tokens.len(), 4);
    // Each output was compared bit-for-bit against a separate tiny CPU model,
    // with its physical KV frontier checked inside NumericalExecutor.
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 4);
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert!(fixture.engine.inner.sequences.is_empty());
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
}

async fn saturated_fanout(detail: ObservabilityProfileDetail) {
    let directory = tempfile::tempdir().unwrap();
    let blocked_path = directory.path().join("blocked.jsonl");
    let journal_config = JsonlJournalConfig::default();
    let blocked = JsonlJournal::<BlockWriter>::open(
        &blocked_path,
        JsonlJournalOpenMode::Truncate,
        journal_config,
    )
    .unwrap();
    let (entered, waiting) = tokio::sync::oneshot::channel();
    let (release, resumed) = std::sync::mpsc::channel();
    // The legacy tool producer deliberately holds the original writer. The
    // Engine attaches to this same physical journal and uses only try_enqueue.
    blocked
        .enqueue(BlockWriter {
            entered: Mutex::new(Some(entered)),
            release: Mutex::new(resumed),
        })
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    let filler = JsonlJournal::<serde_json::Value>::append(&blocked_path).unwrap();
    // Saturate the actual shared product budget. A conflicting one-slot
    // writer is now a configuration error, not a valid transport-loss fixture.
    for index in 0..journal_config.queue_capacity {
        filler
            .try_enqueue(serde_json::json!({"queued": index}))
            .unwrap();
    }
    let fixture = fixture::Fixture::with_config(|config| {
        config.runtime.profile_detail = detail;
        config.runtime.profile_jsonl = Some(blocked_path.clone());
    });
    // Declared after the fixture: unwind releases the real writer before any
    // fixture/journal owner can synchronously join it.
    let mut release = ReleaseWriter(Some(release));
    let blocked_profile = fixture.engine.inner.profile_trace_jsonl.clone().unwrap();
    let healthy = fixture.engine.inner.scheduler_trace_jsonl.clone().unwrap();
    let sink = VNextProfileExecutionEventSink::with_journals(
        vec![blocked_profile.clone(), healthy.clone()],
        ProfileEntrypoint::Run,
        &fixture.engine.inner.config,
    );
    let (_, _, accepted) = vnext_profile_test_event();
    let error = sink.enqueue_events(vec![accepted.clone()]).unwrap_err();
    assert_eq!(error.kind(), ExecutionEventSinkErrorKind::Transport);
    assert_eq!(blocked_profile.delivery_stats().dropped, 1);
    assert_eq!(
        healthy.delivery_stats().accepted,
        1,
        "failed first destination must not skip the healthy one"
    );
    assert_eq!(
        sink.delivery_policy(),
        if detail.requires_complete_evidence() {
            ExecutionEventDeliveryPolicy::RequiredEvidence
        } else {
            ExecutionEventDeliveryPolicy::OptionalTelemetry
        }
    );

    actual_cpu_request(&fixture).await;
    release.release();
    blocked.flush().unwrap();
    healthy.flush().unwrap();
    // Full was temporary: the original destination can accept later events.
    sink.enqueue_events(vec![accepted]).unwrap();
    blocked_profile.flush().unwrap();
    let inner = Arc::downgrade(&fixture.engine.inner);
    let observer = fixture
        .engine
        .driver
        .shutdown(&fixture.engine.inner)
        .unwrap();
    let report = tokio::time::timeout(Duration::from_secs(3), observer.wait())
        .await
        .unwrap();
    let stats = &report
        .journal_deliveries()
        .iter()
        .find(|journal| journal.path == blocked.path())
        .unwrap()
        .stats;
    assert_eq!(
        stats.dropped, 1,
        "successful close cannot erase earlier sample loss"
    );
    assert_eq!(stats.accepted, stats.flushed);
    assert!(stats.worker_finished);
    assert_eq!(stats.pending_payload_bytes, 0);
    if detail.requires_complete_evidence() {
        assert!(report.result().is_err());
    } else {
        report.result().unwrap();
        assert!(!report.telemetry_degradations().is_empty());
    }
    assert!(!report.retains_unproved_owners());
    drop(fixture.engine);
    assert!(
        inner.upgrade().is_none(),
        "diagnostic status must not retain a clean EngineInner"
    );
}

#[tokio::test]
async fn managed_observability_optional_full_fanout_preserves_real_cpu_and_history() {
    saturated_fanout(ObservabilityProfileDetail::Basic).await;
}

#[tokio::test]
async fn managed_observability_required_full_reports_loss_without_retaining_cpu_owner() {
    saturated_fanout(ObservabilityProfileDetail::Verify).await;
}

async fn closed_journal(detail: ObservabilityProfileDetail) {
    let fixture = fixture::Fixture::with_config(|config| config.runtime.profile_detail = detail);
    actual_cpu_request(&fixture).await;
    let journal = fixture.engine.inner.scheduler_trace_jsonl.as_ref().unwrap();
    let path = journal.path().to_path_buf();
    journal.close().unwrap();
    let sink = VNextProfileExecutionEventSink::new(
        journal.clone(),
        ProfileEntrypoint::Run,
        &fixture.engine.inner.config,
    );
    let (_, _, accepted) = vnext_profile_test_event();
    let error = sink.enqueue_events(vec![accepted]).unwrap_err();
    assert_eq!(error.kind(), ExecutionEventSinkErrorKind::Transport);
    let inner = Arc::downgrade(&fixture.engine.inner);
    let observer = fixture
        .engine
        .driver
        .shutdown(&fixture.engine.inner)
        .unwrap();
    let report = tokio::time::timeout(Duration::from_secs(3), observer.wait())
        .await
        .unwrap();
    let delivery = report
        .journal_deliveries()
        .iter()
        .find(|journal| journal.path == path)
        .unwrap();
    assert_eq!(delivery.stats.dropped, 1);
    assert!(delivery.stats.worker_finished);
    if detail.requires_complete_evidence() {
        assert!(report.result().is_err());
    } else {
        report.result().unwrap();
        assert!(!report.telemetry_degradations().is_empty());
    }
    assert!(!report.retains_unproved_owners());
    drop(fixture.engine);
    assert!(inner.upgrade().is_none());
}

#[tokio::test]
async fn managed_observability_optional_closed_sink_is_not_execution_failure() {
    closed_journal(ObservabilityProfileDetail::Basic).await;
}

#[tokio::test]
async fn managed_observability_required_closed_sink_reports_without_false_custody() {
    closed_journal(ObservabilityProfileDetail::Full).await;
}

#[tokio::test]
async fn managed_observability_worker_panic_is_contract_but_does_not_retain_clean_cpu() {
    let fixture = fixture::Fixture::with_config(|config| {
        config.runtime.profile_detail = ObservabilityProfileDetail::Basic
    });
    actual_cpu_request(&fixture).await;
    let path = fixture
        .engine
        .inner
        .scheduler_trace_jsonl
        .as_ref()
        .unwrap()
        .path()
        .to_path_buf();
    let poison = JsonlJournal::<PanicWriter>::append(&path).unwrap();
    poison.enqueue(PanicWriter).unwrap();
    let inner = Arc::downgrade(&fixture.engine.inner);
    let report = fixture
        .engine
        .driver
        .shutdown(&fixture.engine.inner)
        .unwrap()
        .wait()
        .await;
    let delivery = report
        .journal_deliveries()
        .iter()
        .find(|journal| journal.path == path)
        .unwrap();
    assert_eq!(
        delivery.stats.failure_kind,
        Some(JsonlJournalErrorKind::Panic)
    );
    assert!(delivery.stats.worker_finished);
    assert!(report.result().is_err());
    assert!(report.telemetry_degradations().is_empty());
    assert!(!report.retains_unproved_owners());
    drop(fixture.engine);
    assert!(inner.upgrade().is_none());
}

#[tokio::test]
async fn managed_observability_contract_is_rejected_before_enqueue_even_when_optional() {
    let fixture = fixture::Fixture::with_config(|config| {
        config.runtime.profile_detail = ObservabilityProfileDetail::Basic
    });
    actual_cpu_request(&fixture).await;
    let journal = fixture.engine.inner.scheduler_trace_jsonl.as_ref().unwrap();
    let sink = VNextProfileExecutionEventSink::new(
        journal.clone(),
        ProfileEntrypoint::Run,
        &fixture.engine.inner.config,
    );
    let (_, _, accepted) = vnext_profile_test_event();
    let mut invalid = sink.profile_event(&accepted).unwrap();
    invalid.event_id.clear();
    let before = journal.delivery_stats().accepted;
    let error = journal.enqueue(invalid).unwrap_err();
    assert_eq!(error.kind(), ExecutionEventSinkErrorKind::Contract);
    assert_eq!(journal.delivery_stats().accepted, before);
    fixture.engine.inner.report_observation_failure(error);
    // This is the production control-consumption boundary, independent of
    // whether a stream is being read or a journal is being flushed.
    let error = match fixture.engine.inner.run_iteration().await {
        Err(error) => error,
        Ok(_) => panic!("profile contract failure must reach the driver control boundary"),
    };
    assert!(error.to_string().contains("event_id"));
    let inner = Arc::downgrade(&fixture.engine.inner);
    let report = fixture
        .engine
        .driver
        .shutdown(&fixture.engine.inner)
        .unwrap()
        .wait()
        .await;
    assert!(report.result().is_err());
    assert!(report.telemetry_degradations().is_empty());
    assert!(!report.retains_unproved_owners());
    drop(fixture.engine);
    assert!(inner.upgrade().is_none());
}

#[tokio::test]
async fn managed_observability_optional_open_failure_is_observable_and_cpu_continues() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("not-a-directory");
    std::fs::write(&file, b"existing file").unwrap();
    let invalid = file.join("profile.jsonl");
    let fixture = fixture::Fixture::with_config(|config| {
        config.runtime.profile_detail = ObservabilityProfileDetail::Basic;
        config.runtime.profile_jsonl = Some(invalid.clone());
    });
    assert!(fixture.engine.inner.profile_trace_jsonl.is_none());
    actual_cpu_request(&fixture).await;
    let report = fixture
        .engine
        .driver
        .shutdown(&fixture.engine.inner)
        .unwrap()
        .wait()
        .await;
    report.result().unwrap();
    assert_eq!(report.optional_observation_failures(), 1);
    assert!(!report.telemetry_degradations().is_empty());
    let health = ObservationHealth::default();
    assert!(open_observation_journal(
        Some(&invalid),
        ObservabilityProfileDetail::Replay,
        &health,
        |path| SchedulerTraceJournal::create(path.to_path_buf(), Default::default())
    )
    .is_err());
}

#[tokio::test]
async fn managed_observability_reopening_failed_shared_writer_preserves_contract_kind() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed-writer.jsonl");
    let poison = JsonlJournal::<PanicWriter>::create(&path).unwrap();
    poison.enqueue(PanicWriter).unwrap();
    // A real barrier observes the worker failure without closing its registry
    // owner; the next open must attach/reject that same failed physical writer.
    let failure = poison.flush().unwrap_err();
    assert_eq!(failure.kind(), JsonlJournalErrorKind::Panic);
    let health = ObservationHealth::default();
    let reopened = open_observation_journal(
        Some(&path),
        ObservabilityProfileDetail::Basic,
        &health,
        |path| SchedulerTraceJournal::create(path.to_path_buf(), Default::default()),
    );
    assert!(
        reopened.is_err(),
        "optional mode cannot hide a shared writer contract failure"
    );
    let (required, optional, count) = health.snapshot();
    assert!(required.is_some());
    assert!(optional.is_none());
    assert_eq!(count, 0);
    assert_eq!(
        poison.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Panic
    );
}

#[test]
fn profile_journal_explicit_budget_conflict_is_contract_in_optional_mode() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("different-budget.jsonl");
    let original = JsonlJournal::<serde_json::Value>::open(
        &path,
        JsonlJournalOpenMode::Truncate,
        JsonlJournalConfig {
            queue_capacity: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let health = ObservationHealth::default();
    let opened = open_observation_journal(
        Some(&path),
        ObservabilityProfileDetail::Basic,
        &health,
        |path| SchedulerTraceJournal::create(path.to_path_buf(), Default::default()),
    );
    assert!(opened.is_err());
    let (required, optional, count) = health.snapshot();
    assert!(required.is_some());
    assert!(optional.is_none());
    assert_eq!(count, 0);
    original.close().unwrap();
}
