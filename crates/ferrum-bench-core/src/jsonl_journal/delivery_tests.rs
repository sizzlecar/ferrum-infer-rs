use super::*;
use serde::ser::SerializeStruct;
use std::sync::atomic::AtomicUsize;
use std::sync::Condvar;

fn path(label: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "ferrum-jsonl-try-{label}-{}-{}-{}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[derive(Default)]
pub(super) struct BlockingGate {
    state: Mutex<(bool, bool)>,
    wake: Condvar,
}
impl BlockingGate {
    pub(super) fn block(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.wake.notify_all();
        while !state.1 {
            state = self.wake.wait(state).unwrap();
        }
    }
    pub(super) fn wait(&self) {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .wake
            .wait_timeout_while(state, Duration::from_secs(2), |state| !state.0)
            .unwrap();
        assert!(state.0, "worker never reached the real serialization gate");
    }
    pub(super) fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.wake.notify_all();
    }
}
pub(super) struct ReleaseGate(pub(super) Arc<BlockingGate>);
impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.release();
    }
}

pub(super) fn nonblocking<T: Send + 'static>(operation: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = sync_channel(1);
    let task = thread::spawn(move || {
        let _ = tx.send(operation());
    });
    let result = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("try delivery must return while the original owner remains blocked");
    task.join().unwrap();
    result
}

pub(super) struct GatedEvent {
    text: String,
    gate: Arc<BlockingGate>,
    visits: AtomicUsize,
    measurement_first: bool,
}
impl Serialize for GatedEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let visit = self.visits.fetch_add(1, Ordering::Relaxed);
        if !self.measurement_first || visit > 0 {
            self.gate.block();
        }
        let mut value = serializer.serialize_struct("GatedEvent", 1)?;
        value.serialize_field("text", &self.text)?;
        value.end()
    }
}
pub(super) fn gated(gate: &Arc<BlockingGate>, measurement_first: bool) -> GatedEvent {
    GatedEvent {
        text: "x".repeat(160),
        gate: gate.clone(),
        visits: AtomicUsize::new(0),
        measurement_first,
    }
}

#[test]
fn try_delivery_full_and_gate_busy_return_without_poisoning_the_journal() {
    let path = path("full");
    let config = JsonlJournalConfig {
        queue_capacity: 1,
        ..Default::default()
    };
    let mut blocking =
        OwnedJsonlJournal::open(path.clone(), JsonlJournalOpenMode::Truncate, config).unwrap();
    let journal = blocking.scope.producer::<serde_json::Value>();
    let gate = Arc::new(BlockingGate::default());
    let release = ReleaseGate(gate.clone());
    blocking.enqueue(gated(&gate, false)).unwrap();
    gate.wait();
    journal
        .try_enqueue(serde_json::json!({"ordinal": 1}))
        .unwrap();
    let bytes = journal.delivery_stats().pending_payload_bytes;
    let send = journal.clone();
    let (tx, rx) = sync_channel(1);
    let worker = thread::spawn(move || {
        tx.send(send.try_enqueue(serde_json::json!({"ordinal": 2})))
            .unwrap();
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(1))
            .expect("full queue must not wait")
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Full
    );
    worker.join().unwrap();
    assert_eq!(journal.delivery_stats().pending_payload_bytes, bytes);
    let held_gate = journal.inner.send_gate.lock().unwrap();
    let send = journal.clone();
    assert_eq!(
        nonblocking(move || send.try_enqueue(serde_json::json!({"ordinal": 3})))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::GateBusy
    );
    drop(held_gate);
    drop(release);
    journal.flush().result().unwrap();
    journal
        .try_enqueue(serde_json::json!({"ordinal": 4}))
        .unwrap();
    blocking.close().unwrap().result().unwrap();
    let stats = journal.delivery_stats();
    assert_eq!((stats.accepted, stats.dropped, stats.failed), (2, 2, 0));
    assert_eq!(
        (stats.written, stats.flushed, stats.unbounded_accepted),
        (2, 2, 1)
    );
    assert_eq!(stats.pending_payload_bytes, 0);
    assert!(stats.worker_finished);
    assert_eq!(stats.worker_state, JsonlJournalWorkerState::Stopped);
    assert!(!stats.flush_pending);
    assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 3);
    let _ = std::fs::remove_file(path);
}

fn concurrent_measurement(retire_during_measurement: bool) {
    let path = path("concurrent-measurement");
    let owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let mut measured_scope = owner.producer_scope().unwrap();
    let mut peer_scope = owner.producer_scope().unwrap();
    let measured = measured_scope.producer::<GatedEvent>();
    let peer = peer_scope.producer::<serde_json::Value>();
    let gate = Arc::new(BlockingGate::default());
    let release = ReleaseGate(gate.clone());
    let pending_gate = gate.clone();
    let measuring = thread::spawn(move || {
        measured.try_enqueue_batch(vec![
            gated(&pending_gate, false),
            gated(&pending_gate, false),
        ])
    });
    gate.wait();
    // The original batch has not published anything. A real peer publication
    // and its FIFO flush must complete while that batch's Serialize is held.
    let sending = peer.clone();
    let peer_receipt = nonblocking(move || {
        sending.try_enqueue(serde_json::json!({"peer": 1})).unwrap();
        sending.flush()
    });
    peer_receipt.result().unwrap();
    assert_eq!(
        (
            peer_receipt.cut.accepted,
            peer_receipt.flushed,
            peer_receipt.cut.dropped
        ),
        (1, 1, 0)
    );
    assert_eq!(peer.delivery_stats().pending_payload_bytes, 0);
    let (mut owner, extra_scope) = nonblocking(move || {
        let scope = owner.producer_scope().unwrap();
        (owner, scope)
    });
    let extra_receipt = nonblocking(move || {
        let mut scope = extra_scope;
        scope.finish()
    });
    extra_receipt.result().unwrap();
    assert!(extra_receipt.scope_retired);
    assert_eq!(extra_receipt.cut.accepted, 1);

    let retired_receipt = if retire_during_measurement {
        let (scope, receipt) = nonblocking(move || {
            let receipt = measured_scope.finish();
            (measured_scope, receipt)
        });
        measured_scope = scope;
        receipt.result().unwrap();
        assert!(receipt.scope_retired && receipt.barrier_observed);
        Some(receipt)
    } else {
        None
    };
    drop(release);
    let result = measuring.join().unwrap();
    if let Some(receipt) = retired_receipt {
        assert_eq!(result.unwrap_err().kind(), JsonlJournalErrorKind::Closed);
        assert_eq!(
            (receipt.cut.accepted, receipt.cut.dropped, receipt.flushed),
            (1, 0, 1)
        );
        assert_eq!(
            measured_scope.finish(),
            receipt,
            "a late rejected batch cannot move the retired scope's cut"
        );
        assert_eq!(
            (
                peer.delivery_stats().accepted,
                peer.delivery_stats().dropped
            ),
            (1, 2)
        );
    } else {
        result.unwrap();
        let receipt = measured_scope.finish();
        receipt.result().unwrap();
        assert_eq!(
            (receipt.cut.accepted, receipt.cut.dropped, receipt.flushed),
            (3, 0, 3)
        );
    }
    peer_scope.finish().result().unwrap();
    let final_receipt = owner.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined);
    let stats = peer.delivery_stats();
    assert_eq!(stats.pending_payload_bytes, 0);
    assert_eq!(stats.failed, 0);
    assert_eq!(stats.accepted, stats.flushed);
    assert_eq!(stats.unbounded_accepted, 0);
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records[0], serde_json::json!({"peer":1}));
    assert_eq!(records.len(), if retire_during_measurement { 1 } else { 3 });
    let _ = std::fs::remove_file(path);
}

#[test]
fn try_delivery_measurement_does_not_hold_peer_publication_or_scope_barriers() {
    concurrent_measurement(false);
}

#[test]
fn try_delivery_scope_retired_during_measurement_rejects_whole_batch_and_returns_credit() {
    concurrent_measurement(true);
}

#[test]
fn try_delivery_measurement_cannot_publish_after_close_into_a_reopened_epoch() {
    let path = path("measurement-reopened");
    let owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let scope = owner.producer_scope().unwrap();
    let old = scope.producer::<serde_json::Value>();
    old.try_enqueue(serde_json::json!({"epoch": "old"}))
        .unwrap();
    old.flush().result().unwrap();
    let measured = scope.producer::<GatedEvent>();
    let gate = Arc::new(BlockingGate::default());
    let release = ReleaseGate(gate.clone());
    let pending_gate = gate.clone();
    let measuring = thread::spawn(move || measured.try_enqueue(gated(&pending_gate, false)));
    gate.wait();

    let (mut owner, mut scope, scope_receipt, final_receipt) = nonblocking(move || {
        let mut scope = scope;
        let scope_receipt = scope.finish();
        let mut owner = owner;
        let final_receipt = owner.close().unwrap();
        (owner, scope, scope_receipt, final_receipt)
    });
    scope_receipt.result().unwrap();
    final_receipt.result().unwrap();
    assert!(scope_receipt.scope_retired && scope_receipt.barrier_observed);
    assert!(final_receipt.worker_joined);
    assert_eq!(
        (
            final_receipt.flush.cut.accepted,
            final_receipt.flush.cut.dropped
        ),
        (1, 0)
    );

    let mut reopened =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let mut new_scope = reopened.producer_scope().unwrap();
    let current = new_scope.producer::<serde_json::Value>();
    assert!(!Arc::ptr_eq(&old.inner, &current.inner));
    current
        .try_enqueue(serde_json::json!({"epoch": "new"}))
        .unwrap();
    current.flush().result().unwrap();
    drop(release);
    assert_eq!(
        measuring.join().unwrap().unwrap_err().kind(),
        JsonlJournalErrorKind::Closed
    );
    assert_eq!(scope.finish(), scope_receipt);
    assert_eq!(
        owner.close().unwrap(),
        final_receipt,
        "late rejection cannot rewrite final evidence"
    );
    let old_stats = old.delivery_stats();
    assert_eq!(
        (old_stats.accepted, old_stats.flushed, old_stats.dropped),
        (1, 1, 1)
    );
    assert_eq!(old_stats.pending_payload_bytes, 0);
    new_scope.finish().result().unwrap();
    let new_final = reopened.close().unwrap();
    new_final.result().unwrap();
    assert!(new_final.worker_joined);
    assert_eq!(
        (
            new_final.flush.cut.accepted,
            new_final.flush.cut.dropped,
            new_final.flush.flushed
        ),
        (1, 0, 1)
    );
    assert_eq!(new_final.stats.pending_payload_bytes, 0);
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records, vec![serde_json::json!({"epoch": "new"})]);
    let _ = std::fs::remove_file(path);
}

#[test]
fn try_delivery_payload_credit_covers_the_active_job_and_releases_after_write() {
    let path = path("active-credit");
    let mut journal = OwnedJsonlJournal::open(
        path.clone(),
        JsonlJournalOpenMode::Truncate,
        JsonlJournalConfig {
            payload_capacity_bytes: 256,
            max_record_bytes: 256,
            max_batch_bytes: 256,
            ..Default::default()
        },
    )
    .unwrap();
    let gate = Arc::new(BlockingGate::default());
    let release = ReleaseGate(gate.clone());
    journal.try_enqueue(gated(&gate, true)).unwrap();
    gate.wait();
    let stats = journal.delivery_stats();
    assert_eq!(stats.accepted, 1);
    assert!(stats.pending_payload_bytes > 128);
    assert_eq!(stats.written, 0);
    let send = journal.producer().clone();
    let next_gate = gate.clone();
    assert_eq!(
        nonblocking(move || send.try_enqueue(gated(&next_gate, true)))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Full
    );
    assert_eq!(
        journal.delivery_stats().pending_payload_bytes,
        stats.pending_payload_bytes
    );
    drop(release);
    journal.flush().result().unwrap();
    assert_eq!(journal.delivery_stats().pending_payload_bytes, 0);
    journal.try_enqueue(gated(&gate, true)).unwrap();
    journal.close().unwrap().result().unwrap();
    let stats = journal.delivery_stats();
    assert_eq!(
        (stats.accepted, stats.dropped, stats.failed, stats.flushed),
        (2, 1, 0, 2)
    );
    assert!(stats.peak_payload_bytes <= 256);
    assert_eq!(stats.pending_payload_bytes, 0);
    let _ = std::fs::remove_file(path);
}

enum Event {
    Text(String),
    Failing,
}

#[test]
fn direct_records_keep_serialization_and_existing_worker_failure_categories() {
    let path = path("direct-error-kind");
    let error =
        write_jsonl_records(&path, JsonlJournalOpenMode::Truncate, &[Event::Failing]).unwrap_err();
    assert_eq!(error.kind(), JsonlJournalErrorKind::Encoding);
    assert!(
        !path.exists(),
        "invalid records must not create an artifact"
    );

    let mut journal = OwnedJsonlJournal::create(path.clone()).unwrap();
    journal.enqueue(Event::Failing).unwrap();
    assert_eq!(
        journal.flush().result().unwrap_err().kind(),
        JsonlJournalErrorKind::Encoding
    );
    let error =
        write_jsonl_records_to(journal.producer(), &[Event::Text("later".into())]).unwrap_err();
    assert_eq!(
        error.kind(),
        JsonlJournalErrorKind::Encoding,
        "attaching to the failed original writer must retain its cause"
    );
    assert_eq!(
        journal.close().unwrap().result().unwrap_err().kind(),
        JsonlJournalErrorKind::Encoding
    );
    let _ = std::fs::remove_file(path);
}
impl Serialize for Event {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(text) => text.serialize(serializer),
            Self::Failing => Err(serde::ser::Error::custom("intentional measurement failure")),
        }
    }
}

#[test]
fn try_delivery_limits_and_measurement_failures_reject_whole_batches_without_latching() {
    let path = path("limits");
    let mut journal = OwnedJsonlJournal::open(
        path.clone(),
        JsonlJournalOpenMode::Truncate,
        JsonlJournalConfig {
            payload_capacity_bytes: 200,
            max_record_bytes: 64,
            max_batch_bytes: 100,
            max_batch_records: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        journal
            .try_enqueue(Event::Text("x".repeat(64)))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Limit
    );
    assert_eq!(
        journal
            .try_enqueue_batch(vec![
                Event::Text("x".repeat(50)),
                Event::Text("y".repeat(50))
            ])
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Limit
    );
    assert_eq!(
        journal
            .try_enqueue_batch(vec![
                Event::Text("".into()),
                Event::Text("".into()),
                Event::Text("".into())
            ])
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Limit
    );
    assert_eq!(
        journal
            .try_enqueue_batch(vec![Event::Text("valid".into()), Event::Failing])
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Encoding
    );
    assert_eq!(journal.delivery_stats().pending_payload_bytes, 0);
    assert_eq!(journal.delivery_stats().failure_kind, None);
    journal
        .try_enqueue_batch(vec![
            Event::Text("字节".into()),
            Event::Text("complete".into()),
        ])
        .unwrap();
    journal.close().unwrap().result().unwrap();
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows,
        vec![serde_json::json!("字节"), serde_json::json!("complete")]
    );
    assert_eq!(journal.delivery_stats().dropped, 8);
    assert_eq!(journal.delivery_stats().flushed, 2);
    let _ = std::fs::remove_file(path);
}

struct ChangingEvent {
    visits: AtomicUsize,
    panic_on_worker: bool,
}
impl Serialize for ChangingEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.visits.fetch_add(1, Ordering::Relaxed) == 0 {
            return "small".serialize(serializer);
        }
        assert!(!self.panic_on_worker, "actual worker serializer panic");
        "larger than the measured payload".serialize(serializer)
    }
}

#[test]
fn try_delivery_worker_encoding_failure_or_panic_releases_exact_accepted_credit() {
    for panic_on_worker in [false, true] {
        let path = path("worker-failure");
        let mut journal = OwnedJsonlJournal::create(path.clone()).unwrap();
        journal
            .try_enqueue(ChangingEvent {
                visits: AtomicUsize::new(0),
                panic_on_worker,
            })
            .unwrap();
        let error = journal.close().unwrap().result().unwrap_err();
        assert_eq!(
            error.kind(),
            if panic_on_worker {
                JsonlJournalErrorKind::Panic
            } else {
                JsonlJournalErrorKind::Encoding
            }
        );
        let stats = journal.delivery_stats();
        assert_eq!(
            (stats.accepted, stats.failed, stats.written, stats.flushed),
            (1, 1, 0, 0)
        );
        assert_eq!(stats.pending_payload_bytes, 0);
        assert!(stats.worker_finished);
        assert!(stats.failure_kind.is_some());
        assert!(std::fs::read(&path).unwrap().is_empty());
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn try_delivery_close_waits_for_the_actual_worker_and_rejects_later_publication() {
    let path = path("close-race");
    let journal = OwnedJsonlJournal::create(path.clone()).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let release = ReleaseGate(gate.clone());
    journal.try_enqueue(gated(&gate, true)).unwrap();
    gate.wait();
    let OwnedJsonlJournal {
        producer: journal,
        scope,
        mut owner,
    } = journal;
    drop(scope);
    let (tx, rx) = sync_channel(1);
    let close = thread::spawn(move || {
        tx.send(owner.close()).unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while !journal.inner.closed.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    let send = journal.clone();
    let next_gate = gate.clone();
    assert_eq!(
        nonblocking(move || send.try_enqueue(gated(&next_gate, true)))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Closed
    );
    let stats = journal.delivery_stats();
    assert_eq!(stats.worker_state, JsonlJournalWorkerState::Closing);
    assert!(!stats.worker_finished);
    assert!(stats.flush_pending);
    assert!(stats.pending_payload_bytes > 0);
    assert!(rx.try_recv().is_err());
    drop(release);
    rx.recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap()
        .result()
        .unwrap();
    close.join().unwrap();
    let stats = journal.delivery_stats();
    assert!(stats.worker_finished);
    assert_eq!(
        (stats.flushed, stats.dropped, stats.pending_payload_bytes),
        (1, 1, 0)
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn try_delivery_flush_failure_is_not_reported_as_a_complete_artifact() {
    let path = path("flush-failure");
    drop(File::create(&path).unwrap());
    // The real BufWriter accepts this small payload, but its real File refuses
    // the eventual write. No platform-specific device or synthetic IO error.
    let mut journal = readonly_journal::<serde_json::Value>(&path, JsonlJournalConfig::default());
    journal
        .try_enqueue(serde_json::json!({"accepted": true}))
        .unwrap();
    assert_eq!(
        journal.flush().result().unwrap_err().kind(),
        JsonlJournalErrorKind::Worker
    );
    let stats = journal.delivery_stats();
    assert_eq!((stats.accepted, stats.written, stats.flushed), (1, 1, 0));
    assert_eq!(stats.failure_kind, Some(JsonlJournalErrorKind::Worker));
    assert!(stats.flush_pending);
    assert_eq!(stats.pending_payload_bytes, 0);
    assert_eq!(
        journal
            .try_enqueue(serde_json::json!({"later": true}))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Worker
    );
    assert!(journal.close().unwrap().result().is_err());
    assert!(journal.delivery_stats().worker_finished);
    assert!(std::fs::read(&path).unwrap().is_empty());
    let _ = std::fs::remove_file(path);
}

pub(super) fn readonly_journal<T: Serialize + Send + 'static>(
    path: &Path,
    config: JsonlJournalConfig,
) -> OwnedJsonlJournal<T> {
    let file = File::open(path).unwrap();
    let (sender, receiver) = sync_channel(config.queue_capacity);
    let failure = Arc::new(JournalFailure::new());
    let delivery = Arc::new(DeliveryState::new());
    let worker_failure = failure.clone();
    let worker_delivery = delivery.clone();
    let worker_path = path.to_path_buf();
    let worker = thread::spawn(move || {
        run_writer(
            file,
            receiver,
            worker_failure,
            &worker_path,
            config.buffer_capacity_bytes,
            config.flush_interval,
            worker_delivery,
        );
    });
    let owner = JsonlJournalOwner {
        inner: Arc::new(JsonlJournalInner {
            path: path.to_path_buf(),
            sender,
            send_gate: Mutex::new(PublicationState::default()),
            worker: Mutex::new(Some(worker)),
            failure,
            closed: AtomicBool::new(false),
            config,
            delivery,
        }),
    };
    let scope = owner.producer_scope().unwrap();
    let producer = scope.producer();
    OwnedJsonlJournal {
        producer,
        scope,
        owner,
    }
}

#[test]
fn journal_registry_poison_retains_typed_contract_source() {
    // Poison only a local registry, using the same lock boundary as open().
    let registry = Mutex::new(JournalRegistry::new());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = registry.lock().unwrap();
        panic!("local registry owner failed");
    }));
    assert!(poisoned.is_err());
    for _ in 0..2 {
        let error = match lock_journal_registry(&registry) {
            Ok(_) => panic!("a poisoned registry must not authorize opening a writer"),
            Err(error) => error,
        };
        let source = error
            .get_ref()
            .and_then(|source| source.downcast_ref::<JsonlJournalError>())
            .expect("the IO carrier must preserve its typed source");
        assert_eq!(source.kind(), JsonlJournalErrorKind::Panic);
        assert!(registry.is_poisoned());
    }
    let healthy = Mutex::new(JournalRegistry::new());
    assert!(lock_journal_registry(&healthy).unwrap().is_empty());
}

struct PanickingDropEvent;
impl Serialize for PanickingDropEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        "rejected payload".serialize(serializer)
    }
}
impl Drop for PanickingDropEvent {
    fn drop(&mut self) {
        panic!("rejected producer payload drop failed");
    }
}

#[test]
fn journal_producer_poison_still_drains_and_joins_on_close_or_unwind() {
    for explicit_close in [true, false] {
        let path = path("producer-poison");
        let config = JsonlJournalConfig {
            queue_capacity: 1,
            ..Default::default()
        };
        let blocking =
            OwnedJsonlJournal::open(path.clone(), JsonlJournalOpenMode::Truncate, config).unwrap();
        let journal = blocking.scope.producer::<serde_json::Value>();
        let crashing = blocking.scope.producer::<PanickingDropEvent>();
        assert!(Arc::ptr_eq(&blocking.producer.inner, &journal.inner));
        assert!(Arc::ptr_eq(&crashing.inner, &journal.inner));
        let gate = Arc::new(BlockingGate::default());
        let release = ReleaseGate(gate.clone());
        blocking.enqueue(gated(&gate, false)).unwrap();
        gate.wait();
        journal
            .try_enqueue(serde_json::json!({"ordinal": 1}))
            .unwrap();
        let original_bytes = journal.delivery_stats().pending_payload_bytes;
        assert!(original_bytes > 0);

        // Measurement succeeds. The full queue drops the rejected job while
        // publication still owns the gate: this is an actual producer unwind.
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crashing.try_enqueue(PanickingDropEvent)
        }));
        assert!(rejected.is_err());
        assert!(journal.inner.send_gate.is_poisoned());
        assert_eq!(
            journal
                .try_enqueue(serde_json::json!({"later": true}))
                .unwrap_err()
                .kind(),
            JsonlJournalErrorKind::Panic
        );
        assert_eq!(
            journal
                .enqueue(serde_json::json!({"blocking_later": true}))
                .unwrap_err()
                .kind(),
            JsonlJournalErrorKind::Panic
        );
        let stats = journal.delivery_stats();
        assert_eq!((stats.accepted, stats.dropped), (1, 2));
        assert_eq!(stats.pending_payload_bytes, original_bytes);
        assert!(!stats.worker_finished);
        assert_eq!(stats.failure_kind, Some(JsonlJournalErrorKind::Panic));

        let delivery = journal.inner.delivery.clone();
        let failure = journal.inner.failure.clone();
        let weak = Arc::downgrade(&journal.inner);
        drop(crashing);
        drop(journal);
        let mut journal = blocking;
        let (tx, rx) = sync_channel(1);
        let closing = thread::spawn(move || {
            if explicit_close {
                let error = journal.close().unwrap().result().unwrap_err();
                assert_eq!(error.kind(), JsonlJournalErrorKind::Panic);
                assert!(journal.producer.inner.send_gate.is_poisoned());
                assert!(journal.producer.inner.worker.lock().unwrap().is_none());
                drop(journal);
            } else {
                // The last journal owner is dropped during an unrelated unwind.
                // Teardown must join, not panic a second time or detach work.
                let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _last_owner = journal;
                    panic!("caller is unwinding");
                }));
                assert!(unwound.is_err());
            }
            tx.send(()).unwrap();
        });
        assert!(rx.try_recv().is_err());
        assert!(!delivery.snapshot(false, None).worker_finished);
        drop(release);
        rx.recv_timeout(Duration::from_secs(2))
            .expect("poison cleanup must finish");
        closing.join().unwrap();
        assert!(weak.upgrade().is_none());
        let stats = delivery.snapshot(true, failure.current_ref().map(JsonlJournalError::kind));
        assert!(stats.worker_finished);
        assert_eq!((stats.accepted, stats.written, stats.flushed), (1, 1, 1));
        assert_eq!(stats.pending_payload_bytes, 0);
        assert_eq!(stats.failure_kind, Some(JsonlJournalErrorKind::Panic));
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            rows,
            vec![
                serde_json::json!({"text": "x".repeat(160)}),
                serde_json::json!({"ordinal": 1}),
            ]
        );
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn journal_poisoned_worker_handle_is_recovered_only_to_join() {
    let path = path("worker-handle-poison");
    let mut journal = OwnedJsonlJournal::create(path.clone()).unwrap();
    journal
        .try_enqueue(serde_json::json!({"accepted": true}))
        .unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = journal.producer.inner.worker.lock().unwrap();
        panic!("local worker-handle owner failed");
    }));
    assert!(poisoned.is_err());
    assert_eq!(
        journal.close().unwrap().result().unwrap_err().kind(),
        JsonlJournalErrorKind::Panic
    );
    assert!(journal.producer.inner.worker.is_poisoned());
    let worker = match journal.producer.inner.worker.lock() {
        Ok(_) => panic!("cleanup must not clear poison"),
        Err(poisoned) => poisoned.into_inner(),
    };
    assert!(
        worker.is_none(),
        "the original worker must have been joined"
    );
    drop(worker);
    let stats = journal.delivery_stats();
    assert!(stats.worker_finished);
    assert_eq!(stats.pending_payload_bytes, 0);
    assert_eq!((stats.accepted, stats.written, stats.flushed), (1, 1, 1));
    assert_eq!(stats.failure_kind, Some(JsonlJournalErrorKind::Panic));
    drop(journal);
    let _ = std::fs::remove_file(path);
}

enum MixedFailureEvent {
    Write(Arc<BlockingGate>),
    Fail { panic: bool },
}
impl Serialize for MixedFailureEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Write(gate) => {
                gate.block();
                "first real file write".serialize(serializer)
            }
            Self::Fail { panic: true } => panic!("queued serializer panicked after IO failure"),
            Self::Fail { panic: false } => Err(serde::ser::Error::custom(
                "queued encoding failed after IO failure",
            )),
        }
    }
}

#[test]
fn journal_contract_failure_after_real_io_failure_is_not_transport() {
    for panic in [false, true] {
        let path = path("mixed-contract-failure");
        drop(File::create(&path).unwrap());
        let mut journal = readonly_journal::<MixedFailureEvent>(
            &path,
            JsonlJournalConfig {
                buffer_capacity_bytes: 1,
                ..Default::default()
            },
        );
        let bounded = journal.scope.producer::<serde_json::Value>();
        let gate = Arc::new(BlockingGate::default());
        let release = ReleaseGate(gate.clone());
        journal
            .enqueue(MixedFailureEvent::Write(gate.clone()))
            .unwrap();
        gate.wait();
        // Both later jobs are accepted before the first real write fails.
        journal.enqueue(MixedFailureEvent::Fail { panic }).unwrap();
        bounded
            .try_enqueue(serde_json::json!({"queued_owner": true}))
            .unwrap();
        assert!(bounded.delivery_stats().pending_payload_bytes > 0);
        drop(release);
        let error = journal.close().unwrap().result().unwrap_err();
        let expected = if panic {
            JsonlJournalErrorKind::Panic
        } else {
            JsonlJournalErrorKind::Encoding
        };
        assert_eq!(error.kind(), expected);
        let first = journal.producer.inner.failure.error.get().unwrap();
        assert_eq!(first.kind(), JsonlJournalErrorKind::Worker);
        assert!(first.message().contains("write JSONL event"));
        assert_eq!(
            journal
                .producer
                .inner
                .failure
                .contract_error
                .get()
                .unwrap()
                .kind(),
            expected
        );
        let stats = journal.delivery_stats();
        assert_eq!(stats.failure_kind, Some(expected));
        assert!(stats.worker_finished);
        assert_eq!(stats.unbounded_accepted, 2);
        assert_eq!(
            (stats.accepted, stats.failed, stats.written, stats.flushed),
            (1, 1, 0, 0)
        );
        assert_eq!(stats.pending_payload_bytes, 0);
        assert!(std::fs::read(&path).unwrap().is_empty());
        drop(bounded);
        drop(journal);
        let _ = std::fs::remove_file(path);
    }
}

struct PanickingMeasurement;
impl Serialize for PanickingMeasurement {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        panic!("measurement failed before publication");
    }
}

#[test]
fn journal_measurement_panic_does_not_poison_or_authorize_the_record() {
    let path = path("measurement-panic");
    let mut journal = OwnedJsonlJournal::<PanickingMeasurement>::create(path.clone()).unwrap();
    let valid = journal.scope.producer::<serde_json::Value>();
    assert_eq!(
        journal
            .try_enqueue(PanickingMeasurement)
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Panic
    );
    assert!(!journal.producer.inner.send_gate.is_poisoned());
    assert!(journal.producer.inner.failure.current().is_none());
    let rejected = journal.delivery_stats();
    assert_eq!(
        (
            rejected.accepted,
            rejected.dropped,
            rejected.pending_payload_bytes
        ),
        (0, 1, 0)
    );
    valid
        .try_enqueue(serde_json::json!({"valid": true}))
        .unwrap();
    journal.close().unwrap().result().unwrap();
    let stats = valid.delivery_stats();
    assert_eq!((stats.accepted, stats.flushed), (1, 1));
    assert_eq!(stats.failure_kind, None);
    assert_eq!(stats.pending_payload_bytes, 0);
    drop(valid);
    drop(journal);
    let _ = std::fs::remove_file(path);
}

#[test]
fn journal_reopen_waits_for_real_close_join_before_starting_another_epoch() {
    for poison_producer in [false, true] {
        let path = path("reopen-closing");
        let config = JsonlJournalConfig {
            queue_capacity: 1,
            ..Default::default()
        };
        let journal = OwnedJsonlJournal::<serde_json::Value>::open(
            path.clone(),
            JsonlJournalOpenMode::Truncate,
            config,
        )
        .unwrap();
        let sentinel = serde_json::json!({"original_epoch": true});
        journal.try_enqueue(sentinel.clone()).unwrap();
        journal.flush().result().unwrap();
        let original_file = std::fs::read(&path).unwrap();
        let blocking = journal.scope.producer::<GatedEvent>();
        assert!(Arc::ptr_eq(&journal.producer.inner, &blocking.inner));
        let gate = Arc::new(BlockingGate::default());
        let release = ReleaseGate(gate.clone());
        blocking.enqueue(gated(&gate, false)).unwrap();
        gate.wait();
        journal
            .try_enqueue(serde_json::json!({"ordinal": 1}))
            .unwrap();
        if poison_producer {
            let crashing = journal.scope.producer::<PanickingDropEvent>();
            // A full queue drops the rejected payload under the actual gate.
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crashing.try_enqueue(PanickingDropEvent)
            }))
            .is_err());
            assert!(journal.producer.inner.send_gate.is_poisoned());
        }
        let OwnedJsonlJournal {
            producer: journal,
            scope,
            mut owner,
        } = journal;
        drop(scope);
        let (tx, rx) = sync_channel(1);
        let close = thread::spawn(move || {
            tx.send(owner.close()).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while !journal.inner.closed.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "close never acquired its original gate"
            );
            thread::yield_now();
        }
        for mode in [JsonlJournalOpenMode::Append, JsonlJournalOpenMode::Truncate] {
            let reopen_path = path.clone();
            let error = nonblocking(move || {
                OwnedJsonlJournal::<serde_json::Value>::open(reopen_path, mode, config)
                    .err()
                    .expect("an unjoined writer must not be replaced")
            });
            let error = error
                .get_ref()
                .and_then(|source| source.downcast_ref::<JsonlJournalError>())
                .expect("closing rejection must preserve its typed source");
            assert_eq!(
                error.kind(),
                if poison_producer {
                    JsonlJournalErrorKind::Panic
                } else {
                    JsonlJournalErrorKind::Closed
                }
            );
            assert_eq!(std::fs::read(&path).unwrap(), original_file);
            let registered = lock_journal_registry(JSONL_JOURNAL_REGISTRY.get().unwrap())
                .unwrap()
                .get(journal.path())
                .and_then(Weak::upgrade)
                .unwrap();
            assert!(Arc::ptr_eq(&registered, &journal.inner));
            assert!(journal.inner.worker.lock().unwrap().is_some());
            assert!(!journal.delivery_stats().worker_finished);
            assert!(rx.try_recv().is_err());
        }
        drop(release);
        let result = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        close.join().unwrap();
        if poison_producer {
            assert_eq!(
                result.unwrap().result().unwrap_err().kind(),
                JsonlJournalErrorKind::Panic
            );
            assert!(journal.inner.send_gate.is_poisoned());
        } else {
            result.unwrap().result().unwrap();
        }
        assert!(journal.inner.worker.lock().unwrap().is_none());
        assert!(journal.delivery_stats().worker_finished);
        assert_eq!(journal.delivery_stats().pending_payload_bytes, 0);
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(
            rows,
            vec![
                sentinel,
                serde_json::json!({"text": "x".repeat(160)}),
                serde_json::json!({"ordinal": 1}),
            ]
        );
        if poison_producer {
            let error = OwnedJsonlJournal::<serde_json::Value>::append(path.clone())
                .err()
                .expect("Append must retain the failed artifact's original cause");
            let error = error
                .get_ref()
                .and_then(|source| source.downcast_ref::<JsonlJournalError>())
                .unwrap();
            assert_eq!(error, &journal.inner.failure.current().unwrap());
            assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 3);
        } else {
            let mut appended =
                OwnedJsonlJournal::<serde_json::Value>::append(path.clone()).unwrap();
            assert!(!Arc::ptr_eq(&appended.producer.inner, &journal.inner));
            appended
                .try_enqueue(serde_json::json!({"next_append": true}))
                .unwrap();
            appended.close().unwrap().result().unwrap();
            assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 4);
        }
        let mut next = OwnedJsonlJournal::<serde_json::Value>::open(
            path.clone(),
            JsonlJournalOpenMode::Truncate,
            config,
        )
        .unwrap();
        assert!(!Arc::ptr_eq(&next.producer.inner, &journal.inner));
        assert!(std::fs::read(&path).unwrap().is_empty());
        next.try_enqueue(serde_json::json!({"new_epoch": true}))
            .unwrap();
        next.close().unwrap().result().unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"new_epoch\":true}\n"
        );
        assert_eq!(journal.inner.send_gate.is_poisoned(), poison_producer);
        drop(next);
        drop(blocking);
        drop(journal);
        let _ = std::fs::remove_file(path);
    }
}
