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
struct BlockingGate {
    state: Mutex<(bool, bool)>,
    wake: Condvar,
}
impl BlockingGate {
    fn block(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        self.wake.notify_all();
        while !state.1 {
            state = self.wake.wait(state).unwrap();
        }
    }
    fn wait(&self) {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .wake
            .wait_timeout_while(state, Duration::from_secs(2), |state| !state.0)
            .unwrap();
        assert!(state.0, "worker never reached the real serialization gate");
    }
    fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.wake.notify_all();
    }
}
struct ReleaseGate(Arc<BlockingGate>);
impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn nonblocking<T: Send + 'static>(operation: impl FnOnce() -> T + Send + 'static) -> T {
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

struct GatedEvent {
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
fn gated(gate: &Arc<BlockingGate>, measurement_first: bool) -> GatedEvent {
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
    let blocking =
        JsonlJournal::open(path.clone(), JsonlJournalOpenMode::Truncate, config).unwrap();
    let journal = JsonlJournal::<serde_json::Value>::append(path.clone()).unwrap();
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
    journal.flush().unwrap();
    journal
        .try_enqueue(serde_json::json!({"ordinal": 4}))
        .unwrap();
    journal.close().unwrap();
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

#[test]
fn try_delivery_payload_credit_covers_the_active_job_and_releases_after_write() {
    let path = path("active-credit");
    let journal = JsonlJournal::open(
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
    let send = journal.clone();
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
    journal.flush().unwrap();
    assert_eq!(journal.delivery_stats().pending_payload_bytes, 0);
    journal.try_enqueue(gated(&gate, true)).unwrap();
    journal.close().unwrap();
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

    let journal = JsonlJournal::create(path.clone()).unwrap();
    journal.enqueue(Event::Failing).unwrap();
    assert_eq!(
        journal.flush().unwrap_err().kind(),
        JsonlJournalErrorKind::Encoding
    );
    let error = write_jsonl_records(
        &path,
        JsonlJournalOpenMode::Append,
        &[Event::Text("later".into())],
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        JsonlJournalErrorKind::Encoding,
        "attaching to the failed original writer must retain its cause"
    );
    assert_eq!(
        journal.close().unwrap_err().kind(),
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
    let journal = JsonlJournal::open(
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
    journal.close().unwrap();
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
        let journal = JsonlJournal::create(path.clone()).unwrap();
        journal
            .try_enqueue(ChangingEvent {
                visits: AtomicUsize::new(0),
                panic_on_worker,
            })
            .unwrap();
        let error = journal.close().unwrap_err();
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
    let journal = JsonlJournal::create(path.clone()).unwrap();
    let gate = Arc::new(BlockingGate::default());
    let release = ReleaseGate(gate.clone());
    journal.try_enqueue(gated(&gate, true)).unwrap();
    gate.wait();
    let closing = journal.clone();
    let (tx, rx) = sync_channel(1);
    let close = thread::spawn(move || {
        tx.send(closing.close()).unwrap();
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
    rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
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
    let file = File::open(&path).unwrap();
    let config = JsonlJournalConfig::default();
    let (sender, receiver) = sync_channel(config.queue_capacity);
    let failure = Arc::new(JournalFailure::new());
    let delivery = Arc::new(DeliveryState::new());
    let worker_failure = failure.clone();
    let worker_delivery = delivery.clone();
    let worker_path = path.clone();
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
    let journal = JsonlJournal::<serde_json::Value> {
        inner: Arc::new(JsonlJournalInner {
            path: path.clone(),
            sender,
            send_gate: Mutex::new(()),
            worker: Mutex::new(Some(worker)),
            failure,
            closed: AtomicBool::new(false),
            config,
            delivery,
        }),
        marker: PhantomData,
    };
    journal
        .try_enqueue(serde_json::json!({"accepted": true}))
        .unwrap();
    assert_eq!(
        journal.flush().unwrap_err().kind(),
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
    assert!(journal.close().is_err());
    assert!(journal.delivery_stats().worker_finished);
    assert!(std::fs::read(&path).unwrap().is_empty());
    let _ = std::fs::remove_file(path);
}
