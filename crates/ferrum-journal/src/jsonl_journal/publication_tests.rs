//! Concurrent producers share the original publication cutoff and writer.
use super::delivery_tests::{gated, nonblocking, BlockingGate, GatedEvent, ReleaseGate};
use super::*;

fn path(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "ferrum-journal-publication-{label}-{}-{}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

struct GatedDropEvent(Arc<BlockingGate>);

impl Serialize for GatedDropEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        "rejected publication".serialize(serializer)
    }
}

impl Drop for GatedDropEvent {
    fn drop(&mut self) {
        self.0.block();
    }
}

#[test]
fn journal_concurrent_producer_publishes_while_peer_releases_rejected_payload() {
    let path = path("concurrent-cutoff");
    let mut owner = JsonlJournalOwner::open(
        &path,
        JsonlJournalOpenMode::Truncate,
        JsonlJournalConfig {
            queue_capacity: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut first = owner.producer_scope().unwrap();
    let mut peer = owner.producer_scope().unwrap();
    let blocking = first.producer::<GatedEvent>();
    let late = first.producer::<serde_json::Value>();
    let rejecting = first.producer::<GatedDropEvent>();
    let publishing = peer.producer::<serde_json::Value>();
    let first_gate = Arc::new(BlockingGate::default());
    let first_release = ReleaseGate(first_gate.clone());
    let second_gate = Arc::new(BlockingGate::default());
    let second_release = ReleaseGate(second_gate.clone());
    let rejected_gate = Arc::new(BlockingGate::default());
    let rejected_release = ReleaseGate(rejected_gate.clone());

    // A real blocked serializer and one queued job make the next actual
    // try_send fail Full. Its payload destructor pauses before the original
    // publication permit and byte credit can be relinquished.
    blocking.enqueue(gated(&first_gate, false)).unwrap();
    first_gate.wait();
    blocking.try_enqueue(gated(&second_gate, true)).unwrap();
    let credit_before_rejection = owner.delivery_stats().pending_payload_bytes;
    let (rejected_tx, rejected_rx) = sync_channel(1);
    let drop_gate = rejected_gate.clone();
    let reject_thread = thread::spawn(move || {
        rejected_tx
            .send(rejecting.try_enqueue(GatedDropEvent(drop_gate)))
            .unwrap();
    });
    rejected_gate.wait();
    assert_eq!(owner.delivery_stats().dropped, 1);
    assert!(owner.delivery_stats().pending_payload_bytes > credit_before_rejection);

    // The worker moves to the second real job, leaving an empty queue while
    // the rejected producer still owns its publication permit. An independent
    // component must publish successfully without waiting for that producer.
    drop(first_release);
    second_gate.wait();
    let send = publishing.clone();
    nonblocking(move || send.try_enqueue(serde_json::json!({"peer": 1}))).unwrap();
    assert_eq!(owner.delivery_stats().accepted, 2);
    assert!(rejected_rx.try_recv().is_err());

    // Retirement cannot pass an in-progress publication. It must take the
    // completed acceptance/rollback cut and queue its original FIFO barrier.
    let retired = first.state.clone();
    let (started_tx, started_rx) = sync_channel(1);
    let (finished_tx, finished_rx) = sync_channel(1);
    let finish_thread = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let receipt = first.finish();
        finished_tx.send((first, receipt)).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(!retired.retired.load(Ordering::Acquire));
    assert!(finished_rx.try_recv().is_err());
    drop(rejected_release);
    assert_eq!(
        rejected_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Full
    );
    reject_thread.join().unwrap();
    drop(second_release);
    let (mut first, receipt) = finished_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    finish_thread.join().unwrap();
    receipt.result().unwrap();
    assert!(receipt.scope_retired && receipt.barrier_observed);
    assert_eq!(
        receipt.cut,
        JsonlPublicationCut {
            accepted: 2,
            unbounded_accepted: 1,
            dropped: 1,
        }
    );
    assert_eq!((receipt.flushed, receipt.failed), (2, 0));
    assert_eq!(owner.delivery_stats().pending_payload_bytes, 0);
    assert_eq!(
        late.try_enqueue(serde_json::json!({"late": true}))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Closed
    );
    publishing
        .try_enqueue(serde_json::json!({"peer": 2}))
        .unwrap();
    assert_eq!(first.finish(), receipt, "the earlier cut is immutable");
    peer.finish().result().unwrap();
    let final_receipt = owner.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined && final_receipt.stats.worker_finished);
    assert_eq!(
        (
            final_receipt.stats.accepted,
            final_receipt.stats.flushed,
            final_receipt.stats.dropped
        ),
        (3, 3, 2)
    );
    assert_eq!(final_receipt.stats.pending_payload_bytes, 0);
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        records,
        vec![
            serde_json::json!({"text": "x".repeat(160)}),
            serde_json::json!({"text": "x".repeat(160)}),
            serde_json::json!({"peer": 1}),
            serde_json::json!({"peer": 2}),
        ]
    );
    drop((blocking, late, publishing, first, peer, owner));
    let _ = std::fs::remove_file(path);
}

#[test]
fn journal_exclusive_publication_poison_rejects_producers_but_still_joins() {
    let path = path("exclusive-poison");
    let mut journal = OwnedJsonlJournal::<serde_json::Value>::create(&path).unwrap();
    journal
        .try_enqueue(serde_json::json!({"original": true}))
        .unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _gate = journal.producer.inner.send_gate.write().unwrap();
        panic!("cold publication owner failed");
    }))
    .is_err());
    assert!(journal.producer.inner.send_gate.is_poisoned());
    assert_eq!(
        journal
            .try_enqueue(serde_json::json!({"late": true}))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Panic
    );
    let receipt = journal.close().unwrap();
    assert_eq!(
        receipt.result().unwrap_err().kind(),
        JsonlJournalErrorKind::Panic
    );
    assert!(receipt.worker_joined && receipt.stats.worker_finished);
    assert!(journal.producer.inner.worker.lock().unwrap().is_none());
    assert_eq!(receipt.stats.pending_payload_bytes, 0);
    assert_eq!(receipt.stats.accepted, 1);
    drop(journal);
    let _ = std::fs::remove_file(path);
}
