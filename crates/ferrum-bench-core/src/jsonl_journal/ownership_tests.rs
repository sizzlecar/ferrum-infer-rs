use super::delivery_tests::{gated, nonblocking, BlockingGate, GatedEvent, ReleaseGate};
use super::*;

fn path(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "ferrum-journal-ownership-{label}-{}-{}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn journal_scope_finish_keeps_peer_live_and_final_close_requires_all_scopes() {
    let path = path("two-scopes");
    let mut owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let mut a = owner.producer_scope().unwrap();
    let mut b = owner.producer_scope().unwrap();
    let pa = a.producer::<serde_json::Value>();
    let pb = b.producer::<serde_json::Value>();
    let late = pa.clone();
    assert!(Arc::ptr_eq(&pa.inner, &pb.inner));
    pa.try_enqueue(serde_json::json!({"a": 1})).unwrap();
    pb.enqueue(serde_json::json!({"b": 1})).unwrap();
    assert_eq!(
        owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config
    );
    assert!(!owner.inner.closed.load(Ordering::Acquire));
    let first = a.finish();
    first.result().unwrap();
    assert!(first.scope_retired && first.barrier_observed);
    assert_eq!(
        first.cut,
        JsonlPublicationCut {
            accepted: 1,
            unbounded_accepted: 1,
            dropped: 0
        }
    );
    assert_eq!(first.flushed, 1);
    assert_eq!(
        late.try_enqueue(serde_json::json!({"late": true}))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Closed
    );
    pb.try_enqueue(serde_json::json!({"b": 2})).unwrap();
    assert_eq!(
        a.finish(),
        first,
        "finish must return its original immutable prefix"
    );
    assert_eq!(
        owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config
    );
    let second = b.finish();
    second.result().unwrap();
    assert_eq!(
        second.cut,
        JsonlPublicationCut {
            accepted: 2,
            unbounded_accepted: 1,
            dropped: 1
        }
    );
    let final_receipt = owner.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined && final_receipt.stats.worker_finished);
    assert_eq!(final_receipt.stats.pending_payload_bytes, 0);
    assert_eq!(final_receipt.stats.dropped, 1);
    assert_eq!(owner.close().unwrap(), final_receipt);
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows,
        vec![
            serde_json::json!({"a":1}),
            serde_json::json!({"b":1}),
            serde_json::json!({"b":2})
        ]
    );
    drop((pa, pb, late, a, b, owner));
    let _ = std::fs::remove_file(path);
}

#[test]
fn journal_scope_receipt_keeps_enqueue_cut_when_peer_has_later_pending_work_and_drops() {
    let path = path("prefix-cut");
    let mut owner = JsonlJournalOwner::open(
        &path,
        JsonlJournalOpenMode::Truncate,
        JsonlJournalConfig {
            queue_capacity: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let mut a = owner.producer_scope().unwrap();
    let mut b = owner.producer_scope().unwrap();
    let pa = a.producer::<serde_json::Value>();
    let blocked_a = a.producer::<GatedEvent>();
    let pb = b.producer::<GatedEvent>();
    let first_gate = Arc::new(BlockingGate::default());
    let first_release = ReleaseGate(first_gate.clone());
    let second_gate = Arc::new(BlockingGate::default());
    let second_release = ReleaseGate(second_gate.clone());
    pa.try_enqueue(serde_json::json!({"before": true})).unwrap();
    blocked_a.enqueue(gated(&first_gate, false)).unwrap();
    first_gate.wait();
    let state = a.state.clone();
    let (tx, rx) = sync_channel(1);
    let finishing = thread::spawn(move || {
        let receipt = a.finish();
        tx.send((a, receipt)).unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while !state.retired.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    // Acquiring the same gate proves finish has queued its barrier. The first
    // actual serializer is still blocked, so the worker has not reached it.
    let second = pb.clone();
    let next_gate = second_gate.clone();
    nonblocking(move || {
        let gate = second.inner.send_gate.lock().unwrap();
        drop(gate);
        second.try_enqueue(gated(&next_gate, true))
    })
    .unwrap();
    assert_eq!(
        pa.try_enqueue(serde_json::json!({"late": true}))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Closed
    );
    drop(first_release);
    let (a, receipt) = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    finishing.join().unwrap();
    second_gate.wait();
    receipt.result().unwrap();
    assert_eq!(
        receipt.cut,
        JsonlPublicationCut {
            accepted: 1,
            unbounded_accepted: 1,
            dropped: 0
        }
    );
    assert_eq!((receipt.flushed, receipt.failed), (1, 0));
    assert!(receipt.scope_retired && receipt.barrier_observed);
    let stats = owner.delivery_stats();
    assert_eq!(stats.unbounded_accepted, 1);
    assert_eq!(stats.accepted, 2);
    assert!(stats.pending_payload_bytes > 0);
    assert_eq!(stats.dropped, 1);
    assert!(!stats.worker_finished);
    assert_eq!(
        owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config
    );
    drop(second_release);
    b.finish().result().unwrap();
    owner.close().unwrap().result().unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 3);
    drop((pa, blocked_a, pb, a, b, owner));
    let _ = std::fs::remove_file(path);
}

#[test]
fn journal_owner_drop_preserves_live_scope_and_scope_drop_only_revokes_publication() {
    let path = path("owner-drop");
    let owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let scope = owner.producer_scope().unwrap();
    let producer = scope.producer::<serde_json::Value>();
    let weak = Arc::downgrade(&owner.inner);
    drop(owner);
    producer
        .try_enqueue(serde_json::json!({"after_owner_drop": true}))
        .unwrap();
    assert!(!producer.inner.closed.load(Ordering::Acquire));
    drop(scope);
    assert!(producer.scope.retired.load(Ordering::Acquire));
    assert_eq!(
        producer
            .try_enqueue(serde_json::json!({"after_scope_drop": true}))
            .unwrap_err()
            .kind(),
        JsonlJournalErrorKind::Closed
    );
    assert!(!producer.flush().barrier_observed);
    assert!(weak.upgrade().is_some());
    drop(producer);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "{\"after_owner_drop\":true}\n"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn journal_scope_retirement_and_final_join_survive_real_file_flush_failure() {
    let path = path("flush-failure");
    drop(File::create(&path).unwrap());
    let OwnedJsonlJournal {
        producer,
        mut scope,
        mut owner,
    } = super::delivery_tests::readonly_journal::<serde_json::Value>(&path, Default::default());
    let mut peer = owner.producer_scope().unwrap();
    producer
        .try_enqueue(serde_json::json!({"actual_io": true}))
        .unwrap();
    let receipt = scope.finish();
    assert!(receipt.scope_retired && receipt.barrier_observed);
    assert_eq!(
        receipt.result().unwrap_err().kind(),
        JsonlJournalErrorKind::Worker
    );
    assert_eq!(scope.finish(), receipt);
    assert_eq!(
        owner.close().unwrap_err().kind(),
        JsonlJournalErrorKind::Config
    );
    let peer_receipt = peer.finish();
    assert!(peer_receipt.scope_retired && peer_receipt.barrier_observed);
    assert_eq!(
        peer_receipt.result().unwrap_err().kind(),
        JsonlJournalErrorKind::Worker
    );
    let final_receipt = owner.close().unwrap();
    assert!(final_receipt.worker_joined && final_receipt.stats.worker_finished);
    assert_eq!(
        final_receipt.result().unwrap_err().kind(),
        JsonlJournalErrorKind::Worker
    );
    assert_eq!(final_receipt.stats.pending_payload_bytes, 0);
    drop((producer, scope, peer, owner));
    let _ = std::fs::remove_file(path);
}

struct WorkerPanic;
impl Serialize for WorkerPanic {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        panic!("actual worker serialization panic")
    }
}
#[test]
fn journal_lost_barrier_keeps_scope_retired_and_real_panicked_worker_joined() {
    let path = path("lost-barrier");
    let mut owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Truncate, Default::default()).unwrap();
    let mut scope = owner.producer_scope().unwrap();
    let producer = scope.producer::<WorkerPanic>();
    producer.enqueue(WorkerPanic).unwrap();
    let receipt = scope.finish();
    assert!(receipt.scope_retired);
    assert!(!receipt.barrier_observed);
    assert_eq!(
        receipt.result().unwrap_err().kind(),
        JsonlJournalErrorKind::Panic
    );
    let final_receipt = owner.close().unwrap();
    assert!(final_receipt.worker_joined && final_receipt.stats.worker_finished);
    assert_eq!(
        final_receipt.result().unwrap_err().kind(),
        JsonlJournalErrorKind::Panic
    );
    assert_eq!(final_receipt.stats.pending_payload_bytes, 0);
    drop((producer, scope, owner));
    let _ = std::fs::remove_file(path);
}
