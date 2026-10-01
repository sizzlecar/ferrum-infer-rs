use super::*;
use std::sync::{mpsc, Barrier};
use std::time::Duration;

#[test]
fn queue_handoff_consumer_control_never_rejects_producer() {
    let sink = sink(4, 64);
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let (held, waiting) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    let worker_sink = Arc::clone(&sink);
    let worker = std::thread::spawn(move || {
        worker_sink.with_locked_consumer(|| {
            held.send(()).unwrap();
            resume.recv_timeout(Duration::from_secs(3)).unwrap();
        })
    });
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    for _ in 0..3 {
        assert_eq!(
            completed(&shape, &sink).finish(),
            CostCallDisposition::Queued
        );
    }
    release.send(()).unwrap();
    worker.join().unwrap();
    for ordinal in 1..=3 {
        assert_eq!(sink.pop_resolved_bound().unwrap().0, ordinal);
    }
    assert!(sink.pop_resolved_bound().is_none());
    assert!(!sink.stats().has_lost_samples());
    assert_eq!(sink.stats().entries_dropped_contention, 0);
}

#[test]
fn queue_handoff_uncommitted_slot_is_invisible_and_full_send_never_accepts() {
    let sink = sink(1, 64);
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let (sent, waiting) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    sink.on_next_send(move || {
        sent.send(()).unwrap();
        resume.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let call = completed(&shape, &sink);
    let producer = std::thread::spawn(move || call.finish());
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(sink.pop_resolved_bound().is_none());
    assert!(
        sink.needs_drain(0),
        "an in-progress producer still needs drain"
    );
    release.send(()).unwrap();
    assert_eq!(producer.join().unwrap(), CostCallDisposition::Queued);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    );
    assert_eq!(sink.pop_resolved_bound().unwrap().0, 1);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Queued
    );
    assert_eq!(sink.pop_resolved_bound().unwrap().0, 2);
    let stats = sink.stats();
    assert_eq!(stats.entries_published, 2);
    assert_eq!(stats.entries_dropped_capacity, 1);
    assert_eq!(stats.memory.queued_raw_bytes, 0);
}

#[test]
fn queue_handoff_producer_unwind_advances_failed_position_without_training() {
    let sink = sink(2, 64);
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    sink.on_next_send(|| panic!("injected producer unwind after successful channel send"));
    let failed = completed(&shape, &sink);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| failed.finish())).is_err());
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Queued
    );
    let (ordinal, entry, ticket, _, proof) = sink.pop_resolved_bound().unwrap();
    assert_eq!(ordinal, 1);
    assert!(entry.is_none() && ticket.is_none() && proof.is_none());
    let (ordinal, entry, ..) = sink.pop_resolved_bound().unwrap();
    assert_eq!(ordinal, 2);
    assert!(entry.is_some());
    drop(entry);
    assert!(!sink.needs_drain(2));
    let stats = sink.stats();
    assert_eq!(stats.entries_abandoned, 1);
    assert_eq!(stats.raw_abandoned, 1);
    assert_eq!(stats.raw_lost, 1);
    assert_eq!(stats.raw_resolved, 1);
    assert_eq!(stats.raw_pending, 0);
    assert_eq!(stats.published, 1);
    assert!(stats.has_lost_samples());
    assert_eq!(stats.memory.queued_raw_bytes, 0);
}

#[test]
fn queue_handoff_worker_stop_waits_for_inflight_publication_and_releases_leases() {
    let sink = sink(2, 64);
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let (sent, waiting) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    sink.on_next_send(move || {
        sent.send(()).unwrap();
        resume.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let call = completed(&shape, &sink);
    let producer = std::thread::spawn(move || call.finish());
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    let waiter = sink.request_checkpoint().unwrap();
    let stopper_sink = Arc::clone(&sink);
    let started = Arc::new(Barrier::new(2));
    let stopper_started = Arc::clone(&started);
    let stopper = std::thread::spawn(move || {
        stopper_started.wait();
        stopper_sink.worker_stopped();
    });
    started.wait();
    release.send(()).unwrap();
    assert_eq!(producer.join().unwrap(), CostCallDisposition::Queued);
    stopper.join().unwrap();
    drop(waiter);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::WorkerStopped)
    );
    let stats = sink.stats();
    assert_eq!(stats.raw_pending, 0);
    assert_eq!(stats.raw_abandoned, 1);
    assert_eq!(stats.memory.queued_raw_bytes, 0);
    assert!(matches!(
        sink.request_checkpoint(),
        Err(super::super::checkpoint::CheckpointRequestError::Closing)
    ));
}

#[test]
fn queue_handoff_popped_raw_lease_still_blocks_byte_admission() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let probe = sink(2, 64);
    assert_eq!(
        completed(&shape, &probe).finish(),
        CostCallDisposition::Queued
    );
    let one_entry = probe.stats().memory.queued_raw_bytes;
    assert!(one_entry > 0);
    probe.pop_resolved_bound().unwrap();
    let sink = Arc::new(
        BoundedCostSampleSink::new_with_byte_limits(
            CostSampleSinkLimits {
                max_samples: 2,
                max_shape_rows: 64,
            },
            CostRecorderByteLimits {
                maximum_retained_bytes: one_entry,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Queued
    );
    let (popped, waiting) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    sink.on_next_pop(move || {
        popped.send(()).unwrap();
        resume.recv_timeout(Duration::from_secs(3)).unwrap();
    });
    let consumer_sink = Arc::clone(&sink);
    let consumer = std::thread::spawn(move || consumer_sink.pop_resolved_bound().unwrap().0);
    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(sink.stats().memory.queued_raw_bytes, one_entry);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    );
    release.send(()).unwrap();
    assert_eq!(consumer.join().unwrap(), 1);
    assert_eq!(sink.stats().memory.queued_raw_bytes, 0);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Queued
    );
    assert_eq!(sink.pop_resolved_bound().unwrap().0, 2);
}
