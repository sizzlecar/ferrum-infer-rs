use super::*;
use ferrum_interfaces::execution_cost::{
    PendingActualWave, PendingActualWaveProjection, PendingWaveBounds,
};
use std::sync::atomic::AtomicUsize;

struct CountedProjection {
    shape: ActualWaveShape,
    calls: Arc<AtomicUsize>,
    panic: bool,
}
impl PendingActualWaveProjection for CountedProjection {
    fn rows(&self) -> &[ActualWaveRow] {
        &self.shape.rows
    }
    fn bounds(&self) -> PendingWaveBounds {
        PendingWaveBounds {
            retained_bytes: std::mem::size_of::<Self>()
                + self.shape.rows.capacity() * std::mem::size_of::<ActualWaveRow>(),
            maximum_resolved_bytes: 4096,
            retained_rows: self.shape.rows.capacity(),
        }
    }
    fn project(&self) -> Result<ActualWaveShape, ActualWaveEvidenceUnknown> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(!self.panic, "controlled consumer projection panic");
        Ok(self.shape.clone())
    }
}
fn raw_call(
    sink: &Arc<BoundedCostSampleSink>,
    calls: Arc<AtomicUsize>,
    panic: bool,
) -> (
    EngineCostCall,
    Arc<CostCalibrationCapture>,
    Arc<VirtualClock>,
) {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let (mut call, clock) = begin_with_retained_capacity(&shape, sink, 4096);
    let capture = Arc::new(CostCalibrationCapture::default());
    call.attach_calibration_capture(capture.clone());
    {
        let mut context = call.context().unwrap();
        context.physical_wave_pending(
            PendingActualWave::new(Arc::new(CountedProjection {
                shape: shape.clone(),
                calls,
                panic,
            })),
            Some(3),
        );
        clock.set(6);
        context.terminal(ActualWaveOutcome::Completed, None);
        context.finish_call(ObservedCallOutcome::Completed);
    }
    call.record_host_result(committed(&shape.rows[0], 9));
    clock.set(10);
    (call, capture, clock)
}

#[test]
fn async_observation_projects_once_after_fifo_and_keeps_original_clock() {
    let sink = sink(4, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let (call, capture, clock) = raw_call(&sink, calls.clone(), false);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    assert!(matches!(capture.status(), CostCalibrationStatus::Pending));
    let before = sink.stats();
    assert_eq!(
        (before.raw_accepted, before.raw_pending, before.published),
        (1, 1, 0)
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    clock.set(100);
    let (ordinal, entry, ticket, _, _proof) = sink.pop_resolved_bound().unwrap();
    assert_eq!(ordinal, 1);
    assert!(ticket.is_none());
    let resolved = entry.expect("complete original raw wave");
    let CostEvidenceEntry::Training { sample, .. } = resolved.entry() else {
        panic!("training receipt");
    };
    assert_eq!(sample.observed_at_ns, 10);
    assert_eq!(sample.timing.wall_total_ns, 8);
    let _ = resolved.entry();
    let _ = resolved.structured();
    let _ = resolved.structured();
    let _ = capture.status();
    let _ = format!("{:?}", capture.actual_evidence_diagnostic());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let after = sink.stats();
    assert_eq!(
        (after.raw_resolved, after.raw_pending, after.published),
        (1, 0, 1)
    );
    assert_eq!(
        (after.raw_queue_wait_ns_total, after.raw_queue_wait_ns_max),
        (90, 90)
    );
}

#[test]
fn async_observation_queue_loss_never_projects_or_claims_published() {
    let sink = sink(1, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let (first, _, _) = raw_call(&sink, calls.clone(), false);
    let (second, dropped, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(first.finish(), CostCallDisposition::Queued);
    assert_eq!(
        second.finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let CostCalibrationStatus::Complete(result) = dropped.status() else {
        panic!("lost waiter completed");
    };
    assert!(matches!(
        result.as_ref(),
        CostCalibrationResult::UnresolvedDropped(CostSampleDrop::Capacity)
    ));
    assert_eq!(sink.stats().published, 0);
    assert_eq!(sink.pop_resolved_bound().unwrap().0, 1);
    let (third, _, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(third.finish(), CostCallDisposition::Queued);
    assert_eq!(sink.pop_resolved_bound().unwrap().0, 2);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    let stats = sink.stats();
    assert_eq!(
        (
            stats.raw_offered,
            stats.raw_accepted,
            stats.raw_lost,
            stats.raw_pending
        ),
        (3, 2, 1, 0)
    );
}

#[test]
fn async_observation_worker_stop_completes_pending_and_future_waiters() {
    let sink = sink(4, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let (call, capture, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    sink.worker_stopped();
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        panic!("stopped waiter completed");
    };
    assert!(matches!(
        result.as_ref(),
        CostCalibrationResult::UnresolvedDropped(CostSampleDrop::WorkerStopped)
    ));
    let (late, late_capture, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(
        late.finish(),
        CostCallDisposition::Dropped(CostSampleDrop::WorkerStopped)
    );
    assert!(matches!(
        late_capture.status(),
        CostCalibrationStatus::Complete(_)
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let stats = sink.stats();
    assert_eq!(
        (
            stats.raw_accepted,
            stats.raw_abandoned,
            stats.raw_lost,
            stats.raw_pending
        ),
        (1, 1, 2, 0)
    );
}

#[test]
fn async_observation_projection_unwind_cannot_leave_a_pending_capture() {
    let sink = sink(4, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let (call, capture, _) = raw_call(&sink, calls.clone(), true);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink.pop_resolved_bound()))
            .is_err()
    );
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        panic!("unwound waiter completed");
    };
    assert!(matches!(
        result.as_ref(),
        CostCalibrationResult::Rejected(CostCallRejection::Abandoned)
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(sink.stats().memory.queued_raw_bytes, 0);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, 0);
    let stats = sink.stats();
    assert_eq!(
        (
            stats.raw_resolved,
            stats.raw_abandoned,
            stats.raw_lost,
            stats.raw_pending
        ),
        (0, 1, 1, 0)
    );
}

#[tokio::test]
async fn async_observation_cancelled_ack_rejoins_original_fifo_completion() {
    use std::future::Future;
    use std::task::{Context, Poll};
    let sink = sink(4, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let (call, capture, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    {
        let mut wait = Box::pin(capture.wait_resolved());
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
        // Cancellation drops only this waiter, never the accepted observation.
    }
    assert_eq!(sink.stats().raw_pending, 1);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let original = sink.pop_resolved_bound().unwrap();
    assert_eq!(original.0, 1);
    capture.wait_resolved().await;
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        panic!("same completed capture");
    };
    assert!(matches!(
        result.as_ref(),
        CostCalibrationResult::Observed {
            accepted_ordinal: Some(1),
            disposition: CostCallDisposition::Published,
            ..
        }
    ));
    assert!(sink.pop_resolved_bound().is_none());
}

#[test]
fn async_observation_result_bytes_follow_ack_and_stage_owners() {
    let sink = sink(4, 64);
    let calls = Arc::new(AtomicUsize::new(0));
    let (call, capture, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    assert!(sink.stats().memory.queued_raw_bytes > 0);
    let (_, resolved, _, _, _proof) = sink.pop_resolved_bound().unwrap();
    assert_eq!(sink.stats().memory.queued_raw_bytes, 0);
    let CostCalibrationStatus::Complete(ack) = capture.status() else {
        panic!("resolved original ack");
    };
    let stages = capture.host_stages();
    let after = sink.stats().memory;
    assert!(after.worker_and_result_bytes > 0);
    assert!(
        after.worker_and_result_bytes < after.accepted_working_bytes_peak,
        "temporary expansion must not remain charged for an entire calibration phase"
    );
    drop(resolved);
    drop(capture);
    assert!(sink.stats().memory.worker_and_result_bytes > 0);
    drop(ack);
    if stages.is_some() {
        assert!(sink.stats().memory.worker_and_result_bytes > 0);
    }
    drop(stages);
    assert_eq!(sink.stats().memory.worker_and_result_bytes, 0);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn async_observation_working_byte_rejection_preserves_fifo_and_completes_ack() {
    let sink = Arc::new(
        BoundedCostSampleSink::new_with_byte_limits(
            CostSampleSinkLimits {
                max_samples: 4,
                max_shape_rows: 64,
            },
            CostRecorderByteLimits {
                maximum_retained_bytes: 1024 * 1024,
                maximum_working_bytes: 4096,
            },
        )
        .unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let (call, capture, _) = raw_call(&sink, calls.clone(), false);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    let (ordinal, resolved, _, _, _proof) = sink.pop_resolved_bound().unwrap();
    assert_eq!(ordinal, 1);
    assert!(resolved.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let CostCalibrationStatus::Complete(ack) = capture.status() else {
        panic!("rejected original ack");
    };
    assert!(matches!(
        ack.as_ref(),
        CostCalibrationResult::UnresolvedDropped(CostSampleDrop::Capacity)
    ));
    let stats = sink.stats();
    assert_eq!(stats.raw_resolution_failed, 1);
    assert_eq!(stats.raw_lost, 1);
    assert_eq!(stats.memory.queued_raw_bytes, 0);
    assert_eq!(stats.memory.worker_and_result_bytes, 0);
    let failure = stats.memory.first_capture_rejection.unwrap();
    assert_eq!(failure.gate, "worker_and_result_bytes");
    assert!(failure.diagnostic.requested.unwrap() > failure.diagnostic.limit);
}
