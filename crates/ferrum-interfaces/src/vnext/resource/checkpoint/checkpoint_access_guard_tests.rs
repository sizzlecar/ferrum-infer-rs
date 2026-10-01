use super::*;
use crate::execution_cost::{GuardedNotSubmittedReason, HostSubmissionRejection};
use crate::vnext::{
    CheckpointTransferObservationStart, CheckpointTransferSubmissionGuard,
    NativeCheckpointTransferKind, PreparedCheckpointTransfer,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Gate {
    allowed: AtomicBool,
    checks: AtomicUsize,
    kind: NativeCheckpointTransferKind,
}
impl CheckpointTransferSubmissionGuard for Gate {
    fn check(
        &self,
        prepared: &PreparedCheckpointTransfer<'_>,
    ) -> Result<(), GuardedNotSubmittedReason> {
        self.checks.fetch_add(1, Ordering::Relaxed);
        assert_eq!(prepared.identity().kind(), self.kind);
        assert_eq!(prepared.cost_domain().kind(), self.kind);
        assert!(prepared.cost_domain().geometry().copy_bytes() > 0);
        assert_eq!(
            prepared.source_capture_identity().is_some(),
            self.kind == NativeCheckpointTransferKind::Restore
        );
        if self.allowed.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::Cancelled,
            ))
        }
    }
}
fn gate(kind: NativeCheckpointTransferKind, allowed: bool) -> Arc<Gate> {
    Arc::new(Gate {
        allowed: AtomicBool::new(allowed),
        checks: AtomicUsize::new(0),
        kind,
    })
}
fn start_guarded_capture(
    h: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    gate: &Gate,
) -> NativeCheckpointStart<TestRuntime> {
    reaper
        .try_capture_sequence_checkpoint_guarded(
            &h.fixture.plan,
            &h.root.trusted_runtime_binding().unwrap(),
            Arc::clone(&h.session),
            Arc::clone(lane),
            DeviceTimingMode::Off,
            CheckpointTransferObservationStart::now(),
            gate,
        )
        .unwrap()
}

#[test]
fn checkpoint_final_guard_observes_epoch_change_after_encode_without_submission() {
    let h = prefix_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    h.runtime
        .checkpoint_guard_enabled
        .store(true, Ordering::Release);
    let sink = observe_transfers(&reaper, 1);
    let guard = gate(NativeCheckpointTransferKind::Capture, true);
    let changed = guard.clone();
    *h.runtime.checkpoint_guard_hook.lock().unwrap() = Some(Box::new(move || {
        changed.allowed.store(false, Ordering::Release);
    }));
    let submitted_before = h.runtime.submitted_timing_modes.lock().unwrap().len();
    let retained_before = h
        .root
        .dynamic_pools
        .logical_admission
        .checkpoint_retained_bytes()
        .unwrap();
    assert!(matches!(
        start_guarded_capture(&h, &lane, &reaper, &guard),
        NativeCheckpointStart::GuardRejected(GuardedNotSubmittedReason::HostRejected(
            HostSubmissionRejection::Cancelled
        ))
    ));
    assert_eq!(guard.checks.load(Ordering::Acquire), 1);
    assert_eq!(
        h.runtime.submitted_timing_modes.lock().unwrap().len(),
        submitted_before
    );
    assert_eq!(
        h.root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        retained_before
    );
    assert_eq!(reaper.retained_count(), 0);
    assert!(sink.samples.lock().unwrap().is_empty());
    guard.allowed.store(true, Ordering::Release);
    let mut transfer = access_submitted(start_guarded_capture(&h, &lane, &reaper, &guard));
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Captured(mut checkpoint)) = transfer.take_result().unwrap()
    else {
        panic!("capture");
    };
    assert!(sink.samples.lock().unwrap().is_empty());
    checkpoint
        .take_publication_acknowledgement()
        .unwrap()
        .acknowledge(&checkpoint)
        .unwrap();
    assert_eq!(sink.samples.lock().unwrap().len(), 1);
    drop(transfer);
    drop(checkpoint);
    drop(sink);
    drop(reaper);
    drop(lane);
    h.close();
}

#[test]
fn checkpoint_guard_unsupported_and_restore_rejection_preserve_original_owners() {
    let h = prefix_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    let capture_guard = gate(NativeCheckpointTransferKind::Capture, true);
    let submitted_before = h.runtime.submitted_timing_modes.lock().unwrap().len();
    assert!(matches!(
        start_guarded_capture(&h, &lane, &reaper, &capture_guard),
        NativeCheckpointStart::GuardRejected(GuardedNotSubmittedReason::AttributionUnavailable)
    ));
    assert_eq!(capture_guard.checks.load(Ordering::Acquire), 0);
    assert_eq!(
        h.runtime.submitted_timing_modes.lock().unwrap().len(),
        submitted_before
    );
    // Existing unguarded API remains usable on an unsupported backend.
    let checkpoint = access_capture(&h, &lane, &reaper);
    let target = admitted_full_target(&h, "guarded-restore-target", &[19, 23]);
    h.runtime
        .checkpoint_guard_enabled
        .store(true, Ordering::Release);
    let restore_guard = gate(NativeCheckpointTransferKind::Restore, false);
    let submitted_before = h.runtime.submitted_timing_modes.lock().unwrap().len();
    let start = reaper
        .try_restore_sequence_checkpoint_guarded(
            &h.fixture.plan,
            Arc::clone(&target),
            &checkpoint,
            Arc::from([19, 23]),
            Arc::clone(&lane),
            DeviceTimingMode::Off,
            CheckpointTransferObservationStart::now(),
            restore_guard.as_ref(),
        )
        .unwrap();
    assert!(matches!(start, NativeCheckpointStart::GuardRejected(_)));
    assert_eq!(restore_guard.checks.load(Ordering::Acquire), 1);
    assert_eq!(
        h.runtime.submitted_timing_modes.lock().unwrap().len(),
        submitted_before
    );
    restore_guard.allowed.store(true, Ordering::Release);
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint_guarded(
                &h.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
                DeviceTimingMode::Off,
                CheckpointTransferObservationStart::now(),
                restore_guard.as_ref(),
            )
            .unwrap(),
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("restore");
    };
    publication.acknowledge().unwrap();
    drop(transfer);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    h.close();
}

#[test]
fn checkpoint_guard_rechecks_native_source_cancellation_at_final_commit() {
    let h = prefix_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    h.runtime
        .checkpoint_guard_enabled
        .store(true, Ordering::Release);
    let source = Arc::clone(&h.session);
    *h.runtime.checkpoint_guard_hook.lock().unwrap() = Some(Box::new(move || {
        source.request_cancel().unwrap();
    }));
    let guard = gate(NativeCheckpointTransferKind::Capture, true);
    let submitted_before = h.runtime.submitted_timing_modes.lock().unwrap().len();
    assert!(matches!(
        start_guarded_capture(&h, &lane, &reaper, &guard),
        NativeCheckpointStart::GuardRejected(GuardedNotSubmittedReason::HostRejected(
            HostSubmissionRejection::Cancelled
        ))
    ));
    assert_eq!(
        guard.checks.load(Ordering::Acquire),
        0,
        "native owner rejection precedes the host witness"
    );
    assert_eq!(
        h.runtime.submitted_timing_modes.lock().unwrap().len(),
        submitted_before
    );
    assert_eq!(reaper.retained_count(), 0);
    drop(reaper);
    drop(lane);
    h.close();
}

#[test]
fn checkpoint_capture_publication_is_once_and_bound_to_the_actual_capture() {
    let h = prefix_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    let sink = observe_transfers(&reaper, 2);
    let mut first = access_capture(&h, &lane, &reaper);
    let mut first_clone = first.clone();
    let mut second = access_capture(&h, &lane, &reaper);
    assert!(sink.samples.lock().unwrap().is_empty());
    assert!(
        first_clone.take_publication_acknowledgement().is_none(),
        "a cloned cache owner cannot duplicate timing authority"
    );
    let first_publication = first.take_publication_acknowledgement().unwrap();
    assert!(first.take_publication_acknowledgement().is_none());
    assert!(first_publication.acknowledge(&second).is_err());
    assert!(sink.samples.lock().unwrap().is_empty());
    second
        .take_publication_acknowledgement()
        .unwrap()
        .acknowledge(&second)
        .unwrap();
    assert_eq!(sink.samples.lock().unwrap().len(), 1);
    let unpublished = access_capture(&h, &lane, &reaper);
    drop(unpublished);
    assert_eq!(
        sink.samples.lock().unwrap().len(),
        1,
        "dropping native-ready ownership is not product publication"
    );
    drop(first_clone);
    drop(first);
    drop(second);
    drop(sink);
    drop(reaper);
    drop(lane);
    h.close();
}
