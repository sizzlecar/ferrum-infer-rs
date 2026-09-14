//! Real public native lifecycle; this CPU fixture reports controlled device
//! timing but does not execute copied values. Metal continuation tests cover
//! actual copies and backend timestamps.

use super::*;
use crate::vnext::{
    DeviceExecutionTiming, DeviceTimingMeasurement, DeviceTimingMode, DeviceTimingUnavailableReason,
};

fn start_capture(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    mode: DeviceTimingMode,
) -> NativeCheckpointStart<TestRuntime> {
    reaper
        .try_capture_sequence_checkpoint_with_timing(
            &harness.fixture.plan,
            &harness.root.trusted_runtime_binding().unwrap(),
            Arc::clone(&harness.session),
            Arc::clone(lane),
            mode,
        )
        .unwrap()
}

#[test]
fn checkpoint_access_device_timing_and_copy_geometry_follow_capture_and_restore() {
    let harness = prefix_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    *harness.runtime.device_timing.lock().unwrap() =
        DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(37));
    harness.runtime.encoded_copy_regions.lock().unwrap().clear();
    // Kernel diagnostics must still use just Completion for native copies.
    let mut transfer = access_submitted(start_capture(
        &harness,
        &lane,
        &reaper,
        DeviceTimingMode::Kernel,
    ));
    assert_eq!(
        harness
            .runtime
            .submitted_timing_modes
            .lock()
            .unwrap()
            .last(),
        Some(&DeviceTimingMode::Completion)
    );
    assert_eq!(
        transfer.wait_for_recovery().unwrap(),
        NativeCheckpointObservation::Ready
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Captured(checkpoint)) = transfer.take_result().unwrap() else {
        panic!("capture must publish the measured successful transfer");
    };
    let capture = reaper.checkpoint_timing_snapshot().capture;
    assert_eq!(capture.device_execution.measured.samples, 1);
    assert_eq!(capture.device_execution.measured.total_ns, 37);
    assert_eq!(capture.device_execution.not_requested, 0);
    assert_eq!(capture.submitted_copies.samples, 1);
    assert_eq!(
        capture.submitted_copies.total_bytes,
        checkpoint.logical_bytes()
    );
    assert_eq!(
        capture.submitted_copies.total_commands,
        harness.runtime.encoded_copy_regions.lock().unwrap().len() as u64
    );
    assert_eq!(
        capture.submitted_copies.total_bytes,
        harness
            .runtime
            .encoded_copy_regions
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.length_bytes())
            .sum::<u64>()
    );
    assert!(
        transfer.poll().is_err(),
        "consumed results remain single-use"
    );
    assert_eq!(reaper.checkpoint_timing_snapshot().capture, capture);
    assert_eq!(harness.runtime.timing_queries.load(Ordering::Relaxed), 1);
    drop(transfer);

    let target = admitted_full_target(&harness, "timed-target", &[19, 23]);
    harness.runtime.encoded_copy_regions.lock().unwrap().clear();
    // A backend's measured zero remains distinguishable from unavailable.
    *harness.runtime.device_timing.lock().unwrap() =
        DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(0));
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint_with_timing(
                &harness.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
                DeviceTimingMode::Completion,
            )
            .unwrap(),
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("restore must remain gated until consumer acknowledgement");
    };
    assert_transfer_gate(&target);
    publication.acknowledge().unwrap();
    let restore = reaper.checkpoint_timing_snapshot().restore;
    assert_eq!(restore.device_execution.measured.samples, 1);
    assert_eq!(restore.device_execution.measured.total_ns, 0);
    assert_eq!(restore.device_execution.unavailable, 0);
    assert_eq!(
        restore.submitted_copies.total_bytes,
        checkpoint.logical_bytes()
    );
    assert_eq!(
        restore.submitted_copies.total_commands,
        harness.runtime.encoded_copy_regions.lock().unwrap().len() as u64
    );
    assert_eq!(harness.runtime.timing_queries.load(Ordering::Relaxed), 2);
    let (step, _) = execute_continuation(&harness, &target, &lane, &reaper, &[19, 23], 1..2, true);
    step.try_retire_normal().unwrap();
    drop(transfer);
    drop(checkpoint);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    reaper.reset_checkpoint_timings();
    assert_eq!(reaper.checkpoint_timing_snapshot(), Default::default());
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_unsupported_device_timing_is_not_zero_duration() {
    let harness = prefix_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    *harness.runtime.device_timing.lock().unwrap() =
        DeviceTimingMeasurement::Unavailable(DeviceTimingUnavailableReason::BackendUnsupported);
    let mut transfer = access_submitted(start_capture(
        &harness,
        &lane,
        &reaper,
        DeviceTimingMode::Completion,
    ));
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let device = reaper.checkpoint_timing_snapshot().capture.device_execution;
    assert_eq!(device.measured.samples, 0);
    assert_eq!(device.unavailable, 1);
    assert_eq!(
        device.last_unavailable_reason,
        Some(DeviceTimingUnavailableReason::BackendUnsupported)
    );
    assert_eq!(device.not_requested, 0);
    drop(transfer.take_result().unwrap());
    drop(transfer);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_failed_and_unknown_transfers_never_report_successful_device_time() {
    for (submit, fence) in [
        (
            TestSubmitBehavior::DefinitelyNotSubmitted,
            TestFenceBehavior::Succeeded,
        ),
        (
            TestSubmitBehavior::PossiblySubmittedPanic,
            TestFenceBehavior::Succeeded,
        ),
        (
            TestSubmitBehavior::Submitted,
            TestFenceBehavior::FailedButQuiescent,
        ),
        (
            TestSubmitBehavior::Submitted,
            TestFenceBehavior::Indeterminate,
        ),
    ] {
        let harness = prefix_harness(Default::default());
        let lane = harness.root.create_execution_lane().unwrap();
        let reaper = CompletionReaper::new();
        prove_prefix_source(&harness, &lane, &reaper);
        *harness.runtime.device_timing.lock().unwrap() =
            DeviceTimingMeasurement::Measured(DeviceExecutionTiming::device_event_elapsed(999));
        harness.runtime.set_submit_behavior(submit);
        harness.runtime.set_fence_behavior(fence);
        let start = start_capture(&harness, &lane, &reaper, DeviceTimingMode::Completion);
        let submitted = matches!(submit, TestSubmitBehavior::Submitted);
        let unknown = matches!(submit, TestSubmitBehavior::PossiblySubmittedPanic);
        if let NativeCheckpointStart::NotSubmitted(_) = start {
            assert!(!submitted && !unknown);
        } else {
            let mut transfer = match start {
                NativeCheckpointStart::Submitted(t) | NativeCheckpointStart::Indeterminate(t) => t,
                _ => panic!("expected an owned native transfer"),
            };
            let mut observation = transfer.wait_for_recovery().unwrap();
            if observation != NativeCheckpointObservation::Ready {
                assert_eq!(
                    reaper
                        .checkpoint_timing_snapshot()
                        .capture
                        .device_execution
                        .measured
                        .samples,
                    0
                );
                assert_transfer_gate(&harness.session);
                observation = transfer.recover_by_draining_lane().unwrap();
            }
            assert_eq!(observation, NativeCheckpointObservation::Ready);
            assert!(matches!(
                transfer.take_result().unwrap(),
                Some(NativeCheckpointResult::Failed(_))
            ));
        }
        let capture = reaper.checkpoint_timing_snapshot().capture;
        assert_eq!(capture.submitted_copies.samples, u64::from(submitted));
        assert_eq!(capture.indeterminate_copies.samples, u64::from(unknown));
        assert_eq!(capture.device_execution.measured.samples, 0);
        assert_eq!(capture.device_execution.measured.total_ns, 0);
        assert_eq!(
            capture.device_execution.failed_or_unproven,
            u64::from(submitted || unknown)
        );
        assert_checkpoint_budget_released(&harness);
        drop(reaper);
        drop(lane);
        harness.close();
    }
}
