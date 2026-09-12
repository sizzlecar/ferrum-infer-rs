//! Real native reaper protocol with an authenticated plan and the shared CPU
//! runtime. Its copy/zero commands are no-ops: these tests prove ownership,
//! admission and terminal delivery, not copied bytes or model numerics.

use super::completed_boundary_tests::submit_fixture_wave_through_reaper;
use super::restore_initialization_tests::{checkpoint_fixture, reserve_restore, RestoreHarness};
use super::*;
use crate::vnext::{
    CapturedCheckpoint, CompletionObservation, CompletionReaper, OperationCompletionDisposition,
    SequenceCheckpointBytePlan, StateTransferFailureReason, StateTransferHandle,
    StateTransferObservation, StateTransferResult, StateTransferSubmission,
};

#[path = "restore_frontier_tests.rs"]
mod restore_frontier_tests;

fn checkpoint_harness() -> RestoreHarness {
    RestoreHarness::new(checkpoint_fixture::Spec {
        checkpoint_capacity: Some(crate::vnext::CheckpointCapacityPolicy::new(1024).unwrap()),
        ..Default::default()
    })
}

fn reserve_capture(
    session: &Arc<SequenceSession<TestRuntime>>,
) -> PreparedSequenceStateTransfer<TestRuntime> {
    match session
        .try_prepare_state_transfer(
            SequenceStateTransferKind::CaptureRead,
            session.resources().backing_generation().unwrap(),
        )
        .unwrap()
    {
        SequenceStateTransferPreparation::Prepared(guard) => guard,
        _ => panic!("idle source must reserve its exact backing"),
    }
}

fn prove_source(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) {
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&harness.session)]).unwrap();
    let span = TokenSpanWork::from_token_ids(&[19], 0..1)
        .unwrap()
        .with_checkpoint_tokens(Arc::from([19]))
        .unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let StepResourceAdmissionDecision::Admitted(step) =
        batch.try_begin_step(request, lane).unwrap()
    else {
        panic!("resident source model step must admit")
    };
    let StepSubmissionWaveAdmissionDecision::Prepared(wave) = step
        .try_prepare_full_plan_submission_wave(
            Arc::new(step.work_shape().clone()),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    else {
        panic!("source full-plan wave must prepare")
    };
    let handle = submit_fixture_wave_through_reaper(
        &harness.root,
        &[Arc::clone(&harness.session)],
        lane,
        wave,
        reaper,
    );
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("CPU source full-plan fence must complete")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    assert_eq!(
        reserve_capture(&harness.session)
            .completed_boundary()
            .unwrap()
            .token_prefix(),
        &[19]
    );
}

fn allocate_checkpoint(
    harness: &RestoreHarness,
) -> (
    Arc<CheckpointBackingOwner<TestRuntime>>,
    Arc<SequenceCheckpointBytePlan>,
) {
    let bytes = Arc::new(harness.fixture.plan.checkpoint_byte_plan(1).unwrap());
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let CheckpointBackingAllocationDecision::Allocated(owner) = binding
        .try_allocate_checkpoint_backing(&bytes.backing_requests().unwrap())
        .unwrap()
    else {
        panic!("resident native checkpoint must allocate")
    };
    (owner, bytes)
}

fn submitted(submission: StateTransferSubmission<TestRuntime>) -> StateTransferHandle<TestRuntime> {
    match submission {
        StateTransferSubmission::Submitted(handle) => handle,
        _ => panic!("CPU submission must install a tracked fence"),
    }
}

fn complete_capture(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> Arc<CapturedCheckpoint<TestRuntime>> {
    let (owner, bytes) = allocate_checkpoint(harness);
    let handle = submitted(
        reaper
            .submit_capture(
                reserve_capture(&harness.session),
                owner.try_reserve_capture().unwrap(),
                bytes,
                Arc::clone(lane),
            )
            .unwrap(),
    );
    assert_eq!(handle.poll().unwrap(), StateTransferObservation::Ready);
    let Some(StateTransferResult::Captured(checkpoint)) = handle.take().unwrap() else {
        panic!("successful native fence must publish one captured owner")
    };
    assert!(handle.take().is_err());
    assert_eq!(reaper.retained_count(), 0);
    assert!(owner.try_reserve_capture().is_err());
    checkpoint
}

fn assert_transfer_gate(session: &Arc<SequenceSession<TestRuntime>>) {
    assert!(matches!(
        session
            .try_prepare_state_transfer(
                SequenceStateTransferKind::CaptureRead,
                session.resources().backing_generation().unwrap(),
            )
            .unwrap(),
        SequenceStateTransferPreparation::Busy
    ));
    assert!(session.try_abort_if_quiescent().is_err());
}

fn assert_checkpoint_budget_released(harness: &RestoreHarness) {
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        0
    );
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_checkpoint_claims(),
        0
    );
    assert!(harness
        .root
        .maintenance_controller
        .status()
        .unwrap()
        .pools()
        .iter()
        .all(|pool| pool
            .live_occupancy()
            .transient()
            .checkpoint()
            .physical_bytes()
            == 0));
}

#[test]
fn native_capture_fence_outbox_restore_preserves_gates_and_releases_execution_owners() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let source_refs = Arc::strong_count(&harness.session);
    let (owner, bytes) = allocate_checkpoint(&harness);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Pending);
    let handle = submitted(
        reaper
            .submit_capture(
                reserve_capture(&harness.session),
                owner.try_reserve_capture().unwrap(),
                bytes,
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(handle.poll().unwrap(), StateTransferObservation::Pending);
    assert!(handle.take().unwrap().is_none());
    assert_transfer_gate(&harness.session);
    assert!(owner.try_reserve_capture().is_err());
    assert_eq!(lane.in_flight_count(), 1);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    assert_eq!(handle.poll().unwrap(), StateTransferObservation::Ready);
    let Some(StateTransferResult::Captured(checkpoint)) = handle.take().unwrap() else {
        panic!("native capture must publish its owner")
    };
    assert_eq!(Arc::strong_count(&harness.session), source_refs);
    assert_eq!(checkpoint.boundary().token_prefix(), &[19]);
    assert_eq!(lane.in_flight_count(), 0);
    assert!(owner.try_reserve_capture().is_err());
    drop(owner);

    let target = harness.new_session("native-restore");
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Pending);
    let restore = submitted(
        reaper
            .submit_restore(
                reserve_restore(&target),
                Arc::clone(&checkpoint),
                harness.fixture.layout(),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(restore.poll().unwrap(), StateTransferObservation::Pending);
    assert!(restore.take().unwrap().is_none());
    assert_transfer_gate(&target);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    assert_eq!(restore.poll().unwrap(), StateTransferObservation::Ready);
    assert_transfer_gate(&target);
    let Some(StateTransferResult::RestoreReady(pending)) = restore.take().unwrap() else {
        panic!("restore must retain its target until conditional engine commit")
    };
    assert!(Arc::ptr_eq(pending.checkpoint(), &checkpoint));
    assert_transfer_gate(&target);
    assert_eq!(reaper.retained_count(), 0);
    drop(pending);
    assert!(target
        .try_prepare_state_transfer(
            SequenceStateTransferKind::RestoreWrite,
            target.resources().backing_generation().unwrap(),
        )
        .is_err());
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    drop(restore);
    drop(handle);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn native_capture_failed_fence_poisoned_destination_releases_source_and_capacity() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let (owner, bytes) = allocate_checkpoint(&harness);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::FailedButQuiescent);
    let handle = submitted(
        reaper
            .submit_capture(
                reserve_capture(&harness.session),
                owner.try_reserve_capture().unwrap(),
                bytes,
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(handle.poll().unwrap(), StateTransferObservation::Ready);
    let Some(StateTransferResult::Failed(failure)) = handle.take().unwrap() else {
        panic!("failed fence cannot publish a checkpoint")
    };
    assert!(matches!(
        failure.reason(),
        StateTransferFailureReason::FailedButQuiescent(_)
    ));
    assert!(owner.try_reserve_capture().is_err());
    assert!(reserve_capture(&harness.session)
        .completed_boundary()
        .is_ok());
    drop(owner);
    assert_checkpoint_budget_released(&harness);
    drop(handle);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn native_detached_unknown_capture_recovers_by_exact_scheduler_slot_and_keeps_all_owners() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let (owner, bytes) = allocate_checkpoint(&harness);
    let retained_bytes = owner.extent_bytes();
    let weak_owner = Arc::downgrade(&owner);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Indeterminate);
    let handle = submitted(
        reaper
            .submit_capture(
                reserve_capture(&harness.session),
                owner.try_reserve_capture().unwrap(),
                bytes,
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    let slot = handle.slot_id();
    drop(handle);
    drop(owner);
    let sweep = reaper.poll_bounded(1).unwrap();
    assert_eq!(sweep.state_transfers().len(), 1);
    assert_eq!(sweep.state_transfers()[0].slot_id, slot);
    assert_eq!(
        sweep.state_transfers()[0].observation,
        StateTransferObservation::Indeterminate
    );
    assert!(weak_owner.upgrade().is_some());
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        retained_bytes
    );
    assert_transfer_gate(&harness.session);
    assert_eq!(
        reaper.wait_state_transfer_for_recovery(slot).unwrap(),
        StateTransferObservation::Indeterminate
    );
    harness.runtime.fail_synchronize(1);
    assert_eq!(
        reaper
            .recover_state_transfer_by_draining_lane(slot)
            .unwrap(),
        StateTransferObservation::Quarantined
    );
    let timings = reaper.checkpoint_timing_snapshot();
    assert_eq!(timings.capture.fence_recovery.samples, 3);
    assert_eq!(timings.capture.publication.samples, 0);
    assert!(weak_owner.upgrade().is_some());
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        retained_bytes
    );
    assert_transfer_gate(&harness.session);
    assert_eq!(
        reaper
            .recover_state_transfer_by_draining_lane(slot)
            .unwrap(),
        StateTransferObservation::Ready
    );
    let timings = reaper.checkpoint_timing_snapshot();
    assert_eq!(timings.capture.fence_recovery.samples, 4);
    assert_eq!(timings.capture.publication.samples, 1);
    assert_eq!(timings.restore, Default::default());
    let Some(StateTransferResult::Failed(failure)) =
        reaper.take_completed_state_transfer(slot).unwrap()
    else {
        panic!("drain proves quiescence, never successful capture")
    };
    assert!(matches!(
        failure.reason(),
        StateTransferFailureReason::AbandonedAfterDrain
    ));
    assert!(weak_owner.upgrade().is_none());
    assert_checkpoint_budget_released(&harness);
    assert!(reserve_capture(&harness.session)
        .completed_boundary()
        .is_ok());
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn native_ready_restore_outbox_survives_handle_drop_and_reaper_cleanup_cancels_target() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = harness.new_session("native-abandoned-ready");
    let restore = submitted(
        reaper
            .submit_restore(
                reserve_restore(&target),
                Arc::clone(&checkpoint),
                harness.fixture.layout(),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(restore.poll().unwrap(), StateTransferObservation::Ready);
    drop(restore);
    assert_transfer_gate(&target);
    drop(reaper);
    assert_eq!(harness.root.deferred_cleanup_status().pending(), 1);
    assert_transfer_gate(&target);
    assert_eq!(
        harness
            .root
            .maintain_deferred_cleanups(1)
            .unwrap()
            .completed(),
        1
    );
    assert!(!target.request_cancel().unwrap().state_transfer_pending());
    assert!(target
        .try_prepare_state_transfer(
            SequenceStateTransferKind::RestoreWrite,
            target.resources().backing_generation().unwrap(),
        )
        .is_err());
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    drop(lane);
    harness.close();
}

#[test]
fn native_cancelled_restore_stays_gated_through_unknown_and_failed_drain() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = harness.new_session("native-cancelled-unknown");
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Indeterminate);
    let restore = submitted(
        reaper
            .submit_restore(
                reserve_restore(&target),
                Arc::clone(&checkpoint),
                harness.fixture.layout(),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert!(target.request_cancel().unwrap().state_transfer_pending());
    let checkpoint_weak = Arc::downgrade(checkpoint.backing());
    drop(checkpoint);
    drop(restore);
    drop(reaper);
    harness.runtime.fail_synchronize(1);
    let first = harness.root.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(first.status_after().pending(), 1);
    assert!(checkpoint_weak.upgrade().is_some());
    assert!(target.request_cancel().unwrap().state_transfer_pending());
    assert!(target.try_abort_if_quiescent().is_err());
    assert_eq!(
        harness
            .root
            .maintain_deferred_cleanups(1)
            .unwrap()
            .completed(),
        1
    );
    assert!(!target.request_cancel().unwrap().state_transfer_pending());
    assert!(checkpoint_weak.upgrade().is_none());
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    assert_checkpoint_budget_released(&harness);
    drop(lane);
    harness.close();
}

#[test]
fn native_definitely_not_submitted_rolls_back_capture_and_restore_without_writing() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let (owner, bytes) = allocate_checkpoint(&harness);
    let permit = owner.try_reserve_capture().unwrap();
    let old_attempt = permit.attempt_id();
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::DefinitelyNotSubmitted);
    assert!(reaper
        .submit_capture(
            reserve_capture(&harness.session),
            permit,
            bytes,
            Arc::clone(&lane),
        )
        .is_err());
    let retry = owner.try_reserve_capture().unwrap();
    assert_ne!(retry.attempt_id(), old_attempt);
    drop(retry);
    let target = harness.new_session("native-not-submitted");
    assert!(reaper
        .submit_restore(
            reserve_restore(&target),
            Arc::clone(&checkpoint),
            harness.fixture.layout(),
            Arc::clone(&lane),
        )
        .is_err());
    reserve_restore(&target)
        .ensure_fresh_restore_target()
        .unwrap();
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(owner);
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn native_submit_panic_keeps_capture_owned_until_drain_and_never_reopens_destination() {
    let harness = checkpoint_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_source(&harness, &lane, &reaper);
    let (owner, bytes) = allocate_checkpoint(&harness);
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::PossiblySubmittedPanic);
    let StateTransferSubmission::Indeterminate(handle) = reaper
        .submit_capture(
            reserve_capture(&harness.session),
            owner.try_reserve_capture().unwrap(),
            bytes,
            Arc::clone(&lane),
        )
        .unwrap()
    else {
        panic!("submit panic must retain an unknown native record")
    };
    assert_eq!(
        handle.poll().unwrap(),
        StateTransferObservation::Indeterminate
    );
    assert_transfer_gate(&harness.session);
    assert!(owner.try_reserve_capture().is_err());
    assert_eq!(reaper.retained_count(), 1);
    assert_eq!(
        handle.recover_by_draining_lane().unwrap(),
        StateTransferObservation::Ready
    );
    let Some(StateTransferResult::Failed(failure)) = handle.take().unwrap() else {
        panic!("successful drain must not claim the copy executed")
    };
    assert!(matches!(
        failure.reason(),
        StateTransferFailureReason::AbandonedAfterDrain
    ));
    assert!(owner.try_reserve_capture().is_err());
    assert!(reserve_capture(&harness.session)
        .completed_boundary()
        .is_ok());
    drop(owner);
    assert_checkpoint_budget_released(&harness);
    drop(handle);
    drop(reaper);
    drop(lane);
    harness.close();
}
