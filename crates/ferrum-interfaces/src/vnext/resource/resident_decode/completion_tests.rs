//! Exercise the resident owner through the original completion registry and
//! actual mock-runtime submission outcomes, without issuing test-only seals.
use super::*;
use crate::vnext::{
    BatchOperationIdentity, CompletionObservation, CompletionReaper, CompletionRecoveryOutcome,
    DeviceTimingMode, LaneSubmitOutcome, OperationCompletionDisposition,
};

fn reserve(
    harness: &Harness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    mut use_: ResidentDecodeUse<TestRuntime>,
) -> crate::vnext::completion::CompletionReservation<TestRuntime> {
    let active = use_
        .participant_sessions()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = BatchOperationIdentity::test_only_for_resident(
        &use_,
        &harness.root.dynamic_pools.nodes,
        &active,
    )
    .unwrap();
    use_.begin_dispatch().unwrap();
    CompletionReaper::reserve_resident(reaper, use_, Arc::clone(lane), identity).unwrap()
}

fn submit(
    harness: &Harness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    completion: &mut crate::vnext::completion::CompletionReservation<TestRuntime>,
) -> LaneSubmitOutcome<TestFence, TestRuntimeError> {
    let mut commands = DeviceCommandBatch::with_capacity(1);
    assert!(completion
        .encode_backing_initializations(harness.runtime.as_ref(), &mut commands)
        .is_ok());
    completion.mark_submission_started();
    lane.reserve_enqueue().unwrap().submit(commands)
}

#[test]
fn resident_completion_success_keeps_frame_until_original_fence_and_retirement() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-completion", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 3);
    let owner = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (use_, retirement) = ready(&owner, &batch, &req);
    let reaper = CompletionReaper::new();
    let mut completion = reserve(&harness, &lane, &reaper, use_);
    let fence = match submit(&harness, &lane, &mut completion) {
        LaneSubmitOutcome::Submitted(fence) => fence,
        _ => panic!("default mock submit must return a tracked fence"),
    };
    let handle = completion.arm(fence, DeviceTimingMode::Off).unwrap();
    assert_eq!(reaper.retained_count(), 1);
    let retirement = retirement
        .try_retire_normal()
        .unwrap_err()
        .into_retirement();
    assert!(!owner.is_idle());
    assert!(
        matches!(handle.wait().unwrap(), CompletionObservation::Terminal(receipt)
        if receipt.disposition() == &OperationCompletionDisposition::Succeeded)
    );
    assert_eq!(reaper.retained_count(), 0);
    assert!(
        !owner.is_idle(),
        "terminal observation does not consume the model's retirement ticket"
    );
    retirement.try_retire_normal().unwrap();
    assert!(owner.is_idle());
    // A second actual frame verifies that a retired whole-wave ledger is not
    // mistaken for authority to reuse the previous frame.
    let (next, ticket) = ready(&owner, &batch, &req);
    drop(next);
    ticket.try_rollback_unsubmitted().unwrap();
    drop(handle);
    drop(reaper);
    drop(owner);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_completion_rejection_retries_fresh_attempt_without_releasing_reserved_capacity() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-completion-retry", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 5);
    let owner = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (mut use_, retirement) = ready(&owner, &batch, &req);
    let old_attempt = use_.batch_invocation_id();
    let packet = Arc::new(11_u32);
    let packet_weak = Arc::downgrade(&packet);
    use_.install_terminal_readback_payload(packet);
    let reaper = CompletionReaper::new();
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::DefinitelyNotSubmitted);
    let mut completion = reserve(&harness, &lane, &reaper, use_);
    assert!(matches!(
        submit(&harness, &lane, &mut completion),
        LaneSubmitOutcome::DefinitelyNotSubmitted(_)
    ));
    let use_ = completion.definitely_not_submitted_resident().unwrap();
    assert_ne!(old_attempt, use_.batch_invocation_id());
    assert!(packet_weak.upgrade().is_none());
    assert_eq!(reaper.retained_count(), 0);
    assert!(!owner.is_idle());
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::Submitted);
    let mut completion = reserve(&harness, &lane, &reaper, use_);
    let fence = match submit(&harness, &lane, &mut completion) {
        LaneSubmitOutcome::Submitted(fence) => fence,
        _ => panic!("retry must submit its fresh attempt"),
    };
    let handle = completion.arm(fence, DeviceTimingMode::Off).unwrap();
    assert!(
        matches!(handle.wait().unwrap(), CompletionObservation::Terminal(receipt)
        if receipt.disposition() == &OperationCompletionDisposition::Succeeded)
    );
    retirement.try_retire_normal().unwrap();
    assert!(owner.is_idle());
    drop(handle);
    drop(reaper);
    drop(owner);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_completion_unknown_submit_quarantines_until_actual_lane_drain() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-completion-unknown", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 7);
    let owner = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let owner_weak = Arc::downgrade(&owner);
    let (use_, retirement) = ready(&owner, &batch, &req);
    let reaper = CompletionReaper::new();
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::PossiblySubmittedPanic);
    let mut completion = reserve(&harness, &lane, &reaper, use_);
    assert!(matches!(
        submit(&harness, &lane, &mut completion),
        LaneSubmitOutcome::PossiblySubmittedPanic
    ));
    let recovery = completion.submission_indeterminate();
    drop(owner);
    assert_eq!(reaper.retained_count(), 1);
    let retirement = retirement.try_abort().unwrap_err().into_retirement();
    assert!(owner_weak.upgrade().is_some());
    harness.runtime.fail_synchronize(1);
    assert!(matches!(
        recovery.recover_by_draining_lane().unwrap(),
        CompletionRecoveryOutcome::Quarantined(_)
    ));
    assert_eq!(reaper.retained_count(), 1);
    let retirement = retirement.try_abort().unwrap_err().into_retirement();
    assert!(matches!(
        recovery.recover_by_draining_lane().unwrap(),
        CompletionRecoveryOutcome::Drained(_)
    ));
    assert_eq!(reaper.retained_count(), 0);
    assert!(harness.runtime.synchronize_calls() >= 2);
    retirement.try_abort().unwrap();
    assert!(owner_weak.upgrade().is_none());
    drop(recovery);
    drop(reaper);
    session.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}
