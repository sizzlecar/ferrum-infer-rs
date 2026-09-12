//! Public access calls over the existing authenticated plan and CPU native
//! fixture. Copy commands are no-ops: these prove protocol/ownership, not values.

use super::*;
use crate::vnext::{
    CheckpointAccessSkipReason, NativeCheckpointFailure, NativeCheckpointObservation,
    NativeCheckpointResult, NativeCheckpointStart, NativeCheckpointTransfer, SequenceCheckpoint,
};

#[path = "checkpoint_access_abandon_tests.rs"]
mod checkpoint_access_abandon_tests;

#[path = "checkpoint_access_maintenance_tests.rs"]
mod checkpoint_access_maintenance_tests;

#[path = "checkpoint_access_timing_tests.rs"]
mod checkpoint_access_timing_tests;

fn access_submitted(
    start: NativeCheckpointStart<TestRuntime>,
) -> NativeCheckpointTransfer<TestRuntime> {
    match start {
        NativeCheckpointStart::Submitted(transfer) => transfer,
        _ => panic!("CPU access must submit one tracked native transfer"),
    }
}

fn access_capture(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> SequenceCheckpoint<TestRuntime> {
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let mut transfer = access_submitted(
        reaper
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &binding,
                Arc::clone(&harness.session),
                Arc::clone(lane),
            )
            .unwrap(),
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Captured(checkpoint)) = transfer.take_result().unwrap() else {
        panic!("public capture must publish immutable checkpoint ownership")
    };
    assert!(transfer.take_result().is_err());
    checkpoint
}

#[test]
fn checkpoint_access_capture_restore_ack_and_full_plan_continuation() {
    verify_public_continuation(CheckpointPartitionNumerics::BitwiseEquivalent);
}

#[test]
fn checkpoint_access_captured_execution_adopts_only_successful_source_state() {
    verify_public_continuation(CheckpointPartitionNumerics::CapturedExecutionContinuation);
}

fn verify_public_continuation(numerics: CheckpointPartitionNumerics) {
    let harness = prefix_harness(checkpoint_fixture::Spec {
        numerics,
        ..Default::default()
    });
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let source_refs = Arc::strong_count(&harness.session);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    let timings = reaper.checkpoint_timing_snapshot();
    assert_eq!(timings.capture.prepare_claim.samples, 1);
    assert_eq!(timings.capture.encode_submit.samples, 1);
    assert_eq!(timings.capture.fence_recovery.samples, 1);
    assert_eq!(timings.capture.publication.samples, 1);
    assert_eq!(timings.capture.device_execution.not_requested, 1);
    assert_eq!(timings.capture.device_execution.measured.samples, 0);
    assert_eq!(harness.runtime.timing_queries.load(Ordering::Relaxed), 0);
    assert_eq!(timings.restore, Default::default());
    assert_eq!(Arc::strong_count(&harness.session), source_refs);
    assert_eq!(checkpoint.token_prefix(), &[19]);
    assert_eq!(checkpoint.full_input(), &[19, 31]);
    assert_eq!(checkpoint.completed_tokens(), 1);
    assert!(checkpoint.retained_bytes() >= checkpoint.logical_bytes());
    let retained = checkpoint.retained_bytes();
    let clone = checkpoint.clone();
    assert_eq!(clone.authority(), checkpoint.authority());
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        retained
    );
    drop(clone);

    let target = admitted_full_target(&harness, "public-access-target", &[19, 23]);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Pending);
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint(
                &harness.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(
        transfer.poll().unwrap(),
        NativeCheckpointObservation::Pending
    );
    assert!(transfer.take_result().unwrap().is_none());
    assert_transfer_gate(&target);
    let timings = reaper.checkpoint_timing_snapshot();
    assert_eq!(timings.restore.prepare_claim.samples, 1);
    assert_eq!(timings.restore.encode_submit.samples, 1);
    assert_eq!(timings.restore.fence_recovery.samples, 1);
    assert_eq!(timings.restore.publication.samples, 0);
    // Reset is observation-only, even while a native write owns its target.
    reaper.reset_checkpoint_timings();
    assert_eq!(reaper.checkpoint_timing_snapshot(), Default::default());
    assert_eq!(reaper.retained_count(), 1);
    assert_eq!(lane.in_flight_count(), 1);
    assert_transfer_gate(&target);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let timings = reaper.checkpoint_timing_snapshot();
    assert_eq!(timings.capture, Default::default());
    assert_eq!(timings.restore.prepare_claim.samples, 0);
    assert_eq!(timings.restore.encode_submit.samples, 0);
    assert_eq!(timings.restore.fence_recovery.samples, 1);
    assert_eq!(timings.restore.publication.samples, 1);
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    assert_eq!(reaper.checkpoint_timing_snapshot(), timings);
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("public restore must keep outer publication gated")
    };
    assert!(publication.matches_target(&target));
    assert!(!publication.matches_target(&harness.session));
    assert_eq!(
        publication.target_sequence_authority(),
        target.sequence_authority()
    );
    assert_eq!(publication.target_epoch(), target.epoch());
    assert_eq!(publication.token_prefix(), &[19]);
    assert_eq!(publication.completed_tokens(), 1);
    assert_transfer_gate(&target);
    publication.acknowledge().unwrap();
    let boundary = reserve_capture(&target).completed_boundary().unwrap();
    assert!(matches!(
        boundary.provenance(),
        CompletedSequenceProvenance::ImportedCheckpoint { .. }
    ));
    let (step, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &[19, 23], 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    assert_eq!(
        reserve_capture(&target)
            .completed_boundary()
            .unwrap()
            .token_prefix(),
        &[19, 23]
    );
    drop(transfer);
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_rejects_wrong_sequence_input_before_any_submission() {
    verify_wrong_sequence_input(CheckpointPartitionNumerics::BitwiseEquivalent);
}

#[test]
fn checkpoint_access_captured_execution_still_checks_targets_own_input() {
    verify_wrong_sequence_input(CheckpointPartitionNumerics::CapturedExecutionContinuation);
}

fn verify_wrong_sequence_input(numerics: CheckpointPartitionNumerics) {
    let harness = prefix_harness(checkpoint_fixture::Spec {
        numerics,
        ..Default::default()
    });
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    // Parent request agrees with supplied input; the child sequence does not.
    let target = admitted_target(&harness, "public-wrong-child", &[19, 23], &[19, 29]);
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::PossiblySubmittedPanic);
    for tokens in [[19, 23], [31, 29]] {
        assert!(reaper
            .try_restore_sequence_checkpoint(
                &harness.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from(tokens),
                Arc::clone(&lane),
            )
            .is_err());
        assert_eq!(reaper.retained_count(), 0);
        assert_eq!(lane.in_flight_count(), 0);
        reserve_restore(&target)
            .ensure_fresh_restore_target()
            .unwrap();
    }
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::Submitted);
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint(
                &harness.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 29]),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("correct child input must still restore after pre-submit rejection")
    };
    publication.acknowledge().unwrap();
    drop(transfer);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_not_submitted_preserves_source_and_fresh_target() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let boundary = reserve_capture(&harness.session)
        .completed_boundary()
        .unwrap();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::DefinitelyNotSubmitted);
    assert!(matches!(
        reaper
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &binding,
                Arc::clone(&harness.session),
                Arc::clone(&lane),
            )
            .unwrap(),
        NativeCheckpointStart::NotSubmitted(_)
    ));
    assert!(Arc::ptr_eq(
        &boundary,
        &reserve_capture(&harness.session)
            .completed_boundary()
            .unwrap()
    ));
    assert_eq!(reaper.retained_count(), 0);
    assert_checkpoint_budget_released(&harness);
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::Submitted);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "public-not-submitted", &[19, 23]);
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::DefinitelyNotSubmitted);
    assert!(matches!(
        reaper
            .try_restore_sequence_checkpoint(
                &harness.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
            )
            .unwrap(),
        NativeCheckpointStart::NotSubmitted(_)
    ));
    reserve_restore(&target)
        .ensure_fresh_restore_target()
        .unwrap();
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(binding);
    assert_checkpoint_budget_released(&harness);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_unknown_submission_stays_owned_until_failed_drain() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "public-unknown", &[19, 23]);
    harness
        .runtime
        .set_submit_behavior(TestSubmitBehavior::PossiblySubmittedPanic);
    let NativeCheckpointStart::Indeterminate(mut transfer) = reaper
        .try_restore_sequence_checkpoint(
            &harness.fixture.plan,
            Arc::clone(&target),
            &checkpoint,
            Arc::from([19, 23]),
            Arc::clone(&lane),
        )
        .unwrap()
    else {
        panic!("possibly submitted panic must preserve a recovery handle")
    };
    assert_eq!(
        transfer.poll().unwrap(),
        NativeCheckpointObservation::Indeterminate
    );
    assert!(transfer.take_result().unwrap().is_none());
    assert_transfer_gate(&target);
    drop(checkpoint);
    assert!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap()
            > 0
    );
    assert_eq!(
        transfer.recover_by_draining_lane().unwrap(),
        NativeCheckpointObservation::Ready
    );
    assert!(matches!(
        transfer.take_result().unwrap(),
        Some(NativeCheckpointResult::Failed(
            NativeCheckpointFailure::AbandonedAfterDrain
        ))
    ));
    assert!(target
        .try_prepare_state_transfer(
            SequenceStateTransferKind::CaptureRead,
            target.resources().backing_generation().unwrap()
        )
        .is_err());
    assert_eq!(reaper.retained_count(), 0);
    assert_checkpoint_budget_released(&harness);
    drop(transfer);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_publication_drop_and_cancel_never_reopen_target() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    for (suffix, explicit_cancel) in [("public-drop", false), ("public-cancel", true)] {
        let target = admitted_full_target(&harness, suffix, &[19, 23]);
        let mut transfer = access_submitted(
            reaper
                .try_restore_sequence_checkpoint(
                    &harness.fixture.plan,
                    Arc::clone(&target),
                    &checkpoint,
                    Arc::from([19, 23]),
                    Arc::clone(&lane),
                )
                .unwrap(),
        );
        assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
        let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
        else {
            panic!("successful copy must hold publication until explicit ack")
        };
        assert_transfer_gate(&target);
        if explicit_cancel {
            target.request_cancel().unwrap();
            assert!(publication.acknowledge().is_err());
        } else {
            drop(publication);
        }
        assert!(target
            .try_prepare_state_transfer(
                SequenceStateTransferKind::CaptureRead,
                target.resources().backing_generation().unwrap()
            )
            .is_err());
        drop(transfer);
        target.try_abort_if_quiescent().unwrap();
        drop(target);
    }
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_policy_and_unproven_partition_skip_without_state_claims() {
    for (spec, expected_partition) in [
        (checkpoint_fixture::Spec::default(), false),
        (
            checkpoint_fixture::Spec {
                checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1024).unwrap()),
                numerics: CheckpointPartitionNumerics::SamePartitionOnly,
                ..Default::default()
            },
            true,
        ),
    ] {
        let harness = RestoreHarness::new(spec);
        let lane = harness.root.create_execution_lane().unwrap();
        let reaper = CompletionReaper::new();
        let binding = harness.root.trusted_runtime_binding().unwrap();
        let start = reaper
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &binding,
                Arc::clone(&harness.session),
                Arc::clone(&lane),
            )
            .unwrap();
        assert!(matches!(
            (start, expected_partition),
            (
                NativeCheckpointStart::Skipped(CheckpointAccessSkipReason::Disabled),
                false
            ) | (
                NativeCheckpointStart::Skipped(
                    CheckpointAccessSkipReason::MissingPartitionEvidence
                ),
                true
            )
        ));
        assert_eq!(reaper.retained_count(), 0);
        reserve_restore(&harness.session)
            .ensure_fresh_restore_target()
            .unwrap();
        assert_checkpoint_budget_released(&harness);
        drop(binding);
        drop(reaper);
        drop(lane);
        harness.close();
    }
}

#[test]
fn checkpoint_access_rejects_foreign_live_plan_before_submission() {
    let first = prefix_harness(checkpoint_fixture::Spec::default());
    let second = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = first.root.create_execution_lane().unwrap();
    let other_lane = second.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&first, &lane, &reaper);
    let foreign_binding = second.root.trusted_runtime_binding().unwrap();
    assert!(reaper
        .try_capture_sequence_checkpoint(
            &first.fixture.plan,
            &foreign_binding,
            Arc::clone(&first.session),
            Arc::clone(&lane),
        )
        .is_err());
    assert_checkpoint_budget_released(&second);
    let checkpoint = access_capture(&first, &lane, &reaper);
    let target = admitted_full_target(&second, "public-foreign-target", &[19, 23]);
    assert!(matches!(
        reaper
            .try_restore_sequence_checkpoint(
                &second.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&other_lane),
            )
            .unwrap(),
        NativeCheckpointStart::NotSubmitted(_)
    ));
    reserve_restore(&target)
        .ensure_fresh_restore_target()
        .unwrap();
    assert_eq!(reaper.retained_count(), 0);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(foreign_binding);
    drop(checkpoint);
    assert_checkpoint_budget_released(&first);
    drop(reaper);
    drop(lane);
    drop(other_lane);
    first.close();
    second.close();
}
