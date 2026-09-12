use super::*;

#[test]
fn checkpoint_access_abandoned_ready_capture_is_reclaimed_by_public_sweep() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let transfer = access_submitted(
        reaper
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &binding,
                Arc::clone(&harness.session),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    drop(transfer);
    assert_eq!(reaper.retained_count(), 1);
    let sweep = reaper.poll_bounded(1).unwrap();
    assert!(sweep.entries().is_empty());
    assert_eq!(sweep.retained_after(), 0);
    assert_checkpoint_budget_released(&harness);
    // Abandoning capture never cancels its source's execution authority.
    assert_eq!(
        reserve_capture(&harness.session)
            .completed_boundary()
            .unwrap()
            .token_prefix(),
        &[19]
    );
    drop(binding);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_abandoned_ready_restore_cancels_before_public_sweep_releases_gate() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "abandoned-ready", &[19, 23]);
    let transfer = access_submitted(
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
    drop(checkpoint);
    drop(transfer);
    assert!(target.request_cancel().unwrap().state_transfer_pending());
    assert!(target.try_abort_if_quiescent().is_err());
    let sweep = reaper.poll_bounded(1).unwrap();
    assert_eq!(sweep.retained_after(), 0);
    assert!(!target.request_cancel().unwrap().state_transfer_pending());
    assert!(target
        .try_prepare_state_transfer(
            SequenceStateTransferKind::CaptureRead,
            target.resources().backing_generation().unwrap(),
        )
        .is_err());
    assert_checkpoint_budget_released(&harness);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_abandoned_pending_restore_keeps_owners_until_fence_then_public_sweep() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "abandoned-pending", &[19, 23]);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Pending);
    let transfer = access_submitted(
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
    drop(checkpoint);
    drop(transfer);
    let sweep = reaper.poll_bounded(1).unwrap();
    assert_eq!(sweep.retained_after(), 1);
    assert_eq!(harness.runtime.synchronize_calls(), 0);
    assert_eq!(lane.in_flight_count(), 1);
    assert!(target.request_cancel().unwrap().state_transfer_pending());
    assert!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap()
            > 0
    );
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    assert_eq!(reaper.poll_bounded(1).unwrap().retained_after(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(harness.runtime.synchronize_calls(), 0);
    assert!(!target.request_cancel().unwrap().state_transfer_pending());
    assert_checkpoint_budget_released(&harness);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

enum UnknownTransfer {
    Fence,
    Submission,
}

fn abandoned_unknown_restore_requires_successful_recovery(unknown: UnknownTransfer) {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "abandoned-unknown", &[19, 23]);
    match unknown {
        UnknownTransfer::Fence => harness
            .runtime
            .set_fence_behavior(TestFenceBehavior::Indeterminate),
        UnknownTransfer::Submission => harness
            .runtime
            .set_submit_behavior(TestSubmitBehavior::PossiblySubmittedPanic),
    }
    let start = reaper
        .try_restore_sequence_checkpoint(
            &harness.fixture.plan,
            Arc::clone(&target),
            &checkpoint,
            Arc::from([19, 23]),
            Arc::clone(&lane),
        )
        .unwrap();
    let transfer = match (unknown, start) {
        (UnknownTransfer::Fence, NativeCheckpointStart::Submitted(transfer))
        | (UnknownTransfer::Submission, NativeCheckpointStart::Indeterminate(transfer)) => transfer,
        _ => panic!("unknown path must preserve its typed submission outcome"),
    };
    drop(checkpoint);
    drop(transfer);
    assert_eq!(reaper.poll_bounded(1).unwrap().retained_after(), 1);
    assert_eq!(harness.runtime.synchronize_calls(), 0);
    assert!(target.request_cancel().unwrap().state_transfer_pending());
    harness.runtime.fail_synchronize(1);
    let failed_drain = reaper.recover_abandoned_checkpoints(1).unwrap();
    assert_eq!(failed_drain.retained_after(), 1);
    assert_eq!(failed_drain.quarantined_after(), 1);
    assert_eq!(harness.runtime.synchronize_calls(), 1);
    assert!(target.request_cancel().unwrap().state_transfer_pending());
    assert!(target.try_abort_if_quiescent().is_err());
    assert!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap()
            > 0
    );
    let recovered = reaper.recover_abandoned_checkpoints(1).unwrap();
    assert_eq!(recovered.retained_after(), 0);
    assert_eq!(recovered.quarantined_after(), 0);
    assert_eq!(harness.runtime.synchronize_calls(), 2);
    assert!(!target.request_cancel().unwrap().state_transfer_pending());
    assert_checkpoint_budget_released(&harness);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_abandoned_unknown_fence_recovers_only_after_successful_drain() {
    abandoned_unknown_restore_requires_successful_recovery(UnknownTransfer::Fence);
}

#[test]
fn checkpoint_access_abandoned_unknown_submission_recovers_only_after_successful_drain() {
    abandoned_unknown_restore_requires_successful_recovery(UnknownTransfer::Submission);
}

#[test]
fn checkpoint_access_abandonment_does_not_cross_reapers_or_consume_live_handles() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let proof_reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &proof_reaper);
    drop(proof_reaper);
    let first = CompletionReaper::new();
    let second = CompletionReaper::new();
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let first_transfer = access_submitted(
        first
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &binding,
                Arc::clone(&harness.session),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(
        first_transfer.poll().unwrap(),
        NativeCheckpointObservation::Ready
    );
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Indeterminate);
    let mut second_transfer = access_submitted(
        second
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &binding,
                Arc::clone(&harness.session),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(first_transfer.slot_id(), second_transfer.slot_id());
    drop(first_transfer);
    // An indeterminate result with a live consumer must not be drained by the
    // abandoned-result worker, even when another reaper abandoned the same id.
    let untouched = second.recover_abandoned_checkpoints(1).unwrap();
    assert_eq!(untouched.retained_after(), 1);
    assert_eq!(harness.runtime.synchronize_calls(), 0);
    assert_eq!(first.poll_bounded(1).unwrap().retained_after(), 0);
    assert_eq!(second.retained_count(), 1);
    harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    assert_eq!(
        second_transfer.poll().unwrap(),
        NativeCheckpointObservation::Ready
    );
    let Some(NativeCheckpointResult::Captured(checkpoint)) = second_transfer.take_result().unwrap()
    else {
        panic!("live consumer must retain its successful capture result")
    };
    assert_eq!(checkpoint.token_prefix(), &[19]);
    drop(second_transfer);
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    drop(binding);
    drop(first);
    drop(second);
    drop(lane);
    harness.close();
}
