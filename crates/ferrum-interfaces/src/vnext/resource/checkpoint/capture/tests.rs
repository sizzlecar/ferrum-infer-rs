use super::*;
use crate::vnext::SuccessfulCheckpointCaptureSeal;

fn owner_fixture() -> (
    Harness,
    ResourceId,
    Arc<CheckpointBackingOwner<TestRuntime>>,
) {
    let (harness, resource) = fixture(linear_profile(), 64);
    initialize(&harness);
    let owner = allocate(&harness.root, &requests(&harness.root, &resource, 17));
    (harness, resource, owner)
}

fn assert_released(harness: &Harness) {
    let logical = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(logical.active_checkpoint_claims(), 0);
    assert_eq!(logical.active_requests(), 0);
    assert_eq!(logical.active_sequences(), 0);
    assert_eq!(
        // Inspect the retained ledger directly: the public telemetry API must
        // reject a closing plan even while existing owners release normally.
        harness
            .root
            .maintenance_controller
            .status()
            .unwrap()
            .pools()[0]
            .live_occupancy()
            .transient()
            .checkpoint()
            .physical_bytes(),
        0
    );
}

#[test]
fn capture_reservation_is_exclusive_and_retry_uses_a_new_attempt_without_new_claims() {
    let (harness, _, owner) = owner_fixture();
    let first = owner.try_reserve_capture().unwrap();
    let first_id = first.attempt_id();
    let competitor = Arc::clone(&owner);
    std::thread::spawn(move || assert!(competitor.try_reserve_capture().is_err()))
        .join()
        .unwrap();
    let logical = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(logical.active_checkpoint_claims(), 1);
    assert_eq!(logical.active_requests(), 0);
    assert_eq!(logical.active_sequences(), 0);
    assert_eq!(first_id.checkpoint_authority(), owner.authority());
    drop(first);
    let next = owner.try_reserve_capture().unwrap();
    assert_ne!(next.attempt_id(), first_id);
    assert!(next.attempt_id().serial() > first_id.serial());
    drop(next);
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn capture_pre_submit_unwind_releases_permission_and_all_budget_normally() {
    let (harness, _, owner) = owner_fixture();
    let mut previous = None;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let permit = owner.try_reserve_capture().unwrap();
        previous = Some(permit.attempt_id());
        panic!("injected encoder failure before submit");
    }))
    .is_err());
    let retry = owner.try_reserve_capture().unwrap();
    assert_ne!(Some(retry.attempt_id()), previous);
    drop(retry);
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn capture_definitely_not_submitted_restores_fresh_storage_but_not_attempt_identity() {
    let (harness, _, owner) = owner_fixture();
    let mut permit = owner.try_reserve_capture().unwrap();
    let first_id = permit.attempt_id();
    permit.mark_possibly_submitted().unwrap();
    assert!(owner.try_reserve_capture().is_err());
    permit.definitely_not_submitted().unwrap();
    let retry = owner.try_reserve_capture().unwrap();
    assert_ne!(retry.attempt_id(), first_id);
    drop(retry);
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn captured_owner_and_retained_views_can_never_reserve_another_write() {
    let (harness, resource, owner) = owner_fixture();
    let view = owner.view(&resource).unwrap();
    let retention = view.segment_bindings()[0].retention();
    drop(view);
    let mut permit = owner.try_reserve_capture().unwrap();
    let attempt = permit.attempt_id();
    permit.mark_possibly_submitted().unwrap();
    let captured = permit
        .finish_succeeded(&SuccessfulCheckpointCaptureSeal::for_test(attempt))
        .unwrap();
    assert_eq!(captured.attempt_id(), attempt);
    assert!(Arc::ptr_eq(captured.backing(), &owner));
    assert!(owner.try_reserve_capture().is_err());
    assert!(captured.backing().try_reserve_capture().is_err());
    drop(captured);
    assert!(owner.try_reserve_capture().is_err());
    drop(owner);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_checkpoint_claims(),
        1
    );
    drop(retention);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn capture_success_requires_submission_and_rejected_transition_returns_the_permit() {
    let (harness, _, owner) = owner_fixture();
    let permit = owner.try_reserve_capture().unwrap();
    let attempt = permit.attempt_id();
    let failure = permit
        .finish_succeeded(&SuccessfulCheckpointCaptureSeal::for_test(attempt))
        .unwrap_err();
    let (_, permit) = failure.into_parts();
    assert!(owner.try_reserve_capture().is_err());
    drop(permit); // No submit was entered, so this is an ordinary safe rollback.
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn foreign_owner_and_stale_attempt_success_cannot_publish_captured_storage() {
    let (harness, resource, first_owner) = owner_fixture();
    let second_owner = allocate(&harness.root, &requests(&harness.root, &resource, 17));
    let other = second_owner.try_reserve_capture().unwrap();
    let mut first = first_owner.try_reserve_capture().unwrap();
    let stale = first.attempt_id();
    first.mark_possibly_submitted().unwrap();
    let failure = first
        .finish_succeeded(&SuccessfulCheckpointCaptureSeal::for_test(
            other.attempt_id(),
        ))
        .unwrap_err();
    let (_, first) = failure.into_parts();
    first.definitely_not_submitted().unwrap(); // The simulated backend did not submit this attempt.
    let mut next = first_owner.try_reserve_capture().unwrap();
    next.mark_possibly_submitted().unwrap();
    let failure = next
        .finish_succeeded(&SuccessfulCheckpointCaptureSeal::for_test(stale))
        .unwrap_err();
    let (_, next) = failure.into_parts();
    next.finish_failed_but_quiescent().unwrap();
    assert!(first_owner.try_reserve_capture().is_err());
    drop(other);
    drop(second_owner);
    drop(first_owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn duplicate_submit_mark_does_not_reopen_or_lose_the_live_permit() {
    let (harness, _, owner) = owner_fixture();
    let mut permit = owner.try_reserve_capture().unwrap();
    permit.mark_possibly_submitted().unwrap();
    assert!(permit.mark_possibly_submitted().is_err());
    assert!(owner.try_reserve_capture().is_err());
    permit.finish_failed_but_quiescent().unwrap();
    assert!(owner.try_reserve_capture().is_err());
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn capture_serial_exhaustion_never_wraps_to_an_earlier_attempt() {
    let (harness, _, owner) = owner_fixture();
    owner.test_only_exhaust_capture_serial();
    let last = owner.try_reserve_capture().unwrap();
    assert_eq!(last.attempt_id().serial(), u64::MAX);
    drop(last);
    assert!(owner.try_reserve_capture().is_err());
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn poisoned_capture_mutex_rejects_success_but_proven_quiescent_cleanup_releases_budget() {
    let (harness, _, owner) = owner_fixture();
    let mut permit = owner.try_reserve_capture().unwrap();
    permit.mark_possibly_submitted().unwrap();
    let attempt = permit.attempt_id();
    owner.test_only_poison_capture_mutex();
    let failure = permit
        .finish_succeeded(&SuccessfulCheckpointCaptureSeal::for_test(attempt))
        .unwrap_err();
    let (_, permit) = failure.into_parts();
    permit.finish_failed_but_quiescent().unwrap();
    assert!(owner.try_reserve_capture().is_err());
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn plan_close_blocks_new_capture_attempts_but_allows_in_flight_safe_release() {
    let (harness, _, owner) = owner_fixture();
    let mut permit = owner.try_reserve_capture().unwrap();
    permit.mark_possibly_submitted().unwrap();
    assert!(matches!(
        PlanRuntimeResources::close(Arc::clone(&harness.root)),
        Ok(PlanRuntimeCloseOutcome::Referenced { .. })
    ));
    assert!(owner.try_reserve_capture().is_err());
    permit.finish_failed_but_quiescent().unwrap();
    drop(owner);
    assert_released(&harness);
    close_dynamic_test_root(harness.root);
}

#[test]
fn unknown_permit_drop_keeps_poisoned_backing_instead_of_reopening_or_freeing_it() {
    let (harness, _, owner) = owner_fixture();
    let mut permit = owner.try_reserve_capture().unwrap();
    permit.mark_possibly_submitted().unwrap();
    let weak = Arc::downgrade(&owner);
    drop(permit);
    assert!(owner.try_reserve_capture().is_err());
    drop(owner);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap()
            .active_checkpoint_claims(),
        1
    );
    // This deliberately exercises protocol misuse, not normal recovery. No
    // terminal exists, so the bounded fixture owner must remain retained. All
    // normal terminal/rollback tests above verify complete budget release.
}
