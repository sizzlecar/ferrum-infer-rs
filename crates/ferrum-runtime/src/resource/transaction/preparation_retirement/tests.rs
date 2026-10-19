use super::*;
use crate::test_support::resource_support::{
    configured_driver, isolated_static_initialization_plan, transaction, PostAllocationBehavior,
    TestDriver, Trace,
};
use std::sync::{Arc, Mutex};

fn fixture() -> (
    crate::ExecutionPlan,
    ResourceTransaction<TestDriver, TransactionNew>,
    Arc<Mutex<Trace>>,
    Vec<ResourceId>,
) {
    let (_, plan) = isolated_static_initialization_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let original = transaction(&plan, driver, "preparation-retirement");
    let ids = original
        .reservations()
        .reservations()
        .iter()
        .map(|entry| entry.resource_id().clone())
        .collect::<Vec<_>>();
    assert!(
        ids.len() >= 2,
        "fixture has independent real allocation slots"
    );
    (plan, original, trace, ids)
}

fn calls(trace: &Arc<Mutex<Trace>>, action: &str) -> Vec<String> {
    trace
        .lock()
        .unwrap()
        .calls
        .iter()
        .filter(|call| call.starts_with(&format!("{action}:")))
        .cloned()
        .collect()
}

#[test]
fn preparation_retirement_reserve_retries_original_compensation_then_releases_claim() {
    let (plan, original, trace, ids) = fixture();
    let last = ids.last().unwrap();
    let undone = &ids[ids.len() - 2];
    trace
        .lock()
        .unwrap()
        .failures
        .insert(format!("reserve:{last}"), 1);
    trace
        .lock()
        .unwrap()
        .failures
        .insert(format!("undo-reserve:{undone}"), 1);
    let mut original = ResourcePreparationRetirement::from_reserve(
        original.reserve().err().expect("injected reserve failure"),
    );
    let cause = original.failure().clone();
    assert!(original.retire_in_place().is_err());
    assert!(!original.is_retired());
    assert_eq!(trace.lock().unwrap().runtime_allocate_calls, 0);
    let (peer_driver, _) = configured_driver(&plan, &[], &[]);
    let peer = transaction(&plan, peer_driver, "reserve-peer");
    let status = peer.maintenance_controller().status().unwrap();
    assert_eq!(
        status.process_claimed_bytes(),
        2 * status.budget_claimed_bytes()
    );
    original.retire_in_place().unwrap();
    assert!(original.is_retired());
    assert_eq!(original.failure(), &cause);
    let status = peer.maintenance_controller().status().unwrap();
    assert_eq!(
        status.process_claimed_bytes(),
        status.budget_claimed_bytes()
    );
    let expected = std::iter::once(format!("undo-reserve:{undone}"))
        .chain(
            ids[..ids.len() - 1]
                .iter()
                .rev()
                .map(|id| format!("undo-reserve:{id}")),
        )
        .collect::<Vec<_>>();
    assert_eq!(calls(&trace, "undo-reserve"), expected);
    assert!(trace.lock().unwrap().abandon.is_empty());
    // The peer has actual reservations only after reserve; close those normally.
    drop(peer.reserve().unwrap().rollback().unwrap());
}

#[test]
fn preparation_retirement_commit_preserves_prefix_and_partial_rollback() {
    let (_plan, original, trace, ids) = fixture();
    let last = ids.last().unwrap();
    let undone = &ids[ids.len() - 2];
    trace
        .lock()
        .unwrap()
        .failures
        .insert(format!("commit:{last}"), 1);
    trace
        .lock()
        .unwrap()
        .failures
        .insert(format!("undo-commit:{undone}"), 1);
    trace
        .lock()
        .unwrap()
        .failures
        .insert(format!("rollback:{last}"), 1);
    let failure = original
        .reserve()
        .unwrap()
        .commit()
        .err()
        .expect("injected commit failure");
    assert!(matches!(
        failure,
        ResourceCommitTransitionError::Recoverable(_)
    ));
    let mut original = ResourcePreparationRetirement::from_commit(failure);
    let cause = original.failure().clone();
    assert!(original.retire_in_place().is_err());
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    assert!(
        original.retire_in_place().is_err(),
        "second attempt reaches actual rollback failure"
    );
    assert_eq!(trace.lock().unwrap().buffer_drops, (ids.len() - 1) as u64);
    assert!(!original.is_retired());
    original.retire_in_place().unwrap();
    assert_eq!(original.failure(), &cause);
    assert!(original.is_retired());
    let expected = ids
        .iter()
        .map(|id| format!("rollback:{id}"))
        .chain(std::iter::once(format!("rollback:{last}")))
        .collect::<Vec<_>>();
    assert_eq!(calls(&trace, "rollback"), expected);
    assert_eq!(
        trace.lock().unwrap().runtime_allocate_calls,
        (ids.len() - 1) as u64
    );
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    assert!(trace.lock().unwrap().abandon.is_empty());
}

#[test]
fn preparation_retirement_poisoned_commit_reconciles_actual_slot_before_compensation() {
    let (_plan, original, trace, ids) = fixture();
    let last = ids.last().unwrap();
    trace
        .lock()
        .unwrap()
        .post_allocation
        .insert(last.to_string(), PostAllocationBehavior::DropThenError);
    trace
        .lock()
        .unwrap()
        .failures
        .insert(format!("reconcile:{last}"), 1);
    let failure = original
        .reserve()
        .unwrap()
        .commit()
        .err()
        .expect("injected commit failure");
    assert!(matches!(
        failure,
        ResourceCommitTransitionError::Poisoned(_)
    ));
    let mut original = ResourcePreparationRetirement::from_commit(failure);
    let cause = original.failure().clone();
    assert!(original.retire_in_place().is_err());
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    assert!(calls(&trace, "undo-commit").is_empty());
    original.retire_in_place().unwrap();
    assert!(original.is_retired());
    assert_eq!(original.failure(), &cause);
    assert_eq!(trace.lock().unwrap().buffer_drops, ids.len() as u64);
    assert_eq!(
        calls(&trace, "reconcile"),
        vec![format!("reconcile:{last}"), format!("reconcile:{last}")]
    );
    assert_eq!(
        calls(&trace, "undo-commit"),
        ids[..ids.len() - 1]
            .iter()
            .rev()
            .map(|id| format!("undo-commit:{id}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    assert!(trace.lock().unwrap().abandon.is_empty());
}

#[test]
fn preparation_retirement_hook_panic_drop_preserves_actual_resources_and_capacity() {
    let _serial = crate::static_initialization::STATIC_PROVISION_CLEANUP_TEST
        .lock()
        .unwrap();
    let (plan, original, trace, ids) = fixture();
    let last = ids.last().unwrap();
    trace
        .lock()
        .unwrap()
        .post_allocation
        .insert(last.to_string(), PostAllocationBehavior::DropThenError);
    trace
        .lock()
        .unwrap()
        .hook_panics
        .insert(format!("reconcile:{last}"));
    let mut original = ResourcePreparationRetirement::from_commit(
        original
            .reserve()
            .unwrap()
            .commit()
            .err()
            .expect("injected commit failure"),
    );
    assert_eq!(
        original.retire_in_place().unwrap_err().code(),
        "preparation_retirement_panic"
    );
    let before = trace.lock().unwrap().calls.clone();
    assert!(original.retire_in_place().is_err());
    assert_eq!(trace.lock().unwrap().calls, before);
    drop(original);
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    let (peer_driver, _) = configured_driver(&plan, &[], &[]);
    let peer = transaction(&plan, peer_driver, "poison-peer");
    let status = peer.maintenance_controller().status().unwrap();
    assert_eq!(
        status.process_claimed_bytes(),
        2 * status.budget_claimed_bytes()
    );
    crate::maintain_static_initialization_cleanups(
        crate::MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS,
    )
    .unwrap();
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    assert_eq!(
        trace.lock().unwrap().calls,
        before,
        "unknown side effects are never replayed"
    );
    assert!(trace.lock().unwrap().abandon.is_empty());
    drop(peer.reserve().unwrap().rollback().unwrap());
}
