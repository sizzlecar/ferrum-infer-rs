use super::*;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    task::Wake,
};
mod at_most;

fn account(alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            512,
            HostMemoryClassLimits {
                input_preprocess_bytes: 512,
                output_bytes: 512,
                transfer_staging_bytes: 512,
                weight_backing_bytes: 512,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn claims(
    input: usize,
    output: usize,
    staging: usize,
    weights: usize,
) -> [(HostMemoryClass, usize); 4] {
    [
        (HostMemoryClass::InputPreprocess, input),
        (HostMemoryClass::Output, output),
        (HostMemoryClass::TransferStaging, staging),
        (HostMemoryClass::WeightBacking, weights),
    ]
}
fn snapshot(scope: &HostMemoryScope) -> (Counters, Counters, Counters, Counters) {
    let state = scope.account.0.state.lock().unwrap();
    let id = scope.account.1.as_ref().unwrap().id;
    let credit = &state.scopes[&id];
    (
        state.committed,
        credit.capacity,
        credit.remaining,
        credit.inherited,
    )
}

#[test]
fn phase_commit_returns_only_unused_padded_bytes_and_keeps_live_owners() {
    let parent = account(8);
    let p = parent.backing_bytes(3).unwrap();
    let w = parent.backing_bytes(9).unwrap();
    let scope = parent
        .reserve_scope(&claims(4 * p, 3 * p, 2 * p, 2 * w))
        .unwrap();
    let view = scope.account();
    let input = view
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 3)
        .unwrap();
    let output = view.reserve(HostMemoryClass::Output, 3).unwrap();
    let weights = view
        .allocate_zeroed(HostMemoryClass::WeightBacking, 9)
        .unwrap()
        .freeze();
    let before = parent.usage().unwrap();
    parent
        .shrink_scope_unused(&scope, &claims(p, 0, p, 0))
        .unwrap();
    let after = parent.usage().unwrap();
    assert_eq!(after.allocated_bytes(), before.allocated_bytes());
    assert_eq!(after.committed_bytes(), 4 * p + w);
    assert_eq!(
        after.class_committed_bytes(HostMemoryClass::InputPreprocess),
        2 * p
    );
    assert_eq!(after.class_committed_bytes(HostMemoryClass::Output), p);
    assert_eq!(
        after.class_committed_bytes(HostMemoryClass::TransferStaging),
        p
    );
    assert_eq!(
        after.class_committed_bytes(HostMemoryClass::WeightBacking),
        w
    );
    assert!(parent.owns(&weights));
    assert!(matches!(
        view.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::ScopeCapacity { available: 0, .. })
    ));
    let extra_input = view.reserve(HostMemoryClass::InputPreprocess, 3).unwrap();
    assert!(view.reserve(HostMemoryClass::InputPreprocess, 1).is_err());
    // Returned phase credit can now serve an independent consumer on the same
    // parent, while the existing padded allocations retain their exact charge.
    let parent_use = parent.reserve(HostMemoryClass::Output, 3).unwrap();
    drop(weights);
    let recycled = view.reserve(HostMemoryClass::WeightBacking, 9).unwrap();
    scope.close().unwrap();
    assert!(matches!(
        view.reserve(HostMemoryClass::InputPreprocess, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        3 * p + w + parent_use.backing_bytes()
    );
    drop((input, output, extra_input, recycled, parent_use));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn shrink_after_chained_exchange_preserves_original_and_inherited_refund_routes() {
    let parent = account(8);
    let p = parent.backing_bytes(3).unwrap();
    let w = parent.backing_bytes(9).unwrap();
    let a = parent.reserve_scope(&claims(64, 32, 0, 64)).unwrap();
    let a_view = a.account();
    let input = a_view
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 3)
        .unwrap();
    let a_weights = a_view
        .allocate_zeroed(HostMemoryClass::WeightBacking, 9)
        .unwrap();
    let b = parent.exchange_scope(a, &claims(0, 0, 0, 0)).unwrap();
    let b_view = b.account();
    let b_output = b_view
        .allocate_zeroed(HostMemoryClass::Output, 3)
        .unwrap()
        .freeze();
    let output_clone = b_output.clone();
    let b_weights = b_view
        .allocate_zeroed(HostMemoryClass::WeightBacking, 3)
        .unwrap();
    let c = parent.exchange_scope(b, &claims(0, 0, 0, 0)).unwrap();
    let inherited = snapshot(&c).3;
    parent.shrink_scope_unused(&c, &claims(0, 0, 0, 0)).unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 3 * p + w);
    assert_eq!(snapshot(&c).3, inherited);
    assert!(matches!(
        a_view.reserve(HostMemoryClass::InputPreprocess, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    assert!(matches!(
        b_view.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    drop(input);
    let replacement_input = c
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 3)
        .unwrap();
    drop(a_weights);
    let replacement_a = c
        .account()
        .reserve(HostMemoryClass::WeightBacking, 9)
        .unwrap();
    drop(b_weights);
    let replacement_b = c
        .account()
        .reserve(HostMemoryClass::WeightBacking, 3)
        .unwrap();
    drop(b_output);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 3 * p + w);
    c.close().unwrap();
    assert!(parent.0.state.lock().unwrap().scope_refunds.is_empty());
    // A slice/clone of the original immutable output still owns its allocation.
    assert_eq!(output_clone.backing_bytes(), p);
    drop((
        replacement_input,
        replacement_a,
        replacement_b,
        output_clone,
    ));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn late_class_growth_rejection_is_atomic_and_does_not_change_exchange_floor() {
    let parent = account(1);
    let old = parent.reserve_scope(&claims(30, 20, 0, 10)).unwrap();
    let used = old
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 10)
        .unwrap();
    let scope = parent.exchange_scope(old, &claims(0, 0, 0, 0)).unwrap();
    let before = snapshot(&scope);
    assert!(matches!(
        parent.shrink_scope_unused(&scope, &claims(5, 21, 0, 0)),
        Err(HostMemoryError::ScopeCapacity {
            class: HostMemoryClass::Output,
            requested: 21,
            available: 20
        })
    ));
    assert_eq!(snapshot(&scope), before);
    // Exchange itself still preserves old free credit until an explicit commit.
    let next = parent.exchange_scope(scope, &claims(1, 1, 0, 0)).unwrap();
    assert_eq!(snapshot(&next).2, before.2);
    parent
        .shrink_scope_unused(&next, &claims(5, 1, 0, 0))
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 16);
    drop(used);
    let all_input = next
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 15)
        .unwrap();
    next.close().unwrap();
    drop(all_input);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn committed_smaller_phase_releases_admission_capacity_without_recreating_owners() {
    let parent = account(1);
    let persistent = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 64)])
        .unwrap();
    let sample = persistent
        .account()
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 64)
        .unwrap()
        .freeze();
    let initialization = parent.reserve_scope(&claims(192, 128, 64, 32)).unwrap();
    let weights = initialization
        .account()
        .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
        .unwrap()
        .freeze();
    let next_phase = claims(64, 32, 0, 0);
    let candidate = claims(96, 64, 16, 16);
    // Provisional exchange preserves the old floor while the caller's native
    // replacement is still fallible; it cannot prematurely promise that credit.
    let steady = parent.exchange_scope(initialization, &next_phase).unwrap();
    assert!(parent.check_scope_admission(&candidate).is_err());
    parent.shrink_scope_unused(&steady, &next_phase).unwrap();
    let second = parent.reserve_scope(&candidate).unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 64 + 8 + 96 + 192);
    assert!(parent.owns(&sample) && parent.owns(&weights));
    assert_eq!(sample.bytes(), &[0; 64]);
    assert_eq!(weights.bytes(), &[0; 8]);
    drop((second, steady, persistent));
    // Closing grants never retires their live payload on the caller's behalf.
    assert_eq!(parent.usage().unwrap().committed_bytes(), 72);
    drop((sample, weights));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn foreign_nested_unscoped_empty_and_overflow_targets_leave_authority_unchanged() {
    let parent = account(1);
    let foreign = account(1);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 40)])
        .unwrap();
    let view = scope.account();
    let before = snapshot(&scope);
    assert_eq!(
        foreign.shrink_scope_unused(&scope, &[(HostMemoryClass::InputPreprocess, 1)]),
        Err(HostMemoryError::InvalidReservation)
    );
    assert_eq!(
        view.shrink_scope_unused(&scope, &[(HostMemoryClass::InputPreprocess, 1)]),
        Err(HostMemoryError::NestedScope)
    );
    assert_eq!(
        parent.shrink_scope_unused(&scope, &[]),
        Err(HostMemoryError::InvalidReservation)
    );
    assert_eq!(
        parent.shrink_scope_unused(&scope, &[(HostMemoryClass::Output, 0)]),
        Err(HostMemoryError::InvalidReservation)
    );
    assert_eq!(
        parent.shrink_scope_unused(
            &scope,
            &[
                (HostMemoryClass::InputPreprocess, usize::MAX),
                (HostMemoryClass::InputPreprocess, 1)
            ]
        ),
        Err(HostMemoryError::ArithmeticOverflow)
    );
    assert_eq!(snapshot(&scope), before);
    // Duplicate claims use the same checked sum as ordinary scope creation.
    parent
        .shrink_scope_unused(
            &scope,
            &[
                (HostMemoryClass::InputPreprocess, 7),
                (HostMemoryClass::InputPreprocess, 3),
            ],
        )
        .unwrap();
    let held = view.reserve(HostMemoryClass::InputPreprocess, 10).unwrap();
    assert!(view.reserve(HostMemoryClass::InputPreprocess, 1).is_err());
    scope.close().unwrap();
    drop(held);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}

struct ReentrantWake {
    parent: HostMemoryAccount,
    wakes: AtomicUsize,
}
impl Wake for ReentrantWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        // A notification must run after the Host ledger lock is released.
        self.parent.usage().unwrap();
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn only_released_credit_notifies_waiters_after_unlock() {
    let parent = account(1);
    let scope = parent.reserve_scope(&claims(30, 20, 0, 0)).unwrap();
    let wake = Arc::new(ReentrantWake {
        parent: parent.clone(),
        wakes: AtomicUsize::new(0),
    });
    let _subscription = parent.subscribe_capacity(Waker::from(wake.clone()));
    parent
        .shrink_scope_unused(&scope, &claims(30, 20, 0, 0))
        .unwrap();
    assert_eq!(wake.wakes.load(Ordering::SeqCst), 0);
    parent
        .shrink_scope_unused(&scope, &claims(10, 0, 0, 0))
        .unwrap();
    assert_eq!(wake.wakes.load(Ordering::SeqCst), 1);
    assert!(parent
        .shrink_scope_unused(&scope, &claims(11, 0, 0, 0))
        .is_err());
    assert_eq!(wake.wakes.load(Ordering::SeqCst), 1);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn closed_and_fail_closed_scopes_reject_shrink_without_any_accounting_change() {
    let parent = account(1);
    let scope = parent.reserve_scope(&claims(30, 20, 0, 0)).unwrap();
    let output = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 10)
        .unwrap();
    parent
        .shrink_scope_unused(&scope, &claims(10, 0, 0, 0))
        .unwrap();
    // Model a revoked authority while retaining its original guard for the
    // defensive closed-authority API check. This is the same ledger transition
    // as close, without manufacturing a second guard or account.
    let id = scope.account.1.as_ref().unwrap().id;
    {
        let mut state = parent.0.state.lock().unwrap();
        let credit = state.scopes.remove(&id).unwrap();
        state.committed = state.committed.subtract(credit.remaining).unwrap();
    }
    let before = parent.usage().unwrap();
    assert_eq!(
        parent.shrink_scope_unused(&scope, &claims(0, 0, 0, 0)),
        Err(HostMemoryError::ScopeClosed)
    );
    assert_eq!(parent.usage().unwrap(), before);
    scope.close().unwrap();
    drop(output);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);

    let scope = parent.reserve_scope(&claims(30, 20, 0, 0)).unwrap();
    let before = snapshot(&scope);
    parent.0.state.lock().unwrap().fail_closed = true;
    assert_eq!(
        parent.shrink_scope_unused(&scope, &claims(0, 0, 0, 0)),
        Err(HostMemoryError::Poisoned)
    );
    assert_eq!(snapshot(&scope), before);
    assert_eq!(scope.close(), Err(HostMemoryError::Poisoned));
    // Failure does not claim to have released the original authorization.
    assert_eq!(parent.0.state.lock().unwrap().committed, before.0);
}
