use super::*;

#[test]
fn live_output_can_consume_a_phase_floor_without_turning_an_at_most_cap_into_growth() {
    let parent = account(1);
    let scope = parent.reserve_scope(&claims(40, 40, 16, 8)).unwrap();
    let input = scope
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 12)
        .unwrap();
    let pixels = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 24)
        .unwrap()
        .freeze();
    let target = claims(32, 32, 0, 0);
    let before = snapshot(&scope);
    assert!(matches!(
        parent.shrink_scope_unused(&scope, &target),
        Err(HostMemoryError::ScopeCapacity { .. })
    ));
    assert_eq!(snapshot(&scope), before);
    parent
        .shrink_scope_unused_to_at_most(&scope, &target)
        .unwrap();
    assert_eq!(
        snapshot(&scope).2,
        Counters {
            total: 44,
            classes: [28, 0, 16, 0]
        }
    );
    assert_eq!(parent.usage().unwrap().committed_bytes(), 80);
    // The cap did not invent another full floor alongside existing pixels.
    assert!(matches!(
        scope.account().reserve(HostMemoryClass::Output, 17),
        Err(HostMemoryError::ScopeCapacity { available: 16, .. })
    ));
    drop((input, pixels));
    // Refunds keep their original route, so another idle-boundary cap may
    // release bytes after those independent owners really retire.
    parent
        .shrink_scope_unused_to_at_most(&scope, &target)
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 64);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn inherited_pixels_keep_their_exact_charge_and_refund_after_at_most_commit() {
    let parent = account(8);
    let physical = parent.backing_bytes(9).unwrap();
    let original = parent
        .reserve_scope(&claims(0, physical * 3, 0, 0))
        .unwrap();
    let original_view = original.account();
    let pixels = original_view
        .allocate_zeroed(HostMemoryClass::Output, 9)
        .unwrap()
        .freeze();
    let clone = pixels.clone();
    let scope = parent
        .exchange_scope(original, &claims(0, physical, 0, 0))
        .unwrap();
    parent
        .shrink_scope_unused_to_at_most(&scope, &claims(0, physical, 0, 0))
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), physical * 2);
    assert_eq!(
        snapshot(&scope).3.classes[HostMemoryClass::Output.index()],
        physical
    );
    let next = scope.account().reserve(HostMemoryClass::Output, 9).unwrap();
    parent
        .shrink_scope_unused_to_at_most(&scope, &claims(0, physical, 0, 0))
        .unwrap();
    assert_eq!(snapshot(&scope).2.total, 0);
    drop(pixels);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), physical);
    drop(clone);
    let recycled = scope.account().reserve(HostMemoryClass::Output, 9).unwrap();
    assert!(parent.0.state.lock().unwrap().scope_refunds.is_empty());
    assert!(matches!(
        original_view.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    scope.close().unwrap();
    drop((next, recycled));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn at_most_invalid_targets_are_atomic_and_large_valid_maxima_do_not_notify_or_grant() {
    let parent = account(1);
    let foreign = account(1);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 40)])
        .unwrap();
    let wake = Arc::new(ReentrantWake {
        parent: parent.clone(),
        wakes: AtomicUsize::new(0),
    });
    let _subscription = parent.subscribe_capacity(Waker::from(wake.clone()));
    let before = snapshot(&scope);
    for target in [
        vec![],
        vec![(HostMemoryClass::Output, 0)],
        vec![
            (HostMemoryClass::InputPreprocess, usize::MAX),
            (HostMemoryClass::InputPreprocess, 1),
        ],
    ] {
        assert!(parent
            .shrink_scope_unused_to_at_most(&scope, &target)
            .is_err());
        assert_eq!(snapshot(&scope), before);
    }
    assert_eq!(
        foreign.shrink_scope_unused_to_at_most(&scope, &[(HostMemoryClass::InputPreprocess, 40)]),
        Err(HostMemoryError::InvalidReservation)
    );
    assert_eq!(
        scope
            .account()
            .shrink_scope_unused_to_at_most(&scope, &[(HostMemoryClass::InputPreprocess, 40)]),
        Err(HostMemoryError::NestedScope)
    );
    parent
        .shrink_scope_unused_to_at_most(&scope, &[(HostMemoryClass::InputPreprocess, 100)])
        .unwrap();
    assert_eq!(snapshot(&scope), before);
    assert_eq!(wake.wakes.load(Ordering::SeqCst), 0);
    let exact = scope
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 40)
        .unwrap();
    assert!(scope
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 1)
        .is_err());
    drop(exact);
    parent
        .shrink_scope_unused_to_at_most(&scope, &[(HostMemoryClass::InputPreprocess, 10)])
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 10);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}
