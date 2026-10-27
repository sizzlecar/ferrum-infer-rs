use super::*;

fn account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            100,
            HostMemoryClassLimits {
                input_preprocess_bytes: 100,
                transfer_staging_bytes: 100,
                output_bytes: 100,
                weight_backing_bytes: 100,
            },
        )
        .unwrap(),
    )
}

#[test]
fn output_scope_excludes_parent_consumers_and_recycles_freed_backing() {
    let parent = account();
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, 80)])
        .unwrap();
    let view = scope.account();
    let unrelated = account();
    assert!(view.same_parent(&parent));
    assert!(!view.same_parent(&unrelated));
    let external = parent
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 20)
        .unwrap();
    // The ordinary HTTP encoder sees the committed grant and cannot steal it.
    assert!(matches!(
        parent.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::Capacity { .. })
    ));
    let mut reserved = view
        .reserve_many(&[(HostMemoryClass::Output, 30), (HostMemoryClass::Output, 20)])
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    let smaller = reserved.pop().unwrap().allocate_zeroed().unwrap();
    let output = reserved.pop().unwrap().allocate_zeroed().unwrap().freeze();
    assert!(parent.owns(&output));
    assert!(view.owns(&output));
    assert!(!unrelated.owns(&output));
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 70);
    drop(smaller);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 50);
    let recycled = view.reserve(HostMemoryClass::Output, 50).unwrap();
    assert!(matches!(
        view.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::ScopeCapacity { .. })
    ));
    scope.close().unwrap();
    // Existing unallocated reservations remain authorized after close too.
    let recycled = recycled.allocate_zeroed().unwrap();
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 100);
    drop((external, recycled));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 30);
    let slice = output.slice(0..1).unwrap();
    drop(output);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 30);
    drop(slice);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn scope_drop_returns_unused_credit_even_when_cached_views_and_outputs_live() {
    let parent = account();
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, 80)])
        .unwrap();
    let cached = scope.account();
    let output = cached
        .allocate_zeroed(HostMemoryClass::Output, 30)
        .unwrap()
        .freeze();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 80);
    drop(scope);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 30);
    assert!(matches!(
        cached.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    assert!(parent.owns(&output));
    let normal = parent.reserve(HostMemoryClass::Output, 70).unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    drop((normal, output, cached));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert!(parent.0.state.lock().unwrap().scopes.is_empty());
}

#[test]
fn mixed_scope_and_parent_reservation_failure_is_atomic() {
    let parent = account();
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, 80)])
        .unwrap();
    let view = scope.account();
    let external = parent
        .reserve(HostMemoryClass::TransferStaging, 20)
        .unwrap();
    assert!(matches!(
        view.reserve_many(&[
            (HostMemoryClass::Output, 10),
            (HostMemoryClass::TransferStaging, 1)
        ]),
        Err(HostMemoryError::Capacity { .. })
    ));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    // Failed admission consumed no scoped credit.
    drop(view.reserve(HostMemoryClass::Output, 80).unwrap());
    drop(external);
    let mixed = view
        .reserve_many(&[
            (HostMemoryClass::Output, 70),
            (HostMemoryClass::TransferStaging, 20),
        ])
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    assert!(matches!(
        view.reserve(HostMemoryClass::Output, 11),
        Err(HostMemoryError::ScopeCapacity { available: 10, .. })
    ));
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 90);
    drop(mixed);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn scope_reserves_physical_padding_once_for_each_future_allocation() {
    let parent = HostMemoryAccount::with_allocation_alignment(account().limits(), 8).unwrap();
    let physical = parent.backing_bytes(1).unwrap();
    assert_eq!(physical, 15);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, physical * 2)])
        .unwrap();
    let outputs = scope
        .account()
        .reserve_many(&[(HostMemoryClass::Output, 1), (HostMemoryClass::Output, 1)])
        .unwrap()
        .into_iter()
        .map(|r| r.allocate_zeroed().unwrap().freeze())
        .collect::<Vec<_>>();
    assert_eq!(parent.usage().unwrap().committed_bytes(), physical * 2);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), physical * 2);
    for output in &outputs {
        assert_eq!(output.backing_bytes(), physical);
        assert_eq!(output.bytes().as_ptr() as usize % 8, 0);
    }
    scope.close().unwrap();
    drop(outputs);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cloned_scope_views_cannot_overdraw_credit_concurrently() {
    let parent = account();
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, 80)])
        .unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let threads = (0..2)
        .map(|_| {
            let view = scope.account();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                view.reserve(HostMemoryClass::Output, 50)
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let results = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, Err(HostMemoryError::ScopeCapacity { .. })))
            .count(),
        1
    );
    assert_eq!(parent.usage().unwrap().committed_bytes(), 80);
    drop(scope);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 50);
    drop(results);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn close_racing_a_reservation_preserves_exact_outstanding_charge() {
    let parent = account();
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::Output, 80)])
        .unwrap();
    let view = scope.account();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let gate = barrier.clone();
    let reserve = std::thread::spawn(move || {
        gate.wait();
        view.reserve(HostMemoryClass::Output, 50)
    });
    barrier.wait();
    scope.close().unwrap();
    match reserve.join().unwrap() {
        Ok(reservation) => {
            assert_eq!(parent.usage().unwrap().committed_bytes(), 50);
            drop(reservation.allocate_zeroed().unwrap());
        }
        Err(HostMemoryError::ScopeClosed) => {}
        Err(error) => panic!("unexpected scope close race: {error}"),
    }
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert!(parent.0.state.lock().unwrap().scopes.is_empty());
}

#[test]
fn invalid_or_over_capacity_scope_has_no_charge_and_multiple_classes_close_cleanly() {
    let parent = account();
    assert!(matches!(
        parent.reserve_scope(&[]),
        Err(HostMemoryError::InvalidReservation)
    ));
    assert!(matches!(
        parent.reserve_scope(&[(HostMemoryClass::Output, 101)]),
        Err(HostMemoryError::Capacity { .. })
    ));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::Output, 20),
            (HostMemoryClass::Output, 30),
            (HostMemoryClass::TransferStaging, 40),
        ])
        .unwrap();
    let view = scope.account();
    assert!(matches!(
        view.reserve_scope(&[(HostMemoryClass::Output, 1)]),
        Err(HostMemoryError::NestedScope)
    ));
    let allocations = view
        .reserve_many(&[
            (HostMemoryClass::Output, 50),
            (HostMemoryClass::TransferStaging, 40),
        ])
        .unwrap();
    drop(allocations);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 90);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    // Classes absent from the grant retain their ordinary parent behavior.
    let unaffected = view.reserve(HostMemoryClass::InputPreprocess, 100).unwrap();
    drop(unaffected);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn explicit_zero_scope_ceiling_denies_new_backing_without_falling_back_to_parent() {
    let parent = account();
    for output_bytes in [0, 20] {
        let scope = parent
            .reserve_scope(&[
                (HostMemoryClass::Output, output_bytes),
                (HostMemoryClass::WeightBacking, 0),
            ])
            .unwrap();
        let view = scope.account();
        assert!(matches!(
            view.reserve(HostMemoryClass::WeightBacking, 1),
            Err(HostMemoryError::ScopeCapacity {
                class: HostMemoryClass::WeightBacking,
                requested: 1,
                available: 0,
            })
        ));
        let parent_backing = parent.reserve(HostMemoryClass::WeightBacking, 1).unwrap();
        if output_bytes > 0 {
            let output = view.reserve(HostMemoryClass::Output, output_bytes).unwrap();
            assert_eq!(parent.usage().unwrap().committed_bytes(), output_bytes + 1);
            drop(output);
        }
        drop(parent_backing);
        drop(scope);
        assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
        assert!(matches!(
            view.reserve(HostMemoryClass::WeightBacking, 1),
            Err(HostMemoryError::ScopeClosed)
        ));
    }
}
