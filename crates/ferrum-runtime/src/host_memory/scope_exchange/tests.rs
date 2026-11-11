use super::*;
fn account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            100,
            HostMemoryClassLimits {
                input_preprocess_bytes: 100,
                output_bytes: 100,
                transfer_staging_bytes: 100,
                weight_backing_bytes: 100,
            },
        )
        .unwrap(),
    )
}

#[test]
fn exchange_at_full_capacity_preserves_used_weights_once_and_invalidates_old_credit() {
    let parent = account();
    let old = parent
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 64),
            (HostMemoryClass::WeightBacking, 16),
        ])
        .unwrap();
    let old_view = old.account();
    let weights = old_view
        .allocate_zeroed(HostMemoryClass::WeightBacking, 16)
        .unwrap()
        .freeze();
    let external = parent.reserve(HostMemoryClass::Output, 20).unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    assert!(parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 64)])
        .is_err());
    let next = parent
        .exchange_scope(
            old,
            &[
                (HostMemoryClass::InputPreprocess, 64),
                (HostMemoryClass::WeightBacking, 0),
            ],
        )
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    assert_eq!(
        parent
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::WeightBacking),
        16
    );
    assert!(matches!(
        old_view.reserve(HostMemoryClass::InputPreprocess, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    let inputs = next
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 64)
        .unwrap();
    assert!(parent.reserve(HostMemoryClass::InputPreprocess, 1).is_err());
    assert!(parent.owns(&weights));
    drop(inputs);
    drop(weights);
    // The original weight's charge returns to the new scope, rather than
    // disappearing or making already promised replacement capacity unusable.
    let replacements = next
        .account()
        .reserve(HostMemoryClass::WeightBacking, 16)
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 100);
    drop(replacements);
    next.close().unwrap();
    drop(external);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn failed_exchange_returns_unchanged_authority_claims_and_refund_lineage() {
    let parent = account();
    let initial = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 40)])
        .unwrap();
    let original_view = initial.account();
    let retained = original_view
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 10)
        .unwrap();
    let old = parent
        .exchange_scope(initial, &[(HostMemoryClass::InputPreprocess, 30)])
        .unwrap();
    let old_view = old.account();
    let external = parent.reserve(HostMemoryClass::Output, 60).unwrap();
    let before = parent.usage().unwrap();
    let failure = parent
        .exchange_scope(old, &[(HostMemoryClass::InputPreprocess, 50)])
        .err()
        .expect("growth cannot fit");
    assert!(matches!(failure.error(), HostMemoryError::Capacity { .. }));
    assert_eq!(parent.usage().unwrap(), before);
    let (_, old) = failure.into_parts();
    assert!(matches!(
        original_view.reserve(HostMemoryClass::InputPreprocess, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    let remaining = old_view
        .reserve(HostMemoryClass::InputPreprocess, 30)
        .unwrap();
    drop(remaining);
    drop(retained);
    let full = old
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 40)
        .unwrap();
    drop(full);
    old.close().unwrap();
    drop(external);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn chained_exchanges_refund_direct_and_inherited_backing_exactly_once() {
    let parent = account();
    let a = parent
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 32),
            (HostMemoryClass::WeightBacking, 16),
        ])
        .unwrap();
    let a_view = a.account();
    let a_bytes = a_view
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap();
    let b = parent
        .exchange_scope(
            a,
            &[
                (HostMemoryClass::InputPreprocess, 32),
                (HostMemoryClass::WeightBacking, 16),
            ],
        )
        .unwrap();
    let b_view = b.account();
    let b_input = b_view
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
        .unwrap();
    let b_weights = b_view
        .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
        .unwrap();
    let c = parent
        .exchange_scope(
            b,
            &[
                (HostMemoryClass::InputPreprocess, 16),
                (HostMemoryClass::WeightBacking, 4),
            ],
        )
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 56);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 20);
    assert_eq!(
        parent
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        12
    );
    assert_eq!(
        parent
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::WeightBacking),
        8
    );
    assert!(matches!(
        b_view.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    drop(a_bytes);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 12);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 56);
    drop(b_input);
    drop(b_weights);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    let all = c
        .account()
        .reserve_many(&[
            (HostMemoryClass::InputPreprocess, 40),
            (HostMemoryClass::WeightBacking, 16),
        ])
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 56);
    assert!(parent.0.state.lock().unwrap().scope_refunds.is_empty());
    c.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 56);
    drop(all);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn unscoped_original_charges_do_not_refund_into_a_newly_scoped_class() {
    let parent = account();
    let a = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 20)])
        .unwrap();
    let view = a.account();
    let inherited = view
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap();
    let ambient = view.allocate_zeroed(HostMemoryClass::Output, 10).unwrap();
    let b = parent
        .exchange_scope(
            a,
            &[
                (HostMemoryClass::InputPreprocess, 12),
                (HostMemoryClass::Output, 5),
            ],
        )
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 35);
    drop(ambient);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 25);
    let exact = b.account().reserve(HostMemoryClass::Output, 5).unwrap();
    assert!(matches!(
        b.account().reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::ScopeCapacity { .. })
    ));
    drop(exact);
    drop(inherited);
    b.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn closing_successor_leaves_original_live_bytes_charged_until_their_last_owner() {
    let parent = account();
    let a = parent
        .reserve_scope(&[(HostMemoryClass::Output, 32)])
        .unwrap();
    let bytes = a
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 12)
        .unwrap()
        .freeze();
    let clone = bytes.clone();
    let b = parent
        .exchange_scope(a, &[(HostMemoryClass::Output, 20)])
        .unwrap();
    b.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 12);
    assert!(parent.0.state.lock().unwrap().scope_refunds.is_empty());
    drop(bytes);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 12);
    drop(clone);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn exchange_rejects_foreign_parent_and_nested_receiver_without_consuming_old_scope() {
    let parent = account();
    let foreign = account();
    let a = parent
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 20)])
        .unwrap();
    let failure = foreign
        .exchange_scope(a, &[(HostMemoryClass::InputPreprocess, 20)])
        .unwrap_err();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 20);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    let (_, a) = failure.into_parts();
    let view = a.account();
    let failure = view
        .exchange_scope(a, &[(HostMemoryClass::InputPreprocess, 20)])
        .unwrap_err();
    assert_eq!(failure.error(), &HostMemoryError::NestedScope);
    let (_, a) = failure.into_parts();
    let held = a
        .account()
        .reserve(HostMemoryClass::InputPreprocess, 20)
        .unwrap();
    drop(held);
    a.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}
