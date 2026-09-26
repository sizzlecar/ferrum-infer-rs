use super::*;

fn fixture() -> (Harness, LogicalBackingSliceAuthority) {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Request,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    let maintenance = &harness.root.maintenance_controller;
    maintenance.initialize_pool(&harness.pool_ids[0]).unwrap();
    maintenance.grow_pool(&harness.pool_ids[0], 64).unwrap();
    maintenance.grow_pool(&harness.pool_ids[0], 64).unwrap();
    let pool = Arc::clone(&harness.root.dynamic_pools.pools[&harness.pool_ids[0]]);
    let authority = claim_size(&harness.root.dynamic_pools, &pool, 192);
    (harness, authority)
}

#[test]
fn fresh_backing_validation_matches_real_view_and_does_not_retain_the_claim() {
    let (harness, authority) = fixture();
    let pools = &harness.root.dynamic_pools;
    let lease = Arc::downgrade(&authority.segment_lease);
    {
        let fresh = pools
            .validate_view_many(std::slice::from_ref(&authority))
            .unwrap();
        let view = pools.view(&authority).unwrap();
        assert_eq!(fresh.size_bytes(), view.size_bytes());
        assert_eq!(fresh.capacity_size_bytes(), view.capacity_size_bytes());
        assert_eq!(fresh.alignment_bytes(), view.alignment_bytes());
        assert_eq!(fresh.usage(), view.usage());
        assert_eq!(fresh.element_type(), view.element_type());
        assert_eq!(fresh.storage_profile(), view.storage_profile());
        assert_eq!(fresh.segment_count(), view.segment_bindings().len());
        assert_eq!(fresh.physical_coverage_bytes().unwrap(), 192);
        // Independent old full-view oracle includes runtime descriptors and a
        // window crossing three real chunks, rather than just metadata Eq.
        crate::vnext::operation::test_only_backing_window_coverage(
            harness.runtime.as_ref(),
            view,
            96,
            63,
            &[(0, 96), (95, 1)],
        )
        .unwrap();
    }
    drop(authority);
    assert!(
        lease.upgrade().is_none(),
        "numeric inspection retained a physical claim"
    );
}

#[test]
fn fresh_backing_validation_rereads_runtime_descriptors_and_pool_poison() {
    let (harness, authority) = fixture();
    let pools = &harness.root.dynamic_pools;
    pools
        .validate_view_many(std::slice::from_ref(&authority))
        .unwrap();
    harness
        .runtime
        .backing_descriptor_changed
        .store(true, Ordering::Release);
    assert!(pools
        .validate_view_many(std::slice::from_ref(&authority))
        .is_err());
    assert!(crate::vnext::operation::test_only_backing_window_coverage(
        harness.runtime.as_ref(),
        pools.view(&authority).unwrap(),
        96,
        63,
        &[(0, 96)],
    )
    .is_err());
    harness
        .runtime
        .backing_descriptor_changed
        .store(false, Ordering::Release);
    pools
        .validate_view_many(std::slice::from_ref(&authority))
        .unwrap();
    let pool = &pools.pools[&harness.pool_ids[0]];
    pool.state.lock().unwrap().poisoned = true;
    assert!(pools.view(&authority).is_err());
    assert!(pools
        .validate_view_many(std::slice::from_ref(&authority))
        .is_err());
    // Restore the injected state only after checking both failure paths, so
    // the real fixture can perform its normal claim cleanup.
    pool.state.lock().unwrap().poisoned = false;
}

#[test]
fn fresh_backing_validation_rejects_other_pool_claim_and_invalid_logical_extent() {
    let (harness, authority) = fixture();
    let (other, _) = fixture();
    assert!(other.root.dynamic_pools.view(&authority).is_err());
    assert!(other
        .root
        .dynamic_pools
        .validate_view_many(std::slice::from_ref(&authority))
        .is_err());
    let pools = &harness.root.dynamic_pools;
    let mut malformed = authority.retained();
    malformed.evidence.logical_size_bytes = 0;
    assert!(pools.view(&malformed).is_err());
    assert!(pools
        .validate_view_many(std::slice::from_ref(&malformed))
        .is_err());
    malformed.evidence.logical_size_bytes = authority.capacity_size_bytes() + 1;
    assert!(pools.view(&malformed).is_err());
    assert!(pools
        .validate_view_many(std::slice::from_ref(&malformed))
        .is_err());
    assert!(pools.validate_view_many(&[]).is_err());
    // Neither failed inspection consumes or weakens the original authority.
    pools
        .validate_view_many(std::slice::from_ref(&authority))
        .unwrap();
}
