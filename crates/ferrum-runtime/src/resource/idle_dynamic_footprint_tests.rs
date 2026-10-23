use super::*;

#[test]
fn partial_retention_footprint_accounts_for_each_cold_pool_without_double_counting() {
    let catalogs = ['a', 'b'].map(|layout| {
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            layout,
            1,
            256,
            TestDemand::Fixed,
        )
    });
    let catalog = combine_catalogs(&catalogs);
    let runtime = new_runtime(&catalog, 512);
    let h = harness(runtime, catalog, 512, false);
    let observe = |expected| {
        let status = h.root.dynamic_pool_status().unwrap();
        let allocations = h.runtime.allocate_calls();
        assert_eq!(
            h.root
                .minimum_runnable_dynamic_bytes_with_residency()
                .unwrap(),
            expected
        );
        assert_eq!(h.root.dynamic_pool_status().unwrap(), status);
        assert_eq!(h.runtime.allocate_calls(), allocations);
    };
    // Both pools are cold, but each still has a canonical 64-byte requirement.
    observe(128);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 192)
        .unwrap();
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[1], 64)
        .unwrap();
    observe(256);
    let IdleDynamicBackingOutcome::Retained(receipt) =
        h.root.try_retain_largest_idle_dynamic_chunks(192).unwrap()
    else {
        panic!("the large chunk must remain while the small pool becomes cold");
    };
    assert_eq!(receipt.retained_bytes(), 192);
    assert_eq!(receipt.released().unwrap().reclaimed_bytes(), 64);
    assert_eq!(
        h.root.dynamic_pool_status().unwrap().budget_claimed_bytes(),
        192
    );
    // max(sum minima=128, old total=192) would incorrectly predict 192.
    // The actual next runnable footprint is max(64,192) + max(64,0) = 256.
    observe(256);
    let allocations = h.runtime.allocate_calls();
    h.root
        .maintenance_controller
        .initialize_pool(&h.pool_ids[1])
        .unwrap();
    assert_eq!(h.runtime.allocate_calls(), allocations + 1);
    assert_eq!(
        h.root.dynamic_pool_status().unwrap().budget_claimed_bytes(),
        256
    );
    observe(256);
    let request = admitted_request(&h.root, "partial-retention-pool-footprint");
    drop(request);
    assert!(matches!(
        h.root.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    observe(128);
    close_dynamic_test_root(h.root);
}
