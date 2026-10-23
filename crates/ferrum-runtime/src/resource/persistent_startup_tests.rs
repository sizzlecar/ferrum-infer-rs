use super::*;

fn configured_plan(catalog: &PoolCatalog, budget: u64) -> crate::MemoryPlan {
    crate::MemoryPlan::from_core(
        budget,
        budget,
        0,
        8,
        vec![],
        catalog.descriptors.clone(),
        &[],
        None,
    )
    .unwrap()
}

#[test]
fn persistent_startup_allocates_exact_pool_targets_through_original_authority() {
    let kv = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        2,
        8192,
        TestDemand::Tokens,
    );
    let fixed = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'b',
        2,
        8192,
        TestDemand::Fixed,
    );
    let mut catalog = combine_catalogs(&[kv, fixed]);
    let memory = configured_plan(&catalog, 8192);
    // Use exactly the compiled pool catalog, including its original ceilings.
    catalog.pools = memory.dynamic_pools().to_vec();
    let h = harness(new_runtime(&catalog, 8192), catalog, 8192, false);
    assert_eq!(h.runtime.allocate_calls(), 0);
    let plan = memory.persistent_preallocation_plan(4, 8, 8).unwrap();
    assert_eq!(plan.persistent_bytes(), 2 * 256 * 8 + 2 * 64 * 8);
    assert_eq!(h.runtime.allocate_calls(), 0, "planning must not allocate");
    let receipt = h
        .root
        .maintenance_controller
        .grow_pools(plan.initial_growth_requests().unwrap())
        .unwrap();
    assert_eq!(h.runtime.allocate_calls(), 2);
    for expected in plan.pools() {
        let actual = receipt
            .growths()
            .iter()
            .find(|g| g.pool_id() == expected.pool_id())
            .unwrap();
        assert_eq!(actual.chunk_bytes(), expected.target_bytes());
        assert_eq!(actual.published_capacity_bytes(), expected.target_bytes());
    }
    assert_eq!(
        h.observed_persistent_capacity(&[], &[4; 16], 8, Default::default()),
        Ok(())
    );
    // No persistent floor is added: the original owner's core minimum remains
    // smaller than this explicit full configured envelope.
    for pool in memory.dynamic_pools() {
        let target = plan
            .pools()
            .iter()
            .find(|p| p.pool_id() == pool.pool_id())
            .unwrap();
        assert!(pool.provisioning().minimum_resident_bytes() < target.target_bytes());
    }
}

#[test]
fn persistent_startup_full_context_budget_failure_precedes_allocation() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        2048,
        TestDemand::Tokens,
    );
    let memory = configured_plan(&catalog, 1024);
    let h = harness(new_runtime(&catalog, 1024), catalog, 1024, false);
    // Shallow decode fits; the configured full contexts do not.
    assert!(memory.startup_workload_peak_bytes(4, 1, 8, 8).unwrap() <= 1024);
    assert!(matches!(
        memory.persistent_preallocation_plan(4, 8, 8),
        Err(VNextError::PlanCapacityExceeded {
            required_bytes: 2048,
            usable_capacity_bytes: 1024
        })
    ));
    assert_eq!(h.runtime.allocate_calls(), 0);
    assert!(h
        .root
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .all(|p| p.resident_bytes() == 0));
}

#[test]
fn persistent_startup_rejects_unsupported_geometry_and_invalid_dimensions() {
    for (profile, lifetime, demand) in [
        (
            linear_profile(),
            AllocationLifetime::Sequence,
            TestDemand::Tokens,
        ),
        (
            linear_profile(),
            AllocationLifetime::Request,
            TestDemand::Fixed,
        ),
    ] {
        let catalog = pool_catalog(profile, lifetime, 'a', 1, 4096, demand);
        let memory = configured_plan(&catalog, 4096);
        assert!(memory.persistent_preallocation_plan(4, 8, 8).is_err());
    }
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        4096,
        TestDemand::Tokens,
    );
    let memory = configured_plan(&catalog, 4096);
    for (context, width, tokens) in [(0, 8, 8), (4, 0, 8), (4, 9, 8), (4, 8, 0), (5, 8, 8)] {
        assert!(memory
            .persistent_preallocation_plan(context, width, tokens)
            .is_err());
    }
}

#[test]
fn persistent_startup_allocation_failure_has_no_partial_publication_or_claim() {
    let mut catalog = combine_catalogs(&[
        pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'a',
            1,
            8192,
            TestDemand::Tokens,
        ),
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Sequence,
            'b',
            1,
            4096,
            TestDemand::Fixed,
        ),
    ]);
    let memory = configured_plan(&catalog, 8192);
    catalog.pools = memory.dynamic_pools().to_vec();
    let runtime = new_runtime(&catalog, 8192);
    runtime.fail_on_call(2);
    let h = harness(Arc::clone(&runtime), catalog, 8192, false);
    let plan = memory.persistent_preallocation_plan(4, 8, 8).unwrap();
    let before = h.root.dynamic_pool_status().unwrap();
    assert!(h
        .root
        .maintenance_controller
        .grow_pools(plan.initial_growth_requests().unwrap())
        .is_err());
    let after = h.root.dynamic_pool_status().unwrap();
    assert_eq!(after.epochs(), before.epochs());
    assert_eq!(after.budget_claimed_bytes(), 0);
    assert!(after
        .pools()
        .iter()
        .all(|p| p.resident_bytes() == 0 && p.pending_growth_bytes() == 0));
    runtime.fail_on_call(0);
    assert!(h
        .root
        .maintenance_controller
        .grow_pools(plan.initial_growth_requests().unwrap())
        .is_ok());
}
