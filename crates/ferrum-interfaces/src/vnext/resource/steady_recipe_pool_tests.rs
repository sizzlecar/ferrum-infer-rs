//! Direct real-pool coverage of the new multi-pool materializer.
use super::*;

fn two_pools() -> (Harness, Vec<LogicalBackingSliceAuthority>) {
    let catalog = combine_catalogs(&['a', 'b'].map(|digit| {
        pool_catalog(
            linear_profile(),
            AllocationLifetime::Request,
            digit,
            1,
            256,
            TestDemand::Fixed,
        )
    }));
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(runtime, catalog, 512, false);
    let mut authorities = Vec::new();
    for id in &harness.pool_ids {
        harness
            .root
            .maintenance_controller
            .initialize_pool(id)
            .unwrap();
        authorities.push(claim_size(
            &harness.root.dynamic_pools,
            &harness.root.dynamic_pools.pools[id],
            64,
        ));
    }
    (harness, authorities)
}

#[test]
fn steady_pool_permit_checks_all_pools_and_releases_partial_retention_on_failure() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let first = &pools.pools[&harness.pool_ids[0]];
    let second = &pools.pools[&harness.pool_ids[1]];
    let first_ordinal = authorities[0].evidence().segments()[0].chunk_ordinal();
    let second_ordinal = authorities[1].evidence().segments()[0].chunk_ordinal();
    let first_chunk = Arc::clone(&first.state.lock().unwrap().chunks[&first_ordinal].backing);
    let baseline = Arc::strong_count(&first_chunk);
    let groups = authorities
        .iter()
        .map(std::slice::from_ref)
        .collect::<Vec<_>>();
    let forward = pools.steady_backing_batch(&groups).unwrap();
    let reverse = pools.steady_backing_batch(&[groups[1], groups[0]]).unwrap();
    assert_eq!(forward.resources.len(), 2);
    assert_eq!(forward.bindings.len(), 2);
    for (index, authority) in authorities.iter().enumerate() {
        let old = pools.view(authority).unwrap();
        assert!(Arc::ptr_eq(
            &forward.bindings[index].chunk,
            &old.bindings[0].chunk
        ));
        assert!(Arc::ptr_eq(
            &reverse.bindings[1 - index].chunk,
            &old.bindings[0].chunk
        ));
        assert_eq!(
            forward.resources[index].logical_size_bytes,
            old.logical_size_bytes
        );
        assert_eq!(
            forward.resources[index].capacity_size_bytes,
            old.capacity_size_bytes
        );
    }
    drop(forward);
    drop(reverse);
    assert_eq!(Arc::strong_count(&first_chunk), baseline);

    // The first group is retained before validation reaches the failed second
    // group. Error cleanup must release both locks and all temporary retentions.
    second.state.lock().unwrap().poisoned = true;
    assert!(pools.steady_backing_batch(&groups).is_err());
    assert!(first.state.try_lock().is_ok());
    assert!(second.state.try_lock().is_ok());
    assert_eq!(Arc::strong_count(&first_chunk), baseline);
    second.state.lock().unwrap().poisoned = false;

    let removed = second
        .state
        .lock()
        .unwrap()
        .chunks
        .remove(&second_ordinal)
        .unwrap();
    assert!(pools.steady_backing_batch(&groups).is_err());
    assert!(first.state.try_lock().is_ok());
    assert!(second.state.try_lock().is_ok());
    assert_eq!(Arc::strong_count(&first_chunk), baseline);
    second
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(second_ordinal, removed);

    let (foreign, foreign_authorities) = two_pools();
    assert_eq!(harness.pool_ids, foreign.pool_ids);
    assert!(pools
        .steady_backing_batch(&[groups[0], std::slice::from_ref(&foreign_authorities[1])])
        .is_err());
    assert!(first.state.try_lock().is_ok());
    assert!(second.state.try_lock().is_ok());
    assert_eq!(Arc::strong_count(&first_chunk), baseline);
    drop(pools.steady_backing_batch(&groups).unwrap());
    drop(foreign_authorities);
    close_dynamic_test_root(foreign.root);
    drop(first_chunk);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn steady_pool_permit_uses_actual_current_allocation_and_preserves_old_view_checks() {
    let (harness, authorities) = two_pools();
    let (donor, donor_authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let donor_pool = &donor.root.dynamic_pools.pools[&donor.pool_ids[0]];
    let ordinal = authorities[0].evidence().segments()[0].chunk_ordinal();
    let old_view = pools.view(&authorities[0]).unwrap();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let mut replacement = donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .remove(&ordinal)
        .unwrap();
    let donor_identity = replacement.backing.identity.clone();
    let donor_descriptor = replacement.backing.descriptor.clone();
    let current = Arc::get_mut(&mut replacement.backing).unwrap();
    current.identity = original.backing.identity.clone();
    current.descriptor = original.backing.descriptor.clone();
    assert!(!Arc::ptr_eq(&replacement.backing, &original.backing));
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);

    // A fresh materialization is allowed to use a valid current allocation.
    // It must not silently return the old retained Arc because IDs agree.
    let fresh = pools.view(&authorities[0]).unwrap();
    let batch = pools
        .steady_backing_batch(&[std::slice::from_ref(&authorities[0])])
        .unwrap();
    assert!(Arc::ptr_eq(
        &batch.bindings[0].chunk,
        &fresh.bindings[0].chunk
    ));
    assert!(!Arc::ptr_eq(
        &batch.bindings[0].chunk,
        &old_view.bindings[0].chunk
    ));
    assert!(pools.revalidate_view(&authorities[0], &old_view).is_err());
    drop(batch);
    drop(fresh);

    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let current = Arc::get_mut(&mut replacement.backing).unwrap();
    current.identity = donor_identity;
    current.descriptor = donor_descriptor;
    donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    pools.revalidate_view(&authorities[0], &old_view).unwrap();
    drop(old_view);
    drop(authorities);
    drop(donor_authorities);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(donor.root);
}
