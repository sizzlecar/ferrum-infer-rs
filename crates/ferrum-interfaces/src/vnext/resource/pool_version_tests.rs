use super::*;

#[test]
fn pool_version_requires_exact_authority_owner_and_unmoved_payload_proof() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let mut backing = pools.view_with_pool_version(&authority).unwrap();
    assert!(backing.has_pool_version_proof());
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
    // The existing complete read-side validation must not destroy a hit.
    pools.revalidate_view(&authority, &backing).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());

    let equal_authority = authority.retained();
    assert_eq!(authority.evidence(), equal_authority.evidence());
    assert!(pools
        .revalidate_view_with_pool_version(&equal_authority, &backing)
        .is_err());
    let mut equal_backing = pools.view_with_pool_version(&equal_authority).unwrap();
    // A sibling can move a payload, but cannot retarget its sealed authority.
    // The physical mappings remain valid, so both must take the full fallback.
    std::mem::swap(
        &mut backing.payload.bindings,
        &mut equal_backing.payload.bindings,
    );
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
    assert!(!pools
        .revalidate_view_with_pool_version(&equal_authority, &equal_backing)
        .unwrap());

    let (foreign, _foreign_authority) = paged_window_fixture();
    assert_eq!(harness.pool_ids, foreign.pool_ids);
    assert!(foreign
        .root
        .dynamic_pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .is_err());
    let ordinary = pools.view(&authority).unwrap();
    assert!(!ordinary.has_pool_version_proof());
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &ordinary)
        .unwrap());
}

#[test]
fn pool_version_rechecks_poison_missing_chunk_growth_and_actual_lease_release() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let mut backing = pools.view_with_pool_version(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());

    pool.state.lock().unwrap().poisoned = true;
    // When both predicates fail, preserve the old poison-before-metadata error.
    backing.payload.logical_size_bytes -= 1;
    let expected = pools.revalidate_view(&authority, &backing).unwrap_err();
    let actual = pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    backing.payload.logical_size_bytes += 1;
    pool.state.lock().unwrap().poisoned = false;
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());

    let current = pools.view_with_pool_version(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &current)
        .unwrap());
    let ordinal = authority.evidence().segments()[1].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &current)
        .is_err());
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &current)
        .unwrap());

    let before_growth = pools.view_with_pool_version(&authority).unwrap();
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &before_growth)
        .unwrap());
    let other = claim_size(pools, pool, 64);
    let before_release = pools.view_with_pool_version(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &before_release)
        .unwrap());
    drop(other);
    // This is the real BackingSegmentLease drop/release path, not a stamp edit.
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &before_release)
        .unwrap());
    assert!(pools
        .revalidate_view_with_pool_version(
            &authority,
            &pools.view_with_pool_version(&authority).unwrap()
        )
        .unwrap());
}

#[test]
fn pool_version_rejects_an_independently_admitted_same_generation_allocation() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let backing = pools.view_with_pool_version(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
    let ordinal = authority.evidence().segments()[1].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let (donor, _donor_authority) = paged_window_fixture();
    let donor_pool = &donor.root.dynamic_pools.pools[&donor.pool_ids[0]];
    let mut replacement = donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .remove(&ordinal)
        .unwrap();
    let donor_identity = replacement.backing.identity.clone();
    let donor_descriptor = replacement.backing.descriptor.clone();
    let replacement_backing = Arc::get_mut(&mut replacement.backing).unwrap();
    replacement_backing.identity = original.backing.identity.clone();
    replacement_backing.descriptor = original.backing.descriptor.clone();
    assert!(!Arc::ptr_eq(&original.backing, &replacement.backing));
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);

    // A new full view accepts the new current mapping; the old retained owner
    // is still alive and must not be accepted just because metadata is equal.
    pools.view(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .is_err());

    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let restored_donor = Arc::get_mut(&mut replacement.backing).unwrap();
    restored_donor.identity = donor_identity;
    restored_donor.descriptor = donor_descriptor;
    donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
}

#[test]
fn pool_version_mutable_bindings_clear_proof_and_scalar_metadata_still_validates() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let mut backing = pools.view_with_pool_version(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
    // Matches checkpoint's mutable retention traversal, even without a change.
    assert_eq!(backing.payload.bindings.iter_mut().count(), 3);
    assert!(!backing.has_pool_version_proof());
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());

    let mut metadata = pools.view_with_pool_version(&authority).unwrap();
    metadata.payload.logical_size_bytes -= 1;
    assert!(!metadata.has_pool_version_proof());
    let expected = pools.revalidate_view(&authority, &metadata).unwrap_err();
    let actual = pools
        .revalidate_view_with_pool_version(&authority, &metadata)
        .unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());

    let mut bad_segment = pools.view_with_pool_version(&authority).unwrap();
    let segment = bad_segment.payload.bindings[1].segment.clone();
    bad_segment.payload.bindings[1].segment = BackingSegment::from_chunk(
        segment.pool_id(),
        segment.chunk_ordinal(),
        segment.chunk_generation() + 1,
        segment.offset_bytes(),
        segment.length_bytes(),
    )
    .unwrap();
    assert!(!bad_segment.has_pool_version_proof());
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &bad_segment)
        .is_err());
    let mut truncated = pools.view_with_pool_version(&authority).unwrap();
    truncated.payload.bindings.pop();
    assert!(!truncated.has_pool_version_proof());
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &truncated)
        .is_err());
}

#[test]
fn pool_version_hit_does_not_replace_current_runtime_later_page_validation() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let runtime = harness.runtime.as_ref();
    let backing = pools.view_with_pool_version(&authority).unwrap();
    let last_buffer = backing.payload.bindings[2].buffer() as *const TestBuffer as usize;
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
    runtime
        .descriptor_mismatch_buffer
        .store(last_buffer, Ordering::Relaxed);
    // Pool metadata did not change. Runtime evidence is a separate obligation,
    // including pages beyond the one-byte logical window consumed here.
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &backing)
        .unwrap());
    let result = test_only_backing_window_coverage(runtime, backing, 1, 0, &[(0, 1)]);
    runtime
        .descriptor_mismatch_buffer
        .store(0, Ordering::Relaxed);
    assert!(result.is_err());
    test_only_backing_window_coverage(
        runtime,
        pools.view_with_pool_version(&authority).unwrap(),
        1,
        0,
        &[(0, 1)],
    )
    .unwrap();
}
