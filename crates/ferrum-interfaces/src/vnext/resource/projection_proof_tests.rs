//! Projection proof boundaries exercised through real admitted pool claims.
use super::segment_permit_tests::{expectations, two_pools, windows};
use super::*;

// A distinct immutable allocation with identical values is deliberately not
// the same allocation owner. Keep this construction local to adversarial tests.
fn copy_allocation(
    a: &LogicalBackingSliceAllocationEvidence,
) -> LogicalBackingSliceAllocationEvidence {
    LogicalBackingSliceAllocationEvidence {
        domain_id: a.domain_id,
        pool_id: a.pool_id.clone(),
        resource_id: a.resource_id.clone(),
        pool_instance_id: a.pool_instance_id,
        physical_claim_identity: a.physical_claim_identity.clone(),
        reusable_execution_bucket_id: a.reusable_execution_bucket_id.clone(),
        segment_generation: a.segment_generation,
        segments: a.segments.clone(),
        physical_offset_bytes: a.physical_offset_bytes,
        capacity_size_bytes: a.capacity_size_bytes,
        physical_size_bytes: a.physical_size_bytes,
        alignment_bytes: a.alignment_bytes,
        usage: a.usage,
        element_type: a.element_type,
        storage_profile: a.storage_profile,
        initialization: a.initialization,
        fingerprint: a.fingerprint.clone(),
    }
}

#[test]
fn projection_proof_first_invalid_projection_is_not_sealed_and_wire_evidence_is_unchanged() {
    let (harness, mut authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let original = Arc::clone(&authorities[0].evidence.allocation);
    let before = authorities[0].evidence.clone();
    let wire_before = serde_json::to_value(&before).unwrap();
    let proof = authorities[0].projection_proof_for_test();
    assert!(proof.get().is_none());

    // Identity and logical bounds fit, but the claimed 63-byte projection
    // disagrees with the real 64-byte segment list and must not seal a proof.
    let mut malformed = copy_allocation(&original);
    malformed.capacity_size_bytes -= 1;
    authorities[0].evidence.logical_size_bytes = malformed.capacity_size_bytes;
    authorities[0].evidence.allocation = Arc::new(malformed);
    assert!(pools.view(&authorities[0]).is_err());
    assert!(proof.get().is_none());

    authorities[0].evidence.allocation = original;
    authorities[0].evidence.logical_size_bytes = before.logical_size_bytes;
    drop(pools.view(&authorities[0]).unwrap());
    assert!(proof.get().is_some());
    assert!(authorities[0].projection_proof_matches(pool));
    assert_eq!(authorities[0].evidence(), &before);
    assert_eq!(
        serde_json::to_value(authorities[0].evidence()).unwrap(),
        wire_before
    );
    drop(before);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn projection_proof_retained_and_lane_authorities_still_validate_current_logical_size() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    drop(pools.view(&authorities[0]).unwrap());
    let proof = authorities[0].projection_proof_for_test();
    let mut retained = authorities[0].retained();
    let lane = harness.root.create_execution_lane().unwrap();
    let lane_retained = authorities[0].retained_for_lane(lane.id());
    assert!(Arc::ptr_eq(&proof, &retained.projection_proof_for_test()));
    assert!(Arc::ptr_eq(
        &proof,
        &lane_retained.projection_proof_for_test()
    ));
    assert!(retained.projection_proof_matches(pool));
    assert!(lane_retained.projection_proof_matches(pool));
    drop(pools.view(&lane_retained).unwrap());

    retained.evidence.logical_size_bytes = 0;
    assert!(pools.view(&retained).is_err());
    retained.evidence.logical_size_bytes = retained.capacity_size_bytes() + 1;
    assert!(pools.view(&retained).is_err());
    retained.evidence.logical_size_bytes = 32;
    let current = pools.view(&retained).unwrap();
    assert_eq!(current.logical_size_bytes, 32);
    assert!(retained.projection_proof_matches(pool));
    drop(current);
    drop(retained);
    drop(lane_retained);
    drop(lane);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn projection_proof_foreign_allocation_lease_and_pool_cannot_reuse_exact_proof() {
    let (harness, mut authorities) = two_pools();
    let (foreign, foreign_authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    drop(pools.view(&authorities[0]).unwrap());
    let original = Arc::clone(&authorities[0].evidence.allocation);
    let equal = Arc::new(copy_allocation(&original));
    assert_eq!(equal, original);
    assert!(!Arc::ptr_eq(&equal, &original));
    authorities[0].evidence.allocation = equal;
    assert!(!authorities[0].projection_proof_matches(pool));
    drop(pools.view(&authorities[0]).unwrap()); // Valid replacement takes full validation.
    assert!(!authorities[0].projection_proof_matches(pool)); // First seal stays exact.

    let mut malformed = copy_allocation(&original);
    malformed.capacity_size_bytes -= 1;
    authorities[0].evidence.logical_size_bytes = malformed.capacity_size_bytes;
    authorities[0].evidence.allocation = Arc::new(malformed);
    assert!(!authorities[0].projection_proof_matches(pool));
    assert!(pools.view(&authorities[0]).is_err());
    authorities[0].evidence.allocation = original;
    authorities[0].evidence.logical_size_bytes = 64;

    let original_lease = Arc::clone(&authorities[0].segment_lease);
    authorities[0].segment_lease = Arc::clone(&foreign_authorities[0].segment_lease);
    assert!(!authorities[0].projection_proof_matches(pool));
    assert!(pools.view(&authorities[0]).is_err());
    authorities[0].segment_lease = original_lease;
    assert!(authorities[0].projection_proof_matches(pool));
    let foreign_pool = &foreign.root.dynamic_pools.pools[&harness.pool_ids[0]];
    assert!(!authorities[0].projection_proof_matches(foreign_pool));
    assert!(foreign.root.dynamic_pools.view(&authorities[0]).is_err());
    drop(pools.view(&authorities[0]).unwrap());
    drop(authorities);
    drop(foreign_authorities);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(foreign.root);
}

#[test]
fn projection_proof_does_not_skip_current_expected_windows_or_pool_poison() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let groups = authorities
        .iter()
        .map(std::slice::from_ref)
        .collect::<Vec<_>>();
    let expected = expectations(&groups);
    let requested = windows(&expected);
    drop(
        pools
            .segment_backing_batch(&groups, &expected, &requested)
            .unwrap(),
    );
    for authority in &authorities {
        assert!(authority.projection_proof_matches(&pools.pools[authority.evidence.pool_id()]));
    }
    let mut bad = expected.clone();
    bad[1].logical_bytes -= 1;
    assert!(pools
        .segment_backing_batch(&groups, &bad, &requested)
        .is_err());
    bad = expected.clone();
    bad[1].element_type = if bad[1].element_type == ElementType::U8 {
        ElementType::F32
    } else {
        ElementType::U8
    };
    assert!(pools
        .segment_backing_batch(&groups, &bad, &requested)
        .is_err());
    bad = expected.clone();
    bad[1].alignment_bytes *= 2;
    assert!(pools
        .segment_backing_batch(&groups, &bad, &requested)
        .is_err());

    let mut bad_window = requested.clone();
    bad_window[1].offset_bytes = expected[1].logical_bytes;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &bad_window)
        .is_err());
    bad_window = requested.clone();
    bad_window[1].element_type = if expected[1].element_type == ElementType::U8 {
        ElementType::F32
    } else {
        ElementType::U8
    };
    assert!(pools
        .segment_backing_batch(&groups, &expected, &bad_window)
        .is_err());
    bad_window = requested.clone();
    bad_window[1].alignment_bytes = expected[1].alignment_bytes * 2;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &bad_window)
        .is_err());
    let second = &pools.pools[&harness.pool_ids[1]];
    second.state.lock().unwrap().poisoned = true;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    second.state.lock().unwrap().poisoned = false;
    drop(
        pools
            .segment_backing_batch(&groups, &expected, &requested)
            .unwrap(),
    );
    for pool in pools.pools.values() {
        assert!(pool.state.try_lock().is_ok());
    }
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn projection_proof_weak_sidecar_does_not_keep_lease_or_allocation_alive() {
    let (harness, mut authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    drop(pools.view(&authorities[0]).unwrap());
    // With no other strong owner, the proof's Weak seals alone must prevent
    // in-place mutation of the successfully checked allocation and lease.
    assert_eq!(Arc::strong_count(&authorities[0].evidence.allocation), 1);
    assert_eq!(Arc::strong_count(&authorities[0].segment_lease), 1);
    assert!(Arc::get_mut(&mut authorities[0].evidence.allocation).is_none());
    assert!(Arc::get_mut(&mut authorities[0].segment_lease).is_none());
    let proof = authorities[0].projection_proof_for_test();
    let lease = Arc::downgrade(&authorities[0].segment_lease);
    let allocation = Arc::downgrade(&authorities[0].evidence.allocation);
    let owner = Arc::downgrade(pool);
    assert!(proof.get().is_some());
    assert!(
        pool.state
            .lock()
            .unwrap()
            .live_occupancy
            .total()
            .claim_count()
            > 0
    );
    drop(authorities);
    assert!(lease.upgrade().is_none());
    assert!(allocation.upgrade().is_none());
    assert_eq!(
        pool.state
            .lock()
            .unwrap()
            .live_occupancy
            .total()
            .claim_count(),
        0
    );
    // The retained proof must not prevent the actual released extent from
    // satisfying another claim, nor keep the pool alive across root close.
    drop(claim_size(pools, pool, 64));
    close_dynamic_test_root(harness.root);
    assert!(owner.upgrade().is_none());
    assert!(proof.get().is_some());
}
