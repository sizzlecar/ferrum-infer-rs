use super::*;
use crate::vnext::operation::test_only_backing_window_coverage;

fn paged_window_fixture() -> (Harness, LogicalBackingSliceAuthority) {
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
    assert_eq!(authority.evidence().segments().len(), 3);
    for segment in authority.evidence().segments() {
        assert_eq!((segment.offset_bytes(), segment.length_bytes()), (0, 64));
    }
    (harness, authority)
}

#[test]
fn full_coverage_proof_preserves_real_paged_window_subranges() {
    let (harness, authority) = paged_window_fixture();
    let backing = harness.root.dynamic_pools.view(&authority).unwrap();
    let results = test_only_backing_window_coverage(
        harness.runtime.as_ref(),
        backing,
        96,
        63,
        &[
            (0, 96),
            (1, 64),
            (63, 33),
            (95, 1),
            (96, 1),
            (95, 2),
            (0, 0),
            (u64::MAX, 1),
        ],
    )
    .unwrap();
    // Independent physical oracle: the logical window starts at byte63 of
    // the first64-byte page, crosses the next page, and ends31 bytes into
    // the third. All physical pages have separately checked zero offsets.
    let expected = [
        vec![(0, 63, 1), (1, 0, 64), (65, 0, 31)],
        vec![(1, 0, 64)],
        vec![(63, 62, 2), (65, 0, 31)],
        vec![(95, 30, 1)],
    ];
    for (result, expected) in results[..4].iter().zip(expected) {
        assert_eq!(result.as_ref().unwrap(), &expected);
    }
    // Backing retains192 bytes, but proof authority ends at logical byte96.
    assert!(results[4..].iter().all(Result::is_err));
}

#[test]
fn full_coverage_proof_rejects_later_page_evidence_and_invalid_windows() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let mut backing = pools.view(&authority).unwrap();
    let segment = backing.payload.bindings[1].segment.clone();
    backing.payload.bindings[1].segment = BackingSegment::from_chunk(
        segment.pool_id(),
        segment.chunk_ordinal(),
        segment.chunk_generation() + 1,
        segment.offset_bytes(),
        segment.length_bytes(),
    )
    .unwrap();
    assert!(test_only_backing_window_coverage(
        harness.runtime.as_ref(),
        backing,
        96,
        63,
        &[(0, 1)]
    )
    .is_err());
    let mut truncated = pools.view(&authority).unwrap();
    truncated.payload.bindings.pop();
    assert!(test_only_backing_window_coverage(
        harness.runtime.as_ref(),
        truncated,
        96,
        63,
        &[(0, 1)]
    )
    .is_err());
    for (logical_bytes, window) in [(96, 97), (1, u64::MAX), (0, 63)] {
        assert!(test_only_backing_window_coverage(
            harness.runtime.as_ref(),
            pools.view(&authority).unwrap(),
            logical_bytes,
            window,
            &[(0, 1)]
        )
        .is_err());
    }
}

#[test]
fn full_coverage_checks_unused_later_pages_and_current_descriptors() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let runtime = harness.runtime.as_ref();
    // The one-byte prefix consumes only page0, but page2 remains part of the
    // committed backing and must receive the same runtime checks in both arms.
    let backing = pools.view(&authority).unwrap();
    let last_buffer = backing.payload.bindings[2].buffer() as *const TestBuffer as usize;
    let before = runtime.descriptor_queries.load(Ordering::Relaxed);
    test_only_backing_window_coverage(runtime, backing, 1, 0, &[(0, 1)]).unwrap();
    // The helper compares reference and candidate, then constructs the proof.
    // Each validator must visit ALL three current descriptors exactly once.
    assert_eq!(
        runtime.descriptor_queries.load(Ordering::Relaxed) - before,
        3 * 3
    );

    for page in 1..3 {
        let mut backing = pools.view(&authority).unwrap();
        let segment = &backing.payload.bindings[page].segment;
        backing.payload.bindings[page].segment = BackingSegment::from_chunk(
            segment.pool_id(),
            segment.chunk_ordinal(),
            segment.chunk_generation() + 1,
            segment.offset_bytes(),
            segment.length_bytes(),
        )
        .unwrap();
        assert!(test_only_backing_window_coverage(runtime, backing, 1, 0, &[(0, 1)]).is_err());
    }
    // Change the runtime descriptor only after legitimate materialization.
    let backing = pools.view(&authority).unwrap();
    runtime
        .descriptor_mismatch_buffer
        .store(last_buffer, Ordering::Relaxed);
    assert!(test_only_backing_window_coverage(runtime, backing, 1, 0, &[(0, 1)]).is_err());
    runtime
        .descriptor_mismatch_buffer
        .store(0, Ordering::Relaxed);
    test_only_backing_window_coverage(runtime, pools.view(&authority).unwrap(), 1, 0, &[(0, 1)])
        .unwrap();
}

#[test]
fn full_coverage_matches_reference_across_paged_windows_and_invalid_extents() {
    let (harness, authority) = paged_window_fixture();
    for offset in [0, 1, 63, 64, 65, 127, 128, 191, 192, u64::MAX] {
        for bytes in [0, 1, 2, 63, 64, 65, 96, 128, 192, 193, u64::MAX] {
            let backing = harness.root.dynamic_pools.view(&authority).unwrap();
            let result = test_only_backing_window_coverage(
                harness.runtime.as_ref(),
                backing,
                bytes,
                offset,
                &[(0, bytes)],
            );
            let expected = bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= 192);
            assert_eq!(result.is_ok(), expected, "offset={offset} bytes={bytes}");
        }
    }
}

#[test]
fn retained_materialization_rechecks_exact_authority_and_pool_poison() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let backing = pools.view(&authority).unwrap();
    pools.revalidate_view(&authority, &backing).unwrap();
    let pool = Arc::clone(&pools.pools[&harness.pool_ids[0]]);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    let other = claim_size(pools, &pool, 64);
    assert!(pools.revalidate_view(&other, &backing).is_err());
    // A retained chunk keeps memory alive; it cannot make a poisoned pool usable.
    pool.state.lock().unwrap().poisoned = true;
    assert!(pools.revalidate_view(&authority, &backing).is_err());
    pool.state.lock().unwrap().poisoned = false;
    pools.revalidate_view(&authority, &backing).unwrap();
}

#[test]
fn retained_materialization_rejects_generation_range_and_metadata_drift() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let mut backing = pools.view(&authority).unwrap();
    backing.payload.logical_size_bytes -= 1;
    assert!(pools.revalidate_view(&authority, &backing).is_err());
    let mut backing = pools.view(&authority).unwrap();
    let segment = &backing.payload.bindings[1].segment;
    backing.payload.bindings[1].segment = BackingSegment::from_chunk(
        segment.pool_id(),
        segment.chunk_ordinal(),
        segment.chunk_generation() + 1,
        segment.offset_bytes(),
        segment.length_bytes(),
    )
    .unwrap();
    assert!(pools.revalidate_view(&authority, &backing).is_err());
    let mut backing = pools.view(&authority).unwrap();
    backing.payload.bindings.pop();
    assert!(pools.revalidate_view(&authority, &backing).is_err());
    pools
        .revalidate_view(&authority, &pools.view(&authority).unwrap())
        .unwrap();
}

#[test]
fn retained_materialization_rejects_equal_authority_clones_and_foreign_pools() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let backing = pools.view(&authority).unwrap();
    let equal_authority = authority.retained();
    assert_eq!(authority.evidence(), equal_authority.evidence());
    // Value equality cannot transfer the proof to another authority object.
    assert!(pools.revalidate_view(&equal_authority, &backing).is_err());
    pools.view(&equal_authority).unwrap();

    let (foreign, foreign_authority) = paged_window_fixture();
    assert_eq!(harness.pool_ids, foreign.pool_ids);
    assert_ne!(
        authority.evidence().pool_instance_id(),
        foreign_authority.evidence().pool_instance_id()
    );
    // Even the exact original slice cannot authorize another live pool with
    // equal string identifiers. The retained path must check pool instances.
    assert!(foreign
        .root
        .dynamic_pools
        .revalidate_view(&authority, &backing)
        .is_err());
    pools.revalidate_view(&authority, &backing).unwrap();
}

#[test]
fn retained_materialization_rechecks_current_chunk_generation_bounds_and_identity() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let mut backing = pools.view(&authority).unwrap();
    let index = 1;
    let ordinal = authority.evidence().segments()[index].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    assert!(pools.revalidate_view(&authority, &backing).is_err());

    // Use an independently allocated backing to exercise current pool state,
    // while keeping both real allocation grants alive throughout the test.
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
    Arc::get_mut(&mut replacement.backing).unwrap().identity = BackingChunkIdentity::from_parts(
        original.backing.identity.pool_id().clone(),
        ordinal,
        original.backing.identity.generation() + 1,
    )
    .unwrap();
    // Make the retained chunk pointer agree with current state so rejection
    // specifically requires comparing the live generation to authority.
    backing.payload.bindings[index].chunk = Arc::clone(&replacement.backing);
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(pools.revalidate_view(&authority, &backing).is_err());

    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    backing.payload.bindings[index].chunk = Arc::clone(&original.backing);
    let current = Arc::get_mut(&mut replacement.backing).unwrap();
    current.identity = original.backing.identity.clone();
    current.descriptor.size_bytes = original.backing.descriptor.size_bytes - 1;
    backing.payload.bindings[index].chunk = Arc::clone(&replacement.backing);
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    // Identity, retained segment and Arc now agree; only the live bounds fail.
    assert!(pools.revalidate_view(&authority, &backing).is_err());

    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    backing.payload.bindings[index].chunk = Arc::clone(&original.backing);
    Arc::get_mut(&mut replacement.backing).unwrap().descriptor =
        original.backing.descriptor.clone();
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    // Equal chunk identity and bounds cannot substitute a different allocation.
    assert!(pools.revalidate_view(&authority, &backing).is_err());

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
    pools.revalidate_view(&authority, &backing).unwrap();
}

#[test]
fn first_materialization_still_checks_claim_membership_and_physical_projection() {
    let (harness, mut authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let original_claim = authority.evidence.physical_claim_identity.clone();
    let different_claim = PhysicalBackingClaimIdentity::new(
        original_claim.pool_id().clone(),
        vec![ResourceId::new("resource.unclaimed").unwrap()],
    )
    .unwrap();
    Arc::get_mut(&mut authority.evidence.allocation)
        .unwrap()
        .physical_claim_identity = different_claim.clone();
    assert!(pools.view(&authority).is_err());
    // Matching claim identities do not prove membership of this resource.
    Arc::get_mut(&mut authority.segment_lease)
        .unwrap()
        .claim_identity = different_claim;
    assert!(pools.view(&authority).is_err());
    Arc::get_mut(&mut authority.segment_lease)
        .unwrap()
        .claim_identity = original_claim.clone();
    Arc::get_mut(&mut authority.evidence.allocation)
        .unwrap()
        .physical_claim_identity = original_claim;

    let original_segment = authority.evidence.segments[1].clone();
    Arc::get_mut(&mut authority.evidence.allocation)
        .unwrap()
        .segments[1] = BackingSegment::from_chunk(
        original_segment.pool_id(),
        original_segment.chunk_ordinal(),
        original_segment.chunk_generation(),
        original_segment.offset_bytes() + 1,
        original_segment.length_bytes() - 1,
    )
    .unwrap();
    // This region fits its chunk but no longer equals the claimed projection.
    assert!(pools.view(&authority).is_err());
    Arc::get_mut(&mut authority.evidence.allocation)
        .unwrap()
        .segments[1] = original_segment;
    let backing = pools.view(&authority).unwrap();
    pools.revalidate_view(&authority, &backing).unwrap();
}

#[path = "pool_version_tests.rs"]
mod pool_version_tests;

#[path = "sealed_pool_proof_tests.rs"]
mod sealed_pool_proof_tests;
