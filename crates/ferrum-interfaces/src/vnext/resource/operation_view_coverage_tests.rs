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
    let segment = backing.bindings[1].segment.clone();
    backing.bindings[1].segment = BackingSegment::from_chunk(
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
    truncated.bindings.pop();
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
