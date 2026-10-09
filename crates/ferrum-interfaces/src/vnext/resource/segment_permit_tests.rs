//! Actual multi-pool admission, replacement and whole-segment failure boundaries.
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

fn expectations(groups: &[&[LogicalBackingSliceAuthority]]) -> Vec<SegmentBackingExpectation> {
    groups
        .iter()
        .map(|group| {
            let first = &group[0].evidence;
            SegmentBackingExpectation {
                logical_bytes: group.iter().map(|a| a.evidence.logical_size_bytes).sum(),
                logical_size_rule: SegmentLogicalSizeRule::Exact,
                usage: first.usage,
                storage_profile: first.storage_profile,
                element_type: first.element_type,
                alignment_bytes: first.alignment_bytes,
            }
        })
        .collect()
}

fn windows(expected: &[SegmentBackingExpectation]) -> Vec<SegmentBackingWindow> {
    expected
        .iter()
        .enumerate()
        .map(|(resource_index, e)| SegmentBackingWindow {
            resource_index,
            offset_bytes: 0,
            length_bytes: e.logical_bytes,
            element_type: e.element_type,
            alignment_bytes: e.alignment_bytes,
        })
        .collect()
}

fn physical_facts(
    batch: &SegmentBackingBatch<<TestRuntime as DeviceRuntime>::Buffer>,
    index: usize,
) -> Vec<(usize, std::ops::Range<u64>, u64)> {
    batch
        .window(index)
        .unwrap()
        .physical_regions()
        .map(|region| {
            let (buffer, range, _retention) = region.buffer_and_physical_range();
            (
                buffer as *const _ as usize,
                range,
                region.logical_offset_bytes(),
            )
        })
        .collect()
}

#[test]
fn segment_permit_repeated_windows_keep_order_and_distinct_current_views() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let groups = [
        std::slice::from_ref(&authorities[0]),
        std::slice::from_ref(&authorities[1]),
        std::slice::from_ref(&authorities[0]),
    ];
    let mut expected = expectations(&groups);
    expected[2].logical_bytes = 32;
    expected[2].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    let first = SegmentBackingWindow {
        length_bytes: 16,
        ..windows(&expected)[0]
    };
    let requested = [
        first,
        SegmentBackingWindow {
            resource_index: 1,
            ..first
        },
        first,
        SegmentBackingWindow {
            offset_bytes: 16,
            ..first
        },
        SegmentBackingWindow {
            resource_index: 2,
            ..first
        },
        first,
        SegmentBackingWindow {
            alignment_bytes: 1,
            ..first
        },
    ];
    let batch = pools
        .segment_backing_batch(&groups, &expected, &requested)
        .unwrap();
    for (index, window) in requested.iter().enumerate() {
        let separate = pools
            .segment_backing_batch(&groups, &expected, std::slice::from_ref(window))
            .unwrap();
        assert_eq!(physical_facts(&batch, index), physical_facts(&separate, 0));
    }
    assert_eq!(physical_facts(&batch, 0), physical_facts(&batch, 2));
    assert_eq!(physical_facts(&batch, 0), physical_facts(&batch, 5));
    assert_ne!(physical_facts(&batch, 0), physical_facts(&batch, 1));
    assert_ne!(physical_facts(&batch, 0), physical_facts(&batch, 3));
    assert!(batch.window(requested.len()).is_none());
    drop(batch);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn segment_permit_window_alias_cannot_hide_type_alignment_or_current_prefix_errors() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let groups = [
        std::slice::from_ref(&authorities[0]),
        std::slice::from_ref(&authorities[0]),
    ];
    let mut expected = expectations(&groups);
    expected[1].logical_bytes = 16;
    expected[1].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    let first = windows(&expected)[0];
    let wrong_type = if first.element_type == ElementType::U8 {
        ElementType::F16
    } else {
        ElementType::U8
    };
    let invalid = [
        SegmentBackingWindow {
            element_type: wrong_type,
            ..first
        },
        SegmentBackingWindow {
            alignment_bytes: first.alignment_bytes * 2,
            ..first
        },
        SegmentBackingWindow {
            resource_index: 1,
            ..first
        },
        SegmentBackingWindow {
            resource_index: groups.len(),
            ..first
        },
        SegmentBackingWindow {
            length_bytes: 0,
            ..first
        },
        SegmentBackingWindow {
            offset_bytes: u64::MAX,
            ..first
        },
        SegmentBackingWindow {
            length_bytes: first.length_bytes + 1,
            ..first
        },
    ];
    for bad in invalid {
        let separate = pools
            .segment_backing_batch(&groups, &expected, &[bad])
            .err()
            .unwrap()
            .to_string();
        let after_alias = pools
            .segment_backing_batch(&groups, &expected, &[first, first, bad, bad])
            .err()
            .unwrap()
            .to_string();
        assert_eq!(after_alias, separate);
        for pool in pools.pools.values() {
            assert!(pool.state.try_lock().is_ok());
        }
    }
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn segment_permit_repeated_windows_observe_new_physical_page_and_current_prefix() {
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
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let first = claim_size(pools, pool, 64);
    let old_groups = [std::slice::from_ref(&first)];
    let old_expected = expectations(&old_groups);
    let old_window = windows(&old_expected)[0];
    let old_batch = pools
        .segment_backing_batch(&old_groups, &old_expected, &[old_window, old_window])
        .unwrap();
    maintenance.grow_pool(&harness.pool_ids[0], 64).unwrap();
    let next = claim_size(pools, pool, 64);
    assert_ne!(
        first.evidence.segments[0].chunk_ordinal(),
        next.evidence.segments[0].chunk_ordinal()
    );
    let authorities = [first, next];
    let groups = [&authorities[..]];
    let mut expected = expectations(&groups);
    expected[0].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    expected[0].logical_bytes = 65;
    let current = windows(&expected)[0];
    let batch = pools
        .segment_backing_batch(&groups, &expected, &[current, current])
        .unwrap();
    let facts = physical_facts(&batch, 0);
    assert_eq!(facts, physical_facts(&batch, 1));
    assert_eq!(facts.len(), 2);
    assert_eq!(facts[0].1.end - facts[0].1.start, 64);
    assert_eq!(facts[1].1.end - facts[1].1.start, 1);
    assert_eq!(facts[1].2, 64);
    assert_eq!(physical_facts(&old_batch, 0).len(), 1);
    let beyond_current = SegmentBackingWindow {
        length_bytes: 66,
        ..current
    };
    assert!(pools
        .segment_backing_batch(&groups, &expected, &[current, current, beyond_current])
        .is_err());
    drop(batch);
    drop(old_batch);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn segment_permit_deduplicates_real_authorities_and_bounds_every_window() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let groups = [
        std::slice::from_ref(&authorities[0]),
        std::slice::from_ref(&authorities[1]),
        std::slice::from_ref(&authorities[0]),
    ];
    let mut expected = expectations(&groups);
    expected[2].logical_bytes = 32;
    expected[2].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    let mut requested = windows(&expected);
    requested[2].offset_bytes = 16;
    requested[2].length_bytes = 16;
    let queries = harness.runtime.descriptor_queries.load(Ordering::Acquire);
    let batch = pools
        .segment_backing_batch(&groups, &expected, &requested)
        .unwrap();
    assert_eq!(
        batch.bindings().len(),
        2,
        "one materialization per exact authority slice"
    );
    assert_eq!(batch.resources().len(), 3);
    for (index, authority) in authorities.iter().enumerate() {
        let original = pools.view(authority).unwrap();
        let region = batch
            .window(index)
            .unwrap()
            .physical_regions()
            .next()
            .unwrap();
        assert!(std::ptr::eq(
            region.buffer_and_physical_range().0,
            original.segment_bindings()[0].buffer()
        ));
        assert_eq!(region.length_bytes(), 64);
    }
    let region = batch.window(2).unwrap().physical_regions().next().unwrap();
    assert_eq!(region.length_bytes(), 16);
    assert_eq!(batch.resource(2).unwrap().view_size_bytes(), 32);
    assert_eq!(
        harness.runtime.descriptor_queries.load(Ordering::Acquire),
        queries,
        "the pool permit never invokes arbitrary runtime metadata getters"
    );
    drop(batch);
    requested[2].length_bytes = 17; // Actual captured logical64 does not authorize view33.
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    for pool in pools.pools.values() {
        assert!(pool.state.try_lock().is_ok());
    }
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn segment_permit_releases_partial_retention_before_error_cleanup() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let first = &pools.pools[&harness.pool_ids[0]];
    let second = &pools.pools[&harness.pool_ids[1]];
    let ordinal = authorities[0].evidence.segments[0].chunk_ordinal();
    let chunk = Arc::clone(&first.state.lock().unwrap().chunks[&ordinal].backing);
    let baseline = Arc::strong_count(&chunk);
    let groups = authorities
        .iter()
        .map(std::slice::from_ref)
        .collect::<Vec<_>>();
    let expected = expectations(&groups);
    let mut requested = windows(&expected);
    second.state.lock().unwrap().poisoned = true;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    assert!(first.state.try_lock().is_ok());
    assert!(second.state.try_lock().is_ok());
    assert_eq!(Arc::strong_count(&chunk), baseline);
    second.state.lock().unwrap().poisoned = false;
    requested[1].offset_bytes = u64::MAX;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    assert!(first.state.try_lock().is_ok());
    assert!(second.state.try_lock().is_ok());
    assert_eq!(Arc::strong_count(&chunk), baseline);
    drop(chunk);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn segment_permit_rejects_foreign_stale_and_missing_authorities() {
    let (harness, authorities) = two_pools();
    let (foreign, foreign_authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let groups = [
        std::slice::from_ref(&authorities[0]),
        std::slice::from_ref(&foreign_authorities[1]),
    ];
    let expected = expectations(&groups);
    assert_eq!(harness.pool_ids, foreign.pool_ids);
    assert!(pools
        .segment_backing_batch(&groups, &expected, &windows(&expected))
        .is_err());
    let pool = &pools.pools[&harness.pool_ids[0]];
    let ordinal = authorities[0].evidence.segments[0].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let groups = [std::slice::from_ref(&authorities[0])];
    let expected = expectations(&groups);
    assert!(pools
        .segment_backing_batch(&groups, &expected, &windows(&expected))
        .is_err());
    assert!(pool.state.try_lock().is_ok());
    let mut original = original;
    let old_identity = original.backing.identity.clone();
    Arc::get_mut(&mut original.backing).unwrap().identity = BackingChunkIdentity::from_parts(
        old_identity.pool_id().clone(),
        old_identity.ordinal(),
        old_identity.generation() + 1,
    )
    .unwrap();
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    assert!(
        pools
            .segment_backing_batch(&groups, &expected, &windows(&expected))
            .is_err(),
        "an admitted old claim cannot authorize a new current chunk generation"
    );
    let mut original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    Arc::get_mut(&mut original.backing).unwrap().identity = old_identity;
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    drop(
        pools
            .segment_backing_batch(&groups, &expected, &windows(&expected))
            .unwrap(),
    );
    drop(authorities);
    drop(foreign_authorities);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(foreign.root);
}

#[test]
fn segment_permit_returns_current_arc_and_retains_it_after_unlock() {
    let (harness, authorities) = two_pools();
    let (donor, donor_authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let donor_pool = &donor.root.dynamic_pools.pools[&donor.pool_ids[0]];
    let ordinal = authorities[0].evidence.segments[0].chunk_ordinal();
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
    let replacement_probe = Arc::downgrade(&replacement.backing);
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    let groups = [std::slice::from_ref(&authorities[0])];
    let expected = expectations(&groups);
    let repeated = [windows(&expected)[0]; 2];
    let batch = pools
        .segment_backing_batch(&groups, &expected, &repeated)
        .unwrap();
    assert!(Arc::ptr_eq(
        &batch.bindings()[0].chunk,
        &replacement_probe.upgrade().unwrap()
    ));
    assert!(!Arc::ptr_eq(
        &batch.bindings()[0].chunk,
        &old_view.bindings[0].chunk
    ));
    assert!(pools.revalidate_view(&authorities[0], &old_view).is_err());
    assert_eq!(physical_facts(&batch, 0), physical_facts(&batch, 1));
    // A later writer may run immediately; no until-submit stability is claimed.
    pool.state.try_lock().unwrap().poisoned = true;
    assert_eq!(
        batch
            .window(0)
            .unwrap()
            .physical_regions()
            .next()
            .unwrap()
            .length_bytes(),
        64
    );
    assert!(pools
        .segment_backing_batch(&groups, &expected, &repeated)
        .is_err());
    pool.state.lock().unwrap().poisoned = false;
    drop(batch);
    drop(replacement_probe);
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
    drop(old_view);
    drop(authorities);
    drop(donor_authorities);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(donor.root);
}
