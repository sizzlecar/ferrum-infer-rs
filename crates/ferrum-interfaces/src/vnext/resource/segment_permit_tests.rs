//! Actual multi-pool admission, replacement and whole-segment failure boundaries.
use super::*;
use crate::vnext::{
    DeviceSubmissionStage, DeviceSubmissionTimingSink, SubmissionWaveDispatchStage,
    SubmissionWaveDispatchTimingSink,
};

struct PermitTiming<F> {
    before_record: F,
    stages: Mutex<Vec<SubmissionWaveDispatchStage>>,
}

impl<F: Fn() + Send + Sync> DeviceSubmissionTimingSink for PermitTiming<F> {
    const ENABLED: bool = true;

    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("a resource permit does not submit device commands");
    }
}

impl<F: Fn() + Send + Sync> SubmissionWaveDispatchTimingSink for PermitTiming<F> {
    fn record(&self, stage: SubmissionWaveDispatchStage, _: std::time::Duration) {
        (self.before_record)();
        self.stages.lock().unwrap().push(stage);
    }
}

pub(super) fn two_pools() -> (Harness, Vec<LogicalBackingSliceAuthority>) {
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

pub(super) fn expectations(
    groups: &[&[LogicalBackingSliceAuthority]],
) -> Vec<SegmentBackingExpectation> {
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

pub(super) fn windows(expected: &[SegmentBackingExpectation]) -> Vec<SegmentBackingWindow> {
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

#[test]
fn segment_permit_timing_flushes_after_two_pool_unlock_and_preserves_views() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    // Deliberately present the requests in reverse pool order. Timing must not
    // change canonical lock acquisition or caller resource/window indices.
    let groups = [
        std::slice::from_ref(&authorities[1]),
        std::slice::from_ref(&authorities[0]),
    ];
    let expected = expectations(&groups);
    let requested = windows(&expected);
    let sink = PermitTiming {
        before_record: || {
            for pool in pools.pools.values() {
                assert!(
                    pool.state.try_lock().is_ok(),
                    "observer ran under a pool lock"
                );
            }
        },
        stages: Mutex::new(Vec::new()),
    };
    let queries = harness.runtime.descriptor_queries.load(Ordering::Acquire);
    let batch = pools
        .segment_backing_batch_with_timing(&groups, &expected, &requested, &sink)
        .unwrap();
    for (index, group) in groups.iter().enumerate() {
        let original = pools.view(&group[0]).unwrap();
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
        assert_eq!(region.length_bytes(), expected[index].logical_bytes);
    }
    assert_eq!(
        *sink.stages.lock().unwrap(),
        [
            SubmissionWaveDispatchStage::SegmentBackingDedupReserveAndPoolResolution,
            SubmissionWaveDispatchStage::SegmentBackingLockAcquisition,
            SubmissionWaveDispatchStage::SegmentBackingLockedValidation,
            SubmissionWaveDispatchStage::SegmentBackingWindowIntersections,
        ]
    );
    assert_eq!(
        harness.runtime.descriptor_queries.load(Ordering::Acquire),
        queries
    );
    drop(batch);
    drop(sink);
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn segment_permit_timing_error_flush_releases_partial_retention_and_marks_entered_stage() {
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
    let sink = PermitTiming {
        before_record: || {
            for pool in pools.pools.values() {
                match pool.state.try_lock() {
                    Ok(_) | Err(std::sync::TryLockError::Poisoned(_)) => {}
                    Err(std::sync::TryLockError::WouldBlock) => {
                        panic!("observer ran under a pool lock")
                    }
                }
            }
            assert_eq!(
                Arc::strong_count(&chunk),
                baseline,
                "partial retention outlived error cleanup"
            );
        },
        stages: Mutex::new(Vec::new()),
    };
    second.state.lock().unwrap().poisoned = true;
    assert!(pools
        .segment_backing_batch_with_timing(&groups, &expected, &requested, &sink)
        .is_err());
    let stages = std::mem::take(&mut *sink.stages.lock().unwrap());
    assert_eq!(
        stages.last(),
        Some(&SubmissionWaveDispatchStage::SegmentBackingLockedValidation)
    );
    assert!(!stages.contains(&SubmissionWaveDispatchStage::SegmentBackingWindowIntersections));
    second.state.lock().unwrap().poisoned = false;

    requested[1].offset_bytes = u64::MAX;
    assert!(pools
        .segment_backing_batch_with_timing(&groups, &expected, &requested, &sink)
        .is_err());
    let stages = std::mem::take(&mut *sink.stages.lock().unwrap());
    assert_eq!(
        stages.last(),
        Some(&SubmissionWaveDispatchStage::SegmentBackingWindowIntersections)
    );

    // A poisoned second mutex aborts collect after acquiring the first mutex.
    // The partial guard vector must be destroyed before any observer callback.
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = second.state.lock().unwrap();
        panic!("poison the second real pool mutex");
    }))
    .is_err());
    assert!(pools
        .segment_backing_batch_with_timing(&groups, &expected, &requested, &sink)
        .is_err());
    let stages = std::mem::take(&mut *sink.stages.lock().unwrap());
    assert_eq!(
        stages,
        [
            SubmissionWaveDispatchStage::SegmentBackingDedupReserveAndPoolResolution,
            SubmissionWaveDispatchStage::SegmentBackingLockAcquisition,
        ]
    );
    second.state.clear_poison();
    drop(sink);
    drop(chunk);
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
    assert!(authorities[0].projection_proof_matches(pool));
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
    let batch = pools
        .segment_backing_batch(&groups, &expected, &windows(&expected))
        .unwrap();
    assert!(authorities[0].projection_proof_matches(pool));
    assert!(Arc::ptr_eq(
        &batch.bindings()[0].chunk,
        &replacement_probe.upgrade().unwrap()
    ));
    assert!(!Arc::ptr_eq(
        &batch.bindings()[0].chunk,
        &old_view.bindings[0].chunk
    ));
    assert!(pools.revalidate_view(&authorities[0], &old_view).is_err());
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
        .segment_backing_batch(&groups, &expected, &windows(&expected))
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
