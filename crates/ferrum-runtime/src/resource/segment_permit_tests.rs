//! Actual multi-pool admission, replacement and whole-segment failure boundaries.
use super::*;
use crate::{
    DeviceSubmissionStage, DeviceSubmissionTimingSink, DisabledSubmissionWaveDispatchTimingSink,
    SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink,
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

#[test]
fn segment_owner_identity_distinguishes_real_leases_and_retains_only_when_cloned() {
    // Two declared resources make the real initial chunk 128 bytes. Merely
    // setting a 256-byte maximum does not make that capacity resident.
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Request,
        'a',
        2,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .initialize_pool(&harness.pool_ids[0])
        .unwrap();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let authorities = vec![claim_size(pools, pool, 64)];
    let second = claim_size(pools, pool, 64);
    let first_view = pools.view(&authorities[0]).unwrap();
    let second_view = pools.view(&second).unwrap();
    let first = &first_view.segment_bindings()[0];
    let other = &second_view.segment_bindings()[0];
    assert!(std::ptr::eq(first.buffer(), other.buffer()));
    assert!(first.retention_ref().same_owners(&first.retention()));
    assert!(!first.retention_ref().same_owners(other.retention_ref()));
    // Reusing the same allocation and scalar range must never make distinct
    // lease owners equal. The range is not part of opaque owner identity.
    let same_range = first.segment().offset_bytes()
        ..first.segment().offset_bytes() + first.segment().length_bytes();
    let first_fact = (first.buffer(), same_range.clone(), first.retention());
    let foreign_fact = (other.buffer(), same_range, other.retention());
    assert!(std::ptr::eq(first_fact.0, foreign_fact.0));
    assert_eq!(first_fact.1, foreign_fact.1);
    assert!(!first_fact.2.same_owners(&foreign_fact.2));
    drop(first_fact);
    drop(foreign_fact);
    drop(first_view);
    drop(second_view);

    let groups = [std::slice::from_ref(&authorities[0])];
    let mut expected = expectations(&groups);
    expected[0].logical_bytes = 32;
    expected[0].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    let mut requested = windows(&expected);
    requested[0].offset_bytes = 16;
    requested[0].length_bytes = 16;
    let batch = pools
        .segment_backing_batch(&groups, &expected, &requested)
        .unwrap();
    let lease_count = Arc::strong_count(&authorities[0].segment_lease);
    let chunk_count = Arc::strong_count(&batch.bindings()[0].chunk);
    let window = batch.window(0).unwrap();
    let borrowed = window.physical_region(0).unwrap();
    let (buffer, range, retention) = borrowed.borrowed_buffer_and_physical_range();
    assert!(std::ptr::eq(buffer, batch.bindings()[0].buffer()));
    let base = authorities[0].evidence.segments[0].offset_bytes();
    assert_eq!(range, base + 16..base + 32);
    assert_eq!(borrowed.logical_offset_bytes(), 0);
    assert_eq!(window.region_count(), 1);
    assert!(window.physical_region(1).is_none());
    assert_eq!(
        Arc::strong_count(&authorities[0].segment_lease),
        lease_count
    );
    assert_eq!(Arc::strong_count(&batch.bindings()[0].chunk), chunk_count);
    let retained = retention.clone();
    let lease = Arc::downgrade(&authorities[0].segment_lease);
    drop(batch);
    requested[0].length_bytes = 17;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    drop(second);
    drop(authorities);
    assert!(
        lease.upgrade().is_some(),
        "command-owned capability retains the live lease"
    );
    drop(retained);
    assert!(
        lease.upgrade().is_none(),
        "borrowed metadata does not prolong ownership"
    );
    close_dynamic_test_root(harness.root);
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

pub(super) fn sequence_catalog_permit_fixture(
    suffix: &str,
) -> (
    Harness,
    Arc<AdmittedSequenceResources<TestRuntime>>,
    Arc<SequenceBackingSnapshot<TestRuntime>>,
) {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Fixed,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, suffix, 2);
    let snapshot = sequence.backing_snapshot().unwrap();
    (harness, sequence, snapshot)
}

#[test]
fn sequence_catalog_permit_reuses_physical_bindings_but_checks_fresh_windows_and_poison() {
    let (harness, sequence, snapshot) = sequence_catalog_permit_fixture("catalog-hot-window");
    let pools = &harness.root.dynamic_pools;
    let resource = snapshot.backing_slices()[0].resource_id();
    let groups = [snapshot.segment_group(0, resource).unwrap()];
    let mut expected = expectations(&[snapshot.backing_slices()]);
    let mut requested = windows(&expected);
    let sink = PermitTiming {
        before_record: || {
            for pool in pools.pools.values() {
                assert!(
                    pool.state.try_lock().is_ok(),
                    "catalog callback ran under pool lock"
                );
            }
        },
        stages: Mutex::new(Vec::new()),
    };
    let cold = pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .unwrap();
    let hot = pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .unwrap();
    assert!(
        std::ptr::eq(&cold.bindings()[0], &hot.bindings()[0]),
        "the second actual permit must share the checked physical group"
    );

    expected[0].logical_bytes = 32;
    expected[0].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    requested[0].offset_bytes = 16;
    requested[0].length_bytes = 16;
    let narrower = pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .unwrap();
    assert!(std::ptr::eq(&hot.bindings()[0], &narrower.bindings()[0]));
    assert_eq!(narrower.resource(0).unwrap().view_size_bytes(), 32);
    let region = narrower.window(0).unwrap().physical_region(0).unwrap();
    let base = snapshot.backing_slices()[0].evidence.segments[0].offset_bytes();
    assert_eq!(region.buffer_and_physical_range().1, base + 16..base + 32);
    requested[0].length_bytes = 17;
    assert!(
        pools
            .segment_backing_catalog_batch(
                &groups,
                &expected,
                SegmentBackingWindowInput::Slice(&requested),
                &sink,
            )
            .is_err(),
        "cached physical capacity must not authorize the old larger logical view"
    );
    requested[0].length_bytes = 16;
    let element_type = expected[0].element_type;
    expected[0].element_type = ElementType::F32;
    assert!(
        pools
            .segment_backing_catalog_batch(
                &groups,
                &expected,
                SegmentBackingWindowInput::Slice(&requested),
                &sink,
            )
            .is_err(),
        "current metadata is validated even for a cached group"
    );
    expected[0].element_type = element_type;
    let pool = &pools.pools[&harness.pool_ids[0]];
    pool.state.lock().unwrap().poisoned = true;
    assert!(pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .is_err());
    // Existing permits own their backing; poisoning rejects the next use only.
    assert_eq!(
        narrower
            .window(0)
            .unwrap()
            .physical_region(0)
            .unwrap()
            .length_bytes(),
        16
    );
    pool.state.lock().unwrap().poisoned = false;
    drop(narrower);
    drop(hot);
    drop(cold);
    drop(sink);
    drop(snapshot);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_catalog_permit_rechecks_foreign_missing_generation_and_replacement_arc() {
    let (harness, sequence, snapshot) = sequence_catalog_permit_fixture("catalog-replace");
    let (donor, donor_sequence, donor_snapshot) = sequence_catalog_permit_fixture("catalog-donor");
    let pools = &harness.root.dynamic_pools;
    let resource = snapshot.backing_slices()[0].resource_id();
    let groups = [snapshot.segment_group(0, resource).unwrap()];
    let expected = expectations(&[snapshot.backing_slices()]);
    let requested = windows(&expected);
    let sink = DisabledSubmissionWaveDispatchTimingSink;
    let cold = pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .unwrap();
    assert!(
        donor
            .root
            .dynamic_pools
            .segment_backing_catalog_batch(
                &groups,
                &expected,
                SegmentBackingWindowInput::Slice(&requested),
                &sink,
            )
            .is_err(),
        "equal resource/pool labels do not authorize another pool instance"
    );
    let pool = &pools.pools[&harness.pool_ids[0]];
    let donor_pool = &donor.root.dynamic_pools.pools[&donor.pool_ids[0]];
    let ordinal = snapshot.backing_slices()[0].evidence.segments[0].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    assert!(
        pools
            .segment_backing_catalog_batch(
                &groups,
                &expected,
                SegmentBackingWindowInput::Slice(&requested),
                &sink,
            )
            .is_err(),
        "a cached owner does not substitute for a missing current chunk"
    );
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
    current.identity = BackingChunkIdentity::from_parts(
        original.backing.identity.pool_id().clone(),
        ordinal,
        original.backing.identity.generation() + 1,
    )
    .unwrap();
    current.descriptor = original.backing.descriptor.clone();
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(
        pools
            .segment_backing_catalog_batch(
                &groups,
                &expected,
                SegmentBackingWindowInput::Slice(&requested),
                &sink,
            )
            .is_err(),
        "cached proof cannot authorize a changed current generation"
    );
    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    Arc::get_mut(&mut replacement.backing).unwrap().identity = original.backing.identity.clone();
    let replacement_probe = Arc::downgrade(&replacement.backing);
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    let current = pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .unwrap();
    assert!(Arc::ptr_eq(
        &current.bindings()[0].chunk,
        &replacement_probe.upgrade().unwrap()
    ));
    assert!(!Arc::ptr_eq(
        &current.bindings()[0].chunk,
        &cold.bindings()[0].chunk
    ));
    let retained = current.bindings()[0].retention();
    pool.state.lock().unwrap().poisoned = true;
    assert!(pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &sink,
        )
        .is_err());
    assert_eq!(
        current
            .window(0)
            .unwrap()
            .physical_region(0)
            .unwrap()
            .length_bytes(),
        64
    );
    pool.state.lock().unwrap().poisoned = false;
    drop(current);
    assert!(replacement_probe.upgrade().is_some());
    drop(retained);
    drop(replacement_probe);
    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    let backing = Arc::get_mut(&mut replacement.backing).unwrap();
    backing.identity = donor_identity;
    backing.descriptor = donor_descriptor;
    donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    drop(cold);
    drop(snapshot);
    drop(donor_snapshot);
    drop(sequence);
    drop(donor_sequence);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(donor.root);
}

#[test]
fn sequence_catalog_permit_does_not_authorize_foreign_or_cancelled_participant() {
    let (harness, sequence, snapshot) = sequence_catalog_permit_fixture("catalog-participant");
    let other = admitted_sequence(&harness.root, "catalog-other-participant");
    let session = sequence.open_session().unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    let (batch, step) = begin_request_state_test_step(vec![Arc::clone(&session)], &lane);
    let current = step
        .participant_backing_snapshot(BatchParticipantAuthority::new(
            sequence.sequence_authority(),
            sequence.request_authority(),
        ))
        .unwrap();
    assert!(Arc::ptr_eq(current, &snapshot));
    let groups = [current
        .segment_group(0, current.backing_slices()[0].resource_id())
        .unwrap()];
    let expected = expectations(&[current.backing_slices()]);
    let requested = windows(&expected);
    let cold = harness
        .root
        .dynamic_pools
        .segment_backing_catalog_batch(
            &groups,
            &expected,
            SegmentBackingWindowInput::Slice(&requested),
            &DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap();
    for participant in [
        BatchParticipantAuthority::new(other.sequence_authority(), other.request_authority()),
        BatchParticipantAuthority::new(sequence.sequence_authority(), other.request_authority()),
    ] {
        assert!(step.participant_backing_snapshot(participant).is_err());
    }
    drop(cold);
    step.try_retire_normal().unwrap();
    session.request_cancel().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(1)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(
        batch.try_begin_step(request, &lane).is_err(),
        "a warmed physical catalog cannot admit a cancelled participant"
    );
    session.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(lane);
    drop(snapshot);
    drop(sequence);
    drop(other);
    close_dynamic_test_root(harness.root);
}
