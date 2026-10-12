//! Actual multi-pool admission, replacement and whole-segment failure boundaries.
use super::*;
use crate::vnext::operation::DisabledSubmissionWaveDispatchTimingSink;
use crate::vnext::resource::{SegmentBackingRow, SegmentBackingWindowInput};
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

fn fresh_batch<R: DeviceRuntime>(
    pools: &DynamicPoolSet<R>,
    groups: &[&[LogicalBackingSliceAuthority]],
    expected: &[SegmentBackingExpectation],
    windows: &[SegmentBackingWindow],
) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
    assert_eq!(groups.len(), expected.len());
    let rows = groups
        .iter()
        .zip(expected)
        .map(|(authorities, expected)| SegmentBackingRow {
            authorities,
            expected: *expected,
        })
        .collect::<Vec<_>>();
    pools.segment_backing_batch_with_rows(
        &rows,
        SegmentBackingWindowInput::Slice(windows),
        &DisabledSubmissionWaveDispatchTimingSink,
    )
}

fn assert_batch_equivalence<B>(
    old: &SegmentBackingBatch<B>,
    new: &SegmentBackingBatch<B>,
    window_count: usize,
) {
    assert_eq!(old.resources().len(), new.resources().len());
    for (old, new) in old.resources().iter().zip(new.resources()) {
        assert_eq!(old.logical_size_bytes(), new.logical_size_bytes());
        assert_eq!(old.capacity_size_bytes(), new.capacity_size_bytes());
        assert_eq!(old.view_size_bytes(), new.view_size_bytes());
        assert_eq!(old.alignment_bytes(), new.alignment_bytes());
        assert_eq!(old.usage(), new.usage());
        assert_eq!(old.element_type(), new.element_type());
        assert_eq!(old.storage_profile(), new.storage_profile());
    }
    assert_eq!(old.bindings().len(), new.bindings().len());
    for (old, new) in old.bindings().iter().zip(new.bindings()) {
        assert_eq!(old.segment(), new.segment());
        assert!(Arc::ptr_eq(&old.chunk, &new.chunk));
        assert_eq!(old.descriptor(), new.descriptor());
        assert!(old.retention_ref().same_owners(new.retention_ref()));
    }
    for index in 0..window_count {
        let old = old.window(index).unwrap();
        let new = new.window(index).unwrap();
        assert_eq!(old.region_count(), new.region_count());
        for (old, new) in old.physical_regions().zip(new.physical_regions()) {
            assert_eq!(old.logical_offset_bytes(), new.logical_offset_bytes());
            assert_eq!(old.length_bytes(), new.length_bytes());
            assert_eq!(old.descriptor(), new.descriptor());
            let (old_buffer, old_range, old_owner) = old.borrowed_buffer_and_physical_range();
            let (new_buffer, new_range, new_owner) = new.borrowed_buffer_and_physical_range();
            assert!(std::ptr::eq(old_buffer, new_buffer));
            assert_eq!(old_range, new_range);
            assert!(old_owner.same_owners(new_owner));
        }
    }
    assert!(new.window(window_count).is_none());
    let old_buffers = old
        .bindings()
        .iter()
        .map(|binding| binding.buffer() as *const B)
        .collect::<std::collections::BTreeSet<_>>();
    let representatives = new
        .metadata_bindings()
        .map(|binding| binding.buffer() as *const B)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(old_buffers, representatives);
    assert_eq!(new.metadata_bindings().len(), representatives.len());
}

#[test]
fn fresh_rows_match_legacy_for_duplicate_authorities_and_independent_view_bounds() {
    let (harness, authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let groups = [
        std::slice::from_ref(&authorities[1]),
        std::slice::from_ref(&authorities[0]),
        std::slice::from_ref(&authorities[1]),
        std::slice::from_ref(&authorities[1]),
    ];
    let mut expected = expectations(&groups);
    expected[2].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    expected[2].logical_bytes = 32;
    expected[3].logical_size_rule = SegmentLogicalSizeRule::AtLeast;
    expected[3].logical_bytes = 16;
    let mut requested = windows(&expected);
    requested[2].offset_bytes = 16;
    requested[2].length_bytes = 16;
    requested.push(SegmentBackingWindow {
        resource_index: 2,
        offset_bytes: 16,
        length_bytes: 16,
        element_type: expected[2].element_type,
        alignment_bytes: expected[2].alignment_bytes,
    });
    let old = pools
        .segment_backing_batch(&groups, &expected, &requested)
        .unwrap();
    let new = fresh_batch(pools, &groups, &expected, &requested).unwrap();
    assert_batch_equivalence(&old, &new, requested.len());
    drop(old);
    drop(new);

    // Sharing an authority cannot let this row borrow another row's full view.
    requested.last_mut().unwrap().length_bytes = 17;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    assert!(fresh_batch(pools, &groups, &expected, &requested).is_err());
    requested.last_mut().unwrap().length_bytes = 16;
    expected[2].logical_size_rule = SegmentLogicalSizeRule::Exact;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    assert!(fresh_batch(pools, &groups, &expected, &requested).is_err());
    for pool in pools.pools.values() {
        assert!(pool.state.try_lock().is_ok());
    }
    drop(authorities);
    close_dynamic_test_root(harness.root);
}

#[test]
fn fresh_rows_keep_each_real_lease_and_lane_while_representing_shared_buffer_metadata() {
    // The two resources declare an actually resident 128-byte initial chunk.
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
    let original = claim_size(pools, pool, 64);
    let mut independent = claim_size(pools, pool, 64);
    let lane = original.retained_for_lane(crate::vnext::ExecutionLaneId::mint().unwrap());
    let other_lane = original.retained_for_lane(crate::vnext::ExecutionLaneId::mint().unwrap());
    let groups = [&original, &independent, &lane, &other_lane, &original].map(std::slice::from_ref);
    let expected = expectations(&groups);
    let requested = windows(&expected);
    let old = pools
        .segment_backing_batch(&groups, &expected, &requested)
        .unwrap();
    let new = fresh_batch(pools, &groups, &expected, &requested).unwrap();
    assert_batch_equivalence(&old, &new, requested.len());
    let first_binding = &new.bindings()[0];
    let second_binding = &new.bindings()[1];
    assert!(std::ptr::eq(
        first_binding.buffer(),
        second_binding.buffer()
    ));
    assert!(!first_binding
        .retention_ref()
        .same_owners(second_binding.retention_ref()));
    assert!(!new.bindings()[2]
        .retention_ref()
        .same_owners(new.bindings()[3].retention_ref()));
    drop(old);
    drop(new);
    drop(other_lane);
    drop(lane);
    // Keep the later real claim's immutable projection and lease in agreement,
    // but require a chunk generation that the current shared buffer lacks.
    // Validating the first range cannot authorize this later range implicitly.
    let segment = independent.evidence.segments[0].clone();
    let stale = BackingSegment::from_chunk(
        segment.pool_id(),
        segment.chunk_ordinal(),
        segment.chunk_generation() + 1,
        segment.offset_bytes(),
        segment.length_bytes(),
    )
    .unwrap();
    Arc::get_mut(&mut independent.evidence.allocation)
        .unwrap()
        .segments[0] = stale.clone();
    Arc::get_mut(&mut independent.segment_lease)
        .unwrap()
        .segments[0] = stale;
    let groups = [&original, &independent].map(std::slice::from_ref);
    let expected = expectations(&groups);
    let requested = windows(&expected);
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    assert!(fresh_batch(pools, &groups, &expected, &requested).is_err());
    assert!(pool.state.try_lock().is_ok());
    Arc::get_mut(&mut independent.evidence.allocation)
        .unwrap()
        .segments[0] = segment.clone();
    Arc::get_mut(&mut independent.segment_lease)
        .unwrap()
        .segments[0] = segment;
    drop(independent);
    drop(original);
    close_dynamic_test_root(harness.root);
}

#[test]
fn fresh_rows_reject_late_authority_and_current_chunk_changes_without_retention_leaks() {
    let (harness, mut authorities) = two_pools();
    let pools = &harness.root.dynamic_pools;
    let first_pool = &pools.pools[&harness.pool_ids[0]];
    let second_pool = &pools.pools[&harness.pool_ids[1]];
    let first_ordinal = authorities[0].evidence.segments[0].chunk_ordinal();
    let first_chunk = Arc::clone(&first_pool.state.lock().unwrap().chunks[&first_ordinal].backing);
    let first_chunk_owners = Arc::strong_count(&first_chunk);
    let first_lease_owners = Arc::strong_count(&authorities[0].segment_lease);
    let reject = |authorities: &[LogicalBackingSliceAuthority]| {
        let groups = authorities
            .iter()
            .map(std::slice::from_ref)
            .collect::<Vec<_>>();
        let expected = expectations(&groups);
        let requested = windows(&expected);
        assert!(pools
            .segment_backing_batch(&groups, &expected, &requested)
            .is_err());
        assert!(fresh_batch(pools, &groups, &expected, &requested).is_err());
        for pool in pools.pools.values() {
            assert!(pool.state.try_lock().is_ok());
        }
        assert_eq!(Arc::strong_count(&first_chunk), first_chunk_owners);
        assert_eq!(
            Arc::strong_count(&authorities[0].segment_lease),
            first_lease_owners
        );
    };

    let generation = authorities[1].evidence.segment_generation;
    Arc::get_mut(&mut authorities[1].evidence.allocation)
        .unwrap()
        .segment_generation = generation + 1;
    reject(&authorities);
    {
        // A bad earlier requirement cannot hide the later authority failure.
        let groups = authorities
            .iter()
            .map(std::slice::from_ref)
            .collect::<Vec<_>>();
        let mut expected = expectations(&groups);
        expected[0].logical_bytes /= 2;
        let requested = windows(&expected);
        let legacy = pools
            .segment_backing_batch(&groups, &expected, &requested)
            .err()
            .expect("the stale authority must reject the whole permit");
        let fresh = fresh_batch(pools, &groups, &expected, &requested)
            .err()
            .expect("the fresh row permit must also reject");
        match (legacy, fresh) {
            (
                VNextError::InvalidExecutionPlan { reason: legacy },
                VNextError::InvalidExecutionPlan { reason: fresh },
            ) => assert_eq!(legacy, fresh),
            errors => panic!("authority error attribution differs: {errors:?}"),
        }
    }
    Arc::get_mut(&mut authorities[1].evidence.allocation)
        .unwrap()
        .segment_generation = generation;

    let resource_id = authorities[1].evidence.resource_id.clone();
    Arc::get_mut(&mut authorities[1].evidence.allocation)
        .unwrap()
        .resource_id = ResourceId::new("resource.foreign-claim-member").unwrap();
    reject(&authorities);
    Arc::get_mut(&mut authorities[1].evidence.allocation)
        .unwrap()
        .resource_id = resource_id;

    let segment = authorities[1].evidence.segments[0].clone();
    Arc::get_mut(&mut authorities[1].evidence.allocation)
        .unwrap()
        .segments[0] = BackingSegment::from_chunk(
        segment.pool_id(),
        segment.chunk_ordinal(),
        segment.chunk_generation(),
        segment.offset_bytes() + 8,
        segment.length_bytes() - 8,
    )
    .unwrap();
    reject(&authorities);
    Arc::get_mut(&mut authorities[1].evidence.allocation)
        .unwrap()
        .segments[0] = segment;

    let ordinal = authorities[1].evidence.segments[0].chunk_ordinal();
    let identity = {
        let mut state = second_pool.state.lock().unwrap();
        let backing = Arc::get_mut(&mut state.chunks.get_mut(&ordinal).unwrap().backing).unwrap();
        let identity = backing.identity.clone();
        backing.identity = BackingChunkIdentity::from_parts(
            identity.pool_id().clone(),
            identity.ordinal(),
            identity.generation() + 1,
        )
        .unwrap();
        identity
    };
    reject(&authorities);
    {
        let mut state = second_pool.state.lock().unwrap();
        Arc::get_mut(&mut state.chunks.get_mut(&ordinal).unwrap().backing)
            .unwrap()
            .identity = identity;
    }
    let descriptor_size = {
        let mut state = second_pool.state.lock().unwrap();
        let backing = Arc::get_mut(&mut state.chunks.get_mut(&ordinal).unwrap().backing).unwrap();
        let size = backing.descriptor.size_bytes;
        backing.descriptor.size_bytes = 32;
        size
    };
    reject(&authorities);
    {
        let mut state = second_pool.state.lock().unwrap();
        Arc::get_mut(&mut state.chunks.get_mut(&ordinal).unwrap().backing)
            .unwrap()
            .descriptor
            .size_bytes = descriptor_size;
    }
    second_pool.state.lock().unwrap().poisoned = true;
    reject(&authorities);
    second_pool.state.lock().unwrap().poisoned = false;

    let groups = authorities
        .iter()
        .map(std::slice::from_ref)
        .collect::<Vec<_>>();
    let expected = expectations(&groups);
    let mut requested = windows(&expected);
    requested[1].offset_bytes = u64::MAX;
    assert!(pools
        .segment_backing_batch(&groups, &expected, &requested)
        .is_err());
    assert!(fresh_batch(pools, &groups, &expected, &requested).is_err());
    for pool in pools.pools.values() {
        assert!(pool.state.try_lock().is_ok());
    }
    assert_eq!(Arc::strong_count(&first_chunk), first_chunk_owners);
    assert_eq!(
        Arc::strong_count(&authorities[0].segment_lease),
        first_lease_owners
    );

    requested = windows(&expected);
    let old = pools
        .segment_backing_batch(&groups, &expected, &requested)
        .unwrap();
    let new = fresh_batch(pools, &groups, &expected, &requested).unwrap();
    assert_batch_equivalence(&old, &new, requested.len());
    drop(old);
    drop(new);
    drop(first_chunk);
    drop(authorities);
    close_dynamic_test_root(harness.root);
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
    let legacy = pools
        .segment_backing_batch(&groups, &expected, &windows(&expected))
        .unwrap();
    let batch = fresh_batch(pools, &groups, &expected, &windows(&expected)).unwrap();
    assert_batch_equivalence(&legacy, &batch, expected.len());
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
    assert!(fresh_batch(pools, &groups, &expected, &windows(&expected)).is_err());
    pool.state.lock().unwrap().poisoned = false;
    drop(legacy);
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
