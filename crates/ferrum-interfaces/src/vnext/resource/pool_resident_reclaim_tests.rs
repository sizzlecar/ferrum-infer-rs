use super::*;
use crate::vnext::DynamicPoolResidentPressure;

const UNIT: u64 = 64;
const LARGE: u64 = 8 * UNIT;

fn mixed_residency() -> (
    Harness,
    (
        Vec<LogicalBackingSliceAuthority>,
        Arc<ExecutionLane<TestRuntime>>,
    ),
) {
    mixed_residency_with_resources(1)
}

fn mixed_residency_with_resources(
    resource_count: usize,
) -> (
    Harness,
    (
        Vec<LogicalBackingSliceAuthority>,
        Arc<ExecutionLane<TestRuntime>>,
    ),
) {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '7',
        resource_count,
        24 * UNIT,
        TestDemand::Tokens,
    );
    let buckets = [1, 2, 4].map(|units| {
        ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("test.token-mask-resident-buckets").unwrap(),
            ReusableExecutionCapacity::new(1, units, 1).unwrap(),
        )
        .unwrap()
    });
    let resolved = buckets
        .iter()
        .zip([1, 2, 4])
        .map(|(bucket, units)| {
            ResolvedReusableExecutionBucket::new(
                bucket.clone(),
                vec![
                    ReusablePoolWorkspaceBudget::new(catalog.pool_id.clone(), units * UNIT, 0)
                        .unwrap(),
                ],
            )
            .unwrap()
        })
        .collect();
    let memory = ReusableExecutionMemoryPlan::new(1, 1, resolved).unwrap();
    let runtime = new_runtime(&catalog, 40 * UNIT);
    let h = harness_with_reusable(runtime, catalog, 40 * UNIT, memory);
    let lane = h.root.create_execution_lane().unwrap();
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let mut held = Vec::new();
    // Reduced token-mask history: resident 20 units, pinned reusable buckets
    // 1 + 2 + 4, free 13, largest 5, ceiling 24; a new 8-unit claim fragments.
    for (index, units) in [1, 1, 2, 3, 4, 4, 5].into_iter().enumerate() {
        h.root
            .maintenance_controller
            .grow_pool(&h.pool_ids[0], units * UNIT)
            .unwrap();
        if matches!(index, 0 | 2 | 4) {
            let mut request = evaluated_request(pool, units * UNIT);
            request.reusable_execution_bucket_id = Some(buckets[held.len()].bucket_id().clone());
            let LaneBackingPrepareDecision::Prepared(prepared) = h
                .root
                .dynamic_pools
                .prepare_lane_stable_claim(&lane, &[request])
                .unwrap()
            else {
                panic!("the newly resident reusable bucket must fit")
            };
            let (slices, slot) = prepared.commit().into_parts();
            held.extend(slices);
            drop(slot);
        }
    }
    h.runtime.set_reusable_resident_executables(3);
    (h, (held, lane))
}

fn deferred(h: &Harness) -> DynamicBackingDeferred {
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    match h
        .root
        .dynamic_pools
        .prepare_claim(&[evaluated_request(pool, LARGE)])
        .unwrap()
    {
        BackingPrepareDecision::Deferred(deferred) => deferred,
        _ => panic!("the large contiguous claim must initially fragment"),
    }
}

fn resident_pressure(
    h: &Harness,
    deferred: &DynamicBackingDeferred,
) -> DynamicPoolResidentPressure {
    match h
        .root
        .dynamic_pools
        .maintain_pools(vec![DynamicPoolGrowthIntent::RevalidatedDeferral(
            deferred.blockers()[0].clone(),
        )]) {
        Err(VNextError::DynamicPoolResidentUnavailable(pressure)) => pressure,
        outcome => panic!("expected pool-local pressure, got {outcome:?}"),
    }
}

fn reclaim(
    h: &Harness,
    pressure: &DynamicPoolResidentPressure,
    deferred: &DynamicBackingDeferred,
) -> Result<Option<DynamicPoolRebalanceReceipt>, VNextError> {
    h.root
        .dynamic_pools
        .reclaim_idle_chunks_for_pool_resident_pressure(
            pressure,
            deferred.protected_immediate(),
            deferred.protected_packing_envelopes(),
        )
        .map(|receipt| receipt.map(|(rebalance, _)| rebalance))
}

fn multi_claim_requests<'a>(
    pool: &'a Arc<DynamicBackingPool<TestRuntime>>,
    claims: &[u64],
) -> Vec<EvaluatedBackingRequest<'a>> {
    claims
        .iter()
        .enumerate()
        .map(|(index, bytes)| evaluated_descriptor_request(pool, index, *bytes))
        .collect()
}

fn multi_claim_deferred(h: &Harness, claims: &[u64]) -> DynamicBackingDeferred {
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    match h
        .root
        .dynamic_pools
        .prepare_claim(&multi_claim_requests(pool, claims))
        .unwrap()
    {
        BackingPrepareDecision::Deferred(deferred) => deferred,
        _ => panic!("the complete contiguous transaction must initially fragment"),
    }
}

fn claim_transaction(h: &Harness, claims: &[u64]) -> Vec<LogicalBackingSliceAuthority> {
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let BackingPrepareDecision::Prepared(prepared) = h
        .root
        .dynamic_pools
        .prepare_claim(&multi_claim_requests(pool, claims))
        .unwrap()
    else {
        panic!("ordinary admission must grant every companion claim after maintenance")
    };
    let slices = prepared.commit();
    assert_eq!(slices.len(), claims.len());
    slices
}

fn idle_multiclaim_residency(chunks: &[u64], ceiling: u64, resources: usize) -> Harness {
    let mut catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '9',
        resources,
        ceiling.min(256 * 64 * resources as u64),
        TestDemand::Tokens,
    );
    // The base fixture declares only four 64-byte tokens per resource. Larger
    // byte-scale regressions also need a consistent protocol demand ceiling;
    // changing just resident capacity would correctly fail plan validation.
    let maximum_tokens = ceiling.div_ceil(64).max(4);
    for descriptor in &mut catalog.descriptors {
        let mut wire = serde_json::to_value(&*descriptor).unwrap();
        wire["demand"]["tokens"]["maximum_tokens"] = json!(maximum_tokens);
        *descriptor = serde_json::from_value(wire).unwrap();
    }
    let mut pool = serde_json::to_value(&catalog.pools[0]).unwrap();
    pool["theoretical_ceiling_bytes"] =
        json!((64 * u128::from(maximum_tokens) * 64 * resources as u128).to_string());
    pool["provisioning"]["maximum_resident_bytes"] = json!(ceiling);
    catalog.pools[0] = serde_json::from_value(pool).unwrap();
    // Isolate pool residency from the separate device-capacity reclaim policy.
    let budget = ceiling * 2;
    let runtime = new_runtime(&catalog, budget);
    let h = harness(runtime, catalog, budget, false);
    for bytes in chunks {
        h.root
            .maintenance_controller
            .grow_pool(&h.pool_ids[0], *bytes)
            .unwrap();
    }
    h
}

#[test]
fn pool_resident_reclaim_admits_companion_claims_beside_pinned_lane_buckets() {
    let (h, held) = mixed_residency_with_resources(4);
    let claims = [LARGE, 16, 16, 16];
    let deferred = multi_claim_deferred(&h, &claims);
    assert_eq!(
        deferred.protected_packing_envelopes()[0].claim_bytes_descending(),
        claims
    );
    let before = h.root.dynamic_pool_status().unwrap();
    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .unwrap()
    else {
        panic!("all companion claims must be preserved during larger-bucket maintenance")
    };
    assert_eq!(receipt.growths()[0].chunk_bytes(), LARGE);
    assert_eq!(receipt.rebalance().unwrap().reclaimed_bytes(), 4 * UNIT);
    let boundary = receipt.maintenance_boundary().unwrap();
    assert_eq!(
        boundary.protected_packing_envelopes(),
        deferred.protected_packing_envelopes()
    );
    assert_eq!(boundary.pools()[0].protected_immediate_bytes(), LARGE + 48);
    assert_eq!(
        boundary.pools()[0].resident_floor_bytes(),
        7 * UNIT + LARGE + 48
    );
    let after = h.root.dynamic_pool_status().unwrap();
    assert_eq!(
        after.pools()[0].live_occupancy(),
        before.pools()[0].live_occupancy()
    );
    assert_eq!(after.pools()[0].resident_bytes(), 24 * UNIT);
    assert_eq!(h.runtime.reusable_trim_calls(), 0);
    drop(claim_transaction(&h, &claims));
    drop(held);
}

#[test]
fn pool_resident_reclaim_admits_a_large_mask_with_its_three_companion_claims() {
    // A four-resource step pool can have enough aggregate free space while
    // every whole idle chunk is smaller than its next mask bucket.
    let h = idle_multiclaim_residency(
        &[18512, 18480, 2112, 32768, 49152, 65536, 65536, 81920, 98304],
        460112,
        4,
    );
    let mut owners = Vec::new();
    for claims in [
        [16384, 16, 16, 16],
        [32768, 32, 32, 16],
        [65536, 64, 64, 16],
    ] {
        owners.extend(claim_transaction(&h, &claims));
    }
    let before = h.root.dynamic_pool_status().unwrap();
    assert_eq!(before.pools()[0].resident_bytes(), 432320);
    assert_eq!(before.pools()[0].free_bytes(), 317360);
    assert_eq!(before.pools()[0].largest_contiguous_bytes(), 98304);
    let claims = [131072, 128, 128, 32];
    let deferred = multi_claim_deferred(&h, &claims);
    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .unwrap()
    else {
        panic!("whole redundant chunks must admit the complete larger transaction")
    };
    assert_eq!(receipt.growths()[0].chunk_bytes(), 131072);
    let after = h.root.dynamic_pool_status().unwrap();
    assert_eq!(
        after.pools()[0].live_occupancy(),
        before.pools()[0].live_occupancy()
    );
    assert!(after.pools()[0].resident_bytes() <= 460112);
    drop(claim_transaction(&h, &claims));
    drop(owners);
}

#[test]
fn pool_resident_reclaim_preserves_an_extent_needed_by_a_companion_claim() {
    let h = idle_multiclaim_residency(&[256, 192, 192, 192, 192, 192], 1408, 2);
    let claims = [512, 256];
    let deferred = multi_claim_deferred(&h, &claims);
    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .unwrap()
    else {
        panic!("the two redundant smaller chunks must make room for the large claim")
    };
    let boundary = receipt.maintenance_boundary().unwrap();
    let keeper = boundary.pools()[0]
        .chunks()
        .iter()
        .find(|chunk| chunk.bytes() == 256)
        .unwrap();
    assert!(keeper.protected_packing());
    assert!(!keeper.reclaim_candidate());
    assert!(!boundary.selected_chunks().contains(keeper.identity()));
    assert_eq!(receipt.rebalance().unwrap().reclaimed_bytes(), 384);
    assert_eq!(receipt.growths()[0].chunk_bytes(), 512);
    drop(claim_transaction(&h, &claims));
}

#[test]
fn pool_resident_reclaim_checks_packing_after_the_combined_chunk_selection() {
    let h = idle_multiclaim_residency(&[256, 256, 256, 192, 192, 192, 192], 1664, 3);
    let claims = [512, 256, 256];
    let deferred = multi_claim_deferred(&h, &claims);
    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .unwrap()
    else {
        panic!("one redundant 256-byte and one 192-byte chunk safely cover the deficit")
    };
    let boundary = receipt.maintenance_boundary().unwrap();
    let large_candidates = boundary.pools()[0]
        .chunks()
        .iter()
        .filter(|chunk| chunk.bytes() == 256)
        .collect::<Vec<_>>();
    // Each is individually redundant, but two remain necessary for both
    // companion claims. Checking candidates independently would lose one.
    assert!(large_candidates
        .iter()
        .all(|chunk| chunk.reclaim_candidate()));
    assert_eq!(
        large_candidates
            .iter()
            .filter(|chunk| boundary.selected_chunks().contains(chunk.identity()))
            .count(),
        1
    );
    assert_eq!(receipt.rebalance().unwrap().reclaimed_bytes(), 448);
    assert_eq!(receipt.growths()[0].chunk_bytes(), 512);
    assert_eq!(
        h.root.dynamic_pool_status().unwrap().pools()[0].resident_bytes(),
        1600
    );
    drop(claim_transaction(&h, &claims));
}

#[test]
fn pool_resident_reclaim_leaves_a_packing_unsafe_combination_unchanged() {
    let h = idle_multiclaim_residency(&[256, 256, 256, 192, 192, 192, 192], 1664, 3);
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let pins = pool
        .state
        .lock()
        .unwrap()
        .chunks
        .values()
        .filter(|chunk| chunk.backing._grant.bytes() == 192)
        .map(|chunk| Arc::clone(&chunk.backing))
        .collect::<Vec<_>>();
    let deferred = multi_claim_deferred(&h, &[512, 256, 256]);
    let pressure = resident_pressure(&h, &deferred);
    let before = h.root.dynamic_pool_status().unwrap();
    assert!(reclaim(&h, &pressure, &deferred).unwrap().is_none());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    drop(pins);
}

#[test]
fn pool_resident_reclaim_admits_larger_claim_beside_pinned_lane_buckets() {
    let (h, held) = mixed_residency();
    let deferred = deferred(&h);
    let before = h.root.dynamic_pool_status().unwrap();
    let initial = &before.pools()[0];
    assert_eq!(initial.resident_bytes(), 20 * UNIT);
    assert_eq!(initial.free_bytes(), 13 * UNIT);
    assert_eq!(initial.largest_contiguous_bytes(), 5 * UNIT);
    assert_eq!(
        initial
            .live_occupancy()
            .lane_stable()
            .total()
            .physical_bytes(),
        7 * UNIT
    );
    let sequence = admitted_sequence(&h.root, "pool-resident-event");
    let session = sequence.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();

    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .unwrap()
    else {
        panic!("unreferenced whole chunks must make room for the larger bucket")
    };
    let rebalance = receipt.rebalance().unwrap();
    assert_eq!(rebalance.reclaimed_bytes(), 4 * UNIT);
    assert_eq!(rebalance.reclaimed_chunks(), 1);
    assert_eq!(receipt.growths()[0].chunk_bytes(), LARGE);
    // This was real pool pressure; the successful event retains that exact
    // pre-mutation boundary rather than inventing device-capacity pressure.
    let boundary = receipt.maintenance_boundary().unwrap();
    assert!(boundary.reclaim_attempted());
    assert!(boundary.pressure().pool_resident().is_some());
    assert!(boundary.pressure().device_capacity().is_none());
    assert_eq!(boundary.pools()[0].resident_bytes(), 20 * UNIT);
    assert_eq!(boundary.pools()[0].resident_floor_bytes(), 15 * UNIT);
    let event = BoundExecutionResourceMaintenance::bind(
        ExecutionResourceMaintenanceStage::StepAdmission,
        [&active],
        receipt.clone(),
    )
    .unwrap();
    let mut missing_attempt = receipt.clone();
    missing_attempt
        .maintenance_boundary
        .as_mut()
        .unwrap()
        .reclaim_attempted = false;
    assert!(BoundExecutionResourceMaintenance::bind(
        ExecutionResourceMaintenanceStage::StepAdmission,
        [&active],
        missing_attempt,
    )
    .is_err());
    let serialized = serde_json::to_value(&event).unwrap();
    assert_eq!(
        serialized["receipt"]["maintenance_boundary"]["schema_version"],
        DYNAMIC_POOL_MAINTENANCE_BOUNDARY_SCHEMA_VERSION
    );
    assert_eq!(
        serialized["receipt"]["maintenance_boundary"]["pressure"]["kind"],
        "pool_resident"
    );
    assert_eq!(
        event.receipt().rebalance().unwrap().reclaimed_bytes(),
        4 * UNIT
    );
    let after = h.root.dynamic_pool_status().unwrap();
    assert_eq!(after.pools()[0].resident_bytes(), 24 * UNIT);
    assert_eq!(after.pools()[0].live_occupancy(), initial.live_occupancy());
    assert_eq!(after.budget_claimed_bytes(), 24 * UNIT);
    assert_eq!(h.runtime.reusable_trim_calls(), 0);
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let larger = claim_size(&h.root.dynamic_pools, pool, LARGE);
    let previous = claim_size(&h.root.dynamic_pools, pool, 5 * UNIT);
    drop((larger, previous, held));
    drop(active);
    session.try_abort_if_quiescent().unwrap();
}

#[test]
fn pool_resident_reclaim_handles_an_additional_idle_chunk_before_the_large_bucket() {
    let (h, held) = mixed_residency();
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], UNIT)
        .unwrap();
    let deferred = deferred(&h);
    let before = h.root.dynamic_pool_status().unwrap();
    assert_eq!(before.pools()[0].resident_bytes(), 21 * UNIT);
    assert_eq!(before.pools()[0].free_bytes(), 14 * UNIT);
    assert_eq!(before.pools()[0].largest_contiguous_bytes(), 5 * UNIT);
    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .unwrap()
    else {
        panic!("the 21-unit history must also admit the larger bucket")
    };
    assert_eq!(receipt.rebalance().unwrap().reclaimed_bytes(), 5 * UNIT);
    assert_eq!(
        receipt.maintenance_boundary().unwrap().pools()[0].reclaimable_bytes(),
        6 * UNIT
    );
    let after = h.root.dynamic_pool_status().unwrap();
    assert_eq!(after.pools()[0].resident_bytes(), 24 * UNIT);
    assert_eq!(
        after.pools()[0].live_occupancy(),
        before.pools()[0].live_occupancy()
    );
    assert_eq!(h.runtime.reusable_trim_calls(), 0);
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let larger = claim_size(&h.root.dynamic_pools, pool, LARGE);
    // Best-single legitimately replaces the idle 5-unit chunk here. Existing
    // pinned 1/2/4 buckets survive; the previous free 4-unit extent also fits.
    let previous = claim_size(&h.root.dynamic_pools, pool, 4 * UNIT);
    drop((larger, previous, held));
}

#[test]
fn pool_resident_reclaim_never_removes_partial_or_externally_referenced_chunks() {
    for partial in [true, false] {
        let (h, held) = mixed_residency();
        let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
        let partial_owner = partial.then(|| claim_size(&h.root.dynamic_pools, pool, 16));
        let protected_chunk =
            {
                let state = pool.state.lock().unwrap();
                state
                    .chunks
                    .values()
                    .find(|chunk| {
                        if partial {
                            chunk.live_segments != 0
                                && chunk.backing._grant.bytes() == UNIT
                                && state.allocator.by_offset.keys().any(|(ordinal, _)| {
                                    *ordinal == chunk.backing.identity.ordinal()
                                })
                        } else {
                            chunk.live_segments == 0 && chunk.backing._grant.bytes() == 4 * UNIT
                        }
                    })
                    .unwrap()
                    .backing
                    .identity
                    .clone()
            };
        let pin = (!partial).then(|| {
            Arc::clone(&pool.state.lock().unwrap().chunks[&protected_chunk.ordinal()].backing)
        });
        let deferred = deferred(&h);
        let pressure = resident_pressure(&h, &deferred);
        let receipt = reclaim(&h, &pressure, &deferred).unwrap().unwrap();
        assert!(!receipt.pools()[0].chunks().contains(&protected_chunk));
        assert!(pool
            .state
            .lock()
            .unwrap()
            .chunks
            .contains_key(&protected_chunk.ordinal()));
        drop((pin, partial_owner, held));
    }
}

#[test]
fn pool_resident_reclaim_leaves_an_insufficient_candidate_subset_unchanged() {
    let (h, held) = mixed_residency();
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let pins: Vec<_> = pool
        .state
        .lock()
        .unwrap()
        .chunks
        .values()
        .filter(|chunk| chunk.live_segments == 0 && chunk.backing._grant.bytes() > UNIT)
        .map(|chunk| Arc::clone(&chunk.backing))
        .collect();
    let deferred = deferred(&h);
    let pressure = resident_pressure(&h, &deferred);
    let before = h.root.dynamic_pool_status().unwrap();
    assert!(reclaim(&h, &pressure, &deferred).unwrap().is_none());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    drop((pins, held));
}

#[test]
fn pool_resident_reclaim_rechecks_competing_live_demand_and_stale_pressure() {
    let (h, held) = mixed_residency();
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    let deferred = deferred(&h);
    let pressure = resident_pressure(&h, &deferred);
    let first = claim_size(&h.root.dynamic_pools, pool, UNIT);
    let second = claim_size(&h.root.dynamic_pools, pool, 3 * UNIT);
    let before = h.root.dynamic_pool_status().unwrap();
    assert!(reclaim(&h, &pressure, &deferred).unwrap().is_none());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    drop((first, second));
    assert!(reclaim(&h, &pressure, &deferred).unwrap().is_some());
    let reclaimed = h.root.dynamic_pool_status().unwrap();
    // Reusing the stale pressure cannot discard another chunk.
    assert!(reclaim(&h, &pressure, &deferred).unwrap().is_none());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), reclaimed);
    drop(held);
}

#[test]
fn pool_resident_reclaim_allocation_failure_preserves_owners_and_releases_grant() {
    let (h, held) = mixed_residency();
    let deferred = deferred(&h);
    h.runtime.fail_on_call(h.runtime.allocate_calls() + 1);
    assert!(h
        .root
        .maintenance_controller
        .maintain_for_live_deferred(&deferred)
        .is_err());
    let status = h.root.dynamic_pool_status().unwrap();
    assert_eq!(status.pools()[0].resident_bytes(), 16 * UNIT);
    assert_eq!(status.pools()[0].pending_growth_bytes(), 0);
    assert_eq!(
        status.pools()[0]
            .live_occupancy()
            .lane_stable()
            .total()
            .physical_bytes(),
        7 * UNIT
    );
    assert_eq!(status.budget_claimed_bytes(), 16 * UNIT);
    assert_eq!(status.process_claimed_bytes(), 16 * UNIT);
    let pool = &h.root.dynamic_pools.pools[&h.pool_ids[0]];
    drop(claim_size(&h.root.dynamic_pools, pool, 5 * UNIT));
    h.runtime.fail_on_call(0);
    assert!(matches!(
        h.root
            .maintenance_controller
            .maintain_for_live_deferred(&deferred)
            .unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    drop(claim_size(&h.root.dynamic_pools, pool, LARGE));
    drop(held);
}

#[test]
fn pool_resident_reclaim_retry_retains_real_competing_device_reservation_boundary() {
    let (h, held) = mixed_residency();
    let deferred = deferred(&h);
    let pressure = resident_pressure(&h, &deferred);
    reclaim(&h, &pressure, &deferred).unwrap().unwrap();
    let pools = &h.root.dynamic_pools;
    let competing = DeviceCapacityReservation::reserve(&pools.budget, 20 * UNIT).unwrap();
    let mut blocked = None;
    let result = pools.maintain_pools_observed(
        vec![DynamicPoolGrowthIntent::RevalidatedDeferral(
            deferred.blockers()[0].clone(),
        )],
        &mut blocked,
    );
    assert!(matches!(
        result,
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    let snapshot = pools
        .logical_admission
        .wait_snapshot_for_domains(vec![pools.pools[&h.pool_ids[0]].domain.domain_id])
        .unwrap();
    let outcome = h
        .root
        .maintenance_controller
        .capacity_wait_outcome(
            snapshot,
            blocked.unwrap(),
            None,
            deferred.protected_immediate(),
            deferred.protected_packing_envelopes(),
        )
        .unwrap();
    let DynamicDeferredMaintenanceOutcome::WaitForRelease {
        current_epochs,
        wait_condition,
        pressure,
        maintenance_boundary,
    } = outcome
    else {
        panic!("the competing reservation must produce a typed wait")
    };
    assert!(pressure.device_capacity().is_some());
    let boundary = maintenance_boundary.as_ref().unwrap();
    assert!(!boundary.reclaim_attempted());
    assert_eq!(boundary.pressure(), &pressure);
    assert!(boundary.pools().is_empty());
    assert!(boundary.selected_chunks().is_empty());
    assert_eq!(
        boundary.protected_immediate(),
        deferred.protected_immediate()
    );
    assert_eq!(
        boundary.protected_packing_envelopes(),
        deferred.protected_packing_envelopes()
    );
    let executor = ExecutorExecutionCapacityDeferral::from_backing_maintenance(
        &deferred,
        ExecutorAdmissionEpochs::from_capacity(current_epochs),
        wait_condition,
        pressure.clone(),
        maintenance_boundary,
        ExecutorExecutionCapacityStage::StepAdmission,
    )
    .unwrap();
    assert_eq!(executor.backing_pressure(), Some(&pressure));
    assert!(!executor.maintenance_boundary().unwrap().reclaim_attempted());
    let status = h.root.dynamic_pool_status().unwrap();
    assert_eq!(status.pools()[0].pending_growth_bytes(), 0);
    assert_eq!(status.pools()[0].resident_bytes(), 16 * UNIT);
    drop(competing);
    assert!(matches!(
        h.root
            .maintenance_controller
            .maintain_for_live_deferred(&deferred)
            .unwrap(),
        DynamicDeferredMaintenanceOutcome::Maintained(_)
    ));
    drop(held);
}

#[test]
fn pool_resident_reclaim_rejects_foreign_or_inconsistent_pressure() {
    let (h, held) = mixed_residency();
    let deferred = deferred(&h);
    let valid = resident_pressure(&h, &deferred);
    let foreign = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '8',
        1,
        24 * UNIT,
        TestDemand::Tokens,
    )
    .pool_id;
    let before = h.root.dynamic_pool_status().unwrap();
    for pressure in [
        DynamicPoolResidentPressure::new(foreign, LARGE, 20 * UNIT, 24 * UNIT).unwrap(),
        DynamicPoolResidentPressure::new(valid.pool_id().clone(), LARGE, 20 * UNIT, 23 * UNIT)
            .unwrap(),
        DynamicPoolResidentPressure::new(valid.pool_id().clone(), 9 * UNIT, 20 * UNIT, 24 * UNIT)
            .unwrap(),
    ] {
        assert!(reclaim(&h, &pressure, &deferred).is_err());
        assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    }
    assert!(h
        .root
        .dynamic_pools
        .reclaim_idle_chunks_for_pool_resident_pressure(&valid, deferred.protected_immediate(), &[])
        .unwrap()
        .is_none());
    assert_eq!(h.root.dynamic_pool_status().unwrap(), before);
    drop(held);
}
