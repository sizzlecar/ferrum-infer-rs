use super::*;
use crate::vnext::DynamicBackingPressure;

const CLAIM_UNIT: u64 = 64;

#[test]
fn actual_sequence_backing_pressure_survives_token_narrowing_until_hold_releases() {
    // Two real cohorts share a pool whose declared residency admits one cohort.
    // Only token spans change between the blocked cohort's two admission probes.
    let cohort_rows = 2_u32;
    let pool_bytes = CLAIM_UNIT * u64::from(cohort_rows);
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        'd',
        1,
        pool_bytes,
        TestDemand::ActualSequences(cohort_rows),
    );
    let runtime = new_runtime(&catalog, pool_bytes * 2);
    let h = harness(runtime, catalog, pool_bytes * 2, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], pool_bytes)
        .unwrap();
    let lane = h.root.create_execution_lane().unwrap();
    let input = [11_u32, 17, 29, 43];
    let sessions = (0..cohort_rows * 2)
        .map(|index| {
            admitted_sequence_with_ceiling(
                &h.root,
                &format!("sequence-scaled-{index}"),
                input.len(),
            )
            .open_session()
            .unwrap()
        })
        .collect::<Vec<_>>();
    for session in &sessions {
        assert!(matches!(
            session
                .try_ensure_backing_covers(
                    SequenceResourceExtensionRequest::new(
                        work(input.len()),
                        AdmissionPressureAction::WaitForRelease,
                    )
                    .unwrap(),
                )
                .unwrap(),
            SequenceResourceExtensionDecision::Current(_)
                | SequenceResourceExtensionDecision::Extended(_)
        ));
    }
    let holding_batch =
        ExecutionBatchParticipants::new(sessions[..cohort_rows as usize].to_vec()).unwrap();
    let blocked_batch =
        ExecutionBatchParticipants::new(sessions[cohort_rows as usize..].to_vec()).unwrap();
    let request = |batch: &ExecutionBatchParticipants<TestRuntime>, tokens: usize| {
        let spans = batch
            .sessions()
            .iter()
            .map(|_| TokenSpanWork::from_token_ids(&input, 0..tokens).unwrap())
            .collect();
        StepResourceAdmissionRequest::new(
            batch.bind_work_shape(spans).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    };
    let StepResourceAdmissionDecision::Admitted(holding_step) = holding_batch
        .try_begin_step(request(&holding_batch, input.len()), &lane)
        .unwrap()
    else {
        panic!("the first real cohort must fit the declared resident region");
    };
    let full_request = request(&blocked_batch, input.len());
    let StepResourceAdmissionDecision::BackingDeferred(full_deferred) = blocked_batch
        .try_begin_step(full_request.clone(), &lane)
        .unwrap()
    else {
        panic!("the second cohort must encounter the occupied physical region");
    };
    assert!(matches!(
        full_deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
    ));
    let execution_deferral = ExecutorExecutionCapacityDeferral::from_backing(
        full_deferred.evidence(),
        ExecutorExecutionCapacityStage::StepAdmission,
    )
    .unwrap();
    assert!(execution_deferral.shortfalls().is_empty());
    // Keep this resource invariant independent of the model's narrowing heuristic.
    let narrower_tokens = input.len() / 2;
    let StepResourceAdmissionDecision::BackingDeferred(narrow_deferred) = blocked_batch
        .try_begin_step(request(&blocked_batch, narrower_tokens), &lane)
        .unwrap()
    else {
        panic!("fewer tokens must not create another participant-scaled region");
    };
    assert_eq!(
        narrow_deferred.evidence().blockers(),
        full_deferred.evidence().blockers(),
        "the same live hold must yield the same pool, requested bytes and free packing"
    );
    assert_eq!(
        narrow_deferred.evidence().protected_packing_envelopes(),
        full_deferred.evidence().protected_packing_envelopes(),
        "halving tokens cannot shrink the participant-scaled physical transaction"
    );
    assert_eq!(
        full_deferred.evidence().blockers()[0].requested_bytes(),
        pool_bytes
    );
    assert_eq!(
        full_deferred.evidence().protected_packing_envelopes()[0].claim_bytes_descending(),
        &[pool_bytes]
    );
    assert!(matches!(
        narrow_deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
    ));

    holding_step.try_rollback_unsubmitted().unwrap();
    let StepResourceAdmissionDecision::Admitted(retried) =
        blocked_batch.try_begin_step(full_request, &lane).unwrap()
    else {
        panic!("releasing the actual hold must admit the unchanged full token range");
    };
    assert_eq!(retried.participant_count(), cohort_rows);
    assert_eq!(
        retried.backing_slices()[0].capacity_size_bytes(),
        pool_bytes
    );
    retried.try_rollback_unsubmitted().unwrap();
    drop(narrow_deferred);
    drop(full_deferred);
    drop(blocked_batch);
    drop(holding_batch);
    for session in &sessions {
        session.try_abort_if_quiescent().unwrap();
    }
    drop(sessions);
    drop(lane);
    close_dynamic_test_root(h.root);
}

fn cache_idle_pool_slot(
    h: &Harness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    pool_id: &DynamicBackingPoolId,
    bucket: &ReusableExecutionBucketSpec,
) -> Vec<LogicalBackingSliceAuthority> {
    let pool = &h.root.dynamic_pools.pools[pool_id];
    let mut request = evaluated_request(pool, CLAIM_UNIT);
    request.reusable_execution_bucket_id = Some(bucket.bucket_id().clone());
    let LaneBackingPrepareDecision::Prepared(prepared) = h
        .root
        .dynamic_pools
        .prepare_lane_stable_claim(lane, &[request])
        .unwrap()
    else {
        panic!("the initialized single-claim pool must prepare its idle cache slot");
    };
    let (slices, slot) = prepared.commit().into_parts();
    drop(slot);
    slices
}

fn retained_claims(h: &Harness, pool_id: &DynamicBackingPoolId) -> u64 {
    h.root
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .find(|pool| pool.pool_id() == pool_id)
        .unwrap()
        .live_occupancy()
        .lane_stable()
        .total()
        .claim_count()
}

struct TwoPoolIdleSlots {
    harness: Harness,
    lane: Arc<ExecutionLane<TestRuntime>>,
    unrelated_id: DynamicBackingPoolId,
    target_id: DynamicBackingPoolId,
}

impl TwoPoolIdleSlots {
    fn new() -> (Self, Vec<LogicalBackingSliceAuthority>) {
        // Pool-cache API fixture: each certified slot owns one selected pool.
        // A complete Step with this catalog would own both pools; it is not a
        // full-model Step reproduction. Product Step and Invocation lifetimes
        // can separately retain the disjoint pool ownership modeled here.
        let unrelated = pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            'a',
            1,
            CLAIM_UNIT,
            TestDemand::Fixed,
        );
        let target = pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            'b',
            1,
            CLAIM_UNIT,
            TestDemand::Fixed,
        );
        let unrelated_id = unrelated.pool_id.clone();
        let target_id = target.pool_id.clone();
        let catalog = combine_catalogs(&[unrelated, target]);
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("test.pool-resident-idle-target").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        let mut workspace = vec![
            ReusablePoolWorkspaceBudget::new(unrelated_id.clone(), CLAIM_UNIT, 0).unwrap(),
            ReusablePoolWorkspaceBudget::new(target_id.clone(), CLAIM_UNIT, 0).unwrap(),
        ];
        workspace.sort_by(|left, right| left.pool_id().cmp(right.pool_id()));
        let memory = ReusableExecutionMemoryPlan::new(
            1,
            1,
            vec![ResolvedReusableExecutionBucket::new(bucket.clone(), workspace).unwrap()],
        )
        .unwrap();
        // Two single-claim pool ceilings; device room for four claims.
        // A failed target grow is pool residency pressure, not device pressure.
        let budget = CLAIM_UNIT * 4;
        let runtime = new_runtime(&catalog, budget);
        let harness = harness_with_reusable(runtime, catalog, budget, memory);
        harness
            .root
            .maintenance_controller
            .initialize_pools(&harness.pool_ids)
            .unwrap();
        let lane = harness.root.create_execution_lane().unwrap();
        drop(cache_idle_pool_slot(
            &harness,
            &lane,
            &unrelated_id,
            &bucket,
        ));
        let target_pin = cache_idle_pool_slot(&harness, &lane, &target_id, &bucket);
        assert_eq!(retained_claims(&harness, &unrelated_id), 1);
        assert_eq!(retained_claims(&harness, &target_id), 1);
        (
            Self {
                harness,
                lane,
                unrelated_id,
                target_id,
            },
            target_pin,
        )
    }

    fn target_deferral(&self) -> PlanBackingDeferral<TestRuntime> {
        let pool = &self.harness.root.dynamic_pools.pools[&self.target_id];
        let request = evaluated_request(pool, CLAIM_UNIT);
        let BackingPrepareDecision::Deferred(evidence) = self
            .harness
            .root
            .dynamic_pools
            .prepare_claim(std::slice::from_ref(&request))
            .unwrap()
        else {
            panic!("eager backing cannot borrow a differently keyed cached slot");
        };
        assert!(evidence
            .blockers()
            .iter()
            .all(|blocker| blocker.pool_id() == &self.target_id));
        match self
            .harness
            .root
            .maintenance_controller
            .maintain_for_live_deferred(&evidence)
            .unwrap()
        {
            DynamicDeferredMaintenanceOutcome::WaitForRelease { pressure, .. } => {
                assert!(matches!(
                    pressure,
                    DynamicBackingPressure::PoolResident(ref resident)
                        if resident.pool_id() == &self.target_id
                ));
            }
            _ => panic!("the retained target slot must cause pool-local resident pressure"),
        }
        PlanBackingDeferral::new(Arc::clone(&self.harness.root), evidence).unwrap()
    }

    fn assert_target_admits(&self) {
        let pool = &self.harness.root.dynamic_pools.pools[&self.target_id];
        let request = evaluated_request(pool, CLAIM_UNIT);
        let BackingPrepareDecision::Prepared(prepared) = self
            .harness
            .root
            .dynamic_pools
            .prepare_claim(std::slice::from_ref(&request))
            .unwrap()
        else {
            panic!("the same target transaction must now admit without further growth");
        };
        drop(prepared.commit());
    }

    fn close(self) {
        drop(self.lane);
        close_dynamic_test_root(self.harness.root);
    }
}

#[test]
fn pool_resident_maintenance_preserves_older_idle_slot_in_unrelated_pool() {
    let (fixture, target_pin) = TwoPoolIdleSlots::new();
    drop(target_pin);
    let deferred = fixture.target_deferral();
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
    ));
    assert_eq!(
        retained_claims(&fixture.harness, &fixture.unrelated_id),
        1,
        "target-pool pressure must preserve the older unrelated cached slot"
    );
    assert_eq!(
        retained_claims(&fixture.harness, &fixture.target_id),
        0,
        "one targeted maintenance should release the reclaimable blocking slot"
    );
    fixture.assert_target_admits();
    drop(deferred);
    fixture.close();
}

#[test]
fn pool_resident_external_pin_waits_without_evicting_unrelated_idle_slot() {
    let (fixture, target_pin) = TwoPoolIdleSlots::new();
    let deferred = fixture.target_deferral();
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::WaitForRelease {
            pressure: DynamicBackingPressure::PoolResident(ref pressure), ..
        } if pressure.pool_id() == &fixture.target_id
    ));
    assert_eq!(
        retained_claims(&fixture.harness, &fixture.unrelated_id),
        1,
        "an unrelated eviction cannot resolve a pinned target-pool limit"
    );
    assert_eq!(retained_claims(&fixture.harness, &fixture.target_id), 1);

    drop(target_pin);
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
    ));
    assert_eq!(retained_claims(&fixture.harness, &fixture.unrelated_id), 1);
    assert_eq!(retained_claims(&fixture.harness, &fixture.target_id), 0);
    fixture.assert_target_admits();
    drop(deferred);
    fixture.close();
}
