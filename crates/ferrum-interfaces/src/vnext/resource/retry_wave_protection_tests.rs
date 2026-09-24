//! Real Step -> full-plan wave -> rollback -> allocator-maintenance history.
//! No GPU, fabricated growth receipt, or unchecked backing reservation.
use super::*;
use crate::vnext::{CompletionObservation, CompletionReaper, OperationCompletionDisposition};

const UNIT: u64 = 64;

struct RetryWaveFixture {
    h: Harness,
    lane: Arc<ExecutionLane<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    step_pool: DynamicBackingPoolId,
    wave_pool: DynamicBackingPoolId,
    donor_pool: Option<DynamicBackingPoolId>,
    bucket: Option<ReusableExecutionBucketSpec>,
}

impl RetryWaveFixture {
    fn new(step_is_resident: bool, with_donor: bool) -> Self {
        Self::with_bucket(step_is_resident, with_donor, false)
    }

    fn with_bucket(step_is_resident: bool, with_donor: bool, reusable: bool) -> Self {
        let step = pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            '8',
            1,
            8 * UNIT,
            TestDemand::Tokens,
        );
        let mut invocation = pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            '9',
            1,
            8 * UNIT,
            TestDemand::Tokens,
        );
        // Same typed Invocation conversion used by the workspace projection
        // fixture. The immutable plan actually declares this node's scratch.
        let node = NodeId::new("node/retry-wave-scratch").unwrap();
        for descriptor in &mut invocation.descriptors {
            let mut value = serde_json::to_value(&*descriptor).unwrap();
            value["lifetime"] = json!("invocation");
            *descriptor = serde_json::from_value(value).unwrap();
        }
        let mut pool = serde_json::to_value(&invocation.pools[0]).unwrap();
        pool["minimum_step_bytes"] = json!(0);
        pool["minimum_invocation_peak_bytes"] = json!(UNIT);
        pool["step_resource_slots"] = json!([]);
        pool["invocation_liveness_mode"] = json!("total_order_reuse");
        pool["invocation_liveness"] = json!([{
            "node_id": node,
            "resource_ids": [invocation.descriptors[0].base_resource_id()]
        }]);
        invocation.pools[0] = serde_json::from_value(pool).unwrap();
        let step_pool = step.pool_id.clone();
        let wave_pool = invocation.pool_id.clone();
        let mut catalogs = vec![step, invocation];
        let donor_pool = with_donor.then(|| {
            let donor = pool_catalog(
                linear_profile(),
                AllocationLifetime::Request,
                'a',
                1,
                8 * UNIT,
                TestDemand::Fixed,
            );
            let id = donor.pool_id.clone();
            catalogs.push(donor);
            id
        });
        let catalog = combine_catalogs(&catalogs);
        // Without a donor, the simultaneous 4+4 claims cannot fit the six-unit
        // device. With one, 4+4+1 fits twelve units after stale donor reclamation.
        let budget = if with_donor { 12 * UNIT } else { 6 * UNIT };
        let bucket = reusable.then(|| reusable_step_memory_plan(step_pool.clone()).1);
        let memory = bucket.as_ref().map(|bucket| {
            let budgets = catalog
                .pools
                .iter()
                .filter(|p| p.pool_id() == &step_pool || p.pool_id() == &wave_pool)
                .map(|pool| {
                    let step = pool.pool_id() == &step_pool;
                    ReusablePoolWorkspaceBudget::new(
                        pool.pool_id().clone(),
                        if step { 4 * UNIT } else { 0 },
                        if step { 0 } else { 4 * UNIT },
                    )
                    .unwrap()
                })
                .collect();
            ReusableExecutionMemoryPlan::new(
                1,
                1,
                vec![ResolvedReusableExecutionBucket::new(bucket.clone(), budgets).unwrap()],
            )
            .unwrap()
        });
        let runtime = new_runtime(&catalog, budget);
        let h = harness_with_nodes_and_reusable(
            runtime,
            catalog,
            budget,
            false,
            Arc::from(vec![PlanNode::resource_test_node(node)]),
            memory,
        );
        for pool in [&step_pool, &wave_pool] {
            h.root.maintenance_controller.grow_pool(pool, UNIT).unwrap();
        }
        let resident = if step_is_resident {
            &step_pool
        } else {
            &wave_pool
        };
        h.root
            .maintenance_controller
            .grow_pool(resident, 4 * UNIT)
            .unwrap();
        if let Some(pool) = &donor_pool {
            for bytes in [UNIT, 5 * UNIT] {
                h.root
                    .maintenance_controller
                    .grow_pool(pool, bytes)
                    .unwrap();
            }
        }
        let lane = h.root.create_execution_lane().unwrap();
        let session = admitted_sequence_with_ceiling(&h.root, "whole-wave-retry", 4)
            .open_session()
            .unwrap();
        Self {
            h,
            lane,
            session,
            step_pool,
            wave_pool,
            donor_pool,
            bucket,
        }
    }

    fn step(&self, complete: bool) -> StepResourceAdmissionDecision<TestRuntime> {
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&self.session)]).unwrap();
        let mut request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(vec![token_span(4)]).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        if complete {
            request = request.with_full_plan_transient_retry_protection();
        }
        if let Some(bucket) = &self.bucket {
            request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
        }
        batch.try_begin_step(request, &self.lane).unwrap()
    }

    fn wave_deferred(&self) -> ReconciledSubmissionWaveMaintenance<TestRuntime> {
        let StepResourceAdmissionDecision::Admitted(step) = self.step(true) else {
            panic!("the initial exact Step is resident");
        };
        let StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) = step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        else {
            panic!("Invocation scratch must really defer");
        };
        // Before rollback the Step is still live; the original acquire evidence
        // protects Invocation only. Reconciliation seals the retry bundle.
        assert!(deferred
            .evidence()
            .protected_packing_envelopes()
            .iter()
            .all(|e| e.pool_id() == &self.wave_pool));
        deferred.reconcile_for_maintenance(step).unwrap()
    }

    fn assert_bundle(&self, evidence: &DynamicBackingDeferred) {
        let envelopes = evidence.protected_packing_envelopes();
        assert_eq!(envelopes.len(), 2);
        for pool in [&self.step_pool, &self.wave_pool] {
            let envelope = envelopes.iter().find(|e| e.pool_id() == pool).unwrap();
            assert_eq!(envelope.claim_bytes_descending(), [4 * UNIT]);
            assert_eq!(
                evidence
                    .protected_immediate()
                    .entries()
                    .iter()
                    .find(|entry| entry.domain() == envelope.domain_id())
                    .unwrap()
                    .units()
                    .get(),
                4 * UNIT
            );
        }
    }

    fn assert_wait(outcome: DynamicDeferredMaintenanceOutcome) {
        let DynamicDeferredMaintenanceOutcome::WaitForRelease {
            maintenance_boundary: Some(boundary),
            ..
        } = outcome
        else {
            panic!("insufficient whole-wave capacity must wait, not claim growth");
        };
        // A planned subset may be listed, but an insufficient proposal must
        // never be applied (the tests also verify actual pools/allocations).
        assert!(!boundary.reclaim_sufficient());
    }

    fn submit_fresh_wave(&self) {
        // Maintenance is not an admission/execute permit: both real admissions
        // must run again before the CPU runtime submits its tracked fence.
        let StepResourceAdmissionDecision::Admitted(step) = self.step(true) else {
            panic!("protected Step backing should still admit");
        };
        let StepSubmissionWaveAdmissionDecision::Prepared(wave) = step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        else {
            panic!("maintained Invocation backing should admit afresh");
        };
        let reaper = CompletionReaper::new();
        let handle = super::completed_boundary_tests::submit_fixture_wave_through_reaper(
            &self.h.root,
            std::slice::from_ref(&self.session),
            &self.lane,
            wave,
            &reaper,
        );
        let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
            panic!("fixture must reach its actual terminal fence");
        };
        assert!(matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        drop(receipt);
        drop(handle);
        step.try_retire_normal().unwrap();
        assert_eq!(reaper.retained_count(), 0);
        assert_eq!(self.lane.in_flight_count(), 0);
    }

    fn close(self) {
        self.session.try_abort_if_quiescent().unwrap();
        drop(self.session);
        drop(self.lane);
        close_dynamic_test_root(self.h.root);
    }
}

#[test]
fn complete_retry_wave_insufficient_bundle_waits_without_reciprocal_growth() {
    let f = RetryWaveFixture::new(true, false);
    let deferred = f.wave_deferred();
    f.assert_bundle(deferred.evidence());
    let allocations = f.h.runtime.allocate_calls();
    let pools = f.h.root.dynamic_pool_status().unwrap().pools().to_vec();
    for _ in 0..2 {
        RetryWaveFixture::assert_wait(deferred.maintain().unwrap());
        assert_eq!(f.h.runtime.allocate_calls(), allocations);
        assert_eq!(
            f.h.root
                .dynamic_pool_status()
                .unwrap()
                .budget_claimed_bytes(),
            6 * UNIT
        );
        assert_eq!(f.h.root.dynamic_pool_status().unwrap().pools(), pools);
    }
    drop(deferred);
    // No ghost reservation: the original Step admits, and the still-missing
    // Invocation remains a real deferral on a fresh exact retry.
    let retry = f.wave_deferred();
    f.assert_bundle(retry.evidence());
    drop(retry);
    f.close();
}

#[test]
fn complete_retry_step_insufficient_bundle_preserves_invocation_backing() {
    let f = RetryWaveFixture::new(false, false);
    let StepResourceAdmissionDecision::BackingDeferred(deferred) = f.step(true) else {
        panic!("Step must require real physical growth");
    };
    f.assert_bundle(deferred.evidence());
    let allocations = f.h.runtime.allocate_calls();
    let pools = f.h.root.dynamic_pool_status().unwrap().pools().to_vec();
    for _ in 0..2 {
        RetryWaveFixture::assert_wait(deferred.maintain().unwrap());
        assert_eq!(f.h.runtime.allocate_calls(), allocations);
        assert_eq!(f.h.root.dynamic_pool_status().unwrap().pools(), pools);
    }
    drop(deferred);
    // A caller that did not declare a full wave retains the original narrow
    // admission contract; no global/static protection has been installed.
    let StepResourceAdmissionDecision::BackingDeferred(ordinary) = f.step(false) else {
        panic!("Step remains physically unavailable");
    };
    assert!(ordinary
        .evidence()
        .protected_packing_envelopes()
        .iter()
        .all(|e| e.pool_id() == &f.step_pool));
    drop(ordinary);
    f.close();
}

#[test]
fn complete_retry_bundle_reclaims_unrelated_residency_and_submits_after_fresh_admission() {
    for step_is_resident in [true, false] {
        let f = RetryWaveFixture::new(step_is_resident, true);
        let outcome = if step_is_resident {
            let deferred = f.wave_deferred();
            f.assert_bundle(deferred.evidence());
            deferred.maintain().unwrap()
        } else {
            let StepResourceAdmissionDecision::BackingDeferred(deferred) = f.step(true) else {
                panic!("Step must defer before donor reclamation");
            };
            f.assert_bundle(deferred.evidence());
            deferred.maintain().unwrap()
        };
        let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = outcome else {
            panic!("full bundle fits after reclaiming the unrelated idle chunk");
        };
        assert_eq!(receipt.growths().len(), 1);
        let grown = if step_is_resident {
            &f.wave_pool
        } else {
            &f.step_pool
        };
        assert_eq!(receipt.growths()[0].pool_id(), grown);
        assert_eq!(receipt.growths()[0].chunk_bytes(), 4 * UNIT);
        let rebalance = receipt.rebalance().unwrap();
        assert_eq!(rebalance.pools().len(), 1);
        assert_eq!(
            rebalance.pools()[0].pool_id(),
            f.donor_pool.as_ref().unwrap()
        );
        assert_eq!(rebalance.pools()[0].reclaimed_bytes(), 5 * UNIT);
        assert_eq!(
            f.h.root
                .dynamic_pool_status()
                .unwrap()
                .budget_claimed_bytes(),
            11 * UNIT
        );
        f.submit_fresh_wave();
        f.close();
    }
}

#[test]
fn complete_retry_cancelled_owner_invalidates_both_maintenance_continuations() {
    for step_is_resident in [true, false] {
        let f = RetryWaveFixture::new(step_is_resident, false);
        let allocations = f.h.runtime.allocate_calls();
        if step_is_resident {
            let deferred = f.wave_deferred();
            f.session.request_cancel().unwrap();
            assert!(deferred.maintain().is_err());
            drop(deferred);
        } else {
            let StepResourceAdmissionDecision::BackingDeferred(deferred) = f.step(true) else {
                panic!("Step must defer");
            };
            f.session.request_cancel().unwrap();
            assert!(deferred.maintain().is_err());
            drop(deferred);
        }
        assert_eq!(f.h.runtime.allocate_calls(), allocations);
        f.close();
    }
}

#[test]
fn complete_retry_transient_opt_in_keeps_retained_bucket_evidence_unchanged() {
    let f = RetryWaveFixture::with_bucket(false, false, true);
    let StepResourceAdmissionDecision::BackingDeferred(old) = f.step(false) else {
        panic!("bucket Step must defer");
    };
    let StepResourceAdmissionDecision::BackingDeferred(opted) = f.step(true) else {
        panic!("transient opt-in cannot mint a resident bucket");
    };
    assert_eq!(old.evidence(), opted.evidence());
    drop(old);
    drop(opted);
    f.close();

    let f = RetryWaveFixture::with_bucket(true, false, true);
    let StepResourceAdmissionDecision::Admitted(step) = f.step(true) else {
        panic!("resident bucket Step must admit");
    };
    let StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) = step
        .try_prepare_full_plan_submission_wave(
            Arc::new(step.work_shape().clone()),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    else {
        panic!("bucket Invocation must defer");
    };
    let original = deferred.evidence().clone();
    let deferred = deferred.reconcile_for_maintenance(step).unwrap();
    // The released Step still has retained lane backing. Its old occupied
    // bytes must not be counted again as additional free transient demand.
    assert_eq!(&original, deferred.evidence());
    assert!(
        f.h.root
            .dynamic_pool_status()
            .unwrap()
            .pools()
            .iter()
            .find(|p| p.pool_id() == &f.step_pool)
            .unwrap()
            .live_occupancy()
            .lane_stable()
            .total()
            .claim_count()
            > 0
    );
    drop(deferred);
    f.close();
}
