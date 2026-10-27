//! Real shared Step/Invocation pool pressure, with no backend submission.
use super::*;
use crate::{
    DynamicStorageContract, DynamicStorageRequirement, MemoryPlan, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
};
use std::{sync::mpsc, time::Duration};

fn shared_pool_harness() -> Harness {
    shared_pool_harness_with_reuse(false).0
}

fn shared_pool_harness_with_reuse(cached: bool) -> (Harness, Option<ReusableExecutionBucketSpec>) {
    let profile = linear_profile();
    let step = pool_catalog_with_options(
        profile,
        AllocationLifetime::Step,
        '5',
        1,
        192,
        TestDemand::Fixed,
        "scratch",
        false,
        StateInitialization::None,
    );
    let node_id = NodeId::new("node/shared-capacity-wait").unwrap();
    let resource_id = ResourceId::new("scratch/shared-capacity-wait").unwrap();
    let node = PlanNode::resource_test_node_with_scratch(
        node_id.clone(),
        resource_id.clone(),
        ProviderWorkspaceRequirement::new(
            64,
            16,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )
        .unwrap(),
        vec![],
    )
    .resource_test_with_resources(
        step.descriptors
            .iter()
            .map(|resource| resource.base_resource_id().clone())
            .collect(),
    );
    let scratch = DynamicResourceDescriptor::resource_test_scratch(
        resource_id,
        DynamicResourceDemand::tokens(64, 1).unwrap(),
        16,
        node_id,
        DynamicStorageContract::resource_test_contract(profile, "5".repeat(64)).unwrap(),
        8,
    )
    .unwrap();
    let mut descriptors = step.descriptors;
    descriptors.push(scratch);
    descriptors.sort_by(|a, b| a.base_resource_id().cmp(b.base_resource_id()));
    let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &[node.clone()], 192).unwrap();
    assert_eq!(
        pools.len(),
        1,
        "the regression must use a genuinely shared lifetime domain"
    );
    let catalog = PoolCatalog {
        pool_id: pools[0].pool_id().clone(),
        pools,
        descriptors,
        profile,
    };
    let runtime = new_runtime(&catalog, 192);
    let bucket = cached.then(|| {
        ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("class/shared-capacity-wait").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap()
    });
    let reusable = bucket.as_ref().map(|bucket| {
        ReusableExecutionMemoryPlan::new(
            1,
            1,
            vec![ResolvedReusableExecutionBucket::new(
                bucket.clone(),
                vec![ReusablePoolWorkspaceBudget::new(catalog.pool_id.clone(), 64, 64).unwrap()],
            )
            .unwrap()],
        )
        .unwrap()
    });
    let h =
        harness_with_nodes_and_reusable(runtime, catalog, 192, false, vec![node].into(), reusable);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 192)
        .unwrap();
    (h, bucket)
}

fn begin(
    session: Arc<SequenceSession<TestRuntime>>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    bucket: Option<&ReusableExecutionBucketSpec>,
) -> (
    ExecutionBatchParticipants<TestRuntime>,
    Arc<StepResourceLease<TestRuntime>>,
) {
    let batch = ExecutionBatchParticipants::new(vec![session]).unwrap();
    let mut request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(1)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    if let Some(bucket) = bucket {
        request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
    }
    let StepResourceAdmissionDecision::Admitted(step) =
        batch.try_begin_step(request, lane).unwrap()
    else {
        panic!("resident Step backing fits");
    };
    assert_eq!(
        step.claimed_backing().lane_stable_slot_identity().is_some(),
        bucket.is_some(),
        "cached cases must exercise an actual lane slot"
    );
    (batch, step)
}

fn prepare(
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> StepSubmissionWaveAdmissionDecision<TestRuntime> {
    let work = step
        .shared_all_invocation_work_shape(&[token_span(1)])
        .unwrap();
    step.try_prepare_full_plan_submission_wave(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
}

#[test]
fn shared_domain_rollback_does_not_self_wake_and_external_releases_are_not_lost() {
    // Release races on both sides of rollback/registration, plus the parked case.
    for (cached, release_phase) in [false, true]
        .into_iter()
        .flat_map(|cached| (0..3).map(move |phase| (cached, phase)))
    {
        let (h, bucket) = shared_pool_harness_with_reuse(cached);
        let lane = h.root.create_execution_lane().unwrap();
        let first = admitted_sequence(&h.root, "capacity-holder");
        let first_session = first.open_session().unwrap();
        let (first_batch, first_step) = begin(first_session.clone(), &lane, bucket.as_ref());
        let StepSubmissionWaveAdmissionDecision::Prepared(first_wave) = prepare(&first_step) else {
            panic!("first wave fits");
        };
        let second = admitted_sequence(&h.root, "capacity-waiter");
        let second_session = second.open_session().unwrap();
        let root = h.root.clone();
        let retry_lane = lane.clone();
        let retry_session = second_session.clone();
        let attempts = Arc::new(AtomicU64::new(0));
        let observed = attempts.clone();
        let (phase_tx, phase_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            let mut frame = None;
            loop {
                let attempt = observed.fetch_add(1, Ordering::SeqCst);
                let (batch, step) = begin(retry_session.clone(), &retry_lane, bucket.as_ref());
                let current_frame = step.participant_frames().next().unwrap().frame_id();
                if let Some(first) = frame {
                    assert_eq!(first, current_frame);
                } else {
                    frame = Some(current_frame);
                }
                match prepare(&step) {
                    StepSubmissionWaveAdmissionDecision::Prepared(wave) => {
                        drop(wave);
                        step.try_retire_normal().unwrap();
                        drop(batch);
                        done_tx.send(()).unwrap();
                        break;
                    }
                    StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                        let DynamicDeferredMaintenanceOutcome::WaitForRelease {
                            wait_condition,
                            ..
                        } = deferred.maintain().unwrap()
                        else {
                            panic!("occupied bounded pool must wait");
                        };
                        drop(deferred);
                        if attempt == 0 && release_phase == 0 {
                            phase_tx.send(()).unwrap();
                            continue_rx.recv().unwrap();
                        }
                        let (_, condition) = step
                            .try_rollback_unsubmitted_for_capacity_wait(&wait_condition)
                            .unwrap();
                        drop(batch);
                        if attempt == 0 && release_phase == 1 {
                            phase_tx.send(()).unwrap();
                            continue_rx.recv().unwrap();
                        }
                        let waiter = root.register_capacity_waiter(&condition).unwrap();
                        if attempt == 0 && release_phase == 2 {
                            assert!(
                                !waiter.recheck().unwrap().should_retry(),
                                "own rollback is not external capacity"
                            );
                            phase_tx.send(()).unwrap();
                        }
                        runtime.block_on(waiter.wait_for_change()).unwrap();
                    }
                    _ => panic!("shared-pool pressure must remain retryable"),
                }
            }
        });
        phase_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        if release_phase == 2 {
            assert!(done_rx.recv_timeout(Duration::from_millis(30)).is_err());
            assert_eq!(
                attempts.load(Ordering::SeqCst),
                1,
                "no external event means no admission replay"
            );
        }
        drop(first_wave);
        if release_phase < 2 {
            continue_tx.send(()).unwrap();
        }
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        thread.join().unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        first_step.try_retire_normal().unwrap();
        drop(first_batch);
        first_session.try_complete().unwrap();
        second_session.try_complete().unwrap();
        drop((first_session, second_session, first, second, lane));
        close_dynamic_test_root(h.root);
    }
}

#[test]
fn logical_denial_discounts_the_provisional_physical_claim_before_waiting() {
    for (cached, wave_denial) in [false, true]
        .into_iter()
        .flat_map(|cached| [false, true].map(move |wave_denial| (cached, wave_denial)))
    {
        let (h, bucket) = shared_pool_harness_with_reuse(cached);
        let lane = h.root.create_execution_lane().unwrap();
        let sequence = admitted_sequence(&h.root, "provisional-physical");
        let session = sequence.open_session().unwrap();
        let binding = h.root.trusted_runtime_binding().unwrap();
        let (demand, _) = binding
            .scoped_demand(
                AllocationLifetime::Step,
                None,
                shape(1),
                shape(1),
                None,
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap();
        let step = if wave_denial {
            Some(begin(session.clone(), &lane, bucket.as_ref()).1)
        } else {
            None
        };
        // Logical competitors deliberately own no physical extents, exercising
        // the physical-prepare-success/logical-denial ordering directly.
        let mut competitors = Vec::new();
        for _ in 0..if wave_denial { 2 } else { 3 } {
            let BatchCapacityClaimDecision::Claimed(claim) = binding
                .logical_admission()
                .try_claim_for_sequences(&[sequence.logical_lease()], &demand)
                .unwrap()
            else {
                panic!("competitor logical claim fits");
            };
            competitors.push(claim);
        }
        let condition = if let Some(step) = step {
            let StepSubmissionWaveAdmissionDecision::Deferred(deferred) = prepare(&step) else {
                panic!("wave must fail logical admission after preparing physical backing");
            };
            step.try_rollback_unsubmitted_for_capacity_wait(deferred.wait_condition())
                .unwrap()
                .1
        } else {
            let batch = ExecutionBatchParticipants::new(vec![session.clone()]).unwrap();
            let mut request = StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![token_span(1)]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap();
            if let Some(bucket) = &bucket {
                request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
            }
            let StepResourceAdmissionDecision::Deferred(deferred) =
                batch.try_begin_step(request, &lane).unwrap()
            else {
                panic!("step must fail logical admission after preparing physical backing");
            };
            deferred.wait_condition().clone()
        };
        let waiter = h.root.register_capacity_waiter(&condition).unwrap();
        assert!(
            !waiter.recheck().unwrap().should_retry(),
            "provisional release must not count as external progress"
        );
        drop(competitors.pop());
        assert!(waiter.recheck().unwrap().should_retry());
        drop((waiter, competitors, binding));
        session.try_abort_if_quiescent().unwrap();
        drop((session, sequence, lane));
        close_dynamic_test_root(h.root);
    }
}

#[test]
fn rollback_failure_retains_original_step_and_never_returns_a_wait_receipt() {
    let h = shared_pool_harness();
    let lane = h.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence(&h.root, "rollback-owner");
    let session = sequence.open_session().unwrap();
    let (batch, step) = begin_request_state_test_step(vec![session.clone()], &lane);
    let domain = h.root.dynamic_pools.pools[&h.pool_ids[0]].domain.domain_id;
    let condition = h
        .root
        .dynamic_pools
        .logical_admission
        .wait_snapshot_for_domains([domain])
        .unwrap()
        .wait_condition()
        .clone();
    let clone = step.clone();
    let failure = step
        .try_rollback_unsubmitted_for_capacity_wait(&condition)
        .unwrap_err();
    let step = failure.into_step();
    assert!(Arc::ptr_eq(&step, &clone));
    drop(clone);
    let (_, condition) = step
        .try_rollback_unsubmitted_for_capacity_wait(&condition)
        .unwrap();
    assert!(!h
        .root
        .register_capacity_waiter(&condition)
        .unwrap()
        .recheck()
        .unwrap()
        .should_retry());
    drop(batch);
    session.try_abort_if_quiescent().unwrap();
    drop((session, sequence, lane));
    close_dynamic_test_root(h.root);
}
