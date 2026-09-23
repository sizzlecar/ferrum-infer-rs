//! Pure forecasts versus real retained Step slots, with no GPU or fake leases.
use super::*;

#[path = "planning_expected_work_tests.rs"]
mod expected_work;
use crate::vnext::{
    ExecutionCostRouteUnknown, ProductTokenMaskResidencyEntry, ProductTokenMaskResidencySnapshot,
};

fn lane_view(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    session: &SequenceSession<TestRuntime>,
    lane: &ExecutionLane<TestRuntime>,
) -> ResourcePlanningView {
    lane_view_with_limits(root, session, lane, ResourcePlanningLimits::default())
}

fn lane_view_with_limits(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    session: &SequenceSession<TestRuntime>,
    lane: &ExecutionLane<TestRuntime>,
    limits: ResourcePlanningLimits,
) -> ResourcePlanningView {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match root.resource_planning_view_on_lane(&[session], lane, limits, &mut || true) {
            ResourcePlanningAvailability::Known(view) => return view,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(_))
                if std::time::Instant::now() < deadline =>
            {
                std::thread::yield_now()
            }
            other => panic!("workspace snapshot: {other:?}"),
        }
    }
}

#[path = "planning_workspace_coalesced_tests.rs"]
mod coalesced;
fn setup(
    resident: u64,
) -> (
    Harness,
    ReusableExecutionBucketSpec,
    Arc<ExecutionLane<TestRuntime>>,
) {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '6',
        1,
        256,
        TestDemand::Tokens,
    );
    let (memory, bucket) = reusable_step_memory_plan(catalog.pool_id.clone());
    let runtime = new_runtime(&catalog, 256);
    let harness = harness_with_reusable(runtime, catalog, 256, memory);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], resident)
        .unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    (harness, bucket, lane)
}
fn step(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    tokens: usize,
    bucket: Option<&ReusableExecutionBucketSpec>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let mut request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(tokens)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    if let Some(bucket) = bucket {
        request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
    }
    match batch.try_begin_step(request, lane).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident fixture Step must admit"),
    }
}

#[test]
fn planning_workspace_cold_retention_and_hot_reuse_match_actual_slots() {
    let (harness, bucket, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "workspace-cold", 4);
    let session = sequence.open_session().unwrap();
    let before = harness.runtime.allocate_calls();
    let cold = lane_view(&harness.root, &session, &lane);
    assert_eq!(cold.plan_hash(), harness.root.planning_plan_hash());
    assert_eq!(
        cold.coordinator_id(),
        harness.root.dynamic_pools.logical_admission.id()
    );
    assert_eq!(cold.lane_id(), Some(lane.id()));
    assert!(cold.participants()[0].matches_session_identity(&session));
    let first = known(harness.root.project_resource_wave_with_bucket(
        &cold,
        &cold.initial_state(),
        &[row(0, 0, 1)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    assert_eq!(
        first.domains[0].transient_peak_bytes, 64,
        "logical demand remains the immediate span"
    );
    assert_eq!(
        first.state.available_in_domain(first.domains[0].domain),
        Some(256)
    );
    let second = known(harness.root.project_resource_wave_with_bucket(
        &cold,
        &first.state,
        &[row(0, 1, 2)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    assert_eq!(second.domains[0].transient_peak_bytes, 128);
    assert!(
        matches!(
            harness.root.project_resource_wave_with_bucket(
                &cold,
                &first.state,
                &[row(0, 1, 1)],
                None,
                &mut || true
            ),
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
        ),
        "retained physical capacity cannot be spent as an eager extent"
    );
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let actual = step(&batch, &lane, 1, Some(&bucket));
    assert_eq!(
        first.selected_step_slot(),
        actual
            .claimed_backing()
            .lane_stable_slot_identity()
            .as_ref(),
        "the forecast must name the slot actually selected by the allocator"
    );
    assert_eq!(first.selected_step_slot(), second.selected_step_slot());
    assert_eq!(actual.backing_slices()[0].size_bytes(), 64);
    assert_eq!(actual.backing_slices()[0].capacity_size_bytes(), 256);
    let physical = actual.backing_slices()[0].evidence().segments().to_vec();
    actual.try_retire_normal().unwrap();
    let hot = lane_view(&harness.root, &session, &lane);
    assert!(!cold.same_live_evidence(&hot));
    known(harness.root.project_resource_wave_with_bucket(
        &hot,
        &hot.initial_state(),
        &[row(0, 1, 2)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    let actual = step(&batch, &lane, 2, Some(&bucket));
    assert_eq!(actual.backing_slices()[0].size_bytes(), 128);
    assert_eq!(actual.backing_slices()[0].evidence().segments(), physical);
    actual.try_retire_normal().unwrap();
    assert_eq!(harness.runtime.allocate_calls(), before);
    session.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root); // retained pure views must not pin slots
    assert_eq!(second.state.projected_waves(), 2);
}

fn real_mask_step_identity() -> LaneStableArenaSlotIdentity {
    let (harness, bucket, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "mask-slot", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let actual = step(&batch, &lane, 1, Some(&bucket));
    let identity = actual
        .claimed_backing()
        .lane_stable_slot_identity()
        .unwrap();
    actual.try_retire_normal().unwrap();
    session.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
    identity // a copied identity must not pin the now-closed resource root
}

#[path = "planning_workspace_tests/selection_masks.rs"]
mod selection_masks;

#[path = "planning_workspace_tests/shared_capture.rs"]
mod shared_capture;

#[test]
fn planning_token_masks_roll_forward_per_slot_and_per_candidate() {
    let slot = real_mask_step_identity();
    let initial = ProductTokenMaskResidencySnapshot::new(true, 4, vec![], &mut || true).unwrap();
    let mut candidate = initial.clone();
    assert_eq!(
        candidate
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [true, true]
    );
    assert_eq!(
        candidate
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [false, false]
    );
    let mut sibling = initial;
    assert_eq!(
        sibling
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [true, true]
    );
    assert_eq!(
        candidate
            .project_uploads(None, 17, 2, &mut || true)
            .unwrap(),
        [true, true]
    );
    assert_eq!(
        candidate
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [false, false]
    );
    // A wider batch cannot reuse a neighboring row's cached mask.
    assert_eq!(
        candidate
            .project_uploads(Some(&slot), 17, 3, &mut || true)
            .unwrap(),
        [false, false, true]
    );
}

#[test]
fn planning_token_masks_preserve_nonmatching_entries_and_exact_eviction() {
    let slot = real_mask_step_identity();
    let mut snapshot = ProductTokenMaskResidencySnapshot::new(
        true,
        2,
        vec![
            ProductTokenMaskResidencyEntry::new(slot.clone(), 0, Some(17)),
            ProductTokenMaskResidencyEntry::new(slot.clone(), 2, None),
        ],
        &mut || true,
    )
    .unwrap();
    // The unrelated selection-mask entry counts towards capacity. Actual
    // publish clears the ledger, then publishes only this wave's uploads;
    // the row-zero hit is deliberately NOT republished after that clear.
    assert_eq!(
        snapshot
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [false, true]
    );
    assert_eq!(
        snapshot
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [true, false]
    );
    assert_eq!(
        snapshot
            .project_uploads(Some(&slot), 17, 2, &mut || true)
            .unwrap(),
        [false, false]
    );
    assert_eq!(
        snapshot
            .project_uploads(Some(&slot), 19, 2, &mut || true)
            .unwrap(),
        [true, true]
    );
    // More uploads than the ledger can hold clear it without publishing any.
    assert_eq!(
        snapshot
            .project_uploads(Some(&slot), 23, 3, &mut || true)
            .unwrap(),
        [true, true, true]
    );
    assert_eq!(
        snapshot
            .project_uploads(Some(&slot), 23, 1, &mut || true)
            .unwrap(),
        [true]
    );
}

#[test]
fn planning_token_masks_validate_full_identity_limits_and_budget() {
    let slot = real_mask_step_identity();
    let other = real_mask_step_identity();
    assert_eq!(slot.slot_id(), other.slot_id());
    assert_ne!(slot, other);
    let entry = ProductTokenMaskResidencyEntry::new(slot.clone(), 0, Some(17));
    assert!(ProductTokenMaskResidencySnapshot::new(
        true,
        2,
        vec![entry.clone(), entry.clone()],
        &mut || true
    )
    .is_err());
    assert!(ProductTokenMaskResidencySnapshot::new(true, 0, vec![], &mut || true).is_err());
    let mut snapshot =
        ProductTokenMaskResidencySnapshot::new(true, 2, vec![entry.clone()], &mut || true).unwrap();
    let before = snapshot.clone();
    assert_eq!(
        snapshot.project_uploads(Some(&slot), 17, 1, &mut || false),
        Err(ExecutionCostRouteUnknown::BudgetExhausted)
    );
    assert_eq!(snapshot, before);
    assert_eq!(
        snapshot
            .project_uploads(Some(&other), 17, 1, &mut || true)
            .unwrap(),
        [true]
    );
    let mut ineligible =
        ProductTokenMaskResidencySnapshot::new(false, 2, vec![entry], &mut || true).unwrap();
    assert_eq!(
        ineligible
            .project_uploads(Some(&slot), 17, 1, &mut || true)
            .unwrap(),
        [true]
    );
}

#[test]
fn planning_workspace_cold_capacity_requires_real_resident_backing() {
    let (harness, bucket, lane) = setup(128);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "workspace-short", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = lane_view(&harness.root, &session, &lane);
    let before = harness.runtime.allocate_calls();
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 1)],
            Some(bucket.bucket_id()),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
    ));
    assert_eq!(harness.runtime.allocate_calls(), before);
    assert!(snapshot.same_live_evidence(&lane_view(&harness.root, &session, &lane)));
    // Explicit no-bucket eager work still fits the actual immediate size.
    known(harness.root.project_resource_wave_with_bucket(
        &snapshot,
        &snapshot.initial_state(),
        &[row(0, 0, 1)],
        None,
        &mut || true,
    ));
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    step(&batch, &lane, 1, None).try_retire_normal().unwrap();
    session.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_workspace_busy_slot_is_not_assumed_released() {
    let (harness, bucket, lane) = setup(256);
    let first = admitted_sequence_with_ceiling(&harness.root, "workspace-busy-a", 4);
    let second = admitted_sequence_with_ceiling(&harness.root, "workspace-busy-b", 4);
    let a = first.open_session().unwrap();
    let b = second.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&a)]).unwrap();
    let held = step(&batch, &lane, 1, Some(&bucket));
    let busy = lane_view(&harness.root, &b, &lane);
    assert!(busy.participants()[0].matches_session_identity(&b));
    assert!(!busy.participants()[0].matches_session_identity(&a));
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &busy,
            &busy.initial_state(),
            &[row(0, 0, 1)],
            Some(bucket.bucket_id()),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
    ));
    held.try_retire_normal().unwrap();
    let idle = lane_view(&harness.root, &b, &lane);
    assert!(!busy.same_live_evidence(&idle));
    known(harness.root.project_resource_wave_with_bucket(
        &idle,
        &idle.initial_state(),
        &[row(0, 0, 1)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    a.try_abort_if_quiescent().unwrap();
    b.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(a);
    drop(b);
    drop(first);
    drop(second);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_workspace_bucket_shape_lane_identity_and_budget_are_checked() {
    let (harness, bucket, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "workspace-bounds", 8);
    let session = sequence.open_session().unwrap();
    let snapshot = lane_view(&harness.root, &session, &lane);
    let foreign = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("test.foreign-packed-token").unwrap(),
        bucket.capacity(),
    )
    .unwrap();
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 1)],
            Some(foreign.bucket_id()),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::InvalidInput)
    ));
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 5)],
            Some(bucket.bucket_id()),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::InvalidInput)
    ));
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 1)],
            Some(bucket.bucket_id()),
            &mut || false
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    let other_lane = harness.root.create_execution_lane().unwrap();
    let other = lane_view(&harness.root, &session, &other_lane);
    assert!(!snapshot.same_live_evidence(&other));
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &other,
            &snapshot.initial_state(),
            &[row(0, 0, 1)],
            Some(bucket.bucket_id()),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    // A held arena read cannot cause a blocking lock acquisition in capture.
    let held = harness
        .root
        .dynamic_pools
        .lane_stable_arenas
        .lock()
        .unwrap();
    assert!(matches!(
        harness.root.resource_planning_view_on_lane(
            &[&session],
            &lane,
            ResourcePlanningLimits::default(),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(
            ResourcePlanningReadStage::LaneWorkspace
        ))
    ));
    drop(held);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(other_lane);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_workspace_step_and_invocation_capacity_remain_retained_together() {
    let step_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '8',
        1,
        256,
        TestDemand::Tokens,
    );
    let mut invocation = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '9',
        1,
        256,
        TestDemand::Tokens,
    );
    let node_id = NodeId::new("node/workspace-invocation").unwrap();
    for descriptor in &mut invocation.descriptors {
        let mut value = serde_json::to_value(&*descriptor).unwrap();
        value["lifetime"] = json!("invocation");
        *descriptor = serde_json::from_value(value).unwrap();
    }
    let mut pool = serde_json::to_value(&invocation.pools[0]).unwrap();
    pool["minimum_step_bytes"] = json!(0);
    pool["minimum_invocation_peak_bytes"] = json!(64);
    pool["step_resource_slots"] = json!([]);
    pool["invocation_liveness_mode"] = json!("total_order_reuse");
    pool["invocation_liveness"] = json!([{"node_id": node_id, "resource_ids": [invocation.descriptors[0].base_resource_id()]}]);
    invocation.pools[0] = serde_json::from_value(pool).unwrap();
    let catalog = combine_catalogs(&[step_catalog.clone(), invocation]);
    let (_, bucket) = reusable_step_memory_plan(step_catalog.pool_id.clone());
    let budgets = catalog
        .pools
        .iter()
        .map(|pool| {
            let step = pool.pool_id() == &step_catalog.pool_id;
            ReusablePoolWorkspaceBudget::new(
                pool.pool_id().clone(),
                if step { 256 } else { 0 },
                if step { 0 } else { 256 },
            )
            .unwrap()
        })
        .collect();
    let memory = ReusableExecutionMemoryPlan::new(
        1,
        1,
        vec![ResolvedReusableExecutionBucket::new(bucket.clone(), budgets).unwrap()],
    )
    .unwrap();
    let runtime = new_runtime(&catalog, 512);
    let harness = harness_with_nodes_and_reusable(
        runtime,
        catalog,
        512,
        false,
        Arc::from(vec![PlanNode::resource_test_node(node_id)]),
        Some(memory),
    );
    for id in &harness.pool_ids {
        harness
            .root
            .maintenance_controller
            .grow_pool(id, 256)
            .unwrap();
    }
    let lane = harness.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "step-invocation", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = lane_view(&harness.root, &session, &lane);
    let predicted = known(harness.root.project_resource_wave_with_bucket(
        &snapshot,
        &snapshot.initial_state(),
        &[row(0, 0, 1)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    assert_eq!(predicted.domains.len(), 2);
    assert!(predicted
        .domains
        .iter()
        .all(|domain| domain.transient_peak_bytes == 64 && domain.persistent_bytes == 0));
    known(harness.root.project_resource_wave_with_bucket(
        &snapshot,
        &predicted.state,
        &[row(0, 1, 2)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let (_, step_requests) = binding
        .scoped_demand(
            AllocationLifetime::Step,
            None,
            shape(1),
            shape(1),
            Some(&bucket),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
    let (_, wave_requests) = binding
        .submission_wave_demand(
            shape(1),
            shape(1),
            Some(&bucket),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
    let claim = |requests: &[EvaluatedBackingRequest<'_>]| match harness
        .root
        .dynamic_pools
        .prepare_lane_stable_claim(&lane, requests)
        .unwrap()
    {
        LaneBackingPrepareDecision::Prepared(claim) => claim.commit().into_parts(),
        _ => panic!("resident real lane claim deferred"),
    };
    let (step_slices, step_slot) = claim(&step_requests);
    let (wave_slices, wave_slot) = claim(&wave_requests);
    assert_eq!(step_slices[0].capacity_size_bytes(), 256);
    assert_eq!(wave_slices[0].capacity_size_bytes(), 256);
    drop(wave_slices);
    drop(wave_slot);
    drop(step_slices);
    drop(step_slot);
    let actual = lane_view(&harness.root, &session, &lane);
    assert!(!snapshot.same_live_evidence(&actual));
    known(harness.root.project_resource_wave_with_bucket(
        &actual,
        &actual.initial_state(),
        &[row(0, 1, 2)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    assert!(harness
        .root
        .maintenance_controller
        .status()
        .unwrap()
        .pools()
        .iter()
        .all(|pool| pool.live_occupancy().lane_stable().total().physical_bytes() == 256));
    drop(step_requests);
    drop(wave_requests);
    drop(binding);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}
