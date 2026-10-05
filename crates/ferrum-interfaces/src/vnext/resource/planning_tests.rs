//! Uses the real resource allocator/admission fixtures, without a GPU.
use super::*;

#[path = "planning_pool_cow_tests.rs"]
mod pool_cow;

fn known<T: std::fmt::Debug>(value: ResourcePlanningAvailability<T>) -> T {
    match value {
        ResourcePlanningAvailability::Known(value) => value,
        ResourcePlanningAvailability::Unknown(reason) => panic!("expected known: {reason:?}"),
    }
}

pub(super) fn view(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    sessions: &[&SequenceSession<TestRuntime>],
) -> ResourcePlanningView {
    view_with_limits(root, sessions, ResourcePlanningLimits::default())
}

fn view_with_limits(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    sessions: &[&SequenceSession<TestRuntime>],
    limits: ResourcePlanningLimits,
) -> ResourcePlanningView {
    known_read(|| root.resource_planning_view(sessions, limits, &mut || true))
}

pub(super) fn known_read<T: std::fmt::Debug>(
    mut read: impl FnMut() -> ResourcePlanningAvailability<T>,
) -> T {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match read() {
            ResourcePlanningAvailability::Known(view) => return view,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(
                stage,
            )) => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "resource read remained contended: {stage:?}"
                );
                std::thread::yield_now();
            }
            ResourcePlanningAvailability::Unknown(reason) => {
                panic!("expected known resource view: {reason:?}")
            }
        }
    }
}

fn row(participant_index: usize, start_token: u64, token_count: u64) -> ResourcePlanningRow {
    ResourcePlanningRow {
        participant_index,
        start_token,
        token_count,
    }
}

#[test]
fn planning_shared_persistent_pool_is_charged_across_rows_and_rollout() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        320,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 320);
    let harness = harness(Arc::clone(&runtime), catalog, 320, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 320)
        .unwrap();
    let first = admitted_sequence_with_ceiling(&harness.root, "first", 4);
    let second = admitted_sequence_with_ceiling(&harness.root, "second", 4);
    let a = first.open_session().unwrap();
    let b = second.open_session().unwrap();
    let before = runtime.allocate_calls();
    let snapshot = view(&harness.root, &[&a, &b]);
    let state = snapshot.initial_state();
    let first_only = known(harness.root.project_resource_wave(
        &snapshot,
        &state,
        &[row(0, 1, 2)],
        &mut || true,
    ));
    let second_only = known(harness.root.project_resource_wave(
        &snapshot,
        &state,
        &[row(1, 1, 2)],
        &mut || true,
    ));
    assert_eq!(first_only.domains[0].persistent_bytes, 128);
    assert_eq!(second_only.domains[0].persistent_bytes, 128);
    assert_eq!(first_only.state.covered_tokens(0), Some(3));
    assert!(matches!(
        harness.root.project_resource_wave(
            &snapshot,
            &first_only.state,
            &[row(1, 1, 2)],
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::LogicalCapacity)
    ));
    assert!(matches!(
        harness.root.project_resource_wave(
            &snapshot,
            &state,
            &[row(0, 1, 2), row(1, 1, 2)],
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::LogicalCapacity)
    ));
    assert_eq!(state.covered_tokens(0), Some(1));
    assert_eq!(runtime.allocate_calls(), before);
    assert_eq!(
        snapshot.participants(),
        view(&harness.root, &[&a, &b]).participants()
    );
    a.try_abort_if_quiescent().unwrap();
    b.try_abort_if_quiescent().unwrap();
    drop(a);
    drop(b);
    drop(first);
    drop(second);
    // Keep the snapshots and simulated claims alive: none may pin the plan.
    close_dynamic_test_root(harness.root);
    assert_eq!(first_only.state.covered_tokens(0), Some(3));
}

#[test]
fn planning_shared_step_slot_matches_actual_admission_and_releases_transient_only() {
    let catalog = shared_step_activation_catalog(linear_profile());
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(Arc::clone(&runtime), catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "step", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = view(&harness.root, &[&session]);
    let projected = known(harness.root.project_resource_wave(
        &snapshot,
        &snapshot.initial_state(),
        &[row(0, 0, 2)],
        &mut || true,
    ));
    assert_eq!(projected.domains[0].transient_peak_bytes, 128);
    assert_eq!(projected.domains[0].persistent_bytes, 0);
    assert_eq!(
        projected
            .state
            .available_in_domain(projected.domains[0].domain),
        Some(256)
    );
    let before = runtime.allocate_calls();
    let lane = harness.root.create_execution_lane().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span(2)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let StepResourceAdmissionDecision::Admitted(step) =
        batch.try_begin_step(request, &lane).unwrap()
    else {
        panic!("projected eager step should fit");
    };
    assert_eq!(
        step.backing_slices()[0].capacity_size_bytes(),
        projected.domains[0].transient_peak_bytes
    );
    assert_eq!(runtime.allocate_calls(), before);
    step.try_retire_normal().unwrap();
    drop(batch);
    drop(lane);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_contiguous_fragmentation_is_not_total_free_capacity() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        'b',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "fragmented", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = view(&harness.root, &[&session]);
    let before = runtime.allocate_calls();
    assert!(matches!(
        harness.root.project_resource_wave(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 2)],
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
    ));
    known(harness.root.project_resource_wave(
        &snapshot,
        &snapshot.initial_state(),
        &[row(0, 0, 1)],
        &mut || true,
    ));
    assert_eq!(runtime.allocate_calls(), before);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_busy_limits_and_budget_never_wait_or_allocate() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        'c',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 128)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "bounds", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = view(&harness.root, &[&session]);
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    let guard = pool.state.lock().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match harness.root.resource_planning_view(
            &[&session],
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(
                ResourcePlanningReadStage::PhysicalPool,
            )) => break,
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(
                stage,
            )) => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "earlier read remained unavailable: {stage:?}"
                );
                std::thread::yield_now();
            }
            other => panic!("locked physical pool must not produce a known view: {other:?}"),
        }
    }
    drop(guard);
    assert!(matches!(
        harness.root.resource_planning_view(
            &[&session],
            ResourcePlanningLimits::default(),
            &mut || false
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    let mut polls = 0;
    assert!(matches!(
        harness.root.project_resource_wave(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 1)],
            &mut || {
                polls += 1;
                polls < 3
            }
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    let mut limits = ResourcePlanningLimits::default();
    limits.maximum_projected_waves = 1;
    let short = view_with_limits(&harness.root, &[&session], limits);
    let once = known(harness.root.project_resource_wave(
        &short,
        &short.initial_state(),
        &[row(0, 0, 1)],
        &mut || true,
    ));
    assert!(matches!(
        harness
            .root
            .project_resource_wave(&short, &once.state, &[row(0, 0, 1)], &mut || true),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::LimitExceeded)
    ));
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_rejects_foreign_state_overflow_duplicate_and_excess_context() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        'd',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(runtime, catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 128)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "invalid", 4);
    let session = sequence.open_session().unwrap();
    let a = view(&harness.root, &[&session]);
    let b = view(&harness.root, &[&session]);
    assert_eq!(a.participants(), b.participants());
    assert!(matches!(
        harness
            .root
            .project_resource_wave(&a, &b.initial_state(), &[row(0, 0, 1)], &mut || true),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    for rows in [
        vec![row(0, u64::MAX, 1)],
        vec![row(0, 4, 1)],
        vec![row(0, 0, 1), row(0, 0, 1)],
        vec![row(9, 0, 1)],
    ] {
        assert!(matches!(
            harness
                .root
                .project_resource_wave(&a, &a.initial_state(), &rows, &mut || true),
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::InvalidInput)
        ));
    }
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_real_backing_extension_invalidates_view_and_uses_same_delta() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'e',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "generation", 4);
    let session = sequence.open_session().unwrap();
    let before = view(&harness.root, &[&session]);
    let predicted = known(harness.root.project_resource_wave(
        &before,
        &before.initial_state(),
        &[row(0, 1, 1)],
        &mut || true,
    ));
    session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap();
    let after = view(&harness.root, &[&session]);
    assert!(!before.same_live_evidence(&after));
    assert_ne!(
        before.participants()[0].backing_generation(),
        after.participants()[0].backing_generation()
    );
    assert_eq!(
        predicted.state.covered_tokens(0),
        Some(after.participants()[0].covered_tokens())
    );
    assert_eq!(
        predicted
            .state
            .available_in_domain(predicted.domains[0].domain),
        after
            .initial_state()
            .available_in_domain(predicted.domains[0].domain)
    );
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_reusable_lane_retention_remains_explicit_unknown() {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '7',
        1,
        256,
        TestDemand::Tokens,
    );
    let (memory, _) = reusable_step_memory_plan(catalog.pool_id.clone());
    let runtime = new_runtime(&catalog, 256);
    let harness = harness_with_reusable(runtime, catalog, 256, memory);
    let sequence = admitted_sequence(&harness.root, "reusable");
    let session = sequence.open_session().unwrap();
    assert!(matches!(
        harness.root.resource_planning_view(
            &[&session],
            ResourcePlanningLimits::default(),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReusableExecution)
    ));
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_reversed_product_order_matches_real_canonical_extensions() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'f',
        1,
        448,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 448);
    let harness = harness(runtime, catalog, 448, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 448)
        .unwrap();
    let first = admitted_sequence_with_ceiling(&harness.root, "ordered-a", 4);
    let second = admitted_sequence_with_ceiling(&harness.root, "ordered-b", 4);
    let a = first.open_session().unwrap();
    let b = second.open_session().unwrap();
    // The product registry order need not be runtime authority order.
    let snapshot = view(&harness.root, &[&b, &a]);
    let reversed = known(harness.root.project_resource_wave(
        &snapshot,
        &snapshot.initial_state(),
        &[row(0, 1, 1), row(1, 1, 2)],
        &mut || true,
    ));
    let canonical = known(harness.root.project_resource_wave(
        &snapshot,
        &snapshot.initial_state(),
        &[row(1, 1, 2), row(0, 1, 1)],
        &mut || true,
    ));
    assert_eq!(reversed.domains, canonical.domains);
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&b), Arc::clone(&a)]).unwrap();
    assert_eq!(
        batch.sessions()[0].sequence_authority(),
        a.sequence_authority()
    );
    for session in batch.sessions() {
        let target = if session.sequence_authority() == a.sequence_authority() {
            3
        } else {
            2
        };
        assert!(matches!(
            session
                .try_ensure_backing_covers(
                    SequenceResourceExtensionRequest::new(
                        work(target),
                        AdmissionPressureAction::WaitForRelease
                    )
                    .unwrap()
                )
                .unwrap(),
            SequenceResourceExtensionDecision::Extended(_)
        ));
    }
    let actual = view(&harness.root, &[&b, &a]);
    for index in 0..2 {
        assert_eq!(
            reversed.state.covered_tokens(index),
            Some(actual.participants()[index].covered_tokens())
        );
    }
    let domain = reversed.domains[0].domain;
    assert_eq!(
        reversed.state.available_in_domain(domain),
        actual.initial_state().available_in_domain(domain)
    );
    // Replay the next wave on predicted and real allocators, including another
    // persistent allocation: their remaining usable capacity must agree.
    let predicted_next = known(harness.root.project_resource_wave(
        &snapshot,
        &reversed.state,
        &[row(0, 2, 1)],
        &mut || true,
    ));
    let actual_next = known(harness.root.project_resource_wave(
        &actual,
        &actual.initial_state(),
        &[row(0, 2, 1)],
        &mut || true,
    ));
    assert_eq!(predicted_next.domains, actual_next.domains);
    assert_eq!(
        predicted_next.state.available_in_domain(domain),
        actual_next.state.available_in_domain(domain)
    );
    drop(batch);
    a.try_abort_if_quiescent().unwrap();
    b.try_abort_if_quiescent().unwrap();
    drop(a);
    drop(b);
    drop(first);
    drop(second);
    close_dynamic_test_root(harness.root);
}

#[path = "planning_workspace_tests.rs"]
mod workspace;

#[test]
fn planning_unmaterialized_capacity_stays_unknown_until_real_growth() {
    // Incremental growth creates two distinct 64-byte backing chunks. Paged
    // storage can use both; a contiguous 128-byte request must remain blocked
    // and is covered by planning_contiguous_fragmentation_is_not_total_free_capacity.
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Step,
        '8',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 128);
    let harness = harness(Arc::clone(&runtime), catalog, 128, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "materialize", 4);
    let session = sequence.open_session().unwrap();
    let before = view(&harness.root, &[&session]);
    let allocated = runtime.allocate_calls();
    assert!(matches!(
        harness.root.project_resource_wave(
            &before,
            &before.initial_state(),
            &[row(0, 0, 2)],
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::UnmaterializedCapacity)
    ));
    assert_eq!(
        runtime.allocate_calls(),
        allocated,
        "projection must not materialize capacity"
    );
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    let after = view(&harness.root, &[&session]);
    assert!(!before.same_live_evidence(&after));
    // Growth does not update an earlier immutable capture.
    assert!(matches!(
        harness.root.project_resource_wave(
            &before,
            &before.initial_state(),
            &[row(0, 0, 2)],
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::UnmaterializedCapacity)
    ));
    known(harness.root.project_resource_wave(
        &after,
        &after.initial_state(),
        &[row(0, 0, 2)],
        &mut || true,
    ));
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_later_row_hard_ceiling_is_not_hidden_by_first_growable_claim() {
    for maximum in [320, 384] {
        let catalog = pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            '9',
            1,
            maximum,
            TestDemand::Tokens,
        );
        let runtime = new_runtime(&catalog, maximum);
        let harness = harness(Arc::clone(&runtime), catalog, maximum, false);
        harness
            .root
            .maintenance_controller
            .grow_pool(&harness.pool_ids[0], 192)
            .unwrap();
        let first = admitted_sequence_with_ceiling(&harness.root, "first-unmaterialized", 4);
        let second = admitted_sequence_with_ceiling(&harness.root, "second-unmaterialized", 4);
        let a = first.open_session().unwrap();
        let b = second.open_session().unwrap();
        let snapshot = view(&harness.root, &[&a, &b]);
        let allocated = runtime.allocate_calls();
        // The two live one-token rows own 128 bytes. Each future row requires
        // another 128. Current availability is 64; both original ceilings fit
        // one future row, but only the 384-byte ceiling fits them together.
        assert!(matches!(
            harness.root.project_resource_wave(
                &snapshot,
                &snapshot.initial_state(),
                &[row(0, 1, 2)],
                &mut || true
            ),
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::UnmaterializedCapacity)
        ));
        let expected = if maximum == 320 {
            ResourcePlanningUnknown::LogicalCapacity
        } else {
            ResourcePlanningUnknown::UnmaterializedCapacity
        };
        for rows in [[row(0, 1, 2), row(1, 1, 2)], [row(1, 1, 2), row(0, 1, 2)]] {
            assert!(
                matches!(harness.root.project_resource_wave(&snapshot, &snapshot.initial_state(), &rows, &mut || true),
                ResourcePlanningAvailability::Unknown(reason) if reason == expected)
            );
        }
        assert_eq!(runtime.allocate_calls(), allocated);
        harness
            .root
            .maintenance_controller
            .grow_pool(&harness.pool_ids[0], maximum - 192)
            .unwrap();
        let materialized = view(&harness.root, &[&a, &b]);
        let projected = harness.root.project_resource_wave(
            &materialized,
            &materialized.initial_state(),
            &[row(0, 1, 2), row(1, 1, 2)],
            &mut || true,
        );
        if maximum == 384 {
            let projected = known(projected);
            assert_eq!(projected.state.covered_tokens(0), Some(3));
            assert_eq!(projected.state.covered_tokens(1), Some(3));
        } else {
            assert!(matches!(
                projected,
                ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::LogicalCapacity)
            ));
        }
        a.try_abort_if_quiescent().unwrap();
        b.try_abort_if_quiescent().unwrap();
        drop(a);
        drop(b);
        drop(first);
        drop(second);
        close_dynamic_test_root(harness.root);
    }
}

#[test]
fn planning_step_and_invocation_share_one_current_and_maximum_wave_ledger() {
    for maximum in [192, 256] {
        let mut catalog = pool_catalog(
            linear_profile(),
            AllocationLifetime::Step,
            '6',
            2,
            maximum,
            TestDemand::Tokens,
        );
        let node_id = NodeId::try_from("node.shared-step-invocation-capacity".to_owned()).unwrap();
        let step_id = catalog.descriptors[0].base_resource_id().clone();
        let invocation_id = catalog.descriptors[1].base_resource_id().clone();
        let mut descriptor = serde_json::to_value(&catalog.descriptors[1]).unwrap();
        descriptor["lifetime"] = json!("invocation");
        catalog.descriptors[1] = serde_json::from_value(descriptor).unwrap();
        let mut pool = serde_json::to_value(&catalog.pools[0]).unwrap();
        pool["minimum_step_bytes"] = json!(64);
        pool["minimum_invocation_peak_bytes"] = json!(64);
        pool["step_resource_slots"] = json!([{"kind": "dedicated", "resource_ids": [step_id]}]);
        pool["invocation_liveness_mode"] = json!("total_order_reuse");
        pool["invocation_liveness"] =
            json!([{"node_id": node_id, "resource_ids": [invocation_id]}]);
        catalog.pools[0] = serde_json::from_value(pool).unwrap();
        let runtime = new_runtime(&catalog, maximum);
        let harness = harness_with_nodes(
            Arc::clone(&runtime),
            catalog,
            maximum,
            false,
            Arc::from(vec![PlanNode::resource_test_node(node_id)]),
        );
        harness
            .root
            .maintenance_controller
            .grow_pool(&harness.pool_ids[0], 128)
            .unwrap();
        let sequence = admitted_sequence_with_ceiling(&harness.root, "overlapping-lifetimes", 4);
        let session = sequence.open_session().unwrap();
        let snapshot = view(&harness.root, &[&session]);
        let allocated = runtime.allocate_calls();
        let expected = if maximum == 192 {
            ResourcePlanningUnknown::LogicalCapacity
        } else {
            ResourcePlanningUnknown::UnmaterializedCapacity
        };
        assert!(
            matches!(harness.root.project_resource_wave(&snapshot, &snapshot.initial_state(), &[row(0, 0, 2)], &mut || true),
            ResourcePlanningAvailability::Unknown(reason) if reason == expected)
        );
        assert_eq!(runtime.allocate_calls(), allocated);
        if maximum == 256 {
            harness
                .root
                .maintenance_controller
                .grow_pool(&harness.pool_ids[0], 128)
                .unwrap();
            let materialized = view(&harness.root, &[&session]);
            let projected = known(harness.root.project_resource_wave(
                &materialized,
                &materialized.initial_state(),
                &[row(0, 0, 2)],
                &mut || true,
            ));
            assert_eq!(projected.domains.len(), 1);
            assert_eq!(projected.domains[0].transient_peak_bytes, 256);
            assert_eq!(
                projected
                    .state
                    .available_in_domain(projected.domains[0].domain),
                Some(256)
            );
        }
        session.try_abort_if_quiescent().unwrap();
        drop(session);
        drop(sequence);
        close_dynamic_test_root(harness.root);
    }
}
