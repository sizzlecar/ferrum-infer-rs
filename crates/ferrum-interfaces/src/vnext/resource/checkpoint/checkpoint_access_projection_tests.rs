//! Numeric allocator decisions are compared with the existing native owner.
//! This CPU backend encodes no-op copies; it does not establish device speed.
use super::*;
use crate::vnext::{
    NativeCheckpointTransferCostDomain, NativeCheckpointTransferKind, ResourcePlanningAvailability,
    ResourcePlanningLimits, ResourcePlanningState, ResourcePlanningUnknown, ResourcePlanningView,
};

/// Each independent fake device has its own capacity account. The production
/// registry deliberately shares accounts for equal device IDs, which would make
/// parallel fixtures contend on a device that they do not actually share.
fn projection_harness(mut spec: checkpoint_fixture::Spec) -> RestoreHarness {
    static NEXT_DEVICE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT_DEVICE
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |value| value.checked_add(1),
        )
        .expect("projection fixture device identity exhausted");
    spec.device_id = Some(DeviceId::new(format!("checkpoint-projection-{id}")).unwrap());
    prefix_harness(spec)
}

fn known<T>(value: ResourcePlanningAvailability<T>) -> T {
    match value {
        ResourcePlanningAvailability::Known(value) => value,
        ResourcePlanningAvailability::Unknown(reason) => {
            panic!("expected numeric Known: {reason:?}")
        }
    }
}
fn view(h: &RestoreHarness, target: &SequenceSession<TestRuntime>) -> ResourcePlanningView {
    known(h.root.resource_planning_view(
        &[h.session.as_ref(), target],
        ResourcePlanningLimits::default(),
        &mut || true,
    ))
}
fn free_bytes(h: &RestoreHarness) -> u64 {
    h.root
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .map(|pool| pool.free_bytes())
        .sum()
}

#[test]
fn retained_checkpoint_binds_only_to_a_fresh_accounted_view_without_reallocation() {
    let h = projection_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    let target = admitted_full_target(&h, "retained-view-target", &[19, 23]);
    let before_capture = view(&h, &target);
    let checkpoint = observed_capture(&h, &lane, &reaper);
    assert!(matches!(
        h.root.bind_retained_checkpoint(
            &before_capture,
            &before_capture.initial_state(),
            &h.fixture.plan,
            &checkpoint,
            0,
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    let fresh = view(&h, &target);
    let initial = fresh.initial_state();
    let allocations = h.runtime.allocate_calls();
    let free = free_bytes(&h);
    let bound = known(h.root.bind_retained_checkpoint(
        &fresh,
        &initial,
        &h.fixture.plan,
        &checkpoint,
        0,
        &mut || true,
    ));
    assert_eq!(h.runtime.allocate_calls(), allocations);
    assert_eq!(free_bytes(&h), free);
    assert_eq!(bound.state.projected_waves(), 0);
    assert_actual_domains(&h, &bound.state);
    assert!(matches!(
        h.root.bind_retained_checkpoint(
            &fresh,
            &bound.state,
            &h.fixture.plan,
            &checkpoint,
            0,
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::InvalidInput)
    ));
    assert!(matches!(
        h.root.bind_retained_checkpoint(
            &fresh,
            &initial,
            &h.fixture.plan,
            &checkpoint,
            1,
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    assert!(matches!(
        h.root.bind_retained_checkpoint(
            &fresh,
            &initial,
            &h.fixture.plan,
            &checkpoint,
            0,
            &mut || false
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    let other = view(&h, &target);
    assert!(matches!(
        h.root.project_checkpoint_restore(
            &other,
            &other.initial_state(),
            &h.fixture.plan,
            &bound.checkpoint,
            1,
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    let _restored = known(h.root.project_checkpoint_restore(
        &fresh,
        &bound.state,
        &h.fixture.plan,
        &bound.checkpoint,
        1,
        &mut || true,
    ));
    assert!(!known(h.root.checkpoint_restore_completed(
        &fresh,
        &checkpoint,
        1,
        &mut || true
    )));
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint(
                &h.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(
        transfer.wait_for_recovery().unwrap(),
        NativeCheckpointObservation::Ready
    );
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("native restore must produce publication");
    };
    assert!(matches!(
        h.root.resource_planning_view(
            &[h.session.as_ref(), target.as_ref()],
            ResourcePlanningLimits::default(),
            &mut || true
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BusyOrUnavailable)
    ));
    publication.acknowledge().unwrap();
    let after_ack = view(&h, &target);
    assert!(known(h.root.checkpoint_restore_completed(
        &after_ack,
        &checkpoint,
        1,
        &mut || true
    )));
    assert!(!known(h.root.checkpoint_restore_completed(
        &fresh,
        &checkpoint,
        1,
        &mut || true
    )));
    target.try_abort_if_quiescent().unwrap();
}

#[test]
fn ready_checkpoint_survives_retired_producer_with_native_restore_and_ack() {
    let h = projection_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    let target = admitted_full_target(&h, "ready-after-retirement", &[19, 23]);
    let before = known(h.root.resource_planning_view(
        &[target.as_ref()],
        ResourcePlanningLimits::default(),
        &mut || true,
    ));
    let checkpoint = observed_capture(&h, &lane, &reaper);
    assert!(matches!(
        h.root.bind_ready_checkpoint(
            &before,
            &before.initial_state(),
            &h.fixture.plan,
            &checkpoint,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    // Close the real native source; there is no producer row in this capture.
    h.session.try_abort_if_quiescent().unwrap();
    let fresh = known(h.root.resource_planning_view(
        &[target.as_ref()],
        ResourcePlanningLimits::default(),
        &mut || true,
    ));
    assert_eq!(fresh.participants().len(), 1);
    let allocations = h.runtime.allocate_calls();
    let free = free_bytes(&h);
    let retained = h
        .root
        .dynamic_pools
        .logical_admission
        .checkpoint_retained_bytes()
        .unwrap();
    let bound = known(h.root.bind_ready_checkpoint(
        &fresh,
        &fresh.initial_state(),
        &h.fixture.plan,
        &checkpoint,
        &mut || true,
    ));
    assert_eq!(bound.checkpoint.source_participant(), None);
    assert_eq!(bound.state.projected_waves(), 0);
    let projected = known(h.root.project_checkpoint_restore(
        &fresh,
        &bound.state,
        &h.fixture.plan,
        &bound.checkpoint,
        0,
        &mut || true,
    ));
    assert_eq!(h.runtime.allocate_calls(), allocations);
    assert_eq!(free_bytes(&h), free);
    assert_eq!(
        h.root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        retained
    );
    assert!(!known(h.root.checkpoint_restore_completed(
        &fresh,
        &checkpoint,
        0,
        &mut || true
    )));
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint(
                &h.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(
        transfer.wait_for_recovery().unwrap(),
        NativeCheckpointObservation::Ready
    );
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("ready checkpoint must restore through native publication");
    };
    assert!(matches!(
        h.root.resource_planning_view(
            &[target.as_ref()],
            ResourcePlanningLimits::default(),
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BusyOrUnavailable)
    ));
    publication.acknowledge().unwrap();
    let after = known(h.root.resource_planning_view(
        &[target.as_ref()],
        ResourcePlanningLimits::default(),
        &mut || true,
    ));
    assert!(known(h.root.checkpoint_restore_completed(
        &after,
        &checkpoint,
        0,
        &mut || true
    )));
    assert_actual_domains(&h, &projected.state);
    target.try_abort_if_quiescent().unwrap();
}

#[test]
fn ready_checkpoint_keeps_owner_fence_budget_and_no_self_restore_guards() {
    let h = projection_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&h, &lane, &reaper);
    let checkpoint = observed_capture(&h, &lane, &reaper);
    let target = admitted_full_target(&h, "ready-owner-guards", &[19, 23]);
    let fresh = view(&h, &target);
    let initial = fresh.initial_state();
    let bound = known(h.root.bind_ready_checkpoint(
        &fresh,
        &initial,
        &h.fixture.plan,
        &checkpoint,
        &mut || true,
    ));
    assert!(matches!(
        h.root.project_checkpoint_restore(
            &fresh,
            &bound.state,
            &h.fixture.plan,
            &bound.checkpoint,
            0,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    assert!(matches!(
        h.root
            .bind_ready_checkpoint(&fresh, &initial, &h.fixture.plan, &checkpoint, &mut || {
                false
            },),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    let sibling = view(&h, &target);
    assert!(matches!(
        h.root.project_checkpoint_restore(
            &sibling,
            &sibling.initial_state(),
            &h.fixture.plan,
            &bound.checkpoint,
            1,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    let foreign = projection_harness(Default::default());
    assert!(matches!(
        foreign.root.bind_ready_checkpoint(
            &fresh,
            &initial,
            &foreign.fixture.plan,
            &checkpoint,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    target.try_abort_if_quiescent().unwrap();
}

#[test]
fn completed_capture_boundary_is_proven_in_the_current_resource_bracket() {
    use crate::vnext::{
        ExecutionCostRouteAvailability, ExecutionCostRouteUnknown, ExecutionCostRouteView,
    };
    let h = projection_harness(Default::default());
    let lane = h.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    let target = admitted_full_target(&h, "completed-boundary-target", &[19, 23]);
    let unproven = view(&h, &target);
    assert!(unproven.participants()[0]
        .completed_checkpoint_boundary()
        .is_none());
    prove_prefix_source(&h, &lane, &reaper);
    let fresh = view(&h, &target);
    let proof = fresh.participants()[0]
        .completed_checkpoint_boundary()
        .unwrap();
    assert_eq!(
        (
            proof.span_start(),
            proof.completed_tokens(),
            proof.prompt_tokens()
        ),
        (0, 1, 2)
    );
    assert!(!proof.restored());
    let route = ExecutionCostRouteView {
        fence: Arc::new(()),
        resources: fresh,
        structured_capture: false,
        initial_frontiers: vec![1, 0],
        readback_available_bytes: 0,
        lane_id: lane.id(),
        token_masks: None,
        graph_stream_state: None,
        graph_catalog: None,
    };
    let initial = route.initial_state();
    assert!(matches!(
        h.root.project_future_checkpoint_capture(
            &route,
            &initial,
            &h.fixture.plan,
            lane.descriptor(),
            0,
            0,
            1,
            2,
            &mut || true
        ),
        ExecutionCostRouteAvailability::Known(_)
    ));
    assert!(matches!(
        h.root.project_future_checkpoint_capture(
            &route,
            &initial,
            &h.fixture.plan,
            lane.descriptor(),
            0,
            1,
            1,
            2,
            &mut || true
        ),
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::InvalidInput)
    ));
    let mut stale = route.clone();
    stale.resources = unproven;
    assert!(matches!(
        h.root.project_future_checkpoint_capture(
            &stale,
            &stale.initial_state(),
            &h.fixture.plan,
            lane.descriptor(),
            0,
            0,
            1,
            2,
            &mut || true
        ),
        ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::InvalidInput)
    ));
    target.try_abort_if_quiescent().unwrap();
}
fn assert_actual_domains(h: &RestoreHarness, numeric: &ResourcePlanningState) {
    for domain in h
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap()
        .domains()
    {
        if let Some(available) = numeric.available_in_domain(domain.domain()) {
            assert_eq!(
                available,
                domain.available().get(),
                "numeric and actual domain {:?} differ",
                domain.domain()
            );
        }
    }
}

#[test]
fn projected_checkpoint_copy_and_initialization_match_native_publication() {
    let harness = projection_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "projected-target", &[19, 23]);
    let sink = observe_transfers(&reaper, 2);
    let initial = view(&harness, &target);
    let state = initial.initial_state();
    let allocated_before = harness.runtime.allocate_calls();
    let free_before = free_bytes(&harness);
    let retained_before = harness
        .root
        .dynamic_pools
        .logical_admission
        .checkpoint_retained_bytes()
        .unwrap();
    let captured = known(harness.root.project_checkpoint_capture(
        &initial,
        &state,
        &harness.fixture.plan,
        0,
        1,
        &mut || true,
    ));
    let sibling = known(harness.root.project_checkpoint_capture(
        &initial,
        &state,
        &harness.fixture.plan,
        0,
        1,
        &mut || true,
    ));
    assert!(matches!(
        harness.root.project_checkpoint_restore(
            &initial,
            &sibling.state,
            &harness.fixture.plan,
            &captured.checkpoint,
            1,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    let restored = known(harness.root.project_checkpoint_restore(
        &initial,
        &captured.state,
        &harness.fixture.plan,
        &captured.checkpoint,
        1,
        &mut || true,
    ));
    assert_eq!(harness.runtime.allocate_calls(), allocated_before);
    assert_eq!(free_bytes(&harness), free_before);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        retained_before
    );
    assert_eq!(state.projected_waves(), 0);

    let checkpoint = observed_capture(&harness, &lane, &reaper);
    {
        let observations = sink.samples.lock().unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].geometry(), &captured.geometry);
        let domain = NativeCheckpointTransferCostDomain::from_projection(
            captured.checkpoint.byte_plan(),
            lane.descriptor(),
            NativeCheckpointTransferKind::Capture,
            captured.geometry,
        );
        assert_eq!(observations[0].cost_domain(), &domain);
    }
    assert_eq!(
        free_before - free_bytes(&harness),
        checkpoint.retained_bytes()
    );
    assert_actual_domains(&harness, &captured.state);
    let mut transfer = access_submitted(
        reaper
            .try_restore_sequence_checkpoint(
                &harness.fixture.plan,
                Arc::clone(&target),
                &checkpoint,
                Arc::from([19, 23]),
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(transfer.poll().unwrap(), NativeCheckpointObservation::Ready);
    let Some(NativeCheckpointResult::Restored(publication)) = transfer.take_result().unwrap()
    else {
        panic!("expected native restoration");
    };
    assert_eq!(sink.samples.lock().unwrap().len(), 1);
    publication.acknowledge().unwrap();
    {
        let observations = sink.samples.lock().unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[1].geometry(), &restored.geometry);
        let domain = NativeCheckpointTransferCostDomain::from_projection(
            captured.checkpoint.byte_plan(),
            lane.descriptor(),
            NativeCheckpointTransferKind::Restore,
            restored.geometry,
        );
        assert_eq!(observations[1].cost_domain(), &domain);
    }
    assert_actual_domains(&harness, &restored.state);
    assert_eq!(
        free_before - free_bytes(&harness),
        checkpoint.retained_bytes()
    );
    drop(transfer);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(sink);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn numeric_checkpoint_rejects_foreign_capture_sibling_and_exhausted_budget_without_allocation() {
    let harness = projection_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "projection-invalid-target", &[19, 23]);
    let initial = view(&harness, &target);
    let foreign = view(&harness, &target);
    let state = initial.initial_state();
    let allocated_before = harness.runtime.allocate_calls();
    let free_before = free_bytes(&harness);
    let captured = known(harness.root.project_checkpoint_capture(
        &initial,
        &state,
        &harness.fixture.plan,
        0,
        1,
        &mut || true,
    ));
    assert!(matches!(
        harness.root.project_checkpoint_restore(
            &initial,
            &state,
            &harness.fixture.plan,
            &captured.checkpoint,
            1,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    assert!(matches!(
        harness.root.project_checkpoint_restore(
            &foreign,
            &foreign.initial_state(),
            &harness.fixture.plan,
            &captured.checkpoint,
            1,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    assert!(matches!(
        harness.root.project_checkpoint_capture(
            &initial,
            &state,
            &harness.fixture.plan,
            0,
            1,
            &mut || false,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    assert!(matches!(
        harness.root.project_checkpoint_restore(
            &initial,
            &captured.state,
            &harness.fixture.plan,
            &captured.checkpoint,
            1,
            &mut || false,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::BudgetExhausted)
    ));
    assert_eq!(harness.runtime.allocate_calls(), allocated_before);
    assert_eq!(free_bytes(&harness), free_before);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        0
    );
    assert_actual_domains(&harness, &state);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_numeric_retention_bound_matches_native_skip() {
    let harness = projection_harness(checkpoint_fixture::Spec {
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1).unwrap()),
        ..Default::default()
    });
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "retention-target", &[19, 23]);
    let initial = view(&harness, &target);
    let before = free_bytes(&harness);
    let allocated_before = harness.runtime.allocate_calls();
    assert!(matches!(
        harness.root.project_checkpoint_capture(
            &initial,
            &initial.initial_state(),
            &harness.fixture.plan,
            0,
            1,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::LogicalCapacity)
    ));
    assert!(matches!(
        reaper
            .try_capture_sequence_checkpoint(
                &harness.fixture.plan,
                &harness.root.trusted_runtime_binding().unwrap(),
                Arc::clone(&harness.session),
                Arc::clone(&lane),
            )
            .unwrap(),
        NativeCheckpointStart::Skipped(CheckpointAccessSkipReason::Retention(_))
    ));
    assert_eq!(free_bytes(&harness), before);
    assert_eq!(harness.runtime.allocate_calls(), allocated_before);
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .checkpoint_retained_bytes()
            .unwrap(),
        0
    );
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_restore_rejects_completed_zero_initialization_without_a_model_frame() {
    use crate::vnext::resource::backing_initialization::PreparedBackingInitializations;
    let harness = projection_harness(Default::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "initialized-zero-target", &[19, 23]);
    let guard = match target
        .try_prepare_state_transfer(
            SequenceStateTransferKind::RestoreWrite,
            target.resources().backing_generation().unwrap(),
        )
        .unwrap()
    {
        SequenceStateTransferPreparation::Prepared(guard) => guard,
        _ => panic!("the fresh target must reserve"),
    };
    let byte_plan = harness.fixture.plan.checkpoint_byte_plan(1).unwrap();
    let mut zeros = PreparedBackingInitializations::prepare_restore(
        &guard,
        harness.fixture.layout(),
        &byte_plan,
        "projection-initialized-target",
    )
    .unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(0);
    assert!(
        zeros
            .encode_restore(&guard, harness.runtime.as_ref(), &mut commands)
            .unwrap()
            > 0
    );
    // Drive the existing initialization participant's complete terminal protocol,
    // as its own contract tests do. No model Step or retired frame is created.
    zeros.mark_in_flight().unwrap();
    zeros.finish(true).unwrap();
    assert!(guard
        .backing()
        .backing_slices()
        .iter()
        .filter_map(|slice| slice.initialization_status().unwrap())
        .all(|status| status == BackingInitializationStatus::Initialized));
    drop(zeros);
    drop(guard);
    let initial = view(&harness, &target);
    let capture = known(harness.root.project_checkpoint_capture(
        &initial,
        &initial.initial_state(),
        &harness.fixture.plan,
        0,
        1,
        &mut || true,
    ));
    assert!(matches!(
        harness.root.project_checkpoint_restore(
            &initial,
            &capture.state,
            &harness.fixture.plan,
            &capture.checkpoint,
            1,
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::InvalidInput)
    ));
    let checkpoint = observed_capture(&harness, &lane, &reaper);
    let submitted = harness.runtime.submitted_timing_modes.lock().unwrap().len();
    let result = reaper
        .try_restore_sequence_checkpoint(
            &harness.fixture.plan,
            Arc::clone(&target),
            &checkpoint,
            Arc::from([19, 23]),
            Arc::clone(&lane),
        )
        .unwrap();
    // Native pre-submit failures are an explicit outcome, not an outer Err.
    // Require the actual pending-initialization rejection and no device submit.
    match result {
        NativeCheckpointStart::NotSubmitted(VNextError::InvalidExecutionPlan { reason }) => {
            assert_eq!(
                reason,
                "fresh restore requires pending target initialization cells"
            );
        }
        NativeCheckpointStart::NotSubmitted(error) => {
            panic!("restore rejected for a different reason: {error}")
        }
        NativeCheckpointStart::Skipped(reason) => {
            panic!("restore must reject initialization, not skip: {reason:?}")
        }
        _ => panic!("native restore must reject initialized backing before submit"),
    }
    assert_eq!(
        harness.runtime.submitted_timing_modes.lock().unwrap().len(),
        submitted
    );
    let after_rejection = view(&harness, &target);
    assert_eq!(
        after_rejection.participants()[1].authority(),
        initial.participants()[1].authority()
    );
    assert_eq!(
        after_rejection.participants()[1].covered_tokens(),
        initial.participants()[1].covered_tokens()
    );
    assert!(after_rejection.participants()[1].matches_session_identity(target.as_ref()));
    assert!(after_rejection.participants()[1]
        .completed_checkpoint_boundary()
        .is_none());
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}
