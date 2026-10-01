//! Real CPU resource fixtures: sharing changes storage, never reservations.
use super::*;

#[test]
fn planning_pool_cow_copies_only_the_touched_pool_and_replays_from_capture() {
    let growing = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let fixed = pool_catalog(
        linear_profile(),
        AllocationLifetime::Sequence,
        'b',
        1,
        256,
        TestDemand::Fixed,
    );
    let growing_id = growing.pool_id.clone();
    let fixed_id = fixed.pool_id.clone();
    let catalog = combine_catalogs(&[growing, fixed]);
    let runtime = new_runtime(&catalog, 512);
    let harness = harness(Arc::clone(&runtime), catalog, 512, false);
    for id in &harness.pool_ids {
        harness
            .root
            .maintenance_controller
            .grow_pool(id, 256)
            .unwrap();
    }
    let sequence = admitted_sequence_with_ceiling(&harness.root, "cow-branches", 4);
    let session = sequence.open_session().unwrap();
    let captured = view(&harness.root, &[&session]);
    let initial = captured.initial_state();
    let cloned = initial.clone();
    for id in &harness.pool_ids {
        assert!(initial.shares_pool_allocator(&cloned, id));
        assert!(initial.shares_pool_allocator(&captured.initial_state(), id));
    }
    let physical_allocations = runtime.allocate_calls();
    let left = known(harness.root.project_resource_wave(
        &captured,
        &initial,
        &[row(0, 1, 1)],
        &mut || true,
    ));
    let right = known(harness.root.project_resource_wave(
        &captured,
        &initial,
        &[row(0, 1, 2)],
        &mut || true,
    ));
    assert!(!initial.shares_pool_allocator(&left.state, &growing_id));
    assert!(!initial.shares_pool_allocator(&right.state, &growing_id));
    assert!(!left.state.shares_pool_allocator(&right.state, &growing_id));
    for branch in [&left.state, &right.state] {
        assert!(initial.shares_pool_allocator(branch, &fixed_id));
    }
    assert_eq!(initial.covered_tokens(0), Some(1));
    assert_eq!(left.state.covered_tokens(0), Some(2));
    assert_eq!(right.state.covered_tokens(0), Some(3));
    let replay_root = captured.initial_state();
    assert!(initial
        .same_future_state(&replay_root, &mut || true)
        .unwrap());
    let replay = known(harness.root.project_resource_wave(
        &captured,
        &replay_root,
        &[row(0, 1, 1)],
        &mut || true,
    ));
    assert!(left
        .state
        .same_future_state(&replay.state, &mut || true)
        .unwrap());
    assert_eq!(left.domains, replay.domains);
    assert!(captured.same_live_evidence(&view(&harness.root, &[&session])));
    assert_eq!(runtime.allocate_calls(), physical_allocations);
    // The same extension performed physically must still invalidate the old
    // capture and agree with its predicted committed coverage/logical demand.
    session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap();
    let fresh = view(&harness.root, &[&session]);
    assert!(!captured.same_live_evidence(&fresh));
    assert_eq!(
        left.state.covered_tokens(0),
        Some(fresh.participants()[0].covered_tokens())
    );
    for demand in &left.domains {
        assert_eq!(
            left.state.available_in_domain(demand.domain),
            fresh.initial_state().available_in_domain(demand.domain)
        );
    }
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn planning_pool_cow_contiguous_failure_and_transient_release_leave_capture_unchanged() {
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
    let pool_id = harness.pool_ids[0].clone();
    for _ in 0..2 {
        harness
            .root
            .maintenance_controller
            .grow_pool(&pool_id, 64)
            .unwrap();
    }
    let sequence = admitted_sequence_with_ceiling(&harness.root, "cow-fragmented", 4);
    let session = sequence.open_session().unwrap();
    let captured = view(&harness.root, &[&session]);
    let initial = captured.initial_state();
    let before = runtime.allocate_calls();
    assert!(matches!(
        harness
            .root
            .project_resource_wave(&captured, &initial, &[row(0, 0, 2)], &mut || true,),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
    ));
    assert!(initial
        .same_future_state(&captured.initial_state(), &mut || true)
        .unwrap());
    assert!(initial.shares_pool_allocator(&captured.initial_state(), &pool_id));
    let first = known(harness.root.project_resource_wave(
        &captured,
        &initial,
        &[row(0, 0, 1)],
        &mut || true,
    ));
    assert!(!initial.shares_pool_allocator(&first.state, &pool_id));
    let replay = known(harness.root.project_resource_wave(
        &captured,
        &captured.initial_state(),
        &[row(0, 0, 1)],
        &mut || true,
    ));
    assert!(first
        .state
        .same_future_state(&replay.state, &mut || true)
        .unwrap());
    assert_eq!(first.domains[0].persistent_bytes, 0);
    assert_eq!(
        first.state.available_in_domain(first.domains[0].domain),
        initial.available_in_domain(first.domains[0].domain)
    );
    // The transient extent was released and the next whole wave can reserve
    // it again. Allocation search counters remain branch-local evidence.
    let next = known(harness.root.project_resource_wave(
        &captured,
        &first.state,
        &[row(0, 1, 1)],
        &mut || true,
    ));
    assert_eq!(next.domains, first.domains);
    assert!(captured.same_live_evidence(&view(&harness.root, &[&session])));
    assert_eq!(runtime.allocate_calls(), before);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}
