//! Poll work, rather than machine time, bounds a real coalesced-slot capture.
use super::*;

fn capture_poll_count(projections: usize) -> usize {
    let catalog = pool_catalog_with_options(
        linear_profile(),
        AllocationLifetime::Step,
        'b',
        projections,
        256,
        TestDemand::Tokens,
        "activations",
        true,
        StateInitialization::None,
    );
    let (memory, bucket) = reusable_step_memory_plan(catalog.pool_id.clone());
    let harness = harness_with_reusable(new_runtime(&catalog, 256), catalog, 256, memory);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "linear-capture", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let actual = step(&batch, &lane, 1, Some(&bucket));
    assert_eq!(actual.backing_slices().len(), projections);
    for slice in actual.backing_slices() {
        assert!(slice
            .evidence()
            .physical_claim_identity()
            .shares_resource_id_storage(&slice.segment_lease.claim_identity));
    }
    actual.try_retire_normal().unwrap();
    let allocations = harness.runtime.allocate_calls();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let (view, polls) =
        loop {
            let mut polls = 0;
            let captured = harness.root.resource_planning_view_on_lane(
                &[&session],
                &lane,
                ResourcePlanningLimits::default(),
                &mut || {
                    polls += 1;
                    true
                },
            );
            match captured {
                ResourcePlanningAvailability::Known(view) => break (view, polls),
                ResourcePlanningAvailability::Unknown(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                ) if std::time::Instant::now() < deadline => std::thread::yield_now(),
                other => panic!("actual shared-claim capture failed: {other:?}"),
            }
        };
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    session.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
    assert_eq!(
        view.participants().len(),
        1,
        "numeric evidence pins no lease"
    );
    polls
}

#[test]
fn planning_workspace_shared_claim_capture_poll_work_scales_linearly() {
    let small = capture_poll_count(16);
    let large = capture_poll_count(64);
    assert!(large > small, "each real projection still needs validation");
    assert!(
        large <= small * 4,
        "four times as many shared projections must not cause quadratic identity scans: {small} -> {large}"
    );
}
