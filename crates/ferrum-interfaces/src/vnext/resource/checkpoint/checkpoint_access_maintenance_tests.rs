//! Public capture retry against initially full foreground state pools. The
//! shared CPU runtime proves resource/terminal ownership, not copied values.

use super::*;
use crate::vnext::{CheckpointCapacityMaintenance, CheckpointCapacityMaintenanceOutcome};

fn full_foreground_harness() -> (RestoreHarness, Vec<Arc<SequenceSession<TestRuntime>>>) {
    let mut spec = checkpoint_fixture::Spec::default();
    // One fixed boundary state makes the ordinary runnable allocation exact;
    // no spare token-prefix capacity can mask the first checkpoint shortage.
    spec.states.remove(0);
    spec.locations.remove(0);
    spec.layouts.remove(0);
    spec.checkpoint_capacity = Some(CheckpointCapacityPolicy::new(1024).unwrap());
    let mut harness = RestoreHarness::with_initial_sequences(spec, Some(3));
    let source = admitted_target(&harness, "maintenance-source", &[19, 31], &[19, 31]);
    harness.session.try_abort_if_quiescent().unwrap();
    harness.session = source;
    let peers = vec![
        harness.new_session("peer-two"),
        harness.new_session("peer-three"),
    ];
    (harness, peers)
}

fn capture_maintenance(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> CheckpointCapacityMaintenance<TestRuntime> {
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let NativeCheckpointStart::CapacityMaintenance {
        reason,
        maintenance,
    } = reaper
        .try_capture_sequence_checkpoint(
            &harness.fixture.plan,
            &binding,
            Arc::clone(&harness.session),
            Arc::clone(lane),
        )
        .unwrap()
    else {
        panic!("full foreground pools must expose an authentic maintenance owner")
    };
    assert!(matches!(
        reason,
        CheckpointAccessSkipReason::CapacityDeferred(_)
    ));
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_checkpoint_budget_released(harness);
    maintenance
}

fn release_peers(peers: Vec<Arc<SequenceSession<TestRuntime>>>) {
    for peer in peers {
        peer.try_abort_if_quiescent().unwrap();
    }
}

#[test]
fn checkpoint_access_first_capture_grows_without_a_spare_execution_slot() {
    let (harness, peers) = full_foreground_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let boundary = reserve_capture(&harness.session)
        .completed_boundary()
        .unwrap();
    let maintenance = capture_maintenance(&harness, &lane, &reaper);
    // The public maintenance owner holds no source gate or copy authority.
    assert!(Arc::ptr_eq(
        &boundary,
        &reserve_capture(&harness.session)
            .completed_boundary()
            .unwrap()
    ));
    assert!(matches!(
        maintenance.try_maintain().unwrap(),
        CheckpointCapacityMaintenanceOutcome::Ready(_)
    ));
    assert_checkpoint_budget_released(&harness);
    let checkpoint = access_capture(&harness, &lane, &reaper);
    assert_eq!(checkpoint.token_prefix(), &[19]);
    assert_eq!(checkpoint.completed_tokens(), boundary.completed_tokens());
    assert!(Arc::ptr_eq(
        &boundary,
        &reserve_capture(&harness.session)
            .completed_boundary()
            .unwrap()
    ));
    drop(checkpoint);
    assert_checkpoint_budget_released(&harness);
    release_peers(peers);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_dropped_maintenance_leaves_source_and_capacity_unchanged() {
    let (harness, peers) = full_foreground_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let boundary = reserve_capture(&harness.session)
        .completed_boundary()
        .unwrap();
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    drop(capture_maintenance(&harness, &lane, &reaper));
    assert_eq!(
        harness
            .root
            .dynamic_pools
            .logical_admission
            .snapshot()
            .unwrap(),
        before
    );
    assert!(Arc::ptr_eq(
        &boundary,
        &reserve_capture(&harness.session)
            .completed_boundary()
            .unwrap()
    ));
    // Dropping did not secretly grow or reserve storage; a new full capture
    // still reports the same real pressure and can be independently abandoned.
    drop(capture_maintenance(&harness, &lane, &reaper));
    release_peers(peers);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn checkpoint_access_maintenance_does_not_authorize_a_cancelled_source() {
    let (harness, peers) = full_foreground_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let maintenance = capture_maintenance(&harness, &lane, &reaper);
    harness.session.request_cancel().unwrap();
    assert!(matches!(
        maintenance.try_maintain().unwrap(),
        CheckpointCapacityMaintenanceOutcome::Ready(_)
    ));
    let binding = harness.root.trusted_runtime_binding().unwrap();
    assert!(reaper
        .try_capture_sequence_checkpoint(
            &harness.fixture.plan,
            &binding,
            Arc::clone(&harness.session),
            Arc::clone(&lane),
        )
        .is_err());
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_checkpoint_budget_released(&harness);
    release_peers(peers);
    drop(binding);
    drop(reaper);
    drop(lane);
    harness.close();
}
