//! Real resource ownership and frame lifecycle, without pretending to execute
//! device work. The CUDA fixtures cover actual completion and committed state.
use super::*;
use crate::vnext::{DecodeSessionPrepareDecision, DecodeSessionReservation};

fn setup() -> (
    Harness,
    Arc<ExecutionLane<TestRuntime>>,
    ReusableExecutionBucketSpec,
) {
    let catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '7',
        1,
        256,
        TestDemand::Tokens,
    );
    let (memory, bucket) = reusable_step_memory_plan(catalog.pool_id.clone());
    let runtime = new_runtime(&catalog, 256);
    let harness = harness_with_nodes_and_reusable(
        runtime,
        catalog,
        256,
        false,
        Arc::from(vec![PlanNode::resource_test_node(
            NodeId::new("node/decode-session").unwrap(),
        )]),
        Some(memory),
    );
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    (harness, lane, bucket)
}

fn request(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    bucket: &ReusableExecutionBucketSpec,
    token: u32,
) -> StepResourceAdmissionRequest {
    StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![TokenSpanWork::from_token_ids(&[token], 0..1).unwrap()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
    .with_reusable_execution_bucket(bucket.bucket_id().clone())
}

fn prepare(
    reservation: &mut DecodeSessionReservation<TestRuntime>,
    request: &StepResourceAdmissionRequest,
) -> (
    Arc<StepResourceLease<TestRuntime>>,
    PreparedStepSubmissionWave<TestRuntime>,
) {
    match reservation.try_prepare_step(request).unwrap() {
        DecodeSessionPrepareDecision::Prepared { step, wave } => (step, wave),
        _ => panic!("eligible current resource request must prepare from its reservation"),
    }
}

fn assert_unpoisoned(harness: &Harness) {
    assert!(!harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap()
        .poisoned());
    assert!(
        !harness.root.dynamic_pools.pools[&harness.pool_ids[0]]
            .state
            .lock()
            .unwrap()
            .poisoned
    );
}

#[test]
fn decode_session_idle_capacity_retains_parents_until_close() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence(&harness.root, "session-idle-parent");
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 3);
    let reservation = batch
        .try_reserve_decode_session(&req, &lane)
        .unwrap()
        .unwrap();
    assert!(reservation.is_idle());
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    let held = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(held.active_sequences(), 1);
    assert_eq!(held.active_requests(), 1);
    assert!(held.domains().iter().any(|domain| domain.used().get() > 0));
    assert!(!harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    drop(reservation);
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(released.active_sequences(), 0);
    assert_eq!(released.active_requests(), 0);
    assert!(released
        .domains()
        .iter()
        .all(|domain| domain.used().get() == 0));
    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert_unpoisoned(&harness);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn decode_session_fresh_steps_retire_and_release_busy_without_retaining_old_step() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence(&harness.root, "session-fresh-steps");
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let first_request = request(&batch, &bucket, 3);
    // Exercise the real pristine rollback before reserving the same slot.
    let ordinary = match batch.try_begin_step(first_request.clone(), &lane).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident ordinary step must admit"),
    };
    ordinary.try_rollback_unsubmitted().unwrap();
    let mut reservation = batch
        .try_reserve_decode_session(&first_request, &lane)
        .unwrap()
        .unwrap();
    let mut previous_frame = None;
    for token in [3, 9] {
        let req = request(&batch, &bucket, token);
        let (step, wave) = prepare(&mut reservation, &req);
        assert!(!reservation.is_idle());
        assert!(reservation.try_prepare_step(&req).is_err());
        assert_eq!(step.work_shape(), req.work_shape());
        let weak_step = Arc::downgrade(&step);
        // No submission occurred: dropping the prepared wave rolls its flights
        // back. Normal frame retirement here is not a successful GPU receipt.
        drop(wave);
        let retirement = step.try_retire_normal().unwrap();
        let frame = retirement.participants()[0].assignment().frame_id();
        if let Some(previous) = previous_frame {
            assert!(frame > previous);
        }
        previous_frame = Some(frame);
        assert!(weak_step.upgrade().is_none());
        assert!(reservation.is_idle());
    }
    assert_eq!(reservation.prepared_steps(), 2);
    drop(reservation);
    session.try_complete().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert_unpoisoned(&harness);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn decode_session_outer_drop_does_not_release_prepared_step_slot() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence(&harness.root, "session-drop-prepared");
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 4);
    let mut reservation = batch
        .try_reserve_decode_session(&req, &lane)
        .unwrap()
        .unwrap();
    let (step, wave) = prepare(&mut reservation, &req);
    drop(reservation);
    assert!(!harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert!(harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap()
        .domains()
        .iter()
        .any(|domain| domain.used().get() > 0));
    drop(wave);
    assert!(!harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    step.try_retire_normal().unwrap();
    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    session.try_complete().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    assert_unpoisoned(&harness);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn decode_session_same_width_foreign_cohort_falls_back_and_cancel_is_rejected() {
    let (harness, lane, bucket) = setup();
    let first_sequence = admitted_sequence(&harness.root, "session-first");
    let second_sequence = admitted_sequence(&harness.root, "session-replacement");
    let first = first_sequence.open_session().unwrap();
    let second = second_sequence.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&first)]).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second)]).unwrap();
    let first_request = request(&first_batch, &bucket, 1);
    let second_request = request(&second_batch, &bucket, 1);
    let mut reservation = first_batch
        .try_reserve_decode_session(&first_request, &lane)
        .unwrap()
        .unwrap();
    assert!(!reservation.matches_participants(&second_batch));
    assert!(matches!(
        reservation.try_prepare_step(&second_request).unwrap(),
        DecodeSessionPrepareDecision::Fallback
    ));
    assert!(reservation.is_idle());
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    pool.state.lock().unwrap().poisoned = true;
    assert!(reservation.try_prepare_step(&first_request).is_err());
    assert!(reservation.is_idle());
    pool.state.lock().unwrap().poisoned = false;
    first.request_cancel().unwrap();
    assert!(reservation.try_prepare_step(&first_request).is_err());
    // Cancellation may revoke the reservation fail-closed. Either way no new
    // frame can be prepared and dropping it must release the reserved capacity.
    assert!(reservation.try_prepare_step(&first_request).is_err());
    drop(reservation);
    first.try_abort().unwrap();
    // Changed membership must make progress through ordinary fresh admission
    // after the caller releases the old reservation.
    let step = match second_batch.try_begin_step(second_request, &lane).unwrap() {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("released reservation must permit the replacement cohort"),
    };
    step.try_rollback_unsubmitted().unwrap();
    second.request_cancel().unwrap();
    second.try_abort().unwrap();
    drop(first_batch);
    drop(second_batch);
    drop(first);
    drop(second);
    drop(first_sequence);
    drop(second_sequence);
    assert!(harness
        .root
        .dynamic_pools
        .try_reclaim_one_idle_lane_slot()
        .unwrap());
    assert_unpoisoned(&harness);
    drop(lane);
    close_dynamic_test_root(harness.root);
}
