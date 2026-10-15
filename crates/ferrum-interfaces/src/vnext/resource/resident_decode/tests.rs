//! Real admission, reserved allocations and fresh frame ownership. No test
//! claims GPU completion merely by preparing or dropping a resident use.
use super::*;
use crate::vnext::{CompletionObservation, CompletionReaper, OperationCompletionDisposition};

#[path = "completion_tests.rs"]
mod completion_tests;

pub(super) fn setup() -> (
    Harness,
    Arc<ExecutionLane<TestRuntime>>,
    ReusableExecutionBucketSpec,
) {
    let step = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        '7',
        1,
        256,
        TestDemand::Tokens,
    );
    let sequence = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        '8',
        1,
        256,
        TestDemand::Tokens,
    );
    let (memory, bucket) = reusable_step_memory_plan(step.pool_id.clone());
    let catalog = combine_catalogs(&[step, sequence]);
    let runtime = new_runtime(&catalog, 512);
    runtime
        .physical_binding_tables_enabled
        .store(true, Ordering::Relaxed);
    let harness = harness_with_nodes_and_reusable(
        runtime,
        catalog,
        512,
        false,
        Arc::from(vec![PlanNode::resource_test_node(
            NodeId::new("node/resident-decode").unwrap(),
        )]),
        Some(memory),
    );
    for pool in &harness.pool_ids {
        harness
            .root
            .maintenance_controller
            .grow_pool(pool, 256)
            .unwrap();
    }
    let lane = harness.root.create_execution_lane().unwrap();
    (harness, lane, bucket)
}

pub(super) fn request(
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

pub(super) fn ready(
    reservation: &Arc<ResidentDecodeReservation<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
    request: &StepResourceAdmissionRequest,
) -> (
    ResidentDecodeUse<TestRuntime>,
    ResidentFrameRetirement<TestRuntime>,
) {
    match reservation.try_begin_use(batch, request).unwrap() {
        ResidentDecodeDecision::Ready { use_, retirement } => (use_, retirement),
        _ => panic!(
            "current exact physical table and fresh participant must issue a real resident use"
        ),
    }
}

fn close_session(session: &Arc<SequenceSession<TestRuntime>>) {
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
}

#[test]
fn resident_decode_bootstrap_and_hot_uses_share_allocations_but_not_authority() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-fresh", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let first = request(&batch, &bucket, 3);
    let reservation = batch
        .try_reserve_resident_decode(&first, &lane)
        .unwrap()
        .unwrap();
    assert!(reservation.is_idle());
    let (step, wave) = match reservation.prepare_bootstrap(&first).unwrap() {
        ResidentBootstrapDecision::Prepared { step, wave } => (step, wave),
        _ => panic!("cold bootstrap must use its reserved slots"),
    };
    assert!(reservation.matches_bootstrap(&wave).unwrap());
    let slot = step.claimed_backing().lane_stable_slot_identity();
    assert!(slot.is_some());
    let weak_step = Arc::downgrade(&step);
    let reaper = CompletionReaper::new();
    let completion = completed_boundary_tests::submit_fixture_wave_through_reaper(
        &harness.root,
        std::slice::from_ref(&session),
        &lane,
        wave,
        &reaper,
    );
    assert!(
        matches!(completion.wait().unwrap(), CompletionObservation::Terminal(receipt)
        if receipt.disposition() == &OperationCompletionDisposition::Succeeded)
    );
    drop(completion);
    assert_eq!(reaper.retained_count(), 0);
    drop(reaper);
    step.try_retire_normal().unwrap();
    assert!(
        weak_step.upgrade().is_none(),
        "hot use must not keep the old Step alive"
    );
    let mut prior_step = None;
    let mut prior_fingerprint = None;
    for token in [3, 9] {
        let req = request(&batch, &bucket, token);
        let (use_, retirement) = ready(&reservation, &batch, &req);
        assert_eq!(use_.work_shape().as_ref(), req.work_shape());
        assert_eq!(use_.lane_stable_slot_identity(), slot);
        assert!(use_.belongs_to_reservation(&reservation));
        assert_eq!(use_.sequence_physical_tables().len(), 1);
        assert!(reservation.try_begin_use(&batch, &req).is_err());
        // Unsubmitted rollback may reuse the frame ordinal, but the fresh
        // Step/attempt authority must never be reused.
        let step = use_.batch_step_id();
        if let Some(old) = prior_step {
            assert_ne!(step, old);
        }
        if let Some(old) = prior_fingerprint {
            assert_ne!(use_.fingerprint(), old);
        }
        prior_step = Some(step);
        prior_fingerprint = Some(use_.fingerprint().to_owned());
        let fresh_owner = use_.retention();
        drop(use_);
        let failure = retirement.try_rollback_unsubmitted().unwrap_err();
        assert!(
            !reservation.is_idle(),
            "retained commands/readbacks still own the fresh frame"
        );
        drop(fresh_owner);
        failure
            .into_retirement()
            .try_rollback_unsubmitted()
            .unwrap();
        assert!(reservation.is_idle());
    }
    drop(reservation);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_retry_keeps_program_but_clears_attempt_payload() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-retry", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 5);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (mut use_, retirement) = ready(&reservation, &batch, &req);
    let program = Arc::new(7_u32);
    use_.retain_resident_program(DeviceBufferRetention::plan(Arc::clone(&program)))
        .unwrap();
    use_.retain_resident_program(DeviceBufferRetention::plan(Arc::clone(&program)))
        .unwrap();
    assert!(use_
        .retain_resident_program(DeviceBufferRetention::plan(Arc::new(8_u32)))
        .is_err());
    // The packet owner is a stand-in for an allocation; no bytes or completion
    // are fabricated. Definitely-not-submitted is the only retry reset seam.
    let packet = Arc::new(9_u32);
    let weak_packet = Arc::downgrade(&packet);
    use_.install_terminal_readback_payload(packet);
    let first_attempt = use_.batch_invocation_id();
    use_.begin_dispatch().unwrap();
    use_ = use_.definitely_not_submitted().unwrap();
    assert_ne!(first_attempt, use_.batch_invocation_id());
    assert!(weak_packet.upgrade().is_none());
    assert!(!use_.has_prepared_terminal_readbacks());
    use_.retain_resident_program(DeviceBufferRetention::plan(Arc::clone(&program)))
        .unwrap();
    drop(use_);
    retirement.try_rollback_unsubmitted().unwrap();
    assert_eq!(Arc::strong_count(&program), 1);
    assert!(reservation.is_idle());
    drop(reservation);
    close_session(&session);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_rejects_poison_cancel_and_physical_growth_before_enqueue() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-current", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 5);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let foreign_sequence = admitted_sequence_with_ceiling(&harness.root, "resident-foreign", 4);
    let foreign_session = foreign_sequence.open_session().unwrap();
    let foreign_batch =
        ExecutionBatchParticipants::new(vec![Arc::clone(&foreign_session)]).unwrap();
    let foreign_request = request(&foreign_batch, &bucket, 5);
    assert!(matches!(
        reservation
            .try_begin_use(&foreign_batch, &foreign_request)
            .unwrap(),
        ResidentDecodeDecision::Fallback
    ));
    assert!(reservation.is_idle());
    close_session(&foreign_session);
    drop(foreign_batch);
    drop(foreign_session);
    drop(foreign_sequence);
    let pool = &harness.root.dynamic_pools.pools[&harness.pool_ids[0]];
    pool.state.lock().unwrap().poisoned = true;
    assert!(reservation.try_begin_use(&batch, &req).is_err());
    assert!(reservation.is_idle());
    pool.state.lock().unwrap().poisoned = false;
    let snapshot = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("real larger Sequence physical allocation must be admitted"),
    };
    assert!(matches!(
        reservation.try_begin_use(&batch, &req).unwrap(),
        ResidentDecodeDecision::Fallback
    ));
    assert!(
        reservation.is_idle(),
        "physical mismatch must roll back the fresh unsubmitted frame"
    );
    session.request_cancel().unwrap();
    assert!(reservation.try_begin_use(&batch, &req).is_err());
    assert!(reservation.is_idle());
    drop(reservation);
    drop(snapshot);
    session.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn resident_decode_unknown_retention_and_revoke_never_reopen_capacity_early() {
    let (harness, lane, bucket) = setup();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "resident-unknown", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let req = request(&batch, &bucket, 7);
    let reservation = batch
        .try_reserve_resident_decode(&req, &lane)
        .unwrap()
        .unwrap();
    let (mut use_, retirement) = ready(&reservation, &batch, &req);
    use_.begin_dispatch().unwrap();
    use_.mark_submission_indeterminate();
    let quarantine = use_.retention();
    reservation.revoke();
    assert!(reservation.try_begin_use(&batch, &req).is_err());
    let failure = retirement.try_abort().unwrap_err();
    drop(quarantine);
    drop(use_);
    // Unknown device use cannot be declared an unsubmitted rollback. After
    // the explicit simulated drain drops all owners, abort retires the frame.
    let failure = failure
        .into_retirement()
        .try_rollback_unsubmitted()
        .unwrap_err();
    failure.into_retirement().try_abort().unwrap();
    assert!(
        !reservation.is_idle(),
        "late frame retirement cannot undo revocation"
    );
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
    drop(reservation);
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert_eq!(released.active_sequences(), 0);
    assert_eq!(released.active_requests(), 0);
    drop(lane);
    close_dynamic_test_root(harness.root);
}
