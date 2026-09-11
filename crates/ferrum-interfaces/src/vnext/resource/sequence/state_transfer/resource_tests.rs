use super::*;

fn transfer_harness() -> Harness {
    let catalog = pool_catalog_with_options(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
        "state",
        false,
        StateInitialization::Zero,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 256)
        .unwrap();
    harness
}

fn reserve(
    session: &Arc<SequenceSession<TestRuntime>>,
    kind: SequenceStateTransferKind,
) -> PreparedSequenceStateTransfer<TestRuntime> {
    match session
        .try_prepare_state_transfer(kind, session.resources().backing_generation().unwrap())
        .unwrap()
    {
        SequenceStateTransferPreparation::Prepared(prepared) => prepared,
        _ => panic!("idle resident sequence must reserve state transfer"),
    }
}

#[test]
fn checkpoint_transfer_blocks_real_frame_and_extension_without_changing_capacity() {
    let harness = transfer_harness();
    let lane = harness.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "transfer-frame", 2);
    let session = sequence.open_session().unwrap();
    let initial = sequence.backing_snapshot().unwrap();
    let transfer = reserve(&session, SequenceStateTransferKind::CaptureRead);
    assert_eq!(transfer.kind(), SequenceStateTransferKind::CaptureRead);
    assert!(Arc::ptr_eq(transfer.backing(), &initial));
    assert!(Arc::ptr_eq(transfer.session(), &session));
    assert!(matches!(
        session
            .try_prepare_state_transfer(
                SequenceStateTransferKind::RestoreWrite,
                initial.generation()
            )
            .unwrap(),
        SequenceStateTransferPreparation::Busy
    ));

    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let begin_step = || {
        batch.try_begin_step(
            StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![token_span(1)]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            &lane,
        )
    };
    assert!(begin_step().is_err());
    let extension = || {
        session.try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
    };
    assert!(matches!(
        extension().unwrap(),
        SequenceResourceExtensionDecision::RetryRequired(_)
    ));
    assert_eq!(sequence.backing_generation().unwrap(), initial.generation());
    assert!(!session.resources().is_poisoned());

    drop(transfer);
    let frame = match begin_step().unwrap() {
        StepResourceAdmissionDecision::Admitted(frame) => frame,
        _ => panic!("rolled back reservation must permit the original frame"),
    };
    assert!(matches!(
        session
            .try_prepare_state_transfer(
                SequenceStateTransferKind::CaptureRead,
                initial.generation()
            )
            .unwrap(),
        SequenceStateTransferPreparation::Busy
    ));
    frame.try_retire_normal().unwrap();
    let extended = match extension().unwrap() {
        SequenceResourceExtensionDecision::Extended(extended) => extended,
        _ => panic!("retired frame and transfer must permit extension"),
    };
    assert!(matches!(
        session
            .try_prepare_state_transfer(
                SequenceStateTransferKind::CaptureRead,
                initial.generation()
            )
            .unwrap(),
        SequenceStateTransferPreparation::StaleBacking
    ));
    assert_eq!(extended.generation().get(), initial.generation().get() + 1);
    session.try_complete().unwrap();
    drop(extended);
    drop(initial);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_transfer_cancellation_waits_for_preparation_and_does_not_reopen_target() {
    let harness = transfer_harness();
    let sequence = admitted_sequence(&harness.root, "transfer-cancel");
    let session = sequence.open_session().unwrap();
    let generation = sequence.backing_generation().unwrap();
    let transfer = reserve(&session, SequenceStateTransferKind::RestoreWrite);
    assert!(session.try_abort_if_quiescent().is_err());
    let cancelled = session.request_cancel().unwrap();
    assert!(cancelled.state_transfer_pending());
    assert!(cancelled.active_frame().is_none());
    assert_eq!(cancelled.participant_flights(), 0);
    assert!(session.try_abort().is_err());
    drop(transfer);
    assert!(!session.request_cancel().unwrap().state_transfer_pending());
    assert!(session
        .try_prepare_state_transfer(SequenceStateTransferKind::RestoreWrite, generation)
        .is_err());
    session.try_abort().unwrap();
    drop(session);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}

#[test]
fn checkpoint_transfer_preparation_retains_exact_resources_after_external_owner_drop() {
    let harness = transfer_harness();
    let sequence = admitted_sequence(&harness.root, "transfer-retention");
    let session = sequence.open_session().unwrap();
    let weak_sequence = Arc::downgrade(&sequence);
    let weak_session = Arc::downgrade(&session);
    let transfer = reserve(&session, SequenceStateTransferKind::CaptureRead);
    session.request_cancel().unwrap();
    drop(session);
    drop(sequence);
    assert!(weak_sequence.upgrade().is_some());
    assert!(weak_session.upgrade().is_some());
    assert!(transfer.session().try_abort().is_err());
    let retained_session = Arc::clone(transfer.session());
    drop(transfer);
    retained_session.try_abort().unwrap();
    drop(retained_session);
    assert!(weak_sequence.upgrade().is_none());
    assert!(weak_session.upgrade().is_none());
    close_dynamic_test_root(harness.root);
}
