use super::*;

pub(crate) fn execute_sequence(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
    frames: u64,
) -> SequenceEvidence {
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let plan = resolved.execution_plan();
    let mut submissions = Vec::new();
    let mut completions = Vec::new();
    let mut sequence = 3_u64;
    let mut invocation = 1_u64;
    for frame in 1..=frames {
        let step = begin_single_participant_step_on_lane(&batch, &lane);
        let frame_id = step.participant_frames().next().unwrap().frame_id();
        assert_eq!(frame_id.get(), frame);
        sequence += 1;
        for node_index in 0..plan.payload().nodes().len() {
            sequence += 1;
            let operation_event = node_event(
                plan,
                &active,
                active.run_id(),
                active.request_id(),
                sequence,
                frame,
                invocation,
                node_index,
                ExecutionEventKind::OperationSubmitted,
            );
            let node = &plan.payload().nodes()[node_index];
            let provider = registry.bind(resolved, node.id()).unwrap();
            let completion = encode_and_submit_single(
                &provider,
                resolved,
                operation_event.identity(),
                &frame_id,
                &NodeInvocationId::try_from(invocation).unwrap(),
                node.id(),
                &active,
                admit_single_participant_invocation(plan_resources, &step, node.id()),
                &lane,
                &reaper,
            )
            .unwrap();
            let submission = completion.receipt().clone();
            let completion = match completion.poll().unwrap() {
                CompletionObservation::Terminal(receipt) => receipt,
                _ => panic!("event fixture operation did not reach a terminal fence"),
            };
            assert_eq!(completion.submission(), &submission);
            submissions.push(submission);
            completions.push(completion);
            sequence += 2;
            invocation += 1;
        }
        step.try_retire_normal().unwrap();
        sequence += 1;
    }
    assert_eq!(reaper.retained_count(), 0);
    let completion_receipt = session.try_complete().unwrap();
    let completed =
        TrustedCompletedSequenceBinding::from_session_receipt(&completion_receipt, &active)
            .unwrap();
    SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    }
}

pub(crate) fn execute_failure_then_complete(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
    fail_fence: bool,
) -> FailureSequenceEvidence {
    let plan = resolved.execution_plan();
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let step = begin_single_participant_step_on_lane(&batch, &lane);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let reaper = CompletionReaper::new();
    let operation_event = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let provider = registry
        .bind(resolved, plan.payload().nodes()[0].id())
        .unwrap();
    if fail_fence {
        runtime.fail_next_fence();
    }
    let submission = encode_and_submit_single(
        &provider,
        resolved,
        operation_event.identity(),
        &frame_id,
        &NodeInvocationId::try_from(1).unwrap(),
        plan.payload().nodes()[0].id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, plan.payload().nodes()[0].id()),
        &lane,
        &reaper,
    )
    .unwrap();
    let submitted_operation = submission.receipt().clone();
    let operation_completion = match submission.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        _ => panic!("event failure fixture operation did not reach a terminal fence"),
    };
    let failure = match operation_completion.disposition() {
        OperationCompletionDisposition::FailedButQuiescent(failures) => {
            failures.first().expect("one participant failure").clone()
        }
        OperationCompletionDisposition::Succeeded => IdentifiedFailure::new(
            submitted_operation.participants()[0].identity().clone(),
            FailureEnvelope::new(
                FailureDomain::Device,
                "synthetic_success_reversal",
                "successful fence was incorrectly reported as failed",
                false,
            )
            .unwrap(),
        )
        .unwrap(),
        disposition => panic!("event failure fixture received {disposition:?}"),
    };
    assert_eq!(reaper.retained_count(), 0);
    step.try_retire_normal().unwrap();
    let completion_receipt = session.try_complete().unwrap();
    let completed =
        TrustedCompletedSequenceBinding::from_session_receipt(&completion_receipt, &active)
            .unwrap();
    let failure_event = operation_failure_event(&failure, 6);
    let journal = vec![
        accepted_event(active.run_id(), active.request_id()),
        plan_event(plan, active.run_id(), active.request_id()),
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
        operation_event,
        failure_event,
        sequence_completed_event(plan, &active, &completed, 7),
        request_failed_terminal_event(plan, &active, Some(&completed), None, &failure, 8),
    ];
    FailureSequenceEvidence {
        active,
        completed: Some(completed),
        aborted: None,
        submissions: vec![submitted_operation],
        completions: vec![operation_completion],
        journal,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_terminal_failure_then_complete(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
    mode: ReplayTerminalFixtureMode,
) -> ReplayTerminalFailureEvidence {
    let plan = resolved.execution_plan();
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let step = begin_single_participant_step_on_lane(&batch, &lane);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let reaper = CompletionReaper::new();
    let had_submission_fence = !matches!(
        mode,
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained
    );
    let operation_sequence = if had_submission_fence { 5 } else { 4 };
    let operation_event = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        operation_sequence,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let provider = registry
        .bind(resolved, plan.payload().nodes()[0].id())
        .unwrap();
    match mode {
        ReplayTerminalFixtureMode::ContractFailed => runtime.contract_fail_next_fence(),
        ReplayTerminalFixtureMode::Drained | ReplayTerminalFixtureMode::Quarantined => {
            runtime.make_next_fence_indeterminate();
        }
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained => runtime.panic_next_submit(),
    }
    let dispatch = suppress_expected_panic_hook(|| {
        encode_and_submit_single(
            &provider,
            resolved,
            operation_event.identity(),
            &frame_id,
            &NodeInvocationId::try_from(1).unwrap(),
            plan.payload().nodes()[0].id(),
            &active,
            admit_single_participant_invocation(
                plan_resources,
                &step,
                plan.payload().nodes()[0].id(),
            ),
            &lane,
            &reaper,
        )
    });

    let mut submissions = Vec::new();
    let mut completions = Vec::new();
    let mut drains = Vec::new();
    let mut quarantines = Vec::new();
    let terminal_identity = match mode {
        ReplayTerminalFixtureMode::ContractFailed => {
            let handle = dispatch.unwrap();
            submissions.push(handle.receipt().clone());
            let receipt = match handle.poll().unwrap() {
                CompletionObservation::Terminal(receipt) => receipt,
                _ => panic!("contract terminal fixture did not complete"),
            };
            assert!(matches!(
                receipt.disposition(),
                OperationCompletionDisposition::ContractFailedButQuiescent(_)
            ));
            let identity = receipt.submission().participants()[0].identity().clone();
            completions.push(receipt);
            runtime.reset_stream_failure();
            identity
        }
        ReplayTerminalFixtureMode::Drained | ReplayTerminalFixtureMode::Quarantined => {
            let handle = dispatch.unwrap();
            let slot_id = handle.slot_id();
            submissions.push(handle.receipt().clone());
            assert!(matches!(
                handle.poll().unwrap(),
                CompletionObservation::Indeterminate(_)
            ));
            assert!(matches!(
                handle.wait().unwrap(),
                CompletionObservation::Indeterminate(_)
            ));
            if matches!(mode, ReplayTerminalFixtureMode::Quarantined) {
                runtime.set_synchronize_fails(true);
            }
            match reaper.recover_slot_by_draining_lane(slot_id).unwrap() {
                CompletionRecoveryOutcome::Drained(receipt) => {
                    assert!(matches!(mode, ReplayTerminalFixtureMode::Drained));
                    drains.push(receipt);
                }
                CompletionRecoveryOutcome::Quarantined(receipt) => {
                    assert!(matches!(mode, ReplayTerminalFixtureMode::Quarantined));
                    quarantines.push(receipt);
                    runtime.set_synchronize_fails(false);
                    assert!(matches!(
                        reaper.recover_slot_by_draining_lane(slot_id).unwrap(),
                        CompletionRecoveryOutcome::Drained(_)
                    ));
                }
            }
            submissions[0].participants()[0].identity().clone()
        }
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained => {
            let recovery = match dispatch {
                Err(OperationDispatchError::SubmissionIndeterminate { recovery }) => recovery,
                _ => panic!("submit panic fixture did not retain recovery authority"),
            };
            let receipt = match recovery.recover_by_draining_lane().unwrap() {
                CompletionRecoveryOutcome::Drained(receipt) => receipt,
                CompletionRecoveryOutcome::Quarantined(_) => {
                    panic!("submit panic fixture unexpectedly quarantined")
                }
            };
            let identity = receipt.batch_identity().participants()[0]
                .identity()
                .clone();
            drains.push(receipt);
            identity
        }
    };
    runtime.set_synchronize_fails(false);
    runtime.reset_stream_failure();
    assert_eq!(reaper.retained_count(), 0);
    step.try_retire_normal().unwrap();
    let completion_receipt = session.try_complete().unwrap();
    let completed =
        TrustedCompletedSequenceBinding::from_session_receipt(&completion_receipt, &active)
            .unwrap();
    let failure = IdentifiedFailure::new(
        terminal_identity,
        FailureEnvelope::new(
            FailureDomain::Device,
            match mode {
                ReplayTerminalFixtureMode::ContractFailed => "contract_terminal_failure",
                ReplayTerminalFixtureMode::Drained => "drained_terminal_failure",
                ReplayTerminalFixtureMode::Quarantined => "quarantined_terminal_failure",
                ReplayTerminalFixtureMode::SubmissionIndeterminateDrained => {
                    "submission_indeterminate_terminal_failure"
                }
            },
            "operation did not produce a successful replay terminal",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let failure_sequence = if had_submission_fence { 6 } else { 5 };
    let sequence_disposition = failure_sequence + 1;
    let request_terminal = failure_sequence + 2;
    let mut journal = vec![
        accepted_event(active.run_id(), active.request_id()),
        plan_event(plan, active.run_id(), active.request_id()),
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
    ];
    if had_submission_fence {
        journal.push(operation_event);
    }
    journal.push(operation_failure_event(&failure, failure_sequence));
    journal.push(sequence_completed_event(
        plan,
        &active,
        &completed,
        sequence_disposition,
    ));
    journal.push(request_failed_terminal_event(
        plan,
        &active,
        Some(&completed),
        None,
        &failure,
        request_terminal,
    ));
    ReplayTerminalFailureEvidence {
        sequence: FailureSequenceEvidence {
            active,
            completed: Some(completed),
            aborted: None,
            submissions,
            completions,
            journal,
        },
        drains,
        quarantines,
    }
}

pub(crate) fn emit_pre_active_prefix(
    emitter: &mut ExecutionEventEmitter<'_>,
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
) {
    let accepted = accepted_event(active.run_id(), active.request_id());
    emitter
        .emit(
            accepted.clone(),
            &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        )
        .unwrap();
    let planned = plan_event(plan, active.run_id(), active.request_id());
    emitter
        .emit(
            planned.clone(),
            &TrustedExecutionEventContext::bound(
                active.run_id(),
                active.request_id(),
                &TrustedExecutionTopology::from_plan(plan).unwrap(),
            ),
        )
        .unwrap();
}

pub(crate) fn live_witness_emitter_contract(
    passed: &mut usize,
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) {
    let plan = resolved.execution_plan();
    let topology = TrustedExecutionTopology::from_plan(plan).unwrap();

    let resources = logical_resources(plan_resources, "run.emitter.live", "request.emitter.live");
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let step = begin_single_participant_step_on_lane(&batch, &lane);
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                frame.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_ok()
            && emitter.cursor().last_sequence() == 3
            && sink.kinds.lock().unwrap().last() == Some(&ExecutionEventKind::FrameStarted),
    );
    step.try_retire_normal().unwrap();
    session.try_complete().unwrap();
    drop(emitter);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.completed-stale",
        "request.emitter.completed-stale",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step_on_lane(&batch, &lane);
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    step.try_retire_normal().unwrap();
    session.try_complete().unwrap();
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                frame.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && !emitter.sink_failed()
            && sink.kinds.lock().unwrap().len() == 2,
    );
    drop(emitter);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.aborted-stale",
        "request.emitter.aborted-stale",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                frame.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && sink.kinds.lock().unwrap().len() == 2,
    );
    drop(emitter);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.dropped-stale",
        "request.emitter.dropped-stale",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    drop(session);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                frame.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && sink.kinds.lock().unwrap().len() == 2,
    );
    drop(emitter);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.cancel-progress",
        "request.emitter.cancel-progress",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step_on_lane(&batch, &lane);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    for event in [
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
    ] {
        emitter
            .emit(
                event.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .unwrap();
    }
    let reaper = CompletionReaper::new();
    let first_operation = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let first_node = &plan.payload().nodes()[0];
    let first_provider = registry.bind(resolved, first_node.id()).unwrap();
    let first_handle = encode_and_submit_single(
        &first_provider,
        resolved,
        first_operation.identity(),
        &frame_id,
        &NodeInvocationId::try_from(1).unwrap(),
        first_node.id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, first_node.id()),
        &lane,
        &reaper,
    )
    .unwrap();
    emitter
        .emit(
            first_operation.clone(),
            &TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                first_handle.receipt(),
            ),
        )
        .unwrap();
    let first_completion = match first_handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("first node did not complete: {other:?}"),
    };
    let first_retired = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        6,
        1,
        1,
        0,
        ExecutionEventKind::NodeRetired,
    );
    emitter
        .emit(
            first_retired.clone(),
            &TrustedExecutionEventContext::node_retired(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &first_completion.participants()[0],
            ),
        )
        .unwrap();
    let second_started = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        7,
        1,
        2,
        1,
        ExecutionEventKind::NodeStarted,
    );
    emitter
        .emit(
            second_started.clone(),
            &TrustedExecutionEventContext::active(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
            ),
        )
        .unwrap();
    let second_operation = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        8,
        1,
        2,
        1,
        ExecutionEventKind::OperationSubmitted,
    );
    let second_node = &plan.payload().nodes()[1];
    let second_provider = registry.bind(resolved, second_node.id()).unwrap();
    let second_handle = encode_and_submit_single(
        &second_provider,
        resolved,
        second_operation.identity(),
        &frame_id,
        &NodeInvocationId::try_from(2).unwrap(),
        second_node.id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, second_node.id()),
        &lane,
        &reaper,
    )
    .unwrap();
    session.request_cancel().unwrap();
    emitter
        .emit(
            second_operation.clone(),
            &TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                second_handle.receipt(),
            ),
        )
        .unwrap();
    let second_completion = match second_handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("second node did not complete: {other:?}"),
    };
    let second_retired = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        9,
        1,
        2,
        1,
        ExecutionEventKind::NodeRetired,
    );
    emitter
        .emit(
            second_retired.clone(),
            &TrustedExecutionEventContext::node_retired(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &second_completion.participants()[0],
            ),
        )
        .unwrap();
    let frame_completed = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        10,
        1,
        ExecutionEventKind::FrameCompleted,
    );
    emitter
        .emit(
            frame_completed.clone(),
            &TrustedExecutionEventContext::active(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
            ),
        )
        .unwrap();
    check(
        passed,
        emitter.cursor().last_sequence() == 10
            && sink.kinds.lock().unwrap().last() == Some(&ExecutionEventKind::FrameCompleted),
    );
    step.try_abort().unwrap();
    session.try_abort().unwrap();
    drop(emitter);
    drop(second_handle);
    drop(first_handle);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.cancel-start",
        "request.emitter.cancel-start",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(plan_resources, &batch);
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    emitter
        .emit(
            frame.clone(),
            &TrustedExecutionEventContext::active(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
            ),
        )
        .unwrap();
    session.request_cancel().unwrap();
    let node = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        4,
        1,
        1,
        0,
        ExecutionEventKind::NodeStarted,
    );
    check(
        passed,
        emitter
            .emit(
                node.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 3
            && sink.kinds.lock().unwrap().last() == Some(&ExecutionEventKind::FrameStarted),
    );
    step.try_abort().unwrap();
    session.try_abort().unwrap();
    drop(emitter);
    drop(batch);
    drop(session);
    drop(resources);

    let resources = logical_resources(
        plan_resources,
        "run.emitter.legacy-rejected",
        "request.emitter.legacy-rejected",
    );
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let active = TrustedActiveSequenceBinding::from_permit(&permit).unwrap();
    let sink = RecordingSink::default();
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emit_pre_active_prefix(&mut emitter, plan, &active);
    let frame = frame_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        1,
        ExecutionEventKind::FrameStarted,
    );
    check(
        passed,
        emitter
            .emit(
                frame.clone(),
                &TrustedExecutionEventContext::active(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                ),
            )
            .is_err()
            && emitter.cursor().last_sequence() == 2
            && sink.kinds.lock().unwrap().len() == 2,
    );
    let _legacy_completion = permit.synchronize().unwrap().complete().unwrap();
    drop(emitter);
    drop(stream);
    drop(resources);
}

pub(crate) fn execute_failure_then_abort(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    runtime: &Arc<TestRuntime>,
    pool_journal: Vec<ResourcePoolEvent>,
    pool_evidence: ResourcePoolEvidence,
    resolved: &ResolvedModelPlan,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    run: &str,
    request: &str,
) -> (
    FailureSequenceEvidence,
    ResourcePoolEvidence,
    Vec<ResourcePoolEvent>,
) {
    let plan = resolved.execution_plan();
    let resources = logical_resources(plan_resources, run, request);
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = ExecutionLane::create(Arc::clone(runtime)).unwrap();
    let step = begin_single_participant_step_on_lane(&batch, &lane);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let reaper = CompletionReaper::new();
    let operation_event = node_event(
        plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::OperationSubmitted,
    );
    let provider = registry
        .bind(resolved, plan.payload().nodes()[0].id())
        .unwrap();
    runtime.fail_next_fence();
    let submission = encode_and_submit_single(
        &provider,
        resolved,
        operation_event.identity(),
        &frame_id,
        &NodeInvocationId::try_from(1).unwrap(),
        plan.payload().nodes()[0].id(),
        &active,
        admit_single_participant_invocation(plan_resources, &step, plan.payload().nodes()[0].id()),
        &lane,
        &reaper,
    )
    .unwrap();
    let submitted_operation = submission.receipt().clone();
    let operation_completion = match submission.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        _ => panic!("event abort fixture operation did not reach a terminal fence"),
    };
    let failure = match operation_completion.disposition() {
        OperationCompletionDisposition::FailedButQuiescent(failures) => {
            failures.first().expect("one participant failure").clone()
        }
        disposition => panic!("event abort fixture received {disposition:?}"),
    };
    assert_eq!(reaper.retained_count(), 0);
    let failure_event = operation_failure_event(&failure, 6);
    step.try_abort().unwrap();
    let abort_receipt = session.try_abort().unwrap();
    let aborted =
        TrustedAbortedSequenceBinding::from_session_receipt(&abort_receipt, &active).unwrap();
    let journal = vec![
        accepted_event(active.run_id(), active.request_id()),
        plan_event(plan, active.run_id(), active.request_id()),
        frame_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            3,
            1,
            ExecutionEventKind::FrameStarted,
        ),
        node_event(
            plan,
            &active,
            active.run_id(),
            active.request_id(),
            4,
            1,
            1,
            0,
            ExecutionEventKind::NodeStarted,
        ),
        operation_event,
        failure_event,
        sequence_aborted_event(plan, &active, &aborted, 7),
        request_failed_terminal_event(plan, &active, None, Some(&aborted), &failure, 8),
    ];
    (
        FailureSequenceEvidence {
            active,
            completed: None,
            aborted: Some(aborted),
            submissions: vec![submitted_operation],
            completions: vec![operation_completion],
            journal,
        },
        pool_evidence,
        pool_journal,
    )
}

pub(crate) fn failure_recovery_pair(
    plan: &ExecutionPlan,
    topology: &TrustedExecutionTopology,
    suffix: &str,
) -> (
    ResourcePoolEvidence,
    ResourcePoolEvent,
    ResourcePoolEvent,
    ResourcePoolEvent,
    ResourceFailureReceipt,
    ResourceFailureReceipt,
) {
    let (transaction, _, _) = transaction(
        plan,
        &format!("run.failure.{suffix}"),
        &format!("transaction.failure.{suffix}"),
        &format!("request.failure.{suffix}"),
        CommitBehavior::InvalidFirst,
    );
    let reserved = transaction.reserve().unwrap();
    let evidence =
        ResourcePoolEvidence::from_external(topology, reserved.admission(), reserved.identity())
            .unwrap();
    let opened = ResourcePoolEvent::opened(1, pool_timestamp(1), &evidence).unwrap();
    let reserve_event = ResourcePoolEvent::transition(
        2,
        pool_timestamp(2),
        &evidence,
        reserved.receipts().last().unwrap(),
        reserved.latest_transition_validation_context().unwrap(),
    )
    .unwrap();
    let recovery_owner = match reserved.commit() {
        Err(ResourceCommitTransitionError::Recoverable(recovery)) => recovery,
        Err(ResourceCommitTransitionError::Poisoned(_)) => {
            panic!("expected recoverable commit failure")
        }
        Ok(_) => panic!("expected recoverable commit failure"),
    };
    let anchor = recovery_owner.failure().clone();
    let failed = ResourcePoolEvent::failed(3, pool_timestamp(3), &evidence, &anchor).unwrap();
    let recovered_transaction = recovery_owner.recover().unwrap();
    let recovery = recovered_transaction
        .recovery_history()
        .last()
        .unwrap()
        .clone();
    let _rolled_back = recovered_transaction.rollback().unwrap();
    (evidence, opened, reserve_event, failed, anchor, recovery)
}

#[track_caller]
pub(crate) fn check(passed: &mut usize, condition: bool) {
    assert!(condition);
    *passed += 1;
}
