mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn cancel_dispatch_linearization_contract(initial_fixture: Fixture, passed: &mut usize) {
    let start = *passed;
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = initial_fixture;
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let resources = logical_resources(
        &plan_resources,
        "run.cancel-before-dispatch",
        "request.cancel-before-dispatch",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let invocation = admit_single_participant_invocation(&plan_resources, &step, node.id());
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();

    let cancel = session.request_cancel().unwrap();
    check(
        passed,
        cancel.active_frame() == Some(frame_id) && cancel.participant_flights() == 1,
    );
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                invocation,
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    check(
        passed,
        reaper.retained_count() == 0 && lane.in_flight_count() == 0,
    );
    let retired = step.try_retire_normal().unwrap();
    check(
        passed,
        retired.participants()[0].disposition()
            == StepParticipantRetirementDisposition::DiscardedCancelled,
    );
    let terminal = session.try_abort().unwrap();
    check(
        passed,
        terminal.disposition() == SequenceSessionTerminalDisposition::Aborted,
    );
    drop(terminal);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    close_plan_runtime(plan_resources, passed);

    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture();
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let resources = logical_resources(
        &plan_resources,
        "run.dispatch-before-cancel",
        "request.dispatch-before-cancel",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let completion = encode_and_submit_single(
        &provider,
        &resolved,
        &identity,
        &frame_id,
        &invocation_id,
        node.id(),
        &active,
        admit_single_participant_invocation(&plan_resources, &step, node.id()),
        &lane,
        &reaper,
    )
    .unwrap();

    let cancel = session.request_cancel().unwrap();
    check(
        passed,
        cancel.active_frame() == Some(frame_id) && cancel.participant_flights() == 1,
    );
    check(
        passed,
        matches!(completion.poll(), Ok(CompletionObservation::Terminal(_))),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 1);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 1);
    check(
        passed,
        reaper.retained_count() == 0 && lane.in_flight_count() == 0,
    );
    let retired = step.try_retire_normal().unwrap();
    check(
        passed,
        retired.participants()[0].disposition()
            == StepParticipantRetirementDisposition::DiscardedCancelled,
    );
    let terminal = session.try_abort().unwrap();
    check(
        passed,
        terminal.disposition() == SequenceSessionTerminalDisposition::Aborted,
    );
    drop(terminal);
    drop(completion);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    close_plan_runtime(plan_resources, passed);
    assert_eq!(*passed - start, EXPECTED_CANCEL_DISPATCH_CASES);
}

#[test]
fn device_operation_cancel_contract_is_exhaustive() {
    let mut passed = 0;
    cancel_dispatch_linearization_contract(fixture(), &mut passed);
    assert_eq!(passed, EXPECTED_CANCEL_DISPATCH_CASES);
    println!("\nVNEXT DEVICE OPERATION CANCEL PASS: {passed}/{EXPECTED_CANCEL_DISPATCH_CASES}");
}
