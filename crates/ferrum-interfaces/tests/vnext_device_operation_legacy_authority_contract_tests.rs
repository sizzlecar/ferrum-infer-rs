mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn legacy_source_seals_sequence_and_cannot_authorize_other_session(
    fixture: Fixture,
    passed: &mut usize,
) {
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
    } = fixture;
    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.legacy-authority",
        "request.device-operation.legacy-authority",
    );
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let legacy_binding = TrustedActiveSequenceBinding::from_permit(&permit).unwrap();
    let legacy_completion = permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        legacy_completion.activation_epoch() == legacy_binding.activation_epoch()
            && legacy_completion.sequence_authority() == legacy_binding.sequence_authority(),
    );
    drop(stream);

    check(
        passed,
        matches!(
            resources.open_session(),
            Err(error)
                if error
                    .to_string()
                    .contains("permanently selected for legacy streams")
        ),
    );

    let session_resources = logical_resources(
        &plan_resources,
        "run.device-operation.session-authority",
        "request.device-operation.session-authority",
    );
    let session = session_resources.open_session().unwrap();
    check(
        passed,
        legacy_binding.sequence_authority() != session.sequence_authority(),
    );
    check(
        passed,
        legacy_binding.run_id() != session_resources.run_id(),
    );
    check(
        passed,
        legacy_binding.request_id() != session_resources.request_id(),
    );
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &legacy_binding, frame_id, invocation_id);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
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
                &legacy_binding,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    check(passed, reaper.retained_count() == 0);
    check(passed, lane.in_flight_count() == 0);
    check(passed, step.try_retire_normal().is_ok());
    check(passed, session.try_complete().is_ok());
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(session_resources);
    drop(resources);
    drop(runtime);
    close_plan_runtime(plan_resources, passed);
    assert_eq!(*passed - start, EXPECTED_LEGACY_AUTHORITY_CASES);
}

#[test]
fn device_operation_legacy_authority_contract_is_exhaustive() {
    let mut passed = 0;
    legacy_source_seals_sequence_and_cannot_authorize_other_session(fixture(), &mut passed);
    assert_eq!(passed, EXPECTED_LEGACY_AUTHORITY_CASES);
    println!("\nVNEXT DEVICE OPERATION LEGACY AUTHORITY PASS: {passed}/{EXPECTED_LEGACY_AUTHORITY_CASES}");
}
