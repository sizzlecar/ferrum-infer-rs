mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn chunked_token_span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[10, 11, 12, 13], 2..3).unwrap()
}

fn admit_batch_step(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![chunked_token_span(); batch.len() as usize])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            _ => panic!("batch step admission did not converge"),
        }
    }
    unreachable!("bounded batch step admission returns or panics")
}

fn admit_batch_invocation(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_id: &NodeId,
) -> InvocationResourceLease<TestRuntime> {
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        node_id.clone(),
        step.bind_all_invocation_work_shape(vec![
            chunked_token_span();
            step.participant_count() as usize
        ])
        .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => return invocation,
            InvocationResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            _ => panic!("batch invocation admission did not converge"),
        }
    }
    unreachable!("bounded batch invocation admission returns or panics")
}

#[test]
fn thirty_two_participant_dispatch_is_one_physical_submission() {
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
    let resources = (0..32)
        .map(|index| {
            logical_resources(
                &plan_resources,
                &format!("run.device-operation.batch32.{index}"),
                &format!("request.device-operation.batch32.{index}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resources| resources.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let active_bindings = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let step = admit_batch_step(&plan_resources, &batch);
    let node = &plan.payload().nodes()[0];
    let invocation = admit_batch_invocation(&plan_resources, &step, node.id());
    let packed_ranges = invocation.work_shape().participant_token_ranges();
    assert_eq!(packed_ranges.len(), 32);
    for (index, range) in packed_ranges.iter().enumerate() {
        assert_eq!(
            range.participant(),
            invocation.work_shape().participants()[index]
        );
        assert_eq!(
            range.immediate_token_range(),
            index as u64..index as u64 + 1
        );
        assert_eq!(range.immediate_tokens(), 1);
        assert_eq!(range.source_token_range(), 2..3);
        assert_eq!(range.full_input_tokens(), 4);
    }
    let identities = step
        .participant_frames()
        .zip(&active_bindings)
        .enumerate()
        .map(|(index, (frame, active))| {
            operation_identity(
                &plan,
                active,
                frame.frame_id(),
                NodeInvocationId::try_from(index as u64 + 1).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let batch_identity = OperationDispatch::bind_batch_identity(
        &resolved,
        identities,
        &active_bindings,
        &invocation,
        &lane,
    )
    .unwrap();
    assert_eq!(batch_identity.participants().len(), 32);
    assert_eq!(
        batch_identity
            .single_node()
            .expect("single invocation binds one node")
            .work_shape_fingerprint(),
        invocation.work_shape().fingerprint()
    );
    let handle = OperationDispatch::encode_and_submit(
        &provider,
        &resolved,
        &batch_identity,
        &active_bindings,
        invocation,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(runtime_trace.lock().unwrap().submit_calls, 1);
    let trace = provider_trace.lock().unwrap();
    assert_eq!(trace.encode_calls, 1);
    assert_eq!(trace.last_participant_count, 32);
    assert_eq!(trace.last_work_sequences, 32);
    drop(trace);
    assert_eq!(handle.receipt().participants().len(), 32);
    assert!(handle
        .receipt()
        .participants()
        .iter()
        .all(|participant| participant.batch_submission_fingerprint()
            == handle.receipt().fingerprint()));
    let completion = match handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("32-participant batch did not complete: {other:?}"),
    };
    assert_eq!(completion.participants().len(), 32);
    assert!(completion
        .participants()
        .iter()
        .all(|participant| participant.batch_completion_fingerprint() == completion.fingerprint()));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(handle);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(resources);
    drop(reaper);
    drop(lane);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
