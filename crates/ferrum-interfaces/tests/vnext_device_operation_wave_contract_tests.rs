mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn prepare_wave(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> PreparedStepSubmissionWave<TestRuntime> {
    let requests = plan
        .payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                step.bind_all_invocation_work_shape(vec![
                    one_token_span();
                    step.participant_count() as usize
                ])
                .unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for attempt in 0..=3 {
        match step.try_prepare_submission_wave(requests.clone()).unwrap() {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => return wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            _ => panic!("submission wave admission did not converge"),
        }
    }
    unreachable!("bounded wave admission returns or panics")
}

fn setup() -> (
    Fixture,
    Arc<AdmittedSequenceResources<TestRuntime>>,
    Arc<SequenceSession<TestRuntime>>,
    ExecutionBatchParticipants<TestRuntime>,
    Arc<StepResourceLease<TestRuntime>>,
) {
    let fixture = fixture();
    let sequence = logical_resources(
        &fixture.plan_resources,
        "run.device-operation.wave",
        "request.device-operation.wave",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&fixture.plan_resources, &batch);
    (fixture, sequence, session, batch, step)
}

fn wave_identity_inputs(
    plan: &ExecutionPlan,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    session: &Arc<SequenceSession<TestRuntime>>,
) -> (
    Vec<Vec<ExecutionIdentityEnvelope>>,
    Vec<Vec<TrustedActiveSequenceBinding>>,
) {
    let active = TrustedActiveSequenceBinding::from_session(session).unwrap();
    let mut identities = Vec::with_capacity(wave.node_count());
    let mut active_bindings = Vec::with_capacity(wave.node_count());
    for (node_index, node) in wave.nodes().iter().enumerate() {
        assert_eq!(node.participant_frames().len(), 1);
        identities.push(vec![operation_identity_for_node(
            plan,
            node_index,
            &active,
            node.participant_frames()[0].frame_id(),
            NodeInvocationId::try_from(u64::try_from(node_index).unwrap() + 1).unwrap(),
        )]);
        active_bindings.push(vec![active.clone()]);
    }
    (identities, active_bindings)
}

fn teardown(
    fixture: Fixture,
    sequence: Arc<AdmittedSequenceResources<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    step: Arc<StepResourceLease<TestRuntime>>,
) {
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn immutable_plan_nodes_prepare_one_owned_submission_wave() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);

    assert_eq!(wave.batch_step_id(), step.batch_step_id());
    assert_eq!(wave.node_count(), fixture.plan.payload().nodes().len());
    assert_eq!(wave.fingerprint().len(), 64);
    assert_eq!(wave.prepared_participant_flight_count(), wave.node_count());
    assert_eq!(
        wave.nodes()[0].node_id(),
        fixture.plan.payload().nodes()[0].id()
    );
    assert_eq!(wave.nodes()[0].work_shape(), step.work_shape());
    assert_eq!(wave.nodes()[0].participant_count(), 1);
    assert_eq!(wave.claimed_backing().fingerprint().len(), 64);

    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn wrong_wave_topology_rejects_before_legal_wave_can_prepare() {
    let (fixture, sequence, session, batch, step) = setup();
    let wrong = InvocationResourceAdmissionRequest::for_all_step_participants(
        NodeId::new("node/wrong-wave-topology").unwrap(),
        step.bind_all_invocation_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let error = match step.try_prepare_submission_wave(vec![wrong]) {
        Err(error) => error,
        Ok(_) => panic!("unknown wave node unexpectedly prepared"),
    };
    assert!(error
        .to_string()
        .contains("cover every plan node exactly once"));

    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn all_plan_nodes_encode_into_one_submission_and_one_completion() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let (identities, active_bindings) = wave_identity_inputs(&fixture.plan, &wave, &session);
    let lane = ExecutionLane::create(Arc::clone(&fixture.runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        identities,
        &active_bindings,
        &wave,
        &lane,
    )
    .unwrap();
    assert_eq!(batch_identity.nodes().len(), 2);
    assert_eq!(batch_identity.participants().len(), 2);

    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        &active_bindings,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 1);
        assert_eq!(trace.submitted_command_counts, vec![2]);
        assert_eq!(trace.next_fence, 1);
    }
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 2);
    assert_eq!(handle.receipt().participants().len(), 2);
    assert_eq!(lane.in_flight_count(), 1);
    assert_eq!(reaper.retained_count(), 1);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn terminal_wave_reads_output_before_releasing_backing() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let (identities, active_bindings) = wave_identity_inputs(&fixture.plan, &wave, &session);
    let lane = ExecutionLane::create(Arc::clone(&fixture.runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        identities,
        &active_bindings,
        &wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        &active_bindings,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let request = CompletionReadbackRequest::new(
        id("node.tail"),
        0,
        id("resource.output"),
        0,
        HostTransferLayout::new(ElementType::F32, 4).unwrap(),
    )
    .unwrap();
    let receipt = match handle.wait_with_readback(request).unwrap() {
        CompletionReadbackObservation::Terminal(receipt) => receipt,
        other => panic!("wave output readback did not terminate: {other:?}"),
    };
    assert!(matches!(
        receipt.completion().disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let output = match receipt.disposition() {
        CompletionReadbackDisposition::Succeeded(output) => output,
        other => panic!("wave output readback failed: {other:?}"),
    };
    assert_eq!(output.request().node_id(), &id("node.tail"));
    assert_eq!(output.request().resource_id(), &id("resource.output"));
    assert_eq!(output.bytes(), &[0; 16]);
    assert_eq!(output.sha256().len(), 64);
    assert_eq!(receipt.fingerprint().len(), 64);
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert!(trace.readback_calls >= 1);
        assert_eq!(trace.readback_lengths.iter().sum::<u64>(), 16);
    }
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    assert!(handle.wait().is_err(), "terminal slot must be reaped once");

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn definitely_not_submitted_retries_the_same_whole_wave() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let first_attempt = wave.batch_invocation_id();
    let topology_fingerprint = wave.fingerprint().to_owned();
    let (identities, active_bindings) = wave_identity_inputs(&fixture.plan, &wave, &session);
    let lane = ExecutionLane::create(Arc::clone(&fixture.runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::DefinitelyNotSubmitted;
    let first_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        identities.clone(),
        &active_bindings,
        &wave,
        &lane,
    )
    .unwrap();
    let (failures, retry) = match OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &first_identity,
        &active_bindings,
        wave,
        &lane,
        &reaper,
    ) {
        Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { failures, retry }) => {
            (failures, retry)
        }
        other => panic!("wave did not return typed definitely-not-submitted: {other:?}"),
    };
    assert_eq!(failures.len(), 2);
    assert_eq!(retry.prior_attempt(), first_attempt);
    assert_eq!(retry.topology_fingerprint(), topology_fingerprint);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    let retry_wave = retry.retry().unwrap();
    assert_ne!(retry_wave.batch_invocation_id(), first_attempt);
    assert_eq!(retry_wave.fingerprint(), topology_fingerprint);
    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::Success;
    let retry_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        identities,
        &active_bindings,
        &retry_wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &retry_identity,
        &active_bindings,
        retry_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 2);
        assert_eq!(trace.submitted_command_counts, vec![2, 2]);
    }
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 4);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(reaper.retained_count(), 0);

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
