mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn prepare_wave(
    _plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
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
                deferred.maintain().unwrap();
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
    setup_with_fixture(fixture())
}

fn setup_with_fixture(
    fixture: Fixture,
) -> (
    Fixture,
    Arc<AdmittedSequenceResources<TestRuntime>>,
    Arc<SequenceSession<TestRuntime>>,
    ExecutionBatchParticipants<TestRuntime>,
    Arc<StepResourceLease<TestRuntime>>,
) {
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

fn wave_active_bindings(
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    session: &Arc<SequenceSession<TestRuntime>>,
) -> Vec<TrustedActiveSequenceBinding> {
    let active = TrustedActiveSequenceBinding::from_session(session).unwrap();
    for node in wave.nodes() {
        assert_eq!(node.participant_frames().len(), 1);
    }
    vec![active]
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
fn immutable_plan_prebinds_owned_providers_in_node_order() {
    let fixture = fixture();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let expected = fixture
        .resolved
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .map(|node| node.selection().selected_provider())
        .collect::<Vec<_>>();
    let actual = providers
        .providers()
        .iter()
        .map(|provider| provider.descriptor().provider_id())
        .collect::<Vec<_>>();

    assert_eq!(providers.len(), expected.len());
    assert_eq!(actual, expected);

    drop(fixture.registry);
    assert_eq!(
        providers.providers()[0].descriptor().provider_id(),
        expected[0]
    );

    drop(providers);
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
fn unsubmitted_step_retry_keeps_the_first_physical_journal_identity() {
    let (fixture, sequence, session, batch, first_step) = setup();
    let first_step_id = first_step.batch_step_id();
    let first_frame = first_step
        .participant_frames()
        .next()
        .expect("single-participant step owns one frame")
        .frame_id();
    first_step.try_rollback_unsubmitted().unwrap();

    let retry = begin_single_participant_step(&fixture.plan_resources, &batch);
    assert_ne!(retry.batch_step_id(), first_step_id);
    assert_eq!(
        retry
            .participant_frames()
            .next()
            .expect("single-participant retry owns one frame")
            .frame_id(),
        first_frame
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &retry);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = ExecutionLane::create(Arc::clone(&fixture.runtime)).unwrap();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let first_operation = batch_identity.nodes()[0].participants()[0]
        .identity()
        .parts();
    assert_eq!(first_operation.frame_id, Some(first_frame));
    assert_eq!(first_operation.sequence, 5);
    assert_eq!(first_operation.node_invocation_id.unwrap().get(), 1);

    drop(batch_identity);
    drop(lane);
    drop(active_bindings);
    drop(wave);
    teardown(fixture, sequence, session, batch, retry);
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
    let active_bindings = wave_active_bindings(&wave, &session);
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
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    assert_eq!(batch_identity.nodes().len(), 2);
    assert_eq!(batch_identity.participants().len(), 2);
    let first = batch_identity.nodes()[0].participants()[0]
        .identity()
        .parts();
    let second = batch_identity.nodes()[1].participants()[0]
        .identity()
        .parts();
    assert_eq!(first.sequence, 5);
    assert_eq!(first.node_invocation_id.unwrap().get(), 1);
    assert!(first.span_id.as_str().ends_with("/operation"));
    assert_eq!(
        first.parent_span_id.as_ref().unwrap().as_str(),
        &first.span_id.as_str()[..first.span_id.as_str().len() - "/operation".len()]
    );
    assert_eq!(second.sequence, 8);
    assert_eq!(second.node_invocation_id.unwrap().get(), 2);

    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
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
    let completion = match handle.poll().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("wave did not terminate: {other:?}"),
    };
    assert_eq!(
        completion.fence_timing().timing_mode(),
        DeviceTimingMode::Off
    );
    assert_eq!(
        completion.fence_timing().device_execution(),
        DeviceTimingMeasurement::NotRequested
    );
    assert_eq!(
        completion.fence_timing().blocking_wait_host_ns(),
        DeviceTimingMeasurement::NotRequested
    );
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
fn typed_input_upload_precedes_the_plan_in_one_submission() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
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
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let upload = SubmissionWaveInputUpload::new(
        id("node.main"),
        0,
        0,
        0,
        HostTransferLayout::new(ElementType::F32, 4).unwrap(),
        vec![0; 16],
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[upload],
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_command_counts,
        vec![3]
    );
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

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
    let executable = ExecutablePlan::new(
        fixture.plan.clone(),
        fixture.resolved.parts().capabilities.clone(),
    )
    .unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = ExecutionLane::create(Arc::clone(&fixture.runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&executable, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &executable,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &executable,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
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
    let duplicate = CompletionReadbackBatchRequest::new(vec![request.clone(), request.clone()]);
    assert!(duplicate.is_err());
    assert_eq!(reaper.retained_count(), 1);

    let out_of_range = CompletionReadbackBatchRequest::new(vec![
        request.clone(),
        CompletionReadbackRequest::new(
            id("node.tail"),
            1,
            id("resource.output"),
            0,
            HostTransferLayout::new(ElementType::F32, 4).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    assert!(handle.wait_with_readbacks(out_of_range).is_err());
    assert_eq!(reaper.retained_count(), 1);

    let foreign = CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
        id("node.foreign"),
        0,
        id("resource.output"),
        0,
        HostTransferLayout::new(ElementType::F32, 4).unwrap(),
    )
    .unwrap()])
    .unwrap();
    assert!(handle.wait_with_readbacks(foreign).is_err());
    assert_eq!(reaper.retained_count(), 1);

    let receipt = match handle.wait_with_readback(request).unwrap() {
        CompletionReadbackObservation::Terminal(receipt) => receipt,
        other => panic!("wave output readback did not terminate: {other:?}"),
    };
    assert!(matches!(
        receipt.completion().disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let fence_timing = receipt.completion().fence_timing();
    assert_eq!(fence_timing.timing_mode(), DeviceTimingMode::Completion);
    assert!(matches!(
        fence_timing.device_execution(),
        DeviceTimingMeasurement::Measured(timing) if timing.elapsed_ns() == 1_000_000
    ));
    assert!(matches!(
        fence_timing.blocking_wait_host_ns(),
        DeviceTimingMeasurement::Measured(_)
    ));
    assert!(matches!(
        receipt.readback_timing(),
        Some(DeviceTimingMeasurement::Measured(timing))
            if timing.calls() == 1 && timing.bytes() == 16
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
    let active_bindings = wave_active_bindings(&wave, &session);
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
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let (failures, retry) = match OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &first_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
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
        active_bindings.iter(),
        &retry_wave,
        &lane,
    )
    .unwrap();
    assert_eq!(first_identity.nodes(), retry_identity.nodes());
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &retry_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
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

#[test]
fn zero_state_initialization_is_ordered_retried_and_not_repeated_after_success() {
    let (fixture, sequence, session, batch, first_step) =
        setup_with_fixture(fixture_with_zero_state(true));
    let first_wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &first_step);
    let active_bindings = wave_active_bindings(&first_wave, &session);
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
        active_bindings.iter(),
        &first_wave,
        &lane,
    )
    .unwrap();
    let retry = match OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &first_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        first_wave,
        &lane,
        &reaper,
    ) {
        Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        other => panic!("zero-state wave did not return retry authority: {other:?}"),
    };

    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::Success;
    let retry_wave = retry.retry().unwrap();
    let retry_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &retry_wave,
        &lane,
    )
    .unwrap();
    let first_completion = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &retry_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        retry_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        first_completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_commands,
            vec![
                vec![
                    TestCommand::Zero,
                    TestCommand::Provider,
                    TestCommand::Provider
                ],
                vec![
                    TestCommand::Zero,
                    TestCommand::Provider,
                    TestCommand::Provider
                ],
            ]
        );
    }

    drop(first_completion);
    drop(active_bindings);
    first_step.try_retire_normal().unwrap();

    let second_step = begin_single_participant_step(&fixture.plan_resources, &batch);
    let second_wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &second_step);
    let second_active_bindings = wave_active_bindings(&second_wave, &session);
    let second_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        second_active_bindings.iter(),
        &second_wave,
        &lane,
    )
    .unwrap();
    let second_completion = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &second_identity,
        second_active_bindings.iter(),
        DeviceTimingMode::Off,
        second_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        second_completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().submitted_commands,
        vec![
            vec![
                TestCommand::Zero,
                TestCommand::Provider,
                TestCommand::Provider
            ],
            vec![
                TestCommand::Zero,
                TestCommand::Provider,
                TestCommand::Provider
            ],
            vec![TestCommand::Provider, TestCommand::Provider],
        ]
    );

    drop(second_completion);
    drop(second_active_bindings);
    drop(providers);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, second_step);
}
