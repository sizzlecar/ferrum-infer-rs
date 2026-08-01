mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

#[test]
fn definitely_not_submitted_retries_the_same_whole_wave() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let first_attempt = wave.batch_invocation_id();
    let topology_fingerprint = wave.fingerprint().to_owned();
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
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
fn provider_scratch_is_zeroed_before_every_reused_lane_invocation() {
    let (fixture, sequence, session, batch, first_step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ScratchZeroed),
    );
    let lane = Arc::clone(first_step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    let first_wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &first_step);
    let first_active_bindings = wave_active_bindings(&first_wave, &session);
    let first_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        first_active_bindings.iter(),
        &first_wave,
        &lane,
    )
    .unwrap();
    let first_completion = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &first_identity,
        first_active_bindings.iter(),
        DeviceTimingMode::Off,
        first_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        first_completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(first_completion);
    drop(first_active_bindings);
    first_step.try_retire_normal().unwrap();

    let second_step = begin_single_participant_step_on_lane(&batch, &lane);
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

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_commands,
            vec![
                vec![
                    TestCommand::Zero,
                    TestCommand::Zero,
                    TestCommand::ScratchProvider,
                    TestCommand::ScratchProvider,
                ],
                vec![
                    TestCommand::Zero,
                    TestCommand::Zero,
                    TestCommand::ScratchProvider,
                    TestCommand::ScratchProvider,
                ],
            ]
        );
        assert_eq!(
            trace.submitted_command_phases,
            vec![
                vec![
                    DeviceCommandPhase::Initialization,
                    DeviceCommandPhase::Initialization,
                    DeviceCommandPhase::Compute,
                    DeviceCommandPhase::Compute,
                ],
                vec![
                    DeviceCommandPhase::Initialization,
                    DeviceCommandPhase::Initialization,
                    DeviceCommandPhase::Compute,
                    DeviceCommandPhase::Compute,
                ],
            ]
        );
        assert_eq!(
            trace.submitted_command_node_indices,
            vec![
                vec![Some(0), Some(1), Some(0), Some(1)],
                vec![Some(0), Some(1), Some(0), Some(1)],
            ]
        );
        assert_eq!(
            trace.scratch_observations,
            vec![(0, 0), (1, 0), (0, 0), (1, 0)]
        );
    }

    drop(second_completion);
    drop(second_active_bindings);
    drop(providers);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, second_step);
}

#[test]
fn typed_eager_policy_zeroes_overwrite_scratch_independently_from_timing() {
    let (fixture, sequence, session, batch, first_step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ScratchOverwrite),
    );
    let lane = Arc::clone(first_step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    let first_wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &first_step);
    let first_active_bindings = wave_active_bindings(&first_wave, &session);
    let first_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        first_active_bindings.iter(),
        &first_wave,
        &lane,
    )
    .unwrap();
    let first_completion = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &first_identity,
        first_active_bindings.iter(),
        DeviceTimingMode::Off,
        first_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        first_completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(first_completion);
    drop(first_active_bindings);
    first_step.try_retire_normal().unwrap();

    let second_step = begin_single_participant_step_on_lane(&batch, &lane);
    let second_wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &second_step);
    let second_active_bindings = wave_active_bindings(&second_wave, &session);
    let second_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        second_active_bindings.iter(),
        &second_wave,
        &lane,
    )
    .unwrap();
    let second_completion = OperationDispatch::encode_and_submit_wave_with_inputs_and_policy(
        &providers,
        &fixture.resolved,
        &second_identity,
        second_active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::determinism_eager(0),
        second_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        second_completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_commands,
            vec![
                vec![TestCommand::ScratchProvider, TestCommand::ScratchProvider],
                vec![
                    TestCommand::Zero,
                    TestCommand::Zero,
                    TestCommand::ScratchProvider,
                    TestCommand::ScratchProvider,
                ],
            ]
        );
        assert_eq!(
            trace.submitted_command_node_indices,
            vec![
                vec![Some(0), Some(1)],
                vec![Some(0), Some(1), Some(0), Some(1)],
            ]
        );
        assert_eq!(
            trace.scratch_observations,
            vec![(0, 0xa5), (1, 0xa5), (0, 0), (1, 0)]
        );
    }

    drop(second_completion);
    drop(second_active_bindings);
    drop(providers);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, second_step);
}

#[test]
fn typed_eager_policy_poison_fills_overwrite_scratch_before_compute() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ScratchOverwrite),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    let completion = OperationDispatch::encode_and_submit_wave_with_inputs_and_policy(
        &providers,
        &fixture.resolved,
        &identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::determinism_eager(0xa5),
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::Upload(0xa5, BufferUsage::Scratch),
                TestCommand::Upload(0xa5, BufferUsage::Scratch),
                TestCommand::ScratchProvider,
                TestCommand::ScratchProvider,
            ]]
        );
        assert_eq!(
            trace.submitted_command_phases,
            vec![vec![
                DeviceCommandPhase::Initialization,
                DeviceCommandPhase::Initialization,
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::Compute,
            ]]
        );
        assert_eq!(trace.scratch_observations, vec![(0, 0xa5), (1, 0xa5)]);
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::EagerOnly]
        );
        assert_eq!(
            trace
                .uploaded_payloads
                .iter()
                .filter(|payload| !payload.is_empty() && payload.iter().all(|byte| *byte == 0xa5))
                .count(),
            2
        );
    }

    drop(completion);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn replayed_only_policy_rejects_an_eager_wave_before_provider_encoding() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    let error = OperationDispatch::encode_and_submit_wave_with_inputs_and_policy(
        &providers,
        &fixture.resolved,
        &identity,
        active_bindings.iter(),
        DeviceTimingMode::Replay,
        &[],
        SubmissionExecutionPolicy::determinism_replayed(0),
        wave,
        &lane,
        &reaper,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains(
                "replayed-only submission requires one sealed reusable execution program"
            )
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

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
    let lane = Arc::clone(first_step.execution_lane());
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
            trace.submitted_command_phases,
            vec![
                vec![
                    DeviceCommandPhase::Initialization,
                    DeviceCommandPhase::Compute,
                    DeviceCommandPhase::Compute,
                ],
                vec![
                    DeviceCommandPhase::Initialization,
                    DeviceCommandPhase::Compute,
                    DeviceCommandPhase::Compute,
                ],
            ]
        );
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

    let second_step = begin_single_participant_step_on_lane(&batch, &lane);
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
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_command_phases,
        vec![
            vec![
                DeviceCommandPhase::Initialization,
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::Compute,
            ],
            vec![
                DeviceCommandPhase::Initialization,
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::Compute,
            ],
            vec![DeviceCommandPhase::Compute, DeviceCommandPhase::Compute],
        ]
    );

    drop(second_completion);
    drop(second_active_bindings);
    drop(providers);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, second_step);
}
