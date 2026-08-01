mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

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
    assert_eq!(
        wave.prepared_participant_flight_count(),
        step.participant_count() as usize
    );
    assert_eq!(
        wave.node_participant_projection_count(),
        wave.node_count() * step.participant_count() as usize
    );
    assert_eq!(wave.physical_invocation_ledger_entry_count(), 1);
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
fn compiled_wave_identity_defers_full_node_and_participant_materialization() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let batch_identity = OperationDispatch::bind_compiled_submission_wave_identity(
        &topology,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    assert_eq!(topology.node_count(), fixture.plan.payload().nodes().len());
    assert_eq!(batch_identity.node_count(), topology.node_count());
    assert_eq!(
        batch_identity.materialization_snapshot().logical_nodes(),
        u32::try_from(topology.node_count()).unwrap()
    );
    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        0
    );
    assert!(!batch_identity
        .materialization_snapshot()
        .full_participant_projection());
    assert_eq!(
        batch_identity.node_id_at(0),
        Some(fixture.plan.payload().nodes()[0].id())
    );
    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        0
    );

    let eager_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    assert_eq!(batch_identity.nodes().len(), topology.node_count());
    assert_eq!(batch_identity.nodes(), eager_identity.nodes());
    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        u32::try_from(topology.node_count()).unwrap()
    );
    assert_eq!(
        batch_identity.participants().len(),
        topology.node_count() * active_bindings.len()
    );
    assert_eq!(batch_identity.participants(), eager_identity.participants());
    assert!(batch_identity
        .materialization_snapshot()
        .full_participant_projection());

    drop(eager_identity);
    drop(batch_identity);
    drop(topology);
    drop(lane);
    drop(active_bindings);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);
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
    let lane = Arc::clone(retry.execution_lane());
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let batch_identity = OperationDispatch::bind_compiled_submission_wave_identity(
        &topology,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        0
    );
    assert!(!batch_identity
        .materialization_snapshot()
        .full_participant_projection());
    let first_operation = batch_identity.nodes()[0].participants()[0]
        .identity()
        .parts();
    assert_eq!(first_operation.frame_id, Some(first_frame));
    assert_eq!(first_operation.sequence, 5);
    assert_eq!(first_operation.node_invocation_id.unwrap().get(), 1);

    drop(batch_identity);
    drop(topology);
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
fn full_plan_wave_rejects_participant_subset_before_legal_wave_can_prepare() {
    let fixture = fixture();
    let resources = (0..2)
        .map(|index| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.device-operation.wave-subset.{index}"),
                &format!("request.device-operation.wave-subset.{index}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resources| resources.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step_request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![one_token_span(), one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let step = (0..=3)
        .find_map(
            |attempt| match batch.try_begin_step(step_request.clone(), &lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => Some(step),
                StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("two-participant step admission did not converge"),
            },
        )
        .expect("bounded step admission returns or panics");
    let subset_shape = Arc::new(
        step.bind_invocation_work_shape(vec![(
            BatchParticipantAuthority::new(
                sessions[0].sequence_authority(),
                sessions[0].request_authority(),
            ),
            one_token_span(),
        )])
        .unwrap(),
    );
    let subset_requests = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| {
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                Arc::clone(&subset_shape),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let error = match step.try_prepare_submission_wave(subset_requests) {
        Err(error) => error,
        Ok(_) => panic!("participant-subset full-plan wave unexpectedly prepared"),
    };
    assert!(error
        .to_string()
        .contains("must bind every step participant exactly once"));

    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    assert_eq!(wave.prepared_participant_flight_count(), 2);
    assert_eq!(wave.node_participant_projection_count(), 4);
    assert_eq!(wave.physical_invocation_ledger_entry_count(), 1);
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in sessions {
        session.try_complete().unwrap();
    }
    drop(resources);
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
    let lane = Arc::clone(step.execution_lane());
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
    let lane = Arc::clone(step.execution_lane());
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
    let timing = RecordingSubmissionTimingSink::default();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[upload],
        SubmissionExecutionPolicy::adaptive(),
        &timing,
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
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_command_phases,
        vec![vec![
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::Compute,
        ]]
    );
    assert_eq!(
        *timing.stages.lock().unwrap(),
        vec![
            SubmissionWaveDispatchStage::ContractValidateAndReserve,
            SubmissionWaveDispatchStage::BackingAndInputEncode,
            SubmissionWaveDispatchStage::ProviderNodeEncode,
            SubmissionWaveDispatchStage::LaneReserve,
            SubmissionWaveDispatchStage::DeviceRuntimeSubmit,
            SubmissionWaveDispatchStage::CompletionArm,
            SubmissionWaveDispatchStage::LaneReserveSubmitAndArm,
        ]
    );
    let (handle, attribution) = handle.into_parts();
    assert!(attribution.is_none());
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
