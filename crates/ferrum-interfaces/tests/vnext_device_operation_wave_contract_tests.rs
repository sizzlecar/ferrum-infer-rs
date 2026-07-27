mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

#[derive(Default)]
struct RecordingSubmissionTimingSink {
    stages: Mutex<Vec<SubmissionWaveDispatchStage>>,
}

impl DeviceSubmissionTimingSink for RecordingSubmissionTimingSink {
    const ENABLED: bool = true;

    fn record_device_submission(&self, _stage: DeviceSubmissionStage, _elapsed: Duration) {}
}

impl SubmissionWaveDispatchTimingSink for RecordingSubmissionTimingSink {
    fn record(&self, stage: SubmissionWaveDispatchStage, _elapsed: Duration) {
        self.stages.lock().unwrap().push(stage);
    }
}

fn prepare_wave(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> PreparedStepSubmissionWave<TestRuntime> {
    let node_ids = plan
        .payload()
        .nodes()
        .iter()
        .map(|node| node.id().clone())
        .collect::<Vec<_>>();
    prepare_wave_for_node_scope(plan_resources, plan, step, &node_ids, false)
}

fn prepare_determinism_wave(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> PreparedStepSubmissionWave<TestRuntime> {
    let node_ids = plan
        .payload()
        .nodes()
        .iter()
        .map(|node| node.id().clone())
        .collect::<Vec<_>>();
    prepare_determinism_wave_for_nodes(plan_resources, plan, step, &node_ids)
}

fn prepare_determinism_wave_for_nodes(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_ids: &[NodeId],
) -> PreparedStepSubmissionWave<TestRuntime> {
    prepare_wave_for_node_scope(plan_resources, plan, step, node_ids, true)
}

fn prepare_wave_for_node_scope(
    _plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_ids: &[NodeId],
    determinism: bool,
) -> PreparedStepSubmissionWave<TestRuntime> {
    let participant_work = step
        .work_shape()
        .participant_work()
        .iter()
        .map(|work| work.token_span().clone())
        .collect::<Vec<_>>();
    let requests = node_ids
        .iter()
        .map(|node| {
            let node = plan
                .payload()
                .nodes()
                .iter()
                .find(|candidate| candidate.id() == node)
                .expect("test wave node belongs to its exact plan");
            InvocationResourceAdmissionRequest::for_all_step_participants(
                node.id().clone(),
                step.bind_all_invocation_work_shape(participant_work.clone())
                    .unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for attempt in 0..=3 {
        let decision = if determinism {
            step.try_prepare_determinism_submission_wave(requests.clone())
        } else {
            step.try_prepare_submission_wave(requests.clone())
        }
        .unwrap();
        match decision {
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
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
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

fn determinism_restore(
    fixture: &Fixture,
    providers: &[BoundOperationProvider<'_, TestRuntime>],
    batch_identity: &BatchOperationIdentity,
    active_bindings: &[TrustedActiveSequenceBinding],
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    fill_byte: u8,
) -> SubmissionWaveDeterminismRestore {
    let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
        fixture.runtime.as_ref(),
        providers,
        &fixture.resolved,
        batch_identity,
        active_bindings.iter(),
        wave,
    )
    .unwrap();
    assert!(
        !layout.witness_plan().initializations().is_empty(),
        "determinism fixture must have at least one typed initialization"
    );
    let participant_payloads = determinism_payloads(&layout, fill_byte);
    layout.bind(participant_payloads).unwrap()
}

fn determinism_payloads(
    layout: &SubmissionWaveDeterminismRestoreLayout,
    fill_byte: u8,
) -> Vec<Vec<Vec<u8>>> {
    (0..layout.participant_count())
        .map(|participant_index| {
            layout
                .participant_initialization_ranges(participant_index)
                .unwrap()
                .iter()
                .map(|range| {
                    vec![
                        fill_byte;
                        usize::try_from(range.length_bytes())
                            .expect("test initialization length fits usize")
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect()
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

#[test]
fn determinism_eager_submission_restores_the_complete_typed_denominator() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ScratchOverwrite),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
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
    let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
        fixture.runtime.as_ref(),
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        &wave,
    )
    .unwrap();
    let valid_payloads = determinism_payloads(&layout, 0x31);
    let mut invalid_payloads = valid_payloads.clone();
    invalid_payloads[0][0].pop();
    assert!(layout.clone().bind(invalid_payloads).is_err());
    assert!(layout.clone().bind(vec![Vec::new()]).is_err());
    let restore = layout.bind(valid_payloads).unwrap();

    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Kernel,
        &restore,
        0xa5,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let restore_fingerprint = restore.logical_fingerprint().unwrap();
    assert_eq!(handle.restore_fingerprint(), restore_fingerprint);
    let attribution = handle
        .attribution()
        .expect("determinism eager path must preserve attribution");
    let submission_fingerprint = attribution.submission_fingerprint().to_owned();
    let expected_readbacks = handle.readback_plan().collection_request().request_count();
    let expected_witnesses = handle.readback_plan().witness_count();
    assert_eq!(
        expected_witnesses,
        fixture
            .plan
            .determinism_witness_plan()
            .unwrap()
            .witnesses()
            .len()
    );

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        let phases = &trace.submitted_command_phases[0];
        let first_compute = phases
            .iter()
            .position(|phase| *phase == DeviceCommandPhase::Compute)
            .expect("determinism submission has provider compute");
        assert!(phases[..first_compute]
            .iter()
            .all(|phase| *phase == DeviceCommandPhase::Initialization));
        assert!(phases[first_compute..]
            .iter()
            .all(|phase| *phase == DeviceCommandPhase::Compute));
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::EagerOnly]
        );
        assert_eq!(
            trace.submitted_attribution_requirements,
            vec![DeviceSubmissionAttributionRequirement::LogicalExecutionPath]
        );
        assert_eq!(trace.scratch_observations, vec![(0, 0xa5), (1, 0xa5)]);
        assert!(trace
            .uploaded_payloads
            .iter()
            .any(|payload| !payload.is_empty() && payload.iter().all(|byte| *byte == 0x31)));
    }
    let compute_rows = attribution
        .device()
        .commands()
        .iter()
        .filter(|command| command.command_phase() == DeviceCommandPhase::Compute)
        .collect::<Vec<_>>();
    assert_eq!(compute_rows.len(), providers.len());
    assert!(compute_rows
        .iter()
        .all(|command| command.execution_path() == DeviceExecutionPath::Eager));
    let evidence = handle.wait_into_evidence().unwrap();
    assert_eq!(evidence.restore_fingerprint(), restore_fingerprint);
    assert_eq!(
        evidence.expected_execution_path(),
        DeviceExecutionPath::Eager
    );
    assert_eq!(evidence.physical_readbacks().len(), expected_readbacks);
    assert_eq!(evidence.witnesses().len(), expected_witnesses);
    assert!(evidence
        .physical_readbacks()
        .iter()
        .all(|readback| !readback.bytes().is_empty() && readback.raw_sha256().len() == 64));
    assert_eq!(
        evidence.attribution().submission_fingerprint(),
        submission_fingerprint
    );

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_logical_restore_identity_ignores_fresh_physical_authority() {
    fn collect(fill_byte: u8) -> String {
        let (fixture, sequence, session, batch, step) = setup_with_fixture(
            fixture_with_determinism_provider_behavior(false, ProviderBehavior::ScratchOverwrite),
        );
        let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active_bindings = wave_active_bindings(&wave, &session);
        let lane = Arc::clone(step.execution_lane());
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
        let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
            fixture.runtime.as_ref(),
            &providers,
            &fixture.resolved,
            &batch_identity,
            active_bindings.iter(),
            &wave,
        )
        .unwrap();
        let payloads = determinism_payloads(&layout, fill_byte);
        let fingerprint = layout
            .bind(payloads)
            .unwrap()
            .logical_fingerprint()
            .unwrap();

        drop(batch_identity);
        drop(providers);
        drop(active_bindings);
        drop(wave);
        drop(lane);
        teardown(fixture, sequence, session, batch, step);
        fingerprint
    }

    let first = collect(0x31);
    let second = collect(0x31);
    let changed_payload = collect(0x32);
    assert_eq!(first, second);
    assert_ne!(first, changed_payload);
}

#[test]
fn determinism_submission_rejects_unretained_outputs_before_provider_encode() {
    let (fixture, sequence, session, batch, step) = setup();
    assert!(fixture
        .plan
        .payload()
        .retained_completion_values()
        .is_empty());
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
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
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x31,
    );

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0xa5,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("unretained determinism output unexpectedly reached provider encode"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("is not retained as one exact terminal witness")
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
fn determinism_readback_collects_writable_state_with_exact_typed_usage() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(true, ProviderBehavior::Success),
    );
    let node_ids = [id("node.tail")];
    let witness_plan = fixture
        .plan
        .determinism_witness_plan_for_nodes(&node_ids)
        .unwrap();
    assert_eq!(witness_plan.node_ids(), &node_ids);
    assert!(witness_plan.initializations().iter().any(|initialization| {
        matches!(
            initialization.kind(),
            ExecutionDeterminismInitializationKind::ExternalInput { value_id }
                if value_id.as_str() == "value.intermediate"
        )
    }));
    assert!(witness_plan.witnesses().iter().any(|witness| {
        matches!(
            witness.kind(),
            ExecutionDeterminismWitnessKind::StateEffect { .. }
        ) && witness.location().usage() == BufferUsage::State
    }));
    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x27,
    );

    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(handle.readback_plan().targets().iter().any(|target| {
        target.witnesses().iter().any(|witness| {
            matches!(
                witness.kind(),
                ExecutionDeterminismWitnessKind::StateEffect { .. }
            )
        }) && target
            .batch()
            .requests()
            .iter()
            .all(|request| request.expected_usage() == BufferUsage::State)
    }));

    let readback = match handle.wait_with_determinism_readback().unwrap() {
        CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
        other => panic!("writable-state determinism readback did not terminate: {other:?}"),
    };
    assert!(readback.dispositions().iter().any(|disposition| {
        matches!(
            disposition,
            CompletionReadbackDisposition::Succeeded(output)
                if output.request().expected_usage() == BufferUsage::State
        )
    }));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_chunk_binds_the_exact_paged_state_prefix_from_the_provider_view() {
    let fixture = fixture_with_token_scaled_paged_state();
    let full_input = [11, 12, 13, 14];
    let sequence = logical_resources_with_work(
        &fixture.plan_resources,
        "run.device-operation.paged-determinism",
        "request.device-operation.paged-determinism",
        TokenSpanWork::from_token_ids(&full_input, 0..full_input.len()).unwrap(),
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = begin_single_participant_step_on_lane_with_bucket_and_work(
        &batch,
        &lane,
        None,
        TokenSpanWork::from_token_ids(&full_input, 2..3).unwrap(),
    );
    let node_ids = [id("node.tail")];
    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
        fixture.runtime.as_ref(),
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        &wave,
    )
    .unwrap();

    let state_initialization_index = layout
        .witness_plan()
        .initializations()
        .iter()
        .position(|initialization| initialization.location().usage() == BufferUsage::State)
        .expect("token-scaled state must have one deterministic initialization");
    let state_initialization = &layout.witness_plan().initializations()[state_initialization_index];
    assert_eq!(state_initialization.location().declared_length_bytes(), 4);
    assert!(matches!(
        state_initialization.location().extent(),
        ExecutionDeterminismValueExtent::ActiveTokenPrefix {
            bytes_per_token: 4,
            maximum_tokens: 16,
            maximum_storage_bytes: 64,
        }
    ));
    let state_range =
        layout.participant_initialization_ranges(0).unwrap()[state_initialization_index];
    assert_eq!(state_range.logical_offset_bytes(), 0);
    assert_eq!(state_range.length_bytes(), 16);
    assert!(state_range.length_bytes() > 3 * 4);
    assert_eq!(state_range.length_bytes() % TEST_PAGED_BLOCK_BYTES, 0);

    let external_input_index = layout
        .witness_plan()
        .initializations()
        .iter()
        .position(|initialization| {
            matches!(
                initialization.kind(),
                ExecutionDeterminismInitializationKind::ExternalInput { .. }
            )
        })
        .expect("tail chunk must bind its external activation input");
    let external_input_range =
        layout.participant_initialization_ranges(0).unwrap()[external_input_index];
    assert_eq!(external_input_range.logical_offset_bytes(), 0);
    assert_eq!(external_input_range.length_bytes(), 4);

    let state_witness_index = layout
        .witness_plan()
        .witnesses()
        .iter()
        .position(|witness| {
            matches!(
                witness.kind(),
                ExecutionDeterminismWitnessKind::StateEffect { .. }
            )
        })
        .expect("tail chunk must retain its writable state witness");
    assert_eq!(
        layout
            .witness_participant_ranges(state_witness_index)
            .unwrap()[0],
        state_range
    );

    let participant_payloads = determinism_payloads(&layout, 0x27);
    let restore = layout.bind(participant_payloads).unwrap();
    let reaper = CompletionReaper::new();
    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let state_readback = handle
        .readback_plan()
        .targets()
        .iter()
        .flat_map(|target| target.batch().requests())
        .find(|request| request.expected_usage() == BufferUsage::State)
        .expect("token-scaled state must have one exact terminal readback");
    assert_eq!(state_readback.logical_offset_bytes(), 0);
    assert_eq!(state_readback.output_layout().byte_len().unwrap(), 16);

    let readback = match handle.wait_with_determinism_readback().unwrap() {
        CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
        other => panic!("paged-state determinism readback did not terminate: {other:?}"),
    };
    assert!(readback.dispositions().iter().any(|disposition| {
        matches!(
            disposition,
            CompletionReadbackDisposition::Succeeded(output)
                if output.request().expected_usage() == BufferUsage::State
                    && output.bytes().len() == 16
        )
    }));
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert!(trace
            .uploaded_payloads
            .iter()
            .any(|payload| { payload.len() == 16 && payload.iter().all(|byte| *byte == 0x27) }));
        assert!(trace.readback_lengths.contains(&16));
    }

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_submission_rejects_state_overwritten_later_in_the_same_wave() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(true, ProviderBehavior::Success),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
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
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x27,
    );

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("overwritten state witness unexpectedly reached provider encode"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("is overwritten later in the same terminal readback scope")
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
fn determinism_submission_rejects_restore_for_a_different_wave_scope() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    let node_ids = [id("node.tail")];
    let lane = Arc::clone(step.execution_lane());
    let source_wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let source_active_bindings = wave_active_bindings(&source_wave, &session);
    let source_providers = source_wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let source_batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        source_active_bindings.iter(),
        &source_wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &source_providers,
        &source_batch_identity,
        &source_active_bindings,
        &source_wave,
        0x27,
    );
    drop(source_providers);
    drop(source_active_bindings);
    drop(source_wave);
    step.try_retire_normal().unwrap();
    let target_step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );

    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &target_step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let reaper = CompletionReaper::new();
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("foreign determinism restore scope unexpectedly reached provider encode"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("differs from its prepared wave or physical batch identity")
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, target_step);
}

#[test]
fn determinism_single_node_replay_preserves_the_immutable_plan_node_index() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let node_ids = [id("node.tail")];
    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x42,
    );
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("single-node determinism probe has a reusable program identity");
    let program = DeviceReusableExecutionProgram::new(
        program_id,
        vec![DeviceReusableExecutionSegment::new(0, 0, 1, 1).unwrap()],
        vec![0],
    )
    .unwrap();

    let handle = OperationDispatch::encode_and_submit_determinism_replayed_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Replay,
        &restore,
        0xa5,
        &program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(handle.readback_plan().node_ids(), &node_ids);
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .program_binding_slots
            .keys()
            .copied()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([1])
    );
    assert!(matches!(
        handle.wait_with_determinism_readback().unwrap(),
        CompletionReadbackCollectionObservation::Terminal(_)
    ));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_single_node_scopes_cannot_alias_static_reusable_program_identity() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let lane = Arc::clone(step.execution_lane());

    let main_node_ids = [id("node.main")];
    let main_wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &main_node_ids,
    );
    let main_providers = main_wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let main_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &main_providers,
        &fixture.resolved,
        &main_wave,
        &lane,
    )
    .unwrap()
    .expect("static main-node probe has a reusable program identity");
    drop(main_providers);
    drop(main_wave);
    step.try_retire_normal().unwrap();

    let tail_node_ids = [id("node.tail")];
    let tail_step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let tail_wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &tail_step,
        &tail_node_ids,
    );
    let tail_providers = tail_wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let tail_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &tail_providers,
        &fixture.resolved,
        &tail_wave,
        &lane,
    )
    .unwrap()
    .expect("static tail-node probe has a reusable program identity");
    assert_ne!(main_program_id, tail_program_id);

    drop(tail_providers);
    drop(tail_wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, tail_step);
}

#[test]
fn determinism_probe_wave_cannot_escape_through_the_product_dispatch_path() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
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

    let error = match OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("determinism probe wave unexpectedly escaped through product dispatch"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("purpose differs from its product or determinism dispatch path")
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
fn determinism_dispatch_rejects_a_product_wave_before_provider_encode() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
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
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x42,
    );

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0xa5,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("product wave unexpectedly entered determinism dispatch"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("purpose differs from its product or determinism dispatch path")
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
fn determinism_readback_rejects_a_foreign_plan_before_submission() {
    let foreign_fixture =
        fixture_with_determinism_provider_behavior(true, ProviderBehavior::Success);
    let foreign_witness_plan = foreign_fixture.plan.determinism_witness_plan().unwrap();
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    assert_ne!(foreign_witness_plan.plan_hash(), fixture.plan.plan_hash());
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
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
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x27,
    );

    let error = SubmissionWaveDeterminismReadbackPlan::from_restore(
        &foreign_fixture.resolved,
        &batch_identity,
        &wave,
        &restore,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("differs from the exact immutable plan initialization denominator"));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);

    drop(providers);
    drop(active_bindings);
    drop(lane);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);

    drop(foreign_fixture.registry);
    drop(foreign_fixture.impostor_registry);
    drop(foreign_fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(foreign_fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn kernel_profile_binds_native_work_to_exact_plan_nodes() {
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
    let timing = RecordingSubmissionTimingSink::default();

    let profiled = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Kernel,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        &timing,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (handle, attribution) = profiled.into_parts();
    let attribution = attribution.expect("kernel profile must return bound native work");
    assert_eq!(
        attribution.submission_fingerprint(),
        handle.receipt().fingerprint()
    );
    let node_rows = attribution
        .device()
        .commands()
        .iter()
        .filter(|command| command.node_index().is_some())
        .collect::<Vec<_>>();
    assert_eq!(node_rows.len(), batch_identity.nodes().len());
    for (expected_index, command) in node_rows.into_iter().enumerate() {
        assert_eq!(command.node_index(), Some(expected_index as u32));
        assert_eq!(command.native_op_id(), "test_provider");
        assert_eq!(command.execution_path(), DeviceExecutionPath::Eager);
        assert_eq!(command.batching_form(), DeviceBatchingForm::Scalar);
        assert_eq!(command.participant_count(), 1);
        assert_eq!(command.token_count(), 1);
        assert_eq!(command.compute_dispatch_count(), 1);
        assert_eq!(command.transfer_command_count(), 0);
    }
    let completion = match handle.wait().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("kernel-profiled wave did not terminate: {other:?}"),
    };
    let attribution = attribution
        .bind_terminal_timing(completion.submission_timing().clone())
        .unwrap();
    let DeviceTimingMeasurement::Measured(timing) = attribution.terminal_timing() else {
        panic!("kernel-profiled wave must bind terminal command timing")
    };
    assert_eq!(
        timing.command_count() as usize,
        attribution.device().commands().len()
    );
    for (timing, command) in timing.spans().iter().zip(attribution.device().commands()) {
        assert_eq!(timing.start_command_index(), command.command_index());
        assert_eq!(timing.end_command_index(), command.command_index() + 1);
        assert_eq!(
            timing.kind(),
            ferrum_interfaces::vnext::DeviceExecutionSpanKind::EagerCommand
        );
        assert_eq!(timing.measurement().intervals().unwrap().len(), 1);
        assert_eq!(timing.measurement().elapsed_ns(), Some(10));
    }

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
    let lane = Arc::clone(step.execution_lane());
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
fn terminal_wave_reads_multiple_canonical_node_groups_under_one_fence() {
    let (fixture, sequence, session, batch, step) = setup();
    let executable = ExecutablePlan::new(
        fixture.plan.clone(),
        fixture.resolved.parts().capabilities.clone(),
    )
    .unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
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
    let group = |node: &str, resource: &str| {
        CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
            id(node),
            0,
            id(resource),
            0,
            HostTransferLayout::new(ElementType::F32, 4).unwrap(),
        )
        .unwrap()])
        .unwrap()
    };
    assert!(CompletionReadbackCollectionRequest::new(Vec::new()).is_err());
    assert!(CompletionReadbackCollectionRequest::new(vec![
        group("node.tail", "resource.output"),
        group("node.tail", "resource.output"),
    ])
    .is_err());
    let collection = CompletionReadbackCollectionRequest::new(vec![
        group("node.tail", "resource.output"),
        group("node.main", "resource.intermediate"),
    ])
    .unwrap();
    assert_eq!(collection.len(), 2);
    assert_eq!(collection.request_count(), 2);

    let receipt = match handle.wait_with_readback_collection(collection).unwrap() {
        CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
        other => panic!("wave collection readback did not terminate: {other:?}"),
    };
    assert_eq!(receipt.dispositions().len(), 2);
    assert!(receipt
        .dispositions()
        .iter()
        .all(|disposition| matches!(disposition, CompletionReadbackDisposition::Succeeded(_))));
    let groups = receipt
        .dispositions()
        .iter()
        .map(|disposition| match disposition {
            CompletionReadbackDisposition::Succeeded(output) => (
                output.request().node_id().as_str(),
                output.request().resource_id().as_str(),
            ),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        groups,
        vec![
            ("node.main", "resource.intermediate"),
            ("node.tail", "resource.output"),
        ]
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
fn provider_declared_binding_compute_and_result_phases_share_one_wave() {
    let (fixture, sequence, session, batch, step) = setup();
    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::SplitPhases;
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

    let trace = fixture.runtime_trace.lock().unwrap();
    assert_eq!(trace.submitted_command_counts, vec![6]);
    assert_eq!(
        trace.submitted_command_phases,
        vec![vec![
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::ResultBinding,
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::ResultBinding,
        ]]
    );
    drop(trace);
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
fn provider_program_bindings_are_coalesced_once_before_all_wave_compute() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    assert!(batch_identity.nodes().iter().all(|node| {
        node.provider_execution_semantics()
            == ProviderExecutionSemantics::bitwise_eager_and_replay()
    }));
    let expected_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave must expose reusable capture identity");

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
        assert_eq!(trace.program_binding_coalesce_calls, 1);
        assert_eq!(trace.program_binding_input_counts, vec![2]);
        assert_eq!(trace.submitted_command_counts, vec![3]);
        assert_eq!(
            trace.submitted_command_phases,
            vec![vec![
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::Compute,
            ]]
        );
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::Provider,
                TestCommand::Provider,
            ]]
        );
        assert_eq!(trace.submitted_reusable_captures.len(), 1);
        let capture = trace.submitted_reusable_captures[0]
            .as_ref()
            .expect("full encode must carry reusable capture metadata");
        assert_eq!(capture.program_id(), &expected_program_id);
        assert_eq!(capture.per_wave_binding_node_indices(), &[0, 1]);
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.program_binding_slots.len(), 2);
        for (node_index, node) in fixture.plan.payload().nodes().iter().enumerate() {
            assert_eq!(
                trace.program_binding_slots.get(&node_index),
                node.binding_resource()
            );
        }
        assert_eq!(
            trace.program_binding_plan_hashes,
            BTreeSet::from([fixture.plan.plan_hash().as_str().to_owned()])
        );
        assert_eq!(trace.program_binding_layout_fingerprints.len(), 1);
        assert_eq!(trace.program_binding_lane_slot_ids.len(), 1);
        assert_eq!(
            trace.program_binding_lifetimes,
            vec![
                AllocationLifetime::Invocation,
                AllocationLifetime::Invocation
            ]
        );
    }
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
fn sealed_reusable_program_encodes_only_bindings_and_one_direct_segment() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave must have a reusable program identity");
    let node_count = u32::try_from(providers.len()).unwrap();
    let segment = DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap();
    let program =
        DeviceReusableExecutionProgram::new(program_id, vec![segment], (0..node_count).collect())
            .unwrap();

    let handle = OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_policy(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &program,
        SubmissionExecutionPolicy::determinism_replayed(0xa5),
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(!handle.receipt().has_materialized_participant_receipts());

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.program_binding_coalesce_calls, 1);
        assert_eq!(trace.program_binding_input_counts, vec![2]);
        assert_eq!(trace.submitted_command_counts, vec![2]);
        assert_eq!(
            trace.submitted_command_phases,
            vec![vec![
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::Compute,
            ]]
        );
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::ReusableExecution,
            ]]
        );
        assert_eq!(trace.submitted_reusable_captures, vec![None]);
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::ReplayedOnly]
        );
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 0);
        assert_eq!(trace.reusable_binding_encode_calls, 2);
        assert_eq!(trace.program_binding_slots.len(), 2);
    }
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert!(!batch_identity
        .materialization_snapshot()
        .full_participant_projection());

    drop(handle);
    drop(topology);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_replay_submission_restores_state_and_enforces_replayed_compute() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
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
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x42,
    );
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave has a reusable identity");
    let node_count = u32::try_from(providers.len()).unwrap();
    let program = DeviceReusableExecutionProgram::new(
        program_id,
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        (0..node_count).collect(),
    )
    .unwrap();

    let handle = OperationDispatch::encode_and_submit_determinism_replayed_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Replay,
        &restore,
        0xa5,
        &program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let attribution = handle
        .attribution()
        .expect("determinism replay must retain actual-path attribution");
    assert_eq!(attribution.device().replayed_segments().len(), 1);
    assert_eq!(
        attribution.device().replayed_segments()[0]
            .logical_commands()
            .len(),
        providers.len()
    );

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::ReplayedOnly]
        );
        assert_eq!(
            trace.submitted_attribution_requirements,
            vec![DeviceSubmissionAttributionRequirement::LogicalExecutionPath]
        );
        assert!(trace
            .uploaded_payloads
            .iter()
            .any(|payload| !payload.is_empty() && payload.iter().all(|byte| *byte == 0x42)));
        let phases = &trace.submitted_command_phases[0];
        let first_compute = phases
            .iter()
            .position(|phase| *phase == DeviceCommandPhase::Compute)
            .expect("determinism replay has one compute command");
        assert!(phases[..first_compute]
            .iter()
            .all(|phase| *phase != DeviceCommandPhase::Compute));
        assert_eq!(phases[first_compute..], [DeviceCommandPhase::Compute]);
        assert_eq!(
            trace.submitted_commands[0][first_compute],
            TestCommand::ReusableExecution
        );
    }
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    let expected_readbacks = handle.readback_plan().collection_request().request_count();
    let evidence = handle.wait_into_evidence().unwrap();
    assert_eq!(
        evidence.expected_execution_path(),
        DeviceExecutionPath::Replayed
    );
    assert_eq!(evidence.physical_readbacks().len(), expected_readbacks);
    assert!(evidence
        .physical_readbacks()
        .iter()
        .all(|readback| !readback.bytes().is_empty() && readback.raw_sha256().len() == 64));

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn reusable_topology_states_cannot_alias_resident_program_authority() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    let dynamic_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("dynamic topology must produce a reusable program identity");

    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let static_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("static topology must retain the base reusable program identity");
    assert_ne!(dynamic_program_id, static_program_id);

    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::ProgramBindingIneligible;
    assert!(
        OperationDispatch::reusable_execution_program_id_for_wave(
            &providers,
            &fixture.resolved,
            &wave,
            &lane,
        )
        .unwrap()
        .is_none(),
        "one ineligible provider must veto resident reuse for the complete wave"
    );

    drop(providers);
    drop(wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn eager_only_provider_cannot_authorize_reusable_topology() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_provider_behavior_and_execution_semantics(
            false,
            ProviderBehavior::ProgramBinding,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ExecutionDeterminismRequirement::BitwiseSameRuntime,
        ));
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    let error = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("without a bitwise eager-equivalence contract"));
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert!(fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submitted_command_counts
        .is_empty());

    drop(providers);
    drop(wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn stale_reusable_topology_is_rejected_before_dispatch_or_encoding() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let batch_identity = OperationDispatch::bind_compiled_submission_wave_identity(
        &topology,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let live_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave must have a reusable program identity");
    let stale_program_id = live_program_id.with_topology_fingerprint(
        DeviceReusableExecutionTopologyFingerprint::from_sha256([0xff; 32]),
    );
    let node_count = u32::try_from(providers.len()).unwrap();
    let stale_program = DeviceReusableExecutionProgram::new(
        stale_program_id,
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        (0..node_count).collect(),
    )
    .unwrap();

    let error = OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &stale_program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains(
                "reusable execution program differs from the exact wave topology"
            )
    ));
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 0);
        assert!(trace.submitted_commands.is_empty());
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 0);
        assert_eq!(trace.reusable_binding_encode_calls, 0);
    }
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(stale_program);
    drop(topology);
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
