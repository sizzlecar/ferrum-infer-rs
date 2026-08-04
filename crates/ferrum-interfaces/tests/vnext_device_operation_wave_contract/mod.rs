use super::vnext_device_operation_contract::*;

pub(super) fn test_reusable_program(
    program_id: DeviceReusableExecutionProgramId,
    node_count: u32,
    eager_boundary_node_indices: Vec<u32>,
    segments: Vec<DeviceReusableExecutionSegment>,
    per_wave_binding_node_indices: Vec<u32>,
    gaps: Vec<DeviceReusableExecutionProgramGap>,
) -> DeviceReusableExecutionProgram {
    let capture = DeviceReusableExecutionCapture::new(
        program_id,
        node_count,
        eager_boundary_node_indices,
        per_wave_binding_node_indices.clone(),
    )
    .unwrap();
    DeviceReusableExecutionProgram::new(&capture, segments, per_wave_binding_node_indices, gaps)
        .unwrap()
}

#[derive(Default)]
pub(super) struct RecordingSubmissionTimingSink {
    pub(super) stages: Mutex<Vec<SubmissionWaveDispatchStage>>,
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

pub(super) fn prepare_wave(
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

pub(super) fn prepare_determinism_wave(
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

pub(super) fn prepare_determinism_wave_for_nodes(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    plan: &ExecutionPlan,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_ids: &[NodeId],
) -> PreparedStepSubmissionWave<TestRuntime> {
    prepare_wave_for_node_scope(plan_resources, plan, step, node_ids, true)
}

pub(super) fn prepare_wave_for_node_scope(
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

pub(super) fn setup() -> (
    Fixture,
    Arc<AdmittedSequenceResources<TestRuntime>>,
    Arc<SequenceSession<TestRuntime>>,
    ExecutionBatchParticipants<TestRuntime>,
    Arc<StepResourceLease<TestRuntime>>,
) {
    setup_with_fixture(fixture())
}

pub(super) fn setup_with_fixture(
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

pub(super) fn wave_active_bindings(
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    session: &Arc<SequenceSession<TestRuntime>>,
) -> Vec<TrustedActiveSequenceBinding> {
    let active = TrustedActiveSequenceBinding::from_session(session).unwrap();
    for node in wave.nodes() {
        assert_eq!(node.participant_frames().len(), 1);
    }
    vec![active]
}

pub(super) fn determinism_restore(
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

pub(super) fn determinism_payloads(
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

pub(super) fn teardown(
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
