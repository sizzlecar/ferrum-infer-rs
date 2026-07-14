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
    assert_eq!(wave.nodes()[0].claimed_backing().fingerprint().len(), 64);

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
