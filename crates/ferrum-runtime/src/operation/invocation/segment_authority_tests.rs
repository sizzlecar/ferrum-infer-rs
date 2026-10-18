//! The common segment gate consumes real admitted waves and active sessions.
use super::*;
use ferrum_types::InvocationPreparationStrategy;

pub(super) fn segment_identity(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    active: &[TrustedActiveSequenceBinding],
) -> BatchOperationIdentity {
    let lane = wave.step_resources().execution_lane();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
    OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        wave,
        lane,
        InvocationPreparationStrategy::DecodeSegment,
    )
    .unwrap()
}

fn validate(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    identity: &BatchOperationIdentity,
    active: &[TrustedActiveSequenceBinding],
) -> Result<(), VNextError> {
    validate_segment_wave_authority(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        identity,
        wave,
        active.iter(),
    )
}

#[test]
fn segment_wave_authority_matches_constructor_and_rejects_full_or_reordered_active() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, full, active| {
        let identity = segment_identity(fixture, wave, active);
        assert_eq!(identity.materialization_snapshot().materialized_nodes(), 0);
        validate(fixture, wave, &identity, active).unwrap();
        assert_eq!(identity.materialization_snapshot().materialized_nodes(), 1);
        assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
        assert!(validate(fixture, wave, full, active).is_err());
        let reversed = active.iter().rev().cloned().collect::<Vec<_>>();
        assert!(validate(fixture, wave, &identity, &reversed).is_err());
        assert!(validate(fixture, wave, &identity, &active[..1]).is_err());
        for (node_index, prepared) in wave.nodes().iter().enumerate() {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, prepared.node_id())
                .unwrap();
            let node = identity.materialize_node(node_index).unwrap();
            let make = |bindings: &[TrustedActiveSequenceBinding]| {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    &identity,
                    node,
                    OperationInvocationResources::Wave { wave, node_index },
                    bindings.iter(),
                    false,
                    true,
                )
            };
            assert!(make(active).is_ok());
            assert!(make(&reversed).is_err());
        }
        assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn segment_wave_authority_rejects_previous_frames_and_replaced_live_sessions() {
    let fixture = fixture();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let sequences = (0..2)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.segment.old.{i}"),
                &format!("request.segment.old.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = sequences
        .iter()
        .map(|sequence| sequence.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let active = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let first_step = step_for(&batch, &lane, vec![one_token_span(), one_token_span()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &first_step);
    let previous = segment_identity(&fixture, &wave, &active);
    validate(&fixture, &wave, &previous, &active).unwrap();
    let frames = wave.nodes()[0]
        .participant_frames()
        .iter()
        .map(|frame| frame.frame_id())
        .collect::<Vec<_>>();
    drop(wave);
    first_step.try_retire_normal().unwrap();
    // Retirement advances the live session snapshot; mint current active bindings.
    let active = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();

    let step = step_for(&batch, &lane, vec![one_token_span(), one_token_span()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    assert!(wave.nodes()[0]
        .participant_frames()
        .iter()
        .zip(&frames)
        .all(|(current, old)| current.frame_id() > *old));
    let current = segment_identity(&fixture, &wave, &active);
    assert!(validate(&fixture, &wave, &previous, &active).is_err());
    validate(&fixture, &wave, &current, &active).unwrap();
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(sequences);

    let sequences = (0..2)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.segment.new.{i}"),
                &format!("request.segment.new.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = sequences
        .iter()
        .map(|sequence| sequence.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let fresh = batch
        .sessions()
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let step = step_for(&batch, &lane, vec![one_token_span(), one_token_span()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let identity = segment_identity(&fixture, &wave, &fresh);
    assert_ne!(
        fresh[1].sequence_authority(),
        active[1].sequence_authority()
    );
    let mixed = vec![fresh[0].clone(), active[1].clone()];
    assert!(validate(&fixture, &wave, &identity, &mixed).is_err());
    assert!(validate(&fixture, &wave, &current, &fresh).is_err());
    validate(&fixture, &wave, &identity, &fresh).unwrap();
    let provider = fixture
        .registry
        .bind(&fixture.resolved, wave.nodes()[0].node_id())
        .unwrap();
    assert!(BatchedOperationInvocation::from_resources(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        provider.dispatch(),
        &identity,
        identity.materialize_node(0).unwrap(),
        OperationInvocationResources::Wave {
            wave: &wave,
            node_index: 0
        },
        mixed.iter(),
        false,
        true,
    )
    .is_err());
    drop(provider);
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
}
