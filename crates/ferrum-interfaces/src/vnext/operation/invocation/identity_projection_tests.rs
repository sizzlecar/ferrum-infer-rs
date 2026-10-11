//! Real admitted waves exercise the projection without manufacturing authority.
use super::*;
use ferrum_types::InvocationPreparationStrategy;
use std::sync::Barrier;
use vnext_device_operation_wave_contract::{setup, teardown, wave_active_bindings};

fn projected_identity(
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
        InvocationPreparationStrategy::IdentityProjection,
    )
    .unwrap()
}

fn first_envelope(identity: &BatchOperationIdentity) -> &ExecutionIdentityEnvelope {
    identity.materialize_node(0).unwrap().participants()[0].identity()
}

#[test]
fn identity_projection_preserves_owned_wire_debug_and_cold_warm_equality() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, eager, active| {
        let projected = projected_identity(fixture, wave, active);
        let separately_bound = projected_identity(fixture, wave, active);
        for node_index in 0..wave.node_count() {
            let projected_node = projected.materialize_node(node_index).unwrap();
            let eager_node = eager.materialize_node(node_index).unwrap();
            let other_node = separately_bound.materialize_node(node_index).unwrap();
            for ((projection, reference), independent) in projected_node
                .participants()
                .iter()
                .zip(eager_node.participants())
                .zip(other_node.participants())
            {
                let projection = projection.identity();
                let reference = reference.identity();
                let independent = independent.identity();
                assert_eq!(projection.projected_parts_materialized(), Some(false));
                assert_eq!(reference.projected_parts_materialized(), None);
                let shared_clone = projection.clone();
                assert_eq!(*projection, shared_clone);
                assert_eq!(projection.projected_parts_materialized(), Some(false));

                // The legacy envelope wire is the flat parts struct; its Debug
                // was the derived one-field wrapper, not a representation tag.
                let expected_wire = serde_json::to_vec(reference.parts()).unwrap();
                let expected_debug = format!(
                    "ExecutionIdentityEnvelope {{ parts: {:?} }}",
                    reference.parts()
                );
                assert_eq!(serde_json::to_vec(projection).unwrap(), expected_wire);
                assert_eq!(projection.projected_parts_materialized(), Some(true));
                assert_eq!(projection.parts(), reference.parts());
                assert_eq!(format!("{projection:?}"), expected_debug);
                assert_eq!(projection, reference);
                assert_eq!(reference, projection);
                assert_eq!(projection, independent);
                assert_eq!(independent, reference);
                assert_eq!(serde_json::to_vec(&shared_clone).unwrap(), expected_wire);
            }
        }
        let node = projected.materialize_node(0).unwrap();
        assert_ne!(
            node.participants()[0].identity(),
            node.participants()[1].identity(),
            "different real request/sequence authorities must not compare equal"
        );
        assert_ne!(
            node.participants()[0].identity(),
            projected.materialize_node(1).unwrap().participants()[0].identity(),
            "the same participant at a different actual plan node is distinct"
        );
    });
}

#[test]
fn identity_projection_concurrent_first_parts_and_wire_share_one_published_value() {
    with_live_wave(2, |fixture, wave, eager, active| {
        let projected = projected_identity(fixture, wave, active);
        let identity = first_envelope(&projected);
        let expected = serde_json::to_vec(first_envelope(eager).parts()).unwrap();
        let barrier = Barrier::new(4);
        assert_eq!(identity.projected_parts_materialized(), Some(false));
        let pointers = std::thread::scope(|scope| {
            (0..4)
                .map(|index| {
                    let identity = identity.clone();
                    let barrier = &barrier;
                    let expected = &expected;
                    scope.spawn(move || {
                        barrier.wait();
                        if index % 2 == 0 {
                            assert_eq!(serde_json::to_vec(&identity).unwrap(), *expected);
                        } else {
                            assert_eq!(serde_json::to_vec(identity.parts()).unwrap(), *expected);
                        }
                        identity.parts() as *const ExecutionIdentityParts as usize
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(pointers.iter().all(|pointer| *pointer == pointers[0]));
        assert_eq!(identity.projected_parts_materialized(), Some(true));
    });
}

#[test]
fn identity_projection_successful_views_stay_lazy_and_reject_changed_current_inputs() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, eager, active| {
        let projected = projected_identity(fixture, wave, active);
        for node_index in 0..wave.node_count() {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[node_index].node_id())
                .unwrap();
            let projection = projected.materialize_node(node_index).unwrap();
            let reference = eager.materialize_node(node_index).unwrap();
            let make = |identity, node, bindings: &[TrustedActiveSequenceBinding]| {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    node,
                    OperationInvocationResources::Wave { wave, node_index },
                    bindings.iter(),
                    false,
                    true,
                )
            };
            let reference_view = make(eager, reference, active).unwrap();
            let projected_view = make(&projected, projection, active).unwrap();
            assert_eq!(snapshot(&projected_view), snapshot(&reference_view));
            assert!(projection
                .participants()
                .iter()
                .all(
                    |participant| participant.identity().projected_parts_materialized()
                        == Some(false)
                ));
            assert_eq!(projected.preparation_snapshot().parts_materialized, 0);
            drop(projected_view);
            drop(reference_view);

            let mut reversed = active.to_vec();
            reversed.reverse();
            assert!(make(&projected, projection, &reversed).is_err());
            fixture
                .runtime_trace
                .lock()
                .unwrap()
                .tamper_buffer_descriptor = true;
            assert!(make(&projected, projection, active).is_err());
            fixture
                .runtime_trace
                .lock()
                .unwrap()
                .tamper_buffer_descriptor = false;
            assert!(make(&projected, projection, active).is_ok());
        }
        assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn identity_projection_rejects_previous_frame_on_the_same_live_sequence() {
    let (fixture, sequence, session, batch, first_step) = setup();
    let lane = Arc::clone(first_step.execution_lane());
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &first_step);
    let active = wave_active_bindings(&wave, &session);
    let previous = projected_identity(&fixture, &wave, &active);
    let old_node = previous.materialize_node(0).unwrap().clone();
    let first_frame = old_node.participants()[0].node_key().frame_id();
    drop(wave);
    first_step.try_retire_normal().unwrap();

    let step = step_for(&batch, &lane, vec![one_token_span()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let current = projected_identity(&fixture, &wave, &active);
    let fresh_node = current.materialize_node(0).unwrap();
    assert!(fresh_node.participants()[0].node_key().frame_id() > first_frame);
    let provider = fixture
        .registry
        .bind(&fixture.resolved, wave.nodes()[0].node_id())
        .unwrap();
    let make = |node| {
        BatchedOperationInvocation::from_resources(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            &current,
            node,
            OperationInvocationResources::Wave {
                wave: &wave,
                node_index: 0,
            },
            active.iter(),
            false,
            true,
        )
    };
    assert!(make(&old_node).is_err());
    assert!(make(fresh_node).is_ok());
    assert_ne!(
        old_node.participants()[0].identity(),
        fresh_node.participants()[0].identity()
    );
    drop(provider);
    drop(current);
    drop(wave);
    drop(active);
    drop(lane);
    // Retaining both old projected identities must not retain a lane, session,
    // backing allocation, or logical resource lease needed by real close.
    teardown(fixture, sequence, session, batch, step);
    assert_eq!(
        first_envelope(&previous).parts().frame_id,
        Some(first_frame)
    );
    assert_eq!(
        old_node.participants()[0].identity().parts().frame_id,
        Some(first_frame)
    );
}

#[test]
fn identity_projection_retained_metadata_does_not_authorize_a_new_admission() {
    let (fixture, sequence, session, batch, step) = setup();
    let lane = Arc::clone(step.execution_lane());
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let old_active = wave_active_bindings(&wave, &session);
    let old_authority = session.sequence_authority();
    let previous = projected_identity(&fixture, &wave, &old_active);
    let old_node = previous.materialize_node(0).unwrap().clone();
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);

    let sequence = logical_resources(
        &fixture.plan_resources,
        "run.projection.next-admission",
        "request.projection.next-admission",
    );
    let session = sequence.open_session().unwrap();
    assert_ne!(session.sequence_authority(), old_authority);
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = step_for(&batch, &lane, vec![one_token_span()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    assert!(
        OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
            &topology,
            old_active.iter(),
            &wave,
            &lane,
            InvocationPreparationStrategy::IdentityProjection,
        )
        .is_err()
    );
    let current = projected_identity(&fixture, &wave, &active);
    let fresh_node = current.materialize_node(0).unwrap();
    let provider = fixture
        .registry
        .bind(&fixture.resolved, wave.nodes()[0].node_id())
        .unwrap();
    let make = |node| {
        BatchedOperationInvocation::from_resources(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            &current,
            node,
            OperationInvocationResources::Wave {
                wave: &wave,
                node_index: 0,
            },
            active.iter(),
            false,
            true,
        )
    };
    assert!(make(&old_node).is_err());
    assert!(make(fresh_node).is_ok());
    drop(provider);
    drop(current);
    drop(wave);
    drop(active);
    drop(old_active);
    drop(topology);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
    // The metadata is still readable after all actual runtime owners closed.
    assert!(first_envelope(&previous)
        .parts()
        .active_sequence_slot
        .is_some());
    assert!(old_node.participants()[0]
        .identity()
        .parts()
        .admission_generation
        .is_some());
}

#[test]
fn identity_projection_last_handle_releases_its_real_owner_without_a_recipe_cycle() {
    let mut retained = None;
    let mut alive = None;
    with_live_wave(2, |fixture, wave, _, active| {
        let projected = projected_identity(fixture, wave, active);
        let envelope = first_envelope(&projected).clone();
        alive = envelope.projected_owner_alive_probe();
        assert!(alive.as_ref().unwrap()());
        retained = Some(envelope);
    });
    let alive = alive.unwrap();
    let envelope = retained.take().unwrap();
    assert!(alive());
    // A caller may preserve an ordinary owned identity after the compiled
    // recipe and its participant metadata have been released.
    let owned = ExecutionIdentityEnvelope::new(envelope.parts().clone()).unwrap();
    assert_eq!(envelope, owned);
    drop(envelope);
    assert!(
        !alive(),
        "the real projected Arc must not participate in a cycle"
    );
    assert_eq!(owned.projected_parts_materialized(), None);
}

struct ProjectionOffTiming;
impl DeviceSubmissionTimingSink for ProjectionOffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("Off submission must not call device timing callbacks");
    }
}
impl SubmissionWaveDispatchTimingSink for ProjectionOffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("Off submission must not call host timing callbacks");
    }
}
#[derive(Default)]
struct PreparationCapture(Mutex<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for PreparationCapture {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.lock().unwrap().push(stats);
    }
}

#[test]
fn identity_projection_off_submission_stays_lazy_until_explicit_observation() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let identity = projected_identity(&fixture, &wave, &active);
    let expected = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let capture = PreparationCapture::default();
    let profiled = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::IdentityProjection,
        &ProjectionOffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (handle, _) = profiled.into_parts();
    let at_return = identity.preparation_snapshot();
    assert!(at_return.projected_identities > 0);
    assert_eq!(at_return.parts_materialized, 0);
    assert_eq!(*capture.0.lock().unwrap(), vec![at_return]);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(identity.preparation_snapshot().parts_materialized, 0);

    // A later public diagnostic observation is allowed to materialize parts.
    // It cannot rewrite the already emitted through-dispatch-return snapshot.
    assert_eq!(
        serde_json::to_vec(first_envelope(&identity)).unwrap(),
        serde_json::to_vec(first_envelope(&expected).parts()).unwrap()
    );
    assert!(identity.preparation_snapshot().parts_materialized > 0);
    assert_eq!(*capture.0.lock().unwrap(), vec![at_return]);
    drop(handle);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
