use super::*;

struct IgnorePreparation;

impl InvocationPreparationSink for IgnorePreparation {
    fn record_preparation(&self, _stats: InvocationPreparationStats) {}
}

fn submit(
    prepared: PreparedSubmissionWaveTopology<'_, '_, TestRuntime>,
    identity: &BatchOperationIdentity,
    bindings: &[TrustedActiveSequenceBinding],
    program: Option<&DeviceReusableExecutionProgram>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> Result<CompletionHandle<TestRuntime>, SubmissionWaveDispatchError<TestRuntime>> {
    prepared
        .encode_and_submit_with_inputs_and_preparation(
            identity,
            bindings.iter(),
            DeviceTimingMode::Off,
            &[],
            program,
            SubmissionExecutionPolicy::adaptive(),
            ferrum_types::InvocationPreparationStrategy::Full,
            &RecordingSubmissionTimingSink::default(),
            &IgnorePreparation,
            reaper,
        )
        .map(|profiled| profiled.into_parts().0)
}

#[test]
fn prepared_topology_keeps_fingerprint_and_eager_boundaries_with_one_query_per_node() {
    for behavior in [
        ProviderBehavior::ProgramBindingWithScratchTail,
        ProviderBehavior::ProgramBinding,
        ProviderBehavior::ProgramBindingFirstNodeEagerBoundary,
    ] {
        let (fixture, sequence, session, batch, step) =
            setup_with_fixture(fixture_with_provider_behavior(false, behavior));
        fixture.provider_trace.lock().unwrap().topology_stability =
            ReusableTopologyStability::SameWaveStable;
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let bindings = wave_active_bindings(&wave, &session);
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
            bindings.iter(),
            &wave,
            &lane,
        )
        .unwrap();
        let expected = OperationDispatch::reusable_execution_program_id_for_wave(
            &providers,
            &fixture.resolved,
            &wave,
            &lane,
        )
        .unwrap();
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls = 0;
        let prepared = OperationDispatch::prepare_submission_wave_topology(
            &providers,
            &fixture.resolved,
            wave,
            &lane,
        )
        .unwrap();
        assert_eq!(prepared.program_id(), expected.as_ref());
        assert_eq!(
            prepared.program_id(),
            expected.as_ref(),
            "lookup must not recompute topology"
        );
        let handle = submit(prepared, &identity, &bindings, None, &reaper).unwrap();
        assert_eq!(
            fixture
                .provider_trace
                .lock()
                .unwrap()
                .reusable_topology_calls,
            providers.len() as u64
        );
        {
            let trace = fixture.runtime_trace.lock().unwrap();
            let capture = trace.submitted_reusable_captures[0].as_ref().unwrap();
            assert_eq!(Some(capture.program_id()), expected.as_ref());
            let boundaries: &[u32] =
                if behavior == ProviderBehavior::ProgramBindingFirstNodeEagerBoundary {
                    &[0]
                } else {
                    &[]
                };
            assert_eq!(capture.eager_boundary_node_indices(), boundaries);
        }
        assert!(matches!(
            handle.wait().unwrap(),
            CompletionObservation::Terminal(_)
        ));
        drop(handle);
        drop(providers);
        drop(bindings);
        drop(reaper);
        drop(lane);
        teardown(fixture, sequence, session, batch, step);
    }
}

#[test]
fn prepared_topology_revalidates_mutable_provider_before_replay() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let bindings = wave_active_bindings(&wave, &session);
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
        bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let prepared = OperationDispatch::prepare_submission_wave_topology(
        &providers,
        &fixture.resolved,
        wave,
        &lane,
    )
    .unwrap();
    let count = providers.len() as u32;
    let program = test_reusable_program(
        prepared.program_id().unwrap().clone(),
        count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap()],
        (0..count).collect(),
        vec![],
    );
    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let error = submit(prepared, &identity, &bindings, Some(&program), &reaper).unwrap_err();
    assert!(error
        .to_string()
        .contains("differs from the exact wave topology"));
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls,
        2 * u64::from(count)
    );
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    drop(providers);
    drop(bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn prepared_topology_without_lookup_queries_uncertified_provider_only_at_submit() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let bindings = wave_active_bindings(&wave, &session);
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
        bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let prepared = OperationDispatch::defer_submission_wave_topology(
        &providers,
        &fixture.resolved,
        wave,
        &lane,
    );
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls,
        0
    );
    *fixture.provider_behavior.lock().unwrap() =
        ProviderBehavior::ProgramBindingFirstNodeEagerBoundary;
    let handle = submit(prepared, &identity, &bindings, None, &reaper).unwrap();
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls,
        providers.len() as u64
    );
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_reusable_captures[0]
            .as_ref()
            .unwrap()
            .eager_boundary_node_indices(),
        &[0]
    );
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(handle);
    drop(providers);
    drop(bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn prepared_topology_drops_wave_on_error_and_on_unused_preparation() {
    for fail in [false, true] {
        let fixture = if fail {
            fixture_with_provider_behavior_and_execution_semantics(
                false,
                ProviderBehavior::ProgramBinding,
                ProviderExecutionSemantics::bitwise_eager_only(),
                ExecutionDeterminismRequirement::BitwiseSameRuntime,
            )
        } else {
            fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding)
        };
        let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture);
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let lane = Arc::clone(step.execution_lane());
        let providers = fixture
            .plan
            .payload()
            .nodes()
            .iter()
            .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
            .collect::<Vec<_>>();
        let prepared = OperationDispatch::prepare_submission_wave_topology(
            &providers,
            &fixture.resolved,
            wave,
            &lane,
        );
        let step = if fail {
            let error = prepared
                .err()
                .expect("ineligible topology must fail preparation");
            assert!(error
                .to_string()
                .contains("without a bitwise eager-equivalence contract"));
            step
        } else {
            let prepared = prepared.unwrap();
            // The prepared topology must retain the original wave's Step.
            let step = match step.try_retire_normal() {
                Err(failure) => failure.into_step(),
                Ok(_) => panic!("an owned unsubmitted wave must prevent Step retirement"),
            };
            drop(prepared);
            step
        };
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
        assert_eq!(lane.in_flight_count(), 0);
        drop(providers);
        drop(lane);
        // Preparing a wave creates invocation-registry entries, so pristine
        // pre-invocation rollback is deliberately unavailable. Use the same
        // terminal retirement/plan-close checks as the existing dropped-wave
        // tests; retained flights or owners would prevent this teardown.
        teardown(fixture, sequence, session, batch, step);
    }
}

#[test]
fn prepared_topology_retry_requires_a_new_observation_and_current_identity() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    fixture.provider_trace.lock().unwrap().topology_stability =
        ReusableTopologyStability::SameWaveStable;
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let first_attempt = wave.batch_invocation_id();
    let bindings = wave_active_bindings(&wave, &session);
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
        bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let prepared = OperationDispatch::prepare_submission_wave_topology(
        &providers,
        &fixture.resolved,
        wave,
        &lane,
    )
    .unwrap();
    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::DefinitelyNotSubmitted;
    let retry = match submit(prepared, &identity, &bindings, None, &reaper) {
        Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        other => panic!("expected definitely-not-submitted result: {other:?}"),
    };
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls,
        providers.len() as u64
    );
    let retry_wave = retry.retry().unwrap();
    assert_ne!(retry_wave.batch_invocation_id(), first_attempt);
    let retry_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        bindings.iter(),
        &retry_wave,
        &lane,
    )
    .unwrap();
    let prepared = OperationDispatch::prepare_submission_wave_topology(
        &providers,
        &fixture.resolved,
        retry_wave,
        &lane,
    )
    .unwrap();
    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::Success;
    let handle = submit(prepared, &retry_identity, &bindings, None, &reaper).unwrap();
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls,
        2 * providers.len() as u64
    );
    assert!(fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submitted_reusable_captures
        .last()
        .unwrap()
        .as_ref()
        .unwrap()
        .eager_boundary_node_indices()
        .is_empty());
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(handle);
    drop(providers);
    drop(bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
