use super::*;
use ferrum_types::InvocationPreparationStrategy;

fn versioned_identity(
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
        InvocationPreparationStrategy::PoolVersion,
    )
    .unwrap()
}

#[test]
fn pool_version_actual_shared_views_match_full_at_unequal_windows() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, full, active| {
        let identity = versioned_identity(fixture, wave, active);
        let counts = super::super::super::preparation::PoolVersionCounts::default();
        for node_index in 0..wave.node_count() {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[node_index].node_id())
                .unwrap();
            let reference = BatchedOperationInvocation::from_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                full,
                full.materialize_node(node_index).unwrap(),
                wave,
                node_index,
                active.iter(),
            )
            .unwrap();
            let invocation = BatchedOperationInvocation::from_wave_node_with_pool_version(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                identity.materialize_node(node_index).unwrap(),
                wave,
                node_index,
                active.iter(),
                false,
                Some(&counts),
            )
            .unwrap();
            assert_eq!(snapshot(&reference), snapshot(&invocation));
            assert!(!Arc::ptr_eq(
                &reference.retained_dependency_scope,
                &invocation.retained_dependency_scope
            ));
        }
        assert!(counts.proofs.get() > 0);
        assert!(counts.hits.get() > 0);
        assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
    });
}

struct OffTiming;
impl DeviceSubmissionTimingSink for OffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("Off timing callback");
    }
}
impl SubmissionWaveDispatchTimingSink for OffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("Off timing callback");
    }
}
#[derive(Default)]
struct Capture(Mutex<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for Capture {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.lock().unwrap().push(stats);
    }
}

#[test]
fn pool_version_off_dispatch_counts_real_checks_and_releases_resources() {
    let fixture = fixture();
    let resources = (0..2)
        .map(|index| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.pool-version.{index}"),
                &format!("request.pool-version.{index}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|r| r.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = step_for(
        &batch,
        &lane,
        vec![
            TokenSpanWork::from_token_ids(&[1, 2, 3], 0..1).unwrap(),
            TokenSpanWork::from_token_ids(&[1, 2, 3], 0..3).unwrap(),
        ],
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = versioned_identity(&fixture, &wave, &active);
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    let capture = Capture::default();
    let profiled = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::PoolVersion,
        &OffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let records = capture.0.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert!(records[0].pool_version_proofs > 0);
    assert!(records[0].pool_version_hits > 0);
    assert_eq!(records[0].parts_materialized, 0);
    drop(records);
    let (handle, _) = profiled.into_parts();
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(handle);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
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
