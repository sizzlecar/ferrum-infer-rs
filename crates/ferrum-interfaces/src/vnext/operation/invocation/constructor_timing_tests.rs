use super::*;
use crate::vnext::{
    DeviceSubmissionStage, DeviceSubmissionTimingSink, InvocationConstructorOutcome,
    InvocationConstructorTimingBreakdown, SubmissionWaveDispatchStage,
};
use std::{sync::Mutex, time::Duration};

#[derive(Default)]
struct Capture<const ENABLED: bool> {
    nodes: Mutex<Vec<InvocationConstructorTimingBreakdown>>,
}
impl<const ENABLED: bool> DeviceSubmissionTimingSink for Capture<ENABLED> {
    const ENABLED: bool = ENABLED;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        assert!(ENABLED, "Off cannot call a timing sink");
    }
}
impl<const ENABLED: bool> SubmissionWaveDispatchTimingSink for Capture<ENABLED> {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        assert!(ENABLED, "Off cannot call a timing sink");
    }
    fn record_invocation_construct_breakdown(&self, value: InvocationConstructorTimingBreakdown) {
        assert!(ENABLED, "Off cannot flush a constructor breakdown");
        self.nodes.lock().unwrap().push(value);
    }
}

fn elapsed_sum(value: InvocationConstructorTimingBreakdown) -> Duration {
    value.node_setup_assemble.elapsed
        + value.participant_common_validate.elapsed
        + value.resource_view_build.elapsed
        + value.runtime_full_coverage.elapsed
        + value.component_workspace_validate.elapsed
}

fn check_success<const ENABLED: bool>() {
    with_live_wave_spans(vec![1, 3, 2], |fixture, wave, _, active| {
        let lane = wave.step_resources().execution_lane();
        let topology =
            OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
        let identity = OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
            &topology,
            active.iter(),
            wave,
            lane,
            ferrum_types::InvocationPreparationStrategy::IdentityProjection,
        )
        .unwrap();
        let capture = Capture::<ENABLED>::default();
        for node_index in 0..wave.node_count() {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[node_index].node_id())
                .unwrap();
            let node = identity.materialize_node(node_index).unwrap();
            let original = BatchedOperationInvocation::from_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                node,
                wave,
                node_index,
                active.iter(),
            )
            .unwrap();
            let start = Instant::now();
            let timed = BatchedOperationInvocation::from_wave_node_with_timing(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                node,
                wave,
                node_index,
                active.iter(),
                &capture,
            )
            .unwrap();
            let outer = start.elapsed();
            assert_eq!(snapshot(&timed), snapshot(&original));
            assert!(!Arc::ptr_eq(
                &timed.retained_dependency_scope,
                &original.retained_dependency_scope
            ));
            assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
            let records = capture.nodes.lock().unwrap();
            if ENABLED {
                assert_eq!(records.len(), node_index + 1);
                let value = records[node_index];
                assert_eq!(value.outcome, InvocationConstructorOutcome::Success);
                assert_eq!(value.node_setup_assemble.visits, active.len() as u64 + 1);
                for phase in [
                    value.participant_common_validate,
                    value.resource_view_build,
                    value.runtime_full_coverage,
                    value.component_workspace_validate,
                ] {
                    assert_eq!(phase.visits, active.len() as u64);
                }
                assert!(
                    elapsed_sum(value) <= outer,
                    "exclusive phases must fit the enclosing call"
                );
            } else {
                assert!(records.is_empty());
            }
        }
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn constructor_timing_enabled_matches_admitted_views_and_preserves_lazy_identity() {
    check_success::<true>();
}

#[test]
fn constructor_timing_disabled_admitted_path_never_calls_any_sink() {
    check_success::<false>();
}

#[test]
fn constructor_timing_flushes_one_failure_at_the_original_failed_phase() {
    with_live_wave(2, |fixture, wave, identity, active| {
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        let check = |prepared: &PreparedOperationDispatchBinding,
                     bindings: &[TrustedActiveSequenceBinding],
                     expected_visits: [u64; 4]| {
            let original = BatchedOperationInvocation::from_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                prepared,
                identity,
                node,
                wave,
                0,
                bindings.iter(),
            )
            .err()
            .expect("original constructor must reject the actual invalid inputs");
            let capture = Capture::<true>::default();
            let timed = BatchedOperationInvocation::from_wave_node_with_timing(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                prepared,
                identity,
                node,
                wave,
                0,
                bindings.iter(),
                &capture,
            )
            .err()
            .expect("timed constructor must preserve the rejection");
            assert_eq!(
                std::mem::discriminant(&timed),
                std::mem::discriminant(&original)
            );
            assert_eq!(timed.to_string(), original.to_string());
            let records = capture.nodes.lock().unwrap();
            assert_eq!(records.len(), 1);
            let value = records[0];
            assert_eq!(value.outcome, InvocationConstructorOutcome::Error);
            assert_eq!(value.node_setup_assemble.visits, 1);
            let phases = [
                value.participant_common_validate,
                value.resource_view_build,
                value.runtime_full_coverage,
                value.component_workspace_validate,
            ];
            for (phase, expected) in phases.into_iter().zip(expected_visits) {
                assert_eq!(phase.visits, expected);
                if expected == 0 {
                    assert_eq!(phase.elapsed, Duration::ZERO);
                }
            }
        };
        check(provider.dispatch(), &[], [0, 0, 0, 0]);
        let reversed = active.iter().rev().cloned().collect::<Vec<_>>();
        check(provider.dispatch(), &reversed, [1, 0, 0, 0]);
        let mut resource = provider.dispatch().clone();
        resource.resources[0].source = PreparedOperationResourceSource::Dynamic {
            descriptor_index: usize::MAX,
        };
        check(&resource, active, [1, 1, 0, 0]);
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = true;
        check(provider.dispatch(), active, [1, 1, 1, 0]);
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = false;
        let mut component = provider.dispatch().clone();
        assert!(component.binding_component_views.pop().is_some());
        check(&component, active, [1, 1, 1, 1]);
        assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn constructor_timing_panic_skips_flush_and_fresh_construction_still_works() {
    with_live_wave(2, |fixture, wave, identity, active| {
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        let capture = Capture::<true>::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut seen = 0;
            let bindings = active.iter().inspect(|_| {
                seen += 1;
                if seen == 2 {
                    panic!("test panic after one real participant completed");
                }
            });
            let _ = BatchedOperationInvocation::from_wave_node_with_timing(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                identity,
                node,
                wave,
                0,
                bindings,
                &capture,
            );
        }));
        assert!(result.is_err());
        assert!(capture.nodes.lock().unwrap().is_empty());
        let invocation = BatchedOperationInvocation::from_wave_node_with_timing(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            wave,
            0,
            active.iter(),
            &capture,
        )
        .unwrap();
        assert_eq!(invocation.participants().len(), active.len());
        assert_eq!(capture.nodes.lock().unwrap().len(), 1);
    });
}

fn check_complete_dispatch<const ENABLED: bool>() {
    use vnext_device_operation_wave_contract::{setup, teardown, wave_active_bindings};
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let expected_nodes = wave.node_count();
    let capture = Capture::<ENABLED>::default();
    let reaper = CompletionReaper::new();
    let profiled = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (completion, _) = profiled.into_parts();
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert!(matches!(
        completion.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    let records = capture.nodes.lock().unwrap();
    assert_eq!(records.len(), if ENABLED { expected_nodes } else { 0 });
    assert!(records
        .iter()
        .all(|record| record.outcome == InvocationConstructorOutcome::Success));
    drop(records);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(completion);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn constructor_timing_dispatch_flushes_enabled_nodes_and_keeps_off_silent() {
    check_complete_dispatch::<true>();
    check_complete_dispatch::<false>();
}
