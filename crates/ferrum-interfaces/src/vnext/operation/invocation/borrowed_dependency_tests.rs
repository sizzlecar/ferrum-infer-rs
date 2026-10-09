//! Real admitted multi-participant Plan weights and preserved validation banks.
use super::*;
use crate::vnext::operation::retained_dependency::append_dependencies;
use ferrum_types::InvocationPreparationStrategy as Strategy;

fn identity_for(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    active: &[TrustedActiveSequenceBinding],
    strategy: Strategy,
) -> BatchOperationIdentity {
    let lane = wave.step_resources().execution_lane();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
    OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        wave,
        lane,
        strategy,
    )
    .unwrap()
}

fn spec(component: &WeightId) -> RetainedPlanDependencySpec<'_> {
    RetainedPlanDependencySpec {
        input_ordinal: 1,
        component_id: component,
        source_offset_bytes: 0,
        source_length_bytes: 8,
        persistent_offset_bytes: 0,
        persistent_length_bytes: 4,
        alignment_bytes: 4,
        validation_identity: "fixture.exact-weight-validation.v1",
    }
}

#[test]
fn borrowed_dependency_matches_full_owned_identity_and_counts_successful_comparisons() {
    for width in [1, 3, 32] {
        with_live_wave_fixture(
            fixture_with_padded_persistent(24),
            vec![1; width],
            |fixture, wave, _, active| {
                let full = identity_for(fixture, wave, active, Strategy::Full);
                let ip = identity_for(fixture, wave, active, Strategy::IdentityProjection);
                let borrowed = identity_for(fixture, wave, active, Strategy::BorrowedDependency);
                let component = id("weight.component.left");
                for node in 0..wave.node_count() {
                    let provider = fixture
                        .registry
                        .bind(&fixture.resolved, wave.nodes()[node].node_id())
                        .unwrap();
                    let mut expected = None;
                    for identity in [&full, &ip, &borrowed] {
                        let invocation = BatchedOperationInvocation::from_wave_node(
                            fixture.runtime.as_ref(),
                            &fixture.resolved,
                            provider.dispatch(),
                            identity,
                            identity.materialize_node(node).unwrap(),
                            wave,
                            node,
                            active.iter(),
                        )
                        .unwrap();
                        let authority = invocation
                            .retained_plan_dependency(spec(&component))
                            .unwrap();
                        let wire = serde_json::to_value(&authority.identity).unwrap();
                        if let Some(expected) = &expected {
                            assert_eq!(&wire, expected);
                        } else {
                            expected = Some(wire);
                        }
                    }
                }
                assert_eq!(
                    full.preparation_snapshot().borrowed_dependency_comparisons,
                    0
                );
                assert_eq!(ip.preparation_snapshot().borrowed_dependency_comparisons, 0);
                assert_eq!(
                    borrowed
                        .preparation_snapshot()
                        .borrowed_dependency_comparisons,
                    (width as u64 - 1) * wave.node_count() as u64
                );
                assert_eq!(borrowed.preparation_snapshot().parts_materialized, 0);
            },
        );
    }
}

#[test]
fn borrowed_dependency_keeps_range_error_order_and_does_not_count_failed_authorities() {
    with_live_wave_fixture(
        fixture_with_padded_persistent(24),
        vec![1; 3],
        |fixture, wave, _, active| {
            let component = id("weight.component.left");
            let mut expected_errors = None;
            for strategy in [Strategy::IdentityProjection, Strategy::BorrowedDependency] {
                let identity = identity_for(fixture, wave, active, strategy);
                let provider = fixture
                    .registry
                    .bind(&fixture.resolved, wave.nodes()[0].node_id())
                    .unwrap();
                let mut invocation = BatchedOperationInvocation::from_wave_node(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    &identity,
                    identity.materialize_node(0).unwrap(),
                    wave,
                    0,
                    active.iter(),
                )
                .unwrap();
                let errors = (0..9)
                    .map(|change| {
                        let mut invalid = spec(&component);
                        match change {
                            0 => invalid.input_ordinal = 0,
                            1 => invalid.source_offset_bytes = 4,
                            2 => invalid.source_length_bytes = 0,
                            3 => invalid.persistent_offset_bytes = u64::MAX,
                            4 => invalid.persistent_offset_bytes = 1,
                            5 => invalid.persistent_length_bytes = 3,
                            6 => invalid.alignment_bytes = 3,
                            7 => invalid.validation_identity = "",
                            // Both fail: source range must still be observed first.
                            8 => {
                                invalid.source_offset_bytes = u64::MAX;
                                invalid.persistent_offset_bytes = u64::MAX;
                            }
                            _ => unreachable!(),
                        }
                        invocation
                            .retained_plan_dependency(invalid)
                            .err()
                            .unwrap()
                            .to_string()
                    })
                    .collect::<Vec<_>>();
                if let Some(expected) = &expected_errors {
                    assert_eq!(&errors, expected);
                } else {
                    expected_errors = Some(errors);
                }
                assert_eq!(
                    identity
                        .preparation_snapshot()
                        .borrowed_dependency_comparisons,
                    0
                );
                // The third real participant fails after one later-P comparison.
                // This authority must publish no partial success count.
                invocation.participants[2].persistent_view = None;
                assert!(invocation
                    .retained_plan_dependency(spec(&component))
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("persistent view is absent"));
                assert_eq!(
                    identity
                        .preparation_snapshot()
                        .borrowed_dependency_comparisons,
                    0
                );
            }
        },
    );
}

#[test]
fn borrowed_dependency_rejects_foreign_admitted_plan_ranges_after_first_participant() {
    with_live_wave_fixture(
        fixture_with_padded_persistent(24),
        vec![1; 3],
        |fixture, wave, _, active| {
            with_live_wave_fixture(
                fixture_with_padded_persistent(24),
                vec![1; 3],
                |donor, donor_wave, _, donor_active| {
                    let component = id("weight.component.left");
                    for replace_persistent in [false, true] {
                        let mut expected_error = None;
                        for strategy in [Strategy::IdentityProjection, Strategy::BorrowedDependency]
                        {
                            let identity = identity_for(fixture, wave, active, strategy);
                            let other = identity_for(donor, donor_wave, donor_active, strategy);
                            let provider = fixture
                                .registry
                                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                                .unwrap();
                            let donor_provider = donor
                                .registry
                                .bind(&donor.resolved, donor_wave.nodes()[0].node_id())
                                .unwrap();
                            let mut invocation = BatchedOperationInvocation::from_wave_node(
                                fixture.runtime.as_ref(),
                                &fixture.resolved,
                                provider.dispatch(),
                                &identity,
                                identity.materialize_node(0).unwrap(),
                                wave,
                                0,
                                active.iter(),
                            )
                            .unwrap();
                            let mut foreign = BatchedOperationInvocation::from_wave_node(
                                donor.runtime.as_ref(),
                                &donor.resolved,
                                donor_provider.dispatch(),
                                &other,
                                other.materialize_node(0).unwrap(),
                                donor_wave,
                                0,
                                donor_active.iter(),
                            )
                            .unwrap();
                            // Both independently admitted worlds are valid before swapping.
                            assert!(foreign.retained_plan_dependency(spec(&component)).is_ok());
                            let usage = if replace_persistent {
                                BufferUsage::Persistent
                            } else {
                                BufferUsage::Weights
                            };
                            let index = invocation.participants[2]
                                .views
                                .iter()
                                .position(|view| view.descriptor().usage == usage)
                                .unwrap();
                            let donor_index = foreign.participants[2]
                                .views
                                .iter()
                                .position(|view| view.descriptor().usage == usage)
                                .unwrap();
                            assert_eq!(
                                invocation.participants[2].views[index].descriptor(),
                                foreign.participants[2].views[donor_index].descriptor()
                            );
                            std::mem::swap(
                                &mut invocation.participants[2].views[index],
                                &mut foreign.participants[2].views[donor_index],
                            );
                            let error = invocation
                                .retained_plan_dependency(spec(&component))
                                .err()
                                .unwrap()
                                .to_string();
                            assert!(
                                error.contains("do not share exact Plan allocations"),
                                "{error}"
                            );
                            if let Some(expected) = &expected_error {
                                assert_eq!(&error, expected);
                            } else {
                                expected_error = Some(error);
                            }
                            assert_eq!(
                                identity
                                    .preparation_snapshot()
                                    .borrowed_dependency_comparisons,
                                0
                            );
                        }
                    }
                },
            );
        },
    );
}

#[test]
fn borrowed_dependency_keeps_fresh_invocation_scope_and_owned_plan_retention() {
    let mut held = None;
    let mut weak = None;
    with_live_wave_fixture(
        fixture_with_padded_persistent(24),
        vec![1; 3],
        |fixture, wave, _, active| {
            let identity = identity_for(fixture, wave, active, Strategy::BorrowedDependency);
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let make = || {
                BatchedOperationInvocation::from_wave_node(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    &identity,
                    identity.materialize_node(0).unwrap(),
                    wave,
                    0,
                    active.iter(),
                )
                .unwrap()
            };
            let first = make();
            let second = make();
            let component = id("weight.component.left");
            let stale = first
                .retained_plan_dependency(spec(&component))
                .unwrap()
                .encode(());
            let error = append_dependencies(
                &second.retained_dependency_scope,
                vec![stale],
                &mut vec![],
                &mut vec![],
                &mut vec![],
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("not issued by this live invocation"));
            weak = Some(Arc::downgrade(&fixture.plan_resources));
            held = Some(
                second
                    .retained_plan_dependency(spec(&component))
                    .unwrap()
                    .encode(()),
            );
        },
    );
    let weak = weak.unwrap();
    assert!(
        weak.upgrade().is_some(),
        "the returned command owns the admitted source and destination Plan"
    );
    drop(held);
    assert!(
        weak.upgrade().is_none(),
        "no cache or cycle keeps the Plan alive after command drop"
    );
}

struct OffTiming;
impl DeviceSubmissionTimingSink for OffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("Off must not invoke device timing callbacks");
    }
}
impl SubmissionWaveDispatchTimingSink for OffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("Off must not invoke host timing callbacks");
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
fn borrowed_dependency_off_dispatch_publishes_successful_authority_counts_and_releases_plan() {
    let fixture = fixture_with_retained_dependencies(24, DependencyMode::Valid);
    let sequences = (0..3)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.bd.{i}"),
                &format!("request.bd.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = sequences
        .iter()
        .map(|s| s.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = step_for(&batch, &lane, vec![one_token_span(); sessions.len()]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let node_count = wave.node_count();
    let active = sessions
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = identity_for(&fixture, &wave, &active, Strategy::BorrowedDependency);
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    let capture = PreparationCapture::default();
    let profiled = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        Strategy::BorrowedDependency,
        &OffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (handle, _) = profiled.into_parts();
    let stats = identity.preparation_snapshot();
    // The real provider issues two valid authorities per node after eight
    // rejected specifications. Deduplication does not undo successful issuance.
    assert_eq!(
        stats.borrowed_dependency_comparisons,
        node_count as u64 * 2 * (sessions.len() as u64 - 1)
    );
    assert!(stats.projected_identities > 0);
    assert_eq!(stats.parts_materialized, 0);
    assert_eq!(*capture.0.lock().unwrap(), vec![stats]);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(handle);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(sessions);
    drop(sequences);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
#[ignore = "diagnostic timing of the complete admitted dependency constructor; no performance assertion"]
fn borrowed_dependency_paired_constructor_diagnostic() {
    with_live_wave_fixture(
        fixture_with_padded_persistent(24),
        vec![1; 32],
        |fixture, wave, _, active| {
            let ip = identity_for(fixture, wave, active, Strategy::IdentityProjection);
            let borrowed = identity_for(fixture, wave, active, Strategy::BorrowedDependency);
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let make = |identity| {
                BatchedOperationInvocation::from_wave_node(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    identity.materialize_node(0).unwrap(),
                    wave,
                    0,
                    active.iter(),
                )
                .unwrap()
            };
            let invocations = [make(&ip), make(&borrowed)];
            let component = id("weight.component.left");
            for invocation in &invocations {
                for _ in 0..100 {
                    black_box(
                        invocation
                            .retained_plan_dependency(spec(&component))
                            .unwrap(),
                    );
                }
            }
            for repetition in 0..6 {
                for arm in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = Instant::now();
                    for _ in 0..2000 {
                        black_box(
                            invocations[arm]
                                .retained_plan_dependency(black_box(spec(&component)))
                                .unwrap(),
                        );
                    }
                    eprintln!("borrowed_dependency_diagnostic repetition={repetition} mode={} participants=32 calls=2000 elapsed_ns={}", if arm == 0 { "identity-projection" } else { "borrowed-dependency" }, start.elapsed().as_nanos());
                }
            }
        },
    );
}
