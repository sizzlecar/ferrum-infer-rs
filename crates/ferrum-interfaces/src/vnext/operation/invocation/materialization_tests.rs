//! Uses the same admitted TestRuntime/Plan fixture as the public wave contract tests.
#[path = "../../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use super::*;
use std::hint::black_box;
use std::time::Instant;
use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::prepare_wave;

#[path = "coverage_tests.rs"]
mod coverage_tests;
#[path = "descriptor_agreement_tests.rs"]
mod descriptor_agreement_tests;

fn step_for(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    spans: Vec<TokenSpanWork>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(spans).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            _ => panic!("fixture step admission cannot progress"),
        }
    }
    unreachable!("bounded admission returns or reports failure")
}

fn snapshot(invocation: &BatchedOperationInvocation<'_, TestBuffer>) -> Vec<serde_json::Value> {
    invocation.participants().iter().map(|participant| {
        serde_json::json!({"views":participant.views().iter().map(|view| {
            serde_json::json!({"descriptor":view.descriptor(),"regions":view.translate(0,view.descriptor().size_bytes).unwrap().iter().map(|r|
                (r.logical_offset_bytes(),r.buffer_and_physical_range().1.start,r.length_bytes())).collect::<Vec<_>>()})
        }).collect::<Vec<_>>()})
    }).collect()
}

fn with_live_wave(
    width: usize,
    run: impl FnOnce(
        &Fixture,
        &PreparedStepSubmissionWave<TestRuntime>,
        &BatchOperationIdentity,
        &[TrustedActiveSequenceBinding],
    ),
) {
    with_live_wave_spans(vec![1; width], run)
}

fn with_live_wave_spans(
    lengths: Vec<usize>,
    run: impl FnOnce(
        &Fixture,
        &PreparedStepSubmissionWave<TestRuntime>,
        &BatchOperationIdentity,
        &[TrustedActiveSequenceBinding],
    ),
) {
    with_live_wave_fixture(fixture(), lengths, run)
}

fn with_live_wave_fixture(
    fixture: Fixture,
    lengths: Vec<usize>,
    run: impl FnOnce(
        &Fixture,
        &PreparedStepSubmissionWave<TestRuntime>,
        &BatchOperationIdentity,
        &[TrustedActiveSequenceBinding],
    ),
) {
    let width = lengths.len();
    let resources = (0..width)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.materialize.{i}"),
                &format!("request.materialize.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|r| r.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let spans = lengths
        .iter()
        .map(|n| TokenSpanWork::from_token_ids(&[1, 2, 3, 4], 0..*n).unwrap())
        .collect();
    let step = step_for(&batch, &lane, spans);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    run(&fixture, &wave, &identity, &active);
}

#[test]
fn same_wave_materialization_matches_reference_and_keeps_runtime_checks() {
    for width in [1, 8, 32] {
        with_live_wave(width, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = |share| {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0,
                    },
                    active.iter(),
                    false,
                    share,
                )
            };
            let reference = make(false).unwrap();
            let shared = make(true).unwrap();
            assert_eq!(snapshot(&reference), snapshot(&shared));
            let shared_view = shared
                .participants()
                .iter()
                .flat_map(|p| p.views())
                .find_map(|v| v.shared_backing());
            if width == 1 {
                assert!(shared_view.is_none());
            } else {
                let retained = shared_view
                    .expect("actual Step/Invocation fixture must exercise shared backing");
                assert!(shared
                    .participants()
                    .iter()
                    .skip(1)
                    .flat_map(|p| p.views())
                    .filter_map(|v| v.shared_backing())
                    .any(|v| Arc::ptr_eq(retained, v)));
                let weak = Arc::downgrade(retained);
                // A second constructor on the SAME live wave must not retrieve a cached owner.
                let next = make(true).unwrap();
                assert!(next
                    .participants()
                    .iter()
                    .flat_map(|p| p.views())
                    .filter_map(|v| v.shared_backing())
                    .all(|v| !Arc::ptr_eq(retained, v)));
                drop(shared);
                assert!(weak.upgrade().is_none());
                drop(next);
            }
            if width > 1 {
                let mut count = 0;
                let changed = active.iter().inspect(|_| {
                    count += 1;
                    if count == 2 {
                        fixture
                            .runtime_trace
                            .lock()
                            .unwrap()
                            .tamper_buffer_descriptor = true;
                    }
                });
                assert!(BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0
                    },
                    changed,
                    false,
                    true
                )
                .is_err());
                fixture
                    .runtime_trace
                    .lock()
                    .unwrap()
                    .tamper_buffer_descriptor = false;
                assert!(BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0
                    },
                    active.iter().rev(),
                    false,
                    true
                )
                .is_err());
            }
            assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
            assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
        });
    }
}

#[test]
#[ignore = "paired CPU elapsed diagnostic, no speed threshold"]
fn paired_live_wave_invocation_materialization() {
    for width in [1, 8, 32] {
        with_live_wave(width, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = |reference: bool| {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0,
                    },
                    active.iter(),
                    false,
                    !reference,
                )
                .unwrap()
            };
            paired_invocation_elapsed("invocation_materialization_pair", width, make);
        });
    }
}

fn paired_invocation_elapsed<'a>(
    kind: &str,
    width: usize,
    make: impl Fn(bool) -> BatchedOperationInvocation<'a, TestBuffer>,
) {
    assert_eq!(snapshot(&make(false)), snapshot(&make(true)));
    for round in 0..6 {
        for reference in if round % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            let mut construct_ns = 0_u128;
            let mut total_ns = 0_u128;
            for _ in 0..128 {
                let start = Instant::now();
                let value = black_box(make(reference));
                construct_ns += start.elapsed().as_nanos();
                black_box(value.participants().len());
                drop(value);
                total_ns += start.elapsed().as_nanos();
            }
            if round >= 2 {
                println!(
                    "{}",
                    serde_json::json!({"kind":kind,"physical_rows":width,"participants":width,"pair":round-2,"reference":reference,"iterations":128,"construct_ns":construct_ns,"construct_and_drop_ns":total_ns,"scope":"same admitted wave; fresh per-call proofs; no encode/submit; host elapsed, not inference throughput"})
                );
            }
        }
    }
}

#[test]
fn shared_materialization_preserves_unequal_participant_token_windows() {
    with_live_wave_spans(vec![1, 3, 2], |fixture, wave, identity, active| {
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        let make = |share| {
            BatchedOperationInvocation::from_resources(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                identity,
                &node,
                OperationInvocationResources::Wave {
                    wave,
                    node_index: 0,
                },
                active.iter(),
                false,
                share,
            )
            .unwrap()
        };
        assert_eq!(snapshot(&make(false)), snapshot(&make(true)));
        let ranges = wave.nodes()[0].work_shape().participant_token_ranges();
        let widths = ranges
            .iter()
            .map(|r| r.immediate_tokens())
            .collect::<Vec<_>>();
        assert_eq!(widths, [1, 3, 2]);
        assert_eq!(ranges[0].immediate_token_range(), 0..1);
        assert_eq!(ranges[1].immediate_token_range(), 1..4);
        assert_eq!(ranges[2].immediate_token_range(), 4..6);
    });
}

#[path = "identity_projection_tests.rs"]
mod identity_projection_tests;

#[path = "prepared_view_workspace_tests.rs"]
mod prepared_view_workspace_tests;
