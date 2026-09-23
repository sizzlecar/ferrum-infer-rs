mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("cost observation must not enable timing");
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("cost observation must not enable timing");
    }
}

#[derive(Debug, PartialEq)]
struct PhysicalSubmission {
    commands: Vec<Vec<TestCommand>>,
    phases: Vec<Vec<DeviceCommandPhase>>,
    nodes: Vec<Vec<Option<u32>>>,
    policies: Vec<DeviceComputePathRequirement>,
    captures: Vec<Option<DeviceReusableExecutionCapture>>,
    encode_calls: u64,
    binding_calls: u64,
}

fn compare(direct: bool, fail_direct: bool) {
    let (fixture, sequence, session, batch, mut step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let mut results = Vec::new();
    for observed in [false, true] {
        // Both arms use the same actual lane and plan. A fresh step carries fresh
        // invocation authority while retaining the lane's reusable-program identity.
        if observed {
            step.try_retire_normal().unwrap();
            step = begin_single_participant_step_on_lane_with_bucket(
                &batch,
                &lane,
                fixture.reusable_execution_bucket.as_ref(),
            );
        }
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active_bindings = wave_active_bindings(&wave, &session);
        let trace_start = fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_commands
            .len();
        let (encode_start, binding_start) = {
            let trace = fixture.provider_trace.lock().unwrap();
            (trace.encode_calls, trace.reusable_binding_encode_calls)
        };
        let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
            &providers,
            &fixture.resolved,
            &wave,
            &lane,
        )
        .unwrap()
        .unwrap();
        let count = u32::try_from(providers.len()).unwrap();
        let program = test_reusable_program(
            program_id,
            count,
            vec![],
            vec![DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap()],
            (0..count).collect(),
            vec![],
        );
        if fail_direct {
            assert!(direct);
            fixture.runtime_trace.lock().unwrap().submit_behavior =
                SubmitBehavior::DefinitelyNotSubmitted;
        }
        let mut wave = wave;
        let mut use_direct = direct;
        let (handle, evidence) = loop {
            // DefinitelyNotSubmitted retry creates a new invocation. Rebind the
            // exact returned wave, as the production dispatcher does.
            let identity = OperationDispatch::bind_submission_wave_identity(
                &fixture.resolved,
                active_bindings.iter(),
                &wave,
                &lane,
            )
            .unwrap();
            let submitted = if observed {
                OperationDispatch::encode_and_submit_wave_with_cost_observation(
                    &providers,
                    &fixture.resolved,
                    &identity,
                    active_bindings.iter(),
                    DeviceTimingMode::Off,
                    &[],
                    SubmissionExecutionPolicy::adaptive(),
                    use_direct.then_some(&program),
                    &NoTiming,
                    wave,
                    &lane,
                    &reaper,
                )
                .map(ProfiledSubmissionHandle::into_parts)
            } else if use_direct {
                OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_policy(
                    &providers,
                    &fixture.resolved,
                    &identity,
                    active_bindings.iter(),
                    DeviceTimingMode::Off,
                    &[],
                    &program,
                    SubmissionExecutionPolicy::adaptive(),
                    wave,
                    &lane,
                    &reaper,
                )
                .map(|handle| (handle, None))
            } else {
                OperationDispatch::encode_and_submit_wave_with_inputs_and_policy(
                    &providers,
                    &fixture.resolved,
                    &identity,
                    active_bindings.iter(),
                    DeviceTimingMode::Off,
                    &[],
                    SubmissionExecutionPolicy::adaptive(),
                    wave,
                    &lane,
                    &reaper,
                )
                .map(|handle| (handle, None))
            };
            match submitted {
                Ok(result) => break result,
                Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. })
                    if use_direct && fail_direct =>
                {
                    wave = retry.retry().unwrap();
                    use_direct = false;
                    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::Success;
                }
                Err(error) => panic!("unexpected dispatch failure: {error}"),
            }
        };
        assert_eq!(evidence.is_some(), observed);
        if let Some(evidence) = evidence {
            let expected = if direct && !fail_direct {
                DeviceExecutionPath::Replayed
            } else {
                DeviceExecutionPath::Eager
            };
            assert!(evidence
                .device()
                .commands()
                .iter()
                .filter(|row| row.command_phase() == DeviceCommandPhase::Compute)
                .all(|row| row.execution_path() == expected));
            assert_eq!(
                !evidence.device().replayed_segments().is_empty(),
                direct && !fail_direct
            );
        }
        let terminal = handle.wait().unwrap();
        assert!(matches!(terminal, CompletionObservation::Terminal(_)));
        let physical = {
            let trace = fixture.runtime_trace.lock().unwrap();
            assert!(trace.submitted_attribution_requirements[trace_start..]
                .iter()
                .all(|requirement| *requirement
                    == if observed {
                        DeviceSubmissionAttributionRequirement::LogicalExecutionPath
                    } else {
                        DeviceSubmissionAttributionRequirement::None
                    }));
            let provider = fixture.provider_trace.lock().unwrap();
            PhysicalSubmission {
                commands: trace.submitted_commands[trace_start..].to_vec(),
                phases: trace.submitted_command_phases[trace_start..].to_vec(),
                nodes: trace.submitted_command_node_indices[trace_start..].to_vec(),
                policies: trace.submitted_compute_path_requirements[trace_start..].to_vec(),
                captures: trace.submitted_reusable_captures[trace_start..].to_vec(),
                encode_calls: provider.encode_calls - encode_start,
                binding_calls: provider.reusable_binding_encode_calls - binding_start,
            }
        };
        results.push(physical);
        drop(handle);
        drop(active_bindings);
    }
    assert_eq!(results[0], results[1]);
    drop(providers);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn timing_off_cost_attribution_preserves_eager_commands_and_policy() {
    compare(false, false);
}

#[test]
fn timing_off_cost_attribution_preserves_direct_reusable_execution() {
    compare(true, false);
}

#[test]
fn direct_reusable_failure_records_only_actual_eager_fallback_path() {
    compare(true, true);
}
