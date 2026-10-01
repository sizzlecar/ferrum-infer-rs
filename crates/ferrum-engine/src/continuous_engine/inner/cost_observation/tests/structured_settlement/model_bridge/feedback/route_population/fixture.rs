use super::vnext_device_operation_contract as core;
use super::vnext_device_operation_wave_contract as wave_core;
use super::*;
#[path = "fixture/guard_rollback.rs"]
mod guard_rollback;
pub(super) use guard_rollback::{record_guard_rollback, GuardRollbackFault};

pub(super) fn record_route(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    outside: bool,
    extra_unknown: bool,
) -> Option<Arc<HostStageEvidenceV1>> {
    let stages = record_route_with_hook(runtime, clock, w, outside, extra_unknown, false, |_| {});
    if extra_unknown {
        assert!(stages.is_none());
    } else {
        assert!(
            stages.as_ref().unwrap().route_evidence.is_some(),
            "original private route is retained"
        );
    }
    stages
}

pub(super) fn record_route_with_hook(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    outside: bool,
    extra_unknown: bool,
    missing_program_binding: bool,
    hook: impl FnOnce(&mut EngineCostCall),
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider(
        runtime,
        clock,
        w,
        outside,
        extra_unknown,
        missing_program_binding,
        false,
        None,
        false,
        hook,
    )
}

pub(super) fn record_cohort_route(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider(
        runtime,
        clock,
        w,
        false,
        false,
        false,
        true,
        None,
        false,
        |_| {},
    )
}

// Keep the real multi-participant provider for an outside-route cohort too.
// ProgramBinding's single-participant command belongs to the single-row fixture.
pub(super) fn record_outside_cohort_route(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider(
        runtime,
        clock,
        w,
        true,
        false,
        false,
        true,
        None,
        false,
        |_| {},
    )
}

pub(super) fn record_unticketed_cohort_with_hook(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    hook: impl FnOnce(&mut EngineCostCall),
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider(
        runtime, clock, w, false, false, false, true, None, true, hook,
    )
}

pub(super) fn record_private_route(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    outside: bool,
    extra_unknown: bool,
    capture: Arc<CostCalibrationCapture>,
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider(
        runtime,
        clock,
        w,
        outside,
        extra_unknown,
        false,
        false,
        Some(capture),
        false,
        |_| {},
    )
}

pub(super) fn record_feedback_route(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    extra_unknown: bool,
) -> Option<Arc<HostStageEvidenceV1>> {
    // The real source is between blocks; no source ticket is invented.
    assert!(runtime.reserve_live_ticket(clock.now_ns()).is_none());
    record_route_with_provider(
        runtime,
        clock,
        w,
        true,
        extra_unknown,
        false,
        false,
        None,
        true,
        |_| {},
    )
}

fn record_route_with_provider(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    outside: bool,
    extra_unknown: bool,
    missing_program_binding: bool,
    cohort: bool,
    capture: Option<Arc<CostCalibrationCapture>>,
    feedback_only: bool,
    hook: impl FnOnce(&mut EngineCostCall),
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider_and_capture(
        runtime,
        clock,
        w,
        outside,
        extra_unknown,
        missing_program_binding,
        cohort,
        capture,
        feedback_only,
        None,
        hook,
    )
}

pub(super) fn record_issued_cohort_route(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    issued: Arc<crate::continuous_engine::inner::cost_observation::ProspectiveCapture>,
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_provider_and_capture(
        runtime,
        clock,
        w,
        false,
        false,
        false,
        true,
        None,
        false,
        Some(issued),
        |_| {},
    )
}

fn record_route_with_provider_and_capture(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    outside: bool,
    extra_unknown: bool,
    missing_program_binding: bool,
    cohort: bool,
    capture: Option<Arc<CostCalibrationCapture>>,
    feedback_only: bool,
    issued: Option<Arc<crate::continuous_engine::inner::cost_observation::ProspectiveCapture>>,
    hook: impl FnOnce(&mut EngineCostCall),
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_settlement(
        runtime,
        clock,
        w,
        outside,
        extra_unknown,
        missing_program_binding,
        cohort,
        capture,
        feedback_only,
        issued,
        None,
        0,
        hook,
    )
}

// Controlled terminal outcome and host time, recorded by the same original
// private settlement writer. Neither evidence nor observed wall is edited.
pub(super) fn record_eos_feedback_cohort(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    extra_host_ns: u64,
) -> Option<Arc<HostStageEvidenceV1>> {
    record_route_with_settlement(
        runtime,
        clock,
        w,
        false,
        false,
        false,
        true,
        None,
        true,
        None,
        Some(ferrum_types::FinishReason::EOS),
        extra_host_ns,
        |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn record_route_with_settlement(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    w: Wave,
    outside: bool,
    extra_unknown: bool,
    missing_program_binding: bool,
    cohort: bool,
    capture: Option<Arc<CostCalibrationCapture>>,
    feedback_only: bool,
    issued: Option<Arc<crate::continuous_engine::inner::cost_observation::ProspectiveCapture>>,
    terminal_override: Option<ferrum_types::FinishReason>,
    extra_host_ns: u64,
    hook: impl FnOnce(&mut EngineCostCall),
) -> Option<Arc<HostStageEvidenceV1>> {
    use ferrum_interfaces::vnext::*;
    let start = clock.now_ns().unwrap() + 100;
    clock.set(start);
    let ticket = (capture.is_none() && !feedback_only).then(|| {
        runtime.reserve_live_ticket(Some(start)).unwrap_or_else(|| {
            panic!(
                "route fixture reservation failed at {start}; live={:#?}",
                runtime.training.live.as_ref().unwrap().audit()
            )
        })
    });
    let mut call = EngineCostCall::begin(
        &runtime.ids,
        clock.clone(),
        runtime.sink.clone(),
        EngineCostCallSpec {
            identity: super::identity(),
            participants: w
                .actual
                .rows
                .iter()
                .map(|row| CostObservationParticipant {
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    work_generation: row.work_generation,
                    input_index: row.input_index,
                    output_policy_signature: Some([6; 32]),
                    host_features: Some(w.host),
                })
                .collect(),
            prepare_started_at_ns: Some(start),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 1,
                max_rows_per_wave: 8,
                max_retained_rows: 128,
            },
        },
    )
    .unwrap()
    .with_structured_capture(true)
    .with_live_ticket(ticket);
    if let Some(issued) = issued {
        call.attach_prospective_capture(issued, std::time::Instant::now());
        assert!(call.prospective_capture.is_some());
    }
    if let Some(capture) = capture {
        call.attach_calibration_capture(capture);
    }
    let fixture = if cohort {
        // This existing CPU provider encodes the actual cohort width/tokens in
        // its command; its runtime attribution preserves those same numbers.
        core::fixture_with_provider_behavior(false, core::ProviderBehavior::ScratchOverwrite)
    } else if missing_program_binding {
        core::fixture()
    } else {
        core::fixture_with_provider_behavior(false, core::ProviderBehavior::ProgramBinding)
    };
    let (fixture, sequences, sessions, batch, step) = if !cohort && w.actual.rows.len() == 1 {
        let (fixture, sequence, session, batch, step) = wave_core::setup_with_fixture(fixture);
        (fixture, vec![sequence], vec![session], batch, step)
    } else {
        setup_cohort(fixture, &w.actual.rows)
    };
    if cohort {
        for (session, row) in sessions.iter().zip(&w.actual.rows) {
            assert_eq!(
                session.resources().request_id().as_str(),
                row.request_id.to_string()
            );
        }
    }
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let prepared_wave = wave_core::prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    assert!(prepared_wave
        .nodes()
        .iter()
        .all(|node| { node.participant_frames().len() == w.actual.rows.len() }));
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &prepared_wave,
        &lane,
    )
    .unwrap();
    let graph_before = DeviceCostGraphStreamState::new(
        if outside || missing_program_binding {
            DeviceCostGraphConfiguration::OnDemand
        } else {
            DeviceCostGraphConfiguration::Unconfigured
        },
        0,
        0,
        0,
    )
    .unwrap();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.cost_route_projection_enabled = true;
        trace.cost_graph_state = Some(graph_before);
        trace.submitted_graph_evidence = Some(
            DeviceSubmissionGraphEvidence::new(graph_before, graph_before, false, 0, 0, 0, 0, 0)
                .unwrap(),
        );
    }
    let catalog = lane
        .reusable_execution_catalog()
        .unwrap()
        .into_index()
        .unwrap();
    let selection = if outside || missing_program_binding {
        let selected = OperationDispatch::select_reusable_execution_for_cost(
            providers.providers(),
            &fixture.resolved,
            &prepared_wave,
            &lane,
            if missing_program_binding && !outside {
                None
            } else {
                Some(&catalog)
            },
            true,
        )
        .unwrap();
        if missing_program_binding && !outside {
            assert_eq!(selected.route().reason(), ferrum_interfaces::execution_cost::PreparedCostRouteReasonV1::ProgramIdentityUnavailable);
        } else if missing_program_binding {
            assert_eq!(selected.route().class(), ferrum_interfaces::execution_cost::PreparedCostRouteClassV1::OutsideProgramLayoutAbsent);
            assert_eq!(
                selected.route().reason(),
                ferrum_interfaces::execution_cost::PreparedCostRouteReasonV1::ProgramLayoutAbsent
            );
        } else {
            assert!(selected.route().class().is_outside());
        }
        let (program, route) = selected.into_parts();
        assert!(program.is_none());
        route
    } else {
        OperationDispatch::observe_non_reusable_cost_route(&lane)
    };
    clock.set(start + 1);
    let mut context = call.context().unwrap();
    context.prepared_route(
        selection,
        Ok(if outside {
            w.actual.rows.clone()
        } else {
            Vec::new()
        }),
    );
    // The same original core identity reaches a real controlled CPU submit.
    // No receipt, stage binding, program id or hash is constructed here.
    clock.set(start + 2);
    let reaper = CompletionReaper::new();
    let submitted = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Kernel,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        &wave_core::RecordingSubmissionTimingSink::default(),
        prepared_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (handle, attribution) = submitted.into_parts();
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    let device = attribution
        .as_ref()
        .expect("actual submitted attribution")
        .device();
    let compute = device
        .commands()
        .iter()
        .filter(|command| command.command_phase() == DeviceCommandPhase::Compute)
        .collect::<Vec<_>>();
    assert!(!compute.is_empty());
    assert!(
        compute.iter().all(|command| {
            command.participant_count() as usize == w.actual.rows.len()
                && command.token_count() as usize == w.actual.rows.len()
        }),
        "every submitted compute command must cover the original physical cohort"
    );
    context.physical_wave(
        if outside {
            Err(ActualWaveEvidenceUnknown::GraphPath)
        } else {
            Ok(w.actual.clone())
        },
        Some(start + 2),
    );
    context.route_submission(attribution.as_ref());
    if extra_unknown {
        context.mark_unknown(ActualWaveEvidenceUnknown::InvalidLifecycle);
    }
    clock.set(start + 5);
    context.terminal(ActualWaveOutcome::Completed, None);
    context.finish_call(ObservedCallOutcome::Completed);
    drop(context);
    drop(handle);
    drop(active);
    drop(identity);
    drop(providers);
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
    for (position, row) in w.actual.rows.iter().enumerate() {
        let row_start = start + 10 + position as u64 * 4 + extra_host_ns;
        clock.set(row_start);
        call.begin_host_row(&row.request_id);
        clock.set(row_start + 1);
        let ActualRowWork::Decode { kv_tokens } = row.work else {
            panic!("decode fixture")
        };
        let commit = HostCommitEvidence {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            input_index: row.input_index,
            outcome: HostCommitOutcome::Committed(HostCommittedWork::Decode {
                kv_tokens_before: kv_tokens,
                kv_tokens_after: kv_tokens + 1,
                generated_tokens_before: w.host.state.generated_tokens_before,
                generated_tokens_after: w.host.state.generated_tokens_before + 1,
            }),
            committed_at_ns: Some(row_start + 1),
        };
        call.note_host_token_commit(&commit);
        let generated_after = w.host.state.generated_tokens_before + 1;
        assert!(generated_after <= w.host.state.maximum_output_tokens);
        if generated_after == w.host.state.maximum_output_tokens || terminal_override.is_some() {
            let mut pending = call.host_publication(&commit, true, true).unwrap();
            clock.set(row_start + 2);
            pending.terminal_handed_off();
            clock.set(row_start + 3);
            let mut terminal = terminal();
            if let Some(reason) = terminal_override {
                terminal.finish_reason = reason;
            }
            terminal.generated_tokens = generated_after;
            terminal.through_output_ordinal = generated_after;
            call.record_settled(pending.settle(owner(row), terminal));
        } else {
            // Same real host commit, now an honest continuing request. A
            // below-capacity row must not manufacture a Length terminal.
            clock.set(row_start + 3);
            assert!(call.host_publication(&commit, false, true).is_none());
            call.record_host_result(commit);
        }
    }
    // Like record_with_hooks, terminal handoff/cleanup is a composite legacy
    // boundary; the private HostSettled receipt remains the numerical source.
    // Without this actual boundary marker make_sample reports HostMissing.
    call.reject(CostCallRejection::Composite);
    clock.set(start + 20 + 4 * (w.actual.rows.len() as u64 - 1) + extra_host_ns);
    hook(&mut call);
    let stages = call.make_host_stages();
    call.finish();
    stages
}

// Same real multi-participant admission used by the core submission-readback
// tests. Every cost row has a distinct admitted session and physical frame.
fn setup_cohort(
    fixture: core::Fixture,
    rows: &[ActualWaveRow],
) -> (
    core::Fixture,
    Vec<Arc<core::AdmittedSequenceResources<core::TestRuntime>>>,
    Vec<Arc<core::SequenceSession<core::TestRuntime>>>,
    core::ExecutionBatchParticipants<core::TestRuntime>,
    Arc<core::StepResourceLease<core::TestRuntime>>,
) {
    use ferrum_interfaces::vnext::*;
    let resources = rows
        .iter()
        .enumerate()
        .map(|(participant, row)| {
            core::logical_resources(
                &fixture.plan_resources,
                &format!("run.route-cohort.{participant}"),
                &row.request_id.to_string(),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|resource| resource.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(rows.iter().map(|_| core::one_token_span()).collect())
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for _ in 0..=3 {
        match batch.try_begin_step(request.clone(), &lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => {
                return (fixture, resources, sessions, batch, step)
            }
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("original multi-row CPU cohort admission failed"),
        }
    }
    panic!("multi-row CPU cohort backing did not converge")
}
