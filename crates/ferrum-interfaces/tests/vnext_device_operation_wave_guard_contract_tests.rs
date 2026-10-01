mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use ferrum_interfaces::execution_cost::{
    CoreReadbackRoute, GuardedNotSubmittedReason, HostSubmissionRejection,
};
use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

struct NoHostTiming;
impl DeviceSubmissionTimingSink for NoHostTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("disabled host timing was called");
    }
}
impl SubmissionWaveDispatchTimingSink for NoHostTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("disabled host timing was called");
    }
}

#[derive(Default)]
struct HostTiming {
    backend_calls: AtomicU64,
    dispatch_calls: AtomicU64,
    completion_arms: AtomicU64,
}
impl DeviceSubmissionTimingSink for HostTiming {
    const ENABLED: bool = true;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        self.backend_calls.fetch_add(1, Ordering::Relaxed);
    }
}
impl SubmissionWaveDispatchTimingSink for HostTiming {
    fn record(&self, stage: SubmissionWaveDispatchStage, _: Duration) {
        self.dispatch_calls.fetch_add(1, Ordering::Relaxed);
        if stage == SubmissionWaveDispatchStage::CompletionArm {
            self.completion_arms.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Guard {
    rejection: Option<GuardedNotSubmittedReason>,
    calls: AtomicU64,
    staging: bool,
    trace: Arc<Mutex<RuntimeTrace>>,
}
impl PreparedWaveSubmissionGuard for Guard {
    fn check(
        &self,
        attribution: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert!(attribution.commands().iter().any(|command| {
            command.command_phase() == DeviceCommandPhase::Compute
                && command.compute_dispatch_count() > 0
        }));
        let trace = self.trace.lock().unwrap();
        assert_eq!(
            trace.submit_calls, 0,
            "guard precedes the first physical submit"
        );
        assert_eq!(trace.guarded_submit_calls, 1);
        if self.staging {
            assert_eq!(readback, CoreReadbackRoute::SubmissionStaged);
            assert_eq!(trace.submission_readback_live, 1);
        }
        drop(trace);
        self.calls.fetch_add(1, Ordering::Relaxed);
        match self.rejection {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }
}

fn dispatch(
    fixture: &Fixture,
    session: &Arc<SequenceSession<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    guard: &Guard,
) -> GuardedWaveSubmissionOutcome<TestRuntime> {
    dispatch_with_timing(fixture, session, step, reaper, guard, &NoHostTiming)
}

fn dispatch_with_timing<S: SubmissionWaveDispatchTimingSink>(
    fixture: &Fixture,
    session: &Arc<SequenceSession<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    guard: &Guard,
    timing: &S,
) -> GuardedWaveSubmissionOutcome<TestRuntime> {
    dispatch_with_timing_and_demand(
        fixture,
        session,
        step,
        reaper,
        guard,
        timing,
        ferrum_interfaces::execution_cost::StructuredCostSampleDemand::RuntimePolicy,
    )
}

fn dispatch_with_timing_and_demand<S: SubmissionWaveDispatchTimingSink>(
    fixture: &Fixture,
    session: &Arc<SequenceSession<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    guard: &Guard,
    timing: &S,
    demand: ferrum_interfaces::execution_cost::StructuredCostSampleDemand,
) -> GuardedWaveSubmissionOutcome<TestRuntime> {
    dispatch_with_device_timing_and_demand(
        fixture,
        session,
        step,
        reaper,
        guard,
        timing,
        demand,
        DeviceTimingMode::Off,
    )
}

fn dispatch_with_device_timing_and_demand<S: SubmissionWaveDispatchTimingSink>(
    fixture: &Fixture,
    session: &Arc<SequenceSession<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    guard: &Guard,
    timing: &S,
    demand: ferrum_interfaces::execution_cost::StructuredCostSampleDemand,
    device_timing: DeviceTimingMode,
) -> GuardedWaveSubmissionOutcome<TestRuntime> {
    let lane = step.execution_lane();
    let mut wave = prepare_wave(&fixture.plan_resources, &fixture.plan, step);
    if guard.staging {
        wave = wave
            .with_submission_readbacks(
                CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
                    id("node.tail"),
                    0,
                    id("resource.output"),
                    0,
                    HostTransferLayout::new(ElementType::F32, 2).unwrap(),
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap();
    }
    let active = wave_active_bindings(&wave, session);
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        lane,
    )
    .unwrap();
    OperationDispatch::encode_and_submit_guarded_wave_with_device_timing_and_cost_evidence_demand(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        device_timing,
        &[],
        None,
        guard,
        ferrum_interfaces::vnext::DeviceCostObservationDemand::Required,
        demand,
        timing,
        wave,
        lane,
        reaper,
    )
}

fn accepted(outcome: GuardedWaveSubmissionOutcome<TestRuntime>) -> CompletionHandle<TestRuntime> {
    match outcome {
        GuardedWaveSubmissionOutcome::Dispatch(Ok(profiled)) => {
            let (handle, attribution) = profiled.into_parts();
            assert!(
                attribution.is_some(),
                "guarded submission retains actual route evidence"
            );
            handle
        }
        GuardedWaveSubmissionOutcome::Dispatch(Err(error)) => panic!("dispatch failed: {error}"),
        GuardedWaveSubmissionOutcome::NotSubmitted(_) => panic!("accepting guard was rejected"),
    }
}

fn rejected_then_retry(missing_attribution: bool) {
    rejected_then_retry_with_timing(missing_attribution, &NoHostTiming);
}

fn rejected_then_retry_with_timing<S: SubmissionWaveDispatchTimingSink>(
    missing_attribution: bool,
    timing: &S,
) {
    rejected_then_retry_with_timing_and_demand(
        missing_attribution,
        timing,
        ferrum_interfaces::execution_cost::StructuredCostSampleDemand::RuntimePolicy,
    )
}

fn rejected_then_retry_with_timing_and_demand<S: SubmissionWaveDispatchTimingSink>(
    missing_attribution: bool,
    timing: &S,
    demand: ferrum_interfaces::execution_cost::StructuredCostSampleDemand,
) {
    let (fixture, sequence, session, batch, step) = setup();
    let lane = Arc::clone(step.execution_lane());
    lane.configure_submission_readback_staging(8).unwrap();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.submission_readback_enabled = true;
        trace.guarded_missing_attribution = missing_attribution;
    }
    let reason = GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired);
    let guard = Guard {
        rejection: Some(reason),
        calls: AtomicU64::new(0),
        staging: true,
        trace: Arc::clone(&fixture.runtime_trace),
    };
    let reaper = CompletionReaper::new();
    let pending = match dispatch_with_timing_and_demand(
        &fixture, &session, &step, &reaper, &guard, timing, demand,
    ) {
        GuardedWaveSubmissionOutcome::NotSubmitted(pending) => pending,
        _ => panic!("prepared guard must reject without submitting"),
    };
    assert_eq!(
        guard.calls.load(Ordering::Relaxed),
        u64::from(!missing_attribution)
    );
    assert!(fixture.provider_trace.lock().unwrap().encode_calls > 0);
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 0);
        assert_eq!(trace.guarded_submit_calls, 1);
        assert_eq!(trace.submission_readback_live, 0);
    }
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.cost_readback_available_bytes(), Some(8));
    let receipt = pending
        .reconcile_step(step)
        .unwrap_or_else(|(error, _)| panic!("exact rejected step must roll back: {error}"));
    assert_eq!(
        receipt.reason(),
        if missing_attribution {
            GuardedNotSubmittedReason::AttributionUnavailable
        } else {
            reason
        }
    );
    // Reuse the original admitted request, session and lane. Neither abort nor
    // replacement can hide a failed frame/initialization/readback rollback.
    let next = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.guarded_missing_attribution = false;
        trace.guarded_submit_calls = 0;
    }
    let accept = Guard {
        rejection: None,
        calls: AtomicU64::new(0),
        staging: true,
        trace: Arc::clone(&fixture.runtime_trace),
    };
    let handle = accepted(dispatch_with_timing(
        &fixture, &session, &next, &reaper, &accept, timing,
    ));
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(accept.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    drop(handle);
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, next);
}

#[test]
fn guarded_core_rejection_releases_staging_and_retries_same_live_request() {
    rejected_then_retry(false);
}

#[test]
fn guarded_core_missing_attribution_rejects_before_host_gate_and_submit() {
    rejected_then_retry(true);
}

#[test]
fn guarded_cpu_row_primitives_match_actual_dispatches_and_output() {
    struct PrimitiveGuard<'a> {
        fill: &'a ControlledCpuFill,
        dispatches: u64,
    }
    impl PreparedWaveSubmissionGuard for PrimitiveGuard<'_> {
        fn check(
            &self,
            actual: &DeviceSubmissionAttribution,
            readback: CoreReadbackRoute,
        ) -> Result<(), GuardedNotSubmittedReason> {
            assert_eq!(self.fill.executed.load(Ordering::Acquire), 0);
            assert_eq!(self.fill.executed_dispatches.load(Ordering::Acquire), 0);
            assert_eq!(readback, CoreReadbackRoute::HostSynchronized);
            assert_eq!(actual.commands().len(), 2);
            for command in actual.commands() {
                assert_eq!(command.participant_count(), 2);
                assert_eq!(command.token_count(), 2);
                assert_eq!(command.compute_dispatch_count(), self.dispatches);
                assert_eq!(command.transfer_command_count(), 0);
                assert_eq!(
                    command.native_op_id(),
                    self.fill.layout.native_op_id(self.fill.full_logits)
                );
            }
            Ok(())
        }
    }

    // The expected dispatch count is independent of the fixture's helper.
    // A-only and B-only each invoke one primitive; mixed rows invoke both.
    for (contexts, expected_dispatches) in [([1, 1], 1), ([2, 2], 1), ([1, 2], 2)] {
        for full_logits in [false, true] {
            let fixture = fixture();
            let sequences = (0..2)
                .map(|index| {
                    logical_resources(
                        &fixture.plan_resources,
                        &format!("run.cpu-primitives.{index}"),
                        &format!("request.cpu-primitives.{index}"),
                    )
                })
                .collect::<Vec<_>>();
            let sessions = sequences
                .iter()
                .map(|sequence| sequence.open_session().unwrap())
                .collect::<Vec<_>>();
            let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
            let lane = fixture.plan_resources.create_execution_lane().unwrap();
            let work = contexts
                .map(|context| {
                    TokenSpanWork::from_token_ids(&vec![5; context + 1], context..context + 1)
                        .unwrap()
                })
                .to_vec();
            let request = StepResourceAdmissionRequest::new(
                batch.bind_work_shape(work).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap();
            let mut admitted = None;
            for _ in 0..4 {
                match batch.try_begin_step(request.clone(), &lane).unwrap() {
                    StepResourceAdmissionDecision::Admitted(step) => {
                        admitted = Some(step);
                        break;
                    }
                    StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                        deferred.maintain().unwrap();
                    }
                    _ => panic!("original CPU primitive step unavailable"),
                }
            }
            let step = admitted.expect("bounded CPU primitive admission");
            let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
            let active = sessions
                .iter()
                .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
                .collect::<Vec<_>>();
            let identity = OperationDispatch::bind_submission_wave_identity(
                &fixture.resolved,
                active.iter(),
                &wave,
                &lane,
            )
            .unwrap();
            let fill = ControlledCpuFill::with_layout(
                2,
                64,
                full_logits,
                ControlledCpuFillLayout::per_row(contexts),
            );
            fixture.provider_trace.lock().unwrap().controlled_cpu_fill = Some(fill.clone());
            {
                let mut trace = fixture.runtime_trace.lock().unwrap();
                trace.controlled_cpu_fill = Some(fill.clone());
                trace.structured_cost_capture_enabled = true;
                trace.cost_route_projection_enabled = true;
            }
            let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
            let reaper = CompletionReaper::new();
            let handle = accepted(OperationDispatch::encode_and_submit_guarded_wave(
                providers.providers(),
                &fixture.resolved,
                &identity,
                active.iter(),
                &[],
                &PrimitiveGuard {
                    fill: &fill,
                    dispatches: expected_dispatches,
                },
                wave,
                &lane,
                &reaper,
            ));
            assert!(matches!(
                handle.wait().unwrap(),
                CompletionObservation::Terminal(_)
            ));
            assert_eq!(fill.executed.load(Ordering::Acquire), 2);
            assert_eq!(
                fill.executed_dispatches.load(Ordering::Acquire),
                2 * expected_dispatches as usize
            );
            let output = fill.take_output();
            assert_eq!(output.len(), 2);
            for row in output {
                if full_logits {
                    assert_eq!(row.len(), 64);
                    assert_eq!(row[6], 1.0);
                    assert!(row.iter().enumerate().all(|(i, v)| i == 6 || *v == 0.0));
                } else {
                    assert_eq!(row, vec![6.0]);
                }
            }
            drop(handle);
            drop(identity);
            drop(active);
            drop(providers);
            step.try_retire_normal().unwrap();
            drop(batch);
            for session in sessions {
                session.try_complete().unwrap();
            }
        }
    }
}

#[test]
fn guarded_host_timing_reaches_backend_without_authorizing_rejected_work() {
    for missing_attribution in [false, true] {
        let timing = HostTiming::default();
        rejected_then_retry_with_timing(missing_attribution, &timing);
        // One refused attempt followed by one successful use of the same live
        // request/lane: both are observed, only the accepted fence is armed.
        assert_eq!(timing.backend_calls.load(Ordering::Relaxed), 2);
        assert!(timing.dispatch_calls.load(Ordering::Relaxed) > 0);
        assert_eq!(timing.completion_arms.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn guarded_eager_wave_uses_program_bindings_without_requesting_capture() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let guard = Guard {
        rejection: None,
        calls: AtomicU64::new(0),
        staging: false,
        trace: Arc::clone(&fixture.runtime_trace),
    };
    let reaper = CompletionReaper::new();
    let handle = accepted(dispatch(&fixture, &session, &step, &reaper, &guard));
    assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 1);
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::EagerOnly]
        );
        assert_eq!(trace.submitted_reusable_captures, vec![None]);
        assert_eq!(trace.program_binding_coalesce_calls, 1);
        assert_eq!(trace.program_binding_input_counts, vec![2]);
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::Provider,
                TestCommand::Provider,
            ]]
        );
    }
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop((handle, reaper));
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn guarded_core_submitted_handle_drop_retains_flight_and_cannot_rollback() {
    let (fixture, sequence, session, batch, step) = setup();
    let lane = Arc::clone(step.execution_lane());
    lane.configure_submission_readback_staging(8).unwrap();
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.submission_readback_enabled = true;
        trace.fence_behavior = FenceBehavior::Pending;
    }
    let guard = Guard {
        rejection: None,
        calls: AtomicU64::new(0),
        staging: true,
        trace: Arc::clone(&fixture.runtime_trace),
    };
    let reaper = CompletionReaper::new();
    let handle = accepted(dispatch(&fixture, &session, &step, &reaper, &guard));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    let failure = step.try_rollback_unsubmitted().unwrap_err();
    let step = failure.into_step();
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        1
    );
    drop(handle); // Cancellation of the caller is not physical cancellation.
    assert_eq!(reaper.retained_count(), 1);
    // This backend uses a synchronous snapshot reader (no readback command), so
    // detaching that reader releases its reservation. The submitted invocation
    // and Step still belong to the reaper until the real fence is terminal.
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    assert_eq!(lane.cost_readback_available_bytes(), None);
    fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    reaper.poll_bounded(1).unwrap();
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_readback_live,
        0
    );
    let failure = step.try_rollback_unsubmitted().unwrap_err();
    assert!(failure.error().to_string().contains("pristine"));
    let step = failure.into_step();
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn guarded_core_rejection_receipt_cannot_reconcile_a_different_step() {
    let (fixture, sequence, session, batch, step) = setup();
    let lane = Arc::clone(step.execution_lane());
    let guard = Guard {
        rejection: Some(GuardedNotSubmittedReason::ActualRouteMismatch),
        calls: AtomicU64::new(0),
        staging: false,
        trace: Arc::clone(&fixture.runtime_trace),
    };
    let reaper = CompletionReaper::new();
    let pending = match dispatch(&fixture, &session, &step, &reaper, &guard) {
        GuardedWaveSubmissionOutcome::NotSubmitted(pending) => pending,
        _ => panic!("expected unsubmitted prepared wave"),
    };
    let old_step_id = step.batch_step_id();
    step.try_rollback_unsubmitted().unwrap();
    let next = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    assert_ne!(old_step_id, next.batch_step_id());
    let (error, next) = match pending.reconcile_step(next) {
        Err(failure) => failure,
        Ok(_) => panic!("an unrelated step must not mint a reconciliation receipt"),
    };
    assert!(error.to_string().contains("another Step"));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, next);
}

#[test]
fn omitted_actual_sample_preserves_fresh_guard_rejection_and_logical_attribution() {
    use ferrum_interfaces::execution_cost::StructuredCostSampleDemand;
    rejected_then_retry_with_timing_and_demand(
        false,
        &NoHostTiming,
        StructuredCostSampleDemand::NotRequested,
    );
    rejected_then_retry_with_timing_and_demand(
        true,
        &NoHostTiming,
        StructuredCostSampleDemand::NotRequested,
    );
}

#[test]
fn guarded_completion_timing_rejects_without_submit_then_retries_once() {
    use ferrum_interfaces::execution_cost::StructuredCostSampleDemand;
    for staging in [false, true] {
        let (fixture, sequence, session, batch, step) = setup();
        let lane = Arc::clone(step.execution_lane());
        if staging {
            lane.configure_submission_readback_staging(8).unwrap();
            fixture
                .runtime_trace
                .lock()
                .unwrap()
                .submission_readback_enabled = true;
        }
        let timing = HostTiming::default();
        let reaper = CompletionReaper::new();
        let guard = Guard {
            rejection: Some(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            )),
            calls: AtomicU64::new(0),
            staging,
            trace: fixture.runtime_trace.clone(),
        };
        let rejected = dispatch_with_device_timing_and_demand(
            &fixture,
            &session,
            &step,
            &reaper,
            &guard,
            &timing,
            StructuredCostSampleDemand::RuntimePolicy,
            DeviceTimingMode::Completion,
        );
        let GuardedWaveSubmissionOutcome::NotSubmitted(pending) = rejected else {
            panic!("guard must reject");
        };
        {
            let trace = fixture.runtime_trace.lock().unwrap();
            assert_eq!(trace.submit_calls, 0);
            assert!(trace.submitted_timing_modes.is_empty());
            assert_eq!(trace.submission_readback_live, 0);
        }
        assert_eq!(timing.completion_arms.load(Ordering::Relaxed), 0);
        assert_eq!(reaper.retained_count(), 0);
        pending
            .reconcile_step(step)
            .unwrap_or_else(|(e, _)| panic!("rollback: {e}"));
        let next = begin_single_participant_step_on_lane_with_bucket(
            &batch,
            &lane,
            fixture.reusable_execution_bucket.as_ref(),
        );
        fixture.runtime_trace.lock().unwrap().guarded_submit_calls = 0;
        let accept = Guard {
            rejection: None,
            calls: AtomicU64::new(0),
            staging,
            trace: fixture.runtime_trace.clone(),
        };
        let handle = accepted(dispatch_with_device_timing_and_demand(
            &fixture,
            &session,
            &next,
            &reaper,
            &accept,
            &timing,
            StructuredCostSampleDemand::RuntimePolicy,
            DeviceTimingMode::Completion,
        ));
        let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
            panic!("terminal required");
        };
        assert!(matches!(
            receipt.fence_timing().device_execution(),
            DeviceTimingMeasurement::Measured(_)
        ));
        assert!(matches!(
            receipt.submission_timing(),
            DeviceTimingMeasurement::NotRequested
        ));
        assert_eq!(
            fixture.runtime_trace.lock().unwrap().submitted_timing_modes,
            [DeviceTimingMode::Completion]
        );
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
        assert_eq!(timing.completion_arms.load(Ordering::Relaxed), 1);
        assert_eq!(accept.calls.load(Ordering::Relaxed), 1);
        drop((handle, reaper, lane));
        teardown(fixture, sequence, session, batch, next);
    }
}
