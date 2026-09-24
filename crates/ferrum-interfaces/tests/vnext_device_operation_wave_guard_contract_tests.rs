mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use ferrum_interfaces::execution_cost::{
    CoreReadbackRoute, GuardedNotSubmittedReason, HostSubmissionRejection,
};
use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

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
    OperationDispatch::encode_and_submit_guarded_wave(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        &[],
        guard,
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
    let pending = match dispatch(&fixture, &session, &step, &reaper, &guard) {
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
    let handle = accepted(dispatch(&fixture, &session, &next, &reaper, &accept));
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
