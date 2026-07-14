mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

struct CompletionHarness {
    registry: OperationRuntimeRegistry<TestRuntime>,
    resolved: ResolvedModelPlan,
    plan: ExecutionPlan,
    runtime: Arc<TestRuntime>,
    runtime_trace: Arc<Mutex<RuntimeTrace>>,
    plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
    resources: Arc<AdmittedSequenceResources<TestRuntime>>,
    session: Arc<SequenceSession<TestRuntime>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    active: TrustedActiveSequenceBinding,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
}

impl CompletionHarness {
    fn step(&self) -> &Arc<StepResourceLease<TestRuntime>> {
        self.step.as_ref().expect("completion harness owns a step")
    }

    fn new() -> Self {
        let Fixture {
            registry,
            resolved,
            plan,
            runtime,
            runtime_trace,
            plan_resources,
            ..
        } = fixture();
        let resources = logical_resources(
            &plan_resources,
            "run.device-operation.completion",
            "request.device-operation.completion",
        );
        let session = resources.open_session().unwrap();
        let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
        let step = begin_single_participant_step(&plan_resources, &batch);
        let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
        let reaper = CompletionReaper::new();
        Self {
            registry,
            resolved,
            plan,
            runtime,
            runtime_trace,
            plan_resources,
            resources,
            session,
            batch,
            step: Some(step),
            active,
            lane,
            reaper,
        }
    }

    fn dispatch(
        &self,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        self.dispatch_on_lane(&self.lane)
    }

    fn dispatch_on_lane(
        &self,
        lane: &Arc<ExecutionLane<TestRuntime>>,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        let node = &self.plan.payload().nodes()[0];
        self.dispatch_invocation_on_lane(
            admit_single_participant_invocation(&self.plan_resources, self.step(), node.id()),
            lane,
        )
    }

    fn dispatch_invocation(
        &self,
        invocation: InvocationResourceLease<TestRuntime>,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        self.dispatch_invocation_on_lane(invocation, &self.lane)
    }

    fn dispatch_invocation_on_lane(
        &self,
        invocation: InvocationResourceLease<TestRuntime>,
        lane: &Arc<ExecutionLane<TestRuntime>>,
    ) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
        let node = &self.plan.payload().nodes()[0];
        let provider = self.registry.bind(&self.resolved, node.id()).unwrap();
        let frame_id = self.step().participant_frames().next().unwrap().frame_id();
        let invocation_id = NodeInvocationId::try_from(97).unwrap();
        let identity = operation_identity(&self.plan, &self.active, frame_id, invocation_id);
        encode_and_submit_single(
            &provider,
            &self.resolved,
            &identity,
            &frame_id,
            &invocation_id,
            node.id(),
            &self.active,
            invocation,
            lane,
            &self.reaper,
        )
    }

    fn set_submit_behavior(&self, behavior: SubmitBehavior) {
        self.runtime_trace.lock().unwrap().submit_behavior = behavior;
    }

    fn set_fence_behavior(&self, behavior: FenceBehavior) {
        self.runtime_trace.lock().unwrap().fence_behavior = behavior;
    }

    fn set_fence_behavior_for(&self, fence: u64, behavior: FenceBehavior) {
        self.runtime_trace
            .lock()
            .unwrap()
            .fence_behaviors
            .insert(fence, behavior);
    }

    fn set_synchronize_fails(&self, fails: bool) {
        self.runtime_trace.lock().unwrap().synchronize_fails = fails;
    }

    fn block_next_wait(&self, entered: Arc<Barrier>, release: Arc<Barrier>) {
        self.runtime_trace.lock().unwrap().wait_fence_block = Some((entered, release));
    }

    fn drift_after_descriptor_reads(&self, reads: u64) {
        self.runtime
            .descriptor_reads_until_drift
            .store(reads, Ordering::Release);
    }

    fn set_stream_failed(&self, failed: bool) {
        self.runtime_trace.lock().unwrap().stream_failed = failed;
    }

    fn set_describe_error_panics(&self, panics: bool) {
        self.runtime_trace.lock().unwrap().describe_error_panics = panics;
    }

    fn finish(mut self, passed: &mut usize) {
        check(passed, self.reaper.retained_count() == 0);
        check(passed, self.reaper.quarantined_count() == 0);
        check(passed, self.lane.in_flight_count() == 0);
        check(
            passed,
            self.step
                .take()
                .expect("completion harness owns a final step")
                .try_retire_normal()
                .is_ok(),
        );
        check(passed, self.session.try_complete().is_ok());
        drop(self.reaper);
        drop(self.lane);
        drop(self.batch);
        drop(self.session);
        drop(self.resources);
        drop(self.runtime);
        match PlanRuntimeResources::close(self.plan_resources) {
            Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
                check(passed, receipt.released_static_resources() == 2)
            }
            Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
                panic!("completion harness retained {strong_count} root references")
            }
            Err(failure) => panic!("completion harness close failed: {:?}", failure.failure()),
        }
    }
}

fn retire_step_after_deferred_cleanup(
    mut step: Arc<StepResourceLease<TestRuntime>>,
) -> Result<StepRetirementReceipt, StepFinalizationFailure<TestRuntime>> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match step.try_retire_normal() {
            Ok(receipt) => return Ok(receipt),
            Err(failure) if Instant::now() < deadline => {
                step = failure.into_step();
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(failure) => return Err(failure),
        }
    }
}

#[test]
fn completion_reaper_drop_defers_blocking_backend_recovery() {
    let harness = CompletionHarness::new();
    harness.set_fence_behavior(FenceBehavior::Pending);
    let completion = harness.dispatch().unwrap();
    assert_eq!(harness.reaper.retained_count(), 1);
    assert_eq!(harness.lane.in_flight_count(), 1);

    let entered = Arc::new(Barrier::new(COMPLETION_DROP_TEST_WORKERS + 1));
    let release = Arc::new(Barrier::new(COMPLETION_DROP_TEST_WORKERS + 1));
    harness.block_next_wait(Arc::clone(&entered), Arc::clone(&release));
    let CompletionHarness {
        runtime,
        runtime_trace,
        plan_resources,
        resources,
        session,
        batch,
        step,
        lane,
        reaper,
        ..
    } = harness;

    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let drop_returned = std::thread::scope(|scope| {
        let drop_worker = std::thread::Builder::new()
            .name("vnext-blocking-reaper-drop".to_owned())
            .spawn_scoped(scope, move || {
                drop(reaper);
                let _ = dropped_tx.send(());
            })
            .expect("the single bounded reaper-drop worker starts");
        let drop_returned = dropped_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        drop_worker
            .join()
            .expect("the bounded reaper-drop worker does not panic");
        drop_returned
    });
    assert!(
        drop_returned,
        "CompletionReaper::drop waited on the backend"
    );
    assert_eq!(plan_resources.deferred_cleanup_status().pending(), 1);
    assert_eq!(runtime_trace.lock().unwrap().wait_fence_calls, 0);

    let maintenance_root = Arc::clone(&plan_resources);
    let (cleanup, fail_closed_while_blocked, in_flight_while_blocked) =
        std::thread::scope(|scope| {
            let cleanup_worker = std::thread::Builder::new()
                .name("vnext-completion-cleanup-recovery".to_owned())
                .spawn_scoped(scope, move || {
                    maintenance_root.maintain_deferred_cleanups(1)
                })
                .expect("the single bounded completion cleanup worker starts");
            entered.wait();
            let fail_closed_while_blocked = lane.is_fail_closed();
            let in_flight_while_blocked = lane.in_flight_count();
            release.wait();
            let cleanup = cleanup_worker
                .join()
                .expect("the bounded completion cleanup worker does not panic")
                .expect("the bounded completion cleanup call is valid");
            (cleanup, fail_closed_while_blocked, in_flight_while_blocked)
        });

    assert!(fail_closed_while_blocked);
    assert_eq!(in_flight_while_blocked, 1);
    assert_eq!(cleanup.completed(), 1);
    assert_eq!(cleanup.status_after().pending(), 0);
    let retirement = retire_step_after_deferred_cleanup(step.expect("harness owns a step"))
        .expect("deferred completion cleanup converges after backend release");
    assert_eq!(retirement.participants().len(), 1);
    assert_eq!(lane.in_flight_count(), 0);
    let trace = runtime_trace.lock().unwrap();
    assert_eq!(trace.wait_fence_calls, 1);
    assert_eq!(trace.synchronize_calls, 1);
    drop(trace);
    session.try_complete().unwrap();
    drop(completion);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));

    let quarantined = CompletionHarness::new();
    quarantined.set_fence_behavior(FenceBehavior::Indeterminate);
    let handle = quarantined.dispatch().unwrap();
    let slot_id = handle.slot_id();
    drop(handle);
    assert!(matches!(
        quarantined.reaper.wait_slot_for_recovery(slot_id),
        Ok(CompletionObservation::Indeterminate(_))
    ));
    quarantined.set_synchronize_fails(true);
    let quarantine = match quarantined
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Quarantined(receipt) => receipt,
        CompletionRecoveryOutcome::Drained(_) => panic!("failed drain released ownership"),
    };
    let CompletionHarness {
        runtime,
        runtime_trace,
        plan_resources,
        resources,
        session,
        batch,
        step,
        lane,
        reaper,
        ..
    } = quarantined;
    drop(reaper);
    assert_eq!(plan_resources.deferred_cleanup_status().pending(), 1);
    let first_cleanup = plan_resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(first_cleanup.quarantined(), 1);
    assert_eq!(first_cleanup.status_after().pending(), 1);
    assert!(quarantine.is_current());
    runtime_trace.lock().unwrap().synchronize_fails = false;
    let second_cleanup = plan_resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(second_cleanup.completed(), 1);
    assert_eq!(second_cleanup.status_after().pending(), 0);
    assert!(!quarantine.is_current());
    assert_eq!(lane.in_flight_count(), 0);
    assert!(step
        .expect("harness owns a step")
        .try_retire_normal()
        .is_ok());
    assert!(session.try_complete().is_ok());
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    assert!(matches!(
        PlanRuntimeResources::close(plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

fn completion_reaper_owns_invocations_until_quiescent_terminal(passed: &mut usize) {
    let start = *passed;
    let wrong_runtime = CompletionHarness::new();
    let (other_runtime, _) = runtime(&catalog());
    let other_lane = ExecutionLane::create(other_runtime).unwrap();
    check(
        passed,
        matches!(
            wrong_runtime.dispatch_on_lane(&other_lane),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, wrong_runtime.reaper.retained_count() == 0);
    drop(other_lane);
    wrong_runtime.finish(passed);

    let definitely_not_submitted = CompletionHarness::new();
    definitely_not_submitted.set_submit_behavior(SubmitBehavior::DefinitelyNotSubmitted);
    let retry = match definitely_not_submitted.dispatch() {
        Err(OperationDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        other => panic!("expected definitely-not-submitted retry authority, got {other:?}"),
    };
    let prior_attempt = retry.prior_attempt();
    let retry_invocation = retry.retry().unwrap();
    check(
        passed,
        retry_invocation.batch_invocation_id() != prior_attempt,
    );
    check(
        passed,
        definitely_not_submitted.reaper.retained_count() == 0,
    );
    definitely_not_submitted.set_submit_behavior(SubmitBehavior::Success);
    let retry_completion = definitely_not_submitted
        .dispatch_invocation(retry_invocation)
        .unwrap();
    assert!(matches!(
        retry_completion.poll(),
        Ok(CompletionObservation::Terminal(_))
    ));
    definitely_not_submitted.finish(passed);

    let pending = CompletionHarness::new();
    pending.set_fence_behavior(FenceBehavior::Pending);
    let handle = pending.dispatch().unwrap();
    let observer = handle.clone();
    drop(handle);
    check(passed, pending.reaper.retained_count() == 1);
    check(
        passed,
        matches!(observer.poll(), Ok(CompletionObservation::Pending)),
    );
    check(passed, pending.lane.in_flight_count() == 1);
    pending.set_fence_behavior(FenceBehavior::Succeeded);
    check(
        passed,
        matches!(observer.poll(), Ok(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    pending.finish(passed);

    let indeterminate = CompletionHarness::new();
    indeterminate.set_fence_behavior(FenceBehavior::Panic);
    let handle = indeterminate.dispatch().unwrap();
    let slot_id = handle.slot_id();
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| handle.poll()),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    check(passed, indeterminate.reaper.retained_count() == 1);
    check(
        passed,
        indeterminate
            .reaper
            .recover_slot_by_draining_lane(slot_id)
            .is_err(),
    );
    check(passed, indeterminate.lane.is_reusable());
    indeterminate.set_fence_behavior(FenceBehavior::Indeterminate);
    let failure = match handle.poll().unwrap() {
        CompletionObservation::Indeterminate(failure) => failure,
        other => panic!("indeterminate fence produced {other:?}"),
    };
    check(
        passed,
        failure.len() == 1 && failure[0].failure().code() == "test_runtime",
    );
    check(
        passed,
        failure[0].failure().message() == "fence-indeterminate",
    );
    check(passed, indeterminate.reaper.retained_count() == 1);
    indeterminate.set_fence_behavior(FenceBehavior::FailedButQuiescent);
    let terminal = match handle.wait().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("failed-but-quiescent fence was not terminal: {other:?}"),
    };
    check(
        passed,
        matches!(terminal.disposition(), OperationCompletionDisposition::FailedButQuiescent(failures) if failures.len() == 1 && failures[0].failure().code() == "test_runtime" && failures[0].failure().message() == "terminal-failure"),
    );
    indeterminate.finish(passed);

    let classification_panic = CompletionHarness::new();
    classification_panic.set_fence_behavior(FenceBehavior::Indeterminate);
    classification_panic.set_describe_error_panics(true);
    let handle = classification_panic.dispatch().unwrap();
    let slot_id = handle.slot_id();
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| handle.poll()),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    check(passed, classification_panic.reaper.retained_count() == 1);
    check(passed, classification_panic.lane.is_reusable());
    check(
        passed,
        classification_panic
            .reaper
            .recover_slot_by_draining_lane(slot_id)
            .is_err(),
    );
    classification_panic.set_fence_behavior(FenceBehavior::FailedButQuiescent);
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| handle.wait()),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    check(passed, classification_panic.reaper.retained_count() == 1);
    classification_panic.set_describe_error_panics(false);
    let drain = match classification_panic
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => {
            panic!("classification panic recovery unexpectedly quarantined")
        }
    };
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::FenceObservationPanicked
            && drain.had_submission_fence(),
    );
    check(passed, classification_panic.reaper.retained_count() == 0);
    check(passed, classification_panic.lane.is_fail_closed());
    classification_panic.finish(passed);

    let terminal_drift = CompletionHarness::new();
    let handle = terminal_drift.dispatch().unwrap();
    terminal_drift.drift_after_descriptor_reads(2);
    let terminal = match handle.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("terminal accounting drift produced {other:?}"),
    };
    check(
        passed,
        matches!(terminal.disposition(), OperationCompletionDisposition::ContractFailedButQuiescent(failure) if failure.reason().contains("terminal accounting")),
    );
    check(passed, terminal_drift.lane.is_fail_closed());
    check(passed, terminal_drift.reaper.retained_count() == 0);
    check(passed, terminal_drift.lane.in_flight_count() == 0);
    terminal_drift
        .runtime
        .use_alternate_descriptor
        .store(false, Ordering::Release);
    terminal_drift.finish(passed);

    let terminal_stream_failure = CompletionHarness::new();
    let handle = terminal_stream_failure.dispatch().unwrap();
    terminal_stream_failure.set_stream_failed(true);
    let terminal = match handle.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("terminal stream failure produced {other:?}"),
    };
    check(
        passed,
        matches!(terminal.disposition(), OperationCompletionDisposition::ContractFailedButQuiescent(failure) if failure.reason().contains("stream entered failed state")),
    );
    check(passed, terminal_stream_failure.lane.is_fail_closed());
    check(passed, terminal_stream_failure.reaper.retained_count() == 0);
    check(passed, terminal_stream_failure.lane.in_flight_count() == 0);
    terminal_stream_failure.set_stream_failed(false);
    terminal_stream_failure.finish(passed);

    let mut multiple = CompletionHarness::new();
    let second_resources = logical_resources(
        &multiple.plan_resources,
        "run.device-operation.completion.second",
        "request.device-operation.completion.second",
    );
    let second_session = second_resources.open_session().unwrap();
    let second_active = TrustedActiveSequenceBinding::from_session(&second_session).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&second_session)]).unwrap();
    let second_step = begin_single_participant_step(&multiple.plan_resources, &second_batch);
    let first = multiple.dispatch().unwrap();
    let node_id = multiple.plan.payload().nodes()[0].id().clone();
    let provider = multiple
        .registry
        .bind(&multiple.resolved, &node_id)
        .unwrap();
    let second_frame_id = second_step.participant_frames().next().unwrap().frame_id();
    let second_invocation_id = NodeInvocationId::try_from(98).unwrap();
    let second_identity = operation_identity(
        &multiple.plan,
        &second_active,
        second_frame_id,
        second_invocation_id,
    );
    let second = encode_and_submit_single(
        &provider,
        &multiple.resolved,
        &second_identity,
        &second_frame_id,
        &second_invocation_id,
        &node_id,
        &second_active,
        admit_single_participant_invocation(&multiple.plan_resources, &second_step, &node_id),
        &multiple.lane,
        &multiple.reaper,
    )
    .unwrap();
    let first_slot = first.slot_id();
    let second_slot = second.slot_id();
    check(passed, first_slot.get() < second_slot.get());
    check(passed, multiple.lane.in_flight_count() == 2);
    check(passed, multiple.reaper.retained_count() == 2);
    multiple.set_fence_behavior_for(1, FenceBehavior::Pending);
    multiple.set_fence_behavior_for(2, FenceBehavior::Succeeded);
    let first_sweep = multiple.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        first_sweep.entries().len() == 1 && first_sweep.entries()[0].slot_id() == first_slot,
    );
    check(
        passed,
        matches!(
            first_sweep.entries()[0].observation(),
            CompletionSweepObservation::Observed(CompletionObservation::Pending)
        ),
    );
    check(passed, first_sweep.retained_after() == 2);
    let second_sweep = multiple.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        second_sweep.entries().len() == 1 && second_sweep.entries()[0].slot_id() == second_slot,
    );
    check(
        passed,
        matches!(second_sweep.entries()[0].observation(), CompletionSweepObservation::Observed(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    check(passed, multiple.lane.in_flight_count() == 1);
    check(passed, multiple.reaper.retained_count() == 1);
    check(
        passed,
        matches!(first.poll(), Ok(CompletionObservation::Pending)),
    );
    multiple.set_fence_behavior_for(1, FenceBehavior::FailedButQuiescent);
    check(
        passed,
        matches!(first.poll(), Ok(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::FailedButQuiescent(_))),
    );
    check(passed, multiple.lane.in_flight_count() == 0);
    check(passed, multiple.reaper.retained_count() == 0);

    drop(second);
    drop(first);
    multiple
        .step
        .take()
        .expect("multiple-slot harness owns its first step")
        .try_retire_normal()
        .expect("first multiple-slot step is quiescent");
    multiple.step = Some(begin_single_participant_step(
        &multiple.plan_resources,
        &multiple.batch,
    ));
    assert!(second_step.try_retire_normal().is_ok());
    let second_step = begin_single_participant_step(&multiple.plan_resources, &second_batch);
    multiple.set_fence_behavior_for(3, FenceBehavior::Indeterminate);
    multiple.set_fence_behavior_for(4, FenceBehavior::Succeeded);
    let drain_target = multiple.dispatch().unwrap();
    let second_frame_id = second_step.participant_frames().next().unwrap().frame_id();
    let second_invocation_id = NodeInvocationId::try_from(100).unwrap();
    let second_identity = operation_identity(
        &multiple.plan,
        &second_active,
        second_frame_id,
        second_invocation_id,
    );
    let drain_sibling = encode_and_submit_single(
        &provider,
        &multiple.resolved,
        &second_identity,
        &second_frame_id,
        &second_invocation_id,
        &node_id,
        &second_active,
        admit_single_participant_invocation(&multiple.plan_resources, &second_step, &node_id),
        &multiple.lane,
        &multiple.reaper,
    )
    .unwrap();
    let drain_target_slot = drain_target.slot_id();
    check(
        passed,
        multiple.lane.in_flight_count() == 2 && multiple.reaper.retained_count() == 2,
    );
    check(
        passed,
        matches!(drain_target.poll(), Ok(CompletionObservation::Indeterminate(failures)) if failures.len() == 1 && failures[0].failure().message() == "fence-indeterminate"),
    );
    check(
        passed,
        matches!(drain_target.wait(), Ok(CompletionObservation::Indeterminate(failures)) if failures.len() == 1 && failures[0].failure().message() == "fence-indeterminate"),
    );
    let drain = match multiple
        .reaper
        .recover_slot_by_draining_lane(drain_target_slot)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => {
            panic!("selective lane drain unexpectedly quarantined")
        }
    };
    check(
        passed,
        drain.slot_id() == drain_target_slot
            && drain.cause() == CompletionRecoveryCause::FenceIndeterminate
            && drain.had_submission_fence(),
    );
    check(
        passed,
        multiple.lane.is_fail_closed() && multiple.lane.in_flight_count() == 1,
    );
    check(passed, multiple.reaper.retained_count() == 1);
    check(
        passed,
        multiple.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(
        passed,
        matches!(drain_sibling.poll(), Ok(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    check(
        passed,
        multiple.lane.in_flight_count() == 0 && multiple.reaper.retained_count() == 0,
    );
    drop(drain_sibling);
    drop(drain_target);
    check(passed, second_step.try_retire_normal().is_ok());
    check(passed, second_session.try_complete().is_ok());
    drop(second_batch);
    drop(second_session);
    drop(second_resources);
    multiple.finish(passed);

    let detached = CompletionHarness::new();
    detached.set_fence_behavior(FenceBehavior::Pending);
    let handle = detached.dispatch().unwrap();
    let slot_id = handle.slot_id();
    drop(handle);
    check(passed, detached.reaper.retained_count() == 1);
    check(passed, detached.reaper.poll_bounded(0).is_err());
    check(
        passed,
        detached
            .reaper
            .poll_bounded(MAX_COMPLETION_SWEEP_SLOTS + 1)
            .is_err(),
    );
    detached.set_fence_behavior(FenceBehavior::Succeeded);
    let sweep = detached.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        sweep.entries().len() == 1 && sweep.entries()[0].slot_id() == slot_id,
    );
    check(
        passed,
        matches!(sweep.entries()[0].observation(), CompletionSweepObservation::Observed(CompletionObservation::Terminal(receipt)) if matches!(receipt.disposition(), OperationCompletionDisposition::Succeeded)),
    );
    check(passed, sweep.retained_after() == 0);
    check(passed, sweep.quarantined_after() == 0);
    check(passed, detached.lane.in_flight_count() == 0);
    detached.finish(passed);

    let recovered = CompletionHarness::new();
    recovered.set_fence_behavior(FenceBehavior::Indeterminate);
    let handle = recovered.dispatch().unwrap();
    let expected_identity = handle.batch_identity().clone();
    drop(handle);
    let sweep = recovered.reaper.poll_bounded(1).unwrap();
    check(passed, sweep.entries().len() == 1);
    let slot_id = sweep.entries()[0].slot_id();
    check(
        passed,
        matches!(
            sweep.entries()[0].observation(),
            CompletionSweepObservation::Observed(CompletionObservation::Indeterminate(_))
        ),
    );
    check(
        passed,
        recovered
            .reaper
            .recover_slot_by_draining_lane(slot_id)
            .is_err(),
    );
    check(passed, recovered.lane.is_reusable());
    check(passed, recovered.reaper.retained_count() == 1);
    check(
        passed,
        matches!(recovered.reaper.wait_slot_for_recovery(slot_id), Ok(CompletionObservation::Indeterminate(failures)) if failures.len() == 1 && failures[0].failure().message() == "fence-indeterminate"),
    );
    let drain = match recovered
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("successful drain was quarantined"),
    };
    check(passed, drain.slot_id() == slot_id);
    check(passed, drain.batch_identity() == &expected_identity);
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::FenceIndeterminate,
    );
    check(passed, drain.had_submission_fence());
    check(
        passed,
        recovered.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(passed, recovered.reaper.retained_count() == 0);
    check(passed, recovered.lane.in_flight_count() == 0);
    check(passed, recovered.lane.is_fail_closed());
    let failed_lane_resources = logical_resources(
        &recovered.plan_resources,
        "run.device-operation.completion.failed-lane",
        "request.device-operation.completion.failed-lane",
    );
    let failed_lane_session = failed_lane_resources.open_session().unwrap();
    let failed_lane_active =
        TrustedActiveSequenceBinding::from_session(&failed_lane_session).unwrap();
    let failed_lane_batch =
        ExecutionBatchParticipants::new(vec![Arc::clone(&failed_lane_session)]).unwrap();
    let failed_lane_step =
        begin_single_participant_step(&recovered.plan_resources, &failed_lane_batch);
    let failed_lane_frame = failed_lane_step
        .participant_frames()
        .next()
        .unwrap()
        .frame_id();
    let failed_lane_invocation_id = NodeInvocationId::try_from(101).unwrap();
    let failed_lane_identity = operation_identity(
        &recovered.plan,
        &failed_lane_active,
        failed_lane_frame,
        failed_lane_invocation_id,
    );
    let failed_lane_node = &recovered.plan.payload().nodes()[0];
    let failed_lane_provider = recovered
        .registry
        .bind(&recovered.resolved, failed_lane_node.id())
        .unwrap();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &failed_lane_provider,
                &recovered.resolved,
                &failed_lane_identity,
                &failed_lane_frame,
                &failed_lane_invocation_id,
                failed_lane_node.id(),
                &failed_lane_active,
                admit_single_participant_invocation(
                    &recovered.plan_resources,
                    &failed_lane_step,
                    failed_lane_node.id(),
                ),
                &recovered.lane,
                &recovered.reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(
        passed,
        recovered.runtime_trace.lock().unwrap().submit_calls == 1,
    );
    assert!(failed_lane_step.try_retire_normal().is_ok());
    assert!(failed_lane_session.try_complete().is_ok());
    drop(failed_lane_batch);
    drop(failed_lane_session);
    drop(failed_lane_resources);
    recovered.finish(passed);

    let wait_panic = CompletionHarness::new();
    wait_panic.set_fence_behavior(FenceBehavior::Panic);
    let handle = wait_panic.dispatch().unwrap();
    let slot_id = handle.slot_id();
    drop(handle);
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| wait_panic.reaper.wait_slot_for_recovery(slot_id)),
            Ok(CompletionObservation::ObservationPanicked)
        ),
    );
    let drain = match wait_panic
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("wait panic drain was quarantined"),
    };
    check(passed, drain.slot_id() == slot_id);
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::FenceObservationPanicked,
    );
    check(passed, drain.had_submission_fence());
    check(
        passed,
        wait_panic.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(passed, wait_panic.reaper.retained_count() == 0);
    check(passed, wait_panic.lane.is_fail_closed());
    wait_panic.finish(passed);

    let quarantined = CompletionHarness::new();
    quarantined.set_fence_behavior(FenceBehavior::Indeterminate);
    let handle = quarantined.dispatch().unwrap();
    let expected_identity = handle.batch_identity().clone();
    drop(handle);
    let sweep = quarantined.reaper.poll_bounded(1).unwrap();
    check(passed, sweep.entries().len() == 1);
    let slot_id = sweep.entries()[0].slot_id();
    check(
        passed,
        matches!(
            sweep.entries()[0].observation(),
            CompletionSweepObservation::Observed(CompletionObservation::Indeterminate(_))
        ),
    );
    check(
        passed,
        matches!(
            quarantined.reaper.wait_slot_for_recovery(slot_id),
            Ok(CompletionObservation::Indeterminate(_))
        ),
    );
    quarantined.set_synchronize_fails(true);
    let quarantine = match quarantined
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Quarantined(receipt) => receipt,
        CompletionRecoveryOutcome::Drained(_) => panic!("failed drain released ownership"),
    };
    let stale_quarantine = quarantine.clone();
    check(
        passed,
        quarantine.is_current() && stale_quarantine.is_current(),
    );
    check(
        passed,
        serde_json::to_value(&quarantine)
            .unwrap()
            .get("freshness")
            .is_none(),
    );
    check(passed, quarantine.slot_id() == slot_id);
    check(passed, quarantine.batch_identity() == &expected_identity);
    check(
        passed,
        quarantine.cause() == CompletionRecoveryCause::FenceIndeterminate,
    );
    check(passed, quarantine.had_submission_fence());
    check(
        passed,
        quarantine.device_id() == &quarantined.lane.descriptor().id,
    );
    check(
        passed,
        quarantine.runtime_implementation_fingerprint()
            == quarantined
                .lane
                .descriptor()
                .runtime_implementation_fingerprint,
    );
    check(passed, quarantined.reaper.retained_count() == 1);
    check(passed, quarantined.reaper.quarantined_count() == 1);
    check(passed, quarantined.lane.is_fail_closed());
    let sweep = quarantined.reaper.poll_bounded(1).unwrap();
    check(
        passed,
        matches!(sweep.entries()[0].observation(), CompletionSweepObservation::Observed(CompletionObservation::Quarantined(receipt)) if receipt == &quarantine),
    );
    check(
        passed,
        sweep.retained_after() == 1 && sweep.quarantined_after() == 1,
    );
    quarantined.set_synchronize_fails(false);
    let drain = match quarantined
        .reaper
        .recover_slot_by_draining_lane(slot_id)
        .unwrap()
    {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("retry drain remained quarantined"),
    };
    check(
        passed,
        drain.slot_id() == slot_id && drain.cause() == CompletionRecoveryCause::FenceIndeterminate,
    );
    check(
        passed,
        !quarantine.is_current() && !stale_quarantine.is_current(),
    );
    check(
        passed,
        quarantined.runtime_trace.lock().unwrap().synchronize_calls == 2,
    );
    check(passed, quarantined.reaper.retained_count() == 0);
    check(passed, quarantined.reaper.quarantined_count() == 0);
    check(passed, quarantined.lane.in_flight_count() == 0);
    quarantined.finish(passed);

    let submit_panic = CompletionHarness::new();
    submit_panic.set_submit_behavior(SubmitBehavior::Panic);
    let recovery = match suppress_expected_panic_hook(|| submit_panic.dispatch()) {
        Err(OperationDispatchError::SubmissionIndeterminate { recovery }) => recovery,
        _ => panic!("submit panic did not retain recovery ownership"),
    };
    let slot_id = recovery.slot_id();
    check(passed, submit_panic.reaper.retained_count() == 1);
    let drain = match recovery.recover_by_draining_lane().unwrap() {
        CompletionRecoveryOutcome::Drained(receipt) => receipt,
        CompletionRecoveryOutcome::Quarantined(_) => panic!("submit panic drain was quarantined"),
    };
    check(passed, drain.slot_id() == slot_id);
    check(
        passed,
        drain.cause() == CompletionRecoveryCause::SubmissionIndeterminate,
    );
    check(passed, !drain.had_submission_fence());
    check(
        passed,
        submit_panic.runtime_trace.lock().unwrap().synchronize_calls == 1,
    );
    check(passed, submit_panic.reaper.retained_count() == 0);
    check(passed, submit_panic.lane.is_fail_closed());
    submit_panic.finish(passed);

    let drop_fallback = CompletionHarness::new();
    drop_fallback.set_submit_behavior(SubmitBehavior::Panic);
    check(
        passed,
        matches!(
            suppress_expected_panic_hook(|| drop_fallback.dispatch()),
            Err(OperationDispatchError::SubmissionIndeterminate { .. })
        ),
    );
    check(passed, drop_fallback.reaper.retained_count() == 1);
    let CompletionHarness {
        reaper,
        lane,
        step,
        session,
        batch,
        resources,
        runtime,
        plan_resources,
        ..
    } = drop_fallback;
    drop(reaper);
    let cleanup = plan_resources.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(cleanup.completed(), 1);
    assert_eq!(cleanup.status_after().pending(), 0);
    check(
        passed,
        retire_step_after_deferred_cleanup(step.expect("harness owns a step")).is_ok()
            && lane.in_flight_count() == 0,
    );
    session.try_complete().unwrap();
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    drop(runtime);
    check(
        passed,
        matches!(
            PlanRuntimeResources::close(plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ),
    );
    assert_eq!(*passed - start, EXPECTED_COMPLETION_CASES);
}

#[test]
fn device_operation_completion_contract_is_exhaustive() {
    let mut passed = 0;
    completion_reaper_owns_invocations_until_quiescent_terminal(&mut passed);
    assert_eq!(passed, EXPECTED_COMPLETION_CASES);
    println!("\nVNEXT DEVICE OPERATION COMPLETION PASS: {passed}/{EXPECTED_COMPLETION_CASES}");
}
