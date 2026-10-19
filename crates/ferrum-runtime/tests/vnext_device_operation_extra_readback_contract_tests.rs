mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

struct SubmittedWave {
    fixture: Fixture,
    sequences: Vec<Arc<AdmittedSequenceResources<TestRuntime>>>,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    step: Arc<StepResourceLease<TestRuntime>>,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
    handle: Option<CompletionHandle<TestRuntime>>,
}

impl SubmittedWave {
    fn new(prepared: bool) -> Self {
        let participant_count = 2;
        let fixture = fixture_with_scalar_u32_readback();
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .terminal_readback_enabled = prepared;
        let sequences = (0..participant_count)
            .map(|ordinal| {
                logical_resources(
                    &fixture.plan_resources,
                    &format!("run.extra-readback.{ordinal}"),
                    &format!("request.extra-readback.{ordinal}"),
                )
            })
            .collect::<Vec<_>>();
        let sessions = sequences
            .iter()
            .map(|sequence| sequence.open_session().unwrap())
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
        let lane = fixture.plan_resources.create_execution_lane().unwrap();
        let step_request = StepResourceAdmissionRequest::new(
            batch
                .bind_work_shape(vec![one_token_span(); participant_count])
                .unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        let mut admitted = None;
        for _ in 0..4 {
            match batch.try_begin_step(step_request.clone(), &lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => {
                    admitted = Some(step);
                    break;
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                }
                _ => panic!("test step admission did not converge"),
            }
        }
        let step = admitted.expect("step backing converges after maintenance");
        let mut wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        wave.set_terminal_readbacks(primary(2, 4)).unwrap();
        let active_bindings = batch
            .sessions()
            .iter()
            .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
            .collect::<Vec<_>>();
        let providers = fixture
            .plan
            .payload()
            .nodes()
            .iter()
            .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
            .collect::<Vec<_>>();
        let identity = OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active_bindings.iter(),
            &wave,
            &lane,
        )
        .unwrap();
        let reaper = CompletionReaper::new();
        let handle = OperationDispatch::encode_and_submit_wave(
            &providers,
            &fixture.resolved,
            &identity,
            active_bindings.iter(),
            DeviceTimingMode::Completion,
            wave,
            &lane,
            &reaper,
        )
        .unwrap();
        drop(providers);
        Self {
            fixture,
            sequences,
            sessions,
            batch,
            step,
            lane,
            reaper,
            handle: Some(handle),
        }
    }

    fn handle(&self) -> &CompletionHandle<TestRuntime> {
        self.handle
            .as_ref()
            .expect("caller still owns its observation handle")
    }

    fn extra(&self, participant: u32, offset: u64, bytes: u64) -> CompletionExtraReadbackRequest {
        CompletionExtraReadbackRequest::new(
            &self.fixture.plan,
            output_request("node.tail", "resource.output", participant, offset, bytes),
            bytes,
        )
        .unwrap()
    }

    fn finish(self) {
        let Self {
            fixture,
            sequences,
            sessions,
            batch,
            step,
            lane,
            reaper,
            handle,
        } = self;
        assert_eq!(reaper.retained_count(), 0);
        // A readback panic poisons the lane mutex. Its public count is then
        // unavailable; the panic test separately proves the lane fails closed.
        if !lane.is_fail_closed() {
            assert_eq!(lane.in_flight_count(), 0);
        }
        drop(handle);
        step.try_retire_normal().unwrap();
        drop(batch);
        for session in &sessions {
            session.try_complete().unwrap();
        }
        drop(sessions);
        drop(sequences);
        drop(reaper);
        drop(lane);
        drop(fixture.registry);
        drop(fixture.impostor_registry);
        drop(fixture.runtime);
        assert!(matches!(
            PlanRuntimeResources::close(fixture.plan_resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
    }
}

fn primary(count: u32, offset: u64) -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new(
        (0..count)
            .map(|ordinal| {
                CompletionReadbackRequest::new_typed(
                    id("node.tail"),
                    ordinal,
                    id("resource.state"),
                    BufferUsage::State,
                    offset,
                    HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn output_request(
    node: &str,
    resource: &str,
    participant: u32,
    offset: u64,
    bytes: u64,
) -> CompletionReadbackRequest {
    CompletionReadbackRequest::new(
        id(node),
        participant,
        id(resource),
        offset,
        HostTransferLayout::new(ElementType::F32, bytes / 4).unwrap(),
    )
    .unwrap()
}

fn terminal(
    observation: CompletionReadbackWithExtraObservation,
) -> CompletionReadbackWithExtraReceipt {
    match observation {
        CompletionReadbackWithExtraObservation::Terminal(receipt) => receipt,
        other => panic!("expected original terminal with readbacks: {other:?}"),
    }
}

fn assert_success(receipt: &CompletionReadbackWithExtraReceipt) {
    assert!(matches!(
        receipt.primary().completion().disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    assert_eq!(receipt.primary().completion(), receipt.extra().completion());
    assert_eq!(receipt.primary().dispositions().len(), 2);
    for (ordinal, disposition) in receipt.primary().dispositions().iter().enumerate() {
        let CompletionReadbackDisposition::Succeeded(output) = disposition else {
            panic!("primary failed: {disposition:?}")
        };
        assert_eq!(output.request(), &primary(2, 4).requests()[ordinal]);
        assert_eq!(output.bytes(), &[0; 4]);
    }
    let CompletionReadbackDisposition::Succeeded(output) = receipt.extra().disposition() else {
        panic!("extra failed: {:?}", receipt.extra().disposition())
    };
    assert_eq!(output.request().participant_index(), 1);
    assert_eq!(output.request().resource_id(), &id("resource.output"));
    assert_eq!(output.bytes(), &[0; 16]);
}

#[test]
fn extra_readback_preserves_complete_primary_packet_and_one_original_wait() {
    for prepared in [true, false] {
        let submitted = SubmittedWave::new(prepared);
        let receipt = terminal(
            submitted
                .handle()
                .wait_with_readbacks_and_extra(primary(2, 4), submitted.extra(1, 0, 16))
                .unwrap(),
        );
        assert_success(&receipt);
        let trace = submitted.fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 1);
        assert_eq!(trace.wait_fence_calls, 1);
        assert_eq!(trace.terminal_tail_calls, usize::from(prepared));
        assert_eq!(trace.terminal_finish_calls, usize::from(prepared));
        assert_eq!(
            trace.terminal_packet_drops.load(Ordering::SeqCst),
            u64::from(prepared)
        );
        assert_eq!(trace.readback_calls, if prepared { 1 } else { 3 });
        assert_eq!(
            trace.readback_lengths,
            if prepared { vec![16] } else { vec![4, 4, 16] }
        );
        drop(trace);
        assert!(submitted.handle().wait().is_err());
        submitted.finish();
    }
}

#[test]
fn extra_readback_rejects_nonretained_budget_range_and_primary_substitution_before_wait() {
    let submitted = SubmittedWave::new(true);
    for request in [
        output_request("node.main", "resource.intermediate", 1, 0, 16),
        output_request("node.absent", "resource.output", 1, 0, 16),
        output_request("node.main", "resource.output", 1, 0, 16),
    ] {
        assert!(CompletionExtraReadbackRequest::new(&submitted.fixture.plan, request, 16).is_err());
    }
    assert!(CompletionExtraReadbackRequest::new(
        &submitted.fixture.plan,
        output_request("node.tail", "resource.output", 1, 0, 16),
        15
    )
    .is_err());
    let foreign = fixture();
    assert_ne!(foreign.plan.plan_hash(), submitted.fixture.plan.plan_hash());
    let foreign_extra = CompletionExtraReadbackRequest::new(
        &foreign.plan,
        output_request("node.tail", "resource.output", 1, 0, 16),
        16,
    )
    .unwrap();
    assert!(submitted
        .handle()
        .wait_with_readbacks_and_extra(primary(2, 4), foreign_extra)
        .is_err());
    drop(foreign.registry);
    drop(foreign.impostor_registry);
    drop(foreign.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(foreign.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    for request in [primary(1, 4), primary(2, 8)] {
        assert!(submitted
            .handle()
            .wait_with_readbacks_and_extra(request, submitted.extra(1, 0, 16))
            .is_err());
    }
    for extra in [
        submitted.extra(2, 0, 16),
        submitted.extra(1, 16, 16),
        submitted.extra(1, 1, 4),
    ] {
        assert!(submitted
            .handle()
            .wait_with_readbacks_and_extra(primary(2, 4), extra)
            .is_err());
    }
    let trace = submitted.fixture.runtime_trace.lock().unwrap();
    assert_eq!(trace.wait_fence_calls, 0);
    assert_eq!(trace.readback_calls, 0);
    assert_eq!(trace.terminal_packet_drops.load(Ordering::SeqCst), 0);
    drop(trace);
    assert_eq!(submitted.reaper.retained_count(), 1);
    // Rejection neither consumes nor replaces the original terminal packet.
    assert_success(&terminal(
        submitted
            .handle()
            .wait_with_readbacks_and_extra(primary(2, 4), submitted.extra(1, 0, 16))
            .unwrap(),
    ));
    submitted.finish();
}

#[test]
fn extra_readback_failure_is_evidence_and_original_resources_retire_once() {
    for behavior in [
        ReadbackBehavior::DeviceFailure,
        ReadbackBehavior::InvalidByteCount,
        ReadbackBehavior::Panic,
    ] {
        let submitted = SubmittedWave::new(true);
        submitted
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .readback_behavior = behavior;
        let receipt = terminal(
            submitted
                .handle()
                .wait_with_readbacks_and_extra(primary(2, 4), submitted.extra(1, 0, 16))
                .unwrap(),
        );
        assert!(receipt
            .primary()
            .dispositions()
            .iter()
            .all(|d| matches!(d, CompletionReadbackDisposition::Succeeded(_))));
        assert_eq!(receipt.primary().completion(), receipt.extra().completion());
        match behavior {
            ReadbackBehavior::DeviceFailure => assert!(
                matches!(receipt.extra().disposition(), CompletionReadbackDisposition::FailedButQuiescent { failures, .. } if !failures.is_empty())
            ),
            _ => assert!(matches!(
                receipt.extra().disposition(),
                CompletionReadbackDisposition::ContractFailedButQuiescent { .. }
            )),
        }
        let trace = submitted.fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 1);
        assert_eq!(trace.wait_fence_calls, 1);
        assert_eq!(trace.terminal_finish_calls, 1);
        assert_eq!(trace.readback_calls, 1);
        assert_eq!(trace.terminal_packet_drops.load(Ordering::SeqCst), 1);
        drop(trace);
        assert!(submitted.lane.is_fail_closed());
        assert!(submitted.handle().wait().is_err());
        submitted.finish();
    }
}

#[test]
fn extra_readback_does_not_hide_failed_primary_or_failed_execution() {
    for bad_packet in [true, false] {
        let submitted = SubmittedWave::new(true);
        {
            let mut trace = submitted.fixture.runtime_trace.lock().unwrap();
            trace.terminal_bad_count = bad_packet;
            if !bad_packet {
                trace.fence_behavior = FenceBehavior::FailedButQuiescent;
            }
        }
        let receipt = terminal(
            submitted
                .handle()
                .wait_with_readbacks_and_extra(primary(2, 4), submitted.extra(1, 0, 16))
                .unwrap(),
        );
        assert!(receipt
            .primary()
            .dispositions()
            .iter()
            .all(|d| if bad_packet {
                matches!(
                    d,
                    CompletionReadbackDisposition::ContractFailedButQuiescent { .. }
                )
            } else {
                matches!(d, CompletionReadbackDisposition::NotAttempted(_))
            }));
        assert!(matches!(
            receipt.extra().disposition(),
            CompletionReadbackDisposition::NotAttempted(_)
        ));
        assert_eq!(
            submitted
                .fixture
                .runtime_trace
                .lock()
                .unwrap()
                .readback_calls,
            0
        );
        submitted.finish();
    }
}

#[test]
fn extra_readback_unknown_and_caller_drop_keep_original_fence_until_recovery() {
    let mut submitted = SubmittedWave::new(true);
    submitted
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .fence_behavior = FenceBehavior::Indeterminate;
    assert!(matches!(
        submitted
            .handle()
            .wait_with_readbacks_and_extra(primary(2, 4), submitted.extra(1, 0, 16))
            .unwrap(),
        CompletionReadbackWithExtraObservation::Indeterminate(_)
    ));
    let slot = submitted.handle().slot_id();
    drop(submitted.handle.take());
    assert_eq!(submitted.reaper.retained_count(), 1);
    assert_eq!(submitted.lane.in_flight_count(), 1);
    assert!(submitted
        .sessions
        .iter()
        .all(|session| session.try_complete().is_err()));
    {
        let mut trace = submitted.fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.readback_calls, 0);
        assert_eq!(trace.terminal_finish_calls, 0);
        assert_eq!(trace.terminal_packet_drops.load(Ordering::SeqCst), 0);
        trace.fence_behavior = FenceBehavior::Succeeded;
    }
    // Cancellation discards optional readback intent, not the existing work.
    assert!(matches!(
        submitted.reaper.wait_slot_for_recovery(slot).unwrap(),
        CompletionObservation::Terminal(_)
    ));
    let trace = submitted.fixture.runtime_trace.lock().unwrap();
    assert_eq!(trace.submit_calls, 1);
    assert_eq!(trace.readback_calls, 0);
    assert_eq!(trace.terminal_packet_drops.load(Ordering::SeqCst), 1);
    drop(trace);
    submitted.finish();
}

#[test]
fn extra_readback_wait_survives_caller_cancellation_and_retires_original_frame() {
    let mut submitted = SubmittedWave::new(true);
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    submitted
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .wait_fence_block = Some((Arc::clone(&entered), Arc::clone(&release)));
    let handle = submitted.handle().clone();
    let extra = submitted.extra(1, 0, 16);
    let waiter =
        std::thread::spawn(move || handle.wait_with_readbacks_and_extra(primary(2, 4), extra));
    entered.wait();
    drop(submitted.handle.take());
    // Do not lock the completion slot while its original blocking wait holds it.
    let trace = submitted.fixture.runtime_trace.lock().unwrap();
    let pending_drops = trace.terminal_packet_drops.load(Ordering::SeqCst);
    let pending_readbacks = trace.readback_calls;
    drop(trace);
    release.wait();
    let receipt = terminal(waiter.join().unwrap().unwrap());
    assert_eq!(pending_drops, 0);
    assert_eq!(pending_readbacks, 0);
    assert_success(&receipt);
    assert_eq!(
        submitted
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .wait_fence_calls,
        1
    );
    submitted.finish();
}
