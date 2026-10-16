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
    handle: CompletionHandle<TestRuntime>,
}

impl SubmittedWave {
    fn new(participant_count: usize) -> Self {
        let fixture = fixture();
        let sequences = (0..participant_count)
            .map(|ordinal| {
                logical_resources(
                    &fixture.plan_resources,
                    &format!("run.readback-selection.{ordinal}"),
                    &format!("request.readback-selection.{ordinal}"),
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
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
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
            handle,
        }
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

fn group(
    node: &str,
    resource: &str,
    count: u32,
    offset: u64,
    elements: u64,
) -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new(
        (0..count)
            .map(|ordinal| {
                CompletionReadbackRequest::new(
                    id(node),
                    ordinal,
                    id(resource),
                    offset,
                    HostTransferLayout::new(ElementType::F32, elements).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn selection(count: u32, choices: Vec<u32>) -> CompletionReadbackSelectionRequest {
    CompletionReadbackSelectionRequest::new(
        vec![
            group("node.tail", "resource.output", count, 0, 4),
            group("node.main", "resource.intermediate", count, 4, 1),
        ],
        choices,
    )
    .unwrap()
}

#[test]
fn selected_terminal_wait_reads_one_exact_range_per_participant_and_retires_once() {
    let submitted = SubmittedWave::new(2);
    submitted
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .readback_fill_pattern = vec![7, 9];
    let requested = selection(2, vec![1, 0]);
    let expected = requested.selected_requests().cloned().collect::<Vec<_>>();
    let receipt = match submitted
        .handle
        .wait_with_readback_selection(requested)
        .unwrap()
    {
        CompletionReadbackSelectionObservation::Terminal(receipt) => receipt,
        other => panic!("selected readback did not terminate: {other:?}"),
    };
    assert_eq!(receipt.dispositions().len(), 2);
    for ((disposition, expected), fill) in receipt.dispositions().iter().zip(&expected).zip([7, 9])
    {
        let CompletionReadbackDisposition::Succeeded(output) = disposition else {
            panic!("selected readback failed: {disposition:?}");
        };
        assert_eq!(output.request(), expected);
        assert_eq!(
            output.bytes(),
            vec![fill; expected.output_layout().byte_len().unwrap() as usize]
        );
    }
    {
        let trace = submitted.fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 1);
        assert_eq!(trace.wait_fence_calls, 1);
        assert_eq!(trace.readback_calls, 2);
        assert_eq!(trace.readback_lengths, [4, 16]);
    }
    assert!(submitted
        .handle
        .wait_with_readback_selection(selection(2, vec![1, 0]))
        .is_err());
    submitted.finish();
}

#[test]
fn incomplete_or_absent_unused_alternatives_are_rejected_before_observing_the_fence() {
    let submitted = SubmittedWave::new(2);
    assert!(submitted
        .handle
        .wait_with_readback_selection(selection(1, vec![0]))
        .is_err());
    let absent = CompletionReadbackSelectionRequest::new(
        vec![
            group("node.tail", "resource.output", 2, 0, 4),
            group("node.absent", "resource.output", 2, 0, 4),
        ],
        vec![0, 0],
    )
    .unwrap();
    assert!(submitted
        .handle
        .wait_with_readback_selection(absent)
        .is_err());
    assert_eq!(submitted.reaper.retained_count(), 1);
    assert_eq!(
        submitted
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .wait_fence_calls,
        0
    );
    assert!(matches!(
        submitted
            .handle
            .wait_with_readback_selection(selection(2, vec![0, 0]))
            .unwrap(),
        CompletionReadbackSelectionObservation::Terminal(_)
    ));
    submitted.finish();
}

#[test]
fn selected_readbacks_preserve_quiescent_device_and_invalid_size_failure_evidence() {
    for behavior in [
        ReadbackBehavior::DeviceFailure,
        ReadbackBehavior::InvalidByteCount,
        ReadbackBehavior::Panic,
    ] {
        let submitted = SubmittedWave::new(2);
        submitted
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .readback_behavior = behavior;
        let receipt = match submitted
            .handle
            .wait_with_readback_selection(selection(2, vec![1, 0]))
            .unwrap()
        {
            CompletionReadbackSelectionObservation::Terminal(receipt) => receipt,
            other => panic!("selected failure did not terminate: {other:?}"),
        };
        assert!(matches!(
            receipt.completion().disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        assert_eq!(receipt.dispositions().len(), 2);
        match behavior {
            ReadbackBehavior::DeviceFailure => assert!(
                matches!(&receipt.dispositions()[0], CompletionReadbackDisposition::FailedButQuiescent { request, failures } if request.participant_index() == 0 && !failures.is_empty())
            ),
            ReadbackBehavior::InvalidByteCount | ReadbackBehavior::Panic => assert!(
                matches!(&receipt.dispositions()[0], CompletionReadbackDisposition::ContractFailedButQuiescent { request, .. } if request.participant_index() == 0)
            ),
            ReadbackBehavior::Succeeded => unreachable!(),
        }
        if matches!(behavior, ReadbackBehavior::Panic) {
            assert!(submitted.lane.is_fail_closed());
        }
        submitted.finish();
    }
}

#[test]
fn indeterminate_or_panicked_observation_retains_selected_wait_ownership_until_terminal() {
    for behavior in [FenceBehavior::Indeterminate, FenceBehavior::Panic] {
        let submitted = SubmittedWave::new(2);
        submitted
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .fence_behavior = behavior;
        let observation = submitted
            .handle
            .wait_with_readback_selection(selection(2, vec![1, 0]))
            .unwrap();
        match behavior {
            FenceBehavior::Indeterminate => assert!(matches!(
                observation,
                CompletionReadbackSelectionObservation::Indeterminate(_)
            )),
            FenceBehavior::Panic => assert!(matches!(
                observation,
                CompletionReadbackSelectionObservation::ObservationPanicked
            )),
            _ => unreachable!(),
        }
        assert_eq!(submitted.reaper.retained_count(), 1);
        assert_eq!(submitted.lane.in_flight_count(), 1);
        assert_eq!(
            submitted
                .fixture
                .runtime_trace
                .lock()
                .unwrap()
                .readback_calls,
            0
        );
        submitted
            .fixture
            .runtime_trace
            .lock()
            .unwrap()
            .fence_behavior = FenceBehavior::Succeeded;
        assert!(matches!(
            submitted
                .handle
                .wait_with_readback_selection(selection(2, vec![1, 0]))
                .unwrap(),
            CompletionReadbackSelectionObservation::Terminal(_)
        ));
        submitted.finish();
    }
}

#[test]
fn failed_terminal_completion_does_not_attempt_selected_transfers() {
    let submitted = SubmittedWave::new(2);
    submitted
        .fixture
        .runtime_trace
        .lock()
        .unwrap()
        .fence_behavior = FenceBehavior::FailedButQuiescent;
    let receipt = match submitted
        .handle
        .wait_with_readback_selection(selection(2, vec![1, 0]))
        .unwrap()
    {
        CompletionReadbackSelectionObservation::Terminal(receipt) => receipt,
        other => panic!("failed terminal did not terminate: {other:?}"),
    };
    assert!(receipt
        .dispositions()
        .iter()
        .all(|disposition| matches!(disposition, CompletionReadbackDisposition::NotAttempted(_))));
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
