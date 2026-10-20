//! Faults in delivery over the actual compiled program and completion worker.
//! The upper crate supplies its real CPU embedding/head fixture and verifies
//! final runtime residency; no mock operation substitutes for model execution.
use super::token_execution::{admit_tokens, execute_one};
use super::*;

struct FailingDelivery {
    policy: ExecutionEventDeliveryPolicy,
    kind: ExecutionEventSinkErrorKind,
    once: AtomicBool,
    fail_completion: bool,
    failures: AtomicU64,
    timing_deliveries: AtomicU64,
    accepted: Mutex<Vec<ExecutionEventKind>>,
}

impl FailingDelivery {
    fn new(
        policy: ExecutionEventDeliveryPolicy,
        kind: ExecutionEventSinkErrorKind,
        fail_completion: bool,
    ) -> Self {
        Self {
            policy,
            kind,
            once: AtomicBool::new(true),
            fail_completion,
            failures: AtomicU64::new(0),
            timing_deliveries: AtomicU64::new(0),
            accepted: Mutex::new(Vec::new()),
        }
    }
}

impl ExecutionEventSink for FailingDelivery {
    fn delivery_policy(&self) -> ExecutionEventDeliveryPolicy {
        self.policy
    }
    fn device_timing_mode(&self) -> DeviceTimingMode {
        if self.fail_completion && self.policy == ExecutionEventDeliveryPolicy::OptionalTelemetry {
            // Completion requests only per-operation timing. Replay also asks
            // the real CPU fence for its physical submission observation,
            // which honestly reports that device clocks are unavailable.
            DeviceTimingMode::Replay
        } else {
            DeviceTimingMode::Off
        }
    }
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::All
    }
    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        true
    }
    fn record_physical_device_submission_timing(
        &self,
        completion: &OperationCompletionReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        assert!(matches!(
            completion.disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        assert!(matches!(
            completion.submission_timing(),
            DeviceTimingMeasurement::Unavailable(
                ferrum_runtime::DeviceTimingUnavailableReason::BackendUnsupported
            )
        ));
        self.timing_deliveries.fetch_add(1, Ordering::Relaxed);
        Err(ExecutionEventSinkError::transport(
            "physical timing delivery unavailable",
        ))
    }
    fn record(
        &self,
        permit: EventEmissionPermit,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let event = permit.event().kind();
        self.accepted.lock().push(event);
        let fail = (event == ExecutionEventKind::FrameCompleted
            && self.once.swap(false, Ordering::AcqRel))
            || (event == ExecutionEventKind::RequestCompleted && self.fail_completion);
        if !fail {
            return Ok(());
        }
        self.failures.fetch_add(1, Ordering::Relaxed);
        Err(match self.kind {
            ExecutionEventSinkErrorKind::Contract => ExecutionEventSinkError::contract(
                "intentional profile contract failure after native completion",
            ),
            ExecutionEventSinkErrorKind::Transport => ExecutionEventSinkError::transport(
                "intentional transport failure after native completion",
            ),
        })
    }
}
impl ProductExecutionEventSink for FailingDelivery {}

pub(super) async fn settled<R: DeviceRuntime>(executor: &ProgramExecutor<R>) {
    for _ in 0..2 {
        executor
            .completion_worker
            .execute(VNextCompletionTaskKind::PostSubmitDrain, || ())
            .await
            .unwrap();
    }
    assert_eq!(executor.sequences.lock().total_len(), 0);
    assert!(executor.sequences.lock().token_index.is_empty());
    assert_eq!(executor.reaper.retained_count(), 0);
    assert_eq!(executor.reaper.quarantined_count(), 0);
    assert_eq!(executor.lane.in_flight_count(), 0);
}

/// Optional queue/write failure cannot invalidate actual products or stop the
/// next request. The final event also fails, after the actual session ended.
pub async fn observability_optional_transport_preserves_actual_tokens<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let sink = Arc::new(FailingDelivery::new(
        ExecutionEventDeliveryPolicy::OptionalTelemetry,
        ExecutionEventSinkErrorKind::Transport,
        true,
    ));
    executor.attach_execution_event_sink(sink.clone());
    let before = executor.metrics.completed_waves.load(Ordering::Relaxed);
    for _ in 0..2 {
        token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor)
            .await;
    }
    assert_eq!(
        executor.metrics.completed_waves.load(Ordering::Relaxed),
        before + 6,
        "delivery failure must not replay actual work"
    );
    assert_eq!(
        sink.failures.load(Ordering::Relaxed),
        3,
        "the missing records remain observable, including both terminal deliveries"
    );
    assert_eq!(
        sink.timing_deliveries.load(Ordering::Relaxed),
        6,
        "the direct timing sink path also preserves successful native completion"
    );
    assert_eq!(
        sink.accepted
            .lock()
            .iter()
            .filter(|kind| **kind == ExecutionEventKind::FrameCompleted)
            .count(),
        6
    );
    settled(&executor).await;
    assert_executor_closed(executor)
}

/// Contract failure is fatal under either policy; required transport failure is
/// equally explicit, but neither can erase the actual successful native range.
pub async fn observability_fatal_sink_after_terminal_preserves_progress<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
    policy: ExecutionEventDeliveryPolicy,
    kind: ExecutionEventSinkErrorKind,
) -> Arc<R> {
    assert!(
        kind == ExecutionEventSinkErrorKind::Contract
            || policy == ExecutionEventDeliveryPolicy::RequiredEvidence
    );
    let sink = Arc::new(FailingDelivery::new(policy, kind, false));
    executor.attach_execution_event_sink(sink.clone());
    let request = RequestId::new();
    let initial = admit_tokens(&executor, &request, &[TokenId::new(0), TokenId::new(1)]).await;
    let before = executor.metrics.completed_waves.load(Ordering::Relaxed);
    let logits = LogitsReturnPolicy::FullLogits;
    let work = TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap());
    let input = ExecutionWork::new(&initial, work, &logits).unwrap();
    let ExecutionBatchOutcome::Interrupted(interrupted) =
        executor.execute_tokens(&[input]).await.unwrap()
    else {
        panic!("a fatal observation must be reported with actual participant progress")
    };
    assert!(interrupted.completed().is_empty());
    assert_eq!(interrupted.remaining().len(), 1);
    let participant = &interrupted.remaining()[0];
    assert_eq!(participant.sequence(), initial.sequence());
    match participant.progress() {
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Successful,
            executed_range: Some(range),
        } => {
            assert_eq!(range.start(), 0);
            assert_eq!(range.end(), 2);
        }
        other => panic!("delivery must not rewrite successful device completion: {other:?}"),
    }
    assert!(interrupted.causes().iter().any(|cause| cause
        .as_error()
        .is_some_and(|error| error.to_string().contains("intentional"))));
    let (_, mut remaining, _) = interrupted.into_parts();
    let (_, _, _, continuation, _) = remaining.pop().unwrap().into_parts();
    match continuation {
        ContinuationDisposition::Ended(ended) => assert_eq!(ended.sequence(), initial.sequence()),
        ContinuationDisposition::Held(held) => {
            let terminal = held.wait().await.unwrap();
            assert_eq!(terminal.ended.sequence(), initial.sequence());
        }
        ContinuationDisposition::Retryable(_) => {
            panic!("a completed native range cannot be replayed")
        }
    }
    assert_eq!(
        executor.metrics.completed_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert!(executor
        .execute_tokens(&[ExecutionWork::new(&initial, work, &logits).unwrap()])
        .await
        .is_err());
    assert_eq!(
        executor.metrics.completed_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert_eq!(sink.failures.load(Ordering::Relaxed), 1);

    // A separate incarnation still executes real model numerics after the
    // failed request's strict emitter has sealed.
    let peer = admit_tokens(&executor, &RequestId::new(), &[TokenId::new(1)]).await;
    let completed = execute_one(
        &executor,
        &peer,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) =
        completed.product()
    else {
        panic!("peer lost logits")
    };
    assert_eq!(values, &[3.0, 2.0, 1.0, -4.0]);
    let next = execute_one(
        &executor,
        completed.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(3),
            position: 1,
        },
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) = next.product()
    else {
        panic!("peer continuation lost logits")
    };
    assert_eq!(values, &[1.0, 1.0009765625, -1.0, -2.0]);
    let close = executor
        .finish_tokens(
            peer.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: 1,
                output_tokens: 1,
            },
        )
        .await
        .unwrap();
    assert!(close.error.is_none());
    assert!(matches!(
        close.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    settled(&executor).await;
    assert_executor_closed(executor)
}

/// Required evidence can fail at the final write without losing the actual
/// Ended receipt or retaining already released native model resources.
pub async fn observability_required_terminal_delivery_retains_actual_ended<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let sink = Arc::new(FailingDelivery::new(
        ExecutionEventDeliveryPolicy::RequiredEvidence,
        ExecutionEventSinkErrorKind::Transport,
        true,
    ));
    sink.once.store(false, Ordering::Release);
    executor.attach_execution_event_sink(sink.clone());
    let initial = admit_tokens(&executor, &RequestId::new(), &[TokenId::new(1)]).await;
    let completed = execute_one(
        &executor,
        &initial,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) =
        completed.product()
    else {
        panic!("missing actual final logits")
    };
    assert_eq!(values, &[3.0, 2.0, 1.0, -4.0]);
    let close = executor
        .finish_tokens(
            initial.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: 1,
                output_tokens: 1,
            },
        )
        .await
        .unwrap();
    assert!(close
        .error
        .as_ref()
        .is_some_and(|error| error.to_string().contains("intentional transport failure")));
    let ExecutionCloseDisposition::Ended(ended) = close.disposition else {
        panic!("failed evidence delivery cannot erase actual native terminality")
    };
    assert_eq!(ended.sequence(), initial.sequence());
    assert!(initial.sequence().observed_end().is_some());
    assert_eq!(sink.failures.load(Ordering::Relaxed), 1);
    settled(&executor).await;
    assert_executor_closed(executor)
}
