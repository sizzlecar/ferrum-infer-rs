//! Actual submission/receipt assertions over the upper crate's numerical CPU
//! program. Scalar snapshots retain no runtime or sequence authority.
use super::lifecycle::WorkerGate;
use super::observability::settled;
use super::token_execution::admit_tokens;
use super::*;

#[derive(Clone, Copy)]
pub enum FactsFailure {
    None,
    Contract,
    RequiredTransport,
    OptionalTransport,
    CallbackPanic,
    PolicyPanic,
    CaptureQueryPanic,
}

#[derive(Debug)]
struct WorkSnapshot {
    nodes: Vec<(u32, Vec<(std::ops::Range<u64>, std::ops::Range<u64>, u64)>)>,
}

struct FactsSink {
    capture: ExecutionFactsCapture,
    failure: FactsFailure,
    fail_once: AtomicBool,
    snapshots: Mutex<Vec<WorkSnapshot>>,
}

impl FactsSink {
    fn new(capture: ExecutionFactsCapture, failure: FactsFailure) -> Self {
        Self {
            capture,
            failure,
            fail_once: AtomicBool::new(true),
            snapshots: Mutex::new(Vec::new()),
        }
    }
}

impl ExecutionEventSink for FactsSink {
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::None
    }
    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        false
    }
    fn record(&self, _: EventEmissionPermit) -> std::result::Result<(), ExecutionEventSinkError> {
        Ok(())
    }
    fn execution_facts_capture(&self) -> ExecutionFactsCapture {
        if matches!(self.failure, FactsFailure::CaptureQueryPanic)
            && self.fail_once.swap(false, Ordering::AcqRel)
        {
            panic!("intentional facts capture-query panic");
        }
        self.capture
    }
    fn device_timing_mode(&self) -> DeviceTimingMode {
        DeviceTimingMode::Off
    }
    fn delivery_policy(&self) -> ExecutionEventDeliveryPolicy {
        if matches!(self.failure, FactsFailure::PolicyPanic) {
            panic!("intentional facts delivery-policy panic");
        }
        if matches!(
            self.failure,
            FactsFailure::OptionalTransport | FactsFailure::Contract
        ) {
            ExecutionEventDeliveryPolicy::OptionalTelemetry
        } else {
            ExecutionEventDeliveryPolicy::RequiredEvidence
        }
    }
    fn record_execution_facts(
        &self,
        facts: TerminalExecutionFacts<'_>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        assert_eq!(facts.work().submission(), facts.completion().submission());
        assert!(matches!(
            facts.completion().disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        assert!(matches!(
            facts.completion().submission_timing(),
            DeviceTimingMeasurement::NotRequested
        ));
        assert!(
            facts.attribution().is_none(),
            "work-only capture must not force attribution"
        );
        assert_eq!(
            facts.device_coverage(),
            DeviceExecutionFactsCoverage::Unavailable(
                DeviceExecutionFactsUnavailableReason::NotCaptured
            )
        );
        assert!(!facts.work().nodes().is_empty());
        self.snapshots.lock().push(WorkSnapshot {
            nodes: facts
                .work()
                .nodes()
                .iter()
                .map(|node| {
                    (
                        node.node_index(),
                        node.participants()
                            .iter()
                            .map(|row| {
                                (
                                    row.immediate_token_range(),
                                    row.source_token_range(),
                                    row.full_input_tokens(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        });
        if !self.fail_once.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        match self.failure {
            FactsFailure::None | FactsFailure::CaptureQueryPanic => Ok(()),
            FactsFailure::Contract => Err(ExecutionEventSinkError::contract(
                "intentional terminal facts contract failure",
            )),
            FactsFailure::RequiredTransport
            | FactsFailure::OptionalTransport
            | FactsFailure::PolicyPanic => Err(ExecutionEventSinkError::transport(
                "intentional terminal facts transport failure",
            )),
            FactsFailure::CallbackPanic => panic!("intentional terminal facts callback panic"),
        }
    }
}
impl ProductExecutionEventSink for FactsSink {}

/// Same partial prefill, final prefill and decode numerical oracle under both
/// independent capture settings, with device timing disabled throughout.
pub async fn execution_facts_on_off_preserve_actual_tokens<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let off = Arc::new(FactsSink::new(
        ExecutionFactsCapture::Off,
        FactsFailure::None,
    ));
    executor.attach_execution_event_sink(off.clone());
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor).await;
    assert!(off.snapshots.lock().is_empty());
    let on = Arc::new(FactsSink::new(
        ExecutionFactsCapture::On,
        FactsFailure::None,
    ));
    executor.attach_execution_event_sink(on.clone());
    token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor).await;
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 6
    );
    let snapshots = on.snapshots.lock();
    assert_eq!(snapshots.len(), 3);
    for (index, snapshot) in snapshots.iter().enumerate() {
        for (_, rows) in &snapshot.nodes {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].0, 0..1);
            assert_eq!(rows[0].1, index as u64..index as u64 + 1);
            assert_eq!(rows[0].2, if index < 2 { 2 } else { 3 });
        }
    }
    drop(snapshots);
    settled(&executor).await;
    assert_executor_closed(executor)
}

pub async fn execution_facts_optional_transport_preserves_actual_tokens<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let sink = Arc::new(FactsSink::new(
        ExecutionFactsCapture::On,
        FactsFailure::OptionalTransport,
    ));
    executor.attach_execution_event_sink(sink.clone());
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor).await;
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 3
    );
    assert_eq!(sink.snapshots.lock().len(), 3);
    settled(&executor).await;
    assert_executor_closed(executor)
}

pub async fn execution_facts_fatal_terminal_preserves_progress<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
    failure: FactsFailure,
) -> Arc<R> {
    let sink = Arc::new(FactsSink::new(ExecutionFactsCapture::On, failure));
    executor.attach_execution_event_sink(sink.clone());
    let initial = admit_tokens(
        &executor,
        &RequestId::new(),
        &[TokenId::new(0), TokenId::new(1)],
    )
    .await;
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let policy = LogitsReturnPolicy::FullLogits;
    let work = TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap());
    let inputs = [ExecutionWork::new(&initial, work, &policy).unwrap()];
    let ExecutionBatchOutcome::Interrupted(interrupted) =
        executor.execute_tokens(&inputs).await.unwrap()
    else {
        panic!("fatal terminal observer must retain actual progress");
    };
    assert!(interrupted.completed().is_empty());
    assert_eq!(interrupted.remaining().len(), 1);
    assert!(
        matches!(interrupted.remaining()[0].progress(), ExecutionProgress::Terminal { disposition: TerminalModelDisposition::Successful, executed_range: Some(range) } if range.start() == 0 && range.end() == 2)
    );
    assert_eq!(
        interrupted.causes().len(),
        1,
        "the primary observer error must not be duplicated from the same sequence settlement"
    );
    assert!(interrupted.causes()[0]
        .as_error()
        .is_some_and(|error| error.to_string().contains("execution observation failed")));
    let (_, mut remaining, _) = interrupted.into_parts();
    let (_, _, _, continuation, _) = remaining.pop().unwrap().into_parts();
    match continuation {
        ContinuationDisposition::Ended(ended) => assert_eq!(ended.sequence(), initial.sequence()),
        ContinuationDisposition::Held(held) => {
            let terminal = held.wait().await.unwrap();
            assert_eq!(terminal.ended.sequence(), initial.sequence());
            assert!(terminal.error.is_some());
        }
        ContinuationDisposition::Retryable(_) => panic!("actual terminal work cannot replay"),
    }
    assert!(executor.execute_tokens(&inputs).await.is_err());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert_eq!(sink.snapshots.lock().len(), 1);
    // The next genuine numerical wave is independent of the failed observer.
    let healthy = Arc::new(FactsSink::new(
        ExecutionFactsCapture::On,
        FactsFailure::None,
    ));
    executor.attach_execution_event_sink(healthy);
    token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor).await;
    settled(&executor).await;
    assert_executor_closed(executor)
}

/// The original CPU submission is real and its wait is queued. Dropping its
/// caller moves the sole observation to the original Held cohort. A later sink
/// is used only for later submissions; no retrofit and no second delivery.
pub async fn execution_facts_cancelled_caller_preserves_original_sink<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
    failure: FactsFailure,
    capture: ExecutionFactsCapture,
) -> Arc<R> {
    let original = Arc::new(FactsSink::new(capture, failure));
    executor.attach_execution_event_sink(original.clone());
    let initial = admit_tokens(&executor, &RequestId::new(), &[TokenId::new(1)]).await;
    let owner = executor
        .sequences
        .lock()
        .token_index
        .get(initial.sequence())
        .unwrap()
        .clone();
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [ExecutionWork::new(
        &initial,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        &policy,
    )
    .unwrap()];
    let gate = WorkerGate::closed(&executor.completion_worker).await;
    let mut pending = Box::pin(executor.execute_tokens(&inputs));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert_eq!(executor.reaper.retained_count(), 1);
    assert!(original.snapshots.lock().is_empty());
    let later = Arc::new(FactsSink::new(
        ExecutionFactsCapture::On,
        FactsFailure::None,
    ));
    executor.attach_execution_event_sink(later.clone());
    drop(pending);
    let close = executor.cancel_tokens(initial.sequence()).unwrap();
    let ExecutionCloseDisposition::Held(held) = close.disposition else {
        panic!("blocked worker must retain original close");
    };
    assert!(close.error.is_none());
    gate.open().await;
    let terminal = held.wait().await.unwrap();
    assert_eq!(terminal.ended.sequence(), initial.sequence());
    let failed = capture == ExecutionFactsCapture::On
        && !matches!(
            failure,
            FactsFailure::None | FactsFailure::OptionalTransport
        );
    assert_eq!(terminal.error.is_some(), failed);
    if failed {
        assert!(terminal
            .error
            .unwrap()
            .to_string()
            .contains("execution observation failed"));
    }
    settled(&executor).await;
    assert!(owner.upgrade().is_none());
    assert_eq!(
        original.snapshots.lock().len(),
        usize::from(capture == ExecutionFactsCapture::On)
    );
    assert!(later.snapshots.lock().is_empty());
    assert!(executor.execute_tokens(&inputs).await.is_err());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 1
    );
    token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor).await;
    assert_eq!(later.snapshots.lock().len(), 3);
    assert_eq!(
        original.snapshots.lock().len(),
        usize::from(capture == ExecutionFactsCapture::On)
    );
    settled(&executor).await;
    assert_executor_closed(executor)
}

/// A panic while selecting capture is still a pre-submit observer failure. The
/// prepared wave drops before its original guard transfers retirement custody.
pub async fn execution_facts_capture_query_panic_closes_original_owner<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let sink = Arc::new(FactsSink::new(
        ExecutionFactsCapture::On,
        FactsFailure::CaptureQueryPanic,
    ));
    executor.attach_execution_event_sink(sink.clone());
    let initial = admit_tokens(&executor, &RequestId::new(), &[TokenId::new(1)]).await;
    let owner = executor
        .sequences
        .lock()
        .token_index
        .get(initial.sequence())
        .unwrap()
        .clone();
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [ExecutionWork::new(
        &initial,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        &policy,
    )
    .unwrap()];
    let gate = WorkerGate::closed(&executor.completion_worker).await;
    let ExecutionBatchOutcome::Interrupted(interrupted) =
        executor.execute_tokens(&inputs).await.unwrap()
    else {
        panic!("capture-query panic must return the original participant error");
    };
    assert!(interrupted.completed().is_empty());
    assert_eq!(interrupted.remaining().len(), 1);
    assert_eq!(interrupted.remaining()[0].sequence(), initial.sequence());
    assert!(matches!(
        interrupted.remaining()[0].progress(),
        ExecutionProgress::NotSubmitted {
            stage: NoSubmissionStage::BeforeSubmit
        }
    ));
    assert!(interrupted.causes().iter().any(|cause| cause
        .as_error()
        .is_some_and(|error| error.to_string().contains("capture contract panicked"))));
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before
    );
    assert_eq!(executor.reaper.retained_count(), 0);
    assert_eq!(executor.lane.in_flight_count(), 0);
    assert!(sink.snapshots.lock().is_empty());
    let (_, mut remaining, _) = interrupted.into_parts();
    let (_, _, _, continuation, _) = remaining.pop().unwrap().into_parts();
    gate.open().await;
    match continuation {
        ContinuationDisposition::Ended(ended) => assert_eq!(ended.sequence(), initial.sequence()),
        ContinuationDisposition::Held(held) => assert_eq!(
            held.wait().await.unwrap().ended.sequence(),
            initial.sequence()
        ),
        ContinuationDisposition::Retryable(_) => {
            panic!("a fatal observer contract cannot silently retry")
        }
    }
    settled(&executor).await;
    assert!(owner.upgrade().is_none());
    assert!(executor.execute_tokens(&inputs).await.is_err());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before
    );
    executor.attach_execution_event_sink(Arc::new(FactsSink::new(
        ExecutionFactsCapture::On,
        FactsFailure::None,
    )));
    token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(&executor).await;
    settled(&executor).await;
    assert_executor_closed(executor)
}
