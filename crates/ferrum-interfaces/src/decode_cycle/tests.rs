use super::*;

fn recorder(capacity: usize) -> (DecodeCycleTrace, DecodeCycleDrainOwner) {
    DecodeCycleTrace::new(NonZeroUsize::new(capacity).expect("nonzero test capacity"))
        .expect("diagnostic clock and bounded allocation")
}

type OwnerFrame = (
    SequenceAuthorityId,
    RequestAuthorityId,
    ExecutionFrameId,
    RequestIdentity,
);

fn admitted_identity(request: &RequestId) -> RequestIdentity {
    RequestIdentity::new(format!("request.unit.{request}")).expect("typed admitted identity")
}

fn mapped_caller(
    request: RequestId,
    ordinal: u32,
    context: Option<DecodeCycleCallContext>,
) -> DecodeCycleCaller {
    let identity = admitted_identity(&request);
    DecodeCycleCaller::new(request, ordinal, context).with_request_identity(identity)
}

fn frame(sequence: u32, request: &RequestId) -> OwnerFrame {
    (
        SequenceAuthorityId::test_only(sequence, 1),
        RequestAuthorityId::test_only(sequence, 1),
        ExecutionFrameId::try_from(1).expect("nonzero frame"),
        admitted_identity(request),
    )
}

fn wave(clock_id: DecodeCycleClockId) -> DecodeCycleWaveWitness {
    DecodeCycleWaveWitness {
        clock_id,
        kind: DecodeCycleWaveKind::Decode,
        batch_step_id: BatchStepId::try_from(1).expect("nonzero step"),
        batch_invocation_id: BatchInvocationId::try_from(1).expect("nonzero invocation"),
        lane_id: ExecutionLaneId::mint().expect("lane identity"),
        device_id: DeviceId::new("device.unit").expect("device identity"),
        runtime_implementation_fingerprint: "runtime.unit".to_owned(),
        prepared_wave_fingerprint: "prepared-wave".to_owned(),
        owners: Vec::new(),
    }
}

#[test]
fn saturation_keeps_accepted_events_and_reports_incomplete_coverage() {
    let (trace, drain) = recorder(1);
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CallStarted,
            DecodeCycleScope::Iteration(7)
        ),
        DecodeCycleRecordStatus::Accepted
    );
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CallReturned,
            DecodeCycleScope::Iteration(7)
        ),
        DecodeCycleRecordStatus::Full
    );
    let capture = drain.finish();
    assert_eq!(capture.events.len(), 1);
    assert_eq!(capture.events[0].stage, DecodeCycleStage::CallStarted);
    assert_eq!(capture.stats.attempted, 2);
    assert_eq!(capture.stats.accepted, 1);
    assert_eq!(capture.stats.full, 1);
    assert!(!capture.has_complete_coverage());
}

#[test]
fn contended_recorder_does_not_wait_and_retains_later_records() {
    let (trace, drain) = recorder(2);
    let held = trace.inner.events.lock().expect("hold recorder lock");
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CallStarted,
            DecodeCycleScope::Iteration(1)
        ),
        DecodeCycleRecordStatus::Busy
    );
    drop(held);
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CallReturned,
            DecodeCycleScope::Iteration(1)
        ),
        DecodeCycleRecordStatus::Accepted
    );
    let capture = drain.finish();
    assert_eq!(capture.stats.busy, 1);
    assert_eq!(capture.events.len(), 1);
    assert!(!capture.has_complete_coverage());
}

#[test]
fn sealed_recorder_rejects_later_events_without_reopening_storage() {
    let (trace, drain) = recorder(1);
    let capture = drain.finish();
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CallReturned,
            DecodeCycleScope::Iteration(1)
        ),
        DecodeCycleRecordStatus::Closed
    );
    assert_eq!(trace.stats().closed, 1);
    assert!(capture.events.is_empty());
    assert!(trace
        .inner
        .events
        .lock()
        .expect("sealed storage")
        .is_empty());
}

#[test]
fn live_worker_producer_makes_early_export_incomplete() {
    let (trace, drain) = recorder(2);
    let worker = trace.producer_guard();
    trace.mark(
        DecodeCycleStage::CompletionWorkerStarted,
        DecodeCycleScope::Iteration(1),
    );
    let capture = drain.finish();
    assert_eq!(capture.stats.outstanding_producers, 1);
    assert_eq!(capture.active_producers_at_close, 1);
    assert!(!capture.has_complete_coverage());
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CompletionResultReady,
            DecodeCycleScope::Iteration(1)
        ),
        DecodeCycleRecordStatus::Closed
    );
    drop(worker);
    assert_eq!(trace.stats().outstanding_producers, 0);
    assert_eq!(trace.stats().closed, 1);
}

#[test]
fn completed_producer_and_shared_timestamps_survive_owned_handoff() {
    let (trace, drain) = recorder(4);
    let call = trace.call_context(2, 1, 0);
    let worker_trace = trace.clone();
    let producer = trace.producer_guard();
    let ready = std::thread::spawn(move || {
        let ready = worker_trace.timestamp();
        assert_eq!(
            worker_trace.mark_at(
                ready,
                DecodeCycleStage::CompletionResultReady,
                DecodeCycleScope::Call(call)
            ),
            DecodeCycleRecordStatus::Accepted
        );
        drop(producer);
        ready
    })
    .join()
    .expect("worker completed");
    let resumed = trace.timestamp();
    assert_eq!(ready.clock_id(), resumed.clock_id());
    assert!(ready.nanos() <= resumed.nanos());
    trace.mark_at(
        resumed,
        DecodeCycleStage::CompletionAwaitResumed,
        DecodeCycleScope::Call(call),
    );
    let capture = drain.finish();
    assert!(capture.has_complete_coverage());
    assert_eq!(capture.events[0].at, ready);
    assert_eq!(capture.events[1].at, resumed);
}

#[test]
fn recorder_poison_is_distinct_from_contention_and_is_not_rethrown() {
    let (trace, drain) = recorder(1);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = trace.inner.events.lock().expect("hold recorder lock");
        panic!("injected diagnostic recorder panic");
    }));
    assert_eq!(
        trace.mark(
            DecodeCycleStage::FailureObserved,
            DecodeCycleScope::Iteration(1)
        ),
        DecodeCycleRecordStatus::Poisoned
    );
    let capture = drain.finish();
    assert_eq!(capture.stats.busy, 0);
    assert!(capture.stats.poisoned > 0);
    assert!(!capture.has_complete_coverage());
}

#[test]
fn foreign_clock_timestamps_and_call_scopes_are_rejected() {
    let (trace, drain) = recorder(2);
    let (foreign, foreign_drain) = recorder(2);
    assert_eq!(
        trace.mark_at(
            foreign.timestamp(),
            DecodeCycleStage::CallStarted,
            DecodeCycleScope::Iteration(1)
        ),
        DecodeCycleRecordStatus::InvalidTimestamp
    );
    assert_eq!(
        trace.mark(
            DecodeCycleStage::CallStarted,
            DecodeCycleScope::Call(foreign.call_context(1, 0, 0))
        ),
        DecodeCycleRecordStatus::InvalidScope
    );
    let capture = drain.finish();
    assert!(capture.events.is_empty());
    assert_eq!(capture.stats.invalid_metadata, 2);
    assert!(!capture.has_complete_coverage());
    drop(foreign_drain);
}

#[test]
fn caller_mapping_preserves_original_ordinals_for_a_split_wave() {
    let (trace, drain) = recorder(2);
    let first = RequestId::new();
    let second = RequestId::new();
    let frames = [frame(1, &first), frame(2, &second)];
    let owners = validate_callers(
        trace.clock().id,
        frames.iter().cloned(),
        vec![
            mapped_caller(first.clone(), 7, Some(trace.call_context(3, 1, 7))),
            mapped_caller(second.clone(), 2, Some(trace.call_context(3, 1, 2))),
        ],
    )
    .expect("split wave can retain gaps and original order");
    assert_eq!(owners[0].canonical_ordinal, 0);
    assert_eq!(owners[0].caller_ordinal, 7);
    assert_eq!(owners[0].request_id, first);
    assert_eq!(owners[1].canonical_ordinal, 1);
    assert_eq!(owners[1].caller_ordinal, 2);
    assert_eq!(owners[1].request_id, second);
    assert_eq!(owners[0].sequence_authority, frames[0].0);
    assert_eq!(owners[0].request_identity, frames[0].3);
    drop(drain);
}

#[test]
fn mixed_prefill_and_decode_caller_domains_are_not_reindexed() {
    let (trace, drain) = recorder(2);
    let prefill = RequestId::new();
    let decode = RequestId::new();
    let frames = [frame(1, &prefill), frame(2, &decode)];
    let owners = validate_callers(
        trace.clock().id,
        frames.iter().cloned(),
        vec![
            mapped_caller(prefill, 0, None),
            mapped_caller(decode, 0, Some(trace.call_context(3, 1, 0))),
        ],
    )
    .expect("original ordinals belong to separate API caller domains");
    assert_eq!(owners[0].caller_ordinal, 0);
    assert_eq!(owners[1].caller_ordinal, 0);
    assert!(owners[0].context.is_none());
    assert!(owners[1].context.is_some());
    drop(drain);
}

#[test]
fn missing_duplicate_swapped_and_foreign_callers_are_rejected() {
    let (trace, drain) = recorder(2);
    let (foreign, foreign_drain) = recorder(2);
    let first = RequestId::new();
    let second = RequestId::new();
    let frames = [frame(1, &first), frame(2, &second)];
    let validate = |callers| validate_callers(trace.clock().id, frames.iter().cloned(), callers);
    assert!(matches!(
        validate(vec![mapped_caller(first.clone(), 0, None)]),
        Err(DecodeCycleTraceError::InvalidCallerMap)
    ));
    assert!(matches!(
        validate(vec![
            mapped_caller(first.clone(), 0, None),
            mapped_caller(second.clone(), 0, None)
        ]),
        Err(DecodeCycleTraceError::InvalidCallerMap)
    ));
    assert!(matches!(
        validate(vec![
            mapped_caller(first.clone(), 0, None),
            mapped_caller(first.clone(), 1, None)
        ]),
        Err(DecodeCycleTraceError::InvalidCallerMap)
    ));
    assert!(matches!(
        validate(vec![
            mapped_caller(second.clone(), 0, None),
            mapped_caller(first.clone(), 1, None)
        ]),
        Err(DecodeCycleTraceError::InvalidCallerMap)
    ));
    assert!(matches!(
        validate(vec![
            DecodeCycleCaller::new(first.clone(), 0, None),
            mapped_caller(second.clone(), 1, None)
        ]),
        Err(DecodeCycleTraceError::InvalidCallerMap)
    ));
    assert!(matches!(
        validate(vec![
            mapped_caller(first.clone(), 0, Some(foreign.call_context(1, 0, 0))),
            mapped_caller(second.clone(), 1, None)
        ]),
        Err(DecodeCycleTraceError::ForeignClock)
    ));
    assert!(matches!(
        validate(vec![
            mapped_caller(first, 0, Some(trace.call_context(1, 0, 1))),
            mapped_caller(second, 1, None)
        ]),
        Err(DecodeCycleTraceError::InvalidCallerMap)
    ));
    drop(drain);
    drop(foreign_drain);
}

#[test]
fn foreign_submission_metadata_cannot_bind_another_prepared_wave() {
    let (trace, drain) = recorder(2);
    let wave = wave(trace.clock().id);
    let check = |step, invocation, lane, fingerprint| {
        validate_submission_metadata(&wave, step, invocation, lane, fingerprint)
    };
    assert!(check(
        wave.batch_step_id,
        wave.batch_invocation_id,
        wave.lane_id,
        &wave.prepared_wave_fingerprint
    )
    .is_ok());
    assert!(matches!(
        check(
            BatchStepId::try_from(2).expect("nonzero"),
            wave.batch_invocation_id,
            wave.lane_id,
            &wave.prepared_wave_fingerprint
        ),
        Err(DecodeCycleTraceError::ForeignSubmission)
    ));
    assert!(matches!(
        check(
            wave.batch_step_id,
            BatchInvocationId::try_from(2).expect("nonzero"),
            wave.lane_id,
            &wave.prepared_wave_fingerprint
        ),
        Err(DecodeCycleTraceError::ForeignSubmission)
    ));
    assert!(matches!(
        check(
            wave.batch_step_id,
            wave.batch_invocation_id,
            ExecutionLaneId::mint().expect("other lane"),
            &wave.prepared_wave_fingerprint
        ),
        Err(DecodeCycleTraceError::ForeignSubmission)
    ));
    assert!(matches!(
        check(
            wave.batch_step_id,
            wave.batch_invocation_id,
            wave.lane_id,
            "other-prepared-wave"
        ),
        Err(DecodeCycleTraceError::ForeignSubmission)
    ));
    drop(drain);
}

#[test]
fn graph_selection_is_shared_per_attempt_and_fresh_for_next_attempt() {
    let (trace, drain) = recorder(2);
    let witness = Arc::new(wave(trace.clock().id));
    let context =
        DecodeCycleWaveContext::new(trace.clone(), Arc::clone(&witness)).expect("same clock");
    let clone = context.clone();
    assert!(context.claim_first_graph_launch());
    assert!(!clone.claim_first_graph_launch());
    let next_attempt =
        DecodeCycleWaveContext::new(trace.clone(), witness).expect("fresh diagnostic context");
    assert!(next_attempt.claim_first_graph_launch());
    drop(drain);
}
