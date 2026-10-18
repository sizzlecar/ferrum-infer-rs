use super::*;

fn scalar_request(offset: u64) -> CompletionReadbackBatchRequest {
    CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new_typed(
        id("node.tail"),
        0,
        id("resource.state"),
        BufferUsage::State,
        offset,
        HostTransferLayout::new(ElementType::U32, 1).unwrap(),
    )
    .unwrap()])
    .unwrap()
}

#[derive(Clone, Copy)]
enum Case {
    Prepared,
    Unsupported,
    PlainWait,
    Indeterminate,
    InvalidRange,
    BadCount,
}

fn exercise(case: Case, timing_mode: DeviceTimingMode) {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_scalar_u32_readback());
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .terminal_readback_enabled = !matches!(case, Case::Unsupported);
    let mut wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let request = scalar_request(if matches!(case, Case::InvalidRange) {
        16
    } else {
        4
    });
    wave.set_terminal_readbacks(request.clone()).unwrap();
    assert!(wave.set_terminal_readbacks(request.clone()).is_err());
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
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
        &lane,
    )
    .unwrap();
    let before_submit = fixture.runtime_trace.lock().unwrap().submit_calls;
    let result = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        timing_mode,
        wave,
        &lane,
        &reaper,
    );
    if matches!(case, Case::InvalidRange) {
        assert!(result.is_err());
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, before_submit);
        assert_eq!(trace.terminal_prepare_calls, 0);
        assert_eq!(trace.terminal_tail_calls, 0);
        drop(trace);
        assert_eq!(reaper.retained_count(), 0);
    } else {
        let handle = result.unwrap();
        let drops = Arc::clone(&fixture.runtime_trace.lock().unwrap().terminal_packet_drops);
        if !matches!(case, Case::Unsupported) {
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            // Equal layout/count is insufficient: the exact declared offset is bound.
            assert!(handle.wait_with_readbacks(scalar_request(8)).is_err());
            assert_eq!(
                fixture.runtime_trace.lock().unwrap().terminal_finish_calls,
                0
            );
            fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Pending;
            assert!(matches!(
                handle.poll().unwrap(),
                CompletionObservation::Pending
            ));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
        }
        match case {
            Case::PlainWait => {
                assert!(matches!(
                    handle.wait().unwrap(),
                    CompletionObservation::Terminal(_)
                ));
                assert_eq!(
                    fixture.runtime_trace.lock().unwrap().terminal_finish_calls,
                    0
                );
                assert_eq!(drops.load(Ordering::SeqCst), 1);
            }
            Case::Indeterminate => {
                fixture.runtime_trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
                assert!(matches!(
                    handle.wait_with_readbacks(request).unwrap(),
                    CompletionReadbackBatchObservation::Indeterminate(_)
                ));
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                assert_eq!(reaper.retained_count(), 1);
                let _ = reaper
                    .recover_slot_by_draining_lane(handle.slot_id())
                    .unwrap();
                assert_eq!(drops.load(Ordering::SeqCst), 1);
                assert_eq!(
                    fixture.runtime_trace.lock().unwrap().terminal_finish_calls,
                    0
                );
            }
            _ => {
                if matches!(case, Case::BadCount) {
                    fixture.runtime_trace.lock().unwrap().terminal_bad_count = true;
                }
                let receipt = match handle.wait_with_readbacks(request).unwrap() {
                    CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
                    other => panic!("expected terminal: {other:?}"),
                };
                if matches!(case, Case::BadCount) {
                    assert!(matches!(
                        receipt.dispositions()[0],
                        CompletionReadbackDisposition::ContractFailedButQuiescent { .. }
                    ));
                    assert_eq!(fixture.runtime_trace.lock().unwrap().readback_calls, 0);
                } else {
                    let CompletionReadbackDisposition::Succeeded(output) =
                        &receipt.dispositions()[0]
                    else {
                        panic!("missing scalar output")
                    };
                    assert_eq!(output.bytes(), &[0; 4]);
                    assert_eq!(output.request().logical_offset_bytes(), 4);
                    match (case, timing_mode) {
                        (_, DeviceTimingMode::Off) => {
                            assert_eq!(output.timing(), DeviceTimingMeasurement::NotRequested)
                        }
                        (Case::Unsupported, _) => assert!(matches!(
                            output.timing(),
                            DeviceTimingMeasurement::Measured(_)
                        )),
                        _ => assert_eq!(
                            output.timing(),
                            DeviceTimingMeasurement::Unavailable(
                                DeviceTimingUnavailableReason::BackendUnsupported
                            )
                        ),
                    }
                }
                let trace = fixture.runtime_trace.lock().unwrap();
                if matches!(case, Case::Unsupported) {
                    assert_eq!(trace.terminal_tail_calls, 0);
                    assert_eq!(trace.terminal_finish_calls, 0);
                    assert_eq!(trace.readback_calls, 1);
                } else {
                    assert_eq!(trace.terminal_tail_calls, 1);
                    assert_eq!(trace.terminal_finish_calls, 1);
                    assert_eq!(trace.terminal_source_offsets.len(), 1);
                    assert_eq!(trace.terminal_source_offsets[0] % 4, 0);
                    assert_eq!(trace.readback_calls, 0);
                    assert_eq!(drops.load(Ordering::SeqCst), 1);
                }
            }
        }
        assert_eq!(reaper.retained_count(), 0);
        assert!(handle.wait().is_err());
        drop(handle);
    }
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn terminal_packet_exact_wave_and_unsupported_readback_preserve_outputs() {
    for mode in [DeviceTimingMode::Off, DeviceTimingMode::Completion] {
        exercise(Case::Prepared, mode);
        exercise(Case::Unsupported, mode);
    }
}

#[test]
fn terminal_packet_plain_wait_and_unknown_completion_do_not_recycle() {
    exercise(Case::PlainWait, DeviceTimingMode::Off);
    exercise(Case::Indeterminate, DeviceTimingMode::Off);
}

#[test]
fn terminal_packet_rejects_invalid_current_range_and_bad_backend_count() {
    exercise(Case::InvalidRange, DeviceTimingMode::Off);
    exercise(Case::BadCount, DeviceTimingMode::Completion);
}

#[test]
fn terminal_packet_definitely_not_submitted_retry_discards_only_attempt_payload() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_scalar_u32_readback());
    let mut wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let request = scalar_request(4);
    wave.set_terminal_readbacks(request.clone()).unwrap();
    let first_attempt = wave.batch_invocation_id();
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
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
        &lane,
    )
    .unwrap();
    let drops = {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.terminal_readback_enabled = true;
        trace.submit_behavior = SubmitBehavior::DefinitelyNotSubmitted;
        Arc::clone(&trace.terminal_packet_drops)
    };
    let retry = match OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Completion,
        wave,
        &lane,
        &reaper,
    ) {
        Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        other => panic!("expected a proven unsubmitted retry: {other:?}"),
    };
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    let retry_wave = retry.retry().unwrap();
    assert_ne!(retry_wave.batch_invocation_id(), first_attempt);
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.submit_behavior = SubmitBehavior::Success;
        // A backend may conservatively decline reuse after discarding the
        // first packet. The fresh attempt must take the ordinary scalar path.
        trace.terminal_readback_enabled = false;
    }
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &retry_wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Completion,
        retry_wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let receipt = match handle.wait_with_readbacks(request).unwrap() {
        CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
        other => panic!("expected successful scalar completion: {other:?}"),
    };
    let CompletionReadbackDisposition::Succeeded(output) = &receipt.dispositions()[0] else {
        panic!("retry did not return current scalar output")
    };
    assert_eq!(output.bytes(), &[0; 4]);
    assert!(matches!(
        output.timing(),
        DeviceTimingMeasurement::Measured(_)
    ));
    let trace = fixture.runtime_trace.lock().unwrap();
    // Both attempts saw the declaration, but only the first had a packet.
    assert_eq!(trace.terminal_prepare_calls, 2);
    assert_eq!(trace.terminal_tail_calls, 1);
    assert_eq!(trace.terminal_finish_calls, 0);
    assert_eq!(trace.readback_calls, 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(trace);
    drop(receipt);
    drop(handle);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
