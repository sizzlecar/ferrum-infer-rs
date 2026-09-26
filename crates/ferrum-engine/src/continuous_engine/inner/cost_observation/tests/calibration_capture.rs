use super::*;

#[test]
fn calibration_actual_unknown_diagnostic_keeps_typed_reason_without_minting_host_stages() {
    for reason in [
        ActualWaveEvidenceUnknown::GraphPath,
        ActualWaveEvidenceUnknown::ProviderPath,
    ] {
        let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
        let sink = sink(4, 16);
        let (mut call, clock) = begin(&shape, &sink);
        let capture = Arc::new(CostCalibrationCapture::default());
        call.attach_calibration_capture(Arc::clone(&capture));
        {
            let mut context = call.context().unwrap();
            context.physical_wave(Err(reason), Some(3));
            clock.set(6);
            context.terminal(ActualWaveOutcome::Completed, None);
            context.finish_call(ObservedCallOutcome::Completed);
        }
        call.record_host_result(committed(&shape.rows[0], 9));
        clock.set(10);
        assert_eq!(
            call.finish(),
            CostCallDisposition::Rejected(CostCallRejection::ActualEvidenceUnknown)
        );
        assert!(capture.host_stages().is_none());
        assert!(sink.pop().is_none());
        let diagnostic = capture.actual_evidence_diagnostic().unwrap();
        assert_eq!(diagnostic.dispatch_unknown, Some(reason));
        assert_eq!(
            (
                diagnostic.physical_waves,
                diagnostic.retained_waves,
                diagnostic.lost_observations
            ),
            (1, 1, 0)
        );
        assert!(diagnostic.retained_wave_details_complete);
        assert_eq!(diagnostic.waves.len(), 1);
        assert_eq!(diagnostic.waves[0].physical_wave_ordinal, 0);
        assert_eq!(diagnostic.waves[0].reason, reason);
        let json = serde_json::to_value(diagnostic.as_ref()).unwrap();
        assert_eq!(json["dispatch_unknown"], format!("{reason:?}"));
        assert_eq!(json["waves"][0]["reason"], format!("{reason:?}"));
        let CostCalibrationStatus::Complete(result) = capture.status() else {
            panic!("finished capture");
        };
        assert!(matches!(
            result.as_ref(),
            CostCalibrationResult::Rejected(CostCallRejection::ActualEvidenceUnknown)
        ));
    }
}

#[test]
fn calibration_actual_unknown_diagnostic_keeps_recorder_loss_bounded_and_first_reason() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(4, 16);
    let (mut call, clock) = begin(&shape, &sink); // original recorder max_waves = 4
    let capture = Arc::new(CostCalibrationCapture::default());
    call.attach_calibration_capture(Arc::clone(&capture));
    {
        let mut context = call.context().unwrap();
        for wave in 0..6_u64 {
            context.physical_wave(
                Err(ActualWaveEvidenceUnknown::GraphPath),
                Some(3 + wave * 3),
            );
            clock.set(4 + wave * 3);
            context.terminal(ActualWaveOutcome::Completed, None);
        }
        context.finish_call(ObservedCallOutcome::Completed);
    }
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
    let diagnostic = capture.actual_evidence_diagnostic().unwrap();
    assert_eq!(
        diagnostic.dispatch_unknown,
        Some(ActualWaveEvidenceUnknown::GraphPath)
    );
    assert_eq!(diagnostic.physical_waves, 6);
    assert_eq!(diagnostic.retained_waves, 4);
    assert_eq!(diagnostic.lost_observations, 2);
    assert_eq!(
        diagnostic
            .waves
            .iter()
            .map(|wave| wave.physical_wave_ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert!(capture.host_stages().is_none());
    assert!(sink.pop().is_none());
}

#[test]
fn calibration_actual_unknown_diagnostic_stays_absent_without_a_typed_unknown() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(4, 16);
    let capture = Arc::new(CostCalibrationCapture::default());
    let (mut call, _) = begin(&shape, &sink);
    call.attach_calibration_capture(Arc::clone(&capture));
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::NoPhysicalWave)
    );
    assert!(capture.actual_evidence_diagnostic().is_none());
    let capture = Arc::new(CostCalibrationCapture::default());
    assert_eq!(
        measured(&shape, &sink, Arc::clone(&capture)).finish(),
        CostCallDisposition::Published
    );
    assert!(capture.actual_evidence_diagnostic().is_none());
}

fn measured(
    shape: &ActualWaveShape,
    sink: &Arc<BoundedCostSampleSink>,
    capture: Arc<CostCalibrationCapture>,
) -> EngineCostCall {
    let (mut call, clock) = begin(shape, sink);
    call.attach_calibration_capture(capture);
    execute(&mut call, &clock, shape.clone());
    for row in &shape.rows {
        call.record_host_result(committed(row, 9));
    }
    clock.set(10);
    call
}

#[test]
fn calibration_capture_preserves_actual_order_commits_and_sample_clock() {
    let shape = shape(&[
        ActualRowWork::Decode { kv_tokens: 32 },
        ActualRowWork::Prefill {
            offset: 2,
            count: 2,
            total_prompt_tokens: 4,
        },
    ]);
    let sink = sink(4, 16);
    let capture = Arc::new(CostCalibrationCapture::default());
    let call = measured(&shape, &sink, Arc::clone(&capture));
    assert!(matches!(capture.status(), CostCalibrationStatus::Pending));
    assert!(matches!(call.finish(), CostCallDisposition::Published));
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        panic!("actual observation");
    };
    let CostCalibrationResult::Observed {
        sample,
        actual_rows,
        commits,
        host_features,
        accepted_ordinal,
        disposition,
    } = result.as_ref()
    else {
        panic!("accepted sample");
    };
    assert!(matches!(disposition, CostCallDisposition::Published));
    assert_eq!(actual_rows, &shape.rows);
    assert_eq!(host_features, &[None, None]);
    assert_eq!(*accepted_ordinal, Some(1));
    for (index, row) in shape.rows.iter().enumerate() {
        assert_eq!(commits[index], committed(row, 9));
    }
    assert_eq!(sample.observed_at_ns, 10);
    assert_eq!(sample.timing.wall_total_ns, 8);
    let queued = sink.pop().unwrap();
    assert_eq!(queued.observed_at_ns, sample.observed_at_ns);
    assert_eq!(queued.timing, sample.timing);
    assert_eq!(queued.actual_shape, sample.actual_shape);
}

#[test]
fn calibration_capture_reports_queue_drop_without_claiming_training() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 16 }]);
    let sink = sink(1, 16);
    assert!(matches!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    ));
    let capture = Arc::new(CostCalibrationCapture::default());
    assert!(matches!(
        measured(&shape, &sink, Arc::clone(&capture)).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    ));
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        panic!("captured disposition");
    };
    assert!(matches!(
        result.as_ref(),
        CostCalibrationResult::Observed {
            disposition: CostCallDisposition::Dropped(CostSampleDrop::Capacity),
            accepted_ordinal: None,
            ..
        }
    ));
    assert_eq!(sink.stats().published, 1);
}

#[test]
fn calibration_capture_abandon_and_cancel_never_create_reference_samples() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 16 }]);
    for cancelled in [false, true] {
        let sink = sink(2, 16);
        let capture = Arc::new(CostCalibrationCapture::default());
        let (mut call, clock) = begin(&shape, &sink);
        call.attach_calibration_capture(Arc::clone(&capture));
        if cancelled {
            execute(&mut call, &clock, shape.clone());
            call.host_cancelled(&shape.rows[0].request_id);
            assert!(matches!(call.finish(), CostCallDisposition::Rejected(_)));
        } else {
            drop(call);
        }
        let CostCalibrationStatus::Complete(result) = capture.status() else {
            panic!("terminal capture");
        };
        assert!(matches!(
            result.as_ref(),
            CostCalibrationResult::Rejected(_)
        ));
        assert!(sink.pop().is_none());
    }
}

#[test]
fn calibration_capture_reuse_is_explicit_conflict_without_overwriting_a_sample() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 16 }]);
    let sink = sink(4, 16);
    let capture = Arc::new(CostCalibrationCapture::default());
    assert!(matches!(
        measured(&shape, &sink, Arc::clone(&capture)).finish(),
        CostCallDisposition::Published
    ));
    let second = measured(&shape, &sink, Arc::clone(&capture));
    // The new call is still live: old completed evidence must already be hidden.
    assert!(matches!(
        capture.status(),
        CostCalibrationStatus::ConflictingCalls
    ));
    assert!(matches!(second.finish(), CostCallDisposition::Published));
    assert_eq!(sink.stats().published, 2);

    let concurrent = Arc::new(CostCalibrationCapture::default());
    let (mut first, _) = begin(&shape, &sink);
    first.attach_calibration_capture(Arc::clone(&concurrent));
    let (mut second, _) = begin(&shape, &sink);
    second.attach_calibration_capture(Arc::clone(&concurrent));
    assert!(matches!(
        concurrent.status(),
        CostCalibrationStatus::ConflictingCalls
    ));
    drop(first);
    drop(second);
    assert!(matches!(
        concurrent.status(),
        CostCalibrationStatus::ConflictingCalls
    ));
}

#[test]
fn calibration_capture_joins_commits_when_physical_rows_reorder_inputs() {
    let prepared = shape(&[
        ActualRowWork::Decode { kv_tokens: 16 },
        ActualRowWork::Decode { kv_tokens: 24 },
    ]);
    let sink = sink(4, 16);
    let capture = Arc::new(CostCalibrationCapture::default());
    let (mut call, clock) = begin(&prepared, &sink);
    let features = [host_features(4), host_features(8)];
    for (participant, features) in call.participants.iter_mut().zip(features) {
        participant.host_features = Some(features);
    }
    call.attach_calibration_capture(Arc::clone(&capture));
    let mut physical = prepared.clone();
    physical.rows.swap(0, 1);
    execute(&mut call, &clock, physical.clone());
    for row in &prepared.rows {
        call.record_host_result(committed(row, 9));
    }
    clock.set(10);
    assert!(matches!(call.finish(), CostCallDisposition::Published));
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        panic!("completed");
    };
    let CostCalibrationResult::Observed {
        actual_rows,
        commits,
        host_features,
        accepted_ordinal,
        ..
    } = result.as_ref()
    else {
        panic!("observed");
    };
    assert_eq!(actual_rows, &physical.rows);
    assert_eq!(host_features, &[Some(features[1]), Some(features[0])]);
    assert_eq!(*accepted_ordinal, Some(1));
    assert_eq!(
        commits
            .iter()
            .map(|row| row.input_index)
            .collect::<Vec<_>>(),
        [1, 0]
    );
    for (row, commit) in actual_rows.iter().zip(commits) {
        assert_eq!(commit, &committed(row, 9));
    }
}

fn host_features(maximum: u64) -> HostCostFeaturesV1 {
    HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: None,
            categorical_signature: [19; 32],
            decoder_text_bytes_per_token: 12,
            decoder_scratch_bytes_per_token: 4,
            raw_token_bytes_bound: 4,
        },
        state: HostCostStateV1 {
            generated_tokens_before: 1,
            maximum_output_tokens: maximum,
            sampling_history_tokens: 1,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    }
}

#[test]
fn calibration_capture_accepted_ordinal_is_sink_order_not_call_identity() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 16 }]);
    let sink = sink(1, 16);
    let first = Arc::new(CostCalibrationCapture::default());
    measured(&shape, &sink, Arc::clone(&first)).finish();
    let dropped = Arc::new(CostCalibrationCapture::default());
    measured(&shape, &sink, Arc::clone(&dropped)).finish();
    assert!(matches!(
        dropped.status(),
        CostCalibrationStatus::Complete(value)
            if matches!(value.as_ref(), CostCalibrationResult::Observed {accepted_ordinal: None, ..})
    ));
    assert_eq!(sink.pop_numbered().unwrap().0, 1);
    let third = Arc::new(CostCalibrationCapture::default());
    measured(&shape, &sink, Arc::clone(&third)).finish();
    let CostCalibrationStatus::Complete(value) = third.status() else {
        panic!("third call finished");
    };
    assert!(matches!(
        value.as_ref(),
        CostCalibrationResult::Observed {
            accepted_ordinal: Some(2),
            disposition: CostCallDisposition::Published,
            ..
        }
    ));
    assert_eq!(sink.pop_numbered().unwrap().0, 2);
}
