//! Original recorder/settlement binding tests. The separate source8 engine
//! regression supplies genuine installed prefixes end to end.
use super::*;
use crate::continuous_engine::inner::cost_observation::resolved::ResolvedCostEntry;

fn labeled(stages: Arc<HostStageEvidenceV1>) -> ResolvedCostEntry {
    ResolvedCostEntry::new(CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection: CostCallRejection::CalibrationPreparation,
    })
}

#[test]
fn preparation_feedback_requires_original_call_and_unchanged_complete_settlement() {
    // The real prefix route is outside ordinary numerical eligibility. Use an
    // actual physical recorder without any statistical or structured recipe.
    let actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
    let (mut call, clock) = begin(&actual, &sink(8, 256));
    call.attach_calibration_capture(Arc::new(
        CostCalibrationCapture::default()
            .with_original_route_capture()
            .unwrap(),
    ));
    call = call.with_structured_capture(true);
    execute(&mut call, &clock, actual.clone());
    settle(&mut call, &clock, &actual.rows[0], 10, None);
    clock.set(20);
    let (ordinary, proof) = call.make_host_stages_with_preparation().unwrap();
    assert!(proof.is_none());
    assert!(
        labeled(ordinary)
            .with_original_preparation(proof.as_ref())
            .preparation_feedback()
            .is_none(),
        "a reason label on an ordinary call cannot establish preparation"
    );

    // This private marker comes from EngineCostPreparation's actual prefix
    // intervention. It alone is insufficient without the original settlement.
    call.rejection = Some(CostCallRejection::CalibrationPreparation);
    let (stages, proof) = call.make_host_stages_with_preparation().unwrap();
    assert!(stages.statistical_evidence.is_none());
    assert!(!matches!(stages.structured_evidence.as_ref(), Some(Ok(_))));
    assert!(
        proof.is_some(),
        "numerical eligibility is not physical settlement"
    );
    let original = labeled(Arc::clone(&stages)).with_original_preparation(proof.as_ref());
    assert_eq!(
        original.preparation_feedback().map(|(_, at)| at),
        stages.finalized_at_ns
    );
    assert!(
        original.structured().is_err(),
        "preparation remains ineligible for training"
    );
    assert!(
        ResolvedCostEntry::new(original.into_entry())
            .preparation_feedback()
            .is_none(),
        "a replayed public entry cannot recreate live exclusion authority"
    );

    let mut changed = stages.as_ref().clone();
    changed.full_wall_ns = Some(stages.full_wall_ns.unwrap() + 1);
    assert!(labeled(Arc::new(changed))
        .with_original_preparation(proof.as_ref())
        .preparation_feedback()
        .is_none());
    let copied = Arc::new(stages.as_ref().clone());
    assert!(
        labeled(copied)
            .with_original_preparation(proof.as_ref())
            .preparation_feedback()
            .is_none(),
        "even an unchanged DTO is not the original receipt"
    );
    let mut incomplete = stages.as_ref().clone();
    incomplete.completeness = HostStageCompleteness::MissingEvidence;
    assert!(labeled(Arc::new(incomplete))
        .with_original_preparation(proof.as_ref())
        .preparation_feedback()
        .is_none());

    call.stage_rejection = Some(CostCallRejection::HostFailed);
    assert!(call
        .make_host_stages_with_preparation()
        .unwrap()
        .1
        .is_none());
    call.stage_rejection = None;
    call.dispatch.outcome = Some(ObservedCallOutcome::Failed);
    assert!(call
        .make_host_stages_with_preparation()
        .unwrap()
        .1
        .is_none());
    call.dispatch.outcome = Some(ObservedCallOutcome::Completed);
    let capture = call.calibration_capture.take();
    assert!(call
        .make_host_stages_with_preparation()
        .unwrap()
        .1
        .is_none());
    call.calibration_capture = capture;
    let (mut pending, pending_clock) = begin(&actual, &sink(8, 256));
    pending.attach_calibration_capture(Arc::new(
        CostCalibrationCapture::default()
            .with_original_route_capture()
            .unwrap(),
    ));
    pending.rejection = Some(CostCallRejection::CalibrationPreparation);
    execute(&mut pending, &pending_clock, actual.clone());
    pending_clock.set(10);
    pending.begin_host_row(&actual.rows[0].request_id);
    pending_clock.set(11);
    pending.note_host_token_commit(&committed(&actual.rows[0], 11));
    pending_clock.set(20);
    assert!(
        pending
            .make_host_stages_with_preparation()
            .unwrap()
            .1
            .is_none(),
        "missing output acknowledgement cannot be excluded"
    );
}

#[test]
fn startup_readiness_feedback_requires_original_completed_training_and_settled_terminal() {
    use ferrum_scheduler::implementations::continuous::cost_model::WaveObservationOutcome;
    for ends_request in [false, true] {
        let actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
        let (mut call, clock) = begin(&actual, &sink(8, 256));
        call.attach_calibration_capture(Arc::new(CostCalibrationCapture::for_startup_readiness()));
        execute(&mut call, &clock, actual.clone());
        settle(
            &mut call,
            &clock,
            &actual.rows[0],
            10,
            ends_request.then(terminal),
        );
        // This unit fixture exercises the private proof boundary. The separate
        // startup-series inventory test runs the real publication method that
        // supplies Composite on terminal handoff, then checks catalog retention.
        if ends_request {
            call.reject(CostCallRejection::Composite);
        }
        clock.set(20);
        let (stages, proof) = call.make_host_stages_with_preparation().unwrap();
        let sample = (!ends_request).then(|| call.make_sample().unwrap());
        assert_eq!(stages.rows[0].terminal.is_some(), ends_request);
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert!(
            proof.is_some(),
            "both ordinary and truly settled terminal readiness are complete"
        );
        let entry = |stages| {
            ResolvedCostEntry::new(match &sample {
                Some(sample) => CostEvidenceEntry::Training {
                    sample: sample.clone(),
                    stages: Some(stages),
                },
                None => CostEvidenceEntry::StagesOnly {
                    stages,
                    legacy_rejection: CostCallRejection::Composite,
                },
            })
        };
        let original = entry(Arc::clone(&stages)).with_original_preparation(proof.as_ref());
        assert_eq!(
            original.preparation_feedback().map(|(_, at)| at),
            stages.finalized_at_ns
        );
        assert!(
            ResolvedCostEntry::new(original.into_entry())
                .preparation_feedback()
                .is_none(),
            "a public entry cannot recreate the private proof"
        );
        assert!(
            entry(Arc::new(stages.as_ref().clone()))
                .with_original_preparation(proof.as_ref())
                .preparation_feedback()
                .is_none(),
            "even identical copied stages are not the original settlement"
        );
        assert!(
            labeled(Arc::clone(&stages))
                .with_original_preparation(proof.as_ref())
                .preparation_feedback()
                .is_none(),
            "readiness cannot be relabeled as prefix intervention"
        );
        for outcome in [
            WaveObservationOutcome::FailedAfterSubmit,
            WaveObservationOutcome::PartiallyCompleted {
                timing_covers_actual_shape_only: false,
            },
        ] {
            let Some(mut failed) = sample.clone() else {
                continue;
            };
            failed.outcome = outcome;
            assert!(ResolvedCostEntry::new(CostEvidenceEntry::Training {
                sample: failed,
                stages: Some(Arc::clone(&stages))
            })
            .with_original_preparation(proof.as_ref())
            .preparation_feedback()
            .is_none());
        }
        call.calibration_capture = None;
        assert!(
            call.make_host_stages_with_preparation()
                .unwrap()
                .1
                .is_none(),
            "ordinary serving with identical successful work is not readiness"
        );
    }
}

#[test]
fn startup_readiness_never_mints_for_failed_partial_unknown_or_unsettled_work() {
    for failure in 0..5 {
        let actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
        let (mut call, clock) = begin(&actual, &sink(8, 256));
        call.attach_calibration_capture(Arc::new(CostCalibrationCapture::for_startup_readiness()));
        execute(&mut call, &clock, actual.clone());
        let mut completed = terminal();
        if failure == 0 {
            completed.output_failed = true;
        } else if failure == 1 {
            completed.cache_completion_work = ExecutorCompletionWork::Unknown;
        }
        if failure == 4 {
            clock.set(10);
            call.begin_host_row(&actual.rows[0].request_id);
            clock.set(11);
            let evidence = committed(&actual.rows[0], 11);
            call.note_host_token_commit(&evidence);
            let _unsettled = call.host_publication(&evidence, true, true).unwrap();
        } else {
            settle(&mut call, &clock, &actual.rows[0], 10, Some(completed));
        }
        if failure == 2 {
            call.dispatch.outcome = Some(ObservedCallOutcome::Failed);
        } else if failure == 3 {
            call.dispatch.unknown = Some(ActualWaveEvidenceUnknown::InvalidLifecycle);
        }
        clock.set(20);
        assert!(
            call.make_host_stages_with_preparation()
                .unwrap()
                .1
                .is_none(),
            "failure case {failure} cannot be excluded from feedback"
        );
    }
    // The old prefix exclusion still rejects a complete ordinary terminal.
    let actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
    let (mut call, clock) = begin(&actual, &sink(8, 256));
    call.attach_calibration_capture(Arc::new(
        CostCalibrationCapture::default()
            .with_original_route_capture()
            .unwrap(),
    ));
    execute(&mut call, &clock, actual.clone());
    settle(&mut call, &clock, &actual.rows[0], 10, Some(terminal()));
    call.rejection = Some(CostCallRejection::CalibrationPreparation);
    clock.set(20);
    assert!(call
        .make_host_stages_with_preparation()
        .unwrap()
        .1
        .is_none());
}
