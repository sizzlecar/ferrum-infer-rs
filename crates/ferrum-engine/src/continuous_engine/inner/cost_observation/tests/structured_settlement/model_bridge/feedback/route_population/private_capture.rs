//! Same original selector and controlled submission, with an explicitly fixed
//! private capture. No live population ticket supplies this route permission.
use super::*;

#[tokio::test]
async fn private_route_diagnostics_preserve_settlement_fifo_and_authority() {
    use ferrum_interfaces::execution_cost::PreparedCostRouteClassV1;

    for (authorized, outside, extra_unknown, missing_host_features) in [
        (true, false, false, false),
        (true, true, false, false),
        (true, true, true, false),
        (true, true, false, true),
        (false, true, false, false),
    ] {
        let mut baseline = None;
        for debug in [false, true] {
            let (f, outcome) =
                tracing::subscriber::with_default(diagnostics::RuntimeDebug(debug), || {
                    let f = Automatic::new(8);
                    let capture = if authorized {
                        CostCalibrationCapture::default()
                            .with_original_route_capture()
                            .unwrap()
                    } else {
                        CostCalibrationCapture::default()
                    };
                    let capture = Arc::new(capture);
                    let before = f.runtime.sink.stats().raw_accepted;
                    let stages = fixture::record_private_route_with_hook(
                        &f.runtime,
                        &f.clock,
                        wave("private.route-diagnostic"),
                        outside,
                        extra_unknown,
                        capture.clone(),
                        |call| {
                            if missing_host_features {
                                call.participants[0].host_features = None;
                            }
                            assert!(call.live_ticket.is_none());
                            let diagnostic = call.recorder.route_diagnostic();
                            assert_eq!(diagnostic.is_some(), authorized && debug);
                            if let Some(diagnostic) = diagnostic {
                                assert_eq!(diagnostic.preparation_attempts, 1);
                                let prepared = diagnostic.prepared.unwrap();
                                if outside {
                                    assert!(prepared.class.is_outside());
                                } else {
                                    assert_eq!(
                                        prepared.class,
                                        PreparedCostRouteClassV1::GraphDisabled
                                    );
                                }
                                let submitted = diagnostic.submitted.unwrap();
                                assert_eq!(submitted.lane_id, prepared.lane_id);
                                assert!(submitted.graph.is_some());
                                assert!(diagnostic.first_rejection.is_none());
                                assert_eq!(diagnostic.other_evidence_unknown, extra_unknown);
                            }
                            let mut gate = None;
                            assert_eq!(
                                call.make_route_evidence_recording_failure(&mut gate)
                                    .is_some(),
                                authorized && !extra_unknown && !missing_host_features
                            );
                            assert_eq!(
                                gate,
                                if debug && authorized && extra_unknown {
                                    Some("private_prepared_route_missing")
                                } else if debug && authorized && missing_host_features {
                                    Some("outside_host_features_missing")
                                } else {
                                    None
                                }
                            );
                        },
                    );
                    let accepted = authorized && !extra_unknown && !missing_host_features;
                    assert_eq!(stages.is_some(), accepted);
                    f.runtime.consume_samples();
                    assert_eq!(capture.host_stages().is_some(), accepted);
                    assert!(!matches!(capture.status(), CostCalibrationStatus::Pending));
                    let queue = capture.host_stage_queue();
                    if accepted {
                        let queue = queue.unwrap();
                        assert_eq!(queue.disposition, HostStageQueueDisposition::Published);
                        assert_eq!(queue.accepted_ordinal, Some(before + 1));
                    }
                    let audit = f.live().audit();
                    assert_eq!(audit.population.issued, 0);
                    assert_eq!(audit.qualified_publications, 0);
                    let outcome = (
                        stages.is_some(),
                        queue.map(|q| (q.disposition, q.accepted_ordinal.map(|n| n - before))),
                        f.runtime.sink.stats().raw_accepted - before,
                    );
                    (f, outcome)
                });
            if let Some(expected) = &baseline {
                assert_eq!(
                    &outcome, expected,
                    "diagnostics cannot alter settlement or FIFO"
                );
            } else {
                baseline = Some(outcome);
            }
            f.runtime.shutdown().await.unwrap();
        }
    }
}

#[tokio::test]
async fn private_route_capture_retains_original_warm_and_outside_receipts_without_live_ticket() {
    for outside in [false, true] {
        let f = Automatic::new(8);
        let capture = Arc::new(
            CostCalibrationCapture::default()
                .with_original_route_capture()
                .unwrap(),
        );
        let stages = fixture::record_private_route(
            &f.runtime,
            &f.clock,
            wave("private.route"),
            outside,
            false,
            capture.clone(),
        )
        .unwrap();
        assert_eq!(
            stages.route_evidence.as_ref().unwrap().is_outside(),
            outside
        );
        f.runtime.consume_samples();
        let resolved = capture.host_stages().expect("original worker settlement");
        assert_eq!(resolved.call_id, stages.call_id);
        assert_eq!(
            resolved.route_evidence.as_ref().unwrap().is_outside(),
            outside
        );
        let queue = capture.host_stage_queue().unwrap();
        assert_eq!(queue.disposition, HostStageQueueDisposition::Published);
        assert!(queue.accepted_ordinal.is_some());
        if outside {
            let receipt = resolved
                .route_evidence
                .as_ref()
                .unwrap()
                .outside_record(
                    &resolved,
                    1,
                    StructuredPhaseV2::Fit,
                    queue.accepted_ordinal.unwrap(),
                )
                .unwrap();
            assert_eq!(receipt.fifo, queue.accepted_ordinal.unwrap());
        }
        assert_eq!(
            f.runtime
                .audit_snapshot()
                .live_calibration
                .unwrap()
                .population
                .issued,
            0
        );
        // Ownership after the original call cannot retroactively enable a new
        // capture policy, even when all other Arc users have completed.
        let completed = Arc::try_unwrap(capture).expect("worker released capture");
        assert!(completed.with_original_route_capture().is_err());
        f.runtime.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn private_route_capture_requires_prior_permission_and_keeps_unknown_failure() {
    for authorized in [false, true] {
        let f = Automatic::new(8);
        let capture = if authorized {
            CostCalibrationCapture::default()
                .with_original_route_capture()
                .unwrap()
        } else {
            CostCalibrationCapture::default()
        };
        let capture = Arc::new(capture);
        // Unrequested capture is insufficient; a requested capture with real
        // additional InvalidLifecycle remains ineligible despite GraphPath.
        let stages = fixture::record_private_route(
            &f.runtime,
            &f.clock,
            wave("private.route.rejected"),
            true,
            authorized,
            capture.clone(),
        );
        assert!(stages.is_none());
        f.runtime.consume_samples();
        assert!(capture.host_stages().is_none());
        assert!(!matches!(capture.status(), CostCalibrationStatus::Pending));
        assert_eq!(
            f.runtime
                .audit_snapshot()
                .live_calibration
                .unwrap()
                .population
                .issued,
            0
        );
        f.runtime.shutdown().await.unwrap();
    }
    let f = Automatic::with_population(8, SloCalibrationRoutePopulationV1::AllAttempts);
    let stages = fixture::record_route_with_hook(
        &f.runtime,
        &f.clock,
        wave("legacy.all-attempts"),
        false,
        false,
        false,
        |_| {},
    )
    .unwrap();
    assert!(stages.route_evidence.is_none());
    f.runtime.consume_samples();
    f.runtime.shutdown().await.unwrap();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixRouteFault {
    None,
    DetachedPrefix,
    UnauthorizedCapture,
    WrongWork,
    WrongHistory,
    MixedDetachedPrefix,
}

// A real installed prefix supplies policy/owner facts and the engine producer
// supplies CalibrationPreparation. Core submission is controlled CPU work;
// the nonzero-offset case isolates suffix settlement, not native-copy execution.
async fn check_installed_prefix_outside_settlement(offset: u32, fault: PrefixRouteFault) {
    use crate::continuous_engine::inner::{
        calibration::{CalibrationLimits, CalibrationPrefixTokensV1, CalibrationSession},
        cost_observation::resolved::ResolvedCostEntry,
        slo_controller::tests::fixture::fixture_with_custom_config,
    };
    use ferrum_interfaces::{output_flow::OutputProjectionContract, slo::InferenceRequestContext};
    use ferrum_types::{
        SamplingParams, SloLiveStructuredCalibration, SloStructuredCostCapture, TokenId,
    };

    let width = if fault == PrefixRouteFault::MixedDetachedPrefix {
        2
    } else {
        1
    };
    let (mut engine, _, _) = fixture_with_custom_config(width, |config| {
        let cost = &mut config.scheduler.slo.cost_observation;
        cost.predictor = ferrum_types::SloCostPredictor::StructuredWholeWaveV2;
        cost.structured_capture = SloStructuredCostCapture::HostSettledV1;
        cost.live_structured_calibration = SloLiveStructuredCalibration::Disabled;
    })
    .await;
    let clock = Arc::new(VirtualClock(AtomicU64::new(100)));
    {
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
        inner.cost_runtime = Some(Arc::new(
            EngineCostRuntime::build(
                identity,
                clock.clone(),
                &inner.config.scheduler.slo.cost_observation,
                true,
            )
            .unwrap(),
        ));
        inner.bg_loop_spawned.store(false, Ordering::Release);
    }
    let mut session = CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::new(width).unwrap()).unwrap(),
    )
    .unwrap();
    let inner = session.test_engine_inner();
    let runtime = inner.cost_runtime.as_ref().unwrap().clone();
    let mut ids = Vec::new();
    let mut outputs = Vec::new();
    for _ in 0..width {
        let mut request = ferrum_types::InferenceRequest::new(
            "test test test",
            session.configuration().model.model_id.clone(),
        );
        request.stream = true;
        request.sampling_params = SamplingParams {
            max_tokens: 4,
            ..SamplingParams::greedy()
        };
        request
            .metadata
            .insert("ferrum_ignore_eos".into(), true.into());
        ids.push(request.id.clone());
        outputs.push(
            session
                .add_request_with_prefix_preparation(
                    request,
                    InferenceRequestContext::from_ingress(std::time::Instant::now()),
                    Arc::new(OutputProjectionContract::cli_text()),
                    CalibrationPrefixTokensV1 {
                        tokenizer_policy_sha256: inner
                            .tokenizer
                            .host_output_policy_identity()
                            .unwrap(),
                        token_ids: vec![TokenId::new(6)],
                        release_generated: 1,
                    },
                )
                .await
                .unwrap(),
        );
    }
    {
        let mut sequences = inner.sequences.write();
        for id in &ids {
            let sequence = &sequences[id];
            assert!(sequence.calibration_prefix.is_some());
            assert!(sequence
                .cost_numeric_policy
                .unwrap()
                .empirical_content_domain
                .is_none());
            assert!(sequence.generated_tokens.is_empty());
        }
        if matches!(
            fault,
            PrefixRouteFault::DetachedPrefix | PrefixRouteFault::MixedDetachedPrefix
        ) {
            // A missing row capability cannot borrow another participant's
            // CalibrationPreparation marker. Retain the now-unsupported policy.
            sequences
                .get_mut(ids.last().unwrap())
                .unwrap()
                .calibration_prefix
                .take();
        }
    }
    let mut preparation = inner.prepare_cost_observation().unwrap();
    {
        let sequences = inner.sequences.read();
        for id in &ids {
            preparation.capture(&sequences[id]);
        }
    }
    let mut call = preparation.begin().unwrap();
    assert_eq!(
        call.rejection == Some(CostCallRejection::CalibrationPreparation),
        fault != PrefixRouteFault::DetachedPrefix
    );
    assert!(call.live_ticket.is_none());
    assert!(call.structured_capture);
    let capture = Arc::new(if fault == PrefixRouteFault::UnauthorizedCapture {
        CostCalibrationCapture::default()
    } else {
        CostCalibrationCapture::default()
            .with_original_route_capture()
            .unwrap()
    });
    let mut work = wave("private.installed-prefix.outside");
    work.host = call.participants[0].host_features.unwrap();
    assert!(work.host.policy.empirical_content_domain.is_none());
    work.actual.rows = call
        .participants
        .iter()
        .map(|p| ActualWaveRow {
            request_id: p.request_id.clone(),
            owner_incarnation: p.owner_incarnation,
            work_generation: p.work_generation,
            input_index: p.input_index,
            work: ActualRowWork::Prefill {
                offset,
                count: 1,
                total_prompt_tokens: offset + 1,
            },
        })
        .collect();
    if fault == PrefixRouteFault::WrongHistory {
        call.participants[0]
            .host_features
            .as_mut()
            .unwrap()
            .state
            .sampling_history_tokens += 1;
    }
    let committed_override =
        (fault == PrefixRouteFault::WrongWork).then_some(HostCommittedWork::Prefill {
            start: offset + 1,
            end: offset + 1,
            total_prompt_tokens: offset + 1,
            generated_tokens_before: 0,
            generated_tokens_after: 1,
        });
    let before = runtime.sink.stats().raw_accepted;
    let mut gate = None;
    let mut complete_private = false;
    let mut original_feedback = false;
    let stages = tracing::subscriber::with_default(diagnostics::RuntimeDebug(true), || {
        call.attach_calibration_capture(capture.clone());
        fixture::record_original_private_route(
            &mut call,
            &clock,
            work,
            committed_override,
            |call| {
                let _ = call.make_route_evidence_recording_failure(&mut gate);
                if let Some((stages, proof)) = call.make_host_stages_with_preparation() {
                    complete_private = proof.is_some();
                    let original = ResolvedCostEntry::new(CostEvidenceEntry::StagesOnly {
                        stages,
                        legacy_rejection: call.rejection.unwrap(),
                    })
                    .with_original_preparation(proof.as_ref());
                    original_feedback = original.preparation_feedback().is_some();
                    assert!(
                        original.structured().is_err(),
                        "private preparation cannot train"
                    );
                }
            },
        )
    });
    drop(call); // Original resolver and FIFO publication, not a DTO replay.
    session.freeze_cost_model().await.unwrap();
    let captured = capture.host_stages();
    let queue = capture.host_stage_queue();
    let status = capture.status();
    let stats = runtime.sink.stats();
    assert_eq!(
        stats.offered, 0,
        "prefix work must not enter numerical training"
    );
    assert!(runtime.snapshot().is_none());
    assert!(runtime
        .training
        .live
        .as_ref()
        .is_none_or(|live| live.audit().population.issued == 0));
    drop(inner);
    drop(outputs);
    session.shutdown().await.unwrap();

    let accepted = fault == PrefixRouteFault::None;
    assert_eq!(
        stages.is_some(),
        accepted,
        "{fault:?}, offset={offset}, gate={gate:?}"
    );
    assert_eq!(
        complete_private, accepted,
        "same-call private proof must reach the resolver"
    );
    assert_eq!(
        original_feedback, accepted,
        "original preparation feedback is the consumption authority"
    );
    assert_eq!(
        captured.is_some(),
        accepted,
        "worker must resolve the actual private capture"
    );
    if accepted {
        let stages = captured.unwrap();
        let proof = capture
            .private_prefix_settlement(&stages)
            .expect("original private proof");
        assert!(proof.outside_preparation(&stages).is_some());
        assert!(
            capture
                .private_prefix_settlement(&Arc::clone(&stages))
                .is_some(),
            "cloning the same original Arc preserves authority"
        );
        assert!(
            capture
                .private_prefix_settlement(&Arc::new((*stages).clone()))
                .is_none(),
            "copied stages cannot acquire original private authority"
        );
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert!(
            stages.actual_shape.is_none(),
            "GraphPath remains numerically unknown"
        );
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_none_or(Result::is_err));
        let queue = queue.unwrap();
        assert_eq!(queue.disposition, HostStageQueueDisposition::Published);
        assert_eq!(queue.accepted_ordinal, Some(before + 1));
        assert!(matches!(status, CostCalibrationStatus::Complete(ref result)
            if matches!(result.as_ref(), CostCalibrationResult::Rejected(CostCallRejection::CalibrationPreparation))));
    } else {
        assert!(queue.is_none());
    }
}

#[tokio::test]
async fn private_installed_prefix_outside_prefill_keeps_settlement_and_fifo() {
    for offset in [0, 2] {
        check_installed_prefix_outside_settlement(offset, PrefixRouteFault::None).await;
    }
}

#[tokio::test]
async fn private_installed_prefix_outside_rejects_unbound_rows_and_bad_host_facts() {
    for fault in [
        PrefixRouteFault::DetachedPrefix,
        PrefixRouteFault::UnauthorizedCapture,
        PrefixRouteFault::WrongWork,
        PrefixRouteFault::WrongHistory,
        PrefixRouteFault::MixedDetachedPrefix,
    ] {
        check_installed_prefix_outside_settlement(2, fault).await;
    }
}
