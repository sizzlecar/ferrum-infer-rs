use super::super::super::calibration_capture::StructuredCaptureSessionBinding;
use super::super::super::trainer::structured::{
    capture_numeric_observation,
    inspect_numeric_observation_for_test as whole_wave_numeric_observation,
};
use super::*;
use ferrum_interfaces::execution_cost::{
    CostLogicalCommand, CostProviderIdentity, KernelReplayGeometryV1,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured::StructuredUnknown;

fn session_at(protocol: [u8; 32], opened_at: u64) -> Arc<StructuredCaptureSessionBinding> {
    Arc::new(
        StructuredCaptureSessionBinding::new(
            protocol,
            model::ExecutionFingerprint {
                model_weights: [1; 32],
                numerical_policy: [2; 32],
                device_runtime: [3; 32],
                execution_config: [4; 32],
            },
            &VirtualClock(AtomicU64::new(opened_at)),
        )
        .unwrap(),
    )
}
fn session() -> Arc<StructuredCaptureSessionBinding> {
    session_at([8; 32], 0)
}

fn algorithm_shape(maxima: &[u64], first_work: u64) -> (ActualWaveShape, Vec<HostCostFeaturesV1>) {
    algorithm_shape_graph(maxima, first_work, None)
}
fn algorithm_shape_graph(
    maxima: &[u64],
    first_work: u64,
    resident: Option<&str>,
) -> (ActualWaveShape, Vec<HostCostFeaturesV1>) {
    let (actual, hosts, _) = algorithm_parts_graph(maxima, first_work, resident);
    (actual, hosts)
}
fn algorithm_parts_graph(
    maxima: &[u64],
    first_work: u64,
    resident: Option<&str>,
) -> (
    ActualWaveShape,
    Vec<HostCostFeaturesV1>,
    ferrum_interfaces::execution_cost::CanonicalWaveCostShape,
) {
    let (mut actual, hosts) = selected_shape(maxima);
    let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(maxima.len() as u64);
    for (name, units) in [
        ("fixture.bridge.first", first_work),
        ("fixture.bridge.second", 32 - first_work),
    ] {
        selected
            .kernel_with_replay_geometry(
                SelectedAlgorithmClassV1::new(name, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: units,
                    padded_units: units,
                    inner_units_per_logical_unit: 32,
                    grid: [1, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
                KernelReplayGeometryV1 {
                    block: [32, 1, 1],
                    dynamic_shared_bytes: 0,
                    fixed_parameters: &[units],
                },
            )
            .unwrap();
    }
    let selected = selected.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.structured",
        command_index: 0,
        node_index: None,
        command_phase: DeviceCommandPhase::Compute,
        provider: None,
        path: if resident.is_some() {
            CostCommandPath::Replayed
        } else {
            CostCommandPath::Eager
        },
        participant_start: 0,
        participant_count: maxima.len() as u32,
        token_count: maxima.len() as u64,
        batching_form: "packed",
        compute_dispatch_count: if resident.is_some() { 1 } else { 2 },
        transfer_command_count: 0,
        reusable_graph_node_count: resident.map(|_| 2),
        statistical_evidence: resident.is_none().then_some(&selected),
    })
    .unwrap();
    if let Some(resident) = resident {
        b.replay_segment(0, resident, 1).unwrap();
        b.logical_command(CostLogicalCommand {
            native_op_id: "fixture.structured",
            logical_command_ordinal: 0,
            node_index: 0,
            provider: CostProviderIdentity {
                provider_id: "fixture.provider",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            },
            participant_count: maxima.len() as u32,
            token_count: maxima.len() as u64,
            batching_form: "packed",
            compute_dispatch_count: 2,
            transfer_command_count: 0,
            reusable_graph_node_count: 2,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    }
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    for host in &hosts {
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 7 },
            host_policy_signature: [6; 32],
            mask_upload_required: false,
            host_features: Some(*host),
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            },
        })
        .unwrap();
    }
    let built = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            if resident.is_some() {
                ActualWaveGraphState::Warm
            } else {
                ActualWaveGraphState::Disabled
            },
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    actual.provider_signature = built.exact.provider_signature;
    actual.graph = built.exact.graph;
    actual.output_policy_signature = built.exact.output_policy_signature;
    actual.numeric_features = built.exact.numeric_features.clone();
    actual.host_content_features = built.exact.host_content_features;
    actual.row_multiset_features = built.exact.row_multiset_features.clone();
    actual.statistical_evidence = Some(built.statistical.unwrap());
    (actual, hosts, built.exact)
}
fn stages(
    maxima: &[u64],
    result: Option<HostTerminalStageV1>,
    work: u64,
) -> Arc<HostStageEvidenceV1> {
    stages_with_capture(maxima, result, work, None)
}
fn stages_with_capture(
    maxima: &[u64],
    result: Option<HostTerminalStageV1>,
    work: u64,
    capture: Option<Arc<CostCalibrationCapture>>,
) -> Arc<HostStageEvidenceV1> {
    stages_with_graph(maxima, result, work, capture, None)
}
fn stages_with_graph(
    maxima: &[u64],
    result: Option<HostTerminalStageV1>,
    work: u64,
    capture: Option<Arc<CostCalibrationCapture>>,
    resident: Option<&str>,
) -> Arc<HostStageEvidenceV1> {
    let (actual, hosts) = algorithm_shape_graph(maxima, work, resident);
    stages_for_actual(actual, hosts, result, capture)
}
fn stages_for_actual(
    actual: ActualWaveShape,
    hosts: Vec<HostCostFeaturesV1>,
    result: Option<HostTerminalStageV1>,
    capture: Option<Arc<CostCalibrationCapture>>,
) -> Arc<HostStageEvidenceV1> {
    // This fixture retains the additional per-algorithm table. Keep the old
    // shared begin default (32) and production limits untouched.
    let (call, clock) = begin_with_retained_capacity(&actual, &sink(8, 256), 64);
    let mut call = call.with_structured_capture(true);
    for (participant, host) in call.participants.iter_mut().zip(&hosts) {
        participant.host_features = Some(*host);
    }
    if let Some(capture) = &capture {
        call.attach_calibration_capture(Arc::clone(capture));
    }
    execute(&mut call, &clock, actual.clone());
    for (position, row) in actual.rows.iter().enumerate() {
        settle(
            &mut call,
            &clock,
            row,
            10 + position as u64 * 10,
            if position == 0 { result.clone() } else { None },
        );
    }
    call.reject(CostCallRejection::Composite);
    clock.set(100);
    let stages = call.make_host_stages().unwrap();
    // Finish once through the real writer. Dropping an unfinished attached call
    // publishes Abandoned; subsequent manual complete_* would mark a conflict.
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
    capture.map_or(stages, |capture| capture.host_stages().unwrap())
}
fn entry(stages: Arc<HostStageEvidenceV1>) -> CostEvidenceEntry {
    CostEvidenceEntry::StagesOnly {
        stages,
        legacy_rejection: CostCallRejection::Composite,
    }
}

#[path = "model_bridge/multi.rs"]
mod multi;

#[test]
fn structured_bridge_v2_warm_graph_uses_private_settlement_and_keeps_legacy_closed() {
    use super::super::super::trainer::structured_v2;
    let stages_a = stages_with_graph(&[3], Some(terminal()), 8, None, Some("resident-A"));
    let stages_b = stages_with_graph(&[3], Some(terminal()), 8, None, Some("resident-B"));
    let input_a = structured_v2::structured_discovery_input_v2(&stages_a).unwrap();
    let input_b = structured_v2::structured_discovery_input_v2(&stages_b).unwrap();
    assert_eq!(input_a, input_b);
    assert_eq!(
        stages_a.actual_shape.as_ref().unwrap().graph_state,
        model::WaveGraphState::Warm
    );
    assert_eq!(stages_a.full_wall_ns, Some(12));
    assert_ne!(stages_a.actual_shape, stages_b.actual_shape);
    assert!(whole_wave_numeric_observation(&entry(stages_a), 1, &session()).is_err());
}

#[test]
fn structured_bridge_v2_warm_graph_rejects_receipt_swap_clock_and_unsupported_terminal() {
    use super::super::super::trainer::structured_v2::structured_discovery_input_v2;
    let original = stages_with_graph(&[3], Some(terminal()), 8, None, Some("resident-A"));
    let (another, _) = algorithm_shape_graph(&[3], 8, Some("resident-B"));
    for failure in 0..3 {
        let mut changed = original.as_ref().clone();
        match failure {
            0 => changed.structured_evidence = None,
            1 => changed.statistical_evidence = another.statistical_evidence.clone(),
            _ => changed.full_wall_ns = Some(999),
        }
        assert!(
            structured_discovery_input_v2(&Arc::new(changed)).is_err(),
            "case {failure}"
        );
    }
    let mut eos = terminal();
    eos.finish_reason = FinishReason::EOS;
    let stages = stages_with_graph(&[3], Some(eos), 8, None, Some("resident-A"));
    assert!(matches!(structured_discovery_input_v2(&stages), Err(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2::UnsupportedScope)));
}

#[test]
fn structured_bridge_uses_real_settlement_wall_and_fifo_metadata() {
    let session = session();
    let stages = stages(&[3, 4], Some(terminal()), 8);
    let wall = stages.full_wall_ns.unwrap();
    let sample = whole_wave_numeric_observation(&entry(Arc::clone(&stages)), 17, &session).unwrap();
    assert_eq!(sample.wall_ns, wall);
    assert_eq!(sample.observed_at_ns, 100);
    assert_eq!(sample.ordinal, 17);
    assert_eq!(sample.call_id, stages.call_id);
    assert_eq!(sample.source, session.identity());
    assert_eq!(sample.protocol, session.protocol());
    assert_eq!(sample.input.physical_host_rows().len(), 2);
    assert_eq!(
        sample.input.physical_host_rows()[0].terminal_expectation,
        HostTerminalExpectationV1::LengthBoundary
    );
}

#[test]
fn structured_bridge_eos_stop_and_unexpected_early_terminal_do_not_become_length() {
    for (maximum, reason) in [
        (3, FinishReason::EOS),
        (3, FinishReason::Stop),
        (20, FinishReason::EOS),
    ] {
        let mut result = terminal();
        result.finish_reason = reason;
        let stages = stages(&[maximum], Some(result), 8);
        assert!(stages.structured_evidence.as_ref().unwrap().is_ok());
        assert!(matches!(
            whole_wave_numeric_observation(&entry(stages), 1, &session()),
            Err(StructuredUnknown::UnsupportedScope)
        ));
    }
}

#[test]
fn structured_bridge_unknown_cleanup_and_cancellation_are_not_samples() {
    for failure in 0..3 {
        let mut result = terminal();
        match failure {
            0 => result.cache_completion_work = ExecutorCompletionWork::Unknown,
            1 => result.admission_cancellation_work = ExecutorCompletionWork::Unknown,
            _ => result.finish_reason = FinishReason::Cancelled,
        }
        let stages = stages(&[3], Some(result), 8);
        assert!(matches!(
            whole_wave_numeric_observation(&entry(stages), 1, &session()),
            Err(StructuredUnknown::MissingEvidence)
        ));
    }
}

#[test]
fn structured_bridge_rejects_same_exact_recipe_exchange_after_private_settlement() {
    let original = stages(&[3], Some(terminal()), 8);
    let (changed, _) = algorithm_shape(&[3], 24);
    assert_eq!(original.statistical_evidence, changed.statistical_evidence);
    let mut exchanged = original.as_ref().clone();
    exchanged.statistical_evidence = changed.statistical_evidence;
    // The legacy stage digest intentionally excludes the capture extension.
    // The new consumer must additionally bind selected's complete sidecar.
    exchanged
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .validate_host_stages(&exchanged)
        .unwrap();
    assert!(matches!(
        whole_wave_numeric_observation(&entry(Arc::new(exchanged)), 1, &session()),
        Err(StructuredUnknown::MissingEvidence)
    ));
}

#[test]
fn structured_bridge_requires_private_qualified_value_and_unchanged_bound_fields() {
    let original = stages(&[3], Some(terminal()), 8);
    for failure in 0..3 {
        let mut changed = original.as_ref().clone();
        match failure {
            0 => changed.structured_evidence = None,
            1 => changed.full_wall_ns = Some(999),
            _ => changed.call_id += 1,
        }
        assert!(whole_wave_numeric_observation(&entry(Arc::new(changed)), 1, &session()).is_err());
    }
    assert!(matches!(
        whole_wave_numeric_observation(&entry(Arc::clone(&original)), 0, &session()),
        Err(StructuredUnknown::WrongSource)
    ));
}

#[test]
fn structured_bridge_capture_requires_published_original_receipt_and_reconciliation() {
    let session = session();
    let capture = Arc::new(CostCalibrationCapture::for_structured_session(Arc::clone(
        &session,
    )));
    let _stages = stages_with_capture(&[3], Some(terminal()), 8, Some(Arc::clone(&capture)));
    assert_eq!(
        capture.host_stage_queue().unwrap().accepted_ordinal,
        Some(1)
    );
    assert_eq!(
        capture_numeric_observation(&capture, true).unwrap().ordinal,
        1
    );
    assert!(matches!(
        capture_numeric_observation(&capture, false),
        Err(StructuredUnknown::InvalidSample)
    ));
    let dropped = CostCalibrationCapture::for_structured_session(Arc::clone(&session));
    dropped.complete_host_stages(stages(&[3], Some(terminal()), 8));
    dropped.complete(CostCalibrationResult::Rejected(
        CostCallRejection::Composite,
    ));
    dropped.complete_host_stage_queue(HostStageQueueReceipt {
        accepted_ordinal: None,
        disposition: HostStageQueueDisposition::DroppedCapacity,
    });
    assert!(matches!(
        capture_numeric_observation(&dropped, true),
        Err(StructuredUnknown::WrongSource)
    ));
}

#[test]
fn structured_bridge_complete_sample_fields_without_qualified_receipt_do_not_qualify() {
    let stages = stages(&[3], Some(terminal()), 8);
    let fake = ferrum_scheduler::implementations::continuous::cost_model::WaveCostObservation {
        fingerprint: stages.fingerprint.as_ref().unwrap().clone(), actual_shape: stages.actual_shape.as_ref().unwrap().clone(),
        boundary: ferrum_scheduler::implementations::continuous::cost_model::CostBoundary::PreparationToHostSettledV1,
        outcome: ferrum_scheduler::implementations::continuous::cost_model::WaveObservationOutcome::Completed,
        timing: ferrum_scheduler::implementations::continuous::cost_model::WaveTiming {
            wall_total_ns: stages.full_wall_ns.unwrap(),device_elapsed_ns: None,stages: Default::default() },
        observed_at_ns: stages.finalized_at_ns.unwrap(),
    };
    assert!(matches!(
        whole_wave_numeric_observation(
            &CostEvidenceEntry::Training {
                sample: fake,
                stages: None
            },
            1,
            &session()
        ),
        Err(StructuredUnknown::MissingEvidence)
    ));
}

fn bound_capture(session: Arc<StructuredCaptureSessionBinding>) -> Arc<CostCalibrationCapture> {
    let capture = Arc::new(CostCalibrationCapture::for_structured_session(session));
    let _stages = stages_with_capture(&[3, 4], Some(terminal()), 8, Some(Arc::clone(&capture)));
    assert!(matches!(
        capture.status(),
        CostCalibrationStatus::Complete(_)
    ));
    assert_eq!(
        capture.host_stage_queue().unwrap().accepted_ordinal,
        Some(1)
    );
    capture
}

#[test]
fn structured_bridge_binding_is_fixed_before_execution_and_not_supplied_at_conversion() {
    let first = session_at([8; 32], 0);
    let second = session_at([9; 32], 0);
    assert_ne!(first.identity(), second.identity());
    for session in [first, second] {
        let capture = bound_capture(Arc::clone(&session));
        let sample = capture_numeric_observation(&capture, true).unwrap();
        assert_eq!(sample.source, session.identity());
        assert_eq!(sample.protocol, session.protocol());
        assert_eq!(&sample.fingerprint, session.fingerprint());
        assert_eq!(sample.input.physical_host_rows().len(), 2);
        assert_eq!(sample.ordinal, 1);
    }
    // An old capture with all valid receipts still has no new session binding.
    let legacy = CostCalibrationCapture::default();
    legacy.complete_host_stages(stages(&[3], Some(terminal()), 8));
    legacy.complete(CostCalibrationResult::Rejected(
        CostCallRejection::Composite,
    ));
    legacy.complete_host_stage_queue(HostStageQueueReceipt {
        accepted_ordinal: Some(17),
        disposition: HostStageQueueDisposition::Published,
    });
    assert!(matches!(
        capture_numeric_observation(&legacy, true),
        Err(StructuredUnknown::WrongSource)
    ));
}

#[test]
fn structured_bridge_rejects_wrong_session_fingerprint_and_preopening_call() {
    let late = bound_capture(session_at([8; 32], 2)); // Real prepare started at 1.
    assert!(matches!(
        capture_numeric_observation(&late, true),
        Err(StructuredUnknown::Clock)
    ));
    let valid = session();
    let mut wrong = valid.fingerprint().clone();
    wrong.device_runtime = [99; 32];
    let binding = Arc::new(
        StructuredCaptureSessionBinding::new([8; 32], wrong, &VirtualClock(AtomicU64::new(0)))
            .unwrap(),
    );
    assert!(matches!(
        capture_numeric_observation(&bound_capture(binding), true),
        Err(StructuredUnknown::WrongFingerprint)
    ));
    assert!(matches!(
        StructuredCaptureSessionBinding::new(
            [0; 32],
            valid.fingerprint().clone(),
            &VirtualClock(AtomicU64::new(0)),
        ),
        Err(StructuredUnknown::WrongProtocol)
    ));
}

#[test]
fn structured_bridge_rejects_observed_receipt_ordinal_disagreement() {
    let capture = Arc::new(CostCalibrationCapture::for_structured_session(session()));
    // Deliberately assemble mismatched FIFO receipts from an independently
    // completed private stage fixture; do not overwrite a real attached capture.
    let stages = stages_with_capture(&[4], None, 8, None);
    let sample = model::WaveCostObservation {
        fingerprint: stages.fingerprint.as_ref().unwrap().clone(),
        actual_shape: stages.actual_shape.as_ref().unwrap().clone(),
        boundary: model::CostBoundary::PreparationToHostSettledV1,
        outcome: model::WaveObservationOutcome::Completed,
        timing: model::WaveTiming {
            wall_total_ns: stages.full_wall_ns.unwrap(),
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: stages.finalized_at_ns.unwrap(),
    };
    capture.complete_host_stages(stages);
    capture.complete(CostCalibrationResult::Observed {
        sample: Box::new(sample),
        actual_rows: Vec::new(),
        commits: Vec::new(),
        host_features: Vec::new(),
        accepted_ordinal: Some(16),
        disposition: CostCallDisposition::Published,
    });
    capture.complete_host_stage_queue(HostStageQueueReceipt {
        accepted_ordinal: Some(17),
        disposition: HostStageQueueDisposition::Published,
    });
    assert!(matches!(
        capture_numeric_observation(&capture, true),
        Err(StructuredUnknown::InvalidSample)
    ));
}
