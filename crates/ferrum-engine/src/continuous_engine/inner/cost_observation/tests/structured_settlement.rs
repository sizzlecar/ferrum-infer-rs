use super::super::host_stages::StructuredSettlementUnknown;
use super::*;
use ferrum_interfaces::{model_executor::ExecutorCompletionWork, vnext::DeviceCommandPhase};
use ferrum_types::{FinishReason, InferenceRequest, TokenId};

fn selected_shape(maxima: &[u64]) -> (ActualWaveShape, Vec<HostCostFeaturesV1>) {
    let mut selected = SelectedCommandCostBuilderV1::new(maxima.len() as u64);
    selected
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.structured", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 32,
                padded_units: 32,
                inner_units_per_logical_unit: 32,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = selected.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.structured",
        command_index: 0,
        node_index: None,
        command_phase: DeviceCommandPhase::Compute,
        provider: None,
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: maxima.len() as u32,
        token_count: maxima.len() as u64,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    let mut hosts = Vec::new();
    for &maximum in maxima {
        let host = HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [9; 32],
                decoder_text_bytes_per_token: 4,
                decoder_scratch_bytes_per_token: 8,
                raw_token_bytes_bound: 4,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 2,
                maximum_output_tokens: maximum,
                sampling_history_tokens: 2,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        };
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 7 },
            host_policy_signature: [6; 32],
            host_features: Some(host),
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            },
        })
        .unwrap();
        hosts.push(host);
    }
    let result = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let mut actual = shape(&vec![ActualRowWork::Decode { kv_tokens: 7 }; maxima.len()]);
    actual.provider_signature = result.exact.provider_signature;
    actual.output_policy_signature = result.exact.output_policy_signature;
    actual.numeric_features = result.exact.numeric_features;
    actual.host_content_features = result.exact.host_content_features;
    actual.row_multiset_features = result.exact.row_multiset_features;
    actual.statistical_evidence = Some(result.statistical.unwrap());
    for (position, row) in actual.rows.iter_mut().enumerate() {
        row.input_index = 2 + position as u32 * 5;
    }
    (actual, hosts)
}
fn start(
    actual: &ActualWaveShape,
    hosts: &[HostCostFeaturesV1],
) -> (EngineCostCall, Arc<VirtualClock>) {
    let (call, clock) = begin(actual, &sink(8, 256));
    let mut call = call.with_structured_capture(true);
    for (participant, host) in call.participants.iter_mut().zip(hosts) {
        participant.host_features = Some(*host);
    }
    execute(&mut call, &clock, actual.clone());
    (call, clock)
}
fn owner(row: &ActualWaveRow) -> crate::continuous_engine::SequenceState {
    let mut request = InferenceRequest::new("input", "model");
    request.id = row.request_id.clone();
    let mut owner = crate::continuous_engine::SequenceState::new(request, vec![TokenId::new(1)]);
    owner.cost_frontier = Some(CostFrontier {
        owner_incarnation: NonZeroU64::new(row.owner_incarnation).unwrap(),
        work_generation: NonZeroU64::new(row.work_generation + 1).unwrap(),
    });
    owner
}
fn terminal() -> HostTerminalStageV1 {
    HostTerminalStageV1 {
        finish_reason: FinishReason::Length,
        generated_tokens: 3,
        through_output_ordinal: 3,
        output_failed: false,
        physical_failed: false,
        scheduler_failed: false,
        terminal_handoff_succeeded: true,
        pending_restore_removed: false,
        admission_cancellation_work: ExecutorCompletionWork::NoAdditionalWork,
        cache_completion_work: ExecutorCompletionWork::NoAdditionalWork,
        other_physical_resources: false,
        request_slot_closed: true,
        owner_matched: true,
    }
}
fn settle(
    call: &mut EngineCostCall,
    clock: &VirtualClock,
    row: &ActualWaveRow,
    at: u64,
    terminal: Option<HostTerminalStageV1>,
) {
    clock.set(at);
    call.begin_host_row(&row.request_id);
    clock.set(at + 1);
    let evidence = committed(row, at + 1);
    call.note_host_token_commit(&evidence);
    if let Some(terminal) = terminal {
        let mut pending = call.host_publication(&evidence, true, true).unwrap();
        clock.set(at + 2);
        pending.terminal_handed_off();
        clock.set(at + 3);
        call.record_settled(pending.settle(owner(row), terminal));
    } else {
        clock.set(at + 3);
        assert!(call.host_publication(&evidence, false, true).is_none());
        call.record_host_result(evidence);
    }
}
#[test]
fn structured_settlement_binds_physical_rows_and_real_reverse_host_order() {
    let (actual, hosts) = selected_shape(&[3, 4]);
    let (mut call, clock) = start(&actual, &hosts);
    settle(&mut call, &clock, &actual.rows[1], 10, None);
    let mut eos = terminal();
    eos.finish_reason = FinishReason::EOS;
    settle(&mut call, &clock, &actual.rows[0], 20, Some(eos));
    call.reject(CostCallRejection::Composite);
    clock.set(25);
    let stages = call.make_host_stages().unwrap();
    let qualified = stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap();
    qualified.validate_host_stages(&stages).unwrap();
    assert_eq!(qualified.full_wall_ns(), 22);
    assert_eq!(
        stages
            .rows
            .iter()
            .map(|r| r.input_index)
            .collect::<Vec<_>>(),
        [2, 7]
    );
    assert_eq!(
        stages
            .rows
            .iter()
            .map(|r| r.host_processing_ordinal)
            .collect::<Vec<_>>(),
        [Some(1), Some(0)]
    );
    assert_eq!(
        qualified.recipe().physical_host_rows()[0].physical_position,
        0
    );
    assert!(
        stages.retained_rows().unwrap()
            >= qualified.recipe().retained_rows() * 2 + stages.rows.capacity()
    );
    let mut changed = stages.as_ref().clone();
    changed.rows[0].work_generation += 1;
    assert!(qualified.validate_host_stages(&changed).is_err());
    changed = stages.as_ref().clone();
    changed.call_id += 1;
    assert!(qualified.validate_host_stages(&changed).is_err());
    changed = stages.as_ref().clone();
    changed.actual_shape.as_mut().unwrap().recurrent_state_bytes += 1;
    assert!(qualified.validate_host_stages(&changed).is_err());
    changed = stages.as_ref().clone();
    changed.rows.swap(0, 1);
    assert!(qualified.validate_host_stages(&changed).is_err());
}
#[test]
fn structured_settlement_unknown_cleanup_cancel_failure_and_missing_terminal_do_not_qualify() {
    for failure in 0..7 {
        let (actual, hosts) = selected_shape(&[3]);
        let (mut call, clock) = start(&actual, &hosts);
        let mut result = terminal();
        match failure {
            0 => result.cache_completion_work = ExecutorCompletionWork::Unknown,
            1 => result.output_failed = true,
            2 => result.pending_restore_removed = true,
            3 => result.owner_matched = false,
            6 => result.finish_reason = FinishReason::Cancelled,
            _ => {}
        }
        settle(
            &mut call,
            &clock,
            &actual.rows[0],
            10,
            if failure == 4 { None } else { Some(result) },
        );
        if failure == 5 {
            call.host_cancelled(&actual.rows[0].request_id);
        }
        let stages = call.make_host_stages().unwrap();
        assert!(
            stages.structured_evidence.as_ref().unwrap().is_err(),
            "failure {failure}"
        );
    }
}
#[test]
fn structured_settlement_private_receipt_cannot_cross_equal_public_call_ids() {
    let (actual, hosts) = selected_shape(&[3]);
    let (mut first, clock) = start(&actual, &hosts);
    let (mut second, second_clock) = start(&actual, &hosts);
    assert_eq!(first.call_id, second.call_id);
    clock.set(10);
    first.begin_host_row(&actual.rows[0].request_id);
    let evidence = committed(&actual.rows[0], 10);
    first.note_host_token_commit(&evidence);
    let mut pending = first.host_publication(&evidence, true, true).unwrap();
    clock.set(12);
    pending.terminal_handed_off();
    second_clock.set(20);
    second.record_settled(pending.settle(owner(&actual.rows[0]), terminal()));
    assert!(second.make_host_stages().is_none());
    assert!(first
        .make_host_stages()
        .unwrap()
        .structured_evidence
        .as_ref()
        .unwrap()
        .is_err());
}
#[test]
fn structured_settlement_rejects_changed_exact_work_and_charges_arc_capacity() {
    let (mut actual, hosts) = selected_shape(&[3]);
    let recipe = actual
        .statistical_evidence
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .unwrap()
        .clone();
    recipe.validate_actual(&actual).unwrap();
    actual.rows[0].work = ActualRowWork::Decode { kv_tokens: 8 };
    assert!(recipe.validate_actual(&actual).is_err());
    let (mut call, clock) = start(&actual, &hosts);
    settle(&mut call, &clock, &actual.rows[0], 10, Some(terminal()));
    assert!(call
        .make_host_stages()
        .unwrap()
        .structured_evidence
        .as_ref()
        .unwrap()
        .is_err());
    let (actual, _) = selected_shape(&[3]);
    let original_rows = actual.rows.capacity()
        + actual.numeric_features.as_ref().unwrap().rows.capacity()
        + actual
            .row_multiset_features
            .as_ref()
            .unwrap()
            .rows
            .capacity();
    let mut recorder = BoundedWaveRecorder::new(
        NonZeroU64::MIN,
        CostRecorderLimits {
            max_waves: 1,
            max_rows_per_wave: 8,
            max_retained_rows: original_rows.max(8),
        },
    )
    .unwrap();
    assert!(recorder
        .begin(
            actual,
            WaveObservationBoundary::IsolatedPreparationToCommit,
            1
        )
        .is_err());
}

#[test]
fn structured_settlement_missing_producer_and_disabled_real_calls_are_distinct() {
    for enabled in [false, true] {
        let actual = shape(&[ActualRowWork::Decode { kv_tokens: 7 }]);
        let (call, _clock) = begin(&actual, &sink(4, 32));
        let mut call = call.with_structured_capture(enabled);
        assert_eq!(
            call.context().unwrap().structured_capture_enabled(),
            enabled
        );
        // A context is single-use. Build the real call once more for execution.
        drop(call);
        let (call, clock) = begin(&actual, &sink(4, 32));
        let mut call = call.with_structured_capture(enabled);
        execute(&mut call, &clock, actual.clone());
        settle(&mut call, &clock, &actual.rows[0], 10, None);
        let stages = call.make_host_stages().unwrap();
        if enabled {
            assert!(matches!(
                stages.structured_evidence,
                Some(Err(StructuredSettlementUnknown::MissingProducer))
            ));
        } else {
            assert!(stages.structured_evidence.is_none());
            assert_eq!(
                serde_json::to_vec(stages.as_ref()).unwrap(),
                serde_json::to_vec(&stages.structured_diagnostic_view()).unwrap()
            );
        }
    }
}
