use super::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_interfaces::vnext::DeviceCommandPhase;
use ferrum_types::RequestId;

fn wave() -> CanonicalStatisticalWave {
    let algorithm = SelectedAlgorithmClassV1::new("fixture.selected", 1, [1; 32], [2; 32]).unwrap();
    let mut command = SelectedCommandCostBuilderV1::new(4);
    command
        .kernel(
            algorithm,
            KernelNumericWorkV1 {
                logical_units: 4,
                padded_units: 8,
                inner_units_per_logical_unit: 16,
                grid: [1, 2, 1],
                scratch_bytes: 64,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let evidence = command.finish().unwrap();
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.selected",
            command_index: 0,
            node_index: None,
            command_phase: DeviceCommandPhase::Compute,
            provider: None,
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 2,
            token_count: 4,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&evidence),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::NoReadback)
        .unwrap();
    for (work, output, generated) in [
        (
            ActualRowWork::Prefill {
                offset: 4,
                count: 3,
                total_prompt_tokens: 12,
            },
            CostRowOutput::Prefill {
                final_logits: false,
            },
            0,
        ),
        (
            ActualRowWork::Decode { kv_tokens: 9 },
            CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            2,
        ),
    ] {
        builder
            .row(CanonicalCostRow {
                work,
                output,
                host_policy_signature: [3; 32],
                mask_upload_required: false,
                host_features: Some(HostCostFeaturesV1 {
                    policy: HostCostPolicyV2 {
                        empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                        categorical_signature: [4; 32],
                        decoder_text_bytes_per_token: 8,
                        decoder_scratch_bytes_per_token: 4,
                        raw_token_bytes_bound: 4,
                    },
                    state: HostCostStateV1 {
                        generated_tokens_before: generated,
                        maximum_output_tokens: 32,
                        sampling_history_tokens: generated,
                        sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                        pending_decoded_utf8: false,
                        completion_state_signature: satisfied_completion_cost_signature(),
                    },
                }),
            })
            .unwrap();
    }
    builder
        .finish_with_statistics(
            ActualWaveKind::Mixed,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap()
}
fn actual(wave: &CanonicalStatisticalWave) -> ActualWaveShape {
    let c = &wave.exact;
    ActualWaveShape {
        kind: c.kind,
        path: c.path,
        graph: c.graph,
        row_order: c.row_order,
        provider_signature: c.provider_signature,
        output_policy_signature: c.output_policy_signature,
        numeric_features: c.numeric_features.clone(),
        host_content_features: c.host_content_features,
        row_multiset_features: c.row_multiset_features.clone(),
        statistical_evidence: Some(wave.statistical.as_ref().unwrap().clone()),
        rows: c
            .rows
            .iter()
            .enumerate()
            .map(|(index, &work)| ActualWaveRow {
                request_id: RequestId::new(),
                owner_incarnation: index as u64 + 1,
                work_generation: 1,
                input_index: index as u32,
                work,
            })
            .collect(),
        recurrent_state_bytes: c.recurrent_state_bytes,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
#[test]
fn actual_and_future_aggregate_the_same_mixed_wave_without_reordering() {
    let wave = wave();
    let future =
        StatisticalModelInputV1::from_future(&wave.exact, wave.statistical.as_ref().unwrap())
            .unwrap();
    let actual = StatisticalModelInputV1::from_actual(&actual(&wave)).unwrap();
    assert_eq!(actual, future);
    let work = actual.host_and_sequence();
    assert_eq!(work.rows, 2);
    assert_eq!(work.prefill_tokens, 3);
    assert_eq!(work.attention_pairs, 5 + 6 + 7 + 10);
    assert_eq!((work.kv_tokens_sum, work.kv_tokens_max), (17, 10));
    assert_eq!(work.prompt_tokens_sum, 12);
    assert_eq!(
        (work.sampling_history_sum, work.sampling_history_max),
        (2, 2)
    );
    assert_eq!(actual.device().inner_work_units, 64);
}
#[test]
fn different_receipt_or_work_never_joins_by_family_alone() {
    let wave = wave();
    let mut receipt = actual(&wave);
    receipt.rows.swap(0, 1);
    assert!(StatisticalModelInputV1::from_actual(&receipt).is_err());
    let mut receipt = actual(&wave);
    receipt.numeric_features.as_mut().unwrap().rows[1].maximum_output_tokens += 1;
    assert_eq!(
        StatisticalModelInputV1::from_actual(&receipt),
        Err(Unknown::ExactBindingMismatch)
    );
    let mut receipt = actual(&wave);
    receipt.statistical_evidence = None;
    assert_eq!(
        StatisticalModelInputV1::from_actual(&receipt),
        Err(Unknown::MissingProducer)
    );
    let mut receipt = actual(&wave);
    receipt.maintenance_units = 1;
    assert_eq!(
        StatisticalModelInputV1::from_actual(&receipt),
        Err(Unknown::UnsupportedWave)
    );
}
