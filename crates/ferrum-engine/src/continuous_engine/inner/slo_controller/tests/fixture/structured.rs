//! Typed controlled CPU producer, enabled only by the live collector test.
//! It describes the executed full-logits fill and the original host rows.
use super::*;

pub(super) fn record(
    context: &mut PlanRuntimeCostObservationContext<'_>,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
) {
    assert!(decodes
        .iter()
        .all(|row| row.logits_policy.requires_full_logits()));
    assert!(prefills.iter().all(|row| row.chunk.is_final()));
    let row_count = prefills.len() + decodes.len();
    let query_tokens = prefills
        .iter()
        .map(|row| row.chunk.tokens_to_process() as u64)
        .sum::<u64>()
        + decodes.len() as u64;
    let mut algorithm = SelectedCommandCostBuilderV1::new_with_algorithm_work(query_tokens);
    algorithm
        .kernel(
            SelectedAlgorithmClassV1::new(
                "fixture.controlled.full_logits_fill",
                1,
                [1; 32],
                [2; 32],
            )
            .unwrap(),
            KernelNumericWorkV1 {
                logical_units: row_count as u64 * 64,
                padded_units: row_count as u64 * 64,
                inner_units_per_logical_unit: 1,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let algorithm = algorithm.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.controlled.full_logits_fill",
            command_index: 0,
            node_index: None,
            command_phase: vnext::DeviceCommandPhase::Compute,
            provider: None,
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: row_count as u32,
            token_count: query_tokens,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&algorithm),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    let mut rows = Vec::with_capacity(row_count);
    for (id, work, output) in prefills
        .iter()
        .map(|row| {
            (
                &row.request_id,
                ActualRowWork::Prefill {
                    offset: row.chunk.tokens_processed() as u32,
                    count: row.chunk.tokens_to_process() as u32,
                    total_prompt_tokens: row.chunk.total_prompt_tokens() as u32,
                },
                CostRowOutput::Prefill { final_logits: true },
            )
        })
        .chain(decodes.iter().map(|row| {
            (
                &row.request_id,
                ActualRowWork::Decode {
                    kv_tokens: row.kv_cache.num_tokens() as u32,
                },
                CostRowOutput::Decode {
                    requires_full_logits: true,
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1_f32.to_bits(),
                },
            )
        }))
    {
        let participant = context.participant(id).unwrap();
        let host = participant
            .host_features
            .expect("installed credited host policy");
        builder
            .row(CanonicalCostRow {
                work,
                host_policy_signature: participant.output_policy_signature.unwrap(),
                mask_upload_required: false,
                host_features: Some(host),
                output,
            })
            .unwrap();
        rows.push(ActualWaveRow {
            request_id: id.clone(),
            owner_incarnation: participant.owner_incarnation,
            work_generation: participant.work_generation,
            input_index: participant.input_index,
            work,
        });
    }
    let kind = match (prefills.is_empty(), decodes.is_empty()) {
        (true, _) => ActualWaveKind::Decode,
        (_, true) => ActualWaveKind::Prefill,
        _ => ActualWaveKind::Mixed,
    };
    let recurrent_state_bytes = 0; // This controlled executor owns no recurrent state.
    let built = builder
        .finish_with_captured_structure(
            kind,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            recurrent_state_bytes,
        )
        .unwrap();
    let shape = ActualWaveShape {
        statistical_evidence: Some(built.statistical.unwrap()),
        kind,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: built.exact.provider_signature,
        output_policy_signature: built.exact.output_policy_signature,
        numeric_features: built.exact.numeric_features,
        host_content_features: built.exact.host_content_features,
        row_multiset_features: built.exact.row_multiset_features,
        rows,
        recurrent_state_bytes,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    };
    shape
        .statistical_evidence
        .as_ref()
        .unwrap()
        .validate_actual(&shape)
        .unwrap();
    let at = context.now_ns();
    context.physical_wave(Ok(shape), at);
}
