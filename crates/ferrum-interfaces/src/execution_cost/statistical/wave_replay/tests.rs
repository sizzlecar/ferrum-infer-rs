use super::*;
use crate::vnext::DeviceCommandPhase;

fn row() -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 8 },
        host_policy_signature: [3; 32],
        mask_upload_required: false,
        output: CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 2,
            repetition_penalty_bits: 1f32.to_bits(),
        },
        host_features: Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [4; 32],
                decoder_text_bytes_per_token: 4,
                decoder_scratch_bytes_per_token: 8,
                raw_token_bytes_bound: 4,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 2,
                maximum_output_tokens: 10,
                sampling_history_tokens: 2,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
    }
}
fn selected(context: u64, fixed_geometry: bool) -> SelectedCommandCostEvidenceV1 {
    let mut b = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    let class = SelectedAlgorithmClassV1::new("fixture.kernel", 1, [1; 32], [2; 32]).unwrap();
    let work = KernelNumericWorkV1 {
        logical_units: 1,
        padded_units: 32,
        inner_units_per_logical_unit: context,
        grid: [1, 1, 1],
        scratch_bytes: 64,
        staged_weight_bytes: 0,
    };
    if fixed_geometry {
        b.kernel_with_replay_geometry(
            class,
            work,
            KernelReplayGeometryV1 {
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &[1, 32],
            },
        )
        .unwrap();
    } else {
        b.kernel(class, work).unwrap();
    }
    b.finish().unwrap()
}
fn physical() -> CostPhysicalCommand<'static> {
    CostPhysicalCommand {
        native_op_id: "fixture.graph-launch",
        command_index: 2,
        node_index: None,
        command_phase: DeviceCommandPhase::Compute,
        provider: None,
        path: CostCommandPath::Replayed,
        participant_start: 0,
        participant_count: 1,
        token_count: 1,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: Some(1),
        statistical_evidence: None,
    }
}
fn logical(evidence: Option<&SelectedCommandCostEvidenceV1>) -> CostLogicalCommand<'_> {
    CostLogicalCommand {
        native_op_id: "fixture.compute",
        logical_command_ordinal: 0,
        node_index: 0,
        provider: CostProviderIdentity {
            provider_id: "provider",
            implementation_fingerprint: "implementation",
            operation_fingerprint: "operation",
        },
        participant_count: 1,
        token_count: 1,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: 1,
        statistical_evidence: evidence,
    }
}
fn build(
    instance: &str,
    evidence: Option<&SelectedCommandCostEvidenceV1>,
    capture: bool,
    graph: ActualWaveGraphState,
) -> CanonicalStructuredWave {
    let mut b = if capture {
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::GreedyToken)
    } else {
        CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken)
    };
    b.physical_command(physical()).unwrap();
    b.replay_segment(2, instance, 1).unwrap();
    b.logical_command(logical(evidence)).unwrap();
    b.core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    b.row(row()).unwrap();
    b.finish_with_structure(
        ActualWaveKind::Decode,
        ActualWavePath::PlanRuntime,
        graph,
        ActualWaveRowOrder::Ordered,
        64,
    )
    .unwrap()
}

#[test]
fn structured_replay_counts_real_logical_work_without_a_graph_launch_kernel() {
    let current = selected(47, true);
    let a = build(
        "resident-a",
        Some(&current),
        true,
        ActualWaveGraphState::Warm,
    );
    let r = a.structured.unwrap();
    let replay = r.device().replay_work().unwrap();
    assert_eq!(replay.replayed_segments(), 1);
    assert_eq!(replay.logical_commands(), 1);
    assert_eq!(replay.native_graph_nodes(), 1);
    assert_eq!(r.device().physical_commands(), 1);
    let work = r.algorithm_work().unwrap();
    assert_eq!(work.selected_command_count(), 1);
    assert_eq!(work.entries().len(), 1);
    assert_eq!(work.aggregate_work(), current.work());
    assert_eq!(a.statistical.unwrap().work(), current.work());
    assert_eq!(work.aggregate_work().inner_work_units, 47);
    assert!(r.validate_exact(&a.exact).is_ok());
}

#[test]
fn structured_replay_shares_model_template_but_not_instance_bound_evidence() {
    let current = selected(47, true);
    let a = build(
        "resident-a",
        Some(&current),
        true,
        ActualWaveGraphState::Warm,
    );
    let b = build(
        "resident-b",
        Some(&current),
        true,
        ActualWaveGraphState::Warm,
    );
    assert_ne!(a.exact.provider_signature, b.exact.provider_signature);
    assert_eq!(
        a.statistical.as_ref().unwrap().family_signature(),
        b.statistical.as_ref().unwrap().family_signature()
    );
    let ar = a.structured.unwrap();
    let br = b.structured.unwrap();
    assert_eq!(
        ar.device().ordered_template(),
        br.device().ordered_template()
    );
    assert_eq!(
        ar.device().provider_grouped_template(),
        br.device().provider_grouped_template()
    );
    assert_eq!(
        ar.algorithm_work().unwrap().entries(),
        br.algorithm_work().unwrap().entries()
    );
    assert_ne!(
        ar.device().replay_work().unwrap().resident_binding(),
        br.device().replay_work().unwrap().resident_binding()
    );
    assert_ne!(
        ar.algorithm_work().unwrap().ordered_command_binding(),
        br.algorithm_work().unwrap().ordered_command_binding()
    );
    assert_eq!(
        ar.validate_exact(&b.exact),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
    assert!(ar
        .algorithm_work()
        .unwrap()
        .validate_structure(&br)
        .is_err());
    // Numeric evidence cannot authorize either instance. The existing device
    // resident/segment/ordinal and final submission guards are unchanged.
}

#[test]
fn structured_replay_missing_geometry_or_producer_and_cold_graph_stay_unknown() {
    let no_geometry = selected(47, false);
    let good = selected(47, true);
    for result in [
        build("resident", None, true, ActualWaveGraphState::Warm),
        build(
            "resident",
            Some(&no_geometry),
            true,
            ActualWaveGraphState::Warm,
        ),
        build("resident", Some(&good), true, ActualWaveGraphState::Cold),
        build(
            "resident",
            Some(&good),
            true,
            ActualWaveGraphState::ConfiguredEager,
        ),
        build(
            "resident",
            Some(&good),
            true,
            ActualWaveGraphState::Disabled,
        ),
        build("resident", Some(&good), false, ActualWaveGraphState::Warm),
    ] {
        assert!(result.structured.is_err());
        assert!(result.statistical.is_err());
    }
    let later = selected(99, true);
    let first = build("resident", Some(&good), true, ActualWaveGraphState::Warm)
        .structured
        .unwrap();
    let second = build("resident", Some(&later), true, ActualWaveGraphState::Warm)
        .structured
        .unwrap();
    assert_eq!(
        first.device().ordered_template(),
        second.device().ordered_template()
    );
    assert_ne!(
        first.device().aggregate_work(),
        second.device().aggregate_work()
    );
    assert_ne!(
        first.algorithm_work().unwrap().ordered_command_binding(),
        second.algorithm_work().unwrap().ordered_command_binding()
    );
}
