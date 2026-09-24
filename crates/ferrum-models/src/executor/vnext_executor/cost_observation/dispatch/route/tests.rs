use super::*;
use ferrum_interfaces::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, DeviceNativeOperationId, DeviceNativeWorkAttribution,
    DeviceSubmissionAttribution,
};

fn provider(_: u32) -> Option<CostProviderIdentity<'static>> {
    Some(CostProviderIdentity {
        provider_id: "fixture.provider",
        implementation_fingerprint: "fixture-implementation",
        operation_fingerprint: "fixture-operation",
    })
}
fn attribution(
    phase: DeviceCommandPhase,
    computes: u64,
    transfers: u64,
) -> DeviceSubmissionAttribution {
    DeviceSubmissionAttribution::new(vec![DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        phase,
        DeviceNativeOperationId::new("fixture.native").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        1,
        1,
        computes,
        transfers,
        None,
    )
    .unwrap()])
    .unwrap()
}
fn row() -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 12 },
        host_policy_signature: host_history_cost_signature([3; 32], 1),
        host_features: None,
        mask_upload_required: true,
        output: CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 0,
            repetition_penalty_bits: 1.0f32.to_bits(),
        },
    }
}
fn finish(mut route: ObservedRoute, retries: u32) -> CanonicalWaveCostShape {
    route.canonical.row(row()).unwrap();
    route
        .canonical
        .finish(
            ActualWaveKind::Decode,
            if retries == 0 {
                ActualWavePath::PlanRuntime
            } else {
                ActualWavePath::UnsupportedFallback
            },
            route.graph,
            ActualWaveRowOrder::Ordered,
            128,
        )
        .unwrap()
}
fn shape(attribution: &DeviceSubmissionAttribution, retries: u32) -> CanonicalWaveCostShape {
    finish(
        actual_route(
            Some(attribution),
            provider,
            DeviceCostGraphCaptureCapability::Unsupported,
            CostProductOutput::GreedyToken,
            retries,
        )
        .unwrap(),
        retries,
    )
}

#[test]
fn production_actual_route_matches_declared_canonical_inputs_losslessly() {
    let actual = attribution(DeviceCommandPhase::Compute, 2, 1);
    let resolved = shape(&actual, 0);
    let mut declared = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    // Independent typed construction is the prospective route side; the actual
    // side above calls the exact adapter consumed by production actual_shape.
    declared
        .physical_command(CostPhysicalCommand {
            statistical_evidence: None,
            native_op_id: "fixture.native",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: provider(0),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            batching_form: "packed",
            compute_dispatch_count: 2,
            transfer_command_count: 1,
            reusable_graph_node_count: None,
        })
        .unwrap();
    declared.row(row()).unwrap();
    let declared = declared
        .finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            128,
        )
        .unwrap();
    assert_eq!(resolved, declared);
}

#[test]
fn actual_attribution_numeric_rows_match_a_full_declared_route() {
    let evidence = attribution(DeviceCommandPhase::Compute, 2, 1);
    let mut observed = actual_route(
        Some(&evidence),
        provider,
        DeviceCostGraphCaptureCapability::Unsupported,
        CostProductOutput::GreedyToken,
        0,
    )
    .unwrap()
    .canonical;
    let mut declared = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    declared
        .physical_command(CostPhysicalCommand {
            statistical_evidence: None,
            native_op_id: "fixture.native",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: provider(0),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            batching_form: "packed",
            compute_dispatch_count: 2,
            transfer_command_count: 1,
            reusable_graph_node_count: None,
        })
        .unwrap();
    let mut actual_row = row();
    actual_row.host_features = Some(HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: None,
            categorical_signature: [4; 32],
            decoder_text_bytes_per_token: 9,
            decoder_scratch_bytes_per_token: 3,
            raw_token_bytes_bound: 3,
        },
        state: HostCostStateV1 {
            generated_tokens_before: 1,
            maximum_output_tokens: 10,
            sampling_history_tokens: 1,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    });
    for builder in [&mut observed, &mut declared] {
        builder
            .core_readback_route(CoreReadbackRoute::SubmissionStaged)
            .unwrap();
        builder.row(actual_row).unwrap();
    }
    let finish = |builder: CanonicalWaveCostBuilder| {
        builder
            .finish(
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Disabled,
                ActualWaveRowOrder::Ordered,
                128,
            )
            .unwrap()
    };
    let observed = finish(observed);
    let declared = finish(declared);
    assert_eq!(observed, declared);
    let features = observed
        .numeric_features
        .expect("complete actual evidence has numeric rows");
    assert_eq!(features.rows[0].decoded_prefix_tokens, 2);
    assert_eq!(features.rows[0].decoded_text_bytes_bound, 18);
    assert_eq!(features.rows[0].decode_scratch_bytes_bound, 6);
}

#[test]
fn production_actual_route_separates_phase_counts_and_fallback() {
    let baseline = shape(&attribution(DeviceCommandPhase::Compute, 1, 1), 0);
    for evidence in [
        attribution(DeviceCommandPhase::DynamicBinding, 1, 1),
        attribution(DeviceCommandPhase::Compute, 2, 1),
        attribution(DeviceCommandPhase::Compute, 1, 2),
    ] {
        assert_ne!(
            baseline.provider_signature,
            shape(&evidence, 0).provider_signature
        );
    }
    let fallback = shape(&attribution(DeviceCommandPhase::Compute, 1, 1), 1);
    assert_ne!(baseline.provider_signature, fallback.provider_signature);
    assert_eq!(fallback.path, ActualWavePath::UnsupportedFallback);
}

#[test]
fn production_actual_route_rejects_missing_provider_graph_and_attribution() {
    let evidence = attribution(DeviceCommandPhase::Compute, 1, 0);
    assert!(matches!(
        actual_route(
            None,
            provider,
            DeviceCostGraphCaptureCapability::Unsupported,
            CostProductOutput::GreedyToken,
            0
        ),
        Err(ActualWaveEvidenceUnknown::GraphPath)
    ));
    assert!(matches!(
        actual_route(
            Some(&evidence),
            |_| None,
            DeviceCostGraphCaptureCapability::Unsupported,
            CostProductOutput::GreedyToken,
            0
        ),
        Err(ActualWaveEvidenceUnknown::ProviderPath)
    ));
    assert!(matches!(
        actual_route(
            Some(&evidence),
            provider,
            DeviceCostGraphCaptureCapability::Unknown,
            CostProductOutput::GreedyToken,
            0
        ),
        Err(ActualWaveEvidenceUnknown::GraphPath)
    ));
}

#[test]
fn production_attention_host_binding_gap_does_not_discard_the_wave() {
    // Metal's argument-buffer binding at index 1 runs on the host and reports
    // no compute/transfer dispatch. DeviceSubmissionAttribution intentionally
    // omits that row while preserving the following compute's actual index.
    let upload = DeviceNativeWorkAttribution::new(
        0,
        None,
        DeviceCommandPhase::DynamicBinding,
        DeviceNativeOperationId::new("host.upload").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Scalar,
        0,
        0,
        0,
        1,
        None,
    )
    .unwrap();
    let compute = DeviceNativeWorkAttribution::new(
        2,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("vnext_causal_paged_attention").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Scalar,
        1,
        1,
        9,
        0,
        None,
    )
    .unwrap();
    let evidence = DeviceSubmissionAttribution::new(vec![upload.clone(), compute.clone()]).unwrap();
    let actual = shape(&evidence, 0);
    let mut declared = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    declared
        .physical_command(CostPhysicalCommand::from_attribution(&upload, None))
        .unwrap();
    declared
        .physical_command(CostPhysicalCommand::from_attribution(&compute, provider(0)))
        .unwrap();
    declared.row(row()).unwrap();
    let declared = declared
        .finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            128,
        )
        .unwrap();
    assert_eq!(actual, declared);
    let renumbered = DeviceNativeWorkAttribution::new(
        1,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("vnext_causal_paged_attention").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Scalar,
        1,
        1,
        9,
        0,
        None,
    )
    .unwrap();
    let wrong = DeviceSubmissionAttribution::new(vec![upload, renumbered]).unwrap();
    assert_ne!(
        actual.provider_signature,
        shape(&wrong, 0).provider_signature
    );
}
