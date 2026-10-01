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

fn captured_attribution(tokens: u64, statistics: bool) -> DeviceSubmissionAttribution {
    let mut command = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("fixture.actual-capture").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        1,
        tokens,
        1,
        0,
        None,
    )
    .unwrap();
    if statistics {
        let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(tokens);
        selected
            .kernel(
                SelectedAlgorithmClassV1::new("fixture.actual-capture", 1, [5; 32], [6; 32])
                    .unwrap(),
                KernelNumericWorkV1 {
                    logical_units: tokens * 64,
                    padded_units: tokens * 64,
                    inner_units_per_logical_unit: 1,
                    grid: [1, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
        command = command
            .with_statistical_evidence(selected.finish().unwrap())
            .unwrap();
    }
    DeviceSubmissionAttribution::new(vec![command]).unwrap()
}

fn project_actual_capture(
    attribution: &DeviceSubmissionAttribution,
    row: CanonicalCostRow,
    product: CostProductOutput,
    readback: CoreReadbackRoute,
    capture: bool,
) -> CanonicalStatisticalWave {
    // This is the production adapter used by actual_shape_from_device_with_capture,
    // including the real canonical builder's optional structured finish path.
    let mut observed = actual_route_with_capture(
        Some(attribution),
        provider,
        DeviceCostGraphCaptureCapability::Unsupported,
        product,
        0,
        capture,
    )
    .unwrap();
    observed.canonical.core_readback_route(readback).unwrap();
    observed.canonical.row(row).unwrap();
    observed
        .canonical
        .finish_with_captured_structure(
            match row.work {
                ActualRowWork::Prefill { .. } => ActualWaveKind::Prefill,
                ActualRowWork::Decode { .. } => ActualWaveKind::Decode,
                _ => unreachable!("inference-only cases"),
            },
            ActualWavePath::PlanRuntime,
            observed.graph,
            ActualWaveRowOrder::Ordered,
            128,
        )
        .unwrap()
}

#[test]
fn production_structured_actual_capture_preserves_legal_exact_routes_and_missing_sidecars() {
    let token_readbacks = [
        CoreReadbackRoute::SubmissionStaged,
        CoreReadbackRoute::HostSynchronized,
        CoreReadbackRoute::SubmissionFallbackSynchronized,
    ];
    let no_readback = [CoreReadbackRoute::NoReadback];
    // Product prefill always requests FullLogits; its body produces no token
    // and has no readback. Final prefill and decode can have the three real
    // readback outcomes. GreedyToken is a decode product, not a prefill guess.
    let cases = [
        (
            ActualRowWork::Prefill {
                offset: 0,
                count: 4,
                total_prompt_tokens: 8,
            },
            CostRowOutput::Prefill {
                final_logits: false,
            },
            CostProductOutput::FullLogits,
            0,
            no_readback.as_slice(),
        ),
        (
            ActualRowWork::Prefill {
                offset: 4,
                count: 4,
                total_prompt_tokens: 8,
            },
            CostRowOutput::Prefill { final_logits: true },
            CostProductOutput::FullLogits,
            0,
            token_readbacks.as_slice(),
        ),
        (
            ActualRowWork::Decode { kv_tokens: 12 },
            CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0f32.to_bits(),
            },
            CostProductOutput::FullLogits,
            1,
            token_readbacks.as_slice(),
        ),
        (
            ActualRowWork::Decode { kv_tokens: 12 },
            CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0f32.to_bits(),
            },
            CostProductOutput::GreedyToken,
            1,
            token_readbacks.as_slice(),
        ),
    ];
    for (work, output, product, generated, readbacks) in cases {
        for &readback in readbacks {
            // Exercise both independently missing inputs. Neither one is
            // permission to erase a valid exact observation or invent a sidecar.
            for (statistics, host_domain) in [(true, true), (false, true), (true, false)] {
                let row = CanonicalCostRow {
                    work,
                    output,
                    host_policy_signature: host_history_cost_signature([3; 32], generated),
                    mask_upload_required: false,
                    host_features: Some(HostCostFeaturesV1 {
                        policy: HostCostPolicyV2 {
                            empirical_content_domain: host_domain
                                .then_some(HostContentDomainV1::PlainTextGreedyV1),
                            categorical_signature: [4; 32],
                            decoder_text_bytes_per_token: 4,
                            decoder_scratch_bytes_per_token: 8,
                            raw_token_bytes_bound: 4,
                        },
                        state: HostCostStateV1 {
                            generated_tokens_before: generated,
                            maximum_output_tokens: 8,
                            sampling_history_tokens: generated,
                            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                            pending_decoded_utf8: false,
                            completion_state_signature: satisfied_completion_cost_signature(),
                        },
                    }),
                };
                let tokens = match work {
                    ActualRowWork::Prefill { count, .. } => u64::from(count),
                    _ => 1,
                };
                let attribution = captured_attribution(tokens, statistics);
                let baseline = project_actual_capture(&attribution, row, product, readback, false);
                let demand = StructuredCostSampleDemand::for_call(
                    ferrum_types::SloStructuredActualCapturePolicy::ConsumerDrivenV1,
                    true,
                    false,
                );
                let captured = project_actual_capture(
                    &attribution,
                    row,
                    product,
                    readback,
                    demand.enabled(ferrum_types::SloStructuredCostCapture::HostSettledV1),
                );
                assert_eq!(captured.exact, baseline.exact,
                    "{work:?}/{product:?}/{readback:?}, statistics={statistics}, host_domain={host_domain}");
                assert_eq!(captured.exact.rows, vec![work]);
                assert_eq!(captured.exact.row_order, ActualWaveRowOrder::Ordered);
                let numeric = captured.exact.numeric_features.as_ref().unwrap();
                numeric.validate(1).unwrap();
                assert_eq!(
                    numeric.rows[0],
                    project_host_cost_features(row.host_features.unwrap(), work, output).unwrap()
                );
                if !statistics {
                    assert!(matches!(
                        captured.statistical,
                        Err(StatisticalEvidenceUnknown::MissingProducer)
                    ));
                    continue;
                }
                if !host_domain {
                    assert!(matches!(
                        captured.statistical,
                        Err(StatisticalEvidenceUnknown::MissingHostDomain)
                    ));
                    continue;
                }
                let selected = captured.statistical.unwrap();
                selected.validate_exact(&captured.exact).unwrap();
                let structured = selected
                    .structured_capture()
                    .expect("requested dynamic sidecar");
                let structured = structured.unwrap();
                structured.validate_exact(&captured.exact).unwrap();
                structured.algorithm_work().unwrap();
                let host = &structured.physical_host_rows()[0];
                assert_eq!(structured.physical_host_rows().len(), 1);
                assert_eq!(host.physical_position, 0);
                assert_eq!(host.installed_policy, row.host_features.unwrap().policy);
                assert_eq!(
                    host.terminal_expectation,
                    if matches!(
                        output,
                        CostRowOutput::Prefill {
                            final_logits: false
                        }
                    ) {
                        HostTerminalExpectationV1::NoTokenProduced
                    } else {
                        HostTerminalExpectationV1::TokenMayTerminate
                    }
                );
            }
        }
    }
}
