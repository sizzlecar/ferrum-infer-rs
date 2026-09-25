use super::*;
use crate::execution_cost::*;

#[test]
fn graph_capable_runtime_requires_actual_unconfigured_empty_stream() {
    let state = |configuration| DeviceCostGraphStreamState::new(configuration, 0, 0, 0).unwrap();
    let empty = state(DeviceCostGraphConfiguration::Unconfigured);
    assert!(graph_policy_is_eager(
        DeviceCostGraphCaptureCapability::Unsupported,
        None
    ));
    assert!(graph_policy_is_eager(
        DeviceCostGraphCaptureCapability::Supported,
        Some(empty)
    ));
    assert!(!graph_policy_is_eager(
        DeviceCostGraphCaptureCapability::Supported,
        None
    ));
    assert!(!graph_policy_is_eager(
        DeviceCostGraphCaptureCapability::Unknown,
        Some(empty)
    ));
    for configured in [
        DeviceCostGraphConfiguration::StartupPreparing,
        DeviceCostGraphConfiguration::StartupReady,
        DeviceCostGraphConfiguration::OnDemand,
    ] {
        assert!(!graph_policy_is_eager(
            DeviceCostGraphCaptureCapability::Supported,
            Some(state(configured))
        ));
    }
}

fn cuda_like() -> DeviceCoreCostCapabilities {
    DeviceCoreCostCapabilities {
        upload_native_operation: "host.upload",
        zero_native_operation: "device.zero",
        single_transfer_commands: true,
        preserves_program_bindings: false,
        staged_host_readback_without_commands: false,
        staged_host_readback_native_operation: Some("host.submission_readback"),
        fallback_readback: CoreReadbackRoute::HostSynchronized,
    }
}

fn finish(
    mut builder: CanonicalWaveCostBuilder,
    readback: CoreReadbackRoute,
) -> CanonicalWaveCostShape {
    builder.core_readback_route(readback).unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 2 },
            host_policy_signature: [3; 32],
            host_features: None,
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0_f32.to_bits(),
            },
        })
        .unwrap();
    builder
        .finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}

#[test]
fn command_readback_exact_fit_matches_actual_scalar_result_binding_attribution() {
    let mut predicted = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let mut index = 7;
    let route = append_readback_route(
        cuda_like(),
        true,
        2,
        8,
        8,
        &mut |_| None,
        &mut predicted,
        &mut index,
        &mut || true,
    )
    .unwrap();
    assert_eq!(route, CoreReadbackRoute::SubmissionStaged);
    assert_eq!(index, 9);
    let mut actual = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    for index in 7..9 {
        actual
            .physical_command(CostPhysicalCommand {
                statistical_evidence: None,
                native_op_id: "host.submission_readback",
                command_index: index,
                node_index: None,
                command_phase: DeviceCommandPhase::ResultBinding,
                provider: None,
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: 0,
                token_count: 0,
                batching_form: DeviceBatchingForm::Scalar.as_str(),
                compute_dispatch_count: 0,
                transfer_command_count: 1,
                reusable_graph_node_count: None,
            })
            .unwrap();
    }
    assert_eq!(finish(predicted, route), finish(actual, route));
}

#[test]
fn staging_fallback_rolls_back_every_readback_command_and_direct_host_is_distinct() {
    for (attempt, bytes, expected) in [
        (true, 9, CoreReadbackRoute::SubmissionFallbackSynchronized),
        (false, 8, CoreReadbackRoute::HostSynchronized),
    ] {
        let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        let mut index = 7;
        assert_eq!(
            append_readback_route(
                cuda_like(),
                attempt,
                2,
                bytes,
                8,
                &mut |_| None,
                &mut builder,
                &mut index,
                &mut || true
            ),
            Ok(expected)
        );
        assert_eq!(
            index, 7,
            "fallback must not retain partial CUDA staging commands"
        );
    }
    let mut metal = cuda_like();
    metal.staged_host_readback_without_commands = true;
    metal.staged_host_readback_native_operation = None;
    let mut index = 7;
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    assert_eq!(
        append_readback_route(
            metal,
            true,
            2,
            8,
            8,
            &mut |_| panic!("Metal staging has no device transfer"),
            &mut builder,
            &mut index,
            &mut || true
        ),
        Ok(CoreReadbackRoute::SubmissionStaged)
    );
    assert_eq!(index, 7, "shared-memory staging emits no CUDA transfer");
}

#[test]
fn unknown_staging_or_expired_command_budget_never_claims_no_work() {
    let mut unknown = cuda_like();
    unknown.staged_host_readback_native_operation = None;
    let mut index = 7;
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    assert_eq!(
        append_readback_route(
            unknown,
            true,
            1,
            4,
            4,
            &mut |_| None,
            &mut builder,
            &mut index,
            &mut || true
        ),
        Err(U::ReadbackState)
    );
    assert_eq!(
        append_readback_route(
            cuda_like(),
            true,
            1,
            4,
            4,
            &mut |_| None,
            &mut builder,
            &mut index,
            &mut || false
        ),
        Err(U::BudgetExhausted)
    );
    assert_eq!(index, 7);
}

fn readback_evidence(bytes: u64) -> SelectedCommandCostEvidenceV1 {
    let mut selected = SelectedCommandCostBuilderV1::new(0);
    selected
        .transfer(
            SelectedAlgorithmClassV1::new("readback.d2h", 1, [4; 32], [5; 32]).unwrap(),
            StatisticalTransferKindV1::DeviceToHost,
            bytes,
        )
        .unwrap();
    selected.finish().unwrap()
}

fn readback_statistics(mut builder: CanonicalWaveCostBuilder) -> CanonicalStatisticalWave {
    builder
        .core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 2 },
            host_policy_signature: [3; 32],
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [7; 32],
                    decoder_text_bytes_per_token: 8,
                    decoder_scratch_bytes_per_token: 4,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 1,
                    maximum_output_tokens: 8,
                    sampling_history_tokens: 1,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0_f32.to_bits(),
            },
        })
        .unwrap();
    builder
        .finish_with_statistics(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap()
}

#[test]
fn staged_readback_statistics_preserve_each_payload_direction_count_and_missing_producer() {
    let payloads = [4, 18, 1024];
    let selected = payloads.map(readback_evidence);
    let mut predicted = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let mut index = 0;
    let mut seen = Vec::new();
    assert_eq!(
        append_readback_route(
            cuda_like(),
            true,
            3,
            1046,
            1046,
            &mut |i| {
                seen.push(i);
                Some(selected[i].clone())
            },
            &mut predicted,
            &mut index,
            &mut || true
        ),
        Ok(CoreReadbackRoute::SubmissionStaged)
    );
    assert_eq!(seen, [0, 1, 2]);
    assert_eq!(index, 3);
    let predicted = readback_statistics(predicted);
    let work = predicted.statistical.as_ref().unwrap().work();
    assert_eq!(work.device_to_host_bytes, 1046);
    assert_eq!(work.host_to_device_bytes, 0);
    assert_eq!(work.device_to_device_bytes, 0);
    assert_eq!(work.fill_bytes, 0);
    let mut actual = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    for (i, evidence) in selected.iter().enumerate() {
        actual
            .physical_command(CostPhysicalCommand {
                native_op_id: "host.submission_readback",
                command_index: i as u32,
                node_index: None,
                command_phase: DeviceCommandPhase::ResultBinding,
                provider: None,
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: 0,
                token_count: 0,
                batching_form: DeviceBatchingForm::Scalar.as_str(),
                compute_dispatch_count: 0,
                transfer_command_count: 1,
                reusable_graph_node_count: None,
                statistical_evidence: Some(evidence),
            })
            .unwrap();
    }
    let actual = readback_statistics(actual);
    assert_eq!(actual.exact, predicted.exact);
    assert_eq!(actual.statistical, predicted.statistical);
    for missing in 0..3 {
        let mut absent = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        let mut index = 0;
        append_readback_route(
            cuda_like(),
            true,
            3,
            1046,
            1046,
            &mut |i| (i != missing).then(|| selected[i].clone()),
            &mut absent,
            &mut index,
            &mut || true,
        )
        .unwrap();
        let absent = readback_statistics(absent);
        assert_eq!(absent.exact, actual.exact);
        assert_eq!(
            absent.statistical,
            Err(StatisticalEvidenceUnknown::MissingProducer)
        );
    }
}
