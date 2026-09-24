use super::*;
use crate::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, DeviceExecutionPath, DeviceNativeOperationId,
    DeviceNativeWorkAttribution, OperationCostCommand,
};

fn algorithm(entry: &str) -> SelectedAlgorithmClassV1 {
    SelectedAlgorithmClassV1::new(entry, 1, [1; 32], [2; 32]).unwrap()
}
fn evidence(tokens: u64, entry: &str) -> SelectedCommandCostEvidenceV1 {
    let mut builder = SelectedCommandCostBuilderV1::new(tokens);
    builder
        .kernel(
            algorithm(entry),
            KernelNumericWorkV1 {
                logical_units: tokens * 64,
                padded_units: tokens.div_ceil(8) * 8 * 64,
                inner_units_per_logical_unit: 256,
                grid: [1, 4, 1],
                scratch_bytes: 8192,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    builder.finish().unwrap()
}
fn row(count: u32, final_logits: bool) -> CanonicalCostRow {
    let total = if final_logits { count } else { count + 16 };
    CanonicalCostRow {
        work: ActualRowWork::Prefill {
            offset: 0,
            count,
            total_prompt_tokens: total,
        },
        host_policy_signature: [7; 32],
        mask_upload_required: false,
        output: CostRowOutput::Prefill { final_logits },
        host_features: Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [8; 32],
                decoder_text_bytes_per_token: 8,
                decoder_scratch_bytes_per_token: 4,
                raw_token_bytes_bound: 4,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 0,
                maximum_output_tokens: 32,
                sampling_history_tokens: 0,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
    }
}
fn provider() -> CostProviderIdentity<'static> {
    CostProviderIdentity {
        provider_id: "actual.backend",
        implementation_fingerprint: "impl1",
        operation_fingerprint: "op1",
    }
}
fn command(tokens: u64) -> OperationCostCommand {
    OperationCostCommand::new(
        "native.selected",
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        1,
        tokens,
        1,
        0,
    )
    .unwrap()
}
fn finish(
    commands: &[CostPhysicalCommand<'_>],
    count: u32,
    final_logits: bool,
    product: CostProductOutput,
) -> CanonicalStatisticalWave {
    let mut builder = CanonicalWaveCostBuilder::new(0, product);
    for &command in commands {
        builder.physical_command(command).unwrap();
    }
    builder
        .core_readback_route(CoreReadbackRoute::NoReadback)
        .unwrap();
    builder.row(row(count, final_logits)).unwrap();
    builder
        .finish_with_statistics(
            ActualWaveKind::Prefill,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            32,
        )
        .unwrap()
}
fn projected(
    tokens: u32,
    entry: &str,
    final_logits: bool,
    product: CostProductOutput,
) -> CanonicalStatisticalWave {
    let cmd = command(tokens.into())
        .with_statistical_evidence(evidence(tokens.into(), entry))
        .unwrap();
    finish(
        &[cmd.canonical_command(0, 0, provider()).unwrap()],
        tokens,
        final_logits,
        product,
    )
}

#[test]
fn same_selected_algorithm_groups_work_without_rewriting_exact_identity() {
    let a = projected(8, "mma.m8", false, CostProductOutput::GreedyToken);
    let b = projected(16, "mma.m8", false, CostProductOutput::GreedyToken);
    let sa = a.statistical.unwrap();
    let sb = b.statistical.unwrap();
    assert_ne!(a.exact.provider_signature, b.exact.provider_signature);
    assert_eq!(sa.family_signature(), sb.family_signature());
    assert_eq!(sb.work().logical_units, sa.work().logical_units * 2);
    assert!(sa.validate_exact(&a.exact).is_ok());
    assert_eq!(
        sa.validate_exact(&b.exact),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
    let changed = projected(8, "mma.m32", false, CostProductOutput::GreedyToken);
    assert_ne!(
        sa.family_signature(),
        changed.statistical.unwrap().family_signature()
    );
    let final_row = projected(8, "mma.m8", true, CostProductOutput::GreedyToken);
    let full = projected(8, "mma.m8", false, CostProductOutput::FullLogits);
    assert_ne!(
        sa.family_signature(),
        final_row.statistical.unwrap().family_signature()
    );
    assert_ne!(
        sa.family_signature(),
        full.statistical.unwrap().family_signature()
    );
}

#[test]
fn attribution_and_projected_selected_command_share_one_evidence_contract() {
    let selected = evidence(8, "mma.m8");
    let actual = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("native.selected").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        1,
        8,
        1,
        0,
        None,
    )
    .unwrap()
    .with_statistical_evidence(selected.clone())
    .unwrap();
    let predicted = command(8).with_statistical_evidence(selected).unwrap();
    let without_statistics = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("native.selected").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        1,
        8,
        1,
        0,
        None,
    )
    .unwrap();
    assert_eq!(
        actual, without_statistics,
        "legacy exact equality excludes passive sidecar"
    );
    assert!(actual.statistical_evidence().is_some());
    assert!(without_statistics.statistical_evidence().is_none());
    let json = serde_json::to_value(&without_statistics).unwrap();
    assert!(
        json.get("statistical_evidence").is_none(),
        "legacy None wire remains unchanged"
    );
    let actual = finish(
        &[CostPhysicalCommand::from_attribution(
            &actual,
            Some(provider()),
        )],
        8,
        false,
        CostProductOutput::GreedyToken,
    );
    let future = finish(
        &[predicted.canonical_command(0, 0, provider()).unwrap()],
        8,
        false,
        CostProductOutput::GreedyToken,
    );
    assert_eq!(actual.exact, future.exact);
    assert_eq!(actual.statistical, future.statistical);
    assert_eq!(
        command(9).with_statistical_evidence(evidence(8, "mma.m8")),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
}

#[test]
fn incomplete_statistics_never_change_legacy_exact_shape() {
    let known = command(8)
        .with_statistical_evidence(evidence(8, "mma.m8"))
        .unwrap();
    let missing = command(8);
    let a = finish(
        &[known.canonical_command(0, 0, provider()).unwrap()],
        8,
        false,
        CostProductOutput::GreedyToken,
    );
    let b = finish(
        &[missing.canonical_command(0, 0, provider()).unwrap()],
        8,
        false,
        CostProductOutput::GreedyToken,
    );
    assert_eq!(a.exact, b.exact);
    assert_eq!(
        b.statistical,
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
    let incomplete = finish(
        &[
            known.canonical_command(0, 0, provider()).unwrap(),
            missing.canonical_command(1, 1, provider()).unwrap(),
        ],
        8,
        false,
        CostProductOutput::GreedyToken,
    );
    assert_eq!(
        incomplete.statistical,
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
}

#[test]
fn count_mismatch_and_capacity_fail_closed_without_invalidating_exact_work() {
    let selected = command(8)
        .with_statistical_evidence(evidence(8, "mma.m8"))
        .unwrap();
    let mut physical = selected.canonical_command(0, 0, provider()).unwrap();
    physical.compute_dispatch_count = 2;
    let result = finish(&[physical], 8, false, CostProductOutput::GreedyToken);
    assert_eq!(
        result.statistical,
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
    let mut builder = SelectedCommandCostBuilderV1::new(8);
    for _ in 0..MAX_COST_COMMANDS {
        builder
            .transfer(
                algorithm("upload"),
                StatisticalTransferKindV1::HostToDevice,
                4,
            )
            .unwrap();
    }
    assert_eq!(
        builder.transfer(
            algorithm("upload"),
            StatisticalTransferKindV1::HostToDevice,
            4
        ),
        Err(StatisticalEvidenceUnknown::Capacity)
    );
    assert_eq!(builder.finish(), Err(StatisticalEvidenceUnknown::Capacity));
    let mut overflow = SelectedCommandCostBuilderV1::new(8);
    overflow
        .transfer(
            algorithm("upload"),
            StatisticalTransferKindV1::HostToDevice,
            u64::MAX,
        )
        .unwrap();
    assert_eq!(
        overflow.transfer(
            algorithm("upload"),
            StatisticalTransferKindV1::HostToDevice,
            1
        ),
        Err(StatisticalEvidenceUnknown::Overflow)
    );
    assert_eq!(overflow.finish(), Err(StatisticalEvidenceUnknown::Overflow));
}

#[test]
fn numerical_and_layout_contracts_are_part_of_algorithm_identity() {
    let original = algorithm("mma.m8");
    assert_ne!(
        original,
        SelectedAlgorithmClassV1::new("mma.m8", 1, [3; 32], [2; 32]).unwrap()
    );
    assert_ne!(
        original,
        SelectedAlgorithmClassV1::new("mma.m8", 1, [1; 32], [3; 32]).unwrap()
    );
    assert!(SelectedAlgorithmClassV1::new("", 1, [1; 32], [2; 32]).is_err());
    assert!(SelectedAlgorithmClassV1::new("mma.m8", 1, [0; 32], [2; 32]).is_err());
    assert_eq!(
        SelectedCommandCostBuilderV1::new(8).finish(),
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
}
