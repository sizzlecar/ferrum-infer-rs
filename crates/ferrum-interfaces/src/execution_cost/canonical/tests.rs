use super::*;

fn command(index: u32) -> CostPhysicalCommand<'static> {
    CostPhysicalCommand {
        statistical_evidence: None,
        native_op_id: "test.compute",
        command_index: index,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "test.provider",
            implementation_fingerprint: "implementation-a",
            operation_fingerprint: "operation-a",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 2,
        token_count: 5,
        batching_form: "unified",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
    }
}
fn row(work: ActualRowWork) -> CanonicalCostRow {
    CanonicalCostRow {
        work,
        host_policy_signature: host_history_cost_signature([9; 32], 1),
        host_features: None,
        mask_upload_required: true,
        output: match work {
            ActualRowWork::Decode { .. } => CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1.0f32.to_bits(),
            },
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => CostRowOutput::Prefill {
                final_logits: offset + count == total_prompt_tokens,
            },
            _ => unreachable!(),
        },
    }
}

fn original_replay_builder() -> CanonicalWaveCostBuilder {
    let mut builder = CanonicalWaveCostBuilder::new_exact(0, CostProductOutput::FullLogits);
    for index in 0..2 {
        let mut original = command(index);
        original.path = CostCommandPath::Replayed;
        builder
            .original_replay_command(
                original,
                DeviceNativeOperationId::new("test.direct").unwrap(),
            )
            .unwrap();
    }
    builder
        .core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    builder
        .row(row(ActualRowWork::Decode { kv_tokens: 8 }))
        .unwrap();
    builder
}

#[test]
fn original_replay_physical_work_never_finishes_as_numerical_evidence() {
    for finish_kind in 0..4 {
        let builder = original_replay_builder();
        assert_eq!(
            builder.validate_physical_structure(ActualWaveKind::Decode),
            Ok(())
        );
        let result = match finish_kind {
            0 => builder
                .finish(
                    ActualWaveKind::Decode,
                    ActualWavePath::PlanRuntime,
                    ActualWaveGraphState::Warm,
                    ActualWaveRowOrder::Ordered,
                    0,
                )
                .map(|_| ()),
            1 => builder
                .finish_with_statistics(
                    ActualWaveKind::Decode,
                    ActualWavePath::PlanRuntime,
                    ActualWaveGraphState::Warm,
                    ActualWaveRowOrder::Ordered,
                    0,
                )
                .map(|_| ()),
            2 => builder
                .finish_with_structure(
                    ActualWaveKind::Decode,
                    ActualWavePath::PlanRuntime,
                    ActualWaveGraphState::Warm,
                    ActualWaveRowOrder::Ordered,
                    0,
                )
                .map(|_| ()),
            _ => builder
                .finish_with_captured_structure(
                    ActualWaveKind::Decode,
                    ActualWavePath::PlanRuntime,
                    ActualWaveGraphState::Warm,
                    ActualWaveRowOrder::Ordered,
                    0,
                )
                .map(|_| ()),
        };
        assert_eq!(result, Err(CanonicalCostError::InvalidRoute));
    }
}

#[test]
fn original_replay_keeps_command_checks_and_compact_expansion_strict() {
    let direct = DeviceNativeOperationId::new("test.direct").unwrap();
    let mut original = command(0);
    original.path = CostCommandPath::Replayed;
    let mut direct_command = original;
    direct_command.native_op_id = direct.as_str();
    let mut wrong_phase = original;
    wrong_phase.command_phase = DeviceCommandPhase::DynamicBinding;
    let mut missing_provider = original;
    missing_provider.provider = None;
    let mut invalid_provider = original;
    invalid_provider.provider.as_mut().unwrap().provider_id = "";
    let mut empty_work = original;
    empty_work.compute_dispatch_count = 0;
    empty_work.transfer_command_count = 0;
    let mut overflow_participants = original;
    overflow_participants.participant_start = u32::MAX;
    for invalid in [
        direct_command,
        wrong_phase,
        missing_provider,
        invalid_provider,
        empty_work,
        overflow_participants,
    ] {
        let mut builder = CanonicalWaveCostBuilder::new_exact(0, CostProductOutput::FullLogits);
        assert!(builder.original_replay_command(invalid, direct).is_err());
        assert!(
            builder.physical_command(command(1)).is_err(),
            "failure remains sticky"
        );
    }

    let mut mixed = CanonicalWaveCostBuilder::new_exact(0, CostProductOutput::FullLogits);
    mixed.original_replay_command(original, direct).unwrap();
    direct_command.command_index = 1;
    direct_command.reusable_graph_node_count = Some(1);
    mixed.physical_command(direct_command).unwrap();
    mixed
        .row(row(ActualRowWork::Decode { kv_tokens: 8 }))
        .unwrap();
    assert_eq!(
        mixed.validate_physical_structure(ActualWaveKind::Decode),
        Err(CanonicalCostError::InvalidRoute)
    );
    mixed
        .replay_segment(1, "actual-sealed-executable", 1)
        .unwrap();
    mixed
        .logical_command(CostLogicalCommand {
            native_op_id: "test.compute",
            logical_command_ordinal: 0,
            node_index: 0,
            provider: original.provider.unwrap(),
            participant_count: original.participant_count,
            token_count: original.token_count,
            batching_form: original.batching_form,
            compute_dispatch_count: original.compute_dispatch_count,
            transfer_command_count: original.transfer_command_count,
            reusable_graph_node_count: 1,
            statistical_evidence: None,
        })
        .unwrap();
    assert_eq!(
        mixed.validate_physical_structure(ActualWaveKind::Decode),
        Ok(())
    );
    assert_eq!(
        mixed.finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Warm,
            ActualWaveRowOrder::Ordered,
            0
        ),
        Err(CanonicalCostError::InvalidRoute)
    );
}
fn finish(
    builder: CanonicalWaveCostBuilder,
    kind: ActualWaveKind,
) -> Result<CanonicalWaveCostShape, CanonicalCostError> {
    builder.finish(
        kind,
        ActualWavePath::PlanRuntime,
        ActualWaveGraphState::Disabled,
        ActualWaveRowOrder::Ordered,
        128,
    )
}
fn mixed(reverse: bool) -> CanonicalWaveCostShape {
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    builder.physical_command(command(0)).unwrap();
    let mut rows = [
        row(ActualRowWork::Decode { kv_tokens: 12 }),
        row(ActualRowWork::Prefill {
            offset: 0,
            count: 4,
            total_prompt_tokens: 8,
        }),
    ];
    if reverse {
        rows.reverse();
    }
    for row in rows {
        builder.row(row).unwrap();
    }
    finish(builder, ActualWaveKind::Mixed).unwrap()
}
#[test]
fn ordered_mixed_roles_survive_separate_scheduler_vectors() {
    let first = mixed(false);
    let second = mixed(true);
    assert_ne!(first.provider_signature, second.provider_signature);
    assert_ne!(
        first.output_policy_signature,
        second.output_policy_signature
    );
    assert_eq!(first, mixed(false));
}
#[test]
fn actual_core_transfer_with_no_logical_rows_remains_evidence() {
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
    let mut transfer = command(0);
    transfer.provider = None;
    transfer.node_index = None;
    transfer.command_phase = DeviceCommandPhase::DynamicBinding;
    transfer.compute_dispatch_count = 0;
    transfer.transfer_command_count = 1;
    transfer.participant_count = 0;
    transfer.token_count = 0;
    builder.physical_command(transfer).unwrap();
    builder
        .row(row(ActualRowWork::Decode { kv_tokens: 3 }))
        .unwrap();
    assert!(finish(builder, ActualWaveKind::Decode).is_ok());
}
#[test]
fn invalid_command_poisoning_and_hard_capacity_are_sticky() {
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
    builder.physical_command(command(1)).unwrap();
    assert_eq!(
        builder.physical_command(command(0)),
        Err(CanonicalCostError::InvalidCommand)
    );
    assert_eq!(
        builder.physical_command(command(2)),
        Err(CanonicalCostError::InvalidCommand)
    );
    assert_eq!(
        finish(builder, ActualWaveKind::Decode),
        Err(CanonicalCostError::InvalidCommand)
    );
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::FullLogits);
    for index in 0..MAX_COST_COMMANDS {
        builder.physical_command(command(index as u32)).unwrap();
    }
    assert_eq!(
        builder.physical_command(command(MAX_COST_COMMANDS as u32)),
        Err(CanonicalCostError::Capacity)
    );
}
#[test]
fn incomplete_replay_and_mismatched_final_output_are_rejected() {
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let mut replay = command(0);
    replay.path = CostCommandPath::Replayed;
    builder.physical_command(replay).unwrap();
    builder
        .row(row(ActualRowWork::Decode { kv_tokens: 3 }))
        .unwrap();
    assert_eq!(
        finish(builder, ActualWaveKind::Decode),
        Err(CanonicalCostError::InvalidRoute)
    );
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let mut invalid = row(ActualRowWork::Prefill {
        offset: 0,
        count: 4,
        total_prompt_tokens: 8,
    });
    invalid.output = CostRowOutput::Prefill { final_logits: true };
    assert_eq!(builder.row(invalid), Err(CanonicalCostError::InvalidRow));
}
#[test]
fn provider_output_and_generated_history_are_independent_dimensions() {
    let shape = |implementation, history, mask| {
        let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        let mut command = command(0);
        command
            .provider
            .as_mut()
            .unwrap()
            .implementation_fingerprint = implementation;
        builder.physical_command(command).unwrap();
        let mut row = row(ActualRowWork::Decode { kv_tokens: 12 });
        row.host_policy_signature = host_history_cost_signature([9; 32], history);
        row.mask_upload_required = mask;
        builder.row(row).unwrap();
        finish(builder, ActualWaveKind::Decode).unwrap()
    };
    let baseline = shape("implementation-a", 1, true);
    assert_ne!(
        baseline.provider_signature,
        shape("implementation-b", 1, true).provider_signature
    );
    assert_ne!(
        baseline.output_policy_signature,
        shape("implementation-a", 2, true).output_policy_signature
    );
    assert_ne!(
        baseline.output_policy_signature,
        shape("implementation-a", 1, false).output_policy_signature
    );
}

fn logical(nodes: u64) -> CostLogicalCommand<'static> {
    CostLogicalCommand {
        native_op_id: "test.compute",
        logical_command_ordinal: 0,
        node_index: 0,
        provider: command(0).provider.unwrap(),
        participant_count: 2,
        token_count: 5,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: nodes,
        statistical_evidence: None,
    }
}
fn replay(nodes: u64) -> CanonicalWaveCostBuilder {
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    let mut physical = command(0);
    physical.path = CostCommandPath::Replayed;
    physical.reusable_graph_node_count = Some(nodes);
    builder.physical_command(physical).unwrap();
    builder.replay_segment(0, "sealed-executable", 1).unwrap();
    builder
}

#[test]
fn replay_node_counts_are_keyed_and_must_match_the_complete_logical_segment() {
    let shape = |nodes| {
        let mut builder = replay(nodes);
        builder.logical_command(logical(nodes)).unwrap();
        builder
            .row(row(ActualRowWork::Decode { kv_tokens: 12 }))
            .unwrap();
        builder
            .finish(
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Warm,
                ActualWaveRowOrder::Ordered,
                128,
            )
            .unwrap()
    };
    assert_ne!(shape(3).provider_signature, shape(4).provider_signature);
    let mut mismatch = replay(3);
    assert_eq!(
        mismatch.logical_command(logical(4)),
        Err(CanonicalCostError::EvidenceMismatch)
    );
    let mut bad_ordinal = replay(3);
    let mut invalid = logical(3);
    invalid.logical_command_ordinal = 1;
    assert_eq!(
        bad_ordinal.logical_command(invalid),
        Err(CanonicalCostError::InvalidCommand)
    );
}

#[test]
fn lossless_attribution_projection_retains_every_physical_counter() {
    use crate::vnext::{DeviceBatchingForm, DeviceNativeOperationId};
    let evidence = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("test.compute").unwrap(),
        DeviceExecutionPath::Replayed,
        DeviceBatchingForm::Packed,
        2,
        5,
        2,
        3,
        Some(7),
    )
    .unwrap();
    let projected = CostPhysicalCommand::from_attribution(&evidence, command(0).provider);
    assert_eq!(projected.compute_dispatch_count, 2);
    assert_eq!(projected.transfer_command_count, 3);
    assert_eq!(projected.reusable_graph_node_count, Some(7));
    assert_eq!(projected.node_index, Some(0));
    assert_eq!(projected.command_phase, DeviceCommandPhase::Compute);
    let logical = DeviceReplayedLogicalCommandAttribution::new(
        0,
        0,
        DeviceNativeOperationId::new("test.compute").unwrap(),
        DeviceBatchingForm::Packed,
        2,
        5,
        2,
        3,
        7,
    )
    .unwrap();
    let projected = CostLogicalCommand::from_attribution(&logical, command(0).provider.unwrap());
    assert_eq!(projected.compute_dispatch_count, 2);
    assert_eq!(projected.transfer_command_count, 3);
    assert_eq!(projected.reusable_graph_node_count, 7);
    assert_eq!(projected.batching_form, "packed");
    assert_eq!(projected.logical_command_ordinal, 0);
}

#[test]
fn physical_index_gaps_survive_and_replay_binds_the_actual_index() {
    let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    builder.physical_command(command(2)).unwrap();
    let mut physical = command(7);
    physical.path = CostCommandPath::Replayed;
    physical.reusable_graph_node_count = Some(3);
    builder.physical_command(physical).unwrap();
    // 7 is a physical command index, not an index into the two evidence rows.
    builder.replay_segment(7, "sealed-executable", 1).unwrap();
    builder.logical_command(logical(3)).unwrap();
    builder
        .row(row(ActualRowWork::Decode { kv_tokens: 12 }))
        .unwrap();
    assert!(builder
        .finish(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Warm,
            ActualWaveRowOrder::Ordered,
            128
        )
        .is_ok());
    let mut invalid = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    invalid.physical_command(physical).unwrap();
    assert_eq!(
        invalid.replay_segment(0, "sealed-executable", 1),
        Err(CanonicalCostError::InvalidRoute)
    );
}
