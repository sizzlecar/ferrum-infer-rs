use super::*;

fn command(index: u32) -> CostPhysicalCommand<'static> {
    CostPhysicalCommand {
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
