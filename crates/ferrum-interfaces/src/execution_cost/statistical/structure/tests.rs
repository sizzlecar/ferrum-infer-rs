use super::*;
use crate::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, DeviceExecutionPath, DeviceNativeOperationId,
    DeviceNativeWorkAttribution, OperationCostCommand,
};

fn algorithm(name: &str) -> SelectedAlgorithmClassV1 {
    SelectedAlgorithmClassV1::new(name, 1, [1; 32], [2; 32]).unwrap()
}
fn selected(rows: u32, name: &str) -> SelectedCommandCostEvidenceV1 {
    let mut builder = SelectedCommandCostBuilderV1::new(rows.into());
    builder
        .kernel(
            algorithm(name),
            KernelNumericWorkV1 {
                logical_units: rows.into(),
                padded_units: rows.into(),
                inner_units_per_logical_unit: 64,
                grid: [rows, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    builder.finish().unwrap()
}
fn command(rows: u32, name: &str) -> OperationCostCommand {
    OperationCostCommand::new(
        "native.structured-test",
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        rows,
        rows.into(),
        1,
        0,
    )
    .unwrap()
    .with_statistical_evidence(selected(rows, name))
    .unwrap()
}
fn provider() -> CostProviderIdentity<'static> {
    CostProviderIdentity {
        provider_id: "test.provider",
        implementation_fingerprint: "impl",
        operation_fingerprint: "op",
    }
}
fn row(length_boundary: bool) -> CanonicalCostRow {
    CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 64 },
        host_policy_signature: host_history_cost_signature([7; 32], 4),
        host_features: Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [8; 32],
                decoder_text_bytes_per_token: 8,
                decoder_scratch_bytes_per_token: 4,
                raw_token_bytes_bound: 4,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 4,
                maximum_output_tokens: if length_boundary { 5 } else { 20 },
                sampling_history_tokens: 4,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
        mask_upload_required: false,
        output: CostRowOutput::Decode {
            requires_full_logits: false,
            repetition_tokens: 4,
            repetition_penalty_bits: 1f32.to_bits(),
        },
    }
}
fn builder(
    commands: &[CostPhysicalCommand<'_>],
    rows: &[CanonicalCostRow],
    product: CostProductOutput,
    readback: CoreReadbackRoute,
    retries: u32,
    opted_in: bool,
) -> CanonicalWaveCostBuilder {
    let mut b = if opted_in {
        CanonicalWaveCostBuilder::new_with_structured_statistics(retries, product)
    } else {
        CanonicalWaveCostBuilder::new(retries, product)
    };
    for command in commands {
        b.physical_command(*command).unwrap();
    }
    b.core_readback_route(readback).unwrap();
    for row in rows {
        b.row(*row).unwrap();
    }
    b
}
fn finish(b: CanonicalWaveCostBuilder) -> CanonicalStructuredWave {
    b.finish_with_structure(
        ActualWaveKind::Decode,
        ActualWavePath::PlanRuntime,
        ActualWaveGraphState::Disabled,
        ActualWaveRowOrder::Ordered,
        64,
    )
    .unwrap()
}

#[test]
fn structured_recipe_actual_attribution_and_future_command_match_without_changing_old_evidence() {
    let actual = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("native.structured-test").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        2,
        2,
        1,
        0,
        None,
    )
    .unwrap()
    .with_statistical_evidence(selected(2, "selected.kernel"))
    .unwrap();
    let future = command(2, "selected.kernel");
    let rows = [row(true), row(false)];
    let a = finish(builder(
        &[CostPhysicalCommand::from_attribution(
            &actual,
            Some(provider()),
        )],
        &rows,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::HostSynchronized,
        0,
        true,
    ));
    let cmd = [future.canonical_command(0, 0, provider()).unwrap()];
    let b = finish(builder(
        &cmd,
        &rows,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::HostSynchronized,
        0,
        true,
    ));
    assert_eq!(a.exact, b.exact);
    assert_eq!(a.structured, b.structured);
    let old = builder(
        &cmd,
        &rows,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::HostSynchronized,
        0,
        false,
    )
    .finish_with_statistics(
        ActualWaveKind::Decode,
        ActualWavePath::PlanRuntime,
        ActualWaveGraphState::Disabled,
        ActualWaveRowOrder::Ordered,
        64,
    )
    .unwrap();
    assert_eq!(old.exact, b.exact);
    let old = old.statistical.unwrap();
    let new = b.statistical.unwrap();
    assert_eq!(old, new);
    assert_eq!(
        old.independent_attention_v2(),
        new.independent_attention_v2()
    );
    assert_eq!(
        serde_json::to_vec(&old.to_wire_v1()).unwrap(),
        serde_json::to_vec(&new.to_wire_v1()).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&old).unwrap(),
        serde_json::to_vec(&new).unwrap()
    );
    assert_eq!(
        b.structured.unwrap().physical_host_rows()[0].physical_position,
        0
    );
}

#[test]
fn structured_recipe_terminal_position_is_host_structure_not_device_route_or_cost_equivalence() {
    let op = command(2, "selected.kernel");
    let commands = [op.canonical_command(0, 0, provider()).unwrap()];
    let a = finish(builder(
        &commands,
        &[row(true), row(false)],
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::HostSynchronized,
        0,
        true,
    ));
    let b = finish(builder(
        &commands,
        &[row(false), row(true)],
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::HostSynchronized,
        0,
        true,
    ));
    assert_ne!(a.exact, b.exact);
    let sa = a.structured.unwrap();
    let sb = b.structured.unwrap();
    assert_eq!(sa.device(), sb.device());
    assert_ne!(sa.physical_host_rows(), sb.physical_host_rows());
    assert_eq!(
        sa.physical_host_rows()[0].terminal_expectation,
        HostTerminalExpectationV1::LengthBoundary
    );
    assert_eq!(
        sa.physical_host_rows()[1].terminal_expectation,
        HostTerminalExpectationV1::TokenMayTerminate
    );
    assert!(sa.validate_exact(&a.exact).is_ok());
    assert_eq!(
        sa.validate_exact(&b.exact),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
    let mut tampered = a.exact;
    tampered.numeric_features.as_mut().unwrap().rows[0].maximum_output_tokens += 1;
    assert_eq!(
        sa.validate_exact(&tampered),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
    // No fitter/settle conversion exists: equality of device recipes does not
    // claim equal total costs or exclude EOS for TokenMayTerminate.
}

#[test]
fn structured_recipe_product_readback_retries_provider_and_algorithm_keep_separate_domains() {
    let a = command(2, "selected.kernel.a");
    let b = command(2, "selected.kernel.b");
    let base_cmd = [a.canonical_command(0, 0, provider()).unwrap()];
    let run = |cmd: &[CostPhysicalCommand<'_>], product, readback, retries| {
        finish(builder(
            cmd,
            &[row(true), row(false)],
            product,
            readback,
            retries,
            true,
        ))
        .structured
        .unwrap()
        .device()
        .ordered_template()
        .to_owned()
    };
    let base = run(
        &base_cmd,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::HostSynchronized,
        0,
    );
    assert_ne!(
        base,
        run(
            &base_cmd,
            CostProductOutput::FullLogits,
            CoreReadbackRoute::HostSynchronized,
            0
        )
    );
    assert_ne!(
        base,
        run(
            &base_cmd,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::NoReadback,
            0
        )
    );
    assert_ne!(
        base,
        run(
            &base_cmd,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::SubmissionFallbackSynchronized,
            0
        )
    );
    assert_ne!(
        base,
        run(
            &base_cmd,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::HostSynchronized,
            1
        )
    );
    assert_ne!(
        base,
        run(
            &[b.canonical_command(0, 0, provider()).unwrap()],
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::HostSynchronized,
            0
        )
    );
    let mut changed = base_cmd;
    changed[0].provider = Some(CostProviderIdentity {
        implementation_fingerprint: "changed",
        ..provider()
    });
    assert_ne!(
        base,
        run(
            &changed,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::HostSynchronized,
            0
        )
    );
}

#[test]
fn structured_recipe_unknown_or_mismatched_selected_work_never_becomes_a_complete_template() {
    let op = command(2, "selected.kernel");
    let cmd = op.canonical_command(0, 0, provider()).unwrap();
    let rows = [row(true), row(false)];
    let mut missing = cmd;
    missing.statistical_evidence = None;
    let a = finish(builder(
        &[cmd],
        &rows,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::NoReadback,
        0,
        true,
    ));
    let b = finish(builder(
        &[missing],
        &rows,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::NoReadback,
        0,
        true,
    ));
    assert_eq!(a.exact, b.exact);
    assert_eq!(
        b.structured,
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
    let mut mismatch = cmd;
    mismatch.compute_dispatch_count += 1;
    assert_eq!(
        finish(builder(
            &[mismatch],
            &rows,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::NoReadback,
            0,
            true
        ))
        .structured,
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
    assert_eq!(
        finish(builder(
            &[cmd],
            &rows,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::NoReadback,
            0,
            false
        ))
        .structured,
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
    assert_eq!(
        finish(builder(
            &[cmd],
            &rows,
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::Unknown,
            0,
            true
        ))
        .structured,
        Err(StatisticalEvidenceUnknown::MissingHostDomain)
    );
    let graph = builder(
        &[cmd],
        &rows,
        CostProductOutput::GreedyToken,
        CoreReadbackRoute::NoReadback,
        0,
        true,
    )
    .finish_with_structure(
        ActualWaveKind::Decode,
        ActualWavePath::PlanRuntime,
        ActualWaveGraphState::Warm,
        ActualWaveRowOrder::Ordered,
        64,
    )
    .unwrap();
    assert_eq!(
        graph.structured,
        Err(StatisticalEvidenceUnknown::UnsupportedReplay)
    );
}

#[test]
fn structured_recipe_order_and_transfers_are_not_erased_by_equal_aggregate_work() {
    let a = command(2, "selected.kernel.a");
    let b = command(2, "selected.kernel.b");
    let forward = [
        a.canonical_command(0, 0, provider()).unwrap(),
        b.canonical_command(1, 1, provider()).unwrap(),
    ];
    let reverse = [
        b.canonical_command(0, 0, provider()).unwrap(),
        a.canonical_command(1, 1, provider()).unwrap(),
    ];
    let run = |commands: &[CostPhysicalCommand<'_>]| {
        finish(builder(
            commands,
            &[row(true), row(false)],
            CostProductOutput::GreedyToken,
            CoreReadbackRoute::NoReadback,
            0,
            true,
        ))
        .structured
        .unwrap()
    };
    let f = run(&forward);
    let r = run(&reverse);
    assert_eq!(f.device().aggregate_work(), r.device().aggregate_work());
    assert_ne!(f.device().ordered_template(), r.device().ordered_template());
    assert_ne!(
        f.device().provider_grouped_template(),
        r.device().provider_grouped_template()
    );
    let mut transfer = SelectedCommandCostBuilderV1::new(0);
    transfer
        .transfer(
            algorithm("actual.upload"),
            StatisticalTransferKindV1::HostToDevice,
            64,
        )
        .unwrap();
    let transfer = transfer.finish().unwrap();
    let upload = CostPhysicalCommand {
        native_op_id: "native.upload",
        command_index: 2,
        node_index: None,
        provider: None,
        command_phase: DeviceCommandPhase::Initialization,
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 0,
        token_count: 0,
        batching_form: "shared",
        compute_dispatch_count: 0,
        transfer_command_count: 1,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&transfer),
    };
    let with_upload = run(&[forward[0], forward[1], upload]);
    assert_ne!(
        f.device().ordered_template(),
        with_upload.device().ordered_template()
    );
    assert_eq!(
        with_upload.device().aggregate_work().host_to_device_bytes,
        64
    );
}

#[test]
fn structured_recipe_host_prefill_mask_policy_and_capacity_are_explicit() {
    let mut partial = row(false);
    partial
        .host_features
        .as_mut()
        .unwrap()
        .state
        .generated_tokens_before = 0;
    partial
        .host_features
        .as_mut()
        .unwrap()
        .state
        .sampling_history_tokens = 0;
    partial.work = ActualRowWork::Prefill {
        offset: 0,
        count: 4,
        total_prompt_tokens: 8,
    };
    partial.output = CostRowOutput::Prefill {
        final_logits: false,
    };
    let mut end = partial;
    end.work = ActualRowWork::Prefill {
        offset: 4,
        count: 4,
        total_prompt_tokens: 8,
    };
    end.output = CostRowOutput::Prefill { final_logits: true };
    end.mask_upload_required = true;
    end.host_features
        .as_mut()
        .unwrap()
        .state
        .pending_decoded_utf8 = true;
    let mut h = StructuredHostAccumulator::new(0);
    h.observe(partial);
    h.observe(end);
    assert!(h.failure.is_none());
    assert!(h.rows[0].initial_prefill && !h.rows[0].final_prefill);
    assert_eq!(
        h.rows[0].terminal_expectation,
        HostTerminalExpectationV1::NoTokenProduced
    );
    assert!(
        !h.rows[1].initial_prefill && h.rows[1].final_prefill && h.rows[1].mask_upload_required
    );
    assert_eq!(h.rows[1].decode_requires_full_logits, None);
    assert!(h.rows[1].pending_decoded_utf8);
    assert_eq!(
        h.rows[1].terminal_expectation,
        HostTerminalExpectationV1::TokenMayTerminate
    );
    assert_eq!(
        h.rows[1].installed_policy,
        end.host_features.unwrap().policy
    );
    let mut h = StructuredHostAccumulator::new(0);
    for _ in 0..MAX_COST_ROWS {
        h.observe(row(false));
    }
    assert_eq!(h.rows.len(), MAX_COST_ROWS);
    assert!(h.rows.capacity() <= MAX_COST_ROWS);
    h.observe(row(false));
    assert_eq!(h.failure, Some(StatisticalEvidenceUnknown::Capacity));
    assert_eq!(h.rows.len(), MAX_COST_ROWS);
    h.observe(partial);
    assert_eq!(h.failure, Some(StatisticalEvidenceUnknown::Capacity));
}

#[test]
fn structured_recipe_transport_keeps_legacy_wire_and_import_empty() {
    let declared = command(1, "selected.kernel");
    let physical = declared.canonical_command(0, 0, provider()).unwrap();
    let build = |enabled| {
        builder(
            &[physical],
            &[row(false)],
            CostProductOutput::FullLogits,
            CoreReadbackRoute::HostSynchronized,
            0,
            enabled,
        )
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap()
    };
    let old = build(false);
    let new = build(true);
    assert_eq!(old.exact, new.exact);
    let exact = new.exact.clone();
    let old = old.statistical.unwrap();
    let new = new.statistical.unwrap();
    assert_eq!(old, new);
    assert_eq!(
        serde_json::to_vec(&old).unwrap(),
        serde_json::to_vec(&new).unwrap()
    );
    assert!(old.structured_capture().is_none());
    let recipe = new.structured_capture().unwrap().unwrap();
    assert!(recipe.retained_rows() >= recipe.physical_host_rows().len());
    assert!(std::sync::Arc::ptr_eq(
        recipe,
        new.clone().structured_capture().unwrap().unwrap()
    ));
    let imported = StatisticalWaveEvidenceV1::from_wire_v1(new.to_wire_v1(), &exact).unwrap();
    assert!(imported.structured_capture().is_none());
}
