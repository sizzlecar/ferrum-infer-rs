//! Typed controlled CPU producer, enabled only by the live collector test.
//! It describes the executed full-logits fill and the original host rows.
use super::*;

pub(super) fn record(
    context: &mut PlanRuntimeCostObservationContext<'_>,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
    providers: Option<&[ferrum_interfaces::execution_cost::CostProviderIdentity<'_>]>,
) {
    assert!(prefills.iter().all(|row| row.chunk.is_final()));
    record_with_layout(
        context,
        prefills,
        decodes,
        providers,
        contract::ControlledCpuFillLayout::Uniform,
        0,
    );
}

/// Actual core-submitted CPU waves may be partial. Their original provider
/// binding is mandatory; the synthetic producer above remains final-only.
pub(super) fn record_native_with_layout(
    context: &mut PlanRuntimeCostObservationContext<'_>,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
    providers: &[ferrum_interfaces::execution_cost::CostProviderIdentity<'_>],
    layout: contract::ControlledCpuFillLayout,
    recurrent_state_bytes: u64,
) {
    assert!(
        !providers.is_empty(),
        "native wave requires real bound providers"
    );
    record_with_layout(
        context,
        prefills,
        decodes,
        Some(providers),
        layout,
        recurrent_state_bytes,
    );
}

/// The opt-in checkpoint program owns a real fixed BoundaryValue in its plan.
/// Actual evidence and future queries report the same declared physical bytes;
/// the older stateless program retains its zero-state domain.
pub(super) fn recurrent_state_bytes(executor: &ControlledExecutor, rows: usize) -> u64 {
    if executor.evidence.prefix.is_none() {
        return 0;
    }
    executor
        .evidence
        .fixture
        .as_ref()
        .unwrap()
        .resolved
        .execution_plan()
        .checkpoint_byte_plan(1)
        .expect("checkpoint CPU program has a declared fixed state")
        .logical_bytes()
        .checked_mul(u64::try_from(rows).unwrap())
        .expect("CPU fixture physical row state fits u64")
}

fn record_with_layout(
    context: &mut PlanRuntimeCostObservationContext<'_>,
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
    providers: Option<&[ferrum_interfaces::execution_cost::CostProviderIdentity<'_>]>,
    layout: contract::ControlledCpuFillLayout,
    recurrent_state_bytes: u64,
) {
    let full_logits = !prefills.is_empty()
        || decodes
            .iter()
            .any(|row| row.logits_policy.requires_full_logits());
    assert!(providers.is_some() || full_logits);
    let row_count = prefills.len() + decodes.len();
    let query_tokens = prefills
        .iter()
        .map(|row| row.chunk.tokens_to_process() as u64)
        .sum::<u64>()
        + decodes.len() as u64;
    let mut builder =
        command_builder_with_layout(row_count, query_tokens, full_logits, providers, layout);
    let mut rows = Vec::with_capacity(row_count);
    let mut ordinary_rows = 0;
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
                CostRowOutput::Prefill {
                    final_logits: row.chunk.is_final(),
                },
            )
        })
        .chain(decodes.iter().map(|row| {
            (
                &row.request_id,
                ActualRowWork::Decode {
                    kv_tokens: row.kv_cache.num_tokens() as u32,
                },
                CostRowOutput::Decode {
                    requires_full_logits: row.logits_policy.requires_full_logits(),
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
        ordinary_rows += usize::from(host.supports_installed_plain_text_content());
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
    let built = builder
        .finish_with_captured_structure(
            kind,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            recurrent_state_bytes,
        )
        .unwrap();
    // Prefix preparation intentionally has no numerical host domain. Preserve
    // its actual command/row identity, but never manufacture a trained sample.
    let statistical_evidence = if ordinary_rows == row_count {
        Some(built.statistical.unwrap())
    } else {
        assert_eq!(
            ordinary_rows, 0,
            "one prepared wave cannot mix preparation and ordinary policies"
        );
        assert!(matches!(
            built.statistical,
            Err(ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown::MissingHostDomain)
        ));
        None
    };
    let shape = ActualWaveShape {
        statistical_evidence,
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
    if let Some(statistical) = shape.statistical_evidence.as_ref() {
        statistical.validate_actual(&shape).unwrap();
    }
    let at = context.now_ns();
    context.physical_wave(Ok(shape), at);
}

/// One cost recipe for the CPU fill actually executed by this fixture.
/// Actual and pre-execution future projections share this algorithm evidence.
pub(super) fn command_builder(row_count: usize, query_tokens: u64) -> CanonicalWaveCostBuilder {
    command_builder_for_product(row_count, query_tokens, true, None)
}

pub(super) fn command_builder_for_product(
    row_count: usize,
    query_tokens: u64,
    full_logits: bool,
    providers: Option<&[ferrum_interfaces::execution_cost::CostProviderIdentity<'_>]>,
) -> CanonicalWaveCostBuilder {
    command_builder_with_layout(
        row_count,
        query_tokens,
        full_logits,
        providers,
        contract::ControlledCpuFillLayout::Uniform,
    )
}

pub(super) fn command_builder_with_layout(
    row_count: usize,
    query_tokens: u64,
    full_logits: bool,
    providers: Option<&[ferrum_interfaces::execution_cost::CostProviderIdentity<'_>]>,
    layout: contract::ControlledCpuFillLayout,
) -> CanonicalWaveCostBuilder {
    let native_op_id = layout.native_op_id(full_logits);
    let mut algorithm = SelectedCommandCostBuilderV1::new_with_algorithm_work(query_tokens);
    for (primitive, rows) in layout.primitive_rows(row_count) {
        if rows == 0 {
            continue;
        }
        algorithm
            .kernel(
                SelectedAlgorithmClassV1::new(
                    primitive.native_op_id(full_logits),
                    1,
                    [1; 32],
                    [2; 32],
                )
                .unwrap(),
                KernelNumericWorkV1 {
                    logical_units: rows as u64 * 64 * if full_logits { 1 } else { 2 },
                    padded_units: rows as u64 * 64 * if full_logits { 1 } else { 2 },
                    inner_units_per_logical_unit: 1,
                    grid: [1, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
    }
    let algorithm = algorithm.finish().unwrap();
    let mut builder = CanonicalWaveCostBuilder::new_with_structured_statistics(
        0,
        if full_logits {
            CostProductOutput::FullLogits
        } else {
            CostProductOutput::GreedyToken
        },
    );
    let commands = providers.map_or(1, |providers| providers.len());
    assert!(commands > 0);
    for command_index in 0..commands {
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id,
                command_index: command_index as u32,
                node_index: providers.map(|_| command_index as u32),
                command_phase: vnext::DeviceCommandPhase::Compute,
                provider: providers.map(|providers| providers[command_index]),
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: row_count as u32,
                token_count: query_tokens,
                batching_form: "packed",
                compute_dispatch_count: layout.compute_dispatch_count(row_count),
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                statistical_evidence: Some(&algorithm),
            })
            .unwrap();
    }
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    builder
}

/// Bind the numerical recipe to the providers used by the original core dispatch.
pub(super) fn provider_identities<'a>(
    providers: &'a [vnext::BoundOperationProvider<'_, contract::TestRuntime>],
) -> Vec<ferrum_interfaces::execution_cost::CostProviderIdentity<'a>> {
    providers
        .iter()
        .map(|provider| {
            let descriptor = provider.descriptor();
            ferrum_interfaces::execution_cost::CostProviderIdentity {
                provider_id: descriptor.provider_id().as_str(),
                implementation_fingerprint: descriptor.provider_implementation_fingerprint(),
                operation_fingerprint: descriptor.operation_fingerprint(),
            }
        })
        .collect()
}
