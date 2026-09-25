use super::*;
use crate::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, DeviceExecutionPath, DeviceNativeOperationId,
    DeviceNativeWorkAttribution, OperationCostCommand,
};

fn class(name: &str) -> SelectedAlgorithmClassV1 {
    SelectedAlgorithmClassV1::new(name, 1, [1; 32], [2; 32]).unwrap()
}
fn kernel(units: u64, scratch: u64) -> KernelNumericWorkV1 {
    KernelNumericWorkV1 {
        logical_units: units,
        padded_units: units,
        inner_units_per_logical_unit: 1,
        grid: [1, 1, 1],
        scratch_bytes: scratch,
        staged_weight_bytes: 0,
    }
}
fn selected(capture: bool, assignments: &[(&str, u64)]) -> SelectedCommandCostEvidenceV1 {
    let mut b = if capture {
        SelectedCommandCostBuilderV1::new_with_algorithm_work(1)
    } else {
        SelectedCommandCostBuilderV1::new(1)
    };
    for &(name, units) in assignments {
        b.kernel(class(name), kernel(units, 32)).unwrap();
    }
    b.finish().unwrap()
}
fn provider() -> CostProviderIdentity<'static> {
    CostProviderIdentity {
        provider_id: "fixture.provider",
        implementation_fingerprint: "impl",
        operation_fingerprint: "op",
    }
}
fn physical<'a>(
    index: u32,
    selected: &'a SelectedCommandCostEvidenceV1,
    compute: u64,
) -> CostPhysicalCommand<'a> {
    CostPhysicalCommand {
        native_op_id: "fixture.wave",
        command_index: index,
        node_index: Some(index),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(provider()),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 1,
        token_count: 1,
        batching_form: "packed",
        compute_dispatch_count: compute,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(selected),
    }
}
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
fn builder(commands: &[CostPhysicalCommand<'_>], capture: bool) -> CanonicalWaveCostBuilder {
    let mut b = if capture {
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::GreedyToken)
    } else {
        CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken)
    };
    for command in commands {
        b.physical_command(*command).unwrap();
    }
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    b.row(row()).unwrap();
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
fn wave_algorithm_equal_totals_do_not_exchange_assignments_or_occurrences() {
    for (left, right) in [
        (vec![("a", 4), ("b", 12)], vec![("a", 12), ("b", 4)]),
        (vec![("a", 4), ("a", 12)], vec![("a", 12), ("a", 4)]),
    ] {
        let a = selected(true, &left);
        let b = selected(true, &right);
        let a = finish(builder(&[physical(0, &a, 2)], true));
        let b = finish(builder(&[physical(0, &b, 2)], true));
        assert_eq!(a.exact, b.exact);
        assert_eq!(a.statistical, b.statistical);
        let a = a.structured.unwrap();
        let b = b.structured.unwrap();
        assert_eq!(a.device().aggregate_work(), b.device().aggregate_work());
        assert_ne!(
            a.algorithm_work().unwrap().ordered_command_binding(),
            b.algorithm_work().unwrap().ordered_command_binding()
        );
        assert_eq!(
            a.algorithm_work().unwrap().validate_structure(&b),
            Err(StatisticalEvidenceUnknown::CommandMismatch)
        );
    }
}
#[test]
fn wave_algorithm_preserves_physical_order_while_summing_same_algorithm() {
    let first = selected(true, &[("a", 4), ("b", 12)]);
    let second = selected(true, &[("a", 12), ("b", 4)]);
    let a = finish(builder(
        &[physical(0, &first, 2), physical(3, &second, 2)],
        true,
    ))
    .structured
    .unwrap();
    let b = finish(builder(
        &[physical(0, &second, 2), physical(3, &first, 2)],
        true,
    ))
    .structured
    .unwrap();
    let work = a.algorithm_work().unwrap();
    let other = b.algorithm_work().unwrap();
    assert_eq!(work.entries(), other.entries());
    assert_ne!(
        work.ordered_command_binding(),
        other.ordered_command_binding()
    );
    assert_eq!(work.physical_command_count(), 2);
    assert_eq!(
        work.entries()
            .iter()
            .find(|e| e.algorithm() == class("a"))
            .unwrap()
            .work()
            .logical_units,
        16
    );
    assert_eq!(work.entries()[0].commands(), 2);
    assert_eq!(work.aggregate_work().peak_scratch_bytes, 32);
    assert_eq!(
        work.validate_structure(&b),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
}
#[test]
fn wave_algorithm_missing_one_producer_drops_partial_allocations_without_changing_old_evidence() {
    let good = selected(true, &[("a", 1)]);
    let missing = selected(false, &[("b", 1)]);
    let partial = finish(builder(
        &[physical(0, &good, 1), physical(1, &missing, 1)],
        true,
    ));
    assert!(partial.statistical.is_ok());
    let recipe = partial.structured.unwrap();
    assert_eq!(
        recipe.algorithm_work(),
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
    assert_eq!(recipe.retained_rows(), recipe.physical_host_capacity());
    let mut accumulator = WaveAlgorithmAccumulator::new();
    accumulator.observe(physical(0, &good, 1));
    assert!(matches!(&accumulator.pending,Ok(p) if p.commands != 0));
    accumulator.observe(physical(1, &missing, 1));
    assert!(matches!(
        accumulator.pending,
        Err(StatisticalEvidenceUnknown::MissingProducer)
    ));
    let plain = finish(builder(&[physical(0, &good, 1)], false));
    let capture = finish(builder(&[physical(0, &good, 1)], true));
    assert_eq!(plain.statistical, capture.statistical);
    assert_eq!(
        serde_json::to_vec(&plain.statistical.unwrap()).unwrap(),
        serde_json::to_vec(&capture.statistical.unwrap()).unwrap()
    );
}
#[test]
fn wave_algorithm_actual_attribution_matches_future_command_and_copy_kinds() {
    let mut b = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    b.kernel(class("kernel"), kernel(4, 32)).unwrap();
    b.transfer(class("copy"), StatisticalTransferKindV1::HostToDevice, 64)
        .unwrap();
    b.transfer(class("copy"), StatisticalTransferKindV1::DeviceToHost, 16)
        .unwrap();
    b.transfer(class("copy"), StatisticalTransferKindV1::DeviceToDevice, 8)
        .unwrap();
    b.transfer(class("copy"), StatisticalTransferKindV1::Fill, 32)
        .unwrap();
    let selected = b.finish().unwrap();
    let actual = DeviceNativeWorkAttribution::new(
        0,
        Some(0),
        DeviceCommandPhase::Compute,
        DeviceNativeOperationId::new("fixture.wave").unwrap(),
        DeviceExecutionPath::Eager,
        DeviceBatchingForm::Packed,
        1,
        1,
        1,
        4,
        None,
    )
    .unwrap()
    .with_statistical_evidence(selected.clone())
    .unwrap();
    let future = OperationCostCommand::new(
        "fixture.wave",
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        1,
        1,
        1,
        4,
    )
    .unwrap()
    .with_statistical_evidence(selected)
    .unwrap();
    let a = finish(builder(
        &[CostPhysicalCommand::from_attribution(
            &actual,
            Some(provider()),
        )],
        true,
    ));
    let f = finish(builder(
        &[future.canonical_command(0, 0, provider()).unwrap()],
        true,
    ));
    assert_eq!(a.exact, f.exact);
    assert_eq!(a.structured, f.structured);
    let recipe = a.structured.unwrap();
    let evidence = recipe.algorithm_work().unwrap();
    evidence.validate_exact(&a.exact).unwrap();
    assert_eq!(evidence.entries().len(), 5);
    assert_eq!(evidence.aggregate_work().host_to_device_bytes, 64);
    assert_eq!(evidence.aggregate_work().device_to_host_bytes, 16);
    assert_eq!(evidence.aggregate_work().device_to_device_bytes, 8);
    assert_eq!(evidence.aggregate_work().fill_bytes, 32);
    let raw = serde_json::to_value(evidence).unwrap();
    assert_eq!(raw["entries"].as_array().unwrap().len(), 5);
}

#[test]
fn wave_algorithm_rejects_count_mismatch_and_replay_without_retaining_partial_work() {
    let good = selected(true, &[("a", 1)]);
    let mut mismatch = physical(1, &good, 2);
    let mut replay = physical(1, &good, 1);
    replay.path = CostCommandPath::Replayed;
    replay.reusable_graph_node_count = Some(1);
    for (bad, expected) in [
        (mismatch, StatisticalEvidenceUnknown::CommandMismatch),
        (replay, StatisticalEvidenceUnknown::UnsupportedReplay),
    ] {
        let mut accumulator = WaveAlgorithmAccumulator::new();
        accumulator.observe(physical(0, &good, 1));
        accumulator.observe(bad);
        accumulator.observe(physical(2, &good, 1));
        assert!(matches!(&accumulator.pending, Err(error) if *error == expected));
    }
    mismatch.statistical_evidence = None;
    let mut accumulator = WaveAlgorithmAccumulator::new();
    accumulator.observe(mismatch);
    assert!(matches!(
        accumulator.pending,
        Err(StatisticalEvidenceUnknown::MissingProducer)
    ));
}

#[test]
fn wave_algorithm_capacity_and_overflow_are_explicit_not_truncated() {
    let assignments = vec![("a", 1); MAX_COST_COMMANDS];
    let full = selected(true, &assignments);
    let accepted = finish(builder(
        &[physical(0, &full, MAX_COST_COMMANDS as u64)],
        true,
    ));
    assert_eq!(
        accepted
            .structured
            .unwrap()
            .algorithm_work()
            .unwrap()
            .entries()[0]
            .commands(),
        MAX_COST_COMMANDS as u64
    );
    let one = selected(true, &[("a", 1)]);
    let excessive = finish(builder(
        &[
            physical(0, &full, MAX_COST_COMMANDS as u64),
            physical(1, &one, 1),
        ],
        true,
    ));
    assert!(excessive.statistical.is_ok());
    assert_eq!(
        excessive.structured.unwrap().algorithm_work(),
        Err(StatisticalEvidenceUnknown::Capacity)
    );
    let huge = selected(true, &[("a", u64::MAX / 2 + 1)]);
    let overflow = finish(builder(
        &[physical(0, &huge, 1), physical(1, &huge, 1)],
        true,
    ));
    assert_eq!(
        overflow.structured,
        Err(StatisticalEvidenceUnknown::Overflow)
    );
}
#[test]
fn wave_algorithm_retention_units_charge_sparse_work_without_becoming_physical_rows() {
    let e = selected(true, &[("a", 4), ("b", 12)]);
    let finished = finish(builder(&[physical(0, &e, 2), physical(1, &e, 2)], true));
    let recipe = finished.structured.unwrap();
    let evidence = recipe.algorithm_work().unwrap();
    let bytes = evidence.retained_dynamic_bytes().unwrap();
    assert!(bytes >= std::mem::size_of_val(evidence.entries()));
    assert_eq!(
        recipe.retained_units().unwrap(),
        recipe.physical_host_capacity()
            + bytes.div_ceil(std::mem::size_of::<CostRowNumericFeatures>())
    );
    assert!(recipe.retained_rows() > recipe.physical_host_capacity());
    let fixed_bytes =
        std::mem::size_of::<UnsettledStructuredWaveEvidenceV1>() + 2 * std::mem::size_of::<usize>();
    assert_eq!(
        recipe.retained_bytes().unwrap(),
        fixed_bytes
            + recipe.physical_host_capacity() * std::mem::size_of::<StructuredHostRowV1>()
            + bytes
    );
    let charged_row_bytes = std::mem::size_of::<StructuredHostRowV1>()
        .max(std::mem::size_of::<CostRowNumericFeatures>());
    assert!(
        recipe.retained_units().unwrap() * charged_row_bytes + fixed_bytes
            >= recipe.retained_bytes().unwrap()
    );
    let exact = finished.exact;
    let statistics = finished
        .statistical
        .unwrap()
        .attach_structured_capture(Ok(recipe), &exact);
    let actual = ActualWaveShape {
        kind: exact.kind,
        path: exact.path,
        graph: exact.graph,
        row_order: exact.row_order,
        provider_signature: exact.provider_signature,
        output_policy_signature: exact.output_policy_signature,
        numeric_features: exact.numeric_features,
        host_content_features: exact.host_content_features,
        row_multiset_features: exact.row_multiset_features,
        statistical_evidence: Some(statistics),
        rows: vec![ActualWaveRow {
            request_id: ferrum_types::RequestId::new(),
            owner_incarnation: 1,
            work_generation: 1,
            input_index: 0,
            work: row().work,
        }],
        recurrent_state_bytes: 64,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    };
    let legacy_units = actual.rows.capacity()
        + actual.numeric_features.as_ref().unwrap().rows.capacity()
        + actual
            .row_multiset_features
            .as_ref()
            .unwrap()
            .rows
            .capacity();
    let retained = legacy_units
        + actual
            .statistical_evidence
            .as_ref()
            .unwrap()
            .structured_retained_rows();
    let physical = actual
        .statistical_evidence
        .as_ref()
        .unwrap()
        .structured_physical_host_capacity()
        .max(actual.numeric_features.as_ref().unwrap().rows.capacity())
        .max(
            actual
                .row_multiset_features
                .as_ref()
                .unwrap()
                .rows
                .capacity(),
        );
    let limits = CostRecorderLimits {
        max_waves: 1,
        max_rows_per_wave: physical,
        max_retained_rows: retained,
    };
    let mut recorder = BoundedWaveRecorder::new(std::num::NonZeroU64::MIN, limits).unwrap();
    recorder
        .begin(
            actual.clone(),
            WaveObservationBoundary::IsolatedPreparationToCommit,
            1,
        )
        .unwrap();
    let mut recorder = BoundedWaveRecorder::new(
        std::num::NonZeroU64::MIN,
        CostRecorderLimits {
            max_retained_rows: retained - 1,
            ..limits
        },
    )
    .unwrap();
    assert_eq!(
        recorder
            .begin(
                actual,
                WaveObservationBoundary::IsolatedPreparationToCommit,
                1
            )
            .unwrap_err(),
        CostRecorderError::RowCapacity
    );
}
