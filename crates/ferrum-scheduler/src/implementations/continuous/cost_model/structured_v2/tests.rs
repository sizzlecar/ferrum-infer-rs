use super::envelope::PendingGenerators;
use super::fit::{FitRow, RowSpaceFit};
use super::input::PendingQuery;
use super::support::JointSupport;
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
mod phases;

fn wave(
    terminal: usize,
    capture: bool,
    work_a: u64,
    algorithm: &str,
    pending: [bool; 2],
) -> CanonicalStructuredWave {
    wave_with_host_policy(
        terminal, capture, work_a, algorithm, pending, [3; 32], [4; 32],
    )
}

fn wave_with_host_policy(
    terminal: usize,
    capture: bool,
    work_a: u64,
    algorithm: &str,
    pending: [bool; 2],
    exact_policy: [u8; 32],
    numeric_policy: [u8; 32],
) -> CanonicalStructuredWave {
    let mut command = if capture {
        SelectedCommandCostBuilderV1::new_with_algorithm_work(2)
    } else {
        SelectedCommandCostBuilderV1::new(2)
    };
    for (entry, work) in [(algorithm, work_a), ("fixture.second", 32 - work_a)] {
        command
            .kernel(
                SelectedAlgorithmClassV1::new(entry, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: work,
                    padded_units: work,
                    inner_units_per_logical_unit: 2,
                    grid: [1, 1, 1],
                    scratch_bytes: 64,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
    }
    let selected = command.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.structured",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "fixture.provider",
            implementation_fingerprint: "impl-v1",
            operation_fingerprint: "op-v1",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 2,
        token_count: 2,
        batching_form: "packed",
        compute_dispatch_count: 2,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    for position in 0..2 {
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 64 },
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: 2,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            host_policy_signature: exact_policy,
            mask_upload_required: false,
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: numeric_policy,
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 2,
                    maximum_output_tokens: if position == terminal || terminal == 2 {
                        3
                    } else {
                        20
                    },
                    sampling_history_tokens: 2,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: pending[position],
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    }
    let wave = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let structured = wave
        .statistical
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .map(|recipe| recipe.as_ref().clone());
    CanonicalStructuredWave {
        exact: wave.exact,
        statistical: wave.statistical,
        structured,
    }
}

fn project(w: &CanonicalStructuredWave) -> Result<StructuredInputV2> {
    let selected = w.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual(
        &w.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
    )
}
#[test]
fn structured_v2_actual_multi_length_and_pending_share_owner_without_erasing_axes() {
    let a = project(&wave(0, true, 8, "fixture.first", [false, false])).unwrap();
    let b = project(&wave(2, true, 8, "fixture.first", [true, false])).unwrap();
    assert_eq!(a.owner(), b.owner());
    assert_eq!(a.domain_signature(), b.domain_signature());
    assert_eq!(b.length_positions, [0, 1]);
    assert_eq!(b.pending_positions, [0]);
    assert_ne!(a.regression_axes(), b.regression_axes());
    assert_ne!(a.physical_host_rows(), b.physical_host_rows());
}
#[test]
fn structured_v2_input_requires_original_attached_arc_and_complete_algorithm_work() {
    let a = wave(9, true, 8, "fixture.first", [false, false]);
    let b = wave(9, true, 24, "fixture.first", [false, false]);
    assert_eq!(a.exact, b.exact);
    let selected = a.statistical.as_ref().unwrap();
    let original = selected.structured_capture().unwrap().unwrap();
    let duplicate = Arc::new(original.as_ref().clone());
    assert!(matches!(
        StructuredInputV2::from_actual(&a.exact, selected, &duplicate),
        Err(StructuredUnknown::MissingEvidence)
    ));
    assert!(matches!(
        StructuredInputV2::from_actual(
            &a.exact,
            selected,
            b.statistical
                .as_ref()
                .unwrap()
                .structured_capture()
                .unwrap()
                .unwrap()
        ),
        Err(StructuredUnknown::MissingEvidence)
    ));
    assert!(project(&wave(9, false, 8, "fixture.first", [false, false])).is_err());
    assert!(project(&a).is_ok());
}
#[test]
fn structured_v2_query_binds_real_forecast_and_preserves_conditional_domain() {
    let w = wave(9, true, 8, "fixture.first", [true, true]);
    let selected = w.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let forecast = HostContentForecastV2::Unresolved(
        HostPendingSetV2::new(
            &w.exact,
            recipe,
            &[0, 1],
            HostPendingConstraintV2::NonEmptySubset,
        )
        .unwrap(),
    );
    let query = StructuredQueryV2::from_future(&w.exact, selected, recipe, &forecast).unwrap();
    assert_eq!(query.pending.as_ref().unwrap().eligible, [0, 1]);
    assert_eq!(
        query.pending.as_ref().unwrap().constraint,
        HostPendingConstraintV2::NonEmptySubset
    );
    assert_eq!(
        query.input.physical_host_rows(),
        recipe.physical_host_rows()
    );
}
#[test]
fn structured_v2_joint_support_cannot_assemble_independent_coordinate_maxima() {
    let points = [vec![0, 0], vec![10, 1], vec![1, 10]];
    let support = JointSupport::new(points.iter().map(Vec::as_slice)).unwrap();
    assert!(support.contains_envelope(&[0, 0], &[10, 1]));
    assert!(!support.contains_envelope(&[0, 0], &[10, 10]));
    assert!(!support.contains_envelope(&[1, 2], &[1, 1]));
}
#[test]
fn structured_v2_nonempty_singleton_does_not_require_unreachable_base_direction() {
    let mut input = project(&wave(9, true, 8, "fixture.first", [true, false])).unwrap();
    // Only position zero varies; position one stays fixed nonpending.
    let basis = input.basis.clone();
    let rows = (0..8)
        .map(|_| FitRow {
            basis: &basis,
            wall_ns: 1000,
        })
        .collect::<Vec<_>>();
    let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
    let generators = PendingGenerators::new(&fit, &input).unwrap();
    let single = PendingQuery {
        eligible: vec![0],
        constraint: HostPendingConstraintV2::NonEmptySubset,
    };
    let prediction = generators.bounds(&fit, &input, Some(&single)).unwrap();
    assert!((1000..=1001).contains(&prediction.upper_ns));
    let optional = PendingQuery {
        eligible: vec![0],
        constraint: HostPendingConstraintV2::AnySubset,
    };
    assert!(matches!(
        generators.bounds(&fit, &input, Some(&optional)),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
    input.pending_positions.clear();
    assert!(input.validate(&StructuredSettingsV2::default()).is_err());
}
#[test]
fn structured_v2_signed_envelope_and_unknown_direction_use_full_fit_rowspace() {
    let inputs = (0..8)
        .map(|mask| {
            project(&wave(
                9,
                true,
                8,
                "fixture.first",
                [mask & 1 != 0, mask & 2 != 0],
            ))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let rows = inputs
        .iter()
        .enumerate()
        .map(|(mask, input)| FitRow {
            basis: &input.basis,
            wall_ns: (1000 + 100 * (mask as i64 & 1) - 200 * ((mask as i64 >> 1) & 1)) as u64,
        })
        .collect::<Vec<_>>();
    let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
    let generators = PendingGenerators::new(&fit, &inputs[0]).unwrap();
    let pending = PendingQuery {
        eligible: vec![0, 1],
        constraint: HostPendingConstraintV2::AnySubset,
    };
    let b = generators.bounds(&fit, &inputs[0], Some(&pending)).unwrap();
    assert!((800..=801).contains(&b.lower_ns));
    assert!((1100..=1101).contains(&b.upper_ns));
    let support = JointSupport::new(inputs.iter().map(|i| i.support.as_slice())).unwrap();
    assert!(support.contains_envelope(&b.support_lower, &b.support_upper));
    let mut unseen = inputs[0].clone();
    unseen.basis[1] += 1.;
    assert!(matches!(
        generators.bounds(&fit, &unseen, Some(&pending)),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
}

#[test]
fn structured_v2_envelope_rejects_negative_reachable_fit_instead_of_clipping() {
    let inputs = (0..9)
        .map(|n| {
            project(&wave(
                9,
                true,
                8,
                "fixture.first",
                match n % 3 {
                    0 => [false, false],
                    1 => [true, false],
                    _ => [false, true],
                },
            ))
            .unwrap()
        })
        .collect::<Vec<_>>();
    let rows = inputs
        .iter()
        .enumerate()
        .map(|(n, input)| FitRow {
            basis: &input.basis,
            wall_ns: if n % 3 == 0 { 110 } else { 40 },
        })
        .collect::<Vec<_>>();
    let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
    let generators = PendingGenerators::new(&fit, &inputs[0]).unwrap();
    let pending = PendingQuery {
        eligible: vec![0, 1],
        constraint: HostPendingConstraintV2::AnySubset,
    };
    assert!(matches!(
        generators.bounds(&fit, &inputs[0], Some(&pending)),
        Err(StructuredUnknown::Numerical)
    ));
}
