use super::envelope::PendingGenerators;
use super::fit::{FitRow, RowSpaceFit};
use super::input::PendingQuery;
use super::support::JointSupport;
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
mod cost_template_policy;
mod phases;
mod repetition_bounds;
mod required_demand;

#[test]
fn structured_v2_payload_counts_spare_input_capacity_and_optional_vectors() {
    fn grow<T>(values: &mut Vec<T>) -> usize {
        let before = values.capacity();
        values.reserve_exact(before + 7);
        (values.capacity() - before) * std::mem::size_of::<T>()
    }
    let mut input = project(&wave(0, true, 8, "fixture.first", [false, false])).unwrap();
    let before = input.retained_payload_bytes().unwrap();
    let additional = grow(&mut input.basis)
        + grow(&mut input.support)
        + grow(&mut input.physical_host_rows)
        + grow(&mut input.pending_positions)
        + grow(&mut input.length_positions);
    assert_eq!(input.retained_payload_bytes(), Some(before + additional));
    assert!(input.settled_terminal_causes.is_none() && input.completion.is_none());
    let causes = Vec::<(u32, ferrum_types::FinishReason)>::with_capacity(11);
    let positions = Vec::<u32>::with_capacity(13);
    let optional_bytes = causes.capacity()
        * std::mem::size_of::<(u32, ferrum_types::FinishReason)>()
        + positions.capacity() * std::mem::size_of::<u32>();
    input.settled_terminal_causes = Some(causes);
    input.completion = Some(super::completion::InputCompletion {
        positions,
        basis_offset: 0,
        support_offset: 0,
        settled: false,
    });
    assert_eq!(
        input.retained_payload_bytes(),
        Some(before + additional + optional_bytes)
    );
}

#[test]
fn structured_v2_payload_query_counts_embedded_input_once_and_pending_capacity() {
    let input = project(&wave(0, true, 8, "fixture.first", [false, false])).unwrap();
    let bytes = std::mem::size_of::<StructuredQueryV2>() + input.retained_payload_bytes().unwrap()
        - std::mem::size_of::<StructuredInputV2>();
    let mut query = StructuredQueryV2::exact(input);
    assert_eq!(query.retained_payload_bytes(), Some(bytes));
    let eligible = Vec::<u32>::with_capacity(19);
    let pending_bytes = eligible.capacity() * std::mem::size_of::<u32>();
    query.pending = Some(PendingQuery {
        eligible,
        constraint: HostPendingConstraintV2::AnySubset,
    });
    assert_eq!(query.retained_payload_bytes(), Some(bytes + pending_bytes));
    query.pending.as_mut().unwrap().eligible.extend([0, 1]);
    assert_eq!(query.retained_payload_bytes(), Some(bytes + pending_bytes));
}

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
    wave_with_history(
        terminal,
        capture,
        work_a,
        algorithm,
        pending,
        exact_policy,
        numeric_policy,
        64,
        2,
    )
}

fn wave_with_history(
    terminal: usize,
    capture: bool,
    work_a: u64,
    algorithm: &str,
    pending: [bool; 2],
    exact_policy: [u8; 32],
    numeric_policy: [u8; 32],
    kv_tokens: u32,
    generated: u64,
) -> CanonicalStructuredWave {
    wave_with_history_domain(
        terminal,
        capture,
        work_a,
        algorithm,
        pending,
        exact_policy,
        numeric_policy,
        kv_tokens,
        generated,
        HostContentDomainV1::PlainTextGreedyV1,
    )
}

fn wave_with_history_domain(
    terminal: usize,
    capture: bool,
    work_a: u64,
    algorithm: &str,
    pending: [bool; 2],
    exact_policy: [u8; 32],
    numeric_policy: [u8; 32],
    kv_tokens: u32,
    generated: u64,
    domain: HostContentDomainV1,
) -> CanonicalStructuredWave {
    wave_with_history_domain_and_mask(
        terminal,
        capture,
        work_a,
        algorithm,
        pending,
        exact_policy,
        numeric_policy,
        kv_tokens,
        generated,
        domain,
        [false; 2],
    )
}

fn wave_with_history_domain_and_mask(
    terminal: usize,
    capture: bool,
    work_a: u64,
    algorithm: &str,
    pending: [bool; 2],
    exact_policy: [u8; 32],
    numeric_policy: [u8; 32],
    kv_tokens: u32,
    generated: u64,
    domain: HostContentDomainV1,
    mask_upload_required: [bool; 2],
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
            work: ActualRowWork::Decode { kv_tokens },
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: generated,
                repetition_penalty_bits: 1f32.to_bits(),
            },
            host_policy_signature: exact_policy,
            mask_upload_required: mask_upload_required[position],
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(domain),
                    categorical_signature: numeric_policy,
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: generated,
                    maximum_output_tokens: if position == terminal || terminal == 2 {
                        generated + 1
                    } else {
                        generated + 18
                    },
                    sampling_history_tokens: generated,
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
fn installed_plain_text_actual_capture_retains_distinct_policy_and_work_axes() {
    for sampling in [
        PlainTextSamplingRouteV2::Greedy {
            repetition_penalty: true,
        },
        PlainTextSamplingRouteV2::FullLogits,
    ] {
        let domain = HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
            sampling,
            model_eos: true,
            user_stop: true,
        });
        let w = wave_with_history_domain(
            9,
            true,
            8,
            "fixture.first",
            [false, true],
            [13; 32],
            [14; 32],
            64,
            3,
            domain,
        );
        let actual = project(&w).unwrap();
        assert!(actual
            .physical_host_rows()
            .iter()
            .all(|row| row.installed_policy.empirical_content_domain == Some(domain)));
        assert_eq!(actual.pending_positions, [1]);
        let later = project(&wave_with_history_domain(
            9,
            true,
            8,
            "fixture.first",
            [false, true],
            [13; 32],
            [14; 32],
            65,
            4,
            domain,
        ))
        .unwrap();
        assert_eq!(actual.owner(), later.owner());
        assert_ne!(actual.regression_axes(), later.regression_axes());
        let other = project(&wave_with_history_domain(
            9,
            true,
            8,
            "fixture.first",
            [false, true],
            [15; 32],
            [16; 32],
            64,
            3,
            domain,
        ))
        .unwrap();
        assert_ne!(actual.owner(), other.owner());
    }
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

#[test]
fn structured_v2_prepared_reuse_matches_source_replay_numeric_and_owner() {
    for (terminal, work, pending) in [
        (9, 8, [false, false]),
        (0, 24, [true, false]),
        (2, 16, [true, true]),
    ] {
        let w = wave(terminal, true, work, "fixture.first", pending);
        let selected = w.statistical.as_ref().unwrap();
        let recipe = selected.structured_capture().unwrap().unwrap();
        let device = recipe.device();
        let algorithms = recipe
            .algorithm_work()
            .unwrap()
            .entries()
            .iter()
            .map(|a| (*a.algorithm().signature(), a.kind(), a.commands(), a.work()))
            .collect::<Vec<_>>();
        let product = match device.product() {
            StructuredCostProductV1::GreedyToken => StructuredProductV2::GreedyToken,
            StructuredCostProductV1::FullLogits => StructuredProductV2::FullLogits,
        };
        // The unchanged source-replay path independently reconstructs the
        // numerical input; equality includes every basis/support/domain field.
        let (replayed, replayed_facts) = StructuredInputV2::from_replay_parts(
            &w.exact,
            selected,
            *device.ordered_template(),
            device.provider_grouped_template().copied(),
            product,
            device.readback(),
            recipe.physical_host_rows(),
            &algorithms,
            device.replay_work().map(|r| {
                [
                    u64::from(r.replayed_segments()),
                    u64::from(r.logical_commands()),
                    r.native_graph_nodes(),
                ]
            }),
            device.retries(),
        )
        .unwrap();
        let actual = StructuredInputV2::from_actual(&w.exact, selected, recipe).unwrap();
        let facts = StructuredOwnerFactsV2::from_prepared(&w.exact, selected, recipe).unwrap();
        assert_eq!(actual, replayed);
        assert_eq!(
            StructuredInputV2::owner_for(&w.exact, selected, recipe).unwrap(),
            *actual.owner()
        );
        assert_eq!(facts.owner_key().unwrap(), *actual.owner());
        assert_eq!(
            serde_json::to_vec(&facts).unwrap(),
            serde_json::to_vec(&replayed_facts).unwrap()
        );
    }
}

#[test]
fn structured_v2_prepared_reuse_keeps_owner_only_validation_boundaries() {
    fn rejects(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    ) {
        assert!(matches!(
            StructuredInputV2::from_actual(exact, selected, recipe),
            Err(StructuredUnknown::MissingEvidence)
        ));
        assert!(matches!(
            StructuredInputV2::owner_for(exact, selected, recipe),
            Err(StructuredUnknown::MissingEvidence)
        ));
        assert!(matches!(
            StructuredOwnerFactsV2::from_prepared(exact, selected, recipe),
            Err(StructuredUnknown::MissingEvidence)
        ));
    }
    let first = wave(9, true, 8, "fixture.first", [false, false]);
    let second = wave(9, true, 24, "fixture.first", [false, false]);
    assert_eq!(first.exact, second.exact);
    let selected = first.statistical.as_ref().unwrap();
    let attached = selected.structured_capture().unwrap().unwrap();
    let duplicate = Arc::new(attached.as_ref().clone());
    rejects(&first.exact, selected, &duplicate);
    rejects(
        &first.exact,
        selected,
        second
            .statistical
            .as_ref()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap(),
    );
    let mut changed_shape = first.exact.clone();
    changed_shape.recurrent_state_bytes += 1;
    rejects(&changed_shape, selected, attached);
    let missing_work = wave(9, false, 8, "fixture.first", [false, false]);
    let missing_selected = missing_work.statistical.as_ref().unwrap();
    rejects(
        &missing_work.exact,
        missing_selected,
        missing_selected.structured_capture().unwrap().unwrap(),
    );
    // A previous rejected call cannot poison the next valid current recipe.
    assert!(StructuredInputV2::from_actual(&first.exact, selected, attached).is_ok());
    assert!(StructuredInputV2::owner_for(&first.exact, selected, attached).is_ok());
}
