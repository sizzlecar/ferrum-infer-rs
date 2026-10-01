//! Pure numerical timings over real canonical-builder inputs. These tests
//! neither manufacture engine receipts nor claim hardware measurements.
use super::super::envelope::PendingGenerators;
use super::super::fit::{FitRow, RowSpaceFit};
use super::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_interfaces::vnext::DeviceCommandPhase;

fn prepared(pending: u32, length: bool, natural: bool) -> StructuredInputV2 {
    prepared_with_repetition(pending, length, natural, None)
}
fn prepared_with_repetition(
    pending: u32,
    length: bool,
    natural: bool,
    repetition: Option<u64>,
) -> StructuredInputV2 {
    let wave = canonical(pending, length, natural, false, repetition);
    let selected = wave.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
    )
    .unwrap()
}
fn canonical(
    pending: u32,
    length: bool,
    model_eos: bool,
    user_stop: bool,
    repetition: Option<u64>,
) -> CanonicalStatisticalWave {
    let mut command = SelectedCommandCostBuilderV1::new_with_algorithm_work(2);
    command
        .kernel(
            SelectedAlgorithmClassV1::new("completion.fixture", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 2,
                padded_units: 2,
                inner_units_per_logical_unit: 2,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = command.finish().unwrap();
    let mut b = CanonicalWaveCostBuilder::new_with_structured_statistics(
        0,
        if repetition.is_some() {
            CostProductOutput::GreedyToken
        } else {
            CostProductOutput::FullLogits
        },
    );
    b.physical_command(CostPhysicalCommand {
        native_op_id: "completion.fixture",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "fixture",
            implementation_fingerprint: "v1",
            operation_fingerprint: "v1",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 2,
        token_count: 2,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    for p in 0..2 {
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 64 },
            output: CostRowOutput::Decode {
                requires_full_logits: repetition.is_none(),
                repetition_tokens: repetition.unwrap_or(0),
                repetition_penalty_bits: if repetition.is_some() {
                    1.1f32.to_bits()
                } else {
                    1f32.to_bits()
                },
            },
            host_policy_signature: [3; 32],
            mask_upload_required: false,
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextInstalledV2(
                        PlainTextPolicyCapabilityV2 {
                            sampling: if repetition.is_some() {
                                PlainTextSamplingRouteV2::Greedy {
                                    repetition_penalty: true,
                                }
                            } else {
                                PlainTextSamplingRouteV2::FullLogits
                            },
                            model_eos,
                            user_stop,
                        },
                    )),
                    categorical_signature: [4; 32],
                    decoder_text_bytes_per_token: 8,
                    decoder_scratch_bytes_per_token: 16,
                    raw_token_bytes_bound: 8,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 2,
                    maximum_output_tokens: if length { 3 } else { 20 },
                    sampling_history_tokens: 2,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: pending & (1 << p) != 0,
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
    wave
}

fn positions(bits: u32) -> Vec<u32> {
    (0..2).filter(|p| bits & (1 << p) != 0).collect()
}
fn observation(input: StructuredInputV2) -> StructuredNumericObservationV2 {
    StructuredObservationV2 {
        source: [1; 32],
        protocol: [2; 32],
        ordinal: 1,
        membership: StructuredMemberBindingV2 {
            rule_signature: [3; 32],
            offered_ordinal: 1,
            member_ordinal: 1,
            phase: StructuredPhaseV2::Fit,
        },
        call_id: 1,
        fingerprint: ExecutionFingerprint {
            model_weights: [4; 32],
            numerical_policy: [5; 32],
            device_runtime: [6; 32],
            execution_config: [7; 32],
        },
        input,
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        observed_at_ns: 1,
        wall_ns: 1000,
    }
}

#[test]
fn installed_completion_requires_settlement_and_preserves_known_length_boundaries() {
    let input = prepared(0, false, true);
    assert!(matches!(
        input.validate_actual_completion(),
        Err(StructuredUnknown::MissingEvidence)
    ));
    let done = input.clone().with_settled_completion(&[1]).unwrap();
    done.validate(&StructuredSettingsV2::default()).unwrap();
    done.validate_actual_completion().unwrap();
    assert_eq!(done.completion.as_ref().unwrap().positions, [1]);
    assert_ne!(input.regression_axes(), done.regression_axes());
    assert_eq!(input.owner(), done.owner());
    assert_eq!(input.domain_signature(), done.domain_signature());
    assert!(prepared(0, true, true)
        .with_settled_completion(&[0])
        .is_err());
    assert!(prepared(0, true, true)
        .with_settled_completion(&[0, 1])
        .is_ok());
    assert!(prepared(0, false, false)
        .with_settled_completion(&[0])
        .is_err());
    assert!(input.clone().with_settled_completion(&[1, 1]).is_err());
    assert!(input.with_settled_completion(&[2]).is_err());
}

#[test]
fn length_completion_cannot_authorize_unobserved_early_stop_or_cross_phase_reuse() {
    let query = prepared(0, false, true);
    let continuation = observation(query.clone().with_settled_completion(&[]).unwrap());
    let early = observation(query.clone().with_settled_completion(&[0, 1]).unwrap());
    let at_length = observation(
        prepared(0, true, true)
            .with_settled_completion(&[0, 1])
            .unwrap(),
    );
    let one_phase = CompletionCoverage::observed(&[continuation.clone(), early]).unwrap();
    assert!(one_phase.authorize(&query).is_ok());
    let missing_early = CompletionCoverage::observed(&[continuation, at_length]).unwrap();
    assert!(matches!(
        one_phase.intersect(missing_early).authorize(&query),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    assert!(CompletionCoverage::observed(&[observation(query)]).is_err());
}

#[test]
fn signed_completion_and_pending_envelope_contains_every_reachable_combination() {
    let inputs = (0..32)
        .map(|i| {
            prepared(i & 3, false, true)
                .with_settled_completion(&positions((i >> 2) & 3))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let wall = |input: &StructuredInputV2| {
        let c = &input.completion.as_ref().unwrap().positions;
        1000 + 31 * i64::from(c.contains(&0)) - 19 * i64::from(c.contains(&1))
            + input.pending_positions.len() as i64 * 7
    };
    let rows = inputs
        .iter()
        .map(|input| FitRow {
            basis: &input.basis,
            wall_ns: wall(input) as u64,
        })
        .collect::<Vec<_>>();
    let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
    let generators = PendingGenerators::new(&fit, &inputs[0]).unwrap();
    let query = prepared(0, false, true);
    let bounds = generators
        .bounds(
            &fit,
            &query,
            Some(&super::super::input::PendingQuery {
                eligible: vec![0, 1],
                constraint: HostPendingConstraintV2::AnySubset,
            }),
        )
        .unwrap();
    assert!(bounds.lower_ns.abs_diff(981) <= 2);
    assert!(bounds.upper_ns.abs_diff(1045) <= 2);
    for input in &inputs {
        let actual = fit.predict(&input.basis).unwrap();
        assert!(actual + 2 >= bounds.lower_ns && actual <= bounds.upper_ns + 2);
        assert!(input
            .support
            .iter()
            .zip(&bounds.support_lower)
            .all(|(v, l)| v >= l));
        assert!(input
            .support
            .iter()
            .zip(&bounds.support_upper)
            .all(|(v, h)| v <= h));
    }
}

#[test]
fn unobserved_completion_direction_cannot_acquire_zero_cost() {
    let inputs = (0..16)
        .map(|_| {
            prepared(0, false, true)
                .with_settled_completion(&[])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let rows = inputs
        .iter()
        .map(|input| FitRow {
            basis: &input.basis,
            wall_ns: 1000,
        })
        .collect::<Vec<_>>();
    let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
    let generators = PendingGenerators::new(&fit, &inputs[0]).unwrap();
    assert!(matches!(
        generators.bounds(&fit, &prepared(0, false, true), None),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
}

#[test]
fn installed_natural_policy_three_fresh_phases_predict_unresolved_next_wave() {
    let contract = StructuredSourceContractV2 {
        capture_identity: [1; 32],
        protocol: [2; 32],
        membership_rule: [3; 32],
        cohort_manifest: [9; 32],
        phase_members: [16; 3],
    };
    let phase = |which: StructuredPhaseV2, offset: usize| {
        (0..16)
            .map(|i| {
                let terminal = (i >> 2) & 3;
                let input = prepared((i & 3) as u32, false, true)
                    .with_settled_completion(&positions(terminal as u32))
                    .unwrap();
                let mut s = observation(input);
                let n = (offset + i + 1) as u64;
                s.ordinal = n * 3;
                s.call_id = n;
                s.observed_at_ns = n * 10;
                s.membership = StructuredMemberBindingV2 {
                    rule_signature: [3; 32],
                    offered_ordinal: n * 4,
                    member_ordinal: n,
                    phase: which,
                };
                s.wall_ns = (1000 + 31 * i64::from(terminal & 1 != 0)
                    - 19 * i64::from(terminal & 2 != 0)
                    + 7 * (i & 3).count_ones() as i64) as u64;
                s
            })
            .collect::<Vec<_>>()
    };
    let fit = phase(StructuredPhaseV2::Fit, 0);
    let residual = phase(StructuredPhaseV2::Residual, 16);
    let qualification = phase(StructuredPhaseV2::Qualification, 32);
    let fp = fit[0].fingerprint.clone();
    let scope = StructuredScopeV2 {
        owner: fit[0].input.owner().clone(),
        numerical_family: None,
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![0, 1],
            authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
            pending_counts: vec![0, 1, 2],
            length_counts: vec![0],
            pending_positions: vec![0, 1],
            length_positions: vec![],
            joint_counts: vec![(0, 0), (1, 0), (2, 0)],
        },
    };
    let settings = StructuredSettingsV2 {
        max_sample_age_ns: 10_000,
        static_margin_ns: 20,
        ..Default::default()
    };
    let qualified = FittedStructuredModelV2::fit(fp.clone(), settings, scope, contract, &fit, 170)
        .unwrap()
        .calibrate(&residual, 330)
        .unwrap()
        .qualify(&qualification, 490)
        .unwrap();
    let signature = qualified.parameters_signature();
    let query = StructuredQueryV2 {
        input: prepared(0, false, true),
        pending: Some(super::super::input::PendingQuery {
            eligible: vec![0, 1],
            constraint: HostPendingConstraintV2::AnySubset,
        }),
        repetition_upper_sum: None,
    };
    let prediction = qualified.predict_query(&fp, &query, 500).unwrap();
    assert!(prediction.fitted_lower_ns.abs_diff(981) <= 2);
    assert!(prediction.fitted_upper_ns.abs_diff(1045) <= 2);
    assert!(prediction.planning_ns >= 1065);
    assert_eq!(qualified.parameters_signature(), signature);
}

#[test]
fn repetition_work_interval_has_signed_extrema_and_requires_observed_direction() {
    for coefficient in [-7i64, 7] {
        let inputs = (0..16)
            .map(|i| {
                prepared_with_repetition(0, false, false, Some(1 + i % 2))
                    .with_settled_completion(&[])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let rows = inputs
            .iter()
            .map(|input| {
                let (_, s) = input.repetition_offsets.unwrap();
                FitRow {
                    basis: &input.basis,
                    wall_ns: (1000 + coefficient * input.support[s] as i64) as u64,
                }
            })
            .collect::<Vec<_>>();
        let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
        let generators = PendingGenerators::new(&fit, &inputs[0]).unwrap();
        let input = prepared_with_repetition(0, false, false, Some(1));
        let point = generators.bounds(&fit, &input, None).unwrap();
        let bound = generators
            .extend_repetition(&input, Some(4), point)
            .unwrap();
        let a = (1000 + coefficient * 2) as u64;
        let b = (1000 + coefficient * 4) as u64;
        assert!(bound.lower_ns.abs_diff(a.min(b)) <= 2);
        assert!(bound.upper_ns.abs_diff(a.max(b)) <= 2);
    }
    let input = prepared_with_repetition(0, false, false, Some(1))
        .with_settled_completion(&[])
        .unwrap();
    let rows = (0..16)
        .map(|_| FitRow {
            basis: &input.basis,
            wall_ns: 1000,
        })
        .collect::<Vec<_>>();
    let fit = RowSpaceFit::fit(&rows, &StructuredSettingsV2::default()).unwrap();
    let generators = PendingGenerators::new(&fit, &input).unwrap();
    let point = generators.bounds(&fit, &input, None).unwrap();
    assert!(matches!(
        generators.extend_repetition(&input, Some(4), point),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
}

fn cause_input(length: bool, model_eos: bool, user_stop: bool) -> StructuredInputV2 {
    let wave = canonical(0, length, model_eos, user_stop, None);
    let selected = wave.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
    )
    .unwrap()
}

#[test]
fn settled_terminal_causes_respect_installed_policy_and_actual_capacity() {
    use ferrum_types::FinishReason;
    for model_eos in [false, true] {
        for user_stop in [false, true] {
            let input = cause_input(false, model_eos, user_stop);
            for (reason, allowed) in [
                (FinishReason::EOS, model_eos),
                (FinishReason::Stop, user_stop),
                (FinishReason::Length, false),
            ] {
                let result = input.clone().with_settled_terminal_causes(&[(0, reason)]);
                assert_eq!(
                    result.is_ok(),
                    allowed,
                    "EOS={model_eos} Stop={user_stop} reason={reason:?}"
                );
                if let Ok(settled) = result {
                    assert_eq!(
                        settled.settled_terminal_causes(),
                        Some([(0, reason)].as_slice())
                    );
                    settled.validate_actual_completion().unwrap();
                }
            }
            // No terminal receipt is a real continuation, not missing cause data.
            let continued = input.with_settled_terminal_causes(&[]).unwrap();
            assert_eq!(continued.settled_terminal_causes(), Some([].as_slice()));
            assert!(continued.completion.as_ref().unwrap().positions.is_empty());
            continued.validate_actual_completion().unwrap();
        }
    }
    let at_capacity = cause_input(true, false, false);
    assert!(at_capacity
        .clone()
        .with_settled_terminal_causes(&[])
        .is_err());
    assert!(at_capacity
        .clone()
        .with_settled_terminal_causes(&[(0, FinishReason::Length)])
        .is_err());
    let lengths = at_capacity
        .with_settled_terminal_causes(&[(0, FinishReason::Length), (1, FinishReason::Length)])
        .unwrap();
    assert_eq!(lengths.completion.as_ref().unwrap().positions, [0, 1]);
    assert_eq!(
        lengths.settled_terminal_causes(),
        Some([(0, FinishReason::Length), (1, FinishReason::Length)].as_slice())
    );
    // Capacity completion cannot satisfy the early-EOS challenge of a below-cap
    // query, even when EOS is enabled at both observed and future frontiers.
    let lengths = cause_input(true, true, false)
        .with_settled_terminal_causes(&[(0, FinishReason::Length), (1, FinishReason::Length)])
        .unwrap();
    let continuation = cause_input(false, true, false)
        .with_settled_terminal_causes(&[])
        .unwrap();
    let coverage =
        CompletionCoverage::observed(&[observation(lengths), observation(continuation)]).unwrap();
    assert!(matches!(
        coverage.authorize(&cause_input(false, true, false)),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}

#[test]
fn settled_terminal_causes_reject_duplicate_unsorted_and_foreign_positions() {
    use ferrum_types::FinishReason;
    let input = cause_input(false, true, true);
    for causes in [
        vec![(0, FinishReason::EOS), (0, FinishReason::EOS)],
        vec![(0, FinishReason::EOS), (0, FinishReason::Stop)],
        vec![(1, FinishReason::EOS), (0, FinishReason::Stop)],
        vec![(2, FinishReason::EOS)],
    ] {
        assert!(input.clone().with_settled_terminal_causes(&causes).is_err());
    }
    let valid = input
        .with_settled_terminal_causes(&[(0, FinishReason::EOS), (1, FinishReason::Stop)])
        .unwrap();
    valid.validate(&StructuredSettingsV2::default()).unwrap();
    valid.validate_actual_completion().unwrap();
}

#[test]
fn terminal_cause_metadata_preserves_original_settled_numeric_projection() {
    use ferrum_types::FinishReason;
    for pending in 0..4 {
        for terminal_bits in 0..4 {
            let input = prepared(pending, false, true);
            let terminal = positions(terminal_bits);
            let old = input.clone().with_settled_completion(&terminal).unwrap();
            let causes = terminal
                .iter()
                .map(|p| (*p, FinishReason::EOS))
                .collect::<Vec<_>>();
            let new = input.with_settled_terminal_causes(&causes).unwrap();
            assert_eq!(old.settled_terminal_causes(), None);
            assert_eq!(new.settled_terminal_causes(), Some(causes.as_slice()));
            assert_eq!(old.regression_axes(), new.regression_axes());
            assert_eq!(
                old.joint_support_coordinates(),
                new.joint_support_coordinates()
            );
            assert_eq!(old.owner(), new.owner());
            assert_eq!(old.domain_signature(), new.domain_signature());
            assert_eq!(old.completion, new.completion);
        }
    }
}

#[test]
fn actual_and_future_domain_projection_keep_original_recipe_binding_and_limits() {
    use std::num::{NonZeroU32, NonZeroU64};
    let wave = canonical(0, false, true, false, None);
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let identity = ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    };
    let limits = CostWorkloadLimitsV1 {
        maximum_rows: NonZeroU32::new(2).unwrap(),
        maximum_context_tokens: NonZeroU32::new(128).unwrap(),
        maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
        output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
        repetition_slot_capacity: 32,
        fixed_state_bytes_per_row: 32,
    };
    let domain = CostWorkloadDomainV1::new_vnext(&identity, limits).unwrap();
    let legacy = StructuredInputV2::from_actual(&wave.exact, selected, recipe).unwrap();
    let actual =
        StructuredInputV2::from_actual_with_domain(&wave.exact, selected, recipe, &domain).unwrap();
    let future = StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        recipe,
        &HostContentForecastV2::Exact,
        &domain,
    )
    .unwrap();
    assert_eq!(legacy.physical_domain_signature(), None);
    assert_eq!(actual.physical_domain_signature(), Some(domain.sha256()));
    assert_eq!(
        future.input.physical_domain_signature(),
        Some(domain.sha256())
    );
    assert_eq!(legacy.regression_axes(), actual.regression_axes());
    assert_eq!(legacy.domain_signature(), actual.domain_signature());
    assert_eq!(actual.regression_axes(), future.input.regression_axes());
    assert_eq!(
        actual.joint_support_coordinates(),
        future.input.joint_support_coordinates()
    );
    for change in 0..4 {
        let mut limits = limits;
        match change {
            0 => limits.maximum_rows = NonZeroU32::new(1).unwrap(),
            1 => limits.maximum_context_tokens = NonZeroU32::new(64).unwrap(),
            2 => limits.maximum_scheduled_tokens_per_wave = NonZeroU64::new(1).unwrap(),
            3 => limits.fixed_state_bytes_per_row = 0,
            _ => unreachable!(),
        }
        let restricted = CostWorkloadDomainV1::new_vnext(&identity, limits).unwrap();
        assert!(matches!(
            StructuredInputV2::from_actual_with_domain(&wave.exact, selected, recipe, &restricted),
            Err(StructuredUnknown::WrongDomain)
        ));
        assert!(matches!(
            StructuredQueryV2::from_future_with_domain(
                &wave.exact,
                selected,
                recipe,
                &HostContentForecastV2::Exact,
                &restricted
            ),
            Err(StructuredUnknown::WrongDomain)
        ));
    }
    let mut larger_limits = limits;
    larger_limits.maximum_context_tokens = NonZeroU32::new(256).unwrap();
    let larger = CostWorkloadDomainV1::new_vnext(&identity, larger_limits).unwrap();
    assert!(larger.matches_execution_identity(&identity));
    assert!(!larger.matches_runtime_domain(&domain));
    let larger_input =
        StructuredInputV2::from_actual_with_domain(&wave.exact, selected, recipe, &larger).unwrap();
    assert_ne!(
        larger_input.physical_domain_signature(),
        actual.physical_domain_signature()
    );
    // A larger domain cannot repair a changed actual shape with an old recipe.
    let mut changed = wave.exact.clone();
    changed.rows[0] = ActualRowWork::Decode { kv_tokens: 65 };
    assert!(
        StructuredInputV2::from_actual_with_domain(&changed, selected, recipe, &larger).is_err()
    );
    assert!(StructuredQueryV2::from_future_with_domain(
        &changed,
        selected,
        recipe,
        &HostContentForecastV2::Exact,
        &larger
    )
    .is_err());
}
