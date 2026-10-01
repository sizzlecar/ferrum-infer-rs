//! Numerical envelope regression over typed canonical inputs and synthetic
//! timings. This tests the model's Cartesian support check, not VNext route
//! reachability: current VNext makes repetition work neutral under FullLogits.
//! No execution receipt, backend measurement, or reachable route is asserted.
use super::*;

fn input(repetition: u64, pending: bool) -> StructuredInputV2 {
    let mut command = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    command
        .kernel(
            SelectedAlgorithmClassV1::new("repetition.bounds.fixture", 1, [1; 32], [2; 32])
                .unwrap(),
            KernelNumericWorkV1 {
                logical_units: 1,
                padded_units: 1,
                inner_units_per_logical_unit: 1,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = command.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "repetition.bounds.fixture",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "numerical-fixture",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 1,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Decode { kv_tokens: 64 },
            output: CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: repetition,
                repetition_penalty_bits: 1.1f32.to_bits(),
            },
            host_policy_signature: [3; 32],
            mask_upload_required: false,
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextInstalledV2(
                        PlainTextPolicyCapabilityV2 {
                            sampling: PlainTextSamplingRouteV2::FullLogits,
                            model_eos: false,
                            user_stop: false,
                        },
                    )),
                    categorical_signature: [4; 32],
                    decoder_text_bytes_per_token: 8,
                    decoder_scratch_bytes_per_token: 16,
                    raw_token_bytes_bound: 8,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 4,
                    maximum_output_tokens: 20,
                    sampling_history_tokens: 4,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: pending,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    let wave = builder
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    let selected = wave.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
    )
    .unwrap()
}

fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [5; 32],
        numerical_policy: [6; 32],
        device_runtime: [7; 32],
        execution_config: [8; 32],
    }
}

fn model(include_joint_high: bool, repetition_coefficient: i64) -> QualifiedStructuredModelV2 {
    model_with_qualification_support(include_joint_high, repetition_coefficient, None)
}

fn model_with_qualification_support(
    include_joint_high: bool,
    repetition_coefficient: i64,
    qualification_upper: Option<u64>,
) -> QualifiedStructuredModelV2 {
    let source = StructuredSourceContractV2 {
        capture_identity: [10; 32],
        protocol: [11; 32],
        membership_rule: [12; 32],
        cohort_manifest: [13; 32],
        phase_members: [16; 3],
    };
    // Both individual directions are independently identified in either model.
    // Only the covered model includes a single complete high/high witness.
    let points = [(1, false), (3, false), (1, true), (3, true)];
    let points = &points[..if include_joint_high { 4 } else { 3 }];
    let phase = |phase: StructuredPhaseV2, offset: u64, points: &[(u64, bool)]| {
        (0..16)
            .map(|i| {
                let (repetition, pending) = points[i % points.len()];
                let n = offset + i as u64 + 1;
                StructuredObservationV2 {
                    source: source.capture_identity,
                    protocol: source.protocol,
                    ordinal: n * 3,
                    membership: StructuredMemberBindingV2 {
                        rule_signature: source.membership_rule,
                        offered_ordinal: n * 4,
                        member_ordinal: n,
                        phase,
                    },
                    call_id: n,
                    fingerprint: fingerprint(),
                    input: input(repetition, pending)
                        .with_settled_completion(&[])
                        .unwrap(),
                    boundary: CostBoundary::PreparationToHostSettledV1,
                    outcome: WaveObservationOutcome::Completed,
                    observed_at_ns: n * 10,
                    wall_ns: (1000
                        + repetition_coefficient * repetition as i64
                        + 11 * i64::from(pending)) as u64,
                }
            })
            .collect::<Vec<_>>()
    };
    let fit = phase(StructuredPhaseV2::Fit, 0, points);
    let residual = phase(StructuredPhaseV2::Residual, 16, points);
    let qualification_points =
        qualification_upper.map(|upper| [(1, false), (upper, false), (1, true), (upper, true)]);
    let qualification = phase(
        StructuredPhaseV2::Qualification,
        32,
        qualification_points
            .as_ref()
            .map_or(points, |points| points.as_slice()),
    );
    let scope = StructuredScopeV2 {
        owner: fit[0].input.owner().clone(),
        numerical_family: None,
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![0],
            authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
            pending_counts: vec![0, 1],
            length_counts: vec![0],
            pending_positions: vec![0],
            length_positions: vec![],
            joint_counts: vec![(0, 0), (1, 0)],
        },
    };
    let settings = StructuredSettingsV2 {
        max_sample_age_ns: 10_000,
        static_margin_ns: 20,
        ..Default::default()
    };
    if qualification_upper.is_some() {
        let contract = StructuredServiceWindowContractV2 {
            capture_identity: source.capture_identity,
            protocol: source.protocol,
            membership_rule: source.membership_rule,
            window_declaration: source.cohort_manifest,
            phase_offered: [64; 3],
            domain_policy: StructuredServiceDomainPolicyV1::FrozenFitSupportV1,
            nonnegative_envelope: None,
        };
        let close = |phase: StructuredPhaseV2, samples: &[StructuredNumericObservationV2]| {
            StructuredServiceWindowCloseV2::new(phase, [phase.index() as u8 + 21; 32], samples)
        };
        FittedStructuredModelV2::fit_service_window(
            fingerprint(),
            settings,
            scope,
            contract,
            close(StructuredPhaseV2::Fit, &fit),
            &fit,
            170,
        )
        .unwrap()
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            330,
        )
        .unwrap()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &qualification),
            &qualification,
            490,
        )
        .unwrap()
    } else {
        FittedStructuredModelV2::fit(fingerprint(), settings, scope, source, &fit, 170)
            .unwrap()
            .calibrate(&residual, 330)
            .unwrap()
            .qualify(&qualification, 490)
            .unwrap()
    }
}

fn query(vary_pending: bool, repetition_upper: u64) -> StructuredQueryV2 {
    StructuredQueryV2 {
        input: input(1, false),
        pending: vary_pending.then(|| PendingQuery {
            eligible: vec![0],
            constraint: HostPendingConstraintV2::AnySubset,
        }),
        repetition_upper_sum: Some(repetition_upper),
    }
}

#[test]
fn numerical_repetition_pending_rectangle_rejects_separate_coordinate_witnesses() {
    for coefficient in [-7, 7] {
        let model = model(false, coefficient);
        // Each interval alone is identified, qualified, and jointly supported.
        // The combined error must be JointSupport, not an unidentifiable fit.
        for query in [query(false, 3), query(true, 1)] {
            assert!(model.predict_query(&fingerprint(), &query, 500).is_ok());
        }
        assert!(matches!(
            model.predict_query(&fingerprint(), &query(true, 3), 500),
            Err(StructuredUnknown::JointSupport)
        ));
    }
}

#[test]
fn numerical_repetition_pending_rectangle_accepts_complete_joint_witness() {
    for coefficient in [-7, 7] {
        let model = model(true, coefficient);
        let signature = model.parameters_signature();
        let prediction = model
            .predict_query(&fingerprint(), &query(true, 3), 500)
            .unwrap();
        let a = 1000 + coefficient;
        let b = 1000 + 3 * coefficient;
        assert!(prediction.fitted_lower_ns.abs_diff(a.min(b) as u64) <= 2);
        assert!(prediction.fitted_upper_ns.abs_diff((a.max(b) + 11) as u64) <= 2);
        assert_eq!(model.parameters_signature(), signature);
    }
}

#[test]
fn numerical_repetition_expansion_is_bounded_by_qualification_support() {
    for coefficient in [-7, 7] {
        // Fit and Residual cover the entire rectangle in both models. Only
        // the heldout phase differs; it cannot authorize its unseen upper edge.
        let narrow = model_with_qualification_support(true, coefficient, Some(1));
        let full = model_with_qualification_support(true, coefficient, Some(3));
        for pending in [false, true] {
            assert!(narrow
                .predict_query(&fingerprint(), &query(pending, 1), 500)
                .is_ok());
            assert!(full
                .predict_query(&fingerprint(), &query(pending, 3), 500)
                .is_ok());
            assert!(matches!(
                narrow.predict_query(&fingerprint(), &query(pending, 3), 500),
                Err(StructuredUnknown::JointSupport)
            ));
        }
    }
}
