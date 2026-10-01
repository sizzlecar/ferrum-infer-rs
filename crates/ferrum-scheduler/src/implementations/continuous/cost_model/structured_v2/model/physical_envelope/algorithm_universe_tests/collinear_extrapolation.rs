//! Synthetic, predeclared cost laws expose an identification ambiguity. These
//! are canonical original shapes and the real three-phase numerical APIs,
//! not GPU timings or proof that any observed hardware followed these laws.
use super::*;
use std::num::NonZeroU64;

mod independent_geometry;

fn input(rows: u32, kv: u32) -> StructuredInputV2 {
    let wave = fixture::Wave {
        rows,
        algorithm: "A",
        ..Default::default()
    }
    .build_with_kv_tokens(kv);
    fixture::checked(&wave)
        .unwrap()
        .original_input()
        .clone()
        .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
        .unwrap()
}

// All are nonnegative linear functions of existing work features: intercept,
// decode-row count, and attention_pairs. At B1/KV64 all equal 1000 ns.
fn compatible_costs(rows: u32, kv: u32) -> [u64; 3] {
    [
        1000,
        1000 * u64::from(rows),
        (1000 * u64::from(rows) * (u64::from(kv) + 1)).div_ceil(65),
    ]
}

fn settings() -> StructuredSettingsV2 {
    let mut settings = StructuredSettingsV2 {
        max_sample_age_ns: 10_000,
        static_margin_ns: 100,
        ..Default::default()
    };
    // Automatic's declared span policy cannot learn missing input dependence
    // from three constant populations: its observed span is exactly zero.
    settings.learned_drift = StructuredLearnedDriftV2::ObservedResidualSpanV1 {
        maximum_span_margin_ns: NonZeroU64::new(settings.max_wave_ns).unwrap(),
    };
    settings
}

fn source_contract(u: &DeclaredAlgorithmUniverseV1) -> StructuredServiceWindowContractV2 {
    source_contract_for(u, NonNegativePlanningEstimatorV1::FittedResidualV1, 24)
}

fn source_contract_for(
    u: &DeclaredAlgorithmUniverseV1,
    estimator: NonNegativePlanningEstimatorV1,
    members: usize,
) -> StructuredServiceWindowContractV2 {
    let mut envelope = contract(u);
    envelope.planning_estimator = estimator;
    StructuredServiceWindowContractV2 {
        capture_identity: [10; 32],
        protocol: [11; 32],
        membership_rule: [12; 32],
        window_declaration: [13; 32],
        phase_offered: [members; 3],
        domain_policy: StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1,
        nonnegative_envelope: Some(envelope),
    }
}

fn populations(u: &DeclaredAlgorithmUniverseV1) -> [Vec<StructuredNumericObservationV2>; 3] {
    [
        StructuredPhaseV2::Fit,
        StructuredPhaseV2::Residual,
        StructuredPhaseV2::Qualification,
    ]
    .map(|phase| {
        let (_, mut samples) = fixture::population(phase, false);
        for sample in &mut samples {
            sample.input = input(1, 64).with_algorithm_universe(u).unwrap();
            sample.wall_ns = compatible_costs(1, 64)[0];
            sample.membership.member_ordinal = sample.ordinal;
        }
        samples
    })
}

fn fitted(
    u: &DeclaredAlgorithmUniverseV1,
    fit: &[StructuredNumericObservationV2],
) -> FittedStructuredModelV2 {
    fitted_using(u, fit, NonNegativePlanningEstimatorV1::FittedResidualV1)
}

fn fitted_using(
    u: &DeclaredAlgorithmUniverseV1,
    fit: &[StructuredNumericObservationV2],
    estimator: NonNegativePlanningEstimatorV1,
) -> FittedStructuredModelV2 {
    FittedStructuredModelV2::fit_service_window(
        fit[0].fingerprint.clone(),
        settings(),
        scope(&fit[0].input),
        source_contract_for(u, estimator, fit.len()),
        close(StructuredPhaseV2::Fit, fit),
        fit,
        fit.last().unwrap().observed_at_ns,
    )
    .unwrap()
}

fn finish(
    fit: FittedStructuredModelV2,
    phases: &[Vec<StructuredNumericObservationV2>; 3],
) -> QualifiedStructuredModelV2 {
    fit.calibrate_service_window(
        close(StructuredPhaseV2::Residual, &phases[1]),
        &phases[1],
        phases[1].last().unwrap().observed_at_ns,
    )
    .unwrap()
    .qualify_service_window(
        close(StructuredPhaseV2::Qualification, &phases[2]),
        &phases[2],
        phases[2].last().unwrap().observed_at_ns,
    )
    .unwrap()
}

fn repeated_populations(
    u: &DeclaredAlgorithmUniverseV1,
    members: usize,
) -> [Vec<StructuredNumericObservationV2>; 3] {
    let original = populations(u);
    std::array::from_fn(|phase| {
        (0..members)
            .map(|i| {
                // Fresh call and membership ordinals in disjoint phases. Only the
                // input geometry and predeclared synthetic duration are repeated.
                let mut sample = original[phase][i % original[phase].len()].clone();
                let ordinal = (phase * members + i + 1) as u64;
                sample.ordinal = ordinal;
                sample.call_id = ordinal;
                sample.membership.offered_ordinal = ordinal;
                sample.membership.member_ordinal = ordinal;
                sample.observed_at_ns = ordinal * 10;
                sample
            })
            .collect()
    })
}

#[test]
fn fitted_residual_collinear_coefficient_envelope_covers_compatible_laws_without_duplicate_confidence(
) {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let queries = [(1, 64), (8, 64), (1, 511), (8, 511)];
    let mut previous = None;
    for members in [24, 48] {
        let phases = repeated_populations(&u, members);
        for pair in phases.windows(2) {
            assert!(pair[0].last().unwrap().ordinal < pair[1][0].ordinal);
            assert!(pair[0].last().unwrap().call_id < pair[1][0].call_id);
        }
        let mut results = Vec::new();
        for estimator in [
            NonNegativePlanningEstimatorV1::FittedResidualV1,
            NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1,
        ] {
            let fit = fitted_using(&u, &phases[0], estimator);
            let certificate = fit.nonnegative_fit_certificate().unwrap().clone();
            assert_eq!(
                certificate.geometry_rank, 1,
                "duplicates add no input direction"
            );
            assert_eq!(certificate.epsilon_ns, 0);
            let replay_fit = FittedStructuredModelV2::fit_service_window_from_certificate(
                phases[0][0].fingerprint.clone(),
                settings(),
                scope(&phases[0][0].input),
                source_contract_for(&u, estimator, members),
                close(StructuredPhaseV2::Fit, &phases[0]),
                &phases[0],
                phases[0].last().unwrap().observed_at_ns,
                certificate,
            )
            .unwrap();
            let model = finish(fit, &phases);
            let replay = finish(replay_fit, &phases);
            assert_eq!(model.parameters_signature(), replay.parameters_signature());
            let now = phases[2].last().unwrap().observed_at_ns + 10;
            for (rows, kv) in queries {
                let query = StructuredQueryV2::exact(input(rows, kv));
                let original = query.input().clone();
                let prediction = model
                    .predict_query(&phases[0][0].fingerprint, &query, now)
                    .unwrap();
                let replay_prediction = replay
                    .predict_query(&phases[0][0].fingerprint, &query, now)
                    .unwrap();
                assert_eq!(prediction.planning_ns, replay_prediction.planning_ns);
                assert_eq!(query.input(), &original);
                assert_eq!(prediction.fit_samples, members);
                assert_eq!(prediction.residual_samples, members);
                assert_eq!(prediction.residual_ns, 0);
                assert_eq!(prediction.fit_error_floor_ns, 0);
                assert_eq!(prediction.learned_span_margin_ns, 0);
                let costs = compatible_costs(rows, kv);
                let expected_base = match estimator {
                    NonNegativePlanningEstimatorV1::FittedResidualV1 => 1000,
                    NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
                    | NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
                    | NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1 => {
                        unreachable!("this comparison preserves the two historical estimators")
                    }
                    // Existing query certificates cover these nonnegative
                    // linear laws; this is not a hardware upper-bound claim.
                    NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1 => {
                        *costs.iter().max().unwrap()
                    }
                };
                assert_eq!(prediction.fitted_upper_ns, expected_base);
                assert_eq!(
                    prediction.planning_ns,
                    expected_base + settings().static_margin_ns
                );
                if estimator == NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1 {
                    assert!(costs.iter().all(|&cost| cost <= prediction.fitted_upper_ns));
                } else if (rows, kv) != (1, 64) {
                    assert!(costs.iter().any(|&cost| cost > prediction.planning_ns));
                }
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "event": "collinear_estimator_comparison_v1", "phase_members": members,
                        "estimator": estimator, "rows": rows, "kv_tokens": kv,
                        "geometry_rank": prediction.identified_rank,
                        "compatible_costs_ns": costs,
                        "base_planning_ns": prediction.fitted_upper_ns,
                        "planning_ns": prediction.planning_ns,
                    })
                );
                results.push((prediction.fitted_upper_ns, prediction.planning_ns));
            }
        }
        if let Some(previous) = &previous {
            assert_eq!(&results, previous,
                "duplicating the same original input geometry must not invent confidence or narrow the ambiguity envelope");
        }
        previous = Some(results);
    }
}

#[test]
fn fitted_residual_collinear_characterization_qualifies_and_replays_but_cannot_identify_unseen_scaling(
) {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let phases = populations(&u);
    for pair in phases.windows(2) {
        assert!(pair[0].last().unwrap().ordinal < pair[1][0].ordinal);
        assert!(pair[0].last().unwrap().call_id < pair[1][0].call_id);
    }
    let schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        24,
        [24; 3],
        [24; 3],
        OwnerInputReadinessV1::new_cached_residual_v2([1; 3], 32_000_000).unwrap(),
    )
    .unwrap();
    let target = OwnerInputTargetV1::from_samples(&phases[0]).unwrap();
    let mut visits = 0;
    assert_eq!(
        schedule
            .assess_inputs(
                StructuredPhaseV2::Fit,
                24,
                &phases[0],
                Some(&target),
                &settings(),
                &mut visits
            )
            .unwrap(),
        OwnerInputReadinessDecisionV1::Freeze
    );

    let fit = fitted(&u, &phases[0]);
    let certificate = fit.nonnegative_fit_certificate().unwrap().clone();
    assert_eq!(certificate.geometry_rank, 1);
    assert_eq!(certificate.epsilon_ns, 0);
    let replay_fit = FittedStructuredModelV2::fit_service_window_from_certificate(
        phases[0][0].fingerprint.clone(),
        settings(),
        scope(&phases[0][0].input),
        source_contract(&u),
        close(StructuredPhaseV2::Fit, &phases[0]),
        &phases[0],
        240,
        certificate,
    )
    .unwrap();
    let model = finish(fit, &phases);
    let replay = finish(replay_fit, &phases);
    assert_eq!(model.parameters_signature(), replay.parameters_signature());
    assert_eq!(compatible_costs(1, 64), [1000; 3]);
    for (rows, kv) in [(1, 64), (8, 64), (1, 511), (8, 511)] {
        let query = StructuredQueryV2::exact(input(rows, kv));
        let original = query.input().clone();
        assert_eq!(
            original.numerical_family_key_for_universe(&u).unwrap(),
            phases[0][0].input.numerical_family_key().unwrap()
        );
        let prediction = model
            .predict_query(&phases[0][0].fingerprint, &query, 730)
            .unwrap();
        let independent = replay
            .predict_query(&phases[0][0].fingerprint, &query, 730)
            .unwrap();
        assert_eq!(prediction.planning_ns, independent.planning_ns);
        assert_eq!(
            query.input(),
            &original,
            "numeric authorization does not alter execution input"
        );
        assert_eq!(prediction.fitted_upper_ns, 1000);
        assert_eq!(prediction.residual_ns, 0);
        assert_eq!(prediction.learned_span_margin_ns, 0);
        assert_eq!(prediction.planning_ns, 1100);
        let costs = compatible_costs(rows, kv);
        assert!(costs[0] <= prediction.planning_ns);
        if (rows, kv) != (1, 64) {
            assert!(
                costs.iter().any(|&cost| cost > prediction.planning_ns),
                "equally compatible nonnegative cost laws disagree outside the observed geometry"
            );
        }
    }
    // The guard does work when the independent Qualification population
    // actually includes the missing direction. It is not a phase bypass.
    let mut challenged = phases[2].clone();
    challenged[0].input = input(8, 511).with_algorithm_universe(&u).unwrap();
    challenged[0].wall_ns = compatible_costs(8, 511)[2];
    let calibrated = fitted(&u, &phases[0])
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &phases[1]),
            &phases[1],
            480,
        )
        .unwrap();
    assert!(matches!(
        calibrated.qualify_service_window(
            close(StructuredPhaseV2::Qualification, &challenged),
            &challenged,
            720,
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

// The original red regression demanded Unknown. A verified coefficient-set
// upper bound is also valid; a narrow fitted point alone is not authority.
#[test]
fn identified_envelope_retains_uncertainty_for_unidentified_width_context() {
    let u = DeclaredAlgorithmUniverseV1::from_inputs([&input(1, 64)], 4096).unwrap();
    let phases = populations(&u);
    let estimator = NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2;
    let fit = fitted_using(&u, &phases[0], estimator);
    let certificate = fit.nonnegative_fit_certificate().unwrap().clone();
    assert!(certificate.signed_basis.is_some());
    let replay_fit = FittedStructuredModelV2::fit_service_window_from_certificate(
        phases[0][0].fingerprint.clone(),
        settings(),
        scope(&phases[0][0].input),
        source_contract_for(&u, estimator, 24),
        close(StructuredPhaseV2::Fit, &phases[0]),
        &phases[0],
        240,
        certificate,
    )
    .unwrap();
    let model = finish(fit, &phases);
    let replay = finish(replay_fit, &phases);
    assert_eq!(model.parameters_signature(), replay.parameters_signature());
    let old_envelope = finish(
        fitted_using(
            &u,
            &phases[0],
            NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1,
        ),
        &phases,
    );
    for (rows, kv) in [(1, 64), (8, 64), (1, 511), (8, 511)] {
        let query = StructuredQueryV2::exact(input(rows, kv));
        let prediction = model
            .predict_query(&phases[0][0].fingerprint, &query, 730)
            .unwrap();
        let replayed = replay
            .predict_query(&phases[0][0].fingerprint, &query, 730)
            .unwrap();
        let prior = old_envelope
            .predict_query(&phases[0][0].fingerprint, &query, 730)
            .unwrap();
        assert_eq!(prediction.planning_ns, replayed.planning_ns);
        assert!(
            compatible_costs(rows, kv)
                .into_iter()
                .all(|cost| cost <= prediction.planning_ns),
            "unidentified directions retain parameter uncertainty"
        );
        assert!(
            prediction.planning_ns <= prior.planning_ns,
            "the additional certificate cannot weaken an existing upper bound"
        );
    }
}
