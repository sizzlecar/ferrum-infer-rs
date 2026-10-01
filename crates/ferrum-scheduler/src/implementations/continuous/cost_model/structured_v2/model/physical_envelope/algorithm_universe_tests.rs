//! Real canonical producers, synthetic whole-wave walls, independent phases.
//! These tests establish protocol/numerical behavior, not hardware accuracy.
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
mod collinear_extrapolation;
#[path = "numerical_family_tests/fixture.rs"]
mod fixture;

fn original(algorithms: &[&'static str], rows: u32, mask: bool) -> StructuredInputV2 {
    let wave = fixture::Wave {
        rows,
        algorithm: algorithms[0],
        extra_algorithms: algorithms[1..].to_vec(),
        mask: if mask { vec![0] } else { Vec::new() },
        ..Default::default()
    }
    .build();
    fixture::checked(&wave).unwrap().original_input().clone()
}
fn universe() -> DeclaredAlgorithmUniverseV1 {
    DeclaredAlgorithmUniverseV1::from_inputs(
        [&original(&["A"], 2, false), &original(&["B"], 8, true)],
        4096,
    )
    .unwrap()
}
fn contract(universe: &DeclaredAlgorithmUniverseV1) -> NonNegativeEnvelopeContractV1 {
    NonNegativeEnvelopeContractV1 {
        planning_estimator: NonNegativePlanningEstimatorV1::FittedResidualV1,
        algorithm_universe: Some(universe.clone()),
        workload_domain: fixture::domain(),
        settings: Default::default(),
        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
        template_policy: StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
        population_policy: StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
    }
}
fn samples(
    phase: StructuredPhaseV2,
    u: &DeclaredAlgorithmUniverseV1,
    mixed: bool,
    mask: bool,
) -> Vec<StructuredNumericObservationV2> {
    let (_, mut out) = fixture::population(phase, false);
    for (i, sample) in out.iter_mut().enumerate() {
        let rows = [2, 4, 8][i % 3];
        let algorithms: &[&str] = if mixed {
            &["A", "B"]
        } else if i % 2 == 0 {
            &["A"]
        } else {
            &["B"]
        };
        let pending_mask = mask && i % 6 == 0;
        sample.input = original(algorithms, rows, pending_mask)
            .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
            .unwrap()
            .with_algorithm_universe(u)
            .unwrap();
        sample.wall_ns =
            1000 + u64::from(rows) * 10 * algorithms.len() as u64 + u64::from(pending_mask) * 7;
        sample.membership.member_ordinal = sample.ordinal;
    }
    out
}
fn scope(input: &StructuredInputV2) -> StructuredScopeV2 {
    StructuredScopeV2 {
        owner: input.owner().clone(),
        numerical_family: Some(input.numerical_family_key().unwrap()),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: Vec::new(),
            authorized_pending_constraints: Vec::new(),
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: Vec::new(),
            length_positions: Vec::new(),
            joint_counts: vec![(0, 0)],
        },
    }
}
fn close(
    phase: StructuredPhaseV2,
    samples: &[StructuredNumericObservationV2],
) -> StructuredServiceWindowCloseV2 {
    StructuredServiceWindowCloseV2::new(phase, [20 + phase.index() as u8; 32], samples)
}
fn fit(
    u: &DeclaredAlgorithmUniverseV1,
    samples: &[StructuredNumericObservationV2],
) -> Result<FittedStructuredModelV2> {
    FittedStructuredModelV2::fit_service_window(
        samples[0].fingerprint.clone(),
        StructuredSettingsV2 {
            max_sample_age_ns: 10_000,
            static_margin_ns: 100,
            ..Default::default()
        },
        scope(&samples[0].input),
        StructuredServiceWindowContractV2 {
            capture_identity: [10; 32],
            protocol: [11; 32],
            membership_rule: [12; 32],
            window_declaration: [13; 32],
            phase_offered: [24; 3],
            domain_policy: StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1,
            nonnegative_envelope: Some(contract(u)),
        },
        close(StructuredPhaseV2::Fit, samples),
        samples,
        240,
    )
}

#[test]
fn algorithm_universe_aligns_real_subset_work_without_changing_original_identity() {
    let u = universe();
    let originals = [
        original(&["A"], 2, false),
        original(&["B"], 3, false),
        original(&["A", "B"], 4, false),
        original(&["B", "A"], 4, false),
    ];
    assert_ne!(
        originals[0].numerical_family_key().unwrap(),
        originals[1].numerical_family_key().unwrap()
    );
    let mapped: Vec<_> = originals
        .iter()
        .map(|i| i.clone().with_algorithm_universe(&u).unwrap())
        .collect();
    for (raw, input) in originals.iter().zip(&mapped) {
        assert_eq!(
            raw.numerical_family_key_for_universe(&u).unwrap(),
            input.numerical_family_key().unwrap()
        );
        assert_eq!(raw.owner(), input.owner());
        assert_eq!(raw.domain_signature(), input.domain_signature());
        assert_eq!(raw.physical_host_rows(), input.physical_host_rows());
        assert_eq!(
            input.numerical_family_key().unwrap(),
            mapped[0].numerical_family_key().unwrap()
        );
        assert_eq!(
            input.retained_payload_bytes(),
            u.projected_input_retained_bytes(raw)
        );
    }
    assert_eq!(mapped[2].regression_axes(), mapped[3].regression_axes());
    assert_eq!(
        mapped[2].joint_support_coordinates(),
        mapped[3].joint_support_coordinates()
    );
    assert!(matches!(
        original(&["A", "C"], 3, false).with_algorithm_universe(&u),
        Err(StructuredUnknown::WrongDomain)
    ));
    let reversed =
        DeclaredAlgorithmUniverseV1::from_inputs([&originals[1], &originals[0]], 4096).unwrap();
    assert_eq!(u, reversed);
    assert_eq!(u.signature(), reversed.signature());
    let wire = serde_json::to_value(&u).unwrap();
    assert_eq!(
        serde_json::from_value::<DeclaredAlgorithmUniverseV1>(wire.clone()).unwrap(),
        u
    );
    let mut bad = wire;
    bad["algorithms"].as_array_mut().unwrap().reverse();
    assert!(serde_json::from_value::<DeclaredAlgorithmUniverseV1>(bad).is_err());
    assert!(matches!(
        DeclaredAlgorithmUniverseV1::from_inputs(originals.iter(), 16),
        Err(StructuredUnknown::Capacity)
    ));
    assert!(DeclaredAlgorithmUniverseBuilderV1::new(4096, 1).is_err());
}

#[test]
fn algorithm_universe_known_composition_uses_independent_residual_and_qualification() {
    let u = universe();
    let fitting = samples(StructuredPhaseV2::Fit, &u, false, true);
    let residual = samples(StructuredPhaseV2::Residual, &u, true, true);
    let heldout = samples(StructuredPhaseV2::Qualification, &u, true, true);
    let model = fit(&u, &fitting)
        .unwrap()
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            480,
        )
        .unwrap()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            720,
        )
        .unwrap();
    // A genuinely fresh width/composition query is projected by the model;
    // no replacement owner, settled outcome, or manually adjusted hash is used.
    let query = StructuredQueryV2::exact(original(&["B", "A"], 3, false));
    let prediction = model
        .predict_query(&fitting[0].fingerprint, &query, 730)
        .unwrap();
    assert!(prediction.planning_ns >= 1060);
    assert!(
        prediction.planning_ns <= 1360,
        "unexpected empirical error: {}",
        prediction.planning_ns
    );
    assert_eq!(prediction.fit_samples, 24);
    assert_eq!(prediction.residual_samples, 24);
    let unknown = StructuredQueryV2::exact(original(&["C"], 3, false));
    assert!(matches!(
        model.predict_query(&fitting[0].fingerprint, &unknown, 730),
        Err(StructuredUnknown::WrongDomain)
    ));
    // Reconstruct the independent numerical transitions from original inputs.
    let replay = fit(&u, &fitting)
        .unwrap()
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            480,
        )
        .unwrap()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            720,
        )
        .unwrap();
    assert_eq!(model.parameters_signature(), replay.parameters_signature());
    let mut nonlinear = heldout;
    nonlinear[0].wall_ns += 1_000_000;
    let candidate = fit(&u, &fitting)
        .unwrap()
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            480,
        )
        .unwrap();
    assert!(matches!(
        candidate.qualify_service_window(
            close(StructuredPhaseV2::Qualification, &nonlinear),
            &nonlinear,
            720
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn algorithm_universe_preserves_missing_direction_and_mask_challenges() {
    let u = universe();
    let clean = samples(StructuredPhaseV2::Fit, &u, false, false);
    let masked = samples(StructuredPhaseV2::Fit, &u, false, true);
    let residual = samples(StructuredPhaseV2::Residual, &u, true, true);
    assert!(matches!(
        fit(&u, &clean).unwrap().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            480
        ),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
    let clean_residual = samples(StructuredPhaseV2::Residual, &u, true, false);
    assert!(matches!(
        fit(&u, &masked).unwrap().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &clean_residual),
            &clean_residual,
            480
        ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let mut only_a = clean;
    for s in &mut only_a {
        s.input = original(&["A"], s.input.owner().rows, false)
            .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
            .unwrap()
            .with_algorithm_universe(&u)
            .unwrap();
    }
    // B is declared in U but has never been measured in Fit.
    assert!(matches!(
        fit(&u, &only_a).unwrap().calibrate_service_window(
            close(StructuredPhaseV2::Residual, &clean_residual),
            &clean_residual,
            480
        ),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
}

#[test]
fn algorithm_universe_none_preserves_old_contract_and_unsupported_exact_population() {
    let u = universe();
    let mut c = contract(&u);
    c.algorithm_universe = None;
    let wire = serde_json::to_value(&c).unwrap();
    assert!(wire.get("algorithm_universe").is_none());
    assert_eq!(
        serde_json::from_value::<NonNegativeEnvelopeContractV1>(wire).unwrap(),
        c
    );
    let wave = fixture::Wave {
        generated: 0,
        ..Default::default()
    }
    .build();
    let selected = wave.statistical.as_ref().unwrap();
    let first = StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &fixture::domain(),
    )
    .unwrap();
    assert_eq!(contract(&u).project_input(first.clone()).unwrap(), first);
    assert!(matches!(
        first.clone().with_algorithm_universe(&u),
        Err(StructuredUnknown::UnsupportedScope)
    ));
    let projected = original(&["A"], 2, false)
        .with_algorithm_universe(&u)
        .unwrap();
    assert!(matches!(
        c.validate_input(&projected),
        Err(StructuredUnknown::WrongDomain)
    ));
}

#[test]
fn algorithm_universe_actual_future_and_independent_replay_align_the_same_checked_work() {
    let u = universe();
    let wave = fixture::Wave {
        rows: 3,
        algorithm: "B",
        extra_algorithms: vec!["A"],
        mask: vec![1],
        ..Default::default()
    }
    .build();
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let actual = StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        recipe,
        &fixture::domain(),
    )
    .unwrap()
    .with_algorithm_universe(&u)
    .unwrap();
    let future = StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        recipe,
        &HostContentForecastV2::Exact,
        &fixture::domain(),
    )
    .unwrap()
    .with_algorithm_universe(&u)
    .unwrap();
    let algorithms: Vec<_> = recipe
        .algorithm_work()
        .unwrap()
        .entries()
        .iter()
        .map(|a| (*a.algorithm().signature(), a.kind(), a.commands(), a.work()))
        .collect();
    let (replayed, _) = StructuredInputV2::from_replay_parts(
        &wave.exact,
        selected,
        *recipe.device().ordered_template(),
        recipe.device().provider_grouped_template().copied(),
        StructuredProductV2::GreedyToken,
        recipe.device().readback(),
        recipe.physical_host_rows(),
        &algorithms,
        None,
        recipe.device().retries(),
    )
    .unwrap();
    let replayed = replayed
        .bind_validated_physical_domain(&wave.exact, &fixture::domain())
        .unwrap()
        .with_algorithm_universe(&u)
        .unwrap();
    assert_eq!(actual, replayed);
    assert_eq!(&actual, future.input());
    assert_eq!(
        actual.numerical_family_key().unwrap(),
        future
            .input()
            .numerical_family_key_for_universe(&u)
            .unwrap()
    );
    // Projection never replaces the original per-algorithm binding validation.
    let foreign = fixture::Wave {
        algorithm: "C",
        rows: 3,
        ..Default::default()
    }
    .build();
    let other = foreign.statistical.as_ref().unwrap();
    assert!(StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        other.structured_capture().unwrap().unwrap(),
        &fixture::domain()
    )
    .is_err());
}

#[test]
fn algorithm_universe_membership_distinguishes_new_primitive_from_invalid_domain() {
    let u = universe();
    let a = original(&["A"], 2, false);
    let combination = original(&["A", "B"], 3, false);
    let unknown = original(&["A", "C"], 3, false);
    assert_eq!(u.contains_checked_algorithms(&a), Ok(true));
    assert_eq!(u.contains_checked_algorithms(&combination), Ok(true));
    assert_eq!(u.contains_checked_algorithms(&unknown), Ok(false));
    assert!(matches!(
        unknown.numerical_family_key_for_universe(&u),
        Err(StructuredUnknown::WrongDomain)
    ));
    assert!(matches!(
        unknown.clone().with_algorithm_universe(&u),
        Err(StructuredUnknown::WrongDomain)
    ));

    let wave = fixture::Wave {
        algorithm: "C",
        ..Default::default()
    }
    .build();
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    let unbound = StructuredInputV2::from_actual(&wave.exact, selected, recipe).unwrap();
    assert_eq!(
        u.contains_checked_algorithms(&unbound),
        Err(StructuredUnknown::WrongDomain)
    );
    let other_domain = CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [9; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        *fixture::domain().limits(),
    )
    .unwrap();
    let wrong =
        StructuredInputV2::from_actual_with_domain(&wave.exact, selected, recipe, &other_domain)
            .unwrap();
    assert_eq!(
        u.contains_checked_algorithms(&wrong),
        Err(StructuredUnknown::WrongDomain)
    );

    let first_wave = fixture::Wave {
        generated: 0,
        ..Default::default()
    }
    .build();
    let first_selected = first_wave.statistical.as_ref().unwrap();
    let first = StructuredInputV2::from_actual_with_domain(
        &first_wave.exact,
        first_selected,
        first_selected.structured_capture().unwrap().unwrap(),
        &fixture::domain(),
    )
    .unwrap();
    assert_eq!(
        u.contains_checked_algorithms(&first),
        Err(StructuredUnknown::UnsupportedScope)
    );
    let different = DeclaredAlgorithmUniverseV1::from_inputs([&a], 4096).unwrap();
    let already_projected = a.with_algorithm_universe(&different).unwrap();
    assert_eq!(
        u.contains_checked_algorithms(&already_projected),
        Err(StructuredUnknown::WrongDomain)
    );
}
