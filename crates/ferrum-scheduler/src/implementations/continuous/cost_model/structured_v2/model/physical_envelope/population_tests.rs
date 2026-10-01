//! Full numerical phase transitions; source8 receipt/cohort authority is tested
//! by its collector. These synthetic walls make no hardware performance claim.
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
#[path = "numerical_family_tests/fixture.rs"]
mod fixture;
#[path = "population_tests/prospective_completion.rs"]
mod prospective_completion;
#[path = "population_tests/query_outcomes.rs"]
mod query_outcomes;

fn samples(phase: StructuredPhaseV2, mask: bool) -> Vec<StructuredNumericObservationV2> {
    let (_, mut samples) = fixture::population(phase, mask);
    for sample in &mut samples {
        sample.membership.member_ordinal = sample.ordinal;
        sample.input = sample
            .input
            .clone()
            .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
            .unwrap();
    }
    samples
}

fn scope(input: &StructuredInputV2, family: bool) -> StructuredScopeV2 {
    StructuredScopeV2 {
        owner: input.owner().clone(),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: Vec::new(),
            authorized_pending_constraints: Vec::new(),
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: Vec::new(),
            length_positions: Vec::new(),
            joint_counts: vec![(0, 0)],
        },
        numerical_family: family.then(|| input.numerical_family_key().unwrap()),
    }
}

fn contract(policy: StructuredPopulationPolicyV1) -> StructuredServiceWindowContractV2 {
    StructuredServiceWindowContractV2 {
        capture_identity: [10; 32],
        protocol: [11; 32],
        membership_rule: [12; 32],
        window_declaration: [13; 32],
        phase_offered: [24; 3],
        domain_policy: StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1,
        nonnegative_envelope: Some(NonNegativeEnvelopeContractV1 {
            algorithm_universe: None,
            planning_estimator: NonNegativePlanningEstimatorV1::FittedResidualV1,
            workload_domain: fixture::domain(),
            settings: Default::default(),
            challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
            template_policy: StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
            population_policy: policy,
        }),
    }
}

fn close(
    phase: StructuredPhaseV2,
    samples: &[StructuredNumericObservationV2],
) -> StructuredServiceWindowCloseV2 {
    StructuredServiceWindowCloseV2::new(phase, [20 + phase.index() as u8; 32], samples)
}

fn fit(
    samples: &[StructuredNumericObservationV2],
    scope: StructuredScopeV2,
    policy: StructuredPopulationPolicyV1,
) -> Result<FittedStructuredModelV2> {
    FittedStructuredModelV2::fit_service_window(
        samples[0].fingerprint.clone(),
        StructuredSettingsV2 {
            max_sample_age_ns: 10_000,
            static_margin_ns: 100,
            ..Default::default()
        },
        scope,
        contract(policy),
        close(StructuredPhaseV2::Fit, samples),
        samples,
        240,
    )
}

fn calibrate(
    fit: FittedStructuredModelV2,
    samples: &[StructuredNumericObservationV2],
) -> Result<CalibratedStructuredModelV2> {
    fit.calibrate_service_window(close(StructuredPhaseV2::Residual, samples), samples, 480)
}

fn qualify(
    model: CalibratedStructuredModelV2,
    samples: &[StructuredNumericObservationV2],
) -> Result<QualifiedStructuredModelV2> {
    model.qualify_service_window(
        close(StructuredPhaseV2::Qualification, samples),
        samples,
        720,
    )
}

#[test]
fn homogeneous_population_runs_all_phases_and_queries_original_other_width() {
    let fit_samples = samples(StructuredPhaseV2::Fit, true);
    let residual = samples(StructuredPhaseV2::Residual, true);
    let heldout = samples(StructuredPhaseV2::Qualification, true);
    // Real B8 discovery representative; first Fit member is original B2.
    let scope = scope(&fit_samples[2].input, true);
    let family = scope.numerical_family.unwrap();
    let fitted = fit(
        &fit_samples,
        scope,
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
    )
    .unwrap();
    let model = qualify(calibrate(fitted, &residual).unwrap(), &heldout).unwrap();
    assert_eq!(model.owner().rows, 8);
    assert_eq!(model.numerical_family_key(), Some(&family));
    assert_eq!(*model.domain_signature(), family.signature().unwrap());
    assert_ne!(
        model.domain_signature(),
        fit_samples[0].input.domain_signature()
    );

    let wave = fixture::Wave {
        rows: 3,
        ..Default::default()
    }
    .build();
    let selected = wave.statistical.as_ref().unwrap();
    let query = StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &HostContentForecastV2::Exact,
        &fixture::domain(),
    )
    .unwrap();
    assert_eq!(query.owner().rows, 3);
    assert!(model
        .predict_query(&fit_samples[0].fingerprint, &query, 730)
        .is_ok());
    assert!(matches!(
        model.predict_query(&fit_samples[0].fingerprint, &query, 20_000),
        Err(StructuredUnknown::Stale)
    ));
    assert!(matches!(
        model.scope().coverage_report(&heldout),
        Err(StructuredUnknown::WrongProtocol)
    ));
}

#[test]
fn startup_extension_preserves_qualified_work_directions_before_replacement() {
    let qualified = |mask| {
        let inputs = samples(StructuredPhaseV2::Fit, mask);
        let fitted = fit(
            &inputs,
            scope(&inputs[0].input, true),
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
        )
        .unwrap();
        qualify(
            calibrate(fitted, &samples(StructuredPhaseV2::Residual, mask)).unwrap(),
            &samples(StructuredPhaseV2::Qualification, mask),
        )
        .unwrap()
    };
    let clean = qualified(false);
    let masked = qualified(true);
    assert_eq!(clean.numerical_family_key(), masked.numerical_family_key());
    assert!(masked.preserves_physical_input_coverage(&clean));
    assert!(masked.preserves_physical_input_coverage(&masked));
    assert!(!clean.preserves_physical_input_coverage(&masked));

    // Both candidates qualified independently, but the clean-only candidate
    // would lose this previously supported, physically checked query.
    let wave = fixture::Wave {
        rows: 3,
        mask: vec![0],
        ..Default::default()
    }
    .build();
    let selected = wave.statistical.as_ref().unwrap();
    let query = StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &HostContentForecastV2::Exact,
        &fixture::domain(),
    )
    .unwrap();
    let fingerprint = &samples(StructuredPhaseV2::Fit, true)[0].fingerprint;
    assert!(masked.predict_query(fingerprint, &query, 730).is_ok());
    assert!(matches!(
        clean.predict_query(fingerprint, &query, 730),
        Err(StructuredUnknown::QualificationCoverage)
            | Err(StructuredUnknown::UnidentifiedDirection)
    ));
}

#[test]
fn numerical_family_requires_declared_policy_and_rejects_changed_identity() {
    let fit_samples = samples(StructuredPhaseV2::Fit, true);
    assert!(matches!(
        fit(
            &fit_samples,
            scope(&fit_samples[0].input, true),
            StructuredPopulationPolicyV1::ExactOwnerV1
        ),
        Err(StructuredUnknown::WrongProtocol)
    ));
    assert!(matches!(
        fit(
            &fit_samples,
            scope(&fit_samples[0].input, false),
            StructuredPopulationPolicyV1::ExactOwnerV1
        ),
        Err(StructuredUnknown::WrongDomain)
    ));
    let mut changed = fit_samples.clone();
    let wave = fixture::Wave {
        algorithm: "fixture.changed",
        ..Default::default()
    }
    .build();
    changed[1].input = fixture::checked(&wave).unwrap().original_input().clone();
    assert!(matches!(
        fit(
            &changed,
            scope(&fit_samples[0].input, true),
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
        ),
        Err(StructuredUnknown::WrongDomain)
    ));
}

#[test]
fn pooled_population_preserves_mask_challenges_and_qualification_failures() {
    let policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    let clean_fit = samples(StructuredPhaseV2::Fit, false);
    let masked_fit = samples(StructuredPhaseV2::Fit, true);
    let masked_residual = samples(StructuredPhaseV2::Residual, true);
    let fitted = fit(&clean_fit, scope(&clean_fit[0].input, true), policy).unwrap();
    assert!(matches!(
        calibrate(fitted, &masked_residual),
        Err(StructuredUnknown::UnidentifiedDirection)
    ));
    let fitted = fit(&masked_fit, scope(&masked_fit[0].input, true), policy).unwrap();
    assert!(matches!(
        calibrate(fitted, &samples(StructuredPhaseV2::Residual, false)),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let fitted = fit(&masked_fit, scope(&masked_fit[0].input, true), policy).unwrap();
    let calibrated = calibrate(fitted, &masked_residual).unwrap();
    let mut heldout = samples(StructuredPhaseV2::Qualification, true);
    heldout[0].wall_ns += 1_000_000;
    assert!(matches!(
        qualify(calibrated, &heldout),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn new_policy_keeps_unsupported_first_decode_on_original_exact_population() {
    let wave = fixture::Wave {
        generated: 0,
        ..Default::default()
    }
    .build();
    let selected = wave.statistical.as_ref().unwrap();
    let input = StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &fixture::domain(),
    )
    .unwrap()
    .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
    .unwrap();
    assert!(matches!(
        input.numerical_family_key(),
        Err(StructuredUnknown::UnsupportedScope)
    ));
    let population = |phase| {
        let mut values = samples(phase, false);
        for sample in &mut values {
            sample.input = input.clone();
            sample.wall_ns = 1000;
        }
        values
    };
    let fit_samples = population(StructuredPhaseV2::Fit);
    let fitted = fit(
        &fit_samples,
        scope(&input, false),
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
    )
    .unwrap();
    let calibrated = calibrate(fitted, &population(StructuredPhaseV2::Residual)).unwrap();
    let model = qualify(calibrated, &population(StructuredPhaseV2::Qualification)).unwrap();
    assert_eq!(model.numerical_family_key(), None);
    assert_eq!(model.domain_signature(), input.domain_signature());
    assert!(model
        .predict_query(
            &fit_samples[0].fingerprint,
            &StructuredQueryV2::exact(input),
            730
        )
        .is_ok());
}

#[test]
fn population_policy_defaults_preserve_wire_omission_and_bind_new_semantics() {
    let old = contract(StructuredPopulationPolicyV1::ExactOwnerV1)
        .nonnegative_envelope
        .unwrap();
    let old_json = serde_json::to_value(&old).unwrap();
    assert!(old_json.get("population_policy").is_none());
    let replay: NonNegativeEnvelopeContractV1 = serde_json::from_value(old_json).unwrap();
    assert_eq!(old, replay);
    let mut changed = old.clone();
    changed.population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    let mut a = Sha256::new();
    let mut b = Sha256::new();
    old.bind(&mut a);
    changed.bind(&mut b);
    assert_ne!(a.finalize(), b.finalize());
    let fit_samples = samples(StructuredPhaseV2::Fit, false);
    let exact = scope(&fit_samples[0].input, false);
    assert!(serde_json::to_value(&exact)
        .unwrap()
        .get("numerical_family")
        .is_none());
    let family = scope(&fit_samples[0].input, true);
    let restored: StructuredScopeV2 =
        serde_json::from_value(serde_json::to_value(&family).unwrap()).unwrap();
    assert_eq!(restored, family);
}
