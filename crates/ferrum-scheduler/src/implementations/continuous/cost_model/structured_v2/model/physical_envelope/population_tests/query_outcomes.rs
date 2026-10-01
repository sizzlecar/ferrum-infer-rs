//! Full Fit/Residual/Qualification using original typed canonical inputs.
//! Synthetic costs prove query contracts, not hardware performance.
use super::*;

fn query(wave: fixture::Wave, domain: Option<&CostWorkloadDomainV1>) -> StructuredQueryV2 {
    let wave = wave.build();
    let selected = wave.statistical.as_ref().unwrap();
    let recipe = selected.structured_capture().unwrap().unwrap();
    match domain {
        Some(domain) => StructuredQueryV2::from_future_with_domain(
            &wave.exact,
            selected,
            recipe,
            &HostContentForecastV2::Exact,
            domain,
        ),
        None => StructuredQueryV2::from_future(
            &wave.exact,
            selected,
            recipe,
            &HostContentForecastV2::Exact,
        ),
    }
    .unwrap()
}

fn qualified_with_cap(maximum: u64) -> (QualifiedStructuredModelV2, ExecutionFingerprint) {
    let f = samples(StructuredPhaseV2::Fit, false);
    let r = samples(StructuredPhaseV2::Residual, false);
    let q = samples(StructuredPhaseV2::Qualification, false);
    let mut contract = contract(StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1);
    contract
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1;
    let fitted = FittedStructuredModelV2::fit_service_window(
        f[0].fingerprint.clone(),
        StructuredSettingsV2 {
            max_wave_ns: maximum,
            max_sample_age_ns: 10_000,
            static_margin_ns: 100,
            ..Default::default()
        },
        scope(&f[0].input, true),
        contract,
        close(StructuredPhaseV2::Fit, &f),
        &f,
        240,
    )
    .unwrap();
    (
        qualify(calibrate(fitted, &r).unwrap(), &q).unwrap(),
        f[0].fingerprint.clone(),
    )
}

#[test]
fn detailed_query_checked_width_history_and_branch_keep_the_original_region() {
    let (model, fingerprint) = qualified_with_cap(60_000_000_000);
    let domain = fixture::domain();
    let original = query(fixture::Wave::default(), Some(&domain));
    let before = model
        .predict_query_detailed(&fingerprint, &original, 730)
        .unwrap();
    for wave in [
        fixture::Wave {
            rows: 3,
            ..Default::default()
        },
        fixture::Wave {
            rows: 8,
            generated: 31,
            ..Default::default()
        },
    ] {
        let input = query(wave, Some(&domain));
        assert!(model
            .predict_query_detailed(&fingerprint, &input, 730)
            .is_ok());
    }
    for wave in [
        fixture::Wave {
            mask: vec![0],
            ..Default::default()
        },
        fixture::Wave {
            pending: vec![0],
            ..Default::default()
        },
        fixture::Wave {
            length: vec![0],
            ..Default::default()
        },
    ] {
        let input = query(wave, Some(&domain));
        let failure = model
            .predict_query_detailed(&fingerprint, &input, 730)
            .unwrap_err();
        assert_eq!(
            failure,
            StructuredQueryFailureV2::OutsideSupport(StructuredUnknown::QualificationCoverage)
        );
        assert_eq!(
            model.predict_query(&fingerprint, &input, 730).unwrap_err(),
            failure.reason()
        );
        // Negative control: invalid numerical evidence cannot be hidden by
        // the otherwise legitimate missing physical branch.
        let mut invalid = input.clone();
        invalid.input.basis[0] = f64::NAN;
        assert_eq!(
            model
                .predict_query_detailed(&fingerprint, &invalid, 730)
                .unwrap_err(),
            StructuredQueryFailureV2::Invalid(StructuredUnknown::InvalidInput)
        );
        assert_eq!(
            model
                .predict_query_detailed(&fingerprint, &input, 20_000)
                .unwrap_err(),
            StructuredQueryFailureV2::Invalid(StructuredUnknown::Stale)
        );
    }
    assert_eq!(
        model
            .predict_query_detailed(&fingerprint, &original, 730)
            .unwrap()
            .planning_ns,
        before.planning_ns
    );
}

#[test]
fn detailed_query_legal_large_history_is_range_unknown_not_damaged_evidence() {
    let (model, fingerprint) = qualified_with_cap(5_000);
    let domain = fixture::domain();
    let original = query(fixture::Wave::default(), Some(&domain));
    let before = model
        .predict_query_detailed(&fingerprint, &original, 730)
        .unwrap();
    // The original physical domain allows this history. Every positive axis
    // was observed; only the coefficient envelope exceeds the declared cap.
    let larger = query(
        fixture::Wave {
            generated: 63,
            ..Default::default()
        },
        Some(&domain),
    );
    assert_eq!(
        larger.input().numerical_family_key(),
        original.input().numerical_family_key()
    );
    let failure = model
        .predict_query_detailed(&fingerprint, &larger, 730)
        .unwrap_err();
    assert_eq!(
        failure,
        StructuredQueryFailureV2::OutsidePredictionRange(StructuredUnknown::Capacity)
    );
    assert_eq!(
        model.predict_query(&fingerprint, &larger, 730).unwrap_err(),
        failure.reason()
    );
    assert_eq!(
        model
            .predict_query_detailed(&fingerprint, &original, 730)
            .unwrap()
            .planning_ns,
        before.planning_ns
    );
}

#[test]
fn detailed_query_bad_domain_and_identity_never_become_coverage_exclusions() {
    let (model, fingerprint) = qualified_with_cap(60_000_000_000);
    let domain = fixture::domain();
    let mut limits = *domain.limits();
    limits.maximum_context_tokens = std::num::NonZeroU32::new(1025).unwrap();
    let other_domain = CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: fingerprint.model_weights,
            numerical_policy: fingerprint.numerical_policy,
            device_runtime: fingerprint.device_runtime,
            execution_config: fingerprint.execution_config,
        },
        limits,
    )
    .unwrap();
    let original = query(fixture::Wave::default(), Some(&domain));
    for bound in [None, Some(&other_domain)] {
        let invalid = query(fixture::Wave::default(), bound);
        assert_eq!(invalid.owner(), original.owner());
        assert_eq!(
            model
                .predict_query_detailed(&fingerprint, &invalid, 730)
                .unwrap_err(),
            StructuredQueryFailureV2::Invalid(StructuredUnknown::WrongDomain)
        );
    }
    let mut wrong_fingerprint = fingerprint.clone();
    wrong_fingerprint.device_runtime = [99; 32];
    assert_eq!(
        model
            .predict_query_detailed(&wrong_fingerprint, &original, 730)
            .unwrap_err(),
        StructuredQueryFailureV2::Invalid(StructuredUnknown::WrongFingerprint)
    );
}
