//! Continuation-only natural-EOS observations train an empirical work model.
//! These controlled walls prove protocol behavior, never EOS-tail coverage.
#[test]
fn prospective_completion_continuous_decode_advances_context_across_all_phases() {
    let populations: [Vec<StructuredNumericObservationV2>; 3] = std::array::from_fn(|phase| {
        let phase = [
            StructuredPhaseV2::Fit,
            StructuredPhaseV2::Residual,
            StructuredPhaseV2::Qualification,
        ][phase];
        let mut values = population(phase);
        for (i, sample) in values.iter_mut().enumerate() {
            let offset = phase.index() as u32 * 24 + i as u32;
            let mut w = natural(1);
            w.generated = 3 + u64::from(offset);
            let wave = w.build_with_kv_tokens(64 + offset);
            sample.input = fixture::checked(&wave)
                .unwrap()
                .original_input()
                .clone()
                .with_settled_terminal_causes(&[])
                .unwrap()
                .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
                .unwrap();
            sample.wall_ns = 1000 + u64::from(64 + offset) * 3;
        }
        values
    });
    let [f, r, q] = &populations;
    let fitted = FittedStructuredModelV2::fit_service_window(
        f[0].fingerprint.clone(),
        settings(),
        scope(&f[0].input, true),
        declared(GLOBAL),
        close(StructuredPhaseV2::Fit, f),
        f,
        240,
    )
    .unwrap();
    for sample in r {
        assert_eq!(
            fitted.service_input_membership(&sample.input).unwrap(),
            StructuredServiceInputMembershipV1::Eligible
        );
    }
    let calibrated = calibrate(fitted, r).unwrap();
    for sample in q {
        assert_eq!(
            calibrated.service_input_membership(&sample.input).unwrap(),
            StructuredServiceInputMembershipV1::Eligible
        );
    }
    let model = qualify(calibrated, q).unwrap();
    // The next token extends both context and host history beyond every Q row;
    // this estimator explicitly permits magnitude extrapolation within its
    // finite declared domain, with empirical Q/feedback rather than a guarantee.
    let mut w = natural(1);
    w.generated = 75;
    let wave = w.build_with_kv_tokens(136);
    let selected = wave.statistical.as_ref().unwrap();
    let query = StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &HostContentForecastV2::Exact,
        &fixture::domain(),
    )
    .unwrap();
    let prediction = model.predict_query(&f[0].fingerprint, &query, 730).unwrap();
    assert_eq!(model.source_contract().phase_members, [24; 3]);
    assert!(prediction.planning_ns >= 1000 + 136 * 3);
}
use super::*;
use ferrum_types::FinishReason;

const GLOBAL: NonNegativePlanningEstimatorV1 =
    NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1;

fn settings() -> StructuredSettingsV2 {
    StructuredSettingsV2 {
        max_sample_age_ns: 10_000,
        static_margin_ns: 100,
        ..Default::default()
    }
}
fn natural(rows: u32) -> fixture::Wave {
    let mut wave = fixture::Wave {
        rows,
        ..Default::default()
    };
    wave.policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextInstalledV2(
        PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::Greedy {
                repetition_penalty: false,
            },
            model_eos: true,
            user_stop: true,
        },
    ));
    wave
}
fn input(wave: &fixture::Wave, causes: &[(u32, FinishReason)]) -> StructuredInputV2 {
    fixture::checked(&wave.build())
        .unwrap()
        .original_input()
        .clone()
        .with_settled_terminal_causes(causes)
        .unwrap()
        .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
        .unwrap()
}
fn population(phase: StructuredPhaseV2) -> Vec<StructuredNumericObservationV2> {
    let mut values = samples(phase, false);
    for (index, sample) in values.iter_mut().enumerate() {
        let rows = [1, 2, 4, 8][index % 4];
        sample.input = input(&natural(rows), &[]);
        sample.wall_ns = 1000 + u64::from(rows) * 10;
    }
    values
}
fn declared(estimator: NonNegativePlanningEstimatorV1) -> StructuredServiceWindowContractV2 {
    let mut value = contract(StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1);
    value
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = estimator;
    value
}
fn fitted(estimator: NonNegativePlanningEstimatorV1) -> Result<FittedStructuredModelV2> {
    let f = population(StructuredPhaseV2::Fit);
    FittedStructuredModelV2::fit_service_window(
        f[0].fingerprint.clone(),
        settings(),
        scope(&f[0].input, true),
        declared(estimator),
        close(StructuredPhaseV2::Fit, &f),
        &f,
        240,
    )
}
fn qualified() -> QualifiedStructuredModelV2 {
    qualify(
        calibrate(
            fitted(GLOBAL).unwrap(),
            &population(StructuredPhaseV2::Residual),
        )
        .unwrap(),
        &population(StructuredPhaseV2::Qualification),
    )
    .unwrap()
}
fn future(wave: &fixture::Wave) -> StructuredQueryV2 {
    let wave = wave.build();
    let selected = wave.statistical.as_ref().unwrap();
    StructuredQueryV2::from_future_with_domain(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &HostContentForecastV2::Exact,
        &fixture::domain(),
    )
    .unwrap()
}

#[test]
fn prospective_completion_old_branch_contract_rejects_continuation_only_fit() {
    let f = population(StructuredPhaseV2::Fit);
    let coverage = ChallengeCoverage::observe(&f).unwrap();
    assert_eq!(
        coverage.validate_branches(&f),
        Err(StructuredUnknown::QualificationCoverage)
    );
    for estimator in [
        NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
        NonNegativePlanningEstimatorV1::FittedResidualV1,
    ] {
        assert!(matches!(
            fitted(estimator),
            Err(StructuredUnknown::QualificationCoverage)
        ));
    }
}

#[test]
fn prospective_completion_independent_phases_qualify_run_and_batched_inputs() {
    let model = qualified();
    assert_eq!(model.source_contract().phase_members, [24; 3]);
    let fp = population(StructuredPhaseV2::Fit)[0].fingerprint.clone();
    // One request and a physical batch use the same canonical producer and
    // model contract; neither supplies the future terminal outcome.
    for rows in [1, 4] {
        let wave = natural(rows);
        let continuing = input(&wave, &[]);
        let eos = input(&wave, &[(0, FinishReason::EOS)]);
        let stop = input(&wave, &[(0, FinishReason::Stop)]);
        assert_ne!(continuing.regression_axes(), eos.regression_axes());
        assert!(continuing.settled_terminal_causes().unwrap().is_empty());
        assert_eq!(
            eos.settled_terminal_causes().unwrap(),
            &[(0, FinishReason::EOS)]
        );
        let prediction = model.predict_query(&fp, &future(&wave), 730).unwrap();
        for actual in [continuing, eos, stop] {
            let settled = model
                .predict_query(&fp, &StructuredQueryV2::exact(actual), 730)
                .unwrap();
            assert_eq!(settled.fitted_upper_ns, prediction.fitted_upper_ns);
            assert_eq!(
                settled.planning_ns, prediction.planning_ns,
                "retrospective feedback must compare the same pre-execution estimator"
            );
            assert_eq!(settled.valid_until_ns, prediction.valid_until_ns);
        }
    }
    let coverage = model.calibrated.fitted.numerical.physical().unwrap();
    for phase in &coverage.coverage {
        let audit = phase.as_ref().unwrap().diagnose_query_coverage(
            &future(&natural(1)),
            &envelope::prospective_query_upper(&future(&natural(1)), &fixture::domain()).unwrap(),
        );
        assert_eq!(audit["phase_branches"]["early"], false);
        assert_eq!(audit["phase_branches"]["continuation"], true);
        assert_eq!(audit["phase_branches"]["early_cause_mask"], 0);
    }
}

#[test]
fn prospective_completion_frozen_certificate_replays_only_identical_features() {
    let f = population(StructuredPhaseV2::Fit);
    let model = fitted(GLOBAL).unwrap();
    let replay = |certificate| {
        FittedStructuredModelV2::fit_service_window_from_certificate(
            f[0].fingerprint.clone(),
            settings(),
            scope(&f[0].input, true),
            declared(GLOBAL),
            close(StructuredPhaseV2::Fit, &f),
            &f,
            240,
            certificate,
        )
    };
    let restored = replay(model.nonnegative_fit_certificate().unwrap().clone()).unwrap();
    assert_eq!(
        restored.parameters_signature(),
        model.parameters_signature()
    );
    // A numerically valid old actual-completion Fit is not a certificate for
    // the new input variables, even with the same raw samples and wall labels.
    let raw: Vec<_> = f
        .iter()
        .map(|s| envelope::axes(&s.input).unwrap())
        .collect();
    let old_samples: Vec<_> = raw
        .iter()
        .zip(&f)
        .map(|(axes, s)| FitSample {
            axes,
            wall_ns: s.wall_ns,
        })
        .collect();
    let old =
        NonNegativeFit::fit_identified(&old_samples, &settings(), Default::default()).unwrap();
    assert_ne!(
        old.certificate().input_digest,
        model.nonnegative_fit_certificate().unwrap().input_digest
    );
    assert!(replay(old.certificate().clone()).is_err());
}

#[test]
fn prospective_completion_slow_eos_stays_in_qualification_and_fails() {
    let fitted = fitted(GLOBAL).unwrap();
    let calibrated = calibrate(fitted, &population(StructuredPhaseV2::Residual)).unwrap();
    let mut q = population(StructuredPhaseV2::Qualification);
    q[0].input = input(&natural(1), &[(0, FinishReason::EOS)]);
    q[0].wall_ns = 1_000_000;
    // The EOS result may neither exclude the wave nor select a new phase cut.
    assert_eq!(
        calibrated.service_input_membership(&q[0].input).unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    assert_eq!(q.len(), 24);
    assert!(matches!(
        qualify(calibrated, &q),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn prospective_completion_keeps_policy_route_domain_and_age_boundaries() {
    let model = qualified();
    let fp = population(StructuredPhaseV2::Fit)[0].fingerprint.clone();
    let original = future(&natural(1));
    let predicted = model.predict_query(&fp, &original, 730).unwrap();
    assert!(matches!(
        model.predict_query(&fp, &original, predicted.valid_until_ns + 1),
        Err(StructuredUnknown::Stale)
    ));
    let mut changed_policy = natural(1);
    changed_policy.policy.categorical_signature = [99; 32];
    assert!(model
        .predict_query(&fp, &future(&changed_policy), 730)
        .is_err());
    let mut changed_route = natural(1);
    changed_route.algorithm = "fixture.unknown.algorithm";
    assert!(model
        .predict_query(&fp, &future(&changed_route), 730)
        .is_err());
    let wave = natural(1).build();
    let selected = wave.statistical.as_ref().unwrap();
    let unscoped = StructuredQueryV2::from_future(
        &wave.exact,
        selected,
        selected.structured_capture().unwrap().unwrap(),
        &HostContentForecastV2::Exact,
    )
    .unwrap();
    assert!(model.predict_query(&fp, &unscoped, 730).is_err());
    let malformed = fixture::checked(&wave)
        .unwrap()
        .original_input()
        .clone()
        .with_settled_completion(&[0])
        .unwrap()
        .with_cost_template_policy(StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1)
        .unwrap();
    let mut q = population(StructuredPhaseV2::Qualification);
    q[0].input = malformed;
    let calibrated = calibrate(
        fitted(GLOBAL).unwrap(),
        &population(StructuredPhaseV2::Residual),
    )
    .unwrap();
    assert!(
        qualify(calibrated, &q).is_err(),
        "unbound terminal cause is still invalid evidence"
    );
}
