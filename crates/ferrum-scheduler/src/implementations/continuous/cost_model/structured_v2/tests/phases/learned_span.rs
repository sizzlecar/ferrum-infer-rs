//! Real typed features with synthetic clocks; these are not live calibration receipts.
use super::*;
use std::num::NonZeroU64;

mod diagnostics;

fn policy(cap: u64) -> StructuredLearnedDriftV2 {
    StructuredLearnedDriftV2::ObservedResidualSpanV1 {
        maximum_span_margin_ns: NonZeroU64::new(cap).unwrap(),
    }
}
fn observations(kind: StructuredPhaseV2) -> Vec<StructuredNumericObservationV2> {
    let mut values = phase(kind);
    for value in &mut values {
        value.input = project(&wave(9, true, 8, "fixture.first", [false, false])).unwrap();
        value.wall_ns = 1000;
    }
    values
}
fn exact_scope(values: &[StructuredNumericObservationV2]) -> StructuredScopeV2 {
    StructuredScopeV2 {
        owner: values[0].input.owner().clone(),
        numerical_family: None,
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![],
            authorized_pending_constraints: vec![],
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: vec![],
            length_positions: vec![],
            joint_counts: vec![(0, 0)],
        },
    }
}
fn fit_with(settings: StructuredSettingsV2) -> FittedStructuredModelV2 {
    let fit = observations(StructuredPhaseV2::Fit);
    FittedStructuredModelV2::fit(fp(), settings, exact_scope(&fit), contract(), &fit, 170).unwrap()
}
fn calibrated_errors(a: i64, b: i64, cap: u64) -> Result<CalibratedStructuredModelV2> {
    let mut s = settings();
    s.learned_drift = policy(cap);
    let mut residual = observations(StructuredPhaseV2::Residual);
    for (i, sample) in residual.iter_mut().enumerate() {
        sample.wall_ns = (1000 + if i % 2 == 0 { a } else { b }) as u64;
    }
    fit_with(s).calibrate(&residual, 330)
}
#[test]
fn structured_v2_learned_span_signed_order_invariant_and_constant_shift() {
    for (a, b, span) in [(-30, -10, 20), (-10, 5, 15), (15, 15, 0)] {
        let left = calibrated_errors(a, b, 100)
            .unwrap()
            .qualify(&observations(StructuredPhaseV2::Qualification), 490)
            .unwrap();
        let right = calibrated_errors(b, a, 100)
            .unwrap()
            .qualify(&observations(StructuredPhaseV2::Qualification), 490)
            .unwrap();
        assert_eq!(left.uncertainty().learned_span_margin_ns, span);
        assert_eq!(left.uncertainty(), right.uncertainty());
        // A span includes negative errors. Positive-only residual remains the old q99.
        assert_eq!(left.uncertainty().residual_ns, a.max(b).max(0) as u64);
    }
    let mut qualification = observations(StructuredPhaseV2::Qualification);
    qualification[0].wall_ns = 1036; // 1000 + residual15 + static20 + span0 + 1.
    assert!(matches!(
        calibrated_errors(15, 15, 100)
            .unwrap()
            .qualify(&qualification, 490),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}
#[test]
fn structured_v2_learned_span_cap_boundary_and_heldout_never_refits() {
    assert!(matches!(
        calibrated_errors(-100, 100, 199),
        Err(StructuredUnknown::Capacity)
    ));
    let calibrated = calibrated_errors(-100, 100, 200).unwrap();
    let before = calibrated_errors(-100, 100, 200)
        .unwrap()
        .qualify(&observations(StructuredPhaseV2::Qualification), 490)
        .unwrap()
        .uncertainty();
    assert_eq!(before.learned_span_margin_ns, 200);
    let mut qualification = observations(StructuredPhaseV2::Qualification);
    qualification[7].wall_ns = 1320; // frozen fit1000 + q99100 + static20 + span200.
    let model = calibrated.qualify(&qualification, 490).unwrap();
    assert_eq!(model.uncertainty(), before);
    let query = StructuredQueryV2::exact(qualification[0].input.clone());
    let prediction = model.predict_query(&fp(), &query, 500).unwrap();
    assert_eq!(prediction.planning_ns, 1320);
    assert_eq!(prediction.learned_span_margin_ns, 200);
    assert_eq!(prediction.valid_until_ns, 10010);
    qualification[7].wall_ns += 1;
    assert!(matches!(
        calibrated_errors(-100, 100, 200)
            .unwrap()
            .qualify(&qualification, 490),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
    assert!(matches!(
        model.predict_query(&fp(), &query, 10011),
        Err(StructuredUnknown::Stale)
    ));
    let outside = project(&wave(9, true, 24, "fixture.first", [false, false])).unwrap();
    assert!(model
        .predict_query(&fp(), &StructuredQueryV2::exact(outside), 500)
        .is_err());
}
#[test]
fn structured_v2_learned_span_opt_in_identity_and_strict_limits() {
    let old = settings();
    let encoded = serde_json::to_value(&old).unwrap();
    assert!(encoded.get("learned_drift").is_none());
    let decoded: StructuredSettingsV2 = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(
        fit_with(old.clone()).parameters_signature(),
        fit_with(decoded).parameters_signature()
    );
    let mut explicit = encoded;
    explicit["learned_drift"] = serde_json::json!({"kind":"disabled"});
    let disabled: StructuredSettingsV2 = serde_json::from_value(explicit).unwrap();
    assert_eq!(
        fit_with(old.clone()).parameters_signature(),
        fit_with(disabled).parameters_signature()
    );
    let mut enabled = old.clone();
    enabled.learned_drift = policy(100);
    assert_ne!(
        fit_with(old.clone()).parameters_signature(),
        fit_with(enabled.clone()).parameters_signature()
    );
    let signature = fit_with(enabled.clone()).parameters_signature();
    enabled.learned_drift = policy(101);
    assert_ne!(signature, fit_with(enabled.clone()).parameters_signature());
    enabled.learned_drift = policy(enabled.max_wave_ns + 1);
    assert!(matches!(
        enabled.validate(),
        Err(StructuredUnknown::InvalidSettings)
    ));
    for value in [
        serde_json::json!({"kind":"disabled","maximum_span_margin_ns":1}),
        serde_json::json!({"kind":"observed_residual_span_v1","maximum_span_margin_ns":0}),
        serde_json::json!({"kind":"observed_residual_span_v1","maximum_span_margin_ns":1,"extra":true}),
    ] {
        assert!(serde_json::from_value::<StructuredLearnedDriftV2>(value).is_err());
    }
    let mut limited = old;
    limited.max_wave_ns = 1100;
    limited.learned_drift = policy(200);
    let mut residual = observations(StructuredPhaseV2::Residual);
    residual[0].wall_ns = 900;
    residual[1].wall_ns = 1100;
    let calibrated = fit_with(limited).calibrate(&residual, 330).unwrap();
    assert!(matches!(
        calibrated.qualify(&observations(StructuredPhaseV2::Qualification), 490),
        Err(StructuredUnknown::Numerical)
    ));
}
#[test]
fn structured_v2_learned_span_is_added_once_to_pending_envelope() {
    let fit = phase(StructuredPhaseV2::Fit);
    let residual = phase(StructuredPhaseV2::Residual);
    let mut enabled_settings = settings();
    enabled_settings.learned_drift = policy(100);
    let mut varied_residual = residual.clone();
    for (i, s) in varied_residual.iter_mut().enumerate() {
        s.wall_ns += if i % 2 == 0 { 5 } else { 35 };
    }
    let build = |s| {
        FittedStructuredModelV2::fit(fp(), s, scope(&fit), contract(), &fit, 170)
            .unwrap()
            .calibrate(&varied_residual, 330)
            .unwrap()
            .qualify(&phase(StructuredPhaseV2::Qualification), 490)
            .unwrap()
    };
    let old = build(settings());
    let new = build(enabled_settings);
    let query = StructuredQueryV2 {
        repetition_upper_sum: None,
        input: fit[0].input.clone(),
        pending: Some(PendingQuery {
            eligible: vec![0, 1],
            constraint: HostPendingConstraintV2::AnySubset,
        }),
    };
    let a = old.predict_query(&fp(), &query, 500).unwrap();
    let b = new.predict_query(&fp(), &query, 500).unwrap();
    assert!((29..=31).contains(&b.learned_span_margin_ns));
    assert_eq!(b.planning_ns, a.planning_ns + b.learned_span_margin_ns);
    assert_eq!(b.fitted_upper_ns, a.fitted_upper_ns);
    assert_eq!(b.valid_until_ns, a.valid_until_ns);
}

#[test]
fn structured_v2_learned_span_uses_frozen_multifeature_prediction_with_negative_slope() {
    let fit = super::fit_floor::varied(StructuredPhaseV2::Fit);
    let residual = super::fit_floor::varied(StructuredPhaseV2::Residual);
    let validation = super::fit_floor::varied(StructuredPhaseV2::Qualification);
    let mut enabled = settings();
    enabled.learned_drift = policy(300);
    let model =
        FittedStructuredModelV2::fit(fp(), enabled, exact_scope(&fit), contract(), &fit, 170)
            .unwrap()
            .calibrate(&residual, 330)
            .unwrap()
            .qualify(&validation, 490)
            .unwrap();
    let predictions = validation[..2]
        .iter()
        .map(|s| {
            model
                .predict_query(&fp(), &StructuredQueryV2::exact(s.input.clone()), 500)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(predictions[0].identified_rank, 2);
    assert!((4200..=4201).contains(&predictions[0].fitted_upper_ns));
    assert!((3400..=3401).contains(&predictions[1].fitted_upper_ns));
    // Residual errors are +30 and +250; raw costs differ by 580 and must not be used.
    assert!((219..=221).contains(&predictions[0].learned_span_margin_ns));
    assert_eq!(
        predictions[0].learned_span_margin_ns,
        predictions[1].learned_span_margin_ns
    );
}
