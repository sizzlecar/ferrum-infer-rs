//! Synthetic clocks and real typed producer features; no live/GPU authority.
use super::*;

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
fn fit_values(values: &[StructuredNumericObservationV2]) -> FittedStructuredModelV2 {
    FittedStructuredModelV2::fit(
        fp(),
        settings(),
        exact_scope(values),
        contract(),
        values,
        170,
    )
    .unwrap()
}
fn fit_tail() -> Vec<StructuredNumericObservationV2> {
    let mut values = observations(StructuredPhaseV2::Fit);
    values[7].wall_ns = 1600;
    values
}
fn fit_tail_calibrated() -> CalibratedStructuredModelV2 {
    fit_values(&fit_tail())
        .calibrate(&observations(StructuredPhaseV2::Residual), 330)
        .unwrap()
}

#[test]
fn structured_v2_fit_floor_preserves_complete_fit_tail_separately_from_residual() {
    let validation = observations(StructuredPhaseV2::Qualification);
    let model = fit_tail_calibrated().qualify(&validation, 490).unwrap();
    let frozen = model.uncertainty();
    assert_eq!(frozen.fit_error_floor_ns, 562); // ceil((15*1000+1600)/16) = 1038
    assert_eq!(frozen.residual_ns, 0);
    assert_eq!(frozen.effective_residual_ns, 562);
    assert_eq!(frozen.static_margin_ns, 20);
    for sample in fit_tail() {
        let p = model
            .predict_query(&fp(), &StructuredQueryV2::exact(sample.input), 500)
            .unwrap();
        assert_eq!(p.fitted_upper_ns, 1038);
        assert_eq!(p.fit_samples, 16);
        assert_eq!(p.residual_samples, 16);
        assert_eq!(p.fit_error_floor_ns, 562);
        assert_eq!(p.residual_ns, 0);
        assert_eq!(p.effective_residual_ns, 562);
        assert_eq!(p.planning_ns, 1620);
        assert!(p.planning_ns >= sample.wall_ns + settings().static_margin_ns);
        assert_eq!(p.valid_until_ns, 10010);
    }
}

#[test]
fn structured_v2_fit_floor_keeps_larger_independent_residual_without_summing_twice() {
    let mut residual = observations(StructuredPhaseV2::Residual);
    residual[3].wall_ns = 1800;
    let validation = observations(StructuredPhaseV2::Qualification);
    let model = fit_values(&fit_tail())
        .calibrate(&residual, 330)
        .unwrap()
        .qualify(&validation, 490)
        .unwrap();
    let p = model
        .predict_query(
            &fp(),
            &StructuredQueryV2::exact(validation[0].input.clone()),
            500,
        )
        .unwrap();
    assert_eq!(p.fit_error_floor_ns, 562);
    assert_eq!(p.residual_ns, 762);
    assert_eq!(p.effective_residual_ns, 762);
    assert_eq!(p.planning_ns, 1820);
    assert_eq!(model.uncertainty().residual_ns, 762);
}

#[test]
fn structured_v2_fit_floor_qualification_can_only_challenge_frozen_uncertainty() {
    let mut validation = observations(StructuredPhaseV2::Qualification);
    let original = fit_tail_calibrated().qualify(&validation, 490).unwrap();
    let query = StructuredQueryV2::exact(validation[0].input.clone());
    let before = original.predict_query(&fp(), &query, 500).unwrap();
    validation[5].wall_ns = before.planning_ns;
    let boundary = fit_tail_calibrated().qualify(&validation, 490).unwrap();
    assert_eq!(boundary.uncertainty(), original.uncertainty());
    assert_eq!(
        boundary
            .predict_query(&fp(), &query, 500)
            .unwrap()
            .planning_ns,
        before.planning_ns
    );
    validation[5].wall_ns += 1;
    assert!(matches!(
        fit_tail_calibrated().qualify(&validation, 490),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
    assert_eq!(
        original
            .predict_query(&fp(), &query, 500)
            .unwrap()
            .planning_ns,
        before.planning_ns
    );
    assert!(matches!(
        original.predict_query(&fp(), &query, 10011),
        Err(StructuredUnknown::Stale)
    ));
    let mut wrong = fp();
    wrong.execution_config[0] ^= 1;
    assert!(matches!(
        original.predict_query(&wrong, &query, 500),
        Err(StructuredUnknown::WrongFingerprint)
    ));
}

fn varied(kind: StructuredPhaseV2) -> Vec<StructuredNumericObservationV2> {
    let mut values = observations(kind);
    for (i, value) in values.iter_mut().enumerate() {
        let work = if i % 2 == 0 { 8 } else { 16 };
        value.input = project(&wave(9, true, work, "fixture.first", [false, false])).unwrap();
        let mean = 5000 - 100 * work; // Deliberately negative slope, two identified points.
        value.wall_ns = match kind {
            StructuredPhaseV2::Fit => {
                if i / 2 % 2 == 0 {
                    mean - 120
                } else {
                    mean + 120
                }
            }
            StructuredPhaseV2::Residual => mean + if work == 8 { 30 } else { 250 },
            StructuredPhaseV2::Qualification => mean + 180,
        };
    }
    values
}
#[test]
fn structured_v2_fit_floor_uses_each_final_feature_prediction_not_population_mean() {
    let fit = varied(StructuredPhaseV2::Fit);
    let validation = varied(StructuredPhaseV2::Qualification);
    let model = fit_values(&fit)
        .calibrate(&varied(StructuredPhaseV2::Residual), 330)
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
    assert!((119..=120).contains(&predictions[0].fit_error_floor_ns));
    assert!((249..=250).contains(&predictions[0].residual_ns));
    assert_eq!(
        predictions[0].effective_residual_ns,
        predictions[0].residual_ns
    );
    for sample in &fit {
        let p = model
            .predict_query(&fp(), &StructuredQueryV2::exact(sample.input.clone()), 500)
            .unwrap();
        assert!(p.planning_ns >= sample.wall_ns + settings().static_margin_ns);
    }
    // A floor supplies no missing joint-support authority, even inside the fitted row space.
    let outside = project(&wave(9, true, 24, "fixture.first", [false, false])).unwrap();
    assert!(matches!(
        model.predict_query(&fp(), &StructuredQueryV2::exact(outside), 500),
        Err(StructuredUnknown::JointSupport)
    ));
}

#[test]
fn structured_v2_fit_floor_does_not_retain_fit_only_support_or_bypass_wave_limit() {
    let fit = varied(StructuredPhaseV2::Fit);
    let mut residual = varied(StructuredPhaseV2::Residual);
    let mut validation = varied(StructuredPhaseV2::Qualification);
    for samples in [&mut residual, &mut validation] {
        let input = samples[0].input.clone();
        let wall = samples[0].wall_ns;
        for sample in samples.iter_mut() {
            sample.input = input.clone();
            sample.wall_ns = wall;
        }
    }
    for sample in &mut validation {
        sample.wall_ns = 4200;
    }
    let model = fit_values(&fit)
        .calibrate(&residual, 330)
        .unwrap()
        .qualify(&validation, 490)
        .unwrap();
    assert!(matches!(
        model.predict_query(&fp(), &StructuredQueryV2::exact(fit[1].input.clone()), 500),
        Err(StructuredUnknown::JointSupport)
    ));
    let mut limits = settings();
    limits.max_wave_ns = 1600; // All observed costs fit; adding the original margin must still reject.
    let fit = fit_tail();
    let fitted =
        FittedStructuredModelV2::fit(fp(), limits, exact_scope(&fit), contract(), &fit, 170)
            .unwrap();
    assert!(matches!(
        fitted
            .calibrate(&observations(StructuredPhaseV2::Residual), 330)
            .unwrap()
            .qualify(&observations(StructuredPhaseV2::Qualification), 490),
        Err(StructuredUnknown::Numerical)
    ));
}
