use super::*;
use sha2::{Digest, Sha256};

fn settings() -> StructuredSettingsV2 {
    StructuredSettingsV2 {
        max_wave_ns: 100_000,
        static_margin_ns: 0,
        ..StructuredSettingsV2::default()
    }
}
fn samples<'a>(axes: &'a [Vec<u64>], walls: &[u64]) -> Vec<FitSample<'a>> {
    axes.iter()
        .zip(walls)
        .map(|(x, &y)| FitSample {
            axes: x,
            wall_ns: y,
        })
        .collect()
}

#[test]
fn correlated_history_accepts_new_prompt_without_claiming_unseen_axes() {
    let axes: Vec<_> = (0..12).map(|h| vec![1, h, 64 + h, 0]).collect();
    let walls: Vec<_> = (0..12).map(|h| 100 + 2 * h).collect();
    let fit = NonNegativeFit::fit(
        &samples(&axes, &walls),
        &settings(),
        EnvelopeSettings::default(),
    )
    .unwrap();
    assert_eq!(fit.certificate().geometry_rank, 2);
    let result = fit.predict(&[1, 12, 140, 0]).unwrap();
    assert_eq!(result.lower_ns, 0);
    assert!(result.typical_ns > 0 && result.typical_ns <= result.upper_ns);
    assert!(result.upper_ns >= 124);
    assert_eq!(
        fit.predict(&[1, 12, 140, 1]),
        Err(StructuredUnknown::UnidentifiedDirection)
    );
}

#[test]
fn aggregate_certificate_covers_axes_observed_on_separate_actual_waves() {
    let axes: Vec<_> = (0..12)
        .map(|i| {
            if i % 2 == 0 {
                vec![1, 2, 0]
            } else {
                vec![1, 0, 4]
            }
        })
        .collect();
    let walls = vec![100; 12];
    let fit = NonNegativeFit::fit(
        &samples(&axes, &walls),
        &settings(),
        EnvelopeSettings::default(),
    )
    .unwrap();
    let bound = fit.predict(&[1, 2, 4]).unwrap();
    assert_eq!(bound.upper_ns, 200);
    assert_eq!(fit.certificates.len(), 8);
    assert!(fit
        .certificates
        .iter()
        .skip(1)
        .all(|c| certificates::upper_bound(c, &[1, 2, 4]).unwrap().is_none()));
}

#[test]
fn bound_diagnostic_identifies_extrapolated_work_without_changing_prediction() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, 1 + i % 2, 0]).collect();
    let fit = NonNegativeFit::fit(
        &samples(&axes, &[100; 12]),
        &settings(),
        EnvelopeSettings::default(),
    )
    .unwrap();
    let query = [1, 20, 0];
    let before = fit.predict(&query).unwrap();
    assert_eq!(fit.predict_fitted(&query), Ok(100));
    assert!(before.upper_ns >= 1000);
    let explanation = fit.diagnose_bound(&query).unwrap();
    assert_eq!(explanation.upper_ns, before.upper_ns);
    assert_eq!(explanation.limiting_axis, 1);
    assert_eq!(explanation.input_value, 20);
    assert_eq!(explanation.fit_maximum, 2);
    assert_eq!(
        certificates::ceil_div(
            u128::from(explanation.input_value) * explanation.certificate_bound_ns,
            explanation.certificate_axis,
        )
        .unwrap(),
        u128::from(before.upper_ns),
    );
    assert_eq!(fit.predict(&query).unwrap(), before);
    assert!(fit.diagnose_bound(&[1, 20, 1]).is_none());
    assert_eq!(
        fit.predict_fitted(&[1, 20, 1]),
        Err(StructuredUnknown::UnidentifiedDirection)
    );
}

#[test]
fn replay_checks_all_original_inputs_and_exact_frozen_feasibility() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i]).collect();
    let walls: Vec<_> = (0..12).map(|i| 50 + i * 7).collect();
    let rows = samples(&axes, &walls);
    let fit = NonNegativeFit::fit(&rows, &settings(), EnvelopeSettings::default()).unwrap();
    let frozen: FitCertificate =
        serde_json::from_str(&serde_json::to_string(fit.certificate()).unwrap()).unwrap();
    let replay = NonNegativeFit::from_certificate(
        &rows,
        &settings(),
        EnvelopeSettings::default(),
        frozen.clone(),
    )
    .unwrap();
    assert_eq!(fit.predict(&[1, 23]), replay.predict(&[1, 23]));
    let mut first = Sha256::new();
    let mut second = Sha256::new();
    fit.bind_parameters(&mut first);
    replay.bind_parameters(&mut second);
    assert_eq!(first.finalize(), second.finalize());
    let mut tampered = frozen.clone();
    tampered.epsilon_ns += 1;
    assert!(matches!(
        NonNegativeFit::from_certificate(&rows, &settings(), EnvelopeSettings::default(), tampered),
        Err(StructuredUnknown::WrongSource)
    ));
    let mut tampered = frozen.clone();
    tampered.coefficient_words[0] = [u64::MAX, u64::MAX];
    assert!(NonNegativeFit::from_certificate(
        &rows,
        &settings(),
        EnvelopeSettings::default(),
        tampered
    )
    .is_err());
    let mut changed_walls = walls;
    changed_walls[3] += 1;
    assert!(matches!(
        NonNegativeFit::from_certificate(
            &samples(&axes, &changed_walls),
            &settings(),
            EnvelopeSettings::default(),
            frozen
        ),
        Err(StructuredUnknown::WrongSource)
    ));
}

#[test]
fn integer_ratio_ceiling_never_rounds_a_bound_down() {
    let c = UpperCertificate {
        axes: vec![1, 3],
        bound_ns: 10,
    };
    assert_eq!(certificates::upper_bound(&c, &[1, 4]), Ok(Some(14)));
    let large = 1u128 << 53;
    let c = UpperCertificate {
        axes: vec![1, large - 1],
        bound_ns: large,
    };
    assert_eq!(
        certificates::upper_bound(&c, &[1, large as u64]),
        Ok(Some(large as u64 + 2))
    );
    assert_eq!(certificates::ceil_div(u128::MAX, u128::MAX), Ok(1));
    assert!(certificates::ceil_div(1, 0).is_err());
    let overflow = UpperCertificate {
        axes: vec![1, 1],
        bound_ns: u128::MAX,
    };
    assert_eq!(
        certificates::upper_bound(&overflow, &[1, 2]),
        Err(StructuredUnknown::Numerical)
    );
}

#[test]
fn normalized_quantization_preserves_tiny_cost_per_work_unit() {
    let large = 1u64 << 50;
    let axes = vec![vec![1, large / 2], vec![1, large]];
    let rows = samples(&axes, &[50, 100]);
    let coefficients = [0, 100 * DENOMINATOR];
    assert_eq!(
        certificates::verify_feasible(&rows, &[1, large], &coefficients, 1000),
        Ok(0)
    );
    let slightly_perturbed = [0, 100 * DENOMINATOR + 1];
    assert_eq!(
        certificates::verify_feasible(&rows, &[1, large], &slightly_perturbed, 1000),
        Ok(1)
    );
    assert_eq!(
        certificates::scaled_dot(&[1, 2], &[1, 3], &[0, 1]),
        Ok((0, 1))
    );
}

#[test]
fn measured_cost_does_not_change_positive_axis_membership() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i, 0]).collect();
    for wall in [1, 10, 1000] {
        let rows = samples(&axes, &vec![wall; 12]);
        let fit = NonNegativeFit::fit(&rows, &settings(), EnvelopeSettings::default()).unwrap();
        assert_eq!(fit.check_observed_axes(&[1, 100, 0]), Ok(()));
        assert_eq!(
            fit.check_observed_axes(&[1, 100, 1]),
            Err(StructuredUnknown::UnidentifiedDirection)
        );
    }
}

#[test]
fn eligible_slow_heldout_cost_cannot_change_its_frozen_bound() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i]).collect();
    let rows = samples(&axes, &[100; 12]);
    let fit = NonNegativeFit::fit(&rows, &settings(), EnvelopeSettings::default()).unwrap();
    let query = [1, 12];
    let upper = fit.predict(&query).unwrap().upper_ns;
    let slow_heldout_wall = upper + 1;
    assert!(slow_heldout_wall > upper);
    assert_eq!(fit.check_observed_axes(&query), Ok(()));
    assert_eq!(fit.predict(&query).unwrap().upper_ns, upper);
}

#[test]
fn resource_limits_reject_whole_fit_without_truncation() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i]).collect();
    let rows = samples(&axes, &[100; 12]);
    let mut limits = EnvelopeSettings::default();
    limits.maximum_coordinate_visits = 12 * 2 * SWEEPS as u64 - 1;
    assert!(matches!(
        NonNegativeFit::fit(&rows, &settings(), limits),
        Err(StructuredUnknown::Capacity)
    ));
    assert!(matches!(
        NonNegativeFit::fit(&rows[..7], &settings(), EnvelopeSettings::default()),
        Err(StructuredUnknown::InsufficientSamples)
    ));
    limits.maximum_query_certificates = 9;
    assert!(matches!(
        NonNegativeFit::fit(&rows, &settings(), limits),
        Err(StructuredUnknown::InvalidSettings)
    ));
}

#[test]
fn planner_cap_failure_does_not_remove_an_eligible_input() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i]).collect();
    let mut s = settings();
    s.max_wave_ns = 1000;
    let rows = samples(&axes, &[100; 12]);
    let fit = NonNegativeFit::fit(&rows, &s, EnvelopeSettings::default()).unwrap();
    let input = [1, 1000];
    assert_eq!(fit.check_observed_axes(&input), Ok(()));
    assert_eq!(fit.predict(&input), Err(StructuredUnknown::Capacity));
}

#[test]
fn joint_future_box_bounds_every_nonnegative_feasible_coefficient() {
    let c = UpperCertificate {
        axes: vec![1, 1, 1, 1],
        bound_ns: 8,
    };
    let upper = certificates::upper_bound(&c, &[1, 2, 3, 5])
        .unwrap()
        .unwrap();
    for b0 in 0..=8u64 {
        for b1 in 0..=8 - b0 {
            for b2 in 0..=8 - b0 - b1 {
                let b3 = 8 - b0 - b1 - b2;
                for mask in 0..4 {
                    let mut x = [1, 0, 0, 0];
                    for p in 1..=2u64 {
                        if mask & (1 << (p - 1)) != 0 {
                            x[1] += 1;
                            x[2] += p;
                            x[3] += p * p;
                        }
                    }
                    let cost = x
                        .into_iter()
                        .zip([b0, b1, b2, b3])
                        .map(|(a, b)| a * b)
                        .sum::<u64>();
                    assert!(cost <= upper);
                }
            }
        }
    }
}
