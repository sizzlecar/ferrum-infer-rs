use super::*;
use sha2::{Digest, Sha256};

fn settings() -> StructuredSettingsV2 {
    StructuredSettingsV2 {
        max_wave_ns: 1_000_000,
        static_margin_ns: 0,
        ..StructuredSettingsV2::default()
    }
}

fn rows<'a>(axes: &'a [Vec<u64>], walls: &'a [u64]) -> Vec<FitSample<'a>> {
    axes.iter()
        .zip(walls)
        .map(|(axes, &wall_ns)| FitSample { axes, wall_ns })
        .collect()
}

// A replayed, exactly feasible certificate isolates the integer proof from
// optimizer convergence. Its epsilon and every coefficient are checked again
// against all original observations by the real replay constructor.
fn exact_fit(axes: &[Vec<u64>], law: &[u64]) -> NonNegativeFit {
    let walls: Vec<_> = axes
        .iter()
        .map(|x| x.iter().zip(law).map(|(x, c)| x * c).sum())
        .collect();
    let samples = rows(axes, &walls);
    let fitted =
        NonNegativeFit::fit_identified(&samples, &settings(), EnvelopeSettings::default()).unwrap();
    let mut frozen = fitted.certificate().clone();
    let coefficients: Vec<u128> = frozen
        .column_maxima
        .iter()
        .zip(law)
        .map(|(&m, &c)| u128::from(m) * u128::from(c) * DENOMINATOR)
        .collect();
    frozen.coefficient_words = coefficients
        .iter()
        .map(|&q| [q as u64, (q >> 64) as u64])
        .collect();
    frozen.epsilon_ns = certificates::verify_feasible(
        &samples,
        &frozen.column_maxima,
        &coefficients,
        settings().max_wave_ns,
    )
    .unwrap();
    assert_eq!(frozen.epsilon_ns, 0);
    NonNegativeFit::from_certificate_identified(
        &samples,
        &settings(),
        EnvelopeSettings::default(),
        frozen,
    )
    .unwrap()
}

#[test]
fn signed_basis_fixed_shape_does_not_identify_batch_or_context_cost() {
    let axes = vec![vec![1, 1, 65]; 24];
    let fit = exact_fit(&axes, &[1000, 0, 0]);
    assert_eq!(fit.certificate().geometry_rank, 1);
    for (query, compatible_cost) in [([1, 8, 520], 8000), ([1, 1, 512], 7877)] {
        assert_eq!(fit.predict_fitted(&query), Ok(1000));
        let bound = fit.predict_identified_envelope_detailed(&query).unwrap();
        // Both laws agree on every Fit row. The point is not an authorization
        // to choose the constant law for a previously unidentifiable direction.
        assert!(bound.upper_ns >= compatible_cost);
        assert!(bound.upper_ns <= fit.predict(&query).unwrap().upper_ns);
    }
}

#[test]
fn signed_basis_shared_overhead_anchor_allows_tight_algorithm_composition() {
    let anchors = [vec![1, 0, 0], vec![1, 2, 0], vec![1, 0, 4]];
    let axes: Vec<_> = anchors.iter().cycle().take(24).cloned().collect();
    let fit = exact_fit(&axes, &[1000, 100, 50]);
    let query = [1, 2, 4];
    assert_eq!(fit.certificate().geometry_rank, 3);
    let bound = fit.predict_identified_envelope_detailed(&query).unwrap();
    assert_eq!(bound.upper_ns, 1400);
    assert!(bound.upper_ns < fit.predict(&query).unwrap().upper_ns);
    // Removing the shared overhead row makes the same combination ambiguous.
    let ambiguous: Vec<_> = anchors[1..].iter().cycle().take(24).cloned().collect();
    let ambiguous_fit = exact_fit(&ambiguous, &[1000, 100, 50]);
    assert_eq!(ambiguous_fit.certificate().geometry_rank, 2);
    assert!(
        ambiguous_fit
            .predict_identified_envelope_detailed(&query)
            .unwrap()
            .upper_ns
            >= 2400
    );
}

#[test]
fn signed_basis_independent_batch_and_kv_directions_support_extrapolation() {
    let mut axes = Vec::new();
    for _ in 0..3 {
        for batch in [1, 2, 4, 8] {
            for kv in [7, 63, 255, 511] {
                axes.push(vec![1, batch, batch * (kv + 1)]);
            }
        }
    }
    let fit = exact_fit(&axes, &[1000, 100, 10]);
    assert_eq!(fit.certificate().geometry_rank, 3);
    for (batch, kv) in [(3, 47), (6, 191), (8, 767)] {
        let query = [1, batch, batch * (kv + 1)];
        let actual = 1000 + 100 * batch + 10 * batch * (kv + 1);
        let bound = fit.predict_identified_envelope_detailed(&query).unwrap();
        assert!(bound.upper_ns >= actual);
        assert!(bound.upper_ns < fit.predict(&query).unwrap().upper_ns);
    }
}

#[test]
fn signed_basis_negative_intermediate_bound_and_fractional_caps_round_outward() {
    let cache = SignedBasis {
        anchors: vec![Anchor {
            axes: vec![1, 1],
            wall_ns: 10,
        }],
        caps: vec![
            AxisCap {
                numerator: 10,
                denominator: 1
            };
            2
        ],
    };
    // b=-10D must stay signed until the positive residual correction is added.
    assert_eq!(
        cache.bound_for_weights(&[1, 0], 0, &[-(WEIGHT_DENOMINATOR as i64)]),
        Some(20)
    );
    let cache = SignedBasis {
        anchors: vec![Anchor {
            axes: vec![1, 3],
            wall_ns: 10,
        }],
        caps: vec![
            AxisCap {
                numerator: 10,
                denominator: 1,
            },
            AxisCap {
                numerator: 10,
                denominator: 3,
            },
        ],
    };
    assert_eq!(
        cache.bound_for_weights(&[1, 4], 0, &[WEIGHT_DENOMINATOR as i64]),
        Some(14)
    );
    // Half a row leaves a fractional, positive remainder on both coordinates.
    assert_eq!(
        cache.bound_for_weights(&[1, 4], 0, &[(WEIGHT_DENOMINATOR / 2) as i64]),
        Some(19)
    );
}

#[test]
fn signed_basis_bad_or_overflowing_hint_cannot_invalidate_a_positive_certificate() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i + 1, 0]).collect();
    let fit = exact_fit(&axes, &[100, 7, 0]);
    let walls: Vec<_> = axes.iter().map(|x| 100 + 7 * x[1]).collect();
    let samples = rows(&axes, &walls);
    let mut frozen = fit.certificate().clone();
    frozen
        .signed_basis
        .as_mut()
        .unwrap()
        .normalized_query_to_anchor_bits
        .fill(f64::MAX.to_bits());
    let replay = NonNegativeFit::from_certificate_identified(
        &samples,
        &settings(),
        EnvelopeSettings::default(),
        frozen,
    )
    .unwrap();
    assert_eq!(
        replay
            .predict_identified_envelope_detailed(&[1, 20, 0])
            .unwrap(),
        replay.predict(&[1, 20, 0]).unwrap()
    );
    assert_eq!(
        replay.predict_identified_envelope_detailed(&[1, 20, 1]),
        Err(StructuredQueryFailureV2::OutsideSupport(
            StructuredUnknown::UnidentifiedDirection
        ))
    );
    let cache = SignedBasis {
        anchors: vec![Anchor {
            axes: vec![1],
            wall_ns: 1,
        }],
        caps: vec![AxisCap {
            numerator: u128::MAX,
            denominator: 1,
        }],
    };
    assert_eq!(cache.bound_for_weights(&[2], 0, &[0]), None);
}

#[test]
fn signed_basis_strict_replay_binds_frozen_advice_and_preserves_legacy_bytes() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i + 1]).collect();
    let walls: Vec<_> = axes.iter().map(|x| 100 + 7 * x[1]).collect();
    let samples = rows(&axes, &walls);
    let old = NonNegativeFit::fit(&samples, &settings(), EnvelopeSettings::default()).unwrap();
    assert!(!serde_json::to_string(old.certificate())
        .unwrap()
        .contains("signed_basis"));
    assert!(matches!(
        NonNegativeFit::from_certificate_identified(
            &samples,
            &settings(),
            EnvelopeSettings::default(),
            old.certificate().clone(),
        ),
        Err(StructuredUnknown::WrongProtocol)
    ));
    let fit =
        NonNegativeFit::fit_identified(&samples, &settings(), EnvelopeSettings::default()).unwrap();
    let frozen: FitCertificate =
        serde_json::from_str(&serde_json::to_string(fit.certificate()).unwrap()).unwrap();
    let replay = NonNegativeFit::from_certificate_identified(
        &samples,
        &settings(),
        EnvelopeSettings::default(),
        frozen.clone(),
    )
    .unwrap();
    assert_eq!(
        fit.predict_identified_envelope_detailed(&[1, 20]),
        replay.predict_identified_envelope_detailed(&[1, 20])
    );
    let mut a = Sha256::new();
    let mut b = Sha256::new();
    fit.bind_parameters(&mut a);
    replay.bind_parameters(&mut b);
    assert_eq!(a.finalize(), b.finalize());
    assert!(fit.retained_heap_bytes().unwrap() > old.retained_heap_bytes().unwrap());
    assert!(matches!(
        NonNegativeFit::from_certificate(
            &samples,
            &settings(),
            EnvelopeSettings::default(),
            frozen.clone(),
        ),
        Err(StructuredUnknown::WrongProtocol)
    ));
    let mut changed_hint = frozen.clone();
    changed_hint
        .signed_basis
        .as_mut()
        .unwrap()
        .normalized_query_to_anchor_bits[0] ^= 1;
    let changed = NonNegativeFit::from_certificate_identified(
        &samples,
        &settings(),
        EnvelopeSettings::default(),
        changed_hint,
    )
    .unwrap();
    let mut original_hash = Sha256::new();
    let mut changed_hash = Sha256::new();
    fit.bind_parameters(&mut original_hash);
    changed.bind_parameters(&mut changed_hash);
    assert_ne!(original_hash.finalize(), changed_hash.finalize());
    for mutation in 0..3 {
        let mut bad = frozen.clone();
        let signed = bad.signed_basis.as_mut().unwrap();
        match mutation {
            0 => signed.anchor_indices[0] = samples.len(),
            1 => signed.normalized_query_to_anchor_bits[0] = f64::NAN.to_bits(),
            _ => {
                signed.normalized_query_to_anchor_bits.pop();
            }
        }
        assert!(matches!(
            NonNegativeFit::from_certificate_identified(
                &samples,
                &settings(),
                EnvelopeSettings::default(),
                bad,
            ),
            Err(StructuredUnknown::WrongSource)
        ));
    }
}

#[test]
fn signed_basis_extra_work_uses_the_original_single_budget() {
    let axes: Vec<_> = (0..12).map(|i| vec![1, i + 1]).collect();
    let walls = vec![100; axes.len()];
    let samples = rows(&axes, &walls);
    let mut envelope = EnvelopeSettings::default();
    envelope.maximum_coordinate_visits = (axes.len() * axes[0].len() * SWEEPS) as u64;
    assert!(NonNegativeFit::fit(&samples, &settings(), envelope).is_ok());
    assert!(matches!(
        NonNegativeFit::fit_identified(&samples, &settings(), envelope),
        Err(StructuredUnknown::Capacity)
    ));
    // n=12,d=2,r=2: 64nd original + 3r²d + 2nd + 4rd + r² additional.
    envelope.maximum_coordinate_visits = 1627;
    assert!(matches!(
        NonNegativeFit::fit_identified(&samples, &settings(), envelope),
        Err(StructuredUnknown::Capacity)
    ));
    envelope.maximum_coordinate_visits += 1;
    assert!(NonNegativeFit::fit_identified(&samples, &settings(), envelope).is_ok());
    assert_eq!(
        check_work(usize::MAX, usize::MAX, usize::MAX, u64::MAX),
        Err(StructuredUnknown::Capacity)
    );
}
