use super::*;

fn gamma(operations: usize) -> f64 {
    let nu = operations as f64 * 2_f64.powi(-24);
    assert!(nu < 1.0);
    nu / (1.0 - nu)
}

pub(super) fn check(case: &Case, values: &[Vec<f32>], actual: &[f32]) {
    let scale = (case.scale.numerator() as f64 / case.scale.denominator() as f64) as f32 as f64;
    assert_eq!(
        actual.len(),
        case.keys.len() * case.sequence * case.heads * case.width
    );
    for batch in 0..case.keys.len() {
        for query in 0..case.sequence {
            let keys: Vec<_> = (0..case.sequence)
                .filter(|&k| case.visible(batch, query, k))
                .collect();
            for head in 0..case.heads {
                let q = ((batch * case.sequence + query) * case.heads + head) * case.width;
                if keys.is_empty() {
                    assert!(
                        actual[q..q + case.width].iter().all(|v| v.to_bits() == 0),
                        "empty row must be positive zero without reading poisoned Q/K/V"
                    );
                    continue;
                }
                let kv_head = head / (case.heads / case.kv_heads);
                let base = |key: usize| {
                    ((batch * case.sequence + key) * case.kv_heads + kv_head) * case.width
                };
                let mut delta = 0.0_f64;
                let scores: Vec<_> = keys
                    .iter()
                    .map(|&key| {
                        let (mut sum, mut absolute) = (0.0_f64, 0.0_f64);
                        for c in 0..case.width {
                            let product = values[0][q + c] as f64 * values[1][base(key) + c] as f64;
                            sum += product;
                            absolute += product.abs();
                        }
                        // FP32 simdgroup dot (at most 2D rounded operations) and
                        // scale. Independent FP64 summation error is included.
                        let dot_error = gamma(2 * case.width) * absolute
                            + (2 * case.width) as f64 * f64::EPSILON * absolute;
                        delta = delta.max(
                            dot_error * scale.abs()
                                + gamma(1) * (sum.abs() + dot_error) * scale.abs(),
                        );
                        let score = sum * scale;
                        // Overflow is a nonfinite score even if its FP64 oracle fits.
                        if sum.abs() > f32::MAX as f64 || score.abs() > f32::MAX as f64 {
                            f64::NAN
                        } else {
                            score
                        }
                    })
                    .collect();
                if scores.iter().any(|v| !v.is_finite()) {
                    assert!(actual[q..q + case.width].iter().all(|v| v.is_nan()), "nonfinite visible score must poison whole head, b={batch} q={query} h={head}");
                    continue;
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| libm::exp(s - maximum)).collect();
                let denominator: f64 = weights.iter().sum();
                // Bound online weight perturbation, including each tile's
                // rescale: score subtraction <= u*2M, exp <=4 FP32 ULP (<=8u
                // relative, Apple MSL 4.1 table 8.1). Normalization takes a
                // ratio, hence twice the accumulated exponent error. FP32
                // dot/sum, rescale, reciprocal and multiply use gamma_n.
                // This fixture has moderate finite scores; no exp underflow.
                let magnitude = scores.iter().map(|s| s.abs()).fold(0.0_f64, f64::max) + delta;
                assert!(scores.iter().all(|s| s - maximum > -80.0));
                let tiles = case.sequence.div_ceil(32);
                let exponent_error =
                    2.0 * delta + (tiles + 1) as f64 * (2.0 * magnitude + 8.0) * 2_f64.powi(-24);
                let weight_error = libm::expm1(2.0 * exponent_error);
                let arithmetic_error = gamma(4 * keys.len() + 8 * tiles + 4);
                let reference_nu = (4 * keys.len()) as f64 * f64::EPSILON;
                let relative = weight_error
                    + (1.0 + weight_error) * arithmetic_error
                    + reference_nu / (1.0 - reference_nu);
                for c in 0..case.width {
                    let mut sum = 0.0_f64;
                    let mut max_value = 0.0_f64;
                    for (&key, &weight) in keys.iter().zip(&weights) {
                        let v = values[2][base(key) + c] as f64;
                        sum += weight * v;
                        max_value = max_value.max(v.abs());
                    }
                    let expected = sum / denominator;
                    let got = actual[q + c];
                    if expected.is_nan() {
                        assert!(got.is_nan(), "visible V NaN channel");
                    } else if expected.is_infinite() {
                        assert_eq!(got as f64, expected, "visible V infinity channel");
                    } else {
                        let error = relative * max_value;
                        let bound = error + (expected.abs() + error) / 256.0 + 2_f64.powi(-134);
                        assert!(got.is_finite() && (got as f64 - expected).abs() <= bound,
                            "attention b={batch} q={query} h={head} c={c}: {got} != {expected}, derived bound={bound}");
                    }
                }
            }
        }
    }
}
