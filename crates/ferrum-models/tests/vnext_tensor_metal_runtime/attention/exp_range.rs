//! Analytic contributions outside the moderate-score oracle's useful bound.
//! This checks the existing exp contract, not the accuracy of a future exp2 recipe.
use super::*;

const DIFFERENCES: [f32; 5] = [63.75, 64.0, 64.5, 87.0, 88.0];
const HOT_KEY: usize = 16;

fn case(heads: usize) -> Case {
    Case {
        sequence: 32,
        heads,
        kv_heads: heads,
        width: 128,
        segments: vec![(32, true)],
        keys: vec![vec![(32, true)]],
        scale: CanonicalRational::new(1, 1).unwrap(),
    }
}

fn empty_values(case: &Case) -> Vec<Vec<f32>> {
    vec![vec![0.0; case.sequence * case.heads * case.width]; 3]
}

fn mask_hot_key(case: &mut Case) {
    case.keys = vec![vec![(HOT_KEY, true), (1, false), (15, true)]];
}

fn amplified_fixture() -> (Case, Vec<Vec<f32>>) {
    let case = case(DIFFERENCES.len());
    let mut values = empty_values(&case);
    let gain = 2_f32.powi(126);
    for (head, &difference) in DIFFERENCES.iter().enumerate() {
        for token in 0..case.sequence {
            let offset = (token * case.heads + head) * case.width;
            values[0][offset] = 1.0;
            values[1][offset] = if token == 0 { 0.0 } else { -difference };
        }
        let hot = (HOT_KEY * case.heads + head) * case.width;
        values[2][hot] = gain;
        values[2][hot + 127] = -gain;
    }
    (case, values)
}

/// Exact BF16 Q/K and scale give scores 0 and -difference without dot error.
/// There is one nonzero V contribution, hence no cancellation: propagate the
/// exp error relative to that contribution, not to the unweighted huge V.
fn contribution_bound(difference: f32, sequence: usize) -> (f64, f64, bool) {
    let probability = libm::exp(-f64::from(difference));
    let expected = probability * 2_f64.powi(126) / (1.0 + (sequence - 1) as f64 * probability);
    let rounded = probability as f32;
    let ulp = f64::from(rounded.next_up() - rounded).max(f64::from(rounded - rounded.next_down()));
    // Same 4-FP32-ULP exp envelope as the original oracle, including exp(0).
    // The tiny FP64 allowance covers reference arithmetic, not a new GPU tolerance.
    let exp_relative =
        ((4.0 * ulp) / probability).max(4.0 * f64::from(f32::EPSILON)) + 4.0 * f64::EPSILON;
    let tiles = sequence.div_ceil(16);
    let exponent = (2 * tiles + 1) as f64;
    // Include online exp(0) rescaling in numerator and denominator; this also
    // encloses the legacy path with fewer, larger key tiles.
    let weight_error =
        libm::pow(1.0 + exp_relative, exponent) / libm::pow(1.0 - exp_relative, exponent) - 1.0;
    let operations = (4 * sequence + 8 * tiles + 4) as f64;
    let nu = operations * 2_f64.powi(-24);
    let reference_nu = operations * f64::EPSILON;
    let relative =
        weight_error + (1.0 + weight_error) * nu / (1.0 - nu) + reference_nu / (1.0 - reference_nu);
    let error = expected.abs() * relative;
    let bound = error + (expected.abs() + error) / 256.0 + 2_f64.powi(-134);
    (expected, bound, probability < f64::from(f32::MIN_POSITIVE))
}

fn check_amplified(case: &Case, actual: &[f32], masked: bool) {
    assert_eq!(actual.len(), case.sequence * case.heads * case.width);
    for query in 0..case.sequence {
        for (head, &difference) in DIFFERENCES.iter().enumerate() {
            let (magnitude, bound, subnormal_probability) =
                contribution_bound(difference, case.sequence);
            for channel in 0..case.width {
                let got = actual[(query * case.heads + head) * case.width + channel];
                if masked || !matches!(channel, 0 | 127) {
                    assert_eq!(
                        got.to_bits(),
                        0,
                        "masked/zero V q={query} h={head} c={channel}"
                    );
                    continue;
                }
                // MSL permits flushing FP32 subnormals. Only this probability
                // class may produce zero; normal exp outputs may never do so.
                // This union is not proof of gradual underflow or A/B bit equality.
                if subnormal_probability && got == 0.0 {
                    continue;
                }
                let expected = if channel == 0 { magnitude } else { -magnitude };
                assert!(
                    got.is_finite() && (f64::from(got) - expected).abs() <= bound,
                    "exp boundary difference=-{difference}, q={query} c={channel}: {got} != {expected}, contribution bound={bound}"
                );
            }
        }
    }
    if !masked {
        let offset = (DIFFERENCES.len() - 1) * case.width;
        eprintln!(
            "Metal attention exp(-88) amplified output bits=0x{:08x}/0x{:08x}; subnormal probability permits FTZ, no gradual-underflow qualification",
            actual[offset].to_bits(), actual[offset + 127].to_bits()
        );
    }
}

fn huge_score_fixture() -> (Case, Vec<Vec<f32>>) {
    let case = case(1);
    let mut values = empty_values(&case);
    for token in 0..case.sequence {
        values[0][token * case.width] = 1.5 * 2_f32.powi(63);
        values[1][token * case.width] = 2_f32.powi(64);
    }
    values[2][HOT_KEY * case.width] = 32.0;
    values[2][HOT_KEY * case.width + 127] = -64.0;
    (case, values)
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
) {
    let (mut case, values) = amplified_fixture();
    for masked in [false, true] {
        if masked {
            mask_hot_key(&mut case);
        }
        run_case(
            runtime,
            registry,
            catalog,
            policy,
            host,
            &case.fixture(MetalTensorOperation::StructuredAttention),
            &values,
            |actual| check_amplified(&case, actual, masked),
        );
    }
    let (mut case, values) = huge_score_fixture();
    for masked in [false, true] {
        if masked {
            mask_hot_key(&mut case);
        }
        run_case(
            runtime,
            registry,
            catalog,
            policy,
            host,
            &case.fixture(MetalTensorOperation::StructuredAttention),
            &values,
            |actual| {
                // Every score is exactly 3*2^126, still finite FP32. Uniform
                // attention averages the one nonzero V over 32 keys. Scaling
                // these scores into base-2 before subtraction would overflow.
                for query in 0..case.sequence {
                    for channel in 0..case.width {
                        let expected: f32 = match (masked, channel) {
                            (false, 0) => 1.0,
                            (false, 127) => -2.0,
                            _ => 0.0,
                        };
                        assert_eq!(
                            actual[query * case.width + channel].to_bits(),
                            expected.to_bits(),
                            "huge finite uniform score q={query} c={channel}, masked={masked}"
                        );
                    }
                }
            },
        );
    }
}

#[test]
fn analytic_exp_fixtures_are_exact_bf16_and_exclude_zero_for_normal_probabilities() {
    let (case, values) = amplified_fixture();
    assert!(values
        .iter()
        .flatten()
        .all(|&v| rounds(v).to_bits() == v.to_bits()));
    for &difference in &DIFFERENCES {
        let (expected, bound, subnormal) = contribution_bound(difference, case.sequence);
        assert!(expected.is_finite() && expected > bound && bound > 0.0);
        assert_eq!(subnormal, difference == 88.0);
    }
    let (_, values) = huge_score_fixture();
    assert!(values
        .iter()
        .flatten()
        .all(|&v| rounds(v).to_bits() == v.to_bits()));
    let score = f64::from(values[0][0]) * f64::from(values[1][0]);
    assert_eq!(score, 3.0 * 2_f64.powi(126));
    assert!(score < f64::from(f32::MAX));
}
