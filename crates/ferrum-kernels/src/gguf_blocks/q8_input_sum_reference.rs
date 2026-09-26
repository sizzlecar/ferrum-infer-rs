//! Independent reference for the explicit Q8 original-input-sum policy.
//! No provider selection or GPU layout is inferred here. The existing Q8
//! references retain their quantized-integer-sum policy unchanged.

use super::q4k_q8_reference::{self, Q4Block};
use super::q56k_q8_reference::Q5Block;
use half::f16;

#[derive(Debug)]
pub(crate) struct PackedRows {
    pub quants: Vec<i8>,
    pub scales: Vec<f32>,
    pub input_sums: Vec<f32>,
}

/// Complete native K256 leaves only: no implicit tail/padding extension.
pub(crate) fn pack_rows(input: &[f16], rows: usize, inputs: usize) -> PackedRows {
    assert!(rows > 0 && inputs > 0 && inputs.is_multiple_of(256));
    assert_eq!(input.len(), rows.checked_mul(inputs).unwrap());
    let packed = q4k_q8_reference::pack_rows(input, rows, inputs);
    let input_sums = input.chunks_exact(32).map(input_sum).collect();
    PackedRows {
        quants: packed.quants,
        scales: packed.scales,
        input_sums,
    }
}

fn input_sum(group: &[f16]) -> f32 {
    assert_eq!(group.len(), 32);
    if group.iter().any(|x| !x.is_finite()) {
        return f32::NAN;
    }
    if group.iter().all(|x| x.to_f32() == 0.0) {
        return 0.0;
    }
    let mut lanes: [f32; 32] = std::array::from_fn(|i| group[i].to_f32());
    for distance in [16, 8, 4, 2, 1] {
        // Use a copy: every lane consumes the previous step, as a synchronous
        // shuffle does. Out-of-range lanes never contribute to lane zero.
        let previous = lanes;
        for lane in 0..32 - distance {
            lanes[lane] = previous[lane] + previous[lane + distance];
        }
    }
    lanes[0]
}

#[derive(Debug, Default)]
pub(crate) struct DotReference {
    /// F64 evaluation with already-rounded F32 delta, a, b and input sum.
    /// The F32 GPU reduction's error is separate from these policy inputs.
    pub policy: f64,
    pub strict_original: f64,
    /// Positive terms before dot/min cancellation for rounding-error checks.
    pub expanded_abs_terms: f64,
    /// Only the positive quantized dot uses reconstructed activations.
    pub activation_error_bound: f64,
    pub input_sum_rounding_error_bound: f64,
    pub weight_reconstruction_error_bound: f64,
}

struct Group<'a> {
    input: &'a [f16],
    quants: &'a [i8],
    delta: f32,
    sum: f32,
    a: f32,
    b: f32,
}

fn accumulate_group(
    result: &mut DotReference,
    group: Group<'_>,
    quant: impl Fn(usize) -> u8,
    strict_weight: impl Fn(usize) -> f32,
) {
    let Group {
        input,
        quants,
        delta,
        sum,
        a,
        b,
    } = group;
    let (delta, a, b, sum) = (f64::from(delta), f64::from(a), f64::from(b), f64::from(sum));
    let mut dot = 0_i64;
    let mut exact_input_sum = 0.0_f64;
    for lane in 0..32 {
        let qw = f64::from(quant(lane));
        let qx = quants[lane];
        let x = f64::from(input[lane].to_f32());
        let xhat = delta * f64::from(qx);
        let strict_w = f64::from(strict_weight(lane));
        dot += i64::from(quant(lane)) * i64::from(qx);
        exact_input_sum += x;
        result.strict_original += strict_w * x;
        result.expanded_abs_terms += (a * qw * xhat).abs();
        result.activation_error_bound += (a * qw).abs() * (xhat - x).abs();
        result.weight_reconstruction_error_bound += ((a * qw - b) - strict_w).abs() * x.abs();
    }
    // Native Q5 is the larger range: 32 * 31 * 127. All I32 dots and their
    // conversion to F32 are exact; Q4 is a strict subset of this range.
    assert!(dot.abs() <= 125_984);
    result.policy += delta * a * dot as f64 - b * sum;
    result.expanded_abs_terms += (b * sum).abs();
    result.input_sum_rounding_error_bound += b.abs() * (exact_input_sum - sum).abs();
}

pub(crate) fn dot_q4(input: &[f16], blocks: &[Q4Block]) -> DotReference {
    assert_eq!(input.len(), blocks.len().checked_mul(256).unwrap());
    let packed = pack_rows(input, 1, input.len());
    let mut result = DotReference::default();
    for (bi, block) in blocks.iter().enumerate() {
        for g in 0..8 {
            let start = bi * 256 + g * 32;
            accumulate_group(
                &mut result,
                Group {
                    input: &input[start..start + 32],
                    quants: &packed.quants[start..start + 32],
                    delta: packed.scales[start / 32],
                    sum: packed.input_sums[start / 32],
                    a: block.d.to_f32() * f32::from(block.scales[g]),
                    b: block.dmin.to_f32() * f32::from(block.minima[g]),
                },
                |lane| block.quants[g * 32 + lane],
                |lane| block.strict_weight(g * 32 + lane),
            );
        }
    }
    result
}

pub(crate) fn dot_q5(input: &[f16], blocks: &[Q5Block]) -> DotReference {
    assert_eq!(input.len(), blocks.len().checked_mul(256).unwrap());
    let packed = pack_rows(input, 1, input.len());
    let mut result = DotReference::default();
    for (bi, block) in blocks.iter().enumerate() {
        for g in 0..8 {
            let start = bi * 256 + g * 32;
            accumulate_group(
                &mut result,
                Group {
                    input: &input[start..start + 32],
                    quants: &packed.quants[start..start + 32],
                    delta: packed.scales[start / 32],
                    sum: packed.input_sums[start / 32],
                    a: block.low.d.to_f32() * f32::from(block.low.scales[g]),
                    b: block.low.dmin.to_f32() * f32::from(block.low.minima[g]),
                },
                |lane| block.quant(g * 32 + lane),
                |lane| block.strict_weight(g * 32 + lane),
            );
        }
    }
    result
}

#[test]
fn q8_input_sum_pack_retains_old_quants_and_scales_and_row_group_boundaries() {
    let input: Vec<_> = (0..1024)
        .map(|i| f16::from_f32(((i * 37 % 509) as f32 - 254.0) / 128.0))
        .collect();
    let before = q4k_q8_reference::pack_rows(&input, 2, 512);
    let after = pack_rows(&input, 2, 512);
    assert_eq!(before.quants, after.quants);
    assert_eq!(before.scales, after.scales);
    assert_eq!(after.input_sums.len(), 32);
    for (g, source) in input.chunks_exact(32).enumerate() {
        // These small multiples of 1/128 sum exactly in every F32 tree.
        assert_eq!(
            f64::from(after.input_sums[g]),
            source.iter().map(|x| f64::from(x.to_f32())).sum::<f64>()
        );
    }
}

#[test]
fn q8_input_sum_uses_the_declared_f32_tree_not_sequential_or_f64_sum() {
    let tiny = f16::from_bits(1);
    let mut input = [f16::ZERO; 256];
    input[0] = f16::MAX;
    input[1] = tiny;
    input[16] = -f16::MAX;
    input[32] = f16::MAX;
    input[33] = -f16::MAX;
    input[48] = tiny;
    let packed = pack_rows(&input, 1, 256);
    assert_eq!(packed.input_sums[0], tiny.to_f32());
    assert_eq!(
        input[..32]
            .iter()
            .map(|x| x.to_f32())
            .fold(0.0_f32, |a, b| a + b),
        0.0
    );
    assert_eq!(packed.input_sums[1].to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        input[32..64]
            .iter()
            .map(|x| f64::from(x.to_f32()))
            .sum::<f64>(),
        f64::from(tiny.to_f32())
    );
}

#[test]
fn q8_input_sum_nonfinite_groups_and_signed_zero_have_explicit_outputs() {
    let mut input = [f16::NEG_ZERO; 256];
    input[32] = f16::INFINITY;
    input[65] = f16::NAN;
    input[98] = f16::NEG_INFINITY;
    let packed = pack_rows(&input, 1, 256);
    assert_eq!(packed.input_sums[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(packed.scales[0].to_bits(), 0.0_f32.to_bits());
    for g in 1..=3 {
        assert!(packed.input_sums[g].is_nan());
        assert!(packed.scales[g].is_nan());
        assert!(packed.quants[g * 32..(g + 1) * 32].iter().all(|q| *q == 0));
    }
    for g in [0, 4, 5, 6, 7] {
        assert_eq!(packed.input_sums[g].to_bits(), 0.0_f32.to_bits());
        assert_eq!(packed.scales[g].to_bits(), 0.0_f32.to_bits());
    }
}

#[test]
fn q8_input_sum_q4_q5_subtract_original_input_once_per_group() {
    let block = Q4Block {
        d: f16::ONE,
        dmin: f16::ONE,
        scales: [1; 8],
        minima: [1; 8],
        quants: [0; 256],
    };
    let q5 = Q5Block {
        low: block.clone(),
        high: [false; 256],
    };
    let mut input = [f16::ZERO; 256];
    for g in 0..8 {
        input[g * 32] = f16::ONE;
        input[g * 32 + 1] = f16::from_f32(1.0 / 256.0);
    }
    let q4 = dot_q4(&input, std::slice::from_ref(&block));
    let q5 = dot_q5(&input, std::slice::from_ref(&q5));
    assert_eq!(q4.policy, -8.0 * 257.0 / 256.0);
    assert_eq!(q5.policy, q4.policy);
    assert_eq!(q4.policy, q4.strict_original);
    assert_eq!(q5.policy, q5.strict_original);
    assert_eq!(q4.activation_error_bound, 0.0);
    assert_ne!(
        q4.policy,
        q4k_q8_reference::dot_reference(&input, &[block]).policy
    );
}

#[test]
fn q8_input_sum_reference_separates_policy_error_from_f32_reduction() {
    use super::q56k_q8_reference::{fixture_q5, four_partial_bound};
    let input: Vec<_> = (0..768)
        .map(|i| f16::from_f32(((i * 29 % 503) as f32 - 251.0) / 83.0))
        .collect();
    let q4: Vec<_> = (0..3)
        .map(|b| q4k_q8_reference::fixture_block(5, b))
        .collect();
    let q5: Vec<_> = (0..3).map(|b| fixture_q5(7, b)).collect();
    for reference in [dot_q4(&input, &q4), dot_q5(&input, &q5)] {
        let bound = reference.activation_error_bound
            + reference.input_sum_rounding_error_bound
            + reference.weight_reconstruction_error_bound;
        let rounding = f64::EPSILON * 32.0 * reference.expanded_abs_terms;
        assert!((reference.policy - reference.strict_original).abs() <= bound + rounding);
    }
    let packed = pack_rows(&input, 1, 768);
    let mut partial = [0.0_f32; 4];
    for (bi, block) in q5.iter().enumerate() {
        for g in 0..8 {
            let group = bi * 8 + g;
            let start = group * 32;
            let dot: i32 = (0..32)
                .map(|i| i32::from(block.quant(g * 32 + i)) * i32::from(packed.quants[start + i]))
                .sum();
            let a = block.low.d.to_f32() * f32::from(block.low.scales[g]);
            let b = block.low.dmin.to_f32() * f32::from(block.low.minima[g]);
            // Canonical complete-K32 rescale, with no FMA or repeated min.
            let term = (packed.scales[group] * a) * dot as f32 - b * packed.input_sums[group];
            partial[group % 4] += term;
        }
    }
    let actual = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    let expected = dot_q5(&input, &q5);
    assert!(
        (f64::from(actual) - expected.policy).abs()
            <= four_partial_bound(24, expected.expanded_abs_terms)
    );
}

#[test]
fn q8_input_sum_leaves_q6_without_a_min_correction() {
    use super::q56k_q8_reference::{dot_q6, fixture_q6};
    let input: Vec<_> = (0..256)
        .map(|i| f16::from_f32(((i * 13 % 61) as f32 - 30.0) / 17.0))
        .collect();
    let block = fixture_q6(3, 0);
    let packed = pack_rows(&input, 1, 256);
    let mut expected = 0.0_f64;
    for g in 0..8 {
        for half in 0..2 {
            let dot: i64 = (0..16)
                .map(|lane| {
                    let i = g * 32 + half * 16 + lane;
                    i64::from(block.quants[i]) * i64::from(packed.quants[i])
                })
                .sum();
            let a = block.d.to_f32() * f32::from(block.scales[g * 2 + half]);
            expected += f64::from(packed.scales[g]) * f64::from(a) * dot as f64;
        }
    }
    let original = dot_q6(&input, &[block]);
    let tolerance = f64::EPSILON * 32.0 * original.expanded_abs_terms;
    assert!((expected - original.policy).abs() <= tolerance);
}

#[test]
fn q8_input_sum_rejects_partial_native_leaves_and_malformed_rows() {
    for (len, rows, inputs) in [
        (32, 1, 32),
        (255, 1, 255),
        (256, 0, 256),
        (255, 1, 256),
        (512, 1, 256),
    ] {
        assert!(
            std::panic::catch_unwind(|| pack_rows(&vec![f16::ZERO; len], rows, inputs)).is_err()
        );
    }
}
