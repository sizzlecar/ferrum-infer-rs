//! Independent CPU policy for the test-only F32-scale activation-Q8 experiment.
//! The stored scale is not llama Q8_1's half scale or Ferrum KV's ties-even ABI.

use crate::gguf_blocks::{GgufBlockFormat, IQ4_NL_VALUES};
use half::f16;

use super::q4k_reference::pack_rows;

/// Read strided F16 rows; return densely packed q rows and one F32 scale per 32
/// values. Padding is neither read nor quantized. A zero block stores positive
/// zero scale and zero codes. The retained historical policy marks a non-finite
/// group with NaN scale and all-zero codes; it never quantizes its other values.
pub(super) fn quantize(
    input: &[f16],
    rows: usize,
    inputs: usize,
    stride: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert!(rows > 0 && inputs > 0 && inputs % 32 == 0);
    assert!(stride >= inputs);
    let span = (rows - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(inputs))
        .expect("activation input span overflow");
    assert!(input.len() >= span, "activation input view is too short");
    let count = rows.checked_mul(inputs).expect("activation count overflow");
    let packed = if stride == inputs {
        pack_rows(&input[..count], rows, inputs)
    } else {
        let mut contiguous = Vec::with_capacity(count);
        for row in 0..rows {
            contiguous.extend_from_slice(&input[row * stride..row * stride + inputs]);
        }
        pack_rows(&contiguous, rows, inputs)
    };
    (packed.quants, packed.scales)
}

fn half_at(block: &[u8], offset: usize) -> f32 {
    let value = f16::from_bits(u16::from_le_bytes([block[offset], block[offset + 1]])).to_f32();
    assert!(value.is_finite(), "weight oracle requires finite scales");
    value
}

/// Exact integer dot followed by F64 scaling of independently reconstructed
/// F32 group coefficients. The second result sums magnitudes of the separate
/// scaled dot/min terms, before cancellation, for a conservative rounding bound.
/// It is not an acceptance tolerance or a semantic-quality threshold.
pub(super) fn block_formula(
    format: GgufBlockFormat,
    block: &[u8],
    group: usize,
    q: &[i8],
    activation_scale: f32,
) -> (f64, f64) {
    assert!(matches!(
        format,
        GgufBlockFormat::Iq4Xs | GgufBlockFormat::Q4K | GgufBlockFormat::Q5K
    ));
    assert_eq!(block.len(), format.block_bytes());
    assert!(group < 8);
    assert_eq!(q.len(), 32);
    assert!(q.iter().all(|&code| code != i8::MIN));
    assert!(activation_scale.is_finite() && activation_scale >= 0.0);
    let (weight_scale, minimum, levels): (f32, f32, [i32; 32]) = match format {
        GgufBlockFormat::Iq4Xs => {
            let high = u16::from_le_bytes([block[2], block[3]]);
            let low = (block[4 + group / 2] >> (4 * (group % 2))) & 15;
            let signed_scale = i32::from(low) + (((high >> (2 * group)) & 3) as i32) * 16 - 32;
            let scale = half_at(block, 0) * signed_scale as f32;
            let levels = std::array::from_fn(|column| {
                let byte = block[8 + group * 16 + column % 16];
                let code = (byte >> (4 * (column / 16))) & 15;
                i32::from(IQ4_NL_VALUES[usize::from(code)])
            });
            (scale, 0.0, levels)
        }
        GgufBlockFormat::Q4K | GgufBlockFormat::Q5K => {
            let (scale, minimum) = if group < 4 {
                (block[4 + group] & 63, block[8 + group] & 63)
            } else {
                (
                    (block[8 + group] & 15) | ((block[group] >> 6) << 4),
                    (block[8 + group] >> 4) | ((block[4 + group] >> 6) << 4),
                )
            };
            let levels = std::array::from_fn(|column| {
                let offset = if format == GgufBlockFormat::Q5K {
                    48
                } else {
                    16
                };
                let low = (block[offset + (group / 2) * 32 + column] >> (4 * (group % 2))) & 15;
                let high = if format == GgufBlockFormat::Q5K {
                    ((block[16 + column] >> group) & 1) * 16
                } else {
                    0
                };
                i32::from(low + high)
            });
            (
                half_at(block, 0) * f32::from(scale),
                half_at(block, 2) * f32::from(minimum),
                levels,
            )
        }
        _ => unreachable!(),
    };
    // Widen the independent oracle, then check the actual per-group I32
    // contract rather than relying on a potentially overflowing reference.
    let integer_dot: i64 = levels
        .iter()
        .zip(q)
        .map(|(&weight, &input)| i64::from(weight) * i64::from(input))
        .sum();
    let integer_sum: i64 = q.iter().map(|&input| i64::from(input)).sum();
    let integer_dot = i32::try_from(integer_dot).expect("Q8 group dot exceeds I32");
    let integer_sum = i32::try_from(integer_sum).expect("Q8 group sum exceeds I32");
    let positive = f64::from(weight_scale) * f64::from(activation_scale);
    let negative = f64::from(minimum) * f64::from(activation_scale);
    let result = positive * f64::from(integer_dot) - negative * f64::from(integer_sum);
    let magnitude = levels
        .iter()
        .zip(q)
        .map(|(&weight, &input)| {
            (positive * f64::from(weight) * f64::from(input)).abs()
                + (negative * f64::from(input)).abs()
        })
        .sum();
    (result, magnitude)
}

#[test]
fn activation_policy_covers_ties_zero_extremes_and_stride() {
    let stride = 67;
    let mut input = vec![f16::NAN; stride * 2];
    input[..64].fill(f16::ZERO);
    input[stride..stride + 64].fill(f16::ZERO);
    for (column, value) in [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
        .into_iter()
        .enumerate()
    {
        input[column] = f16::from_f32(value);
    }
    input[32] = f16::from_bits(0x8000);
    input[stride] = f16::from_bits(1);
    input[stride + 1] = f16::from_bits(0x8001);
    input[stride + 32] = f16::MAX;
    input[stride + 33] = f16::MIN;
    let (q, scales) = quantize(&input, 2, 64, stride);
    assert_eq!(&q[..8], &[127, -127, 1, -1, 2, -2, 127, -127]);
    assert!(q[8..64].iter().all(|&code| code == 0));
    assert_eq!(&q[64..66], &[127, -127]);
    assert_eq!(&q[96..98], &[127, -127]);
    assert_eq!(scales[0].to_bits(), 1.0_f32.to_bits());
    assert_eq!(scales[1].to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        scales[2].to_bits(),
        (f16::from_bits(1).to_f32() / 127.0).to_bits()
    );
    assert_eq!(scales[3].to_bits(), (65504.0_f32 / 127.0).to_bits());
    assert!(
        scales[2] > 0.0,
        "F32 scale preserves the tiny nonzero block"
    );
}

#[test]
fn activation_adapter_preserves_historical_nonfinite_marker() {
    let mut input = [f16::ZERO; 32];
    input[17] = f16::INFINITY;
    let (q, scales) = quantize(&input, 1, 32, 32);
    assert!(q.iter().all(|&code| code == 0));
    assert!(scales[0].is_nan());
}

#[test]
fn iq4_formula_decodes_both_nibbles_and_signed_group_scales() {
    // Literal reconstruction levels independently anchor the shared Rust table.
    let levels = [
        -127_i32, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let group_scales = [-32_i32, -17, -1, 0, 1, 15, 16, 31];
    let mut block = [0_u8; 136];
    block[..2].copy_from_slice(&f16::from_f32(-0.25).to_le_bytes());
    let mut high = 0_u16;
    for (group, scale) in group_scales.into_iter().enumerate() {
        let encoded = (scale + 32) as u8;
        block[4 + group / 2] |= (encoded & 15) << (4 * (group % 2));
        high |= u16::from(encoded >> 4) << (2 * group);
        for column in 0..16 {
            block[8 + group * 16 + column] = column as u8 | (((15 - column) as u8) << 4);
        }
    }
    block[2..4].copy_from_slice(&high.to_le_bytes());
    let q: [i8; 32] = std::array::from_fn(|column| match column {
        0 => 127,
        1 => -127,
        _ => (column as i8 % 7 - 3) * 31,
    });
    for (group, scale) in group_scales.into_iter().enumerate() {
        let terms: Vec<f64> = q
            .iter()
            .enumerate()
            .map(|(column, &input)| {
                let code = if column < 16 { column } else { 31 - column };
                -0.25 * f64::from(scale) * 0.5 * f64::from(levels[code]) * f64::from(input)
            })
            .collect();
        let (actual, magnitude) = block_formula(GgufBlockFormat::Iq4Xs, &block, group, &q, 0.5);
        assert_eq!(actual, terms.iter().sum::<f64>());
        assert_eq!(magnitude, terms.iter().map(|term| term.abs()).sum::<f64>());
    }
}

#[test]
fn q4_formula_preserves_packed_high_bits_and_same_q_minimum_correction() {
    let scales = [0_u8, 1, 15, 63, 16, 31, 47, 62];
    let minima = [63_u8, 47, 31, 16, 62, 15, 1, 0];
    let mut block = [0_u8; 144];
    block[..2].copy_from_slice(&f16::from_f32(-0.5).to_le_bytes());
    block[2..4].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
    for group in 0..4 {
        block[4 + group] = scales[group] | ((scales[group + 4] >> 4) << 6);
        block[8 + group] = minima[group] | ((minima[group + 4] >> 4) << 6);
        block[12 + group] = (scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4);
    }
    for group in 0..8 {
        for column in 0..32 {
            let code = ((column + group * 3) % 16) as u8;
            block[16 + (group / 2) * 32 + column] |= code << (4 * (group % 2));
        }
    }
    let q: [i8; 32] = std::array::from_fn(|column| match column {
        0 => 127,
        1 => -127,
        _ => column as i8 % 7 - 3,
    });
    assert_ne!(q.iter().map(|&code| i32::from(code)).sum::<i32>(), 0);
    for group in 0..8 {
        let mut expected = 0.0;
        let mut magnitude = 0.0;
        for (column, &input) in q.iter().enumerate() {
            let dot = -0.5
                * f64::from(scales[group])
                * 0.125
                * ((column + group * 3) % 16) as f64
                * f64::from(input);
            let minimum = 0.25 * f64::from(minima[group]) * 0.125 * f64::from(input);
            expected += dot - minimum;
            magnitude += dot.abs() + minimum.abs();
        }
        assert_eq!(
            block_formula(GgufBlockFormat::Q4K, &block, group, &q, 0.125),
            (expected, magnitude)
        );
    }
}

#[test]
fn q4_byte_formula_matches_retained_independent_reference() {
    use super::q4k_reference::{dot_reference, fixture_block};

    let input: Vec<_> = (0..512)
        .map(|index| f16::from_f32(((index * 17 % 67) as f32 - 33.0) / 64.0))
        .collect();
    let packed = pack_rows(&input, 1, input.len());
    let mut blocks = [fixture_block(3, 0), fixture_block(7, 1)];
    blocks[0].d = f16::from_bits(1);
    blocks[0].dmin = f16::from_f32(63.0);
    let reference = dot_reference(&input, &blocks);
    let (mut policy, mut magnitude) = (0.0, 0.0);
    for (index, block) in blocks.iter().enumerate() {
        let encoded = block.encode();
        for group in 0..8 {
            let start = index * 256 + group * 32;
            let (value, bound) = block_formula(
                GgufBlockFormat::Q4K,
                &encoded,
                group,
                &packed.quants[start..start + 32],
                packed.scales[start / 32],
            );
            policy += value;
            magnitude += bound;
        }
    }
    // These independent F64 formulas associate scaling differently. This
    // allowance covers only their F64 evaluation, not activation approximation.
    let bound = 64.0 * f64::EPSILON * reference.expanded_abs_terms;
    assert!((policy - reference.policy).abs() <= bound);
    assert!((magnitude - reference.expanded_abs_terms).abs() <= bound);
}
