//! Independent CPU classification/reference for device MarkerV2. Fast-math
//! packing codes/scales are always read back; no CPU ideal-pack bits contract.
use ferrum_native_ops::upstream_linear::{UpstreamLinearAlgorithm as A, UpstreamLinearFormat as F};
use half::f16;
fn h(b: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32()
}
fn rounded(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}
fn affine(b: &[u8], g: usize) -> (i32, i32) {
    if g < 4 {
        ((b[4 + g] & 63) as i32, (b[8 + g] & 63) as i32)
    } else {
        (
            ((b[8 + g] & 15) | ((b[g] >> 6) << 4)) as i32,
            ((b[8 + g] >> 4) | ((b[4 + g] >> 6) << 4)) as i32,
        )
    }
}
pub(super) fn marker_classify_pack(a: A, f: F, input: &[u16; 32]) -> (bool, bool) {
    let x = input.map(|v| f16::from_bits(v).to_f32());
    let nonfinite = x.iter().any(|v| !v.is_finite());
    let zero = !nonfinite && x.iter().all(|v| *v == 0.0);
    // Upstream MMQ: 8 contiguous float4 local left folds, xor 4/2/1.
    let mut sum: [f32; 8] =
        std::array::from_fn(|g| ((x[g * 4] + x[g * 4 + 1]) + x[g * 4 + 2]) + x[g * 4 + 3]);
    for offset in [4, 2, 1] {
        let prev = sum;
        for lane in 0..8 {
            sum[lane] = prev[lane] + prev[lane ^ offset];
        }
    }
    let overflow = a == A::Mmq && f != F::Iq4Xs && !rounded(sum[0]).is_finite();
    (zero, nonfinite || overflow)
}
pub(super) fn classify_weight(a: A, f: F, b: &[u8]) -> bool {
    let d = h(b, 0);
    if !d.is_finite() {
        return true;
    }
    if f == F::Iq4Xs {
        return false;
    }
    let dm = h(b, 2);
    if !dm.is_finite() {
        return true;
    }
    (0..8).any(|g| {
        let (s, m) = affine(b, g);
        let (c, v) = (d * s as f32, -dm * m as f32);
        !c.is_finite()
            || !v.is_finite()
            || (a == A::Mmq && (!rounded(c).is_finite() || !rounded(v).is_finite()))
    })
}
pub(super) fn final_cast_bits(value: f32, row_bad: bool, leaf_bad: bool) -> u16 {
    let result = f16::from_f32(value);
    if row_bad || leaf_bad || !value.is_finite() || !result.is_finite() {
        0x7e00
    } else {
        result.to_bits()
    }
}
pub(super) fn declared_group(
    a: A,
    f: F,
    b: &[u8],
    g: usize,
    q: &[i8; 32],
    scale: f32,
    original_sum: f32,
) -> (f64, f64) {
    const LUT: [i32; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let (coef, min, codes): (f64, f64, [i32; 32]) = if f == F::Iq4Xs {
        let hi = u16::from_le_bytes([b[2], b[3]]);
        let s = ((b[4 + g / 2] >> (4 * (g % 2))) & 15) | (((hi >> (2 * g)) as u8 & 3) << 4);
        (
            (h(b, 0) * (s as i32 - 32) as f32) as f64,
            0.0,
            std::array::from_fn(|i| {
                LUT[((b[8 + g * 16 + i % 16] >> (4 * (i / 16))) & 15) as usize]
            }),
        )
    } else {
        let (s, m) = affine(b, g);
        let (c, v) = (h(b, 0) * s as f32, -h(b, 2) * m as f32);
        let offset = if f == F::Q5K { 48 } else { 16 };
        let codes = std::array::from_fn(|i| {
            let low = (b[offset + (g / 2) * 32 + i] >> (4 * (g % 2))) & 15;
            let high = if f == F::Q5K {
                ((b[16 + i] >> g) & 1) << 4
            } else {
                0
            };
            (low | high) as i32
        });
        (
            if a == A::Mmq { rounded(c) } else { c } as f64,
            if a == A::Mmq { rounded(v) } else { v } as f64,
            codes,
        )
    };
    let dot: i32 = codes.iter().zip(q).map(|(w, q)| *w * i32::from(*q)).sum();
    let sum: i32 = q.iter().map(|v| i32::from(*v)).sum();
    let correction = if a == A::Mmq && f != F::Iq4Xs {
        original_sum as f64
    } else {
        scale as f64 * sum as f64
    };
    let target = coef * scale as f64 * dot as f64 + min * correction;
    let magnitude = codes
        .iter()
        .zip(q)
        .map(|(w, q)| (coef * scale as f64 * (*w as f64) * (*q as f64)).abs())
        .sum::<f64>()
        + (min * correction).abs();
    (target, magnitude)
}
#[test]
fn zero_nonfinite_and_unconsumed_sum_are_distinct() {
    for f in [F::Iq4Xs, F::Q4K, F::Q5K] {
        assert_eq!(
            marker_classify_pack(A::Mmq, f, &[0x8000; 32]),
            (true, false)
        );
        assert_eq!(
            marker_classify_pack(A::Mmvq, f, &[0x7bff; 32]),
            (false, false)
        );
        assert_eq!(
            marker_classify_pack(A::Mmq, f, &[0x7bff; 32]).1,
            f != F::Iq4Xs
        );
        assert_eq!(
            marker_classify_pack(A::Mmq, f, &[0x7e00; 32]),
            (false, true)
        );
        assert_eq!(marker_classify_pack(A::Mmq, f, &[1; 32]), (false, false));
    }
}
#[test]
fn cast_poison_overflow_and_underflow_have_explicit_bits() {
    assert_eq!(final_cast_bits(-0.0, false, false), 0x8000);
    assert_eq!(final_cast_bits(f32::MIN_POSITIVE, false, false), 0);
    for x in [f32::NAN, f32::INFINITY, 70000.0] {
        assert_eq!(final_cast_bits(x, false, false), 0x7e00);
    }
    assert_eq!(final_cast_bits(1.0, true, false), 0x7e00);
    assert_eq!(final_cast_bits(1.0, false, true), 0x7e00);
}
#[test]
fn half_coefficient_overflow_is_mmq_specific() {
    let mut b = vec![0; 144];
    b[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
    b[4..8].fill(2);
    b[12..16].fill(2);
    assert!(classify_weight(A::Mmq, F::Q4K, &b));
    assert!(!classify_weight(A::Mmvq, F::Q4K, &b));
}
