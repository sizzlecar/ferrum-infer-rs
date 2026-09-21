//! Test-only Q5_K/Q6_K references for the experimental F32-scale Q8 policy.
//! Reuses Q4's activation pack; neither reference selects a product provider.
use super::q4k_q8_reference::{fixture_block, pack_rows, DotReference, Q4Block};
use half::f16;

#[derive(Clone)]
pub(crate) struct Q5Block {
    pub low: Q4Block,
    pub high: [bool; 256],
}

impl Q5Block {
    pub fn quant(&self, i: usize) -> u8 {
        self.low.quants[i] + 16 * u8::from(self.high[i])
    }
    pub fn encode(&self) -> [u8; 176] {
        let low = self.low.encode();
        let mut bytes = [0; 176];
        bytes[..16].copy_from_slice(&low[..16]);
        bytes[48..].copy_from_slice(&low[16..]);
        for (i, &high) in self.high.iter().enumerate() {
            bytes[16 + i % 32] |= u8::from(high) << (i / 32);
        }
        bytes
    }
    pub fn strict_weight(&self, i: usize) -> f32 {
        let g = i / 32;
        (self.low.d.to_f32() * f32::from(self.low.scales[g])) * f32::from(self.quant(i))
            - self.low.dmin.to_f32() * f32::from(self.low.minima[g])
    }
}

pub(crate) fn fixture_q5(column: usize, block: usize) -> Q5Block {
    Q5Block {
        low: fixture_block(column, block),
        high: std::array::from_fn(|i| (i * 3 + i / 32 + column + block) % 5 < 2),
    }
}

#[derive(Clone)]
pub(crate) struct Q6Block {
    pub d: f16,
    pub scales: [i8; 16],
    pub quants: [i8; 256],
}

impl Q6Block {
    pub fn encode(&self) -> [u8; 210] {
        let mut bytes = [0; 210];
        for (i, &quant) in self.quants.iter().enumerate() {
            assert!((-32..=31).contains(&quant));
            let q = (i16::from(quant) + 32) as u8;
            let group = (i % 128) / 32;
            bytes[(i / 128) * 64 + (group % 2) * 32 + i % 32] |= (q & 15) << (4 * (group / 2));
            bytes[128 + (i / 128) * 32 + i % 32] |= (q >> 4) << (2 * group);
        }
        for (i, &scale) in self.scales.iter().enumerate() {
            bytes[192 + i] = scale as u8;
        }
        bytes[208..].copy_from_slice(&self.d.to_le_bytes());
        bytes
    }
    pub fn strict_weight(&self, i: usize) -> f32 {
        (self.d.to_f32() * f32::from(self.scales[i / 16])) * f32::from(self.quants[i])
    }
}

pub(crate) fn fixture_q6(column: usize, block: usize) -> Q6Block {
    Q6Block {
        d: fixture_block(column, block).d,
        scales: std::array::from_fn(|g| ((g * 53 + column * 7 + block * 11) % 256) as u8 as i8),
        quants: std::array::from_fn(|i| ((i * 17 + i / 16 + column + block * 3) % 64) as i8 - 32),
    }
}

fn empty_reference() -> DotReference {
    DotReference {
        policy: 0.0,
        strict_original: 0.0,
        strict_quantized: 0.0,
        strict_abs_terms: 0.0,
        expanded_abs_terms: 0.0,
        activation_error_bound: 0.0,
    }
}

fn strict_terms(r: &mut DotReference, w: f32, x: f16, delta: f64, qx: i8) {
    let (w, x, xhat) = (f64::from(w), f64::from(x.to_f32()), delta * f64::from(qx));
    r.strict_original += w * x;
    r.strict_quantized += w * xhat;
    r.strict_abs_terms += (w * x).abs();
    r.activation_error_bound += w.abs() * (xhat - x).abs();
}

pub(crate) fn dot_q5(input: &[f16], blocks: &[Q5Block]) -> DotReference {
    assert_eq!(input.len(), blocks.len() * 256);
    let packed = pack_rows(input, 1, input.len());
    let mut r = empty_reference();
    for (bi, block) in blocks.iter().enumerate() {
        for g in 0..8 {
            let start = bi * 256 + g * 32;
            let delta = f64::from(packed.scales[start / 32]);
            // Preserve the declared F32 coefficient stage explicitly.
            let a = f64::from(block.low.d.to_f32() * f32::from(block.low.scales[g]));
            let b = f64::from(block.low.dmin.to_f32() * f32::from(block.low.minima[g]));
            let (mut dot, mut sum) = (0_i64, 0_i64);
            for lane in 0..32 {
                let i = g * 32 + lane;
                let qx = packed.quants[start + lane];
                let qw = i64::from(block.quant(i));
                dot += qw * i64::from(qx);
                sum += i64::from(qx);
                strict_terms(
                    &mut r,
                    block.strict_weight(i),
                    input[start + lane],
                    delta,
                    qx,
                );
                r.expanded_abs_terms += (delta * a * qw as f64 * f64::from(qx)).abs()
                    + (delta * b * f64::from(qx)).abs();
            }
            assert!(dot.abs() <= 125_984 && sum.abs() <= 4_064);
            r.policy += delta * (a * dot as f64 - b * sum as f64);
        }
    }
    r
}

pub(crate) fn dot_q6(input: &[f16], blocks: &[Q6Block]) -> DotReference {
    assert_eq!(input.len(), blocks.len() * 256);
    let packed = pack_rows(input, 1, input.len());
    let mut r = empty_reference();
    for (bi, block) in blocks.iter().enumerate() {
        for g in 0..8 {
            let start = bi * 256 + g * 32;
            let delta = f64::from(packed.scales[start / 32]);
            let mut dot = [0_i64; 2];
            let mut a = [0.0_f64; 2];
            for half in 0..2 {
                a[half] = f64::from(block.d.to_f32() * f32::from(block.scales[2 * g + half]));
                for lane in 0..16 {
                    let within = half * 16 + lane;
                    let i = g * 32 + within;
                    let qx = packed.quants[start + within];
                    let qw = i64::from(block.quants[i]);
                    dot[half] += qw * i64::from(qx);
                    strict_terms(
                        &mut r,
                        block.strict_weight(i),
                        input[start + within],
                        delta,
                        qx,
                    );
                    r.expanded_abs_terms += (delta * a[half] * qw as f64 * f64::from(qx)).abs();
                }
                assert!(dot[half].abs() <= 65_024);
            }
            let weighted = i64::from(block.scales[2 * g]) * dot[0]
                + i64::from(block.scales[2 * g + 1]) * dot[1];
            assert!(weighted.abs() <= 16_646_144);
            // Same delta for both K16 halves; no re-quantizing each half.
            r.policy += delta * (a[0] * dot[0] as f64 + a[1] * dot[1] as f64);
        }
    }
    r
}

// Applies only to one F32 term per K32 group: four F32 operations per
// rescaling, four interleaved serial sums, and a balanced two-level final sum.
// EPSILON (rather than EPSILON/2) conservatively matches the existing oracle.
pub(crate) fn four_partial_bound(groups32: usize, expanded_abs_terms: f64) -> f64 {
    let nu = (groups32.div_ceil(4) + 6) as f64 * f64::from(f32::EPSILON);
    assert!(nu < 1.0);
    nu / (1.0 - nu) * expanded_abs_terms
}

fn finish_four(partial: [f32; 4]) -> f32 {
    (partial[0] + partial[2]) + (partial[1] + partial[3])
}

#[test]
fn q56_encode_and_strict_coefficients_match_native_decoder() {
    for bi in 0..2 {
        let q5 = fixture_q5(3, bi);
        let mut q6 = fixture_q6(5, bi);
        q6.scales[..2].copy_from_slice(&[-128, 127]);
        q6.quants[..4].copy_from_slice(&[-32, 31, 0, -1]);
        let mut decoded = [0.0_f32; 256];
        super::GgufBlockFormat::Q5K
            .decode(&q5.encode(), &mut decoded)
            .unwrap();
        for (i, &w) in decoded.iter().enumerate() {
            assert_eq!(w.to_bits(), q5.strict_weight(i).to_bits());
        }
        super::GgufBlockFormat::Q6K
            .decode(&q6.encode(), &mut decoded)
            .unwrap();
        for (i, &w) in decoded.iter().enumerate() {
            assert_eq!(w.to_bits(), q6.strict_weight(i).to_bits());
            assert_eq!(
                f64::from(w),
                f64::from(q6.d.to_f32()) * f64::from(q6.scales[i / 16]) * f64::from(q6.quants[i])
            );
        }
    }
}

#[test]
fn q5_min_and_q6_two_halves_use_the_declared_activation_group() {
    let q5 = Q5Block {
        low: Q4Block {
            d: f16::ONE,
            dmin: f16::ONE,
            scales: [1; 8],
            minima: [1; 8],
            quants: [0; 256],
        },
        high: [false; 256],
    };
    let mut x = [f16::ZERO; 256];
    x[0] = f16::ONE;
    x[1] = f16::from_f32(1.0 / 256.0);
    let r = dot_q5(&x, &[q5]);
    assert_eq!(r.policy, -f64::from(1.0_f32 / 127.0) * 127.0);
    assert_eq!(r.strict_original, -257.0 / 256.0);
    let mut q6 = Q6Block {
        d: f16::ONE,
        scales: [0; 16],
        quants: [1; 256],
    };
    q6.scales[..2].copy_from_slice(&[1, -128]);
    x.fill(f16::ZERO);
    x[0] = f16::from_f32(127.0);
    x[16] = f16::from_f32(1.0 / 256.0);
    let r = dot_q6(&x, &[q6]);
    assert_eq!(r.policy, 127.0);
    assert_eq!(r.strict_original, 126.5);
    assert_eq!(r.activation_error_bound, 0.5);
}

#[test]
fn q56_integer_extrema_fit_i32_and_exact_f32_conversion() {
    let x = [f16::from_f32(127.0); 256];
    let q5 = Q5Block {
        low: Q4Block {
            d: f16::ONE,
            dmin: f16::ZERO,
            scales: [1; 8],
            minima: [0; 8],
            quants: [15; 256],
        },
        high: [true; 256],
    };
    assert_eq!(dot_q5(&x, &[q5]).policy, 8.0 * 125_984.0);
    let q6 = Q6Block {
        d: f16::ONE,
        scales: [-128; 16],
        quants: [-32; 256],
    };
    assert_eq!(dot_q6(&x, &[q6]).policy, 8.0 * 16_646_144.0);
}

#[test]
fn q56_cancellation_keeps_expanded_terms() {
    let q5 = Q5Block {
        low: Q4Block {
            d: f16::ONE,
            dmin: f16::ONE,
            scales: [1; 8],
            minima: [31; 8],
            quants: [15; 256],
        },
        high: [true; 256],
    };
    let q6 = Q6Block {
        d: f16::ONE,
        scales: std::array::from_fn(|g| if g % 2 == 0 { 1 } else { -1 }),
        quants: [31; 256],
    };
    for r in [
        dot_q5(&[f16::ONE; 256], &[q5]),
        dot_q6(&[f16::ONE; 256], &[q6]),
    ] {
        assert_eq!(r.policy, 0.0);
        assert_eq!(r.strict_original, 0.0);
        assert!(r.expanded_abs_terms > 7000.0);
    }
}

#[test]
fn q56_four_partial_rounding_is_separate_from_activation_error() {
    let x: Vec<_> = (0..512)
        .map(|i| f16::from_f32((i as i32 % 61 - 30) as f32 / 32.0))
        .collect();
    let p = pack_rows(&x, 1, x.len());
    let mut q5 = vec![fixture_q5(3, 0), fixture_q5(3, 1)];
    q5[0].low.d = f16::from_bits(1);
    q5[0].low.dmin = f16::from_f32(63.0);
    let q6 = vec![fixture_q6(4, 0), fixture_q6(4, 1)];
    let (mut sums5, mut sums6, mut sums6_h) = ([0.0_f32; 4], [0.0_f32; 4], [0.0_f32; 4]);
    for group in 0..16 {
        let (bi, g, start) = (group / 8, group % 8, group * 32);
        let (mut dot5, mut sum5, mut dot6) = (0_i32, 0_i32, [0_i32; 2]);
        for i in 0..32 {
            let u = i32::from(p.quants[start + i]);
            dot5 += i32::from(q5[bi].quant(g * 32 + i)) * u;
            sum5 += u;
            dot6[i / 16] += i32::from(q6[bi].quants[g * 32 + i]) * u;
        }
        let b = &q5[bi].low;
        let a5 = b.d.to_f32() * f32::from(b.scales[g]);
        let b5 = b.dmin.to_f32() * f32::from(b.minima[g]);
        sums5[group % 4] += p.scales[group] * (a5 * dot5 as f32 - b5 * sum5 as f32);
        let a60 = q6[bi].d.to_f32() * f32::from(q6[bi].scales[2 * g]);
        let a61 = q6[bi].d.to_f32() * f32::from(q6[bi].scales[2 * g + 1]);
        sums6[group % 4] += p.scales[group] * (a60 * dot6[0] as f32 + a61 * dot6[1] as f32);
        // H is computed before entering F32. Both signed K16 weight scales
        // share the same K32 activation delta; no scale is folded into i8 q.
        let h = i32::from(q6[bi].scales[2 * g]) * dot6[0]
            + i32::from(q6[bi].scales[2 * g + 1]) * dot6[1];
        assert!(h.abs() <= 16_646_144);
        assert_eq!(h as f32 as i32, h);
        sums6_h[group % 4] += p.scales[group] * (q6[bi].d.to_f32() * h as f32);
    }
    for (r, actual) in [
        (dot_q5(&x, &q5), finish_four(sums5)),
        (dot_q6(&x, &q6), finish_four(sums6)),
        // The two-multiply H path is also covered by the conservative
        // four-operation rescale allowance, against the same F64 policy.
        (dot_q6(&x, &q6), finish_four(sums6_h)),
    ] {
        // Activation error is reported separately, never added to this bound.
        assert!(
            (f64::from(actual) - r.policy).abs() <= four_partial_bound(16, r.expanded_abs_terms)
        );
        let host_rounding = 512.0 * f64::EPSILON * r.expanded_abs_terms;
        assert!(
            (r.strict_quantized - r.strict_original).abs()
                <= r.activation_error_bound + host_rounding
        );
    }
    let q5_result = dot_q5(&x, &q5);
    assert_ne!(q5_result.policy, q5_result.strict_quantized);
}
