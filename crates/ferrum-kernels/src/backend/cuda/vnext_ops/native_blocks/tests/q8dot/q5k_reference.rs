//! Q5-only reference reused from Ferrum 443ab2d6e654e5bf3aa42734d80f1a31be9a151b.
//! Source: gguf_blocks/q56k_q8_reference.rs; Q6/MMA/old reduction bounds excluded.

use super::q4k_reference::{fixture_block, pack_rows, DotReference, Q4Block};
use crate::gguf_blocks::GgufBlockFormat;
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

#[test]
fn q5_bytes_high_bits_and_signed_coefficients_match_decoder() {
    for shift in 0..4 {
        for d in [
            f16::ZERO,
            f16::from_f32(0.5),
            f16::from_f32(-0.5),
            f16::from_bits(1),
            f16::from_bits(0x8001),
        ] {
            let mut block = fixture_q5(3, shift);
            block.low.d = d;
            block.low.dmin = f16::from_f32(-0.25);
            for i in 0..256 {
                let q = [0, 15, 16, 31][(i + i / 32 + shift) % 4];
                block.low.quants[i] = q & 15;
                block.high[i] = q >= 16;
            }
            let encoded = block.encode();
            let mut decoded = [0.0_f32; 256];
            GgufBlockFormat::Q5K.decode(&encoded, &mut decoded).unwrap();
            for (i, &weight) in decoded.iter().enumerate() {
                assert_eq!(weight.to_bits(), block.strict_weight(i).to_bits());
                assert_eq!(
                    (encoded[16 + i % 32] >> (i / 32)) & 1,
                    u8::from(block.high[i])
                );
            }
        }
    }
}

#[test]
fn q5_independent_byte_formula_matches_retained_reference() {
    for inputs in [256, 768, 1280] {
        let x = (0..inputs)
            .map(|i| f16::from_f32((i as i32 % 61 - 30) as f32 / 32.0))
            .collect::<Vec<_>>();
        let mut blocks = (0..inputs / 256)
            .map(|b| fixture_q5(3, b))
            .collect::<Vec<_>>();
        blocks[0].low.d = f16::from_bits(0x8001);
        blocks[0].low.dmin = f16::from_f32(63.0);
        let packed = pack_rows(&x, 1, inputs);
        let reference = dot_q5(&x, &blocks);
        let (mut policy, mut magnitude) = (0.0, 0.0);
        for group in 0..inputs / 32 {
            let (value, abs) = super::oracle::block_formula(
                GgufBlockFormat::Q5K,
                &blocks[group / 8].encode(),
                group % 8,
                &packed.quants[group * 32..group * 32 + 32],
                packed.scales[group],
            );
            policy += value;
            magnitude += abs;
        }
        let slack = inputs as f64 * f64::EPSILON * reference.expanded_abs_terms;
        assert!((policy - reference.policy).abs() <= slack);
        assert!((magnitude - reference.expanded_abs_terms).abs() <= slack);
        assert!(
            (reference.strict_quantized - reference.strict_original).abs()
                <= reference.activation_error_bound + slack
        );
        assert_ne!(
            reference.policy, reference.strict_quantized,
            "fixture must expose weight-reconstruction association separately"
        );
    }
}

#[test]
fn q5_minimum_uses_the_same_quantized_activation_sum() {
    let block = Q5Block {
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
    let r = dot_q5(&x, &[block]);
    assert_eq!(r.policy, -f64::from(1.0_f32 / 127.0) * 127.0);
    assert_eq!(r.strict_original, -257.0 / 256.0);
    assert_ne!(r.policy, r.strict_original);
}

#[test]
fn q5_integer_extrema_fit_i32_and_exact_f32_conversion() {
    let block = Q5Block {
        low: Q4Block {
            d: f16::ONE,
            dmin: f16::ZERO,
            scales: [1; 8],
            minima: [0; 8],
            quants: [15; 256],
        },
        high: [true; 256],
    };
    for sign in [-1, 1] {
        let dot: i32 = sign * 32 * 31 * 127;
        let sum: i32 = sign * 32 * 127;
        assert_eq!(dot.abs(), 125_984_i32);
        assert_eq!(sum.abs(), 4_064_i32);
        assert_eq!(dot as f32 as i32, dot);
        assert_eq!(sum as f32 as i32, sum);
        assert_eq!(
            dot_q5(&[f16::from_f32((sign * 127) as f32); 256], &[block.clone()]).policy,
            8.0 * f64::from(dot)
        );
    }
}

#[test]
fn q5_cancellation_retains_expanded_terms() {
    let block = Q5Block {
        low: Q4Block {
            d: f16::ONE,
            dmin: f16::ONE,
            scales: [1; 8],
            minima: [31; 8],
            quants: [15; 256],
        },
        high: [true; 256],
    };
    let r = dot_q5(&[f16::from_f32(127.0); 256], &[block]);
    assert_eq!(r.policy, 0.0);
    assert_eq!(r.strict_original, 0.0);
    assert_eq!(r.expanded_abs_terms, 2.0 * 31.0 * 127.0 * 256.0);
}
