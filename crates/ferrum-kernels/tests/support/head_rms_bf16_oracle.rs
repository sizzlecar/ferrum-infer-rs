//! Independent host oracle: compact adjacent pairs each round rather than the
//! CUDA kernel's in-place strided shared tree. Every arithmetic boundary is
//! explicitly RN32 via an F64 operation, then two independent BF16 conversions.
use half::bf16;

pub const WIDTHS: [usize; 7] = [3, 128, 4096, 4097, 5120, 6145, 8192];
pub const EPSILON: f32 = (1.0_f64 / 1_000_000.0) as f32;
pub const MATHEMATICAL_MAX_BF16_ULP: u16 = 2;

fn decode(v: u16) -> f32 {
    bf16::from_bits(v).to_f32()
}
fn encode(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}
fn mul(a: f32, b: f32) -> f32 {
    (f64::from(a) * f64::from(b)) as f32
}

pub fn reference(x: &[u16], gamma: &[u16], single_round: bool) -> Vec<u16> {
    assert!(!gamma.is_empty() && x.len() % gamma.len() == 0);
    x.chunks_exact(gamma.len())
        .flat_map(|group| {
            let mut level: Vec<_> = group.iter().map(|&v| mul(decode(v), decode(v))).collect();
            while level.len() > 1 {
                level = level
                    .chunks(2)
                    .map(|p| {
                        if p.len() == 2 {
                            (f64::from(p[0]) + f64::from(p[1])) as f32
                        } else {
                            p[0]
                        }
                    })
                    .collect();
            }
            let variance = (f64::from(level[0]) / gamma.len() as f64) as f32;
            let shifted = (f64::from(variance) + f64::from(EPSILON)) as f32;
            let root = f64::from(shifted).sqrt() as f32;
            let inverse = (1.0_f64 / f64::from(root)) as f32;
            group
                .iter()
                .zip(gamma)
                .map(move |(&x, &gamma)| {
                    let normalized = mul(decode(x), inverse);
                    encode(mul(
                        if single_round {
                            normalized
                        } else {
                            decode(encode(normalized))
                        },
                        decode(gamma),
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Exact mathematical reduction/normalization in F64, retaining the same two
/// BF16 interfaces. This measures arithmetic error, not deployment parity.
pub fn mathematical(x: &[u16], gamma: &[u16]) -> Vec<u16> {
    x.chunks_exact(gamma.len())
        .flat_map(|group| {
            let mean = group
                .iter()
                .map(|&v| f64::from(decode(v)).powi(2))
                .sum::<f64>()
                / gamma.len() as f64;
            let inverse = 1.0 / (mean + f64::from(EPSILON)).sqrt();
            group
                .iter()
                .zip(gamma)
                .map(move |(&x, &gamma)| {
                    let normalized = encode((f64::from(decode(x)) * inverse) as f32);
                    encode((f64::from(decode(normalized)) * f64::from(decode(gamma))) as f32)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn fixture(width: usize, trial: usize) -> (Vec<u16>, Vec<u16>) {
    let gamma: Vec<_> = (0..width)
        .map(|i| encode([0.75, 1.0078125, -1.5, 0.50390625, 1.9921875, -0.0][i % 6]))
        .collect();
    let mut x: Vec<_> = (0..4 * width)
        .map(|i| {
            let n = ((i * 37 + (i / width) * 11 + trial * 19) % 513) as i32 - 256;
            encode(n as f32 / 32.0)
        })
        .collect();
    for group in 0..4 {
        x[group * width] = 0x8000;
        if width > 1 {
            x[group * width + 1] = 0x0001;
        }
    }
    if trial == 2 {
        x[0] = 0x7f80;
        x[width] = 0x7fc1;
        x[2 * width..3 * width].fill(0x8000);
        x[3 * width] = 0x7f7f;
    }
    (x, gamma)
}

pub fn check_bits(actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if decode(e).is_nan() {
            assert!(decode(a).is_nan(), "NaN class at {index}: {a:04x}");
        } else {
            assert_eq!(a, e, "BF16 bits at {index}");
        }
    }
}
pub fn mathematical_error(actual: &[u16], expected: &[u16]) -> u16 {
    let mut maximum = 0;
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(decode(a).is_finite() && decode(e).is_finite());
        if a & 0x7fff == 0 || e & 0x7fff == 0 {
            assert_eq!(a, e, "zero sign/value");
        } else {
            assert_eq!(a & 0x8000, e & 0x8000);
            maximum = maximum.max((a & 0x7fff).abs_diff(e & 0x7fff));
        }
    }
    assert!(maximum<=MATHEMATICAL_MAX_BF16_ULP,"F64 mathematical oracle exceeded predeclared {MATHEMATICAL_MAX_BF16_ULP} BF16 ULP budget: {maximum}");
    maximum
}
#[allow(dead_code)] // GPU fixtures serialize values; pure oracle tests use bits directly.
pub fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_known_values_preserve_both_zero_signs_and_affine_boundary() {
        check_bits(
            &reference(&[0x3f80, 0xbf80, 0, 0x8000], &[0x3fc0], false),
            &[0x3fc0, 0xbfc0, 0, 0x8000],
        );
    }
    #[test]
    fn bounded_fixtures_distinguish_single_round_and_stay_within_mathematical_budget() {
        for width in WIDTHS {
            for trial in 0..2 {
                let (x, gamma) = fixture(width, trial);
                let expected = reference(&x, &gamma, false);
                mathematical_error(&expected, &mathematical(&x, &gamma));
                if width >= 128 {
                    assert_ne!(
                        expected,
                        reference(&x, &gamma, true),
                        "fixture must expose premature affine fusion"
                    );
                }
            }
        }
    }
}
