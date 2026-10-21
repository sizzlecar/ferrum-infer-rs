//! Candidate-independent finite-domain RMS oracle. F64 intervals enclose the
//! mathematical sum, FP32 reduction error and each sqrt/div rounding boundary.
//! BF16 sets retain BOTH normalization and learned-scale conversions.
#[path = "../../../ferrum-kernels/src/backend/reference/transcendental/oracle/rounding.rs"]
mod rounding;

use anyhow::{bail, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::qwen_image21::QwenImage21Block0QkNormFamily as Family;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
}
fn value(bits: u16) -> f64 {
    f64::from(rounding::decode_bf16(bits))
}
fn finite(bits: u16) -> bool {
    bits & 0x7fff < 0x7f80
}
fn up(x: f64) -> f64 {
    x.next_up()
}
fn down(x: f64) -> f64 {
    x.next_down()
}

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn checked(lo: f64, hi: f64) -> Result<Self> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            bail!("oracle interval escaped its finite ordered domain");
        }
        Ok(Self { lo, hi })
    }
    /// RN32 is monotone. Outward F64 operands enclose the exact operation
    /// before endpoint conversion; no double-rounding equality is assumed.
    fn rn32(self) -> Result<Self> {
        Self::checked(f64::from(self.lo as f32), f64::from(self.hi as f32))
    }
    fn divide_positive(self, divisor: f64) -> Result<Self> {
        if divisor <= 0.0 || !divisor.is_finite() {
            bail!("oracle requires a positive finite divisor");
        }
        Self::checked(down(self.lo / divisor).max(0.0), up(self.hi / divisor))?.rn32()
    }
    fn add_positive(self, rhs: f64) -> Result<Self> {
        Self::checked(down(self.lo + rhs).max(0.0), up(self.hi + rhs))?.rn32()
    }
    fn sqrt(self) -> Result<Self> {
        if self.lo <= 0.0 {
            bail!("RMS denominator is not strictly positive");
        }
        // IEEE RN64 sqrt followed by one outward representable step encloses
        // the exact real square root, including an FP32 midpoint.
        Self::checked(down(self.lo.sqrt()), up(self.hi.sqrt()))?.rn32()
    }
    fn reciprocal(self) -> Result<Self> {
        if self.lo <= 0.0 {
            bail!("RMS root is not strictly positive");
        }
        Self::checked(down(1.0 / self.hi), up(1.0 / self.lo))?.rn32()
    }
}

/// Every nonzero square must be exact FP32 on a binary lattice >= 2^-126.
/// Nonnegative sums then have no underflow; S*(1+gamma) bounds every subtree.
fn square_lattice(square: f64) -> Result<Option<i32>> {
    if square == 0.0 {
        return Ok(None);
    }
    let rounded = square as f32;
    if !rounded.is_finite() || f64::from(rounded) != square {
        bail!("RMS square is not exact finite FP32");
    }
    let raw = rounded.to_bits();
    let exponent = ((raw >> 23) & 0xff) as i32;
    let fraction = raw & 0x7f_ffff;
    let (significand, least) = if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent - 150)
    };
    let lattice = least + significand.trailing_zeros() as i32;
    if lattice < -126 {
        bail!("RMS square lattice does not exclude FP32 underflow");
    }
    Ok(Some(lattice))
}

fn inverse_rms(input: &[u16], epsilon: f32) -> Result<Interval> {
    let width = input.len();
    if width == 0 || width > 4096 || !epsilon.is_finite() || !(1e-12_f32..=1.0).contains(&epsilon) {
        bail!("oracle requires 1..=4096 head width and effective epsilon in [1e-12,1]");
    }
    let (mut sum, mut exact_sum, mut least) = (Interval::point(0.0), 0.0_f64, None::<i32>);
    for &x in input {
        if !finite(x) {
            bail!("RMS input is nonfinite");
        }
        let square = value(x) * value(x); // <=16 significant bits: exact F64.
        if let Some(lattice) = square_lattice(square)? {
            least = Some(least.map_or(lattice, |old| old.min(lattice)));
        }
        exact_sum += square;
        sum = Interval::checked(down(sum.lo + square).max(0.0), up(sum.hi + square))?;
    }
    // If all nonnegative products and their total fit 24 bits on one common
    // normal lattice, EVERY partial sum in EVERY reduction tree is exact.
    // The outward upper endpoint certifies this before exact_sum is used.
    let exact_reduction =
        least.is_none_or(|exponent| up(sum.hi / 2.0_f64.powi(exponent)) <= 16_777_216.0);
    let rounded_sum = if exact_reduction {
        Interval::point(exact_sum)
    } else {
        let nu = (width - 1) as f64 / 16_777_216.0; // exact dyadic, <1.
        let gamma = up(nu / down(1.0 - nu));
        Interval::checked(
            down(sum.lo * down(1.0 - gamma)).max(0.0),
            up(sum.hi * up(1.0 + gamma)),
        )?
    };
    if rounded_sum.hi > f64::from(f32::MAX) {
        bail!("cannot prove overflow-free FP32 reduction");
    }
    rounded_sum
        .divide_positive(width as f64)?
        .add_positive(f64::from(epsilon))?
        .sqrt()?
        .reciprocal()
}

fn ordered(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}
fn from_ordered(key: u16) -> u16 {
    if key & 0x8000 != 0 {
        key & 0x7fff
    } else {
        !key
    }
}

fn normalized_choices(input: u16, gamma: u16, inverse: Interval) -> Result<Vec<u16>> {
    if !finite(gamma) {
        bail!("learned RMS scale is nonfinite");
    }
    let mut result = BTreeSet::new();
    if input & 0x7fff == 0 {
        // Dense's valid reduction may choose either zero sign. Both possible
        // signs propagate through normalization and the learned BF16 scale.
        for zero in [0, 0x8000] {
            result.insert(rounding::multiply_bf16(zero, gamma));
        }
    } else {
        let x = value(input);
        // BF16 x times an FP32 endpoint has <=32 significant bits, so these
        // products are EXACT in F64, including values near FP32 subnormal ties.
        let products = [x * inverse.lo, x * inverse.hi];
        let low = rounding::bf16_from_f32((products[0].min(products[1]) as f32).to_bits());
        let high = rounding::bf16_from_f32((products[0].max(products[1]) as f32).to_bits());
        if !finite(low) || !finite(high) {
            bail!("normalization may overflow BF16");
        }
        let (start, end) = (ordered(low), ordered(high));
        if start > end {
            bail!("normalization BF16 interval is inverted");
        }
        // Enumerate all BF16 normalization boundaries before learned scale.
        // Do not multiply gamma into an unrounded real normalization interval.
        for key in start..=end {
            result.insert(rounding::multiply_bf16(from_ordered(key), gamma));
        }
    }
    if result.iter().any(|&x| !finite(x)) {
        bail!("learned-scale output may overflow BF16");
    }
    Ok(result.into_iter().collect())
}

fn rms_choices(input: &[u16], gamma: &[u16], epsilon: f32) -> Result<Vec<Vec<u16>>> {
    if gamma.is_empty() || input.is_empty() || input.len() % gamma.len() != 0 {
        bail!("invalid grouped RMS geometry");
    }
    let mut result = Vec::with_capacity(input.len());
    for head in input.chunks_exact(gamma.len()) {
        let inverse = inverse_rms(head, epsilon)?;
        for (&x, &weight) in head.iter().zip(gamma) {
            result.push(normalized_choices(x, weight, inverse)?);
        }
    }
    Ok(result)
}

pub struct Prepared {
    input: Vec<u8>,
    expected: [Vec<Vec<u16>>; 4],
    epsilon: f32,
    head_dim: usize,
}

impl Prepared {
    pub fn new(hidden: usize, head_dim: usize, epsilon: f32, weights: [&[u8]; 4]) -> Result<Self> {
        let matrix_bytes = hidden
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .context("Q/K weight shape overflows")?;
        let norm_bytes = head_dim
            .checked_mul(2)
            .context("RMS scale size overflows")?;
        if hidden == 0
            || head_dim == 0
            || hidden % head_dim != 0
            || weights[..2].iter().any(|w| w.len() != matrix_bytes)
            || weights[2..].iter().any(|w| w.len() != norm_bytes)
        {
            bail!("oracle requires exact complete Q/K matrices and learned head scales");
        }
        for weight in &weights[..2] {
            for pair in weight.chunks_exact(2) {
                let magnitude = u16::from_le_bytes([pair[0], pair[1]]) & 0x7fff;
                if magnitude >= 0x7f80 || (magnitude > 0 && magnitude < 0x0080) {
                    bail!("basis oracle requires finite normal-or-zero projection weights");
                }
            }
        }
        let columns: [Vec<u16>; 2] = std::array::from_fn(|branch| {
            (0..hidden)
                .map(|row| bits(weights[branch], row * hidden))
                .collect()
        });
        let projection_choices = |column: &[u16]| {
            column
                .iter()
                .map(|&x| {
                    if x & 0x7fff == 0 {
                        vec![0, 0x8000]
                    } else {
                        vec![x]
                    }
                })
                .collect::<Vec<_>>()
        };
        let gamma_q = (0..head_dim)
            .map(|i| bits(weights[2], i))
            .collect::<Vec<_>>();
        let gamma_k = (0..head_dim)
            .map(|i| bits(weights[3], i))
            .collect::<Vec<_>>();
        let expected = [
            projection_choices(&columns[0]),
            projection_choices(&columns[1]),
            rms_choices(&columns[0], &gamma_q, epsilon)?,
            rms_choices(&columns[1], &gamma_k, epsilon)?,
        ];
        let mut input = vec![0; hidden.checked_mul(2).context("input size overflows")?];
        input[..2].copy_from_slice(&0x3f80_u16.to_le_bytes());
        Ok(Self {
            input,
            expected,
            epsilon,
            head_dim,
        })
    }
    pub fn input(&self) -> &[u8] {
        &self.input
    }
    pub fn declaration(&self) -> Value {
        json!({"input":"one basis[0] row; already normalized/modulated attention entrance", "head_dim":self.head_dim, "effective_epsilon_f32_bits":self.epsilon.to_bits(),
            "projection":"all channels exact real OI column[0], with both signs allowed for exact zero",
            "norm":"independent outward F64 sum-squares; exact common-lattice reduction where proved, else gamma(D-1,2^-24); monotone RN32 interval propagation through /D,+epsilon,sqrt,1/root; enumerate BF16 normalization choices, then exact BF16-product -> RN32 -> independent integer BF16 RNE",
            "domain":"head width 1..4096; finite normal-or-zero projection matrices; square exact FP32 and nonzero lattice >=2^-126; no possible FP32 sum/denominator or BF16 boundary overflow; finite learned scales; IEEE RN-even gradual underflow, correctly-rounded sqrt/div",
            "expected_bf16_bits":self.expected, "nonunique_counts":self.expected.iter().map(|values| values.iter().filter(|allowed| allowed.len()>1).count()).collect::<Vec<_>>(),
            "scope":"all four output ports for one basis row; no RoPE, attention, complete block, general-input norm qualification, or Torch/vLLM parity",
            "candidate_independence":"input, intervals and every allowed set prepared before runtime creation; no candidate intermediate supplies an expectation"})
    }
    pub fn compare(&self, outputs: &[TensorExecutionOutput]) -> Result<Value> {
        let ids = [
            Family::QUERY_ID,
            Family::KEY_ID,
            Family::QUERY_NORMALIZED_ID,
            Family::KEY_NORMALIZED_ID,
        ];
        if outputs.len() != ids.len() {
            bail!("Q/K norm requires exactly four output ports");
        }
        let mut reports = Vec::new();
        let mut total_failures = 0;
        for (id, expected) in ids.into_iter().zip(&self.expected) {
            let matching = outputs
                .iter()
                .filter(|output| output.value_id().as_str() == id)
                .collect::<Vec<_>>();
            if matching.len() != 1 || matching[0].bytes().len() != expected.len() * 2 {
                bail!("Q/K output {id} missing or wrong size");
            }
            let raw = matching[0].bytes().bytes();
            let failures = expected.iter().enumerate().filter(|(index, allowed)| !allowed.contains(&bits(raw,*index))).map(|(index, allowed)| json!({"channel":index,"actual_bits":bits(raw,index),"allowed_bits":allowed})).collect::<Vec<_>>();
            total_failures += failures.len();
            reports.push(json!({"port":id, "samples":expected.len(), "failures":failures}));
        }
        Ok(
            json!({"passed":total_failures==0, "failures":total_failures, "ports":reports, "scope":"all Q/K pre/post norm channels of one independently prepared basis row"}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learned_scale_and_groups_have_hand_computed_outputs() {
        let x = [
            0x3f00, 0xbf00, 0x3f00, 0xbf00, 0xbf00, 0x3f00, 0xbf00, 0x3f00,
        ];
        let choices = rms_choices(&x, &[0x4000, 0x4080, 0x40c0, 0x4100], 0.75).unwrap();
        assert_eq!(
            choices,
            [0x3f80, 0xc000, 0x4040, 0xc080, 0xbf80, 0x4000, 0xc040, 0x4080].map(|bits| vec![bits])
        );
    }
    #[test]
    fn normalization_rounds_before_learned_scale_and_rejects_omission() {
        let epsilon = (112375.0_f64 / 147456.0) as f32;
        let choices = rms_choices(&[0x3f81; 128], &[0x3f81; 128], epsilon).unwrap();
        assert!(choices.iter().all(|allowed| allowed == &[0x3f44]));
        assert!(choices.iter().all(|allowed| !allowed.contains(&0x3f43)));
    }
    #[test]
    fn exact_zero_signs_are_independent_allowed_branches() {
        let choices = rms_choices(&[0, 0x8000], &[0x3f80, 0xbf80], 1e-6).unwrap();
        assert_eq!(choices, vec![vec![0, 0x8000], vec![0, 0x8000]]);
    }
    #[test]
    fn invalid_geometry_nonfinite_and_underflow_overflow_domains_fail() {
        assert!(rms_choices(&[0x3f80], &[], 1e-6).is_err());
        assert!(rms_choices(&[0x3f80], &[0x3f80, 0x3f80], 1e-6).is_err());
        for x in [0x7fc0, 0x7f80, 0x7f7f, 0x0080] {
            assert!(rms_choices(&[x], &[0x3f80], 1e-6).is_err());
        }
        assert!(rms_choices(&[0x3f80], &[0x7f80], 1e-6).is_err());
        assert!(rms_choices(&[0x3f80], &[0x3f80], 0.0).is_err());
    }
}
