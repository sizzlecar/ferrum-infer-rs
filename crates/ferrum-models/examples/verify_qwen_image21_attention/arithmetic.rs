//! Outward F64 enclosures for the declared FP32 operation boundaries. Distinct
//! legal input choices are joined, never intersected. No candidate arithmetic
//! routine or host exp/tanh supplies the reference.
use super::{finite, real_interval, rounding, value, Probe};
use anyhow::{bail, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub(super) lo: f64,
    pub(super) hi: f64,
}
impl Interval {
    pub(super) fn new(lo: f64, hi: f64) -> Result<Self> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            bail!("unresolved: finite ordered oracle interval required");
        }
        Ok(Self { lo, hi })
    }
    pub(super) fn point(x: f64) -> Result<Self> {
        Self::new(x, x)
    }
    pub(super) fn choices(xs: &[u16]) -> Result<Self> {
        if xs.is_empty() || xs.iter().any(|&x| !finite(x)) {
            bail!("unresolved: nonempty finite BF16 choices required");
        }
        Self::new(
            xs.iter().map(|&x| value(x)).fold(f64::INFINITY, f64::min),
            xs.iter()
                .map(|&x| value(x))
                .fold(f64::NEG_INFINITY, f64::max),
        )
    }
    fn rn32(self) -> Result<Self> {
        Self::new(f64::from(self.lo as f32), f64::from(self.hi as f32))
    }
    fn add(self, rhs: Self) -> Result<Self> {
        Self::new((self.lo + rhs.lo).next_down(), (self.hi + rhs.hi).next_up())?.rn32()
    }
    fn subtract(self, rhs: Self) -> Result<Self> {
        Self::new((self.lo - rhs.hi).next_down(), (self.hi - rhs.lo).next_up())?.rn32()
    }
    fn multiply(self, rhs: Self) -> Result<Self> {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        Self::new(
            products
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                .next_down(),
            products
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        )?
        .rn32()
    }
    fn reciprocal(self) -> Result<Self> {
        if self.lo <= 0.0 {
            bail!("unresolved: softmax denominator must be strictly positive");
        }
        Self::new((1.0 / self.hi).next_down(), (1.0 / self.lo).next_up())?.rn32()
    }
    fn maximum(self, rhs: Self) -> Self {
        Self {
            lo: self.lo.max(rhs.lo),
            hi: self.hi.max(rhs.hi),
        }
    }
    /// exp is only used for score - max(score), hence <= 0. Enforce that
    /// proven relation after interval subtraction loses the correlation.
    fn nonpositive(self) -> Result<Self> {
        Self::new(self.lo.min(0.0), self.hi.min(0.0))
    }
    pub(super) fn bf16_choices(self) -> Result<Vec<u16>> {
        let lo = rounding::bf16_from_f32((self.lo as f32).to_bits());
        let hi = rounding::bf16_from_f32((self.hi as f32).to_bits());
        if !finite(lo) || !finite(hi) {
            bail!("unresolved: attention BF16 conversion may overflow");
        }
        let start = ordered(lo);
        let end = ordered(hi);
        if start > end || u32::from(end) - u32::from(start) + 1 > 256 {
            bail!("unresolved: attention interval exceeds predeclared 256 BF16 alternatives");
        }
        Ok((start..=end).map(unordered).collect())
    }
}
fn ordered(x: u16) -> u16 {
    if x & 0x8000 != 0 {
        !x
    } else {
        x | 0x8000
    }
}
fn unordered(x: u16) -> u16 {
    if x & 0x8000 != 0 {
        x & 0x7fff
    } else {
        !x
    }
}

/// A conservative UNION envelope for distinct legal FP32 arguments. For every
/// input t in [lo,hi], monotonic true exp and monotonic RN32 put its reference
/// cell between RN32(exp(lo)) and RN32(exp(hi)). Expanding these outer endpoints
/// by four finite neighbors therefore contains every permitted branch. This
/// is intentionally not the same-input uncertainty intersection from R18.
/// exp(0)=1 is exact; finite nonzero classification cannot expand through zero.
fn exponential(input: Interval) -> Result<Interval> {
    if input.lo == 0.0 && input.hi == 0.0 {
        return Interval::point(1.0);
    }
    if input.lo <= -256.0 || input.hi > 0.0 {
        bail!("unresolved: attention exp input outside (-256,0]");
    }
    let lower = real_interval::exp(input.lo).map_err(anyhow::Error::msg)?;
    let upper = if input.hi == 0.0 {
        real_interval::Interval::new(1.0, 1.0).map_err(anyhow::Error::msg)?
    } else {
        real_interval::exp(input.hi).map_err(anyhow::Error::msg)?
    };
    let lo = (lower.lo as f32).to_bits();
    let hi = (upper.hi as f32).to_bits();
    if !(1..0x7f80_0000).contains(&lo) || !(1..0x7f80_0000).contains(&hi) || lo > hi {
        bail!("unresolved: exp reference crosses finite-nonzero classification");
    }
    Interval::new(
        f64::from(f32::from_bits(lo.saturating_sub(4).max(1))),
        f64::from(f32::from_bits(hi.saturating_add(4).min(0x7f7f_ffff))),
    )
}
fn pairwise(products: &[Interval]) -> Result<Interval> {
    match products {
        [] => bail!("unresolved: empty dot"),
        [x] => Ok(*x),
        _ => {
            let mid = products.len() / 2;
            pairwise(&products[..mid])?.add(pairwise(&products[mid..])?)
        }
    }
}
fn score(query: &[Vec<u16>], key: &[Vec<u16>], scale: f32) -> Result<Interval> {
    if query.len() != key.len() || query.is_empty() {
        bail!("unresolved: score dimensions differ");
    }
    let products = query
        .iter()
        .zip(key)
        .map(|(q, k)| Interval::choices(q)?.multiply(Interval::choices(k)?))
        .collect::<Result<Vec<_>>>()?;
    pairwise(&products)?.multiply(Interval::point(f64::from(scale))?)
}

pub(super) fn attention_choices(
    q: &[Vec<u16>],
    k: &[Vec<u16>],
    v: &[Vec<u16>],
    hidden: usize,
    head_dim: usize,
    scale: f32,
    probe: Probe,
) -> Result<Vec<Vec<u16>>> {
    let rows = probe.rows();
    if hidden == 0
        || head_dim == 0
        || hidden % head_dim != 0
        || q.len() != rows * hidden
        || k.len() != q.len()
        || v.len() != q.len()
        || !scale.is_finite()
        || scale <= 0.0
    {
        bail!("unresolved: attention oracle shape or scale mismatch");
    }
    let mut output = vec![Vec::new(); q.len()];
    for row in 0..rows {
        let keys = match probe {
            Probe::Single => 1,
            Probe::CausalPair => row + 1,
            Probe::FullPair => 2,
        };
        for head in 0..hidden / head_dim {
            let first = row * hidden + head * head_dim;
            let base = head * head_dim;
            // Even the one-key path evaluates QK in the provider; prove the
            // score finite before using the exact attention=V identity.
            let s0 = score(
                &q[first..first + head_dim],
                &k[base..base + head_dim],
                scale,
            )?;
            if keys == 1 {
                for c in 0..head_dim {
                    if v[base + c].is_empty() || v[base + c].iter().any(|&x| !finite(x)) {
                        bail!("unresolved: single-key value is nonfinite");
                    }
                    output[first + c] = v[base + c].clone();
                }
                continue;
            }
            let second = hidden + base;
            let s1 = score(
                &q[first..first + head_dim],
                &k[second..second + head_dim],
                scale,
            )?;
            let maximum = s0.maximum(s1);
            let alpha = exponential(s0.subtract(maximum)?.nonpositive()?)?;
            let probability = exponential(s1.subtract(maximum)?.nonpositive()?)?;
            // l starts at exactly 1. Multiplication by 1 is exact; the sum is
            // still one explicit FP32 RN boundary. No stable/F64 rewrite.
            let denominator = alpha.add(probability)?;
            let inverse = denominator.reciprocal()?;
            for c in 0..head_dim {
                let old = Interval::choices(&v[base + c])?.multiply(alpha)?;
                let new = Interval::choices(&v[second + c])?.multiply(probability)?;
                output[first + c] = old.add(new)?.multiply(inverse)?.bf16_choices()?;
            }
        }
    }
    Ok(output)
}
