//! Independent exact BF16 dot product, with checked i128 capacity. A failure
//! to represent the common lattice is unresolved, never a floating fallback.
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Exact {
    #[serde(serialize_with = "decimal")]
    pub numerator: i128,
    pub exponent: i32,
}
fn decimal<S: serde::Serializer>(v: &i128, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(&v.to_string())
}
impl Exact {
    fn new(n: i128, e: i32) -> Self {
        if n == 0 {
            Self {
                numerator: 0,
                exponent: 0,
            }
        } else {
            let z = n.unsigned_abs().trailing_zeros();
            Self {
                numerator: n / (1_i128 << z),
                exponent: e + z as i32,
            }
        }
    }
    pub fn approximate(self) -> f64 {
        self.numerator as f64 * 2_f64.powi(self.exponent)
    }
    pub fn subtract(self, rhs: Self) -> Result<Self> {
        ensure!(
            self.numerator != i128::MIN && rhs.numerator != i128::MIN,
            "unresolved: dyadic operand magnitude exceeds i128::MAX"
        );
        if rhs.numerator == 0 {
            return Ok(self);
        }
        if self.numerator == 0 {
            return Ok(Self::new(-rhs.numerator, rhs.exponent));
        }
        let e = self.exponent.min(rhs.exponent);
        let numerator = scale(self.numerator, (self.exponent - e) as u32)?
            .checked_sub(scale(rhs.numerator, (rhs.exponent - e) as u32)?)
            .filter(|value| *value != i128::MIN)
            .context("unresolved: dyadic subtraction magnitude exceeds i128::MAX")?;
        Ok(Self::new(numerator, e))
    }
    pub fn round_bf16(self) -> Result<u16> {
        let negative = self.numerator < 0;
        let mag = self.numerator.unsigned_abs();
        if mag == 0 {
            return Ok(0);
        }
        let mut top = 127 - mag.leading_zeros() as i32 + self.exponent;
        let quantum = if top >= -126 { top - 7 } else { -133 };
        let shift = quantum - self.exponent;
        let mut q = if shift > 0 {
            round_shift(mag, shift as u32)
        } else {
            mag.checked_mul(
                1_u128
                    .checked_shl((-shift) as u32)
                    .context("unresolved BF16 rounding shift")?,
            )
            .context("unresolved BF16 rounding extent")?
        };
        let sign = if negative { 0x8000 } else { 0 };
        if top < -126 {
            ensure!(q <= 128, "invalid subnormal rounding");
            return Ok(sign | q as u16);
        }
        if q == 256 {
            q = 128;
            top += 1;
        }
        ensure!(top <= 127, "exact result rounds beyond finite BF16 domain");
        ensure!((128..256).contains(&q), "invalid normal significand");
        Ok(sign | (((top + 127) as u16) << 7) | (q as u16 - 128))
    }
}
fn round_shift(n: u128, shift: u32) -> u128 {
    if shift >= 128 {
        return if shift == 128 && n > (1_u128 << 127) {
            1
        } else {
            0
        };
    }
    let q = n >> shift;
    let mask = (1_u128 << shift) - 1;
    let rem = n & mask;
    let half = 1_u128 << (shift - 1);
    q + u128::from(rem > half || (rem == half && q & 1 == 1))
}
fn scale(n: i128, shift: u32) -> Result<i128> {
    let factor = 1_u128
        .checked_shl(shift)
        .context("unresolved: common lattice shift >=128")?;
    let m = n
        .unsigned_abs()
        .checked_mul(factor)
        .filter(|m| *m <= i128::MAX as u128)
        .context("unresolved: common lattice term exceeds i128")?;
    Ok(if n < 0 { -(m as i128) } else { m as i128 })
}
pub fn bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])
}
pub fn value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}
pub fn from_bf16(bits: u16) -> Result<Exact> {
    decode(bits)
}
fn decode(bits: u16) -> Result<Exact> {
    let raw = bits & 0x7fff;
    ensure!(raw < 0x7f80, "nonfinite BF16 operand");
    let exp = (raw >> 7) as i32;
    let m = if exp == 0 {
        raw & 127
    } else {
        128 + (raw & 127)
    };
    Ok(Exact::new(
        if bits & 0x8000 == 0 {
            i128::from(m)
        } else {
            -i128::from(m)
        },
        if exp == 0 { -133 } else { exp - 134 },
    ))
}
fn product(a: u16, b: u16) -> Result<Exact> {
    let a = decode(a)?;
    let b = decode(b)?;
    Ok(Exact::new(
        a.numerator * b.numerator,
        a.exponent + b.exponent,
    ))
}
fn upward(v: f64) -> Result<f64> {
    ensure!(v.is_finite() && v >= 0.0, "nonfinite bound");
    let u = if v == 0.0 { 0.0 } else { v.next_up() };
    ensure!(u.is_finite(), "bound overflow");
    Ok(u)
}

#[derive(Debug, Serialize)]
pub struct Dot {
    pub exact: Exact,
    pub exact_nearest_bf16: u16,
    pub sum_abs_upper: f64,
    pub fp32_error_bound: f64,
    pub bf16_error_bound: f64,
    pub sequential_fp32_bits: u32,
    pub sequential_bf16_bits: u16,
    pub lower_allowed: f64,
    pub upper_allowed: f64,
}
impl Dot {
    pub fn prepare(x: &[u8], w: &[u8]) -> Result<Self> {
        ensure!(
            !x.is_empty() && x.len() == w.len() && x.len() % 2 == 0,
            "dot needs equal nonempty BF16 rows"
        );
        let k = x.len() / 2;
        ensure!(k < 8_388_608, "2K*2^-24 must be <1");
        let mut exponent = 0;
        let mut any = false;
        for i in 0..k {
            let p = product(bits(x, i), bits(w, i))?;
            if p.numerator != 0 {
                ensure!(
                    p.exponent >= -126,
                    "unresolved: product lattice does not exclude FP32 underflow"
                );
                let p64 = p.approximate();
                ensure!(
                    p64.is_finite() && f64::from(p64 as f32) == p64,
                    "product is not exact finite FP32"
                );
                exponent = if any {
                    exponent.min(p.exponent)
                } else {
                    p.exponent
                };
                any = true;
            }
        }
        let mut sum = 0_i128;
        let mut absolute = 0_u128;
        for i in 0..k {
            let p = product(bits(x, i), bits(w, i))?;
            if p.numerator == 0 {
                continue;
            }
            let n = scale(p.numerator, (p.exponent - exponent) as u32)?;
            absolute = absolute
                .checked_add(n.unsigned_abs())
                .filter(|a| *a <= i128::MAX as u128)
                .context("unresolved: exact absolute sum exceeds i128")?;
            sum = sum
                .checked_add(n)
                .context("unresolved: exact dot exceeds i128")?;
        }
        let exact = Exact::new(sum, exponent);
        let s = upward(upward(absolute as f64)? * 2_f64.powi(exponent))?;
        let numerator = upward((2 * k) as f64 * 2_f64.powi(-24))?;
        let gamma = upward(numerator / (1.0 - numerator).next_down())?;
        ensure!(
            upward(s * upward(1.0 + gamma)?)? <= value(0x7f7f),
            "unresolved: overflow-free accumulation/final BF16 domain not established"
        );
        let e32 = upward(gamma * s)?;
        let center = exact.approximate();
        let (lo, hi) = if sum == 0 {
            (0.0, 0.0)
        } else {
            (center.next_down(), center.next_up())
        };
        let rounding = upward(upward(lo.abs().max(hi.abs()) + e32)? / 256.0)?;
        let bound = upward(upward(e32 + rounding)? + 2_f64.powi(-134))?;
        let mut sequential = 0_f32;
        for i in 0..k {
            let p = f32::from_bits(u32::from(bits(x, i)) << 16)
                * f32::from_bits(u32::from(bits(w, i)) << 16);
            sequential = sequential + p;
        }
        ensure!(
            sequential.is_finite(),
            "sequential FP32 replay escaped established finite domain"
        );
        Ok(Self {
            exact,
            exact_nearest_bf16: exact.round_bf16()?,
            sum_abs_upper: s,
            fp32_error_bound: e32,
            bf16_error_bound: bound,
            sequential_fp32_bits: sequential.to_bits(),
            sequential_bf16_bits: round_f32(sequential.to_bits()),
            lower_allowed: (lo - bound).next_down(),
            upper_allowed: (hi + bound).next_up(),
        })
    }
    pub fn compatible(&self, actual: u16) -> bool {
        let y = value(actual);
        y.is_finite() && y >= self.lower_allowed && y <= self.upper_allowed
    }
}
pub fn round_f32(raw: u32) -> u16 {
    if raw & 0x7f80_0000 == 0x7f80_0000 {
        return ((raw >> 16) as u16) | u16::from(raw & 0x007f_ffff != 0) * 0x40;
    }
    ((raw.wrapping_add(0x7fff + ((raw >> 16) & 1))) >> 16) as u16
}

/// Exact signed distance to the midpoint of two observed finite values. A
/// midpoint is a BF16 rounding boundary only when the values are adjacent.
pub fn midpoint_distance(exact: Exact, a: u16, b: u16) -> Result<Exact> {
    let a = decode(a)?;
    let b = decode(b)?;
    let sum = a.subtract(Exact {
        numerator: -b.numerator,
        exponent: b.exponent,
    })?;
    exact.subtract(Exact::new(sum.numerator, sum.exponent - 1))
}
pub fn adjacent(a: u16, b: u16) -> bool {
    let ordered = |v: u16| {
        if v & 0x8000 == 0 {
            u32::from(v) + 0x8000
        } else {
            0x8000 - u32::from(v & 0x7fff)
        }
    };
    value(a).is_finite() && value(b).is_finite() && ordered(a).abs_diff(ordered(b)) == 1
}
