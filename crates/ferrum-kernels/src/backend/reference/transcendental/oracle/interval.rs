//! Independent real-function enclosures. No host or candidate exp/tanh calls.
//!
//! Basic binary64 arithmetic is RN-even with gradual underflow. Moving the
//! rounded result one representable value outwards encloses the exact result.

#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub(super) lo: f64,
    pub(super) hi: f64,
}

impl Interval {
    pub(super) fn new(lo: f64, hi: f64) -> Result<Self, String> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(format!("invalid finite interval [{lo:?}, {hi:?}]"));
        }
        Ok(Self { lo, hi })
    }

    fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn add(self, rhs: Self) -> Result<Self, String> {
        Self::new((self.lo + rhs.lo).next_down(), (self.hi + rhs.hi).next_up())
    }

    fn subtract(self, rhs: Self) -> Result<Self, String> {
        Self::new((self.lo - rhs.hi).next_down(), (self.hi - rhs.lo).next_up())
    }

    fn multiply_nonnegative(self, rhs: Self) -> Result<Self, String> {
        if self.lo < 0.0 || rhs.lo < 0.0 {
            return Err("nonnegative interval multiplication received a negative bound".into());
        }
        Self::new(
            (self.lo * rhs.lo).next_down().max(0.0),
            (self.hi * rhs.hi).next_up(),
        )
    }

    fn divide_nonnegative(self, rhs: Self) -> Result<Self, String> {
        if self.lo < 0.0 || rhs.lo <= 0.0 {
            return Err(
                "interval division requires nonnegative numerator and positive divisor".into(),
            );
        }
        Self::new(
            (self.lo / rhs.hi).next_down().max(0.0),
            (self.hi / rhs.lo).next_up(),
        )
    }

    fn negative(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

/// For -256 < t < 128. Outside that domain the caller uses analytic FP32
/// overflow/underflow classifications, not an approximate finite enclosure.
pub(super) fn exp(t: f64) -> Result<Interval, String> {
    if !t.is_finite() || t <= -256.0 || t >= 128.0 {
        return Err(format!("exp interval argument outside proof domain: {t:?}"));
    }
    // All callers provide binary32 values, so this power-of-two reduction is
    // exact and |a| < 1/2. No ln(2) approximation participates in the proof.
    let a = Interval::point(t.abs() / 512.0);
    let mut term = Interval::point(1.0);
    let mut sum = term;
    for k in 1..=24 {
        term = term
            .multiply_nonnegative(a)?
            .divide_nonnegative(Interval::point(f64::from(k)))?;
        sum = sum.add(term)?;
    }
    // Subsequent term ratios are at most a/26. The lower denominator and upper
    // term25 enclose the entire positive Taylor remainder, including rounding.
    let term25 = term
        .multiply_nonnegative(a)?
        .divide_nonnegative(Interval::point(25.0))?;
    let denominator =
        Interval::point(1.0).subtract(a.divide_nonnegative(Interval::point(26.0))?)?;
    let tail = term25.divide_nonnegative(denominator)?;
    let mut enclosure = Interval::new(sum.lo, (sum.hi + tail.hi).next_up())?;
    for _ in 0..9 {
        enclosure = enclosure.multiply_nonnegative(enclosure)?;
    }
    if t < 0.0 {
        Interval::point(1.0).divide_nonnegative(enclosure)
    } else {
        Ok(enclosure)
    }
}

/// The tiny and saturation domains have separate analytic RN32 decisions.
pub(super) fn tanh(x: f64) -> Result<Interval, String> {
    let magnitude = x.abs();
    if !x.is_finite() || magnitude <= 1.0 / 4096.0 || magnitude >= 16.0 {
        return Err(format!(
            "tanh interval argument outside middle domain: {x:?}"
        ));
    }
    let z = exp(-2.0 * magnitude)?;
    let numerator = Interval::point(1.0).subtract(z)?;
    let denominator = Interval::point(1.0).add(z)?;
    let positive = numerator.divide_nonnegative(denominator)?;
    Ok(if x.is_sign_negative() {
        positive.negative()
    } else {
        positive
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_taylor_enclosure_obeys_independent_rational_bounds() {
        // S5(e) = 163/60; the remaining series is at most (1/6!)/(1-1/7).
        let e = exp(1.0).unwrap();
        assert!(e.lo > 163.0 / 60.0);
        assert!(e.hi < 11743.0 / 4320.0);
        let inverse = exp(-1.0).unwrap();
        assert!(inverse.lo <= 1.0 / e.hi);
        assert!(inverse.hi >= 1.0 / e.lo);
        let zero = exp(0.0).unwrap();
        assert!(zero.lo <= 1.0 && zero.hi >= 1.0);
        // These are domain limits, not saturation guesses or candidate values.
        assert!(exp(-256.0).is_err());
        assert!(exp(128.0).is_err());
        assert!(exp(f64::NAN).is_err());
    }

    #[test]
    fn outward_arithmetic_preserves_underflow_and_reverses_reciprocal_endpoints() {
        let tiny = Interval::point(f64::from_bits(1));
        let product = tiny.multiply_nonnegative(Interval::point(0.5)).unwrap();
        assert_eq!(product.lo, 0.0);
        assert_eq!(product.hi.to_bits(), 1);
        let reciprocal = Interval::point(1.0)
            .divide_nonnegative(Interval::new(2.0, 4.0).unwrap())
            .unwrap();
        assert!(reciprocal.lo <= 0.25 && reciprocal.hi >= 0.5);
        assert!(Interval::new(2.0, 1.0).is_err());
        assert!(Interval::new(0.0, f64::INFINITY).is_err());
        assert!(tiny
            .divide_nonnegative(Interval::new(0.0, 1.0).unwrap())
            .is_err());
    }

    #[test]
    fn tanh_uses_symmetric_finite_intervals_in_its_middle_domain() {
        let positive = tanh(1.0).unwrap();
        let negative = tanh(-1.0).unwrap();
        assert_eq!(negative.lo.to_bits(), (-positive.hi).to_bits());
        assert_eq!(negative.hi.to_bits(), (-positive.lo).to_bits());
        // Independent e bounds propagated through the increasing rational map.
        let e_lo = 163.0 / 60.0;
        let e_hi = 11743.0 / 4320.0;
        assert!(positive.lo > (e_lo * e_lo - 1.0) / (e_lo * e_lo + 1.0));
        assert!(positive.hi < (e_hi * e_hi - 1.0) / (e_hi * e_hi + 1.0));
        assert!(tanh(1.0 / 4096.0).is_err());
        assert!(tanh(16.0).is_err());
    }
}
