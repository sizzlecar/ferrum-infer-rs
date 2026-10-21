//! Classification-first finite-step intersections and small typed output sets.

use super::interval::Interval;
use super::rounding::{bf16_from_f32, multiply_bf16};

const STEPS: u32 = 4;
const MAX_BRANCHES: usize = 9; // Intersection is a subset of one 4-step neighborhood.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Sign {
    Positive,
    Negative,
}

impl Sign {
    fn from_bits(bits: u32) -> Self {
        if bits & 0x8000_0000 == 0 {
            Self::Positive
        } else {
            Self::Negative
        }
    }

    fn bits(self) -> u32 {
        match self {
            Self::Positive => 0,
            Self::Negative => 0x8000_0000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fp32Value {
    FiniteNonZero(u32),
    Zero(Sign),
    Infinity(Sign),
    Nan,
}

impl Fp32Value {
    pub(super) fn from_bits(bits: u32) -> Self {
        let magnitude = bits & 0x7fff_ffff;
        match magnitude {
            0 => Self::Zero(Sign::from_bits(bits)),
            0x7f80_0000 => Self::Infinity(Sign::from_bits(bits)),
            0x7f80_0001.. => Self::Nan,
            _ => Self::FiniteNonZero(bits),
        }
    }

    pub(super) fn bits(self) -> u32 {
        match self {
            Self::FiniteNonZero(bits) => bits,
            Self::Zero(sign) => sign.bits(),
            Self::Infinity(sign) => sign.bits() | 0x7f80_0000,
            Self::Nan => 0x7fc0_0000,
        }
    }
}

fn ordered(bits: u32) -> u32 {
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

fn unordered(key: u32) -> u32 {
    if key & 0x8000_0000 != 0 {
        key ^ 0x8000_0000
    } else {
        !key
    }
}

#[derive(Clone, Debug)]
pub(super) enum AllowedF32 {
    Exact(Fp32Value),
    FiniteSteps {
        reference_lo: u32,
        reference_hi: u32,
        allowed_lo: u32,
        allowed_hi: u32,
    },
}

impl AllowedF32 {
    pub(super) fn exact(bits: u32) -> Self {
        Self::Exact(Fp32Value::from_bits(bits))
    }

    pub(super) fn around_rounded(bits: u32) -> Result<Self, String> {
        Self::from_rounded_range(bits, bits)
    }

    pub(super) fn from_interval(interval: Interval) -> Result<Self, String> {
        Self::from_rounded_range(
            (interval.lo as f32).to_bits(),
            (interval.hi as f32).to_bits(),
        )
    }

    fn from_rounded_range(lo: u32, hi: u32) -> Result<Self, String> {
        let low_class = Fp32Value::from_bits(lo);
        let high_class = Fp32Value::from_bits(hi);
        match (low_class, high_class) {
            (Fp32Value::FiniteNonZero(low), Fp32Value::FiniteNonZero(high)) => {
                if Sign::from_bits(low) != Sign::from_bits(high) {
                    return Err(
                        "unresolved mathematical reference crosses the zero classification".into(),
                    );
                }
                let reference_lo = ordered(low);
                let reference_hi = ordered(high);
                if reference_lo > reference_hi {
                    return Err("mathematical rounded range has reversed endpoints".into());
                }
                // For every possible rounded reference r, candidate must lie
                // in N4(r). The intersection is [max(r)-4, min(r)+4].
                let (class_lo, class_hi) = match Sign::from_bits(low) {
                    Sign::Positive => (ordered(1), ordered(0x7f7f_ffff)),
                    Sign::Negative => (ordered(0xff7f_ffff), ordered(0x8000_0001)),
                };
                let allowed_lo = reference_hi.saturating_sub(STEPS).max(class_lo);
                let allowed_hi = reference_lo.saturating_add(STEPS).min(class_hi);
                if allowed_lo > allowed_hi {
                    return Err(format!(
                        "empty 4-step intersection for [{low:08x},{high:08x}]"
                    ));
                }
                Ok(Self::FiniteSteps {
                    reference_lo,
                    reference_hi,
                    allowed_lo,
                    allowed_hi,
                })
            }
            (Fp32Value::Zero(a), Fp32Value::Zero(b)) if a == b => Ok(Self::Exact(low_class)),
            (Fp32Value::Infinity(a), Fp32Value::Infinity(b)) if a == b => {
                Ok(Self::Exact(low_class))
            }
            _ => Err(format!(
                "unresolved mathematical reference classifications: {low_class:?}, {high_class:?}"
            )),
        }
    }

    pub(super) fn check(&self, actual: u32) -> Result<(), String> {
        let value = Fp32Value::from_bits(actual);
        let accepted = match self {
            Self::Exact(expected) => value == *expected,
            Self::FiniteSteps {
                allowed_lo,
                allowed_hi,
                ..
            } => {
                matches!(value, Fp32Value::FiniteNonZero(_))
                    && (*allowed_lo..=*allowed_hi).contains(&ordered(actual))
            }
        };
        if accepted {
            Ok(())
        } else {
            Err(format!("FP32 candidate {actual:08x} rejected by {self:?}"))
        }
    }

    pub(super) fn try_for_each(
        &self,
        mut visit: impl FnMut(Fp32Value) -> Result<(), String>,
    ) -> Result<(), String> {
        match self {
            Self::Exact(value) => visit(*value),
            Self::FiniteSteps {
                allowed_lo,
                allowed_hi,
                ..
            } => {
                for key in *allowed_lo..=*allowed_hi {
                    visit(Fp32Value::from_bits(unordered(key)))?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Exact(_) => 1,
            Self::FiniteSteps {
                allowed_lo,
                allowed_hi,
                ..
            } => (allowed_hi - allowed_lo + 1) as usize,
        }
    }

    pub(super) fn reference_is_ambiguous(&self) -> bool {
        matches!(self, Self::FiniteSteps { reference_lo, reference_hi, .. } if reference_lo != reference_hi)
    }

    /// An observed upper bound against all possible mathematical RN32 cells.
    /// The caller checks membership first. This is reporting, not a tolerance.
    pub(super) fn observed_step_bound(&self, actual: u32) -> Option<u32> {
        match self {
            Self::Exact(Fp32Value::FiniteNonZero(_)) => Some(0),
            Self::Exact(_) => None,
            Self::FiniteSteps {
                reference_lo,
                reference_hi,
                ..
            } => Some(
                ordered(actual)
                    .abs_diff(*reference_lo)
                    .max(ordered(actual).abs_diff(*reference_hi)),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Bf16Value {
    FiniteNonZero(u16),
    Zero(Sign),
    Infinity(Sign),
    Nan,
}

impl Bf16Value {
    pub(super) fn from_bits(bits: u16) -> Self {
        match Fp32Value::from_bits(u32::from(bits) << 16) {
            Fp32Value::FiniteNonZero(_) => Self::FiniteNonZero(bits),
            Fp32Value::Zero(sign) => Self::Zero(sign),
            Fp32Value::Infinity(sign) => Self::Infinity(sign),
            Fp32Value::Nan => Self::Nan,
        }
    }

    fn bits(self) -> u16 {
        match self {
            Self::FiniteNonZero(bits) => bits,
            Self::Zero(sign) => (sign.bits() >> 16) as u16,
            Self::Infinity(sign) => ((sign.bits() | 0x7f80_0000) >> 16) as u16,
            Self::Nan => 0x7fc0,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct AllowedBf16 {
    values: [Bf16Value; MAX_BRANCHES],
    length: usize,
}

impl AllowedBf16 {
    pub(super) fn empty() -> Self {
        Self {
            values: [Bf16Value::Nan; MAX_BRANCHES],
            length: 0,
        }
    }

    pub(super) fn exact(bits: u16) -> Self {
        let mut result = Self::empty();
        result.values[0] = Bf16Value::from_bits(bits);
        result.length = 1;
        result
    }

    pub(super) fn insert(&mut self, bits: u16) -> Result<(), String> {
        let value = Bf16Value::from_bits(bits);
        if self.values[..self.length].contains(&value) {
            return Ok(());
        }
        if self.length == MAX_BRANCHES {
            return Err("BF16 result set exceeded the proved nine-branch capacity".into());
        }
        self.values[self.length] = value;
        self.length += 1;
        Ok(())
    }

    pub(super) fn from_fp32(allowed: &AllowedF32) -> Result<Self, String> {
        let mut result = Self::empty();
        allowed.try_for_each(|value| result.insert(bf16_from_f32(value.bits())))?;
        Ok(result)
    }

    pub(super) fn multiplied_by(&self, up: u16) -> Result<Self, String> {
        let mut result = Self::empty();
        for value in &self.values[..self.length] {
            result.insert(multiply_bf16(value.bits(), up))?;
        }
        if result.length == 0 {
            return Err("cannot propagate an empty activation set".into());
        }
        Ok(result)
    }

    pub(super) fn check(&self, actual: u16) -> Result<(), String> {
        if self.values[..self.length].contains(&Bf16Value::from_bits(actual)) {
            Ok(())
        } else {
            Err(format!(
                "BF16 candidate {actual:04x} rejected by {:?}",
                &self.values[..self.length]
            ))
        }
    }

    pub(super) fn len(&self) -> usize {
        self.length
    }

    pub(super) fn singleton(&self) -> Option<Bf16Value> {
        (self.length == 1).then_some(self.values[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_step_allowance_is_intersection_and_never_crosses_special_classes() {
        let allowed = AllowedF32::from_rounded_range(0x3f80_0000, 0x3f80_0001).unwrap();
        assert_eq!(allowed.len(), 8);
        allowed.check(0x3f7f_fffd).unwrap();
        allowed.check(0x3f80_0004).unwrap();
        assert!(allowed.check(0x3f7f_fffc).is_err()); // In a union, outside the intersection.
        assert!(allowed.check(0x3f80_0005).is_err());
        assert!(AllowedF32::from_rounded_range(0x3f80_0000, 0x3f80_0009).is_err());
        for center in [1, 0x7f7f_ffff, 0x8000_0001, 0xff7f_ffff] {
            let allowed = AllowedF32::around_rounded(center).unwrap();
            allowed.check(center).unwrap();
            for special in [0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000] {
                assert!(allowed.check(special).is_err());
            }
        }
    }

    #[test]
    fn uncertain_classification_fails_before_candidate_is_available() {
        let half_min = f64::from(f32::from_bits(1)) * 0.5;
        assert!(AllowedF32::from_interval(
            Interval::new(half_min.next_down(), half_min.next_up()).unwrap()
        )
        .is_err());
        assert!(AllowedF32::from_rounded_range(0x8000_0000, 0).is_err());
        assert!(AllowedF32::from_rounded_range(0x7f7f_ffff, 0x7f80_0000).is_err());
        assert!(AllowedF32::from_rounded_range(0x8000_0001, 1).is_err());
        AllowedF32::exact(0x7fc0_0000).check(0xff80_0001).unwrap();
        assert!(AllowedF32::exact(0).check(0x8000_0000).is_err());
    }

    #[test]
    fn output_sets_keep_all_activation_branches_and_special_results() {
        let mut activation = AllowedBf16::empty();
        activation.insert(0).unwrap();
        activation.insert(0x8000).unwrap();
        activation.insert(0x3f80).unwrap();
        let times_inf = activation.multiplied_by(0x7f80).unwrap();
        assert_eq!(times_inf.len(), 2);
        times_inf.check(0x7fc1).unwrap();
        times_inf.check(0x7f80).unwrap();
        assert!(times_inf.check(0).is_err());
        let times_one = activation.multiplied_by(0x3f80).unwrap();
        assert_eq!(times_one.len(), 3);
        times_one.check(0).unwrap();
        times_one.check(0x8000).unwrap();
        times_one.check(0x3f80).unwrap();
        assert!(times_one.check(0xbf80).is_err());
    }
}
