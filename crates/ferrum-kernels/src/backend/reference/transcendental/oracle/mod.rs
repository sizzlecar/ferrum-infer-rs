//! Independent qualification of the declared Reference CPU profile.
//!
//! The real-function interval, four-step qualification, and composed BF16
//! output set are different objects. Expected sets are prepared without any
//! libm, half, or candidate scalar call. A set membership success does not mean
//! correctly rounded transcendental output, unique bits, or deployed GPU parity.
//! Rounding assumptions are RN-even and gradual underflow, with no FTZ/DAZ.

mod allowed;
mod interval;
mod qualification;
mod rounding;

use allowed::{AllowedBf16, AllowedF32, Fp32Value, Sign};
use rounding::{bf16_from_f32, decode_bf16, divide_bf16_by_positive_f32, one_plus_positive};

#[derive(Clone, Copy, Debug)]
enum Primitive {
    Exp,
    Tanh,
}

fn prepare_primitive(primitive: Primitive, input: u16) -> Result<AllowedF32, String> {
    let raw = u32::from(input) << 16;
    let classified = Fp32Value::from_bits(raw);
    match (primitive, classified) {
        (_, Fp32Value::Nan) => return Ok(AllowedF32::exact(0x7fc0_0000)),
        (Primitive::Exp, Fp32Value::Infinity(Sign::Positive)) => {
            return Ok(AllowedF32::exact(0x7f80_0000))
        }
        (Primitive::Exp, Fp32Value::Infinity(Sign::Negative)) => return Ok(AllowedF32::exact(0)),
        (Primitive::Tanh, Fp32Value::Infinity(Sign::Positive)) => {
            return Ok(AllowedF32::exact(0x3f80_0000))
        }
        (Primitive::Tanh, Fp32Value::Infinity(Sign::Negative)) => {
            return Ok(AllowedF32::exact(0xbf80_0000))
        }
        (Primitive::Tanh, Fp32Value::Zero(_)) => return Ok(AllowedF32::exact(raw)),
        _ => {}
    }
    let input = f64::from(decode_bf16(input));
    match primitive {
        Primitive::Exp => {
            // This is a stronger declared profile subdomain, not merely a
            // mathematical RN32 cell surrounded by four permitted neighbors.
            // For a=2^-25, exp(-a)>1-a and exp(a)<1/(1-a)<1+2a.
            if input.abs() <= 1.0 / 33_554_432.0 {
                return Ok(AllowedF32::exact(0x3f80_0000));
            }
            // e>2 suffices for both deliberately conservative tail cutoffs.
            if input >= 128.0 {
                return Ok(AllowedF32::exact(0x7f80_0000));
            }
            if input <= -256.0 {
                return Ok(AllowedF32::exact(0));
            }
            AllowedF32::from_interval(interval::exp(input)?)
        }
        Primitive::Tanh => {
            // x-x^3/3 <= tanh(x) <= x. For |x|<=2^-12 its deficit
            // is < |x|*2^-25, below the narrowest half-ULP at a BF16 value.
            // This fixes the ideal cell; the candidate still has four steps.
            if input.abs() <= 1.0 / 4096.0 {
                return AllowedF32::around_rounded(raw);
            }
            // 1-tanh(x) < 2*exp(-2*x) < 2^-31 for x>=16, below
            // the lower half-ULP of 1. Odd symmetry handles negative input.
            if input.abs() >= 16.0 {
                return AllowedF32::around_rounded(if input < 0.0 {
                    0xbf80_0000
                } else {
                    0x3f80_0000
                });
            }
            AllowedF32::from_interval(interval::tanh(input)?)
        }
    }
}

fn compose_silu(input: u16, exponential: &AllowedF32) -> Result<AllowedBf16, String> {
    let mut result = AllowedBf16::empty();
    exponential.try_for_each(|value| {
        let denominator = one_plus_positive(value.bits())?;
        let quotient = divide_bf16_by_positive_f32(input, denominator)?;
        result.insert(bf16_from_f32(quotient))
    })?;
    if result.len() == 0 {
        return Err("SiLU propagation produced no allowed outcome".into());
    }
    Ok(result)
}

fn prepare_silu(input: u16) -> Result<AllowedBf16, String> {
    match Fp32Value::from_bits(u32::from(input) << 16) {
        Fp32Value::Nan | Fp32Value::Infinity(Sign::Negative) => {
            return Ok(AllowedBf16::exact(0x7fc0))
        }
        Fp32Value::Infinity(Sign::Positive) => return Ok(AllowedBf16::exact(0x7f80)),
        Fp32Value::Zero(_) => return Ok(AllowedBf16::exact(input)),
        Fp32Value::FiniteNonZero(_) => {}
    }
    let exponential = prepare_primitive(Primitive::Exp, input ^ 0x8000)?;
    compose_silu(input, &exponential)
}

fn prepare_tanh(input: u16) -> Result<AllowedBf16, String> {
    AllowedBf16::from_fp32(&prepare_primitive(Primitive::Tanh, input)?)
}

#[cfg(test)]
mod tests {
    use super::allowed::Bf16Value;
    use super::*;

    fn unique(allowed: &AllowedBf16, bits: u16) {
        assert_eq!(
            allowed.singleton(),
            Some(Bf16Value::from_bits(bits)),
            "{allowed:?}"
        );
        allowed.check(bits).unwrap();
    }

    #[test]
    fn analytic_fixtures_have_unique_bits_and_reject_missing_bf16_boundaries() {
        let activation = prepare_silu(0x3f80).unwrap();
        unique(&activation, 0x3f3b); // 187/256, enclosed using independent e bounds.
        let multiplied = activation.multiplied_by(0x3f81).unwrap();
        unique(&multiplied, 0x3f3c);
        assert!(multiplied.check(0x3f3d).is_err()); // Ideal SiLU product without inner B.
        let tanh = prepare_tanh(0x3f80).unwrap();
        unique(&tanh, 0x3f43); // 195/256
        let multiplied = tanh.multiplied_by(0x3f81).unwrap();
        unique(&multiplied, 0x3f45);
        assert!(multiplied.check(0x3f44).is_err()); // Omitted tanh BF16 materialization.
        unique(&prepare_silu(0xc2b4).unwrap(), 0x8000); // -90: exp(90) overflows.
        unique(&prepare_silu(0x0001).unwrap(), 0x0000); // Ideal real SiLU rounds to 0001.
        unique(&prepare_silu(0x8001).unwrap(), 0x8000);
        unique(&prepare_tanh(0x0001).unwrap(), 0x0001);
        unique(&prepare_tanh(0x8001).unwrap(), 0x8001);
    }

    #[test]
    fn tiny_profile_is_stronger_than_generic_steps_at_the_actual_sensitive_boundary() {
        for input in [0, 0x8000, 0x0001, 0x8001, 0x3300, 0xb300] {
            let exp = prepare_primitive(Primitive::Exp, input).unwrap();
            assert_eq!(exp.len(), 1);
            exp.check(0x3f80_0000).unwrap();
            assert!(exp.check(0x3f80_0001).is_err());
        }
        // The smallest BF16 x is NOT a counterexample to using a generic exp
        // neighborhood: the FP32 quotient absorbs those perturbations.
        let generic_exp = AllowedF32::around_rounded(0x3f80_0000).unwrap();
        unique(&compose_silu(0x0001, &generic_exp).unwrap(), 0x0000);
        // Max BF16 subnormal is sensitive. Exact exp=1 gives a BF16 even tie;
        // generic exp=1+4*2^-23 lowers RN32 quotient by one subnormal quantum.
        unique(&prepare_silu(0x007f).unwrap(), 0x0040);
        let generic = compose_silu(0x007f, &generic_exp).unwrap();
        assert_eq!(generic.len(), 2);
        generic.check(0x003f).unwrap();
        generic.check(0x0040).unwrap();
        assert_eq!(one_plus_positive(0x3f80_0004).unwrap(), 0x4000_0002);
        assert_eq!(
            divide_bf16_by_positive_f32(0x007f, 0x4000_0000).unwrap(),
            0x003f_8000
        );
        assert_eq!(
            divide_bf16_by_positive_f32(0x007f, 0x4000_0002).unwrap(),
            0x003f_7fff
        );
    }

    #[test]
    fn special_inputs_preserve_signed_zero_infinity_and_nan_classification() {
        for (input, silu, tanh) in [
            (0, 0, 0),
            (0x8000, 0x8000, 0x8000),
            (0x7f80, 0x7f80, 0x3f80),
            (0xff80, 0x7fc0, 0xbf80),
            (0x7f81, 0x7fc0, 0x7fc0),
            (0xffff, 0x7fc0, 0x7fc0),
        ] {
            unique(&prepare_silu(input).unwrap(), silu);
            unique(&prepare_tanh(input).unwrap(), tanh);
        }
        prepare_silu(0x7f81).unwrap().check(0xffc1).unwrap();
        assert!(prepare_silu(0x8000).unwrap().check(0).is_err());
    }

    #[test]
    fn classification_errors_and_excess_steps_are_rejected_without_relaxing_bounds() {
        let nonzero_subnormal = prepare_primitive(Primitive::Exp, 0xc2c8).unwrap(); // -100
        assert!(nonzero_subnormal.check(0).is_err());
        let overflow = prepare_primitive(Primitive::Exp, 0x42b2).unwrap(); // +89
        overflow.check(0x7f80_0000).unwrap();
        assert!(overflow.check(0x7f7f_ffff).is_err());
        let underflow = prepare_primitive(Primitive::Exp, 0xc2d0).unwrap(); // -104
        underflow.check(0).unwrap();
        assert!(underflow.check(1).is_err());
        let tanh_small = prepare_primitive(Primitive::Tanh, 0x3980).unwrap(); // 2^-12
        tanh_small.check(0x3980_0004).unwrap();
        assert!(tanh_small.check(0x3980_0005).is_err());
        assert!(tanh_small.check(0x7fc0_0000).is_err());
        assert!(tanh_small.check(0x7f80_0000).is_err());
    }
}
