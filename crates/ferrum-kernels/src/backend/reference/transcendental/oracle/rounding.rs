//! Independent local arithmetic. No half conversion or candidate division.

pub(super) fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Integer BF16 RN-even, including finite overflow. NaN payload is deliberately
/// canonicalized because the profile only promises its classification.
pub(super) fn bf16_from_f32(bits: u32) -> u16 {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return if bits & 0x007f_ffff == 0 {
            (bits >> 16) as u16
        } else {
            0x7fc0
        };
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The SiLU denominator. For 2^-24 < e < 2^25, 1+e has at most 48 significant
/// binary digits and is exact in binary64. Outside that interval its RN32
/// result follows analytically from the half-ULP and the even significand of 1.
pub(super) fn one_plus_positive(exp_bits: u32) -> Result<u32, String> {
    if exp_bits == 0x7f80_0000 {
        return Ok(exp_bits);
    }
    if exp_bits & 0x8000_0000 != 0 || exp_bits > 0x7f7f_ffff {
        return Err(format!(
            "invalid exp output for denominator: {exp_bits:08x}"
        ));
    }
    if exp_bits <= 0x3380_0000 {
        return Ok(1.0_f32.to_bits());
    }
    if exp_bits >= 0x4c00_0000 {
        return Ok(exp_bits);
    }
    Ok(((1.0_f64 + f64::from(f32::from_bits(exp_bits))) as f32).to_bits())
}

fn decode_finite_nonzero(bits: u32) -> (u32, i32) {
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0 {
        (fraction, -149)
    } else {
        (fraction | 0x0080_0000, exponent - 127 - 23)
    }
}

/// Exact RN32 division for the actual composition domain: finite BF16 x and
/// positive FP32 d >= 1 (or +Inf). Integer quotient/remainder decides ties.
/// After binade classification, required shifts lie in [-24,47], so all
/// operands fit comfortably in u128. No F64 double-rounding premise is used.
pub(super) fn divide_bf16_by_positive_f32(x_bits: u16, d_bits: u32) -> Result<u32, String> {
    let x_raw = u32::from(x_bits) << 16;
    let sign = x_raw & 0x8000_0000;
    let magnitude = x_raw & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return Err("local division requires a finite BF16 numerator".into());
    }
    if !(0x3f80_0000..=0x7f80_0000).contains(&d_bits) {
        return Err(format!(
            "local divisor must be positive and >=1: {d_bits:08x}"
        ));
    }
    if magnitude == 0 || d_bits == 0x7f80_0000 {
        return Ok(sign);
    }
    let (numerator, numerator_exponent) = decode_finite_nonzero(magnitude);
    let (denominator, denominator_exponent) = decode_finite_nonzero(d_bits);
    let length_difference = denominator.leading_zeros() as i32 - numerator.leading_zeros() as i32;
    let below_binade = if length_difference >= 0 {
        u128::from(numerator) < (u128::from(denominator) << length_difference)
    } else {
        (u128::from(numerator) << -length_difference) < u128::from(denominator)
    };
    let ratio_exponent = length_difference - i32::from(below_binade);
    let mut exponent = numerator_exponent - denominator_exponent + ratio_exponent;
    if exponent < -150 {
        return Ok(sign);
    }
    if exponent > 127 {
        return Err("positive divisor >=1 cannot overflow finite BF16 numerator".into());
    }
    let quantum_exponent = (exponent - 23).max(-149);
    let shift = numerator_exponent - denominator_exponent - quantum_exponent;
    if !(-24..=47).contains(&shift) {
        return Err(format!(
            "division proof failed to bound integer shift: {shift}"
        ));
    }
    let mut integer_numerator = u128::from(numerator);
    let mut integer_denominator = u128::from(denominator);
    if shift >= 0 {
        integer_numerator <<= shift;
    } else {
        integer_denominator <<= -shift;
    }
    let quotient = integer_numerator / integer_denominator;
    let remainder = integer_numerator % integer_denominator;
    let twice_remainder = 2 * remainder;
    let round_up = twice_remainder > integer_denominator
        || (twice_remainder == integer_denominator && quotient & 1 != 0);
    let mut rounded = u32::try_from(quotient + u128::from(round_up))
        .map_err(|_| "division rounded significand exceeds u32".to_string())?;
    if rounded == 0 {
        return Ok(sign);
    }
    if quantum_exponent == -149 && rounded <= 0x0080_0000 {
        // This also encodes a carry from max subnormal into min normal.
        return Ok(sign | rounded);
    }
    if rounded == 0x0100_0000 {
        rounded >>= 1;
        exponent += 1;
    }
    if !(0x0080_0000..0x0100_0000).contains(&rounded) || !(-126..=127).contains(&exponent) {
        return Err("division normalization escaped the proven finite domain".into());
    }
    Ok(sign | (((exponent + 127) as u32) << 23) | (rounded & 0x007f_ffff))
}

/// Each finite BF16 product has at most 16 significant bits and exponent in
/// [-266,256], hence is exact and normal in F64. The subsequent FP32 then BF16
/// rounds are intentional. IEEE special cases retain zero/Inf signs; NaNs are
/// compared by classification by the caller.
pub(super) fn multiply_bf16(lhs: u16, rhs: u16) -> u16 {
    let product = f64::from(decode_bf16(lhs)) * f64::from(decode_bf16(rhs));
    bf16_from_f32((product as f32).to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_division_covers_normal_subnormal_and_signed_ties() {
        for (x, divisor, expected) in [
            (0x3f80, 0x4000_0000, 0x3f00_0000), // 1/2
            (0x4040, 0x4000_0000, 0x3fc0_0000), // 3/2
            (0x3f80, 0x4040_0000, 0x3eaa_aaab), // 1/3
            (0xbf80, 0x4040_0000, 0xbeaa_aaab),
            (0x0001, 0x4000_0000, 0x0000_8000), // 2^-134
            (0x0001, 0x4780_0000, 0x0000_0001), // /2^16 => min subnormal
            (0x0001, 0x4800_0000, 0x0000_0000), // /2^17 => half-min tie
            (0x8001, 0x4800_0000, 0x8000_0000),
            (0x0001, 0x47ff_ffff, 0x0000_0001), // just above half-min
            (0x0001, 0x4800_0001, 0x0000_0000), // just below half-min
            (0x0003, 0x4800_0000, 0x0000_0002), // odd significand tie
            (0x0005, 0x4800_0000, 0x0000_0002), // even significand tie
            (0x7f7f, 0x3f80_0000, 0x7f7f_0000),
            (0x0001, 0x7f7f_ffff, 0x0000_0000),
            (0x8000, 0x7f80_0000, 0x8000_0000),
            (0x3f80, 0x7f7f_ffff, 0x0020_0000),
        ] {
            assert_eq!(
                divide_bf16_by_positive_f32(x, divisor).unwrap(),
                expected,
                "x={x:04x}, divisor={divisor:08x}"
            );
        }
        for d in [0, 0x3f00_0000, 0xbf80_0000, 0x7fc0_0000] {
            assert!(divide_bf16_by_positive_f32(0x3f80, d).is_err());
        }
        assert!(divide_bf16_by_positive_f32(0x7f80, 0x3f80_0000).is_err());
    }

    #[test]
    fn adding_one_resolves_even_midpoints_without_candidate_f32_addition() {
        for (e, expected) in [
            (0, 0x3f80_0000),
            (0x3380_0000, 0x3f80_0000),
            (0x3380_0001, 0x3f80_0001),
            (0x3f80_0000, 0x4000_0000),
            (0x4b80_0000, 0x4b80_0000),
            (0x4b80_0001, 0x4b80_0002),
            (0x4c00_0000, 0x4c00_0000),
            (0x7f7f_ffff, 0x7f7f_ffff),
            (0x7f80_0000, 0x7f80_0000),
        ] {
            assert_eq!(one_plus_positive(e).unwrap(), expected, "e={e:08x}");
        }
        assert!(one_plus_positive(0x8000_0000).is_err());
        assert!(one_plus_positive(0x7fc0_0000).is_err());
    }

    #[test]
    fn independent_bf16_rounding_retains_intermediate_overflow_and_underflow() {
        assert_eq!(bf16_from_f32(0x0000_8000), 0x0000);
        assert_eq!(bf16_from_f32(0x8000_8000), 0x8000);
        assert_eq!(bf16_from_f32(0x0001_8000), 0x0002);
        assert_eq!(multiply_bf16(0x4380, 0x4380), 0x4780); // 256² is BF16 finite
        assert_eq!(multiply_bf16(0x3ffe, 0x7f01), 0x7f80);
        assert_eq!(multiply_bf16(0x8000, 0x3f80), 0x8000);
        assert_eq!(multiply_bf16(0, 0x7f80), 0x7fc0);
    }
}
