//! Scalar encoding oracles. These specify finite round-to-nearest, ties-to-even
//! conversion with saturation. They do not specify a GPU's approximate inverse,
//! input scaling order, FTZ behavior, or a complete activation quantizer.

use ferrum_types::Result;

use super::{decode_e4m3_scale, invalid};

/// Encode a finite F32 value to E2M1, saturating its magnitude at six.
/// The sign of zero is preserved. Nonfinite inputs are rejected explicitly.
pub fn encode_e2m1_rne(value: f32) -> Result<u8> {
    if !value.is_finite() {
        return Err(invalid("E2M1 reference conversion requires a finite input"));
    }
    let sign = if value.is_sign_negative() { 8 } else { 0 };
    let magnitude = value.abs();
    if magnitude >= 6.0 {
        return Ok(sign | 7);
    }
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    Ok(sign
        | nearest(
            magnitude,
            VALUES.into_iter().enumerate().map(|(i, v)| (i as u8, v)),
        ))
}

/// Encode a nonnegative finite scale to E4M3FN, saturating at 448.
/// Negative zero is rejected, matching the stored scale contract. Subnormal
/// E4M3 values are retained; no host or device FTZ policy is implied.
pub fn encode_e4m3_scale_rne(value: f32) -> Result<u8> {
    if !value.is_finite() || value.is_sign_negative() {
        return Err(invalid(
            "E4M3 scale conversion requires a nonnegative finite input",
        ));
    }
    if value >= 448.0 {
        return Ok(126);
    }
    Ok(nearest(
        value,
        (0..=126).map(|code| {
            // The enumerated codes are exactly the nonnegative finite domain.
            (
                code,
                decode_e4m3_scale(code).expect("finite E4M3 scale code"),
            )
        }),
    ))
}

fn nearest(value: f32, candidates: impl Iterator<Item = (u8, f32)>) -> u8 {
    // Inputs have already been saturated. Wider subtraction distinguishes
    // neighbors of each F32 midpoint, including E4M3 subnormal boundaries.
    let mut best = 0;
    let mut distance = f64::INFINITY;
    for (code, represented) in candidates {
        let current = (f64::from(value) - f64::from(represented)).abs();
        if current < distance || (current == distance && code & 1 == 0) {
            best = code;
            distance = current;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvfp4::decode_e2m1;

    #[test]
    fn all_codes_and_signed_zero_round_trip_without_numeric_conversion_loss() {
        for code in 0..16 {
            assert_eq!(encode_e2m1_rne(decode_e2m1(code).unwrap()).unwrap(), code);
        }
        for code in 0..127 {
            assert_eq!(
                encode_e4m3_scale_rne(decode_e4m3_scale(code).unwrap()).unwrap(),
                code
            );
        }
        assert_eq!(encode_e2m1_rne(-0.0).unwrap(), 8);
        assert_eq!(decode_e2m1(8).unwrap().to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn e2m1_midpoints_round_to_even_and_saturate_with_both_signs() {
        // Written from the E2M1 representable set, independent of the encoder.
        for (midpoint, low, even, high) in [
            (0.25f32, 0, 0, 1),
            (0.75, 1, 2, 2),
            (1.25, 2, 2, 3),
            (1.75, 3, 4, 4),
            (2.5, 4, 4, 5),
            (3.5, 5, 6, 6),
            (5.0, 6, 6, 7),
        ] {
            for sign in [0, 8] {
                let polarity = if sign == 0 { 1.0 } else { -1.0 };
                assert_eq!(encode_e2m1_rne(midpoint * polarity).unwrap(), sign | even);
                assert_eq!(
                    encode_e2m1_rne(f32::from_bits(midpoint.to_bits() - 1) * polarity).unwrap(),
                    sign | low
                );
                assert_eq!(
                    encode_e2m1_rne(f32::from_bits(midpoint.to_bits() + 1) * polarity).unwrap(),
                    sign | high
                );
            }
        }
        assert_eq!(encode_e2m1_rne(f32::MAX).unwrap(), 7);
        assert_eq!(encode_e2m1_rne(-f32::MAX).unwrap(), 15);
    }

    #[test]
    fn e4m3_all_midpoints_include_subnormal_and_exponent_boundaries() {
        for low in 0..126 {
            let lo = decode_e4m3_scale(low).unwrap();
            let hi = decode_e4m3_scale(low + 1).unwrap();
            let midpoint = ((f64::from(lo) + f64::from(hi)) / 2.0) as f32;
            assert_eq!(
                encode_e4m3_scale_rne(midpoint).unwrap(),
                if low & 1 == 0 { low } else { low + 1 }
            );
            assert_eq!(
                encode_e4m3_scale_rne(f32::from_bits(midpoint.to_bits() - 1)).unwrap(),
                low
            );
            assert_eq!(
                encode_e4m3_scale_rne(f32::from_bits(midpoint.to_bits() + 1)).unwrap(),
                low + 1
            );
        }
        assert_eq!(encode_e4m3_scale_rne(f32::MAX).unwrap(), 126);
        assert_eq!(encode_e4m3_scale_rne(f32::from_bits(1)).unwrap(), 0);
    }

    #[test]
    fn nonfinite_values_and_negative_scales_are_rejected() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(encode_e2m1_rne(value).is_err());
            assert!(encode_e4m3_scale_rne(value).is_err());
        }
        for value in [-0.0, -f32::MIN_POSITIVE, -1.0] {
            assert!(encode_e4m3_scale_rne(value).is_err());
        }
    }
}
