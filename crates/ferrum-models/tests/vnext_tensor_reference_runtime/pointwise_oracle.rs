//! Independent local arithmetic for the BF16 pointwise Reference profile.
//!
//! Every operand of an add/multiply is a decoded BF16 number (including the
//! constant one). Finite products have at most 16 significant bits and binary
//! exponents between -266 and 256, so F64 represents them exactly. For addition,
//! if the operands' leading exponents differ by at most 31, their exact sum has
//! at most 40 significant bits and fits F64. Otherwise the smaller operand is
//! less than 2^-30 of the larger. The larger is itself an FP32 grid point with
//! at most eight significant bits, NOT an FP32 midpoint. The smaller operand
//! cannot reach even the narrower half-ULP below a power of two (relative
//! 2^-25); F64 rounding cannot bridge that gap either. BF16 subnormal inputs
//! are multiples of 2^-133 and their sums are also exact in F64.
//!
//! Consequently F64 add/multiply followed by the Rust RN-even F32 conversion
//! implements the required local FP32 result without a double-rounding change.
//! Each BF16 boundary then uses integer RNE, independently of half::bf16.
//! This is NOT a single F64 expression rounded only at the final output.
//! Gradual underflow and ordinary IEEE operations are assumed. NaN payloads
//! are deliberately not an oracle; infinity signs and signed zeros are.

use ferrum_runtime::RowBroadcast;

#[derive(Debug)]
pub(super) struct Prepared {
    bits: Vec<u16>,
}

impl Prepared {
    pub(super) fn bits(&self) -> &[u16] {
        &self.bits
    }

    pub(super) fn check(&self, actual: &[u16]) -> Result<(), String> {
        if actual.len() != self.bits.len() {
            return Err("pointwise output extent differs from its oracle".into());
        }
        for (index, (&expected, &actual)) in self.bits.iter().zip(actual).enumerate() {
            let matches = if is_nan(expected) {
                is_nan(actual)
            } else {
                actual == expected
            };
            if !matches {
                return Err(format!(
                    "pointwise element {index}: expected {expected:#06x}, actual {actual:#06x}"
                ));
            }
        }
        Ok(())
    }
}

fn is_nan(bits: u16) -> bool {
    bits & 0x7f80 == 0x7f80 && bits & 0x007f != 0
}

fn decode(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

fn round_local(value: f64) -> u16 {
    let raw = (value as f32).to_bits();
    if raw & 0x7f80_0000 == 0x7f80_0000 {
        return if raw & 0x007f_ffff != 0 {
            0x7fc0
        } else {
            (raw >> 16) as u16
        };
    }
    // The low 16 bits are discarded. Bias 0x7fff plus the retained LSB
    // implements nearest-even, including gradual underflow and overflow.
    ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16) as u16
}

fn geometry(
    base: &[u16],
    parameter: &[u16],
    rows: usize,
    width: usize,
    mode: RowBroadcast,
) -> Result<(), String> {
    let elements = rows
        .checked_mul(width)
        .filter(|&n| rows != 0 && width != 0 && n <= isize::MAX as usize / 2)
        .ok_or("invalid pointwise oracle geometry")?;
    let parameters = match mode {
        RowBroadcast::Elementwise => elements,
        RowBroadcast::Row => width,
    };
    if base.len() != elements || parameter.len() != parameters {
        return Err("pointwise oracle requires exact base and broadcast extents".into());
    }
    Ok(())
}

fn parameter_index(index: usize, width: usize, mode: RowBroadcast) -> usize {
    match mode {
        RowBroadcast::Elementwise => index,
        RowBroadcast::Row => index % width,
    }
}

pub(super) fn modulation(
    base: &[u16],
    scale: &[u16],
    rows: usize,
    width: usize,
    mode: RowBroadcast,
) -> Result<Prepared, String> {
    geometry(base, scale, rows, width, mode)?;
    let bits = base
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            let selected = scale[parameter_index(index, width, mode)];
            let factor = round_local(1.0 + decode(selected));
            round_local(decode(x) * decode(factor))
        })
        .collect();
    Ok(Prepared { bits })
}

pub(super) fn residual(
    base: &[u16],
    gate: &[u16],
    update: &[u16],
    rows: usize,
    width: usize,
    mode: RowBroadcast,
) -> Result<Prepared, String> {
    geometry(base, gate, rows, width, mode)?;
    if update.len() != base.len() {
        return Err("pointwise update extent differs".into());
    }
    let bits = base
        .iter()
        .zip(update)
        .enumerate()
        .map(|(index, (&x, &u))| {
            let g = gate[parameter_index(index, width, mode)];
            let product = round_local(decode(g) * decode(u));
            round_local(decode(x) + decode(product))
        })
        .collect();
    Ok(Prepared { bits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_computed_boundaries_reject_single_rounding_and_wrong_zero_sign() {
        let modulation = modulation(&[0x3fc0], &[0x3b80], 1, 1, RowBroadcast::Row).unwrap();
        modulation.check(&[0x3fc0]).unwrap();
        assert!(modulation.check(&[0x3fc1]).is_err());
        let residual = residual(&[0xbf80], &[0x3f7e], &[0x3f81], 1, 1, RowBroadcast::Row).unwrap();
        residual.check(&[0]).unwrap();
        assert!(residual.check(&[0xb880]).is_err());
        assert!(residual.check(&[0x8000]).is_err());
    }

    #[test]
    fn normal_subnormal_overflow_and_nonfinite_classes_have_independent_expectations() {
        // 65504 is F16's maximum, not BF16's: 256*256 is finite BF16.
        residual(&[0], &[0x4380], &[0x4380], 1, 1, RowBroadcast::Row)
            .unwrap()
            .check(&[0x4780])
            .unwrap();
        // The halfway product 2^-134 becomes even BF16 zero before addition.
        residual(&[1], &[1], &[0x3f00], 1, 1, RowBroadcast::Row)
            .unwrap()
            .check(&[1])
            .unwrap();
        // FP32 multiplication is finite, but its BF16 conversion overflows.
        residual(&[0xff7f], &[0x3f81], &[0x7f7e], 1, 1, RowBroadcast::Row)
            .unwrap()
            .check(&[0x7f80])
            .unwrap();
        // Overflow already occurs in FP32 here; a single FMA would be finite.
        residual(&[0xff7f], &[0x4000], &[0x7f7f], 1, 1, RowBroadcast::Row)
            .unwrap()
            .check(&[0x7f80])
            .unwrap();
        let invalid = modulation(&[0], &[0x7f80], 1, 1, RowBroadcast::Row).unwrap();
        invalid.check(&[0xff81]).unwrap();
        assert!(invalid.check(&[0x7f80]).is_err());
        residual(
            &[0x8000, 0],
            &[0x8000, 0x8000],
            &[0x3f80; 2],
            1,
            2,
            RowBroadcast::Elementwise,
        )
        .unwrap()
        .check(&[0x8000, 0])
        .unwrap();
        modulation(&[0x8000, 0], &[0xc000; 2], 1, 2, RowBroadcast::Elementwise)
            .unwrap()
            .check(&[0, 0x8000])
            .unwrap();
    }

    #[test]
    fn distant_bf16_exponents_do_not_create_a_false_fp32_midpoint() {
        // The exact F64 sums may lose the tiny operand; the correct FP32
        // result is nevertheless the BF16 grid point one in both directions.
        assert_eq!(round_local(decode(0x3f80) + decode(1)), 0x3f80);
        assert_eq!(round_local(decode(0x3f80) + decode(0x8001)), 0x3f80);
        assert_eq!(round_local(decode(0x7f7f) + decode(0x3f80)), 0x7f7f);
        // Genuine BF16 midpoint ties cover both even/odd retained LSBs.
        assert_eq!(round_local(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(round_local(1.0 + 3.0 / 256.0), 0x3f82);
    }

    #[test]
    fn explicit_broadcast_and_exact_extents_are_required() {
        assert!(modulation(&[0; 4], &[0; 2], 2, 2, RowBroadcast::Elementwise).is_err());
        assert!(modulation(&[0; 4], &[0; 4], 2, 2, RowBroadcast::Row).is_err());
        assert!(modulation(&[], &[], 0, 1, RowBroadcast::Row).is_err());
        assert!(modulation(&[], &[], usize::MAX, 2, RowBroadcast::Row).is_err());
        assert!(residual(&[0], &[0], &[], 1, 1, RowBroadcast::Row).is_err());
        let row = modulation(&[0x3f80; 4], &[0, 0x3f80], 2, 2, RowBroadcast::Row).unwrap();
        row.check(&[0x3f80, 0x4000, 0x3f80, 0x4000]).unwrap();
        assert!(row.check(&[0x3f80]).is_err());
    }
}
