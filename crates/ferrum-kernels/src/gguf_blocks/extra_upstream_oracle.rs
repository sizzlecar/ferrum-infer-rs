//! Test-only F64 targets for observed upstream activation packs.
//!
//! Block layouts/codebooks follow the pinned GGML definitions. Copyright
//! (c) 2023-2026 The ggml authors; see LICENSE.ggml (MIT). Only immutable
//! codebooks are shared with the decoder; the byte unpacking below is separate.
//! There is no affine minimum or half-rounded weight coefficient in these
//! three routes. The caller supplies the actual D4 F32 or Q8_1 half scale.

use super::{GgufBlockFormat, IQ3_S_GRID, IQ4_NL_VALUES};
use half::f16;

fn half(bytes: &[u8], offset: usize) -> f64 {
    f64::from(f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32())
}

/// Decode exactly one block, without calling the production/GPU decoders.
/// Finite half values times these small integers are exact in F64. Nonfinite
/// and signed-zero inputs retain IEEE arithmetic rather than claiming rejection.
pub(crate) fn decode_block(format: GgufBlockFormat, block: &[u8]) -> Vec<f64> {
    match format {
        GgufBlockFormat::Q3K => {
            assert_eq!(block.len(), 110);
            let mut scales = [0_i32; 16];
            for i in 0..8 {
                scales[i] = i32::from(block[96 + i] & 15);
                scales[8 + i] = i32::from(block[96 + i] >> 4);
            }
            for plane in 0..4 {
                for lane in 0..4 {
                    scales[4 * plane + lane] +=
                        i32::from((block[104 + lane] >> (2 * plane)) & 3) * 16 - 32;
                }
            }
            let d = half(block, 108);
            let mut output = vec![0.0; 256];
            for chunk in 0..2 {
                for plane in 0..4 {
                    let group32 = 4 * chunk + plane;
                    for lane in 0..32 {
                        let index = group32 * 32 + lane;
                        let low = (block[32 + chunk * 32 + lane] >> (2 * plane)) & 3;
                        let negative = block[lane] & (1 << group32) == 0;
                        let code = i32::from(low) - if negative { 4 } else { 0 };
                        output[index] = (d * f64::from(scales[index / 16])) * f64::from(code);
                    }
                }
            }
            output
        }
        GgufBlockFormat::Iq3S => {
            assert_eq!(block.len(), 110);
            let d = half(block, 0);
            let mut output = vec![0.0; 256];
            for group in 0..8 {
                let scale_nibble = (block[106 + group / 2] >> (4 * (group % 2))) & 15;
                let coefficient = d * f64::from(2 * scale_nibble + 1);
                for quad in 0..8 {
                    let low = usize::from(block[2 + group * 8 + quad]);
                    let high = usize::from((block[66 + group] >> quad) & 1);
                    let levels = IQ3_S_GRID[low + 256 * high].to_le_bytes();
                    let signs = block[74 + group * 4 + quad / 2] >> (4 * (quad % 2));
                    for (lane, level) in levels.into_iter().enumerate() {
                        let sign = if signs & (1 << lane) == 0 { 1.0 } else { -1.0 };
                        output[group * 32 + quad * 4 + lane] =
                            (coefficient * f64::from(level)) * sign;
                    }
                }
            }
            output
        }
        GgufBlockFormat::Iq4Nl => {
            assert_eq!(block.len(), 18);
            let d = half(block, 0);
            let mut output = vec![0.0; 32];
            for (lane, packed) in block[2..].iter().copied().enumerate() {
                output[lane] = d * f64::from(IQ4_NL_VALUES[usize::from(packed & 15)]);
                output[16 + lane] = d * f64::from(IQ4_NL_VALUES[usize::from(packed >> 4)]);
            }
            output
        }
        other => panic!("extra upstream oracle does not cover {other:?}"),
    }
}

/// F64 dot and sum of absolute terms for one actual quantized activation group.
/// The target is decoded weight times `q * activation_scale`, not the original
/// activation. Quantization error and the route's FP32 reduction bound are
/// separate caller responsibilities. NaN/Inf observations are not qualified
/// finite-domain dots and naturally return nonfinite targets/magnitudes.
pub(crate) fn dot_group(
    format: GgufBlockFormat,
    block: &[u8],
    group: usize,
    q: &[i8; 32],
    activation_scale: f32,
) -> (f64, f64) {
    let weights = decode_block(format, block);
    assert!(group < weights.len() / 32, "activation group outside block");
    let mut value = 0.0;
    let mut magnitude = 0.0;
    for (&weight, &code) in weights[group * 32..group * 32 + 32].iter().zip(q) {
        let term = weight * (f64::from(code) * f64::from(activation_scale));
        value += term;
        magnitude += term.abs();
    }
    (value, magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_blocks::fixtures::oracle_blocks;

    const FORMATS: [GgufBlockFormat; 3] = [
        GgufBlockFormat::Q3K,
        GgufBlockFormat::Iq3S,
        GgufBlockFormat::Iq4Nl,
    ];

    fn q3_fixture(scale_seed: usize, d: f16) -> [u8; 110] {
        let mut bytes = [0_u8; 110];
        bytes[108..110].copy_from_slice(&d.to_le_bytes());
        for group16 in 0..16 {
            let scale = ((scale_seed + group16 * 9) % 64) as u8;
            bytes[96 + group16 % 8] |= (scale & 15) << (4 * (group16 / 8));
            bytes[104 + group16 % 4] |= (scale >> 4) << (2 * (group16 / 4));
        }
        for index in 0..256 {
            let code = (index * 5 % 8) as i8 - 4;
            bytes[32 + index / 128 * 32 + index % 32] |=
                ((code as u8) & 3) << (2 * (index / 32 % 4));
            if code >= 0 {
                bytes[index % 32] |= 1 << (index / 32);
            }
        }
        bytes
    }

    #[test]
    fn extra_oracle_matches_shared_decoder_over_complete_codebook_fixtures() {
        // The existing IQ3 fixtures span every one of the 512 grid indices.
        // This cross-check is in tests only; neither oracle API calls decode.
        for format in FORMATS {
            let bytes = oracle_blocks(format);
            for block in bytes.chunks_exact(format.block_bytes()) {
                let mut decoded = vec![0.0; format.block_values()];
                format.decode(block, &mut decoded).unwrap();
                assert_eq!(
                    decode_block(format, block),
                    decoded.into_iter().map(f64::from).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn extra_q3_oracle_exercises_all_signed_scales_codes_and_scale_planes() {
        for seed in 0..64 {
            for d in [0.5, -0.21875] {
                let block = q3_fixture(seed, f16::from_f32(d));
                let decoded = decode_block(GgufBlockFormat::Q3K, &block);
                for (index, actual) in decoded.into_iter().enumerate() {
                    let scale = ((seed + index / 16 * 9) % 64) as i32 - 32;
                    let code = (index * 5 % 8) as i32 - 4;
                    assert_eq!(actual, f64::from(d) * f64::from(scale) * f64::from(code));
                }
            }
        }
    }

    #[test]
    fn extra_iq_oracle_keeps_group_scales_signs_and_both_nibbles() {
        for scale_seed in 0_u8..16 {
            let mut block = [255_u8; 110];
            block[..2].copy_from_slice(&f16::from_f32(-0.5).to_le_bytes());
            block[74..106].fill(0b1010_0101);
            for group in 0..8 {
                let byte = &mut block[106 + group / 2];
                let shift = 4 * (group % 2);
                let nibble = ((usize::from(scale_seed) + group) % 16) as u8;
                *byte = (*byte & !(15 << shift)) | (nibble << shift);
            }
            let output = decode_block(GgufBlockFormat::Iq3S, &block);
            for (index, actual) in output.into_iter().enumerate() {
                // Grid 511 has literal little-endian levels [1, 1, 15, 15].
                let level = [1.0, 1.0, 15.0, 15.0][index % 4];
                let sign = if 0b1010_0101 & (1 << (index % 8)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                let scale = 1 + 2 * ((usize::from(scale_seed) + index / 32) % 16);
                assert_eq!(actual, -0.5 * scale as f64 * level * sign);
            }
        }
        let mut block = [0_u8; 18];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        for i in 0..16 {
            block[2 + i] = i as u8 | ((15 - i) as u8) << 4;
        }
        let levels = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
            69.0, 89.0, 113.0,
        ];
        let decoded = decode_block(GgufBlockFormat::Iq4Nl, &block);
        for i in 0..16 {
            assert_eq!(decoded[i], 0.5 * levels[i]);
            assert_eq!(decoded[16 + i], 0.5 * levels[15 - i]);
        }
    }

    #[test]
    fn extra_oracle_observes_half_boundaries_without_sanitizing_them() {
        for format in FORMATS {
            let source = oracle_blocks(format);
            for bits in [
                0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
                0x7e00, 0x7c01,
            ] {
                let mut block = source[..format.block_bytes()].to_vec();
                let offset = if format == GgufBlockFormat::Q3K {
                    108
                } else {
                    0
                };
                block[offset..offset + 2].copy_from_slice(&u16::to_le_bytes(bits));
                let actual = decode_block(format, &block);
                let mut shared = vec![0.0; format.block_values()];
                format.decode(&block, &mut shared).unwrap();
                for (value, reference) in actual.into_iter().zip(shared) {
                    if reference.is_nan() {
                        assert!(value.is_nan());
                    } else {
                        assert_eq!(value.to_bits(), f64::from(reference).to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn extra_dot_uses_observed_scale_signed_codes_and_absolute_terms() {
        let mut block = [0xf0_u8; 18];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        let mut q = [127_i8; 32];
        q[..16].fill(-128);
        let d = f32::from_bits(1.0_f32.to_bits() + 1);
        let expected = 16.0 * 0.5 * (127.0 * 128.0 + 113.0 * 127.0) * f64::from(d);
        let (value, abs) = dot_group(GgufBlockFormat::Iq4Nl, &block, 0, &q, d);
        assert_eq!(value, expected);
        assert_eq!(abs, expected);
        assert_ne!(
            value,
            dot_group(GgufBlockFormat::Iq4Nl, &block, 0, &q, 1.0).0
        );
        q.fill(127);
        let (value, abs) = dot_group(GgufBlockFormat::Iq4Nl, &block, 0, &q, 1.0);
        assert_eq!(value, 16.0 * 0.5 * 127.0 * (-127.0 + 113.0));
        assert_eq!(abs, 16.0 * 0.5 * 127.0 * (127.0 + 113.0));
        for format in FORMATS {
            let source = oracle_blocks(format);
            let block = &source[..format.block_bytes()];
            for group in 0..format.block_values() / 32 {
                assert_eq!(dot_group(format, block, group, &[0; 32], 0.0), (0.0, 0.0));
                let observed = dot_group(format, block, group, &[0; 32], f32::NAN);
                assert!(observed.0.is_nan() && observed.1.is_nan());
            }
        }
    }

    #[test]
    #[should_panic(expected = "activation group outside block")]
    fn extra_dot_rejects_a_group_outside_the_exact_block() {
        dot_group(GgufBlockFormat::Iq4Nl, &[0; 18], 1, &[0; 32], 1.0);
    }
}
