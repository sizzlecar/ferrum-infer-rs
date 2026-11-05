use super::*;
use half::{bf16, f16};

fn oracle(input: &[u8], output: &mut [u8]) -> Result<(), &'static str> {
    for (source, destination) in input.chunks_exact(2).zip(output.chunks_exact_mut(2)) {
        let source = bf16::from_le_bytes(source.try_into().unwrap()).to_f32();
        let target = f16::from_f32(source);
        destination.copy_from_slice(&target.to_le_bytes());
        if source.is_finite() && !target.is_finite() {
            return Err("finite dense value overflows the target type");
        }
    }
    Ok(())
}

#[test]
fn bf16_to_f16_simd_exhaustive_bit_patterns_preserve_scalar_bits_and_errors() {
    for word in 0..=u16::MAX {
        // Repeat a lane to exercise a real SIMD block for every normal value,
        // while every exceptional block passes through the original converter.
        let source = word.to_le_bytes().repeat(8);
        let mut actual = [0x55; BLOCK_BYTES];
        let mut expected = actual;
        assert_eq!(
            convert(&source, &mut actual),
            oracle(&source, &mut expected),
            "bits {word:#06x}"
        );
        assert_eq!(actual, expected, "bits {word:#06x}");
    }
}

#[test]
fn bf16_to_f16_simd_mixed_blocks_tails_and_unaligned_bytes_stay_exact() {
    let cases = [
        0, 0x8000, 0x3880, 0xb880, 0x3f80, 0xbf80, 0x477f, 0xc77f, 0x0001, 0x8001, 0x387f, 0x7f80,
        0xff80, 0x7f81, 0xffc1, 0x4780,
    ];
    for length in 0..=65 {
        for source_offset in 0..=3 {
            for destination_offset in 0..=3 {
                for shift in 0..cases.len() {
                    let mut source = vec![0xab; source_offset];
                    for i in 0..length {
                        source.extend((cases[(i + shift) % cases.len()] as u16).to_le_bytes());
                    }
                    let mut actual = vec![0xcd; destination_offset + length * 2 + 3];
                    let mut expected = actual.clone();
                    let range = destination_offset..destination_offset + length * 2;
                    assert_eq!(
                        convert(&source[source_offset..], &mut actual[range.clone()]),
                        oracle(&source[source_offset..], &mut expected[range])
                    );
                    assert_eq!(actual, expected, "length={length} source={source_offset} destination={destination_offset} shift={shift}");
                }
            }
        }
    }
    assert!(convert(&[0], &mut [0]).is_err());
    assert!(convert(&[0, 0], &mut []).is_err());
}

#[cfg(all(
    target_endian = "little",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
#[test]
fn bf16_to_f16_simd_block_checks_all_lanes_before_writing() {
    for position in 0..8 {
        let mut input = vec![0u8; BLOCK_BYTES];
        for bytes in input.chunks_exact_mut(2) {
            bytes.copy_from_slice(&0x3f80u16.to_le_bytes());
        }
        input[position * 2..position * 2 + 2].copy_from_slice(&0x387fu16.to_le_bytes());
        let mut output = [0x12; BLOCK_BYTES];
        // SAFETY: both arrays cover the complete block; unaligned is permitted.
        assert!(!unsafe { normal_block(input.as_ptr(), output.as_mut_ptr()) });
        assert_eq!(output, [0x12; BLOCK_BYTES]);
    }
}
