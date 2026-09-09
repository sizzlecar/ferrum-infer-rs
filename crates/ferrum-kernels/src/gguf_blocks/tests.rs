use super::*;

use super::fixtures::{oracle_blocks, FORMATS};

fn spec(format: GgufBlockFormat) -> BlockQuantizationSpec {
    BlockQuantizationSpec {
        format_id: format.format_id().to_owned().try_into().unwrap(),
        logical_values_per_block: format.block_values() as u32,
        bytes_per_block: format.block_bytes() as u32,
    }
}

#[test]
fn decode_checks_full_abi_and_lengths_before_writing_any_output() {
    for format in FORMATS {
        assert_eq!(GgufBlockFormat::from_spec(&spec(format)).unwrap(), format);
        let mut wrong = spec(format);
        wrong.bytes_per_block += 1;
        assert!(GgufBlockFormat::from_spec(&wrong).is_err());
        wrong = spec(format);
        wrong.logical_values_per_block *= 2;
        assert!(GgufBlockFormat::from_spec(&wrong).is_err());
        let mut output = vec![7.0; format.block_values()];
        for bytes in [format.block_bytes() - 1, format.block_bytes() + 1] {
            assert!(format.decode(&vec![0; bytes], &mut output).is_err());
            assert!(output.iter().all(|&value| value == 7.0));
        }
        assert!(format
            .decode(&vec![0; format.block_bytes()], &mut output[..1])
            .is_err());
        assert!(format.decode(&[], &mut []).is_err());
    }
}

#[test]
fn iq4_nl_decodes_both_nibbles_with_nonuniform_levels() {
    let mut block = [0_u8; 18];
    block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
    for index in 0..16 {
        block[2 + index] = index as u8 | ((15 - index) as u8) << 4;
    }
    let mut output = [0.0; 32];
    GgufBlockFormat::Iq4Nl.decode(&block, &mut output).unwrap();
    let expected = [
        -63.5, -52.0, -41.5, -32.5, -24.5, -17.5, -11.0, -5.0, 0.5, 6.5, 12.5, 19.0, 26.5, 34.5,
        44.5, 56.5,
    ];
    assert_eq!(output[..16], expected);
    assert_eq!(output[16..], expected.into_iter().rev().collect::<Vec<_>>());
}

#[test]
fn iq4_xs_keeps_signed_six_bit_scales_and_subblock_order() {
    let mut block = [0_u8; 136];
    block[..2].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
    block[8..].fill(0x88); // Both indices address the codebook value +1.
    let scales = [-32_i32, -17, -1, 0, 1, 15, 16, 31];
    let mut high = 0_u16;
    for (group, scale) in scales.into_iter().enumerate() {
        let encoded = (scale + 32) as u8;
        block[4 + group / 2] |= (encoded & 15) << (4 * (group % 2));
        high |= u16::from(encoded >> 4) << (2 * group);
    }
    block[2..4].copy_from_slice(&high.to_le_bytes());
    let mut output = [0.0; 256];
    GgufBlockFormat::Iq4Xs.decode(&block, &mut output).unwrap();
    for (group, expected) in output.chunks_exact(32).zip(scales) {
        assert!(group.iter().all(|&value| value == expected as f32 * 0.25));
    }
}

#[test]
fn iq3_s_high_codebook_bit_and_signs_are_independent() {
    let mut block = [0_u8; 110];
    block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
    block[2..66].fill(255);
    block[66..74].fill(255); // Index 511 is [1, 1, 15, 15], little endian.
    block[74..106].fill(0b1010_1010);
    block[106..110].fill(0x10); // Adjacent 32-element groups have scales 1 and 3.
    let mut output = [0.0; 256];
    GgufBlockFormat::Iq3S.decode(&block, &mut output).unwrap();
    for (index, value) in output.iter().enumerate() {
        let base = [0.5, -0.5, 7.5, -7.5][index % 4];
        let scale = if (index / 32) % 2 == 0 { 1.0 } else { 3.0 };
        assert_eq!(*value, base * scale);
    }
}

#[test]
fn native_k_quant_decoding_matches_candle_cpu_on_quantized_weights() {
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    let input = Tensor::from_vec(
        (0..512)
            .map(|index| ((index * 19 % 97) as f32 - 48.0) * 0.3125 + (index % 7) as f32 * 0.015625)
            .collect(),
        (2, 256),
        &Device::Cpu,
    )
    .unwrap();
    for (format, dtype) in [
        (GgufBlockFormat::Q3K, GgmlDType::Q3K),
        (GgufBlockFormat::Q4K, GgmlDType::Q4K),
        (GgufBlockFormat::Q5K, GgmlDType::Q5K),
        (GgufBlockFormat::Q6K, GgmlDType::Q6K),
        (GgufBlockFormat::Q8_0, GgmlDType::Q8_0),
    ] {
        let tensor = QTensor::quantize(&input, dtype).unwrap();
        let expected = tensor
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut output = vec![0.0; 512];
        format.decode(&tensor.data().unwrap(), &mut output).unwrap();
        assert_eq!(output, expected, "{format:?}");
    }
}

#[test]
#[ignore = "requires FERRUM_GGML_REFERENCE_LIBRARY pointing to a trusted libggml-base shared library"]
fn native_blocks_match_independent_ggml_dequantizers() {
    let path =
        std::env::var_os("FERRUM_GGML_REFERENCE_LIBRARY").expect("FERRUM_GGML_REFERENCE_LIBRARY");
    // SAFETY: This explicit developer test loads the trusted, locally built
    // oracle supplied by its caller. No downloaded library is executed.
    let library = unsafe { libloading::Library::new(path) }.unwrap();
    type Dequantize = unsafe extern "C" fn(*const std::ffi::c_void, *mut f32, i64);
    for format in FORMATS {
        let symbol = match format {
            GgufBlockFormat::Q3K => "dequantize_row_q3_K",
            GgufBlockFormat::Q4K => "dequantize_row_q4_K",
            GgufBlockFormat::Q5K => "dequantize_row_q5_K",
            GgufBlockFormat::Q6K => "dequantize_row_q6_K",
            GgufBlockFormat::Q8_0 => "dequantize_row_q8_0",
            GgufBlockFormat::Iq3S => "dequantize_row_iq3_s",
            GgufBlockFormat::Iq4Nl => "dequantize_row_iq4_nl",
            GgufBlockFormat::Iq4Xs => "dequantize_row_iq4_xs",
        };
        // SAFETY: These public GGML C functions share this exact signature.
        let decode: libloading::Symbol<'_, Dequantize> =
            unsafe { library.get(symbol.as_bytes()) }.unwrap();
        let bytes = oracle_blocks(format);
        let values = bytes.len() / format.block_bytes() * format.block_values();
        let mut expected = vec![0.0_f32; values];
        let mut actual = vec![0.0_f32; values];
        // Vec<u64> provides sufficient alignment for the C block structs.
        let mut aligned = vec![0_u64; bytes.len().div_ceil(8)];
        // SAFETY: The initialized allocation has at least bytes.len() bytes;
        // input and output are disjoint, and k is a complete number of blocks.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                aligned.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
            decode(
                aligned.as_ptr().cast(),
                expected.as_mut_ptr(),
                values as i64,
            );
        }
        format.decode(&bytes, &mut actual).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{format:?} element {index}: {actual} vs {expected}"
            );
        }
    }
}
