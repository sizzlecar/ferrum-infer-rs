use super::*;

fn unpack(plan: &RnF16FragmentPlanV1, bytes: &[u8], column: usize, kk: usize) -> f16 {
    let format = format(plan);
    let k = plan.k() as usize;
    let step = 128 + (format.ggml_type_id() as usize - 12) * 32;
    let base = (column / 16 * (k / 32) + kk / 32) * (128 + 2 * step);
    let cell = &bytes[base..];
    let fragment = kk % 32 / 16;
    let local = column % 16;
    let lane = local % 8 * 4 + kk % 8 / 2;
    let slot = usize::from(kk % 16 >= 8) * 4 + usize::from(local >= 8) * 2 + kk % 2;
    let codes = &cell[128 + fragment * step..];
    let word = u32::from_le_bytes(codes[lane * 4..lane * 4 + 4].try_into().unwrap());
    let mut q = ((word >> (slot * 4)) & 15) as u8;
    if format != GgufBlockFormat::Q4K {
        q |= ((codes[128 + lane] >> slot) & 1) << 4;
    }
    if format == GgufBlockFormat::Q6K {
        q |= ((codes[160 + lane] >> slot) & 1) << 5;
    }
    let scale_offset = if format == GgufBlockFormat::Q6K {
        fragment * 64
    } else {
        0
    };
    let a = f32::from_le_bytes(
        cell[scale_offset + local * 4..scale_offset + local * 4 + 4]
            .try_into()
            .unwrap(),
    );
    let z = if format == GgufBlockFormat::Q6K {
        0.0
    } else {
        f32::from_le_bytes(cell[64 + local * 4..68 + local * 4].try_into().unwrap())
    };
    rounded(format, a, z, q)
}
fn source(format: GgufBlockFormat, k: usize, n: usize, seed: u64) -> Vec<u8> {
    let mut bytes = vec![0; n * (k / 256) * format.block_bytes()];
    // Distinct hashed bytes throughout each physical matrix, not repeated
    // column templates. These are synthetic weights at real model dimensions.
    let mut state = seed | 1;
    for block in bytes.chunks_exact_mut(format.block_bytes()) {
        for byte in block.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 27) as u8;
        }
        if format == GgufBlockFormat::Q6K {
            let d = f16::from_f32((0.75 + f32::from(block[0]) / 512.0) / 65536.0);
            block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
        } else {
            let d = f16::from_f32((0.75 + f32::from(block[0]) / 512.0) / 16384.0);
            let m = f16::from_f32((0.75 + f32::from(block[1]) / 512.0) / 2048.0);
            block[..2].copy_from_slice(&d.to_bits().to_le_bytes());
            block[2..4].copy_from_slice(&m.to_bits().to_le_bytes());
        }
    }
    bytes
}

#[test]
fn packet_bits_equal_original_rn_and_cross_source_tail_is_not_padded_separately() {
    for (source_format, format) in [
        (RnF16FragmentSourceFormatV1::Q4K, GgufBlockFormat::Q4K),
        (RnF16FragmentSourceFormatV1::Q5K, GgufBlockFormat::Q5K),
        (RnF16FragmentSourceFormatV1::Q6K, GgufBlockFormat::Q6K),
    ] {
        for n in [1, 15, 16, 19] {
            let plan = RnF16FragmentPlanV1::new(source_format, n, 768).unwrap();
            let raw = source(format, 768, n as usize, 71);
            let dense =
                crate::gguf_f16_projection_materializer::convert_rn_f16_diagnostic(format, &raw)
                    .unwrap();
            let packed = pack_checked(&plan, &[&raw], Some(&dense)).unwrap();
            assert_eq!(packed.len() as u64, plan.packed_bytes());
            for (i, expected) in dense.chunks_exact(2).enumerate() {
                assert_eq!(
                    unpack(&plan, &packed, i / 768, i % 768).to_bits(),
                    u16::from_le_bytes([expected[0], expected[1]]),
                    "{format:?} N{n} index{i}"
                );
            }
            if n > 1 {
                let split = (n as usize / 2) * plan.source_row_bytes() as usize;
                assert_eq!(
                    pack_rn_f16_fragments(&plan, &[&raw[..split], &raw[split..]]).unwrap(),
                    packed
                );
            }
            // All unused metadata and quant bits equal zero source rows, even
            // when the logical tail shares a tile with the next source slice.
            let padded_n = n.div_ceil(16) * 16;
            let padded_plan = RnF16FragmentPlanV1::new(source_format, padded_n, 768).unwrap();
            let mut padded = raw.clone();
            padded.resize(padded_plan.source_bytes() as usize, 0);
            assert_eq!(
                pack_rn_f16_fragments(&padded_plan, &[&padded]).unwrap(),
                packed
            );
            let mut wrong_dense = dense;
            wrong_dense[0] ^= 1;
            assert!(pack_checked(&plan, &[&raw], Some(&wrong_dense)).is_err());
        }
    }
}

#[test]
fn packet_source_spans_and_nonfinite_boundaries_fail_closed() {
    for (source_format, format) in [
        (RnF16FragmentSourceFormatV1::Q4K, GgufBlockFormat::Q4K),
        (RnF16FragmentSourceFormatV1::Q5K, GgufBlockFormat::Q5K),
        (RnF16FragmentSourceFormatV1::Q6K, GgufBlockFormat::Q6K),
    ] {
        let plan = RnF16FragmentPlanV1::new(source_format, 2, 256).unwrap();
        let raw = source(format, 256, 2, 93);
        assert!(pack_rn_f16_fragments(&plan, &[]).is_err());
        assert!(pack_rn_f16_fragments(&plan, &[&[]]).is_err());
        assert!(pack_rn_f16_fragments(&plan, &[&raw[..raw.len() - 1]]).is_err());
        assert!(pack_rn_f16_fragments(&plan, &[&raw[..1], &raw[1..]]).is_err());
        for bits in [
            0_u16, 0x8000, 1, 0x03ff, 0x0400, 0x3555, 0x7bff, 0x7c00, 0x7e00,
        ] {
            let mut edge = source(format, 256, 1, 93);
            let offset = if format == GgufBlockFormat::Q6K {
                208
            } else {
                0
            };
            edge[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            let edge_plan = RnF16FragmentPlanV1::new(source_format, 1, 256).unwrap();
            let dense =
                crate::gguf_f16_projection_materializer::convert_rn_f16_diagnostic(format, &edge);
            let packet = pack_rn_f16_fragments(&edge_plan, &[&edge]);
            assert_eq!(dense.is_ok(), packet.is_ok(), "{format:?} {bits:x}");
            if let Ok(dense) = dense {
                assert!(pack_checked(&edge_plan, &[&edge], Some(&dense)).is_ok());
            }
        }
    }
}
