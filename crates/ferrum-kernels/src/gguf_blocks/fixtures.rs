//! Small native block fixtures shared by CPU/oracle and device conformance tests.
use super::GgufBlockFormat;
use half::f16;

pub(crate) const FORMATS: [GgufBlockFormat; 8] = [
    GgufBlockFormat::Q3K,
    GgufBlockFormat::Q4K,
    GgufBlockFormat::Q5K,
    GgufBlockFormat::Q6K,
    GgufBlockFormat::Q8_0,
    GgufBlockFormat::Iq3S,
    GgufBlockFormat::Iq4Nl,
    GgufBlockFormat::Iq4Xs,
];

pub(crate) fn oracle_blocks(format: GgufBlockFormat) -> Vec<u8> {
    // Eight IQ3_S blocks visit all 512 codebook indices. Other layouts use two
    // distinct blocks to cover row continuation and independent scale signs.
    let blocks = if format == GgufBlockFormat::Iq3S {
        8
    } else {
        2
    };
    let mut bytes = vec![0_u8; blocks * format.block_bytes()];
    for (block_index, block) in bytes.chunks_exact_mut(format.block_bytes()).enumerate() {
        for (index, value) in block.iter_mut().enumerate() {
            *value = (block_index * 37 + index * 29 + (index / 3) * 11) as u8;
        }
        let scale_offset = match format {
            GgufBlockFormat::Q3K => 108,
            GgufBlockFormat::Q6K => 208,
            _ => 0,
        };
        let scale = if block_index % 2 == 0 {
            0.125
        } else {
            -0.21875
        };
        block[scale_offset..scale_offset + 2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
        if matches!(format, GgufBlockFormat::Q4K | GgufBlockFormat::Q5K) {
            block[2..4].copy_from_slice(&f16::from_f32(0.0625).to_le_bytes());
        }
        if format == GgufBlockFormat::Iq3S {
            block[66..74].fill(0);
            for index in 0..64 {
                let code = block_index * 64 + index;
                block[2 + index] = code as u8;
                block[66 + index / 8] |= ((code >> 8) as u8) << (index % 8);
            }
        }
    }
    bytes
}
