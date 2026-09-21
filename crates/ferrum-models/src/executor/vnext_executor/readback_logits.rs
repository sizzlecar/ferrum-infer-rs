//! Decode the immutable host readback without changing its floating-point policy.

pub(super) fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    // Runtime buffers normally satisfy u16 alignment, but HostTransferLayout
    // does not promise it for an arbitrary host slice. The checked cast also
    // rejects partial elements; preserve the scalar little-endian fallback.
    #[cfg(target_endian = "little")]
    if let Ok(bits) = bytemuck::try_cast_slice::<u8, u16>(bytes) {
        use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

        // half dispatches SIMD support once per slice instead of once for
        // every logit. No extra half-precision allocation is needed.
        return bits.reinterpret_cast::<half::f16>().to_f32_vec();
    }
    bytes
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::decode_f16;

    #[test]
    fn f16_readback_preserves_every_encoding_and_unaligned_input() {
        let bits = (0..=u16::MAX).collect::<Vec<_>>();
        let bytes = bits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        // A u16 allocation makes the first slice aligned and the second one
        // deliberately unaligned on every supported host, rather than relying
        // on the allocator's incidental alignment for a Vec<u8>.
        let mut storage = vec![0_u16; bits.len() + 1];
        let storage_bytes = bytemuck::cast_slice_mut::<u16, u8>(&mut storage);
        storage_bytes[..bytes.len()].copy_from_slice(&bytes);
        let aligned = decode_f16(&storage_bytes[..bytes.len()]);
        storage_bytes[1..bytes.len() + 1].copy_from_slice(&bytes);
        let unaligned = decode_f16(&storage_bytes[1..bytes.len() + 1]);
        assert_eq!(aligned.len(), bits.len());
        assert_eq!(unaligned.len(), bits.len());
        for ((&bits, aligned), unaligned) in bits.iter().zip(aligned).zip(unaligned) {
            let expected = half::f16::from_bits(bits).to_f32();
            for actual in [aligned, unaligned] {
                if expected.is_nan() {
                    assert!(actual.is_nan(), "encoding {bits:#06x}");
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "encoding {bits:#06x}");
                }
            }
        }
    }

    #[test]
    fn f16_readback_preserves_short_and_partial_slice_behavior() {
        assert!(decode_f16(&[]).is_empty());
        assert!(decode_f16(&[0xff]).is_empty());
        assert_eq!(decode_f16(&[0x00, 0x3c, 0xff]), [1.0]);
        let words = [0x0001_u16, 0x8000, 0x3c00, 0x7bff, 0xfc00];
        for length in 1..=words.len() {
            let bytes = words[..length]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let actual = decode_f16(&bytes);
            assert_eq!(actual.len(), length);
            for (&word, value) in words[..length].iter().zip(actual) {
                assert_eq!(
                    value.to_bits(),
                    half::f16::from_bits(word).to_f32().to_bits()
                );
            }
        }
    }
}
