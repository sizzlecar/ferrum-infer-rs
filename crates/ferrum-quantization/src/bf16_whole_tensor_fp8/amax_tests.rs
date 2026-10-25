use super::*;

// The pre-optimization F32 implementation remains the independent oracle.
// Do not derive its maximum or finite classification from the integer scan.
fn scalar_f32_amax<'a>(sources: impl IntoIterator<Item = &'a [u8]>) -> Result<f32, ()> {
    let mut maximum = 0_f32;
    for source in sources {
        for bytes in source.chunks_exact(2) {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
            if !value.is_finite() {
                return Err(());
            }
            maximum = maximum.max(value.abs());
        }
    }
    Ok(maximum)
}

#[test]
fn integer_amax_matches_scalar_for_every_bf16_encoding() {
    for bits in 0..=u16::MAX {
        let bytes = bits.to_le_bytes();
        let expected = scalar_f32_amax([bytes.as_slice()]);
        let actual = conversion::finite_bf16_amax([bytes.as_slice()]);
        match expected {
            Ok(expected) => assert_eq!(
                actual.unwrap().to_bits(),
                expected.to_bits(),
                "BF16 encoding {bits:04x}"
            ),
            Err(()) => assert!(actual.is_err(), "nonfinite BF16 encoding {bits:04x}"),
        }
    }
    for zeros in [[0_u16, 0x8000], [0x8000, 0]] {
        let bytes: Vec<_> = zeros.into_iter().flat_map(u16::to_le_bytes).collect();
        assert_eq!(
            conversion::finite_bf16_amax([bytes.as_slice()])
                .unwrap()
                .to_bits(),
            0_f32.to_bits()
        );
    }
}

#[test]
fn integer_amax_preserves_unaligned_tails_and_group_reduction() {
    // An odd affine permutation visits every encoding without sorting the
    // magnitudes. The leading byte deliberately misaligns the input address.
    let mut bytes = vec![0x55];
    for index in 0..=u16::MAX {
        let bits = index.wrapping_mul(40503).wrapping_add(17);
        if bf16::from_bits(bits).to_f32().is_finite() {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
    let bytes = &bytes[1..];
    let expected_group = scalar_f32_amax([bytes]).unwrap().to_bits();
    for elements in [
        1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
    ] {
        for chunk in bytes.chunks(elements * 2) {
            assert_eq!(
                conversion::finite_bf16_amax([chunk]).unwrap().to_bits(),
                scalar_f32_amax([chunk]).unwrap().to_bits(),
                "tail/reduction width {elements}"
            );
        }
        assert_eq!(
            conversion::finite_bf16_amax(bytes.chunks(elements * 2))
                .unwrap()
                .to_bits(),
            expected_group,
            "group width {elements}"
        );
    }
    assert_eq!(
        conversion::finite_bf16_amax(std::iter::empty::<&[u8]>())
            .unwrap()
            .to_bits(),
        scalar_f32_amax(std::iter::empty::<&[u8]>())
            .unwrap()
            .to_bits()
    );
    // A nonfinite entry must win over finite maxima in every group position,
    // including a short tail, even though rejection moves after the scan.
    for poison in [0x7f80_u16, 0xff80, 0x7f81, 0xffc1] {
        for position in [0, 1, 31, 32, 64, 68] {
            let mut values = [0x7f7f_u16; 69];
            values[position] = poison;
            let bytes: Vec<_> = values.into_iter().flat_map(u16::to_le_bytes).collect();
            assert!(scalar_f32_amax(bytes.chunks(64)).is_err());
            assert!(conversion::finite_bf16_amax(bytes.chunks(64)).is_err());
        }
    }
}

#[test]
fn integer_amax_output_subset_still_scans_all_sources_without_extra_host_charge() {
    let (_directory, archive, specs) = source(
        &[[0x4360, 0x3f80, 0xbf80, 0x8000], [0x43e0, 0, 0, 0]],
        Dtype::BF16,
    );
    let refs: Vec<_> = specs.iter().collect();
    let derived = plan::derive_group(&refs).unwrap();
    // Only the first matrix and its scale are requested. The other matrix's
    // maximum (448) still defines the shared scale, so 224 must encode as 118.
    let outputs = [&derived[0], &derived[2]];
    let host = account(8);
    let payloads = conversion::materialize(&host, &archive, &refs, &outputs).unwrap();
    assert_eq!(payloads[0].bytes(), &[118, 56, 184, 128]);
    assert_eq!(payloads[1].bytes(), &1_f32.to_le_bytes());
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    drop(payloads);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let below = account(7);
    assert!(conversion::materialize(&below, &archive, &refs, &outputs).is_err());
    assert_eq!(below.usage().unwrap().committed_bytes(), 0);

    let (_directory, archive, specs) = source(
        &[[0x4360, 0x3f80, 0xbf80, 0x8000], [0x7f80, 0, 0, 0]],
        Dtype::BF16,
    );
    let refs: Vec<_> = specs.iter().collect();
    let derived = plan::derive_group(&refs).unwrap();
    let error = conversion::materialize(&host, &archive, &refs, &[&derived[0], &derived[2]])
        .err()
        .expect("an unrequested source's nonfinite value must still reject the group");
    assert!(error.to_string().contains("nonfinite BF16 source"));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
