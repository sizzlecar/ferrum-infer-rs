use super::*;

#[test]
fn bf16_to_f32_widen_preserves_every_encoding_and_unaligned_tails() {
    let words = (0..=u16::MAX)
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    for offset in 0..4 {
        let mut input = vec![0xa5; offset];
        input.extend_from_slice(&words);
        for tail in 0..8 {
            let source = &input[offset..input.len() - tail * 2];
            let bytes = source.len() * 2;
            let mut actual = vec![0x5a; offset + bytes + 7];
            let mut expected = actual.clone();
            let range = offset..offset + bytes;
            convert_floating_scalar(
                ElementType::Bf16,
                ElementType::F32,
                source,
                &mut expected[range.clone()],
                None,
            )
            .unwrap();
            convert_floating(
                ElementType::Bf16,
                ElementType::F32,
                source,
                &mut actual[range],
                None,
            )
            .unwrap();
            assert_eq!(actual, expected, "offset={offset}, tail={tail}");
        }
    }
}

#[test]
fn bf16_to_f32_widen_rejects_mismatched_extents_before_writing() {
    for (input_bytes, output_bytes) in [(1, 4), (2, 3), (2, 8), (4, 4)] {
        let input = vec![0; input_bytes];
        let mut output = vec![0xa5; output_bytes];
        assert!(convert_floating(
            ElementType::Bf16,
            ElementType::F32,
            &input,
            &mut output,
            None,
        )
        .is_err());
        assert_eq!(output, vec![0xa5; output_bytes]);
    }
    widen_bf16_to_f32(&[], &mut []).unwrap();
}
