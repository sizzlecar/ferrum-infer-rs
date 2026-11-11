use super::*;

fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
        .collect()
}

#[test]
fn thw_frequency_interleaving_is_not_contiguous_sections_or_adjacent_pair_rotation() {
    let plan = PH3ImagePromptExpansionRotary::checked(64, [11, 11, 10], 10000., true).unwrap();
    let mut cos = vec![0; 64];
    let mut sin = vec![0; 64];
    plan.write_row([2, 5, 9], &mut cos, &mut sin).unwrap();
    let (cos, sin) = (values(&cos), values(&sin));
    let axes = [
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        0, 1,
    ];
    for (pair, axis) in axes.into_iter().enumerate() {
        assert_eq!(plan.axis(pair), axis);
        // Independent high-precision scalar check; tolerate F16 quantization
        // and the declared F32/native libm frequency calculation.
        let phase = [2., 5., 9.][axis] / 10000f64.powf(pair as f64 / 32.);
        assert!((cos[pair] as f64 - phase.cos()).abs() < 0.001);
        assert!((sin[pair] as f64 - phase.sin()).abs() < 0.001);
    }
}

#[test]
fn zero_position_and_bad_row_extents_have_exact_visible_boundaries() {
    let plan = PH3ImagePromptExpansionRotary::checked(8, [2, 1, 1], 10000., true).unwrap();
    let mut cos = [0u8; 8];
    let mut sin = [0xffu8; 8];
    plan.write_row([0; 3], &mut cos, &mut sin).unwrap();
    assert_eq!(values(&cos), vec![1.; 4]);
    assert_eq!(values(&sin), vec![0.; 4]);
    let before = cos;
    assert!(plan.write_row([1 << 25; 3], &mut cos, &mut sin).is_err());
    assert_eq!(cos, before);
    assert!(plan.write_row([0; 3], &mut cos[..6], &mut sin).is_err());
    assert_eq!(cos, before);
}

#[test]
fn invalid_frequency_sections_or_theta_cannot_create_a_visual_rotary_profile() {
    for (dim, sections, theta, interleaved) in [
        (0, [0, 0, 0], 10000., true),
        (7, [2, 1, 1], 10000., true),
        (8, [1, 1, 1], 10000., true),
        (8, [1, 2, 1], 10000., true),
        (8, [2, 1, 1], f32::NAN, true),
        (8, [2, 1, 1], 0., true),
        (8, [2, 1, 1], 10000., false),
    ] {
        assert!(PH3ImagePromptExpansionRotary::checked(dim, sections, theta, interleaved).is_err());
    }
}

#[test]
fn bf16_rotary_tables_follow_thw_axes_and_direct_rounding() {
    let plan = PH3ImagePromptExpansionRotary::checked(8, [2, 1, 1], 10000., true).unwrap();
    let mut cosine = [0; 8];
    let mut sine = [0; 8];
    plan.write_row_with_precision(
        [7, 31, 13],
        Qwen35DensePrecision::Bf16,
        &mut cosine,
        &mut sine,
    )
    .unwrap();
    for (pair, axis) in [0usize, 1, 2, 0].into_iter().enumerate() {
        let phase = [7f64, 31., 13.][axis] / 10000f64.powf(pair as f64 / 4.);
        let cos =
            bf16::from_le_bytes(cosine[pair * 2..pair * 2 + 2].try_into().unwrap()).to_f32() as f64;
        let sin =
            bf16::from_le_bytes(sine[pair * 2..pair * 2 + 2].try_into().unwrap()).to_f32() as f64;
        assert!((cos - phase.cos()).abs() < 0.0021);
        assert!((sin - phase.sin()).abs() < 0.0021);
    }
}
