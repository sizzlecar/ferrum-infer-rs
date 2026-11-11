use super::*;
use crate::p_h3_image::test_fixture::tiny_transformer;

fn bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect()
}

#[test]
fn three_axis_frequencies_repeat_halves_and_use_geometric_inverse_powers() {
    let rope = PH3ImageRotaryEmbedding::new(&tiny_transformer(16, 2)).unwrap();
    let mut cosine = [0.; 24];
    let mut sine = [0.; 24];
    rope.write_frequencies(&[[0., 0., 0.], [100., 0., 0.]], &mut cosine, &mut sine)
        .unwrap();
    assert_eq!(&cosine[..12], &[1.; 12]);
    assert_eq!(&sine[..12], &[0.; 12]);
    // Second frequency is theta^-1/2=.01, so time100 produces angle1.
    assert!((cosine[13] as f64 - libm::cos(1.)).abs() < 1e-6);
    assert!((sine[13] as f64 - libm::sin(1.)).abs() < 1e-6);
    assert_eq!(&cosine[12..18], &cosine[18..24]);
    assert_eq!(&sine[12..18], &sine[18..24]);
    assert_eq!(rope.rotary_dim(), 12);
    assert_eq!(rope.head_dim(), 16);
}

#[test]
fn partial_half_split_rotates_only_leading_channels_for_every_row_and_head() {
    let rope = PH3ImageRotaryEmbedding::new(&tiny_transformer(8, 1)).unwrap();
    let input = bits(&[
        1., 2., 3., 4., 5., 6., 7., -0., 10., 20., 30., 40., 50., 60., 70., 80., 1., 2., 3., 4.,
        5., 6., 7., -0., 10., 20., 30., 40., 50., 60., 70., 80.,
    ]);
    let mut output = [0; 32];
    let cosine = [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.];
    let sine = [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.];
    rope.apply_bf16(2, 2, &input, &cosine, &sine, &mut output)
        .unwrap();
    assert_eq!(&output[..16], &input[..16]);
    assert_eq!(
        &output[16..],
        bits(&[-4., -5., -6., 1., 2., 3., 7., -0., -40., -50., -60., 10., 20., 30., 70., 80.])
    );
    assert_eq!(output[23], bf16::NEG_ZERO.to_bits());
}

#[test]
fn frequency_cast_and_each_product_retain_bf16_rounding_boundaries() {
    let rope = PH3ImageRotaryEmbedding::new(&tiny_transformer(8, 1)).unwrap();
    let input = bits(&[1.0078125, 0., 0., 1.015625, 0., 0., 0., 0.]);
    let mut output = [0; 8];
    rope.apply_bf16(
        1,
        1,
        &input,
        &[0.50390625; 6],
        &[0.50390625; 6],
        &mut output,
    )
    .unwrap();
    // Products round to .5078125 and .51171875. Their difference is -1/256;
    // delaying the product rounding gives -129/32768 instead.
    assert_eq!(bf16::from_bits(output[0]).to_f32(), -0.00390625);
    assert_ne!(bf16::from_bits(output[0]).to_f32(), -129.0_f32 / 32768.0);
    rope.apply_bf16(1, 1, &input, &[1.00390625; 6], &[0.; 6], &mut output)
        .unwrap();
    // 1.00390625 is the midpoint rounding to BF16 1 before multiplication.
    assert_eq!(output[0], input[0]);
}

#[test]
fn invalid_frequency_inputs_and_shapes_leave_both_destinations_unchanged() {
    let rope = PH3ImageRotaryEmbedding::new(&tiny_transformer(8, 1)).unwrap();
    let mut cosine = [7.; 12];
    let mut sine = [8.; 12];
    assert!(rope
        .write_frequencies(&[[0.; 3], [f64::MAX, 0., 0.]], &mut cosine, &mut sine)
        .is_err());
    assert_eq!(cosine, [7.; 12]);
    assert_eq!(sine, [8.; 12]);
    assert!(rope
        .write_frequencies(&[[0.; 3]], &mut cosine, &mut sine)
        .is_err());
    assert!(rope.frequency_elements(usize::MAX).is_err());
}

#[test]
fn invalid_rotation_or_late_overflow_cannot_partially_write() {
    let rope = PH3ImageRotaryEmbedding::new(&tiny_transformer(8, 1)).unwrap();
    let input = bits(&[1.; 8]);
    let mut output = [7; 8];
    assert!(rope
        .apply_bf16(1, 1, &input, &[1.; 5], &[0.; 6], &mut output)
        .is_err());
    let mut cosine = [1.; 6];
    cosine[5] = f32::MAX;
    assert!(rope
        .apply_bf16(1, 1, &input, &cosine, &[0.; 6], &mut output)
        .is_err());
    assert!(rope
        .apply_bf16(0, 1, &input, &[1.; 6], &[0.; 6], &mut output)
        .is_err());
    assert_eq!(output, [7; 8]);
}
