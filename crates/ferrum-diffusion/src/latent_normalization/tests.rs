use super::*;

// Hand-computed N=2, C=2, H=2, W=3 fixtures. Each contiguous six-element
// plane is one channel; statistics repeat for the second batch item.
const MEAN: [f32; 2] = [10.0, -4.0];
const STD: [f32; 2] = [2.0, 0.5];
const LATENT: [f32; 24] = [
    6.0, 8.0, 10.0, 12.0, 14.0, 16.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 18.0, 20.0, 22.0, 24.0,
    26.0, 28.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
];
const NORMALIZED: [f32; 24] = [
    -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
];

#[test]
fn normalize_matches_hand_computed_multibatch_rectangular_channels() {
    let shape = SingleFrameLatentShape::new(2, 2, 2, 3).unwrap();
    let mut output = [f32::NAN; 24];
    normalize_latent_f32_into(shape, &LATENT, &MEAN, &STD, &mut output).unwrap();
    assert_eq!(output, NORMALIZED);
}

#[test]
fn denormalize_matches_independent_hand_computed_channel_values() {
    let shape = SingleFrameLatentShape::from_ncfhw_dims([2, 2, 1, 2, 3]).unwrap();
    let mut output = [f32::NAN; 24];
    denormalize_latent_f32_into(shape, &NORMALIZED, &MEAN, &STD, &mut output).unwrap();
    assert_eq!(output, LATENT);
}

#[test]
fn singleton_spatial_axis_keeps_channel_statistics_distinct() {
    let shape = SingleFrameLatentShape::new(2, 3, 1, 1).unwrap();
    let means = [1.0, -2.0, 4.0];
    let stds = [2.0, 4.0, 0.25];
    let latent = [3.0, 6.0, 4.75, -1.0, -10.0, 3.25];
    let normalized = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
    let mut output = [0.0; 6];
    normalize_latent_f32_into(shape, &latent, &means, &stds, &mut output).unwrap();
    assert_eq!(output, normalized);
    denormalize_latent_f32_into(shape, &normalized, &means, &stds, &mut output).unwrap();
    assert_eq!(output, latent);
}

#[test]
fn short_and_long_input_or_output_are_rejected_before_writing() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    for transform in [normalize_latent_f32_into, denormalize_latent_f32_into] {
        for input in [&[1.0; 3][..], &[1.0; 5][..]] {
            let mut output = [17.0; 4];
            assert_eq!(
                transform(shape, input, &MEAN, &STD, &mut output),
                Err(LatentNormalizationError::Layout(
                    LatentLayoutError::InputLengthMismatch {
                        expected: 4,
                        actual: input.len(),
                    }
                ))
            );
            assert_eq!(output, [17.0; 4]);
        }
        for output in [&mut [17.0; 3][..], &mut [17.0; 5][..]] {
            assert_eq!(
                transform(shape, &[1.0; 4], &MEAN, &STD, output),
                Err(LatentNormalizationError::Layout(
                    LatentLayoutError::OutputLengthMismatch {
                        expected: 4,
                        actual: output.len(),
                    }
                ))
            );
            assert!(output.iter().all(|&value| value == 17.0));
        }
    }
}

#[test]
fn statistics_must_have_exactly_one_value_per_channel() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    for transform in [normalize_latent_f32_into, denormalize_latent_f32_into] {
        for stats in [&[][..], &[1.0][..], &[1.0; 3][..]] {
            let mut output = [17.0; 4];
            assert_eq!(
                transform(shape, &[1.0; 4], stats, &STD, &mut output),
                Err(LatentNormalizationError::MeanLengthMismatch {
                    expected: 2,
                    actual: stats.len(),
                })
            );
            assert_eq!(output, [17.0; 4]);
            assert_eq!(
                transform(shape, &[1.0; 4], &MEAN, stats, &mut output),
                Err(LatentNormalizationError::StdLengthMismatch {
                    expected: 2,
                    actual: stats.len(),
                })
            );
            assert_eq!(output, [17.0; 4]);
        }
    }
}

#[test]
fn non_finite_means_and_non_positive_or_non_finite_stds_preserve_output_bits() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    let original = [0x7fc0_1234, 0x8000_0000, 0x7f80_0000, 0x3f80_0000];
    for transform in [normalize_latent_f32_into, denormalize_latent_f32_into] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut output = original.map(f32::from_bits);
            assert_eq!(
                transform(shape, &[1.0; 4], &[0.0, invalid], &STD, &mut output),
                Err(LatentNormalizationError::NonFiniteMean { channel: 1 })
            );
            assert_eq!(output.map(f32::to_bits), original);
        }
        for invalid in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut output = original.map(f32::from_bits);
            assert_eq!(
                transform(shape, &[1.0; 4], &MEAN, &[1.0, invalid], &mut output),
                Err(LatentNormalizationError::InvalidStd { channel: 1 })
            );
            assert_eq!(output.map(f32::to_bits), original);
        }
    }
}

#[test]
fn a_non_finite_last_input_does_not_write_an_earlier_valid_result() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    for transform in [normalize_latent_f32_into, denormalize_latent_f32_into] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut output = [17.0; 4];
            assert_eq!(
                transform(shape, &[1.0, 2.0, 3.0, invalid], &MEAN, &STD, &mut output),
                Err(LatentNormalizationError::NonFiniteInput { index: 3 })
            );
            assert_eq!(output, [17.0; 4]);
        }
    }
}

#[test]
fn normalization_subtraction_and_division_overflow_roll_back() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    for (means, stds) in [
        ([0.0, -f32::MAX], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, f32::MIN_POSITIVE]),
    ] {
        let mut output = [17.0; 4];
        assert_eq!(
            normalize_latent_f32_into(
                shape,
                &[1.0, 2.0, 1.0, f32::MAX],
                &means,
                &stds,
                &mut output,
            ),
            Err(LatentNormalizationError::NonFiniteResult { index: 3 })
        );
        assert_eq!(output, [17.0; 4]);
    }
}

#[test]
fn denormalization_multiplication_and_addition_overflow_roll_back() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    for (input, means, stds) in [
        ([1.0, 2.0, 0.25, 2.0], [0.0, 0.0], [1.0, f32::MAX]),
        ([1.0, 2.0, 0.0, f32::MAX], [0.0, f32::MAX], [1.0, 1.0]),
        // A fused operation could cancel overflow; separate FP32 multiplication
        // cannot. This contract must reject instead of silently changing math.
        ([1.0, 2.0, 0.0, 2.0], [0.0, -f32::MAX], [1.0, f32::MAX]),
    ] {
        let mut output = [17.0; 4];
        assert_eq!(
            denormalize_latent_f32_into(shape, &input, &means, &stds, &mut output),
            Err(LatentNormalizationError::NonFiniteResult { index: 3 })
        );
        assert_eq!(output, [17.0; 4]);
    }
}

#[test]
fn fp32_byte_overflow_and_address_limits_precede_slice_length_checks() {
    let maximum = isize::MAX as usize;
    let too_many_bytes = SingleFrameLatentShape::new(1, 1, 1, maximum / 4 + 1).unwrap();
    let overflow_bytes = SingleFrameLatentShape::new(1, 1, 1, maximum).unwrap();
    for transform in [normalize_latent_f32_into, denormalize_latent_f32_into] {
        let mut output = [17.0];
        assert_eq!(
            transform(too_many_bytes, &[1.0], &[0.0], &[1.0], &mut output),
            Err(LatentNormalizationError::Layout(
                LatentLayoutError::AddressSpaceExceeded {
                    bytes: (maximum / 4 + 1) * 4,
                }
            ))
        );
        assert_eq!(output, [17.0]);
        assert_eq!(
            transform(overflow_bytes, &[1.0], &[0.0], &[1.0], &mut output),
            Err(LatentNormalizationError::Layout(
                LatentLayoutError::ByteCountOverflow
            ))
        );
        assert_eq!(output, [17.0]);
    }
}

#[test]
fn arithmetic_keeps_direct_division_and_separate_multiply_add() {
    let shape = SingleFrameLatentShape::new(1, 1, 1, 1).unwrap();
    let mut output = [17.0];
    // Correctly rounded 9/10 differs from multiplying 9 by an f32 reciprocal.
    normalize_latent_f32_into(shape, &[9.0], &[0.0], &[10.0], &mut output).unwrap();
    assert_eq!(output[0].to_bits(), 0x3f66_6666);
    // (1 + 2^-23) * (1 - 2^-23) rounds to 1 before adding -1.
    denormalize_latent_f32_into(
        shape,
        &[f32::from_bits(0x3f80_0001)],
        &[-1.0],
        &[f32::from_bits(0x3f7f_fffe)],
        &mut output,
    )
    .unwrap();
    assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn positive_subnormal_statistics_and_finite_underflow_are_supported() {
    let shape = SingleFrameLatentShape::new(1, 1, 1, 1).unwrap();
    let smallest = f32::from_bits(1);
    let mut output = [17.0];
    normalize_latent_f32_into(shape, &[smallest], &[0.0], &[smallest], &mut output).unwrap();
    assert_eq!(output, [1.0]);
    denormalize_latent_f32_into(shape, &[1.0], &[0.0], &[smallest], &mut output).unwrap();
    assert_eq!(output[0].to_bits(), 1);
    normalize_latent_f32_into(shape, &[smallest], &[0.0], &[2.0], &mut output).unwrap();
    assert_eq!(output[0].to_bits(), 0);
}
