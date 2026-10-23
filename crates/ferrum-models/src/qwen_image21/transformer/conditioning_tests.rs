use super::*;

fn config(causal: bool) -> QwenImage21TransformerConfig {
    super::super::tests::config(causal)
}

#[test]
fn timestep_cos_then_sin_and_single_shared_t0_are_explicit() {
    let checked = config(true);
    let projection = QwenImage21Timestep256::new().unwrap();
    let zero = QwenImage21ModelTimestep::from_normalized(0.0).unwrap();
    let half = QwenImage21ModelTimestep::from_normalized(0.5).unwrap();
    let mut output = [0xffff; 3 * 256];
    projection
        .write_into(&checked, &[zero, half], &mut output)
        .unwrap();
    for offset in [0, 2 * 256] {
        assert_eq!(&output[offset..offset + 128], &[0x3f80; 128]);
        assert_eq!(&output[offset + 128..offset + 256], &[0; 128]);
    }
    // The first frequency is exactly one: cos(500) is in
    // [-0.883850,-0.883848], wholly within the BF16 cell of -113/128.
    assert_eq!(output[256], 0xbf62);
    assert_eq!(projection.frequencies[0].to_bits(), 1.0_f32.to_bits());
    assert!(projection
        .frequencies
        .windows(2)
        .all(|pair| pair[0] > pair[1]));
    assert_eq!(
        QwenImage21Timestep256::output_elements(&config(false)).unwrap(),
        512
    );
}

#[test]
fn early_bf16_time_cast_and_pipeline_division_are_not_algebraically_cancelled() {
    // This exactly representable midpoint rounds to the even lower BF16.
    let halfway = QwenImage21ModelTimestep::from_normalized(0.5 + 1.0 / 512.0).unwrap();
    let half = QwenImage21ModelTimestep::from_normalized(0.5).unwrap();
    assert_eq!(halfway.bf16_bits(), 0x3f00);
    let projection = QwenImage21Timestep256::new().unwrap();
    let mut output = [0; 512];
    projection
        .write_into(&config(false), &[halfway, half], &mut output)
        .unwrap();
    assert_eq!(&output[..256], &output[256..]);

    // 499 + 1/64 rounds to BF16 500 before division. Dividing first is below
    // the midpoint 0.4990234375 between BF16 0.498046875 and 0.5.
    let raw = 499.015625_f32;
    assert_eq!(
        QwenImage21ModelTimestep::from_scheduler_timestep(raw)
            .unwrap()
            .bf16_bits(),
        0x3f00
    );
    assert_eq!(
        QwenImage21ModelTimestep::from_normalized(raw / 1000.0)
            .unwrap()
            .bf16_bits(),
        0x3eff
    );
}

#[test]
fn invalid_time_domains_and_lengths_do_not_write() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.125, 1.01] {
        assert!(QwenImage21ModelTimestep::from_normalized(value).is_err());
    }
    assert!(QwenImage21ModelTimestep::from_scheduler_timestep(1001.0).is_err());
    let config = config(true);
    let projection = QwenImage21Timestep256::new().unwrap();
    let timestep = QwenImage21ModelTimestep::from_normalized(0.5).unwrap();
    let mut output = [0x1234; 768];
    assert!(projection
        .write_into(&config, &[timestep], &mut output)
        .is_err());
    assert_eq!(output, [0x1234; 768]);
    assert!(projection
        .write_into(&config, &[timestep; 2], &mut output[..767])
        .is_err());
    assert_eq!(output, [0x1234; 768]);
    // A user-supplied -0 preserves its sign; the appended t0 remains +0.
    let negative_zero = QwenImage21ModelTimestep::from_normalized(-0.0).unwrap();
    projection
        .write_into(&config, &[negative_zero, timestep], &mut output)
        .unwrap();
    assert_eq!(&output[128..256], &[0x8000; 128]);
    assert_eq!(&output[640..768], &[0; 128]);
}

fn near(actual: f32, expected: f64) {
    // Independent mathematical checks of selected small angles, not a claim
    // of correctly rounded trig over the full input domain.
    assert!(
        (f64::from(actual) - expected).abs() < 1.0e-7,
        "{actual} vs {expected}"
    );
}

#[test]
fn rotary_uses_padded_text_frame_and_adjacent_frame_height_width_pairs() {
    let config = config(true);
    let descriptor = config.rotary_frequencies().unwrap();
    assert_eq!(descriptor.axes(), [2, 2, 4]);
    let mut output = vec![f32::NAN; descriptor.output_elements(config.layout()).unwrap()];
    descriptor.write_into(config.layout(), &mut output).unwrap();
    // Position zero: all pairs are exact [1,+0].
    assert_eq!(&output[..8], &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    // First image token is at [frame,height,width]=[3,-1,-2]. Frame=3
    // includes padded text even though sample 1 has only one valid text token.
    let row = &output[3 * 8..4 * 8];
    near(row[0], -0.9899924966004454);
    near(row[1], 0.1411200080598672);
    near(row[2], 0.5403023058681398);
    near(row[3], -0.8414709848078965);
    near(row[4], -0.4161468365471424);
    near(row[5], -0.9092974268256817);
    // Width's second pair has exponent 2/4: reciprocal(sqrt(10000))=.01,
    // represented as 0x3c23d70a. Its F32 phase is -2 times that value.
    assert_eq!(angular_frequency(1, 4).to_bits(), 0x3c23d70a);
    near(row[6], 0.9998000066665778);
    near(row[7], -0.0199986666933331);
    // The second text token increments all three axes, including width's
    // lower frequency; this catches accidentally freezing text spatial axes.
    near(output[8 + 1], 0.8414709848078965);
    near(output[8 + 3], 0.8414709848078965);
    near(output[8 + 5], 0.8414709848078965);
    near(output[8 + 7], 0.0099998333341667);
}

#[test]
fn rotary_frequency_exponents_and_invalid_destinations_are_checked() {
    let descriptor = QwenImage21RotaryFrequencies::new([16, 56, 56]).unwrap();
    assert_eq!(descriptor.width, 128);
    // At half an axis' pairs, exponent=1/2, independent of axis length.
    for dimension in [16, 56] {
        assert_eq!(angular_frequency(0, dimension).to_bits(), 1.0_f32.to_bits());
        assert_eq!(
            angular_frequency(dimension / 4, dimension).to_bits(),
            0x3c23d70a
        );
    }
    for axes in [[0, 2, 4], [3, 2, 4], [(1 << 24) + 2, 2, 4]] {
        assert!(QwenImage21RotaryFrequencies::new(axes).is_err());
    }
    let layout = config(true).layout().clone();
    let mut output = [123.0; 7];
    assert!(descriptor.write_into(&layout, &mut output).is_err());
    assert_eq!(output, [123.0; 7]);
    let mut output = [123.0; 256];
    assert!(descriptor
        .write_range(&layout, layout.sequence_length() - 1, 2, &mut output)
        .is_err());
    assert_eq!(output, [123.0; 256]);
    assert!(checked_elements::<f32>(u64::MAX, 128).is_err());
    assert!(checked_elements::<f32>(isize::MAX as u64, 1).is_err());
}

#[test]
fn rotary_edit_phases_match_independent_positions_and_analytic_frequencies() {
    use super::super::QwenImage21TransformerPhase;
    let edit = QwenImage21TransformerConfig::new(
        config(true).transformer().clone(),
        QwenImage21T2iLayout::new(5, 2, 2, vec![5]).unwrap(),
    )
    .unwrap()
    .with_reference_image(1, 1, 2, 2)
    .unwrap();
    let positions = [
        [0, 0, 0],
        [1, -1, -1],
        [1, -1, 0],
        [1, 0, -1],
        [1, 0, 0],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, -1, -1],
        [6, -1, 0],
        [6, 0, -1],
        [6, 0, 0],
    ];
    let descriptor = edit.rotary_frequencies().unwrap();
    for (phase, expected_positions) in [
        (
            QwenImage21TransformerPhase::PrefixPrefill,
            positions.as_slice(),
        ),
        (QwenImage21TransformerPhase::PrefixDecode, &positions[8..]),
    ] {
        let config = edit.clone().with_execution_phase(phase).unwrap();
        let mut output = vec![f32::NAN; descriptor.phase_output_elements(&config).unwrap()];
        descriptor.write_phase_into(&config, &mut output).unwrap();
        assert_eq!(output.len(), expected_positions.len() * 8);
        for (row, &[frame, height, width]) in expected_positions.iter().enumerate() {
            // Axes [2,2,4]: only width's second pair has inverse sqrt(10000)=.01.
            let angles = [
                frame as f32,
                height as f32,
                width as f32,
                width as f32 * 0.01,
            ];
            for (pair, angle) in angles.into_iter().enumerate() {
                let offset = row * 8 + pair * 2;
                assert_eq!(output[offset].to_bits(), libm::cosf(angle).to_bits());
                assert_eq!(output[offset + 1].to_bits(), libm::sinf(angle).to_bits());
            }
        }
    }
}

#[test]
fn rotary_wide_axes_preserve_pair_boundaries_without_input_sized_scratch() {
    let descriptor = QwenImage21RotaryFrequencies::new([256, 256, 4]).unwrap();
    let layout = QwenImage21T2iLayout::new(2, 2, 2, vec![1]).unwrap();
    let positions = [
        [0, 0, 0],
        [1, 1, 1],
        [2, -1, -1],
        [2, -1, 0],
        [2, 0, -1],
        [2, 0, 0],
    ];
    let mut output = vec![f32::NAN; descriptor.output_elements(&layout).unwrap()];
    descriptor.write_into(&layout, &mut output).unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
    for (row, positions) in positions.into_iter().enumerate() {
        // Known exponents 0 and 1/2 give exact inverse frequencies 1 and .01.
        // Include both full wide axes and every pair of the final short axis.
        for (pair, axis, inverse) in [
            (0, 0, 1.0),
            (64, 0, 0.01),
            (128, 1, 1.0),
            (192, 1, 0.01),
            (256, 2, 1.0),
            (257, 2, 0.01),
        ] {
            let angle = positions[axis] as f32 * inverse;
            let offset = row * 516 + pair * 2;
            assert_eq!(output[offset].to_bits(), libm::cosf(angle).to_bits());
            assert_eq!(output[offset + 1].to_bits(), libm::sinf(angle).to_bits());
        }
    }
}
