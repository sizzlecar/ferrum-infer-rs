use super::*;
use serde_json::json;

fn config(taps: &[u32], hidden: u64, output: u64) -> PH3ImageTextConnectorConfig {
    PH3ImageTextConnectorConfig::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"PH3ImageTextConnector", "hidden_state_indices":taps,
            "text_hidden_size":hidden, "out_channels":output
        }))
        .unwrap(),
    )
    .unwrap()
}

struct Fixture {
    input_mean: [f32; 4],
    input_std: [f32; 4],
    projection: [f32; 8],
    output_mean: [f32; 2],
    output_std: [f32; 2],
    sink: [f32; 2],
}

impl Fixture {
    fn new() -> Self {
        Self {
            input_mean: [1.0, 2.0, 3.0, 4.0],
            input_std: [1.0, 2.0, 1.0, 4.0],
            projection: [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 2.0, -2.0],
            output_mean: [10.0, -1.0],
            output_std: [2.0, 0.5],
            sink: [99.0, -99.0],
        }
    }

    fn weights(&self) -> PH3ImageTextConnectorWeights<'_> {
        PH3ImageTextConnectorWeights {
            input_mean: &self.input_mean,
            input_std: &self.input_std,
            projection: &self.projection,
            output_mean: &self.output_mean,
            output_std: &self.output_std,
            sink_embedding: &self.sink,
        }
    }
}

#[test]
fn hand_computed_normalization_projection_and_sink_preserve_batches_and_tokens() {
    let config = config(&[2, 1], 2, 2);
    let reference = PH3ImageTextConnectorReference::new(&config, 2, 3).unwrap();
    let state_two = [100., 200., 2., 6., 3., 2., -7., 8., 0., 2., 1., 4.];
    let state_one = [300., 400., 4., 12., 5., 4., 9., 10., 3., 8., 2., 0.];
    let states = [
        PH3ImageHiddenState {
            index: 2,
            values: &state_two,
        },
        PH3ImageHiddenState {
            index: 1,
            values: &state_one,
        },
    ];
    let fixture = Fixture::new();
    let mut output = [f32::NAN; 12];
    reference
        .write_f32(&states, fixture.weights(), &mut output)
        .unwrap();
    // Batch 0, token 1: standardized [1,2,1,2], dots [16,-3],
    // then per-output scale/offset [16*2+10, -3*.5-1] = [42,-2.5].
    // Token 2 standardizes to [2,0,2,0], yielding [26,0].
    // Batch 1 uses [-1,0,0,1] and [0,1,-1,-1], never batch 0's rows.
    assert_eq!(
        output,
        [99., -99., 42., -2.5, 26., 0., 99., -99., 16., -1.5, 0., -1.]
    );
    assert_eq!(reference.shape().state_elements(), state_two.len());
    assert_eq!(
        reference.shape().projection_elements(),
        fixture.projection.len()
    );
}

#[test]
fn declared_layer_order_controls_concatenation_and_mismatched_descriptors_fail() {
    let first_config = config(&[0, 1], 1, 1);
    let swapped_config = config(&[1, 0], 1, 1);
    let first = PH3ImageTextConnectorReference::new(&first_config, 1, 2).unwrap();
    let swapped = PH3ImageTextConnectorReference::new(&swapped_config, 1, 2).unwrap();
    let zero = PH3ImageHiddenState {
        index: 0,
        values: &[999., 2.],
    };
    let one = PH3ImageHiddenState {
        index: 1,
        values: &[-999., 3.],
    };
    let weights = PH3ImageTextConnectorWeights {
        input_mean: &[0., 0.],
        input_std: &[1., 1.],
        projection: &[1., 10.],
        output_mean: &[0.],
        output_std: &[1.],
        sink_embedding: &[7.],
    };
    let mut output = [-1.; 2];
    first.write_f32(&[zero, one], weights, &mut output).unwrap();
    assert_eq!(output, [7., 32.]);
    swapped
        .write_f32(&[one, zero], weights, &mut output)
        .unwrap();
    assert_eq!(output, [7., 23.]);
    let before = output;
    assert!(first.write_f32(&[one, zero], weights, &mut output).is_err());
    assert_eq!(output, before);
}

#[test]
fn fp32_values_are_preserved_without_bf16_normalization_or_projection() {
    let config = config(&[1], 1, 1);
    let reference = PH3ImageTextConnectorReference::new(&config, 1, 2).unwrap();
    let weights = PH3ImageTextConnectorWeights {
        input_mean: &[1.],
        input_std: &[0.5],
        projection: &[3.],
        output_mean: &[0.125],
        output_std: &[2.],
        sink_embedding: &[1.00390625],
    };
    let mut output = [0.; 2];
    reference
        .write_f32(
            &[PH3ImageHiddenState {
                index: 1,
                values: &[0., 1.00390625],
            }],
            weights,
            &mut output,
        )
        .unwrap();
    // (1.00390625-1)/.5 * 3 * 2 + .125 = .171875, exactly in FP32.
    assert_eq!(output, [1.00390625, 0.171875]);
    assert_ne!(half::bf16::from_f32(output[0]).to_f32(), output[0]);
}

#[test]
fn each_sequence_keeps_a_sink_when_it_has_no_other_tokens() {
    let config = config(&[1], 1, 1);
    let reference = PH3ImageTextConnectorReference::new(&config, 2, 1).unwrap();
    let weights = PH3ImageTextConnectorWeights {
        input_mean: &[0.],
        input_std: &[1.],
        projection: &[f32::MAX],
        output_mean: &[0.],
        output_std: &[1.],
        sink_embedding: &[5.],
    };
    let mut output = [0.; 2];
    reference
        .write_f32(
            &[PH3ImageHiddenState {
                index: 1,
                values: &[2., 3.],
            }],
            weights,
            &mut output,
        )
        .unwrap();
    // The discarded projected sink values overflow, but neither is published.
    assert_eq!(output, [5., 5.]);
}

#[test]
fn std_values_are_finite_positive_and_fail_before_any_output_write() {
    let config = config(&[2, 1], 2, 2);
    let reference = PH3ImageTextConnectorReference::new(&config, 1, 2).unwrap();
    let states = [
        PH3ImageHiddenState {
            index: 2,
            values: &[1., 2., 3., 4.],
        },
        PH3ImageHiddenState {
            index: 1,
            values: &[5., 6., 7., 8.],
        },
    ];
    for invalid in [0., -1., f32::NAN, f32::INFINITY] {
        for input_std in [true, false] {
            let mut fixture = Fixture::new();
            if input_std {
                fixture.input_std[1] = invalid;
            } else {
                fixture.output_std[1] = invalid;
            }
            let mut output = [f32::from_bits(0x7fc0_0001); 4];
            let before = output.map(f32::to_bits);
            assert!(reference
                .write_f32(&states, fixture.weights(), &mut output)
                .is_err());
            assert_eq!(output.map(f32::to_bits), before);
        }
    }
}

#[test]
fn nonfinite_checkpoint_or_hidden_values_are_rejected() {
    let config = config(&[2, 1], 2, 2);
    let reference = PH3ImageTextConnectorReference::new(&config, 1, 2).unwrap();
    for field in 0..4 {
        let mut fixture = Fixture::new();
        match field {
            0 => fixture.input_mean[0] = f32::NAN,
            1 => fixture.projection[0] = f32::INFINITY,
            2 => fixture.output_mean[0] = f32::NAN,
            _ => fixture.sink[0] = f32::NAN,
        }
        let mut output = [9.; 4];
        assert!(reference
            .write_f32(
                &[
                    PH3ImageHiddenState {
                        index: 2,
                        values: &[0.; 4]
                    },
                    PH3ImageHiddenState {
                        index: 1,
                        values: &[0.; 4]
                    }
                ],
                fixture.weights(),
                &mut output
            )
            .is_err());
        assert_eq!(output, [9.; 4]);
    }
    let mut output = [9.; 4];
    assert!(reference
        .write_f32(
            &[
                PH3ImageHiddenState {
                    index: 2,
                    values: &[f32::NAN, 0., 0., 0.]
                },
                PH3ImageHiddenState {
                    index: 1,
                    values: &[0.; 4]
                }
            ],
            Fixture::new().weights(),
            &mut output
        )
        .is_err());
    assert_eq!(output, [9.; 4]);
}

#[test]
fn shapes_states_and_weight_extents_are_checked_before_writes() {
    let config = config(&[2, 1], 2, 2);
    let reference = PH3ImageTextConnectorReference::new(&config, 1, 2).unwrap();
    let fixture = Fixture::new();
    let states = [
        PH3ImageHiddenState {
            index: 2,
            values: &[0.; 4],
        },
        PH3ImageHiddenState {
            index: 1,
            values: &[0.; 4],
        },
    ];
    let mut output = [9.; 4];
    assert!(reference
        .write_f32(&states[..1], fixture.weights(), &mut output)
        .is_err());
    assert!(reference
        .write_f32(
            &[
                states[0],
                PH3ImageHiddenState {
                    index: 1,
                    values: &[0.; 3]
                }
            ],
            fixture.weights(),
            &mut output
        )
        .is_err());
    for field in 0..6 {
        let mut weights = fixture.weights();
        match field {
            0 => weights.input_mean = &weights.input_mean[..3],
            1 => weights.input_std = &weights.input_std[..3],
            2 => weights.projection = &weights.projection[..7],
            3 => weights.output_mean = &weights.output_mean[..1],
            4 => weights.output_std = &weights.output_std[..1],
            _ => weights.sink_embedding = &weights.sink_embedding[..1],
        }
        assert!(reference.write_f32(&states, weights, &mut output).is_err());
        assert_eq!(output, [9.; 4]);
    }
    assert!(reference
        .write_f32(&states, fixture.weights(), &mut output[..3])
        .is_err());
    assert_eq!(output, [9.; 4]);
}

#[test]
fn late_projection_overflow_leaves_all_output_bits_unchanged() {
    let config = config(&[1], 1, 1);
    let reference = PH3ImageTextConnectorReference::new(&config, 1, 3).unwrap();
    let weights = PH3ImageTextConnectorWeights {
        input_mean: &[0.],
        input_std: &[1.],
        projection: &[f32::MAX],
        output_mean: &[0.],
        output_std: &[1.],
        sink_embedding: &[5.],
    };
    let mut output = [0.25, -7., f32::from_bits(0x7fc0_0001)];
    let before = output.map(f32::to_bits);
    assert!(reference
        .write_f32(
            &[PH3ImageHiddenState {
                index: 1,
                values: &[0., 1., 2.]
            }],
            weights,
            &mut output
        )
        .is_err());
    assert_eq!(output.map(f32::to_bits), before);
}

#[test]
fn zero_extents_element_overflow_and_fp32_address_space_are_rejected() {
    let config = config(&[1], 1, 1);
    for (batch, tokens) in [
        (0, 1),
        (1, 0),
        (usize::MAX, 2),
        (isize::MAX as usize / std::mem::size_of::<f32>() + 1, 1),
    ] {
        assert!(PH3ImageTextConnectorReference::new(&config, batch, tokens).is_err());
    }
    let shape = PH3ImageTextConnectorShape::new(&config, 2, 3).unwrap();
    assert_eq!(shape.batch(), 2);
    assert_eq!(shape.tokens(), 3);
    assert_eq!(shape.state_elements(), 6);
    assert_eq!(shape.output_elements(), 6);
}

#[test]
fn released_connector_shape_accounts_for_all_taps_and_projection_weights() {
    let config = config(&[13, 19, 25, 31, 36], 2560, 5120);
    let shape = PH3ImageTextConnectorShape::new(&config, 2, 4).unwrap();
    assert_eq!(shape.input_dim(), 12800);
    assert_eq!(shape.output_dim(), 5120);
    assert_eq!(shape.state_elements(), 20480);
    assert_eq!(shape.output_elements(), 40960);
    assert_eq!(shape.projection_elements(), 65_536_000);
}
