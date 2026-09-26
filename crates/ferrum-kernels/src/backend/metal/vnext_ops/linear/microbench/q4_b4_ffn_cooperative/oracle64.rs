//! Parallel diagnostic reference. The original FP32 reference and performance
//! admission rule remain intact. Encoded coefficients, dot products, and SiLU
//! are evaluated in FP64, with direct FP64-to-FP16 stage writebacks.
use super::*;

fn half_at(block: &[u8], offset: usize) -> f64 {
    f16::from_le_bytes(block[offset..offset + 2].try_into().unwrap()).to_f64()
}

pub(in super::super) fn decode_block(
    format: GgufBlockFormat,
    block: &[u8],
    output: &mut [f64; 256],
) {
    assert_eq!(block.len(), format.block_bytes());
    match format {
        GgufBlockFormat::Q4K => {
            let d = half_at(block, 0);
            let dmin = half_at(block, 2);
            let packed = &block[4..16];
            for (group, values) in output.chunks_exact_mut(32).enumerate() {
                let (scale, minimum) = if group < 4 {
                    (packed[group] & 63, packed[group + 4] & 63)
                } else {
                    (
                        (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4),
                        (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4),
                    )
                };
                for (index, value) in values.iter_mut().enumerate() {
                    let quant = (block[16 + group / 2 * 32 + index] >> (4 * (group % 2))) & 15;
                    *value = d * f64::from(scale) * f64::from(quant) - dmin * f64::from(minimum);
                }
            }
        }
        GgufBlockFormat::Q6K => {
            let d = half_at(block, 208);
            for (index, value) in output.iter_mut().enumerate() {
                let quarter = (index % 128) / 32;
                let low = (block[(index / 128) * 64 + (quarter % 2) * 32 + index % 32]
                    >> (4 * (quarter / 2)))
                    & 15;
                let high = (block[128 + (index / 128) * 32 + index % 32] >> (2 * quarter)) & 3;
                let quant = i32::from(low | (high << 4)) - 32;
                *value = d * f64::from(block[192 + index / 16] as i8) * f64::from(quant);
            }
        }
        _ => unreachable!("complete FFN FP64 oracle supports the declared Q4/Q6 formats"),
    }
}

pub(super) fn project(shape: Shape, bytes: &[u8], input: &[f16]) -> Vec<f16> {
    let width = shape.input as usize;
    let output = shape.output as usize;
    assert_eq!(input.len(), 4 * width);
    assert_eq!(width % 256, 0);
    let row_bytes = width / 256 * shape.format.block_bytes();
    assert_eq!(bytes.len(), row_bytes * output);
    let mut decoded = [0.0_f64; 256];
    let mut result = vec![f16::NAN; 4 * output];
    for (column, encoded_row) in bytes.chunks_exact(row_bytes).enumerate() {
        let mut sums = [0.0_f64; 4];
        for (block_index, block) in encoded_row
            .chunks_exact(shape.format.block_bytes())
            .enumerate()
        {
            decode_block(shape.format, block, &mut decoded);
            for (row, sum) in sums.iter_mut().enumerate() {
                let start = row * width + block_index * 256;
                for (&value, &weight) in input[start..start + 256].iter().zip(&decoded) {
                    *sum += value.to_f64() * weight;
                }
            }
        }
        for (row, value) in sums.into_iter().enumerate() {
            result[row * output + column] = f16::from_f64(value);
        }
    }
    assert!(result.iter().all(|value| value.is_finite()));
    result
}

pub(super) fn gate_up(gate: &[u8], up: &[u8], input: &[f16]) -> oracle::GateUp {
    let gate = project(gate_shape(), gate, input);
    let up = project(gate_shape(), up, input);
    let mut projected = Vec::with_capacity(4 * PACKED);
    let mut activated = Vec::with_capacity(4 * INTERMEDIATE);
    for row in 0..4 {
        projected.extend_from_slice(&gate[row * INTERMEDIATE..(row + 1) * INTERMEDIATE]);
        projected.extend_from_slice(&up[row * INTERMEDIATE..(row + 1) * INTERMEDIATE]);
        for column in 0..INTERMEDIATE {
            let gate = gate[row * INTERMEDIATE + column].to_f64();
            let up = up[row * INTERMEDIATE + column].to_f64();
            activated.push(f16::from_f64(gate / (1.0 + (-gate).exp()) * up));
        }
    }
    assert!(activated.iter().all(|value| value.is_finite()));
    oracle::GateUp {
        projected,
        activated,
    }
}

pub(super) fn report(
    rows: usize,
    projected: &[f16],
    activated: &[f16],
    dense: &[f16],
    reference: &oracle::GateUp,
    expected: &[f16],
    reference_f32: &oracle::GateUp,
    expected_f32: &[f16],
) -> serde_json::Value {
    let mut stages = serde_json::Map::new();
    let mut reference_differences = serde_json::Map::new();
    let mut stage_bounds_passed = true;
    for (name, offset) in [("gate", 0), ("up", INTERMEDIATE)] {
        let extract = |values: &[f16]| -> Vec<f16> {
            (0..rows)
                .flat_map(|row| {
                    values[row * PACKED + offset..row * PACKED + offset + INTERMEDIATE]
                        .iter()
                        .copied()
                })
                .collect()
        };
        let expected = extract(&reference.projected);
        let metrics = oracle::metrics(&extract(projected), &expected, INTERMEDIATE);
        stage_bounds_passed &= metrics["linear_bound_violations"].as_u64() == Some(0);
        stages.insert(name.to_owned(), metrics);
        reference_differences.insert(
            name.to_owned(),
            oracle::metrics(&extract(&reference_f32.projected), &expected, INTERMEDIATE),
        );
    }
    let activation_metrics = oracle::metrics(
        activated,
        &reference.activated[..rows * INTERMEDIATE],
        INTERMEDIATE,
    );
    stage_bounds_passed &= activation_metrics["linear_bound_violations"].as_u64() == Some(0);
    stages.insert("activation".to_owned(), activation_metrics);
    stages.insert(
        "down".to_owned(),
        oracle::metrics(dense, &expected[..rows * HIDDEN], HIDDEN),
    );
    reference_differences.insert(
        "activation".to_owned(),
        oracle::metrics(
            &reference_f32.activated[..rows * INTERMEDIATE],
            &reference.activated[..rows * INTERMEDIATE],
            INTERMEDIATE,
        ),
    );
    reference_differences.insert(
        "down".to_owned(),
        oracle::metrics(
            &expected_f32[..rows * HIDDEN],
            &expected[..rows * HIDDEN],
            HIDDEN,
        ),
    );
    // Every FP16 value is exactly representable in FP32. This widening is only
    // the existing metric/catalog API, never part of reference arithmetic.
    let actual: Vec<_> = dense.iter().map(|value| value.to_f32()).collect();
    let expected: Vec<_> = expected[..rows * HIDDEN]
        .iter()
        .map(|value| value.to_f32())
        .collect();
    let full_pipeline = numerical_tolerance::assert_matches(
        "parallel FP64 complete FFN oracle with direct FP16 stage writebacks",
        &actual,
        &[rows, HIDDEN],
        &expected,
        &[rows, HIDDEN],
        numerical_tolerance::LogicalDtype::Fp16,
        FULL_FFN_TOLERANCE,
    );
    serde_json::json!({"passed":stage_bounds_passed && full_pipeline.is_ok(),
        "oracle_arithmetic":"fp64_coefficients_dot_and_silu_direct_fp16_writebacks",
        "metric_input_representation":"exact_fp16_to_fp32_widening_catalog_api",
        "performance_gate":"unchanged_original_fp32_oracle",
        "stages":stages,"f32_reference_vs_f64_reference":reference_differences,
        "full_pipeline_tolerance":FULL_FFN_TOLERANCE,"full_pipeline_error":full_pipeline.err()})
}

#[test]
fn complete_ffn_f64_decodes_actual_signed_quantized_coefficients() {
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let shape = Shape {
            name: "fp64_decode_check",
            input: 256,
            output: 3,
            format,
        };
        for seed in [2, 5, 11] {
            let encoded = oracle::matrix(shape, seed);
            for block in encoded.chunks_exact(format.block_bytes()) {
                let mut decoded = [0.0_f64; 256];
                decode_block(format, block, &mut decoded);
                for (index, value) in decoded.into_iter().enumerate() {
                    // These fixture products fit exactly in FP32 as well;
                    // independent scalar indexing cross-checks every value.
                    assert_eq!(
                        value,
                        f64::from(format.decode_value(block, index)),
                        "format={format:?}, seed={seed}, coefficient={index}"
                    );
                }
            }
        }
    }
}

#[test]
fn complete_ffn_f64_project_matches_independent_scalar_accumulation() {
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let shape = Shape {
            name: "fp64_project_check",
            input: 768,
            output: 5,
            format,
        };
        let input = oracle::input()[..4 * shape.input as usize].to_vec();
        let encoded = oracle::matrix(shape, 11);
        let actual = project(shape, &encoded, &input);
        for row in 0..4 {
            for column in 0..shape.output as usize {
                // The scalar diagnostic uses independent coefficient indexing;
                // exact representability for this fixture is tested above.
                let expected = oracle::scalar_dot(
                    shape,
                    &encoded,
                    &input[row * shape.input as usize..(row + 1) * shape.input as usize],
                    column,
                );
                assert_eq!(
                    actual[row * shape.output as usize + column].to_bits(),
                    f16::from_f64(expected).to_bits(),
                    "format={format:?}, row={row}, column={column}"
                );
            }
        }
    }
}
