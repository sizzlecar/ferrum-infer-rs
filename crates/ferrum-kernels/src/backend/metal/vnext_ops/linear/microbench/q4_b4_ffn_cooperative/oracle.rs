//! Dense, complete-output CPU oracle. Only one quantization block is decoded
//! at a time; no full FP32 copy of any of the three matrices is constructed.
use super::*;

pub(super) fn input() -> Vec<f16> {
    (0..4 * HIDDEN)
        .map(|index| {
            let phase = (index + 1) as f32 * 0.017 + (index / HIDDEN) as f32 * 0.113;
            let value = f16::from_f32(phase.sin() * 0.03125 + (phase * 0.37).cos() * 0.0078125);
            if value == f16::ZERO {
                f16::from_f32(0.000_061_035_156)
            } else {
                value
            }
        })
        .collect()
}

pub(in super::super) fn matrix(shape: Shape, seed: usize) -> Vec<u8> {
    let mut bytes = weights(shape);
    let blocks_per_row = shape.input as usize / shape.format.block_values();
    for (index, block) in bytes
        .chunks_exact_mut(shape.format.block_bytes())
        .enumerate()
    {
        let row = index / blocks_per_row;
        let scale_offset = if shape.format == GgufBlockFormat::Q6K {
            208
        } else {
            0
        };
        // Keep stored quantization scales normal in F16; the experiment targets
        // dense FFN dispatch, not subnormal scale conversion behavior.
        let factor = 0.0625 + ((row * 13 + index * 7 + seed) % 257) as f32 / 16384.0;
        let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
        let old = f16::from_le_bytes(block[scale_offset..scale_offset + 2].try_into().unwrap());
        block[scale_offset..scale_offset + 2]
            .copy_from_slice(&f16::from_f32(old.to_f32() * factor * sign).to_le_bytes());
        if shape.format == GgufBlockFormat::Q4K {
            let old = f16::from_le_bytes(block[2..4].try_into().unwrap());
            block[2..4].copy_from_slice(&f16::from_f32(old.to_f32() * factor).to_le_bytes());
        }
    }
    bytes
}

pub(super) fn project(shape: Shape, bytes: &[u8], input: &[f16]) -> Vec<f16> {
    let width = shape.input as usize;
    let output = shape.output as usize;
    assert_eq!(input.len(), 4 * width);
    let block_values = shape.format.block_values();
    assert_eq!(block_values, 256);
    let row_bytes = width / block_values * shape.format.block_bytes();
    assert_eq!(bytes.len(), row_bytes * output);
    let mut decoded = [0.0_f32; 256];
    let mut result = vec![f16::NAN; 4 * output];
    for (column, encoded_row) in bytes.chunks_exact(row_bytes).enumerate() {
        let mut sums = [0.0_f32; 4];
        for (block_index, block) in encoded_row
            .chunks_exact(shape.format.block_bytes())
            .enumerate()
        {
            shape.format.decode_block(block, &mut decoded);
            for (row, sum) in sums.iter_mut().enumerate() {
                let start = row * width + block_index * block_values;
                for (&value, &weight) in input[start..start + block_values].iter().zip(&decoded) {
                    *sum += value.to_f32() * weight;
                }
            }
        }
        for (row, value) in sums.into_iter().enumerate() {
            result[row * output + column] = f16::from_f32(value);
        }
    }
    assert!(result.iter().all(|value| value.is_finite()));
    result
}

pub(super) struct GateUp {
    pub projected: Vec<f16>,
    pub activated: Vec<f16>,
}

/// Diagnostic only: retain the original per-element linear bounds. A failure
/// is reported after both complete experiment arms have been inspected.
pub(in super::super) fn metrics(
    actual: &[f16],
    expected: &[f16],
    width: usize,
) -> serde_json::Value {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    assert_eq!(actual.len() % width, 0);
    let mut squared_error = 0.0_f64;
    let mut expected_energy = 0.0_f64;
    let mut actual_energy = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut worst_index = 0;
    let mut maximum_error = 0.0_f64;
    let mut maximum_relative_error = 0.0_f64;
    let mut maximum_ulp_distance = 0_u32;
    let mut violations = 0;
    let mut violations_by_batch_column_parity = vec![[0_usize; 2]; actual.len() / width];
    let mut nonfinite = 0;
    let mut first_violation = None;
    let mut actual_argmax = 0;
    let mut expected_argmax = 0;
    let mut actual_max_abs = 0.0_f64;
    let mut expected_max_abs = 0.0_f64;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let af = f64::from(a.to_f32());
        let ef = f64::from(e.to_f32());
        if !af.is_finite() || !ef.is_finite() {
            nonfinite += 1;
            violations += 1;
            violations_by_batch_column_parity[index / width][(index % width) % 2] += 1;
            first_violation.get_or_insert(index);
            continue;
        }
        let error = (af - ef).abs();
        squared_error += error * error;
        expected_energy += ef * ef;
        actual_energy += af * af;
        dot += af * ef;
        if error > maximum_error {
            maximum_error = error;
            worst_index = index;
        }
        maximum_relative_error = maximum_relative_error.max(error / ef.abs().max(1.0e-12));
        let ordered = |value: f16| {
            let bits = value.to_bits();
            if bits & 0x8000 == 0 {
                0x8000_i32 + i32::from(bits)
            } else {
                0x8000_i32 - i32::from(bits & 0x7fff)
            }
        };
        maximum_ulp_distance = maximum_ulp_distance.max(ordered(a).abs_diff(ordered(e)));
        if error > f64::from(linear_tolerance(ElementType::F16, e.to_f32())) {
            violations += 1;
            violations_by_batch_column_parity[index / width][(index % width) % 2] += 1;
            first_violation.get_or_insert(index);
        }
        if af > f64::from(actual[actual_argmax].to_f32()) {
            actual_argmax = index;
        }
        if ef > f64::from(expected[expected_argmax].to_f32()) {
            expected_argmax = index;
        }
        actual_max_abs = actual_max_abs.max(af.abs());
        expected_max_abs = expected_max_abs.max(ef.abs());
    }
    let point = |index: usize| {
        serde_json::json!({"index":index, "row":index / width, "column":index % width,
            "actual":actual[index].to_f32(), "expected":expected[index].to_f32(),
            "actual_bits":actual[index].to_bits(), "expected_bits":expected[index].to_bits()})
    };
    serde_json::json!({"elements":actual.len(), "shape":[actual.len() / width,width],
        "nonfinite_pairs":nonfinite, "linear_bound_violations":violations,
        "linear_bound_violations_by_batch_even_odd_column":violations_by_batch_column_parity,
        "rmse":(squared_error / actual.len() as f64).sqrt(),
        "relative_l2":(squared_error / expected_energy.max(1.0e-24)).sqrt(),
        "cosine":dot / (actual_energy * expected_energy).sqrt().max(1.0e-24),
        "max_abs_error":maximum_error, "max_relative_error":maximum_relative_error,
        "max_f16_ulp_distance":maximum_ulp_distance,
        "actual_max_abs":actual_max_abs, "expected_max_abs":expected_max_abs,
        "first_linear_bound_violation":first_violation.map(point),
        "worst_absolute":point(worst_index), "actual_argmax":point(actual_argmax),
        "expected_argmax":point(expected_argmax)})
}

pub(super) fn diagnostic_indices(metrics: &serde_json::Value) -> Vec<usize> {
    let mut indices = vec![metrics["worst_absolute"]["index"].as_u64().unwrap() as usize];
    if let Some(index) = metrics["first_linear_bound_violation"]["index"].as_u64() {
        indices.push(index as usize);
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

pub(super) fn scalar_swiglu(gate: f16, up: f16) -> f64 {
    let gate = f64::from(gate.to_f32());
    gate / (1.0 + (-gate).exp()) * f64::from(up.to_f32())
}

#[test]
fn complete_ffn_diagnostics_preserve_bounds_and_report_coordinates() {
    let expected = [f16::from_f32(-1.0), f16::from_f32(2.0)];
    let actual = [f16::from_f32(-0.8), f16::from_f32(2.0)];
    let measured = metrics(&actual, &expected, 2);
    assert_eq!(measured["linear_bound_violations"], 1);
    assert_eq!(
        measured["linear_bound_violations_by_batch_even_odd_column"],
        serde_json::json!([[1, 0]])
    );
    assert_eq!(measured["first_linear_bound_violation"]["column"], 0);
    assert_eq!(measured["worst_absolute"]["row"], 0);
    assert_eq!(measured["actual_argmax"]["index"], 1);
    assert_eq!(diagnostic_indices(&measured), [0]);
    assert_eq!(
        metrics(&expected, &expected, 2)["linear_bound_violations"],
        0
    );
    let nonfinite = metrics(&[f16::NAN, f16::INFINITY], &expected, 2);
    assert_eq!(nonfinite["nonfinite_pairs"], 2);
    assert_eq!(nonfinite["linear_bound_violations"], 2);
    let pattern = metrics(
        &[expected[0], expected[1], f16::NAN, expected[1]],
        &[expected[0], expected[1], expected[0], expected[1]],
        2,
    );
    assert_eq!(
        pattern["linear_bound_violations_by_batch_even_odd_column"],
        serde_json::json!([[0, 0], [1, 0]])
    );
}

#[test]
fn complete_ffn_scalar_oracle_agrees_with_block_decode_for_signed_scales() {
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let shape = Shape {
            name: "scalar_oracle_check",
            input: 256,
            output: 3,
            format,
        };
        let input = input()[..4 * 256].to_vec();
        for seed in [2, 5, 11] {
            let encoded = matrix(shape, seed);
            let expected = project(shape, &encoded, &input);
            for row in 0..4 {
                for column in 0..3 {
                    let scalar =
                        scalar_dot(shape, &encoded, &input[row * 256..(row + 1) * 256], column);
                    let scalar_fp16 = f16::from_f64(scalar).to_f32();
                    let block_fp16 = expected[row * 3 + column].to_f32();
                    assert!((scalar_fp16 - block_fp16).abs() <= linear_tolerance(ElementType::F16, block_fp16),
                        "format={format:?}, seed={seed}, row={row}, column={column}, scalar={scalar}, block={block_fp16}");
                }
            }
        }
    }
}

/// Independently decode scalar coefficients and accumulate one reported
/// coordinate in F64. This checks the full block-decode/F32 oracle at precisely
/// the failing coordinate without repeating a full dense matrix product.
pub(super) fn scalar_dot(shape: Shape, bytes: &[u8], input_row: &[f16], column: usize) -> f64 {
    let width = shape.input as usize;
    assert_eq!(input_row.len(), width);
    assert!(column < shape.output as usize);
    let block_values = shape.format.block_values();
    let block_bytes = shape.format.block_bytes();
    let row_bytes = width / block_values * block_bytes;
    let weights = &bytes[column * row_bytes..(column + 1) * row_bytes];
    input_row
        .iter()
        .enumerate()
        .fold(0.0, |sum, (index, value)| {
            let start = index / block_values * block_bytes;
            sum + f64::from(value.to_f32())
                * f64::from(
                    shape
                        .format
                        .decode_value(&weights[start..start + block_bytes], index % block_values),
                )
        })
}

pub(super) fn gate_up(gate: &[u8], up: &[u8], input: &[f16]) -> GateUp {
    let gate = project(gate_shape(), gate, input);
    let up = project(gate_shape(), up, input);
    let mut projected = Vec::with_capacity(4 * PACKED);
    let mut activated = Vec::with_capacity(4 * INTERMEDIATE);
    for row in 0..4 {
        projected.extend_from_slice(&gate[row * INTERMEDIATE..(row + 1) * INTERMEDIATE]);
        projected.extend_from_slice(&up[row * INTERMEDIATE..(row + 1) * INTERMEDIATE]);
        for column in 0..INTERMEDIATE {
            let gate = gate[row * INTERMEDIATE + column].to_f32();
            let up = up[row * INTERMEDIATE + column].to_f32();
            activated.push(f16::from_f32(gate / (1.0 + (-gate).exp()) * up));
        }
    }
    assert!(activated.iter().all(|value| value.is_finite()));
    GateUp {
        projected,
        activated,
    }
}
