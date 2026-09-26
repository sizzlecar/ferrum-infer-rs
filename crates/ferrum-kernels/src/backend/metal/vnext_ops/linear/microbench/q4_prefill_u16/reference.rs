//! Independent complete-output diagnostic for the tiled operand contract:
//! encoded coefficients -> F16 -> F64 dot -> F16 writeback at each stage.
//! This is not the cooperative GEMV's unrounded-coefficient oracle.
use super::*;
use q4_b4_ffn_cooperative::oracle64::decode_block;

pub(super) fn dense_input(rows: usize, width: usize) -> Vec<f16> {
    (0..rows * width)
        .map(|index| {
            let phase = (index + 1) as f32 * 0.017 + (index / width) as f32 * 0.113;
            let value = f16::from_f32(phase.sin() * 0.03125 + (phase * 0.37).cos() * 0.0078125);
            if value == f16::ZERO {
                f16::MIN_POSITIVE
            } else {
                value
            }
        })
        .collect()
}

pub(super) fn half_coefficients(format: GgufBlockFormat, block: &[u8]) -> [f16; 256] {
    let mut values = [0.0_f64; 256];
    decode_block(format, block, &mut values);
    values.map(f16::from_f64)
}

pub(super) fn project(shape: Shape, bytes: &[u8], input: &[f16]) -> Vec<f16> {
    let width = shape.input as usize;
    assert_eq!(input.len() % width, 0);
    let rows = input.len() / width;
    let outputs = shape.output as usize;
    let row_bytes = width / 256 * shape.format.block_bytes();
    assert_eq!(bytes.len(), outputs * row_bytes);
    // Convert input once, not inside billions of coefficient products.
    let input: Vec<_> = input.iter().map(|v| v.to_f64()).collect();
    let mut weight = vec![0.0_f64; width];
    let mut result = vec![f16::NAN; rows * outputs];
    for (column, encoded) in bytes.chunks_exact(row_bytes).enumerate() {
        for (block_index, block) in encoded.chunks_exact(shape.format.block_bytes()).enumerate() {
            for (offset, value) in half_coefficients(shape.format, block)
                .into_iter()
                .enumerate()
            {
                weight[block_index * 256 + offset] = value.to_f64();
            }
        }
        for (row, values) in input.chunks_exact(width).enumerate() {
            let mut sum = 0.0_f64;
            for (&a, &w) in values.iter().zip(&weight) {
                sum += a * w;
            }
            result[row * outputs + column] = f16::from_f64(sum);
        }
    }
    assert!(result.iter().all(|v| v.is_finite()));
    result
}

pub(super) fn activate(gate: &[f16], up: &[f16]) -> Vec<f16> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            let gate = gate.to_f64();
            f16::from_f64(gate / (1.0 + (-gate).exp()) * up.to_f64())
        })
        .collect()
}

#[test]
fn q4_prefill_reference_rounds_coefficients_before_the_dot() {
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let shape = Shape {
            name: "half_operand_oracle",
            input: 512,
            output: 3,
            format,
        };
        let bytes = matrix(shape, 11);
        let input = dense_input(5, 512);
        let actual = project(shape, &bytes, &input);
        let row_bytes = 2 * format.block_bytes();
        for row in 0..5 {
            for column in 0..3 {
                let mut sum = 0.0_f64;
                for k in 0..512 {
                    let start = column * row_bytes + k / 256 * format.block_bytes();
                    // Independent scalar coefficient indexing, not decode_block.
                    let w = f16::from_f32(
                        format.decode_value(&bytes[start..start + format.block_bytes()], k % 256),
                    );
                    sum += input[row * 512 + k].to_f64() * w.to_f64();
                }
                assert_eq!(
                    actual[row * 3 + column].to_bits(),
                    f16::from_f64(sum).to_bits()
                );
            }
        }
    }
}
