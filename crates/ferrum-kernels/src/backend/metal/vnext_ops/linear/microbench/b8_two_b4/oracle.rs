//! Independent encoded-coefficient FP64 reference; no GPU or production dot.
use super::*;

fn coefficient_block(format: GgufBlockFormat, bytes: &[u8]) -> Vec<f64> {
    let half = |offset| f16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()).to_f64();
    match format {
        GgufBlockFormat::Q4K | GgufBlockFormat::Q6K => {
            let mut decoded = [0.0; 256];
            q4_b4_ffn_cooperative::oracle64::decode_block(format, bytes, &mut decoded);
            decoded.to_vec()
        }
        GgufBlockFormat::Q5K => {
            let d = half(0);
            let minimum = half(2);
            let scales = &bytes[4..16];
            (0..256)
                .map(|index| {
                    let group = index / 32;
                    let (scale, min) = if group < 4 {
                        (scales[group] & 63, scales[group + 4] & 63)
                    } else {
                        (
                            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                        )
                    };
                    let low = (bytes[48 + group / 2 * 32 + index % 32] >> (4 * (group % 2))) & 15;
                    let high = (bytes[16 + index % 32] >> group) & 1;
                    d * f64::from(scale) * f64::from(low | (high << 4)) - minimum * f64::from(min)
                })
                .collect()
        }
        GgufBlockFormat::Q8_0 => (0..32)
            .map(|i| half(0) * f64::from(bytes[2 + i] as i8))
            .collect(),
        _ => unreachable!("declared B8 formats only"),
    }
}

pub(in super::super) fn project(
    projection: &Projection,
    input: &[f16],
    half_coefficients: bool,
) -> Vec<f16> {
    let shape = projection.shape;
    let width = shape.input as usize;
    assert_eq!(input.len(), ROWS * width);
    let block_len = shape.format.block_values();
    let row_bytes = width / block_len * shape.format.block_bytes();
    let mut result = vec![f16::NAN; ROWS * shape.output as usize];
    for (column, encoded_row) in projection.bytes.chunks_exact(row_bytes).enumerate() {
        let mut sums = [0.0; ROWS];
        for (block, bytes) in encoded_row
            .chunks_exact(shape.format.block_bytes())
            .enumerate()
        {
            let mut weights = coefficient_block(shape.format, bytes);
            if half_coefficients {
                for w in &mut weights {
                    *w = f16::from_f64(*w).to_f64();
                }
            }
            for (row, sum) in sums.iter_mut().enumerate() {
                let begin = row * width + block * block_len;
                for (&x, &w) in input[begin..begin + block_len].iter().zip(&weights) {
                    *sum += x.to_f64() * w;
                }
            }
        }
        for (row, sum) in sums.into_iter().enumerate() {
            result[row * shape.output as usize + column] = f16::from_f64(sum);
        }
    }
    assert!(result.iter().all(|v| v.is_finite()));
    result
}

pub(super) fn activate(gate: &[f16], up: &[f16]) -> Vec<f16> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            let g = g.to_f64();
            f16::from_f64(g / (1.0 + (-g).exp()) * u.to_f64())
        })
        .collect()
}

#[test]
fn b8_two_b4_independent_coefficients_match_gguf_encoded_values() {
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q8_0,
    ] {
        for bytes in oracle_blocks(format).chunks_exact(format.block_bytes()) {
            let independent = coefficient_block(format, bytes);
            let mut shared = vec![0.0; format.block_values()];
            format.decode_block(bytes, &mut shared);
            for (a, b) in independent.into_iter().zip(shared) {
                assert!((a - f64::from(b)).abs() <= 1.0e-6, "{format:?}: {a} vs {b}");
            }
        }
    }
}
