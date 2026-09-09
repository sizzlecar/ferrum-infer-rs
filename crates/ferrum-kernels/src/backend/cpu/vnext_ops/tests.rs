use super::matrix::{CpuMatrix, CpuMatrixFormat};
use super::scalar::CpuFloat;
use crate::gguf_blocks::{
    fixtures::{oracle_blocks, FORMATS},
    GgufBlockFormat,
};
use ferrum_interfaces::vnext::{BlockQuantizationSpec, ElementType, WeightEncoding};
use half::f16;

pub(super) fn encoded(values: &[f32], dtype: CpuFloat) -> Vec<u8> {
    let mut bytes = vec![0; values.len() * dtype.bytes()];
    for (index, &value) in values.iter().enumerate() {
        dtype.write(&mut bytes, index, value);
    }
    bytes
}

#[test]
fn residual_aliases_preserve_the_declared_precision_without_a_copy() {
    use super::elementwise::{residual_add, residual_add_in_place};
    let left = [0.1234567, -2.000123, 3.5, 0.0];
    let right = encoded(&[0.0625, 1.0, -3.5, 0.03125], CpuFloat::F16);
    for dtype in [CpuFloat::F16, CpuFloat::F32] {
        let original = encoded(&left, dtype);
        let mut expected = vec![0; original.len()];
        residual_add(&original, dtype, &right, &mut expected).unwrap();
        let mut aliased = original.clone();
        residual_add_in_place(&mut aliased, dtype, Some(&right)).unwrap();
        assert_eq!(aliased, expected);
        let unchanged = aliased.clone();
        assert!(residual_add_in_place(&mut aliased, dtype, Some(&right[..6])).is_err());
        assert_eq!(aliased, unchanged);
    }
    let mut doubled = encoded(&left, CpuFloat::F16);
    let mut expected = vec![0; doubled.len()];
    residual_add(&doubled, CpuFloat::F16, &doubled, &mut expected).unwrap();
    residual_add_in_place(&mut doubled, CpuFloat::F16, None).unwrap();
    assert_eq!(doubled, expected);
}

#[test]
fn compressed_cpu_linear_and_embedding_match_decoded_weights() {
    for format in FORMATS {
        let first = oracle_blocks(format);
        let columns = first.len() / format.block_bytes() * format.block_values();
        let mut bytes = first.clone();
        bytes.extend(
            first
                .chunks_exact(format.block_bytes())
                .rev()
                .flatten()
                .copied(),
        );
        let encoding = WeightEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: format.format_id().to_owned().try_into().unwrap(),
            logical_values_per_block: format.block_values() as u32,
            bytes_per_block: format.block_bytes() as u32,
        });
        let matrix = CpuMatrix::new(
            &bytes,
            CpuMatrixFormat::from_encoding(&encoding).unwrap(),
            2,
            columns,
        )
        .unwrap();
        let mut decoded = vec![0.0; columns * 2];
        format.decode(&bytes, &mut decoded).unwrap();
        for dtype in [CpuFloat::F16, CpuFloat::F32] {
            let input_values: Vec<f32> = (0..columns * 3)
                .map(|i| (i as f32 * 0.073).sin() * 0.03125 + 0.0000123)
                .collect();
            let input = encoded(&input_values, dtype);
            let mut output = encoded(&vec![-123.0; 15], dtype);
            matrix
                .linear(&input, dtype, &mut output, dtype, 3, 5, 1)
                .unwrap();
            for row in 0..3 {
                for column in 0..5 {
                    let actual = dtype.read(&output, row * 5 + column);
                    if !(1..3).contains(&column) {
                        assert_eq!(actual, -123.0);
                        continue;
                    }
                    let input_row =
                        &input[row * columns * dtype.bytes()..(row + 1) * columns * dtype.bytes()];
                    let products = (0..columns).map(|i| {
                        f64::from(dtype.read(input_row, i))
                            * f64::from(decoded[(column - 1) * columns + i])
                    });
                    let expected = products.clone().sum::<f64>();
                    let accumulation = columns as f64
                        * f64::from(f32::EPSILON)
                        * products.map(f64::abs).sum::<f64>();
                    let rounding = if dtype == CpuFloat::F16 {
                        expected.abs() / 1024.0 + 2_f64.powi(-24)
                    } else {
                        0.0
                    };
                    assert!(
                        actual.is_finite()
                            && (f64::from(actual) - expected).abs() <= accumulation + rounding,
                        "{format:?} {dtype:?} ({row},{column}): {actual} != {expected}"
                    );
                }
            }
            let mut embeddings = vec![0_u8; 3 * columns * dtype.bytes()];
            let ids: Vec<u8> = [1_u32, 0, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            matrix.embedding(&ids, &mut embeddings, dtype).unwrap();
            for (row, id) in [1, 0, 1].into_iter().enumerate() {
                for column in 0..columns {
                    let expected = decoded[id * columns + column];
                    let expected = if dtype == CpuFloat::F16 {
                        f16::from_f32(expected).to_f32()
                    } else {
                        expected
                    };
                    assert_eq!(
                        dtype.read(&embeddings, row * columns + column).to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_f32_matrix_does_not_narrow_activations_or_accumulation() {
    let mut block = [0x88_u8; 18];
    block[..2].copy_from_slice(&f16::ONE.to_le_bytes());
    let matrix = CpuMatrix::new(
        &block,
        CpuMatrixFormat::Native(GgufBlockFormat::Iq4Nl),
        1,
        32,
    )
    .unwrap();
    let input = encoded(&[1.0002; 32], CpuFloat::F32);
    let mut output = [0; 4];
    matrix
        .linear(&input, CpuFloat::F32, &mut output, CpuFloat::F32, 1, 1, 0)
        .unwrap();
    assert!((CpuFloat::F32.read(&output, 0) - 32.0064).abs() < 0.00004);
}

#[test]
fn cpu_matrix_checks_physical_ranges_before_writing() {
    let bytes = encoded(&[1.0, 2.0, 3.0, 4.0], CpuFloat::F16);
    let dense = CpuMatrixFormat::from_encoding(&WeightEncoding::Dense {
        element_type: ElementType::F16,
    })
    .unwrap();
    let matrix = CpuMatrix::new(&bytes, dense, 2, 2).unwrap();
    for (rows, columns) in [(0, 2), (2, 0), (usize::MAX, 2), (2, 3)] {
        assert!(CpuMatrix::new(&bytes, dense, rows, columns).is_err());
    }
    assert!(CpuMatrix::new(
        &[0; 18],
        CpuMatrixFormat::Native(GgufBlockFormat::Iq4Nl),
        1,
        31
    )
    .is_err());
    let mut output = [0x55; 8];
    for (input, stride, offset) in [
        (&bytes[..2], 2, 0),
        (&bytes[..4], 1, 0),
        (&bytes[..4], 4, 3),
    ] {
        assert!(matrix
            .linear(
                input,
                CpuFloat::F16,
                &mut output,
                CpuFloat::F16,
                1,
                stride,
                offset
            )
            .is_err());
        assert_eq!(output, [0x55; 8]);
    }
    assert!(matrix
        .embedding(&2_u32.to_le_bytes(), &mut output[..4], CpuFloat::F16)
        .is_err());
    assert_eq!(output, [0x55; 8]);
}

#[test]
fn cpu_elementwise_preserves_declared_float_boundaries() {
    use super::elementwise::{residual_add, rms_norm, swiglu};
    let weights = encoded(&[1.0, 0.5, 2.0], CpuFloat::F16);
    for (input_type, output_type) in [
        (CpuFloat::F16, CpuFloat::F16),
        (CpuFloat::F32, CpuFloat::F16),
        (CpuFloat::F32, CpuFloat::F32),
    ] {
        let input = encoded(
            &[0.0002, -0.0031, 0.0073, 2.0002, -3.125, 4.03125],
            input_type,
        );
        let mut output = vec![0; 6 * output_type.bytes()];
        rms_norm(
            &input,
            input_type,
            &weights,
            &mut output,
            output_type,
            3,
            0.00001,
        )
        .unwrap();
        for row in 0..2 {
            let values: Vec<f64> = (0..3)
                .map(|i| f64::from(input_type.read(&input, row * 3 + i)))
                .collect();
            let scale = (values.iter().map(|x| x * x).sum::<f64>() / 3.0 + f64::from(0.00001_f32))
                .sqrt()
                .recip();
            for (column, value) in values.into_iter().enumerate() {
                let expected =
                    (value * scale * f64::from(CpuFloat::F16.read(&weights, column))) as f32;
                let expected = if output_type == CpuFloat::F16 {
                    f16::from_f32(expected).to_f32()
                } else {
                    expected
                };
                assert!((output_type.read(&output, row * 3 + column) - expected).abs() <= 2e-6);
            }
        }
    }
    let left = encoded(&[1.0002, -2.0003], CpuFloat::F32);
    let right = encoded(&[0.5, -0.25], CpuFloat::F16);
    let mut result = [0_u8; 8];
    residual_add(&left, CpuFloat::F32, &right, &mut result).unwrap();
    assert_eq!(CpuFloat::F32.read(&result, 0), 1.0002_f32 + 0.5);
    assert_eq!(CpuFloat::F32.read(&result, 1), -2.0003_f32 - 0.25);
    let gate_up = encoded(
        &[
            0.0, 1.0, -1.0, 10.0, 2.0, 3.0, -1000.0, 1000.0, 2.0, 1.0, 0.25, 4.0,
        ],
        CpuFloat::F16,
    );
    let mut output = [0_u8; 12];
    swiglu(&gate_up, &mut output, 3).unwrap();
    for row in 0..2 {
        for column in 0..3 {
            let gate = CpuFloat::F16.read(&gate_up, row * 6 + column);
            let up = CpuFloat::F16.read(&gate_up, row * 6 + column + 3);
            let expected = f16::from_f32(
                ((f64::from(gate) / (1.0 + (-f64::from(gate)).exp())) * f64::from(up)) as f32,
            )
            .to_f32();
            assert_eq!(CpuFloat::F16.read(&output, row * 3 + column), expected);
        }
    }
}

#[test]
fn cpu_masked_argmax_preserves_logits_and_applies_penalty_once() {
    use super::elementwise::masked_argmax;
    for dtype in [CpuFloat::F16, CpuFloat::F32] {
        let logits = encoded(&[10.0, 3.0, f32::NAN, f32::INFINITY], dtype);
        let original = logits.clone();
        let repeated: Vec<u8> = [0_u32, 0, 50]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let offsets: Vec<u8> = [0_u32, 30].into_iter().flat_map(u32::to_le_bytes).collect();
        let mut workspace = [0_u8; 4];
        assert_eq!(
            masked_argmax(
                &logits,
                dtype,
                &[1; 4],
                &repeated,
                &offsets,
                2.0,
                &mut workspace
            )
            .unwrap(),
            0
        );
        assert_eq!(
            masked_argmax(
                &logits,
                dtype,
                &[0, 1, 1, 1],
                &repeated,
                &offsets,
                2.0,
                &mut workspace
            )
            .unwrap(),
            1
        );
        assert_eq!(
            masked_argmax(
                &logits,
                dtype,
                &[0; 4],
                &repeated,
                &offsets,
                2.0,
                &mut workspace
            )
            .unwrap(),
            u32::MAX
        );
        assert_eq!(logits, original);
        let negative = encoded(&[-2.0, -3.0, -3.0, f32::NAN], dtype);
        assert_eq!(
            masked_argmax(
                &negative,
                dtype,
                &[1; 4],
                &repeated,
                &offsets,
                2.0,
                &mut workspace
            )
            .unwrap(),
            1
        );
        assert!(masked_argmax(
            &negative,
            dtype,
            &[1; 4],
            &repeated,
            &offsets,
            0.0,
            &mut workspace
        )
        .is_err());
    }
}
