use super::*;
use crate::gguf_blocks::{
    fixtures::{oracle_blocks, FORMATS},
    GgufBlockFormat,
};
use cudarc::driver::{CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
use half::f16;

#[test]
#[ignore = "requires an actual CUDA device"]
fn native_block_decoding_matches_shared_ggml_oracle_on_cuda() {
    let context = CudaContext::new(0).expect("native block conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for format in FORMATS {
        let bytes = oracle_blocks(format);
        let count = bytes.len() / format.block_bytes() * format.block_values();
        let mut expected = vec![-12345.0_f32; count + 8];
        format.decode(&bytes, &mut expected[4..4 + count]).unwrap();
        let mut padded = vec![0xcc_u8; 16];
        padded.extend_from_slice(&bytes);
        let input = stream.clone_htod(&padded).unwrap();
        let mut output = stream.clone_htod(&vec![-12345.0_f32; count + 8]).unwrap();
        let input = input.slice(16..);
        let mut region = output.slice_mut(4..);
        let params = [
            count as u32,
            format.ggml_type_id(),
            format.block_values() as u32,
            format.block_bytes() as u32,
        ];
        let mut launch = stream.launch_builder(&kernels.decode);
        launch.arg(&input).arg(&mut region);
        for param in &params {
            launch.arg(param);
        }
        // SAFETY: The kernel guards the logical count; both views retain their
        // allocations and satisfy the recorded native block layout and extent.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32 + 17)) }.unwrap();
        let actual = stream.clone_dtoh(&output).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{format:?}[{index}]");
        }
    }
}

trait Scalar: DeviceRepr + ValidAsZeroBits + Copy + std::fmt::Debug {
    const ROUNDING: f64;
    fn from_f32(value: f32) -> Self;
    fn as_f32(self) -> f32;
}
impl Scalar for f32 {
    const ROUNDING: f64 = f32::EPSILON as f64;
    fn from_f32(value: f32) -> Self {
        value
    }
    fn as_f32(self) -> f32 {
        self
    }
}
impl Scalar for f16 {
    const ROUNDING: f64 = 0.0009765625;
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
    fn as_f32(self) -> f32 {
        self.to_f32()
    }
}

fn matrix(format: GgufBlockFormat, rows: usize, blocks_per_row: usize) -> (Vec<u8>, Vec<f32>) {
    let source = oracle_blocks(format);
    let bytes = source
        .chunks_exact(format.block_bytes())
        .cycle()
        .take(rows * blocks_per_row)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let mut decoded = vec![0.0; rows * blocks_per_row * format.block_values()];
    format.decode(&bytes, &mut decoded).unwrap();
    (bytes, decoded)
}

fn linear<T: Scalar>(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    row_tile: u32,
) -> Vec<Vec<u32>> {
    let mut observations = Vec::new();
    for format in FORMATS {
        for rows in [1_usize, 3, 8, 11] {
            let inputs = 3 * format.block_values();
            let outputs = 7_usize;
            let stride = outputs + 5;
            let offset = 2_usize;
            let (bytes, weight) = matrix(format, outputs, 3);
            let input = (0..rows * inputs)
                .map(|i| T::from_f32(((i * 11 % 37) as f32 - 18.0) / 128.0))
                .collect::<Vec<_>>();
            let input_gpu = stream.clone_htod(&input).unwrap();
            let weight_gpu = stream.clone_htod(&bytes).unwrap();
            let canary = T::from_f32(-12345.0);
            let mut output = stream.clone_htod(&vec![canary; rows * stride + 8]).unwrap();
            let params = [
                rows as u32,
                inputs as u32,
                outputs as u32,
                stride as u32,
                offset as u32,
                format.ggml_type_id(),
                format.block_values() as u32,
                format.block_bytes() as u32,
            ];
            let mut launch = stream.launch_builder(kernel);
            launch.arg(&input_gpu).arg(&weight_gpu).arg(&mut output);
            for param in &params {
                launch.arg(param);
            }
            // SAFETY: Four complete warps per block, contiguous input/weight
            // rows, and the output column interval lies inside every stride.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (outputs as u32).div_ceil(4),
                        (rows as u32).div_ceil(row_tile),
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let actual = stream.clone_dtoh(&output).unwrap();
            for value in &actual[rows * stride..] {
                assert_eq!(value.as_f32().to_bits(), canary.as_f32().to_bits());
            }
            for row in 0..rows {
                for column in 0..stride {
                    let actual = actual[row * stride + column].as_f32();
                    if !(offset..offset + outputs).contains(&column) {
                        assert_eq!(actual.to_bits(), canary.as_f32().to_bits());
                        continue;
                    }
                    let weights = &weight[(column - offset) * inputs..][..inputs];
                    let products = input[row * inputs..][..inputs]
                        .iter()
                        .zip(weights)
                        .map(|(x, w)| f64::from(x.as_f32()) * f64::from(*w));
                    let sum = products.clone().sum::<f64>();
                    let sum_abs = products.map(f64::abs).sum::<f64>();
                    let bound =
                        (inputs as f64 * f32::EPSILON as f64 + T::ROUNDING) * sum_abs + 1e-6;
                    assert!(actual.is_finite() && (f64::from(actual) - sum).abs() <= bound,
                        "{format:?} {rows}x{inputs}x{outputs} row {row} col {column}: {actual}, F64 {sum}, bound {bound}");
                }
            }
            observations.push(
                actual
                    .iter()
                    .map(|value| value.as_f32().to_bits())
                    .collect(),
            );
        }
    }
    observations
}

fn embedding<T: Scalar>(stream: &Arc<CudaStream>, kernel: &CudaFunction) {
    for format in FORMATS {
        let width = 3 * format.block_values();
        let vocabulary = 5_usize;
        let (bytes, weight) = matrix(format, vocabulary, 3);
        let tokens = [3_u32, 0, 4, 3];
        let token_gpu = stream.clone_htod(&tokens).unwrap();
        let weight_gpu = stream.clone_htod(&bytes).unwrap();
        let canary = T::from_f32(-12345.0);
        let count = tokens.len() * width;
        let mut output = stream.clone_htod(&vec![canary; count + 8]).unwrap();
        let mut region = output.slice_mut(4..);
        let params = [
            tokens.len() as u32,
            width as u32,
            vocabulary as u32,
            format.ggml_type_id(),
            format.block_values() as u32,
            format.block_bytes() as u32,
        ];
        let mut launch = stream.launch_builder(kernel);
        launch.arg(&token_gpu).arg(&weight_gpu).arg(&mut region);
        for param in &params {
            launch.arg(param);
        }
        // SAFETY: All token IDs are in range and the guarded launch writes
        // exactly count elements into a view with four trailing guard values.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32 + 17)) }.unwrap();
        let actual = stream.clone_dtoh(&output).unwrap();
        let expected = std::iter::repeat_n(canary, 4)
            .chain(tokens.into_iter().flat_map(|token| {
                weight[token as usize * width..][..width]
                    .iter()
                    .map(|value| T::from_f32(*value))
            }))
            .chain(std::iter::repeat_n(canary, 4));
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(
                actual.as_f32().to_bits(),
                expected.as_f32().to_bits(),
                "{format:?}"
            );
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn native_block_linears_preserve_f16_and_f32_activations_on_cuda() {
    let context = CudaContext::new(0).expect("native block conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // Both paths first satisfy the independent decoded-byte/F64 oracle. Tiling
    // must additionally preserve the existing per-row accumulation bits.
    assert_eq!(
        linear::<f16>(&stream, &kernels.linear_f16, 1),
        linear::<f16>(&stream, &kernels.linear_tiled_f16, LINEAR_ROW_TILE)
    );
    assert_eq!(
        linear::<f32>(&stream, &kernels.linear_f32, 1),
        linear::<f32>(&stream, &kernels.linear_tiled_f32, LINEAR_ROW_TILE)
    );
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn native_block_embeddings_preserve_f16_and_f32_activations_on_cuda() {
    let context = CudaContext::new(0).expect("native block conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    embedding::<f16>(&stream, &kernels.embedding_f16);
    embedding::<f32>(&stream, &kernels.embedding_f32);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn native_matrix_launcher_preserves_mixed_dense_and_block_partitions() {
    use ferrum_interfaces::vnext::ElementType;
    mixed_matrix::<f16>(ElementType::F16);
    mixed_matrix::<f32>(ElementType::F32);
}

fn mixed_matrix<T: Scalar>(activation: ferrum_interfaces::vnext::ElementType) {
    for rows in [1, 3, 8, 11] {
        mixed_matrix_rows::<T>(activation, rows);
    }
}

fn mixed_matrix_rows<T: Scalar>(activation: ferrum_interfaces::vnext::ElementType, rows: usize) {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use ferrum_interfaces::vnext::WeightId;
    use weights::{MatrixFormat, MatrixPart};
    let context = CudaContext::new(0).expect("native matrix conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    let columns = 512_usize;
    let outputs = 3_usize;
    let stride = 2 * outputs + 2;
    let (blocks, decoded) = matrix(GgufBlockFormat::Iq4Xs, outputs, 2);
    let dense = (0..outputs * columns)
        .map(|i| f16::from_f32(((i * 7 % 41) as f32 - 20.0) / 512.0))
        .collect::<Vec<_>>();
    let input = (0..rows * columns)
        .map(|i| T::from_f32(((i * 13 % 29) as f32 - 14.0) / 256.0))
        .collect::<Vec<_>>();
    let x = stream.clone_htod(&input).unwrap();
    let q = stream.clone_htod(&blocks).unwrap();
    let d = stream.clone_htod(&dense).unwrap();
    let canary = T::from_f32(-12345.0);
    let mut output = stream.clone_htod(&vec![canary; rows * stride]).unwrap();
    let (xp, _x_guard) = x.device_ptr(&stream);
    let (qp, _q_guard) = q.device_ptr(&stream);
    let (dp, _d_guard) = d.device_ptr(&stream);
    let (yp, y_guard) = output.device_ptr_mut(&stream);
    for (format, pointer, offset) in [
        (MatrixFormat::Block(GgufBlockFormat::Iq4Xs), qp, 1),
        (MatrixFormat::DenseF16, dp, 1 + outputs as u32),
    ] {
        let part = MatrixPart {
            component_id: WeightId::new("component.matrix").unwrap(),
            format,
            rows: outputs as u32,
            columns: columns as u32,
            output_offset: offset,
        };
        kernels
            .linear(
                &stream,
                xp,
                pointer,
                yp,
                &part,
                rows as u32,
                stride as u32,
                activation,
            )
            .unwrap();
    }
    drop(y_guard);
    let actual = stream.clone_dtoh(&output).unwrap();
    for row in 0..rows {
        assert_eq!(actual[row * stride].as_f32(), canary.as_f32());
        assert_eq!(actual[row * stride + stride - 1].as_f32(), canary.as_f32());
        for col in 0..2 * outputs {
            let products = (0..columns).map(|k| {
                let weight = if col < outputs {
                    decoded[col * columns + k]
                } else {
                    dense[(col - outputs) * columns + k].to_f32()
                };
                f64::from(input[row * columns + k].as_f32()) * f64::from(weight)
            });
            let sum = products.clone().sum::<f64>();
            let bound = (columns as f64 * f32::EPSILON as f64 + T::ROUNDING)
                * products.map(f64::abs).sum::<f64>()
                + 1e-6;
            let value = actual[row * stride + col + 1].as_f32();
            assert!(
                value.is_finite() && (f64::from(value) - sum).abs() <= bound,
                "row {row} col {col}: actual {value}, F64 {sum}, bound {bound}"
            );
        }
    }
}
