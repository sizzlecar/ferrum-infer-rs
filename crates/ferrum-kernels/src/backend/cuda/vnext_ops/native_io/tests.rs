use super::*;
use crate::gguf_blocks::{
    fixtures::{oracle_blocks, FORMATS},
    GgufBlockFormat,
};
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::WeightId;
use half::f16;
use weights::{MatrixFormat, MatrixPart};

fn part(format: MatrixFormat, rows: u32, columns: u32) -> MatrixPart {
    MatrixPart {
        component_id: WeightId::new("component.token-matrix").unwrap(),
        format,
        rows,
        columns,
        output_offset: 0,
    }
}

#[test]
fn token_io_bounds_preserve_nonzero_spans_and_native_launch_capacity() {
    assert_eq!(last_token(7..11).unwrap(), 10);
    assert_eq!(last_token(u64::MAX - 1..u64::MAX).unwrap(), u64::MAX - 1);
    assert!(last_token(7..7).is_err());
    assert!(last_token(std::ops::Range { start: 8, end: 7 }).is_err());
    for width in [1, 33, 4096, 65536, u32::MAX] {
        let table = part(MatrixFormat::DenseF16, 7, width);
        let limit = embedding_chunk_limit(&table).unwrap();
        assert!(limit > 0 && limit <= MAXIMUM_TOKENS_PER_LAUNCH);
        assert!(limit * u64::from(width) <= u64::from(u32::MAX));
        if limit < MAXIMUM_TOKENS_PER_LAUNCH {
            assert!((limit + 1) * u64::from(width) > u64::from(u32::MAX));
        }
    }
    for table in [
        part(MatrixFormat::DenseF16, 1, 0),
        part(MatrixFormat::DenseF16, 0, 512),
        part(MatrixFormat::Block(GgufBlockFormat::Q4K), 1, 257),
        MatrixPart {
            output_offset: 1,
            ..part(MatrixFormat::DenseF16, 1, 512)
        },
    ] {
        assert!(embedding_chunk_limit(&table).is_err());
    }
}

fn matrix(format: MatrixFormat, rows: usize) -> (Vec<u8>, Vec<f32>, usize) {
    match format {
        MatrixFormat::DenseF16 => {
            let width = 33;
            let values = (0..rows * width)
                .map(|i| f16::from_f32(((i * 13 % 31) as f32 - 15.0) / 16.0))
                .collect::<Vec<_>>();
            (
                values
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect(),
                values.iter().map(|value| value.to_f32()).collect(),
                width,
            )
        }
        MatrixFormat::Block(format) => {
            let width = 3 * format.block_values();
            let bytes = oracle_blocks(format)
                .chunks_exact(format.block_bytes())
                .cycle()
                .take(rows * 3)
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let mut values = vec![0.0; rows * width];
            format.decode(&bytes, &mut values).unwrap();
            (bytes, values, width)
        }
    }
}

fn scalar_bytes(value: f32, precision: TokenPrecision) -> Vec<u8> {
    match precision {
        TokenPrecision::F16 => f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
        TokenPrecision::F32 => value.to_le_bytes().to_vec(),
    }
}

fn scalar_value(bytes: &[u8], precision: TokenPrecision) -> f32 {
    match precision {
        TokenPrecision::F16 => {
            f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
        }
        TokenPrecision::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn token_lookup_launcher_preserves_exact_decode_offsets_and_invalid_ids_on_cuda() {
    let context = CudaContext::new(0).expect("native token I/O conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for format in std::iter::once(MatrixFormat::DenseF16).chain(FORMATS.map(MatrixFormat::Block)) {
        let (bytes, decoded, width) = matrix(format, 5);
        let mut padded = vec![0xAB_u8; 16];
        padded.extend_from_slice(&bytes);
        let weight = stream.clone_htod(&padded).unwrap();
        let tokens = [0xBAD_u32, 3, 0, 4, 3, 5, u32::MAX, 0xBAD];
        let token_gpu = stream.clone_htod(&tokens).unwrap();
        let table = part(format, 5, width as u32);
        let (weight_ptr, _weight_guard) = weight.device_ptr(&stream);
        let (token_ptr, _token_guard) = token_gpu.device_ptr(&stream);
        for precision in [TokenPrecision::F16, TokenPrecision::F32] {
            let element_bytes = precision.element().size_bytes() as usize;
            let count = 6 * width * element_bytes;
            let mut output = stream.clone_htod(&vec![0xCD_u8; count + 32]).unwrap();
            let (output_ptr, guard) = output.device_ptr_mut(&stream);
            kernels
                .embedding(
                    &stream,
                    token_ptr + 4,
                    weight_ptr + 16,
                    output_ptr + 16,
                    &table,
                    6,
                    precision.element(),
                )
                .unwrap();
            drop(guard);
            let actual = stream.clone_dtoh(&output).unwrap();
            assert!(actual[..16]
                .iter()
                .chain(&actual[count + 16..])
                .all(|&byte| byte == 0xCD));
            for (row, &id) in tokens[1..7].iter().enumerate() {
                for col in 0..width {
                    let offset = 16 + (row * width + col) * element_bytes;
                    let bytes = &actual[offset..offset + element_bytes];
                    if id >= 5 {
                        assert!(scalar_value(bytes, precision).is_nan());
                    } else {
                        assert_eq!(
                            bytes,
                            scalar_bytes(decoded[id as usize * width + col], precision),
                            "{format:?} token {id} column {col}"
                        );
                    }
                }
            }
            assert_eq!(stream.clone_dtoh(&token_gpu).unwrap(), tokens);
            assert_eq!(stream.clone_dtoh(&weight).unwrap(), padded);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn final_row_projection_launcher_preserves_f32_input_and_guarded_logits_on_cuda() {
    let context = CudaContext::new(0).expect("native token I/O conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for format in std::iter::once(MatrixFormat::DenseF16).chain(FORMATS.map(MatrixFormat::Block)) {
        let (bytes, decoded, width) = matrix(format, 7);
        let weight = stream.clone_htod(&bytes).unwrap();
        let (weight_ptr, _weight_guard) = weight.device_ptr(&stream);
        let table = part(format, 7, width as u32);
        for precision in [TokenPrecision::F16, TokenPrecision::F32] {
            let element_bytes = precision.element().size_bytes() as usize;
            let input = (0..5 * width)
                .flat_map(|i| {
                    scalar_bytes(
                        if i / width == 3 {
                            1.0001 + (i % 7) as f32 * 0.00003125
                        } else {
                            -90.0 - (i / width) as f32
                        },
                        precision,
                    )
                })
                .collect::<Vec<_>>();
            let x = stream.clone_htod(&input).unwrap();
            let mut y = stream
                .clone_htod(&vec![0xCD_u8; 32 + 7 * element_bytes])
                .unwrap();
            let (xp, _x_guard) = x.device_ptr(&stream);
            let (yp, y_guard) = y.device_ptr_mut(&stream);
            let selected = last_token(1..4).unwrap();
            let xp = checked_pointer_offset(
                xp,
                selected * width as u64,
                element_bytes as u64,
                "test selected row",
            )
            .unwrap();
            kernels
                .linear(
                    &stream,
                    xp,
                    weight_ptr,
                    yp + 16,
                    &table,
                    1,
                    7,
                    precision.element(),
                )
                .unwrap();
            drop(y_guard);
            let actual = stream.clone_dtoh(&y).unwrap();
            assert!(actual[..16]
                .iter()
                .chain(&actual[actual.len() - 16..])
                .all(|&byte| byte == 0xCD));
            let row = &input[3 * width * element_bytes..4 * width * element_bytes];
            for (column, bytes) in actual[16..actual.len() - 16]
                .chunks_exact(element_bytes)
                .enumerate()
            {
                let products = row
                    .chunks_exact(element_bytes)
                    .enumerate()
                    .map(|(k, bytes)| {
                        f64::from(scalar_value(bytes, precision))
                            * f64::from(decoded[column * width + k])
                    });
                let reference = products.clone().sum::<f64>();
                let rounding = if precision == TokenPrecision::F16 {
                    0.0009765625
                } else {
                    f32::EPSILON as f64
                };
                let bound = (width as f64 * f32::EPSILON as f64 + rounding)
                    * products.map(f64::abs).sum::<f64>()
                    + 1e-6;
                let actual = f64::from(scalar_value(bytes, precision));
                assert!(
                    actual.is_finite() && (actual - reference).abs() <= bound,
                    "{format:?} column {column}: actual {actual}, F64 {reference}, bound {bound}"
                );
            }
            assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
        }
    }
}
