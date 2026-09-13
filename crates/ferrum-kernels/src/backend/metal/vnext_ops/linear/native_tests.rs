use super::*;
use crate::gguf_blocks::fixtures::{oracle_blocks, FORMATS};
use half::f16;
use metal::{Buffer, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

fn buffer<T>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[test]
fn native_block_linears_preserve_rows_offsets_strides_and_precision_on_real_metal() {
    let device = Device::system_default().expect("native block linear conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for format in FORMATS {
        let blocks = oracle_blocks(format);
        let input_width = blocks.len() / format.block_bytes() * format.block_values();
        let output_width = 7;
        let output_stride = 11;
        let mut weight_bytes = vec![0x55_u8; 16];
        let block_rows: Vec<Vec<u8>> = (0..output_width)
            .map(|row| {
                let mut bytes = blocks.clone();
                bytes.rotate_left(
                    (row % (blocks.len() / format.block_bytes())) * format.block_bytes(),
                );
                bytes
            })
            .collect();
        let weights: Vec<Vec<f32>> = block_rows
            .iter()
            .map(|bytes| {
                let mut values = vec![0.0; input_width];
                format.decode(bytes, &mut values).unwrap();
                weight_bytes.extend_from_slice(bytes);
                values
            })
            .collect();
        let weight = buffer(&device, &weight_bytes);
        for rows in [1, 3, 9] {
            for activation_type in [ElementType::F16, ElementType::F32] {
                // F32 inputs deliberately contain values not representable
                // in F16, so an accidental narrowing changes this oracle.
                let input: Vec<f32> = (0..rows * input_width)
                    .map(|i| {
                        let value = (i as f32 * 0.071).sin() * 0.03125
                            + (i / input_width) as f32 * 0.0012345;
                        if activation_type == ElementType::F16 {
                            f16::from_f32(value).to_f32()
                        } else {
                            value
                        }
                    })
                    .collect();
                let element_bytes = activation_type.size_bytes() as usize;
                let prefix = 16 / element_bytes;
                let mut padded_input = vec![123.0; prefix];
                padded_input.extend_from_slice(&input);
                let input_buffer = if activation_type == ElementType::F16 {
                    buffer(
                        &device,
                        &padded_input
                            .iter()
                            .copied()
                            .map(f16::from_f32)
                            .collect::<Vec<_>>(),
                    )
                } else {
                    buffer(&device, &padded_input)
                };
                let total = prefix + rows * output_stride + prefix;
                let output = if activation_type == ElementType::F16 {
                    buffer(&device, &vec![f16::from_f32(-123.0); total])
                } else {
                    buffer(&device, &vec![-123.0_f32; total])
                };
                // Q5 F32 exercises the production selector added for its
                // formerly missing master-precision projection kernel.
                let physical =
                    if format == GgufBlockFormat::Q5K && activation_type == ElementType::F32 {
                        LinearPhysicalFormat::Q5K
                    } else {
                        LinearPhysicalFormat::Native(format)
                    };
                let params = LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: output_stride as u32,
                    output_column_offset: 2,
                };
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                let (pipeline, kind) = if activation_type == ElementType::F16 {
                    pipelines.linear_pipeline(physical, params.rows, params.out_features)
                } else {
                    (
                        pipelines.f32_linear_pipeline(physical).unwrap(),
                        LinearDispatchKind::CooperativeGemv,
                    )
                };
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(&input_buffer), 16);
                encoder.set_buffer(1, Some(&weight), 16);
                encoder.set_buffer(2, Some(&output), 16);
                bind_linear_params(encoder, params, physical, activation_type);
                dispatch_linear_grid(encoder, params, kind);
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(
                    command.status(),
                    MTLCommandBufferStatus::Completed,
                    "{format:?} {activation_type:?}"
                );
                // SAFETY: Shared, correctly sized aligned buffers; the GPU
                // completed all writes and the buffers remain owned here.
                let actual: Vec<f32> = if activation_type == ElementType::F16 {
                    unsafe { std::slice::from_raw_parts(output.contents().cast::<f16>(), total) }
                        .iter()
                        .map(|x| x.to_f32())
                        .collect()
                } else {
                    unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), total) }
                        .to_vec()
                };
                for (index, actual) in actual.into_iter().enumerate() {
                    let logical = index
                        .checked_sub(prefix)
                        .filter(|i| *i < rows * output_stride);
                    let location = logical.and_then(|i| {
                        (2..2 + output_width)
                            .contains(&(i % output_stride))
                            .then(|| (i / output_stride, i % output_stride - 2))
                    });
                    let Some((row, column)) = location else {
                        assert_eq!(actual, -123.0, "modified padding {format:?} at {index}");
                        continue;
                    };
                    let products = input[row * input_width..(row + 1) * input_width]
                        .iter()
                        .zip(&weights[column])
                        .map(|(&x, &w)| f64::from(x) * f64::from(w));
                    let expected = products.clone().sum::<f64>();
                    let absolute_sum = products.map(f64::abs).sum::<f64>();
                    // Each SIMD lane accumulates ceil(K/32) products, then
                    // a 32-lane reduction; include product rounding and the
                    // final F16 boundary where applicable.
                    let operations = input_width.div_ceil(32) + 8;
                    let accumulation = operations as f64 * f64::from(f32::EPSILON) * absolute_sum;
                    let rounding = if activation_type == ElementType::F16 {
                        expected.abs() / 1024.0 + 2_f64.powi(-24)
                    } else {
                        0.0
                    };
                    assert!(actual.is_finite() && (f64::from(actual) - expected).abs() <= accumulation + rounding, "{format:?} {activation_type:?} rows={rows} ({row},{column}): {actual} != {expected}; bound={}", accumulation + rounding);
                }
            }
        }
    }
}

fn assert_native_prefill(
    device: &Device,
    pipelines: &MetalLinearPipelines,
    queue: &CommandQueueRef,
    format: GgufBlockFormat,
    blocks: &[u8],
    rows: usize,
    columns: usize,
    input_value: fn(usize, usize) -> f16,
) {
    let width = blocks.len() / format.block_bytes() * format.block_values();
    let stride = columns + 5;
    let prefix = 8;
    let guard = f16::from_f32(-123.0);
    let mut input = vec![guard; prefix];
    for row in 0..rows {
        input.extend((0..width).map(|column| input_value(row, column)));
    }
    input.extend([guard; 8]);
    let mut encoded = vec![0x55_u8; 16];
    let mut decoded = Vec::with_capacity(columns);
    for column in 0..columns {
        let mut bytes = blocks.to_vec();
        bytes.rotate_left((column % (blocks.len() / format.block_bytes())) * format.block_bytes());
        let mut values = vec![0.0_f32; width];
        format.decode(&bytes, &mut values).unwrap();
        encoded.extend_from_slice(&bytes);
        decoded.push(values);
    }
    encoded.extend([0x55; 16]);
    let mut initial_output = vec![guard; prefix + rows * stride + 8];
    for row in 0..rows {
        initial_output[prefix + row * stride + 2..prefix + row * stride + 2 + columns]
            .fill(f16::NAN);
    }
    let input_buffer = buffer(device, &input);
    let weight_buffer = buffer(device, &encoded);
    let output = buffer(device, &initial_output);
    let params = LinearParams {
        rows: rows as u32,
        in_features: width as u32,
        out_features: columns as u32,
        output_stride: stride as u32,
        output_column_offset: 2,
    };
    let physical = LinearPhysicalFormat::Native(format);
    let (pipeline, kind) = pipelines.linear_pipeline(physical, params.rows, params.out_features);
    let expected = if rows >= 32 && columns >= 1024 {
        LinearDispatchKind::NativeTiledGemm
    } else {
        LinearDispatchKind::CooperativeGemv
    };
    assert_eq!(kind, expected);
    let (_, f32_kind) = pipelines
        .f32_linear_dispatch(physical, params.rows, params.out_features)
        .unwrap();
    assert!(matches!(f32_kind, LinearDispatchKind::CooperativeGemv));
    // Exercise partial output tiles independently of the performance guard.
    // Wide fixtures below still execute through the actual product selector.
    let (pipeline, kind) = if rows >= 32 && columns < 1024 {
        (
            &pipelines.native.gemm_f16_f32,
            LinearDispatchKind::NativeTiledGemm,
        )
    } else {
        (pipeline, kind)
    };
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 16);
    encoder.set_buffer(1, Some(&weight_buffer), 16);
    encoder.set_buffer(2, Some(&output), 16);
    bind_linear_params(encoder, params, physical, ElementType::F16);
    dispatch_linear_grid(encoder, params, kind);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

    // SAFETY: These shared buffers have the declared lengths and scalar types;
    // the completed command can no longer mutate them.
    let actual = unsafe {
        std::slice::from_raw_parts(output.contents().cast::<f16>(), initial_output.len())
    };
    let input_after =
        unsafe { std::slice::from_raw_parts(input_buffer.contents().cast::<f16>(), input.len()) };
    let weight_after =
        unsafe { std::slice::from_raw_parts(weight_buffer.contents().cast::<u8>(), encoded.len()) };
    assert_eq!(input_after, input, "input mutated: {format:?}");
    assert_eq!(weight_after, encoded, "weight mutated: {format:?}");
    for (index, value) in actual.iter().enumerate() {
        let location = index
            .checked_sub(prefix)
            .filter(|index| *index < rows * stride)
            .and_then(|index| {
                (2..columns + 2)
                    .contains(&(index % stride))
                    .then(|| (index / stride, index % stride - 2))
            });
        let Some((row, column)) = location else {
            assert_eq!(
                *value, guard,
                "output padding overwritten: {format:?} at {index}"
            );
            continue;
        };
        let products = input[prefix + row * width..prefix + (row + 1) * width]
            .iter()
            .zip(&decoded[column])
            .map(|(x, w)| f64::from(x.to_f32()) * f64::from(*w));
        let expected = products.clone().sum::<f64>();
        let absolute_sum = products.map(f64::abs).sum::<f64>();
        // Bound a full sequential F32 product/sum chain, which also bounds
        // regrouped SIMD MMA. Decoded weights remain F32: no F16 weight-cast
        // error is permitted. Account separately for the final F16 store.
        let n_u = (2 * width + 8) as f64 * f64::from(f32::EPSILON) / 2.0;
        let accumulation = n_u / (1.0 - n_u) * absolute_sum;
        let rounding = (expected.abs() + accumulation) * 2_f64.powi(-11) + 2_f64.powi(-25);
        assert!(
            value.is_finite()
                && (f64::from(value.to_f32()) - expected).abs() <= accumulation + rounding,
            "{format:?} rows={rows} ({row},{column}): {} != {expected}; bound={}",
            value.to_f32(),
            accumulation + rounding
        );
    }
}

#[test]
fn native_prefill_preserves_tile_tails_block_layouts_and_output_boundaries() {
    let device = Device::system_default().expect("native prefill conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for format in FORMATS {
        for (rows, columns) in [(31, 7), (32, 63), (33, 65), (65, 67), (33, 1025)] {
            assert_native_prefill(
                &device,
                &pipelines,
                &queue,
                format,
                &oracle_blocks(format),
                rows,
                columns,
                |row, column| f16::from_f32(((row * 17 + column) as f32 * 0.073).sin() * 0.03125),
            );
        }
    }
}

#[test]
fn native_prefill_retains_decoded_weights_beyond_the_f16_range() {
    let device = Device::system_default().expect("native prefill conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let format = GgufBlockFormat::Iq4Nl;
    // Positive/negative coefficients exceed half's finite range, while their
    // products and final output remain finite. A half weight staging route
    // would create infinities and destroy cancellation in this valid input.
    let mut block = vec![0x0f; format.block_bytes()];
    block[..2].copy_from_slice(&f16::from_f32(1024.0).to_le_bytes());
    let mut decoded = vec![0.0; format.block_values()];
    format.decode(&block, &mut decoded).unwrap();
    assert!(decoded.iter().any(|value| value.abs() > f16::MAX.to_f32()));
    assert_native_prefill(
        &device,
        &pipelines,
        &queue,
        format,
        &block,
        33,
        65,
        |row, _| {
            f16::from_f32(if row % 2 == 0 {
                1.0 / 1024.0
            } else {
                -1.0 / 1024.0
            })
        },
    );
}
