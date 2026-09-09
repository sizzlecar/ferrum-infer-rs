use super::*;
use crate::gguf_blocks::fixtures::{oracle_blocks, FORMATS};
use half::f16;
use metal::{Buffer, MTLCommandBufferStatus, MTLResourceOptions};

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
                    pipelines.linear_pipeline(physical, params.rows)
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
