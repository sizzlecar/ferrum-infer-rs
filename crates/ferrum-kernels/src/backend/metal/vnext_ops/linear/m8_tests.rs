//! M8 product selection and exact parity with the retained M32 MMA arithmetic.
//! This is a small backend regression, not a model-quality or throughput gate.
use super::*;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device as CandleDevice, Tensor};
use half::f16;
use metal::{MTLCommandBufferStatus, MTLResourceOptions};

fn shared<T>(device: &Device, values: &[T]) -> metal::Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn bytes(buffer: &metal::BufferRef) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast(), buffer.length() as usize) }
}

#[test]
fn m8_selected_kernel_matches_original_mma_with_offsets_and_tails() {
    let device = Device::system_default().expect("M8 product regression requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let guard = f16::from_f32(-19.5);
    for (format, dtype) in [
        (LinearPhysicalFormat::Q4K, GgmlDType::Q4K),
        (LinearPhysicalFormat::Q5K, GgmlDType::Q5K),
        (LinearPhysicalFormat::Q6K, GgmlDType::Q6K),
    ] {
        for (input_width, output_width) in [(256, 1), (256, 63), (512, 64), (512, 65)] {
            let rows = 8;
            let dense: Vec<_> = (0..input_width * output_width)
                .map(|i| ((i * 17 % 251) as f32 - 125.0) / 512.0)
                .collect();
            let tensor =
                Tensor::from_vec(dense, (output_width, input_width), &CandleDevice::Cpu).unwrap();
            let quantized = QTensor::quantize(&tensor, dtype).unwrap();
            let weight_prefix = 64;
            let mut weights = vec![0xa5; weight_prefix];
            weights.extend_from_slice(&quantized.data().unwrap());
            weights.extend_from_slice(&[0x5a; 64]);
            let weight = shared(&device, &weights);
            for generation in 0..2 {
                let input_prefix = 7;
                let mut input = vec![guard; input_prefix];
                input.extend((0..rows * input_width).map(|i| {
                    f16::from_f32(((i * 13 + generation * 29) % 197) as f32 / 2048.0 - 0.046875)
                }));
                input.extend_from_slice(&[guard; 11]);
                let input_buffer = shared(&device, &input);
                let output_prefix = 9;
                let stride = output_width + 9;
                let column = 4;
                let initial = vec![guard; output_prefix + rows * stride + 13];
                let outputs: [_; 2] = std::array::from_fn(|_| shared(&device, &initial));
                let params = LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: stride as u32,
                    output_column_offset: column as u32,
                };
                let (selected, kind) =
                    pipelines.plain_linear_dispatch(format, ElementType::F16, params);
                assert_eq!(kind, LinearDispatchKind::TiledGemmM8);
                let original = match format {
                    LinearPhysicalFormat::Q4K => &pipelines.k_quant_gemm.q4_k,
                    LinearPhysicalFormat::Q5K => &pipelines.k_quant_gemm.q5_k,
                    LinearPhysicalFormat::Q6K => &pipelines.k_quant_gemm.q6_k,
                    _ => unreachable!(),
                };
                for (arm, (pipeline, dispatch)) in
                    [(original, LinearDispatchKind::TiledGemm), (selected, kind)]
                        .into_iter()
                        .enumerate()
                {
                    let command = queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(pipeline);
                    encoder.set_buffer(0, Some(&input_buffer), (input_prefix * 2) as u64);
                    encoder.set_buffer(1, Some(&weight), weight_prefix as u64);
                    encoder.set_buffer(2, Some(&outputs[arm]), (output_prefix * 2) as u64);
                    bind_linear_params(encoder, params, format, ElementType::F16);
                    dispatch_linear_grid(encoder, params, dispatch);
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                }
                assert_eq!(bytes(&outputs[0]), bytes(&outputs[1]), "{format:?}");
                for (index, pair) in bytes(&outputs[1]).chunks_exact(2).enumerate() {
                    let value = f16::from_bits(u16::from_ne_bytes([pair[0], pair[1]]));
                    assert!(value.is_finite());
                    let relative = index.saturating_sub(output_prefix);
                    let written = index >= output_prefix
                        && relative / stride < rows
                        && (column..column + output_width).contains(&(relative % stride));
                    if !written {
                        assert_eq!(value.to_bits(), guard.to_bits(), "output guard {index}");
                    }
                }
                let input_bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input.len() * 2) };
                assert_eq!(bytes(&input_buffer), input_bytes);
                assert_eq!(bytes(&weight), weights);
            }
        }
    }
}

#[test]
fn m8_product_route_preserves_neighbors_and_existing_plain_splits() {
    let device = Device::system_default().expect("M8 product regression requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    for format in [
        LinearPhysicalFormat::Q4K,
        LinearPhysicalFormat::Q5K,
        LinearPhysicalFormat::Q6K,
        LinearPhysicalFormat::Q8_0,
    ] {
        for rows in [1, 4, 7, 8, 9, 16, 32] {
            let params = LinearParams {
                rows,
                in_features: 256,
                out_features: 65,
                output_stride: 73,
                output_column_offset: 3,
            };
            assert_eq!(
                pipelines
                    .plain_linear_dispatch(format, ElementType::F16, params)
                    .1
                    == LinearDispatchKind::TiledGemmM8,
                rows == 8 && format != LinearPhysicalFormat::Q8_0,
            );
            assert_ne!(
                pipelines
                    .plain_linear_dispatch(format, ElementType::F32, params)
                    .1,
                LinearDispatchKind::TiledGemmM8,
            );
        }
    }
    for (format, input, output, splits) in [
        (LinearPhysicalFormat::Q4K, 1024, 1024, true),
        (LinearPhysicalFormat::Q6K, 2048, 1024, true),
        (LinearPhysicalFormat::Q5K, 2048, 1024, true),
        (LinearPhysicalFormat::Q5K, 1024, 2048, false),
    ] {
        let launch = linear_launch(
            PreparedLinearPart {
                region: 1,
                format,
                output_offset: 0,
                out_features: output,
                transform: None,
            },
            0,
            2,
            8,
            input,
            u64::from(output),
            0,
            0,
        )
        .unwrap();
        assert_eq!(
            matches!(launch.plain_plan, PlainLinearPlan::SplitRows { .. }),
            splits
        );
        assert_eq!(launch.dispatch_count(), if splits { 2 } else { 1 });
        if let Some(parts) = launch.plain_plan.parts(launch) {
            assert_eq!(parts[0].params.rows, 4);
            assert_eq!(parts[1].params.rows, 4);
            for part in parts {
                assert_ne!(
                    pipelines
                        .plain_linear_dispatch(format, ElementType::F16, part.params)
                        .1,
                    LinearDispatchKind::TiledGemmM8,
                );
            }
        }
    }
}
