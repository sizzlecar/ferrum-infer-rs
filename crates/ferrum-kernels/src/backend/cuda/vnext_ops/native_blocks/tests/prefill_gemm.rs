//! Shared-memory GGUF matrix multiplication: independent numeric and extent
//! checks, plus paired event timing on representative projection dimensions.

use super::*;
use cudarc::driver::{sys::CUevent_flags, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{ElementType, WeightId};

fn kernel(context: &Arc<CudaContext>, format: GgufBlockFormat, suffix: &str) -> CudaFunction {
    let module = context
        .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
        .unwrap();
    let format = match format {
        GgufBlockFormat::Q4K => "q4k",
        GgufBlockFormat::Q5K => "q5k",
        GgufBlockFormat::Q6K => "q6k",
        _ => panic!("unsupported shared-memory format"),
    };
    module
        .load_function(&format!("vnext_gguf_gemm_{format}_{suffix}"))
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn check<I: Scalar, O: Scalar>(
    context: &Arc<CudaContext>,
    format: GgufBlockFormat,
    suffix: &str,
    rows: usize,
    inputs: usize,
    outputs: usize,
) {
    let stream = context.default_stream();
    let (mut encoded, _) = matrix(format, outputs, inputs / 256);
    // Give every output column a distinct scale so a column permutation does
    // not pass merely because the underlying block fixture is periodic.
    let scale_offset = if format == GgufBlockFormat::Q6K {
        208
    } else {
        0
    };
    for (block_index, block) in encoded.chunks_exact_mut(format.block_bytes()).enumerate() {
        let column = block_index / (inputs / 256);
        let scale = f16::from_f32((column + 1) as f32 / 512.0).to_bits();
        block[scale_offset..scale_offset + 2].copy_from_slice(&scale.to_le_bytes());
    }
    let mut decoded = vec![0.0_f32; outputs * inputs];
    format.decode(&encoded, &mut decoded).unwrap();
    let mut weights = vec![0xcc_u8; 5];
    weights.extend_from_slice(&encoded);
    weights.extend_from_slice(&[0xcc; 7]);
    let mut input = vec![I::from_f32(-12345.0); 3 + rows * inputs + 5];
    for (i, value) in input[3..3 + rows * inputs].iter_mut().enumerate() {
        // The F32 variants must include values that do not round-trip through
        // F16; a hidden activation downcast must not satisfy this oracle.
        *value = I::from_f32(((i * 11 % 37) as f32 - 18.0) / 127.31);
    }
    let stride = outputs + 5;
    let initial = vec![O::from_f32(-12345.0); 8 + rows * stride + 7];
    let input_gpu = stream.clone_htod(&input).unwrap();
    let weights_gpu = stream.clone_htod(&weights).unwrap();
    let mut output_gpu = stream.clone_htod(&initial).unwrap();
    let selected = kernel(
        context,
        format,
        suffix.strip_prefix("dispatch_").unwrap_or(suffix),
    );
    let kernels = CudaNativeBlockKernels::load(context).unwrap();
    // Repeated launches must fully overwrite each output without relying on
    // initialized shared memory; include partial row and column tiles.
    for _ in 0..2 {
        let input_view = input_gpu.slice(3..3 + rows * inputs);
        let weights_view = weights_gpu.slice(5..5 + encoded.len());
        let mut output_view = output_gpu.slice_mut(8..8 + rows * stride);
        if suffix == "dispatch_f16" {
            let part = weights::MatrixPart {
                component_id: WeightId::new("component.shared-gemm-test").unwrap(),
                format: weights::MatrixFormat::Block(format),
                rows: outputs as u32,
                columns: inputs as u32,
                output_offset: 2,
                transform: None,
                signs_region: None,
            };
            let (input_ptr, _input_guard) = input_view.device_ptr(&stream);
            let (weight_ptr, _weight_guard) = weights_view.device_ptr(&stream);
            let (output_ptr, _output_guard) = output_view.device_ptr_mut(&stream);
            kernels
                .linear(
                    &stream,
                    input_ptr,
                    weight_ptr,
                    output_ptr,
                    &part,
                    rows as u32,
                    stride as u32,
                    ElementType::F16,
                )
                .unwrap();
            continue;
        }
        let params = [
            rows as u32,
            inputs as u32,
            outputs as u32,
            stride as u32,
            2,
            format.ggml_type_id(),
            256,
            format.block_bytes() as u32,
        ];
        let mut launch = stream.launch_builder(&selected);
        launch
            .arg(&input_view)
            .arg(&weights_view)
            .arg(&mut output_view);
        for param in &params {
            launch.arg(param);
        }
        // SAFETY: Byte-safe weight views and complete input/output extents;
        // all blocks participate in barriers, with guarded edge tile loads.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((outputs as u32).div_ceil(64), (rows as u32).div_ceil(64), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
    let actual = stream.clone_dtoh(&output_gpu).unwrap();
    for (index, &value) in actual.iter().enumerate() {
        let position = index
            .checked_sub(8)
            .filter(|&i| i < rows * stride && (2..2 + outputs).contains(&(i % stride)));
        if let Some(position) = position {
            let row = position / stride;
            let col = position % stride - 2;
            let products = input[3 + row * inputs..][..inputs]
                .iter()
                .zip(&decoded[col * inputs..][..inputs])
                .map(|(&x, &w)| f64::from(x.as_f32()) * f64::from(w));
            let expected = products.clone().sum::<f64>();
            let absolute_sum = products.map(f64::abs).sum::<f64>();
            let error_bound =
                (inputs as f64 * f64::from(f32::EPSILON) + O::ROUNDING) * absolute_sum + 1e-6;
            assert!(value.as_f32().is_finite()
                && (f64::from(value.as_f32()) - expected).abs() <= error_bound,
                "{format:?}/{suffix} {rows}x{inputs}x{outputs} at {row},{col}: {} != {expected}, bound {error_bound}", value.as_f32());
        } else {
            assert_eq!(
                value.as_f32().to_bits(),
                initial[index].as_f32().to_bits(),
                "output guard {index}"
            );
        }
    }
    let actual_input = stream.clone_dtoh(&input_gpu).unwrap();
    assert!(actual_input
        .iter()
        .zip(&input)
        .all(|(&a, &b)| a.as_f32().to_bits() == b.as_f32().to_bits()));
    assert_eq!(stream.clone_dtoh(&weights_gpu).unwrap(), weights);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn shared_prefill_gemm_preserves_f32_math_and_buffer_extents_on_cuda() {
    let context = CudaContext::new(0).expect("shared GGUF GEMM requires CUDA");
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        for (rows, inputs, outputs) in [
            (1, 256, 7),
            (7, 768, 31),
            (31, 512, 33),
            (32, 768, 32),
            (33, 256, 65),
            (63, 256, 65),
            (64, 768, 64),
            (65, 768, 7),
        ] {
            check::<f16, f16>(&context, format, "f16", rows, inputs, outputs);
            check::<f32, f32>(&context, format, "f32", rows, inputs, outputs);
            check::<f32, f16>(&context, format, "f32_f16", rows, inputs, outputs);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn shared_prefill_gemm_production_dispatch_preserves_partition_offsets_on_cuda() {
    let context = CudaContext::new(0).expect("shared GGUF GEMM dispatch requires CUDA");
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        check::<f16, f16>(&context, format, "dispatch_f16", 129, 256, 129);
    }
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn shared_prefill_gemm_dispatch_microbench() {
    const REPEATS: u32 = 8;
    let context = CudaContext::new(0).expect("shared GGUF GEMM timing requires CUDA");
    let stream = context.default_stream();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        let shared = kernel(&context, format, "f16");
        let reference = if format == GgufBlockFormat::Q4K {
            &kernels.linear_q4k_tiled_f16
        } else {
            &kernels.linear_tiled_f16
        };
        for (inputs, outputs) in [(4096_usize, 4096_usize), (4096, 12288), (12288, 4096)] {
            let (encoded, decoded) = matrix(format, outputs, inputs / 256);
            let weights = stream.clone_htod(&encoded).unwrap();
            for rows in [32_usize, 128, 512] {
                let input_cpu = (0..rows * inputs)
                    .map(|i| f16::from_f32(((i % inputs * 11 % 37) as f32 - 18.0) / 128.0))
                    .collect::<Vec<_>>();
                let input = stream.clone_htod(&input_cpu).unwrap();
                let mut output = stream.alloc_zeros::<f16>(rows * outputs).unwrap();
                let params = [
                    rows as u32,
                    inputs as u32,
                    outputs as u32,
                    outputs as u32,
                    0,
                    format.ggml_type_id(),
                    256,
                    format.block_bytes() as u32,
                ];
                let probes = [0, outputs / 2, outputs - 1].map(|col| {
                    let products = input_cpu[..inputs]
                        .iter()
                        .zip(&decoded[col * inputs..][..inputs])
                        .map(|(&x, &w)| f64::from(x.to_f32()) * f64::from(w));
                    (
                        col,
                        products.clone().sum::<f64>(),
                        products.map(f64::abs).sum::<f64>(),
                    )
                });
                for round in 0..6 {
                    for use_shared in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let selected = if use_shared { &shared } else { reference };
                        let config = if use_shared {
                            LaunchConfig {
                                grid_dim: (
                                    (outputs as u32).div_ceil(64),
                                    (rows as u32).div_ceil(64),
                                    1,
                                ),
                                block_dim: (16, 16, 1),
                                shared_mem_bytes: 0,
                            }
                        } else {
                            LaunchConfig {
                                grid_dim: (
                                    (outputs as u32).div_ceil(4),
                                    (rows as u32).div_ceil(8),
                                    1,
                                ),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            }
                        };
                        let start = stream
                            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap();
                        for _ in 0..REPEATS {
                            let mut launch = stream.launch_builder(selected);
                            launch.arg(&input).arg(&weights).arg(&mut output);
                            for param in &params {
                                launch.arg(param);
                            }
                            // SAFETY: Both kernels receive complete contiguous matrices and their respective fixed launch geometry.
                            unsafe { launch.launch(config) }.unwrap();
                        }
                        let end = stream
                            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                            .unwrap();
                        end.synchronize().unwrap();
                        let gpu_us = f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0
                            / f64::from(REPEATS);
                        let actual = stream.clone_dtoh(&output).unwrap();
                        for row in [0, rows / 2, rows - 1] {
                            for &(col, expected, sum_abs) in &probes {
                                let observed = actual[row * outputs + col].to_f32();
                                let bound = (inputs as f64 * f64::from(f32::EPSILON)
                                    + <f16 as Scalar>::ROUNDING)
                                    * sum_abs
                                    + 1e-6;
                                assert!(
                                    observed.is_finite()
                                        && (f64::from(observed) - expected).abs() <= bound
                                );
                            }
                        }
                        if round > 0 {
                            println!("gguf_shared_gemm format={format:?} rows={rows} inputs={inputs} outputs={outputs} round={round} shared={use_shared} gpu_us={gpu_us:.3}");
                        }
                    }
                }
            }
        }
    }
}
