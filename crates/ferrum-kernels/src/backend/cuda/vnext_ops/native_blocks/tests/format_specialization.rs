//! Format-constant kernels must preserve the generic arithmetic and launch ABI.

use super::{q4k::Fixture, *};

fn f16_kernels(
    kernels: &CudaNativeBlockKernels,
    format: GgufBlockFormat,
) -> (&CudaFunction, &CudaFunction) {
    match format {
        GgufBlockFormat::Q5K => (&kernels.linear_q5k_f16, &kernels.linear_q5k_tiled_f16),
        GgufBlockFormat::Iq4Xs => (&kernels.linear_iq4xs_f16, &kernels.linear_iq4xs_tiled_f16),
        GgufBlockFormat::Q6K => (&kernels.linear_q6k_f16, &kernels.linear_q6k_tiled_f16),
        _ => panic!("unsupported specialization fixture format: {format:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn check_format<T: Scalar>(
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    format: GgufBlockFormat,
    generic: &CudaFunction,
    generic_tiled: &CudaFunction,
    specialized: &CudaFunction,
    specialized_tiled: &CudaFunction,
) {
    // Single and multiple quant blocks; a partial four-column group and an
    // incomplete final eight-row tile. Dense inputs exercise every encoded
    // value, including independently signed scales and cross-block tails.
    for (rows, inputs, outputs) in [
        (1, 256, 7),
        (2, 768, 17),
        (4, 512, 7),
        (7, 768, 17),
        (8, 2560, 7),
        (9, 768, 17),
        (64, 512, 7),
        (65, 2560, 17),
    ] {
        let mut fixture = Fixture::<T>::for_format(stream, format, rows, inputs, outputs, true);
        fixture.run(stream, generic, 1, 1);
        // CPU GGUF decoding plus F64 multiplication/summation is independent
        // of the CUDA generic and specialized decoder/warp reduction paths.
        let reference = fixture.validate(stream);
        for (kernel, row_tile) in [
            (generic_tiled, LINEAR_ROW_TILE),
            (specialized, 1),
            (specialized_tiled, LINEAR_ROW_TILE),
        ] {
            for iterations in [1, 2] {
                fixture.run(stream, kernel, row_tile, iterations);
                assert_eq!(
                    fixture.validate(stream),
                    reference,
                    "{format:?} {rows}x{inputs}x{outputs}, row_tile={row_tile}, iterations={iterations} changed generic bits"
                );
            }
        }
        // Exercise production precision/format selection with the same odd
        // byte weight view, nonzero column offset and padded output stride.
        for _ in 0..2 {
            fixture.run_dispatch(stream, kernels);
            assert_eq!(
                fixture.validate(stream),
                reference,
                "{format:?} {rows}x{inputs}x{outputs} production dispatch changed bits"
            );
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn format_specializations_preserve_generic_bits_and_f64_oracle_on_cuda() {
    let context = CudaContext::new(0).expect("format specialization requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for format in [
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Iq4Xs,
        GgufBlockFormat::Q6K,
    ] {
        let (scalar, tiled) = f16_kernels(&kernels, format);
        check_format::<f16>(
            &stream,
            &kernels,
            format,
            &kernels.linear_f16,
            &kernels.linear_tiled_f16,
            scalar,
            tiled,
        );
    }
    // The common template also backs the existing F32 Q6K output-head route.
    // Retain its generic bit pattern, not only a loose output tolerance.
    check_format::<f32>(
        &stream,
        &kernels,
        GgufBlockFormat::Q6K,
        &kernels.linear_f32,
        &kernels.linear_tiled_f32,
        &kernels.linear_q6k_f32,
        &kernels.linear_q6k_tiled_f32,
    );
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn format_specialization_dispatch_microbench() {
    const ITERATIONS: u32 = 8;
    let context = CudaContext::new(0).expect("format microbench requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // Synthetic bytes, actual Qwen3.8-27B projection dimensions (K,N):
    // recurrent gate/QKV/output, FFN gate/down, and Q6K key/output/down.
    // Fixture creation, decoding, copies and correctness checks are untimed.
    for (format, inputs, outputs) in [
        (GgufBlockFormat::Q5K, 5120, 6144),
        (GgufBlockFormat::Q5K, 5120, 10240),
        (GgufBlockFormat::Q5K, 6144, 5120),
        (GgufBlockFormat::Iq4Xs, 5120, 17408),
        (GgufBlockFormat::Iq4Xs, 17408, 5120),
        (GgufBlockFormat::Q6K, 5120, 1024),
        (GgufBlockFormat::Q6K, 6144, 5120),
        (GgufBlockFormat::Q6K, 17408, 5120),
    ] {
        let (scalar, tiled) = f16_kernels(&kernels, format);
        for rows in [1, 4, 8, 64] {
            let mut fixture =
                Fixture::<f16>::for_format(&stream, format, rows, inputs, outputs, false);
            let (generic, specialized, row_tile) = if rows == 1 {
                (&kernels.linear_f16, scalar, 1)
            } else {
                (&kernels.linear_tiled_f16, tiled, LINEAR_ROW_TILE)
            };
            fixture.run(&stream, generic, row_tile, 1);
            let reference = fixture.validate(&stream);
            for round in 0..6 {
                // Pair both routes and alternate order to limit drift bias.
                for specialized_route in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let kernel = if specialized_route {
                        specialized
                    } else {
                        generic
                    };
                    let (wall_ns, gpu_ns) = fixture.run(&stream, kernel, row_tile, ITERATIONS);
                    assert_eq!(
                        fixture.validate_output(&stream),
                        reference,
                        "timed route changed bits"
                    );
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark":"native_format_specialization", "format":format!("{format:?}"),
                                "input_dtype":"f16", "output_dtype":"f16", "synthetic_weights":true,
                                "rows":rows, "input":inputs, "output":outputs, "row_tile":row_tile,
                                "specialized":specialized_route, "round":round-2,
                                "projection_iterations":ITERATIONS,
                                "command_wall_ns":wall_ns, "command_gpu_ns":gpu_ns,
                                "wall_ns":wall_ns/f64::from(ITERATIONS), "gpu_ns":gpu_ns/f64::from(ITERATIONS)
                            })
                        );
                    }
                }
            }
            // Check complete input/weight allocations again after every paired
            // series without repeatedly copying large weights during timings.
            assert_eq!(fixture.validate(&stream), reference);
        }
    }
}
