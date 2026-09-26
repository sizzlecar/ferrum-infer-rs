//! Small-row Q5K uses the original arithmetic and is selected by production.
use super::fixed_k::Fixture;
use super::*;

#[test]
#[ignore = "requires an actual CUDA device"]
fn q5k_smallrow_dispatch_keeps_dtype_and_large_row_boundaries() {
    use ferrum_interfaces::vnext::ElementType::{F16, F32};
    let context = CudaContext::new(0).unwrap();
    for rows in [1, 4, 31, 32] {
        super::shared_dispatch::check::<f32, f16>(
            &context,
            GgufBlockFormat::Q5K,
            rows,
            256,
            17,
            F32,
            F16,
            if rows == 1 {
                "vnext_gguf_linear_f32_f16"
            } else {
                "vnext_gguf_linear_tiled_f32_f16"
            },
        );
        super::shared_dispatch::check::<f32, f32>(
            &context,
            GgufBlockFormat::Q5K,
            rows,
            256,
            17,
            F32,
            F32,
            if rows == 1 {
                "vnext_gguf_linear_f32"
            } else {
                "vnext_gguf_linear_tiled_f32"
            },
        );
    }
    super::shared_dispatch::check::<f16, f16>(
        &context,
        GgufBlockFormat::Q5K,
        32,
        256,
        17,
        F16,
        F16,
        "vnext_gguf_linear_tiled_f16",
    );
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn q5k_smallrow_production_preserves_generic_bits_dense_sparse_guards_and_graph() {
    let context = CudaContext::new(0).expect("Q5K specialization requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.new_stream().unwrap();
    for dense in [false, true] {
        for (rows, inputs, outputs) in [
            (1, 256, 7),
            (2, 768, 7),
            (4, 4096, 17),
            (7, 512, 17),
            (8, 768, 17),
            (9, 2560, 7),
            (31, 4096, 7),
        ] {
            let mut fixture =
                Fixture::with_format(&stream, GgufBlockFormat::Q5K, rows, inputs, outputs, dense);
            fixture.run(&stream, &kernels.linear_f16, 1, 1);
            let reference = fixture.validate(&stream);
            for (kernel, tile) in [
                (&kernels.linear_tiled_f16, LINEAR_ROW_TILE),
                (&kernels.linear_q5k_f16, 1),
                (&kernels.linear_q5k_tiled_f16, LINEAR_ROW_TILE),
            ] {
                fixture.run(&stream, kernel, tile, 1);
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "dense={dense}, {rows}x{inputs}x{outputs}"
                );
            }
            for captured in [false, true] {
                fixture.run_production(&stream, &kernels, captured);
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "production captured={captured}"
                );
            }
        }
    }
}

#[test]
#[ignore = "diagnostic only; coordinate exclusive CUDA access"]
fn q5k_smallrow_gdn_projection_paired_microbench() {
    const ITERATIONS: u32 = 32;
    let context = CudaContext::new(0).expect("Q5K timing requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.new_stream().unwrap();
    // Actual Qwen3.5-9B GDN QKV and output projection dimensions. Synthetic
    // source bytes isolate this route; these timings are not a serving claim.
    for outputs in [8192, 4096] {
        for rows in [1, 4] {
            let mut fixture =
                Fixture::with_format(&stream, GgufBlockFormat::Q5K, rows, 4096, outputs, true);
            let (generic, specialized, tile) = if rows == 1 {
                (&kernels.linear_f16, &kernels.linear_q5k_f16, 1)
            } else {
                (
                    &kernels.linear_tiled_f16,
                    &kernels.linear_q5k_tiled_f16,
                    LINEAR_ROW_TILE,
                )
            };
            fixture.run(&stream, generic, tile, 1);
            let reference = fixture.validate(&stream);
            fixture.run(&stream, specialized, tile, 1);
            assert_eq!(
                fixture.validate(&stream),
                reference,
                "pre-timing equivalence"
            );
            fixture.run_production(&stream, &kernels, true);
            assert_eq!(
                fixture.validate(&stream),
                reference,
                "production graph equivalence"
            );
            for round in 0..10 {
                for candidate in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let kernel = if candidate { specialized } else { generic };
                    let (wall_ns, gpu_ns) = fixture.run(&stream, kernel, tile, ITERATIONS);
                    assert_eq!(fixture.validate(&stream), reference, "timed output");
                    assert!(gpu_ns.is_finite() && gpu_ns > 0.0);
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark": "q5k_smallrow_fixed_abi", "rows": rows,
                                "input": 4096, "output": outputs, "row_tile": tile,
                                "candidate": candidate, "round": round - 2,
                                "projection_iterations": ITERATIONS, "bitwise_qualified": true,
                                "command_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
                                "wall_ns": wall_ns / f64::from(ITERATIONS),
                                "gpu_ns": gpu_ns / f64::from(ITERATIONS),
                                "release_approved": false,
                            })
                        );
                    }
                }
            }
        }
    }
}
