//! Q4K specialization against the retained generic kernel and an F64 oracle.

use super::fixed_k::Fixture;
use super::*;

#[test]
#[ignore = "requires an actual CUDA device"]
fn q4k_specialization_preserves_generic_bits_and_f64_oracle_on_cuda() {
    let context = CudaContext::new(0).expect("Q4K specialization requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for (rows, inputs, outputs) in [
        (1, 256, 7),
        (2, 768, 7),
        (7, 512, 17),
        (8, 768, 17),
        (9, 2560, 7),
        (64, 768, 17),
        (65, 2560, 7),
    ] {
        let mut fixture = Fixture::new(&stream, rows, inputs, outputs);
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let expected = fixture.validate(&stream);
        for (kernel, row_tile) in [
            (&kernels.linear_q4k_f16, 1),
            (&kernels.linear_q4k_tiled_f16, LINEAR_ROW_TILE),
        ] {
            fixture.run(&stream, kernel, row_tile, 1);
            assert_eq!(
                fixture.validate(&stream),
                expected,
                "specialization changed accumulation bits"
            );
            fixture.run(&stream, kernel, row_tile, 2);
            assert_eq!(
                fixture.validate(&stream),
                expected,
                "repeated specialization changed bits"
            );
        }
    }
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn q4k_specialization_dispatch_microbench() {
    const ITERATIONS: u32 = 32;
    let context = CudaContext::new(0).expect("Q4K microbench requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // Synthetic source blocks with actual Qwen3.5-4B projection dimensions.
    // Timing is per complete projection, not per token or generated response.
    for (inputs, outputs) in [(2560, 2560), (2560, 9216)] {
        for rows in [1, 4, 64, 65] {
            let mut fixture = Fixture::new(&stream, rows, inputs, outputs);
            let (generic, specialized, row_tile) = if rows == 1 {
                (&kernels.linear_f16, &kernels.linear_q4k_f16, 1)
            } else {
                (
                    &kernels.linear_tiled_f16,
                    &kernels.linear_q4k_tiled_f16,
                    LINEAR_ROW_TILE,
                )
            };
            fixture.run(&stream, generic, row_tile, 1);
            let reference = fixture.validate(&stream);
            for round in 0..10 {
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
                        fixture.validate(&stream),
                        reference,
                        "timed variant changed bits"
                    );
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark": "q4k_format_specialization", "rows": rows, "input": inputs, "output": outputs,
                                "row_tile": row_tile, "specialized": specialized_route, "round": round - 2,
                                "projection_iterations": ITERATIONS,
                                "command_wall_ns": wall_ns, "command_gpu_ns": gpu_ns,
                                "wall_ns": wall_ns / f64::from(ITERATIONS), "gpu_ns": gpu_ns / f64::from(ITERATIONS),
                            })
                        );
                    }
                }
            }
        }
    }
}
