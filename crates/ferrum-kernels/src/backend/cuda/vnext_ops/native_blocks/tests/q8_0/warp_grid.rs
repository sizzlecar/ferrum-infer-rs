//! Isolate columns per CTA without changing Q8_0 arithmetic or row tiling.

use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::tests::captured::{measure, Captured};

fn context() -> (Arc<CudaContext>, Arc<CudaStream>, CudaNativeBlockKernels) {
    let context = CudaContext::new(0).expect("Q8_0 warp-grid checks require CUDA");
    // Match production before allocating. One test thread/stream owns every
    // allocation; explicit fences precede reads and graph/fixture destruction.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    (context, stream, kernels)
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn q8_0_warp_grid_preserves_bits_and_guards_on_cuda() {
    let (_context, stream, kernels) = context();
    let mut shapes = std::collections::BTreeSet::from([
        (1_usize, 32_usize, 1_usize),
        (4, 32, 3),
        (8, 96, 7),
        (9, 96, 47),
        (8, 32, 49),
        (65, 96, 7),
        (4, 5120, 48),
        (8, 5120, 48),
        (16, 5120, 48),
        (32, 5120, 48),
        (8, 5120, 1024),
    ]);
    // No production selector is added. These widths exercise the proposed
    // four-warp-grid/SM boundary, including partial column groups.
    let groups = u64::from(kernels.multiprocessors).div_ceil(8);
    for outputs in [
        groups.saturating_sub(1).max(1) * 4,
        groups * 4,
        groups * 4 + 1,
    ] {
        shapes.insert((8, 96, usize::try_from(outputs).unwrap()));
    }
    for (rows, inputs, outputs) in shapes {
        let mut fixture = if inputs <= 96 {
            Fixture::<f16>::for_encoded_weights(
                &stream,
                GgufBlockFormat::Q8_0,
                rows,
                inputs,
                outputs,
                true,
                boundary_weights(inputs, outputs),
            )
        } else {
            Fixture::<f16>::for_format(&stream, GgufBlockFormat::Q8_0, rows, inputs, outputs, false)
        };
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let reference = fixture.validate(&stream);
        for (kernel, columns_per_block) in [
            (&kernels.linear_q8_0_f16, 4),
            (&kernels.linear_q8_0_warp_f16, 1),
        ] {
            for iterations in [1, 2] {
                fixture.run_with_columns(&stream, kernel, 1, columns_per_block, iterations);
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "Q8_0 {rows}x{inputs}x{outputs}, columns_per_block={columns_per_block}, direct iterations={iterations}"
                );
            }
            let graph = Captured::new(&stream, || {
                fixture.enqueue(&stream, kernel, 1, columns_per_block, 2);
            });
            for _ in 0..2 {
                fixture.reset(&stream);
                graph.launch();
                stream.synchronize().unwrap();
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "Q8_0 {rows}x{inputs}x{outputs}, columns_per_block={columns_per_block}, graph replay"
                );
            }
            // The graph retains raw addresses, so it must die before fixture.
            drop(graph);
        }
        // The production route remains the retained selector and four-warp
        // geometry, including on shapes where this experiment tests one warp.
        fixture.run_dispatch(&stream, &kernels);
        stream.synchronize().unwrap();
        assert_eq!(fixture.validate(&stream), reference);
    }
}

#[test]
#[ignore = "paired captured-graph GPU timing; coordinate exclusive CUDA access"]
fn q8_0_warp_grid_paired_graph_microbench() {
    const KERNELS_PER_GRAPH: u32 = 32;
    const WARM_QUADS: usize = 2;
    const FORMAL_QUADS: usize = 4;
    let (context, stream, kernels) = context();
    let l2_bytes = context
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap();
    let routes = [
        ("four_warps", &kernels.linear_q8_0_f16, 4_u32),
        ("one_warp", &kernels.linear_q8_0_warp_f16, 1_u32),
    ];
    let shapes = [
        (4_usize, 5120_usize, 48_usize),
        (8, 5120, 48),
        (16, 5120, 48),
        (32, 5120, 48),
        (4, 5120, 1024),
        (8, 5120, 1024),
        (16, 5120, 1024),
        (32, 5120, 1024),
        (1, 32, 1),
        (4, 32, 49),
        (8, 96, 47),
        (9, 96, 7),
    ];
    println!(
        "{}",
        serde_json::json!({
            "benchmark":"native_q8_0_warp_grid_graph_plan",
            "format":"Q8_0", "input_dtype":"f16", "output_dtype":"f16", "row_tile":1,
            "shapes":shapes.iter().map(|&(rows, input, output)| serde_json::json!({
                "rows":rows, "input":input, "output":output
            })).collect::<Vec<_>>(),
            "routes":["four_warps", "one_warp"], "columns_per_block":[4,1],
            "kernels_per_graph":KERNELS_PER_GRAPH, "graph_replays_per_observation":1,
            "warm_quads":WARM_QUADS, "formal_quads":FORMAL_QUADS,
            "order":"alternating ABBA and BAAB; A=four_warps, B=one_warp",
            "synthetic_weights":true, "input_mode":"dense_deterministic",
            "weight_mode":"resident_reused", "cache_flush":false,
            "same_fixture_and_allocations":true, "device_l2_bytes":l2_bytes,
            "multiprocessors":kernels.multiprocessors, "event_tracking":false,
            "excluded_from_timing":["allocation", "reset", "capture", "instantiate", "upload", "validation"],
            "gpu_ns_scope":"CUDA events around one graph replay",
            "wall_ns_scope":"host start-event, graph launch, end-event and end synchronization",
            "production_selector_changed":false
        })
    );
    for (rows, inputs, outputs) in shapes {
        let mut fixture =
            Fixture::<f16>::for_format(&stream, GgufBlockFormat::Q8_0, rows, inputs, outputs, true);
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let reference = fixture.validate(&stream);
        // Warm each exact kernel, then capture/upload on the same allocations.
        // Reset/copy, capture, upload and F64/bitwise checking are never timed.
        let graphs = routes
            .iter()
            .map(|&(_, kernel, columns)| {
                fixture.run_with_columns(&stream, kernel, 1, columns, 1);
                assert_eq!(fixture.validate(&stream), reference);
                Captured::new(&stream, || {
                    fixture.enqueue(&stream, kernel, 1, columns, KERNELS_PER_GRAPH);
                })
            })
            .collect::<Vec<_>>();
        for quad in 0..WARM_QUADS + FORMAL_QUADS {
            // Alternate ABBA and BAAB: both routes occupy every order position.
            let order = if quad % 2 == 0 {
                [0, 1, 1, 0]
            } else {
                [1, 0, 0, 1]
            };
            for (position, index) in order.into_iter().enumerate() {
                let (route, _, columns_per_block) = routes[index];
                fixture.reset(&stream);
                let (wall_ns, gpu_ns) = measure(&stream, || graphs[index].launch());
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "Q8_0 graph timing route={route}, {rows}x{inputs}x{outputs} changed bits"
                );
                if let Some(formal_quad) = quad.checked_sub(WARM_QUADS) {
                    println!(
                        "{}",
                        serde_json::json!({
                            "benchmark":"native_q8_0_warp_grid_graph",
                            "format":"Q8_0", "input_dtype":"f16", "output_dtype":"f16",
                            "synthetic_weights":true, "input_mode":"dense_deterministic",
                            "weight_mode":"resident_reused",
                            "rows":rows, "input":inputs, "output":outputs, "row_tile":1,
                            "route":route, "columns_per_block":columns_per_block,
                            "grid":[(outputs as u32).div_ceil(columns_per_block),rows,1],
                            "block":[columns_per_block*32,1,1], "shared_mem_bytes":0,
                            "multiprocessors":kernels.multiprocessors,
                            "four_warp_grid_below_sm":(outputs as u64).div_ceil(4)*(rows as u64)<u64::from(kernels.multiprocessors),
                            "production_selector_changed":false,
                            "logical_weight_bytes":outputs*(inputs/32)*34, "device_l2_bytes":l2_bytes,
                            "event_tracking":false, "mode":"graph_replay",
                            "quad":formal_quad, "order":position,
                            "warm_quads":WARM_QUADS, "formal_quads":FORMAL_QUADS,
                            "graph_nodes":graphs[index].node_count,
                            "graph_replays":1, "kernels_per_graph":KERNELS_PER_GRAPH,
                            "wall_ns":wall_ns, "gpu_ns":gpu_ns,
                            "wall_ns_per_kernel":wall_ns/f64::from(KERNELS_PER_GRAPH),
                            "gpu_ns_per_kernel":gpu_ns/f64::from(KERNELS_PER_GRAPH),
                            "validation":"generic bits plus F64 bounds, full output guards and immutable input/weight bytes"
                        })
                    );
                }
            }
        }
        // Captures reference this exact fixture; all graph handles die first.
        drop(graphs);
    }
}
