//! Same frozen kernels in balanced direct/graph timing, with direct pack-only
//! observations. R2's test and its implicit event-tracking setting stay intact.

pub(super) use super::super::super::captured::{measure, Captured};
use super::*;

const ITERATIONS: u32 = 32;
const WARM_ROUNDS: usize = 4;

fn prepare(
    case: &mut Fixture,
    stream: &Arc<CudaStream>,
    experiment: &Experiment,
    route: Option<Route>,
) {
    case.reset(stream);
    if route.is_some_and(|r| !r.includes_quantization()) {
        case.pack(stream, experiment);
    }
    stream.synchronize().unwrap();
}

fn enqueue(
    case: &mut Fixture,
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &Experiment,
    route: Option<Route>,
    tile: u32,
) {
    for _ in 0..ITERATIONS {
        if route.is_none_or(Route::includes_quantization) {
            case.pack(stream, experiment);
        }
        if let Some(route) = route {
            case.multiply(stream, kernels, experiment, route, tile);
        }
    }
}

pub(super) fn balanced_route(pair: usize, position: usize, count: usize) -> usize {
    let rotation = pair % count;
    if pair / count % 2 == 0 {
        (rotation + position) % count
    } else {
        (rotation + count - position) % count
    }
}

#[test]
#[ignore = "balanced direct/graph GPU timing; coordinate exclusive CUDA access"]
fn q8dot_balanced_direct_graph_and_pack_microbench() {
    run_formats(&[GgufBlockFormat::Iq4Xs, GgufBlockFormat::Q4K], &Route::ALL);
}

#[test]
#[ignore = "Q5-only balanced direct/graph GPU timing; coordinate exclusive CUDA access"]
fn q5_q8_group32_balanced_direct_graph_and_pack_microbench() {
    run_formats(&[GgufBlockFormat::Q5K], &Route::Q5);
}

fn run_formats(formats: &[GgufBlockFormat], routes: &[Route]) {
    // Two Latin cycles (forward/reverse); Q5 has three routes and six pairs,
    // while the existing two-format diagnostic keeps five routes/ten pairs.
    let pairs = 2 * routes.len();
    let context = CudaContext::new(0).expect("Q8 graph diagnostic requires CUDA");
    // Match vnext_runtime::new before creating streams or allocating buffers.
    // SAFETY: one test thread and one stream own every buffer and explicit fence.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let experiment = Experiment::load(&context);
    for &format in formats {
        for (inputs, outputs) in [(5120, 6144), (5120, 17408), (17408, 5120)] {
            for rows in [1, 8] {
                let tile = if rows == 1 { 1 } else { 8 };
                let mut case = Fixture::new(&stream, format, rows, inputs, outputs);
                let mut stable = vec![None; routes.len()];
                // Warm every kernel before capture. These allocations stay fixed;
                // graphs are declared after case and explicitly dropped first.
                let graphs = routes
                    .iter()
                    .copied()
                    .map(|route| {
                        case.run(&stream, &kernels, &experiment, route, tile, 1);
                        case.validate(&stream, route);
                        prepare(&mut case, &stream, &experiment, Some(route));
                        Captured::new(&stream, || {
                            enqueue(&mut case, &stream, &kernels, &experiment, Some(route), tile);
                        })
                    })
                    .collect::<Vec<_>>();
                prepare(&mut case, &stream, &experiment, None);
                let pack_graph = Captured::new(&stream, || {
                    enqueue(&mut case, &stream, &kernels, &experiment, None, tile);
                });
                for round in 0..WARM_ROUNDS + pairs {
                    let formal = round.checked_sub(WARM_ROUNDS);
                    let pair = formal.unwrap_or(round);
                    for position in 0..routes.len() {
                        let index = balanced_route(pair, position, routes.len());
                        let route = routes[index];
                        for mode_position in 0..2 {
                            let graph_mode = (pair + index + mode_position) % 2 == 1;
                            prepare(&mut case, &stream, &experiment, Some(route));
                            let (wall_ns, gpu_ns) = measure(&stream, || {
                                if graph_mode {
                                    graphs[index].launch();
                                } else {
                                    enqueue(
                                        &mut case,
                                        &stream,
                                        &kernels,
                                        &experiment,
                                        Some(route),
                                        tile,
                                    );
                                }
                            });
                            let (bits, errors) = case.validate(&stream, route);
                            if let Some(reference) = &stable[index] {
                                assert_eq!(
                                    &bits, reference,
                                    "direct/graph changed {route:?} output bits"
                                );
                            } else {
                                stable[index] = Some(bits);
                            }
                            if formal.is_some() {
                                println!(
                                    "{}",
                                    serde_json::json!({
                                        "experiment":"activation_q8_balanced_direct_graph",
                                        "format":format!("{format:?}"),"input":inputs,"output":outputs,"rows":rows,"row_tile":tile,
                                        "route":route.name(),"integer_partial_values":route.integer_partial_values(),
                                        "mode":if graph_mode {"graph_replay"} else {"direct"},"event_tracking":false,
                                        "pair":pair,"order":position,"mode_order":mode_position,
                                        "warm_rounds":WARM_ROUNDS,"measured_pairs":pairs,"iterations":ITERATIONS,
                                        "graph_nodes":if graph_mode {Some(graphs[index].node_count)} else {None},
                                        "graph_replays":if graph_mode {1} else {0},
                                        "quantization_included":route.includes_quantization(),"wall_ns":wall_ns,"gpu_ns":gpu_ns,"errors":errors
                                    })
                                );
                            }
                        }
                    }
                    // Pack-only is a separate observation after the balanced
                    // projection routes, never a difference of two elapsed times.
                    for mode_position in 0..2 {
                        let graph_mode = (pair + mode_position) % 2 == 1;
                        prepare(&mut case, &stream, &experiment, None);
                        // reset deliberately leaves NaNs in logical output and
                        // sentinels in guards. Preserve both bitwise; comparing
                        // the whole allocation to one sentinel is incorrect.
                        let output_before = stream
                            .clone_dtoh(&case.output_gpu)
                            .unwrap()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>();
                        let (wall_ns, gpu_ns) = measure(&stream, || {
                            if graph_mode {
                                pack_graph.launch();
                            } else {
                                enqueue(&mut case, &stream, &kernels, &experiment, None, tile);
                            }
                        });
                        case.validate_pack(&stream);
                        assert_eq!(
                            stream
                                .clone_dtoh(&case.output_gpu)
                                .unwrap()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            output_before,
                            "pack-only overwrote output"
                        );
                        if formal.is_some() {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "experiment":"activation_q8_pack_only",
                                    "format_context":format!("{format:?}"),"input":inputs,"output_context":outputs,"rows":rows,
                                    "mode":if graph_mode {"graph_replay"} else {"direct"},"event_tracking":false,
                                    "pair":pair,"mode_order":mode_position,"warm_rounds":WARM_ROUNDS,
                                    "measured_pairs":pairs,"iterations":ITERATIONS,
                                    "graph_nodes":if graph_mode {Some(pack_graph.node_count)} else {None},
                                    "graph_replays":if graph_mode {1} else {0},"wall_ns":wall_ns,"gpu_ns":gpu_ns,
                                    "validation":"all qwords/scale bits, guards, input/weight immutability and untouched output"
                                })
                            );
                        }
                    }
                }
                drop(pack_graph);
                drop(graphs);
            }
        }
    }
}

#[cfg(feature = "cuda-upstream-q6-f32-linear")]
mod q6_mmq_f32;
