use super::*;

const ITERATIONS: u32 = 32;
const WARM_ROUNDS: usize = 4;
const PAIRS: usize = 10;

#[test]
#[ignore = "exclusive CUDA diagnostic; balanced resident IQ4 layout direct/graph and pack"]
fn iq4_group32_word_major_balanced_direct_graph_and_pack_microbench() {
    balanced_formats(&[GgufBlockFormat::Iq4Xs], "iq4");
}

#[test]
#[ignore = "exclusive CUDA diagnostic; balanced resident Q4/Q5 layout direct/graph and pack"]
fn q4_q5_group32_word_major_balanced_direct_graph_and_pack_microbench() {
    balanced_formats(&[GgufBlockFormat::Q4K, GgufBlockFormat::Q5K], "q4q5");
}

fn balanced_formats(formats: &[GgufBlockFormat], label: &str) {
    let context = CudaContext::new(0).expect("layout timing requires CUDA");
    // Match production before creating any stream/allocation. Both direct and
    // graph modes use this setting; prior archived tests retain their behavior.
    // SAFETY: one stream/test thread owns all allocations with explicit fences.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let l2_bytes = context
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap();
    for &format in formats {
        let experiment = LayoutExperiment::load(&context, format);
        for (inputs, outputs) in [(5120, 6144), (5120, 17408), (17408, 5120)] {
            for rows in [1, 8] {
                let tile = if rows == 1 { 1 } else { 8 };
                let mut case = Fixture::new(&stream, format, rows, inputs, outputs);
                let weight_bytes = inputs / 256 * outputs * format.block_bytes();
                // Allocation/reference preparation and graph capture/upload stay
                // outside timing. Weights are reused, never described as streaming.
                let graphs = LayoutRoute::ALL
                    .iter()
                    .copied()
                    .map(|route| {
                        experiment.prepare(&mut case, &stream, route);
                        experiment.enqueue(&mut case, &stream, &kernels, route, tile, 1);
                        experiment.validate(&case, &stream, route);
                        experiment.prepare(&mut case, &stream, route);
                        Captured::new(&stream, || {
                            experiment
                                .enqueue(&mut case, &stream, &kernels, route, tile, ITERATIONS);
                        })
                    })
                    .collect::<Vec<_>>();
                let layouts = [QwordLayout::GroupMajor, QwordLayout::WordMajor];
                let pack_graphs = layouts
                    .iter()
                    .copied()
                    .map(|layout| {
                        case.reset(&stream);
                        experiment.pack(&mut case, &stream, layout);
                        case.validate_pack_layout(&stream, layout);
                        case.reset(&stream);
                        stream.synchronize().unwrap();
                        Captured::new(&stream, || {
                            enqueue_pack(&mut case, &stream, &experiment, layout);
                        })
                    })
                    .collect::<Vec<_>>();
                let mut stable_f32 = None;
                let mut stable_g32 = None;
                for round in 0..WARM_ROUNDS + PAIRS {
                    let formal = round.checked_sub(WARM_ROUNDS);
                    let pair = formal.unwrap_or(round);
                    for position in 0..LayoutRoute::ALL.len() {
                        let index = balanced_route(pair, position, LayoutRoute::ALL.len());
                        let route = LayoutRoute::ALL[index];
                        for mode_order in 0..2 {
                            let graph_mode = (pair + index + mode_order) % 2 == 1;
                            experiment.prepare(&mut case, &stream, route);
                            let (wall_ns, gpu_ns) = measure(&stream, || {
                                if graph_mode {
                                    graphs[index].launch();
                                } else {
                                    experiment.enqueue(
                                        &mut case, &stream, &kernels, route, tile, ITERATIONS,
                                    );
                                }
                            });
                            let (bits, errors) = experiment.validate(&case, &stream, route);
                            let stable = if matches!(route, LayoutRoute::RetainedF32) {
                                &mut stable_f32
                            } else {
                                &mut stable_g32
                            };
                            if let Some(reference) = stable.as_ref() {
                                assert_eq!(
                                    &bits, reference,
                                    "layout/direct/graph changed route bits"
                                );
                            } else {
                                *stable = Some(bits);
                            }
                            if formal.is_some() {
                                println!(
                                    "{}",
                                    serde_json::json!({
                                        "experiment":format!("{label}_qword_layout_balanced_direct_graph"),
                                        "format":format!("{format:?}"),"input":inputs,"output":outputs,"rows":rows,"row_tile":tile,
                                        "route":route.name(),"qword_layout":format!("{:?}",route.layout()),
                                        "integer_partial_values":route.numerical_route().integer_partial_values(),
                                        "mode":if graph_mode {"graph_replay"} else {"direct"},"event_tracking":false,
                                        "pair":pair,"order":position,"mode_order":mode_order,
                                        "warm_rounds":WARM_ROUNDS,"measured_pairs":PAIRS,"iterations":ITERATIONS,
                                        "graph_nodes":if graph_mode {Some(graphs[index].node_count)} else {None},
                                        "graph_replays":if graph_mode {1} else {0},"quantization_included":route.includes_pack(),
                                        "weight_mode":"resident_reused","logical_weight_bytes":weight_bytes,"device_l2_bytes":l2_bytes,
                                        "wall_ns":wall_ns,"gpu_ns":gpu_ns,"errors":errors
                                    })
                                );
                            }
                        }
                    }
                    // Separately measure both pack layouts with alternating order.
                    // These are observations, never inclusive-minus-dot estimates.
                    for position in 0..layouts.len() {
                        let index = (pair + position) % layouts.len();
                        let layout = layouts[index];
                        for mode_order in 0..2 {
                            let graph_mode = (pair + index + mode_order) % 2 == 1;
                            case.reset(&stream);
                            stream.synchronize().unwrap();
                            let before = output_bits(&case, &stream);
                            let (wall_ns, gpu_ns) = measure(&stream, || {
                                if graph_mode {
                                    pack_graphs[index].launch();
                                } else {
                                    enqueue_pack(&mut case, &stream, &experiment, layout);
                                }
                            });
                            case.validate_pack_layout(&stream, layout);
                            assert_eq!(
                                output_bits(&case, &stream),
                                before,
                                "pack changed output bits"
                            );
                            if formal.is_some() {
                                println!(
                                    "{}",
                                    serde_json::json!({
                                        "experiment":format!("{label}_qword_layout_pack_only"),
                                        "format":format!("{format:?}"),"input":inputs,"output_context":outputs,"rows":rows,
                                        "qword_layout":format!("{layout:?}"),
                                        "mode":if graph_mode {"graph_replay"} else {"direct"},"event_tracking":false,
                                        "pair":pair,"order":position,"mode_order":mode_order,
                                        "warm_rounds":WARM_ROUNDS,"measured_pairs":PAIRS,"iterations":ITERATIONS,
                                        "graph_nodes":if graph_mode {Some(pack_graphs[index].node_count)} else {None},
                                        "graph_replays":if graph_mode {1} else {0},"weight_mode":"resident_context",
                                        "wall_ns":wall_ns,"gpu_ns":gpu_ns,
                                        "validation":"all qwords/scale bits, guards, input/weight immutability and untouched output"
                                    })
                                );
                            }
                        }
                    }
                }
                // Drop graph resources before captured Fixture allocations.
                drop(pack_graphs);
                drop(graphs);
            }
        }
    }
}

fn enqueue_pack(
    case: &mut Fixture,
    stream: &Arc<CudaStream>,
    experiment: &LayoutExperiment,
    layout: QwordLayout,
) {
    for _ in 0..ITERATIONS {
        experiment.pack(case, stream, layout);
    }
}
