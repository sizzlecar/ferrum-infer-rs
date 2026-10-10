use super::*;

const SHAPES: [(usize, usize); 4] = [(6144, 5120), (5120, 1024), (5120, 17408), (17408, 5120)];
const ROWS: [usize; 4] = [4, 8, 16, 32];
const WARM: usize = 8;
const PAIRS: usize = 16;

#[test]
#[ignore = "exclusive CUDA synthetic Q6 F16 full-adapter vs strict paired graphs"]
fn q6_f16_adapter_full_cost_vs_strict_paired_working_sets() {
    let (ctx, stream, kernels) = context();
    let l2 = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap() as usize;
    assert!(l2 > 0);
    println!(
        "{}",
        serde_json::json!({"kind":"q6_f16_adapter_paired_plan","synthetic":true,
        "shapes":SHAPES,"rows":ROWS,"regimes":["single_matrix_repeated","distinct_allocation_ring"],
        "input_mode":"dense deterministic F16","matrix_contents":"same deterministic compressed Q6 bytes at distinct addresses in ring",
        "warm_pairs":WARM,"formal_pairs":PAIRS,"order":"AB BA balanced forward/reverse pairs",
        "minimum_projections_per_graph":32,"ring_minimum_weight_working_set":"strictly greater than 3 times actual device L2",
        "timed_adapter":"F16 widening/padding, fresh row flags and D4 pack/guard reset, Q6 dot/fixup, F16 marker cast reading row/weight flags",
        "timed_strict":"unchanged production Q6 F16 T8 math on identical physical weight allocation and logical inputs",
        "weight_validation":"immutable allocation scanned once before capture; scan wall/GPU separately recorded; replay uses retained zero flag",
        "excluded":"plan/allocation/upload/capture/static weight scan/oracle/readbacks and provider/conditional-dependency host orchestration",
        "new_product_route":false,"model_quality":"not evaluated","performance_threshold":null})
    );
    for (k, n) in SHAPES {
        let weight_bytes = n * (k / 256) * 210;
        for m in ROWS {
            for ring in [false, true] {
                let count = if ring { (3 * l2) / weight_bytes + 1 } else { 1 };
                let cases: Vec<_> = (0..count)
                    .map(|_| Case::new(&stream, m, k, n, false, n == 17408))
                    .collect();
                let addresses: Vec<_> = cases
                    .iter()
                    .map(|c| c.weight_ptr(&stream) as usize)
                    .collect();
                let mut unique = addresses.clone();
                unique.sort_unstable();
                unique.dedup();
                assert_eq!(unique.len(), count);
                if ring {
                    assert!(count * weight_bytes > 3 * l2);
                }
                let (scan_wall_ns, scan_gpu_ns) = measure(&stream, || {
                    for c in &cases {
                        c.scan(&stream);
                    }
                });
                let mut stable = Vec::with_capacity(count);
                let mut oracle_evidence = Vec::with_capacity(count);
                for c in &cases {
                    c.inclusive(&stream);
                    c.strict(&stream, &kernels);
                    stream.synchronize().unwrap();
                    oracle_evidence.push(c.validate(&stream, false));
                    stable.push((c.strict_output.read(&stream), c.output.read(&stream)));
                }
                let laps = 32_usize.div_ceil(count);
                let projections = laps * count;
                let graphs = [false, true].map(|adapter| {
                    Captured::new(&stream, || {
                        for _ in 0..laps {
                            for c in &cases {
                                if adapter {
                                    c.inclusive(&stream);
                                } else {
                                    c.strict(&stream, &kernels);
                                }
                            }
                        }
                    })
                });
                let mut samples = Vec::with_capacity(PAIRS * 2);
                for pair in 0..WARM + PAIRS {
                    for order in 0..2 {
                        let route = balanced_route(pair, order, 2);
                        let (wall_ns, gpu_ns) = measure(&stream, || graphs[route].launch());
                        if pair >= WARM {
                            samples.push((pair - WARM, order, route, gpu_ns, wall_ns));
                        }
                    }
                }
                for (index, c) in cases.iter().enumerate() {
                    assert_eq!(
                        stable[index].0,
                        c.strict_output.read(&stream),
                        "strict full-output replay bits"
                    );
                    assert_eq!(
                        stable[index].1,
                        c.output.read(&stream),
                        "adapter full-output replay bits"
                    );
                    c.validate(&stream, false);
                }
                for (pair, order, route, gpu_ns, wall_ns) in samples {
                    println!(
                        "{}",
                        serde_json::json!({"kind":"q6_f16_adapter_paired","m":m,"k":k,"n":n,
                        "weight_mode":if ring {"distinct_allocation_ring"} else {"single_matrix_repeated"},
                        "weight_bytes":weight_bytes,"matrix_count":count,"working_set_bytes":count*weight_bytes,
                        "device_l2_bytes":l2,"working_set_fits_l2":count*weight_bytes<=l2,
                        "same_physical_weights_between_routes":true,"input_mode":"dense F16","input_stride":k,
                        "output_stride":cases[0].ys,"output_offset":cases[0].output_offset,
                        "laps":laps,"projections":projections,"warm_pairs":WARM,"pairs":PAIRS,
                        "pair":pair,"order":order,"route":if route==0 {"strict_f16_t8"} else {"f16_d4_q6_mmq_marker"},
                        "gpu_ns":gpu_ns,"wall_ns":wall_ns,"gpu_ns_per_projection":gpu_ns/projections as f64,
                        "wall_ns_per_projection":wall_ns/projections as f64,"graph_nodes":graphs[route].node_count,
                        "static_scan_gpu_ns_all_matrices":scan_gpu_ns,"static_scan_wall_ns_all_matrices":scan_wall_ns,
                        "all_outputs_finite":true,"all_adapter_cast_bits_checked":true,"same_route_full_output_repeat_bits":true,
                        "oracle_samples_per_matrix":oracle_evidence[0]["checked_f64_outputs"],
                        "cross_algorithm_bit_equality_required":false,"quality_status":"synthetic diagnostic only"})
                    );
                }
                // The captured commands borrow these exact allocations. Destroy
                // both executables before any per-matrix input/scratch/flag dies.
                drop(graphs);
                drop(cases);
            }
        }
    }
}
