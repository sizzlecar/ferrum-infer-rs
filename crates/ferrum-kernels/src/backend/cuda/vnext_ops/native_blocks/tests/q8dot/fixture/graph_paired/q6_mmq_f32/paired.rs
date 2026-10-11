use super::*;
#[test]
#[ignore = "synthetic head geometry paired timing; exclusive CUDA; ~1.05GB compressed host weights"]
fn q6_mmq_f32_head_inclusive_vs_original_strict_paired_on_cuda() {
    let (ctx, s, kernels) = context();
    let l2 = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap() as u64;
    // Actual head K/N, synthetic F32 values and compressed Q6 blocks. Do not
    // decode a dense N*K host matrix. Bounded-N test owns exhaustive correctness.
    for m in [1, 4, 8, 16, 17, 24, 31, 32] {
        let c = Case::new(&s, m, 5120, 248320, false);
        let (scan_wall, scan_gpu) = measure(&s, || c.scan(&s));
        c.inclusive(&s);
        c.strict(&s, &kernels);
        s.synchronize().unwrap();
        let oracle = c.oracle(&s, true);
        let initial_mmq = c.output.read(&s);
        let initial_strict = c.strict_output.read(&s);
        for output in [&initial_mmq, &initial_strict] {
            assert!(floats(output)
                .chunks(c.ys)
                .all(|r| r[..c.n].iter().all(|v| v.is_finite())));
        }
        c.guards(&s);
        const ITERATIONS: usize = 8;
        let strict = Captured::new(&s, || {
            for _ in 0..ITERATIONS {
                c.strict(&s, &kernels);
            }
        });
        let inclusive = Captured::new(&s, || {
            for _ in 0..ITERATIONS {
                c.inclusive(&s);
            }
        });
        let pack = Captured::new(&s, || {
            for _ in 0..ITERATIONS {
                c.pack(&s);
            }
        });
        let dot_publish = Captured::new(&s, || {
            for _ in 0..ITERATIONS {
                c.dot(&s);
            }
        });
        for _ in 0..4 {
            strict.launch();
            inclusive.launch();
        }
        s.synchronize().unwrap();
        let mut samples = Vec::new();
        for pair in 0..12 {
            for position in 0..2 {
                let route = balanced_route(pair, position, 2);
                let (wall, gpu) = measure(&s, || {
                    if route == 0 {
                        strict.launch()
                    } else {
                        inclusive.launch()
                    }
                });
                let route_name = if route == 0 {
                    if m == 1 {
                        "original_strict_scalar"
                    } else {
                        "original_strict_t8"
                    }
                } else {
                    "f32_pad_pack_mmq_publish"
                };
                samples.push(serde_json::json!({"pair":pair,"position":position,"route":route_name,"wall_ns_per_projection":wall/ITERATIONS as f64,"gpu_ns_per_projection":gpu/ITERATIONS as f64}));
            }
        }
        let (pack_wall, pack_gpu) = measure(&s, || pack.launch());
        let (dot_wall, dot_gpu) = measure(&s, || dot_publish.launch());
        c.guards(&s);
        assert_eq!(
            initial_mmq,
            c.output.read(&s),
            "head inclusive/dot replay bits"
        );
        assert_eq!(
            initial_strict,
            c.strict_output.read(&s),
            "head original strict route replay bits"
        );
        println!(
            "{}",
            serde_json::json!({"kind":"q6_mmq_f32_head_paired","synthetic":true,"m":m,"k":c.k,"n":c.n,"j":c.p.j,"fixup":c.p.fixup,
            "weight_bytes":c.p.weight_bytes,"actual_l2_bytes":l2,"weights_exceed_three_l2":c.p.weight_bytes>3*l2,
            "iterations_per_graph":ITERATIONS,"static_scan_gpu_ns":scan_gpu,"static_scan_wall_ns":scan_wall,
            "pack_gpu_ns_per_projection":pack_gpu/ITERATIONS as f64,"pack_wall_ns_per_projection":pack_wall/ITERATIONS as f64,
            "dot_publish_gpu_ns_per_projection":dot_gpu/ITERATIONS as f64,"dot_publish_wall_ns_per_projection":dot_wall/ITERATIONS as f64,
            "samples":samples,"oracle":oracle,"all_logical_outputs_finite":true,"same_route_full_output_repeat_bits":true,"performance_threshold":null,"semantic_quality":"not evaluated"})
        );
        // Captured values drop (and synchronize) before case-owned allocations.
    }
}
