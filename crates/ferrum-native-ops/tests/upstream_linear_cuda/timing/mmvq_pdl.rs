//! Complete MMVQ projection diagnostic with PDL only on quantize -> dot.
//! Clear, F16 widening, MarkerV2 cast and all scratch/weight scopes stay ordinary.
use super::*;

unsafe extern "C" {
    fn ferrum_test_upstream_mmvq_pdl_support_v1(
        plan: *const UpstreamLinearPlanV1,
        supported: *mut u32,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_pdl_pack_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_pdl_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_pdl_graph_edges_v1(
        plan: *const UpstreamLinearPlanV1,
        graph: *mut c_void,
        total: *mut u64,
        programmatic: *mut u64,
        quant_dot: *mut u64,
    ) -> i32;
}

fn ordinary_ops(format: UpstreamLinearFormat, rows: usize) -> MmvqOps {
    MmvqOps {
        pack: ffi::ferrum_upstream_mmvq_pack_v2,
        dot: if rows == 8 {
            shared_weight_decode::selected_dot(format)
        } else {
            ffi::ferrum_upstream_mmvq_dot_v1
        },
    }
}

fn supported(route: &Route) -> bool {
    let mut supported = 0;
    unsafe {
        ok(ferrum_test_upstream_mmvq_pdl_support_v1(
            &route.plan,
            &mut supported,
        ))
    };
    supported != 0
}

fn pdl_ops() -> MmvqOps {
    MmvqOps {
        pack: ferrum_test_upstream_mmvq_pdl_pack_v2,
        dot: ferrum_test_upstream_mmvq_pdl_dot_v1,
    }
}

fn inspect_graph(route: &Route, arm: usize, graph: *mut c_void, projections: usize) {
    let (mut total, mut programmatic, mut quant_dot) = (0, 0, 0);
    unsafe {
        ok(ferrum_test_upstream_mmvq_pdl_graph_edges_v1(
            &route.plan,
            graph,
            &mut total,
            &mut programmatic,
            &mut quant_dot,
        ));
    }
    if arm == 0 {
        assert_eq!(programmatic, 0, "ordinary capture unexpectedly used PDL");
        assert_eq!(quant_dot, 0);
    } else {
        assert_eq!(quant_dot, projections as u64,
            "every candidate projection must capture its quantizer -> selected dot programmatic edge");
    }
    println!(
        "{}",
        serde_json::json!({
            "experiment":"mmvq_partial_pdl", "kind":"capture_edges", "arm":arm,
            "M":route.plan.request.rows,"K":route.plan.request.inputs,"N":route.plan.request.outputs,
            "edges":total,"programmatic_edges":programmatic,"quantize_to_selected_dot_edges":quant_dot,
            "projections":projections,
            "inspection":"cudaGraphGetEdges_v2 with actual edgeData, kernel function pointers and programmatic source port"
        })
    );
}

fn equal_buffers(a: &Route, b: &Route) {
    for (a, b) in [
        (&a.converted, &b.converted),
        (&a.packed, &b.packed),
        (&a.output, &b.output),
        (&a.result, &b.result),
        (&a.rows, &b.rows),
        (&a.flags, &b.flags),
    ] {
        assert_eq!(a.bytes, b.bytes);
        assert_eq!(a.read::<u8>(a.bytes), b.read::<u8>(b.bytes));
        a.guards();
        b.guards();
    }
}

fn enqueue(route: &Route, stream: &Stream, input: &Buffer, weight: *const c_void, ops: MmvqOps) {
    route.enqueue_with_native_ops(
        stream,
        input,
        weight,
        0,
        ffi::ferrum_upstream_mmq_dot_v1,
        ops,
    );
}

fn boundary(format: UpstreamLinearFormat, rows: usize, k: usize, n: usize) {
    let stream = Stream::new();
    let host_weights = weights(format, k, n);
    let weight = Buffer::new(host_weights.len());
    weight.write(&host_weights);
    let input = Buffer::new(rows * k * 2);
    let routes = std::array::from_fn::<_, 2, _>(|_| {
        Route::new_with_result_window(
            UpstreamLinearAlgorithm::Mmvq,
            format,
            rows as u32,
            k as u32,
            n as u32,
            1,
            (n + 7) as u32,
            3,
        )
    });
    let pdl = supported(&routes[1]);
    let normal = ordinary_ops(format, rows);
    let ops = [normal, if pdl { pdl_ops() } else { normal }];
    for route in &routes {
        route.scan(&stream, weight.ptr, 0);
    }
    stream.sync();
    let write_input = |generation: usize, nonfinite: bool| {
        let mut words = (0..rows * k)
            .map(|i| {
                f16::from_f32(((i * 17 + generation * 29) % 113) as f32 / 256.0 - 0.2).to_bits()
            })
            .collect::<Vec<_>>();
        if nonfinite {
            for row in 0..rows {
                words[row * k + row % k] = f16::NAN.to_bits();
            }
        }
        input.write(&words);
    };
    write_input(0, false);
    for arm in 0..2 {
        enqueue(&routes[arm], &stream, &input, weight.ptr, ops[arm]);
    }
    stream.sync();
    equal_buffers(&routes[0], &routes[1]);
    let initial = routes[0].validate(format, &host_weights, 1);
    routes[1].validate(format, &host_weights, 1);
    let graphs = [0, 1].map(|arm| {
        Graph::capture_inspected(
            &stream,
            || enqueue(&routes[arm], &stream, &input, weight.ptr, ops[arm]),
            |graph| {
                if pdl {
                    inspect_graph(&routes[arm], arm, graph, 1);
                }
            },
        )
    });
    // Captured pointers stay fixed while data changes. Poison then finite replay
    // also proves that ordinary clear runs before marker atomics on every replay.
    for (generation, nonfinite) in [(1, false), (2, true), (3, false)] {
        write_input(generation, nonfinite);
        for graph in &graphs {
            graph.launch(&stream);
        }
        stream.sync();
        equal_buffers(&routes[0], &routes[1]);
        let flags = routes[1].rows.read::<u32>(rows);
        if nonfinite {
            assert!(flags.iter().all(|&flag| flag != 0));
            assert!(routes[1]
                .result
                .read::<u16>(routes[1].result.bytes / 2)
                .chunks_exact(n + 7)
                .all(|row| row[3..3 + n].iter().all(|&b| f16::from_bits(b).is_nan())));
        } else {
            assert_eq!(flags, vec![0; rows]);
            assert_ne!(
                routes[0].validate(format, &host_weights, 1),
                initial,
                "capture did not observe changed input"
            );
            routes[1].validate(format, &host_weights, 1);
        }
        input.guards();
        weight.guards();
    }
    println!(
        "{}",
        serde_json::json!({
            "experiment":"mmvq_partial_pdl","kind":"boundary","format":format!("{format:?}"),
            "M":rows,"K":k,"N":n,"pdl_supported":pdl,
            "correctness":"eager then captured finite/nonfinite/finite fresh inputs; full converted/packed/rawF32/F16/flags/guards bit equality; finite actual-pack F64 oracle",
            "fallback":"ordinary pack/dot when actual CC or loaded PTX lacks PDL; no active-PDL performance claim"
        })
    );
}

#[test]
#[ignore = "exclusive CUDA complete-MMVQ partial-PDL diagnostic"]
fn mmvq_partial_pdl_preserves_bits_and_complete_projection_pairs() {
    let _ = caps();
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for rows in [1, 8] {
            // The timing Route owns complete physical rows, including M1's
            // small-K four-row CTA. Keep odd output stride and nonzero offset.
            for (k, n) in [(256, 32), (768, 64)] {
                boundary(format, rows, k, n);
            }
        }
        // M1's two real FFN shapes. M8 is a same-arithmetic correctness scope
        // above (SharedM8 for Q4/Q5, ordinary MMVQ for IQ4), not an FFN route change.
        for (k, n) in [(5120, 17408), (17408, 5120)] {
            let probe = Route::new(UpstreamLinearAlgorithm::Mmvq, format, 1, k, n, 1);
            let active = supported(&probe);
            drop(probe);
            if !active {
                println!(
                    "{}",
                    serde_json::json!({"experiment":"mmvq_partial_pdl","kind":"unsupported",
                    "format":format!("{format:?}"),"M":1,"K":k,"N":n,"timing_skipped":true})
                );
                continue;
            }
            let normal = ordinary_ops(format, 1);
            complete_projection::paired_projection_ring_with_ops(
                "mmvq_partial_pdl_complete_projection",
                format,
                1,
                UpstreamLinearAlgorithm::Mmvq,
                k as usize,
                n as usize,
                [normal, pdl_ops()],
                "mmvq_quantize_dot_pdl",
                Some(inspect_graph),
            );
        }
    }
}
