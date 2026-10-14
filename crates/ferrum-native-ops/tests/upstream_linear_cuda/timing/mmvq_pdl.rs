//! Complete MMVQ projection diagnostics for partial and complete-chain PDL.
//! Both retain ordinary math, scratch extents and once-prepared weight flags.
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
    fn ferrum_test_upstream_mmvq_full_pdl_support_v1(
        plan: *const UpstreamLinearPlanV1,
        supported: *mut u32,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_full_pdl_pack_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_full_pdl_cast_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        weights: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_full_pdl_graph_edges_v1(
        plan: *const UpstreamLinearPlanV1,
        graph: *mut c_void,
        counts: *mut u64,
        counts_len: usize,
    ) -> i32;
    fn cudaMemsetAsync(pointer: *mut c_void, value: i32, bytes: usize, stream: *mut c_void) -> i32;
}

#[derive(Clone, Copy)]
enum PdlMode {
    Partial,
    Full,
}

impl PdlMode {
    fn experiment(self) -> &'static str {
        match self {
            Self::Partial => "mmvq_partial_pdl",
            Self::Full => "mmvq_full_pdl",
        }
    }
    fn supported(self, route: &Route) -> bool {
        let mut supported = 0;
        unsafe {
            ok(match self {
                Self::Partial => {
                    ferrum_test_upstream_mmvq_pdl_support_v1(&route.plan, &mut supported)
                }
                Self::Full => {
                    ferrum_test_upstream_mmvq_full_pdl_support_v1(&route.plan, &mut supported)
                }
            });
        }
        supported != 0
    }
    fn ops(self) -> MmvqOps {
        MmvqOps {
            pack: match self {
                Self::Partial => ferrum_test_upstream_mmvq_pdl_pack_v2,
                Self::Full => ferrum_test_upstream_mmvq_full_pdl_pack_v2,
            },
            dot: ferrum_test_upstream_mmvq_pdl_dot_v1,
            cast: match self {
                Self::Partial => ffi::ferrum_upstream_mmvq_cast_v2,
                Self::Full => ferrum_test_upstream_mmvq_full_pdl_cast_v2,
            },
        }
    }
    fn inspector(self) -> fn(&Route, usize, *mut c_void, usize) {
        match self {
            Self::Partial => inspect_graph,
            Self::Full => inspect_full_graph,
        }
    }
}

fn ordinary_ops(format: UpstreamLinearFormat, rows: usize) -> MmvqOps {
    MmvqOps {
        pack: ffi::ferrum_upstream_mmvq_pack_v2,
        dot: if rows == 8 {
            shared_weight_decode::selected_dot(format)
        } else {
            ffi::ferrum_upstream_mmvq_dot_v1
        },
        cast: ffi::ferrum_upstream_mmvq_cast_v2,
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

fn full_edges(route: &Route, graph: *mut c_void) -> [u64; 7] {
    let mut counts = [0; 7];
    unsafe {
        ok(ferrum_test_upstream_mmvq_full_pdl_graph_edges_v1(
            &route.plan,
            graph,
            counts.as_mut_ptr(),
            counts.len(),
        ));
    }
    counts
}

fn inspect_full_graph(route: &Route, arm: usize, graph: *mut c_void, projections: usize) {
    let counts = full_edges(route, graph);
    if arm == 0 {
        assert_eq!(counts[1], 0, "ordinary capture unexpectedly used PDL");
    } else {
        assert_eq!(counts[2..6], [projections as u64; 4]);
        assert_eq!(
            counts[6],
            projections as u64 - 1,
            "scratch reuse must retain the preceding cast -> next clear programmatic dependency"
        );
        assert_eq!(counts[1], counts[2..].iter().sum::<u64>());
    }
    println!(
        "{}",
        serde_json::json!({
            "experiment":"mmvq_full_pdl","kind":"capture_edges","arm":arm,
            "M":route.plan.request.rows,"K":route.plan.request.inputs,"N":route.plan.request.outputs,
            "edges":counts[0],"programmatic_edges":counts[1],"clear_to_widen":counts[2],
            "widen_to_quantize":counts[3],"quantize_to_dot":counts[4],"dot_to_cast":counts[5],
            "cast_to_next_clear":counts[6],"projections":projections,
            "inspection":"actual edgeData_v2, function pointer pairs and programmatic port for every stage; dot has implicit completion trigger"
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

fn boundary(format: UpstreamLinearFormat, rows: usize, k: usize, n: usize, mode: PdlMode) {
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
    let pdl = mode.supported(&routes[1]);
    let normal = ordinary_ops(format, rows);
    let ops = [normal, if pdl { mode.ops() } else { normal }];
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
                    mode.inspector()(&routes[arm], arm, graph, 1);
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
            "experiment":mode.experiment(),"kind":"boundary","format":format!("{format:?}"),
            "M":rows,"K":k,"N":n,"pdl_supported":pdl,
            "correctness":"eager then captured finite/nonfinite/finite fresh inputs; full converted/packed/rawF32/F16/flags/guards bit equality; finite actual-pack F64 oracle",
            "fallback":"ordinary pack/dot/cast when actual CC or any participating loaded kernel PTX lacks PDL; no active-PDL performance claim"
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
                boundary(format, rows, k, n, PdlMode::Partial);
            }
        }
        // M1's two real FFN shapes. M8 is a same-arithmetic correctness scope
        // above (SharedM8 for Q4/Q5, ordinary MMVQ for IQ4), not an FFN route change.
        for (k, n) in [(5120, 17408), (17408, 5120)] {
            paired(format, 1, k, n, PdlMode::Partial);
        }
    }
}

fn paired(format: UpstreamLinearFormat, rows: usize, k: u32, n: u32, mode: PdlMode) {
    let probe = Route::new(UpstreamLinearAlgorithm::Mmvq, format, rows as u32, k, n, 1);
    let active = mode.supported(&probe);
    drop(probe);
    if !active {
        println!(
            "{}",
            serde_json::json!({
                "experiment":mode.experiment(),"kind":"unsupported","format":format!("{format:?}"),
                "M":rows,"K":k,"N":n,"timing_skipped":true
            })
        );
        return;
    }
    let (experiment, label) = match mode {
        PdlMode::Partial => (
            "mmvq_partial_pdl_complete_projection",
            "mmvq_quantize_dot_pdl",
        ),
        PdlMode::Full => ("mmvq_full_pdl_complete_projection", "mmvq_full_chain_pdl"),
    };
    let normal = ordinary_ops(format, rows);
    complete_projection::paired_projection_ring_with_ops(
        experiment,
        format,
        rows,
        UpstreamLinearAlgorithm::Mmvq,
        k as usize,
        n as usize,
        [normal, mode.ops()],
        label,
        Some(mode.inspector()),
    );
}

// The first raw output is finite; only its deliberately nonzero row flags
// require a NaN cast. A following projection reuses the same scratch/flags.
// Preserve result1 separately so a premature clear cannot hide behind result2.
fn scratch_hazard() {
    let (format, rows, k, n) = (UpstreamLinearFormat::Q4K, 8, 256, 32);
    let stream = Stream::new();
    let host_weights = weights(format, k, n);
    let weight = Buffer::new(host_weights.len());
    weight.write(&host_weights);
    let inputs = [Buffer::new(rows * k * 2), Buffer::new(rows * k * 2)];
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
    let first_results = std::array::from_fn::<_, 2, _>(|_| Buffer::new(rows * (n + 7) * 2));
    let active = PdlMode::Full.supported(&routes[1]);
    let normal = ordinary_ops(format, rows);
    let ops = [normal, if active { PdlMode::Full.ops() } else { normal }];
    let write_inputs = |generation: usize| {
        for (input_index, input) in inputs.iter().enumerate() {
            input.write(
                &(0..rows * k)
                    .map(|i| {
                        f16::from_f32(
                            ((i * 19 + generation * 31 + input_index * 23) % 103) as f32 / 2048.0
                                - 0.024,
                        )
                        .to_bits()
                    })
                    .collect::<Vec<_>>(),
            );
        }
    };
    write_inputs(0);
    for route in &routes {
        route.scan(&stream, weight.ptr, 0);
        enqueue(route, &stream, &inputs[0], weight.ptr, normal);
    }
    stream.sync();
    for route in &routes {
        route.validate(format, &host_weights, 1);
        assert!(route
            .output
            .read::<f32>(rows * n)
            .iter()
            .all(|value| value.is_finite()));
    }
    let run = |arm: usize| unsafe {
        let route = &routes[arm];
        let p = &route.plan;
        ok((ops[arm].pack)(
            p,
            inputs[0].ptr,
            p.request.inputs,
            route.converted.ptr,
            route.packed.ptr,
            route.rows.ptr,
            stream.0,
        ));
        ok((ops[arm].dot)(
            p,
            weight.ptr,
            route.packed.ptr,
            route.output.ptr,
            ptr::null_mut(),
            stream.0,
        ));
        // Correctness-only node: a legal nonzero row-flags input to the cast.
        // It is never part of the paired timing observations.
        ok(cudaMemsetAsync(route.rows.ptr, 1, rows * 4, stream.0));
        ok((ops[arm].cast)(
            p,
            route.output.ptr,
            first_results[arm].ptr.add(route.result_offset as usize * 2),
            route.result_stride,
            route.rows.ptr,
            route.flags.ptr,
            stream.0,
        ));
        enqueue(route, &stream, &inputs[1], weight.ptr, ops[arm]);
    };
    let verify = || {
        equal_buffers(&routes[0], &routes[1]);
        assert_eq!(
            first_results[0].read::<u16>(rows * (n + 7)),
            first_results[1].read::<u16>(rows * (n + 7))
        );
        for arm in 0..2 {
            for row in first_results[arm]
                .read::<u16>(rows * (n + 7))
                .chunks_exact(n + 7)
            {
                assert_eq!(&row[..3], &[0x3535; 3]);
                assert!(row[3..3 + n].iter().all(|&word| word == 0x7e00));
                assert_eq!(&row[3 + n..], &[0x3535; 4]);
            }
            first_results[arm].guards();
            assert_eq!(routes[arm].rows.read::<u32>(rows), vec![0; rows]);
            routes[arm].validate(format, &host_weights, 1);
        }
        for input in &inputs {
            input.guards();
        }
        weight.guards();
    };
    for arm in 0..2 {
        run(arm);
    }
    stream.sync();
    verify();
    let previous = routes[0].validate(format, &host_weights, 1);
    let graphs = [0, 1].map(|arm| Graph::capture_inspected(&stream, || run(arm), |graph| {
        if active {
            let counts = full_edges(&routes[arm], graph);
            if arm == 0 { assert_eq!(counts[1], 0); }
            else {
                assert_eq!(counts[2..5], [2; 3]);
                assert_eq!(counts[5], 1); // The first dot->cast contains the explicit flag poison node.
                assert_eq!(counts[6], 1, "first cast must finish its flag reads before the second clear");
            }
            println!("{}", serde_json::json!({
                "experiment":"mmvq_full_pdl","kind":"scratch_hazard_edges","arm":arm,
                "edges":counts[0],"programmatic_edges":counts[1],"clear_to_widen":counts[2],
                "widen_to_quantize":counts[3],"quantize_to_dot":counts[4],"dot_to_cast":counts[5],
                "cast_to_next_clear":counts[6],"poison_memset":"correctness only; outside paired timing"
            }));
        }
    }));
    write_inputs(1);
    for graph in &graphs {
        graph.launch(&stream);
    }
    stream.sync();
    verify();
    assert_ne!(
        routes[0].validate(format, &host_weights, 1),
        previous,
        "second projection ignored fresh input"
    );
    println!(
        "{}",
        serde_json::json!({
            "experiment":"mmvq_full_pdl","kind":"scratch_hazard","M":rows,"K":k,"N":n,
            "pdl_supported":active,"first_result":"finite raw plus nonzero flags yields retained canonical NaN",
            "second_result":"same scratch naturally cleared; finite actual-pack F64 and full-bit match",
            "replay":"same captured addresses, new input data, independent retained first output window"
        })
    );
}

#[test]
#[ignore = "exclusive CUDA complete-MMVQ full-chain-PDL diagnostic"]
fn mmvq_full_pdl_preserves_bits_and_complete_projection_pairs() {
    let _ = caps();
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for rows in [1, 8] {
            for (k, n) in [(256, 32), (768, 64)] {
                boundary(format, rows, k, n, PdlMode::Full);
            }
        }
    }
    scratch_hazard();
    println!(
        "{}",
        serde_json::json!({
            "experiment":"mmvq_full_pdl","kind":"scope",
            "launch":"clear/widen/quantize/dot/cast all Ex programmatic serialization; dot retains original read-side sync and implicit completion trigger",
            "arithmetic":"same MMVQ pack/dot/MarkerV2 expressions; SharedM8 is held fixed in both M8 Q4/Q5 arms",
            "correctness":"12 eager/fresh/poison/recovery boundaries plus retained first cast with poisoned row flags before a following projection clears reused scratch",
            "timing":"six real M1 FFN groups and two real M8 attention groups; complete pack+dot+cast; allocation/scan/capture/precondition/readback excluded"
        })
    );
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for (k, n) in [(5120, 17408), (17408, 5120)] {
            paired(format, 1, k, n, PdlMode::Full);
        }
    }
    for (format, k, n) in [
        (UpstreamLinearFormat::Q4K, 5120, 10240),
        (UpstreamLinearFormat::Q5K, 6144, 5120),
    ] {
        paired(format, 8, k, n, PdlMode::Full);
    }
}
