//! Original split-MMQ complete-chain PDL. No plan, geometry or arithmetic change.
//! The 173 physical leaves are measured together, not extrapolated from one leaf.
use super::*;

unsafe extern "C" {
    fn ferrum_test_upstream_mmq_pdl_support_v1(
        p: *const UpstreamLinearPlanV1,
        supported: *mut u32,
    ) -> i32;
    fn ferrum_test_upstream_mmq_pdl_pack_f16_small_v2(
        p: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        rows: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmq_pdl_dot_v1(
        p: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmq_pdl_cast_v2(
        p: *const UpstreamLinearPlanV1,
        input: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        weights: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmq_pdl_graph_edges_v1(
        p: *const UpstreamLinearPlanV1,
        graph: *mut c_void,
        counts: *mut u64,
        count: usize,
    ) -> i32;
    fn cudaGraphUpload(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaMemsetAsync(pointer: *mut c_void, value: i32, bytes: usize, stream: *mut c_void) -> i32;
}

const EXPERIMENT: &str = "mmq_original_split_full_chain_pdl";
const PAIRS: usize = 8;
const SPECS: [(UpstreamLinearFormat, &str, usize); 6] = [
    (UpstreamLinearFormat::Iq4Xs, "gate_up", 61),
    (UpstreamLinearFormat::Iq4Xs, "down", 24),
    (UpstreamLinearFormat::Q4K, "gate_up", 28),
    (UpstreamLinearFormat::Q4K, "down", 10),
    (UpstreamLinearFormat::Q5K, "gate_up", 28),
    (UpstreamLinearFormat::Q5K, "down", 22),
];

fn normal() -> MmvqOps {
    MmvqOps {
        pack: ffi::ferrum_upstream_mmq_pack_v2,
        dot: ffi::ferrum_upstream_mmq_dot_v1,
        cast: ffi::ferrum_upstream_mmq_cast_v2,
    }
}

fn candidate() -> MmvqOps {
    MmvqOps {
        pack: ferrum_test_upstream_mmq_pdl_pack_f16_small_v2,
        dot: ferrum_test_upstream_mmq_pdl_dot_v1,
        cast: ferrum_test_upstream_mmq_pdl_cast_v2,
    }
}

fn supported(route: &Route) -> bool {
    let mut yes = 0;
    unsafe {
        ok(ferrum_test_upstream_mmq_pdl_support_v1(
            &route.plan,
            &mut yes,
        ));
    }
    yes != 0
}

fn dot(route: &Route, stream: &Stream, weight: *const c_void, ops: MmvqOps) {
    unsafe {
        ok((ops.dot)(
            &route.plan,
            weight,
            route.packed.ptr,
            route.output.ptr,
            if route.plan.fixup_bytes == 0 {
                ptr::null_mut()
            } else {
                route.fixup.ptr
            },
            stream.0,
        ));
    }
}

fn pack(route: &Route, stream: &Stream, input: &Buffer, ops: MmvqOps) {
    unsafe {
        ok((ops.pack)(
            &route.plan,
            input.ptr,
            route.plan.request.inputs,
            route.converted.ptr,
            route.packed.ptr,
            route.rows.ptr,
            stream.0,
        ));
    }
}

fn cast(route: &Route, stream: &Stream, destination: &Buffer, ops: MmvqOps) {
    unsafe {
        ok((ops.cast)(
            &route.plan,
            route.output.ptr,
            destination.ptr.add(route.result_offset as usize * 2),
            route.result_stride,
            route.rows.ptr,
            route.flags.ptr,
            stream.0,
        ));
    }
}

fn enqueue(route: &Route, stream: &Stream, input: &Buffer, weight: *const c_void, ops: MmvqOps) {
    pack(route, stream, input, ops);
    dot(route, stream, weight, ops); // The original optional fixup is inside this ABI.
    cast(route, stream, &route.result, ops);
}

fn edge_counts(route: &Route, graph: *mut c_void) -> [u64; 8] {
    let mut counts = [0; 8];
    unsafe {
        ok(ferrum_test_upstream_mmq_pdl_graph_edges_v1(
            &route.plan,
            graph,
            counts.as_mut_ptr(),
            counts.len(),
        ));
    }
    counts
}

fn inspect(route: &Route, graph: *mut c_void, active: bool, projections: usize) {
    let c = edge_counts(route, graph);
    if active {
        let n = projections as u64;
        assert_eq!(&c[2..4], &[n, n]);
        if route.plan.fixup != 0 {
            assert_eq!(&c[4..7], &[n, n, 0]);
        } else {
            assert_eq!(&c[4..7], &[0, 0, n]);
        }
        assert_eq!(
            c[7],
            n - 1,
            "each preceding cast must order the next scratch clear"
        );
        assert_eq!(c[1], c[2..].iter().sum::<u64>());
    } else {
        assert_eq!(c[1], 0, "ordinary capture must not acquire PDL edges");
    }
    println!(
        "{}",
        serde_json::json!({"experiment":EXPERIMENT,"kind":"capture_edges","M":route.plan.request.rows,"K":route.plan.request.inputs,"N":route.plan.request.outputs,"active_pdl":active,"projections":projections,"fixup":route.plan.fixup,"counts":c,"count_order":["all","programmatic","clear_pack","pack_main","main_fixup","fixup_cast","main_cast","cast_next_clear"],"inspection":"actual edgeData_v2 plus native function pairs; original plan determines fixup"})
    );
}

fn equal_buffers(a: &Route, b: &Route) {
    for (a, b) in [
        (&a.converted, &b.converted),
        (&a.packed, &b.packed),
        (&a.output, &b.output),
        (&a.fixup, &b.fixup),
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

fn input_words(m: usize, k: usize, seed: usize, generation: usize) -> Vec<u16> {
    let mut v = (0..m * k)
        .map(|i| {
            f16::from_f32(((i * 17 + seed * 13 + generation * 29) % 101) as f32 / 2048.0 - 0.024)
                .to_bits()
        })
        .collect::<Vec<_>>();
    v[0] = 0x8000;
    v[1] = 1;
    v[2] = 0x8001;
    v
}

fn mixed_weights(format: UpstreamLinearFormat, k: usize, n: usize, seed: usize) -> Vec<u8> {
    let mut v = weights(format, k, n);
    let bytes = match format {
        UpstreamLinearFormat::Iq4Xs => 136,
        UpstreamLinearFormat::Q4K => 144,
        UpstreamLinearFormat::Q5K => 176,
    };
    for (i, b) in v.chunks_exact_mut(bytes).enumerate() {
        b[..2].copy_from_slice(
            &f16::from_f32((1.0 + ((i + seed) % 17) as f32 / 32.0) / 4096.0)
                .to_bits()
                .to_le_bytes(),
        );
        if format != UpstreamLinearFormat::Iq4Xs {
            b[2..4].copy_from_slice(
                &f16::from_f32((1.0 + ((i * 3 + seed) % 19) as f32 / 64.0) / 8192.0)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
    }
    v
}

fn new_route(format: UpstreamLinearFormat, m: usize, k: usize, n: usize) -> Route {
    Route::new_with_result_window(
        UpstreamLinearAlgorithm::Mmq,
        format,
        m as u32,
        k as u32,
        n as u32,
        1,
        (n + 7) as u32,
        3,
    )
}

fn verify_markers(route: &Route, bad_rows: bool, bad_weight: bool) {
    let p = &route.plan;
    let (m, n) = (p.request.rows as usize, p.request.outputs as usize);
    assert_eq!(route.flags.read::<u32>(1)[0] != 0, bad_weight);
    assert!(route
        .rows
        .read::<u32>(m)
        .iter()
        .all(|&f| (f != 0) == bad_rows));
    let raw = route.output.read::<f32>(m * n);
    let result = route.result.read::<u16>(route.result.bytes / 2);
    for row in 0..m {
        let dst = &result[row * (n + 7)..(row + 1) * (n + 7)];
        assert_eq!(&dst[..3], &[0x3535; 3]);
        assert_eq!(&dst[3 + n..], &[0x3535; 4]);
        for col in 0..n {
            assert_eq!(
                dst[3 + col],
                oracle::final_cast_bits(raw[row * n + col], bad_rows, bad_weight)
            );
        }
    }
    let packed = route.packed.read::<u8>(route.packed.bytes);
    let tail = m * p.padded_inputs as usize * 9 / 8;
    assert!(
        packed[tail..].iter().all(|&v| v == 0),
        "MMQ overread guard must be zero on every replay"
    );
}

fn boundary(format: UpstreamLinearFormat, m: usize, k: usize, n: usize, expect_no_fixup: bool) {
    let stream = Stream::new();
    let host = mixed_weights(format, k, n, 7);
    let weight = Buffer::new(host.len());
    weight.write(&host);
    let input = Buffer::new(m * k * 2);
    input.write(&input_words(m, k, 7, 0));
    let routes = [new_route(format, m, k, n), new_route(format, m, k, n)];
    if expect_no_fixup {
        assert_eq!(routes[0].plan.fixup, 0);
    }
    let active = supported(&routes[1]);
    let ops = [normal(), if active { candidate() } else { normal() }];
    for (arm, r) in routes.iter().enumerate() {
        r.scan(&stream, weight.ptr, 0);
        enqueue(r, &stream, &input, weight.ptr, ops[arm]);
    }
    stream.sync();
    equal_buffers(&routes[0], &routes[1]);
    let initial = routes[0].validate(format, &host, 1);
    routes[1].validate(format, &host, 1);
    let graphs = [0, 1].map(|arm| {
        Graph::capture_inspected(
            &stream,
            || enqueue(&routes[arm], &stream, &input, weight.ptr, ops[arm]),
            |g| {
                if active {
                    inspect(&routes[arm], g, arm == 1, 1);
                }
            },
        )
    });
    for (generation, bad_rows, bad_weight) in [
        (1, false, false),
        (2, true, false),
        (3, false, true),
        (4, false, false),
    ] {
        let mut words = input_words(m, k, 7, generation);
        if bad_rows {
            for row in 0..m {
                words[row * k + 4] = if row % 2 == 0 { 0x7e00 } else { 0x7c00 };
            }
        }
        input.write(&words);
        let mut current = host.clone();
        if bad_weight {
            current[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        }
        weight.write(&current);
        for r in &routes {
            r.scan(&stream, weight.ptr, 0);
        }
        for g in &graphs {
            g.launch(&stream);
        }
        stream.sync();
        equal_buffers(&routes[0], &routes[1]);
        for r in &routes {
            verify_markers(r, bad_rows, bad_weight);
            if !bad_rows && !bad_weight {
                assert_ne!(
                    r.validate(format, &host, 1),
                    initial,
                    "fresh replay ignored changed input"
                );
            }
        }
        input.guards();
        weight.guards();
    }
    println!(
        "{}",
        serde_json::json!({"experiment":EXPERIMENT,"kind":"boundary","format":format!("{format:?}"),"M":m,"K":k,"N":n,"fixup":routes[0].plan.fixup,"active_pdl":active,"graph_edges_inspected":active,"full_buffer_bit_equality":true,"fresh_finite_nonfinite_weight_marker_finite_recovery":true,"oracle":"finite actual-packed nine-point F64 plus all-output cast/marker/stride/guard checks","fallback":"unsupported uses ordered correctness only; edge metadata not claimed (older SDK may lack API); no PDL performance claim"})
    );
}

fn scratch_hazard(format: UpstreamLinearFormat, m: usize, k: usize, n: usize) {
    let stream = Stream::new();
    let host = mixed_weights(format, k, n, 3);
    let weight = Buffer::new(host.len());
    weight.write(&host);
    let inputs = [Buffer::new(m * k * 2), Buffer::new(m * k * 2)];
    let routes = [new_route(format, m, k, n), new_route(format, m, k, n)];
    let first = [Buffer::new(m * (n + 7) * 2), Buffer::new(m * (n + 7) * 2)];
    let active = supported(&routes[1]);
    let ops = [normal(), if active { candidate() } else { normal() }];
    let update = |generation| {
        for (i, input) in inputs.iter().enumerate() {
            input.write(&input_words(m, k, i + 3, generation));
        }
    };
    update(0);
    for r in &routes {
        r.scan(&stream, weight.ptr, 0);
    }
    let run = |arm: usize| {
        let r = &routes[arm];
        pack(r, &stream, &inputs[0], ops[arm]);
        dot(r, &stream, weight.ptr, ops[arm]);
        // Correctness-only poison after the first clear/main, before its cast.
        // The retained result detects a second clear overtaking the first cast.
        // Leave other rows finite so an early next main/fixup cannot hide its
        // output overwrite behind an all-NaN first cast.
        unsafe {
            ok(cudaMemsetAsync(r.rows.ptr, 1, 4, stream.0));
        }
        cast(r, &stream, &first[arm], ops[arm]);
        enqueue(r, &stream, &inputs[1], weight.ptr, ops[arm]);
    };
    let verify = || {
        equal_buffers(&routes[0], &routes[1]);
        assert_eq!(
            first[0].read::<u16>(m * (n + 7)),
            first[1].read::<u16>(m * (n + 7))
        );
        for arm in 0..2 {
            let retained = first[arm].read::<u16>(m * (n + 7));
            let second = routes[arm].result.read::<u16>(m * (n + 7));
            for (row_index, row) in retained.chunks_exact(n + 7).enumerate() {
                assert_eq!(&row[..3], &[0x3535; 3]);
                assert_eq!(&row[3 + n..], &[0x3535; 4]);
                if row_index == 0 {
                    assert!(row[3..3 + n].iter().all(|&v| v == 0x7e00));
                } else {
                    assert!(row[3..3 + n].iter().all(|&v| f16::from_bits(v).is_finite()));
                }
            }
            assert!(
                retained
                    .chunks_exact(n + 7)
                    .zip(second.chunks_exact(n + 7))
                    .skip(1)
                    .any(|(first, next)| first[3..3 + n] != next[3..3 + n]),
                "finite first rows must independently witness shared raw-output reuse"
            );
            first[arm].guards();
            routes[arm].validate(format, &host, 1);
            verify_markers(&routes[arm], false, false);
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
    let old = routes[0].validate(format, &host, 1);
    let graphs=[0,1].map(|arm|Graph::capture_inspected(&stream,||run(arm),|g|{
        if !active {return;}
        let c=edge_counts(&routes[arm],g);
        if active&&arm==1 {
            assert_eq!(&c[2..4],&[2,2]);
            if routes[arm].plan.fixup!=0 {assert_eq!(&c[4..7],&[2,1,0]);}else{assert_eq!(&c[4..7],&[0,0,1]);}
            assert_eq!(c[7],1,"first cast -> next clear must protect shared flags and scratch");
        }else{assert_eq!(c[1],0);}
        println!("{}",serde_json::json!({"experiment":EXPERIMENT,"kind":"scratch_hazard_edges","arm":arm,"M":m,"K":k,"N":n,"fixup":routes[arm].plan.fixup,"counts":c,"poison":"correctness-only memset interrupts first main/fixup -> cast edge, not cast -> next clear"}));
    }));
    update(1);
    for graph in &graphs {
        graph.launch(&stream);
    }
    stream.sync();
    verify();
    assert_ne!(routes[0].validate(format, &host, 1), old);
    println!(
        "{}",
        serde_json::json!({"experiment":EXPERIMENT,"kind":"scratch_hazard","M":m,"K":k,"N":n,"fixup":routes[0].plan.fixup,"active_pdl":active,"first_poisoned_row_retained_canonical_nan":true,"first_other_rows_finite_distinct_from_second":true,"next_projection_reuses_all_scratch_and_naturally_clears_flags":true,"fresh_input_bit_equality_and_actual_pack_oracle":true})
    );
}

#[test]
#[ignore = "exclusive CUDA original split-MMQ PDL boundary and reused-scratch oracle"]
fn mmq_original_split_pdl_preserves_bits_markers_and_scratch_reuse() {
    let device = caps();
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for m in [2, 8] {
            boundary(format, m, 256, 33, false);
        }
        let probe = new_route(format, 8, 256, 128);
        let n = probe.plan.i as usize * device.multiprocessors as usize;
        drop(probe);
        boundary(format, 8, 256, n, true);
    }
    // Both real split/fixup scratch and an exact SM-count no-fixup plan.
    scratch_hazard(UpstreamLinearFormat::Q5K, 8, 5120, 17408);
    let p = new_route(UpstreamLinearFormat::Q4K, 2, 256, 128);
    let n = p.plan.i as usize * device.multiprocessors as usize;
    drop(p);
    scratch_hazard(UpstreamLinearFormat::Q4K, 2, 256, n);
}

struct Leaf {
    routes: [Route; 2],
    stable: Vec<u16>,
}
struct Group {
    graphs: [Graph; 2], // Graphs must drop before all captured addresses.
    leaves: Vec<Leaf>,
    input: Buffer,
    allocation: Buffer,
    addresses: Vec<*mut c_void>,
    host: Vec<u8>,
    format: UpstreamLinearFormat,
    role: &'static str,
    m: usize,
    k: usize,
    n: usize,
    seed: usize,
}
impl Group {
    fn new(
        stream: &Stream,
        format: UpstreamLinearFormat,
        role: &'static str,
        count: usize,
        m: usize,
        seed: usize,
    ) -> Self {
        let (k, n) = if role == "gate_up" {
            (5120, 17408)
        } else {
            (17408, 5120)
        };
        let host = mixed_weights(format, k, n, seed);
        let stride = host.len().div_ceil(256) * 256;
        let allocation = Buffer::new(count * stride);
        let addresses = (0..count)
            .map(|i| unsafe { allocation.ptr.add(i * stride) })
            .collect::<Vec<_>>();
        for &address in &addresses {
            unsafe {
                ok(cudaMemcpy(address, host.as_ptr().cast(), host.len(), 1));
            }
        }
        let input = Buffer::new(m * k * 2);
        input.write(&input_words(m, k, seed, 0));
        let leaves = addresses
            .iter()
            .map(|&weight| {
                let routes = [new_route(format, m, k, n), new_route(format, m, k, n)];
                assert!(
                    supported(&routes[1]),
                    "each actual selected kernel must cold-qualify, not a double-ordered comparison"
                );
                for arm in 0..2 {
                    routes[arm].scan(stream, weight, 0);
                    enqueue(
                        &routes[arm],
                        stream,
                        &input,
                        weight,
                        if arm == 0 { normal() } else { candidate() },
                    );
                }
                stream.sync();
                equal_buffers(&routes[0], &routes[1]);
                let stable = routes[0].validate(format, &host, 1);
                routes[1].validate(format, &host, 1);
                Leaf { routes, stable }
            })
            .collect::<Vec<_>>();
        let graphs = [0, 1].map(|arm| {
            Graph::capture_inspected(
                stream,
                || {
                    for (leaf, &weight) in leaves.iter().zip(&addresses) {
                        enqueue(
                            &leaf.routes[arm],
                            stream,
                            &input,
                            weight,
                            if arm == 0 { normal() } else { candidate() },
                        );
                    }
                },
                |graph| inspect(&leaves[0].routes[arm], graph, arm == 1, count),
            )
        });
        Self {
            graphs,
            leaves,
            input,
            allocation,
            addresses,
            host,
            format,
            role,
            m,
            k,
            n,
            seed,
        }
    }
    fn enqueue(&self, stream: &Stream, arm: usize) {
        for (leaf, &weight) in self.leaves.iter().zip(&self.addresses) {
            enqueue(
                &leaf.routes[arm],
                stream,
                &self.input,
                weight,
                if arm == 0 { normal() } else { candidate() },
            );
        }
    }
    fn verify(&self, fresh: bool) {
        for leaf in &self.leaves {
            equal_buffers(&leaf.routes[0], &leaf.routes[1]);
            for r in &leaf.routes {
                let actual = r.validate(self.format, &self.host, 1);
                if fresh {
                    assert_ne!(
                        actual, leaf.stable,
                        "capture ignored fresh same-address input"
                    );
                } else {
                    assert_eq!(actual, leaf.stable);
                }
                verify_markers(r, false, false);
            }
        }
        self.input.guards();
        self.allocation.guards();
        println!(
            "{}",
            serde_json::json!({"experiment":EXPERIMENT,"kind":"ring_correctness","M":self.m,"format":format!("{:?}",self.format),"role":self.role,"K":self.k,"N":self.n,"leaves":self.leaves.len(),"fresh":fresh,"all_buffers_bits_equal":true,"oracle":"each leaf finite actual-packed nine-point F64; every output cast, padded window and allocation guard"})
        );
    }
}
struct Ring {
    graphs: [Graph; 2],
    groups: Vec<Group>,
}

fn thread_cpu_ns() -> Option<u128> {
    #[cfg(target_os = "linux")]
    {
        let mut t = std::mem::MaybeUninit::<libc::timespec>::uninit();
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, t.as_mut_ptr()) } != 0 {
            return None;
        }
        let t = unsafe { t.assume_init() };
        Some(u128::try_from(t.tv_sec).ok()? * 1_000_000_000 + u128::try_from(t.tv_nsec).ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
fn timed_pairs(stream: &Stream, graphs: &[Graph; 2], metadata: serde_json::Value) {
    for graph in graphs {
        unsafe {
            ok(cudaGraphUpload(graph.0, stream.0));
        }
    }
    stream.sync();
    for _ in 0..2 {
        for graph in graphs {
            graph.launch(stream);
        }
    }
    stream.sync();
    let (start, end) = (Event::new(), Event::new());
    for pair in 0..PAIRS {
        for order in 0..2 {
            let arm = (pair + order) % 2;
            let wall = Instant::now();
            unsafe {
                ok(cudaEventRecord(start.0, stream.0));
            }
            let launch = Instant::now();
            let cpu = thread_cpu_ns();
            let status = unsafe { cudaGraphLaunch(graphs[arm].0, stream.0) };
            let cpu_end = thread_cpu_ns();
            let launch_ns = launch.elapsed().as_nanos();
            ok(status);
            unsafe {
                ok(cudaEventRecord(end.0, stream.0));
                ok(cudaEventSynchronize(end.0));
            }
            let wall_ns = wall.elapsed().as_nanos();
            let mut ms = 0.0;
            unsafe {
                ok(cudaEventElapsedTime(&mut ms, start.0, end.0));
            }
            println!(
                "{}",
                serde_json::json!({"experiment":EXPERIMENT,"kind":"sample","metadata":metadata,"pair":pair,"order":order,"arm":if arm==0{"ordered"}else{"programmatic"},"gpu_ns":f64::from(ms)*1_000_000.0,"host_wall_ns":wall_ns,"graph_launch_wall_ns":launch_ns,"graph_launch_thread_cpu_ns":cpu.zip(cpu_end).and_then(|(a,b)|b.checked_sub(a))})
            );
        }
    }
}

fn run_ring(m: usize) {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    for &(format, role, _) in &SPECS {
        let (k, n) = if role == "gate_up" {
            (5120, 17408)
        } else {
            (17408, 5120)
        };
        let probe = new_route(format, m, k, n);
        if !supported(&probe) {
            println!(
                "{}",
                serde_json::json!({"experiment":EXPERIMENT,"kind":"unsupported","M":m,"format":format!("{format:?}"),"K":k,"N":n,"timing_skipped":true})
            );
            return;
        }
    }
    let stream = Stream::new();
    let groups = SPECS
        .iter()
        .enumerate()
        .map(|(seed, &(format, role, count))| Group::new(&stream, format, role, count, m, seed))
        .collect::<Vec<_>>();
    let graphs = [0, 1].map(|arm| {
        Graph::capture(&stream, || {
            for group in &groups {
                group.enqueue(&stream, arm);
            }
        })
    });
    let ring = Ring { graphs, groups };
    let bytes = ring
        .groups
        .iter()
        .map(|g| g.host.len() * g.leaves.len())
        .sum::<usize>();
    let leaves = ring.groups.iter().map(|g| g.leaves.len()).sum::<usize>();
    assert!(bytes > 3 * l2 as usize);
    let scope = serde_json::json!({"M":m,"leaves":leaves,"physical_weight_bytes":bytes,"device_l2_bytes":l2,"pairs":PAIRS,"order":"alternating AB/BA","same_physical_weights_both_arms":true,"timed":"complete original direct-F16 clear+pack, original split-MMQ main+optionalfixup and MarkerV2 cast","excluded":"allocation/weightupload/scan/capture/graphupload/validation/readback","geometry":"unchanged normal public plan per format/shape; no full-tile or I32 substitution","scratch":"per-leaf independent in weighted ring; separate captured two-projection hazard explicitly reuses all scratch and preserves first poisoned output","scope":"diagnostic full projection costs, not serving throughput; kernel-sum coverage is not predicted savings"});
    println!(
        "{}",
        serde_json::json!({"experiment":EXPERIMENT,"kind":"scope","metadata":scope})
    );
    for g in &ring.groups {
        let p = &g.leaves[0].routes[0].plan;
        timed_pairs(
            &stream,
            &g.graphs,
            serde_json::json!({"scope":"format_role","M":m,"format":format!("{:?}",g.format),"role":g.role,"K":g.k,"N":g.n,"leaves":g.leaves.len(),"physical_weight_bytes":g.host.len()*g.leaves.len(),"I":p.i,"J":p.j,"threads":p.nthreads,"blocks":p.blocks,"fixup":p.fixup,"fixup_bytes":p.fixup_bytes}),
        );
    }
    timed_pairs(
        &stream,
        &ring.graphs,
        serde_json::json!({"scope":"weighted_full_ring","configuration":scope}),
    );
    stream.sync();
    for g in &ring.groups {
        g.verify(false);
    }
    for g in &ring.groups {
        g.input.write(&input_words(m, g.k, g.seed, 1));
    }
    for graph in &ring.graphs {
        graph.launch(&stream);
    }
    stream.sync();
    for g in &ring.groups {
        g.verify(true);
    }
}

#[test]
#[ignore = "exclusive CUDA original split-MMQ full-chain PDL; complete M8 and M2 173-leaf rings"]
fn mmq_original_split_pdl_m8_m2_complete_173_leaf_pairs() {
    for m in [8, 2] {
        run_ring(m);
    }
}
