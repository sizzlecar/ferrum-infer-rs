//! One representative M1 FFN group: one production RMS feeds gate and up.
//! Both arms use the retained normal five-kernel MMVQ PDL chain. Only the RMS
//! launch changes. This is not a model benchmark or a sum of isolated kernels.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

const K: usize = 5120;
const N: usize = 17408;
const EPSILON: f32 = 1e-6;
const FORMAT: UpstreamLinearFormat = UpstreamLinearFormat::Q4K;

#[link(name = "ferrum_rms_mmvq_pdl_test", kind = "static")]
unsafe extern "C" {
    fn ferrum_test_rms_mmvq_support_v1(supported: *mut u32) -> i32;
    fn ferrum_test_rms_mmvq_launch_v1(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        epsilon: f32,
        arm: u32,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_rms_mmvq_graph_v1(
        graph: *mut c_void,
        nodes: *mut u64,
        node_words: usize,
        edges: *mut u64,
        edge_words: usize,
        counts: *mut u64,
    ) -> i32;
}
unsafe extern "C" {
    fn ferrum_upstream_mmvq_pdl_support_v1(
        plan: *const UpstreamLinearPlanV1,
        dot_kind: u32,
        supported: *mut u32,
    ) -> i32;
    fn ferrum_upstream_mmvq_pdl_pack_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_upstream_mmvq_pdl_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
        dot_kind: u32,
    ) -> i32;
    fn ferrum_upstream_mmvq_pdl_cast_v2(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        weights: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn cudaGraphUpload(graph: *mut c_void, stream: *mut c_void) -> i32;
}

// Declared after graph/buffer owners: synchronize before their destruction,
// including an assertion unwinding after an asynchronous enqueue.
struct Drain(*mut c_void);
impl Drop for Drain {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaStreamSynchronize(self.0);
        }
    }
}

fn input(generation: usize) -> Vec<f32> {
    (0..K)
        .map(|i| ((i * 17 + generation * 29) % 113) as f32 / 64.0 - 0.8)
        .collect()
}
fn norm_weight(generation: usize) -> Vec<u16> {
    (0..K)
        .map(|i| f16::from_f32(0.75 + ((i * 7 + generation) % 31) as f32 / 64.0).to_bits())
        .collect()
}

struct Group {
    input: Buffer,
    norm_weight: Buffer,
    norm: Buffer,
    weights: [Buffer; 2],
    host_weights: [Vec<u8>; 2],
    routes: [Route; 2],
}
impl Group {
    fn new(stream: &Stream, generation: usize) -> Self {
        let host_weights = std::array::from_fn(|leaf| {
            let mut w = weights(FORMAT, K, N);
            for (block_index, block) in w.chunks_exact_mut(144).enumerate() {
                block[..2].copy_from_slice(&f16::from_f32(1.0 / 4096.0).to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&f16::from_f32(1.0 / 8192.0).to_bits().to_le_bytes());
                for (i, code) in block[16..].iter_mut().enumerate() {
                    *code = (i * 31 + block_index * 13 + leaf * 71 + generation * 19) as u8;
                }
            }
            w
        });
        let weights = std::array::from_fn(|i| {
            let w = Buffer::new(host_weights[i].len());
            w.write(&host_weights[i]);
            w
        });
        let routes = std::array::from_fn(|_| {
            Route::new_with_result_window(
                UpstreamLinearAlgorithm::Mmvq,
                FORMAT,
                1,
                K as u32,
                N as u32,
                1,
                (N + 7) as u32,
                3,
            )
        });
        let group = Self {
            input: Buffer::new(K * 4),
            norm_weight: Buffer::new(K * 2),
            norm: Buffer::new(K * 2),
            weights,
            host_weights,
            routes,
        };
        group.write(generation);
        for i in 0..2 {
            group.routes[i].scan(stream, group.weights[i].ptr, 0);
        }
        stream.sync();
        group
    }
    fn write(&self, generation: usize) {
        self.input.write(&input(generation));
        self.norm_weight.write(&norm_weight(generation));
    }
    fn enqueue(&self, stream: &Stream, arm: usize, x: &Buffer, outputs: [&Buffer; 2]) {
        unsafe {
            ok(ferrum_test_rms_mmvq_launch_v1(
                x.ptr,
                self.norm_weight.ptr,
                self.norm.ptr,
                EPSILON,
                arm as u32,
                stream.0,
            ));
            for (i, destination) in outputs.into_iter().enumerate() {
                let r = &self.routes[i];
                assert_eq!(r.plan.fixup_bytes, 0);
                ok(ferrum_upstream_mmvq_pdl_pack_v2(
                    &r.plan,
                    self.norm.ptr,
                    K as u32,
                    r.converted.ptr,
                    r.packed.ptr,
                    r.rows.ptr,
                    stream.0,
                ));
                ok(ferrum_upstream_mmvq_pdl_dot_v1(
                    &r.plan,
                    self.weights[i].ptr,
                    r.packed.ptr,
                    r.output.ptr,
                    ptr::null_mut(),
                    stream.0,
                    0,
                ));
                ok(ferrum_upstream_mmvq_pdl_cast_v2(
                    &r.plan,
                    r.output.ptr,
                    destination.ptr.add(6),
                    (N + 7) as u32,
                    r.rows.ptr,
                    r.flags.ptr,
                    stream.0,
                ));
            }
        }
    }
    fn run(&self, stream: &Stream, arm: usize) {
        self.enqueue(
            stream,
            arm,
            &self.input,
            [&self.routes[0].result, &self.routes[1].result],
        );
    }
    fn buffers(&self) -> Vec<&Buffer> {
        let mut b = vec![&self.input, &self.norm_weight, &self.norm];
        for r in &self.routes {
            b.extend([
                &r.converted,
                &r.packed,
                &r.output,
                &r.fixup,
                &r.result,
                &r.rows,
                &r.flags,
            ]);
        }
        b
    }
    fn snapshot(&self) -> Vec<Vec<u8>> {
        self.buffers()
            .iter()
            .map(|b| {
                b.guards();
                b.read::<u8>(b.bytes)
            })
            .collect()
    }
    fn validate(&self, x: &Buffer, finite: bool) {
        let norm = self.norm.read::<u16>(K);
        for w in &self.weights {
            w.guards();
        }
        for b in self.buffers() {
            b.guards();
        }
        x.guards();
        if finite {
            // Full RMS F64 reference, with reduction/rsqrt and final half-rounding
            // error allowed; bit identity between arms is separately mandatory.
            let x = x.read::<f32>(K);
            let w = self.norm_weight.read::<u16>(K);
            let variance = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / K as f64;
            let inverse = (variance + EPSILON as f64).sqrt().recip();
            let nu = (K.div_ceil(1024) + 16) as f64 * f32::EPSILON as f64;
            for i in 0..K {
                let target = x[i] as f64 * inverse * f16::from_bits(w[i]).to_f64();
                let actual = f16::from_bits(norm[i]).to_f64();
                let bound =
                    target.abs() * (2.0 * nu / (1.0 - nu) + 1.0 / 1024.0) + 2.0_f64.powi(-24);
                assert!(
                    actual.is_finite() && (actual - target).abs() <= bound,
                    "RMS oracle at {i}"
                );
            }
            for i in 0..2 {
                self.routes[i].validate(FORMAT, &self.host_weights[i], 1);
            }
        } else {
            assert!(norm.iter().all(|&v| f16::from_bits(v).is_nan()));
            for r in &self.routes {
                assert_ne!(r.rows.read::<u32>(1)[0], 0);
                assert_eq!(r.flags.read::<u32>(1), [0]);
                let values = r.result.read::<u16>(N + 7);
                assert!(values[3..3 + N].iter().all(|&v| v == 0x7e00));
                assert!(values[..3]
                    .iter()
                    .chain(&values[3 + N..])
                    .all(|&v| v == 0x3535));
            }
        }
    }
}

fn inspect(graph: *mut c_void, arm: usize, groups: usize, label: &str) -> Vec<u64> {
    let mut counts = [0; 4];
    unsafe {
        ok(ferrum_test_rms_mmvq_graph_v1(
            graph,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            counts.as_mut_ptr(),
        ));
    }
    let mut nodes = vec![0; counts[0] as usize * 4];
    let mut edges = vec![0; counts[1] as usize * 5];
    unsafe {
        ok(ferrum_test_rms_mmvq_graph_v1(
            graph,
            nodes.as_mut_ptr(),
            nodes.len(),
            edges.as_mut_ptr(),
            edges.len(),
            counts.as_mut_ptr(),
        ));
    }
    let by_handle: BTreeMap<_, _> = nodes.chunks_exact(4).map(|n| (n[0], n)).collect();
    assert_eq!(by_handle.len(), groups * 11);
    assert!(
        by_handle.values().all(|n| n[2] == 0),
        "complete group must contain only its kernel nodes"
    );
    let targets: BTreeSet<_> = edges.chunks_exact(5).map(|e| e[1]).collect();
    let roots: Vec<_> = by_handle
        .keys()
        .filter(|n| !targets.contains(*n))
        .copied()
        .collect();
    assert_eq!(roots.len(), 1);
    let mut ordered = Vec::new();
    let mut current = roots[0];
    loop {
        assert!(!ordered.contains(&current), "graph contains a cycle");
        ordered.push(current);
        let next: BTreeSet<_> = edges
            .chunks_exact(5)
            .filter(|e| e[0] == current)
            .map(|e| e[1])
            .collect();
        if next.is_empty() {
            break;
        }
        assert_eq!(
            next.len(),
            1,
            "unexpected branching in captured complete group"
        );
        current = *next.first().unwrap();
    }
    assert_eq!(ordered.len(), by_handle.len());
    let functions: Vec<_> = ordered.iter().map(|n| by_handle[n][1]).collect();
    for (i, handle) in ordered.iter().enumerate() {
        let kind = by_handle[handle][3];
        assert_eq!(kind, if i % 11 == 0 { arm as u64 + 1 } else { 0 });
        if i % 11 != 0 {
            let within_leaf = (i % 11 - 1) % 5;
            assert_eq!(
                functions[i],
                functions[1 + within_leaf],
                "both leaves must use the same normal MMVQ kernels"
            );
        }
    }
    for (i, pair) in ordered.windows(2).enumerate() {
        // A's RMS outgoing edge may itself be programmatic (implicit producer
        // completion). Never infer early launch_dependents from edge type.
        if arm == 1 || !matches!(i % 11, 0 | 10) {
            assert!(
                edges.chunks_exact(5).any(|e| e[0] == pair[0]
                    && e[1] == pair[1]
                    && e[2] == counts[2]
                    && e[3] == counts[3]),
                "missing actual PDL edge after stage {}",
                i % 11
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"experiment":"rms_mmvq_group_pdl", "kind":"capture",
        "label":label,"arm":arm,"groups":groups,"roots":roots.len(),"nodes":nodes,
        "edges":edges,"node_record":"handle,function,type,rms_kind", "edge_record":"from,to,type,from_port,to_port",
        "programmatic_type":counts[2],"programmatic_port":counts[3],
        "rms_has_explicit_early_trigger":arm == 1,"projection_source":"retained normal PDL ABI, dot_kind=0"})
    );
    functions[1..6].to_vec()
}

fn capture(
    stream: &Stream,
    arm: usize,
    groups: usize,
    label: &str,
    enqueue: impl FnOnce(),
) -> (Graph, Vec<u64>) {
    let mut functions = Vec::new();
    let graph = Graph::capture_inspected(stream, enqueue, |g| {
        functions = inspect(g, arm, groups, label)
    });
    (graph, functions)
}

fn hazard(stream: &Stream) {
    let cold = Instant::now();
    let arms = [Group::new(stream, 17), Group::new(stream, 17)];
    let next: [Buffer; 2] = std::array::from_fn(|_| {
        let b = Buffer::new(K * 4);
        b.write(&input(18));
        b
    });
    let retained: [[Buffer; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Buffer::new((N + 7) * 2)));
    let mut graphs = Vec::new();
    let _drain = Drain(stream.0);
    let mut eager_first = Vec::new();
    let mut eager_final = Vec::new();
    let mut functions = Vec::new();
    for arm in 0..2 {
        let g = &arms[arm];
        g.run(stream, arm);
        stream.sync();
        g.validate(&g.input, true);
        eager_first.push(
            g.routes
                .iter()
                .map(|r| r.result.read::<u16>(N + 7))
                .collect::<Vec<_>>(),
        );
        g.enqueue(
            stream,
            arm,
            &next[arm],
            [&g.routes[0].result, &g.routes[1].result],
        );
        stream.sync();
        g.validate(&next[arm], true);
        eager_final.push(g.snapshot());
        assert_ne!(eager_first[arm][0], g.routes[0].result.read::<u16>(N + 7));
        let (graph, f) = capture(stream, arm, 2, "shared_scratch_hazard", || {
            g.enqueue(
                stream,
                arm,
                &g.input,
                [&retained[arm][0], &retained[arm][1]],
            );
            g.enqueue(
                stream,
                arm,
                &next[arm],
                [&g.routes[0].result, &g.routes[1].result],
            );
        });
        unsafe {
            ok(cudaGraphUpload(graph.0, stream.0));
        }
        stream.sync();
        functions.push(f);
        graphs.push(graph);
    }
    assert_eq!(functions[0], functions[1]);
    assert_eq!(eager_first[0], eager_first[1]);
    assert_eq!(eager_final[0], eager_final[1]);
    let cold_ms = cold.elapsed().as_secs_f64() * 1000.0;
    for arm in 0..2 {
        graphs[arm].launch(stream);
        stream.sync();
        assert_eq!(arms[arm].snapshot(), eager_final[arm]);
        for leaf in 0..2 {
            assert_eq!(
                retained[arm][leaf].read::<u16>(N + 7),
                eager_first[arm][leaf]
            );
            retained[arm][leaf].guards();
        }
    }
    // Same captured addresses, first result poisoned by the first request only.
    // The second clear/pack must recover while preserving the first cast result.
    for arm in 0..2 {
        let mut bad = input(19);
        bad[113] = f32::NAN;
        arms[arm].input.write(&bad);
        next[arm].write(&input(20));
        graphs[arm].launch(stream);
        stream.sync();
        arms[arm].validate(&next[arm], true);
        for leaf in 0..2 {
            let output = retained[arm][leaf].read::<u16>(N + 7);
            assert!(output[3..3 + N].iter().all(|&v| v == 0x7e00));
            assert!(output[..3]
                .iter()
                .chain(&output[3 + N..])
                .all(|&v| v == 0x3535));
            retained[arm][leaf].guards();
        }
    }
    assert_eq!(arms[0].snapshot(), arms[1].snapshot());
    println!(
        "{}",
        serde_json::json!({"experiment":"rms_mmvq_group_pdl", "kind":"scratch_hazard",
        "cold_setup_capture_upload_oracles_ms":cold_ms,"total_correctness_ms":cold.elapsed().as_secs_f64() * 1000.0,
        "full_bits":true,"retained_first_result":true,
        "distinct_second_input":true,"finite_first_differs_from_second":true,"nonfinite_then_finite":true,
        "guards":true,"f64":"full RMS; projection samples actual packed activation, not full F64"})
    );
}

fn thread_cpu_ns() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) },
        0
    );
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
fn measure(graph: &Graph, stream: &Stream, start: &Event, end: &Event) -> [f64; 4] {
    let block = Instant::now();
    unsafe {
        ok(cudaEventRecord(start.0, stream.0));
    }
    let cpu = thread_cpu_ns();
    let launch = Instant::now();
    graph.launch(stream);
    let launch_ms = launch.elapsed().as_secs_f64() * 1000.0;
    let cpu_ms = (thread_cpu_ns() - cpu) as f64 / 1e6;
    unsafe {
        ok(cudaEventRecord(end.0, stream.0));
        ok(cudaEventSynchronize(end.0));
    }
    let block_ms = block.elapsed().as_secs_f64() * 1000.0;
    let mut gpu_ms = 0.0;
    unsafe {
        ok(cudaEventElapsedTime(&mut gpu_ms, start.0, end.0));
    }
    [gpu_ms as f64, block_ms, launch_ms, cpu_ms]
}

#[test]
#[ignore = "exclusive CUDA TEST archive: complete representative M1 RMS + gate/up, not model throughput"]
fn rms_mmvq_m1_gate_up_group_pdl_pairs() {
    let cold_start = Instant::now();
    let device = caps();
    let p = plan(
        UpstreamLinearAlgorithm::Mmvq,
        FORMAT,
        UpstreamLinearLayout::Columns,
        1,
        K as u32,
        N as u32,
    );
    let (mut rms_supported, mut projection_supported, mut l2) = (0, 0, 0);
    unsafe {
        ok(ferrum_test_rms_mmvq_support_v1(&mut rms_supported));
        ok(ferrum_upstream_mmvq_pdl_support_v1(
            &p,
            0,
            &mut projection_supported,
        ));
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    if rms_supported == 0 || projection_supported == 0 {
        println!(
            "{}",
            serde_json::json!({"experiment":"rms_mmvq_group_pdl","kind":"unsupported",
            "rms_supported":rms_supported,"normal_projection_supported":projection_supported,"timing_skipped":true})
        );
        return;
    }
    assert!(l2 > 0);
    let qualification_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    let stream = Stream::new();
    hazard(&stream);
    let weight_bytes = K / 256 * N * 144;
    let group_count = 3 * l2 as usize / (2 * weight_bytes) + 1;
    let laps = 16_usize.div_ceil(group_count);
    let allocation = Instant::now();
    let arms: [Vec<Group>; 2] =
        std::array::from_fn(|_| (0..group_count).map(|g| Group::new(&stream, g)).collect());
    let allocation_plan_upload_scan_ms = allocation.elapsed().as_secs_f64() * 1000.0;
    let all_weights: BTreeSet<_> = arms
        .iter()
        .flatten()
        .flat_map(|g| g.weights.iter().map(|w| w.ptr as usize))
        .collect();
    assert_eq!(
        all_weights.len(),
        group_count * 4,
        "A/B and every gate/up leaf need distinct physical weights"
    );
    let mut graphs = Vec::new();
    let _drain = Drain(stream.0);
    let correctness = Instant::now();
    for arm in 0..2 {
        for g in &arms[arm] {
            g.run(&stream, arm);
        }
        stream.sync();
        for g in &arms[arm] {
            g.validate(&g.input, true);
        }
    }
    for i in 0..group_count {
        assert_eq!(arms[0][i].snapshot(), arms[1][i].snapshot());
    }
    let eager_validation_ms = correctness.elapsed().as_secs_f64() * 1000.0;
    let mut capture_ms = [0.0; 2];
    let mut upload_ms = [0.0; 2];
    let mut functions = Vec::new();
    for arm in 0..2 {
        let start = Instant::now();
        let (graph, f) = capture(
            &stream,
            arm,
            group_count * laps,
            "physical_weight_ring",
            || {
                for _ in 0..laps {
                    for g in &arms[arm] {
                        g.run(&stream, arm);
                    }
                }
            },
        );
        capture_ms[arm] = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        unsafe {
            ok(cudaGraphUpload(graph.0, stream.0));
        }
        stream.sync();
        upload_ms[arm] = start.elapsed().as_secs_f64() * 1000.0;
        functions.push(f);
        graphs.push(graph);
    }
    assert_eq!(functions[0], functions[1]);
    let validation = Instant::now();
    let before = arms[0][0].routes[0].result.read::<u16>(N + 7);
    for generation in [1, 2, 3] {
        let finite = generation != 2;
        for arm in 0..2 {
            for (i, g) in arms[arm].iter().enumerate() {
                g.write(100 * generation + i);
                if !finite {
                    let mut x = input(100 * generation + i);
                    x[113] = f32::NAN;
                    g.input.write(&x);
                }
                g.norm.write(&vec![0x7e00_u16; K]);
                for r in &g.routes {
                    r.rows.write(&[1_u32]);
                }
            }
            graphs[arm].launch(&stream);
            stream.sync();
            for g in &arms[arm] {
                g.validate(&g.input, finite);
            }
        }
        for i in 0..group_count {
            assert_eq!(arms[0][i].snapshot(), arms[1][i].snapshot());
        }
    }
    assert_ne!(before, arms[0][0].routes[0].result.read::<u16>(N + 7));
    let expected: Vec<_> = arms[0].iter().map(Group::snapshot).collect();
    let fresh_validation_ms = validation.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..4 {
        for g in &graphs {
            g.launch(&stream);
            stream.sync();
        }
    }
    let start = Event::new();
    let end = Event::new();
    let mut paired = [[[0.0; 4]; 2]; 8];
    for (pair, times) in paired.iter_mut().enumerate() {
        for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            times[arm] = measure(&graphs[arm], &stream, &start, &end);
        }
    }
    stream.sync();
    for arm in 0..2 {
        for (i, g) in arms[arm].iter().enumerate() {
            assert_eq!(g.snapshot(), expected[i]);
        }
    }
    for (pair, times) in paired.iter().enumerate() {
        println!(
            "{}",
            serde_json::json!({"experiment":"rms_mmvq_group_pdl","kind":"pair","pair":pair,
            "order":if pair % 2 == 0 {"AB"} else {"BA"},"metrics_ms":["gpu_complete_graph","complete_block","graph_launch_wall","graph_launch_thread_cpu"],
            "arms":times,"groups_per_graph":group_count * laps})
        );
    }
    println!(
        "{}",
        serde_json::json!({"experiment":"rms_mmvq_group_pdl","kind":"summary",
        "format":"Q4_K","M":1,"K":K,"N_per_leaf":N,"leaves_per_rms":2,"rms_threads":1024,"epsilon":EPSILON,
        "architecture":device.architecture,"l2_bytes":l2,"physical_groups_per_arm":group_count,
        "weight_bytes_per_arm":group_count * 2 * weight_bytes,"laps":laps,"pairs":8,"warm_graphs_per_arm":4,
        "same_content_distinct_weights_between_arms":true,"full_bits":true,"fresh_same_capture":true,
        "nonfinite_recovery":true,"guards":true,"stable_after_timing":true,
        "f64":"full RMS, actual-packed projection sample at three columns; not full projection F64",
        "qualification_ms":qualification_ms,"allocation_plan_upload_scan_ms":allocation_plan_upload_scan_ms,
        "eager_validation_ms":eager_validation_ms,"capture_instantiate_inspect_log_ms":capture_ms,
        "graph_upload_sync_ms":upload_ms,"fresh_validation_ms":fresh_validation_ms,
        "scope":"complete representative RMS + two normal-PDL projections; independent physical weight ring, not a model throughput estimate"})
    );
}
