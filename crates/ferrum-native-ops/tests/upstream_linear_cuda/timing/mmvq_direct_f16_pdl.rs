//! Current five-kernel PDL MMVQ versus four-kernel direct-half PDL.
//! Model-derived projection counts, synthetic weights, no whole-model claim.
use super::*;

unsafe extern "C" {
    fn ferrum_test_mmvq_direct_pdl_support_v1(
        p: *const UpstreamLinearPlanV1,
        supported: *mut u32,
    ) -> i32;
    fn ferrum_test_mmvq_direct_pdl_pack_v1(
        p: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_mmvq_direct_pdl_graph_v1(
        graph: *mut c_void,
        counts: *mut u64,
        count: usize,
    ) -> i32;
    fn ferrum_upstream_mmvq_pdl_pack_v2(
        p: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_upstream_mmvq_pdl_dot_v1(
        p: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
        kind: u32,
    ) -> i32;
    fn ferrum_upstream_mmvq_pdl_cast_v2(
        p: *const UpstreamLinearPlanV1,
        input: *const c_void,
        output: *mut c_void,
        stride: u32,
        rows: *const c_void,
        weights: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn cudaGraphUpload(graph: *mut c_void, stream: *mut c_void) -> i32;
}

struct Drain(*mut c_void);
impl Drop for Drain {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaStreamSynchronize(self.0);
        }
    }
}

// Counts from the observed M1 projection inventory; no assertion that arbitrary
// models have these counts. Twenty format/K/N combinations, six output widths.
fn shapes() -> Vec<(UpstreamLinearFormat, usize, usize, usize)> {
    use UpstreamLinearFormat::{Iq4Xs as I, Q4K as Q4, Q5K as Q5};
    vec![
        (Q4, 5120, 10240, 29),
        (Q4, 5120, 6144, 24),
        (Q4, 5120, 1024, 1),
        (Q4, 5120, 17408, 28),
        (Q4, 6144, 5120, 5),
        (Q4, 17408, 5120, 10),
        (Q4, 5120, 12288, 6),
        (Q5, 5120, 10240, 12),
        (Q5, 5120, 6144, 12),
        (Q5, 5120, 1024, 11),
        (Q5, 5120, 17408, 28),
        (Q5, 6144, 5120, 44),
        (Q5, 17408, 5120, 22),
        (Q5, 5120, 12288, 2),
        (I, 5120, 10240, 6),
        (I, 5120, 6144, 12),
        (I, 5120, 17408, 61),
        (I, 6144, 5120, 6),
        (I, 17408, 5120, 24),
        (I, 5120, 12288, 8),
    ]
}

struct Bank {
    format: UpstreamLinearFormat,
    k: usize,
    n: usize,
    calls: usize,
    copies: usize,
    weight_stride: usize,
    host: Vec<u8>,
    weights: [Buffer; 2],
    input: Buffer,
    routes: [Route; 2],
}
impl Bank {
    fn new(
        stream: &Stream,
        format: UpstreamLinearFormat,
        k: usize,
        n: usize,
        calls: usize,
    ) -> Self {
        let host = weights(format, k, n);
        let copies = calls.min(3);
        let weight_stride = host.len().div_ceil(256) * 256;
        let weights = std::array::from_fn(|_| {
            let allocation = Buffer::new(copies * weight_stride);
            for i in 0..copies {
                unsafe {
                    ok(cudaMemcpy(
                        allocation.ptr.add(i * weight_stride),
                        host.as_ptr().cast(),
                        host.len(),
                        1,
                    ));
                }
            }
            allocation
        });
        let routes = std::array::from_fn(|_| {
            Route::new_with_result_window(
                UpstreamLinearAlgorithm::Mmvq,
                format,
                1,
                k as u32,
                n as u32,
                copies,
                (n + 7) as u32,
                3,
            )
        });
        for arm in 0..2 {
            for i in 0..copies {
                routes[arm].scan(
                    stream,
                    unsafe { weights[arm].ptr.add(i * weight_stride) },
                    i,
                );
            }
        }
        let bank = Self {
            format,
            k,
            n,
            calls,
            copies,
            weight_stride,
            host,
            weights,
            input: Buffer::new((k + 10) * 2),
            routes,
        };
        bank.set_input(0, 0);
        bank
    }
    fn words(&self, generation: usize, edge: usize) -> Vec<u16> {
        let mut words = vec![0x3535; self.k + 10];
        for i in 0..self.k {
            words[i + 3] = match edge {
                1 => 0,
                2 => 0x8000,
                3 => {
                    if i % 2 == 0 {
                        1
                    } else {
                        0x8001
                    }
                }
                4 if i == 0 => 0x7c00,
                5 if i == 0 => 0xfc00,
                6 if i == 0 => 0x7e01,
                7 => 0x7bff,
                _ => {
                    f16::from_f32(((i * 17 + generation * 29) % 113) as f32 / 512.0 - 0.1).to_bits()
                }
            };
        }
        words
    }
    fn set_input(&self, generation: usize, edge: usize) {
        self.input.write(&self.words(generation, edge));
    }
    fn enqueue(
        &self,
        stream: &Stream,
        arm: usize,
        index: usize,
        input: &Buffer,
        retained: Option<&Buffer>,
    ) {
        let r = &self.routes[arm];
        unsafe {
            let pack: MmvqPack = if arm == 0 {
                ferrum_upstream_mmvq_pdl_pack_v2
            } else {
                ferrum_test_mmvq_direct_pdl_pack_v1
            };
            ok(pack(
                &r.plan,
                input.ptr.add(6),
                (self.k + 7) as u32,
                r.converted.ptr,
                r.packed.ptr,
                r.rows.ptr,
                stream.0,
            ));
            ok(ferrum_upstream_mmvq_pdl_dot_v1(
                &r.plan,
                self.weights[arm].ptr.add(index * self.weight_stride),
                r.packed.ptr,
                r.output.ptr,
                ptr::null_mut(),
                stream.0,
                0,
            ));
            ok(ferrum_upstream_mmvq_pdl_cast_v2(
                &r.plan,
                r.output.ptr,
                retained.unwrap_or(&r.result).ptr.add(6),
                r.result_stride,
                r.rows.ptr,
                r.flags.ptr.add(index * 4),
                stream.0,
            ));
        }
    }
    fn equal(&self, finite: bool) -> Vec<u16> {
        let a = &self.routes[0];
        let b = &self.routes[1];
        for (x, y) in [
            (&a.packed, &b.packed),
            (&a.output, &b.output),
            (&a.result, &b.result),
            (&a.rows, &b.rows),
            (&a.flags, &b.flags),
            (&a.fixup, &b.fixup),
        ] {
            assert_eq!(x.read::<u8>(x.bytes), y.read::<u8>(y.bytes));
            x.guards();
            y.guards();
        }
        assert_eq!(
            b.converted.read::<u8>(b.converted.bytes),
            vec![0x35; b.converted.bytes],
            "direct pack touched removed intermediate"
        );
        for r in &self.routes {
            r.converted.guards();
            let result = r.result.read::<u16>(self.n + 7);
            assert!(result[..3]
                .iter()
                .chain(&result[3 + self.n..])
                .all(|&x| x == 0x3535));
            let flags = r.rows.read::<u32>(1)[0];
            for (&raw, &bits) in r
                .output
                .read::<f32>(self.n)
                .iter()
                .zip(&result[3..3 + self.n])
            {
                assert_eq!(bits, oracle::final_cast_bits(raw, flags != 0, false));
            }
        }
        for w in &self.weights {
            w.guards();
        }
        self.input.guards();
        if finite {
            let result = a.validate(self.format, &self.host, self.copies);
            assert_eq!(result, b.validate(self.format, &self.host, self.copies));
            result
        } else {
            a.result.read::<u16>(self.n + 7)[3..3 + self.n].to_vec()
        }
    }
}

fn inspect(graph: *mut c_void, arm: usize, calls: usize) {
    let mut c = [0_u64; 13];
    unsafe {
        ok(ferrum_test_mmvq_direct_pdl_graph_v1(
            graph,
            c.as_mut_ptr(),
            c.len(),
        ));
    }
    let calls = calls as u64;
    assert_eq!(c[0], calls * if arm == 0 { 5 } else { 4 });
    assert_eq!(c[1], c[0]);
    assert_eq!(c[2], 0);
    assert_eq!(c[3], calls);
    assert_eq!(c[7], calls);
    assert_eq!(
        (c[4], c[5], c[6]),
        if arm == 0 {
            (calls, calls, 0)
        } else {
            (0, 0, calls)
        }
    );
    assert_eq!(c[8], c[0] - 1);
    assert_eq!(c[9], c[8]);
    assert_eq!(c[10], calls - 1);
    if arm == 1 {
        assert_eq!(c[11], calls);
        assert_eq!(c[12], calls);
    }
    println!(
        "{}",
        serde_json::json!({"kind":"actual_graph","arm":arm,"calls":calls,"counts":c,"fields":["nodes","kernel","other","clear","widen","f32pack","f16pack","cast","edges","programmatic","cast_to_clear","clear_to_direct","direct_to_dot"]})
    );
}

fn boundary(stream: &Stream, format: UpstreamLinearFormat, k: usize, n: usize) {
    let b = Bank::new(stream, format, k, n, 1);
    let _drain = Drain(stream.0);
    let mut supported = 0;
    unsafe {
        ok(ferrum_test_mmvq_direct_pdl_support_v1(
            &b.routes[1].plan,
            &mut supported,
        ));
    }
    assert_eq!(
        supported, 1,
        "this diagnostic requires actual PDL support; no ordinary-vs-ordinary result"
    );
    for arm in 0..2 {
        b.enqueue(stream, arm, 0, &b.input, None);
    }
    stream.sync();
    let initial = b.equal(true);
    let graphs = [0, 1].map(|arm| {
        Graph::capture_inspected(
            stream,
            || b.enqueue(stream, arm, 0, &b.input, None),
            |g| inspect(g, arm, 1),
        )
    });
    for (generation, edge) in [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (7, 6),
        (8, 7),
        (9, 0),
    ] {
        b.set_input(generation, edge);
        for g in &graphs {
            g.launch(stream);
        }
        stream.sync();
        let result = b.equal(edge <= 3);
        if edge == 0 {
            assert_ne!(result, initial, "captured input stayed stale");
        }
        if (4..=6).contains(&edge) {
            assert_ne!(b.routes[1].rows.read::<u32>(1)[0], 0);
            assert!(result.iter().all(|&x| x == 0x7e00));
        }
        if edge == 7 {
            assert!(
                result.contains(&0x7e00),
                "overflow case did not exercise half/packed-sum marker"
            );
        }
        assert_eq!(b.input.read::<u16>(b.k + 10), b.words(generation, edge));
    }
    drop(graphs);
    // Two projections share every scratch span, with different captured input
    // addresses and retained first output. Fresh replay must preserve both.
    let second = Buffer::new(b.input.bytes);
    let first = [
        Buffer::new(b.routes[0].result.bytes),
        Buffer::new(b.routes[1].result.bytes),
    ];
    let _hazard_drain = Drain(stream.0);
    b.set_input(11, 0);
    for arm in 0..2 {
        b.enqueue(stream, arm, 0, &b.input, None);
    }
    stream.sync();
    let expected = b.equal(true);
    second.write(&b.words(12, 0));
    let graphs = [0, 1].map(|arm| {
        Graph::capture_inspected(
            stream,
            || {
                b.enqueue(stream, arm, 0, &b.input, Some(&first[arm]));
                b.enqueue(stream, arm, 0, &second, None);
            },
            |g| inspect(g, arm, 2),
        )
    });
    for generation in [12, 13] {
        second.write(&b.words(generation, 0));
        for r in &b.routes {
            r.rows.write(&[u32::MAX]);
        }
        for g in &graphs {
            g.launch(stream);
        }
        stream.sync();
        let last = b.equal(true);
        assert_ne!(last, expected);
        for retained in &first {
            let all = retained.read::<u16>(b.n + 7);
            assert_eq!(&all[3..3 + b.n], expected.as_slice());
            assert!(all[..3].iter().chain(&all[3 + b.n..]).all(|&x| x == 0x3535));
            retained.guards();
        }
    }
    println!(
        "{}",
        serde_json::json!({"kind":"correctness","format":format!("{format:?}"),"K":k,"N":n,"full_bits_q8_scales_raw_cast_flags":true,"fresh_and_marker_reset":true,"shared_scratch_retained_first":true,"oracle":"actual packed F64 samples, not full F64"})
    );
    drop(graphs);
    stream.sync();
}

fn cpu_ns() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) },
        0
    );
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
fn measure(g: &Graph, s: &Stream, start: &Event, end: &Event) -> [f64; 4] {
    let block = Instant::now();
    unsafe {
        ok(cudaEventRecord(start.0, s.0));
    }
    let cpu = cpu_ns();
    let now = Instant::now();
    g.launch(s);
    let api = now.elapsed().as_secs_f64() * 1000.0;
    let cpu = (cpu_ns() - cpu) as f64 / 1e6;
    unsafe {
        ok(cudaEventRecord(end.0, s.0));
        ok(cudaEventSynchronize(end.0));
    }
    let wall = block.elapsed().as_secs_f64() * 1000.0;
    let mut gpu = 0.0;
    unsafe {
        ok(cudaEventElapsedTime(&mut gpu, start.0, end.0));
    }
    [gpu as f64, wall, api, cpu]
}

#[test]
#[ignore = "exclusive CUDA M1 direct-half versus current full-PDL complete projection collection"]
fn mmvq_m1_direct_f16_pdl_preserves_bits_and_complete_ring_pairs() {
    let device = caps();
    let stream = Stream::new();
    for format in [
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
        UpstreamLinearFormat::Iq4Xs,
    ] {
        boundary(&stream, format, 5120, 17408);
        boundary(&stream, format, 768, 64);
    }
    let banks = shapes()
        .into_iter()
        .map(|(f, k, n, c)| Bank::new(&stream, f, k, n, c))
        .collect::<Vec<_>>();
    let _drain = Drain(stream.0);
    let calls: usize = banks.iter().map(|b| b.calls).sum();
    let weights_per_arm: usize = banks.iter().map(|b| b.copies * b.weight_stride).sum();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    for b in &banks {
        let mut supported = 0;
        unsafe {
            ok(ferrum_test_mmvq_direct_pdl_support_v1(
                &b.routes[1].plan,
                &mut supported,
            ));
        }
        assert_eq!(supported, 1);
        for arm in 0..2 {
            b.enqueue(&stream, arm, 0, &b.input, None);
        }
        stream.sync();
        b.equal(true);
    }
    // Interleave shapes; rotate up to three physical matrices for each shape.
    // The counts match the recorded projection collection, not its exact order.
    let graphs = [0, 1].map(|arm| {
        Graph::capture_inspected(
            &stream,
            || {
                for visit in 0..banks.iter().map(|b| b.calls).max().unwrap() {
                    for b in &banks {
                        if visit < b.calls {
                            b.enqueue(&stream, arm, visit % b.copies, &b.input, None);
                        }
                    }
                }
            },
            |g| inspect(g, arm, calls),
        )
    });
    for g in &graphs {
        unsafe {
            ok(cudaGraphUpload(g.0, stream.0));
        }
        for _ in 0..8 {
            g.launch(&stream);
        }
    }
    stream.sync();
    for b in &banks {
        b.equal(true);
    }
    let start = Event::new();
    let end = Event::new();
    for pair in 0..8 {
        let mut measured = [[0.0; 4]; 2];
        for position in 0..2 {
            let arm = if pair % 2 == 0 {
                position
            } else {
                1 - position
            };
            measured[arm] = measure(&graphs[arm], &stream, &start, &end);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"complete_projection_pair","pair":pair,"order":if pair%2==0 {"AB"} else {"BA"},"calls":calls,"baseline_ms":measured[0],"direct_ms":measured[1],"fields":["gpu_graph","submit_wait","graph_launch_wall","graph_launch_thread_cpu"]})
        );
    }
    let stable = banks.iter().map(|b| b.equal(true)).collect::<Vec<_>>();
    for b in &banks {
        b.set_input(21, 0);
    }
    for g in &graphs {
        g.launch(&stream);
    }
    stream.sync();
    for (b, old) in banks.iter().zip(stable) {
        assert_ne!(b.equal(true), old);
    }
    println!(
        "{}",
        serde_json::json!({"kind":"scope","M":1,"architecture":device.architecture,"projection_calls":calls,"shape_count":banks.len(),"physical_weights_per_arm":banks.iter().map(|b|b.copies).sum::<usize>(),"weight_bytes_per_arm":weights_per_arm,"l2_bytes":l2,"working_set_over_three_l2":weights_per_arm>3*l2 as usize,"weights":"A/B distinct addresses with matched contents; up to three physical matrices per shape reused at model-derived counts; reported L2 ratio is neither proof of eviction nor a correctness gate","order":"round-robin shapes, not model layer order","included":"clear/direct-or-widen+pack/dot/cast; actual graph and completion wall","excluded":"cold allocations/scans/capture/upload/validation, all other model operators","validation":"full final snapshot per shape, not independent snapshot for every repeated logical leaf; fresh captured inputs and actual packed F64 samples"})
    );
    drop(graphs);
    stream.sync();
}
