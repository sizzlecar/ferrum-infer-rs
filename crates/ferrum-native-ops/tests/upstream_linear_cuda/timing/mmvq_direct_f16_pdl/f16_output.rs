//! Complete M1 chains: current PDL, fused output, and direct input + fused output.
//! Reuses the existing collection, buffers, actual-pack oracle and timing scope.
use super::*;

unsafe extern "C" {
    fn ferrum_test_mmvq_f16_output_pdl_support_v1(
        plan: *const UpstreamLinearPlanV1,
        supported: *mut u32,
    ) -> i32;
    fn ferrum_test_mmvq_f16_output_pdl_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        output_stride: u32,
        row_flags: *const c_void,
        weight_flag: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_mmvq_f16_output_pdl_graph_v1(
        graph: *mut c_void,
        counts: *mut u64,
        count: usize,
    ) -> i32;
}

const ARMS: [&str; 3] = ["baseline", "output_only", "input_and_output"];

// Route's ordinary timing constructor intentionally requires padded N == N.
// This local constructor also admits odd logical N, with zero weight padding.
fn route(format: UpstreamLinearFormat, k: usize, n: usize, copies: usize) -> Route {
    let p = plan(
        UpstreamLinearAlgorithm::Mmvq,
        format,
        UpstreamLinearLayout::Columns,
        1,
        k as u32,
        n as u32,
    );
    Route {
        algorithm: UpstreamLinearAlgorithm::Mmvq,
        converted: Buffer::new(p.converted_bytes as usize),
        packed: Buffer::new(p.packed_bytes as usize),
        output: Buffer::new(p.output_bytes as usize),
        fixup: Buffer::new(p.fixup_bytes.max(4) as usize),
        result: Buffer::new((n + 7) * 2),
        rows: Buffer::new(4),
        flags: Buffer::new(copies * 4),
        result_stride: (n + 7) as u32,
        result_offset: 3,
        plan: p,
    }
}

struct CollectionBank {
    base: Bank,
    combined: Route,
    combined_weights: Buffer,
    // An independent strided window for EVERY logical projection, not merely
    // the final snapshot of each physical scratch buffer.
    results: [Buffer; 3],
}
impl CollectionBank {
    fn new(s: &Stream, f: UpstreamLinearFormat, k: usize, n: usize, calls: usize) -> Self {
        let copies = calls.min(3);
        let routes = std::array::from_fn(|_| route(f, k, n, copies));
        let combined = route(f, k, n, copies);
        let padded = combined.plan.padded_outputs as usize;
        let mut host = weights(f, k, padded);
        let row_bytes = host.len() / padded;
        host[n * row_bytes..].fill(0);
        let weight_stride = host.len().div_ceil(256) * 256;
        let make_weights = || {
            let b = Buffer::new(copies * weight_stride);
            for i in 0..copies {
                unsafe {
                    ok(cudaMemcpy(
                        b.ptr.add(i * weight_stride),
                        host.as_ptr().cast(),
                        host.len(),
                        1,
                    ));
                }
            }
            b
        };
        let weights = std::array::from_fn(|_| make_weights());
        let combined_weights = make_weights();
        let base = Bank {
            format: f,
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
        base.set_input(0, 0);
        let b = Self {
            base,
            combined,
            combined_weights,
            results: std::array::from_fn(|_| Buffer::new(calls * (n + 7) * 2)),
        };
        b.scan(s);
        b
    }
    fn r(&self, arm: usize) -> &Route {
        if arm == 2 {
            &self.combined
        } else {
            &self.base.routes[arm]
        }
    }
    fn w(&self, arm: usize) -> &Buffer {
        if arm == 2 {
            &self.combined_weights
        } else {
            &self.base.weights[arm]
        }
    }
    fn scan(&self, s: &Stream) {
        for arm in 0..3 {
            for i in 0..self.base.copies {
                self.r(arm).scan(
                    s,
                    unsafe { self.w(arm).ptr.add(i * self.base.weight_stride) },
                    i,
                );
            }
        }
    }
    fn supported(&self) -> bool {
        let (mut direct, mut output) = (0, 0);
        unsafe {
            ok(ferrum_test_mmvq_direct_pdl_support_v1(
                &self.combined.plan,
                &mut direct,
            ));
            ok(ferrum_test_mmvq_f16_output_pdl_support_v1(
                &self.combined.plan,
                &mut output,
            ));
        }
        assert!(direct <= 1 && output <= 1);
        direct == 1 && output == 1
    }
    fn enqueue(
        &self,
        s: &Stream,
        arm: usize,
        physical: usize,
        input: &Buffer,
        output: *mut c_void,
    ) {
        let r = self.r(arm);
        unsafe {
            let pack: MmvqPack = if arm == 2 {
                ferrum_test_mmvq_direct_pdl_pack_v1
            } else {
                ferrum_upstream_mmvq_pdl_pack_v2
            };
            ok(pack(
                &r.plan,
                input.ptr.add(6),
                (self.base.k + 7) as u32,
                r.converted.ptr,
                r.packed.ptr,
                r.rows.ptr,
                s.0,
            ));
            let weight = self.w(arm).ptr.add(physical * self.base.weight_stride);
            let flag = r.flags.ptr.add(physical * 4);
            if arm == 0 {
                ok(ferrum_upstream_mmvq_pdl_dot_v1(
                    &r.plan,
                    weight,
                    r.packed.ptr,
                    r.output.ptr,
                    ptr::null_mut(),
                    s.0,
                    0,
                ));
                ok(ferrum_upstream_mmvq_pdl_cast_v2(
                    &r.plan,
                    r.output.ptr,
                    output,
                    r.result_stride,
                    r.rows.ptr,
                    flag,
                    s.0,
                ));
            } else {
                ok(ferrum_test_mmvq_f16_output_pdl_dot_v1(
                    &r.plan,
                    weight,
                    r.packed.ptr,
                    output,
                    r.result_stride,
                    r.rows.ptr,
                    flag,
                    s.0,
                ));
            }
        }
    }
    fn leaf(&self, s: &Stream, arm: usize, visit: usize) {
        self.enqueue(s, arm, visit % self.base.copies, &self.base.input, unsafe {
            self.results[arm]
                .ptr
                .add((visit * (self.base.n + 7) + 3) * 2)
        });
    }
    fn eager_expected(&self, s: &Stream, input: &Buffer, finite: bool) -> Vec<u16> {
        let r = self.r(0);
        self.enqueue(s, 0, 0, input, unsafe { r.result.ptr.add(6) });
        s.sync();
        if finite {
            r.validate(self.base.format, &self.base.host, self.base.copies)
        } else {
            r.result.read::<u16>(self.base.n + 7)[3..3 + self.base.n].to_vec()
        }
    }
    fn check_buffer(&self, b: &Buffer, expected: &[u16], calls: usize) {
        for row in b
            .read::<u16>(calls * (self.base.n + 7))
            .chunks_exact(self.base.n + 7)
        {
            assert!(
                row[3..3 + self.base.n] == *expected,
                "complete retained F16 output differs from normal eager oracle"
            );
            assert!(
                row[..3]
                    .iter()
                    .chain(&row[3 + self.base.n..])
                    .all(|&x| x == 0x3535),
                "output stride/prefix overwritten"
            );
        }
        b.guards();
    }
    fn check(&self, expected: &[u16]) {
        for arm in 0..3 {
            self.check_buffer(&self.results[arm], expected, self.base.calls);
            let r = self.r(arm);
            for (x, y) in [
                (&r.packed, &self.r(0).packed),
                (&r.rows, &self.r(0).rows),
                (&r.flags, &self.r(0).flags),
            ] {
                assert!(
                    x.read::<u8>(x.bytes) == y.read::<u8>(y.bytes),
                    "packed input/markers differ"
                );
            }
            if arm > 0 {
                assert!(
                    r.output
                        .read::<u8>(r.output.bytes)
                        .iter()
                        .all(|&x| x == 0x35),
                    "fused dot touched removed raw output scratch"
                );
            }
            assert!(r.fixup.read::<u8>(r.fixup.bytes).iter().all(|&x| x == 0x35));
            for b in [
                &r.converted,
                &r.packed,
                &r.output,
                &r.fixup,
                &r.result,
                &r.rows,
                &r.flags,
                self.w(arm),
            ] {
                b.guards();
            }
        }
        assert!(
            self.combined
                .converted
                .read::<u8>(self.combined.converted.bytes)
                .iter()
                .all(|&x| x == 0x35),
            "direct pack touched removed converted scratch"
        );
        assert!(
            self.r(0).converted.read::<u8>(self.r(0).converted.bytes)
                == self.r(1).converted.read::<u8>(self.r(1).converted.bytes)
        );
        self.base.input.guards();
    }
}

fn inspect_output(graph: *mut c_void, arm: usize, calls: usize) {
    let mut c = [0_u64; 16];
    unsafe {
        ok(ferrum_test_mmvq_f16_output_pdl_graph_v1(
            graph,
            c.as_mut_ptr(),
            c.len(),
        ));
    }
    let n = calls as u64;
    assert_eq!(c[0], n * [5, 4, 3][arm]);
    assert_eq!((c[1], c[2], c[3]), (c[0], 0, n));
    assert_eq!(
        (c[4], c[5], c[6]),
        if arm == 2 { (0, 0, n) } else { (n, n, 0) }
    );
    assert_eq!(
        (c[7], c[8], c[9]),
        if arm == 0 { (n, 0, n) } else { (0, n, 0) }
    );
    assert_eq!((c[10], c[11]), (c[0] - 1, c[0] - 1));
    assert_eq!(
        (c[12], c[13]),
        if arm == 0 { (n - 1, 0) } else { (0, n - 1) }
    );
    assert_eq!(
        (c[14], c[15]),
        match arm {
            1 => (n, 0),
            2 => (0, n),
            _ => (0, 0),
        }
    );
    println!(
        "{}",
        serde_json::json!({"kind":"actual_graph","arm":ARMS[arm],"calls":calls,"counts":c,"fields":["nodes","kernels","other","clear","widen","f32pack","f16pack","normal_dot","fused_dot","cast","edges","programmatic_edges","cast_to_next_clear","fused_dot_to_next_clear","f32pack_to_fused","f16pack_to_fused"]})
    );
}

fn boundary_output(s: &Stream, f: UpstreamLinearFormat, k: usize, n: usize) -> bool {
    let b = CollectionBank::new(s, f, k, n, 1);
    let _drain = Drain(s.0);
    if !b.supported() {
        println!(
            "{}",
            serde_json::json!({"kind":"unsupported","format":format!("{f:?}"),"K":k,"N":n,"timing_collected":false})
        );
        return false;
    }
    let initial = b.eager_expected(s, &b.base.input, true);
    let graphs = [0, 1, 2].map(|arm| {
        Graph::capture_inspected(s, || b.leaf(s, arm, 0), |g| inspect_output(g, arm, 1))
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
        b.base.set_input(generation, edge);
        let expected = b.eager_expected(s, &b.base.input, edge <= 3);
        if edge == 0 {
            assert!(
                expected != initial,
                "fresh eager input does not distinguish stale replay"
            );
        }
        for g in &graphs {
            g.launch(s);
        }
        s.sync();
        b.check(&expected);
        if (4..=6).contains(&edge) {
            assert_ne!(b.r(0).rows.read::<u32>(1)[0], 0);
            assert!(expected.iter().all(|&x| x == 0x7e00));
        }
        if edge == 7 {
            assert!(
                expected.contains(&0x7e00),
                "half overflow was not exercised"
            );
        }
        assert_eq!(
            b.base.input.read::<u16>(k + 10),
            b.base.words(generation, edge)
        );
    }
    // Use the REAL scanner, not a hand-injected flag. Restore the same physical
    // allocation and rescan, then replay the same captured graph to prove reset.
    b.base.set_input(10, 0);
    for bad in [true, false] {
        let bits = if bad {
            0x7c00_u16.to_le_bytes()
        } else {
            [b.base.host[0], b.base.host[1]]
        };
        for arm in 0..3 {
            unsafe {
                ok(cudaMemcpy(b.w(arm).ptr, bits.as_ptr().cast(), 2, 1));
            }
        }
        b.scan(s);
        s.sync();
        for arm in 0..3 {
            assert_eq!(
                b.r(arm).flags.read::<u32>(1)[0] != 0,
                bad,
                "actual weight scanner status"
            );
        }
        let expected = b.eager_expected(s, &b.base.input, !bad);
        for g in &graphs {
            g.launch(s);
        }
        s.sync();
        b.check(&expected);
        if bad {
            assert!(expected.iter().all(|&x| x == 0x7e00));
        }
    }
    drop(graphs);
    // Shared scratch across consecutive projections and different input
    // addresses: keep the first finite result observable after the second dot.
    let second = Buffer::new(b.base.input.bytes);
    let first: [Buffer; 3] = std::array::from_fn(|_| Buffer::new((n + 7) * 2));
    let _hazard_drain = Drain(s.0);
    b.base.set_input(11, 0);
    let expected_first = b.eager_expected(s, &b.base.input, true);
    second.write(&b.base.words(12, 0));
    let graphs = [0, 1, 2].map(|arm| {
        Graph::capture_inspected(
            s,
            || {
                b.enqueue(s, arm, 0, &b.base.input, unsafe { first[arm].ptr.add(6) });
                b.enqueue(s, arm, 0, &second, unsafe { b.results[arm].ptr.add(6) });
            },
            |g| inspect_output(g, arm, 2),
        )
    });
    for generation in [12, 13] {
        second.write(&b.base.words(generation, 0));
        let expected_second = b.eager_expected(s, &second, true);
        assert!(expected_second != expected_first);
        for arm in 0..3 {
            b.r(arm).rows.write(&[u32::MAX]);
        }
        for g in &graphs {
            g.launch(s);
        }
        s.sync();
        b.check(&expected_second);
        for r in &first {
            b.check_buffer(r, &expected_first, 1);
        }
        second.guards();
    }
    // The next clear must also wait for consumption of the PREVIOUS bad-row
    // flag. A finite-only retained result cannot expose premature flag reset.
    b.base.set_input(14, 6);
    let bad_first = b.eager_expected(s, &b.base.input, false);
    assert!(bad_first.iter().all(|&x| x == 0x7e00));
    second.write(&b.base.words(15, 0));
    let finite_second = b.eager_expected(s, &second, true);
    for g in &graphs {
        g.launch(s);
    }
    s.sync();
    b.check(&finite_second);
    for r in &first {
        b.check_buffer(r, &bad_first, 1);
    }
    second.guards();
    println!(
        "{}",
        serde_json::json!({"kind":"correctness","format":format!("{f:?}"),"K":k,"N":n,"full_f16_bits":true,"actual_packed_f64_samples":true,"actual_corrupt_weight_scanner_and_recovery":true,"fresh_matched_eager":true,"shared_scratch_first_result":true,"shared_scratch_nonfinite_then_finite":true,"removed_raw_canary":true,"strides_and_guards":true})
    );
    drop(graphs);
    s.sync();
    true
}

#[test]
#[ignore = "exclusive CUDA complete M1 PDL chains with direct F16 output"]
fn mmvq_m1_f16_output_pdl_preserves_bits_and_complete_ring_pairs() {
    let device = caps();
    let s = Stream::new();
    for f in [
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
        UpstreamLinearFormat::Iq4Xs,
    ] {
        if !boundary_output(&s, f, 5120, 17408) || !boundary_output(&s, f, 768, 33) {
            return;
        }
    }
    let banks = shapes()
        .into_iter()
        .map(|(f, k, n, c)| CollectionBank::new(&s, f, k, n, c))
        .collect::<Vec<_>>();
    let _drain = Drain(s.0);
    let calls: usize = banks.iter().map(|b| b.base.calls).sum();
    let weights_per_arm: usize = banks
        .iter()
        .map(|b| b.base.copies * b.base.weight_stride)
        .sum();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    let mut expected = Vec::new();
    for b in &banks {
        if !b.supported() {
            println!(
                "{}",
                serde_json::json!({"kind":"unsupported","scope":"ring","timing_collected":false})
            );
            return;
        }
        expected.push(b.eager_expected(&s, &b.base.input, true));
    }
    let graphs = [0, 1, 2].map(|arm| {
        Graph::capture_inspected(
            &s,
            || {
                for visit in 0..banks.iter().map(|b| b.base.calls).max().unwrap() {
                    for b in &banks {
                        if visit < b.base.calls {
                            b.leaf(&s, arm, visit);
                        }
                    }
                }
            },
            |g| inspect_output(g, arm, calls),
        )
    });
    for g in &graphs {
        unsafe {
            ok(cudaGraphUpload(g.0, s.0));
        }
        for _ in 0..8 {
            g.launch(&s);
        }
    }
    s.sync();
    for (b, e) in banks.iter().zip(&expected) {
        b.check(e);
    }
    let start = Event::new();
    let end = Event::new();
    for pair in 0..8 {
        // Each candidate has its OWN baseline measurement. Reverse both the
        // comparison order and A/B order on alternate pairs.
        for candidate in if pair % 2 == 0 { [1, 2] } else { [2, 1] } {
            let mut measured = [[0.0; 4]; 2];
            for arm in if pair % 2 == 0 {
                [0, candidate]
            } else {
                [candidate, 0]
            } {
                measured[usize::from(arm != 0)] = measure(&graphs[arm], &s, &start, &end);
            }
            println!(
                "{}",
                serde_json::json!({"kind":"complete_projection_pair","candidate":ARMS[candidate],"pair":pair,"order":if pair%2==0 {"AB"}else{"BA"},"calls":calls,"baseline_ms":measured[0],"candidate_ms":measured[1],"fields":["gpu_graph","submit_wait","graph_launch_wall","graph_launch_thread_cpu"]})
            );
        }
    }
    for (b, e) in banks.iter().zip(&expected) {
        b.check(e);
    }
    for (b, e) in banks.iter().zip(&mut expected) {
        b.base.set_input(21, 0);
        let fresh = b.eager_expected(&s, &b.base.input, true);
        assert!(
            fresh != *e,
            "matched fresh eager output does not distinguish stale ring"
        );
        *e = fresh;
    }
    for g in &graphs {
        g.launch(&s);
    }
    s.sync();
    for (b, e) in banks.iter().zip(&expected) {
        b.check(e);
    }
    println!(
        "{}",
        serde_json::json!({"kind":"scope","M":1,"architecture":device.architecture,"projection_calls":calls,"shape_count":banks.len(),"physical_weights_per_arm":banks.iter().map(|b|b.base.copies).sum::<usize>(),"weight_bytes_per_arm":weights_per_arm,"l2_bytes":l2,"working_set_over_three_l2":weights_per_arm>3*l2 as usize,"arms":ARMS,"weights":"three distinct physical sets with matched contents; up to three matrices per shape; L2 ratio does not prove eviction","order":"round-robin shapes and rotating physical copies, not model layer order","validation":"all logical F16 output windows, full packed/marker equality, sampled actual-packed F64 normal oracle; removed raw scratch canaries","included":"every clear/pack/dot/cast of each complete arm, graph GPU and submit-to-completion wall","excluded":"allocation, scanning, capture/upload, validation and all other model operators","admission":"each candidate separately: >=3% aggregate GPU and submit_wait improvement, both AB/BA groups positive; analysis rule, not a correctness assertion or model gain"})
    );
    drop(graphs);
    s.sync();
}
