//! Full-output-tile MMQ diagnostic. Original public plans, pack and cast stay
//! identical; only the test dot uses full tiles without Stream-K fixup.
use super::*;

unsafe extern "C" {
    fn ferrum_test_upstream_mmq_full_tile_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

const M: usize = 8;
const PAIRS: usize = 8;
const DOTS: [MmvqDot; 2] = [
    ffi::ferrum_upstream_mmq_dot_v1,
    ferrum_test_upstream_mmq_full_tile_dot_v1,
];
const LABELS: [&str; 2] = ["public_mmq_stream_k", "full_tile_no_fixup"];

fn enqueue(route: &Route, stream: &Stream, input: &Buffer, weight: *const c_void, arm: usize) {
    route.enqueue_with_native_dot(
        stream,
        input,
        weight,
        0,
        DOTS[arm],
        ffi::ferrum_upstream_mmvq_dot_v1,
    );
}

fn input_words(k: usize, seed: usize, generation: usize) -> Vec<u16> {
    (0..M * k)
        .map(|i| {
            f16::from_f32(((i * 17 + seed * 13 + generation * 29) % 101) as f32 / 2048.0 - 0.024)
                .to_bits()
        })
        .collect()
}

fn mixed_weights(format: UpstreamLinearFormat, k: usize, n: usize, seed: usize) -> Vec<u8> {
    let mut data = weights(format, k, n);
    let bytes = block_bytes(format);
    for (index, block) in data.chunks_exact_mut(bytes).enumerate() {
        // Mixed representable mantissas exercise changed K partition/FMA order.
        let d = (1.0 + ((index + seed) % 17) as f32 / 32.0) / 4096.0;
        block[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        if format != UpstreamLinearFormat::Iq4Xs {
            let dmin = (1.0 + ((index * 3 + seed) % 19) as f32 / 64.0) / 8192.0;
            block[2..4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        }
    }
    data
}

fn block_bytes(format: UpstreamLinearFormat) -> usize {
    match format {
        UpstreamLinearFormat::Iq4Xs => 136,
        UpstreamLinearFormat::Q4K => 144,
        UpstreamLinearFormat::Q5K => 176,
    }
}

fn same_pack_and_flags(routes: &[Route; 2]) {
    for (left, right) in [
        (&routes[0].converted, &routes[1].converted),
        (&routes[0].packed, &routes[1].packed),
        (&routes[0].rows, &routes[1].rows),
        (&routes[0].flags, &routes[1].flags),
    ] {
        assert_eq!(left.read::<u8>(left.bytes), right.read::<u8>(right.bytes));
        left.guards();
        right.guards();
    }
}

#[derive(Default)]
struct Difference {
    values: usize,
    bit_mismatches: usize,
    max_abs: f64,
    first: Vec<serde_json::Value>,
}

impl Difference {
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        group: usize,
        leaf: usize,
        index: usize,
        n: usize,
        left_bits: u32,
        right_bits: u32,
        left: f64,
        right: f64,
    ) {
        self.values += 1;
        if left_bits != right_bits {
            self.bit_mismatches += 1;
            self.max_abs = self.max_abs.max((left - right).abs());
            if self.first.len() < 4 {
                self.first.push(serde_json::json!({
                    "group":group,"leaf":leaf,"row":index/n,"column":index%n,
                    "baseline_bits":left_bits,"candidate_bits":right_bits,
                    "baseline":left,"candidate":right
                }));
            }
        }
    }

    fn merge(&mut self, other: Self) {
        self.values += other.values;
        self.bit_mismatches += other.bit_mismatches;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.first
            .extend(other.first.into_iter().take(4 - self.first.len()));
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "values":self.values,"bit_mismatches":self.bit_mismatches,
            "max_abs":self.max_abs,"first_mismatches":self.first
        })
    }
}

struct Leaf {
    routes: [Route; 2],
    stable: [Vec<u16>; 2],
}

struct Group {
    // Captured addresses must be released before any routes or allocations.
    graphs: [Graph; 2],
    leaves: Vec<Leaf>,
    input: Buffer,
    allocation: Buffer,
    addresses: Vec<*mut c_void>,
    host_weights: Vec<u8>,
    format: UpstreamLinearFormat,
    role: &'static str,
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
        seed: usize,
    ) -> Self {
        let (k, n) = if role == "gate_up" {
            (5120, 17408)
        } else {
            (17408, 5120)
        };
        let host_weights = mixed_weights(format, k, n, seed);
        let stride = host_weights.len().div_ceil(256) * 256;
        let allocation = Buffer::new(count * stride);
        let addresses = (0..count)
            .map(|index| unsafe { allocation.ptr.add(index * stride) })
            .collect::<Vec<_>>();
        for &address in &addresses {
            unsafe {
                ok(cudaMemcpy(
                    address,
                    host_weights.as_ptr().cast(),
                    host_weights.len(),
                    1,
                ));
            }
        }
        let input = Buffer::new(M * k * 2);
        input.write(&input_words(k, seed, 0));
        let leaves = addresses
            .iter()
            .enumerate()
            .map(|(index, &address)| {
                // Gate/up leaves use alternating windows with other-leaf canaries.
                let (result_stride, result_offset) = if role == "gate_up" {
                    (2 * n + 5, (index % 2) * n)
                } else {
                    (n + 5, 0)
                };
                let routes = std::array::from_fn(|_| {
                    Route::new_with_result_window(
                        UpstreamLinearAlgorithm::Mmq,
                        format,
                        M as u32,
                        k as u32,
                        n as u32,
                        1,
                        result_stride as u32,
                        result_offset as u32,
                    )
                });
                let stable = std::array::from_fn(|arm| {
                    routes[arm].scan(stream, address, 0);
                    enqueue(&routes[arm], stream, &input, address, arm);
                    stream.sync();
                    routes[arm].validate(format, &host_weights, 1)
                });
                same_pack_and_flags(&routes);
                Leaf { routes, stable }
            })
            .collect::<Vec<_>>();
        let graphs = [0, 1].map(|arm| {
            Graph::capture(stream, || {
                for (leaf, &address) in leaves.iter().zip(&addresses) {
                    enqueue(&leaf.routes[arm], stream, &input, address, arm);
                }
            })
        });
        Self {
            graphs,
            leaves,
            input,
            allocation,
            addresses,
            host_weights,
            format,
            role,
            k,
            n,
            seed,
        }
    }

    fn enqueue(&self, stream: &Stream, arm: usize) {
        for (leaf, &address) in self.leaves.iter().zip(&self.addresses) {
            enqueue(&leaf.routes[arm], stream, &self.input, address, arm);
        }
    }

    fn verify(&self, phase: &str, fresh: bool) -> [Difference; 2] {
        let mut differences = [Difference::default(), Difference::default()];
        for (index, leaf) in self.leaves.iter().enumerate() {
            let bits = std::array::from_fn::<_, 2, _>(|arm| {
                let bits = leaf.routes[arm].validate(self.format, &self.host_weights, 1);
                if fresh {
                    assert_ne!(
                        bits, leaf.stable[arm],
                        "capture ignored fresh same-address input"
                    );
                } else {
                    assert_eq!(bits, leaf.stable[arm], "same-runtime replay changed output");
                }
                bits
            });
            same_pack_and_flags(&leaf.routes);
            let raw = [0, 1].map(|arm| leaf.routes[arm].output.read::<f32>(M * self.n));
            for output in 0..M * self.n {
                differences[0].observe(
                    self.seed,
                    index,
                    output,
                    self.n,
                    raw[0][output].to_bits(),
                    raw[1][output].to_bits(),
                    f64::from(raw[0][output]),
                    f64::from(raw[1][output]),
                );
                differences[1].observe(
                    self.seed,
                    index,
                    output,
                    self.n,
                    u32::from(bits[0][output]),
                    u32::from(bits[1][output]),
                    f64::from(f16::from_bits(bits[0][output]).to_f32()),
                    f64::from(f16::from_bits(bits[1][output]).to_f32()),
                );
            }
        }
        self.input.guards();
        self.allocation.guards();
        println!(
            "{}",
            serde_json::json!({
                "experiment":"full_tile_stream_k_m8_complete_173_leaf_ring","kind":"correctness",
                "scope":"format_role","phase":phase,"format":format!("{:?}",self.format),
                "role":self.role,"leaves":self.leaves.len(),"K":self.k,"N":self.n,
                "raw_f32":differences[0].json(),"cast_f16":differences[1].json(),
                "oracle":"each arm actual-packed nine-point F64; all values finite and cast checked; all leaf/window/stride guards; identical pack/row/weight flags",
                "cross_arm_policy":"report differences; K partition changes accumulation order"
            })
        );
        differences
    }
}

struct Ring {
    // Full-ring graphs drop before group graphs and every captured buffer.
    graphs: [Graph; 2],
    groups: Vec<Group>,
}

fn timed_pairs(stream: &Stream, graphs: &[Graph; 2], metadata: &serde_json::Value) {
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
            graphs[arm].launch(stream);
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
                serde_json::json!({
                    "experiment":"full_tile_stream_k_m8_complete_173_leaf_ring","kind":"sample",
                    "metadata":metadata,"pair":pair,"order":order,"arm":LABELS[arm],
                    "gpu_ns":f64::from(ms)*1_000_000.0,"host_wall_ns":wall_ns
                })
            );
        }
    }
}

fn boundary_cases(stream: &Stream) {
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for (kind, bad_weight) in [
            (Input::Finite, false),
            (Input::Zero, false),
            (Input::Subnormal, false),
            (Input::Nonfinite, false),
            (Input::SumOverflow, false),
            (Input::Finite, true),
        ] {
            let (k, n) = (256, 128);
            let mut host = mixed_weights(format, k, n, 7);
            if bad_weight {
                host[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
            }
            let expected_bad_weight = host
                .chunks_exact(block_bytes(format))
                .any(|block| oracle::classify_weight(UpstreamLinearAlgorithm::Mmq, format, block));
            let weight = Buffer::new(host.len());
            weight.write(&host);
            let mut words = input_words(k, 7, 0);
            match kind {
                Input::Zero => words.fill(0x8000),
                Input::Subnormal => {
                    for (i, word) in words.iter_mut().enumerate() {
                        *word = if i % 2 == 0 { 1 } else { 0x8001 };
                    }
                }
                Input::Nonfinite => {
                    words[0] = 0x7c00;
                    words[k] = 0xfc00;
                    words[(M - 1) * k] = 0x7e00;
                }
                Input::SumOverflow => words.fill(0x7bff),
                Input::Finite => {}
            }
            let input = Buffer::new(words.len() * 2);
            input.write(&words);
            let routes = std::array::from_fn::<_, 2, _>(|_| {
                Route::new_with_result_window(
                    UpstreamLinearAlgorithm::Mmq,
                    format,
                    M as u32,
                    k as u32,
                    n as u32,
                    1,
                    n as u32 + 5,
                    0,
                )
            });
            for (arm, route) in routes.iter().enumerate() {
                route.scan(stream, weight.ptr, 0);
                enqueue(route, stream, &input, weight.ptr, arm);
            }
            stream.sync();
            same_pack_and_flags(&routes);
            for route in &routes {
                let flags = route.rows.read::<u32>(M);
                assert_eq!(route.flags.read::<u32>(1)[0] != 0, expected_bad_weight);
                let raw = route.output.read::<f32>(M * n);
                let result = route.result.read::<u16>(M * (n + 5));
                let packed = route.packed.read::<u8>(route.packed.bytes);
                let tail = M * route.plan.padded_inputs as usize * 9 / 8;
                assert!(packed[tail..].iter().all(|&byte| byte == 0));
                for row in 0..M {
                    let mut row_bad = false;
                    for group in 0..k / 32 {
                        let values = std::array::from_fn(|i| words[row * k + group * 32 + i]);
                        let (zero, bad) = oracle::marker_classify_pack(
                            UpstreamLinearAlgorithm::Mmq,
                            format,
                            &values,
                        );
                        row_bad |= bad;
                        let block = (group / 4) * M + row;
                        let meta = &packed[block * 144 + (group % 4) * 4..][..4];
                        let codes = &packed[block * 144 + 16 + (group % 4) * 32..][..32];
                        if zero {
                            assert_eq!(meta, &[0; 4]);
                            assert!(codes.iter().all(|&v| v == 0));
                        }
                        if bad {
                            assert!(codes.iter().all(|&v| v == 0));
                            if route.plan.pack_abi == 1 {
                                assert_eq!(
                                    u32::from_le_bytes(meta.try_into().unwrap()),
                                    0x7fc00000
                                );
                            } else {
                                assert_eq!(
                                    u16::from_le_bytes(meta[..2].try_into().unwrap()),
                                    0x7e00
                                );
                            }
                        }
                    }
                    assert_eq!(flags[row] != 0, row_bad);
                    for col in 0..n {
                        assert_eq!(
                            result[row * (n + 5) + col],
                            oracle::final_cast_bits(
                                raw[row * n + col],
                                row_bad,
                                expected_bad_weight
                            )
                        );
                    }
                    assert!(result[row * (n + 5) + n..(row + 1) * (n + 5)]
                        .iter()
                        .all(|&v| v == 0x3535));
                }
                for buffer in [&route.output, &route.fixup, &route.result] {
                    buffer.guards();
                }
                if !bad_weight && matches!(kind, Input::Finite | Input::Zero | Input::Subnormal) {
                    route.validate(format, &host, 1);
                }
            }
            input.guards();
            weight.guards();
            println!(
                "{}",
                serde_json::json!({
                    "experiment":"full_tile_stream_k_m8_complete_173_leaf_ring","kind":"marker_boundary",
                    "format":format!("{format:?}"),"input":format!("{kind:?}"),"bad_weight":bad_weight,
                    "M":M,"K":k,"N":n,"own_marker_flags_cast_and_guards":true
                })
            );
        }
    }
}

#[test]
#[ignore = "exclusive CUDA test-only full-tile MMQ; complete 173-leaf FFN bank"]
fn full_tile_stream_k_m8_complete_173_leaf_ring() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    let stream = Stream::new();
    boundary_cases(&stream);
    let specs = [
        (UpstreamLinearFormat::Iq4Xs, "gate_up", 61),
        (UpstreamLinearFormat::Iq4Xs, "down", 24),
        (UpstreamLinearFormat::Q4K, "gate_up", 28),
        (UpstreamLinearFormat::Q4K, "down", 10),
        (UpstreamLinearFormat::Q5K, "gate_up", 28),
        (UpstreamLinearFormat::Q5K, "down", 22),
    ];
    let groups = specs
        .into_iter()
        .enumerate()
        .map(|(seed, (format, role, count))| Group::new(&stream, format, role, count, seed))
        .collect::<Vec<_>>();
    let graphs = [0, 1].map(|arm| {
        Graph::capture(&stream, || {
            for group in &groups {
                group.enqueue(&stream, arm);
            }
        })
    });
    let ring = Ring { graphs, groups };
    let physical_bytes = ring
        .groups
        .iter()
        .map(|g| g.host_weights.len() * g.leaves.len())
        .sum::<usize>();
    let leaf_count = ring.groups.iter().map(|g| g.leaves.len()).sum::<usize>();
    assert!(
        physical_bytes > 3 * l2 as usize,
        "complete physical bank must exceed three times device L2"
    );
    let scope = serde_json::json!({
        "M":M,"leaves":leaf_count,"physical_weight_bytes":physical_bytes,"device_l2_bytes":l2,
        "banks":1,"pairs":PAIRS,"order":"alternating AB/BA pairs (ABBA); two warm rounds",
        "timed":"complete public pack_v2 and clears, MMQ main and actual fixup, public cast_v2",
        "excluded":"allocation/upload/cold static weight scan/capture/oracle/readback",
        "weight_contents":"same deterministic mixed-mantissa scales within each format/role group; all leaves have distinct physical addresses; same addresses in both arms",
        "candidate":"test-only full-tile blocks=ceil(N/128), no fixup; original validated Plan",
        "scope":"native diagnostic; no product speed, serving latency, or cache-miss claim"
    });
    println!(
        "{}",
        serde_json::json!({"experiment":"full_tile_stream_k_m8_complete_173_leaf_ring","kind":"scope","metadata":scope})
    );
    for group in &ring.groups {
        let p = &group.leaves[0].routes[0].plan;
        timed_pairs(
            &stream,
            &group.graphs,
            &serde_json::json!({
                "scope":"format_role","format":format!("{:?}",group.format),"role":group.role,
                "M":M,"K":group.k,"N":group.n,"leaves":group.leaves.len(),
                "physical_weight_bytes":group.host_weights.len()*group.leaves.len(),
                "baseline_blocks":p.blocks,"baseline_fixup":p.fixup,"baseline_fixup_bytes":p.fixup_bytes,
                "candidate_blocks":group.n.div_ceil(128),"candidate_fixup":false,
                "result_stride":group.leaves[0].routes[0].result_stride,
                "gate_up_offsets":"alternating 0/N; other leaf and five trailing elements remain canaries"
            }),
        );
    }
    timed_pairs(
        &stream,
        &ring.graphs,
        &serde_json::json!({"scope":"weighted_full_ring","configuration":scope}),
    );
    stream.sync();
    for phase in ["stable", "fresh_same_address_input"] {
        if phase != "stable" {
            for group in &ring.groups {
                group.input.write(&input_words(group.k, group.seed, 1));
            }
            for graph in &ring.graphs {
                graph.launch(&stream);
            }
            stream.sync();
        }
        let mut total = [Difference::default(), Difference::default()];
        for group in &ring.groups {
            let [raw, cast] = group.verify(phase, phase != "stable");
            total[0].merge(raw);
            total[1].merge(cast);
        }
        println!(
            "{}",
            serde_json::json!({
                "experiment":"full_tile_stream_k_m8_complete_173_leaf_ring","kind":"correctness",
                "scope":"weighted_full_ring","phase":phase,"leaves":leaf_count,
                "raw_f32":total[0].json(),"cast_f16":total[1].json()
            })
        );
    }
}
