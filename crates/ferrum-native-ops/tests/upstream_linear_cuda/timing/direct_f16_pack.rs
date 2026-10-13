//! Explicit test-only input pack replacement. Both arms keep the same plan,
//! static weights, selected dot/fixup, cast, and output window.
use super::*;

type Pack = unsafe extern "C" fn(
    *const UpstreamLinearPlanV1,
    *const c_void,
    u32,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
) -> i32;

unsafe extern "C" {
    fn ferrum_test_upstream_mmq_pack_f32_intermediate_v1(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmq_pack_f16_direct_v1(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn ferrum_test_upstream_mmvq_pack_f16_direct_v1(
        plan: *const UpstreamLinearPlanV1,
        input: *const c_void,
        stride: u32,
        converted: *mut c_void,
        packed: *mut c_void,
        flags: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn cudaGraphGetNodes(graph: *mut c_void, nodes: *mut *mut c_void, count: *mut usize) -> i32;
}

fn capture_counted(stream: &Stream, enqueue: impl FnOnce()) -> (Graph, usize) {
    unsafe {
        ok(cudaStreamBeginCapture(stream.0, 0));
        enqueue();
        let mut graph = ptr::null_mut();
        ok(cudaStreamEndCapture(stream.0, &mut graph));
        let mut nodes = 0;
        ok(cudaGraphGetNodes(graph, ptr::null_mut(), &mut nodes));
        let mut exec = ptr::null_mut();
        ok(cudaGraphInstantiateWithFlags(&mut exec, graph, 0));
        ok(cudaGraphDestroy(graph));
        (Graph(exec), nodes)
    }
}

#[derive(Clone, Copy)]
struct Shape {
    algorithm: UpstreamLinearAlgorithm,
    format: UpstreamLinearFormat,
    m: usize,
    k: usize,
    n: usize,
}

impl Shape {
    fn layout(self) -> UpstreamLinearLayout {
        if self.algorithm == UpstreamLinearAlgorithm::Mmvq && self.m == 32 {
            UpstreamLinearLayout::Channels
        } else {
            UpstreamLinearLayout::Columns
        }
    }

    fn dot(self) -> MmvqDot {
        if self.m == 8 {
            shared_weight_decode::selected_dot(self.format)
        } else {
            ffi::ferrum_upstream_mmvq_dot_v1
        }
    }
}

fn real_shapes() -> impl Iterator<Item = Shape> {
    [8, 32].into_iter().flat_map(|m| {
        [
            (
                UpstreamLinearAlgorithm::Mmq,
                UpstreamLinearFormat::Iq4Xs,
                5120,
                17408,
            ),
            (
                UpstreamLinearAlgorithm::Mmq,
                UpstreamLinearFormat::Q4K,
                5120,
                17408,
            ),
            (
                UpstreamLinearAlgorithm::Mmvq,
                UpstreamLinearFormat::Q4K,
                5120,
                10240,
            ),
            (
                UpstreamLinearAlgorithm::Mmvq,
                UpstreamLinearFormat::Q5K,
                6144,
                5120,
            ),
        ]
        .into_iter()
        .map(move |(algorithm, format, k, n)| Shape {
            algorithm,
            format,
            m,
            k,
            n,
        })
    })
}

struct Fixture {
    shape: Shape,
    routes: [Route; 2],
    input: Buffer,
    input_stride: usize,
    host_weights: Vec<u8>,
    weights: Buffer,
    weight_stride: usize,
    count: usize,
}

impl Fixture {
    fn new(shape: Shape, ring_bytes: usize) -> Self {
        let host_weights = weights(shape.format, shape.k, shape.n);
        let count = if ring_bytes == 0 {
            1
        } else {
            ring_bytes / host_weights.len() + 1
        };
        let weight_stride = host_weights.len().div_ceil(256) * 256;
        let weights = Buffer::new(count * weight_stride);
        for index in 0..count {
            unsafe {
                ok(cudaMemcpy(
                    weights.ptr.add(index * weight_stride),
                    host_weights.as_ptr().cast(),
                    host_weights.len(),
                    1,
                ));
            }
        }
        let input_stride = shape.k + 3;
        let input = Buffer::new((1 + shape.m * input_stride) * 2);
        let routes = [0, 1].map(|_| {
            Route::new_with_layout_and_result_window(
                shape.algorithm,
                shape.format,
                shape.layout(),
                shape.m as u32,
                shape.k as u32,
                shape.n as u32,
                count,
                shape.n as u32 + 5,
                1,
            )
        });
        Self {
            shape,
            routes,
            input,
            input_stride,
            host_weights,
            weights,
            weight_stride,
            count,
        }
    }

    fn write_input(&self, seed: usize, edges: bool) -> Vec<u16> {
        let mut input = vec![0x3535; 1 + self.shape.m * self.input_stride];
        for row in 0..self.shape.m {
            for col in 0..self.shape.k {
                input[1 + row * self.input_stride + col] = if edges {
                    match row % 8 {
                        0 => 0,
                        1 => 0x8000,
                        2 => {
                            if col % 2 == 0 {
                                1
                            } else {
                                0x8001
                            }
                        }
                        3 if col == 0 => 0x7c00,
                        4 if col == 0 => 0xfc00,
                        5 if col == 0 => 0x7e01,
                        6 => 0x7bff,
                        _ => f16::from_f32(((col + row * 7) % 19) as f32 / 64.0 - 0.14).to_bits(),
                    }
                } else {
                    f16::from_f32(
                        ((col * 17 + row * 7) % 101) as f32 / 2048.0 - 0.024 + seed as f32 / 32.0,
                    )
                    .to_bits()
                };
            }
        }
        self.input.write(&input);
        input
    }

    fn scan(&self, stream: &Stream) {
        for route in &self.routes {
            for index in 0..self.count {
                route.scan(
                    stream,
                    unsafe { self.weights.ptr.add(index * self.weight_stride) },
                    index,
                );
            }
        }
        stream.sync();
    }

    fn enqueue(&self, stream: &Stream, arm: usize, index: usize) {
        let pack: Pack = match (self.shape.algorithm, arm) {
            (UpstreamLinearAlgorithm::Mmq, 0) => ferrum_test_upstream_mmq_pack_f32_intermediate_v1,
            (UpstreamLinearAlgorithm::Mmq, _) => ferrum_test_upstream_mmq_pack_f16_direct_v1,
            (UpstreamLinearAlgorithm::Mmvq, 0) => ffi::ferrum_upstream_mmvq_pack_v2,
            (UpstreamLinearAlgorithm::Mmvq, _) => ferrum_test_upstream_mmvq_pack_f16_direct_v1,
        };
        self.enqueue_with_pack(stream, arm, index, pack);
    }

    fn enqueue_with_pack(&self, stream: &Stream, arm: usize, index: usize, pack: Pack) {
        let route = &self.routes[arm];
        let p = &route.plan;
        unsafe {
            ok(pack(
                p,
                self.input.ptr.add(2),
                self.input_stride as u32,
                route.converted.ptr,
                route.packed.ptr,
                route.rows.ptr,
                stream.0,
            ));
            let dot = if self.shape.algorithm == UpstreamLinearAlgorithm::Mmq {
                ffi::ferrum_upstream_mmq_dot_v1 as MmvqDot
            } else {
                self.shape.dot()
            };
            ok(dot(
                p,
                self.weights.ptr.add(index * self.weight_stride),
                route.packed.ptr,
                route.output.ptr,
                if p.fixup_bytes == 0 {
                    ptr::null_mut()
                } else {
                    route.fixup.ptr
                },
                stream.0,
            ));
            let cast = if self.shape.algorithm == UpstreamLinearAlgorithm::Mmq {
                ffi::ferrum_upstream_mmq_cast_v2
            } else {
                ffi::ferrum_upstream_mmvq_cast_v2
            };
            ok(cast(
                p,
                route.output.ptr,
                route.result.ptr.add(route.result_offset as usize * 2),
                route.result_stride,
                route.rows.ptr,
                route.flags.ptr.add(index * 4),
                stream.0,
            ));
        }
    }

    fn assert_equal(&self, host_input: &[u16], finite: bool) -> Vec<u16> {
        self.assert_equal_with_converted(host_input, finite, true)
    }

    fn assert_equal_with_converted(
        &self,
        host_input: &[u16],
        finite: bool,
        direct: bool,
    ) -> Vec<u16> {
        let [a, b] = &self.routes;
        for (name, left, right) in [
            ("packed including guard", &a.packed, &b.packed),
            ("F32 output", &a.output, &b.output),
            ("fixup", &a.fixup, &b.fixup),
            ("F16 output and padding", &a.result, &b.result),
            ("row flags", &a.rows, &b.rows),
            ("weight flags", &a.flags, &b.flags),
        ] {
            assert_eq!(left.bytes, right.bytes);
            assert_eq!(
                left.read::<u8>(left.bytes),
                right.read::<u8>(right.bytes),
                "{name}"
            );
            left.guards();
            right.guards();
        }
        let converted = b.converted.read::<u8>(b.converted.bytes);
        if direct {
            assert_eq!(
                converted,
                vec![0x35; b.converted.bytes],
                "direct pack wrote the legacy converted buffer"
            );
        } else {
            assert_eq!(converted, a.converted.read::<u8>(a.converted.bytes));
            assert_ne!(
                converted,
                vec![0x35; b.converted.bytes],
                "large-row product pack did not execute its legacy conversion"
            );
        }
        a.converted.guards();
        b.converted.guards();
        assert_eq!(self.input.read::<u16>(host_input.len()), host_input);
        self.input.guards();
        self.weights.guards();
        assert_eq!(a.flags.read::<u32>(self.count), vec![0; self.count]);
        let row_flags = a.rows.read::<u32>(self.shape.m);
        let output = a.output.read::<u32>(self.shape.m * self.shape.n);
        let result = a.result.read::<u16>(a.result.bytes / 2);
        for row in 0..self.shape.m {
            let bad = (0..self.shape.k / 32).any(|group| {
                let values = std::array::from_fn(|i| {
                    host_input[1 + row * self.input_stride + group * 32 + i]
                });
                oracle::marker_classify_pack(self.shape.algorithm, self.shape.format, &values).1
            });
            assert_eq!(row_flags[row] != 0, bad);
            let start = row * a.result_stride as usize;
            assert_eq!(result[start], 0x3535);
            assert!(
                result[start + 1 + self.shape.n..start + a.result_stride as usize]
                    .iter()
                    .all(|&value| value == 0x3535)
            );
            for col in 0..self.shape.n {
                assert_eq!(
                    result[start + 1 + col],
                    oracle::final_cast_bits(
                        f32::from_bits(output[row * self.shape.n + col]),
                        bad,
                        false
                    )
                );
            }
        }
        if a.plan.guard_blocks > 0 {
            let packed = a.packed.read::<u8>(a.packed.bytes);
            let body = self.shape.m * a.plan.padded_inputs as usize * 9 / 8;
            assert!(packed[body..].iter().all(|&byte| byte == 0));
        }
        if finite {
            for route in &self.routes {
                route.validate(self.shape.format, &self.host_weights, self.count);
            }
        }
        result
    }
}

#[test]
#[ignore = "exclusive CUDA test-only direct F16 pack ABI; no product routing change"]
fn direct_f16_pack_preserves_complete_projection_bits_and_boundaries() {
    let stream = Stream::new();
    // The small K=768 cases add actual K padding; real shapes cover selected
    // shared M8 dot and Channels M32 without changing either arm's arithmetic.
    let padding_shapes = real_shapes()
        .filter(|s| s.m == 8)
        .map(|s| Shape { k: 768, n: 64, ..s });
    for shape in real_shapes().chain(padding_shapes) {
        let fixture = Fixture::new(shape, 0);
        fixture.scan(&stream);
        let initial_input = fixture.write_input(0, false);
        for arm in 0..2 {
            fixture.enqueue(&stream, arm, 0);
        }
        stream.sync();
        let initial = fixture.assert_equal(&initial_input, true);
        let graphs = [0, 1].map(|arm| Graph::capture(&stream, || fixture.enqueue(&stream, arm, 0)));
        let edge_input = fixture.write_input(0, true);
        for graph in &graphs {
            graph.launch(&stream);
        }
        stream.sync();
        fixture.assert_equal(&edge_input, false);
        let fresh_input = fixture.write_input(1, false);
        for graph in &graphs {
            graph.launch(&stream);
        }
        stream.sync();
        let fresh = fixture.assert_equal(&fresh_input, true);
        assert_ne!(initial, fresh, "captured pack ignored changed input");
        println!(
            "{}",
            serde_json::json!({"experiment":"direct_f16_pack_correctness",
            "algorithm":format!("{:?}",shape.algorithm),"format":format!("{:?}",shape.format),
            "M":shape.m,"K":shape.k,"N":shape.n,"layout":format!("{:?}",shape.layout()),
            "checked":"all packed/guard/row flags/F32/fixup/F16 bytes, converted untouched, canaries, odd stride/view offsets, signed zero/subnormals/NaN/Inf/overflow, fresh capture and marker reset"})
        );
    }
}

#[test]
#[ignore = "exclusive CUDA product MMQ pack-v2 small/large row boundary correctness"]
fn product_mmq_pack_v2_preserves_bits_across_small_and_prefill_boundaries() {
    let stream = Stream::new();
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for m in [1, 3, 32, 33, 75] {
            let shape = Shape {
                algorithm: UpstreamLinearAlgorithm::Mmq,
                format,
                m,
                k: 768,
                n: 64,
            };
            let fixture = Fixture::new(shape, 0);
            fixture.scan(&stream);
            let enqueue = |arm| {
                fixture.enqueue_with_pack(
                    &stream,
                    arm,
                    0,
                    if arm == 0 {
                        ferrum_test_upstream_mmq_pack_f32_intermediate_v1
                    } else {
                        ffi::ferrum_upstream_mmq_pack_v2
                    },
                );
            };
            let initial_input = fixture.write_input(0, false);
            for arm in 0..2 {
                enqueue(arm);
            }
            stream.sync();
            let initial = fixture.assert_equal_with_converted(&initial_input, true, m <= 32);
            let graphs = [0, 1].map(|arm| Graph::capture(&stream, || enqueue(arm)));
            let edge_input = fixture.write_input(0, true);
            for graph in &graphs {
                graph.launch(&stream);
            }
            stream.sync();
            fixture.assert_equal_with_converted(&edge_input, false, m <= 32);
            let fresh_input = fixture.write_input(1, false);
            for graph in &graphs {
                graph.launch(&stream);
            }
            stream.sync();
            let fresh = fixture.assert_equal_with_converted(&fresh_input, true, m <= 32);
            assert_ne!(
                initial, fresh,
                "product captured pack ignored changed input"
            );
            println!(
                "{}",
                serde_json::json!({
                "experiment":"product_mmq_pack_v2_boundary_correctness",
                "format":format!("{format:?}"),"M":m,"K":shape.k,"N":shape.n,
                "actual_product_symbol":"ferrum_upstream_mmq_pack_v2",
                "baseline_symbol":"ferrum_test_upstream_mmq_pack_f32_intermediate_v1",
                "expected_path":if m<=32 {"direct_f16; converted untouched"} else {"legacy_f32; converted bytes equal baseline"},
                "checked":"full packed/guard/F32/fixup/F16/flags/padding/canaries, fresh capture, markers then finite reset"})
            );
        }
    }
}

fn thread_cpu_ns() -> Option<u128> {
    #[cfg(target_os = "linux")]
    {
        let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
            return None;
        }
        let time = unsafe { time.assume_init() };
        let seconds = u128::try_from(time.tv_sec).ok()?;
        let nanos = u128::try_from(time.tv_nsec).ok()?;
        (nanos < 1_000_000_000).then_some(seconds * 1_000_000_000 + nanos)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[test]
#[ignore = "exclusive CUDA complete-projection long-graph paired diagnostic; run correctness first"]
fn direct_f16_pack_complete_projection_ring_pairs() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const WARM: usize = 2;
    const PAIRS: usize = 8;
    for shape in real_shapes() {
        let fixture = Fixture::new(shape, 3 * l2 as usize);
        let stream = Stream::new();
        fixture.scan(&stream);
        let host_input = fixture.write_input(0, false);
        for arm in 0..2 {
            for index in 0..fixture.count {
                fixture.enqueue(&stream, arm, index);
            }
        }
        stream.sync();
        let stable = fixture.assert_equal(&host_input, true);
        // Approximately 320 full projections, not a tiny one-projection graph.
        // The physical weight ring exceeds 3x L2 and repeats without host work.
        let laps = 320_usize.div_ceil(fixture.count);
        let graphs = [0, 1].map(|arm| {
            capture_counted(&stream, || {
                for _ in 0..laps {
                    for index in 0..fixture.count {
                        fixture.enqueue(&stream, arm, index);
                    }
                }
            })
        });
        let (start, end) = (Event::new(), Event::new());
        let mut samples = Vec::with_capacity(2 * PAIRS * 2);
        for launches in [1, 2] {
            for round in 0..WARM + PAIRS {
                for order in 0..2 {
                    let arm = (round + order) % 2;
                    // Both blocks begin idle. The queued mode has no wait
                    // between its two graph launches and only one final fence.
                    stream.sync();
                    let mut launch_wall = [0_u128; 2];
                    let mut launch_cpu = [None; 2];
                    let block = Instant::now();
                    unsafe {
                        ok(cudaEventRecord(start.0, stream.0));
                    }
                    for index in 0..launches {
                        let wall = Instant::now();
                        let cpu_start = thread_cpu_ns();
                        let graph = &graphs[arm].0;
                        let status = unsafe { cudaGraphLaunch(graph.0, stream.0) };
                        let cpu_end = thread_cpu_ns();
                        launch_wall[index] = wall.elapsed().as_nanos();
                        launch_cpu[index] =
                            cpu_start.zip(cpu_end).and_then(|(s, e)| e.checked_sub(s));
                        ok(status);
                    }
                    unsafe {
                        ok(cudaEventRecord(end.0, stream.0));
                        ok(cudaEventSynchronize(end.0));
                    }
                    let block_wall_ns = block.elapsed().as_nanos();
                    let mut gpu_ms = 0.0;
                    unsafe {
                        ok(cudaEventElapsedTime(&mut gpu_ms, start.0, end.0));
                    }
                    if round >= WARM {
                        samples.push((
                            launches,
                            round - WARM,
                            order,
                            arm,
                            launch_wall,
                            launch_cpu,
                            block_wall_ns,
                            f64::from(gpu_ms) * 1_000_000.0,
                        ));
                    }
                }
            }
        }
        stream.sync();
        assert_eq!(fixture.assert_equal(&host_input, true), stable);
        let fresh_input = fixture.write_input(1, false);
        for (graph, _) in &graphs {
            graph.launch(&stream);
        }
        stream.sync();
        assert_ne!(fixture.assert_equal(&fresh_input, true), stable);
        for (launches, pair, order, arm, launch_wall, launch_cpu, block_wall_ns, gpu_ns) in samples
        {
            let mut record = serde_json::json!({"experiment":"direct_f16_pack_complete_projection_ring",
                "algorithm":format!("{:?}",shape.algorithm),"format":format!("{:?}",shape.format),
                "M":shape.m,"K":shape.k,"N":shape.n,"layout":format!("{:?}",shape.layout()),
                "dot":if shape.algorithm==UpstreamLinearAlgorithm::Mmvq && shape.m==8 {"current_shared_m8"} else {"unchanged_baseline_dot"},
                "route_scope":if (shape.algorithm==UpstreamLinearAlgorithm::Mmq && shape.format==UpstreamLinearFormat::Q4K) || (shape.algorithm==UpstreamLinearAlgorithm::Mmvq && shape.m==32) {"pack_layout_diagnostic; no assertion this is the current product route"} else {"current_ffn_or_attention_route"},
                "input_stride":fixture.input_stride,"input_offset_bytes":2,
                "output_stride":shape.n+5,"output_offset_elements":1,
                "matrix_count":fixture.count,"weight_bytes":fixture.host_weights.len(),
                "working_set_bytes":fixture.count*fixture.host_weights.len(),"device_l2_bytes":l2,
                "matrix_contents":"same deterministic bytes at distinct ring addresses; both arms same physical weights",
                "laps":laps,"projections_per_graph":laps*fixture.count});
            record["graph"] = serde_json::json!({
                "graph_nodes":graphs[arm].1,"baseline_graph_nodes":graphs[0].1,"candidate_graph_nodes":graphs[1].1,
                "mode":if launches==1 {"idle_single"} else {"idle_then_queued_two"},
                "graph_launches_per_block":launches,"projections_per_block":launches*laps*fixture.count,
                "launch_api":"cudaGraphLaunch runtime API; product cuGraphLaunch is a different API boundary",
                "clocks":"monotonic wall brackets raw launch and adjacent thread-clock reads; optional Linux calling-thread CPU; status check outside launch clocks",
                "warm_pairs":WARM,"pairs":PAIRS,
                "timed":"pack including row flags and MMQ guard/dot including actual fixup/cast",
                "excluded":"allocation, upload, static scan, capture, oracle, checks, readback, pre-block idle fence",
                "scope":"micro mechanism only, not model throughput; event span is not sum of kernel times; CPU may include busy-spin"});
            record["sample"] = serde_json::json!({"pair":pair,"order":order,"arm":if arm==0 {"baseline_f32_intermediate"} else {"direct_f16"},
                    "gpu_ns":gpu_ns,"block_wall_ns":block_wall_ns,
                    "graph_launch_wall_ns":&launch_wall[..launches],"graph_launch_thread_cpu_ns":&launch_cpu[..launches]});
            println!("{record}");
        }
    }
}
