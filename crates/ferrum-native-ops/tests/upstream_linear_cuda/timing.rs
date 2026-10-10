//! Directly paired MarkerV2 costs on identical physical weights. These are
//! controlled working sets, not a measurement of DRAM misses or model speed.
//! No host copies, validation, resets, or allocations occur between timed pairs.
use super::*;
use std::time::Instant;

unsafe extern "C" {
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64)
        -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

struct Graph(*mut c_void);
impl Graph {
    fn capture(stream: &Stream, enqueue: impl FnOnce()) -> Self {
        unsafe {
            ok(cudaStreamBeginCapture(stream.0, 0));
            enqueue();
            let mut graph = ptr::null_mut();
            ok(cudaStreamEndCapture(stream.0, &mut graph));
            let mut exec = ptr::null_mut();
            ok(cudaGraphInstantiateWithFlags(&mut exec, graph, 0));
            ok(cudaGraphDestroy(graph));
            Self(exec)
        }
    }
    fn launch(&self, stream: &Stream) {
        unsafe { ok(cudaGraphLaunch(self.0, stream.0)) }
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaGraphExecDestroy(self.0);
        }
    }
}
struct Event(*mut c_void);
impl Event {
    fn new() -> Self {
        let mut event = ptr::null_mut();
        unsafe { ok(cudaEventCreate(&mut event)) };
        Self(event)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaEventDestroy(self.0);
        }
    }
}

struct Route {
    algorithm: UpstreamLinearAlgorithm,
    plan: UpstreamLinearPlanV1,
    converted: Buffer,
    packed: Buffer,
    output: Buffer,
    fixup: Buffer,
    result: Buffer,
    rows: Buffer,
    flags: Buffer,
}
impl Route {
    fn new(
        algorithm: UpstreamLinearAlgorithm,
        format: UpstreamLinearFormat,
        m: u32,
        k: u32,
        n: u32,
        count: usize,
    ) -> Self {
        let p = plan(algorithm, format, UpstreamLinearLayout::Columns, m, k, n);
        assert_eq!(
            p.padded_outputs, n,
            "benchmark uses complete physical output rows"
        );
        Self {
            algorithm,
            converted: Buffer::new(p.converted_bytes as usize),
            packed: Buffer::new(p.packed_bytes as usize),
            output: Buffer::new(p.output_bytes as usize),
            fixup: Buffer::new(p.fixup_bytes.max(4) as usize),
            result: Buffer::new(m as usize * n as usize * 2),
            rows: Buffer::new(m as usize * 4),
            flags: Buffer::new(count * 4),
            plan: p,
        }
    }
    fn scan(&self, stream: &Stream, weight: *const c_void, index: usize) {
        unsafe {
            let flag = self.flags.ptr.add(index * 4);
            ok(match self.algorithm {
                UpstreamLinearAlgorithm::Mmq => {
                    ffi::ferrum_upstream_mmq_check_weights_v2(&self.plan, weight, flag, stream.0)
                }
                UpstreamLinearAlgorithm::Mmvq => {
                    ffi::ferrum_upstream_mmvq_check_weights_v2(&self.plan, weight, flag, stream.0)
                }
            });
        }
    }
    fn enqueue(&self, stream: &Stream, input: &Buffer, weight: *const c_void, index: usize) {
        let p = &self.plan;
        unsafe {
            let flag = self.flags.ptr.add(index * 4);
            let fixup = if p.fixup_bytes == 0 {
                ptr::null_mut()
            } else {
                self.fixup.ptr
            };
            match self.algorithm {
                UpstreamLinearAlgorithm::Mmq => {
                    ok(ffi::ferrum_upstream_mmq_pack_v2(
                        p,
                        input.ptr,
                        p.request.inputs,
                        self.converted.ptr,
                        self.packed.ptr,
                        self.rows.ptr,
                        stream.0,
                    ));
                    ok(ffi::ferrum_upstream_mmq_dot_v1(
                        p,
                        weight,
                        self.packed.ptr,
                        self.output.ptr,
                        fixup,
                        stream.0,
                    ));
                    ok(ffi::ferrum_upstream_mmq_cast_v2(
                        p,
                        self.output.ptr,
                        self.result.ptr,
                        p.request.outputs,
                        self.rows.ptr,
                        flag,
                        stream.0,
                    ));
                }
                UpstreamLinearAlgorithm::Mmvq => {
                    ok(ffi::ferrum_upstream_mmvq_pack_v2(
                        p,
                        input.ptr,
                        p.request.inputs,
                        self.converted.ptr,
                        self.packed.ptr,
                        self.rows.ptr,
                        stream.0,
                    ));
                    ok(ffi::ferrum_upstream_mmvq_dot_v1(
                        p,
                        weight,
                        self.packed.ptr,
                        self.output.ptr,
                        fixup,
                        stream.0,
                    ));
                    ok(ffi::ferrum_upstream_mmvq_cast_v2(
                        p,
                        self.output.ptr,
                        self.result.ptr,
                        p.request.outputs,
                        self.rows.ptr,
                        flag,
                        stream.0,
                    ));
                }
            }
        }
    }
    fn validate(&self, format: UpstreamLinearFormat, weights: &[u8], count: usize) -> Vec<u16> {
        let p = &self.plan;
        let (m, k, n) = (
            p.request.rows as usize,
            p.request.inputs as usize,
            p.request.outputs as usize,
        );
        assert_eq!(self.rows.read::<u32>(m), vec![0; m]);
        assert_eq!(self.flags.read::<u32>(count), vec![0; count]);
        let values = self.output.read::<f32>(m * n);
        let bits = self.result.read::<u16>(m * n);
        for (value, &bits) in values.iter().zip(&bits) {
            assert!(value.is_finite());
            assert!(f16::from_bits(bits).is_finite());
            assert_eq!(bits, oracle::final_cast_bits(*value, false, false));
        }
        let packed = self.packed.read::<u8>(p.packed_bytes as usize);
        let wb = match format {
            UpstreamLinearFormat::Iq4Xs => 136,
            UpstreamLinearFormat::Q4K => 144,
            UpstreamLinearFormat::Q5K => 176,
        };
        for row in [0, m / 2, m - 1] {
            for col in [0, n / 2, n - 1] {
                let (mut target, mut magnitude) = (0.0, 0.0);
                for group in 0..k / 32 {
                    let (meta, codes) = if self.algorithm == UpstreamLinearAlgorithm::Mmq {
                        let b = (group / 4) * m + row;
                        (
                            &packed[b * 144 + (group % 4) * 4..][..4],
                            &packed[b * 144 + 16 + (group % 4) * 32..][..32],
                        )
                    } else {
                        let b = row * (p.padded_inputs as usize / 32) + group;
                        (&packed[b * 36..][..4], &packed[b * 36 + 4..][..32])
                    };
                    let scale = if p.pack_abi == 1 {
                        f32::from_le_bytes(meta.try_into().unwrap())
                    } else {
                        f16::from_bits(u16::from_le_bytes(meta[..2].try_into().unwrap())).to_f32()
                    };
                    let sum = if p.pack_abi == 1 {
                        0.0
                    } else {
                        f16::from_bits(u16::from_le_bytes(meta[2..].try_into().unwrap())).to_f32()
                    };
                    let offset = (col * (k / 256) + group / 8) * wb;
                    let (v, mag) = oracle::declared_group(
                        self.algorithm,
                        format,
                        &weights[offset..offset + wb],
                        group % 8,
                        &std::array::from_fn(|i| codes[i] as i8),
                        scale,
                        sum,
                    );
                    target += v;
                    magnitude += mag;
                }
                let nu = (k as f64 / 32.0 + 32.0) * f32::EPSILON as f64;
                let bound = nu / (1.0 - nu) * magnitude + 32.0 * f32::MIN_POSITIVE as f64;
                assert!(
                    (values[row * n + col] as f64 - target).abs() <= bound,
                    "actual-pack oracle {:?}/{format:?} row{row} col{col}",
                    self.algorithm
                );
            }
        }
        for b in [
            &self.converted,
            &self.packed,
            &self.output,
            &self.fixup,
            &self.result,
            &self.rows,
            &self.flags,
        ] {
            b.guards();
        }
        bits
    }
}

#[test]
#[ignore = "exclusive CUDA MarkerV2 paired diagnostic; run correctness first"]
fn marker_v2_m8_mmq_mmvq_paired_working_sets() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const WARM: usize = 8;
    const PAIRS: usize = 16;
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        for (k, n) in [(5120, 17408), (17408, 5120)] {
            let m = 8;
            let host_weights = weights(format, k, n);
            let weight_bytes = host_weights.len();
            for mode in ["resident", "arena_offset_ring", "distinct_allocation_ring"] {
                let count = if mode == "resident" {
                    1
                } else {
                    3 * l2 as usize / weight_bytes + 1
                };
                let stride = weight_bytes.div_ceil(256) * 256;
                let allocations = if mode == "arena_offset_ring" {
                    vec![Buffer::new(count * stride)]
                } else {
                    (0..count)
                        .map(|_| Buffer::new(weight_bytes))
                        .collect::<Vec<_>>()
                };
                let addresses = (0..count)
                    .map(|index| unsafe {
                        if mode == "arena_offset_ring" {
                            allocations[0].ptr.add(index * stride)
                        } else {
                            allocations[index].ptr
                        }
                    })
                    .collect::<Vec<_>>();
                for &address in &addresses {
                    unsafe {
                        ok(cudaMemcpy(
                            address,
                            host_weights.as_ptr().cast(),
                            weight_bytes,
                            1,
                        ));
                    }
                }
                let input = Buffer::new(m * k * 2);
                input.write(
                    &(0..m * k)
                        .map(|i| f16::from_f32(((i * 17) % 101) as f32 / 2048.0 - 0.024).to_bits())
                        .collect::<Vec<_>>(),
                );
                let stream = Stream::new();
                let routes = [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq]
                    .map(|a| Route::new(a, format, m as u32, k as u32, n as u32, count));
                let mut stable = Vec::new();
                for route in &routes {
                    for (index, &address) in addresses.iter().enumerate() {
                        route.scan(&stream, address, index);
                        route.enqueue(&stream, &input, address, index);
                    }
                    stream.sync();
                    stable.push(route.validate(format, &host_weights, count));
                }
                // At least 32 projections in each graph. A ring makes multiple
                // complete laps without intervening host validation or resets.
                let laps = 32_usize.div_ceil(count);
                let graphs = routes.each_ref().map(|route| {
                    Graph::capture(&stream, || {
                        for _ in 0..laps {
                            for (index, &address) in addresses.iter().enumerate() {
                                route.enqueue(&stream, &input, address, index);
                            }
                        }
                    })
                });
                let (start, end) = (Event::new(), Event::new());
                // Reserve host storage before timing. Delay all JSON creation
                // and formatting until every measured graph has completed.
                let mut samples = Vec::with_capacity(PAIRS * 2);
                for round in 0..WARM + PAIRS {
                    for order in 0..2 {
                        let route = (round + order) % 2;
                        let wall = Instant::now();
                        unsafe {
                            ok(cudaEventRecord(start.0, stream.0));
                        }
                        graphs[route].launch(&stream);
                        unsafe {
                            ok(cudaEventRecord(end.0, stream.0));
                            ok(cudaEventSynchronize(end.0));
                        }
                        let wall_ns = wall.elapsed().as_nanos();
                        let mut ms = 0.0;
                        unsafe {
                            ok(cudaEventElapsedTime(&mut ms, start.0, end.0));
                        }
                        if round >= WARM {
                            samples.push((
                                round - WARM,
                                order,
                                route,
                                f64::from(ms) * 1_000_000.0,
                                wall_ns,
                            ));
                        }
                    }
                }
                stream.sync();
                for (index, route) in routes.iter().enumerate() {
                    assert_eq!(route.validate(format, &host_weights, count), stable[index]);
                }
                input.guards();
                for allocation in &allocations {
                    allocation.guards();
                }
                for (pair, order, route, gpu_ns, wall_ns) in samples {
                    let sample = serde_json::json!({"pair":pair,"order":order,
                        "algorithm":format!("{:?}",routes[route].algorithm),
                        "gpu_ns":gpu_ns,"wall_ns":wall_ns});
                    println!(
                        "{}",
                        serde_json::json!({"experiment":"marker_v2_direct_paired",
                        "format":format!("{format:?}"),"M":m,"K":k,"N":n,"weight_mode":mode,
                        "weight_bytes":weight_bytes,"matrix_count":count,"working_set_bytes":count*weight_bytes,
                        "device_l2_bytes":l2,"same_physical_weights_between_algorithms":true,
                        "matrix_contents":"same deterministic bytes; distinct addresses in rings",
                        "laps":laps,"projections":laps*count,"warm_rounds":WARM,"pairs":PAIRS,
                        "timed":"F16-to-F32, MarkerV2 row checks/pack, dot/fixup, MarkerV2 cast",
                        "excluded":"cold plan/weight scan, allocation, input upload, oracle and readbacks",
                        "between_timed_routes":"events and graph submission only; no reset/readback",
                        "correctness":"all finite/cast bits/canaries/repeat; nine actual-pack F64 samples per route",
                        "sample":sample})
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "exclusive CUDA MarkerV2 M32 MMQ counter and working-set diagnostic"]
fn marker_v2_m32_mmq_working_sets() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const M: usize = 32;
    const WARM: usize = 8;
    const FORMAL: usize = 16;
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q5K,
        UpstreamLinearFormat::Q4K,
    ] {
        for (k, n) in [(5120, 17408), (17408, 5120)] {
            let host_weights = weights(format, k, n);
            let weight_bytes = host_weights.len();
            for mode in ["resident", "arena_offset_ring"] {
                let count = if mode == "resident" {
                    1
                } else {
                    3 * l2 as usize / weight_bytes + 1
                };
                let stride = weight_bytes.div_ceil(256) * 256;
                let allocation = Buffer::new(count * stride);
                let addresses = (0..count)
                    .map(|index| unsafe { allocation.ptr.add(index * stride) })
                    .collect::<Vec<_>>();
                for &address in &addresses {
                    unsafe {
                        ok(cudaMemcpy(
                            address,
                            host_weights.as_ptr().cast(),
                            weight_bytes,
                            1,
                        ));
                    }
                }
                let input = Buffer::new(M * k * 2);
                input.write(
                    &(0..M * k)
                        .map(|i| f16::from_f32(((i * 17) % 101) as f32 / 2048.0 - 0.024).to_bits())
                        .collect::<Vec<_>>(),
                );
                let stream = Stream::new();
                let route = Route::new(
                    UpstreamLinearAlgorithm::Mmq,
                    format,
                    M as u32,
                    k as u32,
                    n as u32,
                    count,
                );
                // One cold weight scan per physical address, outside capture
                // and all timed replays. Every address is validated before use.
                for (index, &address) in addresses.iter().enumerate() {
                    route.scan(&stream, address, index);
                }
                stream.sync();
                let mut stable = None;
                for (index, &address) in addresses.iter().enumerate() {
                    route.enqueue(&stream, &input, address, index);
                    stream.sync();
                    let bits = route.validate(format, &host_weights, count);
                    if let Some(expected) = &stable {
                        assert_eq!(&bits, expected);
                    } else {
                        stable = Some(bits);
                    }
                }
                input.guards();
                allocation.guards();
                let p = &route.plan;
                let plan_report = serde_json::json!({"request_rows":p.request.rows,
                    "request_inputs":p.request.inputs,"request_outputs":p.request.outputs,
                    "sm_count":p.request.sm_count,"algorithm":p.algorithm,"pack_abi":p.pack_abi,
                    "padded_inputs":p.padded_inputs,"padded_outputs":p.padded_outputs,
                    "j":p.j,"i":p.i,"nthreads":p.nthreads,"nwarps":p.nwarps,
                    "blocks":p.blocks,"tiles_y":p.tiles_y,"shared_bytes":p.shared_bytes,
                    "fixup":p.fixup,"fixup_bytes":p.fixup_bytes,"guard_blocks":p.guard_blocks,
                    "ncols":p.ncols,"channels":p.channels,"rows_per_block":p.rows_per_block,
                    "small_k":p.small_k,"weight_bytes":p.weight_bytes,
                    "converted_bytes":p.converted_bytes,"packed_bytes":p.packed_bytes,
                    "output_bytes":p.output_bytes});
                println!(
                    "{}",
                    serde_json::json!({"experiment":"marker_v2_m32_mmq",
                        "record":"validated_before_replay","format":format!("{format:?}"),
                        "M":M,"K":k,"N":n,"weight_mode":mode,
                        "matrix_count":count,"weight_bytes":weight_bytes,
                        "working_set_bytes":count*weight_bytes,"arena_stride_bytes":stride,
                        "arena_bytes":count*stride,"device_l2_bytes":l2,
                        "matrix_contents":"same deterministic bytes; distinct addresses in ring",
                        "scope":"controlled working set; not measured DRAM misses or model speed",
                        "projection_layout":"standalone Columns leaf; input stride K, output stride N; not FFN joined-bank stride",
                        "validated_outputs_per_matrix":M*n,"actual_pack_f64_points_per_matrix":9,
                        "checks":"all finite, cast bits, row/weight flags, canaries, identical-address-content output bits",
                        "plan":plan_report})
                );
                // A single complete projection per physical matrix per graph:
                // no repeated laps obscure counter launch selection.
                let graph = Graph::capture(&stream, || {
                    for (index, &address) in addresses.iter().enumerate() {
                        route.enqueue(&stream, &input, address, index);
                    }
                });
                let (start, end) = (Event::new(), Event::new());
                let mut samples = Vec::with_capacity(FORMAL);
                for round in 0..WARM + FORMAL {
                    let wall = Instant::now();
                    unsafe {
                        ok(cudaEventRecord(start.0, stream.0));
                    }
                    graph.launch(&stream);
                    unsafe {
                        ok(cudaEventRecord(end.0, stream.0));
                        ok(cudaEventSynchronize(end.0));
                    }
                    let wall_ns = wall.elapsed().as_nanos();
                    let mut ms = 0.0;
                    unsafe {
                        ok(cudaEventElapsedTime(&mut ms, start.0, end.0));
                    }
                    if round >= WARM {
                        samples.push((round - WARM, f64::from(ms) * 1_000_000.0, wall_ns));
                    }
                }
                stream.sync();
                let stable = stable.expect("nonempty physical weight working set");
                assert_eq!(route.validate(format, &host_weights, count), stable);
                for (index, &address) in addresses.iter().enumerate() {
                    route.enqueue(&stream, &input, address, index);
                    stream.sync();
                    assert_eq!(route.validate(format, &host_weights, count), stable);
                }
                input.guards();
                allocation.guards();
                for (replay, gpu_ns, wall_ns) in samples {
                    println!(
                        "{}",
                        serde_json::json!({"experiment":"marker_v2_m32_mmq",
                            "record":"formal_replay","format":format!("{format:?}"),
                            "algorithm":"Mmq","M":M,"K":k,"N":n,"weight_mode":mode,
                            "weight_bytes":weight_bytes,"matrix_count":count,
                            "working_set_bytes":count*weight_bytes,"device_l2_bytes":l2,
                            "laps":1,"projections":count,"warm_replays":WARM,
                            "formal_replays":FORMAL,"replay":replay,
                            "gpu_ns":gpu_ns,"wall_ns":wall_ns,
                            "gpu_ns_per_projection":gpu_ns/count as f64,
                            "wall_ns_per_projection":wall_ns as f64/count as f64,
                            "timed":"full F16 conversion, MarkerV2 pack, MMQ dot/fixup, MarkerV2 cast",
                            "projection_layout":"standalone Columns leaf; input stride K, output stride N",
                            "excluded":"cold plan/weight scan, allocations, input/weight upload, graph capture, validation and readbacks",
                            "wall_scope":"host event/graph enqueue through event completion, not pure CPU",
                            "correctness":"every physical matrix validated before and after; all output bits repeat"})
                    );
                }
            }
        }
    }
}
