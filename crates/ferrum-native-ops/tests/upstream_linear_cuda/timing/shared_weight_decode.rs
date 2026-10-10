//! Test-only Q4K Columns M8 dot alternative. Both arms retain the same
//! MarkerV2 pack, static weight check, cast and physical working set.
use super::*;

unsafe extern "C" {
    fn ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

const DOTS: [MmvqDot; 2] = [
    ffi::ferrum_upstream_mmvq_dot_v1,
    ferrum_upstream_mmvq_q4_shared_weight_decode_dot_v1,
];
const LABELS: [&str; 2] = ["baseline", "shared_weight_decode"];
const FORMAT: UpstreamLinearFormat = UpstreamLinearFormat::Q4K;
const ALGORITHM: UpstreamLinearAlgorithm = UpstreamLinearAlgorithm::Mmvq;

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    converted: Vec<u32>,
    packed: Vec<u8>,
    output: Vec<u32>,
    result: Vec<u16>,
    rows: Vec<u32>,
    weight_flag: Vec<u32>,
}

/// Nonzero aligned view offsets and odd element strides expose accidental
/// use of allocation bases, logical N in place of stride, or padded N writes.
struct Case {
    plan: UpstreamLinearPlanV1,
    input_stride: usize,
    output_stride: usize,
    host_input: Vec<u16>,
    host_weights: Vec<u8>,
    input: Buffer,
    weights: Buffer,
    converted: Buffer,
    packed: Buffer,
    output: Buffer,
    result: Buffer,
    rows: Buffer,
    weight_flag: Buffer,
}

impl Case {
    fn new(k: usize, n: usize, input_kind: Input, bad_weight: bool) -> Self {
        let p = plan(
            ALGORITHM,
            FORMAT,
            UpstreamLinearLayout::Columns,
            8,
            k as u32,
            n as u32,
        );
        let input_stride = k + 3;
        let output_stride = n + 5;
        let mut x = vec![0x3555; 8 * input_stride];
        for row in 0..8 {
            for col in 0..k {
                x[row * input_stride + col] = match input_kind {
                    Input::Zero => 0x8000,
                    Input::Subnormal => {
                        if col % 2 == 0 {
                            1
                        } else {
                            0x8001
                        }
                    }
                    Input::SumOverflow => 0x7bff,
                    _ => f16::from_f32(((col + row * 7) % 19) as f32 / 16.0 - 0.5625).to_bits(),
                };
            }
        }
        if matches!(input_kind, Input::Nonfinite) {
            x[0] = 0x7c00;
            x[input_stride] = 0xfc00;
            x[7 * input_stride] = 0x7e00;
        }
        let mut w = weights(FORMAT, k, p.padded_outputs as usize);
        let row_bytes = k / 256 * 144;
        for (column, row) in w[..n * row_bytes].chunks_exact_mut(row_bytes).enumerate() {
            for (block_index, block) in row.chunks_exact_mut(144).enumerate() {
                block[..2].copy_from_slice(
                    &f16::from_f32(0.0625 + (column % 5) as f32 / 64.0)
                        .to_bits()
                        .to_le_bytes(),
                );
                for (i, code) in block[16..].iter_mut().enumerate() {
                    *code = ((column * 13 + block_index * 7 + i * 17) % 256) as u8;
                }
            }
        }
        w[n * row_bytes..].fill(0);
        if bad_weight {
            w[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        }
        let input = Buffer::new(x.len() * 2 + 2);
        let weights = Buffer::new(w.len() + 4);
        unsafe {
            ok(cudaMemcpy(
                input.ptr.add(2),
                x.as_ptr().cast(),
                x.len() * 2,
                1,
            ));
            ok(cudaMemcpy(
                weights.ptr.add(4),
                w.as_ptr().cast(),
                w.len(),
                1,
            ));
        }
        Self {
            plan: p,
            input_stride,
            output_stride,
            host_input: x,
            host_weights: w,
            input,
            weights,
            converted: Buffer::new(p.converted_bytes as usize),
            packed: Buffer::new(p.packed_bytes as usize),
            output: Buffer::new(p.output_bytes as usize + 4),
            result: Buffer::new(8 * output_stride * 2 + 2),
            rows: Buffer::new(8 * 4),
            weight_flag: Buffer::new(4),
        }
    }

    fn run(&self, stream: &Stream, dot: MmvqDot) -> Snapshot {
        let p = &self.plan;
        unsafe {
            ok(cudaMemset(self.result.ptr, 0x35, self.result.bytes));
            ok(cudaMemset(self.packed.ptr, 0x35, self.packed.bytes));
            ok(ffi::ferrum_upstream_mmvq_check_weights_v2(
                p,
                self.weights.ptr.add(4),
                self.weight_flag.ptr,
                stream.0,
            ));
            ok(ffi::ferrum_upstream_mmvq_pack_v2(
                p,
                self.input.ptr.add(2),
                self.input_stride as u32,
                self.converted.ptr,
                self.packed.ptr,
                self.rows.ptr,
                stream.0,
            ));
            ok(dot(
                p,
                self.weights.ptr.add(4),
                self.packed.ptr,
                self.output.ptr.add(4),
                ptr::null_mut(),
                stream.0,
            ));
            ok(ffi::ferrum_upstream_mmvq_cast_v2(
                p,
                self.output.ptr.add(4),
                self.result.ptr.add(2),
                self.output_stride as u32,
                self.rows.ptr,
                self.weight_flag.ptr,
                stream.0,
            ));
        }
        stream.sync();
        let output = self.output.read::<u32>(self.output.bytes / 4);
        let result = self.result.read::<u16>(self.result.bytes / 2);
        assert_eq!(output[0], 0x35353535);
        assert_eq!(result[0], 0x3535);
        let input = self.input.read::<u16>(self.input.bytes / 2);
        assert_eq!(input[0], 0x3535);
        assert_eq!(&input[1..], self.host_input);
        let weights = self.weights.read::<u8>(self.weights.bytes);
        assert_eq!(&weights[..4], &[0x35; 4]);
        assert_eq!(&weights[4..], self.host_weights);
        for b in [
            &self.input,
            &self.weights,
            &self.converted,
            &self.packed,
            &self.output,
            &self.result,
            &self.rows,
            &self.weight_flag,
        ] {
            b.guards();
        }
        Snapshot {
            converted: self.converted.read::<u32>(self.converted.bytes / 4),
            packed: self.packed.read::<u8>(self.packed.bytes),
            output: output[1..].to_vec(),
            result: result[1..].to_vec(),
            rows: self.rows.read::<u32>(8),
            weight_flag: self.weight_flag.read::<u32>(1),
        }
    }

    fn validate(&self, snapshot: &Snapshot) {
        let p = &self.plan;
        let (k, n) = (p.request.inputs as usize, p.request.outputs as usize);
        let bad_weight = self
            .host_weights
            .chunks_exact(144)
            .any(|b| oracle::classify_weight(ALGORITHM, FORMAT, b));
        assert_eq!(snapshot.weight_flag[0] != 0, bad_weight);
        for row in 0..8 {
            let mut row_bad = false;
            for group in 0..p.padded_inputs as usize / 32 {
                let values = std::array::from_fn(|i| {
                    if group * 32 < k {
                        self.host_input[row * self.input_stride + group * 32 + i]
                    } else {
                        0
                    }
                });
                let (zero, bad) = oracle::marker_classify_pack(ALGORITHM, FORMAT, &values);
                row_bad |= bad;
                let b = row * (p.padded_inputs as usize / 32) + group;
                let metadata = &snapshot.packed[b * 36..][..4];
                let codes = &snapshot.packed[b * 36 + 4..][..32];
                if zero {
                    assert_eq!(metadata, &[0; 4]);
                }
                if zero || bad {
                    assert!(codes.iter().all(|&v| v == 0));
                }
                if bad {
                    assert_eq!(
                        u16::from_le_bytes(metadata[..2].try_into().unwrap()),
                        0x7e00
                    );
                }
            }
            assert_eq!(snapshot.rows[row] != 0, row_bad);
            for col in 0..n {
                let value = f32::from_bits(snapshot.output[row * n + col]);
                assert_eq!(
                    snapshot.result[row * self.output_stride + col],
                    oracle::final_cast_bits(value, row_bad, bad_weight)
                );
                if !row_bad && !bad_weight {
                    assert!(value.is_finite());
                    let (mut target, mut magnitude) = (0.0, 0.0);
                    for group in 0..k / 32 {
                        let b = row * (p.padded_inputs as usize / 32) + group;
                        let meta = &snapshot.packed[b * 36..][..4];
                        let codes = &snapshot.packed[b * 36 + 4..][..32];
                        let scale =
                            f16::from_bits(u16::from_le_bytes(meta[..2].try_into().unwrap()))
                                .to_f32();
                        let sum = f16::from_bits(u16::from_le_bytes(meta[2..].try_into().unwrap()))
                            .to_f32();
                        let offset = (col * (k / 256) + group / 8) * 144;
                        let (v, mag) = oracle::declared_group(
                            ALGORITHM,
                            FORMAT,
                            &self.host_weights[offset..offset + 144],
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
                        (f64::from(value) - target).abs() <= bound,
                        "actual-pack F64 row{row} col{col}: {value} vs {target}, bound{bound}"
                    );
                }
            }
            assert!(
                snapshot.result[row * self.output_stride + n..(row + 1) * self.output_stride]
                    .iter()
                    .all(|&v| v == 0x3535)
            );
        }
    }
}

#[test]
#[ignore = "requires the source-built shared-weight-decode CUDA experiment"]
fn q4_shared_weight_decode_preserves_marker_bits_and_boundaries() {
    let _ = caps();
    let stream = Stream::new();
    for (k, n) in [(256, 17), (512, 33), (768, 17), (5120, 17)] {
        for input in [
            Input::Finite,
            Input::Zero,
            Input::Subnormal,
            Input::Nonfinite,
            Input::SumOverflow,
        ] {
            for bad_weight in [false, true] {
                let case = Case::new(k, n, input, bad_weight);
                let reference = case.run(&stream, DOTS[0]);
                case.validate(&reference);
                for dot in [DOTS[1], DOTS[1], DOTS[0]] {
                    let actual = case.run(&stream, dot);
                    case.validate(&actual);
                    assert_eq!(
                        actual, reference,
                        "full buffers/repeat {k}x{n} {input:?} bad_weight={bad_weight}"
                    );
                }
            }
        }
    }
    // Valid-but-ineligible plans and malformed inputs fail before a launch.
    let case = Case::new(512, 17, Input::Finite, false);
    let invoke = |p: &UpstreamLinearPlanV1, weights, packed, output, fixup| unsafe {
        DOTS[1](p, weights, packed, output, fixup, stream.0)
    };
    let w = unsafe { case.weights.ptr.add(4) }.cast_const();
    let q = case.packed.ptr.cast_const();
    let o = unsafe { case.output.ptr.add(4) };
    let mut malformed = case.plan;
    malformed.nwarps += 1;
    assert_eq!(invoke(&malformed, w, q, o, ptr::null_mut()), -1);
    assert_eq!(invoke(&case.plan, w, q, o, o), -1);
    for p in [
        plan(ALGORITHM, FORMAT, UpstreamLinearLayout::Columns, 4, 512, 17),
        plan(
            ALGORITHM,
            FORMAT,
            UpstreamLinearLayout::Channels,
            8,
            512,
            17,
        ),
        plan(
            ALGORITHM,
            UpstreamLinearFormat::Q5K,
            UpstreamLinearLayout::Columns,
            8,
            512,
            17,
        ),
    ] {
        assert_eq!(invoke(&p, w, q, o, ptr::null_mut()), -2);
    }
    assert_eq!(invoke(&case.plan, ptr::null(), q, o, ptr::null_mut()), -5);
    assert_eq!(invoke(&case.plan, w, ptr::null(), o, ptr::null_mut()), -5);
    assert_eq!(
        invoke(&case.plan, w, q, ptr::null_mut(), ptr::null_mut()),
        -5
    );
    unsafe {
        assert_eq!(DOTS[1](ptr::null(), w, q, o, ptr::null_mut(), stream.0), -1);
        assert_eq!(invoke(&case.plan, w.add(1), q, o, ptr::null_mut()), -5);
        assert_eq!(invoke(&case.plan, w, q.add(1), o, ptr::null_mut()), -5);
        assert_eq!(invoke(&case.plan, w, q, o.add(1), ptr::null_mut()), -5);
    }
    stream.sync();
    assert!(case
        .output
        .read::<u8>(case.output.bytes)
        .iter()
        .all(|&byte| byte == 0x35));
    for buffer in [
        &case.input,
        &case.weights,
        &case.converted,
        &case.packed,
        &case.output,
        &case.result,
        &case.rows,
        &case.weight_flag,
    ] {
        buffer.guards();
    }
}

#[test]
#[ignore = "exclusive CUDA full-projection paired diagnostic; run correctness first"]
fn q4_shared_weight_decode_paired_working_sets() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const WARM: usize = 8;
    const PAIRS: usize = 16;
    let (m, k, n) = (8, 5120, 10240);
    let host_weights = weights(FORMAT, k, n);
    let weight_bytes = host_weights.len();
    println!(
        "{}",
        serde_json::json!({"experiment":"q4_shared_weight_decode", "kind":"plan",
        "M":m,"K":k,"N":n,"format":"Q4K","warm_rounds":WARM,"pairs":PAIRS,
        "order":"alternating AB/BA pairs","modes":["resident","arena_offset_ring"],
        "timed":"same MarkerV2 F16 convert/row checks/pack + selected dot + MarkerV2 cast",
        "excluded":"allocation, input upload, static plan/weight check, graph capture, oracle/readback",
        "scope":"synthetic same physical weights; fixed input; no provider/driver conditional checks or model throughput claim"})
    );
    for mode in ["resident", "arena_offset_ring"] {
        let count = if mode == "resident" {
            1
        } else {
            3 * l2 as usize / weight_bytes + 1
        };
        let stride = weight_bytes.div_ceil(256) * 256;
        let allocation = Buffer::new(count * stride);
        let addresses = (0..count)
            .map(|i| unsafe { allocation.ptr.add(i * stride) })
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
        let routes =
            DOTS.map(|_| Route::new(ALGORITHM, FORMAT, m as u32, k as u32, n as u32, count));
        let mut stable = Vec::new();
        let mut raw = Vec::new();
        for (arm, route) in routes.iter().enumerate() {
            for (index, &address) in addresses.iter().enumerate() {
                route.scan(&stream, address, index);
                route.enqueue_with_mmvq_dot(&stream, &input, address, index, DOTS[arm]);
            }
            stream.sync();
            stable.push(route.validate(FORMAT, &host_weights, count));
            raw.push(route.output.read::<u32>(m * n));
        }
        assert_eq!(stable[0], stable[1]);
        assert_eq!(raw[0], raw[1]);
        assert_eq!(
            routes[0].packed.read::<u8>(routes[0].packed.bytes),
            routes[1].packed.read::<u8>(routes[1].packed.bytes)
        );
        let laps = 32_usize.div_ceil(count);
        let graphs = [0, 1].map(|arm| {
            Graph::capture(&stream, || {
                for _ in 0..laps {
                    for (index, &address) in addresses.iter().enumerate() {
                        routes[arm]
                            .enqueue_with_mmvq_dot(&stream, &input, address, index, DOTS[arm]);
                    }
                }
            })
        });
        let (start, end) = (Event::new(), Event::new());
        let mut samples = Vec::with_capacity(PAIRS * 2);
        for round in 0..WARM + PAIRS {
            for order in 0..2 {
                let arm = (round + order) % 2;
                let wall = Instant::now();
                unsafe {
                    ok(cudaEventRecord(start.0, stream.0));
                }
                graphs[arm].launch(&stream);
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
                        arm,
                        f64::from(ms) * 1_000_000.0,
                        wall_ns,
                    ));
                }
            }
        }
        stream.sync();
        for (arm, route) in routes.iter().enumerate() {
            assert_eq!(route.validate(FORMAT, &host_weights, count), stable[arm]);
            assert_eq!(route.output.read::<u32>(m * n), raw[arm]);
        }
        input.guards();
        allocation.guards();
        for (pair, order, arm, gpu_ns, wall_ns) in samples {
            println!(
                "{}",
                serde_json::json!({"experiment":"q4_shared_weight_decode", "kind":"sample",
                "M":m,"K":k,"N":n,"format":"Q4K","weight_mode":mode,
                "weight_bytes":weight_bytes,"matrix_count":count,"working_set_bytes":count*weight_bytes,
                "device_l2_bytes":l2,"matrix_contents":"same deterministic bytes; different ring addresses",
                "laps":laps,"projections":laps*count,"warm_rounds":WARM,"pairs":PAIRS,
                "same_physical_weights":true,"between_timed_routes":"events/graph submission only; no reset/readback",
                "pair":pair,"order":order,"arm":LABELS[arm],"gpu_ns":gpu_ns,"wall_ns":wall_ns})
            );
        }
        // Captured graphs relinquish device addresses before their allocations.
        drop(graphs);
    }
}
