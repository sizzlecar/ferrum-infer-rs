//! One test-only scheduling change; the production plan and arithmetic stay put.
use super::*;

unsafe extern "C" {
    fn ferrum_test_mmvq_m8_row1_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weights: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Clone, Copy, Debug)]
enum Dot {
    OriginalRow2,
    TestRow1,
}
impl Dot {
    unsafe fn launch(
        self,
        p: &UpstreamLinearPlanV1,
        w: *const c_void,
        q: *const c_void,
        o: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32 {
        unsafe {
            match self {
                Self::OriginalRow2 => ffi::ferrum_upstream_mmvq_dot_v1(p, w, q, o, fixup, stream),
                Self::TestRow1 => ferrum_test_mmvq_m8_row1_dot_v1(p, w, q, o, fixup, stream),
            }
        }
    }
}

fn route(p: UpstreamLinearPlanV1, output_stride: u32, weight_count: usize) -> Route {
    Route {
        algorithm: UpstreamLinearAlgorithm::Mmvq,
        converted: Buffer::new(p.converted_bytes as usize),
        packed: Buffer::new(p.packed_bytes as usize),
        output: Buffer::new(p.output_bytes as usize),
        fixup: Buffer::new(4),
        result: Buffer::new(p.request.rows as usize * output_stride as usize * 2),
        rows: Buffer::new(p.request.rows as usize * 4),
        flags: Buffer::new(weight_count * 4),
        plan: p,
    }
}

fn enqueue(
    r: &Route,
    dot: Dot,
    stream: &Stream,
    input: &Buffer,
    input_stride: u32,
    weight: *const c_void,
    weight_index: usize,
    output_stride: u32,
) {
    unsafe {
        ok(ffi::ferrum_upstream_mmvq_pack_v2(
            &r.plan,
            input.ptr,
            input_stride,
            r.converted.ptr,
            r.packed.ptr,
            r.rows.ptr,
            stream.0,
        ));
        ok(dot.launch(
            &r.plan,
            weight,
            r.packed.ptr,
            r.output.ptr,
            ptr::null_mut(),
            stream.0,
        ));
        ok(ffi::ferrum_upstream_mmvq_cast_v2(
            &r.plan,
            r.output.ptr,
            r.result.ptr,
            output_stride,
            r.rows.ptr,
            r.flags.ptr.add(weight_index * 4),
            stream.0,
        ));
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    converted_bits: Vec<u32>,
    packed: Vec<u8>,
    dot_bits: Vec<u32>,
    result_bits: Vec<u16>,
    rows: Vec<u32>,
    weights: Vec<u32>,
}
fn snapshot(r: &Route) -> Snapshot {
    for b in [
        &r.converted,
        &r.packed,
        &r.output,
        &r.fixup,
        &r.result,
        &r.rows,
        &r.flags,
    ] {
        b.guards();
    }
    Snapshot {
        converted_bits: r.converted.read(r.converted.bytes / 4),
        packed: r.packed.read(r.packed.bytes),
        dot_bits: r.output.read(r.output.bytes / 4),
        result_bits: r.result.read(r.result.bytes / 2),
        rows: r.rows.read(r.rows.bytes / 4),
        weights: r.flags.read(r.flags.bytes / 4),
    }
}

fn distinct_weights(format: UpstreamLinearFormat, k: usize, n: usize) -> Vec<u8> {
    let mut w = weights(format, k, n);
    let block_bytes = if format == UpstreamLinearFormat::Q4K {
        144
    } else {
        176
    };
    for (index, block) in w.chunks_exact_mut(block_bytes).enumerate() {
        block[..2].copy_from_slice(
            &f16::from_f32(0.015625 + (index % 7) as f32 / 1024.0)
                .to_bits()
                .to_le_bytes(),
        );
        block[2..4].copy_from_slice(
            &f16::from_f32(0.0078125 + (index % 3) as f32 / 2048.0)
                .to_bits()
                .to_le_bytes(),
        );
        for (offset, byte) in block[4..].iter_mut().enumerate() {
            *byte = (index.wrapping_mul(29) ^ offset.wrapping_mul(17) ^ (index / (k / 256))) as u8;
        }
    }
    w
}

fn check_case(format: UpstreamLinearFormat, k: u32, n: u32, kind: Input, bad_weight: bool) {
    let p = plan(
        UpstreamLinearAlgorithm::Mmvq,
        format,
        UpstreamLinearLayout::Columns,
        8,
        k,
        n,
    );
    let before = p;
    let stream = Stream::new();
    let (input_stride, output_stride) = (k + 3, n + 5);
    let mut x = vec![0x3555u16; 8 * input_stride as usize];
    for row in 0..8 {
        for col in 0..k as usize {
            x[row * input_stride as usize + col] = match kind {
                Input::Zero => 0x8000,
                Input::Subnormal => {
                    if col % 2 == 0 {
                        1
                    } else {
                        0x8001
                    }
                }
                Input::SumOverflow => 0x7bff,
                _ => f16::from_f32(((row * 17 + col * 7) % 101) as f32 / 512.0 - 0.09).to_bits(),
            };
        }
    }
    if matches!(kind, Input::Nonfinite) {
        x[0] = 0x7c00;
        x[input_stride as usize] = 0xfc00;
        x[7 * input_stride as usize] = 0x7e01;
    }
    let mut w = distinct_weights(format, k as usize, p.padded_outputs as usize);
    let row_bytes = w.len() / p.padded_outputs as usize;
    w[n as usize * row_bytes..].fill(0);
    if bad_weight {
        w[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    }
    let input = Buffer::new(x.len() * 2);
    input.write(&x);
    let weight = Buffer::new(w.len());
    weight.write(&w);
    let routes = [route(p, output_stride, 1), route(p, output_stride, 1)];
    for r in &routes {
        r.scan(&stream, weight.ptr, 0);
    }
    let mut previous = None;
    for repetition in 0..2 {
        for index in if repetition == 0 { [0, 1] } else { [1, 0] } {
            let r = &routes[index];
            r.result.write(&vec![0x3555u16; 8 * output_stride as usize]);
            unsafe {
                ok(cudaMemset(r.output.ptr, 0x35, r.output.bytes));
            }
            enqueue(
                r,
                [Dot::OriginalRow2, Dot::TestRow1][index],
                &stream,
                &input,
                input_stride,
                weight.ptr,
                0,
                output_stride,
            );
        }
        stream.sync();
        let a = snapshot(&routes[0]);
        assert_eq!(
            a,
            snapshot(&routes[1]),
            "full bits {format:?}/{k}/{n}/{kind:?}/bad={bad_weight}"
        );
        assert_eq!(a.weights, vec![u32::from(bad_weight)]);
        for row in 0..8 {
            let mut row_bad = false;
            for group in 0..p.padded_inputs as usize / 32 {
                let values = std::array::from_fn(|lane| {
                    let col = group * 32 + lane;
                    if col < k as usize {
                        x[row * input_stride as usize + col]
                    } else {
                        0
                    }
                });
                let (zero, bad) =
                    oracle::marker_classify_pack(UpstreamLinearAlgorithm::Mmvq, format, &values);
                row_bad |= bad;
                let offset = (row * (p.padded_inputs as usize / 32) + group) * 36;
                if zero {
                    assert_eq!(&a.packed[offset..offset + 36], &[0; 36]);
                }
                if bad {
                    assert_eq!(&a.packed[offset..offset + 2], &0x7e00u16.to_le_bytes());
                    assert!(a.packed[offset + 4..offset + 36].iter().all(|&b| b == 0));
                }
            }
            assert_eq!(a.rows[row], u32::from(row_bad));
            for col in 0..n as usize {
                assert_eq!(
                    a.result_bits[row * output_stride as usize + col],
                    oracle::final_cast_bits(
                        f32::from_bits(a.dot_bits[row * n as usize + col]),
                        row_bad,
                        bad_weight
                    )
                );
            }
            assert!(a.result_bits
                [row * output_stride as usize + n as usize..(row + 1) * output_stride as usize]
                .iter()
                .all(|&v| v == 0x3555));
        }
        if let Some(prev) = &previous {
            assert_eq!(&a, prev, "repeat full bits");
        }
        previous = Some(a);
    }
    assert_eq!(p, before);
    assert_eq!(input.read::<u16>(x.len()), x);
    assert_eq!(weight.read::<u8>(w.len()), w);
    input.guards();
    weight.guards();
}

fn check_rejections() {
    let p = plan(
        UpstreamLinearAlgorithm::Mmvq,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearLayout::Columns,
        8,
        768,
        17,
    );
    let stream = Stream::new();
    let b = Buffer::new(64);
    let unchanged = b.read::<u8>(64);
    let mut malformed = Vec::new();
    let mut q = p;
    q.output_bytes -= 4;
    malformed.push(q);
    let mut q = p;
    q.weight_bytes -= 1;
    malformed.push(q);
    let mut q = p;
    q.rows_per_block = 1;
    malformed.push(q);
    let mut q = p;
    q.nwarps = 4;
    malformed.push(q);
    let mut q = p;
    q.request.inputs = u32::MAX & !255;
    malformed.push(q);
    for q in malformed {
        for dot in [Dot::OriginalRow2, Dot::TestRow1] {
            unsafe {
                assert_eq!(
                    dot.launch(&q, b.ptr, b.ptr, b.ptr, ptr::null_mut(), stream.0),
                    -1
                );
            }
        }
    }
    for dot in [Dot::OriginalRow2, Dot::TestRow1] {
        unsafe {
            for (w, q, o) in [
                (ptr::null_mut(), b.ptr, b.ptr),
                (b.ptr, ptr::null_mut(), b.ptr),
                (b.ptr, b.ptr, ptr::null_mut()),
                (b.ptr.add(1), b.ptr, b.ptr),
                (b.ptr, b.ptr.add(1), b.ptr),
                (b.ptr, b.ptr, b.ptr.add(1)),
            ] {
                assert_eq!(dot.launch(&p, w, q, o, ptr::null_mut(), stream.0), -5);
            }
            assert_eq!(dot.launch(&p, b.ptr, b.ptr, b.ptr, b.ptr, stream.0), -1);
        }
    }
    for (format, layout, rows) in [
        (UpstreamLinearFormat::Q4K, UpstreamLinearLayout::Columns, 4),
        (UpstreamLinearFormat::Q5K, UpstreamLinearLayout::Channels, 8),
        (
            UpstreamLinearFormat::Iq4Xs,
            UpstreamLinearLayout::Columns,
            8,
        ),
    ] {
        let q = plan(UpstreamLinearAlgorithm::Mmvq, format, layout, rows, 768, 17);
        unsafe {
            assert_eq!(
                Dot::TestRow1.launch(&q, b.ptr, b.ptr, b.ptr, ptr::null_mut(), stream.0),
                -2
            );
        }
    }
    stream.sync();
    assert_eq!(b.read::<u8>(64), unchanged, "rejected calls do not launch");
    b.guards();
}

#[test]
#[ignore = "requires the experimental source-built rowtile native archive and an exclusive CUDA device"]
fn mmvq_m8_row1_preserves_bits_markers_tails_and_contracts() {
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        for kind in [
            Input::Finite,
            Input::Zero,
            Input::Nonfinite,
            Input::SumOverflow,
            Input::Subnormal,
        ] {
            check_case(format, 768, 17, kind, false);
        }
        check_case(format, 768, 17, Input::Finite, true);
        for (k, n) in [(256, 1), (5120, 1024), (5120, 10240), (6144, 5120)] {
            check_case(format, k, n, Input::Finite, false);
        }
    }
    check_rejections();
}

#[test]
#[ignore = "exclusive CUDA paired graph diagnostic; run rowtile correctness first"]
fn mmvq_m8_row1_paired_graph_working_sets() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const WARM: usize = 8;
    const PAIRS: usize = 16;
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        for (k, n) in [(5120, 1024), (5120, 10240), (6144, 5120)] {
            let host_weights = distinct_weights(format, k, n);
            let bytes = host_weights.len();
            for mode in ["resident", "arena_offset_ring"] {
                let count = if mode == "resident" {
                    1
                } else {
                    3 * l2 as usize / bytes + 1
                };
                let stride = bytes.div_ceil(256) * 256;
                let allocation = Buffer::new(count * stride);
                let addresses = (0..count)
                    .map(|i| unsafe { allocation.ptr.add(i * stride) })
                    .collect::<Vec<_>>();
                for &address in &addresses {
                    unsafe {
                        ok(cudaMemcpy(address, host_weights.as_ptr().cast(), bytes, 1));
                    }
                }
                let input = Buffer::new(8 * k * 2);
                input.write(
                    &(0..8 * k)
                        .map(|i| f16::from_f32(((i * 17) % 101) as f32 / 2048.0 - 0.024).to_bits())
                        .collect::<Vec<_>>(),
                );
                let stream = Stream::new();
                let routes = [0, 1].map(|_| {
                    Route::new(
                        UpstreamLinearAlgorithm::Mmvq,
                        format,
                        8,
                        k as u32,
                        n as u32,
                        count,
                    )
                });
                for (i, r) in routes.iter().enumerate() {
                    for (index, &address) in addresses.iter().enumerate() {
                        r.scan(&stream, address, index);
                        enqueue(
                            r,
                            [Dot::OriginalRow2, Dot::TestRow1][i],
                            &stream,
                            &input,
                            k as u32,
                            address,
                            index,
                            n as u32,
                        );
                    }
                }
                stream.sync();
                for r in &routes {
                    r.validate(format, &host_weights, count);
                }
                let stable = snapshot(&routes[0]);
                assert_eq!(
                    stable,
                    snapshot(&routes[1]),
                    "full baseline/candidate bits before timing"
                );
                let laps = 32_usize.div_ceil(count);
                let graphs = [0, 1].map(|i| {
                    Graph::capture(&stream, || {
                        for _ in 0..laps {
                            for (index, &address) in addresses.iter().enumerate() {
                                enqueue(
                                    &routes[i],
                                    [Dot::OriginalRow2, Dot::TestRow1][i],
                                    &stream,
                                    &input,
                                    k as u32,
                                    address,
                                    index,
                                    n as u32,
                                );
                            }
                        }
                    })
                });
                let (start, end) = (Event::new(), Event::new());
                let mut samples = Vec::with_capacity(PAIRS * 2);
                for round in 0..WARM + PAIRS {
                    for order in 0..2 {
                        let i = (round + order) % 2;
                        let wall = Instant::now();
                        unsafe {
                            ok(cudaEventRecord(start.0, stream.0));
                        }
                        graphs[i].launch(&stream);
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
                                i,
                                f64::from(ms) * 1_000_000.0,
                                wall_ns,
                            ));
                        }
                    }
                }
                stream.sync();
                for r in &routes {
                    r.validate(format, &host_weights, count);
                    assert_eq!(snapshot(r), stable, "full bits after measured graphs");
                }
                input.guards();
                allocation.guards();
                for (pair, order, i, gpu_ns, wall_ns) in samples {
                    println!(
                        "{}",
                        serde_json::json!({
                            "experiment":"mmvq_m8_output_row_tile", "format":format!("{format:?}"), "M":8,"K":k,"N":n,
                            "variant":format!("{:?}",[Dot::OriginalRow2,Dot::TestRow1][i]),"output_rows_per_cta":if i==0 {2}else{1},
                            "warps_per_cta":2,"original_writer_lane_preserved":true,"baseline_plan_unchanged":true,
                            "weight_mode":mode,"synthetic_weights":true,"weight_bytes":bytes,"matrix_count":count,
                            "working_set_bytes":count*bytes,"device_l2_bytes":l2,"same_physical_weights":true,
                            "laps":laps,"projections":laps*count,"warm_rounds":WARM,"pairs":PAIRS,"pair":pair,"order":order,
                            "gpu_ns":gpu_ns,"wall_ns":wall_ns,
                            "timed":"F16 conversion, unchanged MarkerV2 pack, selected dot, unchanged MarkerV2 cast",
                            "excluded":"cold plan/weight scan, allocation, copies, validation, formatting",
                            "correctness":"complete F32/F16/pack/flags equality before and after; canaries; actual-pack F64 spot oracle",
                            "scope":"primitive diagnostic, not whole-model gain or measured DRAM cache residency"
                        })
                    );
                }
            }
        }
    }
}
