//! Opt-in next-K prefetch experiment. Both routes retain the original MMQ
//! MarkerV2 pack, main/fixup and cast; no product route or precision is changed.
use super::*;

unsafe extern "C" {
    fn ferrum_upstream_mmq_next_k_prefetch_plan_v1(
        request: *const UpstreamLinearRequestV1,
        plan: *mut UpstreamLinearPlanV1,
    ) -> i32;
    fn ferrum_upstream_mmq_next_k_prefetch_dot_v1(
        plan: *const UpstreamLinearPlanV1,
        weight: *const c_void,
        packed: *const c_void,
        output: *mut c_void,
        fixup: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

const FORMATS: [UpstreamLinearFormat; 3] = [
    UpstreamLinearFormat::Iq4Xs,
    UpstreamLinearFormat::Q4K,
    UpstreamLinearFormat::Q5K,
];

fn routes(format: UpstreamLinearFormat, m: u32, k: u32, n: u32, count: usize) -> [Route; 2] {
    let base = Route::new(UpstreamLinearAlgorithm::Mmq, format, m, k, n, count);
    let mut candidate = Route::new(UpstreamLinearAlgorithm::Mmq, format, m, k, n, count);
    let mut actual = UpstreamLinearPlanV1::default();
    // New-kernel shared-memory setup belongs to cold preparation, outside every
    // captured/timed projection. All wire geometry must remain the old plan.
    unsafe {
        ok(ferrum_upstream_mmq_next_k_prefetch_plan_v1(
            &base.plan.request,
            &mut actual,
        ));
    }
    actual
        .validate_identity(&base.plan.request, UpstreamLinearAlgorithm::Mmq)
        .unwrap();
    assert_eq!(
        actual, base.plan,
        "prefetch changed the complete native plan"
    );
    candidate.plan = actual;
    candidate.mmq_dot = Some(ferrum_upstream_mmq_next_k_prefetch_dot_v1);
    [base, candidate]
}

fn varied_weights(format: UpstreamLinearFormat, k: usize, n: usize) -> Vec<u8> {
    let mut bytes = weights(format, k, n);
    let block_bytes = bytes.len() / ((k / 256) * n);
    for (index, block) in bytes.chunks_exact_mut(block_bytes).enumerate() {
        let scale = f16::from_f32((1 + index % 7) as f32 / 512.0);
        block[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
        let last = block.len() - 1;
        block[last] ^= (index * 13) as u8;
    }
    bytes
}

fn finite_input(m: usize, k: usize, stride: usize, salt: usize) -> Vec<u16> {
    let mut values = vec![0x3555; m * stride];
    for row in 0..m {
        for col in 0..k {
            values[row * stride + col] =
                f16::from_f32(((row * 11 + col * 17 + salt) % 101) as f32 / 2048.0 - 0.024)
                    .to_bits();
        }
    }
    values
}

fn enqueue_strided(
    route: &Route,
    stream: &Stream,
    input: &Buffer,
    weight: &Buffer,
    input_stride: u32,
    output_stride: u32,
) {
    let p = &route.plan;
    unsafe {
        ok(ffi::ferrum_upstream_mmq_pack_v2(
            p,
            input.ptr,
            input_stride,
            route.converted.ptr,
            route.packed.ptr,
            route.rows.ptr,
            stream.0,
        ));
        ok(route.mmq_dot.unwrap_or(ffi::ferrum_upstream_mmq_dot_v1)(
            p,
            weight.ptr,
            route.packed.ptr,
            route.output.ptr,
            if p.fixup_bytes == 0 {
                ptr::null_mut()
            } else {
                route.fixup.ptr
            },
            stream.0,
        ));
        ok(ffi::ferrum_upstream_mmq_cast_v2(
            p,
            route.output.ptr,
            route.result.ptr,
            output_stride,
            route.rows.ptr,
            route.flags.ptr,
            stream.0,
        ));
    }
}

fn assert_buffers_equal(routes: &[Route; 2]) {
    for (name, left, right) in [
        ("converted", &routes[0].converted, &routes[1].converted),
        ("packed", &routes[0].packed, &routes[1].packed),
        ("raw output", &routes[0].output, &routes[1].output),
        ("fixup", &routes[0].fixup, &routes[1].fixup),
        ("cast output", &routes[0].result, &routes[1].result),
        ("row flags", &routes[0].rows, &routes[1].rows),
        ("weight flags", &routes[0].flags, &routes[1].flags),
    ] {
        assert_eq!(left.bytes, right.bytes);
        assert_eq!(
            left.read::<u8>(left.bytes),
            right.read::<u8>(right.bytes),
            "complete {name} bits"
        );
        left.guards();
        right.guards();
    }
}

fn full_oracle(
    route: &Route,
    format: UpstreamLinearFormat,
    input: &[u16],
    input_stride: usize,
    weight: &[u8],
    output_stride: usize,
) -> usize {
    let p = &route.plan;
    let (m, k, n) = (
        p.request.rows as usize,
        p.request.inputs as usize,
        p.request.outputs as usize,
    );
    let packed = route.packed.read::<u8>(p.packed_bytes as usize);
    let raw = route.output.read::<f32>(m * n);
    let cast = route.result.read::<u16>(m * output_stride);
    let flags = route.rows.read::<u32>(m);
    let block_bytes = weight.len() / (n * (k / 256));
    let bad_weight = weight
        .chunks_exact(block_bytes)
        .any(|block| oracle::classify_weight(UpstreamLinearAlgorithm::Mmq, format, block));
    assert_eq!(route.flags.read::<u32>(1)[0] != 0, bad_weight);
    let guard_start = m * p.padded_inputs as usize * 9 / 8;
    assert!(packed[guard_start..].iter().all(|&byte| byte == 0));
    let mut checked = 0;
    for row in 0..m {
        let bad_row = (0..p.padded_inputs as usize / 32).any(|group| {
            let values = std::array::from_fn(|i| {
                if group * 32 + i < k {
                    input[row * input_stride + group * 32 + i]
                } else {
                    0
                }
            });
            oracle::marker_classify_pack(UpstreamLinearAlgorithm::Mmq, format, &values).1
        });
        assert_eq!(flags[row] != 0, bad_row);
        for col in 0..n {
            assert_eq!(
                cast[row * output_stride + col],
                oracle::final_cast_bits(raw[row * n + col], bad_row, bad_weight)
            );
            if !bad_row && !bad_weight {
                let (mut target, mut magnitude) = (0.0, 0.0);
                for group in 0..k / 32 {
                    let b = (group / 4) * m + row;
                    let meta = &packed[b * 144 + (group % 4) * 4..][..4];
                    let codes = &packed[b * 144 + 16 + (group % 4) * 32..][..32];
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
                    let offset = (col * (k / 256) + group / 8) * block_bytes;
                    let (v, mag) = oracle::declared_group(
                        UpstreamLinearAlgorithm::Mmq,
                        format,
                        &weight[offset..offset + block_bytes],
                        group % 8,
                        &std::array::from_fn(|i| codes[i] as i8),
                        scale,
                        sum,
                    );
                    target += v;
                    magnitude += mag;
                }
                let nu = (k as f64 / 32.0 + 32.0) * f64::from(f32::EPSILON);
                let bound = nu / (1.0 - nu) * magnitude + 32.0 * f64::from(f32::MIN_POSITIVE);
                assert!(
                    raw[row * n + col].is_finite()
                        && (f64::from(raw[row * n + col]) - target).abs() <= bound,
                    "full actual-pack F64 {format:?} [{row},{col}]"
                );
                checked += 1;
            }
        }
        assert!(cast[row * output_stride + n..(row + 1) * output_stride]
            .iter()
            .all(|&value| value == 0x3555));
    }
    checked
}

fn correctness_case(
    format: UpstreamLinearFormat,
    m: u32,
    k: u32,
    n: u32,
    fresh_capture: bool,
) -> UpstreamLinearPlanV1 {
    let mut routes = routes(format, m, k, n, 1);
    let (input_stride, output_stride) = (k as usize + 3, n as usize + 5);
    for route in &mut routes {
        route.result = Buffer::new(m as usize * output_stride * 2);
    }
    let ordinary = finite_input(m as usize, k as usize, input_stride, 0);
    let good = varied_weights(format, k as usize, n as usize);
    let input = Buffer::new(ordinary.len() * 2);
    let weight = Buffer::new(good.len());
    let stream = Stream::new();
    input.write(&ordinary);
    weight.write(&good);
    for route in &routes {
        route.scan(&stream, weight.ptr, 0);
    }
    stream.sync();
    let graphs = routes.each_ref().map(|route| {
        Graph::capture(&stream, || {
            enqueue_strided(
                route,
                &stream,
                &input,
                &weight,
                input_stride as u32,
                output_stride as u32,
            )
        })
    });
    let mut checked = 0;
    for pattern in 0..if fresh_capture { 8 } else { 1 } {
        let mut x = ordinary.clone();
        let mut w = good.clone();
        match pattern {
            1 => x = finite_input(m as usize, k as usize, input_stride, 23),
            2 => {
                for row in 0..m as usize {
                    x[row * input_stride..row * input_stride + k as usize].fill(0x8000);
                }
            }
            3 => {
                x[0] = 0x7e00;
                x[input_stride] = 0x7c00;
                x[(m as usize - 1) * input_stride] = 0xfc00;
            }
            4 => {
                for row in 0..m as usize {
                    x[row * input_stride..row * input_stride + k as usize].fill(0x7bff);
                }
            }
            5 => {
                for row in 0..m as usize {
                    for col in 0..k as usize {
                        x[row * input_stride + col] = if col % 2 == 0 { 1 } else { 0x8001 };
                    }
                }
            }
            6 => w[..2].copy_from_slice(&0x7c00_u16.to_le_bytes()),
            _ => {}
        }
        input.write(&x);
        weight.write(&w);
        for route in &routes {
            route.scan(&stream, weight.ptr, 0);
            route
                .result
                .write(&vec![0x3555_u16; m as usize * output_stride]);
            unsafe {
                ok(cudaMemset(route.packed.ptr, 0x35, route.packed.bytes));
                ok(cudaMemset(route.fixup.ptr, 0x35, route.fixup.bytes));
            }
            enqueue_strided(
                route,
                &stream,
                &input,
                &weight,
                input_stride as u32,
                output_stride as u32,
            );
        }
        stream.sync();
        assert_buffers_equal(&routes);
        checked += full_oracle(&routes[0], format, &x, input_stride, &w, output_stride);
        let direct = routes[0].result.read::<u8>(routes[0].result.bytes);
        let direct_raw = routes[0].output.read::<u8>(routes[0].output.bytes);
        // Poison after the direct oracle so an omitted/stale graph cannot pass
        // by retaining results just computed outside capture.
        for route in &routes {
            route
                .result
                .write(&vec![0x3555_u16; m as usize * output_stride]);
            for buffer in [
                &route.converted,
                &route.packed,
                &route.output,
                &route.fixup,
                &route.rows,
            ] {
                unsafe {
                    ok(cudaMemset(buffer.ptr, 0x35, buffer.bytes));
                }
            }
        }
        for graph in &graphs {
            graph.launch(&stream);
        }
        stream.sync();
        assert_buffers_equal(&routes);
        assert_eq!(
            routes[1].result.read::<u8>(routes[1].result.bytes),
            direct,
            "fresh captured projection bits"
        );
        assert_eq!(
            routes[1].output.read::<u8>(routes[1].output.bytes),
            direct_raw,
            "fresh captured raw main/fixup bits"
        );
        input.guards();
        weight.guards();
        assert_eq!(input.read::<u16>(x.len()), x);
        assert_eq!(weight.read::<u8>(w.len()), w);
    }
    println!(
        "{}",
        serde_json::json!({"kind":"mmq_next_k_prefetch_correctness","format":format!("{format:?}"),
        "m":m,"k":k,"n":n,"j":routes[0].plan.j,"i":routes[0].plan.i,"blocks":routes[0].plan.blocks,"fixup":routes[0].plan.fixup,
        "complete_plan_equal":true,"all_raw_and_cast_bits_equal":true,"all_scratch_flags_guards_equal":true,
        "fresh_captured_inputs_and_flags":fresh_capture,"full_f64_outputs_checked":checked,"arithmetic":"original MMQ MarkerV2"})
    );
    let p = routes[0].plan;
    drop(graphs);
    stream.sync();
    p
}

#[test]
#[ignore = "next-K prefetch complete projection oracle requires the opt-in native archive and actual CUDA"]
fn mmq_next_k_prefetch_preserves_complete_projection_bits_flags_and_fresh_capture_on_cuda() {
    let device = caps();
    for format in FORMATS {
        for m in [8, 32] {
            let tail = correctness_case(format, m, 768, 17, true);
            let crossing = correctness_case(format, m, 768, tail.i + 1, false);
            let no_fixup_n = (1..=device.multiprocessors)
                .map(|tiles| tiles.checked_mul(tail.i).unwrap())
                .find(|&n| {
                    plan(
                        UpstreamLinearAlgorithm::Mmq,
                        format,
                        UpstreamLinearLayout::Columns,
                        m,
                        256,
                        n,
                    )
                    .fixup
                        == 0
                })
                .expect("an integral SM wave admits original no-fixup MMQ");
            let no_fixup = correctness_case(format, m, 256, no_fixup_n, false);
            assert_eq!(no_fixup.fixup, 0);
            assert!(
                device.multiprocessors == 1 || tail.fixup != 0 || crossing.fixup != 0,
                "actual stream-K partial reduction must execute on a multi-SM device"
            );
        }
    }
}

#[test]
#[ignore = "complete MMQ next-K prefetch paired experiment requires exclusive CUDA; run correctness first"]
fn mmq_next_k_prefetch_m8_m32_inclusive_paired_working_sets_on_cuda() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const LIMIT: usize = 1 << 30;
    const WARM: usize = 8;
    const PAIRS: usize = 8;
    for format in FORMATS {
        for m in [8, 32] {
            for (k, n) in [(5120, 17408), (17408, 5120)] {
                let host_weights = varied_weights(format, k, n);
                let weight_bytes = host_weights.len();
                let p = plan(
                    UpstreamLinearAlgorithm::Mmq,
                    format,
                    UpstreamLinearLayout::Columns,
                    m,
                    k as u32,
                    n as u32,
                );
                let (result_stride, result_offset) =
                    if k == 5120 { (34816, 17408) } else { (n, 0) };
                let fixed = 2
                    * (p.converted_bytes
                        + p.packed_bytes
                        + p.output_bytes
                        + p.fixup_bytes
                        + u64::from(m) * result_stride as u64 * 2
                        + u64::from(m) * 4
                        + 8 * 64) as usize
                    + m as usize * k * 2
                    + 64;
                for mode in ["resident", "distinct_allocation_ring"] {
                    let desired = if mode == "resident" {
                        1
                    } else {
                        (3 * l2 as usize).div_ceil(weight_bytes) + 1
                    };
                    let count = desired
                        .min((LIMIT - fixed) / (weight_bytes + 64 + 8))
                        .max(1);
                    let allocations = (0..count)
                        .map(|_| {
                            let b = Buffer::new(weight_bytes);
                            b.write(&host_weights);
                            b
                        })
                        .collect::<Vec<_>>();
                    let input_values = finite_input(m as usize, k, k, 0);
                    let input = Buffer::new(input_values.len() * 2);
                    input.write(&input_values);
                    let mut routes = routes(format, m, k as u32, n as u32, count);
                    for route in &mut routes {
                        route.result_stride = result_stride as u32;
                        route.result_offset = result_offset as u32;
                        route.result = Buffer::new(m as usize * result_stride * 2);
                    }
                    let stream = Stream::new();
                    for route in &routes {
                        for (index, weight) in allocations.iter().enumerate() {
                            route.scan(&stream, weight.ptr, index);
                            route.enqueue(&stream, &input, weight.ptr, index);
                        }
                    }
                    stream.sync();
                    assert_buffers_equal(&routes);
                    let expected = routes
                        .each_ref()
                        .map(|route| route.validate(format, &host_weights, count));
                    assert_eq!(expected[0], expected[1]);
                    let laps = 32_usize.div_ceil(count);
                    let graphs = routes.each_ref().map(|route| {
                        Graph::capture(&stream, || {
                            for _ in 0..laps {
                                for (index, weight) in allocations.iter().enumerate() {
                                    route.enqueue(&stream, &input, weight.ptr, index);
                                }
                            }
                        })
                    });
                    let (start, end) = (Event::new(), Event::new());
                    let mut samples = Vec::with_capacity(PAIRS * 2);
                    for round in 0..WARM + PAIRS {
                        for position in 0..2 {
                            let route = (round + position) % 2;
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
                                    position,
                                    route,
                                    f64::from(ms) * 1e6,
                                    wall_ns,
                                ));
                            }
                        }
                    }
                    stream.sync();
                    assert_buffers_equal(&routes);
                    for (index, route) in routes.iter().enumerate() {
                        assert_eq!(
                            route.validate(format, &host_weights, count),
                            expected[index]
                        );
                    }
                    input.guards();
                    for weight in &allocations {
                        weight.guards();
                    }
                    for (pair, position, route, gpu_ns, wall_ns) in samples {
                        println!(
                            "{}",
                            serde_json::json!({"kind":"mmq_next_k_prefetch_inclusive_paired","synthetic":true,
                    "format":format!("{format:?}"),"m":m,"k":k,"n":n,"route":if route==0 {"original_mmq_marker_v2"} else {"next_k_l2_prefetch_mmq_marker_v2"},
                    "pair":pair,"position":position,"weight_mode":mode,"matrix_count":count,"weight_bytes":weight_bytes,"working_set_bytes":count*weight_bytes,
                    "result_stride":result_stride,"result_offset":result_offset,"tested_output_window":if k==5120 {"up half of shared gate/up output"} else {"down projection"},
                    "device_l2_bytes":l2,"working_set_exceeds_3x_l2":count*weight_bytes>3*l2 as usize,"device_allocation_limit_bytes":LIMIT,
                    "projections":laps*count,"j":p.j,"i":p.i,"blocks":p.blocks,"fixup":p.fixup,"gpu_ns":gpu_ns,"wall_ns":wall_ns,
                    "gpu_ns_per_projection":gpu_ns/(laps*count) as f64,"wall_ns_per_projection":wall_ns as f64/(laps*count) as f64,
                    "same_physical_weights_and_input":true,"timed":"original F16 convert, MarkerV2 pack/row checks/guard, actual main and stream-K fixup, MarkerV2 cast",
                    "excluded":"cold plan/kernel setup, weight scan, allocations, uploads, oracle/readbacks","between_routes":"events and graph submission only",
                    "correctness":"all raw/cast/scratch/flags bits, canaries; existing nine-point actual-pack F64 per route",
                    "performance_assertion":null,"model_throughput_or_quality":"not evaluated"})
                        );
                    }
                    drop(graphs);
                    stream.sync();
                }
            }
        }
    }
}
