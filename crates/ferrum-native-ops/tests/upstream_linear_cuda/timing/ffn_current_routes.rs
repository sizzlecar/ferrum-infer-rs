//! M8 FFN route candidates using existing native exports on the same weights.
//! Product FFN still selects MMQ; shared MMVQ is only a diagnostic candidate.
use super::*;

fn same_mmvq_buffers(reference: &Route, selected: &Route, count: usize) {
    for (left, right) in [
        (&reference.converted, &selected.converted),
        (&reference.packed, &selected.packed),
        (&reference.output, &selected.output),
        (&reference.result, &selected.result),
        (&reference.rows, &selected.rows),
        (&reference.flags, &selected.flags),
    ] {
        assert_eq!(left.bytes, right.bytes);
        assert_eq!(
            left.read::<u8>(left.bytes),
            right.read::<u8>(right.bytes),
            "shared MMVQ changed a complete baseline buffer"
        );
        left.guards();
        right.guards();
    }
    assert_eq!(reference.flags.read::<u32>(count), vec![0; count]);
}

#[test]
#[ignore = "exclusive CUDA diagnostic with the current Q4/Q5 shared native exports"]
fn marker_v2_m8_ffn_mmq_vs_current_mmvq_complete_projection_pairs() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    // Reuse the qualified full-buffer zero/subnormal/nonfinite/overflow,
    // malformed-span and odd-output shared-vs-baseline assertions.
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        shared_weight_decode::validate_selected_markers(format);
    }
    const M: usize = 8;
    const WARM: usize = 8;
    const PAIRS: usize = 16;
    for format in [
        UpstreamLinearFormat::Iq4Xs,
        UpstreamLinearFormat::Q4K,
        UpstreamLinearFormat::Q5K,
    ] {
        let dot = shared_weight_decode::selected_dot(format);
        let selected_label = if format == UpstreamLinearFormat::Iq4Xs {
            "mmvq_baseline"
        } else {
            "mmvq_shared_m8"
        };
        for (k, n, destination_stride, destination_offset) in
            [(5120, 17408, 34816, 17408), (17408, 5120, 5120, 0)]
        {
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
                let input_words = |seed: usize| {
                    (0..M * k)
                        .map(|i| {
                            f16::from_f32(((i * 17 + seed * 29) % 101) as f32 / 2048.0 - 0.024)
                                .to_bits()
                        })
                        .collect::<Vec<_>>()
                };
                input.write(&input_words(0));
                let stream = Stream::new();
                let routes =
                    [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq].map(|a| {
                        Route::new_with_result_window(
                            a,
                            format,
                            M as u32,
                            k as u32,
                            n as u32,
                            count,
                            destination_stride,
                            destination_offset,
                        )
                    });
                let reference = Route::new_with_result_window(
                    UpstreamLinearAlgorithm::Mmvq,
                    format,
                    M as u32,
                    k as u32,
                    n as u32,
                    count,
                    destination_stride,
                    destination_offset,
                );
                let enqueue = |arm: usize, index: usize, address| {
                    routes[arm].enqueue_with_mmvq_dot(&stream, &input, address, index, dot);
                };
                let mut stable = Vec::new();
                for (arm, route) in routes.iter().enumerate() {
                    for (index, &address) in addresses.iter().enumerate() {
                        route.scan(&stream, address, index);
                        enqueue(arm, index, address);
                    }
                    stream.sync();
                    stable.push(route.validate(format, &host_weights, count));
                }
                for (index, &address) in addresses.iter().enumerate() {
                    reference.scan(&stream, address, index);
                    reference.enqueue(&stream, &input, address, index);
                }
                stream.sync();
                reference.validate(format, &host_weights, count);
                same_mmvq_buffers(&reference, &routes[1], count);
                let laps = 32_usize.div_ceil(count);
                let graphs = [0, 1].map(|arm| {
                    Graph::capture(&stream, || {
                        for _ in 0..laps {
                            for (index, &address) in addresses.iter().enumerate() {
                                enqueue(arm, index, address);
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
                    assert_eq!(route.validate(format, &host_weights, count), stable[arm]);
                }
                same_mmvq_buffers(&reference, &routes[1], count);
                // Reuse the captured plans with genuinely changed input. Each
                // algorithm validates against its own actual quantized bytes.
                input.write(&input_words(1));
                for graph in &graphs {
                    graph.launch(&stream);
                }
                for (index, &address) in addresses.iter().enumerate() {
                    reference.enqueue(&stream, &input, address, index);
                }
                stream.sync();
                for (arm, route) in routes.iter().enumerate() {
                    let fresh = route.validate(format, &host_weights, count);
                    assert_ne!(
                        fresh, stable[arm],
                        "captured projection ignored fresh input"
                    );
                }
                reference.validate(format, &host_weights, count);
                same_mmvq_buffers(&reference, &routes[1], count);
                input.guards();
                allocation.guards();
                for (pair, order, arm, gpu_ns, wall_ns) in samples {
                    println!(
                        "{}",
                        serde_json::json!({
                            "experiment":"ffn_m8_current_complete_projection",
                            "product_status":"MMQ is current FFN routing; shared MMVQ is a candidate using existing native exports; no product routing change",
                            "format":format!("{format:?}"),"M":M,"K":k,"N":n,
                            "destination_stride":destination_stride,"destination_offset":destination_offset,
                            "weight_mode":mode,"weight_bytes":weight_bytes,"matrix_count":count,
                            "working_set_bytes":count*weight_bytes,"device_l2_bytes":l2,
                            "same_physical_weights_between_algorithms":true,
                            "matrix_contents":"same deterministic bytes; distinct addresses in rings",
                            "laps":laps,"projections":laps*count,"warm_rounds":WARM,"pairs":PAIRS,
                            "timed":"complete F16 convert/MarkerV2 row check+pack/dot+actual fixup/cast",
                            "excluded":"cold plan/static weight scan/allocation/upload/capture/oracle/readback",
                            "correctness":"baseline-vs-shared full buffers; each algorithm all finite/cast/guards/repeat/fresh capture; nine actual-pack F64 samples; GateUp other-leaf canaries",
                            "arithmetic":"MMQ vs MMVQ retain different declared activation packing; no cross-algorithm bit equality",
                            "sample":{"pair":pair,"order":order,"route":if arm==0 {"mmq"} else {selected_label},
                                "gpu_ns":gpu_ns,"wall_ns":wall_ns}
                        })
                    );
                }
            }
        }
    }
}
