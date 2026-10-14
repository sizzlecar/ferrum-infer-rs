//! Native projection diagnostic: actual M8 attention shapes, MMQ versus shared MMVQ.
//! Reuses timing::Route, its actual-packed-input F64 oracle and graph helpers.
//! Different activation pack contracts are compared; no cross-route bit claim.
use super::*;

fn shared_matches_reference(reference: &Route, selected: &Route, count: usize) {
    for (a, b) in [
        (&reference.converted, &selected.converted),
        (&reference.packed, &selected.packed),
        (&reference.output, &selected.output),
        (&reference.result, &selected.result),
        (&reference.rows, &selected.rows),
        (&reference.flags, &selected.flags),
    ] {
        assert_eq!(a.bytes, b.bytes);
        assert_eq!(a.read::<u8>(a.bytes), b.read::<u8>(b.bytes));
        a.guards();
        b.guards();
    }
    assert_eq!(selected.flags.read::<u32>(count), vec![0; count]);
}

fn input_words(bank: usize, k: usize, generation: usize) -> Vec<u16> {
    (0..8 * k)
        .map(|i| {
            f16::from_f32(((i * 17 + bank * 13 + generation * 29) % 101) as f32 / 2048.0 - 0.024)
                .to_bits()
        })
        .collect()
}

struct Bank {
    // Graphs must relinquish captured addresses before the allocations below.
    graphs: [Graph; 2],
    routes: [Route; 2],
    reference: Route,
    input: Buffer,
    allocation: Buffer,
    addresses: Vec<*mut c_void>,
    host_weights: Vec<u8>,
    stable: [Vec<u16>; 2],
    count: usize,
    laps: usize,
}

impl Bank {
    fn new(
        stream: &Stream,
        format: UpstreamLinearFormat,
        bank: usize,
        k: usize,
        n: usize,
        count: usize,
        dot: MmvqDot,
    ) -> Self {
        let mut host_weights = weights(format, k, n);
        let block_bytes = if format == UpstreamLinearFormat::Q4K {
            144
        } else {
            176
        };
        let code_start = if format == UpstreamLinearFormat::Q4K {
            16
        } else {
            48
        };
        for (block_index, block) in host_weights.chunks_exact_mut(block_bytes).enumerate() {
            // Retain finite scales; vary actual codes between banks and columns.
            for (i, code) in block[code_start..].iter_mut().enumerate() {
                *code = ((bank * 31 + block_index * 7 + i * 17) % 256) as u8;
            }
        }
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
        let input = Buffer::new(8 * k * 2);
        input.write(&input_words(bank, k, 0));
        // A nonzero physical output window verifies the other leaf and padding.
        let (output_stride, output_offset) = (2 * n + 5, n + 2);
        let routes =
            [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq].map(|algorithm| {
                Route::new_with_result_window(
                    algorithm,
                    format,
                    8,
                    k as u32,
                    n as u32,
                    count,
                    output_stride as u32,
                    output_offset as u32,
                )
            });
        let reference = Route::new_with_result_window(
            UpstreamLinearAlgorithm::Mmvq,
            format,
            8,
            k as u32,
            n as u32,
            count,
            output_stride as u32,
            output_offset as u32,
        );
        let stable = std::array::from_fn(|arm| {
            for (index, &address) in addresses.iter().enumerate() {
                routes[arm].scan(stream, address, index);
                routes[arm].enqueue_with_mmvq_dot(stream, &input, address, index, dot);
            }
            stream.sync();
            routes[arm].validate(format, &host_weights, count)
        });
        for (index, &address) in addresses.iter().enumerate() {
            reference.scan(stream, address, index);
            reference.enqueue(stream, &input, address, index);
        }
        stream.sync();
        reference.validate(format, &host_weights, count);
        shared_matches_reference(&reference, &routes[1], count);
        let laps = 32_usize.div_ceil(count);
        let graphs = [0, 1].map(|arm| {
            Graph::capture(stream, || {
                for _ in 0..laps {
                    for (index, &address) in addresses.iter().enumerate() {
                        routes[arm].enqueue_with_mmvq_dot(stream, &input, address, index, dot);
                    }
                }
            })
        });
        Self {
            graphs,
            routes,
            reference,
            input,
            allocation,
            addresses,
            host_weights,
            stable,
            count,
            laps,
        }
    }

    fn verify(&self, format: UpstreamLinearFormat) {
        for arm in 0..2 {
            assert_eq!(
                self.routes[arm].validate(format, &self.host_weights, self.count),
                self.stable[arm]
            );
        }
        self.reference
            .validate(format, &self.host_weights, self.count);
        shared_matches_reference(&self.reference, &self.routes[1], self.count);
        self.input.guards();
        self.allocation.guards();
    }

    fn fresh_replay(&self, stream: &Stream, format: UpstreamLinearFormat, bank: usize, k: usize) {
        self.input.write(&input_words(bank, k, 1));
        for graph in &self.graphs {
            graph.launch(stream);
        }
        for (index, &address) in self.addresses.iter().enumerate() {
            self.reference.enqueue(stream, &self.input, address, index);
        }
        stream.sync();
        for arm in 0..2 {
            let fresh = self.routes[arm].validate(format, &self.host_weights, self.count);
            assert_ne!(fresh, self.stable[arm], "capture ignored new input");
        }
        self.reference
            .validate(format, &self.host_weights, self.count);
        shared_matches_reference(&self.reference, &self.routes[1], self.count);
        self.input.guards();
        self.allocation.guards();
    }
}

#[test]
#[ignore = "exclusive CUDA current attention MMQ/shared-MMVQ diagnostic"]
fn marker_v2_m8_attention_mmq_vs_shared_mmvq_complete_projection_pairs() {
    let _ = caps();
    let mut l2 = 0;
    unsafe {
        ok(cudaDeviceGetAttribute(&mut l2, 38, 0));
    }
    assert!(l2 > 0);
    const BANKS: usize = 5;
    const WARM: usize = 8;
    const PAIRS: usize = 20;
    for (format, k, n) in [
        (UpstreamLinearFormat::Q4K, 5120, 10240),
        (UpstreamLinearFormat::Q5K, 6144, 5120),
    ] {
        // Qualified helper checks retain per-algorithm zero/marker semantics.
        // These checks and all cold setup are outside measured intervals.
        shared_weight_decode::validate_selected_markers(format);
        for algorithm in [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq] {
            for input in [
                Input::Finite,
                Input::Zero,
                Input::Subnormal,
                Input::Nonfinite,
                Input::SumOverflow,
            ] {
                check_geometry(
                    algorithm,
                    format,
                    UpstreamLinearLayout::Columns,
                    [8, 768, 17],
                    input,
                    false,
                    true,
                );
            }
            check_geometry(
                algorithm,
                format,
                UpstreamLinearLayout::Columns,
                [8, 768, 17],
                Input::Finite,
                true,
                true,
            );
        }
        let weight_bytes = weights(format, k, n).len();
        let count = 3 * l2 as usize / weight_bytes + 1;
        let stream = Stream::new();
        // All five physical banks and both graphs exist before any warm/timing.
        let banks = (0..BANKS)
            .map(|bank| {
                Bank::new(
                    &stream,
                    format,
                    bank,
                    k,
                    n,
                    count,
                    shared_weight_decode::selected_dot(format),
                )
            })
            .collect::<Vec<_>>();
        let plan = serde_json::json!({
            "experiment":"attention_m8_current_complete_projection",
            "scope":"test-only native projection; no provider/model throughput claim",
            "format":format!("{format:?}"),"M":8,"K":k,"N":n,
            "banks":BANKS,"matrices_per_bank":count,"weight_bytes":weight_bytes,
            "physical_working_set_bytes":BANKS*count*weight_bytes,
            "device_l2_bytes":l2,"laps":banks[0].laps,
            "projections_per_observation":banks[0].laps*count,
            "output_stride":2*n+5,"output_offset":n+2,
            "warm_rounds_per_bank":WARM,"pairs":PAIRS,
            "order":"pair%5 bank; alternating AB/BA; each bank/order twice"
        });
        println!("{plan}");
        println!(
            "{}",
            serde_json::json!({
                "experiment":"attention_m8_current_complete_projection","kind":"scope",
                "timed":"public pack_v2 including flags/guards; dot with actual stream-K fixup; cast",
                "mmq":"current public pack_v2 direct-F16 small rows and kernel flag clear",
                "mmvq":"current public pack_v2 F32 activation and kernel flag clear, SharedM8 dot",
                "precondition":"before each measured arm replay the next bank's complete MMQ ring then synchronize; excluded from event and wall intervals; no cache-eviction guarantee",
                "precondition_projection_count":banks[0].laps*count,
                "precondition_weight_address_bytes":count*weight_bytes,
                "correctness":"own actual-pack nine-point F64; all output finite/cast/guards; full shared-vs-baseline-MMVQ buffers; fresh same-address graph replay",
                "arithmetic":"MMQ/MMVQ activation quantization differs; no cross-algorithm bit equality",
                "excluded":"allocation/upload/plan/cold weight scan/capture/precondition/oracle/readback"
            })
        );
        for bank in &banks {
            for _ in 0..WARM {
                for graph in &bank.graphs {
                    graph.launch(&stream);
                }
            }
        }
        stream.sync();
        let (start, end) = (Event::new(), Event::new());
        let mut samples = Vec::with_capacity(PAIRS * 2);
        for pair in 0..PAIRS {
            let selected = pair % BANKS;
            for order in 0..2 {
                let arm = (pair + order) % 2;
                // Both arms receive the same complete ring from one other bank.
                banks[(selected + 1) % BANKS].graphs[0].launch(&stream);
                stream.sync();
                let wall = Instant::now();
                unsafe {
                    ok(cudaEventRecord(start.0, stream.0));
                }
                banks[selected].graphs[arm].launch(&stream);
                unsafe {
                    ok(cudaEventRecord(end.0, stream.0));
                    ok(cudaEventSynchronize(end.0));
                }
                let wall_ns = wall.elapsed().as_nanos();
                let mut ms = 0.0;
                unsafe {
                    ok(cudaEventElapsedTime(&mut ms, start.0, end.0));
                }
                samples.push((
                    pair,
                    selected,
                    order,
                    arm,
                    f64::from(ms) * 1_000_000.0,
                    wall_ns,
                ));
            }
        }
        stream.sync();
        for (index, bank) in banks.iter().enumerate() {
            bank.verify(format);
            bank.fresh_replay(&stream, format, index, k);
        }
        for (pair, bank, order, arm, gpu_ns, wall_ns) in samples {
            println!(
                "{}",
                serde_json::json!({
                    "experiment":"attention_m8_current_complete_projection","kind":"sample",
                    "format":format!("{format:?}"),"M":8,"K":k,"N":n,
                    "bank":bank,"pair":pair,"order":order,
                    "route":if arm==0 {"mmq_current"} else {"mmvq_shared_m8_current"},
                    "gpu_ns":gpu_ns,"wall_ns":wall_ns,
                    "projections":banks[bank].laps*count
                })
            );
        }
    }
}
