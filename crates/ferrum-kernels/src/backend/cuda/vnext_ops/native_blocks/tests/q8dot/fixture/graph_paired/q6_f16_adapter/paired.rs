use super::*;
use std::collections::BTreeSet;

const SHAPES: [(usize, usize); 4] = [(5120, 1024), (6144, 5120), (5120, 17408), (17408, 5120)];
// Current prepared Q6 body order, not a model dependency graph. Every entry
// owns a distinct allocation; the two routes consume the same allocation.
const ORDER: [usize; 23] = [
    1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 3,
];
const PAIRS: usize = 8;

fn encoded_weights(k: usize, n: usize, leaf: usize) -> Vec<u8> {
    // Literal Q6 blocks with signed group scales and mixed finite half scales.
    // These are synthetic values, not sampled model-weight distributions.
    let mut encoded = vec![0_u8; n * (k / 256) * 210];
    let scales = [0x0000_u16, 0x8000, 0x0001, 0x8001, 0x1000, 0x9400, 0x1800];
    let mut pattern = [0_u8; 208];
    for (index, value) in pattern.iter_mut().enumerate() {
        *value = ((index * 37 + leaf * 19) % 256) as u8;
    }
    for (block, bytes) in encoded.chunks_exact_mut(210).enumerate() {
        bytes[..208].copy_from_slice(&pattern);
        bytes[block % 192] ^= (block % 251) as u8;
        bytes[208..].copy_from_slice(&scales[(block + leaf) % scales.len()].to_le_bytes());
    }
    encoded
}

fn set_input(c: &mut Case, stream: &Arc<CudaStream>, leaf: usize, generation: usize) {
    for row in 0..c.m {
        for column in 0..c.k {
            let integer = ((column * 17 + row * 7 + leaf * 11 + generation * 29) % 127) as f32;
            c.x[row * c.xs + column] = f16::from_f32((integer - 63.0) / 8192.0);
        }
    }
    c.upload_input(stream);
}

fn outputs(c: &Case, stream: &Arc<CudaStream>) -> [Vec<u8>; 2] {
    [c.strict_output.read(stream), c.output.read(stream)]
}

struct Bank {
    // Captured executables are destroyed (and their stream fenced) before the
    // exact input, scratch, flags and weight allocations they reference.
    graphs: [Captured; 2],
    cases: Vec<Case>,
    expected: Vec<[Vec<u8>; 2]>,
}

impl Bank {
    fn check_route(&self, stream: &Arc<CudaStream>, route: usize) {
        for (case, expected) in self.cases.iter().zip(&self.expected) {
            let actual = if route == 0 {
                case.strict_output.read(stream)
            } else {
                case.output.read(stream)
            };
            assert_eq!(actual, expected[route], "complete same-route replay bits");
        }
    }
}

#[test]
#[ignore = "exclusive CUDA; current M8 Q6 body ring and the qualified F16 ABI2 artifact"]
fn q6_f16_adapter_current_m8_complete_projection_ring_pairs() {
    complete_projection_ring_pairs(8);
}

#[test]
#[ignore = "exclusive CUDA; current M32 Q6 body ring and the qualified F16 ABI2 artifact"]
fn q6_f16_adapter_current_m32_complete_projection_ring_pairs() {
    complete_projection_ring_pairs(32);
}

fn complete_projection_ring_pairs(m: usize) {
    let (ctx, stream, kernels) = context();
    let l2 = usize::try_from(
        ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .unwrap(),
    )
    .unwrap();
    assert!(l2 > 0);
    let ring_bytes: usize = ORDER
        .iter()
        .map(|&shape| {
            let (k, n) = SHAPES[shape];
            n * (k / 256) * 210
        })
        .sum();
    let bank_count = (3 * l2 + 1).div_ceil(ring_bytes).max(1);
    let mut addresses = BTreeSet::new();
    let mut banks = Vec::with_capacity(bank_count);
    let mut scan_costs = Vec::with_capacity(bank_count);
    for bank_index in 0..bank_count {
        let mut cases: Vec<_> = ORDER
            .iter()
            .enumerate()
            .map(|(index, &shape)| {
                let (k, n) = SHAPES[shape];
                let leaf = bank_index * ORDER.len() + index;
                let mut case =
                    Case::with_encoded_weights(&stream, m, k, n, encoded_weights(k, n, leaf));
                let address = case.weight_ptr(&stream) as usize;
                assert_eq!(address % 4, 0, "Q6 MMQ requires a four-byte weight base");
                assert!(addresses.insert(address), "distinct live weight allocation");
                set_input(&mut case, &stream, leaf, 0);
                case
            })
            .collect();
        let (scan_wall_ns, scan_gpu_ns) = measure(&stream, || {
            for case in &cases {
                case.scan(&stream);
            }
        });
        scan_costs.push((scan_wall_ns, scan_gpu_ns));
        let mut initial = Vec::with_capacity(cases.len());
        for case in &cases {
            case.strict(&stream, &kernels);
            case.inclusive(&stream);
            stream.synchronize().unwrap();
            case.validate(&stream, false);
            initial.push(outputs(case, &stream));
        }
        let graphs = [0, 1].map(|route| {
            Captured::new(&stream, || {
                for case in &cases {
                    if route == 0 {
                        case.strict(&stream, &kernels);
                    } else {
                        case.inclusive(&stream);
                    }
                }
            })
        });
        assert_eq!(graphs[0].node_count, cases.len());
        let mut expected = Vec::with_capacity(cases.len());
        for (index, case) in cases.iter_mut().enumerate() {
            set_input(case, &stream, bank_index * ORDER.len() + index, 1);
            case.strict(&stream, &kernels);
            case.inclusive(&stream);
            stream.synchronize().unwrap();
            let evidence = case.validate(&stream, false);
            let fresh = outputs(case, &stream);
            for route in 0..2 {
                assert_ne!(fresh[route], initial[index][route], "fresh input observed");
            }
            expected.push(fresh);
            println!(
                "{}",
                serde_json::json!({"kind":"q6_f16_adapter_current_ring_oracle",
                    "bank":bank_index,"leaf":index,"m":case.m,"k":case.k,"n":case.n,
                    "input_stride":case.xs,"output_stride":case.ys,"output_offset":case.output_offset,
                    "weight_alignment_bytes":4,"oracle":evidence})
            );
        }
        let mut bank = Bank {
            graphs,
            cases,
            expected,
        };
        // Prove the pre-existing captures consume fresh input, rather than
        // accepting the results left by the fresh direct reference launches.
        for route in 0..2 {
            for case in &mut bank.cases {
                let output = if route == 0 {
                    &mut case.strict_output
                } else {
                    &mut case.output
                };
                output.write(&stream, &vec![0x35; output.bytes]);
            }
            bank.graphs[route].launch();
            stream.synchronize().unwrap();
            bank.check_route(&stream, route);
        }
        banks.push(bank);
    }
    assert!(bank_count * ring_bytes > 3 * l2);
    for bank in &banks {
        for _ in 0..2 {
            bank.graphs[0].launch();
            bank.graphs[1].launch();
        }
    }
    stream.synchronize().unwrap();
    println!(
        "{}",
        serde_json::json!({"kind":"q6_f16_adapter_current_ring_plan",
            "product_status":"test_only_existing_F16_D4_MMQ_boundary","m":m,
            "shapes":SHAPES,"leaf_shape_order":ORDER,"shape_counts":[12,9,1,1],
            "projections_per_observation":ORDER.len(),"physical_banks":bank_count,
            "distinct_physical_weight_leaves":addresses.len(),"compressed_bytes_per_ring":ring_bytes,
            "resident_compressed_bytes":bank_count*ring_bytes,"device_l2_bytes":l2,
            "pairs":PAIRS,"warm_replays_per_arm_per_bank":2,
            "routes":["current_strict_f16_t8","existing_f16_d4_q6_mmq_marker"],
            "graph_nodes":banks.iter().map(|b|[b.graphs[0].node_count,b.graphs[1].node_count]).collect::<Vec<_>>(),
            "cold_scan_wall_gpu_ns_by_bank":scan_costs,
            "weight_values":"distinct synthetic mixed signed Q6 and finite half scales; not sampled model weights",
            "same_physical_weights_between_routes":true,"same_logical_F16_input_between_routes":true,
            "fresh_input_capture_replay":true,"cross_algorithm_bit_equality_required":false,
            "timed_candidate":"widen/pad, row flags and packed guard reset, D4 pack, MMQ/fixup, F16 marker cast",
            "gpu_scope":"events enclosing one complete 23-leaf graph",
            "wall_scope":"event record, graph submission, end event and synchronization; not CPU-only",
            "cache_precondition":"all strict complete banks in fixed cyclic order before each arm, then synchronize; outside timing; not a model-traffic claim",
            "excluded":["allocation","upload","capture","instantiate","weight scan","precondition","readback","oracle","provider dependency orchestration"],
            "performance_threshold":null})
    );
    for pair in 0..PAIRS {
        let bank_index = pair % bank_count;
        for position in 0..2 {
            let route = (pair + position) % 2;
            for offset in 0..bank_count {
                banks[(bank_index + offset) % bank_count].graphs[0].launch();
            }
            stream.synchronize().unwrap();
            let bank = &banks[bank_index];
            let (wall_ns, gpu_ns) = measure(&stream, || bank.graphs[route].launch());
            bank.check_route(&stream, route);
            println!(
                "{}",
                serde_json::json!({"kind":"q6_f16_adapter_current_ring_pair",
                    "m":m,"pair":pair,"position":position,"bank":bank_index,
                    "pair_order":if pair%2==0 {"AB"} else {"BA"},
                    "route":if route==0 {"current_strict_f16_t8"} else {"existing_f16_d4_q6_mmq_marker"},
                    "projections":ORDER.len(),"gpu_ns":gpu_ns,"wall_ns":wall_ns,
                    "graph_nodes":bank.graphs[route].node_count,"same_route_full_output_bits":true})
            );
        }
    }
    for bank in &banks {
        for case in &bank.cases {
            case.validate(&stream, false);
        }
    }
}
