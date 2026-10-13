//! Strict Q6 F16 row tiling: identical arithmetic, independent test-only T4.

use super::super::captured::{measure, Captured};
use super::*;
use cudarc::driver::sys;
use std::collections::BTreeSet;

const SHAPES: [(usize, usize); 4] = [(5120, 1024), (6144, 5120), (5120, 17408), (17408, 5120)];
// Actual strict-Q6 projection order from prepared metadata; this is not a
// dependency graph for the model. Each leaf has its own physical weight/input.
const ORDER: [usize; 23] = [
    1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 3,
];
const PAIRS_PER_BANK: usize = 8;

#[test]
#[ignore = "requires an actual CUDA device"]
fn q6_t4_preserves_strict_bits_and_f64_boundaries() {
    let context = CudaContext::new(0).expect("Q6 T4 boundaries require CUDA");
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    for (rows, k, n) in [
        (1, 256, 7),
        (3, 768, 17),
        (4, 512, 7),
        (5, 768, 17),
        (8, 2560, 7),
        (9, 768, 17),
        (32, 512, 17),
        (33, 768, 7),
    ] {
        let mut fixture =
            Fixture::<f16>::for_format(&stream, GgufBlockFormat::Q6K, rows, k, n, true);
        for generation in 0..2 {
            if generation != 0 {
                fixture.negate_input(&stream);
            }
            fixture.run(&stream, &kernels.linear_q6k_tiled_f16, 8, 1);
            let expected = fixture.validate(&stream);
            fixture.run(&stream, &kernels.linear_q6k_test_t4_f16, 4, 1);
            assert_eq!(
                fixture.validate(&stream),
                expected,
                "Q6 T4 complete bits {rows}x{k}x{n}"
            );
            // Production remains T8 (or the original scalar M1 route).
            fixture.run_dispatch(&stream, &kernels);
            stream.synchronize().unwrap();
            assert_eq!(
                fixture.validate(&stream),
                expected,
                "production route changed"
            );
        }
        println!(
            "{}",
            serde_json::json!({"kind":"q6_t4_boundary", "m":rows,"k":k,"n":n,
            "all_output_bits_equal":true,"f64_oracle":"all outputs, dense finite input",
            "weight_byte_offset":5,"output_offset":2,"output_stride":n+5,"fresh_input":true})
        );
    }
}

fn encoded_weights(k: usize, n: usize, leaf: usize) -> Vec<u8> {
    // Literal 210-byte Q6 blocks, with independently signed scale bytes. No
    // dense host weight expansion or large F64 matrix multiplication.
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

struct Bank {
    // Drop executables before allocations. Every capture references these exact
    // addresses; both routes share the same immutable weights and fresh input.
    graphs: Vec<[Captured; 2]>,
    leaves: Vec<Fixture<f16>>,
    expected: Vec<Vec<u32>>,
}

fn selected(scope: usize, leaf: usize) -> bool {
    scope == 0 || ORDER[leaf] == scope - 1
}

fn validate_scope(bank: &Bank, stream: &Arc<CudaStream>, scope: usize) {
    for (index, fixture) in bank.leaves.iter().enumerate() {
        if selected(scope, index) {
            assert_eq!(
                fixture.validate_output(stream),
                bank.expected[index],
                "whole-output replay bits"
            );
        }
    }
}

#[test]
#[ignore = "paired real-shape Q6 ring; exclusive CUDA; compressed working set exceeds three L2 caches"]
fn q6_t4_complete_projection_ring_pairs() {
    let context = CudaContext::new(0).expect("Q6 T4 ring requires CUDA");
    // This thread/stream owns all buffers, fences all host reads and drops
    // uploaded graphs before their buffers. Avoid tracking events in capture.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let l2 = usize::try_from(
        context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .unwrap(),
    )
    .unwrap();
    let ring_bytes: usize = ORDER
        .iter()
        .map(|&shape| {
            let (k, n) = SHAPES[shape];
            n * (k / 256) * 210
        })
        .sum();
    // Round up complete independent rings only; never multiply a single leaf's
    // elapsed time or relabel repeated addresses as a large working set.
    let bank_count = (3 * l2 + 1).div_ceil(ring_bytes).max(1);
    let mut banks = Vec::with_capacity(bank_count);
    let mut addresses = BTreeSet::new();
    for bank_index in 0..bank_count {
        let mut leaves = ORDER
            .iter()
            .enumerate()
            .map(|(index, &shape)| {
                let (k, n) = SHAPES[shape];
                let fixture = Fixture::<f16>::for_encoded_weights_without_f64(
                    &stream,
                    GgufBlockFormat::Q6K,
                    8,
                    k,
                    n,
                    encoded_weights(k, n, bank_index * ORDER.len() + index),
                );
                assert!(
                    addresses.insert(fixture.weight_address(&stream)),
                    "ring leaves must own distinct live weights"
                );
                fixture
            })
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(leaves.len());
        for fixture in &mut leaves {
            fixture.run(&stream, &kernels.linear_q6k_tiled_f16, 8, 1);
            let reference = fixture.validate_output(&stream);
            fixture.run(&stream, &kernels.linear_q6k_test_t4_f16, 4, 1);
            assert_eq!(
                fixture.validate_output(&stream),
                reference,
                "initial real-shape complete bits"
            );
            expected.push(reference);
        }
        let graphs = (0..=SHAPES.len())
            .map(|scope| {
                [0, 1].map(|route| {
                    let graph = Captured::new(&stream, || {
                        for (index, fixture) in leaves.iter_mut().enumerate() {
                            if selected(scope, index) {
                                fixture.enqueue(
                                    &stream,
                                    if route == 0 {
                                        &kernels.linear_q6k_tiled_f16
                                    } else {
                                        &kernels.linear_q6k_test_t4_f16
                                    },
                                    if route == 0 { 8 } else { 4 },
                                    1,
                                );
                            }
                        }
                    });
                    // Capture must contain every selected complete projection exactly
                    // once and no upload/reset/validation operations.
                    assert_eq!(
                        graph.node_count,
                        (0..ORDER.len()).filter(|&i| selected(scope, i)).count()
                    );
                    graph
                })
            })
            .collect::<Vec<_>>();
        for (index, fixture) in leaves.iter_mut().enumerate() {
            fixture.negate_input(&stream);
            fixture.run(&stream, &kernels.linear_q6k_tiled_f16, 8, 1);
            let fresh = fixture.validate_output(&stream);
            assert_ne!(
                fresh, expected[index],
                "fresh activation must affect projection"
            );
            expected[index] = fresh;
        }
        let mut bank = Bank {
            graphs,
            leaves,
            expected,
        };
        for scope in 0..=SHAPES.len() {
            for route in 0..2 {
                for (index, fixture) in bank.leaves.iter_mut().enumerate() {
                    if selected(scope, index) {
                        fixture.reset_output(&stream);
                    }
                }
                bank.graphs[scope][route].launch();
                stream.synchronize().unwrap();
                validate_scope(&bank, &stream, scope);
            }
        }
        banks.push(bank);
    }
    assert!(bank_count * ring_bytes > 3 * l2);
    for bank in &banks {
        for _ in 0..2 {
            for route in 0..2 {
                bank.graphs[0][route].launch();
            }
        }
    }
    stream.synchronize().unwrap();
    println!(
        "{}",
        serde_json::json!({"kind":"q6_t4_ring_plan", "product_status":"test_only",
        "m":8,"shapes":SHAPES,"leaf_shape_order":ORDER,"shape_counts":[12,9,1,1],
        "physical_banks":bank_count,"distinct_physical_weight_leaves":addresses.len(),
        "compressed_bytes_per_ring":ring_bytes,"resident_compressed_bytes":bank_count*ring_bytes,
        "device_l2_bytes":l2,"greater_than_three_l2":true,"pairs_per_scope_per_bank":PAIRS_PER_BANK,
        "graph_nodes":banks[0].graphs.iter().map(|g|[g[0].node_count,g[1].node_count]).collect::<Vec<_>>(),
        "routes":["strict_q6_t8","strict_q6_test_t4"],"weight_byte_offset":5,"output_offset":2,
        "full_ring_is_actual_23_launches":true,"per_shape_scope":"actual subset graphs, not sums or extrapolated leaf timings",
        "cache_precondition":"before each arm replay every complete T8 bank in fixed cyclic order, then synchronize; outside measured block; not guaranteed cache eviction or model traffic",
        "wall_scope":"start event enqueue, graph launch, end event enqueue and end-event synchronization",
        "gpu_scope":"events enclosing one complete captured projection ring or shape subset",
        "timing_excludes":["allocation","upload","capture","instantiate","precondition","reset","output_readback","validation"],
        "f64_scope":"separate small boundary selector only; real shapes compare every GPU output bit",
        "fresh_input_replay":true,"performance_threshold":null})
    );
    for scope in 0..=SHAPES.len() {
        let count = (0..ORDER.len()).filter(|&i| selected(scope, i)).count();
        for round in 0..PAIRS_PER_BANK {
            for bank_index in 0..bank_count {
                for position in 0..2 {
                    let route = (round + position) % 2;
                    // Identical address/order cache precondition for both arms.
                    for offset in 0..bank_count {
                        banks[(bank_index + offset) % bank_count].graphs[0][0].launch();
                    }
                    stream.synchronize().unwrap();
                    let bank = &mut banks[bank_index];
                    for (index, fixture) in bank.leaves.iter_mut().enumerate() {
                        if selected(scope, index) {
                            fixture.reset_output(&stream);
                        }
                    }
                    stream.synchronize().unwrap();
                    let (wall_ns, gpu_ns) = measure(&stream, || bank.graphs[scope][route].launch());
                    validate_scope(bank, &stream, scope);
                    println!(
                        "{}",
                        serde_json::json!({"kind":"q6_t4_ring_pair","m":8,
                        "scope":if scope==0 { "complete_ring" } else { "shape_subset" },
                        "shape":if scope==0 { None } else { Some(SHAPES[scope-1]) },
                        "projections":count,"bank":bank_index,"round":round,"position":position,
                        "pair_order":if round%2==0 { "AB" } else { "BA" },
                        "route":if route==0 { "strict_q6_t8" } else { "strict_q6_test_t4" },
                        "row_tile":if route==0 {8} else {4},"graph_nodes":bank.graphs[scope][route].node_count,
                        "wall_ns":wall_ns,"gpu_ns":gpu_ns,"all_output_bits_equal":true})
                    );
                }
            }
        }
    }
    // Check complete retained input/weight allocations once after the whole
    // timed series, never download 430 MB between individual paired blocks.
    for bank in &banks {
        for (fixture, expected) in bank.leaves.iter().zip(&bank.expected) {
            assert_eq!(fixture.validate(&stream), *expected);
        }
    }
}
