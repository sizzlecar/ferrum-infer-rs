//! Test-only cold F32 weight expansion with the retained strict Q8 reduction.
//! The 96-leaf warm ring may fit L2; it does not model intervening model traffic.

use super::super::captured::{measure, Captured};
use super::*;
use cudarc::driver::{sys, CudaSlice};

const K: usize = 5120;
const N: usize = 48;
const LAYERS: usize = 48;
const STRIDE: usize = 16480;
const OFFSETS: [usize; 2] = [16384, 16432];
const GUARD: usize = 8;
const WEIGHT_PREFIX: usize = 5;
const WEIGHT_BYTES: usize = N * (K / 32) * 34;
const CANARY: u16 = 0x3535;
const DECODE_CANARY: f32 = -12345.0;

fn read<T: DeviceRepr>(stream: &Arc<CudaStream>, source: &CudaSlice<T>) -> Vec<T> {
    let values = stream.clone_dtoh(source).unwrap();
    // Event tracking is disabled; explicitly finish the copy before host reads.
    stream.synchronize().unwrap();
    values
}

struct Leaf {
    weights: Vec<u8>,
    encoded: CudaSlice<u8>,
}

impl Leaf {
    fn new(stream: &Arc<CudaStream>, leaf: usize) -> Self {
        let mut weights = vec![0xcc; WEIGHT_PREFIX + WEIGHT_BYTES + GUARD];
        // Literal Q8 layout, with both zero signs, subnormal scales, signed
        // extrema, and a different matrix for every physical projection leaf.
        let scales = [0x0000_u16, 0x8000, 0x0001, 0x8001, 0x2000, 0xa400, 0x2800];
        for (block, bytes) in weights[WEIGHT_PREFIX..WEIGHT_PREFIX + WEIGHT_BYTES]
            .chunks_exact_mut(34)
            .enumerate()
        {
            bytes[..2].copy_from_slice(&scales[(block + leaf) % scales.len()].to_le_bytes());
            for lane in 0..32 {
                bytes[2 + lane] = ((block * 13 + lane * 29 + leaf * 7) % 256) as u8;
            }
        }
        Self {
            encoded: stream.clone_htod(&weights).unwrap(),
            weights,
        }
    }

    fn predecode(
        &self,
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        decoded: &mut CudaSlice<f32>,
        offset: usize,
    ) {
        let source = self
            .encoded
            .slice(WEIGHT_PREFIX..WEIGHT_PREFIX + WEIGHT_BYTES);
        let mut destination = decoded.slice_mut(offset..offset + K * N);
        let (k, n) = (K as u32, N as u32);
        let mut launch = stream.launch_builder(kernel);
        launch.arg(&source).arg(&mut destination).arg(&k).arg(&n);
        // SAFETY: both complete matrices stay live, including the deliberately
        // unaligned GGUF byte view; the destination is a guarded F32 matrix.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (((K * N) as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }

    fn validate_decode(&self, stream: &Arc<CudaStream>, actual: &[f32]) {
        assert_eq!(actual.len(), K * N);
        for (block, bytes) in self.weights[WEIGHT_PREFIX..WEIGHT_PREFIX + WEIGHT_BYTES]
            .chunks_exact(34)
            .enumerate()
        {
            let scale = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
            for lane in 0..32 {
                let expected = scale * f32::from(bytes[2 + lane] as i8);
                assert_eq!(
                    actual[block * 32 + lane].to_bits(),
                    expected.to_bits(),
                    "cold Q8 decode block={block} lane={lane}"
                );
            }
        }
        assert_eq!(read(stream, &self.encoded), self.weights);
    }
}

struct Layer {
    input: Vec<f16>,
    input_gpu: CudaSlice<f16>,
    output_initial: Vec<f16>,
    output_gpu: CudaSlice<f16>,
    // Fusion uses one contiguous pair; the older pilots keep two guarded leaves.
    decoded: Vec<CudaSlice<f32>>,
    leaves: [Leaf; 2],
    rows: usize,
}

impl Layer {
    fn new(stream: &Arc<CudaStream>, rows: usize, layer: usize, fusion: bool) -> Self {
        let input = vec![f16::from_bits(CANARY); GUARD + rows * K + GUARD];
        let mut output_initial = vec![f16::from_bits(CANARY); GUARD + rows * STRIDE + GUARD];
        for row in 0..rows {
            for offset in OFFSETS {
                output_initial[GUARD + row * STRIDE + offset..GUARD + row * STRIDE + offset + N]
                    .fill(f16::NAN);
            }
        }
        let mut result = Self {
            input_gpu: stream.clone_htod(&input).unwrap(),
            output_gpu: stream.clone_htod(&output_initial).unwrap(),
            decoded: (0..if fusion { 1 } else { 2 })
                .map(|_| {
                    let elements = GUARD + (if fusion { 2 * K * N } else { K * N }) + GUARD;
                    stream.clone_htod(&vec![DECODE_CANARY; elements]).unwrap()
                })
                .collect(),
            input,
            output_initial,
            leaves: [
                Leaf::new(stream, layer * 2),
                Leaf::new(stream, layer * 2 + 1),
            ],
            rows,
        };
        result.set_input(stream, layer, 0);
        result
    }

    fn set_input(&mut self, stream: &Arc<CudaStream>, layer: usize, generation: usize) {
        for row in 0..self.rows {
            for column in 0..K {
                self.input[GUARD + row * K + column] = f16::from_f32(
                    ((column * 17 + row * 31 + layer * 11 + generation * 23) % 127) as f32 / 4096.0
                        - 63.0 / 4096.0,
                );
            }
        }
        stream
            .memcpy_htod(&self.input, &mut self.input_gpu)
            .unwrap();
    }

    fn reset(&mut self, stream: &Arc<CudaStream>) {
        stream
            .memcpy_htod(&self.output_initial, &mut self.output_gpu)
            .unwrap();
    }

    fn predecode(&mut self, stream: &Arc<CudaStream>, kernel: &CudaFunction) {
        for (index, leaf) in self.leaves.iter().enumerate() {
            let (allocation, offset) = if self.decoded.len() == 1 {
                (0, GUARD + index * K * N)
            } else {
                (index, GUARD)
            };
            leaf.predecode(stream, kernel, &mut self.decoded[allocation], offset);
        }
    }

    fn validate_decode(&self, stream: &Arc<CudaStream>) {
        for (allocation, decoded) in self.decoded.iter().enumerate() {
            let actual = read(stream, decoded);
            let leaves = if self.decoded.len() == 1 { 2 } else { 1 };
            assert!(actual[..GUARD]
                .iter()
                .chain(&actual[GUARD + leaves * K * N..])
                .all(|value| value.to_bits() == DECODE_CANARY.to_bits()));
            for local in 0..leaves {
                let index = if leaves == 2 { local } else { allocation };
                let start = GUARD + local * K * N;
                self.leaves[index].validate_decode(stream, &actual[start..start + K * N]);
            }
        }
    }

    fn enqueue(&mut self, stream: &Arc<CudaStream>, kernel: &CudaFunction, decoded: bool) {
        let input = self.input_gpu.slice(GUARD..GUARD + self.rows * K);
        for (index, (leaf, offset)) in self.leaves.iter().zip(OFFSETS).enumerate() {
            let mut output = self.output_gpu.slice_mut(GUARD..GUARD + self.rows * STRIDE);
            let encoded = leaf
                .encoded
                .slice(WEIGHT_PREFIX..WEIGHT_PREFIX + WEIGHT_BYTES);
            let (allocation, start) = if self.decoded.len() == 1 {
                (0, GUARD + index * K * N)
            } else {
                (index, GUARD)
            };
            let weights_f32 = self.decoded[allocation].slice(start..start + K * N);
            let params = [
                self.rows as u32,
                K as u32,
                N as u32,
                STRIDE as u32,
                offset as u32,
            ];
            let format = [GgufBlockFormat::Q8_0.ggml_type_id(), 32_u32, 34_u32];
            let mut launch = stream.launch_builder(kernel);
            launch.arg(&input);
            if decoded {
                launch.arg(&weights_f32);
            } else {
                launch.arg(&encoded);
            }
            launch.arg(&mut output);
            for param in &params {
                launch.arg(param);
            }
            if !decoded {
                for param in &format {
                    launch.arg(param);
                }
            }
            // SAFETY: full inputs, two disjoint F16 output windows in a shared
            // result, and immutable encoded/expanded matrices remain live.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: ((N as u32).div_ceil(4), self.rows as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
    }

    fn enqueue_fused(&mut self, stream: &Arc<CudaStream>, kernel: &CudaFunction) {
        let input = self.input_gpu.slice(GUARD..GUARD + self.rows * K);
        let weights = self.decoded[0].slice(GUARD..GUARD + 2 * K * N);
        let mut output = self.output_gpu.slice_mut(GUARD..GUARD + self.rows * STRIDE);
        let params = [
            self.rows as u32,
            K as u32,
            (2 * N) as u32,
            STRIDE as u32,
            OFFSETS[0] as u32,
        ];
        let mut launch = stream.launch_builder(kernel);
        launch.arg(&input).arg(&weights).arg(&mut output);
        for param in &params {
            launch.arg(param);
        }
        // SAFETY: the two column-major N48 matrices are contiguous in the
        // same retained allocation; the adjacent output windows form N96.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (((2 * N) as u32).div_ceil(4), self.rows as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }

    fn read_checked(&self, stream: &Arc<CudaStream>) -> Vec<u16> {
        let values = read(stream, &self.output_gpu);
        for (index, value) in values.iter().enumerate() {
            let logical = index
                .checked_sub(GUARD)
                .filter(|&i| i < self.rows * STRIDE)
                .is_some_and(|i| {
                    OFFSETS
                        .iter()
                        .any(|&offset| (offset..offset + N).contains(&(i % STRIDE)))
                });
            if logical {
                assert!(
                    value.is_finite(),
                    "nonfinite or unwritten output at {index}"
                );
            } else {
                assert_eq!(value.to_bits(), CANARY, "shared output guard at {index}");
            }
        }
        assert_eq!(
            read(stream, &self.input_gpu)
                .into_iter()
                .map(f16::to_bits)
                .collect::<Vec<_>>(),
            self.input
                .iter()
                .copied()
                .map(f16::to_bits)
                .collect::<Vec<_>>()
        );
        values.into_iter().map(f16::to_bits).collect()
    }
}

#[test]
#[ignore = "paired CUDA graph pilot; requires exclusive GPU access"]
fn q8_0_predecoded_f32_preserves_bits_and_paired_layer_ring_on_cuda() {
    paired_layer_ring(&[1, 8, 32], 1, 16, false);
}

#[test]
#[ignore = "five physical weight banks; paired CUDA cache-capacity control"]
fn q8_0_predecoded_f32_m8_five_bank_capacity_control_on_cuda() {
    paired_layer_ring(&[8], 5, 20, false);
}

#[test]
#[ignore = "five-bank paired N48/N96 CUDA launch-fusion diagnostic"]
fn q8_0_predecoded_f32_pair_fusion_m8_m32_five_bank_on_cuda() {
    paired_layer_ring(&[8, 32], 5, 20, true);
}

#[test]
#[ignore = "CUDA correctness for paired M2 decode and 75-token prefill"]
fn q8_0_predecoded_f32_pair_fusion_m2_m75_correctness_on_cuda() {
    // Reuse full compressed/independent/fused bits, fresh-input replay and
    // guarded windows. One bank/pair is correctness coverage, not timing evidence.
    paired_layer_ring(&[2, 75], 1, 1, true);
}

struct Bank {
    // Field order also destroys graphs before allocations if validation panics.
    graphs: [Captured; 2],
    layers: Vec<Layer>,
    reference: Vec<Vec<u16>>,
    cold_wall_ns: f64,
    cold_gpu_ns: f64,
}

fn paired_layer_ring(row_counts: &[usize], bank_count: usize, pairs: usize, fusion: bool) {
    const WARM: usize = 8;
    let context = CudaContext::new(0).expect("Q8 predecode pilot requires CUDA");
    // One thread/stream owns every buffer; explicit fences surround host reads
    // and graph destruction, matching the existing captured-kernel fixtures.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
        .unwrap();
    let cold = module
        .load_function("vnext_gguf_q8_0_predecode_f32")
        .unwrap();
    let hot = module
        .load_function("vnext_gguf_linear_q8_0_predecoded_f32_f16")
        .unwrap();
    let l2 = context
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap();
    for &rows in row_counts {
        let mut banks = Vec::with_capacity(bank_count);
        for bank in 0..bank_count {
            let mut layers = (0..LAYERS)
                .map(|layer| Layer::new(&stream, rows, bank * LAYERS + layer, fusion))
                .collect::<Vec<_>>();
            stream.synchronize().unwrap();
            let (cold_wall_ns, cold_gpu_ns) = measure(&stream, || {
                for layer in &mut layers {
                    layer.predecode(&stream, &cold);
                }
            });
            for layer in &layers {
                layer.validate_decode(&stream);
            }
            let mut reference = Vec::new();
            for layer in &mut layers {
                layer.enqueue(&stream, &kernels.linear_q8_0_f16, false);
                reference.push(layer.read_checked(&stream));
                layer.reset(&stream);
                layer.enqueue(&stream, &hot, true);
                assert_eq!(layer.read_checked(&stream), *reference.last().unwrap());
                if fusion {
                    layer.reset(&stream);
                    layer.enqueue_fused(&stream, &hot);
                    assert_eq!(layer.read_checked(&stream), *reference.last().unwrap());
                }
            }
            let graphs = [false, true].map(|candidate| {
                Captured::new(&stream, || {
                    for layer in &mut layers {
                        if fusion && candidate {
                            layer.enqueue_fused(&stream, &hot);
                        } else {
                            let decoded = fusion || candidate;
                            layer.enqueue(
                                &stream,
                                if decoded {
                                    &hot
                                } else {
                                    &kernels.linear_q8_0_f16
                                },
                                decoded,
                            );
                        }
                    }
                })
            });
            // Keep graph addresses fixed, change all activation values, and derive
            // a fresh direct baseline before checking each captured route.
            for (index, layer) in layers.iter_mut().enumerate() {
                layer.set_input(&stream, bank * LAYERS + index, 1);
                layer.reset(&stream);
                layer.enqueue(&stream, &kernels.linear_q8_0_f16, false);
                let fresh = layer.read_checked(&stream);
                assert_ne!(
                    fresh, reference[index],
                    "changed input must affect the result"
                );
                reference[index] = fresh;
            }
            for graph in &graphs {
                for layer in &mut layers {
                    layer.reset(&stream);
                }
                graph.launch();
                stream.synchronize().unwrap();
                for (layer, expected) in layers.iter().zip(&reference) {
                    assert_eq!(
                        layer.read_checked(&stream),
                        *expected,
                        "captured fresh-input bits"
                    );
                }
            }
            banks.push(Bank {
                graphs,
                layers,
                reference,
                cold_wall_ns,
                cold_gpu_ns,
            });
        }
        for warm in 0..WARM * bank_count {
            let bank = warm % bank_count;
            for route in [warm % 2, 1 - warm % 2] {
                banks[bank].graphs[route].launch();
            }
        }
        stream.synchronize().unwrap();
        let mut plan = serde_json::json!({
                "kind":if fusion { "q8_f32_pair_fusion_ring_plan" } else { "q8_f32_predecode_ring_plan" },
                "product_status":"test_only_candidate",
                "m":rows, "k":K, "n":N, "layers":LAYERS, "layers_per_bank":LAYERS,
                "physical_banks":bank_count, "distinct_weight_leaves":bank_count*2*LAYERS,
                "projection_leaves_per_observation":2*LAYERS,
                "decoded_allocations_per_layer":if fusion { 1 } else { 2 },
                "decoded_matrices_per_allocation":if fusion { 2 } else { 1 },
                "timed_routes":if fusion { ["predecoded_f32_two_n48", "predecoded_f32_one_n96"] }
                    else { ["strict_q8_f16", "predecoded_f32_f16"] },
                "compressed_reference_timed":!fusion,
                "kernel_calls_per_observation":if fusion { [2*LAYERS,LAYERS] } else { [2*LAYERS,2*LAYERS] },
                "result_stride":STRIDE, "result_offsets":OFFSETS,
                "additional_resident_f32_bytes":bank_count*2*LAYERS*K*N*4,
                "retained_q8_bytes":bank_count*2*LAYERS*WEIGHT_BYTES,
                "f32_bytes_per_observation":2*LAYERS*K*N*4, "q8_bytes_per_observation":2*LAYERS*WEIGHT_BYTES,
                "cold_predecode_gpu_ns":banks.iter().map(|bank| bank.cold_gpu_ns).sum::<f64>(),
                "cold_predecode_wall_ns":banks.iter().map(|bank| bank.cold_wall_ns).sum::<f64>(),
        });
        let scope = serde_json::json!({
                "cold_scope":"device Q8-to-F32 conversions for all resident banks; allocation and upload excluded",
                "device_l2_bytes":l2,
                "working_set":if bank_count == 1 { "warm 96-leaf ring; may fit L2; no intervening model traffic" }
                    else { "rotating physical banks; cache-capacity control, not guaranteed eviction or model traffic" },
                "bank_selection":"pair modulo physical_banks; both paired arms use the same bank",
                "pair_order":"alternating AB and BA",
                "other_bank_precondition":if fusion { serde_json::json!({
                    "before_each_measured_arm":true,
                    "route":"predecoded_f32_two_n48",
                    "bank_order":"selected bank plus 1..physical_banks modulo physical_banks",
                    "graph_replays":bank_count-1,
                    "f32_weight_bytes":(bank_count-1)*2*LAYERS*K*N*4,
                    "scope":"other-bank graph launches followed by stream synchronization; excluded from measured GPU and host wall time; cache eviction is not guaranteed"
                }) } else { serde_json::Value::Null },
                "graph_nodes":banks[0].graphs.iter().map(|graph| graph.node_count).collect::<Vec<_>>(),
                "graph_nodes_per_bank":banks.iter().map(|bank| bank.graphs.iter().map(|graph| graph.node_count).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "warm_rounds":WARM, "warm_rounds_per_bank":WARM, "formal_pairs":pairs, "graph_replays_per_observation":1,
                "cold_decoded_values_match_literal_q8_oracle":true,
                "all_output_bits_equal":true, "shared_window_guards":true, "fresh_input_replay":true,
                "performance_threshold":null,
                "excluded_from_hot_timing":["allocation","uploads","cold_predecode","graph_capture","reset","validation"]
        });
        plan.as_object_mut()
            .unwrap()
            .extend(scope.as_object().unwrap().clone());
        println!("{plan}");
        for pair in 0..pairs {
            let bank = pair % bank_count;
            for position in 0..2 {
                let route = (pair + position) % 2;
                if fusion {
                    for offset in 1..bank_count {
                        banks[(bank + offset) % bank_count].graphs[0].launch();
                    }
                    stream.synchronize().unwrap();
                }
                let (wall_ns, gpu_ns) = measure(&stream, || banks[bank].graphs[route].launch());
                println!(
                    "{}",
                    serde_json::json!({
                        "kind":if fusion { "q8_f32_pair_fusion_ring_pair" } else { "q8_f32_predecode_ring_pair" },
                        "m":rows, "k":K, "n":N,
                        "pair":pair, "position":position, "bank":bank, "physical_banks":bank_count,
                        "pair_order":if pair % 2 == 0 { "AB" } else { "BA" },
                        "route":if fusion {
                            if route == 0 { "predecoded_f32_two_n48" } else { "predecoded_f32_one_n96" }
                        } else if route == 0 { "strict_q8_f16" } else { "predecoded_f32_f16" },
                        "projection_leaves":2*LAYERS, "graph_replays":1,
                        "kernel_calls":if fusion && route == 1 { LAYERS } else { 2*LAYERS },
                        "other_bank_precondition_replays":if fusion { bank_count-1 } else { 0 },
                        "gpu_ns":gpu_ns, "wall_ns":wall_ns,
                        "gpu_ns_per_leaf":gpu_ns/(2*LAYERS) as f64,
                        "performance_threshold":null
                    })
                );
            }
        }
        for bank in banks {
            for (layer, expected) in bank.layers.iter().zip(&bank.reference) {
                assert_eq!(
                    layer.read_checked(&stream),
                    *expected,
                    "timed replay final bits"
                );
                layer.validate_decode(&stream);
            }
            // Graphs retain raw addresses; destroy them before fixture allocations.
            drop(bank);
        }
    }
}
