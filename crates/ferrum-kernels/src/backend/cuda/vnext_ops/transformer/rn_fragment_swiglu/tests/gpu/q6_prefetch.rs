//! Independent Q6-only production-selector screen against the unchanged global entry.
//! Reuses the prior direct-production paired projection screen and production helpers.
use super::*;
use cudarc::driver::{sys::CUevent_flags, CudaGraph};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

const ENTRIES: [&str; 2] = [
    "vnext_rn_fragment_q6_prefetch_mma",
    "vnext_rn_fragment_coefficients",
];
const ROWS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const WARMUP: usize = 2;
const PAIRS: usize = 6;
const REPEATS: usize = 4;

fn entry(format: RnF16FragmentSourceFormatV1) -> &'static str {
    plan::FragmentKernel::for_format(format).entry()
}

#[test]
fn rn_q6_prefetch_embedded_async_instruction_exists() {
    assert!(compiled_mma_target(crate::ptx::VNEXT_GGUF));
    let clean = crate::ptx::VNEXT_GGUF
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let words = clean
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    for name in [ENTRY].into_iter().chain(ENTRIES) {
        assert!(
            words.windows(2).any(|w| w == [".entry", name]),
            "missing {name}"
        );
    }
    let baseline = ptx_entry(crate::ptx::VNEXT_GGUF, ENTRY);
    let candidate = ptx_entry(crate::ptx::VNEXT_GGUF, ENTRIES[0]);
    assert!(
        candidate.contains("cp.async.ca.shared.global")
            && candidate.contains("cp.async.commit_group")
            && candidate.contains("cp.async.wait_group"),
        "candidate did not produce the requested asynchronous packet-copy mechanism"
    );
    println!(
        "{}",
        json!({"event":"ptx_mechanism_qualification","candidate_async_packet_copy_present":true,"instruction_counts_are_diagnostics_not_correctness_gates":true,
    "baseline_mma_sites":baseline.matches("mma.sync.aligned.m16n8k16").count(),"candidate_mma_sites":candidate.matches("mma.sync.aligned.m16n8k16").count(),"baseline_packed_half_conversions":baseline.matches("cvt.rn.f16x2.f32").count(),"baseline_scalar_half_conversions":baseline.matches("cvt.rn.f16.f32").count(),
    "candidate_scalar_half_conversions":candidate.matches("cvt.rn.f16.f32").count(),
    "candidate_async_copy_sites":candidate.matches("cp.async.ca.shared.global").count(),
    "baseline_entry_sha256":format!("{:x}",Sha256::digest(baseline.as_bytes())),
    "candidate_entry_sha256":format!("{:x}",Sha256::digest(candidate.as_bytes())),
    "static_instruction_counts_are_not_dynamic_counts":true})
    );
}

fn ptx_entry<'a>(ptx: &'a str, name: &str) -> &'a str {
    let declaration = format!(".entry {name}(");
    let start = ptx.find(&declaration).expect("entry declaration");
    let body = ptx[start..].find('{').unwrap() + start;
    let mut depth = 0usize;
    for (offset, byte) in ptx.as_bytes()[body..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &ptx[start..=body + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated PTX entry");
}

#[derive(Clone, Copy)]
struct Args {
    input: u64,
    packet: u64,
    bytes: u64,
    output: u64,
    rows: u32,
    k: u32,
    n: u32,
    stride: u32,
    offset: u32,
    format: u32,
    abi: u32,
}
impl Args {
    fn launch(&self, stream: &Arc<CudaStream>, function: &CudaFunction) {
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&self.input)
            .arg(&self.packet)
            .arg(&self.bytes)
            .arg(&self.output)
            .arg(&self.rows)
            .arg(&self.k)
            .arg(&self.n)
            .arg(&self.stride)
            .arg(&self.offset)
            .arg(&self.format)
            .arg(&self.abi);
        // SAFETY: the fixture owns all guarded allocations until graph drop.
        // Invalid cases below can only return before dereferencing those spans.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.n.max(1).div_ceil(16), self.rows.max(1).div_ceil(8), 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }
}

struct Weight {
    plan: RnF16FragmentPlanV1,
    packet: Live<u8>,
    source_sha: String,
    packet_sha: String,
}
impl Weight {
    fn new(
        stream: &Arc<CudaStream>,
        plan: RnF16FragmentPlanV1,
        salt: usize,
        coeff: &CudaFunction,
    ) -> Self {
        let raw = source(plan, salt);
        let packed = crate::gguf_rn_fragment::pack_rn_f16_fragments(&plan, &[&raw]).unwrap();
        let packet = Live::new(stream, &packed, 0xa5, 128);
        let format = match plan.source_format() {
            RnF16FragmentSourceFormatV1::Q4K => GgufBlockFormat::Q4K,
            RnF16FragmentSourceFormatV1::Q5K => GgufBlockFormat::Q5K,
            RnF16FragmentSourceFormatV1::Q6K => GgufBlockFormat::Q6K,
        };
        let dense =
            crate::gguf_f16_projection_materializer::convert_rn_f16_diagnostic(format, &raw)
                .unwrap()
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect::<Vec<_>>();
        verify_packed_coefficients(stream, coeff, plan, &packet, &dense);
        Self {
            plan,
            packet,
            source_sha: format!("{:x}", Sha256::digest(&raw)),
            packet_sha: format!("{:x}", Sha256::digest(&packed)),
        }
    }
    fn immutable(&self, stream: &Arc<CudaStream>) {
        assert_eq!(
            format!("{:x}", Sha256::digest(self.packet.read(stream))),
            self.packet_sha
        );
    }
}

fn values(count: usize, generation: usize) -> Vec<f16> {
    (0..count)
        .map(|i| f16::from_f32(((i * 13 + generation * 7) % 41) as f32 / 64.0 - 0.3125))
        .collect()
}
fn bits(values: &[f16]) -> Vec<u16> {
    assert!(values.iter().all(|v| v.is_finite()), "nonfinite output");
    values.iter().map(|v| v.to_bits()).collect()
}
fn capture(stream: &Arc<CudaStream>, enqueue: impl Fn()) -> CudaGraph {
    enqueue();
    stream.synchronize().unwrap();
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    enqueue();
    stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .unwrap()
        .unwrap()
}

fn small(
    stream: &Arc<CudaStream>,
    module: &Arc<cudarc::driver::CudaModule>,
    baseline: &CudaFunction,
) {
    let coeff = module
        .load_function("vnext_rn_fragment_coefficients")
        .unwrap();
    for format in [
        RnF16FragmentSourceFormatV1::Q4K,
        RnF16FragmentSourceFormatV1::Q5K,
        RnF16FragmentSourceFormatV1::Q6K,
    ] {
        let candidate = module.load_function(entry(format)).unwrap();
        for (n, k) in [(17, 256), (33, 512), (33, 768)] {
            let plan = RnF16FragmentPlanV1::new(format, n, k).unwrap();
            // Original RN converter and GPU coefficients agree at every half bit.
            // These small views deliberately retain the existing unaligned guards.
            let (dense, dense_device, packet) = matrix(stream, &coeff, plan, 11);
            // Both legal byte-aligned fallback and 16-byte asynchronous paths must
            // qualify at partial N tiles, one/two groups and a reused third slot.
            let packed_bytes = packet.read(stream);
            let aligned_packet = Live::new(stream, &packed_bytes, 0xa5, 128);
            for (packet, alignment) in [(&packet, 5_u64), (&aligned_packet, 0_u64)] {
                assert_eq!(packet.ptr(stream) % 16, alignment);
                for rows in 1..=8 {
                    let mut input = Live::new(
                        stream,
                        &values(rows * k as usize, 0),
                        f16::from_f32(-117.0),
                        3,
                    );
                    let output = [
                        Live::new(
                            stream,
                            &vec![f16::ZERO; rows * n as usize],
                            f16::from_f32(-119.0),
                            3,
                        ),
                        Live::new(
                            stream,
                            &vec![f16::ZERO; rows * n as usize],
                            f16::from_f32(-119.0),
                            3,
                        ),
                    ];
                    // DevicePtr has synchronization side effects. Resolve all pointers
                    // before capture and keep every allocation alive with the graphs.
                    let base = Args {
                        input: input.ptr(stream),
                        packet: packet.ptr(stream),
                        bytes: plan.packed_bytes(),
                        output: 0,
                        rows: rows as u32,
                        k: k as u32,
                        n: n as u32,
                        stride: n as u32,
                        offset: 0,
                        format: plan::format_code(format),
                        abi: plan.packing_abi(),
                    };
                    let args = output.each_ref().map(|y| Args {
                        output: y.ptr(stream),
                        ..base
                    });
                    let functions = [baseline, &candidate];
                    let graphs = [
                        capture(stream, || args[0].launch(stream, functions[0])),
                        capture(stream, || args[1].launch(stream, functions[1])),
                    ];
                    let mut previous = None;
                    for generation in 0..2 {
                        let host = values(rows * k as usize, generation);
                        input.replace(stream, &host);
                        let mut observed = Vec::new();
                        for arm in 0..2 {
                            graphs[arm].launch().unwrap();
                            let actual = output[arm].read(stream);
                            check_projection(&host, &dense, &actual, k as usize, n as usize);
                            let expected = bits(&actual);
                            graphs[arm].launch().unwrap();
                            assert_eq!(bits(&output[arm].read(stream)), expected, "replay bits");
                            observed.push(expected);
                        }
                        assert_eq!(
                            observed[0], observed[1],
                            "packet prefetch changes arithmetic"
                        );
                        if let Some(prior) = previous {
                            assert_ne!(observed[0], prior, "changed input ignored");
                        }
                        previous = Some(observed.remove(0));
                        assert_eq!(input.read(stream), host);
                    }
                    // Guards must leave every output byte unchanged, for both entries.
                    for arm in 0..2 {
                        let before = bits(&output[arm].read(stream));
                        let a = args[arm];
                        for invalid in [
                            Args {
                                bytes: a.bytes - 1,
                                ..a
                            },
                            Args { abi: 0, ..a },
                            Args { format: 0, ..a },
                            Args { rows: 0, ..a },
                            Args { k: a.k - 1, ..a },
                            Args { n: 0, ..a },
                            Args {
                                stride: a.n - 1,
                                ..a
                            },
                            Args {
                                offset: a.stride + 1,
                                ..a
                            },
                        ] {
                            invalid.launch(stream, functions[arm]);
                            assert_eq!(
                                bits(&output[arm].read(stream)),
                                before,
                                "invalid launch wrote output"
                            );
                        }
                    }
                }
            }
            assert_eq!(aligned_packet.read(stream), packed_bytes);
            assert_eq!(dense_device.read(stream), dense);
            assert_eq!(
                packet.read(stream),
                packet.host[packet.prefix..packet.prefix + packet.count]
            );
        }
    }
    println!(
        "{}",
        json!({"event":"small_qualification", "formats":["Q4K","Q5K","Q6K"],
        "all_m_1_to_8":true,"unaligned_views":true,"aligned_async_and_unaligned_global_paths":true,"n_tails":[17,33],"k":[256,512,768],
        "all_coeff_half_bits_match_materializer":true,"all_output_bits_equal":true,
        "all_f64_dots_checked":true,"changed_input_replay":true,"invalid_guards_preserved":true})
    );
}

fn verify_packed_coefficients(
    stream: &Arc<CudaStream>,
    coeff: &CudaFunction,
    plan: RnF16FragmentPlanV1,
    packet: &Live<u8>,
    dense: &[f16],
) {
    let output = Live::new(
        stream,
        &vec![f16::NAN; dense.len()],
        f16::from_f32(-119.0),
        3,
    );
    let (p, bytes, y, k, n, format, abi) = (
        packet.ptr(stream),
        plan.packed_bytes(),
        output.ptr(stream),
        plan.k() as u32,
        plan.n() as u32,
        plan::format_code(plan.source_format()),
        plan.packing_abi(),
    );
    let mut launch = stream.launch_builder(coeff);
    launch
        .arg(&p)
        .arg(&bytes)
        .arg(&y)
        .arg(&k)
        .arg(&n)
        .arg(&format)
        .arg(&abi);
    // SAFETY: the original coefficient entry writes one element per thread; the typed plan bounds N*K and packet spans.
    unsafe { launch.launch(LaunchConfig::for_num_elems(dense.len() as u32)) }.unwrap();
    assert_eq!(
        bits(&output.read(stream)),
        bits(dense),
        "packed coefficient RN bits differ"
    );
}

// Actual production dimensions with synthetic quantized weights. Complete FFN
// includes both original half stage boundaries.
// This correctness phase runs before any event timing; it has no performance claim.
fn full_ffn(
    stream: &Arc<CudaStream>,
    context: &Arc<CudaContext>,
    module: &Arc<cudarc::driver::CudaModule>,
    baseline: &CudaFunction,
    coeff: &CudaFunction,
) {
    let candidate = module.load_function(ENTRIES[0]).unwrap();
    let silu = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap()
        .load_function(SILU_MUL_FUNCTION_NAME)
        .unwrap();
    let gate_plan =
        RnF16FragmentPlanV1::new(RnF16FragmentSourceFormatV1::Q4K, 24576, 4096).unwrap();
    let gates = [
        Weight::new(stream, gate_plan, 307, coeff),
        Weight::new(stream, gate_plan, 1298, coeff),
    ];
    for format in [
        RnF16FragmentSourceFormatV1::Q4K,
        RnF16FragmentSourceFormatV1::Q6K,
    ] {
        let down_plan = RnF16FragmentPlanV1::new(format, 4096, 12288).unwrap();
        let downs = [
            Weight::new(stream, down_plan, 403, coeff),
            Weight::new(stream, down_plan, 1394, coeff),
        ];
        for rows in ROWS {
            for set in 0..2 {
                let mut input =
                    Live::new(stream, &values(rows * 4096, 0), f16::from_f32(-117.0), 64);
                let gate = [0, 1].map(|_| {
                    Live::new(
                        stream,
                        &vec![f16::NAN; rows * 24576],
                        f16::from_f32(-119.0),
                        64,
                    )
                });
                let activation = [0, 1].map(|_| {
                    Live::new(
                        stream,
                        &vec![f16::NAN; rows * 12288],
                        f16::from_f32(-119.0),
                        64,
                    )
                });
                let output = [0, 1].map(|_| {
                    Live::new(
                        stream,
                        &vec![f16::NAN; rows * 4096],
                        f16::from_f32(-119.0),
                        64,
                    )
                });
                let x = input.ptr(stream);
                let g = gate.each_ref().map(|v| v.ptr(stream));
                let a = activation.each_ref().map(|v| v.ptr(stream));
                let y = output.each_ref().map(|v| v.ptr(stream));
                let gp = gates[set].packet.ptr(stream);
                let dp = downs[set].packet.ptr(stream);
                let shape = Shape::new(rows as u64, 4096, 12288, gate_plan.source_format(), format)
                    .unwrap();
                let gate_functions = [
                    baseline,
                    plan::fragment_function(gate_plan, baseline, &candidate),
                ];
                let down_functions = [
                    baseline,
                    plan::fragment_function(down_plan, baseline, &candidate),
                ];
                let enqueue = |arm| {
                    plan::launch(stream, gate_functions[arm], shape, gate_plan, x, gp, g[arm])
                        .unwrap();
                    launch_silu_mul(
                        stream,
                        &silu,
                        g[arm],
                        a[arm],
                        shape.intermediate,
                        shape.activation_elements,
                    )
                    .unwrap();
                    plan::launch(
                        stream,
                        down_functions[arm],
                        shape,
                        down_plan,
                        a[arm],
                        dp,
                        y[arm],
                    )
                    .unwrap();
                };
                let graphs = [
                    capture(stream, || enqueue(0)),
                    capture(stream, || enqueue(1)),
                ];
                let mut previous = None;
                for generation in 0..2 {
                    let host = values(rows * 4096, generation);
                    input.replace(stream, &host);
                    let mut observed = Vec::new();
                    for arm in 0..2 {
                        graphs[arm].launch().unwrap();
                        let actual = [
                            bits(&gate[arm].read(stream)),
                            bits(&activation[arm].read(stream)),
                            bits(&output[arm].read(stream)),
                        ];
                        graphs[arm].launch().unwrap();
                        assert_eq!(
                            actual,
                            [
                                bits(&gate[arm].read(stream)),
                                bits(&activation[arm].read(stream)),
                                bits(&output[arm].read(stream)),
                            ],
                            "full FFN replay bits changed"
                        );
                        observed.push(actual);
                    }
                    assert_eq!(
                        observed[0], observed[1],
                        "full FFN intermediate/output bits differ"
                    );
                    if let Some(prior) = previous {
                        assert_ne!(observed[0][0], prior, "full FFN changed input ignored");
                    }
                    previous = Some(observed[0][0].clone());
                    assert_eq!(input.read(stream), host);
                }
                println!(
                    "{}",
                    json!({"event":"full_ffn_qualification","weights":"synthetic quantized values at actual production dimensions", "down_format":format,
                    "rows":rows, "weight_set":set, "hidden":4096, "intermediate":12288,
                    "all_gate_activation_output_bits_equal":true, "changed_input_graph_replay":true,
                    "gate_and_activation_half_boundaries_unchanged":true})
                );
            }
        }
        for w in &downs {
            w.immutable(stream);
        }
    }
    for w in &gates {
        w.immutable(stream);
    }
}

struct Case {
    // Drop captured graphs before the allocations they retain by raw address.
    graphs: [CudaGraph; 2],
    input: Live<f16>,
    output: [Live<f16>; 2],
    expected: [Vec<u16>; 2],
    rows: usize,
    k: usize,
}
impl Case {
    fn new(
        stream: &Arc<CudaStream>,
        functions: [&CudaFunction; 2],
        weights: &[Weight],
        rows: usize,
    ) -> Self {
        let plan = weights[0].plan;
        let k = plan.k() as usize;
        let n = plan.n() as usize;
        let input = Live::new(stream, &values(rows * k, 0), f16::from_f32(-117.0), 64);
        let output = [
            Live::new(
                stream,
                &vec![f16::ZERO; rows * n * weights.len()],
                f16::from_f32(-119.0),
                64,
            ),
            Live::new(
                stream,
                &vec![f16::ZERO; rows * n * weights.len()],
                f16::from_f32(-119.0),
                64,
            ),
        ];
        let x = input.ptr(stream);
        let y = output.each_ref().map(|y| y.ptr(stream));
        let packets = weights
            .iter()
            .map(|w| w.packet.ptr(stream))
            .collect::<Vec<_>>();
        assert_eq!(x % 128, 0);
        assert!(y.iter().chain(&packets).all(|p| p % 128 == 0));
        let enqueue = |arm: usize| {
            for _ in 0..REPEATS {
                for (set, packet) in packets.iter().enumerate() {
                    Args {
                        input: x,
                        packet: *packet,
                        bytes: plan.packed_bytes(),
                        output: y[arm] + (set * rows * n * 2) as u64,
                        rows: rows as u32,
                        k: k as u32,
                        n: n as u32,
                        stride: n as u32,
                        offset: 0,
                        format: plan::format_code(plan.source_format()),
                        abi: plan.packing_abi(),
                    }
                    .launch(stream, functions[arm]);
                }
            }
        };
        let graphs = [
            capture(stream, || enqueue(0)),
            capture(stream, || enqueue(1)),
        ];
        Self {
            graphs,
            input,
            output,
            expected: [Vec::new(), Vec::new()],
            rows,
            k,
        }
    }
    fn qualify(&mut self, stream: &Arc<CudaStream>) {
        for generation in 0..2 {
            let input = values(self.rows * self.k, generation);
            self.input.replace(stream, &input);
            for arm in 0..2 {
                self.graphs[arm].launch().unwrap();
                let actual = bits(&self.output[arm].read(stream));
                if arm == 0 {
                    self.expected[generation] = actual.clone();
                }
                assert_eq!(
                    actual, self.expected[generation],
                    "full-size cross-arm bit mismatch"
                );
                self.graphs[arm].launch().unwrap();
                assert_eq!(
                    bits(&self.output[arm].read(stream)),
                    actual,
                    "full-size replay mismatch"
                );
            }
            assert_eq!(self.input.read(stream), input);
        }
        assert_ne!(
            self.expected[0], self.expected[1],
            "full-size input change ignored"
        );
    }
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite() && *v > 0.0));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}
fn time(stream: &Arc<CudaStream>, graph: &CudaGraph, count: usize) -> (f64, f64) {
    stream.synchronize().unwrap();
    let wall = Instant::now();
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    graph.launch().unwrap();
    let end = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    (
        f64::from(start.elapsed_ms(&end).unwrap()) * 1e6 / count as f64,
        wall.elapsed().as_secs_f64() * 1e9 / count as f64,
    )
}

#[test]
#[ignore = "exclusive SM80+ CUDA; direct production versus packet asynchronous prefetch; exclusive GPU"]
fn rn_q6_prefetch_production_pair_screen() {
    rn_q6_prefetch_embedded_async_instruction_exists();
    println!(
        "{}",
        json!({"event":"predeclared","schema":1,"candidate":"q6_only_packet_async_prefetch_ahead_one",
        "baseline_entry":ENTRY,"candidate_entries":ENTRIES,"physical_shapes":[["gate_up",24576,4096],["down",4096,12288]],
        "rows":ROWS,"formats":["Q4K","Q5K","Q6K"],"q5_timing_scope":"synthetic legal FFN shape control, not actual 9B FFN inventory","weight_sets":2,"repeats_per_set":REPEATS,
        "warmup_pairs":WARMUP,"measured_pairs":PAIRS,"order":"AB BA alternating",
        "q6_b8_ratio_limit":0.90,"q6_other_ratio_limit":1.05,"unchanged_q4_q5_ratio_limit":1.05,"cross_arm_output_contract":"all half bits identical",
        "arithmetic_layout_reduction_unchanged":true,"full_size_pointer_alignment":128,"prefetch_distance_k32_groups":8,"per_warp_slots":2,"slot_bytes":512,"additional_static_shared_bytes":8192,
        "weights":"two distinct full-size synthetic source sets, not model tensor values",
        "ptx_sha256":format!("{:x}",Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes())),
        "model_quality_passed":false,"serving_slo_measured":false,"candidate_product_selector_changed":true,"repository_product_default_changed":false})
    );
    let context = CudaContext::new(0).unwrap();
    assert!(context.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap() >= 8);
    let l2_bytes = context
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        .unwrap();
    let stream = context.new_stream().unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF))
        .unwrap();
    let baseline = module.load_function(ENTRY).unwrap();
    small(&stream, &module, &baseline);
    let mut passed = true;
    let coeff = module.load_function(ENTRIES[1]).unwrap();
    full_ffn(&stream, &context, &module, &baseline, &coeff);
    for (role, format, n, k) in [
        ("gate_up", RnF16FragmentSourceFormatV1::Q4K, 24576, 4096),
        ("down", RnF16FragmentSourceFormatV1::Q4K, 4096, 12288),
        ("down", RnF16FragmentSourceFormatV1::Q5K, 4096, 12288),
        ("down", RnF16FragmentSourceFormatV1::Q6K, 4096, 12288),
    ] {
        let candidate = module.load_function(entry(format)).unwrap();
        for (arm, function) in [&baseline, &candidate].into_iter().enumerate() {
            println!(
                "{}",
                json!({"event":"function_attributes","role":role,"n":n,"k":k,"format":format,"arm":arm,
                "registers_per_thread":function.num_regs().unwrap(),"local_bytes_per_thread":function.local_size_bytes().unwrap(),
                "static_shared_bytes":function.shared_size_bytes().unwrap(),
                "theoretical_active_blocks_per_sm":function.occupancy_max_active_blocks_per_multiprocessor(256,0,None).unwrap(),
                "achieved_occupancy_or_dram_measured":false})
            );
        }
        let plan = RnF16FragmentPlanV1::new(format, n, k).unwrap();
        let cold = Instant::now();
        let weights = [
            Weight::new(&stream, plan, 307, &coeff),
            Weight::new(&stream, plan, 1298, &coeff),
        ];
        stream.synchronize().unwrap();
        println!(
            "{}",
            json!({"event":"cold_setup","role":role,"n":n,"k":k,"format":format,"wall_ns":cold.elapsed().as_nanos(),
            "scope":"source generation + production packing + materializer/packed GPU coefficient bit validation + transfers + synchronization; not GPU event",
            "packet_bytes_per_set":plan.packed_bytes(),"combined_packet_bytes":plan.packed_bytes()*2,"device_l2_bytes":l2_bytes,"cache_residency_not_controlled_or_measured":true,"source_sha256":weights.iter().map(|w|&w.source_sha).collect::<Vec<_>>(),
            "packet_sha256":weights.iter().map(|w|&w.packet_sha).collect::<Vec<_>>()})
        );
        assert_ne!(weights[0].source_sha, weights[1].source_sha);
        let mut cases = ROWS.map(|rows| {
            let mut case = Case::new(&stream, [&baseline, &candidate], &weights, rows);
            case.qualify(&stream);
            case
        });
        println!(
            "{}",
            json!({"event":"timing_frozen","role":role,"n":n,"k":k,"format":format,"rows":ROWS,
            "all_full_size_output_bits_equal":true,"both_generations_qualified":true,
            "all_replay_bits_stable":true,"cold_and_validation_excluded_from_event":true})
        );
        for case in &mut cases {
            let mut samples = [Vec::new(), Vec::new()];
            let mut wall_samples = [Vec::new(), Vec::new()];
            let mut ratios = Vec::new();
            for pair in 0..WARMUP + PAIRS {
                let generation = pair % 2;
                let input = values(case.rows * case.k, generation);
                case.input.replace(&stream, &input);
                let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut pair_ns = [0.0; 2];
                for arm in order {
                    let (gpu, wall) = time(&stream, &case.graphs[arm], REPEATS * weights.len());
                    assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
                    assert_eq!(
                        bits(&case.output[arm].read(&stream)),
                        case.expected[generation]
                    );
                    pair_ns[arm] = gpu;
                    if pair >= WARMUP {
                        samples[arm].push(gpu);
                        wall_samples[arm].push(wall);
                    }
                    println!(
                        "{}",
                        json!({"event":"paired_sample","role":role,"n":n,"k":k,"format":format,"rows":case.rows,
                        "pair":pair,"warmup":pair<WARMUP,"generation":generation,"order":order,"arm":arm,
                        "gpu_ns_per_projection":gpu,"host_wall_ns_per_projection":wall,"all_output_bits_equal":true})
                    );
                }
                assert_eq!(case.input.read(&stream), input);
                if pair >= WARMUP {
                    ratios.push(pair_ns[1] / pair_ns[0]);
                }
            }
            let ratio = median(&ratios);
            let limit = if format == RnF16FragmentSourceFormatV1::Q6K && case.rows == 8 {
                0.90
            } else {
                1.05
            };
            passed &= ratio <= limit;
            println!(
                "{}",
                json!({"event":"cell_summary","role":role,"n":n,"k":k,"format":format,"rows":case.rows,
                "baseline_ns":samples[0],"candidate_ns":samples[1],"baseline_wall_ns":wall_samples[0],"candidate_wall_ns":wall_samples[1],
                "paired_ratios":ratios,"paired_median_ratio":ratio,"ratio_limit":limit,"screen_eligible":ratio<=limit,
                "baseline_median_ns":median(&samples[0]),"candidate_median_ns":median(&samples[1]),
                "six_pairs_do_not_establish_tail_latency":true})
            );
        }
        for weight in &weights {
            weight.immutable(&stream);
        }
    }
    println!(
        "{}",
        json!({"event":"screen_summary","operator_bitwise_qualified":true,"full_ffn_stages_bitwise_qualified":true,
        "all_cells_meet_predeclared_ratio":passed,"performance_failure_retained":true,
        "model_quality_passed":false,"serving_slo_measured":false,"candidate_product_selector_changed":true,"repository_product_default_changed":false})
    );
}
