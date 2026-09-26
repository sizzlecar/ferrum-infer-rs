//! Opt-in, bounded RN-F16 complete-FFN diagnostic. No product selection changes.
//! Discovery uses complete gate/up -> SiLU -> down graphs, then freezes both
//! choices before changing inputs and collecting independent paired samples.
use super::*;
use cudarc::cublaslt::{result as lt, sys as ls};
use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, DevicePtr};
use serde_json::{json, Value};
use std::{mem::MaybeUninit, sync::Arc};

const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12288;
const WORKSPACE: usize = 4 * 1024 * 1024;
const MAX_ALGORITHMS: usize = 8;
const WEIGHT_SETS: usize = 2;
const GRAPH_REPEATS: usize = 4;
const VALIDATION_PAIRS: usize = 6;
const DECODE_ROWS: [usize; 5] = [1, 2, 4, 7, 8];
const B8_RATIO_LIMIT: f64 = 0.90;
const OTHER_RATIO_LIMIT: f64 = 1.05;

#[derive(Clone, Copy, Debug)]
enum Layout {
    Nk,
    Kn,
}
impl Layout {
    fn index(self) -> usize {
        match self {
            Self::Nk => 0,
            Self::Kn => 1,
        }
    }
}

/// Keeps the existing RN operand types, compute enum and alpha/beta. Only the
/// physical weight layout and explicit vendor algorithm vary. Their reduction
/// trees may differ: operator qualification cannot replace teacher validation.
struct LtPlan {
    handle: ls::cublasLtHandle_t,
    desc: ls::cublasLtMatmulDesc_t,
    a: ls::cublasLtMatrixLayout_t,
    b: ls::cublasLtMatrixLayout_t,
    c: ls::cublasLtMatrixLayout_t,
    preference: ls::cublasLtMatmulPreference_t,
    layout: Layout,
    algorithms: Vec<ls::cublasLtMatmulHeuristicResult_t>,
}

impl LtPlan {
    fn new(rows: usize, outputs: usize, reduction: usize, layout: Layout) -> Self {
        // Empty handles make partial construction unwind safely.
        let mut plan = Self {
            handle: std::ptr::null_mut(),
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
            layout,
            algorithms: Vec::new(),
        };
        plan.handle = lt::create_handle().unwrap();
        plan.desc = lt::create_matmul_desc(
            ls::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
            ls::cudaDataType_t::CUDA_R_32F,
        )
        .unwrap();
        let (a_rows, a_cols, transpose) = match layout {
            Layout::Nk => (reduction, outputs, cublasOperation_t::CUBLAS_OP_T),
            Layout::Kn => (outputs, reduction, cublasOperation_t::CUBLAS_OP_N),
        };
        plan.a = lt::create_matrix_layout(
            ls::cudaDataType_t::CUDA_R_16F,
            a_rows as u64,
            a_cols as u64,
            a_rows as i64,
        )
        .unwrap();
        plan.b = lt::create_matrix_layout(
            ls::cudaDataType_t::CUDA_R_16F,
            reduction as u64,
            rows as u64,
            reduction as i64,
        )
        .unwrap();
        plan.c = lt::create_matrix_layout(
            ls::cudaDataType_t::CUDA_R_16F,
            outputs as u64,
            rows as u64,
            outputs as i64,
        )
        .unwrap();
        plan.preference = lt::create_matmul_pref().unwrap();
        let alignment = 16_u32;
        let mut raw =
            [MaybeUninit::<ls::cublasLtMatmulHeuristicResult_t>::uninit(); MAX_ALGORITHMS];
        let mut returned = 0_i32;
        // SAFETY: All descriptors are live, the preference matches the guarded
        // allocation alignment, and raw has exactly requestedAlgoCount slots.
        unsafe {
            lt::set_matmul_desc_attribute(
                plan.desc,
                ls::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&transpose as *const cublasOperation_t).cast(),
                std::mem::size_of_val(&transpose),
            )
            .unwrap();
            lt::set_matmul_pref_attribute(
                plan.preference,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&WORKSPACE as *const usize).cast(),
                std::mem::size_of::<usize>(),
            )
            .unwrap();
            for attr in [
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            ] {
                lt::set_matmul_pref_attribute(
                    plan.preference,
                    attr,
                    (&alignment as *const u32).cast(),
                    std::mem::size_of_val(&alignment),
                )
                .unwrap();
            }
            let status = ls::cublasLtMatmulAlgoGetHeuristic(
                plan.handle,
                plan.desc,
                plan.a,
                plan.b,
                plan.c,
                plan.c,
                plan.preference,
                MAX_ALGORITHMS as i32,
                raw.as_mut_ptr().cast(),
                &mut returned,
            );
            if status != ls::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                println!(
                    "{}",
                    json!({"event":"heuristic_unavailable", "rows":rows,
                    "outputs":outputs, "reduction":reduction, "layout":format!("{layout:?}"),
                    "status":format!("{status:?}")})
                );
                return plan;
            }
            assert!((0..=MAX_ALGORITHMS as i32).contains(&returned));
            for item in raw.into_iter().take(returned as usize) {
                let candidate = item.assume_init();
                if candidate.state != ls::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    continue;
                }
                let mut checked = MaybeUninit::uninit();
                let status = ls::cublasLtMatmulAlgoCheck(
                    plan.handle,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    plan.c,
                    &candidate.algo,
                    checked.as_mut_ptr(),
                );
                if status != ls::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    println!(
                        "{}",
                        json!({"event":"algorithm_rejected", "rows":rows,
                        "outputs":outputs, "reduction":reduction,
                        "layout":format!("{layout:?}"), "status":format!("{status:?}"),
                        "algo":candidate.algo.data})
                    );
                    continue;
                }
                let checked = checked.assume_init();
                if checked.state == ls::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                    && checked.workspaceSize <= WORKSPACE
                    && candidate.workspaceSize <= WORKSPACE
                {
                    plan.algorithms.push(candidate);
                }
            }
        }
        plan
    }

    fn launch(&self, index: usize, io: &ProjectionIo, stream: &CudaStream) {
        // SAFETY: Ffn owns aligned, offset buffers with full checked matrix
        // spans; their addresses stay fixed throughout capture and replay.
        unsafe {
            lt::matmul(
                self.handle,
                self.desc,
                (&CUDA_GEMM_ALPHA_F32 as *const f32).cast(),
                (&CUDA_GEMM_BETA_F32 as *const f32).cast(),
                io.weights[self.layout.index()] as *const c_void,
                self.a,
                io.input as *const c_void,
                self.b,
                io.output as *const c_void,
                self.c,
                io.output as *mut c_void,
                self.c,
                &self.algorithms[index].algo,
                io.workspace as *mut c_void,
                WORKSPACE,
                stream.cu_stream().cast(),
            )
            .unwrap();
        }
    }
}

impl Drop for LtPlan {
    fn drop(&mut self) {
        // SAFETY: The benchmark synchronizes and destroys graphs before plans.
        unsafe {
            if !self.preference.is_null() {
                let _ = lt::destroy_matmul_pref(self.preference);
            }
            for layout in [self.a, self.b, self.c] {
                if !layout.is_null() {
                    let _ = lt::destroy_matrix_layout(layout);
                }
            }
            if !self.desc.is_null() {
                let _ = lt::destroy_matmul_desc(self.desc);
            }
            if !self.handle.is_null() {
                let _ = lt::destroy_handle(self.handle);
            }
        }
    }
}

/// A full FFN matrix is already larger than L2 on the target device; rotate
/// distinct sets as well. The exact permutation is checked before GPU upload.
struct Weights {
    gate: [Guarded<f16>; 2],
    down: [Guarded<f16>; 2],
    salt: usize,
}
fn coefficient(n: usize, k: usize, salt: usize) -> f16 {
    let mut bits = (n as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (k as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ (salt as u64).wrapping_mul(0x94d049bb133111eb);
    bits ^= bits >> 31;
    let signed = ((bits >> 11) & 2047) as i32 - 1024;
    // Mixed signs, cancellation and several magnitudes without overflowing FFN.
    let divisor = [32768.0_f32, 65536.0, 131072.0, 1048576.0][bits as usize & 3];
    f16::from_f32(signed as f32 / divisor)
}
fn matrix_pair(
    stream: &Arc<CudaStream>,
    outputs: usize,
    reduction: usize,
    salt: usize,
) -> [Guarded<f16>; 2] {
    let mut nk = vec![f16::ZERO; outputs * reduction];
    let mut kn = nk.clone();
    // Tiled host traversal avoids turning the cold transpose into random writes.
    for n0 in (0..outputs).step_by(64) {
        for k0 in (0..reduction).step_by(64) {
            for n in n0..(n0 + 64).min(outputs) {
                for k in k0..(k0 + 64).min(reduction) {
                    let value = coefficient(n, k, salt);
                    nk[n * reduction + k] = value;
                    kn[k * outputs + n] = value;
                }
            }
        }
    }
    for n in 0..outputs {
        for k in 0..reduction {
            assert_eq!(
                nk[n * reduction + k].to_bits(),
                kn[k * outputs + n].to_bits()
            );
        }
    }
    let sentinel = f16::from_f32(-117.0);
    [
        Guarded::new(stream, &nk, sentinel),
        Guarded::new(stream, &kn, sentinel),
    ]
}

struct Frame {
    input: CudaSlice<f16>,
    host_input: Vec<f16>,
    gate: Guarded<f16>,
    activation: Guarded<f16>,
    output: Guarded<f16>,
}
impl Frame {
    fn new(stream: &Arc<CudaStream>, rows: usize, salt: usize) -> Self {
        let sentinel = f16::from_f32(-117.0);
        let host_input = Self::inputs(rows, salt, 0);
        Self {
            input: stream.clone_htod(&host_input).unwrap(),
            host_input,
            gate: Guarded::new(stream, &vec![f16::NAN; rows * INTERMEDIATE * 2], sentinel),
            activation: Guarded::new(stream, &vec![f16::NAN; rows * INTERMEDIATE], sentinel),
            output: Guarded::new(stream, &vec![f16::NAN; rows * HIDDEN], sentinel),
        }
    }
    fn inputs(rows: usize, salt: usize, revision: usize) -> Vec<f16> {
        let mut values = vec![f16::from_f32(-117.0); rows * HIDDEN + 16];
        for (i, value) in values[8..8 + rows * HIDDEN].iter_mut().enumerate() {
            *value = f16::from_f32(
                ((i * 13 + (i / HIDDEN) * 17 + salt * 7 + revision * 29) % 127) as f32 / 128.0
                    - 0.4921875,
            );
        }
        values
    }
    fn change_input(&mut self, stream: &Arc<CudaStream>, rows: usize, salt: usize) {
        self.host_input = Self::inputs(rows, salt, 1);
        stream
            .memcpy_htod(&self.host_input, &mut self.input)
            .unwrap();
    }
}

#[derive(Clone, Copy, Debug, Default)]
enum Choice {
    #[default]
    Baseline,
    Lt {
        plan: usize,
        algorithm: usize,
    },
}
#[derive(Clone, Copy, Debug, Default)]
struct Sequence {
    gate: Choice,
    down: Choice,
}
#[derive(Clone, Copy)]
struct ProjectionIo {
    input: u64,
    weights: [u64; 2],
    output: u64,
    workspace: u64,
}
struct Ffn<'a> {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    silu: CudaFunction,
    rows: usize,
    weights: &'a [Weights],
    frames: Vec<Frame>,
    io: Vec<[ProjectionIo; 2]>,
    gate_plans: Vec<LtPlan>,
    down_plans: Vec<LtPlan>,
    workspace: CudaSlice<u8>,
}
impl Ffn<'_> {
    fn launch_projection(&self, choice: Choice, gate: bool, io: ProjectionIo) {
        let (outputs, reduction, plans) = if gate {
            (2 * INTERMEDIATE, HIDDEN, &self.gate_plans)
        } else {
            (HIDDEN, INTERMEDIATE, &self.down_plans)
        };
        match choice {
            Choice::Baseline => {
                cublas_api::GemmF16ApiPlan::new(self.rows as i32, outputs as i32, reduction as i32)
                    .unwrap()
                    .launch(
                        &self.blas,
                        io.input,
                        io.weights[0],
                        io.output,
                        "FFN layout baseline",
                    )
                    .unwrap()
            }
            Choice::Lt { plan, algorithm } => plans[plan].launch(algorithm, &io, &self.stream),
        }
    }
    fn launch(&self, sequence: Sequence, set: usize) {
        let [gate, down] = self.io[set];
        self.launch_projection(sequence.gate, true, gate);
        launch_silu_mul(
            &self.stream,
            &self.silu,
            gate.output,
            down.input,
            INTERMEDIATE as i32,
            (self.rows * INTERMEDIATE) as u64,
        )
        .unwrap();
        self.launch_projection(sequence.down, false, down);
    }
    fn graph(&self, sequence: Sequence) -> CudaGraph {
        for set in 0..self.weights.len() {
            self.launch(sequence, set);
        }
        self.stream.synchronize().unwrap();
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        for _ in 0..GRAPH_REPEATS {
            for set in 0..self.weights.len() {
                self.launch(sequence, set);
            }
        }
        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .unwrap()
            .unwrap();
        graph.launch().unwrap();
        self.stream.synchronize().unwrap();
        graph
    }
    fn time(&self, graph: &CudaGraph) -> f64 {
        let start = self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        graph.launch().unwrap();
        let end = self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        end.synchronize().unwrap();
        f64::from(start.elapsed_ms(&end).unwrap()) * 1e6
            / (GRAPH_REPEATS * self.weights.len()) as f64
    }
    fn describe(&self, sequence: Sequence) -> Value {
        let describe = |choice, plans: &[LtPlan]| match choice {
            Choice::Baseline => json!({"api":"original_gemm_ex"}),
            Choice::Lt { plan, algorithm } => json!({
                "api":"cublasLtMatmul", "layout":format!("{:?}", plans[plan].layout),
                "heuristic_index":algorithm, "algo":plans[plan].algorithms[algorithm].algo.data,
                "workspace_bytes":plans[plan].algorithms[algorithm].workspaceSize,
            }),
        };
        json!({"gate":describe(sequence.gate, &self.gate_plans),
            "down":describe(sequence.down, &self.down_plans)})
    }
    fn validate(&self) -> Vec<Vec<u16>> {
        let mut outputs = Vec::new();
        for (frame, weights) in self.frames.iter().zip(self.weights) {
            let gates = frame.gate.read(&self.stream);
            let acts = frame.activation.read(&self.stream);
            let out = frame.output.read(&self.stream);
            sample_projection(
                &frame.host_input[8..8 + self.rows * HIDDEN],
                &gates,
                HIDDEN,
                2 * INTERMEDIATE,
                weights.salt,
            );
            for (row, pair) in gates.chunks_exact(2 * INTERMEDIATE).enumerate() {
                for column in 0..INTERMEDIATE {
                    let g = pair[column].to_f64();
                    activation(
                        acts[row * INTERMEDIATE + column],
                        g / (1.0 + (-g).exp()) * pair[INTERMEDIATE + column].to_f64(),
                    );
                }
            }
            sample_projection(&acts, &out, INTERMEDIATE, HIDDEN, weights.salt + 101);
            assert!(gates.iter().chain(&acts).chain(&out).all(|v| v.is_finite()));
            assert!(out
                .chunks_exact(HIDDEN)
                .all(|row| row.iter().any(|v| *v != f16::ZERO)));
            assert_eq!(
                self.stream.clone_dtoh(&frame.input).unwrap(),
                frame.host_input
            );
            outputs.push(out.iter().map(|v| v.to_bits()).collect());
        }
        let workspace = self.stream.clone_dtoh(&self.workspace).unwrap();
        assert!(workspace[..256].iter().all(|&v| v == 0xa5));
        assert!(workspace[256 + WORKSPACE..].iter().all(|&v| v == 0xa5));
        outputs
    }
}

fn sample_projection(input: &[f16], actual: &[f16], reduction: usize, outputs: usize, salt: usize) {
    // Independent F64 dot oracle at both row edges, internal tile edges and
    // both halves of the gate/up matrix. All output/activation values are also
    // checked for finiteness; this is deliberately not full model quality.
    for (row, x) in input.chunks_exact(reduction).enumerate() {
        for n in [0, 1, 15, 16, 127, outputs / 2 - 1, outputs / 2, outputs - 1] {
            let (mut sum, mut magnitude) = (0.0, 0.0);
            for (k, x) in x.iter().enumerate() {
                let product = x.to_f64() * coefficient(n, k, salt).to_f64();
                sum += product;
                magnitude += product.abs();
            }
            let bound = (reduction as f64 * f64::from(f32::EPSILON) + 0.0009765625) * magnitude
                + f16::from_bits(1).to_f64();
            let value = actual[row * outputs + n].to_f64();
            assert!(
                (value - sum).abs() <= bound,
                "F64 oracle row={row} n={n}: actual={value}, expected={sum}, bound={bound}"
            );
        }
    }
}
fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    assert!(sorted.iter().all(|value| value.is_finite() && *value > 0.0));
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}

fn discover(ffn: &Ffn<'_>) -> Sequence {
    // Bounded coordinate search: first gate/up with original down, then down
    // with the selected gate/up. Every measurement includes the entire FFN.
    // The result is local to this geometry; no global optimum is claimed.
    let mut selected = Sequence::default();
    for gate in [true, false] {
        let reference = ffn.graph(selected);
        let plans = if gate {
            &ffn.gate_plans
        } else {
            &ffn.down_plans
        };
        let mut best_ratio = 1.0;
        let mut next = selected;
        for (plan, descriptor) in plans.iter().enumerate() {
            for algorithm in 0..descriptor.algorithms.len() {
                let choice = Choice::Lt { plan, algorithm };
                let candidate = if gate {
                    Sequence {
                        gate: choice,
                        ..selected
                    }
                } else {
                    Sequence {
                        down: choice,
                        ..selected
                    }
                };
                let graph = ffn.graph(candidate);
                ffn.validate();
                let mut pairs = Vec::new();
                for pair in 0..3 {
                    let (base_ns, candidate_ns) = if pair % 2 == 0 {
                        (ffn.time(&reference), ffn.time(&graph))
                    } else {
                        let candidate_ns = ffn.time(&graph);
                        (ffn.time(&reference), candidate_ns)
                    };
                    pairs.push(candidate_ns / base_ns);
                    println!(
                        "{}",
                        json!({"event":"discovery_pair", "rows":ffn.rows,
                        "stage":if gate {"gate"} else {"down"}, "pair":pair,
                        "reference_ns":base_ns, "candidate_ns":candidate_ns,
                        "reference":ffn.describe(selected),
                        "candidate":ffn.describe(candidate)})
                    );
                }
                let ratio = median(&pairs);
                if ratio < best_ratio {
                    best_ratio = ratio;
                    next = candidate;
                }
            }
        }
        selected = next;
    }
    selected
}

#[test]
#[ignore = "requires exclusive CUDA, about 1.3 GiB device memory and 2 GiB host memory; diagnostic only"]
fn rn_f16_complete_ffn_cublaslt_layout_screen() {
    println!(
        "{}",
        json!({"event":"predeclared", "benchmark":"rn_f16_complete_ffn_lt_layout_v1",
        "hidden":HIDDEN, "intermediate":INTERMEDIATE, "rows":DECODE_ROWS,
        "compute":"CUBLAS_COMPUTE_32F_FAST_16F", "operands":"F16", "output":"F16",
        "workspace_limit_bytes":WORKSPACE, "max_heuristics_per_layout":MAX_ALGORITHMS,
        "weight_sets":WEIGHT_SETS, "ffns_per_graph":GRAPH_REPEATS * WEIGHT_SETS,
        "validation_pairs":VALIDATION_PAIRS, "b8_paired_median_ratio_limit":B8_RATIO_LIMIT,
        "other_paired_median_ratio_limit":OTHER_RATIO_LIMIT,
        "selection":"complete_ffn_coordinate_search_then_changed_input_validation",
        "prefill_tested":false, "model_quality_passed":false, "release_approved":false})
    );
    let context = CudaContext::new(0).expect("complete FFN screen requires CUDA");
    // SAFETY: Version query has no pointer arguments or mutable handle state.
    let lt_version = unsafe { ls::cublasLtGetVersion() };
    println!(
        "{}",
        json!({"event":"library_identity", "cublaslt_version":lt_version})
    );
    let stream = context.new_stream().unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    let weights = (0..WEIGHT_SETS)
        .map(|set| {
            let salt = 3 + set * 19;
            Weights {
                gate: matrix_pair(&stream, 2 * INTERMEDIATE, HIDDEN, salt),
                down: matrix_pair(&stream, HIDDEN, INTERMEDIATE, salt + 101),
                salt,
            }
        })
        .collect::<Vec<_>>();
    let mut summaries = Vec::new();
    for rows in DECODE_ROWS {
        let frames = (0..WEIGHT_SETS)
            .map(|set| Frame::new(&stream, rows, set))
            .collect::<Vec<_>>();
        let workspace = stream.clone_htod(&vec![0xa5_u8; WORKSPACE + 512]).unwrap();
        let workspace_ptr = workspace.device_ptr(&stream).0 + 256;
        assert_eq!(workspace_ptr % 256, 0);
        // Resolve pointers once outside capture. Calling DevicePtr within each
        // captured launch would inject cudarc tracking events into this benchmark.
        // All buffers stay alive on this one stream until graphs are destroyed.
        let io = frames
            .iter()
            .zip(&weights)
            .map(|(frame, weight)| {
                [
                    ProjectionIo {
                        input: frame.input.device_ptr(&stream).0 + 16,
                        weights: weight.gate.each_ref().map(|w| w.pointer(&stream)),
                        output: frame.gate.pointer(&stream),
                        workspace: workspace_ptr,
                    },
                    ProjectionIo {
                        input: frame.activation.pointer(&stream),
                        weights: weight.down.each_ref().map(|w| w.pointer(&stream)),
                        output: frame.output.pointer(&stream),
                        workspace: workspace_ptr,
                    },
                ]
            })
            .collect();
        let mut ffn = Ffn {
            stream: stream.clone(),
            blas: CudaBlas::new(stream.clone()).unwrap(),
            silu: silu.clone(),
            rows,
            weights: &weights,
            frames,
            io,
            gate_plans: [Layout::Nk, Layout::Kn]
                .map(|layout| LtPlan::new(rows, 2 * INTERMEDIATE, HIDDEN, layout))
                .into(),
            down_plans: [Layout::Nk, Layout::Kn]
                .map(|layout| LtPlan::new(rows, HIDDEN, INTERMEDIATE, layout))
                .into(),
            workspace,
        };
        let selected = discover(&ffn);
        let baseline = ffn.graph(Sequence::default());
        let candidate = ffn.graph(selected);
        let old_output = ffn.validate();
        for (set, frame) in ffn.frames.iter_mut().enumerate() {
            frame.change_input(&stream, rows, set);
        }
        for set in 0..WEIGHT_SETS {
            ffn.launch(selected, set);
        }
        let changed_eager = ffn.validate();
        candidate.launch().unwrap();
        let changed_replay = ffn.validate();
        assert_eq!(
            changed_eager, changed_replay,
            "changed input graph differs from eager"
        );
        assert_ne!(
            old_output, changed_replay,
            "graph did not consume changed input"
        );
        baseline.launch().unwrap();
        ffn.validate();
        println!(
            "{}",
            json!({"event":"frozen_selection", "rows":rows,
            "selection":ffn.describe(selected), "changed_input_eager_replay_bitwise":true})
        );
        let mut ratios = Vec::new();
        let mut baseline_ns = Vec::new();
        let mut candidate_ns = Vec::new();
        for pair in 0..VALIDATION_PAIRS {
            let (a, b) = if pair % 2 == 0 {
                (ffn.time(&baseline), ffn.time(&candidate))
            } else {
                let b = ffn.time(&candidate);
                (ffn.time(&baseline), b)
            };
            ratios.push(b / a);
            baseline_ns.push(a);
            candidate_ns.push(b);
            println!(
                "{}",
                json!({"event":"validation_pair", "rows":rows, "pair":pair,
                "order":if pair % 2 == 0 {"AB"} else {"BA"},
                "baseline_ns":a, "candidate_ns":b, "paired_ratio":b/a})
            );
        }
        let ratio = median(&ratios);
        let eligible = ratio
            <= if rows == 8 {
                B8_RATIO_LIMIT
            } else {
                OTHER_RATIO_LIMIT
            };
        let summary = json!({"rows":rows, "baseline_median_ns":median(&baseline_ns),
            "candidate_median_ns":median(&candidate_ns), "paired_median_ratio":ratio,
            "paired_ratio_samples":ratios, "screen_eligible":eligible,
            "selection":ffn.describe(selected)});
        println!("{}", json!({"event":"geometry_summary", "result":summary}));
        summaries.push(summary);
        candidate.launch().unwrap();
        ffn.validate();
        stream.synchronize().unwrap();
    }
    for set in &weights {
        for weight in set.gate.iter().chain(&set.down) {
            weight.assert_unchanged(&stream);
        }
    }
    println!(
        "{}",
        json!({"event":"screen_summary", "geometries":summaries,
        "screen_eligible":summaries.iter().all(|s| s["screen_eligible"] == true),
        "weights_unchanged":true, "model_quality_passed":false, "release_approved":false,
        "limitations":"synthetic operator screen; no prefill, teacher, serving or Enforce qualification"})
    );
    // Performance is evidence, never a cargo-test pass/fail oracle. Numerical,
    // memory-boundary and graph correctness failures above remain real failures.
}
