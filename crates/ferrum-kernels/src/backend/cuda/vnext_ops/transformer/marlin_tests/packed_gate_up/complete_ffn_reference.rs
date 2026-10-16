//! Synthetic complete FFN qualification; neither timing nor a serving claim.
//! References decode the original GPTQ/scales/CT-zero arrays, never Marlin bytes.
use super::*;
use crate::backend::cuda::vllm_marlin::{
    launch_marlin_mm_f16_weight, MarlinMmBuffers, MarlinMmExecution, MarlinMmF16WeightRequest,
    MarlinMmProblem,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use std::ffi::c_void;

mod baseline_reference_audit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightReference {
    ExactScaleProduct,
    RoundedF16Weight,
}

// Freeze from the native arithmetic review before running, never from results.
const PRIMARY_REFERENCE: WeightReference = WeightReference::RoundedF16Weight;
const RELATIVE_LIMIT: f64 = 1e-3;
const DEVICE_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

struct CompleteArm {
    gate_up: Arm,
    down: Projection,
    fp32_reduction: Option<Fp32ReductionScratch>,
}

struct Fp32ReductionScratch {
    buffers: Vec<Guarded<f32>>,
    pointers: Vec<u64>,
    bytes_per_projection: usize,
}

impl Fp32ReductionScratch {
    fn elements(rows: usize, sms: i32) -> usize {
        // Pinned marlin.cu:684-691 / marlin.cuh max_thread_n=256.
        // C_tmp is addressed as int4; Guarded<f32>'s 32-byte prefix preserves it.
        assert!(rows > 0 && sms > 0);
        (sms as usize)
            .checked_mul(rows.div_ceil(16).checked_mul(16).unwrap().min(64))
            .and_then(|n| n.checked_mul(256))
            .unwrap()
    }

    fn new(stream: &Arc<CudaStream>, rows: usize, sms: i32, projections: usize) -> Self {
        let elements = Self::elements(rows, sms);
        let buffers: Vec<_> = (0..projections)
            .map(|_| Guarded::new(stream, &vec![f32::NAN; elements], -117.0_f32))
            .collect();
        let pointers: Vec<_> = buffers
            .iter()
            .map(|buffer| buffer.pointer(stream))
            .collect();
        assert!(pointers.iter().all(|pointer| pointer % 16 == 0));
        Self {
            buffers,
            pointers,
            bytes_per_projection: elements.checked_mul(4).unwrap(),
        }
    }
}

fn launch_fp32_reduction_projection(
    projection: &Projection,
    runtime: MarlinProjectionRuntime,
    stream: &Arc<CudaStream>,
    input: u64,
    f: Fixture,
    c_tmp: u64,
) {
    // Same typed request and full lock reset as MarlinProjectionRuntime::launch;
    // only C_tmp and USE_FP32_REDUCE differ. This is test-only arithmetic policy.
    unsafe {
        cudarc::driver::result::memset_d8_async(
            projection.pointers[4],
            0,
            runtime.workspace_bytes().unwrap() as usize,
            stream.cu_stream(),
        )
        .unwrap();
        launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
            weight_type: MarlinF16WeightType::U4,
            buffers: MarlinMmBuffers {
                a: input as *const c_void,
                b: projection.pointers[0] as *const c_void,
                c: projection.pointers[3] as *mut c_void,
                c_tmp: c_tmp as *mut c_void,
                a_scales: std::ptr::null_mut(),
                b_scales: projection.pointers[1] as *mut c_void,
                zero_points: projection.pointers[2] as *mut c_void,
                group_index: std::ptr::null_mut(),
                permutation: std::ptr::null_mut(),
                a_tmp: std::ptr::null_mut(),
                workspace: projection.pointers[4] as *mut c_void,
            },
            problem: MarlinMmProblem {
                m: f.rows as i32,
                n: projection.columns as i32,
                k: f.input_features as i32,
                lda: f.input_features as i32,
                num_groups: (f.input_features / GROUP_SIZE) as i32,
                group_size: GROUP_SIZE as i32,
            },
            execution: MarlinMmExecution {
                device: runtime.device_ordinal,
                stream: stream.cu_stream(),
                sms: runtime.multiprocessor_count,
                has_act_order: false,
                is_k_full: true,
                use_atomic_add: false,
                use_fp32_reduce: true,
            },
        });
    }
}

#[derive(Clone)]
struct Values {
    projection: Vec<f16>,
    activation: Vec<f16>,
    output: Vec<f16>,
}

impl CompleteArm {
    fn enqueue(
        &self,
        runtime: MarlinProjectionRuntime,
        stream: &Arc<CudaStream>,
        gate: Fixture,
        down: Fixture,
        planar: &CudaFunction,
        merged: &CudaFunction,
    ) {
        if let Some(scratch) = &self.fp32_reduction {
            for (index, projection) in self.gate_up.projections.iter().enumerate() {
                launch_fp32_reduction_projection(
                    projection,
                    runtime,
                    stream,
                    self.gate_up.input_pointer,
                    gate,
                    scratch.pointers[index],
                );
            }
            let intermediate = gate.output_features / 2;
            if self.gate_up.projections.len() == 2 {
                launch_planar_silu_mul(
                    stream,
                    planar,
                    self.gate_up.projections[0].pointers[3],
                    self.gate_up.projections[1].pointers[3],
                    self.gate_up.activation_pointer,
                    (gate.rows * intermediate) as u64,
                )
                .unwrap();
            } else {
                launch_silu_mul(
                    stream,
                    merged,
                    self.gate_up.projections[0].pointers[3],
                    self.gate_up.activation_pointer,
                    intermediate as i32,
                    (gate.rows * intermediate) as u64,
                )
                .unwrap();
            }
            launch_fp32_reduction_projection(
                &self.down,
                runtime,
                stream,
                self.gate_up.activation_pointer,
                down,
                scratch.pointers[self.gate_up.projections.len()],
            );
            return;
        }
        self.gate_up.enqueue(runtime, stream, gate, planar, merged);
        self.down
            .launch(runtime, stream, self.gate_up.activation_pointer, down);
    }

    fn refill(&self, stream: &Arc<CudaStream>, input: &[f16], gate: Fixture) {
        self.gate_up.prepare_replay_input(stream, input, gate);
        unsafe {
            if let Some(scratch) = &self.fp32_reduction {
                // Native first contributor must overwrite, not consume stale partials.
                // Poison between eager and graph and between input generations.
                for &pointer in &scratch.pointers {
                    cudarc::driver::result::memset_d8_async(
                        pointer,
                        0xff,
                        scratch.bytes_per_projection,
                        stream.cu_stream(),
                    )
                    .unwrap();
                }
            }
            cudarc::driver::result::memset_d8_async(
                self.down.pointers[3],
                0xff,
                gate.rows * self.down.columns * 2,
                stream.cu_stream(),
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
    }

    fn read(&self, stream: &Arc<CudaStream>, gate: Fixture) -> Values {
        Values {
            projection: self.gate_up.projection_values(stream, gate),
            activation: self.gate_up.activated.read(stream),
            output: self.down.output.read(stream),
        }
    }

    fn guards(&self, stream: &Arc<CudaStream>) {
        for projection in self.gate_up.projections.iter().chain([&self.down]) {
            projection.weights.assert_unchanged(stream);
            projection.scales.assert_unchanged(stream);
            projection.zeros.assert_unchanged(stream);
            projection.output.read(stream);
        }
        self.gate_up.workspace.read(stream);
        self.gate_up.input.read(stream);
        if let Some(scratch) = &self.fp32_reduction {
            for buffer in &scratch.buffers {
                buffer.read(stream);
            }
        }
    }
}

fn same_bits(a: &[f16], b: &[f16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn observed_difference(a: &[f16], b: &[f16]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    let different_bits = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let nonfinite_a = a.iter().filter(|v| !v.is_finite()).count();
    let nonfinite_b = b.iter().filter(|v| !v.is_finite()).count();
    let max_finite_absolute = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a.to_f64() - b.to_f64()).abs())
        .fold(0.0_f64, f64::max);
    json!({"elements":a.len(),"different_bits":different_bits,"exact_bits":different_bits==0,"nonfinite_a":nonfinite_a,"nonfinite_b":nonfinite_b,"max_finite_absolute":max_finite_absolute})
}

fn dequantize(f: Fixture, host: &HostFixture, rule: WeightReference) -> Vec<f64> {
    let k = f.input_features;
    let n = f.output_features;
    let groups = k / GROUP_SIZE;
    (0..k * n)
        .map(|index| {
            let (i, col) = (index / n, index % n);
            let q = ((host.qweight_gptq[(i / 8) * n + col] as u32 >> (4 * (i % 8))) & 15) as i32;
            let z = ((host.zero_points_compressed_tensors[(col / 8) * groups + i / GROUP_SIZE]
                as u32
                >> (4 * (col % 8)))
                & 15) as i32;
            let scale = host.scales_grouped[(i / GROUP_SIZE) * n + col];
            match rule {
                WeightReference::ExactScaleProduct => f64::from(q - z) * scale.to_f64(),
                WeightReference::RoundedF16Weight => {
                    f16::from_f32((q - z) as f32 * scale.to_f32()).to_f64()
                }
            }
        })
        .collect()
}

/// Row-major X[M,K] * W[K,N], using only cublasDgemm through Gemm<f64>.
/// All three F64 device allocations die here before the next stage is allocated.
fn reference_projection(
    stream: &Arc<CudaStream>,
    blas: &CudaBlas,
    f: Fixture,
    host: &HostFixture,
    input: &[f16],
    rule: WeightReference,
    cpu_crosscheck: bool,
) -> Vec<f64> {
    assert_eq!(input.len(), f.rows * f.input_features);
    let weights = dequantize(f, host, rule);
    let input64: Vec<_> = input.iter().map(|v| v.to_f64()).collect();
    let device_weights = stream.clone_htod(&weights).unwrap();
    let device_input = stream.clone_htod(&input64).unwrap();
    let mut device_output = stream
        .alloc_zeros::<f64>(f.rows * f.output_features)
        .unwrap();
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: f.output_features as i32,
        n: f.rows as i32,
        k: f.input_features as i32,
        alpha: 1.0_f64,
        lda: f.output_features as i32,
        ldb: f.input_features as i32,
        beta: 0.0_f64,
        ldc: f.output_features as i32,
    };
    unsafe { blas.gemm(cfg, &device_weights, &device_input, &mut device_output) }.unwrap();
    let output = stream.clone_dtoh(&device_output).unwrap();
    stream.synchronize().unwrap();
    assert!(
        output.iter().all(|v| v.is_finite()),
        "nonfinite F64 reference"
    );
    if cpu_crosscheck {
        let mut max_absolute = 0.0_f64;
        for row in 0..f.rows {
            for col in 0..f.output_features {
                let expected: f64 = (0..f.input_features)
                    .map(|i| {
                        input64[row * f.input_features + i] * weights[i * f.output_features + col]
                    })
                    .sum();
                let delta = (output[row * f.output_features + col] - expected).abs();
                max_absolute = max_absolute.max(delta);
                // Only crosschecks two F64 implementations, not product tolerance.
                assert!(
                    delta <= 1e-10 * (1.0 + expected.abs()),
                    "F64 GEMM orientation/reference mismatch"
                );
            }
        }
        println!(
            "{}",
            json!({"kind":"complete_ffn_f64_cpu_crosscheck","name":f.name,"weight_reference":format!("{rule:?}"),"elements":output.len(),"max_absolute":max_absolute,"all_elements_checked":true})
        );
    }
    output
}

fn reference_activation(projection: &[f64], rows: usize, intermediate: usize) -> Vec<f16> {
    (0..rows * intermediate)
        .map(|index| {
            let (row, col) = (index / intermediate, index % intermediate);
            let gate = f16::from_f64(projection[row * 2 * intermediate + col]).to_f32();
            let up =
                f16::from_f64(projection[row * 2 * intermediate + intermediate + col]).to_f32();
            let silu = gate / (1.0_f32 + (-gate).exp());
            f16::from_f32(silu * up)
        })
        .collect()
}

fn element_errors(actual: &[f16], expected: &[f64]) -> (serde_json::Value, bool) {
    assert_eq!(actual.len(), expected.len());
    let (mut passed, mut failed, mut nonfinite, mut zero_reference_failures) = (0, 0, 0, 0);
    let (mut max_absolute, mut max_relative_nonzero) = (0.0_f64, 0.0_f64);
    let mut worst_absolute_index = 0;
    let mut worst_relative_index = 0;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let value = actual.to_f64();
        if !value.is_finite() || !expected.is_finite() {
            nonfinite += 1;
            failed += 1;
            continue;
        }
        let absolute = (value - expected).abs();
        if absolute > max_absolute {
            max_absolute = absolute;
            worst_absolute_index = index;
        }
        if *expected != 0.0 {
            let relative = absolute / expected.abs();
            if relative > max_relative_nonzero {
                max_relative_nonzero = relative;
                worst_relative_index = index;
            }
        } else if absolute != 0.0 {
            zero_reference_failures += 1;
        }
        if absolute <= RELATIVE_LIMIT * expected.abs() {
            passed += 1;
        } else {
            failed += 1;
        }
    }
    (
        json!({"elements":actual.len(),"relative_limit":RELATIVE_LIMIT,"passed":passed,"failed":failed,"nonfinite":nonfinite,"zero_reference_failures":zero_reference_failures,"zero_rule":"reference zero requires exact actual zero; no absolute floor","max_absolute":max_absolute,"worst_absolute_index":worst_absolute_index,"max_relative_nonzero":max_relative_nonzero,"worst_relative_index":worst_relative_index,"all_finite":nonfinite==0}),
        failed == 0,
    )
}

fn half_values(values: &[f16]) -> Vec<f64> {
    values.iter().map(|v| v.to_f64()).collect()
}

fn projection_halves<T: Copy>(values: &[T], intermediate: usize) -> [Vec<T>; 2] {
    assert_eq!(values.len() % (2 * intermediate), 0);
    std::array::from_fn(|half| {
        values
            .chunks_exact(2 * intermediate)
            .flat_map(|row| {
                row[half * intermediate..(half + 1) * intermediate]
                    .iter()
                    .copied()
            })
            .collect()
    })
}

fn qualify_case(
    context: &Arc<CudaContext>,
    gate: Fixture,
    output_features: usize,
    small: bool,
    primary: WeightReference,
    use_fp32_reduce: bool,
) -> bool {
    let stream = context.new_stream().unwrap();
    let blas = CudaBlas::new(Arc::clone(&stream)).unwrap();
    let runtime = MarlinProjectionRuntime {
        multiprocessor_count: context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap(),
        device_ordinal: 0,
    };
    let down = Fixture {
        name: gate.name,
        rows: gate.rows,
        input_features: gate.output_features / 2,
        output_features,
    };
    let gate_host = source(gate);
    let down_host = build_host_fixture(down);
    let gate_weights = repack_source(gate, &gate_host);
    let down_weights = Repacked::new(
        &down_host.qweight_gptq,
        &down_host.scales_grouped,
        &down_host.zero_points_compressed_tensors,
        down.input_features,
        down.output_features,
    );
    // Largest one-stage F64 allocation plus both independent device arms and
    // conservative guard/output overhead. cuBLAS/context overhead is not timed.
    let reference_peak = [gate, down]
        .iter()
        .map(|f| {
            8 * (f.input_features * f.output_features
                + f.rows * (f.input_features + f.output_features))
        })
        .max()
        .unwrap();
    let arms_bytes = 2
        * (gate_weights[2].bytes()
            + down_weights.bytes()
            + 2 * gate.rows
                * (gate.input_features
                    + gate.output_features
                    + down.input_features
                    + down.output_features)
            + runtime.workspace_bytes().unwrap() as usize);
    let fp32_scratch_peak = if use_fp32_reduce {
        // At most three independent projection scratch buffers per arm.
        2 * 3 * Fp32ReductionScratch::elements(gate.rows, runtime.multiprocessor_count) * 4
    } else {
        0
    };
    let owned_peak_bound = reference_peak + arms_bytes + fp32_scratch_peak + 1024 * 1024;
    assert!(
        owned_peak_bound < DEVICE_BUDGET_BYTES - 256 * 1024 * 1024,
        "fixture device budget"
    );
    let arms: [CompleteArm; 2] = std::array::from_fn(|arm| {
        let gate_up = Arm::new(
            &stream,
            runtime,
            gate,
            &gate_host.input,
            &gate_weights,
            arm == 1,
        );
        let down_projection = Projection::new(
            &stream,
            &down_weights,
            down.rows,
            down.output_features,
            gate_up.projections[0].pointers[4],
        );
        let fp32_reduction = use_fp32_reduce.then(|| {
            Fp32ReductionScratch::new(
                &stream,
                gate.rows,
                runtime.multiprocessor_count,
                gate_up.projections.len() + 1,
            )
        });
        CompleteArm {
            gate_up,
            down: down_projection,
            fp32_reduction,
        }
    });
    let _drain = Drain(Arc::clone(&stream));
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let planar = module.load_function(PLANAR_SILU_MUL_FUNCTION_NAME).unwrap();
    let merged = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    for arm in &arms {
        arm.enqueue(runtime, &stream, gate, down, &planar, &merged);
    }
    stream.synchronize().unwrap();
    let graphs: [Graph; 2] = std::array::from_fn(|arm| {
        Graph::new(&stream, || {
            arms[arm].enqueue(runtime, &stream, gate, down, &planar, &merged)
        })
    });
    println!(
        "{}",
        json!({"kind":"complete_ffn_reference_setup","name":gate.name,"synthetic_geometry":[gate.rows,gate.input_features,down.input_features,down.output_features],"primary_reference":format!("{primary:?}"),"owned_device_peak_bound_bytes":owned_peak_bound,"fp32_scratch_upper_bound_bytes":fp32_scratch_peak,"device_budget_bytes":DEVICE_BUDGET_BYTES,"reference_device_allocations":"one F64 projection stage at a time; synchronized before drop","graph_node_types_A":graphs[0].inventory,"graph_node_types_B":graphs[1].inventory,"atomic_add":false,"fp32_reduce":use_fp32_reduce,"use_fp32_reduce":use_fp32_reduce,"performance_claim":false})
    );
    let mut qualification = true;
    let mut previous: Option<[Values; 2]> = None;
    for generation in 0..2 {
        let input: Vec<_> = gate_host
            .input
            .iter()
            .map(|v| f16::from_f32(v.to_f32() * if generation == 0 { 1.0 } else { 0.5 }))
            .collect();
        let eager: [Values; 2] = std::array::from_fn(|arm| {
            arms[arm].refill(&stream, &input, gate);
            arms[arm].enqueue(runtime, &stream, gate, down, &planar, &merged);
            let values = arms[arm].read(&stream, gate);
            if use_fp32_reduce {
                arms[arm].guards(&stream);
            }
            assert!(
                same_bits(&arms[arm].gate_up.input.read(&stream), &input),
                "eager mutated its input"
            );
            values
        });
        for arm in &arms {
            arm.refill(&stream, &input, gate);
        }
        for graph in &graphs {
            graph.launch();
        }
        stream.synchronize().unwrap();
        let actual: [Values; 2] = std::array::from_fn(|arm| arms[arm].read(&stream, gate));
        for arm in 0..2 {
            assert!(
                same_bits(&actual[arm].projection, &eager[arm].projection),
                "projection eager/graph mismatch"
            );
            assert!(
                same_bits(&actual[arm].activation, &eager[arm].activation),
                "SiLU eager/graph mismatch"
            );
            assert!(
                same_bits(&actual[arm].output, &eager[arm].output),
                "down eager/graph mismatch"
            );
            arms[arm].guards(&stream);
            assert!(
                same_bits(&arms[arm].gate_up.input.read(&stream), &input),
                "graph mutated its input"
            );
            if let Some(previous) = &previous {
                assert!(!same_bits(
                    &actual[arm].projection,
                    &previous[arm].projection
                ));
                assert!(!same_bits(
                    &actual[arm].activation,
                    &previous[arm].activation
                ));
                assert!(!same_bits(&actual[arm].output, &previous[arm].output));
            }
        }
        println!(
            "{}",
            json!({"kind":"complete_ffn_ab_bits","name":gate.name,"generation":generation,"projection":observed_difference(&actual[0].projection,&actual[1].projection),"activation":observed_difference(&actual[0].activation,&actual[1].activation),"output":observed_difference(&actual[0].output,&actual[1].output),"matched_eager_graph_all_bits":true,"fresh_input_observable":generation==1,"ab_bit_equality_is_qualification_gate":false})
        );
        for rule in [
            WeightReference::ExactScaleProduct,
            WeightReference::RoundedF16Weight,
        ] {
            let reference =
                reference_projection(&stream, &blas, gate, &gate_host, &input, rule, small);
            let activation = reference_activation(&reference, gate.rows, down.input_features);
            assert!(activation.iter().all(|v| v.is_finite()));
            let output =
                reference_projection(&stream, &blas, down, &down_host, &activation, rule, small);
            let activation64 = half_values(&activation);
            let rounded_projection: Vec<_> = reference
                .iter()
                .map(|v| f16::from_f64(*v).to_f64())
                .collect();
            let rounded_output: Vec<_> =
                output.iter().map(|v| f16::from_f64(*v).to_f64()).collect();
            let reference_halves = projection_halves(&rounded_projection, down.input_features);
            for arm in 0..2 {
                let (projection_error, _) = element_errors(&actual[arm].projection, &reference);
                let (activation_error, activation_pass) =
                    element_errors(&actual[arm].activation, &activation64);
                let (output_error, _) = element_errors(&actual[arm].output, &output);
                let (projection_boundary_error, projection_pass) =
                    element_errors(&actual[arm].projection, &rounded_projection);
                let (output_boundary_error, output_pass) =
                    element_errors(&actual[arm].output, &rounded_output);
                let actual_halves = projection_halves(&actual[arm].projection, down.input_features);
                let passes = projection_pass && activation_pass && output_pass;
                if rule == primary {
                    qualification &= passes;
                }
                println!(
                    "{}",
                    json!({"kind":"complete_ffn_element_reference","name":gate.name,"generation":generation,"arm":if arm==0 {"separate"} else {"packed"},"weight_reference":format!("{rule:?}"),"primary":rule==primary,"projection_vs_f64":projection_error,"projection_vs_rounded_f16":projection_boundary_error,"gate_vs_rounded_f16":element_errors(&actual_halves[0],&reference_halves[0]).0,"up_vs_rounded_f16":element_errors(&actual_halves[1],&reference_halves[1]).0,"silu_from_independent_reference_f16_projections":activation_error,"ffn_vs_full_chain_f64_down":output_error,"ffn_vs_rounded_f16":output_boundary_error,"all_stages_relative_pass":passes,"qualification_reference":"declared F16 stage outputs; pre-round F64 errors reported separately","silu_math":"F32 gate/(1+exp(-gate))*up then F16; host exp is independent from device expf","down_input":"independent reference activation, never actual arm activation"})
                );
            }
        }
        previous = Some(actual);
    }
    println!(
        "{}",
        json!({"kind":"complete_ffn_qualification","name":gate.name,"use_fp32_reduce":use_fp32_reduce,"primary_reference":format!("{primary:?}"),"qualified":qualification,"scope":"all elements of synthetic complete FFN, both fresh input generations, both arms; not model quality or serving performance"})
    );
    qualification
}

#[test]
#[ignore = "requires actual CUDA and pinned Marlin; independent complete FFN F64 qualification"]
fn marlin_ct_complete_ffn_independent_f64_reference_on_cuda() {
    run_qualification(false);
}

#[test]
#[ignore = "requires actual CUDA and pinned Marlin; FP32 global-reduction arithmetic diagnostic"]
fn marlin_ct_complete_ffn_fp32_reduction_independent_f64_reference_on_cuda() {
    run_qualification(true);
}

fn run_qualification(use_fp32_reduce: bool) {
    let primary = PRIMARY_REFERENCE;
    println!(
        "{}",
        json!({"kind":"complete_ffn_preregistered_contract","use_fp32_reduce":use_fp32_reduce,"primary_reference":format!("{primary:?}"),"relative_limit":RELATIVE_LIMIT,"relative_definition":"abs(actual-reference) <= 1e-3*abs(reference); exact zero required at zero reference","alternate_reference_is_not_a_fallback":true,"half_boundaries":"projection F16 -> F32 SiLU -> F16 activation -> down F16","ab_bits_are_reported_not_required":true,"eager_graph_bits_guards_and_freshness_required":true,"synthetic_not_model_quality":true,"performance_claim":false,"fp32_reduction_scope":"test-only explicit arithmetic change in both arms; original false test and its failed qualification remain separate"})
    );
    let context = CudaContext::new(0).expect("complete FFN qualification requires CUDA");
    let small = qualify_case(
        &context,
        Fixture {
            name: if use_fp32_reduce {
                "small-complete-ffn-fp32-reduce"
            } else {
                "small-complete-ffn"
            },
            rows: 3,
            input_features: 128,
            output_features: 256,
        },
        128,
        true,
        primary,
        use_fp32_reduce,
    );
    let actual_geometry = qualify_case(
        &context,
        Fixture {
            name: if use_fp32_reduce {
                "synthetic-m32-complete-ffn-fp32-reduce"
            } else {
                "synthetic-m32-complete-ffn"
            },
            rows: 32,
            input_features: 5120,
            output_features: 34816,
        },
        5120,
        false,
        primary,
        use_fp32_reduce,
    );
    assert!(
        small && actual_geometry,
        "primary independent all-element relative qualification failed; see complete stage records"
    );
}
