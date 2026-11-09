//! Isolate cuBLAS reduction precision on the existing FFN-down fixture.
//! Both arms call the unchanged production GEMM launcher on one private handle.

use super::*;
use cudarc::cublas::sys;

#[derive(Clone, Copy)]
enum ReductionMode {
    LibraryDefault,
    AccumulatorReduction,
}

impl ReductionMode {
    fn math_mode(self) -> sys::cublasMath_t {
        match self {
            Self::LibraryDefault => sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            // DEFAULT_MATH is zero: this is DEFAULT | DISALLOW_REDUCED.
            Self::AccumulatorReduction => {
                sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
            }
        }
    }

    fn select(self, kernels: &Kernels) -> u32 {
        // A single private handle is changed only after its prior work finishes.
        // Set/Get and this synchronization are outside every timed interval.
        kernels.stream.synchronize().unwrap();
        unsafe {
            sys::cublasSetMathMode(*kernels.blas.handle(), self.math_mode())
                .result()
                .unwrap();
        }
        let observed = observed_mode(&kernels.blas);
        assert_eq!(observed, self.math_mode());
        observed as u32
    }
}

fn observed_mode(blas: &CudaBlas) -> sys::cublasMath_t {
    let mut observed = sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
    unsafe {
        sys::cublasGetMathMode(*blas.handle(), &mut observed)
            .result()
            .unwrap();
    }
    observed
}

#[test]
#[ignore = "requires CUDA; full FFN-down reduction-mode diagnostic, no latency PASS threshold"]
fn cuda_dense_small_m_f16_ffn_down_reduction_modes() {
    const ITERATIONS: usize = 8;
    const PAIRS: usize = 8;
    let modes = [
        ReductionMode::LibraryDefault,
        ReductionMode::AccumulatorReduction,
    ];
    let kernels = Kernels::new();
    assert_eq!(
        observed_mode(&kernels.blas),
        sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
        "fresh diagnostic handle must start with the original library default"
    );
    let mut cublas_version = 0;
    unsafe {
        sys::cublasGetVersion_v2(*kernels.blas.handle(), &mut cublas_version)
            .result()
            .unwrap();
    }
    let maximum = Shape::new("ffn_down", 2, 4096, 12288);
    let x = Buffer::new(&kernels.stream, maximum.rows * maximum.inputs, |i| {
        value(i, 37, 1.0 / 1024.0)
    });
    // One complete 100,663,296-byte matrix serves both modes and both shapes.
    let w = Buffer::new(&kernels.stream, maximum.outputs * maximum.inputs, |i| {
        value(i, 731, 1.0 / 65536.0)
    });
    let outputs = [
        Buffer::new(&kernels.stream, maximum.rows * maximum.outputs, |_| GUARD),
        Buffer::new(&kernels.stream, maximum.rows * maximum.outputs, |_| GUARD),
    ];
    let oracle_started = Instant::now();
    let reference = f64_reference(maximum, x.values(), w.values());
    let oracle_wall_ms = oracle_started.elapsed().as_secs_f64() * 1000.0;
    for rows in [1, 2] {
        let shape = Shape::new("ffn_down", rows, maximum.outputs, maximum.inputs);
        let elements = rows * shape.outputs;
        let prepared = outputs
            .each_ref()
            .map(|output| kernels.prepare(shape, &x, &w, output));
        let mut observed_modes = [0; 2];
        for arm in 0..2 {
            observed_modes[arm] = modes[arm].select(&kernels);
            kernels.launch(false, &prepared[arm]);
        }
        kernels.stream.synchronize().unwrap();
        let actual = outputs
            .each_ref()
            .map(|output| output.read(&kernels.stream));
        for output in &actual {
            assert!(output[elements..].iter().all(|&value| value == GUARD));
        }
        let stats = actual
            .each_ref()
            .map(|output| validate(&output[..elements], &reference[..elements]));
        let bit_mismatches = actual[0][..elements]
            .iter()
            .zip(&actual[1][..elements])
            .filter(|(left, right)| left != right)
            .count();
        println!(
            "{}",
            serde_json::json!({
                "diagnostic":"dense_f16_cublas_reduction_validation",
                "shape":shape.name,"m":rows,"n":shape.outputs,"k":shape.inputs,
                "all_outputs_checked":elements,"cublas_version":cublas_version,
                "library_default":{"math_mode":observed_modes[0],"errors":stats[0]},
                "accumulator_reduction":{"math_mode":observed_modes[1],"errors":stats[1]},
                "between_modes_bit_mismatches":bit_mismatches,"oracle_wall_ms":oracle_wall_ms,
                "gemm":"unchanged launch_gemm_f16; A/B/C=F16, compute=32F, DEFAULT_TENSOR_OP, alpha=1, beta=0",
                "workspace":"original fresh-handle vendor default; not reset or replaced between arms",
                "oracle":"serial F64 dot from the same F16 carriers, then F16 RNE; complete output",
                "qualification":"original forward-error checks remain mandatory; relative 1/1000 versus rounded F64 is reported, not a production CPU-provider qualification",
                "production_selected":false
            })
        );
        // Validation precedes timing; mode changes and warmup are not timed.
        for arm in 0..2 {
            modes[arm].select(&kernels);
            for _ in 0..4 {
                kernels.launch(false, &prepared[arm]);
            }
        }
        kernels.stream.synchronize().unwrap();
        for pair in 0..PAIRS {
            let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut timing = [[0.0_f64; 3]; 2];
            for arm in order {
                observed_modes[arm] = modes[arm].select(&kernels);
                let wall = Instant::now();
                let start = kernels
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                let enqueue = Instant::now();
                for _ in 0..ITERATIONS {
                    kernels.launch(false, &prepared[arm]);
                }
                let enqueue_ns = enqueue.elapsed().as_nanos() as f64 / ITERATIONS as f64;
                let end = kernels
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                end.synchronize().unwrap();
                timing[arm] = [
                    f64::from(start.elapsed_ms(&end).unwrap()) * 1e6 / ITERATIONS as f64,
                    enqueue_ns,
                    wall.elapsed().as_nanos() as f64 / ITERATIONS as f64,
                ];
            }
            println!(
                "{}",
                serde_json::json!({
                    "diagnostic":"dense_f16_cublas_reduction_timing",
                    "shape":shape.name,"m":rows,"n":shape.outputs,"k":shape.inputs,
                    "pair":pair,"order":order,"iterations":ITERATIONS,
                    "cublas_version":cublas_version,"weight_bytes":w.values().len() * 2,
                    "library_default":{"math_mode":observed_modes[0],"cuda_event_ns":timing[0][0],"host_enqueue_ns":timing[0][1],"completion_wall_ns":timing[0][2]},
                    "accumulator_reduction":{"math_mode":observed_modes[1],"cuda_event_ns":timing[1][0],"host_enqueue_ns":timing[1][1],"completion_wall_ns":timing[1][2]},
                    "timing_scope":"same private handle/stream/owners; warmed alternating modes; Set/GetMathMode, prior synchronization, allocation, oracle and readback excluded; events may include host-feed gaps",
                    "performance_threshold":null,"production_selected":false
                })
            );
        }
        // Retain the original guards and verify timing did not change outputs.
        for arm in 0..2 {
            assert_eq!(outputs[arm].read(&kernels.stream), actual[arm]);
        }
        x.assert_read_only(&kernels.stream);
        w.assert_read_only(&kernels.stream);
    }
    ReductionMode::LibraryDefault.select(&kernels);
}
