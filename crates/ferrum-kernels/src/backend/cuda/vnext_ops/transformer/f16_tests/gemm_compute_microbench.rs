//! Diagnostic only: compare cuBLAS compute modes without changing production.
//! Rotate weights for the smaller projections to avoid measuring only a single
//! repeatedly cached matrix. Graph replay excludes host launch-loop overhead.

use super::*;
use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::DevicePtr;

#[derive(Clone, Copy, Debug)]
enum Compute {
    ExistingFast16,
    Float32,
}

impl Compute {
    fn mode(self) -> cublasComputeType_t {
        match self {
            Self::ExistingFast16 => cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
            Self::Float32 => cublasComputeType_t::CUBLAS_COMPUTE_32F,
        }
    }
}

#[test]
#[ignore = "requires CUDA and 1.4 GiB free device memory; diagnostic microbenchmark"]
fn f16_gemm_compute_mode_graph_microbench() {
    let context = CudaContext::new(0).expect("GEMM microbenchmark requires CUDA");
    let stream = context.new_stream().unwrap();
    let blas = CudaBlas::new(stream.clone()).unwrap();
    let guard = f16::from_f32(-117.0);
    let coefficient =
        |matrix: usize, column: usize| ((column + matrix * 3) as i32 % 17 - 8) as f32 / 64.0;

    for (hidden, outputs) in [(2560_usize, 18432_usize), (9216, 2560), (2560, 248320)] {
        let matrix_elements = hidden * outputs;
        let weight_count = (128 * 1024 * 1024_usize).div_ceil(matrix_elements * 2);
        let mut weights = vec![guard; 16 + weight_count * matrix_elements];
        for matrix in 0..weight_count {
            for (column, row) in weights
                [8 + matrix * matrix_elements..8 + (matrix + 1) * matrix_elements]
                .chunks_exact_mut(hidden)
                .enumerate()
            {
                row.fill(f16::from_f32(coefficient(matrix, column)));
            }
        }
        let w = stream.clone_htod(&weights).unwrap();
        drop(weights);
        let (wp, _weight_retention) = w.device_ptr(&stream);
        let iterations = weight_count * 8;

        for rows in [1_usize, 4, 8, 16] {
            let mut input = vec![guard; 16 + rows * hidden];
            let mut sums = Vec::with_capacity(rows);
            for (row, values) in input[8..8 + rows * hidden]
                .chunks_exact_mut(hidden)
                .enumerate()
            {
                for (k, value) in values.iter_mut().enumerate() {
                    *value = f16::from_f32(((k * 13 + row * 7) % 41) as f32 / 32.0 - 0.625);
                }
                sums.push(values.iter().map(|value| value.to_f64()).sum::<f64>());
            }
            let x = stream.clone_htod(&input).unwrap();
            let output_template = vec![guard; 16 + rows * outputs];
            let y = stream.clone_htod(&output_template).unwrap();
            let (xp, _input_retention) = x.device_ptr(&stream);
            let (yp, _output_retention) = y.device_ptr(&stream);
            let launch = |compute: Compute, matrix: usize| unsafe {
                gemm_ex(
                    *blas.handle(),
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    outputs as i32,
                    rows as i32,
                    hidden as i32,
                    &CUDA_GEMM_ALPHA_F32 as *const f32 as *const c_void,
                    (wp + 16 + (matrix * matrix_elements * 2) as u64) as *const c_void,
                    cudaDataType_t::CUDA_R_16F,
                    hidden as i32,
                    (xp + 16) as *const c_void,
                    cudaDataType_t::CUDA_R_16F,
                    hidden as i32,
                    &CUDA_GEMM_BETA_F32 as *const f32 as *const c_void,
                    (yp + 16) as *mut c_void,
                    cudaDataType_t::CUDA_R_16F,
                    outputs as i32,
                    compute.mode(),
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .unwrap();
            };

            let mut graphs = Vec::new();
            for compute in [Compute::ExistingFast16, Compute::Float32] {
                launch(compute, 0);
                stream.synchronize().unwrap();
                stream
                    .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .unwrap();
                for iteration in 0..iterations {
                    launch(compute, iteration % weight_count);
                }
                // cudarc uses cuGraphInstantiateWithFlags; UPLOAD is valid only
                // for WithParams. This graph has no allocation nodes, so the
                // supported AUTO_FREE flag has no effect. The untimed first
                // replay below completes graph upload for both modes.
                let graph = stream
                    .end_capture(
                        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    )
                    .unwrap()
                    .expect("nonempty GEMM graph");
                graph.launch().unwrap();
                stream.synchronize().unwrap();
                graphs.push(graph);
            }

            for round in 0..8 {
                for index in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    graphs[index].launch().unwrap();
                    let end = stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    end.synchronize().unwrap();
                    let gpu_ns =
                        f64::from(start.elapsed_ms(&end).unwrap()) * 1e6 / iterations as f64;

                    // Dyadic inputs permit a complete exact F64 output oracle.
                    // This fixture is not comprehensive floating-point coverage.
                    let actual = stream.clone_dtoh(&y).unwrap();
                    assert_eq!(&actual[..8], &output_template[..8]);
                    assert_eq!(
                        &actual[8 + rows * outputs..],
                        &output_template[8 + rows * outputs..]
                    );
                    for (row, sum) in sums.iter().copied().enumerate() {
                        for column in 0..outputs {
                            let expected = f16::from_f64(
                                sum * f64::from(coefficient(weight_count - 1, column)),
                            );
                            assert_eq!(actual[8 + row * outputs + column], expected);
                        }
                    }
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark": "f16_gemm_compute_mode_graph",
                                "mode": if index == 0 { "existing_fast_16f" } else { "32f" },
                                "rows": rows, "hidden": hidden, "outputs": outputs,
                                "rotating_weights": weight_count, "iterations": iterations,
                                "round": round - 2, "gpu_ns": gpu_ns,
                                "validated_outputs": rows * outputs,
                            })
                        );
                    }
                }
            }
            assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
        }
    }
}
