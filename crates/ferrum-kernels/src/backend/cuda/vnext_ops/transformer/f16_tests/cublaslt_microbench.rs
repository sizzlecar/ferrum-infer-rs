//! Opt-in diagnostics only. Explicit bounded workspace and a single default
//! heuristic per shape/limit; no production allocator or dispatch changes.

use super::*;
use cudarc::cublaslt::{result as lt, sys as ls};
use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::DevicePtr;

struct LtPlan {
    handle: ls::cublasLtHandle_t,
    desc: ls::cublasLtMatmulDesc_t,
    a: ls::cublasLtMatrixLayout_t,
    b: ls::cublasLtMatrixLayout_t,
    c: ls::cublasLtMatrixLayout_t,
    preference: ls::cublasLtMatmulPreference_t,
    heuristic: Option<ls::cublasLtMatmulHeuristicResult_t>,
}

impl LtPlan {
    fn new(hidden: usize, outputs: usize, rows: usize, workspace_limit: usize) -> Self {
        let mut plan = Self {
            handle: std::ptr::null_mut(),
            desc: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
            heuristic: None,
        };
        plan.handle = lt::create_handle().unwrap();
        plan.desc = lt::create_matmul_desc(
            ls::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ls::cudaDataType_t::CUDA_R_32F,
        )
        .unwrap();
        plan.a = lt::create_matrix_layout(
            ls::cudaDataType_t::CUDA_R_16F,
            hidden as u64,
            outputs as u64,
            hidden as i64,
        )
        .unwrap();
        plan.b = lt::create_matrix_layout(
            ls::cudaDataType_t::CUDA_R_16F,
            hidden as u64,
            rows as u64,
            hidden as i64,
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
        let transpose = 1_i32;
        let alignment = 16_u32;
        unsafe {
            lt::set_matmul_desc_attribute(
                plan.desc,
                ls::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&transpose as *const i32).cast(),
                std::mem::size_of_val(&transpose),
            )
            .unwrap();
            lt::set_matmul_pref_attribute(
                plan.preference,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&workspace_limit as *const usize).cast(),
                std::mem::size_of_val(&workspace_limit),
            )
            .unwrap();
            for attribute in [
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                ls::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
            ] {
                lt::set_matmul_pref_attribute(
                    plan.preference,
                    attribute,
                    (&alignment as *const u32).cast(),
                    std::mem::size_of_val(&alignment),
                )
                .unwrap();
            }
            plan.heuristic = Some(
                lt::get_matmul_algo_heuristic(
                    plan.handle,
                    plan.desc,
                    plan.a,
                    plan.b,
                    plan.c,
                    plan.c,
                    plan.preference,
                )
                .unwrap(),
            );
        }
        assert!(plan.heuristic.as_ref().unwrap().workspaceSize <= workspace_limit);
        plan
    }
}

impl Drop for LtPlan {
    fn drop(&mut self) {
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

#[test]
#[ignore = "requires CUDA and 1.4 GiB free device memory; opt-in Lt diagnostic"]
fn f16_cublaslt_default_heuristic_graph_microbench() {
    let context = CudaContext::new(0).expect("Lt microbenchmark requires CUDA");
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

        for rows in [4_usize, 8] {
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

            for workspace_limit in [0_usize, 4 * 1024 * 1024] {
                let plan = LtPlan::new(hidden, outputs, rows, workspace_limit);
                let heuristic = plan.heuristic.as_ref().unwrap();
                let workspace_template = vec![0xa5_u8; workspace_limit + 512];
                let workspace = stream.clone_htod(&workspace_template).unwrap();
                let (sp, _workspace_retention) = workspace.device_ptr(&stream);
                assert_eq!((sp + 256) % 256, 0);
                let workspace_ptr = if workspace_limit == 0 {
                    std::ptr::null_mut()
                } else {
                    (sp + 256) as *mut c_void
                };
                let launch = |index: usize, matrix: usize| unsafe {
                    let a = (wp + 16 + (matrix * matrix_elements * 2) as u64) as *const c_void;
                    let b = (xp + 16) as *const c_void;
                    let c = (yp + 16) as *mut c_void;
                    if index == 0 {
                        gemm_ex(
                            *blas.handle(),
                            cublasOperation_t::CUBLAS_OP_T,
                            cublasOperation_t::CUBLAS_OP_N,
                            outputs as i32,
                            rows as i32,
                            hidden as i32,
                            (&CUDA_GEMM_ALPHA_F32 as *const f32).cast(),
                            a,
                            cudaDataType_t::CUDA_R_16F,
                            hidden as i32,
                            b,
                            cudaDataType_t::CUDA_R_16F,
                            hidden as i32,
                            (&CUDA_GEMM_BETA_F32 as *const f32).cast(),
                            c,
                            cudaDataType_t::CUDA_R_16F,
                            outputs as i32,
                            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                        )
                        .unwrap();
                    } else {
                        lt::matmul(
                            plan.handle,
                            plan.desc,
                            (&CUDA_GEMM_ALPHA_F32 as *const f32).cast(),
                            (&CUDA_GEMM_BETA_F32 as *const f32).cast(),
                            a,
                            plan.a,
                            b,
                            plan.b,
                            c.cast_const(),
                            plan.c,
                            c,
                            plan.c,
                            &heuristic.algo,
                            workspace_ptr,
                            workspace_limit,
                            stream.cu_stream().cast(),
                        )
                        .unwrap();
                    }
                };
                let mut graphs = Vec::new();
                for index in [0, 1] {
                    launch(index, 0);
                    stream.synchronize().unwrap();
                    stream
                        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                        .unwrap();
                    for iteration in 0..iterations {
                        launch(index, iteration % weight_count);
                    }
                    // No allocation nodes; AUTO_FREE is inert. An untimed replay
                    // uploads the graph without WithParams-only UPLOAD flags.
                    let graph = stream
                        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                        .unwrap()
                        .expect("nonempty Lt comparison graph");
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
                        let scratch = stream.clone_dtoh(&workspace).unwrap();
                        assert_eq!(&scratch[..256], &workspace_template[..256]);
                        assert_eq!(
                            &scratch[256 + workspace_limit..],
                            &workspace_template[256 + workspace_limit..]
                        );
                        if round >= 2 {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "benchmark": "f16_cublaslt_default_heuristic_graph",
                                    "mode": if index == 0 { "gemm_ex" } else { "lt" },
                                    "rows": rows, "hidden": hidden, "outputs": outputs,
                                    "workspace_limit": workspace_limit,
                                    "heuristic_workspace": heuristic.workspaceSize,
                                    "heuristic_algo": heuristic.algo.data,
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
}
