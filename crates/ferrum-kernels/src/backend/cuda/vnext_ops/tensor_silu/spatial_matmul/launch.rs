//! Enqueue only: all payload regions and the lane BLAS already have owners.
use super::*;
use cudarc::cublas::{result::gemm_ex, sys, CudaBlas};
use cudarc::driver::CudaStream;
use std::ffi::c_void;

fn flat(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            elements.div_ceil(u64::from(THREADS)).min(MAX_BLOCKS) as u32,
            1,
            1,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn configure(stream: &CudaStream, blas: &CudaBlas) -> Result<(), CudaDeviceRuntimeError> {
    let handle = *blas.handle();
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial GEMM stream", e))?;
        // SetStream resets workspace. Never permit the implicit cuBLAS pool.
        sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial GEMM zero workspace", e))?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial GEMM host scalars", e))?;
        sys::cublasSetMathMode(
            handle,
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )
        .result()
        .map_err(|e| CudaDeviceRuntimeError::blas("spatial GEMM FP32 reduction", e))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial GEMM atomics policy", e))?;
        let mut observed = sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
        sys::cublasGetMathMode(handle, &mut observed)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial GEMM observed math policy", e))?;
        if observed != sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION {
            return Err(CudaDeviceRuntimeError::contract(
                "spatial GEMM math policy was not applied",
            ));
        }
    }
    Ok(())
}

pub(super) fn enqueue(
    stream: &CudaStream,
    blas: &CudaBlas,
    kernels: &RetainedFunctions,
    regions: &[CudaBufferRegion],
    shape: Shape,
) -> Result<(), CudaDeviceRuntimeError> {
    configure(stream, blas)?;
    let handle = *blas.handle();
    let input = regions[0].device_ptr();
    let output = regions[1].device_ptr();
    let values = regions[2].device_ptr();
    let scores = values + shape.score_offset;
    let result = values + shape.result_offset;
    let one = 1.0_f32;
    let zero = 0.0_f32;
    let s = shape.spatial as i32;
    let c = shape.channels as i32;
    for batch in 0..shape.batch {
        let q = input + batch * 3 * shape.elements_per_batch * 2;
        let k = q + shape.elements_per_batch * 2;
        let v = k + shape.elements_per_batch * 2;
        let y = output + batch * shape.elements_per_batch * 2;
        let mut cast = stream.launch_builder(kernels.spatial_matmul_function(0));
        cast.arg(&v).arg(&values).arg(&shape.elements_per_batch);
        unsafe { cast.launch(flat(shape.elements_per_batch)) }
            .map_err(|e| CudaDeviceRuntimeError::driver("spatial GEMM values conversion", e))?;
        let mut start = 0u32;
        while start < shape.spatial {
            let queries = (shape.spatial - start).min(shape.tile);
            let m = queries as i32;
            let query = q + u64::from(start) * 2;
            // NCHW Q/K are column-major [S,C]. Scores are [S,M], so
            // each query's complete key row is contiguous for softmax.
            unsafe {
                gemm_ex(
                    handle,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    s,
                    m,
                    c,
                    &one as *const f32 as *const c_void,
                    k as *const c_void,
                    sys::cudaDataType_t::CUDA_R_16BF,
                    s,
                    query as *const c_void,
                    sys::cudaDataType_t::CUDA_R_16BF,
                    s,
                    &zero as *const f32 as *const c_void,
                    scores as *mut c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    s,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial QK BF16 to FP32 GEMM", e))?;
            let spatial = u64::from(shape.spatial);
            let mut softmax = stream.launch_builder(kernels.spatial_matmul_function(1));
            softmax
                .arg(&scores)
                .arg(&spatial)
                .arg(&queries)
                .arg(&shape.scale);
            unsafe {
                softmax.launch(LaunchConfig {
                    grid_dim: (queries, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("spatial FP32 softmax", e))?;
            // Result is column-major [M,C]. PEDANTIC forbids TF32 or
            // downconverting probabilities to BF16/FP16 for this product.
            unsafe {
                gemm_ex(
                    handle,
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    m,
                    c,
                    s,
                    &one as *const f32 as *const c_void,
                    scores as *const c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    s,
                    values as *const c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    s,
                    &zero as *const f32 as *const c_void,
                    result as *mut c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    m,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::blas("spatial PV pedantic FP32 GEMM", e))?;
            let mut scatter = stream.launch_builder(kernels.spatial_matmul_function(2));
            scatter
                .arg(&result)
                .arg(&scores)
                .arg(&y)
                .arg(&spatial)
                .arg(&shape.channels)
                .arg(&start)
                .arg(&queries);
            unsafe { scatter.launch(flat(u64::from(queries) * u64::from(shape.channels))) }
                .map_err(|e| CudaDeviceRuntimeError::driver("spatial BF16 output scatter", e))?;
            start += queries;
        }
    }
    Ok(())
}
