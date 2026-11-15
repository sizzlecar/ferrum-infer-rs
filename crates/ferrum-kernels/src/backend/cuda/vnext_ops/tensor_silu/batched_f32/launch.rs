use super::*;
use cudarc::cublas::{result::gemm_ex, sys, CudaBlas};
use cudarc::driver::CudaStream;
use std::ffi::c_void;

pub(super) fn matmul(
    stream: &CudaStream,
    blas: &CudaBlas,
    r: &[CudaBufferRegion],
    g: BatchedF32Geometry,
) -> Result<(), CudaDeviceRuntimeError> {
    let handle = *blas.handle();
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("batched F32 stream", e))?;
        sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("batched F32 zero workspace", e))?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("batched F32 scalars", e))?;
        sys::cublasSetMathMode(handle, sys::cublasMath_t::CUBLAS_PEDANTIC_MATH)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("batched F32 pedantic math", e))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("batched F32 atomics", e))?;
    }
    let one = 1f32;
    let zero = 0f32;
    let m = g.m() as i32;
    let n = g.n() as i32;
    let k = g.k() as i32;
    // Row-major C=A*op(B) becomes column-major C^T=op(B)^T*A^T.
    // Iterate the two public batch axes independently; no implicit broadcast.
    for batch in 0..g.batch() {
        for group in 0..g.group() {
            let index = batch * g.group() + group;
            let a = r[0].device_ptr() + index * g.m() * g.k() * 4;
            let b = r[1].device_ptr() + index * g.n() * g.k() * 4;
            let y = r[2].device_ptr() + index * g.m() * g.n() * 4;
            let (transpose, lda) = if g.transpose_rhs() {
                (sys::cublasOperation_t::CUBLAS_OP_T, k)
            } else {
                (sys::cublasOperation_t::CUBLAS_OP_N, n)
            };
            unsafe {
                gemm_ex(
                    handle,
                    transpose,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &one as *const f32 as *const c_void,
                    b as *const c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    lda,
                    a as *const c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k,
                    &zero as *const f32 as *const c_void,
                    y as *mut c_void,
                    sys::cudaDataType_t::CUDA_R_32F,
                    n,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::blas("batched pedantic F32 GEMM", e))?;
        }
    }
    Ok(())
}
pub(super) fn pointwise(
    stream: &CudaStream,
    kernels: &RetainedFunctions,
    r: &[CudaBufferRegion],
    g: BatchedF32Geometry,
) -> Result<(), CudaDeviceRuntimeError> {
    let input = r[0].device_ptr();
    let output = r[1].device_ptr();
    let elements = g.output_elements();
    let rows = g.rows();
    let width = g.n();
    let scale = g.scale();
    let index = if g.op() == BatchedF32Op::Scale { 0 } else { 1 };
    let mut builder = stream.launch_builder(kernels.batched_f32_function(index));
    builder.arg(&input).arg(&output);
    let blocks = if g.op() == BatchedF32Op::Scale {
        builder.arg(&elements).arg(&scale);
        elements.div_ceil(256).min(MAX_BLOCKS)
    } else {
        builder.arg(&rows).arg(&width);
        rows.min(MAX_BLOCKS)
    };
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|e| CudaDeviceRuntimeError::driver("batched F32 scale/softmax", e))
}
