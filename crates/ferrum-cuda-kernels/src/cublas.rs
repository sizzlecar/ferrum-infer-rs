//! cuBLAS GEMM wrapper for LLM linear projections.
//!
//! Wraps cudarc's `Gemm<half::f16>` for the common LLM pattern:
//!   output = input @ weight^T   (no allocation — writes to pre-existing buffer)

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaSlice, CudaStream};

/// Compute output = input @ weight^T using raw cublasGemmEx FFI.
///
/// Bypasses cudarc's DevicePtr/SyncOnDrop which does stream.wait()
/// and event recording that break CUDA Graph capture.
pub fn linear_f16(
    blas: &CudaBlas,
    input: &CudaSlice<half::f16>,
    weight: &CudaSlice<half::f16>,
    output: &mut CudaSlice<half::f16>,
    m: i32,
    n: i32,
    k: i32,
) -> candle_core::Result<()> {
    use cudarc::cublas::sys::*;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // Get raw device pointers, bypassing cudarc's event tracking.
    // We use DevicePtr but forget the SyncOnDrop guard — all buffers are
    // on the same stream so no cross-stream sync is needed.
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let stream = input.stream();
    let (a_ptr, a_guard) = weight.device_ptr(stream);
    std::mem::forget(a_guard);
    let (b_ptr, b_guard) = input.device_ptr(stream);
    std::mem::forget(b_guard);
    let (c_ptr, c_guard) = output.device_ptr_mut(stream);
    std::mem::forget(c_guard);
    let a_ptr = a_ptr as *const std::ffi::c_void;
    let b_ptr = b_ptr as *const std::ffi::c_void;
    let c_ptr = c_ptr as *mut std::ffi::c_void;

    let status = unsafe {
        cublasGemmEx(
            *blas.handle(),
            cublasOperation_t::CUBLAS_OP_T, // weight transposed
            cublasOperation_t::CUBLAS_OP_N, // input not transposed
            n,                              // rows of output (output features)
            m,                              // cols of output (batch)
            k,                              // reduction dim
            &alpha as *const f32 as *const _,
            a_ptr,
            cudaDataType_t::CUDA_R_16F,
            k, // lda
            b_ptr,
            cudaDataType_t::CUDA_R_16F,
            k, // ldb
            &beta as *const f32 as *const _,
            c_ptr,
            cudaDataType_t::CUDA_R_16F,
            n, // ldc
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };

    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Err(candle_core::Error::Msg(format!(
            "cublasGemmEx failed: {status:?}"
        )))
    } else {
        Ok(())
    }
}
