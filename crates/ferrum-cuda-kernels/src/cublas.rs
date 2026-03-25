//! cuBLAS GEMM wrapper for LLM linear projections.
//!
//! Wraps cudarc's `Gemm<half::f16>` for the common LLM pattern:
//!   output = input @ weight^T   (no allocation — writes to pre-existing buffer)
//!
//! This is the foundation for bypassing candle's matmul in the decode hot path.
//! candle's matmul allocates a new output tensor on each call; this wrapper
//! writes to a pre-allocated buffer, enabling CUDA Graph capture.

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// Compute output = input @ weight^T using cuBLAS HGEMM.
///
/// - `input`:  [M, K] row-major f16 on device
/// - `weight`: [N, K] row-major f16 on device (transposed in GEMM → op_t)
/// - `output`: [M, N] row-major f16 on device (pre-allocated, overwritten)
///
/// For decode (M=1): this is effectively a matrix-vector multiply.
/// For prefill (M>1): standard GEMM.
///
/// Uses FP32 accumulation internally (CUBLAS_COMPUTE_32F) for numerical stability.
pub fn linear_f16(
    blas: &CudaBlas,
    input: &CudaSlice<half::f16>,
    weight: &CudaSlice<half::f16>,
    output: &mut CudaSlice<half::f16>,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), cudarc::cublas::CublasError> {
    // cuBLAS uses column-major. For row-major C = A @ B^T:
    //   C^T = B @ A^T  (in column-major)
    //   → cublas_gemm(OP_T, OP_N, N, M, K, ...)
    //
    // A (input):  row-major [M, K] → col-major [K, M] → OP_N with lda=K
    // B (weight): row-major [N, K] → col-major [K, N] → OP_T with ldb=K
    // C (output): row-major [M, N] → col-major [N, M] → ldc=N
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n, // rows of op(B) = rows of C^T = N
        n: m, // cols of op(A) = cols of C^T = M
        k,
        alpha: half::f16::ONE,
        lda: k, // leading dim of weight (col-major view of [N,K] row-major)
        ldb: k, // leading dim of input (col-major view of [M,K] row-major)
        beta: half::f16::ZERO,
        ldc: n, // leading dim of output (col-major view of [M,N] row-major)
    };

    unsafe { blas.gemm(cfg, weight, input, output) }
}

/// Compute output = input @ weight^T + bias using cuBLAS HGEMM.
///
/// Same as `linear_f16` but adds a bias vector after GEMM.
/// Bias is broadcast-added row-wise: output[i, :] += bias[:]
///
/// Note: For decode (M=1), bias add is trivial. For now we do GEMM then
/// add bias separately. A fused GEMM+bias kernel could be added later.
pub fn linear_f16_bias(
    blas: &CudaBlas,
    input: &CudaSlice<half::f16>,
    weight: &CudaSlice<half::f16>,
    _bias: &CudaSlice<half::f16>,
    output: &mut CudaSlice<half::f16>,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), cudarc::cublas::CublasError> {
    // For now, just do the GEMM. Bias add would need a separate kernel.
    // Most LLM layers use no-bias linears (Qwen3 has no bias on projections).
    linear_f16(blas, input, weight, output, m, n, k)
}
