//! cuBLAS GEMM wrapper for LLM linear projections.
//!
//! Wraps cudarc's `Gemm<half::f16>` for the common LLM pattern:
//!   output = input @ weight^T   (no allocation — writes to pre-existing buffer)

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::CudaSlice;

/// Compute output = input @ weight^T using cuBLAS HGEMM.
///
/// - `input`:  [M, K] row-major f16 on device
/// - `weight`: [N, K] row-major f16 on device (transposed in GEMM → op_t)
/// - `output`: [M, N] row-major f16 on device (pre-allocated, overwritten)
pub fn linear_f16(
    blas: &CudaBlas,
    input: &CudaSlice<half::f16>,
    weight: &CudaSlice<half::f16>,
    output: &mut CudaSlice<half::f16>,
    m: i32,
    n: i32,
    k: i32,
) -> candle_core::Result<()> {
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k,
        alpha: half::f16::ONE,
        lda: k,
        ldb: k,
        beta: half::f16::ZERO,
        ldc: n,
    };

    unsafe { blas.gemm(cfg, weight, input, output) }
        .map_err(|e| candle_core::Error::Msg(format!("cuBLAS gemm failed: {e}")))
}
