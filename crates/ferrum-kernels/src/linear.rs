//! `Linear<B>` trait — weight-bearing projection abstraction.
//!
//! Lives in ferrum-kernels alongside `Backend` because:
//!   1. `Backend::layer_forward_fused` and other "standard transformer layer"
//!      helpers want to accept `&dyn Linear<Self>` as their projection
//!      parameter, so the trait must be visible here.
//!   2. Model code in `ferrum-models` depends on both ferrum-kernels and
//!      ferrum-quantization, so keeping the trait in kernels avoids any
//!      circular dependency between kernels and quantization.
//!
//! Concrete implementations (DenseLinear, GptqLinear, AwqLinear, GgufLinear)
//! live in `ferrum-quantization`, which depends on `ferrum-kernels` for this
//! trait and for the `Backend` it parameterises over.

use crate::backend::Backend;

/// A weight-bearing linear projection.
///
/// `forward` computes `out[m, out_features] = input[m, in_features] @ W^T`.
/// Implementations are responsible for calling the right backend kernel
/// (`B::gemm` for dense, `B::gemm_quant` for quantized variants).
pub trait Linear<B: Backend>: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;

    /// Append GEMM work onto `ctx`. Caller flushes the context when results
    /// must be materialised.
    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize);
}
