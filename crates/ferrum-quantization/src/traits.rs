//! Core `Linear` trait.
//!
//! A `Linear<B>` is a single matrix-vector / matrix-matrix projection.
//! Implementations carry whatever weight representation they need (dense
//! f32/f16 buffer, GPTQ packed int4 + scales + zeros, etc.) and dispatch
//! to the appropriate `Backend` op in `forward`.

use ferrum_kernels::backend::Backend;

/// A weight-bearing linear projection.
///
/// `forward` computes `out[m, out_features] = input[m, in_features] @ W^T`.
/// The implementation is responsible for calling the right `Backend` op
/// (e.g. `B::gemm` for dense, `B::gemm_gptq` for GPTQ).
pub trait Linear<B: Backend>: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;

    /// Append GEMM work onto `ctx`. Caller is responsible for eventually
    /// flushing the context.
    fn forward(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        out: &mut B::Buffer,
        m: usize,
    );
}

/// Factory for constructing `Linear<B>` from raw loaded tensors.
///
/// A `WeightLoader` will typically hold a `LinearFactory<B>` and call it
/// for each layer weight. This indirection keeps the loader independent
/// of any specific `Backend`.
pub trait LinearFactory<B: Backend>: Send + Sync {
    /// Build a dense `Linear` from an already-materialised weight buffer.
    ///
    /// `weight_row_major`: `[out_features, in_features]` laid out contiguously.
    fn dense(
        &self,
        weight_row_major: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Box<dyn Linear<B>>;
}
