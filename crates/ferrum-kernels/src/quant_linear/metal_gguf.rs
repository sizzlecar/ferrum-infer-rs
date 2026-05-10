//! `Linear<MetalBackend>` impl for GGUF k-quant weights (Q4_K / Q6_K).
//!
//! Phase 3e/3: replaces the old `BackendQuantGguf::gemm_quant` impl on
//! MetalBackend. The kernel dispatch (decode/prefill split, fused-store
//! handling, q4_k / q6_k mul_mm + gemv variants) lives inside
//! `crate::backend::metal::metal_gemm_quant_dispatch` (the body that
//! used to be the trait method, now `pub fn`).

use crate::backend::metal::{metal_gemm_quant_dispatch, MetalBackend, MetalQuantStore};
use crate::Linear;

/// Metal GGUF Linear: holds a `MetalQuantStore` (Q4_K / Q6_K /
/// Fused-mixed) plus shape, dispatches via `metal_gemm_quant_dispatch`.
pub struct MetalGgufLinear {
    pub store: MetalQuantStore,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear<MetalBackend> for MetalGgufLinear {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn forward(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        input: &<MetalBackend as crate::backend::Backend>::Buffer,
        out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        m: usize,
    ) {
        metal_gemm_quant_dispatch(ctx, input, &self.store, out, m)
            .unwrap_or_else(|e| panic!("MetalGgufLinear forward failed: {e}"));
    }
}
