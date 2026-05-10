//! `QuantLinear<B>` — thin wrapper that delegates to the boxed
//! `Linear<B>` returned by `B::load_quant` / `B::load_quant_fused`.
//!
//! Phase 3e/3: backend-specific kernel dispatch (Metal Q4_K/Q6_K
//! mul_mm, CPU dequant + gemm) lives inside the boxed Linear's
//! `forward()` body, not in a Backend trait method. The historical
//! `QuantLinear<B>` constructors (`from_gguf_bytes`, `from_gguf_fused`)
//! stay so callers don't have to change shape — they just route through
//! the new factory.

use ferrum_kernels::backend::{Backend, BackendQuantGguf, GgufQuantType};
use ferrum_kernels::Linear;
use ferrum_types::Result;

/// Linear projection backed by a GGUF k-quant weight.
///
/// `forward()` is a tail-call to the inner backend-specific Linear
/// (Metal: `MetalGgufLinear`, CPU: `CpuGgufLinear`). LTO inlines through
/// the dispatch.
pub struct QuantLinear<B: Backend + BackendQuantGguf> {
    inner: Box<dyn Linear<B> + Send + Sync>,
}

impl<B: Backend + BackendQuantGguf> QuantLinear<B> {
    /// Build from raw GGUF block bytes.
    ///
    /// `kind`: which k-quant flavour the bytes encode (Q4_K, Q5_K, …).
    /// `bytes`: the on-disk payload, sized by the kind's block layout.
    pub fn from_gguf_bytes(
        kind: GgufQuantType,
        bytes: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        let inner = B::load_quant(kind, bytes, out_features, in_features)?;
        Ok(Self { inner })
    }

    /// Build a fused projection from multiple `(kind, bytes, rows)`
    /// parts that share `in_features`. Each part stays in its own
    /// QuantStore (no byte-concat); forward dispatches one matvec per
    /// part. Used for Qwen3 `qkv_proj` when q+k are Q4_K and v is Q6_K
    /// — the homogeneous fused-Q4 fast path would have to fall back
    /// to eager-fp32, blowing 100 MB per layer.
    pub fn from_gguf_fused(
        parts: &[(GgufQuantType, &[u8], usize)],
        in_features: usize,
    ) -> Result<Self> {
        let inner = B::load_quant_fused(parts, in_features)?;
        Ok(Self { inner })
    }
}

impl<B: Backend + BackendQuantGguf> Linear<B> for QuantLinear<B> {
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        self.inner.forward(ctx, input, out, m);
    }
}
