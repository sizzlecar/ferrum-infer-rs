//! `StackedExpertGgufLinear<B>` — abstraction for "N MoE experts' GGUF
//! quantized weights stored contiguously, dispatched as one batched
//! MoE GEMV/GEMM kernel".
//!
//! Phase 3e/4 (sibling to `Linear<B>`): replaces the `*_moe_id*`
//! trait methods on `BackendQuantGguf` and the `type QuantStore`
//! associated type on `Backend`. Same goal as PR #123 (Marlin Linear
//! cutover): make weight-format a polymorphism point that doesn't
//! leak into the `Backend` trait.
//!
//! Concrete impls live in `quant_linear/`:
//!   - `quant_linear::metal_gguf_moe::MetalStackedExpertGgufLinear`
//!     (wraps Metal `Q4KExperts` / `Q6KExperts` `MetalQuantStore`
//!     variants, dispatches via `dispatch_gemv_moe_id*` /
//!     `dispatch_gemm_moe_id*`).
//!   - CPU has no MoE batched kernel today — the trait default
//!     `Unsupported` is fine; ExpertStack falls back to per-expert
//!     `Linear<B>` calls.
//!   - CUDA's stacked-expert path goes through `make_stacked_expert_linear`
//!     (returns `Box<dyn Linear<Self>>`) since it's Marlin tiles, not
//!     GGUF k-quant. That path stays unchanged.
//!
//! The trait surface is **7 methods**: one GEMV per dispatch shape
//! (single / offset / batched / fused-gate-up-silu / batched-fused) +
//! two GEMM variants (direct / indirect-args). Each is a one-line
//! wrapper over a backend-specific dispatcher in practice.

use crate::backend::Backend;
use ferrum_types::Result;

/// MoE-stacked GGUF-quantized linear: holds N experts' weights for one
/// matmul role (gate / up / down) in one contiguous buffer, dispatches
/// `out[pair] = a[pair] @ dequant(W[ids[pair]])^T` over all `(token,
/// slot)` pairs in a single backend launch.
pub trait StackedExpertGgufLinear<B: Backend>: Send + Sync {
    fn num_experts(&self) -> usize;
    /// Per-expert output features (n_rows).
    fn n_rows(&self) -> usize;
    /// Input features (n_cols). Common across experts.
    fn n_cols(&self) -> usize;

    /// Downcast hook for fused multi-store methods (`gemv_moe_id_gate_up_silu*`)
    /// — the impl needs to reach into the *up* projection's concrete store
    /// to access raw GPU resources. Standard `dyn Any` pattern; only used at
    /// FFN dispatch boundaries (low frequency vs the per-pair kernel work).
    fn as_any(&self) -> &dyn std::any::Any;

    /// MoE indirect-dispatch GEMV (single-token decode path):
    /// `out[i, :] = a[off(i, src1_stride), :] @ dequant(W[ids[i], :])^T` for `i ∈ [0, n_selected)`.
    fn gemv_moe_id(
        &self,
        ctx: &mut B::Context,
        a: &B::Buffer,
        ids: &B::Buffer,
        out: &mut B::Buffer,
        n_selected: usize,
        src1_stride: usize,
    ) -> Result<()>;

    /// Offset-aware variant — `a` starts at `a_offset` (elements), `ids` at
    /// `ids_offset`. Used by the per-item batched decode loop to read
    /// directly from the M-batch buffer without per-iter copies.
    #[allow(clippy::too_many_arguments)]
    fn gemv_moe_id_offset(
        &self,
        ctx: &mut B::Context,
        a: &B::Buffer,
        a_offset: usize,
        ids: &B::Buffer,
        ids_offset: usize,
        out: &mut B::Buffer,
        n_selected: usize,
        src1_stride: usize,
    ) -> Result<()>;

    /// Fused gate+up MoE GEMV with in-register `SiLU(gate) * up`.
    /// Folds 3 dispatches (gate gemv, up gemv, silu_mul) into one.
    /// `other` is the up-projection weight stack (must match shape).
    fn gemv_moe_id_gate_up_silu(
        &self,
        ctx: &mut B::Context,
        a: &B::Buffer,
        other_up: &dyn StackedExpertGgufLinear<B>,
        ids: &B::Buffer,
        silu_out: &mut B::Buffer,
        n_selected: usize,
    ) -> Result<()>;

    /// Batched MoE GEMV — one launch covers all `m * top_k` pairs.
    /// `pair p` reads `a[(p/top_k) * src1_outer_stride + (p%top_k) * src1_inner_stride]`.
    #[allow(clippy::too_many_arguments)]
    fn gemv_moe_id_batched(
        &self,
        ctx: &mut B::Context,
        a: &B::Buffer,
        ids: &B::Buffer,
        out: &mut B::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()>;

    /// Batched fused gate+up GEMV — counterpart of
    /// `gemv_moe_id_gate_up_silu` for the m≥2 batched-decode path.
    #[allow(clippy::too_many_arguments)]
    fn gemv_moe_id_gate_up_silu_batched(
        &self,
        ctx: &mut B::Context,
        a: &B::Buffer,
        other_up: &dyn StackedExpertGgufLinear<B>,
        ids: &B::Buffer,
        silu_out: &mut B::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()>;

    /// MoE 2-D indirect-dispatch GEMM (prefill, m > 1).
    /// `out[token, slot, :] = a[token, slot_or_0, :] @ dequant(W[expert(token, slot)])^T`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_moe_id(
        &self,
        ctx: &mut B::Context,
        a: &B::Buffer,
        ids: &B::Buffer,
        tpe: &B::Buffer,
        out: &mut B::Buffer,
        ne11: usize,
        top_k: usize,
        max_per_expert: usize,
        batch: usize,
    ) -> Result<()>;

    /// Indirect-dispatch GEMM — grid read from `args_buf` (12-byte u32
    /// triple written by `compute_ids_tpe_gpu`) instead of computed from
    /// `max_per_expert`. Eliminates the host-side D2H of `tpe` that the
    /// direct variant needs.
    #[allow(clippy::too_many_arguments)]
    fn gemm_moe_id_indirect(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        ids: &B::Buffer,
        tpe: &B::Buffer,
        out: &mut B::Buffer,
        args_buf: &B::Buffer,
        ne11: usize,
        top_k: usize,
        max_per_expert: usize,
        batch: usize,
    ) -> Result<()>;
}
