//! `MarlinExpertStack<B>` — abstraction for "N MoE experts' Marlin
//! GPTQ-INT4 tiles stored contiguously, dispatched as bucketed batched
//! GEMM or vLLM fused MoE kernel".
//!
//! Phase C sibling to `StackedExpertGgufLinear<B>` (GGUF) and `Linear<B>`
//! (single-tensor). Same goal: drop `type GptqStore` from the `Backend`
//! trait by routing dispatch through a `Box<dyn MarlinExpertStack<B>>`
//! returned by the loader — so future backends only need to implement
//! this trait, not edit the `Backend` supertrait stack.
//!
//! Concrete impls (added in Phase C step 2):
//!   - `quant_linear::cuda_marlin_stack::CudaMarlinExpertStack` wraps
//!     `Arc<GptqStoreCuda>` and dispatches to `marlin_gemm_with_offset_strided`
//!     (bucketed) or `marlin_moe_wna16` (vLLM fused).
//!   - CPU dequant path stays per-Linear (no batched MoE Marlin kernel).
//!
//! The trait surface is intentionally small — three GEMM methods + a
//! workspace zero + an expert-view constructor. Each maps 1:1 to an
//! existing `Backend::moe_gemm_phase_*` method that Phase C step 3
//! will delete from the trait.

use crate::backend::Backend;
use crate::Linear;
use ferrum_types::Result;
use std::sync::Arc;

/// MoE-stacked Marlin INT4 expert tile: holds N experts' weights for one
/// matmul role (gate_up / down) in one contiguous repacked Marlin buffer,
/// dispatches per-expert column-slice GEMMs in a single fused launch
/// (vLLM marlin_moe_wna16) or as a bucketed batched call.
pub trait MarlinExpertStack<B: Backend>: Send + Sync {
    /// Per-expert output width (N tile cols).
    fn n_per_expert(&self) -> usize;
    /// Input width (K), common across experts.
    fn k(&self) -> usize;
    /// Number of experts packed into the tile.
    fn num_experts(&self) -> usize;

    /// Downcast hook — used at FFN dispatch boundaries where the
    /// caller needs to reach into the concrete store to e.g. share
    /// workspace memory across phases. Standard `dyn Any` pattern.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Bulk-zero the per-expert Marlin workspace mutex slots. Call ONCE
    /// before a batch of bucketed `gemm_phase_batched` calls — saves
    /// the per-call cuMemsetD32Async (one launch each → one launch
    /// total). At c=32 with 128 active experts × 2 phases × 48 layers
    /// that's ~12k memset launches/token reduced to ~96.
    fn zero_workspace(&self, ctx: &mut B::Context) -> Result<()>;

    /// Batched per-expert offset GEMM. `dispatches[i] =
    /// (expert_idx, in_row_offset, out_row_offset, m)`. Runs each
    /// expert's `(m × K) @ tile[expert] = m × n_per_expert` slice;
    /// CUDA backend overlaps via multi-stream round-robin.
    #[allow(clippy::too_many_arguments)]
    fn gemm_phase_batched(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        dispatches: &[(usize, usize, usize, usize)],
        output: &mut B::Buffer,
        k: usize,
    ) -> Result<()>;

    /// vLLM `marlin_moe_wna16` fused GEMM (single launch, per-block
    /// expert routing inside the kernel). Caller responsibilities:
    /// - `output` MUST be pre-zeroed (atomic-add path doesn't self-zero).
    /// - `sorted_token_ids` / `expert_ids` / `num_tokens_past_padded`
    ///   come from `moe_align_block_size`.
    /// - `prob_m` is the unique-token count (top_k=1 with pre-gathered
    ///   rows ⇒ equals `total_pairs`).
    /// Backends without vLLM Marlin return `Err(unsupported)`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_phase_vllm(
        &self,
        _ctx: &mut B::Context,
        _input: &B::Buffer,
        _sorted_token_ids: &B::Buffer,
        _expert_ids: &B::Buffer,
        _num_tokens_past_padded: &B::Buffer,
        _output: &mut B::Buffer,
        _prob_m: usize,
        _moe_block_size: usize,
        _top_k: usize,
    ) -> Result<()> {
        Err(ferrum_types::FerrumError::unsupported(
            "MarlinExpertStack::gemm_phase_vllm not implemented for this backend",
        ))
    }

    /// Build a single-expert `Linear<B>` view onto this stack's
    /// `[expert_offset .. expert_offset + expert_n)` column slice.
    /// Used for per-expert dispatch outside the MoE phase batching
    /// (e.g. shared-experts code paths). `expert_offset` and `expert_n`
    /// MUST be multiples of the backend's Marlin N tile (64 on CUDA).
    fn make_expert_linear(
        self: Arc<Self>,
        expert_offset: usize,
        expert_n: usize,
        bias_host: Option<&[f32]>,
    ) -> Result<Box<dyn Linear<B> + Send + Sync>>;
}
