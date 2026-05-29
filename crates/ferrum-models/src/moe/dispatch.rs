//! Expert dispatch — load per-layer expert weights from a GGUF file and run
//! the per-token MoE forward (top-K experts per token, weighted combine).
//!
//! Phase 2 ships a CPU-only implementation (`moe_forward_cpu`). The
//! algorithm is:
//!
//! ```text
//! for each token b in batch:
//!     route token b → (expert_ids[K], weights[K])
//!     out[b] = 0
//!     for each (expert_id, weight) pair:
//!         gate_up = experts.gate_up[expert_id].forward(x[b])     # [2*ffn]
//!         silu_mul = silu(gate_up[..ffn]) * gate_up[ffn..]       # [ffn]
//!         contribution = experts.down[expert_id].forward(silu_mul) # [hidden]
//!         out[b] += weight * contribution
//! ```
//!
//! The fused `gate || up` per-expert layout means we can call
//! `Backend::fused_silu_mul_split` directly on the projection's output
//! — same kernel ferrum already uses for dense Llama-family models.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::quantized::GgmlDType;
use candle_core::{Device, Result as CandleResult};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::{
    Backend, BackendMoeFused, BackendPagedKv, BackendQuantGguf, BackendQuantMarlin, GgufQuantType,
    LlmBackend, QuantLlmBackend,
};
use ferrum_kernels::{Linear, StackedExpertGgufLinear};
use ferrum_quantization::gguf::GgufFile;
use ferrum_quantization::{DenseLinear, QuantLinear};
use ferrum_types::{FerrumError, Result};

use crate::moe::router::RouterOutput;

/// MoE per-op timers. Public so the model wrapper can drain + print at
/// end of decode. Times are in microseconds, atomically accumulated.
/// Toggle via env `FERRUM_MOE_PROFILE=1`.
pub static MOE_SYNC_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_SYNC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_GATE_UP_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_GATE_UP_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_SILU_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_SILU_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_GEMV_DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_SCALED_ADD_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_SCALED_ADD_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_COPY_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_COPY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MOE_HOST_TOPK_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_HOST_TOPK_CALLS: AtomicU64 = AtomicU64::new(0);

// Bucketed-path per-phase timers. Drained by the model wrapper alongside
// the per-pair counters above. Same `FERRUM_MOE_PROFILE=1` gate.
pub static MOE_BUCKET_SYNC_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_D2H_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_ROUTE_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_PLAN_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_GATHER_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_GEMM1_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_SILU_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_GEMM3_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_COMBINE_US: AtomicU64 = AtomicU64::new(0);
pub static MOE_BUCKET_LAYER_CALLS: AtomicU64 = AtomicU64::new(0);

fn moe_profile_enabled() -> bool {
    std::env::var("FERRUM_MOE_PROFILE").is_ok()
}

fn vllm_moe_zero_workspace_enabled() -> bool {
    std::env::var("FERRUM_VLLM_MOE_ZERO_WS").map_or(false, |v| v == "1")
}

fn vllm_moe_pair_ids_enabled() -> bool {
    std::env::var("FERRUM_VLLM_MOE_PAIR_IDS").map_or(false, |v| v == "1")
}

/// Per-layer expert weights, materialised as `[num_experts]`-long vectors
/// of `Box<dyn Linear<B>>`. Each entry runs the corresponding expert's
/// fused `[gate; up]` projection or its `down` projection.
///
/// `B::Buffer` is hidden behind `Linear<B>` so this struct is generic
/// over backend. Production (`Qwen3MoeModel::forward`) dispatches through
/// the generic [`moe_forward<B>`] (this file, line ~960) and
/// [`moe_forward_bucketed<B>`]; the CPU-only `moe_forward_cpu` is the
/// reference path used by parity tests + `Qwen3MoeLayer::forward_cpu`.
pub struct ExpertStack<B: QuantLlmBackend + BackendMoeFused> {
    /// Fused `[gate; up]` projection per expert. Output shape per token:
    /// `[2 * expert_intermediate]` — the lower half is gate, upper is up.
    pub gate_up: Vec<Box<dyn Linear<B>>>,
    /// `down` projection per expert. Output shape per token: `[hidden_size]`.
    pub down: Vec<Box<dyn Linear<B>>>,
    /// Stacked-experts representation for backends that have a batched
    /// MoE indirect-dispatch kernel (Metal `gemv_q4kw_moe_id_f32` /
    /// `gemv_q6kw_moe_id_f32`). Holds **all experts** for one matmul
    /// role behind a `StackedExpertGgufLinear<B>` (typically backed by a
    /// single GPU buffer with byte stride between expert slabs), so a
    /// single dispatch can cover all selected (token, expert) pairs at
    /// decode m=1.
    ///
    /// `None` on backends without the kernel (CPU, CUDA-without-MoE-kernel)
    /// and on quant flavours that don't have a stacked path yet — callers
    /// fall back to the per-expert `gate_up` / `down` Linears in those
    /// cases.
    pub gate_stacked: Option<Box<dyn StackedExpertGgufLinear<B>>>,
    pub up_stacked: Option<Box<dyn StackedExpertGgufLinear<B>>>,
    pub down_stacked: Option<Box<dyn StackedExpertGgufLinear<B>>>,

    /// Stacked Marlin GPTQ expert tiles for the bucketed CUDA path.
    /// When both are Some, [`moe_forward_bucketed`] dispatches expert
    /// GEMMs through trait-object methods (`store.gemm_phase_*` /
    /// `store.zero_workspace`). None on CPU / Metal / GGUF.
    ///
    /// Phase C step 3: replaces `Option<Arc<B::GptqStore>>` with a
    /// `Box<dyn MarlinExpertStack<B>>` trait object — kills the
    /// `type GptqStore` leak through the model layer.
    pub gate_up_marlin_stack: Option<std::sync::Arc<dyn ferrum_kernels::MarlinExpertStack<B>>>,
    pub down_marlin_stack: Option<std::sync::Arc<dyn ferrum_kernels::MarlinExpertStack<B>>>,
}

impl<B: QuantLlmBackend + BackendMoeFused> ExpertStack<B> {
    /// Returns the shared stacked Marlin expert tile for `gate_up` if
    /// loaded via the bucketed/Marlin path. Used by
    /// [`moe_forward_bucketed`].
    pub fn gate_up_stacked_store(
        &self,
        _expert_idx: usize,
    ) -> Option<&std::sync::Arc<dyn ferrum_kernels::MarlinExpertStack<B>>> {
        self.gate_up_marlin_stack.as_ref()
    }

    /// Same for `down`.
    pub fn down_stacked_store(
        &self,
        _expert_idx: usize,
    ) -> Option<&std::sync::Arc<dyn ferrum_kernels::MarlinExpertStack<B>>> {
        self.down_marlin_stack.as_ref()
    }

    // ── MoE GEMV dispatch (hides B::QuantStore + in_stride from callers) ──
    //
    // These wrap `B::gemv_quant_moe_id*` so the MoE forward path goes
    // through the ExpertStack abstraction instead of reaching into
    // `self.gate_stacked` / `self.up_stacked` / `self.down_stacked`
    // directly. The weight + correct in_stride are picked from `self`,
    // so callers only pass activations + routing + scratch out.

    /// Gate projection: `out_stacked[k] = gate_weight[expert_id[k]] · input`,
    /// broadcast input across all top_k slots.
    pub fn gemv_gate(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        out: &mut B::Buffer,
        top_k: usize,
    ) -> Result<()> {
        let weight = self.gate_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_gate: gate_stacked not loaded")
        })?;
        weight.gemv_moe_id(ctx, input, ids, out, top_k, 0)
    }

    /// Up projection: same shape as gate, broadcast input.
    pub fn gemv_up(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        out: &mut B::Buffer,
        top_k: usize,
    ) -> Result<()> {
        let weight = self.up_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_up: up_stacked not loaded")
        })?;
        weight.gemv_moe_id(ctx, input, ids, out, top_k, 0)
    }

    /// Down projection: per-slot input via `in_stride = expert_intermediate`.
    /// Caller's `input` is the SiLU-mul stacked output (`top_k × inter` floats).
    pub fn gemv_down(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        out: &mut B::Buffer,
        top_k: usize,
        expert_intermediate: usize,
    ) -> Result<()> {
        let weight = self.down_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_down: down_stacked not loaded")
        })?;
        weight.gemv_moe_id(ctx, input, ids, out, top_k, expert_intermediate)
    }

    /// Fused gate + up + SiLU·gate: replaces 3 separate dispatches with 1.
    /// Backend must support the fused path
    /// (`B::supports_fused_moe_gate_up_silu()`); caller checks first.
    pub fn gemv_gate_up_silu_fused(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        out_silu_stacked: &mut B::Buffer,
        top_k: usize,
    ) -> Result<()> {
        let gate = self.gate_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported(
                "ExpertStack::gemv_gate_up_silu_fused: gate_stacked not loaded",
            )
        })?;
        let up = self.up_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_gate_up_silu_fused: up_stacked not loaded")
        })?;
        gate.gemv_moe_id_gate_up_silu(ctx, input, up, ids, out_silu_stacked, top_k)
    }

    // ── Prefill GEMM dispatch (Phase 3d) ──
    //
    // Same role as the gemv wrappers above, but for the m>1 path that
    // emits batched mul_mm_id instead of per-pair gemv. `args_buf`
    // toggles between direct (`gemm_quant_moe_id`) and indirect-grid
    // (`gemm_quant_moe_id_indirect`) dispatch — the indirect form lets
    // `compute_ids_tpe_gpu` produce a tighter grid sized to `max(tpe[e])`.
    //
    // ne11 is fixed by role: gate/up = 1 (broadcast across slots),
    // down = top_k (per-slot src1 read). Callers no longer pass it.

    /// Gate prefill GEMM. `dst` shape: `[batch, top_k, expert_inter]`.
    /// `args_buf=Some` triggers indirect-grid dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        ids: &B::Buffer,
        tpe: &B::Buffer,
        dst: &mut B::Buffer,
        args_buf: Option<&B::Buffer>,
        top_k: usize,
        max_per_expert: usize,
        tokens: usize,
    ) -> Result<()> {
        let weight = self.gate_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemm_gate: gate_stacked not loaded")
        })?;
        match args_buf {
            Some(args) => weight.gemm_moe_id_indirect(
                ctx,
                src1,
                ids,
                tpe,
                dst,
                args,
                1,
                top_k,
                max_per_expert,
                tokens,
            ),
            None => weight.gemm_moe_id(ctx, src1, ids, tpe, dst, 1, top_k, max_per_expert, tokens),
        }
    }

    /// Up prefill GEMM. Same shape contract as [`Self::gemm_gate`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_up(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        ids: &B::Buffer,
        tpe: &B::Buffer,
        dst: &mut B::Buffer,
        args_buf: Option<&B::Buffer>,
        top_k: usize,
        max_per_expert: usize,
        tokens: usize,
    ) -> Result<()> {
        let weight = self.up_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemm_up: up_stacked not loaded")
        })?;
        match args_buf {
            Some(args) => weight.gemm_moe_id_indirect(
                ctx,
                src1,
                ids,
                tpe,
                dst,
                args,
                1,
                top_k,
                max_per_expert,
                tokens,
            ),
            None => weight.gemm_moe_id(ctx, src1, ids, tpe, dst, 1, top_k, max_per_expert, tokens),
        }
    }

    /// Down prefill GEMM. `dst` shape: `[batch, top_k, hidden]`.
    /// ne11=top_k (per-slot src1 read from `silu_stacked[batch, top_k, inter]`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_down(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        ids: &B::Buffer,
        tpe: &B::Buffer,
        dst: &mut B::Buffer,
        args_buf: Option<&B::Buffer>,
        top_k: usize,
        max_per_expert: usize,
        tokens: usize,
    ) -> Result<()> {
        let weight = self.down_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemm_down: down_stacked not loaded")
        })?;
        match args_buf {
            Some(args) => weight.gemm_moe_id_indirect(
                ctx,
                src1,
                ids,
                tpe,
                dst,
                args,
                top_k,
                top_k,
                max_per_expert,
                tokens,
            ),
            None => weight.gemm_moe_id(
                ctx,
                src1,
                ids,
                tpe,
                dst,
                top_k,
                top_k,
                max_per_expert,
                tokens,
            ),
        }
    }

    // ── Batched-decode GEMV dispatch (Phase 3d) ──
    //
    // For the small-m batched-decode range (c=2..32). Single Metal
    // launch covers all m*top_k (token, expert) pairs.

    /// Gate batched gemv: `dst[m * top_k]` with broadcast input
    /// (slots within a token share the activation row).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_gate_batched(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        dst: &mut B::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()> {
        let weight = self.gate_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_gate_batched: gate_stacked not loaded")
        })?;
        weight.gemv_moe_id_batched(
            ctx,
            input,
            ids,
            dst,
            m,
            top_k,
            src1_outer_stride,
            src1_inner_stride,
        )
    }

    /// Up batched gemv: same shape as [`Self::gemv_gate_batched`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_up_batched(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        dst: &mut B::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()> {
        let weight = self.up_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_up_batched: up_stacked not loaded")
        })?;
        weight.gemv_moe_id_batched(
            ctx,
            input,
            ids,
            dst,
            m,
            top_k,
            src1_outer_stride,
            src1_inner_stride,
        )
    }

    /// Down batched gemv: src1 = `silu_stacked[m, top_k, inter]` per-slot read.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_down_batched(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        dst: &mut B::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()> {
        let weight = self.down_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_down_batched: down_stacked not loaded")
        })?;
        weight.gemv_moe_id_batched(
            ctx,
            input,
            ids,
            dst,
            m,
            top_k,
            src1_outer_stride,
            src1_inner_stride,
        )
    }

    /// Fused batched gate + up + SiLU·gate. Single dispatch over `m * top_k`
    /// pairs. Caller gates on `B::supports_batched_moe_gate_up_silu()` first.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_gate_up_silu_batched_fused(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        ids: &B::Buffer,
        silu_out: &mut B::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()> {
        let gate = self.gate_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported(
                "ExpertStack::gemv_gate_up_silu_batched_fused: gate_stacked not loaded",
            )
        })?;
        let up = self.up_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported(
                "ExpertStack::gemv_gate_up_silu_batched_fused: up_stacked not loaded",
            )
        })?;
        gate.gemv_moe_id_gate_up_silu_batched(
            ctx,
            input,
            up,
            ids,
            silu_out,
            m,
            top_k,
            src1_outer_stride,
            src1_inner_stride,
        )
    }

    // ── Per-item offset GEMV (Phase 3d, qwen3_moe.rs decode path) ──
    //
    // Used by the per-item batched-decode loop in `Qwen3MoeModel::forward`
    // when offset variants are supported. Reads `src1` at `src1_offset`
    // floats and `ids` at `ids_offset` ids, writes `dst` from offset 0.

    /// Gate offset gemv. `src1_stride=0` → broadcast.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_gate_offset(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        src1_offset: usize,
        ids: &B::Buffer,
        ids_offset: usize,
        dst: &mut B::Buffer,
        top_k: usize,
        src1_stride: usize,
    ) -> Result<()> {
        let weight = self.gate_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_gate_offset: gate_stacked not loaded")
        })?;
        weight.gemv_moe_id_offset(
            ctx,
            src1,
            src1_offset,
            ids,
            ids_offset,
            dst,
            top_k,
            src1_stride,
        )
    }

    /// Up offset gemv.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_up_offset(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        src1_offset: usize,
        ids: &B::Buffer,
        ids_offset: usize,
        dst: &mut B::Buffer,
        top_k: usize,
        src1_stride: usize,
    ) -> Result<()> {
        let weight = self.up_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_up_offset: up_stacked not loaded")
        })?;
        weight.gemv_moe_id_offset(
            ctx,
            src1,
            src1_offset,
            ids,
            ids_offset,
            dst,
            top_k,
            src1_stride,
        )
    }

    /// Down offset gemv.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_down_offset(
        &self,
        ctx: &mut B::Context,
        src1: &B::Buffer,
        src1_offset: usize,
        ids: &B::Buffer,
        ids_offset: usize,
        dst: &mut B::Buffer,
        top_k: usize,
        src1_stride: usize,
    ) -> Result<()> {
        let weight = self.down_stacked.as_deref().ok_or_else(|| {
            FerrumError::unsupported("ExpertStack::gemv_down_offset: down_stacked not loaded")
        })?;
        weight.gemv_moe_id_offset(
            ctx,
            src1,
            src1_offset,
            ids,
            ids_offset,
            dst,
            top_k,
            src1_stride,
        )
    }
}

impl<B: QuantLlmBackend + BackendMoeFused> ExpertStack<B> {
    /// Build from raw fp32 stacked tensors (test helper). Caller has
    /// already dequantised and laid out the data:
    ///   `gate_stack`: `[num_experts * expert_inter * hidden]`
    ///   `up_stack`:   `[num_experts * expert_inter * hidden]`
    ///   `down_stack`: `[num_experts * hidden * expert_inter]`
    /// Each per-expert slice is row-major in the natural Linear shape.
    pub fn from_dense_stacks(
        gate_stack: &[f32],
        up_stack: &[f32],
        down_stack: &[f32],
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
        let gate_up_per_expert = expert_intermediate * hidden_size;
        let down_per_expert = hidden_size * expert_intermediate;

        check_size(
            gate_stack.len(),
            num_experts * gate_up_per_expert,
            "gate_stack",
        )?;
        check_size(up_stack.len(), num_experts * gate_up_per_expert, "up_stack")?;
        check_size(
            down_stack.len(),
            num_experts * down_per_expert,
            "down_stack",
        )?;

        let mut gate_up = Vec::with_capacity(num_experts);
        let mut down = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let g_off = e * gate_up_per_expert;
            let g_slice = &gate_stack[g_off..g_off + gate_up_per_expert];
            let u_slice = &up_stack[g_off..g_off + gate_up_per_expert];

            // Fused [gate; up] is [2 * expert_inter, hidden] row-major.
            // We concatenate row-blocks so the first expert_inter rows are
            // gate, the next expert_inter rows are up — the layout
            // fused_silu_mul_split expects.
            let mut fused = Vec::with_capacity(2 * gate_up_per_expert);
            fused.extend_from_slice(g_slice);
            fused.extend_from_slice(u_slice);
            gate_up.push(Box::new(DenseLinear::<B>::from_rows(
                &fused,
                2 * expert_intermediate,
                hidden_size,
            )) as Box<dyn Linear<B>>);

            let d_off = e * down_per_expert;
            let d_slice = &down_stack[d_off..d_off + down_per_expert];
            down.push(Box::new(DenseLinear::<B>::from_rows(
                d_slice,
                hidden_size,
                expert_intermediate,
            )) as Box<dyn Linear<B>>);
        }
        Ok(Self {
            gate_up,
            down,
            gate_stacked: None,
            up_stacked: None,
            down_stacked: None,
            gate_up_marlin_stack: None,
            down_marlin_stack: None,
        })
    }

    /// Load all experts for one MoE layer from a GGUF file. Names follow
    /// the GGUF convention: `blk.{layer_idx}.ffn_{gate,up,down}_exps.weight`.
    ///
    /// The loader picks between two strategies based on the on-disk dtype
    /// of the expert tensors:
    ///
    ///   - **Quantised path** (Q4_K / Q6_K only): each expert's
    ///     `gate || up` becomes a single `QuantLinear<B>` (Fused
    ///     QuantStore — gate + up share `n_cols = hidden`), and `down` is
    ///     a plain `QuantLinear<B>`. Block bytes stay compressed in
    ///     backend memory; per-call dequant happens inside `gemm_quant`.
    ///   - **Dense fallback** (everything else, e.g. F32 / F16 / Q5_K
    ///     until a kernel ships): eager-dequant to fp32 and wrap
    ///     `DenseLinear<B>`. Memory inflates ~7× vs Q4_K_M but the
    ///     algorithm is correctness-equivalent and this is the path the
    ///     synthetic-MoE test fixtures need.
    ///
    /// The runtime dispatcher (`moe_forward<B>`) doesn't see which path
    /// was taken — it just calls `Linear::forward` per (token, expert).
    pub fn load_from_gguf(
        gguf: &GgufFile,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
        if let Some(quant) = Self::try_load_quantised(
            gguf,
            layer_idx,
            num_experts,
            hidden_size,
            expert_intermediate,
        )? {
            if std::env::var("FERRUM_MOE_LOAD_TRACE").is_ok() {
                eprintln!("[moe-load] layer {layer_idx} → quantised expert path");
            }
            return Ok(quant);
        }

        if std::env::var("FERRUM_MOE_LOAD_TRACE").is_ok() {
            eprintln!("[moe-load] layer {layer_idx} → eager fp32 dense fallback ⚠");
        }

        let device = Device::Cpu;
        let gate = read_dequant_flat(
            gguf,
            &format!("blk.{layer_idx}.ffn_gate_exps.weight"),
            &device,
        )?;
        let up = read_dequant_flat(
            gguf,
            &format!("blk.{layer_idx}.ffn_up_exps.weight"),
            &device,
        )?;
        let down = read_dequant_flat(
            gguf,
            &format!("blk.{layer_idx}.ffn_down_exps.weight"),
            &device,
        )?;
        // Eager-dense path leaves stacked variants as None — no MoE
        // fast path for synthesised / non-quantised expert tensors.
        Self::from_dense_stacks(
            &gate,
            &up,
            &down,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
    }

    /// Attempt the quantised path. Returns `Ok(None)` if any of the three
    /// tensors isn't a supported k-quant flavour (Q4_K / Q6_K) or if the
    /// shape doesn't match the expected per-expert tile size — caller
    /// then takes the eager-dequant fallback. Returns `Err` only on a
    /// genuine load failure (missing tensor, byte-count mismatch).
    fn try_load_quantised(
        gguf: &GgufFile,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Option<Self>> {
        let device = Device::Cpu;

        let gate_name = format!("blk.{layer_idx}.ffn_gate_exps.weight");
        let up_name = format!("blk.{layer_idx}.ffn_up_exps.weight");
        let down_name = format!("blk.{layer_idx}.ffn_down_exps.weight");

        // Inspect tensor info up front — if any tensor isn't a k-quant
        // flavour the backend can dispatch on, bail to the dense path
        // before paying the byte-read cost.
        let gate_kind = match quant_kind(gguf, &gate_name)? {
            Some(k) => k,
            None => return Ok(None),
        };
        let up_kind = match quant_kind(gguf, &up_name)? {
            Some(k) => k,
            None => return Ok(None),
        };
        let down_kind = match quant_kind(gguf, &down_name)? {
            Some(k) => k,
            None => return Ok(None),
        };

        // Slice the three 3-D quantised expert stacks directly from
        // the mmap. These are the dominant memory cost on Qwen3-MoE
        // (~14 GB for Qwen3-30B-A3B); going through candle's
        // `read_tensor` would copy them into a heap `Vec<u8>` first,
        // then `load_quant_experts` would copy again into the Metal
        // buffer — together doubling the working set and pushing a
        // 32 GB Mac into swap. With this slice + the Metal mmap
        // registry, we avoid both copies (steady state: just the
        // file mmap).
        let gate_bytes = gguf.tensor_byte_slice(&gate_name).ok_or_else(|| {
            FerrumError::model(format!("MoE: tensor_byte_slice failed for '{gate_name}'"))
        })?;
        let up_bytes = gguf.tensor_byte_slice(&up_name).ok_or_else(|| {
            FerrumError::model(format!("MoE: tensor_byte_slice failed for '{up_name}'"))
        })?;
        let down_bytes = gguf.tensor_byte_slice(&down_name).ok_or_else(|| {
            FerrumError::model(format!("MoE: tensor_byte_slice failed for '{down_name}'"))
        })?;
        let _ = device; // candle device no longer needed for the byte read

        // Per-expert byte stride for each tensor. The 3-D layout is
        // contiguous, [num_experts, rows, cols] row-major, so each
        // expert's slab is exactly `total_bytes / num_experts`.
        let gate_per = block_bytes_for(
            gate_kind,
            expert_intermediate * hidden_size,
            "ffn_gate_exps",
        )?;
        let up_per = block_bytes_for(up_kind, expert_intermediate * hidden_size, "ffn_up_exps")?;
        let down_per = block_bytes_for(
            down_kind,
            hidden_size * expert_intermediate,
            "ffn_down_exps",
        )?;

        check_size(
            gate_bytes.len(),
            num_experts * gate_per,
            "ffn_gate_exps bytes",
        )?;
        check_size(up_bytes.len(), num_experts * up_per, "ffn_up_exps bytes")?;
        check_size(
            down_bytes.len(),
            num_experts * down_per,
            "ffn_down_exps bytes",
        )?;

        // Try the stacked-experts fast path FIRST. If the backend has a
        // batched MoE kernel (Metal `gemv_q*kw_moe_id_f32`), we want to
        // hold the experts only as one big stacked buffer per role —
        // not as 128 per-expert MetalQuantStores PLUS the stacked one
        // (that would double-allocate ~17 GB on a 32 GB Mac, which on
        // Qwen3-30B-A3B Q4_K_M sends the model into swap and tanks
        // both load and forward time).
        let gate_stacked = B::load_quant_experts(
            gate_kind,
            gate_bytes,
            num_experts,
            expert_intermediate,
            hidden_size,
        )
        .ok();
        let up_stacked = B::load_quant_experts(
            up_kind,
            up_bytes,
            num_experts,
            expert_intermediate,
            hidden_size,
        )
        .ok();
        let down_stacked = B::load_quant_experts(
            down_kind,
            down_bytes,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
        .ok();

        // Decide the storage shape:
        //   * Stacked-only (Metal MoE fast path): all three stacked
        //     loaders succeeded — skip per-expert and use stacked
        //     for both decode and prefill. Cuts memory in half.
        //   * Per-expert: stacked path is incomplete or unsupported —
        //     load 128-per-layer QuantLinears and let `moe_forward`
        //     drive the per-(token, expert) loop on top of them.
        let stacked_complete =
            gate_stacked.is_some() && up_stacked.is_some() && down_stacked.is_some();

        let (gate_up, down) = if stacked_complete {
            // No per-expert needed — `moe_forward_stacked_decode_impl`
            // and the per-token prefill loop both use the stacked buffers.
            (Vec::new(), Vec::new())
        } else {
            let mut gate_up: Vec<Box<dyn Linear<B>>> = Vec::with_capacity(num_experts);
            let mut down: Vec<Box<dyn Linear<B>>> = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let g_slice = &gate_bytes[e * gate_per..(e + 1) * gate_per];
                let u_slice = &up_bytes[e * up_per..(e + 1) * up_per];
                let d_slice = &down_bytes[e * down_per..(e + 1) * down_per];

                let parts: [(GgufQuantType, &[u8], usize); 2] = [
                    (gate_kind, g_slice, expert_intermediate),
                    (up_kind, u_slice, expert_intermediate),
                ];
                let gate_up_e = match QuantLinear::<B>::from_gguf_fused(&parts, hidden_size) {
                    Ok(q) => q,
                    Err(_) => return Ok(None),
                };
                gate_up.push(Box::new(gate_up_e) as Box<dyn Linear<B>>);

                let down_e = match QuantLinear::<B>::from_gguf_bytes(
                    down_kind,
                    d_slice,
                    hidden_size,
                    expert_intermediate,
                ) {
                    Ok(q) => q,
                    Err(_) => return Ok(None),
                };
                down.push(Box::new(down_e) as Box<dyn Linear<B>>);
            }
            (gate_up, down)
        };

        Ok(Some(Self {
            gate_up,
            down,
            gate_stacked,
            up_stacked,
            down_stacked,
            gate_up_marlin_stack: None,
            down_marlin_stack: None,
        }))
    }

    /// Convenience: open a GGUF and load layer `layer_idx`. The GGUF
    /// stays open inside this call only — for multi-layer loads use
    /// [`Self::load_from_gguf`] with a shared [`GgufFile`].
    pub fn open_and_load(
        path: impl AsRef<Path>,
        layer_idx: usize,
        num_experts: usize,
        hidden_size: usize,
        expert_intermediate: usize,
    ) -> Result<Self> {
        let gguf = GgufFile::open(path).map_err(candle_to_ferrum)?;
        Self::load_from_gguf(
            &gguf,
            layer_idx,
            num_experts,
            hidden_size,
            expert_intermediate,
        )
    }

    /// `num_experts` for the layer (consistency check helper).
    ///
    /// Returns the per-expert Vec length, OR — when the stacked-only
    /// path is in effect (Metal MoE fast path with empty per-expert
    /// Vecs) — falls back to a stored count via the stacked variants.
    /// In the stacked-only case there's no Vec to count, so this method
    /// is mostly used by tests on the per-expert path.
    pub fn num_experts(&self) -> usize {
        debug_assert_eq!(
            self.gate_up.len(),
            self.down.len(),
            "ExpertStack: gate_up and down disagree on expert count"
        );
        self.gate_up.len()
    }
}

/// Backend-generic MoE forward.
///
/// Equivalent of [`moe_forward_cpu`] but parameterised on `B: Backend`
/// so Metal / CUDA paths can dispatch the same per-(token, expert) loop
/// using their own kernels for the gemv + silu + scaled-add primitives.
///
/// The caller pre-supplies all scratch buffers — this function does no
/// allocation, which matters because it's invoked from inside the
/// transformer's `forward_layer` where allocation during graph capture
/// (CUDA) would corrupt the captured graph.
///
/// Buffer contract (lengths, sized at scratch alloc time):
///   - `x`            : `[batch * hidden]` post-RMSNorm activations
///   - `router_logits`: `[batch * num_experts]` raw router output
///   - `out`          : `[batch * hidden]` — caller is responsible for
///                      zeroing this before the call (we accumulate,
///                      not assign)
///   - `x_single`     : `[hidden]` per-token input slice
///   - `acc_buf`      : `[hidden]` per-token output accumulator (kept
///                      separate from `x_single` so the gate_up gemv
///                      can consume `x_single` repeatedly across the
///                      top_k loop without an inter-pair restore)
///   - `gate_up_buf`  : `[2 * expert_inter]` per-(token, expert) gemv out
///   - `silu_buf`     : `[expert_inter]`
///   - `down_buf`     : `[hidden]` per-(token, expert) accumulate src
///
/// Routing (softmax + top-K + optional renorm) runs host-side using
/// `B::to_vec(router_logits, …)` — the routing computation is small
/// (`batch * num_experts` floats) and the top-K is a sort, both of
/// which dwarf in cost any plausible host↔device transfer.
///
/// Per-pair dispatch budget (m=1, Metal):
///   gate_up Fused gemv (2 parts) + silu + down gemv + scaled_add
///   = 5 dispatches/pair. Plus 2 copy_slice/token (load x_single,
///   write acc_buf back to out[b]). With top_k=8 and 48 layers, that's
///   8×5 + 2 = 42 dispatches/layer × 48 ≈ 2k/token (vs. ~3.5k in the
///   pre-PR scheme that round-tripped through `out` per pair).
#[allow(clippy::too_many_arguments)]
pub fn moe_forward<B: QuantLlmBackend + BackendMoeFused>(
    ctx: &mut B::Context,
    x: &B::Buffer,
    router_logits: &B::Buffer,
    out: &mut B::Buffer,
    batch: usize,
    hidden_size: usize,
    expert_intermediate: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    experts: &ExpertStack<B>,
    x_single: &mut B::Buffer,
    acc_buf: &mut B::Buffer,
    gate_up_buf: &mut B::Buffer,
    silu_buf: &mut B::Buffer,
    down_buf: &mut B::Buffer,
    zero_hidden: &B::Buffer,
) -> Result<()> {
    let n_experts = experts.num_experts();
    if n_experts != num_experts {
        return Err(FerrumError::model(format!(
            "moe_forward: experts.num_experts() = {n_experts} != cfg.num_experts = {num_experts}"
        )));
    }

    let prof = moe_profile_enabled();

    // Routing on host. Sized batch*num_experts (e.g. 512*128 = 64k floats
    // per layer for Qwen3-30B-A3B prefill); cheap relative to the per-
    // expert gemvs that follow.
    let t0 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    B::sync(ctx);
    if let Some(t) = t0 {
        MOE_SYNC_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        MOE_SYNC_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    let t0 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let logits_host = B::to_vec(router_logits, batch * num_experts);
    let route_out =
        crate::moe::router::route(&logits_host, batch, num_experts, top_k, norm_topk_prob);
    if let Some(t) = t0 {
        MOE_HOST_TOPK_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        MOE_HOST_TOPK_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    for b in 0..batch {
        // Load x[b] into x_single + reset accumulator.
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::copy_slice(ctx, x, b * hidden_size, x_single, 0, hidden_size);
        B::copy_slice(ctx, zero_hidden, 0, acc_buf, 0, hidden_size);
        if let Some(t) = t0 {
            MOE_COPY_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            MOE_COPY_CALLS.fetch_add(2, Ordering::Relaxed);
        }

        for k in 0..top_k {
            let pair = b * top_k + k;
            let expert_id = route_out.expert_ids[pair] as usize;
            let weight = route_out.expert_weights[pair];
            if expert_id >= num_experts {
                return Err(FerrumError::model(format!(
                    "moe_forward: routed expert {expert_id} >= num_experts {num_experts}"
                )));
            }

            // Fused gate||up gemv → [2 * expert_inter]
            let t0 = if prof {
                B::sync(ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            experts.gate_up[expert_id].forward(ctx, x_single, gate_up_buf, 1);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_GEMV_GATE_UP_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_GEMV_GATE_UP_CALLS.fetch_add(1, Ordering::Relaxed);
            }

            // SiLU(gate) * up → [expert_inter]
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            B::fused_silu_mul_split(ctx, gate_up_buf, silu_buf, 1, expert_intermediate);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_SILU_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_SILU_CALLS.fetch_add(1, Ordering::Relaxed);
            }

            // down gemv → [hidden]
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            experts.down[expert_id].forward(ctx, silu_buf, down_buf, 1);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_GEMV_DOWN_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_GEMV_DOWN_CALLS.fetch_add(1, Ordering::Relaxed);
            }

            // acc_buf += weight * down_buf
            let t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            B::scaled_add_inplace(ctx, acc_buf, down_buf, weight, hidden_size);
            if let Some(t) = t0 {
                B::sync(ctx);
                MOE_SCALED_ADD_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                MOE_SCALED_ADD_CALLS.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Final write: out[b] = acc_buf
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::copy_slice(ctx, acc_buf, 0, out, b * hidden_size, hidden_size);
        if let Some(t) = t0 {
            MOE_COPY_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            MOE_COPY_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    Ok(())
}

/// Largest moe_block_size we'd ever pick. Drives Qwen3MoeScratch
/// `route_sorted_tokens_dev` sizing (allocates `t*top_k + n_exp*MAX`).
pub const MOE_BLOCK_SIZE_MAX: usize = 64;

/// Pick `moe_block_size` ∈ {16, 32, 64} based on routing distribution.
///
/// Marlin-MoE templates instantiate for thread_m_blocks ∈ {1, 2, 3, 4}
/// → block_size ∈ {16, 32, 48, 64}. Larger block_size enables the
/// "large_batch tile" path (thread_n=256, num_threads=256, 8× more work
/// per kernel launch) but pads each expert's tokens up to a multiple of
/// block_size — sparse routing wastes most of that.
///
/// Decision: pick the largest size whose **total padded tokens** stays
/// within 30% of actual. If we can't keep overhead below the threshold,
/// stick with 16. Skips block_size=48 for simplicity (rare sweet spot).
///
/// Device-routing path doesn't expose `plan` host-side; fall back to 16
/// (no regression vs pre-PR behaviour).
fn pick_moe_block_size(
    plan: Option<&MoeBucketPlan>,
    num_experts: usize,
    use_device_route: bool,
) -> usize {
    const CANDIDATES: &[usize] = &[64, 32, 16];
    const PADDING_BUDGET: f64 = 1.30; // ≤ 30% overhead vs actual tokens
                                      // Manual override (testing / autotuning): FERRUM_MOE_BLOCK_SIZE=8/16/32/48/64.
                                      // vLLM 0.20.2 often selects 8 for small-M MoE; keep it override-only
                                      // until full-model correctness + throughput beats the 16 default.
    if let Some(bs) = std::env::var("FERRUM_MOE_BLOCK_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|bs| matches!(*bs, 8 | 16 | 32 | 48 | 64))
    {
        return bs;
    }
    if use_device_route {
        // Empirical 2026-05-13: block_size=64 (`thread_m_blocks=4`,
        // matching vLLM's tile) regresses M3 c=32 by 5.7% on RTX 4090
        // because sparse routing (top_k=8 / num_experts=128 / m=32 ≈
        // 2 pairs per active expert) pads each expert's tile by ~32×,
        // and the wasted sentinel-row compute exceeds the tile-width
        // win. block_size=32 is within noise of 16. Keep 16 as default;
        // FERRUM_MOE_BLOCK_SIZE override stays for future autotuning
        // when m / routing density changes (e.g. dense Llama at m=32).
        return 16;
    }
    let Some(plan) = plan else {
        return 16;
    };
    let m_e: Vec<usize> = (0..num_experts)
        .map(|e| plan.expert_offsets[e + 1] - plan.expert_offsets[e])
        .collect();
    let total_actual: usize = m_e.iter().sum();
    if total_actual == 0 {
        return 16;
    }
    for &bs in CANDIDATES {
        let total_padded: usize = m_e.iter().map(|&m| m.div_ceil(bs) * bs).sum();
        if (total_padded as f64) <= (total_actual as f64) * PADDING_BUDGET {
            return bs;
        }
    }
    16
}

/// Bucket plan: per-expert lists of which (token, k_slot) pairs route
/// through that expert. Built host-side from the router output and used
/// by [`moe_forward_bucketed`] to issue ONE m=tokens_per_expert Marlin
/// GEMM per active expert instead of `batch * top_k` m=1 GEMMs.
pub struct MoeBucketPlan {
    /// `expert_offsets[e+1] - expert_offsets[e]` = tokens routed to expert e.
    /// Length: `num_experts + 1`. `expert_offsets[num_experts]` = total_pairs
    /// (always `batch * top_k`).
    pub expert_offsets: Vec<usize>,
    /// `[total_pairs]` flat: which input token each packed-row gathers
    /// from. Index into `x[batch, hidden]`.
    pub packed_token_idx: Vec<u32>,
    /// `[batch, top_k]` row-major: for each (b, k_slot), which row of the
    /// packed buffers carries that pair's contribution. Used by
    /// `B::moe_combine` to scatter weighted sums back to `out[b]`.
    pub pairs_by_token: Vec<i32>,
    /// `[batch, top_k]` row-major: combine weight for the (b, k_slot)
    /// pair, copied verbatim from the router output. Used by
    /// `B::moe_combine`.
    pub pair_weights: Vec<f32>,
    /// Cached cursor scratch for [`Self::rebuild_into`] — sized to
    /// `num_experts` on first build and reused (one alloc total instead
    /// of one per call).
    cursors: Vec<usize>,
}

impl MoeBucketPlan {
    /// Empty plan with no allocation. Use [`Self::rebuild_into`] before
    /// reuse — this is the cheap constructor for putting the plan in a
    /// scratch struct.
    pub fn empty() -> Self {
        Self {
            expert_offsets: Vec::new(),
            packed_token_idx: Vec::new(),
            pairs_by_token: Vec::new(),
            pair_weights: Vec::new(),
            cursors: Vec::new(),
        }
    }

    /// Allocate a fresh plan. Convenience wrapper over [`Self::rebuild_into`]
    /// for tests and code paths that don't care about reuse.
    pub fn build(route: &RouterOutput, batch: usize, num_experts: usize, top_k: usize) -> Self {
        let mut p = Self::empty();
        p.rebuild_into(route, batch, num_experts, top_k);
        p
    }

    /// Allocation-free rebuild. Reuses the existing `expert_offsets`,
    /// `packed_token_idx`, `pairs_by_token`, `pair_weights` buffers via
    /// `clear() + resize()`. Uses the trailing tail of `expert_offsets`
    /// as the host-side cursor scratch (saves the per-call `cursors.clone()`).
    pub fn rebuild_into(
        &mut self,
        route: &RouterOutput,
        batch: usize,
        num_experts: usize,
        top_k: usize,
    ) {
        debug_assert_eq!(route.expert_ids.len(), batch * top_k);
        debug_assert_eq!(route.expert_weights.len(), batch * top_k);
        let total_pairs = batch * top_k;

        self.expert_offsets.clear();
        self.expert_offsets.resize(num_experts + 1, 0);
        self.packed_token_idx.clear();
        self.packed_token_idx.resize(total_pairs, 0);
        self.pairs_by_token.clear();
        self.pairs_by_token.resize(total_pairs, -1);

        // Pass 1: count pairs per expert. Stored into expert_offsets[1..]
        // so the inclusive-prefix-sum in Pass 2 can run in place — no
        // separate `counts` Vec.
        for &eid in &route.expert_ids {
            self.expert_offsets[eid as usize + 1] += 1;
        }

        // Pass 2: in-place inclusive prefix sum → expert_offsets[].
        for e in 0..num_experts {
            self.expert_offsets[e + 1] += self.expert_offsets[e];
        }

        // Pass 3: fill packed_token_idx + pairs_by_token by walking pairs
        // in (b, k) order and bucketing. The `cursors` scratch tracks how
        // many pairs each expert has already received; on first call it
        // grows to `num_experts`, subsequent calls reuse the allocation.
        self.cursors.clear();
        self.cursors
            .extend_from_slice(&self.expert_offsets[..num_experts]);

        for b in 0..batch {
            for k in 0..top_k {
                let pair_flat = b * top_k + k;
                let eid = route.expert_ids[pair_flat] as usize;
                let slot = self.cursors[eid];
                self.cursors[eid] += 1;
                self.packed_token_idx[slot] = b as u32;
                self.pairs_by_token[pair_flat] = slot as i32;
            }
        }

        // Pair weights: replicate from RouterOutput. Reuse self's vector
        // via clear() + extend rather than the per-call `clone()`.
        self.pair_weights.clear();
        self.pair_weights.extend_from_slice(&route.expert_weights);
    }
}

/// Reusable host-side scratch for [`moe_forward_bucketed`]. Holds the
/// router output, softmax scratch buffer, and bucket plan, all reused
/// across layers so the inner MoE forward path is allocation-free.
///
/// At c=32 / Qwen3-MoE / 48 layers, the previous fresh-`Vec`-per-layer
/// pattern accounted for ~10 ms / token of pure CPU softmax+sort+alloc
/// (25% of MoE wallclock — see `docs/bench/cuda-rtx4090-2026-05-08-m3-moe`).
pub struct MoeRouteScratch {
    pub output: RouterOutput,
    /// Softmax buffer reused across all rows of all layers — sized to
    /// `num_experts` on first use.
    pub probs: Vec<f32>,
    pub plan: MoeBucketPlan,
}

impl MoeRouteScratch {
    pub fn new() -> Self {
        Self {
            output: RouterOutput::empty(),
            probs: Vec::new(),
            plan: MoeBucketPlan::empty(),
        }
    }
}

impl Default for MoeRouteScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle of pre-allocated device buffers for the graph-capturable
/// device-routing path in [`moe_forward_bucketed`]. Pass `Some` to
/// take the device path (under `FERRUM_MOE_DEVICE_ROUTE=1`); pass
/// `None` for the legacy host-mediated path (used by tests + the
/// non-vLLM CUDA bucketed path).
///
/// Pre-allocated on Qwen3MoeScratch (`route_pairs_dev` etc.) so the
/// per-layer call doesn't alloc inside a captured stream.
pub struct DeviceRouteScratch<'a, B: crate::moe::dispatch::Backend> {
    pub selected_ids: &'a mut B::Buffer,
    pub pair_weights: &'a mut B::Buffer,
    pub pairs_by_token: &'a mut B::Buffer,
    pub packed_token_idx: &'a mut B::Buffer,
    pub expert_offsets: &'a mut B::Buffer,
    // Phase 2: moe_align_block_size outputs for the vLLM marlin_moe
    // fused GEMM path. Same shape as host `vllm_routing` builder
    // produces, but device-resident.
    pub sorted_tokens: &'a mut B::Buffer,
    pub block_ids: &'a mut B::Buffer,
    pub total_post_pad: &'a mut B::Buffer,
}

/// Bucketed MoE forward: gather → per-expert m=N Marlin GEMM → silu_mul →
/// per-expert m=N Marlin GEMM → moe_combine.
///
/// Replaces the `batch × top_k` m=1 dispatch loop in [`moe_forward`] with
/// `num_active_experts × 2` m=tokens_per_expert dispatches. For prefill
/// (m=512+), this is a 30× reduction in GEMM launches AND each GEMM runs
/// at a much more efficient m than the m=1 path. For decode (m=1), the
/// number of dispatches is similar but we still benefit from the
/// gather/combine kernel pattern (one launch each instead of 2 per pair).
///
/// **Requires**: scratch buffers `x_packed [total_pairs, hidden]`,
/// `gate_up_packed [total_pairs, 2*expert_inter]`,
/// `silu_packed [total_pairs, expert_inter]`, and
/// `down_packed [total_pairs, hidden]` provisioned by the caller. The
/// caller is responsible for sizing these to `batch * top_k` rows
/// (worst-case all top_k pairs alive).
#[allow(clippy::too_many_arguments)]
pub fn moe_forward_bucketed<B: QuantLlmBackend + BackendMoeFused>(
    ctx: &mut B::Context,
    x: &B::Buffer,
    router_logits: &B::Buffer,
    out: &mut B::Buffer,
    batch: usize,
    hidden_size: usize,
    expert_intermediate: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    experts: &ExpertStack<B>,
    x_packed: &mut B::Buffer,
    gate_up_packed: &mut B::Buffer,
    silu_packed: &mut B::Buffer,
    down_packed: &mut B::Buffer,
    route_scratch: &mut MoeRouteScratch,
    // Optional device routing scratch — when Some AND
    // FERRUM_MOE_DEVICE_ROUTE=1 AND FERRUM_VLLM_MOE=1, runs the
    // graph-capturable device-routing branch. None / unset = legacy
    // host-mediated path (used by tests + non-vLLM path).
    device_route: Option<DeviceRouteScratch<'_, B>>,
) -> Result<()> {
    if experts.num_experts() != num_experts {
        return Err(FerrumError::model(format!(
            "moe_forward_bucketed: experts {} != num_experts {num_experts}",
            experts.num_experts()
        )));
    }

    // Bucket profiling fires on either FERRUM_MOE_PROFILE=1 (legacy)
    // or FERRUM_DECODE_OP_PROFILE=1 (the gate the print site uses).
    let prof = moe_profile_enabled() || std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
    if prof {
        MOE_BUCKET_LAYER_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    // ── Device-route fast path (opt-in via FERRUM_MOE_DEVICE_ROUTE=1
    //     + FERRUM_VLLM_MOE=1 + device_route Some) ────────────────────
    //
    // Skips ALL host round-trips in the routing + bucket-plan stages:
    //   1. B::route_topk_softmax → device expert_ids + weights
    //   2. B::moe_build_pairs_by_token → device pairs / packed_idx /
    //      expert_offsets
    //   3. Gather via B::embedding_lookup_dev (device packed_idx)
    //   4. (rest of function reuses these device buffers; the vLLM
    //      MoE GEMM consumes them directly)
    //
    // This is the prerequisite for CUDA Graph capture over the MoE
    // layer loop in Qwen3MoeModel::decode_batch_internal.
    let use_vllm_moe = std::env::var("FERRUM_VLLM_MOE").map_or(false, |v| v == "1");
    // Device-routing path: enabled whenever the caller passes
    // pre-allocated `DeviceRouteScratch` AND `FERRUM_VLLM_MOE=1` is on.
    // No separate env var — the device path is strictly faster than
    // the host path (+15.4% c=32 on Qwen3-30B-A3B-GPTQ-Int4, RTX 4090
    // bench docs/bench/moe-phase3-vast-2026-05-12); the host path's
    // per-layer `try_gpu_route_topk_into_host` (D2H + cuStreamSynchronize)
    // was a per-layer GPU stall that compounded over 48 layers.
    //
    // Requires use_vllm_moe because the non-vLLM bucketed path needs
    // host phase1_dispatches / phase3_dispatches lists (one entry per
    // active expert with expert-id-dependent shape), which can't be
    // built on-device.
    //
    // Callers that need to force the host path for diagnostics can set
    // FERRUM_MOE_HOST_ROUTE=1 (opt-out).
    let use_device_route = device_route.is_some()
        && use_vllm_moe
        && !std::env::var("FERRUM_MOE_HOST_ROUTE")
            .map(|v| v == "1")
            .unwrap_or(false);
    let use_vllm_pair_ids = use_device_route && vllm_moe_pair_ids_enabled();

    // Run device-side routing kernels EARLY so `dr.packed_token_idx`
    // is available for the device-buffer gather (embedding_lookup_dev),
    // `dr.pairs_by_token` / `dr.pair_weights` for moe_combine, and
    // `dr.sorted_tokens` / `dr.block_ids` / `dr.total_post_pad` (via
    // moe_align_block_size below) for the vLLM marlin_moe GEMM phases.
    // Kept alive in `dr_kept` until end of function.
    let mut dr_kept: Option<DeviceRouteScratch<'_, B>> = if use_device_route {
        let dr = device_route.expect("device_route is Some when use_device_route");
        B::route_topk_softmax(
            ctx,
            router_logits,
            dr.selected_ids,
            dr.pair_weights,
            batch,
            num_experts,
            top_k,
            norm_topk_prob,
        )?;
        if !use_vllm_pair_ids {
            B::moe_build_pairs_by_token(
                ctx,
                dr.selected_ids,
                dr.pairs_by_token,
                dr.packed_token_idx,
                dr.expert_offsets,
                batch * top_k,
                num_experts,
                top_k,
            )?;
        }
        Some(dr)
    } else {
        None
    };

    // ── Routing + bucket plan (host) ─────────────────────────────────
    //
    // Skipped entirely under use_device_route — the device kernels run
    // by `dr_kept` above produce equivalent on-device buffers. The host
    // path stays for the legacy non-vllm bucketed dispatch and for
    // tests (where device_route is None).
    //
    // GPU fast-path: `try_gpu_route_topk_into_host` runs the same
    // route_topk_softmax kernel and D2Hs only `[batch, top_k]` ids +
    // weights (~1 KB at c=32) into RouterOutput. Host fallback covers
    // backends without the override (CPU / Metal / future).
    let plan: Option<&crate::moe::MoeBucketPlan> = if !use_device_route {
        let t_route_total = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let gpu_route = B::try_gpu_route_topk_into_host(
            ctx,
            router_logits,
            &mut route_scratch.output.expert_ids,
            &mut route_scratch.output.expert_weights,
            batch,
            num_experts,
            top_k,
            norm_topk_prob,
        );
        if gpu_route.is_err() {
            let t_sync = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            B::sync(ctx);
            if let Some(t) = t_sync {
                MOE_BUCKET_SYNC_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            let t_d2h = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let logits_host = B::to_vec(router_logits, batch * num_experts);
            if let Some(t) = t_d2h {
                MOE_BUCKET_D2H_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            let t_route = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            crate::moe::router::route_into(
                &logits_host,
                batch,
                num_experts,
                top_k,
                norm_topk_prob,
                &mut route_scratch.output,
                &mut route_scratch.probs,
            );
            if let Some(t) = t_route {
                MOE_BUCKET_ROUTE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        } else if let Some(t) = t_route_total {
            MOE_BUCKET_ROUTE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let t_plan = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        route_scratch
            .plan
            .rebuild_into(&route_scratch.output, batch, num_experts, top_k);
        if let Some(t) = t_plan {
            MOE_BUCKET_PLAN_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        Some(&route_scratch.plan)
    } else {
        None
    };

    // ── Gather: x_packed[i] = x[packed_token_idx[i]] ───────────────────
    // Under use_device_route, read packed_token_idx from device (no
    // host roundtrip → graph-capturable). Else use the host plan.
    if !use_vllm_pair_ids {
        let t_gather = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if let Some(ref dr) = dr_kept {
            B::embedding_lookup_dev(
                ctx,
                x,
                dr.packed_token_idx,
                x_packed,
                batch * top_k,
                hidden_size,
            );
        } else {
            let plan = plan.expect("plan is Some when !use_device_route");
            B::embedding_lookup(ctx, x, &plan.packed_token_idx, x_packed, hidden_size);
        }
        if let Some(t) = t_gather {
            B::sync(ctx);
            MOE_BUCKET_GATHER_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    // ── Per-expert dispatch: gate_up + down GEMMs at m=tokens_per_expert
    //
    // Uses the strided GPTQ + silu_mul methods so we can pump through
    // the BIG packed buffers (allocated once at scratch alloc) without
    // any per-expert copies. Each expert gets its column-slice of the
    // shared stacked Marlin tile via expert_offset; the row-slice of
    // the packed input/output buffers via in_row_offset / out_row_offset.
    let gate_up_dim_per_expert = 2 * expert_intermediate;
    let down_n_per_expert = hidden_size;
    // Bulk-zero the gate_up workspace ONCE before phase 1 for the
    // non-vLLM Marlin paths. The vLLM marlin_moe_wna16 kernel resets
    // its lock slots internally on the reduce path; vLLM itself only
    // zeros this workspace at allocation time. Keep
    // FERRUM_VLLM_MOE_ZERO_WS=1 as an A/B escape hatch.
    let gu_store = experts.gate_up_stacked_store(0).ok_or_else(|| {
        FerrumError::model(
            "moe_forward_bucketed requires stacked gate_up store \
             (load via Qwen3MoeModel::new_safetensors)",
        )
    })?;
    let zero_marlin_workspace = !use_vllm_moe || vllm_moe_zero_workspace_enabled();
    if zero_marlin_workspace {
        let _ = gu_store.zero_workspace(ctx);
    }

    // Decide path: vLLM marlin_moe_wna16 fused or per-expert bucketed
    // GEMMs. Under use_device_route, build routing on-device via
    // `moe_align_block_size`; under use_vllm_moe alone, host-build it.
    // Either way the GEMM dispatcher takes `&Buffer` for the 3 routing
    // arrays.
    let total_pairs_active = batch * top_k;
    // ── Dynamic moe_block_size policy ────────────────────────────────────
    //
    // Marlin-MoE kernel template is instantiated for thread_m_blocks ∈
    // {1, 2, 3, 4} via COMMON_GET_IF_M1 / COMMON_GET_IF_M234 in
    // vllm_marlin_moe/ops.cu. Each maps to block_size = thread_m_blocks
    // × 16 ∈ {16, 32, 48, 64}. Picking the right one is a classic
    // throughput-vs-padding-waste tradeoff:
    //
    //   block_size=16 :  16 thread_n=128 num_threads=128 (small_batch tile)
    //   block_size=32+:  16 thread_n=256 num_threads=256 (large_batch tile,
    //                    8× more work per kernel launch)
    //
    // Larger tile = more arithmetic per memory load = higher DRAM
    // utilization. But each expert pads its actual token count up to
    // a multiple of block_size — sparse routing (many experts, few
    // tokens each) bleeds into massive padding waste.
    //
    // Test data (commit ccba35f static block=64 vs reverted block=16):
    //   bench/v0.2-cuda dmon @ c=32:
    //     block=16  →  SM=99%  DRAM=50%  (mem-stalled, tile too small)
    //     block=64  →  varies wildly:
    //                    same-prompt c=32  : 2078 tok/s (+100% vs block=16)
    //                    apples c=32 diverse: 921 tok/s (-11% vs block=16)
    //
    // Decision rule: pick the largest block_size whose padding overhead
    // would still be ≤ ~30%. The host-routing path has `plan.expert_offsets`
    // which gives exact m_e per expert — pick by actual data. The
    // device-routing path doesn't have host visibility so falls back to
    // a conservative 16 (matches pre-PR behaviour, no regression).
    //
    // Worst-case scratch sizing: 64 (the largest block_size we'd pick).
    // `Qwen3MoeScratch.route_sorted_tokens_dev` capacity is allocated
    // assuming this upper bound.
    let max_block_size: usize = 64;
    let moe_block_size: usize = pick_moe_block_size(plan, num_experts, use_device_route);
    debug_assert!(
        moe_block_size <= max_block_size,
        "moe_block_size {moe_block_size} exceeds scratch worst-case {max_block_size}"
    );
    // sorted_max bound — passed to moe_align as a runtime cap so it
    // never writes past `total_padded` for the chosen block_size. We use
    // the picked `moe_block_size`, not `max_block_size`, so moe_align
    // doesn't sentinel-fill the slack between the actual padded count
    // and the worst-case buffer capacity (saves ~6 KB of writes per
    // layer × 48 layers × 32 layer-loop iters when block_size lands at 16).
    // The buffer itself is sized for max_block_size in qwen3_moe.rs.
    let sorted_max_size = batch * top_k + num_experts * moe_block_size;
    let vllm_routing_owned: Option<ferrum_kernels::backend::MoeRouting<B>> =
        if use_vllm_moe && !use_device_route {
            let plan = plan.expect("plan is Some when host vllm builder runs");
            let mut padded_offsets = Vec::with_capacity(num_experts + 1);
            let mut acc = 0usize;
            for e in 0..num_experts {
                padded_offsets.push(acc);
                let m_e = plan.expert_offsets[e + 1] - plan.expert_offsets[e];
                let pe = m_e.div_ceil(moe_block_size) * moe_block_size;
                acc += pe;
            }
            padded_offsets.push(acc);
            let total_padded = acc;
            let total_blocks = total_padded / moe_block_size;
            let sentinel = total_pairs_active as i32;

            let mut sorted_token_ids = vec![sentinel; total_padded];
            let mut expert_ids = vec![0i32; total_blocks];
            for e in 0..num_experts {
                let m_e = plan.expert_offsets[e + 1] - plan.expert_offsets[e];
                if m_e == 0 {
                    continue;
                }
                let p_off = padded_offsets[e];
                let real_off = plan.expert_offsets[e];
                for i in 0..m_e {
                    sorted_token_ids[p_off + i] = (real_off + i) as i32;
                }
                let blocks_for_e = (padded_offsets[e + 1] - p_off) / moe_block_size;
                let block_start = p_off / moe_block_size;
                for b in 0..blocks_for_e {
                    expert_ids[block_start + b] = e as i32;
                }
            }
            let num_tokens_past_padded = vec![total_padded as i32];
            Some(B::upload_moe_routing(
                ctx,
                &sorted_token_ids,
                &expert_ids,
                &num_tokens_past_padded,
            )?)
        } else {
            None
        };

    // Device-side moe_align_block_size — under use_device_route, fill
    // dr.{sorted_tokens, block_ids, total_post_pad} on device from
    // dr.selected_ids. No host roundtrip → captures cleanly.
    if use_device_route {
        let dr = dr_kept
            .as_mut()
            .expect("dr_kept is Some when use_device_route");
        if use_vllm_pair_ids {
            B::moe_align_block_size_pair_ids(
                ctx,
                dr.selected_ids,
                dr.sorted_tokens,
                dr.block_ids,
                dr.total_post_pad,
                batch * top_k,
                num_experts,
                moe_block_size,
                sorted_max_size,
            )?;
        } else {
            B::moe_align_block_size(
                ctx,
                dr.selected_ids,
                dr.sorted_tokens,
                dr.block_ids,
                dr.total_post_pad,
                batch * top_k,
                num_experts,
                moe_block_size,
                sorted_max_size,
            )?;
        }
    }

    // Resolve the 3 routing buffers for vLLM phase 1/3 GEMM. Either
    // from dr_kept (device-built by moe_align_block_size) or from
    // vllm_routing_owned (host-built + uploaded). None → use legacy
    // per-expert batched GEMM path.
    let vllm_refs: Option<(&B::Buffer, &B::Buffer, &B::Buffer)> = if use_device_route {
        let dr = dr_kept
            .as_ref()
            .expect("dr_kept is Some when use_device_route");
        Some((&*dr.sorted_tokens, &*dr.block_ids, &*dr.total_post_pad))
    } else if let Some(r) = vllm_routing_owned.as_ref() {
        Some((
            &r.sorted_token_ids,
            &r.expert_ids,
            &r.num_tokens_past_padded,
        ))
    } else {
        None
    };

    // Phase 1/3 batched-GEMM dispatch lists. Only built (and read) for
    // the non-vLLM path. Under use_device_route the host plan is None
    // anyway, so we'd fail to build them — skip via vllm_refs.is_some.
    let phase1_dispatches: Vec<(usize, usize, usize, usize)> = if vllm_refs.is_none() {
        let plan = plan.expect("plan is Some when batched GEMM path runs");
        let mut v: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let m_e = plan.expert_offsets[e + 1] - plan.expert_offsets[e];
            if m_e == 0 {
                continue;
            }
            let pair_off = plan.expert_offsets[e];
            v.push((e, pair_off, pair_off, m_e));
        }
        v.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
        v
    } else {
        Vec::new()
    };
    let t_gemm1 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    if let Some((sorted_tokens, block_ids, total_post_pad)) = vllm_refs {
        // fp32_reduce path: kernel writes C directly via global reduce.
        if use_vllm_pair_ids {
            gu_store.gemm_phase_vllm(
                ctx,
                x,
                sorted_tokens,
                block_ids,
                total_post_pad,
                gate_up_packed,
                batch,
                moe_block_size,
                top_k,
            )?;
        } else {
            gu_store.gemm_phase_vllm(
                ctx,
                x_packed,
                sorted_tokens,
                block_ids,
                total_post_pad,
                gate_up_packed,
                total_pairs_active,
                moe_block_size,
                1, // top_k=1: pre-gathered rows already index packed input directly
            )?;
        }
    } else {
        gu_store.gemm_phase_batched(
            ctx,
            x_packed,
            &phase1_dispatches,
            gate_up_packed,
            hidden_size,
        )?;
    }
    if let Some(t) = t_gemm1 {
        B::sync(ctx);
        MOE_BUCKET_GEMM1_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    // Phase 2: SiLU(gate) * up — single launch covering ALL active
    // expert rows in the packed buffer. The unused rows (zeros from
    // experts with m_e=0) just produce zeros that the combine step
    // ignores via pairs_by_token. Saves num_active_experts-1 launches
    // per layer.
    let total_pairs_active = batch * top_k;
    let t_silu = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    B::fused_silu_mul_split(
        ctx,
        gate_up_packed,
        silu_packed,
        total_pairs_active,
        expert_intermediate,
    );
    if let Some(t) = t_silu {
        B::sync(ctx);
        MOE_BUCKET_SILU_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    // Phase 3: down GEMM per active expert. Multi-stream batched.
    let d_store = experts.down_stacked_store(0).ok_or_else(|| {
        FerrumError::model(
            "moe_forward_bucketed requires stacked down store \
             (load via Qwen3MoeModel::new_safetensors)",
        )
    })?;
    if zero_marlin_workspace {
        let _ = d_store.zero_workspace(ctx);
    }
    let phase3_dispatches: Vec<(usize, usize, usize, usize)> = if vllm_refs.is_none() {
        let plan = plan.expect("plan is Some when batched GEMM path runs");
        let mut v: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let m_e = plan.expert_offsets[e + 1] - plan.expert_offsets[e];
            if m_e == 0 {
                continue;
            }
            let pair_off = plan.expert_offsets[e];
            v.push((e, pair_off, pair_off, m_e));
        }
        v.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
        v
    } else {
        Vec::new()
    };
    let t_gemm3 = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    if let Some((sorted_tokens, block_ids, total_post_pad)) = vllm_refs {
        d_store.gemm_phase_vllm(
            ctx,
            silu_packed,
            sorted_tokens,
            block_ids,
            total_post_pad,
            down_packed,
            total_pairs_active,
            moe_block_size,
            1,
        )?;
    } else {
        d_store.gemm_phase_batched(
            ctx,
            silu_packed,
            &phase3_dispatches,
            down_packed,
            expert_intermediate,
        )?;
    }
    if let Some(t) = t_gemm3 {
        B::sync(ctx);
        MOE_BUCKET_GEMM3_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    // ── Combine: out[b, h] = Σ_k weights[b,k] * down_packed[pairs_by_token[b,k], h]
    //
    // Two paths for the pairs/weights device buffers:
    //
    //   (a) device-route mode: reuse `dr_kept` populated up top by
    //       B::route_topk_softmax + B::moe_build_pairs_by_token. No
    //       host→device upload, so this is graph-capturable when
    //       wrapped in begin_graph_capture.
    //
    //   (b) legacy: upload host plan (plan.pairs_by_token /
    //       plan.pair_weights) via from_slice_typed. Records host
    //       pointer; captures stale on replay.
    //
    // Both produce mathematically equivalent outputs — device path
    // does the same counting-sort the host plan rebuild does, just
    // on-device via the moe_build_pairs kernel.
    let total_pairs = batch * top_k;
    let t_comb = if prof {
        Some(std::time::Instant::now())
    } else {
        None
    };
    if use_vllm_pair_ids {
        let dr = dr_kept
            .as_ref()
            .expect("dr_kept is Some when use_vllm_pair_ids");
        B::weighted_sum_batched(
            ctx,
            down_packed,
            dr.pair_weights,
            out,
            batch,
            top_k,
            hidden_size,
        )?;
    } else {
        let (pairs_ref, weights_ref);
        let _pairs_owned;
        let _weights_owned;
        if let Some(ref dr) = dr_kept {
            pairs_ref = &*dr.pairs_by_token;
            weights_ref = &*dr.pair_weights;
        } else {
            let plan = plan.expect("plan is Some when host moe_combine runs");
            _pairs_owned = B::from_slice_typed::<i32>(&plan.pairs_by_token);
            _weights_owned = B::from_slice_typed::<f32>(&plan.pair_weights);
            pairs_ref = &_pairs_owned;
            weights_ref = &_weights_owned;
        }
        B::moe_combine(
            ctx,
            down_packed,
            pairs_ref,
            weights_ref,
            out,
            batch,
            hidden_size,
            top_k,
            total_pairs,
        );
    }
    if let Some(t) = t_comb {
        B::sync(ctx);
        MOE_BUCKET_COMBINE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    Ok(())
}

/// Run MoE forward on CPU.
///
/// Inputs:
///   - `x`: `[batch, hidden_size]` row-major hidden states (post-attention,
///          post-residual — i.e. what the dense MLP would normally see).
///   - `router`: top-K assignments + weights from [`super::router::route`].
///   - `experts`: per-layer expert weights from [`ExpertStack::load_from_gguf`].
///
/// Output:
///   - `out`: `[batch, hidden_size]`. Resized + zero-initialised.
///
/// The function recomputes its scratch buffers each call. For tight
/// inner loops, callers will eventually want a pre-allocated workspace
/// (Phase 2F refactor). For now, this is the readable reference.
pub fn moe_forward_cpu(
    x: &[f32],
    batch: usize,
    hidden_size: usize,
    expert_intermediate: usize,
    top_k: usize,
    router: &RouterOutput,
    experts: &ExpertStack<CpuBackend>,
    out: &mut Vec<f32>,
) -> Result<()> {
    let n_experts = experts.num_experts();

    if x.len() != batch * hidden_size {
        return Err(FerrumError::model(format!(
            "moe_forward_cpu: x len {} doesn't match batch*hidden = {}*{} = {}",
            x.len(),
            batch,
            hidden_size,
            batch * hidden_size
        )));
    }
    if router.expert_ids.len() != batch * top_k {
        return Err(FerrumError::model(format!(
            "moe_forward_cpu: router has {} expert_ids but expected batch*top_k = {}*{} = {}",
            router.expert_ids.len(),
            batch,
            top_k,
            batch * top_k
        )));
    }

    out.clear();
    out.resize(batch * hidden_size, 0.0);

    let mut ctx = <CpuBackend as Backend>::new_context();
    let mut x_b: Vec<f32> = vec![0.0; hidden_size];
    let mut gate_up_buf: Vec<f32> = vec![0.0; 2 * expert_intermediate];
    let mut silu_mul_buf: Vec<f32> = vec![0.0; expert_intermediate];
    let mut down_out: Vec<f32> = vec![0.0; hidden_size];

    for b in 0..batch {
        x_b.copy_from_slice(&x[b * hidden_size..(b + 1) * hidden_size]);

        for k in 0..top_k {
            let pair_idx = b * top_k + k;
            let expert_id = router.expert_ids[pair_idx] as usize;
            let weight = router.expert_weights[pair_idx];

            if expert_id >= n_experts {
                return Err(FerrumError::model(format!(
                    "moe_forward_cpu: router selected expert {expert_id} >= num_experts {n_experts}"
                )));
            }

            // Gate||Up projection (fused) → [1, 2*expert_inter]
            experts.gate_up[expert_id].forward(&mut ctx, &x_b, &mut gate_up_buf, 1);

            // SiLU(gate) * up → [1, expert_inter]
            <CpuBackend as Backend>::fused_silu_mul_split(
                &mut ctx,
                &gate_up_buf,
                &mut silu_mul_buf,
                1,
                expert_intermediate,
            );

            // Down projection → [1, hidden]
            experts.down[expert_id].forward(&mut ctx, &silu_mul_buf, &mut down_out, 1);

            // Weighted accumulate into out[b, :]. Done host-side because
            // CpuBackend::Buffer = Vec<f32> and the trait doesn't yet
            // expose scaled-add.
            let out_row = &mut out[b * hidden_size..(b + 1) * hidden_size];
            for (o, d) in out_row.iter_mut().zip(down_out.iter()) {
                *o += weight * *d;
            }
        }
    }

    Ok(())
}

fn check_size(actual: usize, expected: usize, label: &str) -> Result<()> {
    if actual != expected {
        return Err(FerrumError::model(format!(
            "ExpertStack: {label} size mismatch (got {actual}, expected {expected})"
        )));
    }
    Ok(())
}

/// Map candle's `GgmlDType` to the kernel-side `GgufQuantType` for the
/// dtypes a backend can dispatch on. Returns `None` for any other dtype
/// (callers fall back to eager dequant).
fn quant_kind(gguf: &GgufFile, name: &str) -> Result<Option<GgufQuantType>> {
    let info = gguf.tensor_info(name).ok_or_else(|| {
        FerrumError::model(format!("ExpertStack: tensor info missing for '{name}'"))
    })?;
    Ok(match info.ggml_dtype {
        GgmlDType::Q4K => Some(GgufQuantType::Q4K),
        GgmlDType::Q6K => Some(GgufQuantType::Q6K),
        _ => None,
    })
}

/// Per-expert block-byte count for a given k-quant flavour and element
/// count. Q4_K = 144 B / 256 elems, Q6_K = 210 B / 256 elems. Errors if
/// `n_elems` is not a multiple of the super-block size (256) — a Q-quant
/// invariant.
fn block_bytes_for(kind: GgufQuantType, n_elems: usize, label: &str) -> Result<usize> {
    const QK_K: usize = 256;
    if n_elems % QK_K != 0 {
        return Err(FerrumError::model(format!(
            "ExpertStack {label}: per-expert element count {n_elems} not a multiple of {QK_K}"
        )));
    }
    let block_bytes = match kind {
        GgufQuantType::Q4K => 144,
        GgufQuantType::Q6K => 210,
        // Other k-quants are filtered out earlier via `quant_kind`; reaching here
        // with one would be a programming error.
        other => {
            return Err(FerrumError::model(format!(
                "ExpertStack {label}: unsupported k-quant flavour {other:?}"
            )))
        }
    };
    Ok((n_elems / QK_K) * block_bytes)
}

fn read_dequant_flat(gguf: &GgufFile, name: &str, device: &Device) -> Result<Vec<f32>> {
    let qt = gguf.read_tensor(name, device).map_err(candle_to_ferrum)?;
    let dense = qt.dequantize(device).map_err(candle_to_ferrum)?;
    let flat = dense.flatten_all().map_err(candle_to_ferrum)?;
    flat.to_vec1::<f32>().map_err(candle_to_ferrum)
}

fn candle_to_ferrum(e: candle_core::Error) -> FerrumError {
    FerrumError::model(format!("candle: {e}"))
}

// Suppress unused-import warning when this module compiles standalone in
// the lib (the candle Result alias is only used via map_err in Phase 2).
#[allow(dead_code)]
type _CandleResult<T> = CandleResult<T>;
