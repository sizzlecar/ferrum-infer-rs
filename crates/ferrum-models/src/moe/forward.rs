//! MoE forward dispatchers for Qwen3-MoE-style architectures.
//!
//! Free functions (not methods) so callers can split the borrow on the
//! parent struct between the immutable expert layer state and the mutable
//! scratch buffers. Imported by `crate::models::qwen3_moe`.
//!
//! These are the three production paths the model selects between at
//! runtime based on token count + batched-decode env flags:
//!   - `moe_forward_stacked_decode`  — m=1 single-token decode
//!   - `moe_forward_batched_prefill` — m>=8 prefill / large-batch decode
//!   - `moe_forward_batched_decode`  — 2 <= m < 32 batched decode (mid)

use ferrum_kernels::backend::{BackendMoeFused, QuantLlmBackend};
use ferrum_types::Result;

use crate::models::qwen3_moe::{Qwen3MoeLayerState, Qwen3MoeScratch};

use crate::models::qwen3_moe::{
    DEC_DOWN_US, DEC_GATE_US, DEC_ROUTE_US, DEC_SILU_US, DEC_UP_US, DEC_WSUM_US,
    MOE_BATCHED_DECODE_DOWN_US, MOE_BATCHED_DECODE_GATE_US, MOE_BATCHED_DECODE_ROUTE_US,
    MOE_BATCHED_DECODE_SILU_US, MOE_BATCHED_DECODE_UP_US, MOE_BATCHED_DECODE_WSUM_US,
    MOE_PREFILL_DOWN_CALLS, MOE_PREFILL_DOWN_US, MOE_PREFILL_GATE_CALLS, MOE_PREFILL_GATE_US,
    MOE_PREFILL_HOST_TOPK_CALLS, MOE_PREFILL_HOST_TOPK_US, MOE_PREFILL_SILU_CALLS,
    MOE_PREFILL_SILU_US, MOE_PREFILL_UP_CALLS, MOE_PREFILL_UP_US, MOE_PREFILL_WSUM_CALLS,
    MOE_PREFILL_WSUM_US,
};
use std::sync::atomic::AtomicU64;
use std::sync::OnceLock;

/// Batched MoE FFN — decode (m=1) and per-token-prefill (m>1 looped).
///
/// Three batched `gemv_quant_moe_id` dispatches per token: gate (broadcast
/// activation), up (broadcast activation), down (per-slot activation —
/// each expert sees its own silu·up). The per-(token, expert) outer loop
/// shrinks from `top_k * 4` dispatches per layer to **3 batched + 1
/// silu_mul_split + 1 weighted_sum_dispatch_loop**.
///
/// For prefill (m > 1) we loop over tokens externally — each token's
/// router output drives a single batched call. Still much faster than
/// the per-(token, expert) per-Linear path because the gemvs are batched.
///
/// Free function (not a method) so the caller can split the borrow on
/// `self` between `moe_layers[li]` (immutable) and `scratch` (mutable).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_forward_stacked_decode_impl<B: QuantLlmBackend + BackendMoeFused>(
    ctx: &mut B::Context,
    moe_layer: &Qwen3MoeLayerState<B>,
    scratch: &mut Qwen3MoeScratch<B>,
    h: usize,
    inter: usize,
    top_k: usize,
    n_exp: usize,
    norm_topk_prob: bool,
    tokens: usize,
    residual: &mut B::Buffer,
    // If `Some`, fold the NEXT layer's leading rms_norm into the
    // weighted-sum-residual tail using `weighted_sum_residual_norm_stacked`.
    next_norm_w: Option<&B::Buffer>,
    eps: f32,
) -> Result<()> {
    // GPU-side routing: one Metal launch reads router_logits and writes
    // selected ids + combine weights directly into device-side scratch
    // buffers. Eliminates the per-layer `B::sync + B::to_vec(router_logits)
    // + host route()` round trip — the dominant remaining cost in the
    // decode hot path (~10% of total decode latency).
    let prof = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
    let stage_t0 = || -> Option<std::time::Instant> {
        if prof {
            Some(std::time::Instant::now())
        } else {
            None
        }
    };
    let stage_end = |t0: Option<std::time::Instant>, ctx: &mut B::Context, c: &AtomicU64| {
        if let Some(t) = t0 {
            B::sync(ctx);
            c.fetch_add(
                t.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    };

    let t0 = stage_t0();
    B::route_topk_softmax(
        ctx,
        &scratch.router_logits,
        &mut scratch.ids_buf,
        &mut scratch.weights_buf,
        tokens,
        n_exp,
        top_k,
        norm_topk_prob,
    )?;
    stage_end(t0, ctx, &DEC_ROUTE_US);

    // moe_forward_stacked_decode_impl is only called when `tokens == 1`
    // (the branch in `forward_layer` routes prefill m>1 through
    // `moe_forward_batched_prefill_impl` instead). The for-b loop and
    // the copy norm_out[b*h] → x_single were vestigial scaffolding;
    // for tokens=1 norm_out[0..h] IS the activation row, and we can
    // pass it straight to the gemv kernel via src1_stride=0 broadcast.
    debug_assert_eq!(
        tokens, 1,
        "moe_forward_stacked_decode_impl expects tokens=1 (prefill goes through moe_forward_batched_prefill_impl)"
    );
    let _ = tokens; // silence unused-warning when assertion is compiled out

    {
        // ids_buf and weights_buf populated by the GPU router above —
        // no host writes needed here in the decode path.

        // Fused-vs-unfused gate+up+silu selection.
        //
        // Default: when the backend advertises support (Metal Q4KExperts),
        // run the single fused dispatch — saves 2 dispatches and the
        // entire round-trip through gate_out_stacked / up_out_stacked
        // scratch (≈4× [top_k, ffn] of intermediate bandwidth).
        //
        // Opt-out: `FERRUM_MOE_FUSED_GATE_UP_SILU=0` forces the legacy
        // 3-dispatch path. Used for A/B benchmarking and as a kill switch
        // if the fused kernel ever produces divergent outputs.
        // Cache the env-flag read once per process — the decode hot
        // path calls this fn ~48 layers × ~steps_per_run times.
        static FUSED_DISABLED: OnceLock<bool> = OnceLock::new();
        let fused_disabled = *FUSED_DISABLED
            .get_or_init(|| std::env::var("FERRUM_MOE_FUSED_GATE_UP_SILU").as_deref() == Ok("0"));
        let use_fused = B::supports_fused_moe_gate_up_silu() && !fused_disabled;

        if use_fused {
            // 1+2+3 fused: silu_stacked = SiLU(gate · norm_out) * (up · norm_out)
            let t0 = stage_t0();
            moe_layer.experts.gemv_gate_up_silu_fused(
                ctx,
                &scratch.norm_out,
                &scratch.ids_buf,
                &mut scratch.silu_stacked,
                top_k,
            )?;
            stage_end(t0, ctx, &DEC_SILU_US);
        } else {
            // 1. Batched gate gemv — broadcast input across top_k slots.
            let t0 = stage_t0();
            moe_layer.experts.gemv_gate(
                ctx,
                &scratch.norm_out,
                &scratch.ids_buf,
                &mut scratch.gate_out_stacked,
                top_k,
            )?;
            stage_end(t0, ctx, &DEC_GATE_US);

            // 2. Batched up gemv — also broadcast.
            let t0 = stage_t0();
            moe_layer.experts.gemv_up(
                ctx,
                &scratch.norm_out,
                &scratch.ids_buf,
                &mut scratch.up_out_stacked,
                top_k,
            )?;
            stage_end(t0, ctx, &DEC_UP_US);

            // 3. Stacked SiLU·gate → silu_stacked. Single dispatch covers
            //    all top_k slots — replaces the per-slot loop's
            //    (3 copy_slice + 1 silu_mul) × 8 = 32 dispatches.
            let t0 = stage_t0();
            B::silu_mul_stacked(
                ctx,
                &scratch.gate_out_stacked,
                &scratch.up_out_stacked,
                &mut scratch.silu_stacked,
                top_k,
                inter,
            )?;
            stage_end(t0, ctx, &DEC_SILU_US);
        }

        // 4. Batched down gemv — per-slot input via in_stride = inter.
        //    silu_stacked[k * inter ..] is the activation row for slot k.
        let t0 = stage_t0();
        moe_layer.experts.gemv_down(
            ctx,
            &scratch.silu_stacked,
            &scratch.ids_buf,
            &mut scratch.down_out_stacked,
            top_k,
            inter,
        )?;
        stage_end(t0, ctx, &DEC_DOWN_US);

        // 5. Fused weighted-sum + residual-add (+ optional next-layer
        //    rms_norm). Two paths:
        //
        //    * `next_norm_w = Some(_)` (cross-layer fusion): one kernel
        //      computes residual[i] += Σ_k w[k] · down[k, i] AND
        //      norm_out[i] = residual[i] · scale · next_norm_w[i].
        //      The next layer's leading rms_norm is skipped. Saves an
        //      additional dispatch per layer transition.
        //    * `next_norm_w = None` (last layer): just residual-add.
        let t0 = stage_t0();
        if let Some(nnw) = next_norm_w {
            B::weighted_sum_residual_norm_stacked(
                ctx,
                &scratch.down_out_stacked,
                &scratch.weights_buf,
                residual,
                nnw,
                &mut scratch.norm_out,
                top_k,
                h,
                eps,
            )?;
        } else {
            B::weighted_sum_residual_stacked(
                ctx,
                &scratch.down_out_stacked,
                &scratch.weights_buf,
                residual,
                top_k,
                h,
            )?;
        }
        stage_end(t0, ctx, &DEC_WSUM_US);
    }

    Ok(())
}

/// Batched MoE FFN for prefill (m > 1).
///
/// One pass through the expert dispatch — replaces the per-token loop
/// with three batched 2-D mul_mm_id dispatches (gate, up, down) where
/// each expert's slab of (token, slot) pairs runs as one gemm tile.
/// Per-layer dispatch count: ~6 (router + 3 mul_mm_id + silu + wsum)
/// independent of `tokens`. Compare to the decode-style stacked path
/// that emits ~10 per token.
///
/// Free function so the caller can split the borrow on `self` between
/// `moe_layers[li]` (immutable) and `scratch` (mutable).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_forward_batched_prefill_impl<B: QuantLlmBackend + BackendMoeFused>(
    ctx: &mut B::Context,
    moe_layer: &Qwen3MoeLayerState<B>,
    scratch: &mut Qwen3MoeScratch<B>,
    h: usize,
    inter: usize,
    top_k: usize,
    n_exp: usize,
    norm_topk_prob: bool,
    tokens: usize,
) -> Result<()> {
    let prof = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
    let stage_t0 = || -> Option<std::time::Instant> {
        if prof {
            Some(std::time::Instant::now())
        } else {
            None
        }
    };
    let stage_end =
        |t0: Option<std::time::Instant>, ctx: &mut B::Context, us: &AtomicU64, n: &AtomicU64| {
            if let Some(t) = t0 {
                B::sync(ctx);
                us.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        };

    // GPU-side routing: keep the whole pipeline device-resident. Two
    // dispatches replace the per-layer `B::sync + to_vec(router_logits)
    // + host route() + host compute_ids_tpe + write_back` round trip.
    //
    //   1. `route_topk_softmax` writes selected expert IDs (flat
    //      `[batch, top_k]`) into `selected_ids_buf` and the post-renorm
    //      combine weights directly into `weights_2d`.
    //   2. `compute_ids_tpe_gpu` buckets those pairs into `tpe_buf` and
    //      `ids_2d` using device-side atomic_fetch_add slot claims. The
    //      `ids_2d` row stride is the worst-case `tokens * top_k`; the
    //      consumer GEMM stops at `tpe[e]` so the over-strided columns
    //      cost only launch overhead, not real compute.
    //
    // `FERRUM_MOE_HOST_TOPK=1`        → legacy CPU softmax+topk+bucket
    // `FERRUM_MOE_DIRECT_DISPATCH=1`  → GPU topk but worst-case GEMM grid
    // (default)                       → GPU topk + indirect-dispatched GEMM
    //                                    (grid sized from max(tpe[e]))
    let use_gpu_topk = std::env::var("FERRUM_MOE_HOST_TOPK").as_deref() != Ok("1");
    let use_indirect_dispatch =
        use_gpu_topk && std::env::var("FERRUM_MOE_DIRECT_DISPATCH").as_deref() != Ok("1");
    let max_per_expert = if use_gpu_topk {
        let t0 = stage_t0();
        B::route_topk_softmax(
            ctx,
            &scratch.router_logits,
            &mut scratch.selected_ids_buf,
            &mut scratch.weights_2d,
            tokens,
            n_exp,
            top_k,
            norm_topk_prob,
        )?;
        B::compute_ids_tpe_gpu(
            ctx,
            &scratch.selected_ids_buf,
            &mut scratch.tpe_buf,
            &mut scratch.ids_2d,
            &mut scratch.gate_up_args_buf,
            &mut scratch.down_args_buf,
            tokens,
            n_exp,
            top_k,
            inter,
            h,
        )?;
        stage_end(
            t0,
            ctx,
            &MOE_PREFILL_HOST_TOPK_US,
            &MOE_PREFILL_HOST_TOPK_CALLS,
        );
        // Worst-case ids row stride; matches `dispatch_compute_ids_tpe`.
        tokens * top_k
    } else {
        use ferrum_kernels::moe_host::compute_ids_tpe;
        let t0 = stage_t0();
        B::sync(ctx);
        let logits_host = B::to_vec(&scratch.router_logits, tokens * n_exp);
        let route = crate::moe::router::route(&logits_host, tokens, n_exp, top_k, norm_topk_prob);
        let (tpe_host, ids_host, max_per_expert) =
            compute_ids_tpe(&route.expert_ids, n_exp, tokens, top_k);
        B::write_i32_into(&mut scratch.tpe_buf, &tpe_host);
        B::write_i32_into(&mut scratch.ids_2d, &ids_host);
        B::write_f32_into(&mut scratch.weights_2d, &route.expert_weights);
        stage_end(
            t0,
            ctx,
            &MOE_PREFILL_HOST_TOPK_US,
            &MOE_PREFILL_HOST_TOPK_CALLS,
        );
        max_per_expert
    };

    // 1. Batched gate gemm — one launch covers all (token, expert) pairs.
    //    src1 layout: [batch, ne11=1, K] (broadcast: each pair reads its
    //    token's row, slot index ignored).
    //    dst layout:  [batch, top_k, expert_inter] — natural.
    let gate_up_args = use_indirect_dispatch.then_some(&scratch.gate_up_args_buf);
    let down_args = use_indirect_dispatch.then_some(&scratch.down_args_buf);
    let t0 = stage_t0();
    moe_layer.experts.gemm_gate(
        ctx,
        &scratch.norm_out,
        &scratch.ids_2d,
        &scratch.tpe_buf,
        &mut scratch.gate_out_stacked,
        gate_up_args,
        top_k,
        max_per_expert,
        tokens,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_GATE_US, &MOE_PREFILL_GATE_CALLS);

    // 2. Batched up gemm — same shape as gate.
    let t0 = stage_t0();
    moe_layer.experts.gemm_up(
        ctx,
        &scratch.norm_out,
        &scratch.ids_2d,
        &scratch.tpe_buf,
        &mut scratch.up_out_stacked,
        gate_up_args,
        top_k,
        max_per_expert,
        tokens,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_UP_US, &MOE_PREFILL_UP_CALLS);

    // 3. SiLU·gate over [tokens * top_k, expert_inter] flat layout.
    let total_pairs = tokens * top_k;
    let t0 = stage_t0();
    B::silu_mul_batched(
        ctx,
        &scratch.gate_out_stacked,
        &scratch.up_out_stacked,
        &mut scratch.silu_stacked,
        total_pairs,
        inter,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_SILU_US, &MOE_PREFILL_SILU_CALLS);

    // 4. Batched down gemm — src1 is [batch, top_k, expert_inter] from
    //    silu_stacked. ne11 = top_k → each pair reads its own row.
    let t0 = stage_t0();
    moe_layer.experts.gemm_down(
        ctx,
        &scratch.silu_stacked,
        &scratch.ids_2d,
        &scratch.tpe_buf,
        &mut scratch.down_out_stacked,
        down_args,
        top_k,
        max_per_expert,
        tokens,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_DOWN_US, &MOE_PREFILL_DOWN_CALLS);

    // 5. Per-batch weighted sum: moe_out[b, h] = Σ_k w[b,k] · down[b,k,h]
    let t0 = stage_t0();
    B::weighted_sum_batched(
        ctx,
        &scratch.down_out_stacked,
        &scratch.weights_2d,
        &mut scratch.moe_out,
        tokens,
        top_k,
        h,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_WSUM_US, &MOE_PREFILL_WSUM_CALLS);

    Ok(())
}

/// Batched MoE FFN for the **small-m decode** range (typically c=2..32).
///
/// Mirrors llama.cpp's `kernel_mul_mv_id` strategy: hold the dispatch
/// count flat as concurrency scales by emitting **one** batched GEMV
/// per linear (gate / up / down) that covers all `m * top_k`
/// (token, expert) pairs in a single Metal launch. Replaces the
/// per-token outer loop in `forward_layer` (which emitted ~5
/// dispatches × m tokens per layer) with a fixed-shape pipeline.
///
/// Compared to [`moe_forward_batched_prefill_impl`]:
///   * no `compute_ids_tpe_gpu` bucketing kernel (the new pair-indexed
///     GEMV reads `selected_ids_buf` directly)
///   * uses GEMV not GEMM (better tile utilisation when tokens-per-expert
///     is small — at c=16 with top_k=8 each expert sees ~1-3 token rows,
///     well below the simdgroup_matmul tile width)
///   * fewer Metal dispatches per layer (5: route + 3 gemv + silu + wsum)
///
/// Per-layer dispatch budget: 5 (independent of m). At c=16 / 48 layers
/// that's 240 dispatches per decode step vs the per-token loop's ~3,840.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_forward_batched_decode_impl<B: QuantLlmBackend + BackendMoeFused>(
    ctx: &mut B::Context,
    moe_layer: &Qwen3MoeLayerState<B>,
    scratch: &mut Qwen3MoeScratch<B>,
    h: usize,
    inter: usize,
    top_k: usize,
    n_exp: usize,
    norm_topk_prob: bool,
    tokens: usize,
) -> Result<()> {
    let prof = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
    let stage_t0 = || -> Option<std::time::Instant> {
        if prof {
            Some(std::time::Instant::now())
        } else {
            None
        }
    };
    let stage_end = |t0: Option<std::time::Instant>, ctx: &mut B::Context, c: &AtomicU64| {
        if let Some(t) = t0 {
            B::sync(ctx);
            c.fetch_add(
                t.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    };

    let total_pairs = tokens * top_k;

    // 1. Single batched router pass — fills selected_ids_buf [m * top_k]
    //    and weights_2d [m * top_k] in one Metal dispatch.
    let t0 = stage_t0();
    B::route_topk_softmax(
        ctx,
        &scratch.router_logits,
        &mut scratch.selected_ids_buf,
        &mut scratch.weights_2d,
        tokens,
        n_exp,
        top_k,
        norm_topk_prob,
    )?;
    stage_end(t0, ctx, &MOE_BATCHED_DECODE_ROUTE_US);

    // 2+3+4. Fused gate+up+silu — single Metal dispatch covers all
    // m*top_k pairs. Falls back to the 3-dispatch sequence on backends
    // that don't have the fused-batched kernel.
    if B::supports_batched_moe_gate_up_silu() {
        let t0 = stage_t0();
        moe_layer.experts.gemv_gate_up_silu_batched_fused(
            ctx,
            &scratch.norm_out,
            &scratch.selected_ids_buf,
            &mut scratch.silu_stacked,
            tokens,
            top_k,
            h, // outer stride: K floats per token
            0, // inner stride: 0 (slots within a token broadcast)
        )?;
        // Charge the whole fused step to the SiLU bucket — keeps the
        // profile counter additive with the unfused path's silu line.
        stage_end(t0, ctx, &MOE_BATCHED_DECODE_SILU_US);
    } else {
        // 2. Batched gate gemv — one launch covers all m*top_k pairs.
        let t0 = stage_t0();
        moe_layer.experts.gemv_gate_batched(
            ctx,
            &scratch.norm_out,
            &scratch.selected_ids_buf,
            &mut scratch.gate_out_stacked,
            tokens,
            top_k,
            h,
            0,
        )?;
        stage_end(t0, ctx, &MOE_BATCHED_DECODE_GATE_US);

        // 3. Batched up gemv.
        let t0 = stage_t0();
        moe_layer.experts.gemv_up_batched(
            ctx,
            &scratch.norm_out,
            &scratch.selected_ids_buf,
            &mut scratch.up_out_stacked,
            tokens,
            top_k,
            h,
            0,
        )?;
        stage_end(t0, ctx, &MOE_BATCHED_DECODE_UP_US);

        // 4. SiLU·gate.
        let t0 = stage_t0();
        B::silu_mul_batched(
            ctx,
            &scratch.gate_out_stacked,
            &scratch.up_out_stacked,
            &mut scratch.silu_stacked,
            total_pairs,
            inter,
        )?;
        stage_end(t0, ctx, &MOE_BATCHED_DECODE_SILU_US);
    }

    // 5. Batched down gemv — src1 = silu_stacked [m, top_k, ffn]: each
    //    pair has its own row, outer = top_k * ffn, inner = ffn.
    let t0 = stage_t0();
    moe_layer.experts.gemv_down_batched(
        ctx,
        &scratch.silu_stacked,
        &scratch.selected_ids_buf,
        &mut scratch.down_out_stacked,
        tokens,
        top_k,
        top_k * inter, // outer: top_k * ffn floats per token
        inter,         // inner: ffn floats per slot
    )?;
    stage_end(t0, ctx, &MOE_BATCHED_DECODE_DOWN_US);

    // 6. Per-token weighted sum across slots → moe_out [m, h]. Caller
    //    does residual += moe_out at the end of forward_layer.
    let t0 = stage_t0();
    B::weighted_sum_batched(
        ctx,
        &scratch.down_out_stacked,
        &scratch.weights_2d,
        &mut scratch.moe_out,
        tokens,
        top_k,
        h,
    )?;
    stage_end(t0, ctx, &MOE_BATCHED_DECODE_WSUM_US);

    Ok(())
}
