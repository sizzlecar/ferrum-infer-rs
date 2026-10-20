//! `BackendPagedKv for CudaBackend` — paged-KV decode attention,
//! split-K dispatch, varlen prefill/decode, paged KV append.
//!
//! Extracted from `cuda/mod.rs` (#8 Phase 4). Owns:
//!
//! - `SplitKScratch` instance-owned f32 partial buffers for split-K
//!   phase 1 + `with_split_k_scratch`.
//! - `paged_varlen_split_k_dispatch` (split-K varlen path used by
//!   prefill on long contexts).
//! - `paged_batched_flash_dispatch` (Stage 13a batched paged-decode
//!   flash with split-K — `FERRUM_PAGED_FLASH=1`).
//! - `paged_batched_decode_single_pass` (default batched paged-decode
//!   single-pass path).
//! - `impl BackendPagedKv for CudaBackend` (~666 lines): the trait
//!   methods (`supports_paged_kv`, `paged_decode_attention`,
//!   `kv_append_paged`, `split_qkv_norm_rope_into_paged_cache_varlen`,
//!   etc.) that fan out to the dispatchers above.
//!
//! All helpers stay private to this module; the trait impl is the
//! only public surface.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use ferrum_types::{FerrumError, Result};
use half::f16;

use super::{CudaBackend, CudaState, BATCHED_SCRATCH_CAP};
use crate::backend::{Backend, BackendPagedKv, PagedAttentionLaunchGeometry};
use crate::ptx;

// ────────────────────────────────────────────────────────────────────────
// Paged-varlen split-K scratch + dispatch
// ────────────────────────────────────────────────────────────────────────

/// Per-context scratch for split-K phase1 outputs. Three buffers
/// (partial_out f32, partial_m f32, partial_l f32) sized to the largest
/// shape ever requested. Same lazy-grow pattern as Marlin gather scratch.
pub(super) struct SplitKScratch {
    partial_out: CudaSlice<f32>, // [M_total * num_q_heads * num_splits * head_dim]
    partial_m: CudaSlice<f32>,   // [M_total * num_q_heads * num_splits]
    partial_l: CudaSlice<f32>,   // [M_total * num_q_heads * num_splits]
    out_capacity: usize,
    ml_capacity: usize,
}
fn with_split_k_scratch<R>(
    ctx: &mut CudaState,
    out_required: usize,
    ml_required: usize,
    body: impl FnOnce(&mut CudaSlice<f32>, &mut CudaSlice<f32>, &mut CudaSlice<f32>) -> R,
) -> R {
    let need_new = ctx
        .split_k_scratch
        .as_ref()
        .is_none_or(|s| s.out_capacity < out_required || s.ml_capacity < ml_required);
    if need_new {
        assert!(
            !ctx.capture_in_flight,
            "split-K scratch growth during CUDA capture"
        );
        let partial_out = unsafe { ctx.stream.alloc::<f32>(out_required) }
            .expect("split-K partial_out allocation");
        let partial_m =
            unsafe { ctx.stream.alloc::<f32>(ml_required) }.expect("split-K partial_m allocation");
        let partial_l =
            unsafe { ctx.stream.alloc::<f32>(ml_required) }.expect("split-K partial_l allocation");
        let next = SplitKScratch {
            partial_out,
            partial_m,
            partial_l,
            out_capacity: out_required,
            ml_capacity: ml_required,
        };
        if let Some(previous) = ctx.split_k_scratch.replace(next) {
            ctx.captured_storage.push(Box::new(previous));
        }
    }
    let scratch = ctx
        .split_k_scratch
        .as_mut()
        .expect("prepared split-K scratch");
    body(
        &mut scratch.partial_out,
        &mut scratch.partial_m,
        &mut scratch.partial_l,
    )
}

/// Both preparation and dispatch use the original context's frozen policy.
fn varlen_geometry(
    ctx: &CudaState,
    num_seqs: usize,
    tokens: usize,
    max_kv: usize,
) -> PagedAttentionLaunchGeometry {
    PagedAttentionLaunchGeometry::cuda_varlen(
        ctx.configuration.cuda.split_k_attn,
        num_seqs,
        tokens,
        max_kv,
    )
}

fn varlen_phase(
    ctx: &mut CudaState,
    geometry: PagedAttentionLaunchGeometry,
) -> Result<CudaFunction> {
    let (module, name) = match geometry {
        PagedAttentionLaunchGeometry::SinglePass { .. } => {
            ("paged_varlen_attn", "paged_varlen_attn_f16")
        }
        PagedAttentionLaunchGeometry::SplitK { .. } => (
            "paged_varlen_split_k_phase1",
            "paged_varlen_attn_split_k_phase1_f16",
        ),
    };
    let function = ctx.func(module, ptx::PAGED_VARLEN_ATTENTION, name);
    let shared = geometry.shared_bytes()?;
    if shared > 48 * 1024 {
        let shared = i32::try_from(shared)
            .map_err(|_| FerrumError::model("paged attention dynamic shared extent exceeds i32"))?;
        function.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared,
        ).map_err(|error| FerrumError::model(format!("paged attention shared-memory opt-in: {error}")))?;
    }
    Ok(function)
}

#[allow(clippy::too_many_arguments)]
fn paged_varlen_split_k_dispatch(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    // Integer scratch: post-B-2 these come from `alloc_u32` → typed
    // `CudaSlice<u32>` storage. Old signature was `&CudaSlice<f16>`
    // (bit-tunnel through the f16-typed buffer) — kernels read these
    // as `int*` regardless, so the wire-level behavior is unchanged.
    cu_seqlens_q: &CudaSlice<u32>,
    pos_offsets: &CudaSlice<u32>,
    block_tables: &CudaSlice<u32>,
    num_seqs: usize,
    total_q_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    geometry: PagedAttentionLaunchGeometry,
) -> Result<()> {
    let PagedAttentionLaunchGeometry::SplitK { num_splits, .. } = geometry else {
        return Err(FerrumError::model(
            "split-K dispatcher received a different launch route",
        ));
    };
    let (out_required, ml_required) = geometry
        .split_scratch(total_q_tokens, num_heads, head_dim)?
        .expect("split-K geometry");
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();

    let phase1 = varlen_phase(ctx, geometry)?;
    let reduce = ctx.func(
        "paged_varlen_split_k_reduce",
        ptx::PAGED_VARLEN_ATTENTION,
        "paged_varlen_split_k_reduce_f16",
    );

    with_split_k_scratch(
        ctx,
        out_required,
        ml_required,
        |partial_out, partial_m, partial_l| {
            let qv = q.slice(..);
            let kp = k_pool.slice(..);
            let vp = v_pool.slice(..);
            let csq = cu_seqlens_q.slice(..);
            let po = pos_offsets.slice(..);
            let bt = block_tables.slice(..);
            let pout = partial_out.slice(..);
            let pm = partial_m.slice(..);
            let pl = partial_l.slice(..);
            let ns = num_seqs as i32;
            let nqi = num_heads as i32;
            let nkvi = num_kv_heads as i32;
            let hdi = head_dim as i32;
            let mbps = max_blocks_per_seq as i32;
            let bsi = block_size as i32;
            let swi = sliding_window as i32;
            let nsp = num_splits as i32;

            // Phase 1: (num_heads, M_total, num_splits)
            let mut b1 = stream.launch_builder(&phase1);
            b1.arg(&qv);
            b1.arg(&kp);
            b1.arg(&vp);
            b1.arg(&csq);
            b1.arg(&po);
            b1.arg(&bt);
            b1.arg(&pout);
            b1.arg(&pm);
            b1.arg(&pl);
            b1.arg(&ns);
            b1.arg(&nqi);
            b1.arg(&nkvi);
            b1.arg(&hdi);
            b1.arg(&mbps);
            b1.arg(&bsi);
            b1.arg(&scale);
            b1.arg(&swi);
            b1.arg(&nsp);
            let shmem1 = geometry.shared_bytes()?;
            unsafe {
                b1.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, total_q_tokens as u32, num_splits as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shmem1,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_varlen_split_k_phase1: {e}")))?;

            // Phase 2: (num_heads, M_total)
            let pout2 = partial_out.slice(..);
            let pm2 = partial_m.slice(..);
            let pl2 = partial_l.slice(..);
            let mut b2 = stream.launch_builder(&reduce);
            b2.arg(&pout2);
            b2.arg(&pm2);
            b2.arg(&pl2);
            b2.arg(out);
            b2.arg(&nqi);
            b2.arg(&hdi);
            b2.arg(&nsp);
            unsafe {
                b2.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, total_q_tokens as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_varlen_split_k_reduce: {e}")))?;

            Ok::<(), FerrumError>(())
        },
    )
}

// ────────────────────────────────────────────────────────────────────────
// Stage 13a — Batched paged-decode flash (split-K)
//
// Same idea as the varlen split-K path but for q_len=1 batched decode
// (gridDim.y = num_seqs). Phase 1 splits each seq's kv across `num_splits`
// chunks; phase 2 reduces partials per (seq, head).
//
// FERRUM_PAGED_FLASH=1 selects this over the single-pass kernel. Default
// OFF.
// ────────────────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn paged_batched_flash_dispatch(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    // See paged_varlen_split_k_dispatch — block_tables / valid_kv_lens
    // are u32 storage after B-2 (was f16-bit-tunneled).
    block_tables: &CudaSlice<u32>,
    valid_kv_lens: &CudaSlice<u32>,
    num_seqs: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    // Pick num_splits. Heuristic combines two effects:
    //   1. SM occupancy: a (num_q_heads, num_seqs, splits) grid wants
    //      total blocks ≳ 2 × SM count for full pipelining. When the
    //      base grid (num_seqs × num_heads) already saturates SMs,
    //      splits ≥ 2 just add launch + reduce overhead.
    //   2. kv_len: longer kv → more inherent serial work in step 3 →
    //      split-K helps even at moderate occupancy.
    //
    // Bench (RTX 4090, M3, 32 q_heads):
    //   c=1  splits=8 → +22% (grid was 1/4 wave, splits saturate)
    //   c=16 splits=2 → -3.7% (grid already 4 waves)
    //   c=32 splits=4 → +4.7% (grid 8 waves, kv split still helps — 2026-05-27 clean-regime
    //                          re-test post-DECODE_OP_PROFILE removal showed splits=2
    //                          actually -5% and splits=4 is the right answer at this wave
    //                          count, even when kv≤768; old "splits=2 +5%" claim was
    //                          measured under sync-instrumented bench).
    // Override via FERRUM_PAGED_FLASH_SPLITS for tuning.
    let force_splits = ctx.configuration.cuda.paged_flash_splits;
    // 128 SMs on Ada/Hopper-class; conservative under-estimate.
    const SM_TARGET: usize = 128;
    let base_grid = num_seqs * num_heads;
    let saturated = base_grid >= 2 * SM_TARGET;
    let waves = base_grid / SM_TARGET; // c=16:4, c=32:8 on 32-q-head models
    let num_splits: usize = force_splits.unwrap_or_else(|| {
        if saturated {
            // Heavy concurrency (waves ≥ 8 = c≥32 on 32-q-head): even short
            // kv benefits from splits=4 because V-loop work per (q_head,seq)
            // is still serial inside one CTA; splitting recovers parallelism
            // the saturated base_grid can't expose.
            // Mid concurrency (waves 4-7 = c≈16): splits=2 hurts (-3.7%),
            // keep splits=1 for short kv.
            match (max_kv_len, waves) {
                (kv, _) if kv > 2048 => 4,
                (kv, _) if kv <= 768 && waves >= 8 => 4,
                (kv, _) if kv <= 768 => 1,
                _ => 2,
            }
        } else {
            // Low concurrency: aggressive splits to fill SMs.
            let needed = (SM_TARGET + base_grid - 1) / base_grid;
            let by_kv = match max_kv_len {
                kv if kv <= 256 => 4,
                kv if kv <= 1024 => 8,
                _ => 16,
            };
            needed.max(1).min(by_kv).min(16)
        }
    });
    if num_splits <= 1 {
        // Caller's main path is the single-pass kernel.
        // FERRUM_PAGED_FLASH=1 still routes here, so do the single-pass
        // launch inline (avoids env-flag round-trip).
        return paged_batched_decode_single_pass(
            ctx,
            q,
            k_pool,
            v_pool,
            out,
            block_tables,
            valid_kv_lens,
            num_seqs,
            max_kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
        );
    }

    let chunk = (max_kv_len + num_splits - 1) / num_splits;
    let total_qh = num_seqs * num_heads;
    let out_required = total_qh * num_splits * head_dim;
    let ml_required = total_qh * num_splits;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();

    let phase1 = ctx.func(
        "paged_batched_flash_attn",
        ptx::PAGED_DECODE_ATTENTION,
        "paged_batched_flash_decode_attn_f16",
    );
    let phase2 = ctx.func(
        "paged_batched_flash_reduce",
        ptx::PAGED_DECODE_ATTENTION,
        "paged_batched_flash_decode_reduce_f16",
    );
    let safe_kv = ctx.configuration.cuda.kv_capacity;

    with_split_k_scratch(
        ctx,
        out_required,
        ml_required,
        |partial_out, partial_m, partial_l| {
            let qv = q.slice(..);
            let kp = k_pool.slice(..);
            let vp = v_pool.slice(..);
            let bt = block_tables.slice(..);
            let kvl = valid_kv_lens.slice(..);
            let pout = partial_out.slice(..);
            let pm = partial_m.slice(..);
            let pl = partial_l.slice(..);
            let nqi = num_heads as i32;
            let nkvi = num_kv_heads as i32;
            let hdi = head_dim as i32;
            let mbps = max_blocks_per_seq as i32;
            let bsi = block_size as i32;
            let nsp = num_splits as i32;

            let mut b1 = stream.launch_builder(&phase1);
            b1.arg(&qv);
            b1.arg(&kp);
            b1.arg(&vp);
            b1.arg(&bt);
            b1.arg(&kvl);
            b1.arg(&pout);
            b1.arg(&pm);
            b1.arg(&pl);
            b1.arg(&nqi);
            b1.arg(&nkvi);
            b1.arg(&hdi);
            b1.arg(&mbps);
            b1.arg(&bsi);
            b1.arg(&scale);
            b1.arg(&nsp);
            // Match graph-capture sizing rationale used elsewhere — size
            // shared to FERRUM_KV_CAPACITY ceiling, not current chunk.
            let safe_chunk = (safe_kv + num_splits - 1) / num_splits;
            let shmem1 = (safe_chunk.max(chunk).max(1) as u32) * 4;
            unsafe {
                b1.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, num_seqs as u32, num_splits as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shmem1,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_batched_flash phase1: {e}")))?;

            let pout2 = partial_out.slice(..);
            let pm2 = partial_m.slice(..);
            let pl2 = partial_l.slice(..);
            let mut b2 = stream.launch_builder(&phase2);
            b2.arg(&pout2);
            b2.arg(&pm2);
            b2.arg(&pl2);
            b2.arg(out);
            b2.arg(&nqi);
            b2.arg(&hdi);
            b2.arg(&nsp);
            unsafe {
                b2.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, num_seqs as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_batched_flash phase2: {e}")))?;

            Ok::<(), FerrumError>(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn paged_batched_decode_single_pass(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    // U32 typed storage post-B-2; kernel reads as int*.
    block_tables: &CudaSlice<u32>,
    valid_kv_lens: &CudaSlice<u32>,
    num_seqs: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    let func = ctx.func(
        "paged_batched_decode_attn",
        ptx::PAGED_DECODE_ATTENTION,
        "paged_batched_decode_attn_f16",
    );
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();
    let qv = q.slice(..);
    let kp = k_pool.slice(..);
    let vp = v_pool.slice(..);
    let bt = block_tables.slice(..);
    let kvl = valid_kv_lens.slice(..);
    let nqi = num_heads as i32;
    let nkvi = num_kv_heads as i32;
    let hdi = head_dim as i32;
    let mbps = max_blocks_per_seq as i32;
    let bsi = block_size as i32;
    // Size shared to FERRUM_KV_CAPACITY ceiling (graph-replay safe; see the
    // comment block in `paged_varlen_attention`). Compute before the launch
    // builder so we can opt into the extended dynamic-shared limit if the
    // request exceeds the 48 KB default — without the opt-in cuLaunchKernel
    // returns CUDA_ERROR_INVALID_VALUE.
    let safe_kv_max = ctx.configuration.cuda.kv_capacity;
    let shared_kv = safe_kv_max.max(max_kv_len).max(1);
    let shared_bytes = (shared_kv as u32) * 4;
    if shared_bytes > 48 * 1024 {
        let _ = func.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_bytes as i32,
        );
    }
    let mut b = stream.launch_builder(&func);
    b.arg(&qv);
    b.arg(&kp);
    b.arg(&vp);
    b.arg(&bt);
    b.arg(&kvl);
    b.arg(out);
    b.arg(&nqi);
    b.arg(&nkvi);
    b.arg(&hdi);
    b.arg(&mbps);
    b.arg(&bsi);
    b.arg(&scale);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_bytes,
        })
    }
    .map(|_| ())
    .map_err(|e| FerrumError::model(format!("paged_batched_decode_attn: {e}")))
}

impl BackendPagedKv for CudaBackend {
    /// Default ON for CUDA. Mixed-batch unified_forward path requires
    /// paged_pools; without it the engine's run_unified_iter falls back
    /// to serial prefill that stalls in-flight decoders (~50% of bench
    /// wall time at c=16). Override via FERRUM_METAL_PAGED_KV=0 if a
    /// caller specifically wants legacy contig KV.
    fn supports_paged_kv() -> bool {
        true
    }
    fn supports_windowed_paged_attention() -> bool {
        true
    }
    fn supports_varlen_qkv() -> bool {
        true
    }
    fn supports_qwen35_paged_qkv() -> bool {
        true
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    fn supports_qwen35_paged_qkv_vllm() -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_split_qkv_norm_rope_into_paged_cache_varlen(
        ctx: &mut Self::Context,
        query_raw: &Self::Buffer,
        key_raw: &Self::Buffer,
        value_raw: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        token_seq_indices: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        total_q_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        q_proj_stride: usize,
        q_head_stride: usize,
        kv_proj_stride: usize,
        eps: f32,
        qk_mode: i32,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 || total_q_tokens == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "qwen35_paged_qkv",
            ptx::QWEN35_PAGED_QKV,
            "qwen35_split_qkv_norm_rope_into_paged_cache_varlen_f16",
        );
        let stream = ctx.stream.clone();
        let num_seqs_i32 = num_seqs as i32;
        let total_q_tokens_i32 = total_q_tokens as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let rope_dim_i32 = rope_dim as i32;
        let q_proj_stride_i32 = q_proj_stride as i32;
        let q_head_stride_i32 = q_head_stride as i32;
        let kv_proj_stride_i32 = kv_proj_stride as i32;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let qk_mode_i32 = qk_mode;
        let mut b = stream.launch_builder(&func);
        b.arg(query_raw);
        b.arg(key_raw);
        b.arg(value_raw);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(cu_seqlens_q);
        b.arg(token_seq_indices);
        b.arg(pos_offsets);
        b.arg(block_tables);
        b.arg(&num_seqs_i32);
        b.arg(&total_q_tokens_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&rope_dim_i32);
        b.arg(&q_proj_stride_i32);
        b.arg(&q_head_stride_i32);
        b.arg(&kv_proj_stride_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_q_tokens as u32, (q_heads + 2 * kv_heads) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| {
            FerrumError::model(format!(
                "qwen35_split_qkv_norm_rope_into_paged_cache_varlen: {e}"
            ))
        })
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm(
        ctx: &mut Self::Context,
        query_raw: &Self::Buffer,
        key_raw: &Self::Buffer,
        value_raw: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        token_seq_indices: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        total_q_tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        q_proj_stride: usize,
        q_head_stride: usize,
        kv_proj_stride: usize,
        eps: f32,
        qk_mode: i32,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 || total_q_tokens == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "qwen35_paged_qkv",
            ptx::QWEN35_PAGED_QKV,
            "qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16",
        );
        let stream = ctx.stream.clone();
        let num_seqs_i32 = num_seqs as i32;
        let total_q_tokens_i32 = total_q_tokens as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let rope_dim_i32 = rope_dim as i32;
        let q_proj_stride_i32 = q_proj_stride as i32;
        let q_head_stride_i32 = q_head_stride as i32;
        let kv_proj_stride_i32 = kv_proj_stride as i32;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let qk_mode_i32 = qk_mode;
        let mut b = stream.launch_builder(&func);
        b.arg(query_raw);
        b.arg(key_raw);
        b.arg(value_raw);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(cu_seqlens_q);
        b.arg(token_seq_indices);
        b.arg(pos_offsets);
        b.arg(block_tables);
        b.arg(&num_seqs_i32);
        b.arg(&total_q_tokens_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&rope_dim_i32);
        b.arg(&q_proj_stride_i32);
        b.arg(&q_head_stride_i32);
        b.arg(&kv_proj_stride_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_q_tokens as u32, (q_heads + 2 * kv_heads) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| {
            FerrumError::model(format!(
                "qwen35_split_qkv_norm_rope_into_paged_cache_varlen_vllm: {e}"
            ))
        })
    }

    fn populate_batched_pointers(
        ctx: &mut Self::Context,
        k_caches: &[&Self::Buffer],
        v_caches: &[&Self::Buffer],
        num_layers: usize,
        m: usize,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        if num_layers == 0 || m == 0 {
            return Ok(());
        }
        if num_layers > super::MAX_LAYERS_FOR_GRAPH {
            return Err(FerrumError::model(format!(
                "populate_batched_pointers: num_layers={num_layers} > MAX_LAYERS_FOR_GRAPH={}",
                super::MAX_LAYERS_FOR_GRAPH
            )));
        }
        if m > BATCHED_SCRATCH_CAP {
            return Err(FerrumError::model(format!(
                "populate_batched_pointers: m={m} > BATCHED_SCRATCH_CAP={BATCHED_SCRATCH_CAP}",
            )));
        }
        if k_caches.len() != num_layers * m || v_caches.len() != num_layers * m {
            return Err(FerrumError::model(
                "populate_batched_pointers: k/v_caches length != num_layers * m",
            ));
        }

        if ctx.capture_in_flight {
            return Err(FerrumError::backend(
                "batched pointer preparation must precede capture",
            ));
        }
        let stream = ctx.stream.clone();
        // These are the SAME original-context arrays consumed by append and
        // attention nodes, with distinct slots for every layer and K/V role.
        // Graph nodes upload them on each replay; no second device-table owner
        // or temporary host allocation is introduced here.
        super::with_batched_scratch_mut(&mut ctx.batched_scratch, |tables| {
            for li in 0..num_layers {
                let k_off = li * BATCHED_SCRATCH_CAP;
                let v_off = (li + super::MAX_LAYERS_FOR_GRAPH) * BATCHED_SCRATCH_CAP;
                for i in 0..m {
                    let (kp, _) = k_caches[li * m + i].as_f16().device_ptr(&stream);
                    let (vp, _) = v_caches[li * m + i].as_f16().device_ptr(&stream);
                    tables.host_cache_ptrs[k_off + i] = kp;
                    tables.host_cache_ptrs[v_off + i] = vp;
                    tables.host_k_ptrs[k_off + i] = kp;
                    tables.host_v_ptrs[k_off + i] = vp;
                }
            }
        });
        Ok(())
    }

    fn prepare_paged_varlen_attention(
        ctx: &mut Self::Context,
        num_seqs: usize,
        total_q_tokens: usize,
        max_kv_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<PagedAttentionLaunchGeometry> {
        if ctx.capture_in_flight {
            return Err(FerrumError::unsupported(
                "varlen attention preparation must precede capture",
            ));
        }
        let geometry = varlen_geometry(ctx, num_seqs, total_q_tokens, max_kv_len);
        // Load the same phase and validate extended shared memory before capture.
        let _ = varlen_phase(ctx, geometry)?;
        if let Some((out, ml)) = geometry.split_scratch(total_q_tokens, num_heads, head_dim)? {
            with_split_k_scratch(ctx, out, ml, |_, _, _| ());
            let _ = ctx.func(
                "paged_varlen_split_k_reduce",
                ptx::PAGED_VARLEN_ATTENTION,
                "paged_varlen_split_k_reduce_f16",
            );
        }
        Ok(geometry)
    }

    fn paged_varlen_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        total_q_tokens: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 || total_q_tokens == 0 {
            return Ok(());
        }

        // Auto-tune: split-K helps under-occupied grids at low concurrency
        // or with long context. The heuristic gates it to those regions.
        // FERRUM_SPLIT_K_ATTN=1 forces on, FERRUM_SPLIT_K_ATTN=0 forces off.
        //
        // total_q_tokens cap: the unified prefill path can pack
        // O(thousands) of q tokens into a single call (32 prefills × ~150
        // tokens each). partial_out = total_q × heads × splits × head_dim
        // f32 grows to hundreds of MB — observed OOM at M3 c=32 unified
        // (4800 × 32 × 8 × 128 × 4 ≈ 628 MB). Skip split-K when the grid
        // is already well-occupied via q-tokens (>64) — the per-call
        // transient cost dominates any kernel win.
        let geometry = varlen_geometry(ctx, num_seqs, total_q_tokens, max_kv_len);

        if matches!(geometry, PagedAttentionLaunchGeometry::SplitK { .. }) {
            return paged_varlen_split_k_dispatch(
                ctx,
                q.as_f16(),
                k_pool.as_f16(),
                v_pool.as_f16(),
                out.as_f16_mut(),
                cu_seqlens_q.as_u32(),
                pos_offsets.as_u32(),
                block_tables.as_u32(),
                num_seqs,
                total_q_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                block_size,
                max_blocks_per_seq,
                geometry,
            );
        }

        let func = varlen_phase(ctx, geometry)?;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        // After B-2 typed-buffer migration: integer scratch
        // (cu_seqlens_q / pos_offsets / block_tables) is stored in
        // CudaBuf::U32, the kernel reads them as `int*` — same wire-level
        // behavior as the prior f16-bit-tunneled layout but with a
        // dtype tag that prevents byte-count miscalculation.
        let qv = q.as_f16().slice(..);
        let kp = k_pool.as_f16().slice(..);
        let vp = v_pool.as_f16().slice(..);
        let csq = cu_seqlens_q.as_u32().slice(..);
        let po = pos_offsets.as_u32().slice(..);
        let bt = block_tables.as_u32().slice(..);
        let ns = num_seqs as i32;
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_blocks_per_seq as i32;
        let bsi = block_size as i32;
        let swi = sliding_window as i32;
        // Preparation, dispatch and graph equality share this exact extent.
        let shared_bytes = geometry.shared_bytes()?;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&csq);
        b.arg(&po);
        b.arg(&bt);
        b.arg(out);
        b.arg(&ns);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mbps);
        b.arg(&bsi);
        b.arg(&scale);
        b.arg(&swi);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, total_q_tokens as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("paged_varlen_attn: {e}")))
    }

    fn paged_varlen_attention_fa2_ffi(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        lse: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        seq_lens: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        total_q_tokens: usize,
        max_q_len: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        #[cfg(feature = "fa2-source")]
        if ctx.configuration.cuda.fa2_source {
            return super::fa2_source::paged_varlen_attention_fa2_source(
                super::fa2_source::Fa2SourcePagedVarlenArgs {
                    stream: &ctx.stream,
                    q: q.as_f16(),
                    k_pool: k_pool.as_f16(),
                    v_pool: v_pool.as_f16(),
                    out: out.as_f16_mut(),
                    lse: lse.as_f32_mut(),
                    cu_seqlens_q: cu_seqlens_q.as_u32(),
                    seq_lens: seq_lens.as_u32(),
                    block_tables: block_tables.as_u32(),
                    num_seqs,
                    total_q_tokens,
                    max_q_len,
                    max_kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    max_blocks_per_seq,
                },
            );
        }

        super::fa2_ffi::paged_varlen_attention_fa2_ffi(
            ctx.fa2_shim.as_ref().ok_or_else(|| {
                FerrumError::unsupported("FA2 direct FFI requires an explicitly configured shim")
            })?,
            &ctx.stream,
            q.as_f16(),
            k_pool.as_f16(),
            v_pool.as_f16(),
            out.as_f16_mut(),
            lse.as_f32_mut(),
            cu_seqlens_q.as_u32(),
            seq_lens.as_u32(),
            block_tables.as_u32(),
            num_seqs,
            total_q_tokens,
            max_q_len,
            max_kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
        )
    }

    /// Paged attention dispatcher (CUDA-only, replaces missing native
    /// `paged_decode_attention`). Routes:
    ///   - q_len==1 (decode for any num_seqs): paged_batched_decode_attention.
    ///     The layouts coincide for q_len==1 — a [num_seqs, heads, dim]
    ///     buffer is identical to [heads, num_seqs, dim] when seen as a
    ///     single seq's contribution, so no transpose is needed.
    ///   - q_len>1 (prefill, single-seq only): paged_varlen_attention.
    ///     Caller's q is `[heads, q_len, dim]` (head-major) but varlen
    ///     reads `[q_len, heads, dim]` (token-major), so we transpose
    ///     in/out around the call. Cold path (prefill is rare per token).
    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        context_lens: &Self::Buffer,
        num_seqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
        q_len: usize,
        sliding_window: usize,
        host_context_lens: Option<&[u32]>,
    ) -> Result<()> {
        if q_len == 0 || num_seqs == 0 {
            return Ok(());
        }
        let max_kv_len = block_size
            .checked_mul(max_num_blocks_per_seq)
            .ok_or_else(|| FerrumError::model("paged attention capacity overflow"))?;

        if q_len == 1 && sliding_window == 0 {
            return Self::paged_batched_decode_attention(
                ctx,
                q,
                k_pool,
                v_pool,
                out,
                block_tables,
                context_lens,
                num_seqs,
                max_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_num_blocks_per_seq,
            );
        }

        // q_len > 1: this prefill wrapper accepts one sequence per call.
        if q_len > 1 && num_seqs != 1 {
            return Err(FerrumError::model(format!(
                "paged_decode_attention(CUDA): q_len={q_len} num_seqs={num_seqs} \
                 not supported (caller must split prefill into per-seq calls)"
            )));
        }

        // This wrapper reads host lengths and prepares metadata. A graph must
        // instead call the device-staged varlen entry with durable caller views.
        if ctx.capture_in_flight {
            return Err(FerrumError::unsupported(
                "windowed/prefill paged wrapper must prepare metadata outside graph capture",
            ));
        }
        let total_q_tokens = num_seqs
            .checked_mul(q_len)
            .ok_or_else(|| FerrumError::model("paged query extent overflow"))?;
        // The producer already owns these exact lengths. Do not read them
        // back from the device, which would add a barrier at every local layer.
        // Preserve only the existing full-attention prefill fallback for direct
        // callers that do not have a host projection.
        let copied_lengths;
        let cl_host = match host_context_lens {
            Some(lengths) if lengths.len() == num_seqs => lengths,
            Some(_) => return Err(FerrumError::model("paged host context length count mismatch")),
            None if sliding_window != 0 => return Err(FerrumError::unsupported(
                "windowed paged attention requires original host lengths or device-staged varlen metadata",
            )),
            None => {
                let stream = ctx.stream.clone();
                let view = context_lens.as_u32().slice(0..num_seqs);
                let mut h = vec![0u32; num_seqs];
                stream.memcpy_dtoh(&view, h.as_mut_slice())
                    .map_err(|e| FerrumError::model(format!("dtoh context_lens: {e}")))?;
                stream.synchronize()
                    .map_err(|e| FerrumError::model(format!("dtoh sync: {e}")))?;
                copied_lengths = h;
                &copied_lengths
            }
        };
        let mut cu_seqlens = Vec::with_capacity(num_seqs + 1);
        for seq in 0..=num_seqs {
            cu_seqlens.push(
                u32::try_from(
                    seq.checked_mul(q_len)
                        .ok_or_else(|| FerrumError::model("paged query offset overflow"))?,
                )
                .map_err(|_| FerrumError::model("paged query offset exceeds u32"))?,
            );
        }
        let positions: Vec<u32> = cl_host
            .iter()
            .map(|&len| {
                (len as usize)
                    .checked_sub(q_len)
                    .and_then(|pos| u32::try_from(pos).ok())
                    .ok_or_else(|| FerrumError::model("paged query exceeds final KV length"))
            })
            .collect::<Result<_>>()?;
        let final_kv_len = cl_host.iter().copied().max().unwrap() as usize;
        if final_kv_len > max_kv_len {
            return Err(FerrumError::model(
                "paged context exceeds declared block-table capacity",
            ));
        }
        if ctx.paged_attn_metadata_capacity < num_seqs {
            let grown = num_seqs.checked_next_power_of_two().unwrap_or(num_seqs);
            let stream = ctx.stream.clone();
            let next_cu = crate::backend::CudaBuf::from_u32(
                stream
                    .alloc_zeros::<u32>(grown + 1)
                    .map_err(|e| FerrumError::model(format!("paged query metadata: {e}")))?,
            );
            if let Some(previous) = ctx.paged_attn_cu_seqlens.replace(next_cu) {
                ctx.captured_storage.push(Box::new(previous));
            }
            let next_pos = crate::backend::CudaBuf::from_u32(
                stream
                    .alloc_zeros::<u32>(grown)
                    .map_err(|e| FerrumError::model(format!("paged position metadata: {e}")))?,
            );
            if let Some(previous) = ctx.paged_attn_pos_offsets.replace(next_pos) {
                ctx.captured_storage.push(Box::new(previous));
            }
            ctx.paged_attn_metadata_capacity = grown;
        }
        // The original context owns these allocations through all queued work.
        // Backend writes/varlen dispatch do not mutate these metadata fields.
        let cu_ptr = ctx.paged_attn_cu_seqlens.as_mut().unwrap() as *mut crate::backend::CudaBuf;
        let pos_ptr = ctx.paged_attn_pos_offsets.as_mut().unwrap() as *mut crate::backend::CudaBuf;
        unsafe {
            <Self as Backend>::write_typed::<u32>(ctx, &mut *cu_ptr, &cu_seqlens);
            <Self as Backend>::write_typed::<u32>(ctx, &mut *pos_ptr, &positions);
        }
        if q_len == 1 {
            // For one query per sequence, token/head-major layouts coincide.
            return unsafe {
                Self::paged_varlen_attention(
                    ctx,
                    q,
                    k_pool,
                    v_pool,
                    out,
                    &*cu_ptr,
                    &*pos_ptr,
                    block_tables,
                    num_seqs,
                    total_q_tokens,
                    final_kv_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    sliding_window,
                    block_size,
                    max_num_blocks_per_seq,
                )
            };
        }

        // The paged-write kernel produces q_out[token, head, dim], which
        // paged_varlen_attention reads directly. Its output is also
        // token-major; this wrapper's output contract is head-major, so
        // transpose token→head into `out` after attention completes.
        let q_n = q_len * num_heads * head_dim;

        // Lazy-grow the cached token-major output scratch. Stable
        // address across calls — required to avoid stream-ordered
        // free / kernel-still-running races at higher concurrency.
        if ctx.paged_attn_out_tm_capacity < q_n {
            let stream = ctx.stream.clone();
            let n_grown = q_n.next_power_of_two().max(q_n);
            assert!(
                !ctx.capture_in_flight,
                "paged output scratch growth during CUDA capture"
            );
            let next = crate::backend::CudaBuf::from_f16(
                stream
                    .alloc_zeros::<f16>(n_grown)
                    .map_err(|e| FerrumError::model(format!("alloc paged_attn_out_tm: {e}")))?,
            );
            if let Some(previous) = ctx.paged_attn_out_tm.replace(next) {
                ctx.captured_storage.push(Box::new(previous));
            }
            ctx.paged_attn_out_tm_capacity = n_grown;
        }

        // SAFETY: varlen dispatch and transpose may update other context
        // scratch, but do not replace the metadata/output fields above.
        // We take a raw pointer to the cached
        // buffer so we can pass it as a normal &mut/& while ctx is also
        // borrowed by the kernel-call methods.
        let out_tm_ptr: *mut crate::backend::CudaBuf =
            ctx.paged_attn_out_tm
                .as_mut()
                .expect("paged_attn_out_tm allocated") as *mut _;
        unsafe {
            Self::paged_varlen_attention(
                ctx,
                q,
                k_pool,
                v_pool,
                &mut *out_tm_ptr,
                &*cu_ptr,
                &*pos_ptr,
                block_tables,
                1,
                q_len,
                final_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window,
                block_size,
                max_num_blocks_per_seq,
            )?;

            // Restore head-major layout: [q_len, heads, hd] → [heads, q_len, hd]
            // → caller's `out` buffer.
            <Self as Backend>::transpose_token_to_head(
                ctx,
                &*out_tm_ptr,
                out,
                q_len,
                num_heads,
                head_dim,
            );
        }

        Ok(())
    }
    fn paged_batched_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        valid_kv_lens: &Self::Buffer,
        num_seqs: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 {
            return Ok(());
        }

        // Stage 13a: split-K path for batched paged decode. Default ON;
        // smart heuristic auto-tunes splits based on (num_seqs × num_heads)
        // and kv_len, so it falls back to single-pass when grid already
        // saturates SMs at low kv. Bench M3 c=1/8/16/32 across +21% / +3% /
        // +3.5% / +10.8% over the single-pass kernel — every concurrency
        // wins. Set FERRUM_PAGED_FLASH=0 to opt out.
        if ctx.configuration.cuda.paged_flash {
            return paged_batched_flash_dispatch(
                ctx,
                q.as_f16(),
                k_pool.as_f16(),
                v_pool.as_f16(),
                out.as_f16_mut(),
                block_tables.as_u32(),
                valid_kv_lens.as_u32(),
                num_seqs,
                max_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
            );
        }

        let func = ctx.func(
            "paged_batched_decode_attn",
            ptx::PAGED_DECODE_ATTENTION,
            "paged_batched_decode_attn_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        let qv = q.as_f16().slice(..);
        let kp = k_pool.as_f16().slice(..);
        let vp = v_pool.as_f16().slice(..);
        // block_tables / valid_kv_lens are CudaBuf::U32 (typed since B-2);
        // slice the U32 variant to give the kernel an int* it can read.
        let bt = block_tables.as_u32().slice(..);
        let kvl = valid_kv_lens.as_u32().slice(..);
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_blocks_per_seq as i32;
        let bsi = block_size as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&bt);
        b.arg(&kvl);
        b.arg(out);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mbps);
        b.arg(&bsi);
        b.arg(&scale);
        // Same shared-mem sizing rationale as paged_varlen_attention
        // (graph capture freezes shared_mem_bytes; size to
        // FERRUM_KV_CAPACITY ceiling).
        let shared_kv = ctx.configuration.cuda.shared_kv_for(max_kv_len);
        let shared_bytes = (shared_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, num_seqs as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("paged_batched_decode_attn: {e}")))
    }
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        qkv_byte_offset: u64,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        q_out_byte_offset: u64,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        block_table: &Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "split_qkv_norm_rope_into_paged_cache",
            ptx::SPLIT_QKV_NORM_ROPE_INTO_PAGED_CACHE,
            "split_qkv_norm_rope_into_paged_cache_f16",
        );
        let stream = ctx.stream.clone();
        let qkv_byte_offset_u64 = qkv_byte_offset;
        let q_out_byte_offset_u64 = q_out_byte_offset;
        let tokens_i32 = tokens as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let pos_offset_i32 = pos_offset as i32;
        let cache_len_i32 = cache_len as i32;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let qk_mode_i32 = qk_mode;
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(&qkv_byte_offset_u64);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(&q_out_byte_offset_u64);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(block_table);
        b.arg(&tokens_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&pos_offset_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&cache_len_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        let total_heads = (q_heads + 2 * kv_heads) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, total_heads, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("split_qkv_norm_rope_into_paged_cache: {e}")))
    }
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_varlen(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        m_total: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
        qk_mode: i32,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if m_total == 0 || num_seqs == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "split_qkv_norm_rope_into_paged_cache_varlen",
            ptx::SPLIT_QKV_NORM_ROPE_INTO_PAGED_CACHE,
            "split_qkv_norm_rope_into_paged_cache_varlen_f16",
        );
        let stream = ctx.stream.clone();
        let num_seqs_i32 = num_seqs as i32;
        let m_total_i32 = m_total as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let qk_mode_i32 = qk_mode;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(cu_seqlens_q);
        b.arg(pos_offsets);
        b.arg(block_tables);
        b.arg(&num_seqs_i32);
        b.arg(&m_total_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        let total_heads = (q_heads + 2 * kv_heads) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (m_total as u32, total_heads, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| {
            FerrumError::model(format!("split_qkv_norm_rope_into_paged_cache_varlen: {e}"))
        })
    }

    // ─────────── vLLM-layout paged write + paged_attention_v2 ───────────

    #[cfg(feature = "vllm-paged-attn-v2")]
    fn supports_vllm_paged_attn() -> bool {
        true
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_vllm(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        qkv_byte_offset: u64,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        q_out_byte_offset: u64,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        block_table: &Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "split_qkv_norm_rope_into_paged_cache_vllm",
            ptx::SPLIT_QKV_NORM_ROPE_INTO_PAGED_CACHE_VLLM,
            "split_qkv_norm_rope_into_paged_cache_vllm_f16",
        );
        let stream = ctx.stream.clone();
        let qkv_byte_offset_u64 = qkv_byte_offset;
        let q_out_byte_offset_u64 = q_out_byte_offset;
        let tokens_i32 = tokens as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let pos_offset_i32 = pos_offset as i32;
        let cache_len_i32 = cache_len as i32;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let qk_mode_i32 = qk_mode;
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(&qkv_byte_offset_u64);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(&q_out_byte_offset_u64);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(block_table);
        b.arg(&tokens_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&pos_offset_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&cache_len_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        let total_heads = (q_heads + 2 * kv_heads) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, total_heads, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("split_qkv_norm_rope_into_paged_cache_vllm: {e}")))
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_varlen_vllm(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        m_total: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
        qk_mode: i32,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if m_total == 0 || num_seqs == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "split_qkv_norm_rope_into_paged_cache_varlen_vllm",
            ptx::SPLIT_QKV_NORM_ROPE_INTO_PAGED_CACHE_VLLM,
            "split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16",
        );
        let stream = ctx.stream.clone();
        let num_seqs_i32 = num_seqs as i32;
        let m_total_i32 = m_total as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let qk_mode_i32 = qk_mode;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(cu_seqlens_q);
        b.arg(pos_offsets);
        b.arg(block_tables);
        b.arg(&num_seqs_i32);
        b.arg(&m_total_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        let total_heads = (q_heads + 2 * kv_heads) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (m_total as u32, total_heads, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| {
            FerrumError::model(format!(
                "split_qkv_norm_rope_into_paged_cache_varlen_vllm: {e}"
            ))
        })
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention_v2(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        context_lens: &Self::Buffer,
        num_seqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
        max_seq_len: usize,
    ) -> Result<()> {
        if num_seqs == 0 {
            return Ok(());
        }
        super::vllm_paged_attn::dispatch_paged_attention_v2(
            ctx,
            out.as_f16_mut(),
            q.as_f16(),
            k_pool.as_f16(),
            v_pool.as_f16(),
            block_tables.as_u32(),
            context_lens.as_u32(),
            num_seqs,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks_per_seq,
            max_seq_len,
        )
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_vllm_layout(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        context_lens: &Self::Buffer,
        num_seqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
        q_len: usize,
    ) -> Result<()> {
        if q_len == 0 || num_seqs == 0 {
            return Ok(());
        }
        if num_seqs != 1 {
            return Err(FerrumError::model(format!(
                "paged_varlen_attention_vllm_layout(CUDA): q_len={q_len} num_seqs={num_seqs} \
                 not supported yet"
            )));
        }

        let final_kv_len = {
            let stream = ctx.stream.clone();
            let view = context_lens.as_u32().slice(0..1);
            let mut host = vec![0u32; 1];
            stream
                .memcpy_dtoh(&view, host.as_mut_slice())
                .map_err(|e| FerrumError::model(format!("dtoh context_lens vllm: {e}")))?;
            stream
                .synchronize()
                .map_err(|e| FerrumError::model(format!("dtoh sync vllm: {e}")))?;
            host[0] as usize
        };
        if final_kv_len < q_len {
            return Err(FerrumError::model(format!(
                "paged_varlen_attention_vllm_layout(CUDA): final_kv_len={final_kv_len} < q_len={q_len}"
            )));
        }
        let pos_offset = (final_kv_len - q_len) as u32;
        let mut cu_seqlens_q_buf =
            <Self as Backend>::alloc_typed_in_context(ctx, crate::backend::Dtype::U32, 2);
        <Self as Backend>::write_typed::<u32>(ctx, &mut cu_seqlens_q_buf, &[0u32, q_len as u32]);
        let mut pos_offsets_buf =
            <Self as Backend>::alloc_typed_in_context(ctx, crate::backend::Dtype::U32, 1);
        <Self as Backend>::write_typed::<u32>(ctx, &mut pos_offsets_buf, &[pos_offset]);

        let q_n = q_len * num_heads * head_dim;
        if ctx.paged_attn_out_tm_capacity < q_n {
            let stream = ctx.stream.clone();
            let n_grown = q_n.next_power_of_two().max(q_n);
            assert!(
                !ctx.capture_in_flight,
                "paged output scratch growth during CUDA capture"
            );
            let next =
                crate::backend::CudaBuf::from_f16(stream.alloc_zeros::<f16>(n_grown).map_err(
                    |e| FerrumError::model(format!("alloc paged_attn_out_tm vllm: {e}")),
                )?);
            if let Some(previous) = ctx.paged_attn_out_tm.replace(next) {
                ctx.captured_storage.push(Box::new(previous));
            }
            ctx.paged_attn_out_tm_capacity = n_grown;
        }

        let func = ctx.func(
            "paged_varlen_attn_vllm",
            ptx::PAGED_VARLEN_ATTENTION_VLLM,
            "paged_varlen_attn_vllm_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        let shared_kv = ctx.configuration.cuda.shared_kv_for(final_kv_len);
        let shared_bytes = (shared_kv as u32) * 4;
        if shared_bytes > 48 * 1024 {
            let _ = func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            );
        }

        let out_tm_ptr: *mut crate::backend::CudaBuf =
            ctx.paged_attn_out_tm
                .as_mut()
                .expect("paged_attn_out_tm allocated") as *mut _;
        unsafe {
            let qv = q.as_f16().slice(..);
            let kp = k_pool.as_f16().slice(..);
            let vp = v_pool.as_f16().slice(..);
            let csq = cu_seqlens_q_buf.as_u32().slice(..);
            let po = pos_offsets_buf.as_u32().slice(..);
            let bt = block_tables.as_u32().slice(..);
            let ns = num_seqs as i32;
            let nqi = num_heads as i32;
            let nkvi = num_kv_heads as i32;
            let hdi = head_dim as i32;
            let mbps = max_num_blocks_per_seq as i32;
            let bsi = block_size as i32;
            let mut b = stream.launch_builder(&func);
            b.arg(&qv);
            b.arg(&kp);
            b.arg(&vp);
            b.arg(&csq);
            b.arg(&po);
            b.arg(&bt);
            b.arg(&mut *out_tm_ptr);
            b.arg(&ns);
            b.arg(&nqi);
            b.arg(&nkvi);
            b.arg(&hdi);
            b.arg(&mbps);
            b.arg(&bsi);
            b.arg(&scale);
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, q_len as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
            .map_err(|e| FerrumError::model(format!("paged_varlen_attn_vllm: {e}")))?;

            <Self as Backend>::transpose_token_to_head(
                ctx,
                &*out_tm_ptr,
                out,
                q_len,
                num_heads,
                head_dim,
            );
        }
        Ok(())
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_vllm(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        total_q_tokens: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 || total_q_tokens == 0 {
            return Ok(());
        }

        let func = ctx.func(
            "paged_varlen_attn_vllm",
            ptx::PAGED_VARLEN_ATTENTION_VLLM,
            "paged_varlen_attn_vllm_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        let shared_kv = ctx.configuration.cuda.shared_kv_for(max_kv_len);
        let shared_bytes = (shared_kv as u32) * 4;
        if shared_bytes > 48 * 1024 {
            let _ = func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            );
        }

        let qv = q.as_f16().slice(..);
        let kp = k_pool.as_f16().slice(..);
        let vp = v_pool.as_f16().slice(..);
        let csq = cu_seqlens_q.as_u32().slice(..);
        let po = pos_offsets.as_u32().slice(..);
        let bt = block_tables.as_u32().slice(..);
        let ns = num_seqs as i32;
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_num_blocks_per_seq as i32;
        let bsi = block_size as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&csq);
        b.arg(&po);
        b.arg(&bt);
        b.arg(out);
        b.arg(&ns);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mbps);
        b.arg(&bsi);
        b.arg(&scale);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, total_q_tokens as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("paged_varlen_attn_vllm: {e}")))
    }

    #[cfg(feature = "vllm-paged-attn-v2")]
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_vllm_tiled_q4(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        tile_seqs: &Self::Buffer,
        tile_starts: &Self::Buffer,
        num_tiles: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_tiles == 0 {
            return Ok(());
        }

        let func = ctx.func(
            "paged_varlen_attn_vllm",
            ptx::PAGED_VARLEN_ATTENTION_VLLM,
            "paged_varlen_attn_vllm_tiled_q4_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        let shared_kv = ctx.configuration.cuda.shared_kv_for(max_kv_len);
        let shared_bytes = (4 * shared_kv as u32) * 4;
        if shared_bytes > 48 * 1024 {
            let _ = func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            );
        }

        let qv = q.as_f16().slice(..);
        let kp = k_pool.as_f16().slice(..);
        let vp = v_pool.as_f16().slice(..);
        let csq = cu_seqlens_q.as_u32().slice(..);
        let po = pos_offsets.as_u32().slice(..);
        let bt = block_tables.as_u32().slice(..);
        let ts = tile_seqs.as_u32().slice(..);
        let tst = tile_starts.as_u32().slice(..);
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_num_blocks_per_seq as i32;
        let bsi = block_size as i32;
        let score_stride = shared_kv as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&csq);
        b.arg(&po);
        b.arg(&bt);
        b.arg(&ts);
        b.arg(&tst);
        b.arg(out);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mbps);
        b.arg(&bsi);
        b.arg(&score_stride);
        b.arg(&scale);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, num_tiles as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("paged_varlen_attn_vllm_tiled_q4: {e}")))
    }
}
