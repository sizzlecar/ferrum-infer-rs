//! `BackendPagedKv for CudaBackend` — paged-KV decode attention,
//! split-K dispatch, varlen prefill/decode, paged KV append.
//!
//! Extracted from `cuda/mod.rs` (#8 Phase 4). Owns:
//!
//! - `SplitKScratch` process-global f32 partial buffers for split-K
//!   phase 1 + `split_k_scratch_slot` / `with_split_k_scratch` helpers.
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

use super::{CudaBackend, CudaState, BATCHED_SCRATCH_CAP, HOST_STAGING_TOTAL};
use crate::backend::{Backend, BackendPagedKv};
use crate::ptx;

// ────────────────────────────────────────────────────────────────────────
// Paged-varlen split-K scratch + dispatch
// ────────────────────────────────────────────────────────────────────────

/// Process-global scratch for split-K phase1 outputs. Three buffers
/// (partial_out f32, partial_m f32, partial_l f32) sized to the largest
/// shape ever requested. Same lazy-grow pattern as Marlin gather scratch.
struct SplitKScratch {
    partial_out: CudaSlice<f32>, // [M_total * num_q_heads * num_splits * head_dim]
    partial_m: CudaSlice<f32>,   // [M_total * num_q_heads * num_splits]
    partial_l: CudaSlice<f32>,   // [M_total * num_q_heads * num_splits]
    out_capacity: usize,
    ml_capacity: usize,
}
unsafe impl Send for SplitKScratch {}
unsafe impl Sync for SplitKScratch {}

static SPLIT_K_SCRATCH: std::sync::OnceLock<std::sync::RwLock<Option<SplitKScratch>>> =
    std::sync::OnceLock::new();

fn split_k_scratch_slot() -> &'static std::sync::RwLock<Option<SplitKScratch>> {
    SPLIT_K_SCRATCH.get_or_init(|| std::sync::RwLock::new(None))
}

fn with_split_k_scratch<R>(
    stream: &Arc<CudaStream>,
    out_required: usize,
    ml_required: usize,
    body: impl FnOnce(&mut CudaSlice<f32>, &mut CudaSlice<f32>, &mut CudaSlice<f32>) -> R,
) -> R {
    let slot = split_k_scratch_slot();
    {
        let g = slot.read().expect("SPLIT_K_SCRATCH poisoned");
        if let Some(ref s) = *g {
            if s.out_capacity >= out_required && s.ml_capacity >= ml_required {
                drop(g);
                let mut w = slot.write().expect("SPLIT_K_SCRATCH poisoned");
                let s = w.as_mut().expect("just observed Some");
                return body(&mut s.partial_out, &mut s.partial_m, &mut s.partial_l);
            }
        }
    }
    let mut w = slot.write().expect("SPLIT_K_SCRATCH poisoned");
    let need_new = match &*w {
        Some(s) => s.out_capacity < out_required || s.ml_capacity < ml_required,
        None => true,
    };
    if need_new {
        let partial_out = unsafe { stream.alloc::<f32>(out_required) }
            .expect("SPLIT_K_SCRATCH partial_out alloc");
        let partial_m =
            unsafe { stream.alloc::<f32>(ml_required) }.expect("SPLIT_K_SCRATCH partial_m alloc");
        let partial_l =
            unsafe { stream.alloc::<f32>(ml_required) }.expect("SPLIT_K_SCRATCH partial_l alloc");
        *w = Some(SplitKScratch {
            partial_out,
            partial_m,
            partial_l,
            out_capacity: out_required,
            ml_capacity: ml_required,
        });
    }
    let s = w.as_mut().expect("just allocated");
    body(&mut s.partial_out, &mut s.partial_m, &mut s.partial_l)
}

#[allow(clippy::too_many_arguments)]
fn paged_varlen_split_k_dispatch(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    cu_seqlens_q: &CudaSlice<f16>,
    pos_offsets: &CudaSlice<f16>,
    block_tables: &CudaSlice<f16>,
    num_seqs: usize,
    total_q_tokens: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    // Pick num_splits based on kv_len. Microbench peak points:
    //   kv ≤ 384  → N=2 (or skip if c≥16)
    //   kv ≤ 1024 → N=4
    //   kv ≤ 2048 → N=8
    //   else      → N=16
    let num_splits: usize = match max_kv_len {
        kv if kv <= 384 => 2,
        kv if kv <= 1024 => 4,
        kv if kv <= 2048 => 8,
        _ => 16,
    };

    let chunk = (max_kv_len + num_splits - 1) / num_splits;
    let out_required = total_q_tokens * num_heads * num_splits * head_dim;
    let ml_required = total_q_tokens * num_heads * num_splits;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();

    let phase1 = ctx.func(
        "paged_varlen_split_k_phase1",
        ptx::PAGED_VARLEN_ATTENTION,
        "paged_varlen_attn_split_k_phase1_f16",
    );
    let reduce = ctx.func(
        "paged_varlen_split_k_reduce",
        ptx::PAGED_VARLEN_ATTENTION,
        "paged_varlen_split_k_reduce_f16",
    );

    with_split_k_scratch(
        &stream,
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
            b1.arg(&nsp);
            let shmem1 = (chunk.max(1) as u32) * 4;
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
    block_tables: &CudaSlice<f16>,
    valid_kv_lens: &CudaSlice<f16>,
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
    //   c=32 splits=2 → +5% (grid 8 waves, kv split still helps)
    // Override via FERRUM_PAGED_FLASH_SPLITS for tuning.
    let force_splits = std::env::var("FERRUM_PAGED_FLASH_SPLITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    // 128 SMs on Ada/Hopper-class; conservative under-estimate.
    const SM_TARGET: usize = 128;
    let base_grid = num_seqs * num_heads;
    let saturated = base_grid >= 2 * SM_TARGET;
    let num_splits: usize = force_splits.unwrap_or_else(|| {
        if saturated {
            // Only split when kv is so long that the V loop dominates.
            match max_kv_len {
                kv if kv <= 768 => 1,
                kv if kv <= 2048 => 2,
                _ => 4,
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

    with_split_k_scratch(
        &stream,
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
            let safe_kv: usize = std::env::var("FERRUM_KV_CAPACITY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(512);
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
    block_tables: &CudaSlice<f16>,
    valid_kv_lens: &CudaSlice<f16>,
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
    let safe_kv_max: usize = std::env::var("FERRUM_KV_CAPACITY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let shared_kv = safe_kv_max.max(max_kv_len).max(1);
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

impl BackendPagedKv for CudaBackend {
    /// Default ON for CUDA. Mixed-batch unified_forward path requires
    /// paged_pools; without it the engine's run_unified_iter falls back
    /// to serial prefill that stalls in-flight decoders (~50% of bench
    /// wall time at c=16). Override via FERRUM_METAL_PAGED_KV=0 if a
    /// caller specifically wants legacy contig KV.
    fn supports_paged_kv() -> bool {
        true
    }
    fn supports_varlen_qkv() -> bool {
        true
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

        let stream = ctx.stream.clone();
        // Lazy-alloc all three device scratch buffers to HOST_STAGING_TOTAL
        // u64 elements. Done outside any captured stream — sync allocs only.
        if ctx.batched_scratch_u64_cache.is_none() {
            ctx.batched_scratch_u64_cache = Some(
                stream
                    .alloc_zeros::<u64>(HOST_STAGING_TOTAL)
                    .map_err(|e| FerrumError::model(format!("alloc cache_ptrs: {e}")))?,
            );
        }
        if ctx.batched_scratch_u64_k.is_none() {
            ctx.batched_scratch_u64_k = Some(
                stream
                    .alloc_zeros::<u64>(HOST_STAGING_TOTAL)
                    .map_err(|e| FerrumError::model(format!("alloc k_ptrs: {e}")))?,
            );
        }
        if ctx.batched_scratch_u64_v.is_none() {
            ctx.batched_scratch_u64_v = Some(
                stream
                    .alloc_zeros::<u64>(HOST_STAGING_TOTAL)
                    .map_err(|e| FerrumError::model(format!("alloc v_ptrs: {e}")))?,
            );
        }
        // Fill host arrays at every slot we'll launch from. Layout:
        //   K-append (kv_cache_append): slot = li → host_cache_ptrs[li * CAP ..]
        //   V-append (kv_cache_append): slot = li + MAX_LAYERS_FOR_GRAPH
        //   flash_attn:                 slot = li → host_k_ptrs / host_v_ptrs
        for li in 0..num_layers {
            let k_off = li * BATCHED_SCRATCH_CAP;
            let v_off = (li + super::MAX_LAYERS_FOR_GRAPH) * BATCHED_SCRATCH_CAP;
            for i in 0..m {
                let (kp, _) = k_caches[li * m + i].device_ptr(&stream);
                let (vp, _) = v_caches[li * m + i].device_ptr(&stream);
                ctx.batched_host_cache_ptrs[k_off + i] = kp;
                ctx.batched_host_cache_ptrs[v_off + i] = vp;
                ctx.batched_host_k_ptrs[k_off + i] = kp;
                ctx.batched_host_v_ptrs[k_off + i] = vp;
            }
        }
        // Bind context for sync memcpys (tokio thread-shift safe).
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("populate bind_to_thread: {e}")))?;

        // Sync memcpy each entire host array to its device scratch in one shot.
        // cuMemcpyHtoD_v2 is on the legacy default (null) stream → NOT
        // captured by stream capture, so the captured graph contains
        // only kernel launches. Device scratch is fresh before every
        // call, including pure-replay (which doesn't re-enter Rust).
        let total_bytes = HOST_STAGING_TOTAL * std::mem::size_of::<u64>();
        unsafe {
            use cudarc::driver::{sys, DevicePtrMut};
            let scratch_cache = ctx.batched_scratch_u64_cache.as_mut().unwrap();
            let (dst, _g) = scratch_cache.device_ptr_mut(&stream);
            let st = sys::cuMemcpyHtoD_v2(
                dst,
                ctx.batched_host_cache_ptrs.as_ptr() as *const std::ffi::c_void,
                total_bytes,
            );
            if st != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::unsupported(format!(
                    "populate cache_ptrs sync memcpy: {st:?}"
                )));
            }
            let scratch_k = ctx.batched_scratch_u64_k.as_mut().unwrap();
            let (dst, _g) = scratch_k.device_ptr_mut(&stream);
            let st = sys::cuMemcpyHtoD_v2(
                dst,
                ctx.batched_host_k_ptrs.as_ptr() as *const std::ffi::c_void,
                total_bytes,
            );
            if st != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::unsupported(format!(
                    "populate k_ptrs sync memcpy: {st:?}"
                )));
            }
            let scratch_v = ctx.batched_scratch_u64_v.as_mut().unwrap();
            let (dst, _g) = scratch_v.device_ptr_mut(&stream);
            let st = sys::cuMemcpyHtoD_v2(
                dst,
                ctx.batched_host_v_ptrs.as_ptr() as *const std::ffi::c_void,
                total_bytes,
            );
            if st != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::unsupported(format!(
                    "populate v_ptrs sync memcpy: {st:?}"
                )));
            }
        }
        Ok(())
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
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 || total_q_tokens == 0 {
            return Ok(());
        }

        // Auto-tune: split-K helps under-occupied grids — low concurrency
        // OR long context. Microbench (scripts/microbench_split_k.cu)
        // shows c=1/kv=4096 9× speedup, c=4/kv=384 +103%, c=16/kv=384
        // marginal/-2%. Heuristic gates split-K to regions where it wins.
        // FERRUM_SPLIT_K_ATTN=1 forces on, FERRUM_SPLIT_K_ATTN=0 forces off.
        let split_k_force = std::env::var("FERRUM_SPLIT_K_ATTN").ok();
        let use_split_k = match split_k_force.as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => num_seqs <= 4 || max_kv_len >= 768,
        };

        if use_split_k {
            return paged_varlen_split_k_dispatch(
                ctx,
                q.as_f16(),
                k_pool.as_f16(),
                v_pool.as_f16(),
                out.as_f16_mut(),
                cu_seqlens_q.as_f16(),
                pos_offsets.as_f16(),
                block_tables.as_f16(),
                num_seqs,
                total_q_tokens,
                max_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
            );
        }

        let func = ctx.func(
            "paged_varlen_attn",
            ptx::PAGED_VARLEN_ATTENTION,
            "paged_varlen_attn_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        // CudaBackend::Buffer is monomorphic CudaSlice<f16>; i32 data
        // (cu_seqlens_q / pos_offsets / block_tables) is stored in
        // f16-typed buffers via `from_slice_i32` + matching alloc, the
        // kernel reads them as `int*`. Same pattern as kv_lens in
        // `flash_attention_batched_per_cache`.
        let qv = q.slice(..);
        let kp = k_pool.slice(..);
        let vp = v_pool.slice(..);
        let csq = cu_seqlens_q.slice(..);
        let po = pos_offsets.slice(..);
        let bt = block_tables.slice(..);
        let ns = num_seqs as i32;
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_blocks_per_seq as i32;
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
        // CUDA graph capture freezes `shared_mem_bytes` at capture time;
        // graph keys at the engine level are (m_total, num_seqs) — they
        // do NOT distinguish kv_len buckets. So a graph captured at
        // kv_len=300 (shared=300*4) replays unchanged at kv_len=600 →
        // kernel writes scores[300..600] OOB into shared.
        // compute-sanitizer caught it:
        //   "Invalid __shared__ write of size 4 bytes at paged_varlen_attn_f16
        //    Address 0x84c is out of bounds (in captured graph replay)".
        //
        // Allocate the worst-case kv slot length for ANY future decode
        // iter that may replay this graph. FERRUM_KV_CAPACITY caps it
        // (default 512, bench sets 2048). 8 KB shared = 2048 floats
        // — well within Ada's 96 KB/SM and Hopper's 228 KB/SM budgets.
        // For models with longer effective contexts the cap raises with
        // capacity at the cost of one alloc, never per-launch.
        let safe_kv_max: usize = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let shared_kv = safe_kv_max.max(max_kv_len).max(1);
        let shared_bytes = (shared_kv as u32) * 4;
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
    ) -> Result<()> {
        let max_kv_len = block_size * max_num_blocks_per_seq;

        if q_len == 1 {
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

        // q_len > 1: prefill. Only single-seq is exercised — the only
        // caller (Qwen3MoeModel::forward_layer) always passes num_seqs=1.
        if num_seqs != 1 {
            return Err(FerrumError::model(format!(
                "paged_decode_attention(CUDA): q_len={q_len} num_seqs={num_seqs} \
                 not supported (caller must split prefill into per-seq calls)"
            )));
        }

        // Build cu_seqlens_q = [0, q_len] and pos_offsets = [final_kv_len - q_len].
        // alloc_u32 + write_u32 (NOT from_slice_i32 — that default goes
        // through f32→f16 and zeroes the bit pattern).
        // Need final_kv_len from context_lens[0] — D2H 4 bytes (cold path).
        // Post-Phase-B-2 context_lens is `CudaBuf::U32` (typed), so we read
        // the U32 variant directly — the prior `transmute::<u32>` route was
        // for the f16-bit-tunneled allocation that B-2 retired.
        let cl_host: Vec<u32> = {
            let stream = ctx.stream.clone();
            let view = context_lens.as_u32().slice(0..1);
            let mut h = vec![0u32; 1];
            stream
                .memcpy_dtoh(&view, h.as_mut_slice())
                .map_err(|e| FerrumError::model(format!("dtoh context_lens: {e}")))?;
            stream
                .synchronize()
                .map_err(|e| FerrumError::model(format!("dtoh sync: {e}")))?;
            h
        };
        let final_kv_len = cl_host[0] as usize;
        if final_kv_len < q_len {
            return Err(FerrumError::model(format!(
                "paged_decode_attention(CUDA): final_kv_len={final_kv_len} < q_len={q_len}"
            )));
        }
        let pos_offset = (final_kv_len - q_len) as u32;
        let mut cu_seqlens_q_buf = <Self as Backend>::alloc_u32(2);
        <Self as Backend>::write_u32(ctx, &mut cu_seqlens_q_buf, &[0u32, q_len as u32]);
        let mut pos_offsets_buf = <Self as Backend>::alloc_u32(1);
        <Self as Backend>::write_u32(ctx, &mut pos_offsets_buf, &[pos_offset]);

        // The caller's q buffer (despite being named `q_head_major` in
        // Qwen3MoeModel) is ALREADY token-major in paged mode: the
        // paged-write kernel split_qkv_norm_rope_into_paged_cache_f16
        // writes `q_out[tok, head, hd]` (see kernel comment at
        // kernels/split_qkv_norm_rope_into_paged_cache.cu:102). So
        // paged_varlen_attention can read q directly. No Q transpose.
        //
        // Output, however, is written by paged_varlen as
        // `[M_total, num_q_heads, head_dim]` token-major (kernel:16),
        // while Qwen3MoeModel's downstream code does
        // `transpose_head_to_token(attn_head_major_out → attn_flat)`,
        // expecting head-major. We transpose token→head into `out`.
        let q_n = q_len * num_heads * head_dim;

        // Lazy-grow the cached token-major output scratch. Stable
        // address across calls — required to avoid stream-ordered
        // free / kernel-still-running races at higher concurrency.
        if ctx.paged_attn_out_tm_capacity < q_n {
            let stream = ctx.stream.clone();
            let n_grown = q_n.next_power_of_two().max(q_n);
            ctx.paged_attn_out_tm = Some(crate::backend::CudaBuf::from_f16(
                stream
                    .alloc_zeros::<f16>(n_grown)
                    .map_err(|e| FerrumError::model(format!("alloc paged_attn_out_tm: {e}")))?,
            ));
            ctx.paged_attn_out_tm_capacity = n_grown;
        }

        // SAFETY: paged_varlen_attention only touches ctx.modules and
        // ctx.stream (disjoint from paged_attn_out_tm). Same for
        // transpose_token_to_head. We take a raw pointer to the cached
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
                &cu_seqlens_q_buf,
                &pos_offsets_buf,
                block_tables,
                1,
                q_len,
                final_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
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
        if std::env::var("FERRUM_PAGED_FLASH").map_or(true, |v| v != "0") {
            return paged_batched_flash_dispatch(
                ctx,
                q.as_f16(),
                k_pool.as_f16(),
                v_pool.as_f16(),
                out.as_f16_mut(),
                block_tables.as_f16(),
                valid_kv_lens.as_f16(),
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
        let safe_kv_max: usize = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let shared_kv = safe_kv_max.max(max_kv_len).max(1);
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
}
