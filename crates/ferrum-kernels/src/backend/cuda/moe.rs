//! `BackendMoeFused for CudaBackend` — fused MoE routing kernels
//! (`route_topk_softmax`, `moe_align_block_size`, `route_topk`,
//! `moe_combine`).
//!
//! Extracted from `cuda/mod.rs` (#8 Phase 5). The trait body is
//! the only public surface — no private helpers leak from here.
//!
//! All kernels here run on `crate::ptx::MOE_*` PTX modules:
//! `MOE_ROUTER`, `MOE_ALIGN_BLOCK_SIZE`, `MOE_COMBINE`. The fused
//! Marlin GEMM phase (`moe_gemm_phase_*` impl methods) lives on
//! `BackendQuantMarlin` and is in `cuda/quant.rs`.

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use ferrum_bench_core::{global_profile, profile_fields_from_json};
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::{moe_shared_aux, CudaBackend, CudaState, HOST_STAGING_TOTAL};
use crate::backend::{Backend, BackendMoeFused, CudaBuf};
use crate::ptx;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoeDumpRuntimeConfig {
    enabled: bool,
    batch_x_topk_filter: Option<usize>,
}

fn moe_dump_runtime_config() -> &'static MoeDumpRuntimeConfig {
    static CONFIG: OnceLock<MoeDumpRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mut config = MoeDumpRuntimeConfig {
            enabled: false,
            batch_x_topk_filter: None,
        };
        for (name, value) in std::env::vars() {
            match name.as_str() {
                "FERRUM_MOE_DUMP" => config.enabled = true,
                "FERRUM_MOE_DUMP_BATCH_X_TOPK" => {
                    config.batch_x_topk_filter = value.parse::<usize>().ok();
                }
                _ => {}
            }
        }
        config
    })
}

fn maybe_dump_moe_routing(
    kind: &str,
    stream: &Arc<CudaStream>,
    sorted_token_ids: &CudaBuf,
    block_ids: &CudaBuf,
    total_tokens_post_pad: &CudaBuf,
    batch_x_topk: usize,
    num_experts: usize,
    block_size: usize,
) {
    let config = moe_dump_runtime_config();
    if !config.enabled {
        return;
    }
    if config
        .batch_x_topk_filter
        .is_some_and(|target| target != batch_x_topk)
    {
        return;
    }

    use std::sync::atomic::{AtomicBool, Ordering};
    static DUMPED: AtomicBool = AtomicBool::new(false);
    if DUMPED.swap(true, Ordering::Relaxed) {
        return;
    }

    let read_i32 = |buf: &CudaBuf, len: usize| -> Vec<i32> {
        let n = len.min(buf.len());
        if n == 0 {
            return Vec::new();
        }
        let view = buf.as_i32().slice(0..n);
        let mut host = vec![0i32; n];
        if stream.memcpy_dtoh(&view, host.as_mut_slice()).is_err() {
            return Vec::new();
        }
        if stream.synchronize().is_err() {
            return Vec::new();
        }
        host
    };

    let st = read_i32(sorted_token_ids, sorted_token_ids.len());
    let bi = read_i32(block_ids, block_ids.len());
    let tp = read_i32(total_tokens_post_pad, 1);
    let total_post_pad = tp.first().copied().unwrap_or(-1);
    let total_blocks = if total_post_pad > 0 {
        ((total_post_pad as usize) / block_size).min(bi.len())
    } else {
        0
    };
    let mut seen = vec![false; num_experts];
    let mut unique_experts = 0usize;
    for &expert_id in bi.iter().take(total_blocks) {
        if expert_id >= 0 {
            let expert_idx = expert_id as usize;
            if expert_idx < seen.len() && !seen[expert_idx] {
                seen[expert_idx] = true;
                unique_experts += 1;
            }
        }
    }
    let n_show = 48.min(st.len());
    let n_bi = 32.min(bi.len());
    eprintln!(
        "[MOE_DUMP:{kind}] batch_x_topk={batch_x_topk} block_size={block_size} \
         num_experts={num_experts} total_post_pad={total_post_pad} \
         active_blocks={total_blocks} unique_experts={unique_experts}",
    );
    eprintln!(
        "[MOE_DUMP:{kind}] sorted_token_ids[0..{n_show}] = {:?}",
        &st[..n_show]
    );
    eprintln!("[MOE_DUMP:{kind}] block_ids[0..{n_bi}] = {:?}", &bi[..n_bi]);
    let profile = global_profile();
    if profile.is_enabled() {
        let _ = profile.push_event(
            "moe_dump",
            profile_fields_from_json(serde_json::json!({
                "kind": kind,
                "batch_x_topk": batch_x_topk,
                "block_size": block_size,
                "num_experts": num_experts,
                "total_post_pad": total_post_pad,
                "active_blocks": total_blocks,
                "unique_experts": unique_experts,
                "sorted_token_ids_preview": &st[..n_show],
                "block_ids_preview": &bi[..n_bi],
            })),
            profile_fields_from_json(serde_json::json!({})),
            false,
        );
    }
}

impl BackendMoeFused for CudaBackend {
    fn try_launch_moe_aux_stream<F>(
        ctx: &mut Self::Context,
        _tokens: usize,
        body: F,
    ) -> Result<bool>
    where
        F: FnOnce(&mut Self::Context) -> Result<()>,
    {
        if ctx.capture_in_flight {
            return Ok(false);
        }

        let (aux_stream, aux_blas, entry_event, exit_event) = moe_shared_aux(ctx);
        let default_stream = ctx.stream.clone();
        unsafe {
            cudarc::driver::sys::cuEventRecord(entry_event, default_stream.cu_stream());
            cudarc::driver::sys::cuStreamWaitEvent(aux_stream.cu_stream(), entry_event, 0);
        }

        let mut aux_ctx = CudaState {
            ordinal: ctx.ordinal,
            ctx: ctx.ctx.clone(),
            stream: aux_stream.clone(),
            blas: aux_blas,
            modules: HashMap::new(),
            use_dev_state: ctx.use_dev_state,
            capture_in_flight: false,
            batched_scratch_u64_k: None,
            batched_scratch_u64_v: None,
            batched_scratch_u64_cache: None,
            batched_scratch_i32_kv_lens: None,
            batched_scratch_i32_cache_lens: None,
            batched_host_k_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            batched_host_v_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            batched_host_cache_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            batched_host_kv_lens: Box::new([0i32; HOST_STAGING_TOTAL]),
            batched_host_cache_lens: Box::new([0i32; HOST_STAGING_TOTAL]),
            moe_streams: None,
            moe_entry_event: None,
            moe_exit_events: None,
            moe_route_ids: None,
            moe_route_weights: None,
            moe_route_capacity: 0,
            paged_attn_out_tm: None,
            paged_attn_out_tm_capacity: 0,
        };

        let result = body(&mut aux_ctx);
        unsafe {
            cudarc::driver::sys::cuEventRecord(exit_event, aux_stream.cu_stream());
        }
        if result.is_err() {
            unsafe {
                cudarc::driver::sys::cuStreamWaitEvent(default_stream.cu_stream(), exit_event, 0);
            }
        }
        result?;
        Ok(true)
    }

    fn wait_moe_aux_stream(ctx: &mut Self::Context) -> Result<()> {
        if ctx.capture_in_flight {
            return Ok(());
        }
        let (_, _, _, exit_event) = moe_shared_aux(ctx);
        let default_stream = ctx.stream.clone();
        unsafe {
            cudarc::driver::sys::cuStreamWaitEvent(default_stream.cu_stream(), exit_event, 0);
        }
        Ok(())
    }

    fn route_topk_softmax(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        out_ids: &mut Self::Buffer,
        out_weights: &mut Self::Buffer,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<()> {
        // Block: one warp (32 threads), one block per row.
        // Shared mem: num_experts × 4 bytes (per-row probability vector
        // in fp32). At Qwen3-MoE num_experts=128 this is 512 bytes /
        // block — far below the 48 KB / SM limit. Larger MoE configs
        // (DeepSeek 256 experts) still only use 1 KB.
        let func = ctx.func(
            "moe_router_topk_softmax",
            ptx::MOE_ROUTER,
            "moe_router_topk_softmax_f16",
        );
        let batch_i32 = batch as i32;
        let n_exp_i32 = num_experts as i32;
        let top_k_i32 = top_k as i32;
        let norm_i32 = if norm_topk_prob { 1i32 } else { 0i32 };
        let smem_bytes = (num_experts as u32) * 4;

        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(logits);
        b.arg(out_ids);
        b.arg(out_weights);
        b.arg(&batch_i32);
        b.arg(&n_exp_i32);
        b.arg(&top_k_i32);
        b.arg(&norm_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: smem_bytes,
            })
        }
        .map_err(|e| FerrumError::model(format!("moe_router launch: {e}")))?;
        Ok(())
    }
    fn try_gpu_route_topk_into_host(
        ctx: &mut Self::Context,
        logits_dev: &Self::Buffer,
        out_ids_host: &mut Vec<u32>,
        out_weights_host: &mut Vec<f32>,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<()> {
        let total_pairs = batch * top_k;

        // Lazy-init the scratch device buffers. Sized to total_pairs;
        // grow if a larger shape shows up. i32 storage = 4*total_pairs
        // bytes = 2*total_pairs f16 elements; f32 storage same (4 bytes
        // per element).
        if ctx.moe_route_capacity < total_pairs {
            let stream = ctx.stream.clone();
            // 2 × total_pairs because each i32 / f32 element needs 4
            // bytes = 2 f16 slots in the underlying CudaSlice<f16>.
            let nf16 = 2 * total_pairs;
            ctx.moe_route_ids = Some(
                stream
                    .alloc_zeros::<f16>(nf16)
                    .map_err(|e| FerrumError::model(format!("alloc moe_route_ids: {e}")))?,
            );
            ctx.moe_route_weights = Some(
                stream
                    .alloc_zeros::<f16>(nf16)
                    .map_err(|e| FerrumError::model(format!("alloc moe_route_weights: {e}")))?,
            );
            ctx.moe_route_capacity = total_pairs;
        }

        // 1. Launch the kernel into the cached scratch. Scoped so the
        // launch_builder (which moves the &mut buffer references) drops
        // before we re-borrow them immutably for the D2H phase.
        let func = ctx.func(
            "moe_router_topk_softmax",
            ptx::MOE_ROUTER,
            "moe_router_topk_softmax_f16",
        );
        let batch_i32 = batch as i32;
        let n_exp_i32 = num_experts as i32;
        let top_k_i32 = top_k as i32;
        let norm_i32 = if norm_topk_prob { 1i32 } else { 0i32 };
        let smem_bytes = (num_experts as u32) * 4;

        let stream = ctx.stream.clone();
        {
            let ids_dev = ctx
                .moe_route_ids
                .as_mut()
                .expect("moe_route_ids should be allocated");
            let weights_dev = ctx
                .moe_route_weights
                .as_mut()
                .expect("moe_route_weights should be allocated");

            let mut b = stream.launch_builder(&func);
            b.arg(logits_dev);
            b.arg(ids_dev);
            b.arg(weights_dev);
            b.arg(&batch_i32);
            b.arg(&n_exp_i32);
            b.arg(&top_k_i32);
            b.arg(&norm_i32);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (batch as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: smem_bytes,
                })
            }
            .map_err(|e| FerrumError::model(format!("moe_router launch: {e}")))?;
        }

        // 2. D2H ids (i32) and weights (f32) into the host destinations.
        out_ids_host.clear();
        out_ids_host.resize(total_pairs, 0u32);
        out_weights_host.clear();
        out_weights_host.resize(total_pairs, 0.0f32);

        let ids_dev = ctx
            .moe_route_ids
            .as_ref()
            .expect("moe_route_ids should be allocated");
        let weights_dev = ctx
            .moe_route_weights
            .as_ref()
            .expect("moe_route_weights should be allocated");

        // Reinterpret the f16-typed scratch as i32 / f32 views. transmute
        // verifies byte-fit (returns None if undersized).
        let ids_view = unsafe {
            ids_dev
                .transmute::<i32>(total_pairs)
                .ok_or_else(|| FerrumError::model("ids transmute size mismatch"))?
        };
        let weights_view = unsafe {
            weights_dev
                .transmute::<f32>(total_pairs)
                .ok_or_else(|| FerrumError::model("weights transmute size mismatch"))?
        };

        // out_ids_host is Vec<u32>; reinterpret as &mut [i32] for the
        // memcpy. Same byte pattern.
        let out_ids_i32: &mut [i32] = unsafe {
            std::slice::from_raw_parts_mut(out_ids_host.as_mut_ptr() as *mut i32, total_pairs)
        };
        stream
            .memcpy_dtoh(&ids_view, out_ids_i32)
            .map_err(|e| FerrumError::model(format!("dtoh route ids: {e}")))?;
        stream
            .memcpy_dtoh(&weights_view, out_weights_host.as_mut_slice())
            .map_err(|e| FerrumError::model(format!("dtoh route weights: {e}")))?;
        // Synchronize so the host can read the results immediately.
        stream
            .synchronize()
            .map_err(|e| FerrumError::model(format!("dtoh sync: {e}")))?;

        Ok(())
    }
    fn moe_build_pairs_by_token(
        ctx: &mut Self::Context,
        expert_ids: &Self::Buffer,
        pairs_by_token: &mut Self::Buffer,
        packed_token_idx: &mut Self::Buffer,
        expert_offsets: &mut Self::Buffer,
        batch_x_topk: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<()> {
        if num_experts > 256 {
            return Err(FerrumError::model(format!(
                "moe_build_pairs_by_token: num_experts={num_experts} > MAX 256 (shmem limit)"
            )));
        }
        let func = ctx.func(
            "moe_build_pairs_by_token",
            ptx::MOE_BUILD_PAIRS,
            "moe_build_pairs_by_token",
        );
        let n = batch_x_topk as i32;
        let ne = num_experts as i32;
        let tk = top_k as i32;
        let smem = (num_experts as u32) * 4; // i32 counts per expert
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(expert_ids);
        b.arg(pairs_by_token);
        b.arg(packed_token_idx);
        b.arg(expert_offsets);
        b.arg(&n);
        b.arg(&ne);
        b.arg(&tk);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: smem,
            })
        }
        .map_err(|e| FerrumError::model(format!("moe_build_pairs_by_token launch: {e}")))?;
        Ok(())
    }

    fn moe_align_block_size(
        ctx: &mut Self::Context,
        expert_ids_per_pair: &Self::Buffer,
        sorted_token_ids: &mut Self::Buffer,
        block_ids: &mut Self::Buffer,
        total_tokens_post_pad: &mut Self::Buffer,
        batch_x_topk: usize,
        num_experts: usize,
        block_size: usize,
        sorted_max_size: usize,
    ) -> Result<()> {
        if num_experts > 256 {
            return Err(FerrumError::model(format!(
                "moe_align_block_size: num_experts={num_experts} exceeds compile-time MAX_NUM_EXPERTS=256"
            )));
        }
        let func = ctx.func(
            "moe_align_block_size",
            ptx::MOE_ALIGN_BLOCK_SIZE,
            "moe_align_block_size_f32",
        );
        let n = batch_x_topk as i32;
        let ne = num_experts as i32;
        let bs = block_size as i32;
        let smax = sorted_max_size as i32;
        let stream = ctx.stream.clone();
        // Single block — algorithm uses shared mem for counts + offsets,
        // sized to MAX_NUM_EXPERTS=256. Use 256 threads to cover the
        // ≤256-experts and ≤1024-pair Qwen3-MoE configs cleanly.
        {
            let mut b = stream.launch_builder(&func);
            b.arg(&*expert_ids_per_pair);
            b.arg(&mut *sorted_token_ids);
            b.arg(&mut *block_ids);
            b.arg(&mut *total_tokens_post_pad);
            b.arg(&n);
            b.arg(&ne);
            b.arg(&bs);
            b.arg(&smax);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("moe_align_block_size launch: {e}")))?;
        }

        maybe_dump_moe_routing(
            "packed",
            &stream,
            sorted_token_ids,
            block_ids,
            total_tokens_post_pad,
            batch_x_topk,
            num_experts,
            block_size,
        );
        Ok(())
    }

    fn moe_align_block_size_pair_ids(
        ctx: &mut Self::Context,
        expert_ids_per_pair: &Self::Buffer,
        sorted_token_ids: &mut Self::Buffer,
        block_ids: &mut Self::Buffer,
        total_tokens_post_pad: &mut Self::Buffer,
        batch_x_topk: usize,
        num_experts: usize,
        block_size: usize,
        sorted_max_size: usize,
    ) -> Result<()> {
        if num_experts > 256 {
            return Err(FerrumError::model(format!(
                "moe_align_block_size_pair_ids: num_experts={num_experts} exceeds compile-time MAX_NUM_EXPERTS=256"
            )));
        }
        let func = ctx.func(
            "moe_align_block_size_pair_ids",
            ptx::MOE_ALIGN_BLOCK_SIZE_PAIR_IDS,
            "moe_align_block_size_pair_ids_f32",
        );
        let n = batch_x_topk as i32;
        let ne = num_experts as i32;
        let bs = block_size as i32;
        let smax = sorted_max_size as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(&*expert_ids_per_pair);
        b.arg(&mut *sorted_token_ids);
        b.arg(&mut *block_ids);
        b.arg(&mut *total_tokens_post_pad);
        b.arg(&n);
        b.arg(&ne);
        b.arg(&bs);
        b.arg(&smax);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("moe_align_block_size_pair_ids launch: {e}")))?;
        maybe_dump_moe_routing(
            "pair_ids",
            &stream,
            sorted_token_ids,
            block_ids,
            total_tokens_post_pad,
            batch_x_topk,
            num_experts,
            block_size,
        );
        Ok(())
    }

    fn moe_combine(
        ctx: &mut Self::Context,
        packed_down: &Self::Buffer,
        pairs_by_token: &Self::Buffer,
        pair_weights: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        hidden: usize,
        top_k: usize,
        _total_pairs: usize,
    ) {
        // Phase D follow-up: device-buffer routing — no clone_htod here.
        // Callers (moe_forward_bucketed) upload pairs/weights to device
        // once per call (or, eventually, build them entirely device-side
        // via B::moe_build_pairs_by_token + route_topk_softmax — that
        // path unlocks CUDA Graph capture).
        let func = ctx.func("moe_combine", ptx::MOE_COMBINE, "moe_combine_f16");
        let batch_i32 = batch as i32;
        let hidden_i32 = hidden as i32;
        let top_k_i32 = top_k as i32;

        let block = 256u32;
        let grid_x = ((hidden as u32) + block - 1) / block;

        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(packed_down);
        b.arg(pairs_by_token);
        b.arg(pair_weights);
        b.arg(out);
        b.arg(&batch_i32);
        b.arg(&hidden_i32);
        b.arg(&top_k_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid_x, batch as u32, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("moe_combine launch");
    }

    fn weighted_sum_batched(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        let func = ctx.func(
            "weighted_sum_batched",
            ptx::MOE_COMBINE,
            "weighted_sum_batched_f16",
        );
        let batch_i32 = batch as i32;
        let top_k_i32 = top_k as i32;
        let hidden_i32 = hidden as i32;

        let block = 256u32;
        let grid_x = ((hidden as u32) + block - 1) / block;

        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(slots);
        b.arg(weights);
        b.arg(out);
        b.arg(&batch_i32);
        b.arg(&top_k_i32);
        b.arg(&hidden_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid_x, batch as u32, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("weighted_sum_batched launch: {e}")))?;
        Ok(())
    }

    #[cfg(feature = "vllm-moe-marlin")]
    fn upload_moe_routing(
        ctx: &mut Self::Context,
        sorted_token_ids: &[i32],
        expert_ids: &[i32],
        num_tokens_past_padded: &[i32],
    ) -> Result<crate::backend::traits::MoeRouting<Self>> {
        use cudarc::driver::CudaSlice;

        // Phase D step 2+3: B::Buffer is now CudaBuf (typed enum),
        // so we can store the i32 routing buffers directly in
        // CudaBuf::I32 instead of leaking through f16 upgrade_device_ptr.
        // No more mem::forget leak — the I32 slices own their memory.
        let stream = ctx.stream.clone();
        let st: CudaSlice<i32> = stream
            .clone_htod(sorted_token_ids)
            .map_err(|e| FerrumError::model(format!("htod sorted_token_ids: {e}")))?;
        let eid: CudaSlice<i32> = stream
            .clone_htod(expert_ids)
            .map_err(|e| FerrumError::model(format!("htod expert_ids: {e}")))?;
        let npp: CudaSlice<i32> = stream
            .clone_htod(num_tokens_past_padded)
            .map_err(|e| FerrumError::model(format!("htod num_tokens_past_padded: {e}")))?;

        Ok(crate::backend::traits::MoeRouting {
            sorted_token_ids: crate::backend::CudaBuf::from_i32(st),
            expert_ids: crate::backend::CudaBuf::from_i32(eid),
            num_tokens_past_padded: crate::backend::CudaBuf::from_i32(npp),
        })
    }
}
