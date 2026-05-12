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

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ferrum_types::{FerrumError, Result};
use half::f16;

use super::CudaBackend;
use crate::backend::{Backend, BackendMoeFused};
use crate::ptx;

impl BackendMoeFused for CudaBackend {
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
        expert_offsets: &mut Self::Buffer,
        batch_x_topk: usize,
        num_experts: usize,
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
        let smem = (num_experts as u32) * 4; // i32 counts per expert
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(expert_ids);
        b.arg(pairs_by_token);
        b.arg(expert_offsets);
        b.arg(&n);
        b.arg(&ne);
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
        let mut b = stream.launch_builder(&func);
        b.arg(expert_ids_per_pair);
        b.arg(sorted_token_ids);
        b.arg(block_ids);
        b.arg(total_tokens_post_pad);
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
