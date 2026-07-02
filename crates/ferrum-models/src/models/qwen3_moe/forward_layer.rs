use super::*;

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    /// Run one full transformer layer (attention + MoE FFN).
    pub(crate) fn forward_layer(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        pos_offset: usize,
        tokens: usize,
        // If `Some(idx)` and we land on the decode fast path, fold the
        // next layer's leading rms_norm into this layer's MoE tail
        // (cross-layer norm fusion). The next layer's caller must pass
        // `prev_did_norm_fusion = true` so it skips its own rms_norm.
        next_layer_idx: Option<usize>,
        // If `true`, skip step 1's input rms_norm — the previous
        // layer's tail already populated `scratch.norm_out`.
        prev_did_norm_fusion: bool,
    ) -> Result<bool> {
        let cfg_base = &self.cfg.base;
        let h = cfg_base.hidden_size;
        let nh = cfg_base.num_heads;
        let nkv = cfg_base.num_kv_heads;
        let hd = cfg_base.head_dim;
        let eps = cfg_base.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let attn_layer = &self.attn_layers[li];
        let moe_layer = &self.moe_layers[li];

        // PLAYBOOK § 1.2 — was: `B::sync(ctx); Some(Instant::now())`.
        // Migrated onto BackendTimer (CUDA: cuEvents, async, no sync paid).
        let attn_t0 = ferrum_kernels::backend::timer::start_probe_timer_if::<B>(
            self.runtime_env.decode_op_profile,
            ctx,
        );

        // 1. Input RMSNorm — skipped when the previous layer's MoE tail
        //    fused this norm via `weighted_sum_residual_norm_stacked`.
        if !prev_did_norm_fusion {
            B::rms_norm(
                ctx,
                residual,
                &attn_layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                tokens,
                h,
            );
        }

        // 2. Fused QKV
        attn_layer.qkv_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.qkv_out,
            tokens,
        );

        // 3-4. Fused split-QKV + QK-norm + RoPE + head-major transpose.
        //
        // One Metal dispatch replaces (split_qkv → 3× qk_norm_rope), the
        // four-launch chain that used to dominate the attention prelude.
        // Reads qkv_out once, writes head-major Q/K (norm+RoPE) and V
        // (transpose only) directly into attention scratch. Saves 3
        // dispatches per layer (×48 = 144 dispatches per decode token).
        //
        // CPU and other backends without the fused kernel return
        // Unsupported and we fall through to the original four-launch
        // path. q_buf / k_buf / v_buf stay in scratch because that path
        // and the per-expert MoE fallback still want them.
        let qk_mode: i32 = if cfg_base.has_qk_norm { 1 } else { 2 };
        let dummy = &attn_layer.input_ln_w;
        let q_norm_w = attn_layer.q_norm_w.as_ref().unwrap_or(dummy);
        let k_norm_w = attn_layer.k_norm_w.as_ref().unwrap_or(dummy);

        // 5. Grab the per-layer KV cache up front — the deepest fused
        //    variant writes K/V straight into it, avoiding a trailing
        //    `kv_cache_append_head_major` dispatch.
        //
        // Paged mode: extract a raw pointer to the layer's pool buffers
        // BEFORE the &mut cache borrow, so we can pass &mut to the
        // paged kernel below without holding two simultaneous mutable
        // borrows on `self`. Safety: `paged_pools` is allocated once at
        // first ensure_kv call and never resized; the only concurrent
        // mutation is the pool's own kernel writes (sequenced via
        // command buffers), so the raw pointer remains valid for the
        // duration of this layer call.
        let paged_pool_ptr: Option<(*mut B::Buffer, *mut B::Buffer)> =
            if let Some(pools) = self.paged_pools.as_mut() {
                let pool = &mut pools[li];
                Some((&mut pool.0 as *mut _, &mut pool.1 as *mut _))
            } else {
                None
            };
        let caches = self
            .kv_caches
            .get_mut(cache_id)
            .expect("ensure_kv must be called before forward_layer");
        let cache = &mut caches[li];
        let cache_len_before = cache.len;
        let cache_capacity = cache.capacity;

        // Defense in depth: refuse to write past the KV buffer. Silent
        // overflow has visible failure modes (garbage output, stale token
        // attention, slowdowns from reading uninitialised memory). The
        // graceful path is the caller pre-checking via `kv_capacity()` and
        // either compacting or refusing the request; this panic only
        // fires when that contract is broken.
        if cache_len_before + tokens > cache_capacity {
            panic!(
                "KV cache overflow on layer {li}: would write tokens [{cache_len_before}..{}) but capacity is {cache_capacity} (cache_id={cache_id:?}). Increase FERRUM_KV_CAPACITY or call /clear in the REPL.",
                cache_len_before + tokens
            );
        }

        // Try the deepest fusion: fused split-QKV-norm-rope that writes
        // K/V directly into the cache slot. Paged mode writes into the
        // shared pool via block_table indirection; contiguous mode
        // writes into the per-cache_id k/v buffers directly.
        let used_qkv_into_cache = if cache.block_size > 0 {
            // Paged path.
            let bt = cache
                .block_table
                .as_ref()
                .expect("paged cache missing block_table");
            let num_blocks_per_seq = cache.capacity / cache.block_size;
            let (pool_k_ptr, pool_v_ptr) =
                paged_pool_ptr.expect("paged_pools must be allocated when block_size > 0");
            // SAFETY: pools allocated-once, see paged_pool_ptr setup above.
            let pool_k = unsafe { &mut *pool_k_ptr };
            let pool_v = unsafe { &mut *pool_v_ptr };
            let dispatch = if self.use_vllm_paged_attn {
                B::split_qkv_norm_rope_into_paged_cache_vllm
            } else {
                B::split_qkv_norm_rope_into_paged_cache
            };
            dispatch(
                ctx,
                &self.scratch.qkv_out,
                0,
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                &mut self.scratch.q_head_major,
                0,
                pool_k,
                pool_v,
                bt,
                tokens,
                nh,
                nkv,
                hd,
                pos_offset,
                eps,
                qk_mode,
                cache_len_before,
                cache.block_size,
                num_blocks_per_seq,
            )
            .is_ok()
        } else {
            B::split_qkv_norm_rope_into_cache(
                ctx,
                &self.scratch.qkv_out,
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                &mut self.scratch.q_head_major,
                &mut cache.k,
                &mut cache.v,
                tokens,
                nh,
                nkv,
                hd,
                pos_offset,
                eps,
                qk_mode,
                cache_len_before,
                cache_capacity,
            )
            .is_ok()
        };
        if !used_qkv_into_cache {
            // Fallback 1: fused split-QKV-norm-rope to head-major scratch
            // (Metal pre-decode-fusion path), then explicit cache append.
            let used_fused_qkv = B::split_qkv_norm_rope(
                ctx,
                &self.scratch.qkv_out,
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                &mut self.scratch.q_head_major,
                &mut self.scratch.k_head_major,
                &mut self.scratch.v_head_major,
                tokens,
                nh,
                nkv,
                hd,
                pos_offset,
                eps,
                qk_mode,
            )
            .is_ok();
            if !used_fused_qkv {
                // Fallback 2: original four-launch chain.
                B::split_qkv(
                    ctx,
                    &self.scratch.qkv_out,
                    &mut self.scratch.q_buf,
                    &mut self.scratch.k_buf,
                    &mut self.scratch.v_buf,
                    tokens,
                    q_dim,
                    kv_dim,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.q_buf,
                    q_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    &mut self.scratch.q_head_major,
                    tokens,
                    nh,
                    hd,
                    pos_offset,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.k_buf,
                    k_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    &mut self.scratch.k_head_major,
                    tokens,
                    nkv,
                    hd,
                    pos_offset,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.v_buf,
                    dummy,
                    &self.rope.cos,
                    &self.rope.sin,
                    &mut self.scratch.v_head_major,
                    tokens,
                    nkv,
                    hd,
                    pos_offset,
                    eps,
                    0,
                );
            }
            B::kv_cache_append_head_major(
                ctx,
                &mut cache.k,
                &mut cache.v,
                cache.len,
                cache.capacity,
                &self.scratch.k_head_major,
                &self.scratch.v_head_major,
                tokens,
                nkv,
                hd,
            );
        }
        cache.len += tokens;
        let kv_len = cache.len;
        let kv_stride = cache.capacity;

        if cache.block_size > 0 {
            // Paged decode: read from the shared pool via block_table.
            let bt = cache
                .block_table
                .as_ref()
                .expect("paged cache missing block_table");
            let cl_buf = cache
                .context_lens
                .as_mut()
                .expect("paged cache missing context_lens");
            let num_blocks_per_seq = cache.capacity / cache.block_size;
            let (pool_k_ptr, pool_v_ptr) =
                paged_pool_ptr.expect("paged_pools must be allocated when block_size > 0");
            // SAFETY: see paged_pool_ptr setup above.
            let pool_k = unsafe { &*pool_k_ptr };
            let pool_v = unsafe { &*pool_v_ptr };
            let final_kv_len = cache.len as u32;
            B::write_typed::<u32>(ctx, cl_buf, &[final_kv_len]);
            if self.use_vllm_paged_attn {
                let force_varlen_decode = self.runtime_env.vllm_decode_varlen;
                if tokens == 1 && !force_varlen_decode {
                    B::paged_decode_attention_v2(
                        ctx,
                        &self.scratch.q_head_major,
                        pool_k,
                        pool_v,
                        &mut self.scratch.attn_head_major_out,
                        bt,
                        cl_buf,
                        1, // num_seqs
                        nh,
                        nkv,
                        hd,
                        cache.block_size,
                        num_blocks_per_seq,
                        cache.len, // max_seq_len = current kv_len for single-seq decode
                    )
                    .expect("paged_decode_attention_v2 (single-seq decode)");
                } else {
                    B::paged_varlen_attention_vllm_layout(
                        ctx,
                        &self.scratch.q_head_major,
                        pool_k,
                        pool_v,
                        &mut self.scratch.attn_head_major_out,
                        bt,
                        cl_buf,
                        1, // num_seqs
                        nh,
                        nkv,
                        hd,
                        cache.block_size,
                        num_blocks_per_seq,
                        tokens,
                    )
                    .expect("paged_varlen_attention_vllm_layout (single-seq prefill)");
                }
            } else {
                B::paged_decode_attention(
                    ctx,
                    &self.scratch.q_head_major,
                    pool_k,
                    pool_v,
                    &mut self.scratch.attn_head_major_out,
                    bt,
                    cl_buf,
                    1, // num_seqs (single-seq m=1 path)
                    nh,
                    nkv,
                    hd,
                    cache.block_size,
                    num_blocks_per_seq,
                    tokens,
                )
                .expect("paged_decode_attention");
            }
            let _ = kv_stride; // consumed by contig path only
        } else {
            let attn_cfg = ferrum_kernels::backend::AttnConfig {
                num_heads: nh,
                num_kv_heads: nkv,
                head_dim: hd,
                causal: true,
                scale: 1.0 / (hd as f32).sqrt(),
                kv_seq_stride: kv_stride,
                sliding_window: cfg_base.sliding_window,
            };
            B::flash_attention(
                ctx,
                &self.scratch.q_head_major,
                &cache.k,
                &cache.v,
                &mut self.scratch.attn_head_major_out,
                1,
                tokens,
                kv_len,
                pos_offset,
                &attn_cfg,
            );
        }

        if let Some(us) = ferrum_kernels::backend::timer::finish_probe_timer_traced::<B>(
            attn_t0,
            ctx,
            "attn",
            "attention",
            li as u32,
        ) {
            ATTN_TIME_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
            ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 7. transpose head-major → token-major.
        //
        // For tokens=1 the two layouts are byte-identical: both
        // collapse to the flat [heads * head_dim] vector at offset
        // `head*hd + d`. Skip the dispatch and point o_proj at
        // attn_head_major_out directly. Saves 1 dispatch per layer
        // (×48 = 48 dispatches per decode token) on Qwen3-30B-A3B.
        let attn_token_major = if tokens == 1 {
            &self.scratch.attn_head_major_out
        } else {
            B::transpose_head_to_token(
                ctx,
                &self.scratch.attn_head_major_out,
                &mut self.scratch.attn_flat,
                tokens,
                nh,
                hd,
            );
            &self.scratch.attn_flat
        };

        // 8. O-proj.
        attn_layer
            .o_proj
            .forward(ctx, attn_token_major, &mut self.scratch.o_proj_out, tokens);

        // 9. fused residual-add + post-attention RMSNorm.
        B::fused_add_rms_norm(
            ctx,
            residual,
            &self.scratch.o_proj_out,
            &attn_layer.post_ln_w,
            eps,
            &mut self.scratch.norm_out,
            tokens,
            h,
        );

        // ── MoE FFN block ────────────────────────────────────────────
        // PLAYBOOK § 1.2 — migrated onto BackendTimer.
        let moe_t0 = ferrum_kernels::backend::timer::start_probe_timer_if::<B>(
            self.runtime_env.decode_op_profile,
            ctx,
        );

        // 10. Router gemv: norm_out [tokens, hidden] → router_logits [tokens, num_experts]
        moe_layer.router.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.router_logits,
            tokens,
        );

        // 11. Per-(token, expert) MLP dispatch + weighted combine.
        //
        // Two paths:
        //   - **Batched fast path** (decode m=1, all stacked variants
        //     present): single `gemv_quant_moe_id` dispatch covers all
        //     8 selected expert × 1 token gate gemvs in parallel; same
        //     for up and down. Cuts per-layer expert dispatches from
        //     ~32 (8 × 4 ops/pair) to 4 (gate + up + silu + down + 1 acc).
        //     Routes Qwen3-30B-A3B decode close to llama.cpp's
        //     `kernel_mul_mm_id`.
        //   - **Per-(token, expert) fallback** via `moe_forward` —
        //     used for prefill (m > 1), or when the backend doesn't
        //     populate stacked variants (CPU, synthetic-MoE tests).
        let stacked_path_available = moe_layer.experts.gate_stacked.is_some()
            && moe_layer.experts.up_stacked.is_some()
            && moe_layer.experts.down_stacked.is_some();

        // CUDA Marlin bucketed path: shared GPTQ store per (gate_up, down)
        // role + offset GEMMs per expert. Disabled with FERRUM_MOE_BUCKETED=0.
        let bucketed_path_available = moe_layer.experts.gate_up_marlin_stack.is_some()
            && moe_layer.experts.down_marlin_stack.is_some()
            && self.runtime_env.moe_bucketed;

        // Fast path for decode (tokens=1): the stacked decode impl
        // writes the weighted-sum result *directly* into `residual` via
        // `weighted_sum_residual_stacked`, skipping the moe_out scratch
        // and the trailing `add_inplace`. Saves 1 dispatch per layer.
        // Prefill (m>1) and the per-expert fallback still go through
        // moe_out + add_inplace.
        let decode_fast_path = stacked_path_available && tokens == 1;
        // Cross-layer fusion: when on the decode fast path AND there is
        // a next layer, fold its leading rms_norm into this layer's
        // tail (`weighted_sum_residual_norm_stacked`). Returns whether
        // the fusion ran so the caller can signal the next layer to
        // skip its standalone rms_norm.
        let did_norm_fusion = decode_fast_path && next_layer_idx.is_some();

        if bucketed_path_available {
            // CUDA: gather → per-expert m=N Marlin → silu → per-expert
            // m=N Marlin → moe_combine. Single-launch combine.
            crate::moe::moe_forward_bucketed::<B>(
                crate::moe::dispatch::MoeForwardBucketedParams {
                    ctx,
                    x: &self.scratch.norm_out,
                    router_logits: &self.scratch.router_logits,
                    out: &mut self.scratch.moe_out,
                    batch: tokens,
                    hidden_size: h,
                    expert_intermediate: self.cfg.expert_intermediate_size,
                    num_experts: self.cfg.num_experts,
                    top_k: self.cfg.num_experts_per_tok,
                    norm_topk_prob: self.cfg.norm_topk_prob,
                    experts: &moe_layer.experts,
                    x_packed: &mut self.scratch.x_packed_bucket,
                    gate_up_packed: &mut self.scratch.gate_up_packed_bucket,
                    silu_packed: &mut self.scratch.silu_stacked,
                    down_packed: &mut self.scratch.down_out_stacked,
                    route_scratch: &mut self.scratch.moe_route_scratch,
                    profile_bucket: false,
                    device_route: Some(crate::moe::dispatch::DeviceRouteScratch {
                        selected_ids: &mut self.scratch.selected_ids_buf,
                        pair_weights: &mut self.scratch.route_pair_weights_dev,
                        pairs_by_token: &mut self.scratch.route_pairs_dev,
                        packed_token_idx: &mut self.scratch.route_packed_idx_dev,
                        expert_offsets: &mut self.scratch.route_expert_offsets_dev,
                        sorted_tokens: &mut self.scratch.route_sorted_tokens_dev,
                        block_ids: &mut self.scratch.route_block_ids_dev,
                        total_post_pad: &mut self.scratch.route_total_post_pad_dev,
                    }),
                },
            )?;
        } else if stacked_path_available {
            if tokens > 1 {
                // Prefill: one batched 2-D mul_mm_id covers all
                // (token, expert) pairs in parallel.
                self.moe_forward_batched_prefill(ctx, li, tokens)?;
            } else {
                // Decode m=1: dedicated per-token path that fuses
                // residual-add into the final weighted-sum, and
                // optionally folds the next layer's rms_norm in too.
                self.moe_forward_stacked(ctx, li, tokens, residual, next_layer_idx)?;
            }
        } else {
            moe_forward::<B>(crate::moe::dispatch::MoeForwardParams {
                ctx,
                x: &self.scratch.norm_out,
                router_logits: &self.scratch.router_logits,
                out: &mut self.scratch.moe_out,
                batch: tokens,
                hidden_size: h,
                expert_intermediate: self.cfg.expert_intermediate_size,
                num_experts: self.cfg.num_experts,
                top_k: self.cfg.num_experts_per_tok,
                norm_topk_prob: self.cfg.norm_topk_prob,
                experts: &moe_layer.experts,
                x_single: &mut self.scratch.x_single,
                acc_buf: &mut self.scratch.acc_buf,
                gate_up_buf: &mut self.scratch.gate_up_buf,
                silu_buf: &mut self.scratch.silu_buf,
                down_buf: &mut self.scratch.down_buf,
                zero_hidden: &self.scratch.zero_hidden,
            })?;
        }

        // 12. residual += moe_out (skipped on decode fast path — already
        //     accumulated by `weighted_sum_residual_stacked`).
        if !decode_fast_path {
            B::add_inplace(ctx, residual, &self.scratch.moe_out, tokens * h);
        }

        if let Some(us) = ferrum_kernels::backend::timer::finish_probe_timer_traced::<B>(
            moe_t0, ctx, "moe", "moe", li as u32,
        ) {
            MOE_TIME_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
            MOE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(did_norm_fusion)
    }

    fn moe_forward_stacked(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        tokens: usize,
        residual: &mut B::Buffer,
        next_layer_idx: Option<usize>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        // `next_norm_w` is the next layer's `attn_layer.input_ln_w`.
        // We can't borrow `self.attn_layers[idx]` and pass &mut
        // self.scratch to the impl simultaneously, so collect the raw
        // pointer here. Safety: forward_layer holds &mut self for the
        // call; the borrow scopes are fully sequential.
        let next_norm_w_ptr: Option<*const B::Buffer> =
            next_layer_idx.map(|idx| &self.attn_layers[idx].input_ln_w as *const _);
        // SAFETY: pointer dereference is valid because:
        //   * The buffer lives in `self.attn_layers[idx]` which we
        //     borrowed immutably to take the pointer. We do not mutate
        //     `self.attn_layers` while `next_norm_w_ptr` is in use.
        //   * `&mut self.scratch` and `&self.moe_layers[li]` are disjoint
        //     fields from `self.attn_layers` so this is safe.
        let next_norm_w: Option<&B::Buffer> = next_norm_w_ptr.map(|p| unsafe { &*p });
        crate::moe::forward::moe_forward_stacked_decode_impl::<B>(
            ctx,
            &self.moe_layers[li],
            &mut self.scratch,
            cfg.base.hidden_size,
            cfg.expert_intermediate_size,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            cfg.norm_topk_prob,
            tokens,
            residual,
            next_norm_w,
            cfg.base.rms_norm_eps,
        )
    }

    fn moe_forward_batched_prefill(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        tokens: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        crate::moe::forward::moe_forward_batched_prefill_impl::<B>(
            ctx,
            &self.moe_layers[li],
            &mut self.scratch,
            cfg.base.hidden_size,
            cfg.expert_intermediate_size,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            cfg.norm_topk_prob,
            tokens,
        )
    }
}
