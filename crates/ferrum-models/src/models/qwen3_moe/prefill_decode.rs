use super::*;

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    /// Prefill: process `tokens` prompt tokens, return last-token logits.
    ///
    /// When `FERRUM_PREFIX_CACHE=1` is set, blocks of `tokens` whose
    /// content-addressed hash matches a previously-cached block are
    /// spliced into this seq's KV cache via `try_acquire_prefix_cache`,
    /// and only the unmatched suffix is actually prefilled.
    pub fn prefill_internal(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        assert!(!tokens.is_empty());
        self.ensure_kv(cache_id);

        // Block-level prefix cache. Env-gated for cautious rollout — the
        // engine's older whole-prompt PrefixCache (continuous_engine.rs)
        // still owns the default path; this knob opt-in switches the
        // model layer to block-level matching too.
        let cache_len_before = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(|c| c.len)
            .unwrap_or(0);
        let cached_prefix_tokens = if self.runtime_env.prefix_cache && cache_len_before == 0 {
            self.try_acquire_prefix_cache(cache_id, tokens)
        } else {
            0
        };

        // The suffix is what we actually push through the model. If the
        // entire prompt hit the cache (suffix empty), we leave at least
        // one block's worth to re-run so we still produce final logits.
        // TODO: dedicated "full-hit" path that recomputes only the last
        //   block's logits from cached KV — for now under-cache by 1 block.
        let cached_prefix_tokens = if cached_prefix_tokens >= tokens.len() {
            let block_size = self
                .kv_caches
                .get(cache_id)
                .and_then(|c| c.first())
                .map(|c| c.block_size)
                .unwrap_or(16);
            // Roll back one block so we still have a suffix to prefill.
            cached_prefix_tokens
                .saturating_sub(block_size)
                .min(tokens.len() - 1)
        } else {
            cached_prefix_tokens
        };
        if self.runtime_env.prefix_cache && cache_len_before == 0 {
            self.record_prefix_cache_probe(cached_prefix_tokens);
        }

        // Sync cache.len down if we rolled back. Cheap: 1 u32 write per layer.
        if cached_prefix_tokens > 0 {
            let caches_mut = self.kv_caches.get_mut(cache_id).expect("cache present");
            let mut ctx_tmp = B::new_context();
            for c in caches_mut.iter_mut() {
                if c.len != cached_prefix_tokens {
                    c.len = cached_prefix_tokens;
                    if let Some(cl) = c.context_lens.as_mut() {
                        B::write_typed::<u32>(&mut ctx_tmp, cl, &[cached_prefix_tokens as u32]);
                    }
                }
            }
            B::sync(&mut ctx_tmp);
        }

        let suffix_tokens: &[u32] = &tokens[cached_prefix_tokens..];
        let seq_len = suffix_tokens.len();
        assert!(seq_len > 0, "prefix cache must leave ≥1 suffix token");
        self.ensure_scratch(seq_len);

        let pos_offset = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(|c| c.len)
            .unwrap_or(0);

        let h = self.cfg.base.hidden_size;
        let vocab = self.cfg.base.vocab_size;
        let mut ctx = B::new_context();

        // FERRUM_DECODE_OP_PROFILE doubles as the prefill-profile gate
        // for Qwen3-MoE: when set, dump (attn-us, moe-us, total-us) at
        // the end of prefill so we can attribute the prefill bottleneck
        // between attention and MoE.
        // PLAYBOOK § 1.2 — migrated. Counter reset (when probe is on)
        // stays inline; timer construction goes through BackendTimer.
        let prefill_t0 = if self.runtime_env.decode_op_profile {
            for c in [
                &ATTN_TIME_US,
                &ATTN_CALLS,
                &MOE_TIME_US,
                &MOE_CALLS,
                &MOE_PREFILL_HOST_TOPK_US,
                &MOE_PREFILL_HOST_TOPK_CALLS,
                &MOE_PREFILL_GATE_US,
                &MOE_PREFILL_GATE_CALLS,
                &MOE_PREFILL_UP_US,
                &MOE_PREFILL_UP_CALLS,
                &MOE_PREFILL_SILU_US,
                &MOE_PREFILL_SILU_CALLS,
                &MOE_PREFILL_DOWN_US,
                &MOE_PREFILL_DOWN_CALLS,
                &MOE_PREFILL_WSUM_US,
                &MOE_PREFILL_WSUM_CALLS,
            ] {
                c.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            let mut t = <B as ferrum_kernels::backend::Backend>::make_timer();
            ferrum_kernels::backend::timer::BackendTimer::<B>::record_start(&mut t, &mut ctx);
            Some(t)
        } else {
            None
        };

        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        B::embedding_lookup(&mut ctx, &self.embed, suffix_tokens, &mut residual, h);

        // For prefill (seq_len > 1) the cross-layer norm fusion does
        // not apply (it lives on the decode fast path). We still pass
        // `next_layer_idx = None` so forward_layer emits the regular
        // tail.
        let mut prev_did_norm_fusion = false;
        let num_layers = self.cfg.base.num_layers;
        for li in 0..num_layers {
            let next_layer_idx = if li + 1 < num_layers {
                Some(li + 1)
            } else {
                None
            };
            prev_did_norm_fusion = self
                .forward_layer(
                    &mut ctx,
                    li,
                    cache_id,
                    &mut residual,
                    pos_offset,
                    seq_len,
                    next_layer_idx,
                    prev_did_norm_fusion,
                )
                .expect("forward_layer");
        }

        // Last-token slice → final RMSNorm → lm_head.
        B::copy_slice(
            &mut ctx,
            &residual,
            (seq_len - 1) * h,
            &mut self.scratch.last_hidden,
            0,
            h,
        );
        B::rms_norm(
            &mut ctx,
            &self.scratch.last_hidden,
            &self.final_norm_w,
            self.cfg.base.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );
        self.lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );

        if let Some(mut t0) = prefill_t0 {
            ferrum_kernels::backend::timer::BackendTimer::<B>::record_end(&mut t0, &mut ctx);
            let total_us = (ferrum_kernels::backend::timer::BackendTimer::<B>::elapsed_ms(&t0)
                * 1000.0) as u64;
            let attn_us = ATTN_TIME_US.load(std::sync::atomic::Ordering::Relaxed);
            let attn_n = ATTN_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let moe_us = MOE_TIME_US.load(std::sync::atomic::Ordering::Relaxed);
            let moe_n = MOE_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let other_us = total_us.saturating_sub(attn_us).saturating_sub(moe_us);
            eprintln!(
                "[prefill-profile] tokens={seq_len} total={} ms ({:.0} t/s)",
                total_us / 1000,
                seq_len as f64 * 1e6 / total_us as f64
            );
            let bucket = |label: &str, n: u64, us: u64| {
                if n > 0 {
                    eprintln!(
                        "  {label:>6}: {:7} ms ({:5.1}%) over {n:4} calls",
                        us / 1000,
                        us as f64 * 100.0 / total_us as f64
                    );
                }
            };
            bucket("attn", attn_n, attn_us);
            bucket("moe", moe_n, moe_us);
            bucket("other", 1, other_us);
            // MoE sub-stages — show as % of total prefill time so they
            // reconcile against the `moe` bucket above.
            let host_us = MOE_PREFILL_HOST_TOPK_US.load(std::sync::atomic::Ordering::Relaxed);
            let gate_us = MOE_PREFILL_GATE_US.load(std::sync::atomic::Ordering::Relaxed);
            let up_us = MOE_PREFILL_UP_US.load(std::sync::atomic::Ordering::Relaxed);
            let silu_us = MOE_PREFILL_SILU_US.load(std::sync::atomic::Ordering::Relaxed);
            let down_us = MOE_PREFILL_DOWN_US.load(std::sync::atomic::Ordering::Relaxed);
            let wsum_us = MOE_PREFILL_WSUM_US.load(std::sync::atomic::Ordering::Relaxed);
            let host_n = MOE_PREFILL_HOST_TOPK_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let gate_n = MOE_PREFILL_GATE_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let up_n = MOE_PREFILL_UP_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let silu_n = MOE_PREFILL_SILU_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let down_n = MOE_PREFILL_DOWN_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let wsum_n = MOE_PREFILL_WSUM_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            bucket("  host", host_n, host_us);
            bucket("  gate", gate_n, gate_us);
            bucket("  up", up_n, up_us);
            bucket("  silu", silu_n, silu_us);
            bucket("  down", down_n, down_us);
            bucket("  wsum", wsum_n, wsum_us);
        }
        self.scratch.residual = Some(residual);

        // Register hashes for blocks newly written by this prefill so
        // future requests with the same prompt prefix can reuse them.
        // No-op when prefix cache is disabled (prior_cached_tokens always
        // 0, register fires but `paged_block_alloc.is_none()` short-circuits)
        // or when the cached prefix already covered all blocks (the rolled-
        // back block gets re-registered with its newly-recomputed content).
        if self.runtime_env.prefix_cache {
            self.register_prefix_cache(cache_id, tokens, cached_prefix_tokens);
        }

        B::sync_before_host_readback(&mut ctx);
        B::to_vec(&self.scratch.logits, vocab)
    }

    /// Decode: 1 token at position `pos`, return next-step logits.
    pub fn decode_internal(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let h = self.cfg.base.hidden_size;
        let vocab = self.cfg.base.vocab_size;
        let mut ctx = B::new_context();

        let decode_t0 = if self.runtime_env.moe_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // FERRUM_DECODE_OP_PROFILE gates the per-stage breakdown emitted
        // at the bottom of every decode token. Reuses the same atomic
        // counters that `forward_layer` already populates (ATTN_TIME_US,
        // MOE_TIME_US — drained here per-token instead of per-prefill).
        // PLAYBOOK § 1.2 — migrated. See site 1959 for the same pattern.
        let stage_t0 = if self.runtime_env.decode_op_profile {
            for c in [
                &ATTN_TIME_US,
                &ATTN_CALLS,
                &MOE_TIME_US,
                &MOE_CALLS,
                &DEC_ROUTE_US,
                &DEC_GATE_US,
                &DEC_UP_US,
                &DEC_SILU_US,
                &DEC_DOWN_US,
                &DEC_WSUM_US,
                &DEC_EMBED_US,
                &DEC_FINAL_NORM_US,
                &DEC_LM_HEAD_US,
            ] {
                c.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            let mut t = <B as ferrum_kernels::backend::Backend>::make_timer();
            ferrum_kernels::backend::timer::BackendTimer::<B>::record_start(&mut t, &mut ctx);
            Some(t)
        } else {
            None
        };
        let prof = stage_t0.is_some();
        // PLAYBOOK § 1.2 — migrated. Replaces `Instant::now()` + `B::sync`
        // with BackendTimer (CUDA event-accurate) and pushes a chrome-trace
        // event so visualize_layerwise.py shows embed / final_norm / lm_head
        // as separate stages within the decode_step category.
        let decode_op_profile = self.runtime_env.decode_op_profile;
        let stage_start =
            |ctx: &mut B::Context| -> Option<<B as ferrum_kernels::backend::Backend>::Timer> {
                ferrum_kernels::backend::timer::start_probe_timer_if::<B>(decode_op_profile, ctx)
            };
        let stage_finish = |t: Option<<B as ferrum_kernels::backend::Backend>::Timer>,
                            ctx: &mut B::Context,
                            name: &str,
                            c: &AtomicU64| {
            if let Some(us) = ferrum_kernels::backend::timer::finish_probe_timer_traced::<B>(
                t,
                ctx,
                name,
                "decode_step",
                0,
            ) {
                c.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
            }
        };
        let mt0 = std::time::Instant::now();

        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let t0 = stage_start(&mut ctx);
        B::embedding_lookup(&mut ctx, &self.embed, &[token], &mut residual, h);
        stage_finish(t0, &mut ctx, "embed", &DEC_EMBED_US);
        let _ = mt0; // silence if unused on non-profile builds

        // Cross-layer rms_norm fusion: layer L's MoE tail folds the
        // next layer's leading rms_norm into its weighted-sum-residual
        // when the decode fast path applies. The flag carries forward.
        let mut prev_did_norm_fusion = false;
        let num_layers = self.cfg.base.num_layers;
        for li in 0..num_layers {
            let next_layer_idx = if li + 1 < num_layers {
                Some(li + 1)
            } else {
                None
            };
            prev_did_norm_fusion = self
                .forward_layer(
                    &mut ctx,
                    li,
                    cache_id,
                    &mut residual,
                    pos as usize,
                    1,
                    next_layer_idx,
                    prev_did_norm_fusion,
                )
                .expect("forward_layer");
        }

        let t0 = stage_start(&mut ctx);
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.base.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );
        stage_finish(t0, &mut ctx, "final_norm", &DEC_FINAL_NORM_US);

        let t0 = stage_start(&mut ctx);
        self.lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );
        stage_finish(t0, &mut ctx, "lm_head", &DEC_LM_HEAD_US);

        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);

        // FERRUM_DECODE_OP_PROFILE: per-token decode breakdown.
        if let Some(mut t0) = stage_t0 {
            use std::sync::atomic::Ordering;
            ferrum_kernels::backend::timer::BackendTimer::<B>::record_end(&mut t0, &mut ctx);
            let total_us = (ferrum_kernels::backend::timer::BackendTimer::<B>::elapsed_ms(&t0)
                * 1000.0) as u64;
            let attn_us = ATTN_TIME_US.swap(0, Ordering::Relaxed);
            let moe_us = MOE_TIME_US.swap(0, Ordering::Relaxed);
            let route = DEC_ROUTE_US.swap(0, Ordering::Relaxed);
            let gate = DEC_GATE_US.swap(0, Ordering::Relaxed);
            let up = DEC_UP_US.swap(0, Ordering::Relaxed);
            let silu = DEC_SILU_US.swap(0, Ordering::Relaxed);
            let down = DEC_DOWN_US.swap(0, Ordering::Relaxed);
            let wsum = DEC_WSUM_US.swap(0, Ordering::Relaxed);
            let embed = DEC_EMBED_US.swap(0, Ordering::Relaxed);
            let fnorm = DEC_FINAL_NORM_US.swap(0, Ordering::Relaxed);
            let lmhead = DEC_LM_HEAD_US.swap(0, Ordering::Relaxed);
            let other = total_us.saturating_sub(attn_us + moe_us + embed + fnorm + lmhead);
            let pct = |us: u64| -> f64 {
                if total_us == 0 {
                    0.0
                } else {
                    100.0 * us as f64 / total_us as f64
                }
            };
            eprintln!(
                "[decode-prof] total={} ms | attn={} ({:.1}%) | moe={} ({:.1}%) [route={} gate={} up={} silu={} down={} wsum={}] | embed={} fnorm={} lmhead={} other={} ({:.1}%)",
                total_us / 1000,
                attn_us / 1000, pct(attn_us),
                moe_us / 1000, pct(moe_us),
                route / 1000, gate / 1000, up / 1000, silu / 1000, down / 1000, wsum / 1000,
                embed / 1000, fnorm / 1000, lmhead / 1000,
                other / 1000, pct(other),
            );
        }

        // Drain MoE per-op counters every decode step. The counters
        // accumulate across all 48 layers; printing per-step gives a
        // per-token breakdown.
        if let Some(t0) = decode_t0 {
            use crate::moe::dispatch::*;
            use std::sync::atomic::Ordering;
            let total_us = t0.elapsed().as_micros() as u64;
            let sync_us = MOE_SYNC_US.swap(0, Ordering::Relaxed);
            let sync_n = MOE_SYNC_CALLS.swap(0, Ordering::Relaxed);
            let topk_us = MOE_HOST_TOPK_US.swap(0, Ordering::Relaxed);
            let topk_n = MOE_HOST_TOPK_CALLS.swap(0, Ordering::Relaxed);
            let gu_us = MOE_GEMV_GATE_UP_US.swap(0, Ordering::Relaxed);
            let gu_n = MOE_GEMV_GATE_UP_CALLS.swap(0, Ordering::Relaxed);
            let silu_us = MOE_SILU_US.swap(0, Ordering::Relaxed);
            let silu_n = MOE_SILU_CALLS.swap(0, Ordering::Relaxed);
            let dn_us = MOE_GEMV_DOWN_US.swap(0, Ordering::Relaxed);
            let dn_n = MOE_GEMV_DOWN_CALLS.swap(0, Ordering::Relaxed);
            let sa_us = MOE_SCALED_ADD_US.swap(0, Ordering::Relaxed);
            let sa_n = MOE_SCALED_ADD_CALLS.swap(0, Ordering::Relaxed);
            let cp_us = MOE_COPY_US.swap(0, Ordering::Relaxed);
            let cp_n = MOE_COPY_CALLS.swap(0, Ordering::Relaxed);
            eprintln!(
                "[moe-prof] decode total={} ms | sync={} ms ({}x) | host_topk={} ms ({}x) | gate_up={} ms ({}x) | silu={} ms ({}x) | down={} ms ({}x) | scaled_add={} ms ({}x) | copy={} ms ({}x)",
                total_us / 1000,
                sync_us / 1000, sync_n,
                topk_us / 1000, topk_n,
                gu_us / 1000, gu_n,
                silu_us / 1000, silu_n,
                dn_us / 1000, dn_n,
                sa_us / 1000, sa_n,
                cp_us / 1000, cp_n,
            );

            // Bucketed CUDA MoE per-phase breakdown (CUDA M3 path).
            let bk_layers = MOE_BUCKET_LAYER_CALLS.swap(0, Ordering::Relaxed);
            if bk_layers > 0 {
                let bk_sync = MOE_BUCKET_SYNC_US.swap(0, Ordering::Relaxed);
                let bk_d2h = MOE_BUCKET_D2H_US.swap(0, Ordering::Relaxed);
                let bk_route = MOE_BUCKET_ROUTE_US.swap(0, Ordering::Relaxed);
                let bk_plan = MOE_BUCKET_PLAN_US.swap(0, Ordering::Relaxed);
                let bk_gather = MOE_BUCKET_GATHER_US.swap(0, Ordering::Relaxed);
                let bk_g1 = MOE_BUCKET_GEMM1_US.swap(0, Ordering::Relaxed);
                let bk_silu_us = MOE_BUCKET_SILU_US.swap(0, Ordering::Relaxed);
                let bk_g3 = MOE_BUCKET_GEMM3_US.swap(0, Ordering::Relaxed);
                let bk_comb = MOE_BUCKET_COMBINE_US.swap(0, Ordering::Relaxed);
                let bk_total = bk_sync
                    + bk_d2h
                    + bk_route
                    + bk_plan
                    + bk_gather
                    + bk_g1
                    + bk_silu_us
                    + bk_g3
                    + bk_comb;
                eprintln!(
                    "[bucket-prof] layers={} bk_total={} ms | sync={} d2h={} host_route={} plan={} gather={} gemm1={} silu={} gemm3={} combine={} (us, summed across layers)",
                    bk_layers, bk_total / 1000,
                    bk_sync, bk_d2h, bk_route, bk_plan, bk_gather,
                    bk_g1, bk_silu_us, bk_g3, bk_comb,
                );
                let profile = global_profile();
                if profile.is_enabled() {
                    let _ = profile.push_event(
                        "bucket_prof",
                        profile_fields_from_json(serde_json::json!({
                            "layers": bk_layers,
                        })),
                        profile_fields_from_json(serde_json::json!({
                            "bk_total": bk_total,
                            "sync": bk_sync,
                            "d2h": bk_d2h,
                            "host_route": bk_route,
                            "plan": bk_plan,
                            "gather": bk_gather,
                            "gemm1": bk_g1,
                            "silu": bk_silu_us,
                            "gemm3": bk_g3,
                            "combine": bk_comb,
                        })),
                        false,
                    );
                }
            }
        }

        B::sync_before_host_readback(&mut ctx);
        B::to_vec(&self.scratch.logits, vocab)
    }
}
