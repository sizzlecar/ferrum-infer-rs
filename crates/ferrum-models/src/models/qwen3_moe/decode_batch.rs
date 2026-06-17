use super::*;

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    /// Multi-sequence batched decode (Phase 4b for MoE).
    ///
    /// Mirrors `LlamaFamilyModel::decode_batch_internal` but adapted to
    /// the MoE forward. The wins come from running the GEMM-heavy ops
    /// (qkv_proj, o_proj, router, MoE expert mul_mm_id, lm_head) at
    /// m=M, even though attention stays a per-item loop because
    /// Qwen3-MoE uses contiguous KV — no paged path here.
    ///
    /// Cross-layer rms_norm fusion (the `weighted_sum_residual_norm_stacked`
    /// fast path) is disabled in batched mode: the prefill MoE path
    /// (`moe_forward_batched_prefill_impl`) writes to `moe_out` and we
    /// add to residual explicitly. Each layer's leading rms_norm runs
    /// at m=M, which is one fused dispatch on M rows — cheap.
    pub fn decode_batch_internal(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_internal_with_full_logits(batch, false)
    }

    pub fn decode_batch_internal_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        let m = batch.len();
        if m == 0 {
            return Vec::new();
        }
        if m == 1 {
            let (cid, tok, pos) = &batch[0];
            return vec![self.decode_internal(cid, *tok, *pos)];
        }

        let prof_t0 = if self.runtime_env.decode_op_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };

        for (cid, _, _) in batch {
            self.ensure_kv(cid);
        }
        self.ensure_scratch(m);
        self.scratch.enable_batched_decode_scratch(&self.cfg);

        let h = self.cfg.base.hidden_size;
        let vocab = self.cfg.base.vocab_size;
        let num_layers = self.cfg.base.num_layers;
        let central_paged_len_bump = self.paged_pools.is_some() && !self.supports_varlen_qkv;
        let mut ctx = B::new_context();
        for (cid, _, _) in batch {
            let target_len = self
                .kv_caches
                .get(cid)
                .and_then(|layers| layers.first())
                .map(|cache| cache.len.saturating_add(1))
                .unwrap_or(1);
            self.ensure_paged_kv_capacity_for_cache_id(&mut ctx, cid, target_len)
                .expect("paged KV dynamic grow");
        }

        // 0. Embed all M tokens into residual [M, H]
        let tokens: Vec<u32> = batch.iter().map(|(_, t, _)| *t).collect();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        B::embedding_lookup(&mut ctx, &self.embed, &tokens, &mut residual, h);

        // CUDA varlen/graph path: pre-populate paged-batch scratch
        // (block_tables / context_lens / pos_offsets / cu_seqlens_q) ONCE
        // before the layer loop. Metal has no varlen QKV kernel and must
        // keep the historical PR #81 per-layer metadata/bump flow inside
        // `forward_layer_batched_decode`; moving that work out of the layer
        // loop regressed Qwen3-30B-A3B c16 on Apple Silicon by ~4x.
        let prepopulate_paged_decode = self.paged_pools.is_some() && self.supports_varlen_qkv;
        if prepopulate_paged_decode {
            self.populate_paged_batch_scratch_decode(&mut ctx, batch, m);
        }

        // ── CUDA-graph capture/replay (FERRUM_MOE_GRAPH=1) ───────────
        //
        // Per-shape graph cache keyed by `m_padded`. Captures the layer
        // loop + final rms_norm + lm_head — every kernel that's stable
        // across decode iters. Pre-work (paged scratch upload + embed
        // + marlin gather scratch grow) and post-work (sync + to_vec)
        // stay eager.
        //
        // Requires FERRUM_MOE_DEVICE_ROUTE=1 + FERRUM_VLLM_MOE=1 to be
        // graph-clean (otherwise moe_forward_bucketed has D2H + host
        // pointer writes that fail capture). Recovery: if begin_capture
        // / end_capture errors, set `batched_graph_failed=true` and
        // never retry — stays eager for the rest of this model's life.
        let graph_enabled = self.moe_graph_enabled_graph_clean();
        // Per-m graph cache. Key is `m` exactly, NOT `m_padded`.
        // Captured kernel launches bake the grid_dim / loop bounds for
        // the m used at capture time; replaying the same graph for a
        // different actual m → wrong indexing into per-seq scratch
        // (block_tables / context_lens) → early-EOS garbage tokens.
        // Llama's unified graph keys by `(m_total, num_seqs)` for the
        // same reason. For MoE decode where each seq contributes 1
        // token (q_len=1), m is num_seqs is m_total, so the single
        // u64 m fully identifies the shape.
        //
        // High bit set so this key space never collides with the
        // single-item / batched-graph keys used by LlamaFamilyModel
        // (which share the same DECODE_GRAPHS cache).
        let graph_key: u64 = (1u64 << 63) | (m as u64);
        let cache_has_key = self.batched_graph_keys_seen.contains(&graph_key);

        // Pre-grow marlin gather scratch before capture begins —
        // `with_marlin_gather_scratch`'s in-place grow does
        // `stream.alloc` which CUDA Graph capture forbids inside the
        // captured stream. Bucketed MoE phase 1/3 GEMMs need at most
        // `total_pairs * intermediate_size` (phase 3 is the bigger k).
        let total_pairs = m * self.cfg.num_experts_per_tok;
        let max_marlin_required = total_pairs * self.cfg.expert_intermediate_size;
        if prepopulate_paged_decode {
            B::pregrow_marlin_gather_scratch(&mut ctx, max_marlin_required);
        }

        // Settle pre-work (write_typeds + embed + pregrow) before
        // begin_capture or replay — the captured region must start
        // from a quiescent stream state.
        if prepopulate_paged_decode {
            B::sync(&mut ctx);
        }

        let mut did_pure_replay = false;
        if graph_enabled && cache_has_key && !self.batched_graph_failed {
            match B::replay_graph(&mut ctx, graph_key) {
                Ok(true) => did_pure_replay = true,
                Ok(false) => {}
                Err(e) => {
                    self.batched_graph_failed = true;
                    eprintln!("[moe-graph] replay err: {e}");
                }
            }
        }

        const MOE_GRAPH_WARMUP: usize = 3;
        let should_capture = graph_enabled
            && !self.batched_graph_failed
            && !cache_has_key
            && self.batched_graph_warmup >= MOE_GRAPH_WARMUP
            && !did_pure_replay;
        // Only bump the warmup counter when the flag is on — otherwise
        // we accumulate uselessly across the model's lifetime and the
        // first FERRUM_MOE_GRAPH=1 call would skip warmup.
        if graph_enabled && !did_pure_replay {
            self.batched_graph_warmup += 1;
        }

        if should_capture {
            if let Err(e) = B::begin_graph_capture(&mut ctx) {
                eprintln!("[moe-graph] begin_capture err: {e}");
                self.batched_graph_failed = true;
            }
        }

        // Inner profile: layer loop / final norm / lm_head Rust-side wall.
        // FERRUM_RBD_PROF=1 (set the same time as the engine's per-stage
        // timer) — splits the 17 ms / iter "decode wall" at c=32 to find
        // whether per-layer Rust dispatch (48 × ?us) or final norm/lm_head
        // dominates. Zero overhead when env unset.
        let inner_prof = self.runtime_env.rbd_prof;
        let t_loop = if inner_prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if !did_pure_replay {
            // 1..num_layers: batched forward for each layer.
            // Records into graph if capture is active above.
            for li in 0..num_layers {
                self.forward_layer_batched_decode(&mut ctx, li, batch, &mut residual, m)
                    .expect("forward_layer_batched_decode");
            }

            // Final RMSNorm on [M, H] → norm_out [M, H]
            B::rms_norm(
                &mut ctx,
                &residual,
                &self.final_norm_w,
                self.cfg.base.rms_norm_eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );

            // LM head with m=M → batch_logits [M, vocab]
            self.lm_head.forward(
                &mut ctx,
                &self.scratch.norm_out,
                &mut self.scratch.batch_logits,
                m,
            );
        }
        let loop_us = t_loop.map(|t| t.elapsed().as_micros() as u64);

        if central_paged_len_bump {
            for (cid, _, _) in batch.iter() {
                let caches = self
                    .kv_caches
                    .get_mut(cid)
                    .expect("paged batched: cache missing for central len bump");
                for cache in caches.iter_mut().take(num_layers) {
                    cache.len += 1;
                }
            }
        }

        if should_capture && !self.batched_graph_failed {
            if let Err(e) = B::end_graph_capture(&mut ctx, graph_key) {
                eprintln!("[moe-graph] end_capture err: {e}");
                self.batched_graph_failed = true;
            } else {
                self.batched_graph_keys_seen.insert(graph_key);
                // Post-capture replay — capture only RECORDS, so without
                // this the layer-loop kernels never actually execute and
                // batch_logits stays uninitialised.
                if let Err(e) = B::replay_graph(&mut ctx, graph_key) {
                    eprintln!("[moe-graph] post-capture replay err: {e}");
                    self.batched_graph_failed = true;
                }
            }
        }

        let t_sync = if inner_prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::sync(&mut ctx);
        let sync_us = t_sync.map(|t| t.elapsed().as_micros() as u64);
        self.scratch.residual = Some(residual);

        // Greedy fast path: when `FERRUM_GREEDY_ARGMAX=1` is set, do the
        // argmax on-device and return one f32 per row carrying the
        // token id (cast). Replaces the m × vocab × 2 B logit download +
        // host argmax with one kernel + tiny D2H. At c=32, vocab=152064:
        // 19.5 MB + ~5 ms CPU → 128 B + ~0.3 ms GPU.
        // The engine has a complementary fast path that interprets a
        // size-1 Vec<f32> as `TokenId::new(logits[0] as u32)`, skipping
        // sample_with_processors entirely.
        let greedy = self.runtime_env.greedy_argmax && !force_full_logits;
        // One-shot log on first decode call so the bench / smoke tests
        // can confirm the env var actually wired through. Cheap atomic
        // bool fence; no per-call cost after first hit.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[qwen3_moe] decode_batch_internal: FERRUM_GREEDY_ARGMAX={} (path={})",
                    if greedy { "1" } else { "0" },
                    if greedy { "GPU argmax" } else { "host argmax" }
                );
            }
        }
        let t_argmax = if inner_prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let all = if greedy {
            // No logits download — argmax kernel returns m tokens.
            // Callers expect Vec<f32> per item, so encode tokens as floats.
            // Note: rows of the returned `all` are size 1 (not vocab),
            // and the splitter below reads only 1 element per row.
            let tokens = B::argmax_rows_f16(&mut ctx, &self.scratch.batch_logits, m, vocab)
                .expect("argmax_rows_f16");
            tokens.into_iter().map(|t| t as f32).collect::<Vec<f32>>()
        } else {
            B::to_vec(&self.scratch.batch_logits, m * vocab)
        };
        let argmax_us = t_argmax.map(|t| t.elapsed().as_micros() as u64);
        if inner_prof {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(32) {
                eprintln!(
                    "[moe-inner-prof] iter#{} m={} loop={}us sync={}us argmax={}us",
                    n,
                    m,
                    loop_us.unwrap_or(0),
                    sync_us.unwrap_or(0),
                    argmax_us.unwrap_or(0),
                );
            }
        }

        // Profile dump (one decode_batch_internal call = one decode step
        // covering all m tokens).
        if let Some(t0) = prof_t0 {
            use std::sync::atomic::Ordering;
            let total_us = t0.elapsed().as_micros() as u64;
            let dense = BD_DENSE_US.swap(0, Ordering::Relaxed);
            let attn = BD_ATTN_PERITEM_US.swap(0, Ordering::Relaxed);
            let moe = BD_MOE_US.swap(0, Ordering::Relaxed);
            let layers = BD_LAYER_CALLS.swap(0, Ordering::Relaxed);
            let other = total_us.saturating_sub(dense + attn + moe);
            let pct = |us: u64| -> f64 {
                if total_us == 0 {
                    0.0
                } else {
                    100.0 * us as f64 / total_us as f64
                }
            };
            // MoE sub-stage breakdown — meaningful when
            // moe_forward_batched_decode_impl was used.
            let moe_route = MOE_BATCHED_DECODE_ROUTE_US.swap(0, Ordering::Relaxed);
            let moe_gate = MOE_BATCHED_DECODE_GATE_US.swap(0, Ordering::Relaxed);
            let moe_up = MOE_BATCHED_DECODE_UP_US.swap(0, Ordering::Relaxed);
            let moe_silu = MOE_BATCHED_DECODE_SILU_US.swap(0, Ordering::Relaxed);
            let moe_down = MOE_BATCHED_DECODE_DOWN_US.swap(0, Ordering::Relaxed);
            let moe_wsum = MOE_BATCHED_DECODE_WSUM_US.swap(0, Ordering::Relaxed);
            eprintln!(
                "[batched-decode-prof] m={} layers={} total={} ms | dense={} ({:.1}%) | attn_peritem={} ({:.1}%) | moe={} ({:.1}%) [route={} gate={} up={} silu={} down={} wsum={}] | other={} ({:.1}%)",
                m, layers, total_us / 1000,
                dense / 1000, pct(dense),
                attn / 1000, pct(attn),
                moe / 1000, pct(moe),
                moe_route / 1000, moe_gate / 1000, moe_up / 1000,
                moe_silu / 1000, moe_down / 1000, moe_wsum / 1000,
                other / 1000, pct(other),
            );
            let profile = global_profile();
            if profile.is_enabled() {
                let _ = profile.push_event(
                    "batched_decode_prof",
                    profile_fields_from_json(serde_json::json!({
                        "m": m,
                        "layers": layers,
                    })),
                    profile_fields_from_json(serde_json::json!({
                        "total": total_us,
                        "dense": dense,
                        "attn_peritem": attn,
                        "moe": moe,
                        "route": moe_route,
                        "gate": moe_gate,
                        "up": moe_up,
                        "silu": moe_silu,
                        "down": moe_down,
                        "wsum": moe_wsum,
                        "other": other,
                    })),
                    graph_enabled,
                );
            }

            // Bucketed CUDA MoE per-phase breakdown (FERRUM_MOE_PROFILE=1).
            // Counters are summed across all layers in this decode step.
            use crate::moe::dispatch::*;
            let bk_layers = MOE_BUCKET_LAYER_CALLS.swap(0, Ordering::Relaxed);
            if bk_layers > 0 {
                let bk_sync = MOE_BUCKET_SYNC_US.swap(0, Ordering::Relaxed);
                let bk_d2h = MOE_BUCKET_D2H_US.swap(0, Ordering::Relaxed);
                let bk_route = MOE_BUCKET_ROUTE_US.swap(0, Ordering::Relaxed);
                let bk_plan = MOE_BUCKET_PLAN_US.swap(0, Ordering::Relaxed);
                let bk_gather = MOE_BUCKET_GATHER_US.swap(0, Ordering::Relaxed);
                let bk_g1 = MOE_BUCKET_GEMM1_US.swap(0, Ordering::Relaxed);
                let bk_silu = MOE_BUCKET_SILU_US.swap(0, Ordering::Relaxed);
                let bk_g3 = MOE_BUCKET_GEMM3_US.swap(0, Ordering::Relaxed);
                let bk_comb = MOE_BUCKET_COMBINE_US.swap(0, Ordering::Relaxed);
                let bk_total = bk_sync
                    + bk_d2h
                    + bk_route
                    + bk_plan
                    + bk_gather
                    + bk_g1
                    + bk_silu
                    + bk_g3
                    + bk_comb;
                eprintln!(
                    "[bucket-prof] layers={} bk_total={} ms | sync={} d2h={} host_route={} plan={} gather={} gemm1={} silu={} gemm3={} combine={} (us, summed across layers)",
                    bk_layers, bk_total / 1000,
                    bk_sync, bk_d2h, bk_route, bk_plan, bk_gather,
                    bk_g1, bk_silu, bk_g3, bk_comb,
                );
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
                            "silu": bk_silu,
                            "gemm3": bk_g3,
                            "combine": bk_comb,
                        })),
                        graph_enabled,
                    );
                }
            }
        }

        if greedy {
            // `all` has shape [m] in this branch — each entry is a token
            // id encoded as f32. Wrap each in a 1-elem Vec so the
            // engine's greedy fast path can detect (len==1) and pick
            // `logits[0] as u32` without going through sample_with_processors.
            all.into_iter().map(|t| vec![t]).collect()
        } else {
            (0..m)
                .map(|i| all[i * vocab..(i + 1) * vocab].to_vec())
                .collect()
        }
    }

    /// Pre-populate per-decode-step paged-batch scratch (block_tables /
    /// context_lens / pos_offsets / cu_seqlens_q) on the device side
    /// and bump cache.len for all layers in lockstep.
    ///
    /// Called ONCE before the layer loop in `decode_batch_internal`,
    /// replacing the per-layer write_typed inside
    /// `forward_layer_batched_decode`. This is the prerequisite for
    /// CUDA Graph capture: H2D copies during capture record the host
    /// pointer, so they must happen outside the captured region
    /// against stable scratch addresses.
    ///
    /// Invariant: all layers of a given cache_id share the same
    /// `paged_block_indices` and `len`, so we read layer-0 for the
    /// "shared" host values and bump all layers identically.
    fn populate_paged_batch_scratch_decode(
        &mut self,
        ctx: &mut B::Context,
        batch: &[(String, u32, u32)],
        m: usize,
    ) {
        if !self.paged_pools.is_some() {
            return; // non-paged path doesn't use this scratch
        }
        let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
        let num_layers = self.cfg.base.num_layers;

        let mut stacked_bt: Vec<u32> = vec![0u32; m * max_blocks_per_seq];
        let mut stacked_cl: Vec<u32> = vec![0u32; m];
        let mut pos_offsets_host: Vec<u32> = vec![0u32; m];
        let mut cu_seqlens_host: Vec<u32> = vec![0u32; m + 1];
        for i in 0..=m {
            cu_seqlens_host[i] = i as u32;
        }
        for (i, (cache_id, _, _)) in batch.iter().enumerate() {
            let caches = self
                .kv_caches
                .get_mut(cache_id)
                .expect("paged batched: cache not present");
            // Read shared values from layer-0 (invariant: same len + block
            // indices across all layers for one cache_id).
            let cache0 = &caches[0];
            pos_offsets_host[i] = cache0.len as u32;
            let blocks = &cache0.paged_block_indices;
            let n_to_copy = blocks.len().min(max_blocks_per_seq);
            stacked_bt[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
                .copy_from_slice(&blocks[..n_to_copy]);
            // Bump ALL layers in lockstep (replaces the per-layer bump
            // that used to live in forward_layer_batched_decode).
            for li in 0..num_layers {
                caches[li].len += 1;
            }
            stacked_cl[i] = caches[0].len as u32;
        }

        let bt_buf = self
            .scratch
            .paged_batch_block_tables
            .as_mut()
            .expect("paged_batch_block_tables missing");
        B::write_typed::<u32>(ctx, bt_buf, &stacked_bt);
        let cl_buf = self
            .scratch
            .paged_batch_context_lens
            .as_mut()
            .expect("paged_batch_context_lens missing");
        B::write_typed::<u32>(ctx, cl_buf, &stacked_cl);
        let pos_buf = self
            .scratch
            .paged_batch_pos_offsets
            .as_mut()
            .expect("paged_batch_pos_offsets missing");
        B::write_typed::<u32>(ctx, pos_buf, &pos_offsets_host);
        let cu_buf = self
            .scratch
            .paged_batch_cu_seqlens_q
            .as_mut()
            .expect("paged_batch_cu_seqlens_q missing");
        B::write_typed::<u32>(ctx, cu_buf, &cu_seqlens_host);
    }

    /// One transformer layer over M items: GEMMs at m=M, per-item
    /// attention loop, MoE FFN at m=M via the prefill batched path.
    /// Mirrors `LlamaFamilyModel::forward_layer_batched_decode` minus
    /// the paged branch.
    ///
    /// PRECONDITION (paged path): caller must invoke
    /// `populate_paged_batch_scratch_decode` once before the layer
    /// loop. This function reads the pre-populated scratch buffers
    /// (`paged_batch_block_tables` etc.) and no longer does its own
    /// host gather / write_typed / cache.len bump.
    fn forward_layer_batched_decode(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        batch: &[(String, u32, u32)],
        residual: &mut B::Buffer,
        m: usize,
    ) -> Result<()> {
        let cfg_base = &self.cfg.base;
        let h = cfg_base.hidden_size;
        let nh = cfg_base.num_heads;
        let nkv = cfg_base.num_kv_heads;
        let hd = cfg_base.head_dim;
        let eps = cfg_base.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        let attn_layer = &self.attn_layers[li];
        let qk_mode: i32 = if cfg_base.has_qk_norm { 1 } else { 2 };
        let dummy_w = &attn_layer.input_ln_w;
        let q_norm_w = attn_layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = attn_layer.k_norm_w.as_ref().unwrap_or(dummy_w);

        let prof = self.runtime_env.decode_op_profile;
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
        if prof {
            BD_LAYER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let dense_t0 = stage_t0();

        // 1. rms_norm [M, H] → norm_out
        B::rms_norm(
            ctx,
            residual,
            &attn_layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m,
            h,
        );

        // 2. qkv_proj GEMM at m=M: norm_out [M, H] → qkv_out [M, QKV]
        attn_layer
            .qkv_proj
            .forward(ctx, &self.scratch.norm_out, &mut self.scratch.qkv_out, m);

        // ── Paged batched attention path ───────────────────────────────
        //
        // Mirrors LlamaFamilyModel's Phase 4b paged batched-decode. When
        // `FERRUM_METAL_PAGED_KV=1` was set at ensure_kv time, each
        // cache_id has paged metadata (block_table + context_lens) and
        // K/V live in the shared `paged_pools[layer]` pool. This path:
        //   1. m × `split_qkv_norm_rope_into_paged_cache` writes K/V into
        //      the pool at each item's allocated blocks AND fills
        //      `paged_batch_q[i*q_dim ..]` with that item's head-major Q.
        //   2. Build `paged_batch_block_tables [m, max_blocks_per_seq]`
        //      and `paged_batch_context_lens [m]` host-side, upload.
        //   3. ONE `paged_decode_attention(num_seqs=m)` call reads all m
        //      sequences' K/V from the pool via per-seq block_tables,
        //      writes outputs to `paged_batch_o [m, q_dim]`.
        //   4. Per-item copy_slice paged_batch_o[i] → attn_flat[i*q_dim].
        //
        // This is the structural fix for the c=16 attn_peritem cliff
        // (~55 ms / round of 16 sequential m=1 flash_attn + plumbing).
        let is_paged = self.paged_pools.is_some();
        if is_paged {
            stage_end(dense_t0, ctx, &BD_DENSE_US);
            let attn_t0 = stage_t0();

            let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
            let block_size = 16; // matches PAGED_BLOCK_SIZE in ensure_kv
            let qkv_stride = q_dim + 2 * kv_dim;

            // Paged scratch (block_tables / context_lens / pos_offsets /
            // cu_seqlens_q) was pre-populated by
            // `populate_paged_batch_scratch_decode` before the layer
            // loop — see precondition docstring on this fn. We just
            // claim per-layer pool pointers and proceed to QKV+attn.
            let q_head_major_size_bytes = (q_dim * std::mem::size_of::<f32>()) as u64;
            let _qkv_stride_bytes = (qkv_stride * std::mem::size_of::<f32>()) as u64;
            let _ = q_head_major_size_bytes; // unused in batched path
            let pool_ptr = {
                let pools = self.paged_pools.as_mut().unwrap();
                (
                    &mut pools[li].0 as *mut B::Buffer,
                    &mut pools[li].1 as *mut B::Buffer,
                )
            };
            // SAFETY: pools allocated-once, see paged_pools field comment.
            let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };

            let mut metal_pos_offsets_host: Option<Vec<u32>> = None;
            if !self.supports_varlen_qkv {
                // Historical Metal path from PR #81: gather per-seq
                // metadata and upload the batched block/context tensors
                // before the per-item QKV writes. CUDA does this once
                // outside the layer loop via
                // populate_paged_batch_scratch_decode; Metal has no varlen
                // QKV kernel and must keep the per-layer flow. Do not bump
                // cache.len here; decode_batch_internal does one central
                // post-forward bump so all layers see the same pre-step
                // position.
                let mut stacked_bt: Vec<u32> = vec![0u32; m * max_blocks_per_seq];
                let mut stacked_cl: Vec<u32> = vec![0u32; m];
                let mut pos_offsets_host: Vec<u32> = vec![0u32; m];
                for (i, (cache_id, _, _)) in batch.iter().enumerate() {
                    let caches = self
                        .kv_caches
                        .get_mut(cache_id)
                        .expect("paged batched: cache not present");
                    let cache = &mut caches[li];
                    pos_offsets_host[i] = cache.len as u32;
                    let blocks = &cache.paged_block_indices;
                    let n_to_copy = blocks.len().min(max_blocks_per_seq);
                    stacked_bt[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
                        .copy_from_slice(&blocks[..n_to_copy]);
                    stacked_cl[i] = (cache.len + 1) as u32;
                }

                let bt_buf = self
                    .scratch
                    .paged_batch_block_tables
                    .as_mut()
                    .expect("paged_batch_block_tables missing");
                B::write_typed::<u32>(ctx, bt_buf, &stacked_bt);
                let cl_buf = self
                    .scratch
                    .paged_batch_context_lens
                    .as_mut()
                    .expect("paged_batch_context_lens missing");
                B::write_typed::<u32>(ctx, cl_buf, &stacked_cl);
                metal_pos_offsets_host = Some(pos_offsets_host);
            }

            // Step 1: write K/V into the shared pool + RoPE'd Q into
            // paged_batch_q at offset i × q_dim. Two code paths:
            //
            //   - CUDA (`self.supports_varlen_qkv == true`): ONE batched
            //     `split_qkv_norm_rope_into_paged_cache_varlen` dispatch
            //     — saves (m-1) × launch_overhead per layer × num_layers.
            //   - Metal (no varlen kernel — would panic): per-item loop
            //     of `split_qkv_norm_rope_into_paged_cache` with
            //     `qkv_byte_offset = i * qkv_stride * 2` (FP16). Mirrors
            //     the pattern in `llama_family_forward_batched.rs:182`.
            //     Each call is m=1 so loses the (m-1)x batched-launch
            //     amortization, but Metal's per-item kernel is what
            //     the historical PR #81 bench at c=16 = 79 tok/s used.
            let q_buf_ptr_raw = self.scratch.paged_batch_q.as_mut().unwrap() as *mut B::Buffer;
            // SAFETY: scratch buffers are independent of qkv_out / norm
            // weights / rope and are not re-borrowed by the called fn.
            let q_buf_safe: &mut B::Buffer = unsafe { &mut *q_buf_ptr_raw };

            if self.supports_varlen_qkv {
                let bt_ptr_raw =
                    self.scratch.paged_batch_block_tables.as_ref().unwrap() as *const B::Buffer;
                let pos_ptr_raw =
                    self.scratch.paged_batch_pos_offsets.as_ref().unwrap() as *const B::Buffer;
                let cu_ptr_raw =
                    self.scratch.paged_batch_cu_seqlens_q.as_ref().unwrap() as *const B::Buffer;
                let bt_safe: &B::Buffer = unsafe { &*bt_ptr_raw };
                let pos_safe: &B::Buffer = unsafe { &*pos_ptr_raw };
                let cu_safe: &B::Buffer = unsafe { &*cu_ptr_raw };
                let varlen_dispatch = if self.use_vllm_paged_attn {
                    B::split_qkv_norm_rope_into_paged_cache_varlen_vllm
                } else {
                    B::split_qkv_norm_rope_into_paged_cache_varlen
                };
                varlen_dispatch(
                    ctx,
                    &self.scratch.qkv_out,
                    q_norm_w,
                    k_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    q_buf_safe,
                    pool_k,
                    pool_v,
                    cu_safe,
                    pos_safe,
                    bt_safe,
                    m, // num_seqs
                    m, // m_total — q_len=1 each, so m_total = m
                    nh,
                    nkv,
                    hd,
                    eps,
                    qk_mode,
                    block_size,
                    max_blocks_per_seq,
                )
                .expect("split_qkv_norm_rope_into_paged_cache_varlen (batched)");
            } else {
                // Per-item fallback. Match LlamaFamily's paged batched
                // decode and derive byte strides from the backend
                // activation element size; qkv_out/paged_batch_q layouts
                // can be f16 or f32 depending on backend/model path.
                let elem_size = B::activation_elem_size_bytes();
                let qkv_stride_bytes = (qkv_stride * elem_size) as u64;
                let q_head_major_size_bytes = (q_dim * elem_size) as u64;
                let pos_offsets_host = metal_pos_offsets_host
                    .as_ref()
                    .expect("Metal per-item fallback pos_offsets missing");
                for (i, (cache_id, _, _)) in batch.iter().enumerate() {
                    let caches = self
                        .kv_caches
                        .get(cache_id)
                        .expect("paged batched: cache not present (per-item fallback)");
                    let cache = &caches[li];
                    let bt = cache
                        .block_table
                        .as_ref()
                        .expect("paged batched: cache.block_table missing");
                    let pos_i = pos_offsets_host[i] as usize;
                    let bt_raw = bt as *const B::Buffer;
                    // SAFETY: bt is read-only in the dispatch; we don't
                    // mutate self.kv_caches between this raw deref and
                    // the call.
                    let bt_safe: &B::Buffer = unsafe { &*bt_raw };
                    B::split_qkv_norm_rope_into_paged_cache(
                        ctx,
                        &self.scratch.qkv_out,
                        (i as u64) * qkv_stride_bytes,
                        q_norm_w,
                        k_norm_w,
                        &self.rope.cos,
                        &self.rope.sin,
                        q_buf_safe,
                        (i as u64) * q_head_major_size_bytes,
                        pool_k,
                        pool_v,
                        bt_safe,
                        1, // tokens (one per seq for decode)
                        nh,
                        nkv,
                        hd,
                        pos_i,
                        eps,
                        qk_mode,
                        pos_i,
                        block_size,
                        max_blocks_per_seq,
                    )
                    .expect("split_qkv_norm_rope_into_paged_cache (per-item fallback)");
                }
            }

            // Step 3: one batched paged_decode_attention(num_seqs=m).
            let bt_ptr =
                self.scratch.paged_batch_block_tables.as_ref().unwrap() as *const B::Buffer;
            let cl_ptr =
                self.scratch.paged_batch_context_lens.as_ref().unwrap() as *const B::Buffer;
            let q_ptr = self.scratch.paged_batch_q.as_ref().unwrap() as *const B::Buffer;
            let o_ptr = self.scratch.paged_batch_o.as_mut().unwrap() as *mut B::Buffer;
            // SAFETY: scratch buffers are not aliased; we hold &mut self
            // through this entire block.
            let bt_safe = unsafe { &*bt_ptr };
            let cl_safe = unsafe { &*cl_ptr };
            let q_safe = unsafe { &*q_ptr };
            let o_safe = unsafe { &mut *o_ptr };
            if self.use_vllm_paged_attn {
                // Compute max_kv_len across the batch (post-bump, so layer-0
                // cache.len is the current kv_len). v2 needs this to size
                // the partition reduction.
                let max_kv_len = batch
                    .iter()
                    .map(|(cid, _, _)| {
                        self.kv_caches
                            .get(cid)
                            .and_then(|cs| cs.first())
                            .map(|c| c.len)
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0);
                B::paged_decode_attention_v2(
                    ctx,
                    q_safe,
                    pool_k,
                    pool_v,
                    o_safe,
                    bt_safe,
                    cl_safe,
                    m,
                    nh,
                    nkv,
                    hd,
                    block_size,
                    max_blocks_per_seq,
                    max_kv_len,
                )
                .expect("paged_decode_attention_v2 (batched)");
            } else {
                B::paged_decode_attention(
                    ctx,
                    q_safe,
                    pool_k,
                    pool_v,
                    o_safe,
                    bt_safe,
                    cl_safe,
                    m,
                    nh,
                    nkv,
                    hd,
                    block_size,
                    max_blocks_per_seq,
                    1, // q_len
                )
                .expect("paged batched decode");
            }

            // Step 4: ONE batched copy paged_batch_o[0..m*q_dim] →
            // attn_flat[0..m*q_dim]. Layouts match (both head-major,
            // contiguous m × q_dim), so a single copy replaces the m
            // per-item launches.
            B::copy_slice(
                ctx,
                self.scratch.paged_batch_o.as_ref().unwrap(),
                0,
                &mut self.scratch.attn_flat,
                0,
                m * q_dim,
            );

            stage_end(attn_t0, ctx, &BD_ATTN_PERITEM_US);
        } else {
            // 3. split_qkv [M, QKV] → q_buf [M, Q], k_buf [M, KV], v_buf [M, KV]
            B::split_qkv(
                ctx,
                &self.scratch.qkv_out,
                &mut self.scratch.q_buf,
                &mut self.scratch.k_buf,
                &mut self.scratch.v_buf,
                m,
                q_dim,
                kv_dim,
            );

            // 4-6. Per-item loop: rope + kv_append + attention.
            //      Each item has its own cache_id + pos + kv_len.
            let q_single = self
                .scratch
                .q_single
                .as_ref()
                .expect("q_single missing — enable_batched_decode_scratch not called")
                as *const B::Buffer;
            let k_single =
                self.scratch.k_single.as_ref().expect("k_single missing") as *const B::Buffer;
            let v_single =
                self.scratch.v_single.as_ref().expect("v_single missing") as *const B::Buffer;
            let q_hm_single =
                self.scratch
                    .q_head_major_single
                    .as_mut()
                    .expect("q_head_major_single missing") as *mut B::Buffer;
            let k_hm_single =
                self.scratch
                    .k_head_major_single
                    .as_mut()
                    .expect("k_head_major_single missing") as *mut B::Buffer;
            let v_hm_single =
                self.scratch
                    .v_head_major_single
                    .as_mut()
                    .expect("v_head_major_single missing") as *mut B::Buffer;
            let attn_hm_single =
                self.scratch
                    .attn_head_major_single
                    .as_mut()
                    .expect("attn_head_major_single missing") as *mut B::Buffer;
            // SAFETY: each Option holds a stable B::Buffer; we don't mutate
            // self.scratch in a way that would invalidate them inside the loop
            // (the kv_caches mutation is on a disjoint field).

            // End of dense block (rms_norm + qkv_proj + split_qkv); start
            // per-item attention loop instrumentation.
            stage_end(dense_t0, ctx, &BD_DENSE_US);
            let attn_t0 = stage_t0();

            for (i, (cache_id, _token, pos)) in batch.iter().enumerate() {
                let pos_i = *pos as usize;

                // SAFETY: borrows of disjoint scratch fields, see above.
                let q_single_ref = unsafe { &*q_single };
                let k_single_ref = unsafe { &*k_single };
                let v_single_ref = unsafe { &*v_single };
                let q_hm_single_mut = unsafe { &mut *q_hm_single };
                let k_hm_single_mut = unsafe { &mut *k_hm_single };
                let v_hm_single_mut = unsafe { &mut *v_hm_single };
                let attn_hm_single_mut = unsafe { &mut *attn_hm_single };

                // Extract item i's Q/K/V slice from the batched buffers.
                B::copy_slice(
                    ctx,
                    &self.scratch.q_buf,
                    i * q_dim,
                    // copy_slice signature wants &mut for dst, but q_single
                    // is shared; we need a *mut variant — since enable_*
                    // gives us Option, we can do as_mut() here.
                    self.scratch.q_single.as_mut().unwrap(),
                    0,
                    q_dim,
                );
                B::copy_slice(
                    ctx,
                    &self.scratch.k_buf,
                    i * kv_dim,
                    self.scratch.k_single.as_mut().unwrap(),
                    0,
                    kv_dim,
                );
                B::copy_slice(
                    ctx,
                    &self.scratch.v_buf,
                    i * kv_dim,
                    self.scratch.v_single.as_mut().unwrap(),
                    0,
                    kv_dim,
                );

                // qk_norm_rope with tokens=1, per-item pos.
                B::qk_norm_rope(
                    ctx,
                    q_single_ref,
                    q_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    q_hm_single_mut,
                    1,
                    nh,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    k_single_ref,
                    k_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    k_hm_single_mut,
                    1,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    v_single_ref,
                    dummy_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    v_hm_single_mut,
                    1,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    0,
                );

                // KV append + attention for item i's cache.
                let caches = self
                    .kv_caches
                    .get_mut(cache_id)
                    .expect("ensure_kv must be called before forward_layer_batched");
                let cache = &mut caches[li];
                B::kv_cache_append_head_major(
                    ctx,
                    &mut cache.k,
                    &mut cache.v,
                    cache.len,
                    cache.capacity,
                    k_hm_single_mut,
                    v_hm_single_mut,
                    1,
                    nkv,
                    hd,
                );
                cache.len += 1;
                let kv_len = cache.len;
                let kv_stride = cache.capacity;

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
                    q_hm_single_mut,
                    &cache.k,
                    &cache.v,
                    attn_hm_single_mut,
                    1,
                    1,
                    kv_len,
                    pos_i,
                    &attn_cfg,
                );

                // Untranspose head-major → token-major: for tokens=1 the
                // layouts are byte-identical, so copy_slice straight into
                // attn_flat at the per-item offset (saves a transpose).
                B::copy_slice(
                    ctx,
                    attn_hm_single_mut,
                    0,
                    &mut self.scratch.attn_flat,
                    i * q_dim,
                    q_dim,
                );
            }

            // End of per-item attention loop.
            stage_end(attn_t0, ctx, &BD_ATTN_PERITEM_US);
        } // end of `else` for non-paged path

        let post_attn_t0 = stage_t0();

        // 7. o_proj GEMM at m=M: attn_flat [M, Q] → o_proj_out [M, H]
        attn_layer.o_proj.forward(
            ctx,
            &self.scratch.attn_flat,
            &mut self.scratch.o_proj_out,
            m,
        );

        // 8. fused residual_add + post_attention_layernorm
        B::fused_add_rms_norm(
            ctx,
            residual,
            &self.scratch.o_proj_out,
            &attn_layer.post_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m,
            h,
        );

        // o_proj + post-norm count under DENSE.
        stage_end(post_attn_t0, ctx, &BD_DENSE_US);
        let moe_t0 = stage_t0();

        // 9. Router gemv: norm_out [M, H] → router_logits [M, n_exp]
        let moe_layer = &self.moe_layers[li];
        moe_layer.router.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.router_logits,
            m,
        );

        // 10. MoE expert dispatch — per-item loop using the cheap
        //     stacked decode kernels (gemv_quant_moe_id + silu_mul_stacked
        //     + weighted_sum_batched). NOT the batched prefill path:
        //     `moe_forward_batched_prefill_impl` is tuned for large M
        //     (prefill) and the GPU bucketing overhead
        //     (`compute_ids_tpe_gpu` + indirect-dispatch arg-buffer
        //     setup) costs more than M sequential gemv calls at small M.
        //
        // Strategy: route ALL M tokens once via batched
        // `route_topk_softmax`, then loop M iterations of the stacked
        // decode kernels. Each iteration:
        //   - extract item i's selected ids + weights from the M-batch
        //     buffers via copy_slice
        //   - copy norm_out[i*h..(i+1)*h] → x_single
        //   - 3× gemv_quant_moe_id (gate/up/down) reading from x_single
        //   - silu_mul_stacked
        //   - weighted_sum_batched(batch=1) → acc_buf  (fresh write,
        //     no residual fusion)
        //   - copy_slice acc_buf → moe_out[i*h..(i+1)*h]
        // After the loop, single add_inplace residual += moe_out [M, H].
        let stacked_path_available = moe_layer.experts.gate_stacked.is_some()
            && moe_layer.experts.up_stacked.is_some()
            && moe_layer.experts.down_stacked.is_some();
        // MoE FFN dispatch tiers (m = batch size of this layer call):
        //
        //   m = 1          : `moe_forward_stacked_decode_impl`
        //                    (decode m=1 fast path, fused gate+up+silu)
        //   m ≥ 8 (default): `moe_forward_batched_prefill_impl`
        //                    (GEMM with simdgroup_matmul + GPU bucketing)
        //   else (m=2..7)  : per-item stacked decode loop
        //
        // EXPERIMENTAL — opt-in `FERRUM_MOE_BATCHED_DECODE=1` engages the
        // new `moe_forward_batched_decode_impl` for 2 ≤ m < 32. The
        // kernel itself is bitwise correct and ports llama.cpp's
        // `kernel_mul_mv_id` strategy to ferrum (one indirect-dispatch
        // GEMV per linear covering all m*top_k pairs). Empirically OFF
        // by default because the existing `forward_layer_batched_decode`
        // attention plumbing (per-item copy_slice × m × 6 dispatches)
        // scales linearly with m and overshadows the FFN savings —
        // regression measured at -19% (c=4) and -36% (c=16) on
        // Qwen3-30B-A3B Q4_K_M / M1 Max. Closing that gap requires a
        // batched attention path with offset-aware QKV slicing, which
        // is the next PR's job. Until then the kernel sits as
        // infrastructure.
        // Two independent thresholds:
        //   * `FERRUM_MOE_BATCH_THRESHOLD` (default 4) — m above which
        //     the LEGACY non-experimental path uses the prefill GEMM.
        //     Shared with `decode_batch`'s engine-level gate, so users
        //     who set it to a small value to engage batched decode
        //     don't accidentally also push the inner FFN to GEMM.
        //   * `FERRUM_MOE_PREFILL_THRESHOLD` (default 32) — m above
        //     which the EXPERIMENTAL batched-decode path defers to the
        //     prefill GEMM path. Mirrors llama.cpp's `ne21_mm_id_min=32`
        //     GEMV→GEMM boundary.
        let legacy_prefill_threshold = self.runtime_env.moe_batch_threshold;
        let new_prefill_threshold = self.runtime_env.moe_prefill_threshold;
        // 0.7.2: default to ON when paged-KV is also on (which is now
        // the default for Metal). The historical regression for this
        // flag (-19% c=4 / -36% c=16) was measured in the pre-paged-KV
        // world where `forward_layer_batched_decode`'s per-item
        // copy_slice × m × 6 attention dispatches cost more than the
        // batched MoE FFN saved. Once paged-KV is on, attention runs as
        // one `paged_decode_attention(num_seqs=m)` dispatch, the
        // plumbing cost drops, and the batched MoE GEMV's win net out
        // to ~+50% at c=16. `FERRUM_MOE_BATCHED_DECODE=0` forces off.
        let new_batched_default = stacked_path_available && self.supports_batched_moe_gemv;
        let new_batched_enabled =
            new_batched_default && self.runtime_env.moe_batched_decode_enabled;

        // When the new path is opted in, it owns the m=2..new_prefill_threshold
        // range; the legacy threshold is overridden upward.
        let use_prefill_batched = if new_batched_enabled {
            stacked_path_available && m >= new_prefill_threshold
        } else {
            stacked_path_available && m >= legacy_prefill_threshold
        };
        let use_batched_decode = new_batched_enabled && !use_prefill_batched && m >= 2;

        if use_prefill_batched {
            crate::moe::forward::moe_forward_batched_prefill_impl::<B>(
                ctx,
                moe_layer,
                &mut self.scratch,
                h,
                self.cfg.expert_intermediate_size,
                self.cfg.num_experts_per_tok,
                self.cfg.num_experts,
                self.cfg.norm_topk_prob,
                m,
            )?;
        } else if use_batched_decode {
            crate::moe::forward::moe_forward_batched_decode_impl::<B>(
                ctx,
                moe_layer,
                &mut self.scratch,
                h,
                self.cfg.expert_intermediate_size,
                self.cfg.num_experts_per_tok,
                self.cfg.num_experts,
                self.cfg.norm_topk_prob,
                m,
            )?;
        } else if stacked_path_available {
            let inter = self.cfg.expert_intermediate_size;
            let top_k = self.cfg.num_experts_per_tok;
            let n_exp = self.cfg.num_experts;
            let norm_topk_prob = self.cfg.norm_topk_prob;

            // Single batched router pass: writes selected_ids_buf [M, top_k]
            // and weights_2d [M, top_k]. Replaces M individual route calls.
            B::route_topk_softmax(
                ctx,
                &self.scratch.router_logits,
                &mut self.scratch.selected_ids_buf,
                &mut self.scratch.weights_2d,
                m,
                n_exp,
                top_k,
                norm_topk_prob,
            )?;

            // Per-item loop using offset-aware kernel APIs — eliminates
            // the 4 copy_slice round-trips per iteration that the
            // earlier implementation needed (ids, weights, x_single,
            // moe_out). At c=16 / 48 layers that's ~3,072 dispatches
            // saved per token. Uses `gemv_*_offset` to read
            // `selected_ids_buf` at the i-th `top_k` block and
            // `norm_out` at the i-th hidden row directly. Falls back
            // to copy_slice path if backend doesn't support offsets.
            for i in 0..m {
                let ids_offset = i * top_k;
                let activation_offset = i * h;
                let weights_offset = i * top_k;
                let moe_out_offset = i * h;

                // Stacked gate / up gemvs — broadcast item i's row of
                // norm_out across top_k slots, read item i's ids.
                let gate_res = moe_layer.experts.gemv_gate_offset(
                    ctx,
                    &self.scratch.norm_out,
                    activation_offset,
                    &self.scratch.selected_ids_buf,
                    ids_offset,
                    &mut self.scratch.gate_out_stacked,
                    top_k,
                    0,
                );
                if gate_res.is_err() {
                    // Backend doesn't support offset variants — fall back
                    // to the legacy copy_slice path. Same as before.
                    B::copy_slice(
                        ctx,
                        &self.scratch.selected_ids_buf,
                        ids_offset,
                        &mut self.scratch.ids_buf,
                        0,
                        top_k,
                    );
                    B::copy_slice(
                        ctx,
                        &self.scratch.weights_2d,
                        weights_offset,
                        &mut self.scratch.weights_buf,
                        0,
                        top_k,
                    );
                    B::copy_slice(
                        ctx,
                        &self.scratch.norm_out,
                        activation_offset,
                        &mut self.scratch.x_single,
                        0,
                        h,
                    );
                    moe_layer.experts.gemv_gate(
                        ctx,
                        &self.scratch.x_single,
                        &self.scratch.ids_buf,
                        &mut self.scratch.gate_out_stacked,
                        top_k,
                    )?;
                    moe_layer.experts.gemv_up(
                        ctx,
                        &self.scratch.x_single,
                        &self.scratch.ids_buf,
                        &mut self.scratch.up_out_stacked,
                        top_k,
                    )?;
                    B::silu_mul_stacked(
                        ctx,
                        &self.scratch.gate_out_stacked,
                        &self.scratch.up_out_stacked,
                        &mut self.scratch.silu_stacked,
                        top_k,
                        inter,
                    )?;
                    moe_layer.experts.gemv_down(
                        ctx,
                        &self.scratch.silu_stacked,
                        &self.scratch.ids_buf,
                        &mut self.scratch.down_out_stacked,
                        top_k,
                        inter,
                    )?;
                    B::weighted_sum_batched(
                        ctx,
                        &self.scratch.down_out_stacked,
                        &self.scratch.weights_buf,
                        &mut self.scratch.acc_buf,
                        1,
                        top_k,
                        h,
                    )?;
                    B::copy_slice(
                        ctx,
                        &self.scratch.acc_buf,
                        0,
                        &mut self.scratch.moe_out,
                        moe_out_offset,
                        h,
                    );
                    continue;
                }
                // Fast path: offset-aware all the way through.
                moe_layer.experts.gemv_up_offset(
                    ctx,
                    &self.scratch.norm_out,
                    activation_offset,
                    &self.scratch.selected_ids_buf,
                    ids_offset,
                    &mut self.scratch.up_out_stacked,
                    top_k,
                    0,
                )?;
                B::silu_mul_stacked(
                    ctx,
                    &self.scratch.gate_out_stacked,
                    &self.scratch.up_out_stacked,
                    &mut self.scratch.silu_stacked,
                    top_k,
                    inter,
                )?;
                moe_layer.experts.gemv_down_offset(
                    ctx,
                    &self.scratch.silu_stacked,
                    0, // silu_stacked itself stays at offset 0 each iter
                    &self.scratch.selected_ids_buf,
                    ids_offset,
                    &mut self.scratch.down_out_stacked,
                    top_k,
                    inter,
                )?;
                // Write directly into moe_out at the per-item offset —
                // skips the copy_slice from acc_buf.
                B::weighted_sum_batched_offset(
                    ctx,
                    &self.scratch.down_out_stacked,
                    &self.scratch.weights_2d,
                    weights_offset,
                    &mut self.scratch.moe_out,
                    moe_out_offset,
                    1,
                    top_k,
                    h,
                )?;
            }
        } else if moe_layer.experts.gate_up_marlin_stack.is_some()
            && moe_layer.experts.down_marlin_stack.is_some()
            && self.runtime_env.moe_bucketed
        {
            // CUDA Marlin bucketed path (decode_batch m ≥ 1 entry).
            crate::moe::moe_forward_bucketed::<B>(
                crate::moe::dispatch::MoeForwardBucketedParams {
                    ctx,
                    x: &self.scratch.norm_out,
                    router_logits: &self.scratch.router_logits,
                    out: &mut self.scratch.moe_out,
                    batch: m,
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
        } else {
            // Backend without stacked variants — fall back to the legacy
            // per-(token, expert) host-routed path.
            moe_forward::<B>(crate::moe::dispatch::MoeForwardParams {
                ctx,
                x: &self.scratch.norm_out,
                router_logits: &self.scratch.router_logits,
                out: &mut self.scratch.moe_out,
                batch: m,
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

        // 11. residual += moe_out [M, H]
        B::add_inplace(ctx, residual, &self.scratch.moe_out, m * h);

        // Close MoE-block instrumentation (router + FFN + residual add).
        stage_end(moe_t0, ctx, &BD_MOE_US);

        Ok(())
    }
}
