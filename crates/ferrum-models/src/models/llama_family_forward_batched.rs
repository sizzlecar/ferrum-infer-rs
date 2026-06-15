//! Batched-decode forward methods for `LlamaFamilyModel`.
//!
//! Split out of `llama_family.rs` to keep that file under 3000 lines.
//! These three methods are the heaviest forward paths and reference
//! many private fields on the parent struct, so they live in a separate
//! `impl` block in a peer file (Rust allows multiple `impl` blocks for
//! the same type across the crate).

use std::sync::atomic::Ordering;

use ferrum_interfaces::{
    kv_dtype::KvFp16,
    model_executor::{LogitsReturnPolicy, TokenSelectionMask},
};
use ferrum_kernels::backend::{
    Backend, BackendGraph, BackendMoeFused, BackendPagedKv, BackendQuantGguf, BackendQuantMarlin,
    KvCache, MoeLlmBackend, MAX_LAYERS_FOR_GRAPH,
};
use ferrum_quantization::Linear;
use ferrum_types::Result;

use super::llama_family::{
    elapsed_micros_u64_floor1, LlamaFamilyModel, LlamaStageHiddenBridgeTiming, ATTN_CALLS,
    ATTN_TIME_US, BATCHED_GRAPH_EAGER_COUNT, BATCHED_GRAPH_REPLAY_COUNT, MATMUL_CALLS,
    MATMUL_TIME_US, NORM_CALLS, NORM_TIME_US, OTHER_CALLS, OTHER_TIME_US, QKR_CALLS, QKR_TIME_US,
    SINGLE_ITEM_GRAPH_KEY,
};
use super::llama_family_pipeline::{LlamaPipelineStageBatchOps, PipelineHidden};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LlamaBatchedRuntimeConfig {
    decode_op_profile: bool,
    unified_graph: bool,
    unified_profile: bool,
    batched_graph: bool,
    batched_trace: bool,
    greedy_argmax: bool,
}

#[derive(Debug, Clone, Copy)]
enum DecodeLogitsReturn<'a> {
    Full,
    LegacyDefault,
    GreedyArgmax {
        token_mask: Option<&'a TokenSelectionMask>,
    },
}

#[derive(Debug, Clone, Copy)]
enum ArgmaxMode<'a> {
    Raw,
    Masked(&'a TokenSelectionMask),
}

impl LlamaBatchedRuntimeConfig {
    /// Resolve from the process-wide snapshot installed at the composition root
    /// (was a direct `std::env::vars()` read). The model holds this in a
    /// `batched_cfg` field, resolved once at construction.
    pub(crate) fn from_env() -> Self {
        Self::from_runtime_config_snapshot(&ferrum_types::active_runtime_snapshot())
    }

    fn from_runtime_config_snapshot(snapshot: &ferrum_types::RuntimeConfigSnapshot) -> Self {
        Self::from_env_vars(
            snapshot
                .entries
                .iter()
                .map(|e| (e.key.as_str(), e.effective_value.as_str())),
        )
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self {
            decode_op_profile: false,
            unified_graph: false,
            unified_profile: false,
            batched_graph: false,
            batched_trace: false,
            greedy_argmax: false,
        };
        for (name, value) in vars {
            let value = value.as_ref();
            match name.as_ref() {
                "FERRUM_DECODE_OP_PROFILE" => config.decode_op_profile = true,
                "FERRUM_UNIFIED_GRAPH" => config.unified_graph = value == "1",
                "FERRUM_UNIFIED_PROFILE" => config.unified_profile = true,
                "FERRUM_BATCHED_GRAPH" => config.batched_graph = value != "0",
                "FERRUM_BATCHED_TRACE" => config.batched_trace = true,
                "FERRUM_GREEDY_ARGMAX" => config.greedy_argmax = value == "1",
                _ => {}
            }
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaBatchedRuntimeConfig;

    #[test]
    fn llama_batched_runtime_config_parses_startup_knobs() {
        let config = LlamaBatchedRuntimeConfig::from_env_vars([
            ("FERRUM_DECODE_OP_PROFILE", "0"),
            ("FERRUM_UNIFIED_GRAPH", "1"),
            ("FERRUM_UNIFIED_PROFILE", ""),
            ("FERRUM_BATCHED_GRAPH", "1"),
            ("FERRUM_BATCHED_TRACE", "0"),
            ("FERRUM_GREEDY_ARGMAX", "1"),
        ]);

        assert!(config.decode_op_profile);
        assert!(config.unified_graph);
        assert!(config.unified_profile);
        assert!(config.batched_graph);
        assert!(config.batched_trace);
        assert!(config.greedy_argmax);
    }

    #[test]
    fn llama_batched_runtime_config_preserves_opt_out_values() {
        let config = LlamaBatchedRuntimeConfig::from_env_vars([
            ("FERRUM_UNIFIED_GRAPH", "true"),
            ("FERRUM_BATCHED_GRAPH", "0"),
            ("FERRUM_GREEDY_ARGMAX", "true"),
        ]);

        assert!(!config.decode_op_profile);
        assert!(!config.unified_graph);
        assert!(!config.unified_profile);
        assert!(!config.batched_graph);
        assert!(!config.batched_trace);
        assert!(!config.greedy_argmax);
    }
}

// Batched / unified-forward paths are FP16-only. Pinning the impl to
// K = KvFp16 lets us access `KvCache<B, KvFp16>` fields directly without
// trait-method indirection. K = KvInt8 falls back to per-item decode at
// the engine level (DecoderOnlyLLM::decode_batch + unified_forward report
// Unsupported in the K=KvInt8 specialization).
impl<B: MoeLlmBackend> LlamaFamilyModel<B, KvFp16> {
    fn prepare_batched_decode_stage(
        &mut self,
        batch: &[(String, u32, u32)],
    ) -> (usize, usize, B::Context) {
        let m = batch.len();
        debug_assert!(m > 0);
        for (cid, _, _) in batch {
            self.ensure_kv(cid);
        }
        self.ensure_scratch(m);

        let mut ctx = B::new_context();
        let positions: Vec<u32> = batch.iter().map(|(_, _, p)| *p).collect();
        let kv_pre: Vec<u32> = batch
            .iter()
            .map(|(cid, _, _)| self.kv_caches.get(cid).expect("kv_caches missing")[0].len as u32)
            .collect();
        let kv_post: Vec<u32> = kv_pre.iter().map(|&x| x + 1).collect();
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_positions, &positions);
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_kv_lens_pre, &kv_pre);
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_kv_lens_post, &kv_post);

        (m, self.cfg.hidden_size, ctx)
    }

    fn bump_local_batched_decode_kv_lengths(&mut self, batch: &[(String, u32, u32)]) {
        let local_layers = self.local_layer_count();
        for (cid, _, _) in batch {
            let caches = self.kv_caches.get_mut(cid).expect("kv_caches missing");
            for cache in caches.iter_mut().take(local_layers) {
                cache.len += 1;
            }
        }
    }

    /// One transformer layer over M items, GEMMs batched + per-item attention.
    pub(crate) fn forward_layer_batched_decode(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        batch: &[(String, u32, u32)],
        residual: &mut B::Buffer,
        m: usize,
        host_residual: Option<&mut [f32]>,
        device_residual: Option<&mut B::Buffer>,
        device_branch: Option<&mut B::Buffer>,
    ) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        let source_li = self.source_layer_index(li);
        let layer = &self.layers[li];
        let qk_mode: i32 = if cfg.has_qk_norm {
            1
        } else if cfg.rope_interleaved {
            3
        } else {
            2
        };
        let dummy_w = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy_w);

        // Match the single-sequence Gemma3 attention schedule. Local layers
        // use the local RoPE table and a sliding window; global layers use
        // the main RoPE table and full causal attention.
        let pattern = cfg.sliding_window_pattern;
        let is_global_layer = pattern == 0 || (source_li + 1) % pattern == 0;
        let layer_window = if pattern == 0 {
            cfg.sliding_window
        } else if is_global_layer {
            0
        } else {
            cfg.sliding_window
        };
        let (rope_cos, rope_sin) = match (&self.rope_local, is_global_layer) {
            (Some(local), false) => (&local.cos, &local.sin),
            _ => (&self.rope.cos, &self.rope.sin),
        };

        let _bp = self.batched_cfg.decode_op_profile;

        // 1. rms_norm [M, H]  → norm_out
        let _t = if _bp {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        if let Some(host) = host_residual.as_deref() {
            Self::rms_norm_host_to_activation(
                ctx,
                host,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
        } else if let Some(device) = device_residual.as_ref() {
            B::rms_norm_f32_to_activation(
                ctx,
                &**device,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
        } else {
            B::rms_norm(
                ctx,
                residual,
                &layer.input_ln_w,
                eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
        }
        if let Some(t0) = _t {
            B::sync(ctx);
            NORM_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 2. qkv_proj (GEMM m=M): norm_out [M, H] → qkv_out [M, QKV]
        let _t = if _bp {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.batched_layer.qkv_proj");
            layer
                .qkv_proj
                .forward(ctx, &self.scratch.norm_out, &mut self.scratch.qkv_out, m);
        }
        if let Some(t0) = _t {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // ── Paged-KV batched path (Phase 4b) ──────────────────────────
        // When paged is on, we skip the contig split_qkv + per-item
        // qk_norm_rope + kv_append + flash_attention loop entirely.
        // Instead:
        //   1. Per item: split_qkv_norm_rope_into_paged_cache with
        //      qkv_byte_offset = i * qkv_stride * 4 reads item i's
        //      slice of qkv_out, writes K/V into the shared pool at
        //      its block_table-resolved position, and stores the
        //      RoPE'd Q at paged_batch_q[i * q_dim .. (i+1) * q_dim].
        //   2. Build batched block_tables [M, max_blocks_per_seq] +
        //      context_lens [M] host-side, write to scratch device
        //      buffers.
        //   3. Single paged_decode_attention(num_seqs=M) reads all M
        //      seqs' K/V via per-seq block_tables, writes to
        //      paged_batch_o.
        //   4. Per item: copy paged_batch_o[i] → attn_flat[i * q_dim].
        //
        // This is the "real" multi-seq decode — one heavy attention
        // dispatch covering all sequences instead of M sequential ones.
        if let Some(pools) = self.paged_pools.as_mut() {
            let pool_ptr = (
                &mut pools[li].0 as *mut B::Buffer,
                &mut pools[li].1 as *mut B::Buffer,
            );
            // SAFETY: pools allocated once; not concurrently mutated.
            let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };

            let qkv_stride = q_dim + 2 * kv_dim;
            let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
            let block_size = 16; // matches PAGED_BLOCK_SIZE in ensure_kv

            // Step 1: per-item paged write. We collect cache_len + block_indices
            // up front for step 2. Note: this loop borrows self.kv_caches mutably
            // per iteration, so we extract the batched-write parameters first then
            // do the dispatches.
            let mut item_state: Vec<(u32, Vec<u32>)> = Vec::with_capacity(m);
            for (cache_id, _, _) in batch.iter() {
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("ensure_kv must be called before forward_layer_batched");
                let cache = &caches[li];
                item_state.push((cache.len as u32, cache.paged_block_indices.clone()));
            }

            let elem_size = B::activation_elem_size_bytes();
            let q_head_major_size_bytes = (q_dim * elem_size) as u64;
            let qkv_stride_bytes = (qkv_stride * elem_size) as u64;
            let _t_qkr = if _bp {
                B::sync(ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            for (i, (cache_id, _, pos)) in batch.iter().enumerate() {
                let pos_i = *pos as usize;
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("paged batched: cache not present");
                let cache = &caches[li];
                let bt = cache
                    .block_table
                    .as_ref()
                    .expect("paged batched: block_table missing");
                let cache_len_before = cache.len;
                let block_table_ref = bt as *const B::Buffer;
                // SAFETY: bt is read-only in the dispatch; we don't
                // mutate self.kv_caches between this raw deref and the
                // call.
                let bt_safe: &B::Buffer = unsafe { &*block_table_ref };
                B::split_qkv_norm_rope_into_paged_cache(
                    ctx,
                    &self.scratch.qkv_out,
                    (i as u64) * qkv_stride_bytes,
                    q_norm_w,
                    k_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    self.scratch
                        .paged_batch_q
                        .as_mut()
                        .expect("paged_batch_q missing"),
                    (i as u64) * q_head_major_size_bytes,
                    pool_k,
                    pool_v,
                    bt_safe,
                    1,
                    nh,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                    cache_len_before,
                    block_size,
                    max_blocks_per_seq,
                )
                .expect("paged batched write");
            }
            if let Some(t0) = _t_qkr {
                B::sync(ctx);
                QKR_TIME_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                QKR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            // Step 2: build the stacked block_tables + post-append
            // context_lens host-side, then upload to device scratch. Do not
            // mutate cache.len here: decode_batch_internal does one central
            // bump after all layers so eager and graph replay stay in sync.
            let mut stacked_bt: Vec<u32> = vec![0u32; m * max_blocks_per_seq];
            let mut stacked_cl: Vec<u32> = vec![0u32; m];
            for (i, (cache_id, _, _)) in batch.iter().enumerate() {
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("paged batched: cache not present");
                let cache = &caches[li];
                stacked_cl[i] = (cache.len + 1) as u32;
                let blocks = &cache.paged_block_indices;
                let n_to_copy = blocks.len().min(max_blocks_per_seq);
                stacked_bt[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
                    .copy_from_slice(&blocks[..n_to_copy]);
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

            // Step 3: one batched paged_decode_attention(num_seqs=m).
            let bt_ptr =
                self.scratch.paged_batch_block_tables.as_ref().unwrap() as *const B::Buffer;
            let cl_ptr =
                self.scratch.paged_batch_context_lens.as_ref().unwrap() as *const B::Buffer;
            let q_ptr = self.scratch.paged_batch_q.as_ref().unwrap() as *const B::Buffer;
            let o_ptr = self.scratch.paged_batch_o.as_mut().unwrap() as *mut B::Buffer;
            // SAFETY: the four scratch buffers above are not aliased
            // by anything else; we only deref while &mut self is held.
            let bt_safe = unsafe { &*bt_ptr };
            let cl_safe = unsafe { &*cl_ptr };
            let q_safe = unsafe { &*q_ptr };
            let o_safe = unsafe { &mut *o_ptr };
            let _t = if _bp {
                B::sync(ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
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
            if let Some(t0) = _t {
                B::sync(ctx);
                ATTN_TIME_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            // Step 4: per-item copy paged_batch_o[i] → attn_flat[i * q_dim].
            // Both have q_dim floats per item; same head-major-equals-token-major
            // identity collapse used in the contig path.
            for i in 0..m {
                B::copy_slice(
                    ctx,
                    self.scratch.paged_batch_o.as_ref().unwrap(),
                    i * q_dim,
                    &mut self.scratch.attn_flat,
                    i * q_dim,
                    q_dim,
                );
            }

            // Skip the contig split_qkv + per-item loop below.
            return self.forward_layer_batched_decode_post_attn(
                ctx,
                li,
                residual,
                m,
                host_residual,
                device_residual,
                device_branch,
            );
        }

        // 3. split_qkv [M, QKV] → q_buf [M, Q], k_buf [M, KV], v_buf [M, KV]
        let _t = if _bp {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
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
        if let Some(t0) = _t {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 4. Try the batched per-item qk_norm_rope path: one launch
        //    each for Q/K/V instead of M sequential per-item launches.
        //    Saves 3*(M-1) qk_norm_rope dispatches per layer (and at
        //    M=16 with 32 layers that's ~1500 launches, ~10 ms TPOT).
        //    Backends that don't implement it return Err and we drop
        //    back into the per-item loop unchanged.
        // batch_positions, batch_kv_lens_pre, batch_kv_lens_post are
        // populated by `decode_batch_internal` ONCE per step before the
        // layer loop — required so a captured CUDA graph reads from
        // stable buffer contents on replay (per-layer write_u32 inside
        // the captured region would freeze the values).
        let _t_qkr_contig = if _bp {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        let q_batched = B::qk_norm_rope_batched_per_item(
            ctx,
            &self.scratch.q_buf,
            q_norm_w,
            rope_cos,
            rope_sin,
            &mut self.scratch.q_normed_batched,
            &self.scratch.batch_positions,
            m,
            nh,
            hd,
            eps,
            qk_mode,
        );
        let k_batched = B::qk_norm_rope_batched_per_item(
            ctx,
            &self.scratch.k_buf,
            k_norm_w,
            rope_cos,
            rope_sin,
            &mut self.scratch.k_normed_batched,
            &self.scratch.batch_positions,
            m,
            nkv,
            hd,
            eps,
            qk_mode,
        );
        // V's "qk_norm_rope" runs in mode=0 (transpose-only). For
        // tokens-per-item=1 in batched decode this is a memcpy — kept
        // for layout-equivalence with the per-item path. Cheap.
        let v_batched = B::qk_norm_rope_batched_per_item(
            ctx,
            &self.scratch.v_buf,
            dummy_w,
            rope_cos,
            rope_sin,
            &mut self.scratch.v_normed_batched,
            &self.scratch.batch_positions,
            m,
            nkv,
            hd,
            eps,
            0,
        );
        let use_batched_qkr = q_batched.is_ok() && k_batched.is_ok() && v_batched.is_ok();
        if let Some(t0) = _t_qkr_contig {
            B::sync(ctx);
            QKR_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            QKR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // One-time diagnostic so we can verify in server logs that the
        // batched qkr path is actually being taken (vs. silently falling
        // back to the per-item loop). Prints once total per process.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static REPORTED: AtomicBool = AtomicBool::new(false);
            if !REPORTED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[batched-qkr] first batched_decode call: m={} use_batched_qkr={} (q={:?} k={:?} v={:?})",
                    m,
                    use_batched_qkr,
                    q_batched.as_ref().err().map(|e| e.to_string()),
                    k_batched.as_ref().err().map(|e| e.to_string()),
                    v_batched.as_ref().err().map(|e| e.to_string()),
                );
            }
        }

        // 5. Batched kv_cache_append (when use_batched_qkr is on): one
        //    launch each for K and V replaces M sequential per-item
        //    kv_append calls. Reads k/v_normed_batched directly so the
        //    M K/V copy_slice dispatches into single buffers also go
        //    away. cache_lens captured BEFORE the bump.
        let mut kv_lens_host: Vec<u32> = Vec::with_capacity(m);
        let mut batched_kv_append_ok = false;
        let _t_kvapp = if _bp {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        if use_batched_qkr {
            let mut k_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut v_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut pre_append_lens: Vec<u32> = Vec::with_capacity(m);
            let mut capacity_first: usize = 0;
            for (cache_id, _, _) in batch.iter() {
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("kv_caches must be present");
                let cache = &caches[li];
                k_caches_ref.push(&cache.k);
                v_caches_ref.push(&cache.v);
                pre_append_lens.push(cache.len as u32);
                if capacity_first == 0 {
                    capacity_first = cache.capacity;
                }
            }
            // batch_kv_lens_pre is pre-populated by decode_batch_internal.
            // Per-layer slot for K/V append. cache_ptrs is shared
            // between the K and V calls, so V uses an offset slot
            // (layer_idx + MAX_LAYERS_FOR_GRAPH) to keep its captured
            // memcpy reading from a distinct host region. See
            // ferrum-kernels::backend::cuda for the rationale (graph
            // capture records host POINTERS; same slot → all replays
            // read whichever value was written last).
            let k_append_res = B::kv_cache_append_batched_per_cache(
                ctx,
                &k_caches_ref,
                &self.scratch.k_normed_batched,
                &self.scratch.batch_kv_lens_pre,
                capacity_first,
                m,
                nkv,
                hd,
                li,
            );
            let v_append_res = B::kv_cache_append_batched_per_cache(
                ctx,
                &v_caches_ref,
                &self.scratch.v_normed_batched,
                &self.scratch.batch_kv_lens_pre,
                capacity_first,
                m,
                nkv,
                hd,
                li + MAX_LAYERS_FOR_GRAPH,
            );
            batched_kv_append_ok = k_append_res.is_ok() && v_append_res.is_ok();
            // One-time diag
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static REPORTED_KV: AtomicBool = AtomicBool::new(false);
                if !REPORTED_KV.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[batched-kv-append] first call: m={} ok={} k_err={:?} v_err={:?}",
                        m,
                        batched_kv_append_ok,
                        k_append_res.as_ref().err().map(|e| e.to_string()),
                        v_append_res.as_ref().err().map(|e| e.to_string()),
                    );
                }
            }
            // Note: cache.len bump moved to decode_batch_internal post-forward
            // (so a graph replay doesn't double-bump). No-op here.
        }
        if let Some(t0) = _t_kvapp {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // kv_lens_host no longer used; flash_attn reads
        // scratch.batch_kv_lens_post (also pre-populated).
        let _ = kv_lens_host;

        // 6. Per-item loop: only runs when the batched paths are NOT in
        // effect, or when batched_kv_append failed (Err fallback). Gemma
        // local-window layers still use the batched attention kernel:
        // `layer_window` is passed into flash_attention_batched_per_cache.
        for (i, (cache_id, _token, pos)) in batch.iter().enumerate() {
            if use_batched_qkr && batched_kv_append_ok {
                // Already handled by batched kv_append above. Skip
                // per-item copy + kv_append + per-item flash_attn.
                continue;
            }
            let pos_i = *pos as usize;

            if use_batched_qkr {
                // batched_kv_append fallback: still need per-item
                // copy_slice for K/V into single buffers for the
                // per-item kv_append below.
                if !batched_kv_append_ok {
                    B::copy_slice(
                        ctx,
                        &self.scratch.k_normed_batched,
                        i * kv_dim,
                        &mut self.scratch.k_head_major_single,
                        0,
                        kv_dim,
                    );
                    B::copy_slice(
                        ctx,
                        &self.scratch.v_normed_batched,
                        i * kv_dim,
                        &mut self.scratch.v_head_major_single,
                        0,
                        kv_dim,
                    );
                }
            } else {
                // Fallback: extract item i's Q/K/V then run per-item
                // qk_norm_rope. Same dispatch budget as before this
                // commit — used on backends without the batched kernel.
                B::copy_slice(
                    ctx,
                    &self.scratch.q_buf,
                    i * q_dim,
                    &mut self.scratch.q_single,
                    0,
                    q_dim,
                );
                B::copy_slice(
                    ctx,
                    &self.scratch.k_buf,
                    i * kv_dim,
                    &mut self.scratch.k_single,
                    0,
                    kv_dim,
                );
                B::copy_slice(
                    ctx,
                    &self.scratch.v_buf,
                    i * kv_dim,
                    &mut self.scratch.v_single,
                    0,
                    kv_dim,
                );

                B::qk_norm_rope(
                    ctx,
                    &self.scratch.q_single,
                    q_norm_w,
                    rope_cos,
                    rope_sin,
                    &mut self.scratch.q_head_major_single,
                    1,
                    nh,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.k_single,
                    k_norm_w,
                    rope_cos,
                    rope_sin,
                    &mut self.scratch.k_head_major_single,
                    1,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    qk_mode,
                );
                B::qk_norm_rope(
                    ctx,
                    &self.scratch.v_single,
                    dummy_w,
                    rope_cos,
                    rope_sin,
                    &mut self.scratch.v_head_major_single,
                    1,
                    nkv,
                    hd,
                    pos_i,
                    eps,
                    0,
                );
            }

            // KV append for item i's cache.
            let caches = self
                .kv_caches
                .get_mut(cache_id)
                .expect("ensure_kv must be called before forward_layer_batched");
            let cache = &mut caches[li];
            if !(use_batched_qkr && batched_kv_append_ok) {
                B::kv_cache_append_head_major(
                    ctx,
                    &mut cache.k,
                    &mut cache.v,
                    cache.len,
                    cache.capacity,
                    &self.scratch.k_head_major_single,
                    &self.scratch.v_head_major_single,
                    1,
                    nkv,
                    hd,
                );
            }
            // cache.len bump moved to decode_batch_internal post-forward
            // for graph-replay correctness. flash_attn below uses
            // cache.len + 1 directly.
            let kv_len = cache.len + 1;
            let kv_stride = cache.capacity;
            kv_lens_host.push(kv_len as u32);

            // Per-item flash_attn runs only when the batched qkr fallback is
            // in use. If batched qkr works, the single-launch batched
            // attention below covers both full and local-window layers.
            if !use_batched_qkr {
                let attn_cfg = ferrum_kernels::backend::AttnConfig {
                    num_heads: nh,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    causal: true,
                    scale: 1.0 / (hd as f32).sqrt(),
                    kv_seq_stride: kv_stride,
                    sliding_window: layer_window,
                };
                B::flash_attention(
                    ctx,
                    &self.scratch.q_head_major_single,
                    &cache.k,
                    &cache.v,
                    &mut self.scratch.attn_head_major_single,
                    1,
                    1,
                    kv_len,
                    pos_i,
                    &attn_cfg,
                );
                // For tokens=1 head-major and token-major are
                // byte-identical, so just copy into the per-item slot
                // of attn_flat without a transpose dispatch.
                B::copy_slice(
                    ctx,
                    &self.scratch.attn_head_major_single,
                    0,
                    &mut self.scratch.attn_flat,
                    i * q_dim,
                    q_dim,
                );
            }
        }

        // 7. Batched flash_attention: one launch covers all M items.
        //    Reads q_normed_batched directly (item-major) and writes
        //    output straight into attn_flat at [m, q_dim] item-major.
        //    On Err (backend lacks the batched kernel) we fall through
        //    to a per-item flash_attn loop that mirrors the original
        //    code path.
        if use_batched_qkr {
            let mut k_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut v_caches_ref: Vec<&B::Buffer> = Vec::with_capacity(m);
            let mut max_kv = 0usize;
            let mut capacity_for_kernel = 0usize;
            for (cache_id, _, _) in batch.iter() {
                let caches = self
                    .kv_caches
                    .get(cache_id)
                    .expect("kv_caches must be present");
                let cache = &caches[li];
                k_caches_ref.push(&cache.k);
                v_caches_ref.push(&cache.v);
                // POST-append valid_kv_len: kv_cache_append_batched_per_cache
                // wrote position cache.len, so attention reads cache.len + 1
                // entries. Pre-append cache.len under-sized the shared mem
                // and corrupted s_scores (silent garbage tokens at m≥2).
                let post_len = cache.len + 1;
                if post_len > max_kv {
                    max_kv = post_len;
                }
                if capacity_for_kernel == 0 {
                    capacity_for_kernel = cache.capacity;
                }
            }
            let scale = 1.0 / (hd as f32).sqrt();
            // batch_kv_lens_post pre-populated by decode_batch_internal.
            // flash_attn_batched uses its own k_ptrs/v_ptrs host
            // arrays in CudaState (separate from kv_cache_append's
            // cache_ptrs), so per-layer slot = li is sufficient.
            let _t_attn = if _bp {
                B::sync(ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            let batched_attn_res = B::flash_attention_batched_per_cache(
                ctx,
                &self.scratch.q_normed_batched,
                &k_caches_ref,
                &v_caches_ref,
                &self.scratch.batch_kv_lens_post,
                &mut self.scratch.attn_flat,
                nh,
                nkv,
                hd,
                scale,
                max_kv,
                capacity_for_kernel,
                layer_window,
                li,
            );
            if let Some(t0) = _t_attn {
                B::sync(ctx);
                ATTN_TIME_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // One-time diagnostic
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static REPORTED_ATTN: AtomicBool = AtomicBool::new(false);
                if !REPORTED_ATTN.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[batched-attn] first call: m={} ok={} err={:?}",
                        m,
                        batched_attn_res.is_ok(),
                        batched_attn_res.as_ref().err().map(|e| e.to_string()),
                    );
                }
            }
            if batched_attn_res.is_err() {
                // Per-item flash_attn fallback for backends that
                // implement the batched qkr but not the batched attn.
                for (i, (cache_id, _, pos)) in batch.iter().enumerate() {
                    let pos_i = *pos as usize;
                    // Populate q_head_major_single from the normed Q.
                    B::copy_slice(
                        ctx,
                        &self.scratch.q_normed_batched,
                        i * q_dim,
                        &mut self.scratch.q_head_major_single,
                        0,
                        q_dim,
                    );
                    let caches = self
                        .kv_caches
                        .get(cache_id)
                        .expect("kv_caches must be present");
                    let cache = &caches[li];
                    let kv_stride = cache.capacity;
                    let attn_cfg = ferrum_kernels::backend::AttnConfig {
                        num_heads: nh,
                        num_kv_heads: nkv,
                        head_dim: hd,
                        causal: true,
                        scale: 1.0 / (hd as f32).sqrt(),
                        kv_seq_stride: kv_stride,
                        sliding_window: layer_window,
                    };
                    B::flash_attention(
                        ctx,
                        &self.scratch.q_head_major_single,
                        &cache.k,
                        &cache.v,
                        &mut self.scratch.attn_head_major_single,
                        1,
                        1,
                        cache.len + 1,
                        pos_i,
                        &attn_cfg,
                    );
                    B::copy_slice(
                        ctx,
                        &self.scratch.attn_head_major_single,
                        0,
                        &mut self.scratch.attn_flat,
                        i * q_dim,
                        q_dim,
                    );
                }
            }
        }

        self.forward_layer_batched_decode_post_attn(
            ctx,
            li,
            residual,
            m,
            host_residual,
            device_residual,
            device_branch,
        );
    }

    pub(crate) fn forward_layer_batched_decode_post_attn(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        residual: &mut B::Buffer,
        m: usize,
        host_residual: Option<&mut [f32]>,
        device_residual: Option<&mut B::Buffer>,
        device_branch: Option<&mut B::Buffer>,
    ) {
        let layer = &self.layers[li];
        let _bp = self.batched_cfg.decode_op_profile;

        // 7. o_proj (GEMM m=M): attn_flat [M, Q] → o_proj_out [M, H]
        let _t = if _bp {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        layer.o_proj.forward(
            ctx,
            &self.scratch.attn_flat,
            &mut self.scratch.o_proj_out,
            m,
        );
        if let Some(t0) = _t {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        self.forward_layer_post_o_proj_with_residual_shadow(
            ctx,
            li,
            None,
            residual,
            m,
            host_residual,
            device_residual,
            device_branch,
        );
    }

    /// Unified mixed-batch forward (chunked-prefill workhorse).
    ///
    /// Each item is `(cache_id, q_tokens, pos_offset, is_final_chunk)`:
    /// - `q_tokens.len() == 1` is a decode step
    /// - `q_tokens.len() >= 1` with `pos_offset > 0` is a continuing
    ///   prefill chunk
    /// - `is_final_chunk == true` ⇒ logits returned for sampling, else
    ///   `None` (intermediate prefill chunks just advance KV state)
    ///
    /// Concatenates all q_tokens into a single `[M_total, hidden]`
    /// forward, dispatches per-item `split_qkv_norm_rope_into_paged_cache`
    /// to write per-seq K/V into the paged pool with correct RoPE per
    /// token position, then a single `paged_varlen_attention` call
    /// (Step 4 kernel) handles attention for all q-tokens with per-seq
    /// causal masks.
    ///
    /// REQUIRES paged KV (self.paged_pools.is_some()). Caller (the
    /// public `unified_forward` impl on the trait) returns
    /// `Err(unsupported)` for the contig path so the engine falls
    /// back to legacy dispatch.
    pub(crate) fn unified_forward_internal(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Vec<Option<Vec<f32>>> {
        if items.is_empty() {
            return Vec::new();
        }
        // Snapshot cfg fields into Copy locals so we can take &mut self later
        // without a long-lived `&self.cfg` borrow conflicting.
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let im = self.cfg.intermediate_size;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv_stride = q_dim + 2 * kv_dim;
        let eps = self.cfg.rms_norm_eps;
        let qk_mode: i32 = if self.cfg.has_qk_norm { 1 } else { 2 };
        let vocab = self.cfg.vocab_size;
        let num_layers = self.cfg.num_layers;
        let num_seqs = items.len();

        // Per-item bookkeeping (host) — shared helpers from
        // `common::decoder_unified` so Qwen3-MoE / future decoder
        // families don't re-implement cu_seqlens construction etc.
        let (q_lens, cu_seqlens_q, m_total) =
            crate::common::decoder_unified::compute_cu_seqlens_q(items);
        let pos_offsets = crate::common::decoder_unified::compute_pos_offsets(items);
        let max_kv_len = crate::common::decoder_unified::compute_max_kv_len(items);

        // Ensure all items' KV caches exist.
        for (cid, _, _, _) in items {
            self.ensure_kv(cid);
        }

        // Concatenated input tokens for one embedding_lookup.
        let all_tokens = crate::common::decoder_unified::concat_q_tokens(items);
        debug_assert_eq!(all_tokens.len(), m_total);

        // Paged path requirements.
        let pools_present = self.paged_pools.is_some();
        if !pools_present {
            // Caller should have routed to fallback; defensive.
            panic!("unified_forward_internal called without paged_pools — caller must check");
        }
        let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
        let block_size = 16usize; // matches PAGED_BLOCK_SIZE in ensure_kv

        // Make sure scratch fits this batch. Use the engine-side
        // `paged_max_seqs` cap (set in `enable_paged_batch` from
        // `FERRUM_PAGED_MAX_SEQS`, default 32) so the index buffers
        // (`unified_cu_seqlens_q`, `unified_pos_offsets`,
        // `unified_block_tables`, `unified_packed_*`) are sized for the
        // largest batch the engine will ever submit. Earlier we used
        // `num_seqs` and the buffers were pinned at the FIRST call's
        // batch size — c=1 warmup → c=16 bench would write 17 ints into
        // a 2-int buffer (CUDA_ERROR_INVALID_VALUE).
        debug_assert!(
            self.scratch.paged_max_seqs >= num_seqs,
            "unified_forward batch ({} items) exceeds engine paged_max_seqs ({})",
            num_seqs,
            self.scratch.paged_max_seqs,
        );
        let max_seqs = self.scratch.paged_max_seqs.max(num_seqs);
        // Take cfg ref only for the immediate ensure call (Copy fields
        // already snapshotted above into h/nh/etc — cfg is just for the
        // shape constants ensure_unified_scratch needs).
        let cfg_for_alloc = self.cfg.clone();
        self.scratch
            .ensure_unified_scratch(&cfg_for_alloc, m_total, max_seqs, max_blocks_per_seq);

        let mut ctx = B::new_context();

        // Embed all q-tokens into the unified residual buffer.
        let mut residual = self
            .scratch
            .unified_residual
            .take()
            .expect("unified_residual missing after ensure");
        let embed = self
            .embed
            .as_ref()
            .expect("unified_forward_internal called on backbone-only model");
        B::embedding_lookup(&mut ctx, embed, &all_tokens, &mut residual, h);

        // Upload index buffers.
        {
            let csq = self
                .scratch
                .unified_cu_seqlens_q
                .as_mut()
                .expect("unified_cu_seqlens_q missing");
            B::write_typed::<u32>(&mut ctx, csq, &cu_seqlens_q);
        }
        {
            let po = self
                .scratch
                .unified_pos_offsets
                .as_mut()
                .expect("unified_pos_offsets missing");
            B::write_typed::<u32>(&mut ctx, po, &pos_offsets);
        }
        // Stack per-seq block tables host-side, then upload.
        let stacked =
            crate::common::decoder_unified::stack_block_tables(items, max_blocks_per_seq, |cid| {
                self.kv_caches
                    .get(cid)
                    .expect("kv cache missing for unified item")[0]
                    .paged_block_indices
                    .clone()
            });
        {
            let bt = self
                .scratch
                .unified_block_tables
                .as_mut()
                .expect("unified_block_tables missing");
            B::write_typed::<u32>(&mut ctx, bt, &stacked);
        }

        // ── CUDA-graph capture/replay control ────────────────────────
        // `FERRUM_UNIFIED_GRAPH=1` opts in. Per-shape graph cache keyed
        // by (m_total, num_seqs); high bit set so we never collide with
        // legacy keys (SINGLE_ITEM = 0, batched = m_padded). The captured
        // region covers the layer loop + final rms_norm + per-item
        // copy_slice + lm_head — every kernel that's stable across iters.
        // Pre-work (write_u32 + embedding_lookup) and post-work (sync +
        // to_vec) stay eager.
        //
        // Status (2026-05-05): WORKING. The earlier
        // `gather_columns launch: CUDA_ERROR_INVALID_VALUE` was caused
        // by `with_marlin_gather_scratch` doing an in-place
        // `stream.alloc::<f16>` when its slot was too small — runtime
        // alloc inside a captured stream is illegal. Fix:
        // `B::pregrow_marlin_gather_scratch(m_total * intermediate_size)`
        // called eagerly above, BEFORE this section.
        //
        // Bench (RTX 4090, Llama-3.1-8B GPTQ-INT4, c=16):
        //   varlen-only:     680 tok/s, TPOT 19.3 ms
        //   varlen + graph:  714 tok/s, TPOT 18.3 ms  (+5% / -5%)
        let graph_enabled = self.batched_cfg.unified_graph;
        let graph_key = crate::common::decoder_unified::unified_graph_key(m_total, num_seqs);
        let cache_has_key = self.unified_graph_keys_seen.contains(&graph_key);

        // Pre-grow the marlin gather scratch slot to the worst-case
        // size for THIS call's matmul shapes. Done eagerly BEFORE
        // begin_capture: `with_marlin_gather_scratch`'s in-place grow
        // does `stream.alloc` which is forbidden inside a captured
        // stream (CUDA_ERROR_INVALID_VALUE on the next launch).
        // `down_proj` has the largest k = intermediate_size, so
        // m_total * intermediate_size is the upper bound across all
        // 4 matmuls in the layer.
        let max_marlin_required = m_total * im;
        B::pregrow_marlin_gather_scratch(&mut ctx, max_marlin_required);

        // Sync eager pre-work (write_u32 + embedding + scratch grow)
        // before either a replay or a capture starts — buffer state
        // must be settled.
        B::sync(&mut ctx);

        let unified_profile = self.batched_cfg.unified_profile;
        let layer_t0 = if unified_profile {
            B::sync(&mut ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };

        let mut did_pure_replay = false;
        if graph_enabled && cache_has_key && !self.unified_graph_failed {
            match B::replay_graph(&mut ctx, graph_key) {
                Ok(true) => did_pure_replay = true,
                Ok(false) => {}
                Err(e) => {
                    self.unified_graph_failed = true;
                    eprintln!("[unified-graph] replay err: {e}");
                }
            }
        }

        const UNIFIED_GRAPH_WARMUP: usize = 3;
        let should_capture = graph_enabled
            && !self.unified_graph_failed
            && !cache_has_key
            && self.unified_graph_warmup >= UNIFIED_GRAPH_WARMUP
            && !did_pure_replay;
        if !did_pure_replay {
            self.unified_graph_warmup += 1;
        }

        if should_capture {
            if let Err(e) = B::begin_graph_capture(&mut ctx) {
                eprintln!("[unified-graph] begin_capture err: {e}");
                self.unified_graph_failed = true;
            }
        }

        if !did_pure_replay {
            for li in 0..num_layers {
                self.unified_forward_layer(
                    &mut ctx,
                    li,
                    items,
                    &q_lens,
                    &cu_seqlens_q,
                    &pos_offsets,
                    &mut residual,
                    m_total,
                    max_kv_len,
                    num_seqs,
                    max_blocks_per_seq,
                    block_size,
                    qkv_stride,
                    q_dim,
                    kv_dim,
                    nh,
                    nkv,
                    hd,
                    im,
                    h,
                    eps,
                    qk_mode,
                );
            }
        }
        if let Some(t0) = layer_t0 {
            B::sync(&mut ctx);
            static UNIFIED_PROF_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = UNIFIED_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(64) {
                let total_us = t0.elapsed().as_micros();
                let per_layer = total_us / num_layers as u128;
                eprintln!(
                    "[unified-prof] call#{n} m_total={m_total} num_seqs={num_seqs} layers={num_layers} total={total_us}us per_layer={per_layer}us"
                );
            }
        }
        // Drain per-op counters every 64 calls when DECODE_OP_PROFILE is on.
        // The op-profile macro inside unified_forward_layer adds to these
        // global atomics; without a swap the totals would just keep growing.
        if self.batched_cfg.decode_op_profile {
            static OP_DRAIN_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = OP_DRAIN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(64) {
                use std::sync::atomic::Ordering;
                let norm_us = NORM_TIME_US.swap(0, Ordering::Relaxed);
                let norm_n = NORM_CALLS.swap(0, Ordering::Relaxed);
                let mm_us = MATMUL_TIME_US.swap(0, Ordering::Relaxed);
                let mm_n = MATMUL_CALLS.swap(0, Ordering::Relaxed);
                let qkr_us = QKR_TIME_US.swap(0, Ordering::Relaxed);
                let qkr_n = QKR_CALLS.swap(0, Ordering::Relaxed);
                let attn_us = ATTN_TIME_US.swap(0, Ordering::Relaxed);
                let attn_n = ATTN_CALLS.swap(0, Ordering::Relaxed);
                let other_us = OTHER_TIME_US.swap(0, Ordering::Relaxed);
                let other_n = OTHER_CALLS.swap(0, Ordering::Relaxed);
                let bucket = |label: &str, n: u64, us: u64| {
                    if n > 0 {
                        eprintln!(
                            "[unified-op] {label}: {} calls {} ms total avg={}us",
                            n,
                            us / 1000,
                            us / n.max(1)
                        );
                    }
                };
                eprintln!("[unified-op] drain#{n} (over last 64 unified_forward calls):");
                bucket("rms_norm/fused_rms", norm_n, norm_us);
                bucket("matmul (qkv/o/gate_up/down)", mm_n, mm_us);
                bucket("split_qkv_norm_rope_paged", qkr_n, qkr_us);
                bucket("paged_varlen_attention", attn_n, attn_us);
                bucket("silu/residual_add", other_n, other_us);
            }
        }

        // Identify per-item last-token global indices for is_final_chunk
        // items. Pure host work — same shape across iters at c=16
        // steady state, so the captured copy_slice + lm_head launches
        // record stable offsets.
        let final_indices =
            crate::common::decoder_unified::compute_final_indices(items, &cu_seqlens_q);
        let num_sampled = final_indices.len();

        // Take scratch buffers we'll either record into the graph or
        // restore on return — outside the !did_pure_replay branch so
        // both paths can put them back.
        let final_norm_w = &self.final_norm_w;
        let mut norm_out = self
            .scratch
            .unified_norm_out
            .take()
            .expect("unified_norm_out missing");

        if !did_pure_replay {
            B::rms_norm(
                &mut ctx,
                &residual,
                final_norm_w,
                eps,
                &mut norm_out,
                m_total,
                h,
            );
            if num_sampled > 0 {
                let packed_normed = self
                    .scratch
                    .unified_packed_normed
                    .as_mut()
                    .expect("unified_packed_normed missing");
                for (j, &(_, global)) in final_indices.iter().enumerate() {
                    B::copy_slice(&mut ctx, &norm_out, global * h, packed_normed, j * h, h);
                }
                let lm_head = self
                    .lm_head
                    .as_ref()
                    .expect("unified_forward_internal called on backbone-only model");
                let packed_logits = self
                    .scratch
                    .unified_packed_logits
                    .as_mut()
                    .expect("unified_packed_logits missing");
                {
                    #[cfg(feature = "cuda")]
                    let _alloc_label =
                        ferrum_kernels::backend::cuda::push_alloc_label("llama.unified.lm_head");
                    lm_head.forward(&mut ctx, packed_normed, packed_logits, num_sampled);
                }
            }
        }

        // End capture (if active), install graph, and immediately
        // replay it — capture only RECORDS the launches, so without a
        // post-capture replay the layer loop's kernels never actually
        // execute and packed_logits stays uninitialised. Mirrors the
        // legacy `forward_layer_batched_decode_post_attn` pattern.
        if should_capture && !self.unified_graph_failed {
            if let Err(e) = B::end_graph_capture(&mut ctx, graph_key) {
                eprintln!("[unified-graph] end_capture err: {e}");
                self.unified_graph_failed = true;
            } else {
                self.unified_graph_keys_seen.insert(graph_key);
                if let Err(e) = B::replay_graph(&mut ctx, graph_key) {
                    eprintln!("[unified-graph] post-capture replay err: {e}");
                    self.unified_graph_failed = true;
                }
            }
        }

        // Sync + readback. ALWAYS eager — to_vec needs the host result.
        B::sync(&mut ctx);
        let mut out: Vec<Option<Vec<f32>>> = (0..items.len()).map(|_| None).collect();
        if num_sampled > 0 {
            let packed_logits = self
                .scratch
                .unified_packed_logits
                .as_ref()
                .expect("unified_packed_logits missing");
            let logits_flat = B::to_vec(packed_logits, num_sampled * vocab);
            for (j, &(orig_idx, _)) in final_indices.iter().enumerate() {
                let row = logits_flat[j * vocab..(j + 1) * vocab].to_vec();
                out[orig_idx] = Some(row);
            }
        }

        // Bump cache.len for each item (we wrote q_lens[i] tokens into seq i's
        // KV pool inside the layer loop). Centralised post-loop bump matches
        // the pattern in decode_batch_internal.
        for (i, (cid, _, _, _)) in items.iter().enumerate() {
            let caches = self
                .kv_caches
                .get_mut(cid)
                .expect("kv cache missing for unified item post-loop");
            for c in caches.iter_mut() {
                c.len += q_lens[i];
            }
        }

        // Restore scratch for next call.
        self.scratch.unified_residual = Some(residual);
        self.scratch.unified_norm_out = Some(norm_out);

        out
    }

    /// One transformer layer for the unified mixed-batch forward.
    /// Mirrors `forward_layer_batched_decode` paged path but operates
    /// on `[M_total, *]` tensors and uses `paged_varlen_attention`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn unified_forward_layer(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        items: &[(String, Vec<u32>, usize, bool)],
        q_lens: &[usize],
        cu_seqlens_q: &[u32],
        pos_offsets: &[u32],
        residual: &mut B::Buffer,
        m_total: usize,
        max_kv_len: usize,
        num_seqs: usize,
        max_blocks_per_seq: usize,
        block_size: usize,
        qkv_stride: usize,
        q_dim: usize,
        kv_dim: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        im: usize,
        h: usize,
        eps: f32,
        qk_mode: i32,
    ) {
        let layer = &self.layers[li];
        let dummy_w = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy_w);
        let op_prof = self.batched_cfg.decode_op_profile;

        macro_rules! time_op {
            ($bucket_us:expr, $bucket_n:expr, $body:block) => {{
                if op_prof {
                    B::sync(ctx);
                    let _t0 = std::time::Instant::now();
                    $body
                    B::sync(ctx);
                    $bucket_us.fetch_add(_t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                    $bucket_n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    $body
                }
            }};
        }

        // 1. rms_norm [M_total, h] → unified_norm_out
        time_op!(NORM_TIME_US, NORM_CALLS, {
            let norm_out = self
                .scratch
                .unified_norm_out
                .as_mut()
                .expect("unified_norm_out missing");
            B::rms_norm(ctx, residual, &layer.input_ln_w, eps, norm_out, m_total, h);
        });

        // 2. qkv_proj GEMM (m=M_total): unified_norm_out → unified_qkv_out
        time_op!(MATMUL_TIME_US, MATMUL_CALLS, {
            let norm_out = self
                .scratch
                .unified_norm_out
                .as_ref()
                .expect("unified_norm_out missing");
            let qkv_out = self
                .scratch
                .unified_qkv_out
                .as_mut()
                .expect("unified_qkv_out missing");
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.unified_layer.qkv_proj");
            layer.qkv_proj.forward(ctx, norm_out, qkv_out, m_total);
        });

        // 3. Single-launch varlen split_qkv_norm_rope. Reads
        //    cu_seqlens_q / pos_offsets / block_tables from device
        //    buffers (graph-capturable; kernel scalars are stable across
        //    iters). Replaces the prior per-item dispatch loop (16 small
        //    launches per layer at c=16) with one launch.
        let pools = self
            .paged_pools
            .as_mut()
            .expect("unified_forward_layer requires paged_pools");
        let pool_ptr = (
            &mut pools[li].0 as *mut B::Buffer,
            &mut pools[li].1 as *mut B::Buffer,
        );
        // SAFETY: pools allocated once; not concurrently mutated.
        let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };
        // Suppress unused warning until the legacy per-item path goes.
        let _ = items;
        let _ = q_lens;
        let _ = cu_seqlens_q;
        let _ = pos_offsets;
        let _ = qkv_stride;

        time_op!(QKR_TIME_US, QKR_CALLS, {
            let qkv_out = self
                .scratch
                .unified_qkv_out
                .as_ref()
                .expect("unified_qkv_out missing");
            let packed_q = self
                .scratch
                .unified_packed_q
                .as_mut()
                .expect("unified_packed_q missing");
            let cu_seqlens_buf = self
                .scratch
                .unified_cu_seqlens_q
                .as_ref()
                .expect("unified_cu_seqlens_q missing");
            let pos_offsets_buf = self
                .scratch
                .unified_pos_offsets
                .as_ref()
                .expect("unified_pos_offsets missing");
            let bt_buf = self
                .scratch
                .unified_block_tables
                .as_ref()
                .expect("unified_block_tables missing");
            B::split_qkv_norm_rope_into_paged_cache_varlen(
                ctx,
                qkv_out,
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                packed_q,
                pool_k,
                pool_v,
                cu_seqlens_buf,
                pos_offsets_buf,
                bt_buf,
                num_seqs,
                m_total,
                nh,
                nkv,
                hd,
                eps,
                qk_mode,
                block_size,
                max_blocks_per_seq,
            )
            .expect("paged unified: split_qkv_norm_rope_into_paged_cache_varlen");
        });

        // 4. paged_varlen_attention: one call covering all M_total tokens.
        time_op!(ATTN_TIME_US, ATTN_CALLS, {
            let packed_q = self
                .scratch
                .unified_packed_q
                .as_ref()
                .expect("unified_packed_q missing");
            let cu_seqlens_buf = self
                .scratch
                .unified_cu_seqlens_q
                .as_ref()
                .expect("unified_cu_seqlens_q missing");
            let pos_offsets_buf = self
                .scratch
                .unified_pos_offsets
                .as_ref()
                .expect("unified_pos_offsets missing");
            let bt_buf = self
                .scratch
                .unified_block_tables
                .as_ref()
                .expect("unified_block_tables missing");
            let attn_out = self
                .scratch
                .unified_attn_out
                .as_mut()
                .expect("unified_attn_out missing");
            B::paged_varlen_attention(
                ctx,
                packed_q,
                pool_k,
                pool_v,
                attn_out,
                cu_seqlens_buf,
                pos_offsets_buf,
                bt_buf,
                num_seqs,
                m_total,
                max_kv_len,
                nh,
                nkv,
                hd,
                block_size,
                max_blocks_per_seq,
            )
            .expect("paged_varlen_attention");
        });

        // 5. o_proj (M_total): attn_out → o_proj_out
        time_op!(MATMUL_TIME_US, MATMUL_CALLS, {
            let attn_out = self
                .scratch
                .unified_attn_out
                .as_ref()
                .expect("unified_attn_out missing");
            let o_proj_out = self
                .scratch
                .unified_o_proj_out
                .as_mut()
                .expect("unified_o_proj_out missing");
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.unified_layer.o_proj");
            layer.o_proj.forward(ctx, attn_out, o_proj_out, m_total);
        });

        // 6. fused_add_rms_norm: residual = residual + o_proj_out;
        //    norm_out = rms_norm(new_residual, post_ln_w)
        time_op!(NORM_TIME_US, NORM_CALLS, {
            let o_proj_out = self
                .scratch
                .unified_o_proj_out
                .as_ref()
                .expect("unified_o_proj_out missing");
            let norm_out = self
                .scratch
                .unified_norm_out
                .as_mut()
                .expect("unified_norm_out missing");
            B::fused_add_rms_norm(
                ctx,
                residual,
                o_proj_out,
                &layer.post_ln_w,
                eps,
                norm_out,
                m_total,
                h,
            );
        });

        // 7. gate_up_proj
        time_op!(MATMUL_TIME_US, MATMUL_CALLS, {
            let norm_out = self
                .scratch
                .unified_norm_out
                .as_ref()
                .expect("unified_norm_out missing");
            let gate_up_out = self
                .scratch
                .unified_gate_up_out
                .as_mut()
                .expect("unified_gate_up_out missing");
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.unified_layer.gate_up_proj");
            layer
                .gate_up_proj
                .forward(ctx, norm_out, gate_up_out, m_total);
        });

        // 8. SwiGLU
        time_op!(OTHER_TIME_US, OTHER_CALLS, {
            let gate_up_out = self
                .scratch
                .unified_gate_up_out
                .as_ref()
                .expect("unified_gate_up_out missing");
            let silu_out = self
                .scratch
                .unified_silu_out
                .as_mut()
                .expect("unified_silu_out missing");
            B::fused_silu_mul_split(ctx, gate_up_out, silu_out, m_total, im);
        });

        // 9. down_proj
        time_op!(MATMUL_TIME_US, MATMUL_CALLS, {
            let silu_out = self
                .scratch
                .unified_silu_out
                .as_ref()
                .expect("unified_silu_out missing");
            let mlp_out = self
                .scratch
                .unified_mlp_out
                .as_mut()
                .expect("unified_mlp_out missing");
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.unified_layer.down_proj");
            layer.down_proj.forward(ctx, silu_out, mlp_out, m_total);
        });

        // 10. residual_add
        time_op!(OTHER_TIME_US, OTHER_CALLS, {
            let mlp_out = self
                .scratch
                .unified_mlp_out
                .as_ref()
                .expect("unified_mlp_out missing");
            B::add_inplace(ctx, residual, mlp_out, m_total * h);
        });
    }
    /// Batched decode: process M concurrent requests at potentially different
    /// positions in one forward pass. GEMM-heavy ops (qkv_proj, o_proj,
    /// gate_up, down) run with m=M for natural batching; rope + KV append +
    /// attention loop per-item (each has its own KV cache at a different
    /// kv_len, and potentially different pos).
    ///
    /// Returns M logit vectors in the same order as `batch`.
    pub fn decode_batch_internal(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_internal_with_full_logits(batch, false)
    }

    pub fn decode_batch_internal_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        let logits_return = if force_full_logits {
            DecodeLogitsReturn::Full
        } else {
            DecodeLogitsReturn::LegacyDefault
        };
        self.decode_batch_internal_with_logits_return(batch, logits_return)
    }

    pub fn decode_batch_internal_with_logits_policy(
        &mut self,
        batch: &[(String, u32, u32)],
        policies: &[LogitsReturnPolicy],
    ) -> Vec<Vec<f32>> {
        if policies.len() != batch.len() {
            return self.decode_batch_internal_with_full_logits(batch, true);
        }

        let mut selected_mask: Option<&TokenSelectionMask> = None;
        let mut saw_unmasked = false;
        for policy in policies {
            match policy {
                LogitsReturnPolicy::FullLogits => {
                    return self.decode_batch_internal_with_full_logits(batch, true);
                }
                LogitsReturnPolicy::GreedyArgmax { token_mask } => match token_mask.as_ref() {
                    Some(mask) => {
                        if saw_unmasked {
                            return self.decode_batch_internal_with_full_logits(batch, true);
                        }
                        if let Some(selected) = selected_mask {
                            if selected.fingerprint != mask.fingerprint
                                || selected.len() != mask.len()
                            {
                                return self.decode_batch_internal_with_full_logits(batch, true);
                            }
                        } else {
                            selected_mask = Some(mask);
                        }
                    }
                    None => {
                        if selected_mask.is_some() {
                            return self.decode_batch_internal_with_full_logits(batch, true);
                        }
                        saw_unmasked = true;
                    }
                },
            }
        }

        self.decode_batch_internal_with_logits_return(
            batch,
            DecodeLogitsReturn::GreedyArgmax {
                token_mask: selected_mask,
            },
        )
    }

    fn decode_batch_internal_with_logits_return(
        &mut self,
        batch: &[(String, u32, u32)],
        logits_return: DecodeLogitsReturn<'_>,
    ) -> Vec<Vec<f32>> {
        let force_full_logits = matches!(logits_return, DecodeLogitsReturn::Full);
        let m = batch.len();
        if m == 0 {
            return Vec::new();
        }
        self.decode_batch_stats.record_call(m, force_full_logits);
        if m == 1 && !force_full_logits {
            let (cid, tok, pos) = &batch[0];
            self.decode_batch_stats.record_singleton_fast_path();
            return vec![self.decode_internal(cid, *tok, *pos)];
        }
        if !self.supports_batched_decode {
            // Some backends do not yet produce correct follow-up logits in
            // the optimized dense batched decode path under concurrent
            // serving. Preserve user-visible correctness by falling back to
            // the known-good per-item decode path until that backend's
            // batched kernels pass the dedicated multi-turn gate.
            self.decode_batch_stats.record_unsupported_fallback();
            return batch
                .iter()
                .map(|(cid, tok, pos)| self.decode_internal(cid, *tok, *pos))
                .collect();
        }
        if batch
            .iter()
            .any(|(cid, _, _)| self.active_lora_adapter_for_cache(cid).is_some())
        {
            self.decode_batch_stats.record_lora_fallback();
            return batch
                .iter()
                .map(|(cid, tok, pos)| self.decode_internal(cid, *tok, *pos))
                .collect();
        }

        // Ensure all caches exist and scratch is sized for M tokens.
        for (cid, _, _) in batch {
            self.ensure_kv(cid);
        }
        self.ensure_scratch(m);
        self.scratch.ensure_batch_logits(&self.cfg, m);
        // Phase 4b: when paged mode is on, ensure_kv has already
        // populated the batched scratch buffers (paged_batch_q etc.).
        // The forward path branches on `paged_pools.is_some()` inside
        // each layer.

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let num_layers = self.cfg.num_layers;
        let mut ctx = B::new_context();

        // Pre-step state (positions, kv_lens_pre, kv_lens_post). All
        // 32 layers' batched kernels read these from scratch device
        // buffers — written ONCE here, NEVER inside the layer loop, so
        // the captured graph below replays with stable buffer addresses
        // and the next step's content update reaches the kernels.
        let positions: Vec<u32> = batch.iter().map(|(_, _, p)| *p as u32).collect();
        let tokens: Vec<u32> = batch.iter().map(|(_, t, _)| *t).collect();
        let kv_pre: Vec<u32> = batch
            .iter()
            .map(|(cid, _, _)| self.kv_caches.get(cid).expect("kv_caches missing")[0].len as u32)
            .collect();
        let kv_post: Vec<u32> = kv_pre.iter().map(|&x| x + 1).collect();
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_positions, &positions);
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_kv_lens_pre, &kv_pre);
        B::write_typed::<u32>(&mut ctx, &mut self.scratch.batch_kv_lens_post, &kv_post);

        // Pre-populate per-slot device-pointer scratch for the batched
        // kernels (kv_cache_append_batched, flash_attention_batched).
        // Done OUTSIDE any captured forward — sync memcpy on the legacy
        // null stream is not captured by stream capture, so the captured
        // graph contains only kernel launches. Without this, the
        // captured `stream.memcpy_htod` records host pointers and the
        // 2nd pure-replay reads stale/corrupted data → ILLEGAL_ADDRESS.
        //
        // populate_batched_pointers was intended to support
        // FERRUM_BATCHED_GRAPH=1 (Phase 4d graph capture) by pre-filling
        // device scratch outside the captured forward. Phase 4d is
        // shelved (CUDA driver state accumulates and SIGSEGVs after
        // ~14 launches even with all the right knobs), so the populate
        // call adds overhead with no benefit on the OFF path. Skip it —
        // the kv_cache_append / flash_attn batched impls fall back to
        // their inline captured memcpy when scratch isn't pre-populated.
        // batched_pointers_for kept for future use; revisit when we
        // either (a) get graph capture working, or (b) move scratch to
        // process-global static.
        let _ = &self.batched_pointers_for;

        // 0. Embed all M tokens into residual [M, H]. Eager, OUTSIDE
        //    any captured graph (host tokens slice; embedding_lookup_dyn
        //    is single-item only).
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("decode_batch_internal called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, &tokens, &mut residual, h);
        if let Some(scale) = self.cfg.embed_scale {
            B::scale_inplace(&mut ctx, &mut residual, scale, m * h);
        }
        let use_host_residual_shadow = self.use_host_residual_shadow();
        let use_device_residual_shadow = self.use_device_residual_shadow();
        let mut host_residual = if use_host_residual_shadow {
            B::sync(&mut ctx);
            Some(B::to_vec(&residual, m * h))
        } else {
            None
        };
        let mut device_residual_shadow = if use_device_residual_shadow {
            let mut shadow = self
                .scratch
                .residual_f32_shadow
                .take()
                .expect("device F32 residual shadow scratch missing");
            B::activation_to_f32_shadow(&mut ctx, &residual, &mut shadow, m * h);
            Some(shadow)
        } else {
            None
        };
        let mut device_branch_shadow = if use_device_residual_shadow {
            Some(
                self.scratch
                    .sandwich_branch_f32
                    .take()
                    .expect("device F32 sandwich branch scratch missing"),
            )
        } else {
            None
        };

        // ── Phase 4d: CUDA-graph replay path ─────────────────────────
        // gated on FERRUM_BATCHED_GRAPH=1; skipped on backends without
        // graph support (begin_graph_capture returns Err).
        let graph_enabled = self.batched_cfg.batched_graph
            && !use_host_residual_shadow
            && !use_device_residual_shadow;
        let m_padded = m.next_power_of_two();
        // Per-m_padded graph cache: each batch shape gets its own
        // captured graph instead of thrashing a single slot. Native
        // CUDA microbench (graph_upload_bench) confirmed multi-slot
        // replay is stable.
        let graph_key = m_padded as u64;
        let cache_has_key = self.batched_graph_keys_seen.contains(&graph_key);

        let mut did_pure_replay = false;
        if graph_enabled && cache_has_key && !self.batched_graph_failed {
            // Sync stream first so embedding_lookup (just queued) plus
            // any null-stream cuMemcpyHtoD_v2's from write_u32 are all
            // settled before cuGraphLaunch.
            B::sync(&mut ctx);
            match B::replay_graph(&mut ctx, graph_key) {
                Ok(true) => {
                    did_pure_replay = true;
                    BATCHED_GRAPH_REPLAY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(false) => {}
                Err(e) => {
                    self.batched_graph_failed = true;
                    eprintln!("[batched-trace] replay err: {}", e);
                }
            }
        }
        if graph_enabled && !did_pure_replay {
            BATCHED_GRAPH_EAGER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // Periodic stats (every 256 calls).
        let total = BATCHED_GRAPH_REPLAY_COUNT.load(std::sync::atomic::Ordering::Relaxed)
            + BATCHED_GRAPH_EAGER_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        if graph_enabled && total > 0 && total.is_multiple_of(256) {
            eprintln!(
                "[batched-graph-stats] m={m} m_padded={m_padded} replays={} eagers={} keys_seen={:?}",
                BATCHED_GRAPH_REPLAY_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                BATCHED_GRAPH_EAGER_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                self.batched_graph_keys_seen,
            );
        }

        if !did_pure_replay {
            const BATCHED_GRAPH_WARMUP: usize = 3;
            let should_capture = graph_enabled
                && !self.batched_graph_failed
                && self.batched_graph_warmup >= BATCHED_GRAPH_WARMUP;
            if should_capture {
                tracing::debug!("[batched-graph] BEGIN CAPTURE m_padded={m_padded}");
                if let Err(e) = B::begin_graph_capture(&mut ctx) {
                    eprintln!("[batched-trace] begin_capture err: {}", e);
                    self.batched_graph_failed = true;
                }
            }
            self.batched_graph_warmup += 1;

            // Trace mode (env): sync after each major op so that the
            // first panicking sync localises which kernel/section faulted.
            // Off by default (adds 32 syncs per token = pipeline serialisation).
            let trace = self.batched_cfg.batched_trace;
            macro_rules! tracesync {
                ($label:expr) => {
                    if trace {
                        B::sync(&mut ctx);
                        eprintln!("[trace-batched] {}", $label);
                    }
                };
            }
            tracesync!("entry-after-writes-and-embed");

            // Op-profile: time the entire batched forward (eager only —
            // pure replay short-circuits above and does not increment
            // the per-op counters since the wrapped ops aren't executed
            // by the Rust dispatch path).
            let batched_profile = self.batched_cfg.decode_op_profile;
            let batched_iter_t0 = if batched_profile {
                // Drain shared counters first so this iter's print isn't
                // contaminated by prior prefill/single-decode contributions.
                ATTN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                ATTN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                QKR_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                QKR_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                MATMUL_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                MATMUL_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                OTHER_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                OTHER_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                B::sync(&mut ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Eager forward (records into graph if capture is active).
            for li in 0..num_layers {
                self.forward_layer_batched_decode(
                    &mut ctx,
                    li,
                    batch,
                    &mut residual,
                    m,
                    host_residual.as_deref_mut(),
                    device_residual_shadow.as_mut(),
                    device_branch_shadow.as_mut(),
                );
                tracesync!(format!("after layer {}", li));
            }
            let _t0_norm = if batched_profile {
                B::sync(&mut ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            if let Some(host) = host_residual.as_deref() {
                Self::rms_norm_host_to_activation(
                    &mut ctx,
                    host,
                    &self.final_norm_w,
                    self.cfg.rms_norm_eps,
                    &mut self.scratch.norm_out,
                    m,
                    h,
                );
            } else if let Some(device) = device_residual_shadow.as_ref() {
                B::rms_norm_f32_to_activation(
                    &mut ctx,
                    device,
                    &self.final_norm_w,
                    self.cfg.rms_norm_eps,
                    &mut self.scratch.norm_out,
                    m,
                    h,
                );
            } else {
                B::rms_norm(
                    &mut ctx,
                    &residual,
                    &self.final_norm_w,
                    self.cfg.rms_norm_eps,
                    &mut self.scratch.norm_out,
                    m,
                    h,
                );
            }
            if let Some(t0) = _t0_norm {
                B::sync(&mut ctx);
                NORM_TIME_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            tracesync!("after final rms_norm");
            let lm_head = self
                .lm_head
                .as_ref()
                .expect("decode_batch_internal called on backbone-only model (no lm_head)");
            let _t0_lm = if batched_profile {
                B::sync(&mut ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            {
                #[cfg(feature = "cuda")]
                let _alloc_label =
                    ferrum_kernels::backend::cuda::push_alloc_label("llama.batched.lm_head");
                lm_head.forward(
                    &mut ctx,
                    &self.scratch.norm_out,
                    &mut self.scratch.batch_logits,
                    m,
                );
            }
            if let Some(t0) = _t0_lm {
                B::sync(&mut ctx);
                MATMUL_TIME_US.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            tracesync!("after lm_head");

            if let Some(t0) = batched_iter_t0 {
                B::sync(&mut ctx);
                let total_us = t0.elapsed().as_micros() as u64;
                let attn_us = ATTN_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                let attn_n = ATTN_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                let qkr_us = QKR_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                let qkr_n = QKR_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                let mm_us = MATMUL_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                let mm_n = MATMUL_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                let norm_us = NORM_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                let norm_n = NORM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                let other_us = OTHER_TIME_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                let other_n = OTHER_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
                let wrapped_us = attn_us + qkr_us + mm_us + norm_us + other_us;
                let unwrapped_us = total_us.saturating_sub(wrapped_us);
                eprintln!(
                    "[batched-op-profile] m={} total={}us  matmul={}us({}) attn={}us({}) qkr={}us({}) norm={}us({}) other={}us({})  unwrapped={}us",
                    m,
                    total_us,
                    mm_us, mm_n,
                    attn_us, attn_n,
                    qkr_us, qkr_n,
                    norm_us, norm_n,
                    other_us, other_n,
                    unwrapped_us,
                );
            }

            if should_capture && !self.batched_graph_failed {
                if let Err(e) = B::end_graph_capture(&mut ctx, graph_key) {
                    eprintln!("[batched-trace] end_capture err: {}", e);
                    self.batched_graph_failed = true;
                } else {
                    self.batched_graph_keys_seen.insert(graph_key);
                    if let Err(e) = B::replay_graph(&mut ctx, graph_key) {
                        eprintln!("[batched-trace] post-capture replay err: {}", e);
                        self.batched_graph_failed = true;
                    }
                }
            }
        }

        // Bump cache.len for all (m × num_layers) caches. forward_layer
        // no longer bumps (so a graph replay's lack of Rust execution
        // doesn't desync it). One central bump covers eager and replay.
        for (cid, _, _) in batch.iter() {
            let caches = self.kv_caches.get_mut(cid).expect("kv_caches missing");
            for li in 0..num_layers {
                caches[li].len += 1;
            }
        }

        // Sync before to_vec (Metal: no internal sync on buffer read).
        B::sync(&mut ctx);
        if use_device_residual_shadow {
            self.scratch.residual_f32_shadow = device_residual_shadow;
            self.scratch.sandwich_branch_f32 = device_branch_shadow;
        }
        self.scratch.residual = Some(residual);

        // Greedy fast path: FERRUM_GREEDY_ARGMAX=1 → GPU argmax + tiny
        // D2H (m × 4 B) instead of full logit download (m × vocab × 2 B).
        // Saves ~5 ms / iter at c=32 on Qwen3 vocab=152064. Engine has a
        // matching size-1-Vec fast path in run_batch_decode that picks
        // `logits[0] as u32` and skips sample_with_processors entirely.
        let argmax_mode = match logits_return {
            DecodeLogitsReturn::Full => None,
            DecodeLogitsReturn::LegacyDefault if self.batched_cfg.greedy_argmax => {
                Some(ArgmaxMode::Raw)
            }
            DecodeLogitsReturn::LegacyDefault => None,
            DecodeLogitsReturn::GreedyArgmax { token_mask } => {
                token_mask.map(ArgmaxMode::Masked).or(Some(ArgmaxMode::Raw))
            }
        };
        if let Some(argmax_mode) = argmax_mode {
            let tokens = match argmax_mode {
                ArgmaxMode::Raw => {
                    B::argmax_rows_f16(&mut ctx, &self.scratch.batch_logits, m, vocab)
                }
                ArgmaxMode::Masked(mask) => {
                    self.scratch.ensure_argmax_token_mask(&mut ctx, mask);
                    let mask_len = self.scratch.argmax_token_mask_len;
                    let device_mask = self
                        .scratch
                        .argmax_token_mask
                        .as_ref()
                        .expect("argmax token mask upload failed");
                    B::argmax_rows_f16_masked(
                        &mut ctx,
                        &self.scratch.batch_logits,
                        device_mask,
                        mask_len,
                        m,
                        vocab,
                    )
                }
            };
            if let Ok(tokens) = tokens {
                if tokens.iter().all(|&token| token != u32::MAX) {
                    return tokens.into_iter().map(|t| vec![t as f32]).collect();
                }
            }
        }

        let all = B::to_vec(&self.scratch.batch_logits, m * vocab);
        (0..m)
            .map(|i| all[i * vocab..(i + 1) * vocab].to_vec())
            .collect()
    }
}

impl<B: MoeLlmBackend> LlamaPipelineStageBatchOps<B> for LlamaFamilyModel<B, KvFp16> {
    fn decode_stage_tokens_to_hidden_batch(
        &mut self,
        batch: &[(String, u32, u32)],
    ) -> PipelineHidden<B> {
        if batch.is_empty() {
            return PipelineHidden::host(Vec::new(), 0, self.cfg.hidden_size);
        }
        if batch.len() == 1 || !self.supports_batched_decode || self.cfg.sandwich_norms {
            let h = self.cfg.hidden_size;
            let mut hidden = Vec::with_capacity(batch.len() * h);
            for (cache_id, token, pos) in batch {
                hidden
                    .extend_from_slice(&self.decode_stage_token_to_hidden(cache_id, *token, *pos));
            }
            return PipelineHidden::host(hidden, batch.len(), h);
        }

        let (m, h, mut ctx) = self.prepare_batched_decode_stage(batch);
        let tokens: Vec<u32> = batch.iter().map(|(_, token, _)| *token).collect();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("decode_stage_tokens_to_hidden_batch called on stage without embedding");
        B::embedding_lookup(&mut ctx, embed, &tokens, &mut residual, h);

        for li in 0..self.local_layer_count() {
            self.forward_layer_batched_decode(
                &mut ctx,
                li,
                batch,
                &mut residual,
                m,
                None,
                None,
                None,
            );
        }
        self.bump_local_batched_decode_kv_lengths(batch);

        B::sync(&mut ctx);
        let out = B::to_vec(&residual, m * h);
        self.scratch.residual = Some(residual);
        PipelineHidden::host(out, m, h)
    }

    fn decode_stage_hidden_from_host_batch(
        &mut self,
        batch: &[(String, u32, u32)],
        hidden: &PipelineHidden<B>,
    ) -> (PipelineHidden<B>, LlamaStageHiddenBridgeTiming) {
        if batch.is_empty() {
            return (
                PipelineHidden::host(Vec::new(), 0, self.cfg.hidden_size),
                LlamaStageHiddenBridgeTiming::default(),
            );
        }
        let h = self.cfg.hidden_size;
        let hidden_slice = hidden.host_slice();
        assert_eq!(
            hidden_slice.len(),
            batch.len() * h,
            "hidden length {} != batch * hidden_size {}",
            hidden_slice.len(),
            batch.len() * h
        );
        if batch.len() == 1 || !self.supports_batched_decode || self.cfg.sandwich_norms {
            let mut out = Vec::with_capacity(hidden_slice.len());
            let mut bridge_timing = LlamaStageHiddenBridgeTiming::default();
            for (row, (cache_id, _, pos)) in batch.iter().enumerate() {
                let start = row * h;
                let (row_hidden, row_timing) = self.decode_stage_hidden_from_host_with_timing(
                    cache_id,
                    &hidden_slice[start..start + h],
                    *pos,
                );
                out.extend_from_slice(&row_hidden);
                bridge_timing = bridge_timing.add(row_timing);
            }
            return (PipelineHidden::host(out, batch.len(), h), bridge_timing);
        }

        let (m, h, mut ctx) = self.prepare_batched_decode_stage(batch);
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let bridge_t0 = std::time::Instant::now();
        let host_copy_t0 = std::time::Instant::now();
        B::write_f32_to_activation(&mut ctx, &mut residual, hidden_slice);
        let host_copy_us = elapsed_micros_u64_floor1(host_copy_t0);
        let bridge_timing = LlamaStageHiddenBridgeTiming {
            bridge_us: elapsed_micros_u64_floor1(bridge_t0),
            host_copy_us,
            device_copy_us: 0,
        };

        for li in 0..self.local_layer_count() {
            self.forward_layer_batched_decode(
                &mut ctx,
                li,
                batch,
                &mut residual,
                m,
                None,
                None,
                None,
            );
        }
        self.bump_local_batched_decode_kv_lengths(batch);

        B::sync(&mut ctx);
        let out = B::to_vec(&residual, m * h);
        self.scratch.residual = Some(residual);
        (PipelineHidden::host(out, m, h), bridge_timing)
    }

    fn logits_from_hidden_batch(
        &mut self,
        hidden: &PipelineHidden<B>,
        force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        let row_count = hidden.row_count();
        if row_count == 0 {
            return Vec::new();
        }
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let hidden_slice = hidden.host_slice();
        assert_eq!(
            hidden_slice.len(),
            row_count * h,
            "hidden length {} != row_count * hidden_size {}",
            hidden_slice.len(),
            row_count * h
        );
        if row_count == 1 || !self.supports_batched_decode || self.cfg.sandwich_norms {
            return (0..row_count)
                .map(|row| {
                    let start = row * h;
                    self.logits_from_hidden(&hidden_slice[start..start + h])
                })
                .collect();
        }

        self.ensure_scratch(row_count);
        self.scratch.ensure_batch_logits(&self.cfg, row_count);
        let mut ctx = B::new_context();
        let mut hidden_buf = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing before logits_from_hidden_batch");
        B::write_f32_to_activation(&mut ctx, &mut hidden_buf, hidden_slice);
        B::rms_norm(
            &mut ctx,
            &hidden_buf,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.norm_out,
            row_count,
            h,
        );
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("logits_from_hidden_batch called on stage without lm_head");
        {
            #[cfg(feature = "cuda")]
            let _alloc_label =
                ferrum_kernels::backend::cuda::push_alloc_label("llama.logits_batch.lm_head");
            lm_head.forward(
                &mut ctx,
                &self.scratch.norm_out,
                &mut self.scratch.batch_logits,
                row_count,
            );
        }
        B::sync(&mut ctx);
        self.scratch.residual = Some(hidden_buf);

        let greedy = self.batched_cfg.greedy_argmax && !force_full_logits;
        if greedy {
            let tokens = B::argmax_rows_f16(&mut ctx, &self.scratch.batch_logits, row_count, vocab)
                .expect("argmax_rows_f16");
            tokens.into_iter().map(|t| vec![t as f32]).collect()
        } else {
            let all = B::to_vec(&self.scratch.batch_logits, row_count * vocab);
            (0..row_count)
                .map(|row| all[row * vocab..(row + 1) * vocab].to_vec())
                .collect()
        }
    }
}
