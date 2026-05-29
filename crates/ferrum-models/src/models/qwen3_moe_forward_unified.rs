//! Qwen3-MoE unified mixed-batch forward — vLLM-style.
//!
//! Design (per vLLM v1 `gpu_model_runner.py`):
//! - SHARED scratch: residual / norm_out / qkv_out / moe_out are the
//!   same buffers the legacy decode/prefill paths use, sized for
//!   `max_tokens` at scratch construction. No per-call realloc.
//! - Flat `[M_total, hidden]` activations, ONE forward call per iter.
//! - Per-call index buffers (cu_seqlens_q / pos_offsets / block_tables)
//!   are tiny (max_seqs × few bytes), pre-allocated by
//!   `ensure_unified_scratch`.
//! - Variable q_len handled inside the varlen attention kernel.
//! - MoE block: ONE `moe_forward_bucketed` call at M = M_total — MoE
//!   doesn't care about per-request boundaries (every token routes
//!   independently).
//!
//! Returns one `Option<Vec<f32>>` per item: `Some(logits)` only for
//! items whose `is_final_chunk = true`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ferrum_interfaces::kv_dtype::KvDtypeKind;
use ferrum_kernels::backend::{Backend, BackendPagedKv, MoeLlmBackend};

use super::qwen3_moe::{fa_layout_varlen_enabled, Qwen3MoeModel};

#[derive(Default)]
pub(crate) struct UnifiedLayerProfile {
    input_norm_us: u64,
    qkv_us: u64,
    split_cache_us: u64,
    attn_us: u64,
    o_proj_us: u64,
    post_norm_us: u64,
    router_us: u64,
    moe_us: u64,
    residual_add_us: u64,
    final_norm_us: u64,
    sample_gather_us: u64,
    lm_head_us: u64,
    readback_us: u64,
}

static UNIFIED_LAYER_PROF_CALLS: AtomicU64 = AtomicU64::new(0);

fn unified_layer_prof_enabled() -> bool {
    std::env::var("FERRUM_UNIFIED_LAYER_PROF").map_or(false, |v| v != "0")
}

fn unified_layer_prof_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn unified_layer_prof_selected(m_total: usize, num_seqs: usize) -> bool {
    if !unified_layer_prof_enabled() {
        return false;
    }
    if let Some(max_m) = unified_layer_prof_usize("FERRUM_UNIFIED_LAYER_PROF_MAX_M") {
        if m_total > max_m {
            return false;
        }
    }
    if let Some(min_seqs) = unified_layer_prof_usize("FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS") {
        if num_seqs < min_seqs {
            return false;
        }
    }
    true
}

fn unified_layer_prof_every() -> u64 {
    std::env::var("FERRUM_UNIFIED_LAYER_PROF_EVERY")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(16)
}

fn vllm_varlen_tiled_q4_enabled() -> bool {
    std::env::var("FERRUM_VLLM_VARLEN_TILED_Q4").as_deref() == Ok("1")
}

fn fa_layout_varlen_tiled_q4_enabled() -> bool {
    std::env::var("FERRUM_FA_LAYOUT_VARLEN_TILED_Q4").as_deref() == Ok("1")
}

fn unified_greedy_argmax_enabled() -> bool {
    std::env::var("FERRUM_GREEDY_ARGMAX").as_deref() == Ok("1")
        && std::env::var("FERRUM_UNIFIED_GREEDY_ARGMAX").map_or(true, |v| v != "0")
}

fn unified_stage_start<B: Backend>(ctx: &mut B::Context, enabled: bool) -> Option<Instant> {
    if enabled {
        B::sync(ctx);
        Some(Instant::now())
    } else {
        None
    }
}

fn unified_stage_finish<B: Backend>(ctx: &mut B::Context, start: Option<Instant>) -> u64 {
    if let Some(start) = start {
        B::sync(ctx);
        start.elapsed().as_micros() as u64
    } else {
        0
    }
}

fn log_unified_layer_profile(
    prof: &UnifiedLayerProfile,
    m_total: usize,
    num_seqs: usize,
    num_sampled: usize,
    num_layers: usize,
) {
    let call = UNIFIED_LAYER_PROF_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let every = unified_layer_prof_every();
    if call > 8 && call % every != 0 {
        return;
    }

    let layer_sum = prof.input_norm_us
        + prof.qkv_us
        + prof.split_cache_us
        + prof.attn_us
        + prof.o_proj_us
        + prof.post_norm_us
        + prof.router_us
        + prof.moe_us
        + prof.residual_add_us;
    let final_sum = prof.final_norm_us + prof.sample_gather_us + prof.lm_head_us + prof.readback_us;
    eprintln!(
        "[unified-layer-prof] call#{} m={} seqs={} sampled={} layers={} layer_sum={}us final_sum={}us \
         input_norm={} qkv={} split_cache={} attn={} o_proj={} post_norm={} router={} moe={} residual_add={} \
         final_norm={} sample_gather={} lm_head={} readback={} (us)",
        call,
        m_total,
        num_seqs,
        num_sampled,
        num_layers,
        layer_sum,
        final_sum,
        prof.input_norm_us,
        prof.qkv_us,
        prof.split_cache_us,
        prof.attn_us,
        prof.o_proj_us,
        prof.post_norm_us,
        prof.router_us,
        prof.moe_us,
        prof.residual_add_us,
        prof.final_norm_us,
        prof.sample_gather_us,
        prof.lm_head_us,
        prof.readback_us,
    );
}

impl<B, K> Qwen3MoeModel<B, K>
where
    B: MoeLlmBackend + BackendPagedKv,
    K: KvDtypeKind,
{
    pub(crate) fn unified_forward_internal(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Vec<Option<Vec<f32>>> {
        if items.is_empty() {
            return Vec::new();
        }

        // ── 1. Shared helpers for per-item bookkeeping ──
        let (q_lens, cu_seqlens_q, m_total) =
            crate::common::decoder_unified::compute_cu_seqlens_q(items);
        let pos_offsets = crate::common::decoder_unified::compute_pos_offsets(items);
        let max_kv_len = crate::common::decoder_unified::compute_max_kv_len(items);
        let num_seqs = items.len();
        let final_indices =
            crate::common::decoder_unified::compute_final_indices(items, &cu_seqlens_q);
        let num_sampled = final_indices.len();

        // ── 2. KV cache discovery ──
        for (cid, _, _, _) in items {
            self.ensure_kv(cid);
        }
        if self.paged_pools.is_none() {
            panic!(
                "Qwen3MoeModel::unified_forward_internal called without paged_pools; \
                 set FERRUM_METAL_PAGED_KV=1"
            );
        }

        // ── 3. Snapshot config + ensure unified INDEX scratch ──
        let h = self.cfg.base.hidden_size;
        let nh = self.cfg.base.num_heads;
        let nkv = self.cfg.base.num_kv_heads;
        let hd = self.cfg.base.head_dim;
        let eps = self.cfg.base.rms_norm_eps;
        let qk_mode: i32 = if self.cfg.base.has_qk_norm { 1 } else { 2 };
        let vocab = self.cfg.base.vocab_size;
        let num_layers = self.cfg.base.num_layers;
        let inter = self.cfg.expert_intermediate_size;
        let top_k = self.cfg.num_experts_per_tok;
        let n_exp = self.cfg.num_experts;
        let norm_topk_prob = self.cfg.norm_topk_prob;

        let (paged_max_seqs, max_blocks_per_seq) = self
            .paged_dims
            .expect("paged_dims missing — ensure_kv must set this");
        let block_size = 16usize;
        debug_assert!(paged_max_seqs >= num_seqs);
        debug_assert!(m_total <= self.scratch.max_tokens);
        let max_seqs = paged_max_seqs.max(num_seqs);
        let cfg_clone = self.cfg.clone();
        self.scratch
            .ensure_unified_scratch(&cfg_clone, max_seqs, max_blocks_per_seq);

        let mut ctx = B::new_context();

        // ── 4. Take legacy residual, embed all tokens into it ──
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch.residual missing");
        let all_tokens = crate::common::decoder_unified::concat_q_tokens(items);
        debug_assert_eq!(all_tokens.len(), m_total);
        B::embedding_lookup(&mut ctx, &self.embed, &all_tokens, &mut residual, h);

        // ── 5. Upload index buffers ──
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

        let use_vllm_tiled_q4 =
            self.use_vllm_paged_attn && vllm_varlen_tiled_q4_enabled() && m_total > num_seqs;
        let use_fa_layout_tiled_q4 = self.use_vllm_paged_attn
            && fa_layout_varlen_enabled()
            && fa_layout_varlen_tiled_q4_enabled()
            && m_total > num_seqs;
        let mut tile_q4_count = 0usize;
        if use_vllm_tiled_q4 || use_fa_layout_tiled_q4 {
            let mut tile_seqs = Vec::with_capacity(m_total);
            let mut tile_starts = Vec::with_capacity(m_total);
            for (seq_idx, &q_len) in q_lens.iter().enumerate() {
                for start in (0..q_len).step_by(4) {
                    tile_seqs.push(seq_idx as u32);
                    tile_starts.push(start as u32);
                }
            }
            tile_q4_count = tile_seqs.len();
            let tile_seqs_buf = self
                .scratch
                .unified_tile_q4_seqs
                .as_mut()
                .expect("unified_tile_q4_seqs missing");
            B::write_typed::<u32>(&mut ctx, tile_seqs_buf, &tile_seqs);
            let tile_starts_buf = self
                .scratch
                .unified_tile_q4_starts
                .as_mut()
                .expect("unified_tile_q4_starts missing");
            B::write_typed::<u32>(&mut ctx, tile_starts_buf, &tile_starts);
        }

        // Pre-grow Marlin gather scratch for worst-case GEMM input m.
        let max_marlin_required = m_total * inter;
        B::pregrow_marlin_gather_scratch(&mut ctx, max_marlin_required);
        B::sync(&mut ctx);

        // ── 6. Layer loop ──
        let mut layer_prof = if unified_layer_prof_selected(m_total, num_seqs) {
            Some(UnifiedLayerProfile::default())
        } else {
            None
        };
        for li in 0..num_layers {
            self.unified_forward_layer(
                &mut ctx,
                li,
                &mut residual,
                m_total,
                num_seqs,
                max_kv_len,
                max_blocks_per_seq,
                block_size,
                nh,
                nkv,
                hd,
                h,
                inter,
                eps,
                qk_mode,
                top_k,
                n_exp,
                norm_topk_prob,
                use_vllm_tiled_q4,
                use_fa_layout_tiled_q4,
                tile_q4_count,
                layer_prof.as_mut(),
            );
        }

        // ── 7. Final rms_norm + lm_head on per-item last tokens ──
        let prof_enabled = layer_prof.is_some();
        let t_final_norm = unified_stage_start::<B>(&mut ctx, prof_enabled);
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            eps,
            &mut self.scratch.norm_out,
            m_total,
            h,
        );
        if let Some(prof) = layer_prof.as_mut() {
            prof.final_norm_us += unified_stage_finish::<B>(&mut ctx, t_final_norm);
        }

        if num_sampled > 0 {
            let packed_normed = self
                .scratch
                .unified_packed_normed
                .as_mut()
                .expect("unified_packed_normed missing");
            let t_sample_gather = unified_stage_start::<B>(&mut ctx, prof_enabled);
            for (j, &(_, global)) in final_indices.iter().enumerate() {
                B::copy_slice(
                    &mut ctx,
                    &self.scratch.norm_out,
                    global * h,
                    packed_normed,
                    j * h,
                    h,
                );
            }
            if let Some(prof) = layer_prof.as_mut() {
                prof.sample_gather_us += unified_stage_finish::<B>(&mut ctx, t_sample_gather);
            }
            let packed_logits = self
                .scratch
                .unified_packed_logits
                .as_mut()
                .expect("unified_packed_logits missing");
            let t_lm_head = unified_stage_start::<B>(&mut ctx, prof_enabled);
            self.lm_head
                .forward(&mut ctx, packed_normed, packed_logits, num_sampled);
            if let Some(prof) = layer_prof.as_mut() {
                prof.lm_head_us += unified_stage_finish::<B>(&mut ctx, t_lm_head);
            }
        }

        // ── 8. Sync + readback per-item logits ──
        let t_readback = unified_stage_start::<B>(&mut ctx, prof_enabled);
        B::sync(&mut ctx);
        let mut out: Vec<Option<Vec<f32>>> = (0..items.len()).map(|_| None).collect();
        if num_sampled > 0 {
            let packed_logits = self
                .scratch
                .unified_packed_logits
                .as_ref()
                .expect("unified_packed_logits missing");
            if unified_greedy_argmax_enabled() {
                let tokens = B::argmax_rows_f16(&mut ctx, packed_logits, num_sampled, vocab)
                    .expect("Qwen3Moe unified: argmax_rows_f16");
                for (j, &(orig_idx, _)) in final_indices.iter().enumerate() {
                    out[orig_idx] = Some(vec![tokens[j] as f32]);
                }
            } else {
                let logits_flat = B::to_vec(packed_logits, num_sampled * vocab);
                for (j, &(orig_idx, _)) in final_indices.iter().enumerate() {
                    let row = logits_flat[j * vocab..(j + 1) * vocab].to_vec();
                    out[orig_idx] = Some(row);
                }
            }
        }
        if let Some(prof) = layer_prof.as_mut() {
            prof.readback_us += unified_stage_finish::<B>(&mut ctx, t_readback);
            log_unified_layer_profile(prof, m_total, num_seqs, num_sampled, num_layers);
        }

        // ── 9. Bump cache.len per item ──
        for (i, (cid, _, _, _)) in items.iter().enumerate() {
            let caches = self
                .kv_caches
                .get_mut(cid)
                .expect("kv cache missing post-loop");
            for c in caches.iter_mut() {
                c.len += q_lens[i];
            }
        }

        // ── 10. Restore residual ──
        self.scratch.residual = Some(residual);

        out
    }

    /// One transformer layer for the unified forward. Mirrors Llama's
    /// `unified_forward_layer` for steps 1-6 (shared attention path) and
    /// dispatches MoE via `moe_forward_bucketed` for the FFN block.
    /// Uses LEGACY scratch fields (sized for `max_tokens`) — no per-call
    /// realloc, no buffer duplication.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn unified_forward_layer(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        residual: &mut B::Buffer,
        m_total: usize,
        num_seqs: usize,
        max_kv_len: usize,
        max_blocks_per_seq: usize,
        block_size: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        h: usize,
        inter: usize,
        eps: f32,
        qk_mode: i32,
        top_k: usize,
        n_exp: usize,
        norm_topk_prob: bool,
        use_vllm_tiled_q4: bool,
        use_fa_layout_tiled_q4: bool,
        tile_q4_count: usize,
        mut prof: Option<&mut UnifiedLayerProfile>,
    ) {
        let attn_layer = &self.attn_layers[li];
        let moe_layer = &self.moe_layers[li];
        let dummy_w = &attn_layer.input_ln_w;
        let q_norm_w = attn_layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = attn_layer.k_norm_w.as_ref().unwrap_or(dummy_w);

        // 1. Input rms_norm [m_total, h] → scratch.norm_out
        let prof_enabled = prof.is_some();
        let t_input_norm = unified_stage_start::<B>(ctx, prof_enabled);
        B::rms_norm(
            ctx,
            residual,
            &attn_layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m_total,
            h,
        );
        if let Some(prof) = prof.as_mut() {
            prof.input_norm_us += unified_stage_finish::<B>(ctx, t_input_norm);
        }

        // 2. qkv_proj GEMM (m = M_total): norm_out → qkv_out
        let t_qkv = unified_stage_start::<B>(ctx, prof_enabled);
        attn_layer.qkv_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.qkv_out,
            m_total,
        );
        if let Some(prof) = prof.as_mut() {
            prof.qkv_us += unified_stage_finish::<B>(ctx, t_qkv);
        }

        // 3. varlen split_qkv_norm_rope_into_paged_cache
        let pools = self
            .paged_pools
            .as_mut()
            .expect("unified_forward_layer requires paged_pools");
        let pool_ptr = (
            &mut pools[li].0 as *mut B::Buffer,
            &mut pools[li].1 as *mut B::Buffer,
        );
        let use_fa_layout_varlen = self.use_vllm_paged_attn && fa_layout_varlen_enabled();
        let fa_pool_ptr = if use_fa_layout_varlen {
            let pools = self
                .paged_fa_pools
                .as_mut()
                .expect("FERRUM_FA_LAYOUT_VARLEN=1 requires paged_fa_pools");
            Some((
                &mut pools[li].0 as *mut B::Buffer,
                &mut pools[li].1 as *mut B::Buffer,
            ))
        } else {
            None
        };
        // SAFETY: pools allocated once, no concurrent mutation.
        let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };

        {
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
            let varlen_dispatch = if self.use_vllm_paged_attn {
                B::split_qkv_norm_rope_into_paged_cache_varlen_vllm
            } else {
                B::split_qkv_norm_rope_into_paged_cache_varlen
            };
            let t_split_cache = unified_stage_start::<B>(ctx, prof_enabled);
            varlen_dispatch(
                ctx,
                &self.scratch.qkv_out,
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                &mut self.scratch.q_head_major,
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
            .expect("Qwen3Moe unified: split_qkv_norm_rope_into_paged_cache_varlen");
            if let Some((fa_k_ptr, fa_v_ptr)) = fa_pool_ptr {
                // Maintain an FA-compatible K/V view in parallel with the
                // vLLM decode layout. The block table is shared, so release
                // and cache lifetime rules stay unchanged.
                let (fa_pool_k, fa_pool_v) = unsafe { (&mut *fa_k_ptr, &mut *fa_v_ptr) };
                B::split_qkv_norm_rope_into_paged_cache_varlen(
                    ctx,
                    &self.scratch.qkv_out,
                    q_norm_w,
                    k_norm_w,
                    &self.rope.cos,
                    &self.rope.sin,
                    &mut self.scratch.q_head_major,
                    fa_pool_k,
                    fa_pool_v,
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
                .expect("Qwen3Moe unified: split_qkv_norm_rope_into_paged_cache_varlen fa-layout");
            }
            if let Some(prof) = prof.as_mut() {
                prof.split_cache_us += unified_stage_finish::<B>(ctx, t_split_cache);
            }
        }

        // 4. paged_varlen_attention
        {
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
            let t_attn = unified_stage_start::<B>(ctx, prof_enabled);
            if self.use_vllm_paged_attn {
                if let Some((fa_k_ptr, fa_v_ptr)) = fa_pool_ptr {
                    let (fa_pool_k, fa_pool_v) = unsafe { (&mut *fa_k_ptr, &mut *fa_v_ptr) };
                    if use_fa_layout_tiled_q4 {
                        let tile_seqs = self
                            .scratch
                            .unified_tile_q4_seqs
                            .as_ref()
                            .expect("unified_tile_q4_seqs missing");
                        let tile_starts = self
                            .scratch
                            .unified_tile_q4_starts
                            .as_ref()
                            .expect("unified_tile_q4_starts missing");
                        B::paged_varlen_attention_tiled_q4(
                            ctx,
                            &self.scratch.q_head_major,
                            fa_pool_k,
                            fa_pool_v,
                            &mut self.scratch.attn_head_major_out,
                            cu_seqlens_buf,
                            pos_offsets_buf,
                            bt_buf,
                            tile_seqs,
                            tile_starts,
                            tile_q4_count,
                            max_kv_len,
                            nh,
                            nkv,
                            hd,
                            block_size,
                            max_blocks_per_seq,
                        )
                        .expect("Qwen3Moe unified: paged_varlen_attention_tiled_q4 fa-layout");
                    } else {
                        B::paged_varlen_attention(
                            ctx,
                            &self.scratch.q_head_major,
                            fa_pool_k,
                            fa_pool_v,
                            &mut self.scratch.attn_head_major_out,
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
                        .expect("Qwen3Moe unified: paged_varlen_attention fa-layout");
                    }
                } else if use_vllm_tiled_q4 {
                    let tile_seqs = self
                        .scratch
                        .unified_tile_q4_seqs
                        .as_ref()
                        .expect("unified_tile_q4_seqs missing");
                    let tile_starts = self
                        .scratch
                        .unified_tile_q4_starts
                        .as_ref()
                        .expect("unified_tile_q4_starts missing");
                    B::paged_varlen_attention_vllm_tiled_q4(
                        ctx,
                        &self.scratch.q_head_major,
                        pool_k,
                        pool_v,
                        &mut self.scratch.attn_head_major_out,
                        cu_seqlens_buf,
                        pos_offsets_buf,
                        bt_buf,
                        tile_seqs,
                        tile_starts,
                        tile_q4_count,
                        max_kv_len,
                        nh,
                        nkv,
                        hd,
                        block_size,
                        max_blocks_per_seq,
                    )
                    .expect("Qwen3Moe unified: paged_varlen_attention_vllm_tiled_q4");
                } else {
                    B::paged_varlen_attention_vllm(
                        ctx,
                        &self.scratch.q_head_major,
                        pool_k,
                        pool_v,
                        &mut self.scratch.attn_head_major_out,
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
                    .expect("Qwen3Moe unified: paged_varlen_attention_vllm");
                }
            } else {
                B::paged_varlen_attention(
                    ctx,
                    &self.scratch.q_head_major,
                    pool_k,
                    pool_v,
                    &mut self.scratch.attn_head_major_out,
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
                .expect("Qwen3Moe unified: paged_varlen_attention");
            }
            if let Some(prof) = prof.as_mut() {
                prof.attn_us += unified_stage_finish::<B>(ctx, t_attn);
            }
        }

        // 5. o_proj (m = M_total): attn_head_major_out → o_proj_out
        let t_o_proj = unified_stage_start::<B>(ctx, prof_enabled);
        attn_layer.o_proj.forward(
            ctx,
            &self.scratch.attn_head_major_out,
            &mut self.scratch.o_proj_out,
            m_total,
        );
        if let Some(prof) = prof.as_mut() {
            prof.o_proj_us += unified_stage_finish::<B>(ctx, t_o_proj);
        }

        // 6. fused_add_rms_norm: residual += o_proj_out; norm → scratch.norm_out
        //    (reusing scratch.norm_out as the MoE input — the legacy MoE
        //    forward also reads scratch.norm_out, so this naturally
        //    feeds the next step.)
        let t_post_norm = unified_stage_start::<B>(ctx, prof_enabled);
        B::fused_add_rms_norm(
            ctx,
            residual,
            &self.scratch.o_proj_out,
            &attn_layer.post_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m_total,
            h,
        );
        if let Some(prof) = prof.as_mut() {
            prof.post_norm_us += unified_stage_finish::<B>(ctx, t_post_norm);
        }

        // 7. Router GEMM: norm_out → router_logits
        let t_router = unified_stage_start::<B>(ctx, prof_enabled);
        moe_layer.router.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.router_logits,
            m_total,
        );
        if let Some(prof) = prof.as_mut() {
            prof.router_us += unified_stage_finish::<B>(ctx, t_router);
        }

        // 8. MoE forward (bucketed CUDA path).
        //    Reads scratch.norm_out + scratch.router_logits. Writes scratch.moe_out.
        //    M = m_total; MoE routes each token independently (no per-request boundary).
        let t_moe = unified_stage_start::<B>(ctx, prof_enabled);
        crate::moe::moe_forward_bucketed::<B>(
            ctx,
            &self.scratch.norm_out,
            &self.scratch.router_logits,
            &mut self.scratch.moe_out,
            m_total,
            h,
            inter,
            n_exp,
            top_k,
            norm_topk_prob,
            &moe_layer.experts,
            &mut self.scratch.x_packed_bucket,
            &mut self.scratch.gate_up_packed_bucket,
            &mut self.scratch.silu_stacked,
            &mut self.scratch.down_out_stacked,
            &mut self.scratch.moe_route_scratch,
            Some(crate::moe::dispatch::DeviceRouteScratch {
                selected_ids: &mut self.scratch.selected_ids_buf,
                pair_weights: &mut self.scratch.route_pair_weights_dev,
                pairs_by_token: &mut self.scratch.route_pairs_dev,
                packed_token_idx: &mut self.scratch.route_packed_idx_dev,
                expert_offsets: &mut self.scratch.route_expert_offsets_dev,
                sorted_tokens: &mut self.scratch.route_sorted_tokens_dev,
                block_ids: &mut self.scratch.route_block_ids_dev,
                total_post_pad: &mut self.scratch.route_total_post_pad_dev,
            }),
        )
        .expect("Qwen3Moe unified: moe_forward_bucketed");
        if let Some(prof) = prof.as_mut() {
            prof.moe_us += unified_stage_finish::<B>(ctx, t_moe);
        }

        // 9. residual += moe_out
        let t_residual_add = unified_stage_start::<B>(ctx, prof_enabled);
        B::add_inplace(ctx, residual, &self.scratch.moe_out, m_total * h);
        if let Some(prof) = prof.as_mut() {
            prof.residual_add_us += unified_stage_finish::<B>(ctx, t_residual_add);
        }
    }
}
