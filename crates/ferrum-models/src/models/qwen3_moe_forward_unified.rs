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

use ferrum_interfaces::kv_dtype::KvDtypeKind;
use ferrum_kernels::backend::{Backend, BackendPagedKv, MoeLlmBackend};

use super::qwen3_moe::Qwen3MoeModel;

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

        // Pre-grow Marlin gather scratch for worst-case GEMM input m.
        let max_marlin_required = m_total * inter;
        B::pregrow_marlin_gather_scratch(&mut ctx, max_marlin_required);
        B::sync(&mut ctx);

        // ── 6. Layer loop ──
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
            );
        }

        // ── 7. Final rms_norm + lm_head on per-item last tokens ──
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            eps,
            &mut self.scratch.norm_out,
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
                B::copy_slice(
                    &mut ctx,
                    &self.scratch.norm_out,
                    global * h,
                    packed_normed,
                    j * h,
                    h,
                );
            }
            let packed_logits = self
                .scratch
                .unified_packed_logits
                .as_mut()
                .expect("unified_packed_logits missing");
            self.lm_head
                .forward(&mut ctx, packed_normed, packed_logits, num_sampled);
        }

        // ── 8. Sync + readback per-item logits ──
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
    ) {
        let attn_layer = &self.attn_layers[li];
        let moe_layer = &self.moe_layers[li];
        let dummy_w = &attn_layer.input_ln_w;
        let q_norm_w = attn_layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = attn_layer.k_norm_w.as_ref().unwrap_or(dummy_w);

        // 1. Input rms_norm [m_total, h] → scratch.norm_out
        B::rms_norm(
            ctx,
            residual,
            &attn_layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m_total,
            h,
        );

        // 2. qkv_proj GEMM (m = M_total): norm_out → qkv_out
        attn_layer.qkv_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.qkv_out,
            m_total,
        );

        // 3. varlen split_qkv_norm_rope_into_paged_cache
        let pools = self
            .paged_pools
            .as_mut()
            .expect("unified_forward_layer requires paged_pools");
        let pool_ptr = (
            &mut pools[li].0 as *mut B::Buffer,
            &mut pools[li].1 as *mut B::Buffer,
        );
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
            if self.use_vllm_paged_attn {
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
        }

        // 5. o_proj (m = M_total): attn_head_major_out → o_proj_out
        attn_layer.o_proj.forward(
            ctx,
            &self.scratch.attn_head_major_out,
            &mut self.scratch.o_proj_out,
            m_total,
        );

        // 6. fused_add_rms_norm: residual += o_proj_out; norm → scratch.norm_out
        //    (reusing scratch.norm_out as the MoE input — the legacy MoE
        //    forward also reads scratch.norm_out, so this naturally
        //    feeds the next step.)
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

        // 7. Router GEMM: norm_out → router_logits
        moe_layer.router.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.router_logits,
            m_total,
        );

        // 8. MoE forward (bucketed CUDA path).
        //    Reads scratch.norm_out + scratch.router_logits. Writes scratch.moe_out.
        //    M = m_total; MoE routes each token independently (no per-request boundary).
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

        // 9. residual += moe_out
        B::add_inplace(ctx, residual, &self.scratch.moe_out, m_total * h);
    }
}
