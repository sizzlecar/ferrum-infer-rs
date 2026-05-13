//! Qwen3-MoE unified mixed-batch forward path.
//!
//! Mirrors `llama_family_forward_batched.rs::unified_forward_internal`
//! using the shared `common::decoder_unified` helpers. The only
//! family-specific piece is `unified_forward_layer`'s FFN block, which
//! dispatches MoE expert kernels via `moe_forward_batched_prefill_impl`
//! instead of Llama's MLP gate_up/silu/down chain.
//!
//! Phase 2B per `docs/continuous-batching-redesign.md`. Eager-only;
//! CUDA graph capture is a follow-up (memory `project_moe_phase3_graph_bug.md`
//! is relevant context — graph regresses c=32 by -6% on the ~480-node
//! MoE graph today, so adding graph capture here without re-tuning
//! cuGraphLaunch overhead would be a perf regression).

use ferrum_interfaces::kv_dtype::KvDtypeKind;
use ferrum_kernels::backend::{Backend, BackendMoeFused, BackendPagedKv, MoeLlmBackend, QuantLlmBackend};

use super::qwen3_moe::Qwen3MoeModel;

impl<B, K> Qwen3MoeModel<B, K>
where
    B: MoeLlmBackend
        + QuantLlmBackend
        + BackendMoeFused
        + BackendPagedKv
        + ferrum_kernels::backend::BackendKvDtype<K>,
    K: KvDtypeKind,
{
    /// Unified mixed-batch forward for Qwen3-MoE. See
    /// `LlamaFamilyModel::unified_forward_internal` for the reference
    /// design — this function is its sibling for MoE families, sharing
    /// the same per-iter scaffolding (cu_seqlens, embedding, layer
    /// loop, final norm + lm_head, KV bump) but substituting an MoE
    /// FFN block for the dense MLP at each layer.
    ///
    /// Returns one `Option<Vec<f32>>` per `items[i]`:
    /// - `Some(logits)` iff `items[i].3 == true` (is_final_chunk)
    /// - `None` for intermediate prefill chunks
    pub(crate) fn unified_forward_internal(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Vec<Option<Vec<f32>>> {
        if items.is_empty() {
            return Vec::new();
        }

        // ── 1. Per-item bookkeeping via shared helpers ──
        let (q_lens, cu_seqlens_q, m_total) =
            crate::common::decoder_unified::compute_cu_seqlens_q(items);
        let pos_offsets = crate::common::decoder_unified::compute_pos_offsets(items);
        let max_kv_len = crate::common::decoder_unified::compute_max_kv_len(items);
        let num_seqs = items.len();
        let final_indices =
            crate::common::decoder_unified::compute_final_indices(items, &cu_seqlens_q);
        let num_sampled = final_indices.len();

        // ── 2. Ensure KV caches exist for all items ──
        for (cid, _, _, _) in items {
            self.ensure_kv(cid);
        }
        if self.paged_pools.is_none() {
            // Caller (DecoderOnlyLLM::unified_forward) must have already
            // checked `paged_pools.is_some()` — defensive.
            panic!(
                "Qwen3MoeModel::unified_forward_internal called without paged_pools; \
                 set FERRUM_METAL_PAGED_KV=1"
            );
        }

        // ── 3. Snapshot config + ensure scratch ──
        let h = self.cfg.base.hidden_size;
        let nh = self.cfg.base.num_heads;
        let nkv = self.cfg.base.num_kv_heads;
        let hd = self.cfg.base.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
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
            .expect("paged_dims missing — ensure_kv must have set this");
        let block_size = 16usize; // matches PAGED_BLOCK_SIZE in ensure_kv
        debug_assert!(
            paged_max_seqs >= num_seqs,
            "unified_forward batch ({} items) exceeds paged_max_seqs ({})",
            num_seqs,
            paged_max_seqs,
        );
        let max_seqs = paged_max_seqs.max(num_seqs);
        let cfg_clone = self.cfg.clone();
        self.scratch
            .ensure_unified_scratch(&cfg_clone, m_total, max_seqs, max_blocks_per_seq);

        let mut ctx = B::new_context();

        // ── 4. Concat all q-tokens, embedding_lookup into residual ──
        let mut residual = self
            .scratch
            .unified_residual
            .take()
            .expect("unified_residual missing after ensure");
        let all_tokens = crate::common::decoder_unified::concat_q_tokens(items);
        debug_assert_eq!(all_tokens.len(), m_total);
        B::embedding_lookup(&mut ctx, &self.embed, &all_tokens, &mut residual, h);

        // ── 5. Upload index buffers (cu_seqlens, pos_offsets, block_tables) ──
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
        let stacked = crate::common::decoder_unified::stack_block_tables(
            items,
            max_blocks_per_seq,
            |cid| {
                self.kv_caches
                    .get(cid)
                    .expect("kv cache missing for unified item")[0]
                    .paged_block_indices
                    .clone()
            },
        );
        {
            let bt = self
                .scratch
                .unified_block_tables
                .as_mut()
                .expect("unified_block_tables missing");
            B::write_typed::<u32>(&mut ctx, bt, &stacked);
        }

        // Pre-grow MoE GEMM scratch (Marlin gather + c_tmp) BEFORE any
        // captured region — same rationale as Llama's
        // `pregrow_marlin_gather_scratch`. Worst-case is the down-proj's
        // input size = m_total * expert_intermediate_size; for MoE the
        // gate_up is similar so we use the larger of the two.
        let max_marlin_required = m_total * inter;
        B::pregrow_marlin_gather_scratch(&mut ctx, max_marlin_required);
        B::sync(&mut ctx);

        // ── 6. Layer loop (eager; graph capture deferred to next iteration) ──
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
                q_dim,
                kv_dim,
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
        let final_norm_w = &self.final_norm_w;
        let mut norm_out = self
            .scratch
            .unified_norm_out
            .take()
            .expect("unified_norm_out missing");

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
            let packed_logits = self
                .scratch
                .unified_packed_logits
                .as_mut()
                .expect("unified_packed_logits missing");
            self.lm_head
                .forward(&mut ctx, packed_normed, packed_logits, num_sampled);
        }

        // ── 8. Sync + readback (host-side logits) ──
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

        // ── 9. Bump cache.len for each item (wrote q_lens[i] tokens) ──
        for (i, (cid, _, _, _)) in items.iter().enumerate() {
            let caches = self
                .kv_caches
                .get_mut(cid)
                .expect("kv cache missing for unified item post-loop");
            for c in caches.iter_mut() {
                c.len += q_lens[i];
            }
        }

        // ── 10. Restore scratch for next call ──
        self.scratch.unified_residual = Some(residual);
        self.scratch.unified_norm_out = Some(norm_out);

        out
    }

    /// One transformer layer for the unified mixed-batch forward.
    /// Steps 1-6 mirror Llama's `unified_forward_layer` (shared attention
    /// path via varlen kernels). Steps 7-10 are Qwen3-MoE's expert
    /// dispatch in place of the dense MLP gate_up / silu / down chain.
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
        q_dim: usize,
        kv_dim: usize,
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
        let _ = kv_dim;

        // 1. Input rms_norm [M_total, h] → unified_norm_out
        {
            let norm_out = self
                .scratch
                .unified_norm_out
                .as_mut()
                .expect("unified_norm_out missing");
            B::rms_norm(ctx, residual, &attn_layer.input_ln_w, eps, norm_out, m_total, h);
        }

        // 2. qkv_proj GEMM (m = M_total): unified_norm_out → unified_qkv_out
        {
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
            attn_layer
                .qkv_proj
                .forward(ctx, norm_out, qkv_out, m_total);
        }

        // 3. Single-launch varlen split_qkv_norm_rope_into_paged_cache
        let pools = self
            .paged_pools
            .as_mut()
            .expect("unified_forward_layer requires paged_pools");
        let pool_ptr = (
            &mut pools[li].0 as *mut B::Buffer,
            &mut pools[li].1 as *mut B::Buffer,
        );
        // SAFETY: pools allocated once; no concurrent mutation.
        let (pool_k, pool_v) = unsafe { (&mut *pool_ptr.0, &mut *pool_ptr.1) };

        {
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
            .expect("Qwen3Moe unified: split_qkv_norm_rope_into_paged_cache_varlen");
        }

        // 4. paged_varlen_attention: one call covering all M_total tokens.
        {
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
            .expect("Qwen3Moe unified: paged_varlen_attention");
        }

        // 5. o_proj GEMM (m = M_total): attn_out → o_proj_out
        {
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
            attn_layer
                .o_proj
                .forward(ctx, attn_out, o_proj_out, m_total);
        }

        // 6. fused_add_rms_norm: residual += o_proj_out; then norm into
        //    unified_moe_input for the MoE block's input.
        {
            let o_proj_out = self
                .scratch
                .unified_o_proj_out
                .as_ref()
                .expect("unified_o_proj_out missing");
            let moe_input = self
                .scratch
                .unified_moe_input
                .as_mut()
                .expect("unified_moe_input missing");
            B::fused_add_rms_norm(
                ctx,
                residual,
                o_proj_out,
                &attn_layer.post_ln_w,
                eps,
                moe_input,
                m_total,
                h,
            );
        }

        // 7. Router logits: moe_input → unified_router_logits.
        //    GEMM at m = M_total.
        {
            let moe_input = self
                .scratch
                .unified_moe_input
                .as_ref()
                .expect("unified_moe_input missing");
            let router_logits = self
                .scratch
                .unified_router_logits
                .as_mut()
                .expect("unified_router_logits missing");
            moe_layer
                .router
                .forward(ctx, moe_input, router_logits, m_total);
        }

        // 8. MoE expert dispatch via the existing batched-prefill impl.
        //    This handles route_topk_softmax + compute_ids_tpe (or vLLM
        //    align_block) + per-expert Marlin GEMM + combine, all at
        //    m = M_total. The function reads from `scratch.router_logits`
        //    and writes the combined output into a target buffer the
        //    caller specifies.
        //
        //    For now, copy unified_router_logits into the existing
        //    scratch.router_logits buffer that moe_forward_batched_prefill_impl
        //    reads from. (Future: refactor moe_forward_batched_prefill_impl
        //    to read its inputs by parameter so we don't need this copy.)
        //
        // TODO(Phase 2B step 2): wire moe_forward_batched_prefill_impl
        // for the unified path. The current legacy path uses
        // `scratch.router_logits` + `scratch.norm_out` as inputs and
        // `scratch.moe_out` as output. Unified path uses
        // `unified_router_logits` + `unified_moe_input` + `unified_moe_out`.
        // Re-route through additional buffer parameters or temporary
        // pointer swap.
        let _ = (top_k, n_exp, norm_topk_prob, moe_layer);

        // 9. Residual add: unified_moe_out → residual
        // 10. (Next layer's input_rms_norm picks up residual.)
        // TODO(Phase 2B step 2): once step 8 writes to unified_moe_out,
        // enable:
        //   let moe_out = self.scratch.unified_moe_out.as_ref().unwrap();
        //   B::residual_add(ctx, residual, moe_out, m_total, h);

        let _ = inter; // silence until step 8 wired

        // INCOMPLETE — unified_forward_layer returns without doing the MoE
        // block. The caller (unified_forward_internal) will produce wrong
        // logits. This commit lands the attention path + scaffolding;
        // step 8 (MoE wiring) is the next concrete change.
        panic!(
            "Qwen3MoeModel::unified_forward_layer: MoE block not yet wired. \
             Phase 2B step 2 TODO. Until then, unified_forward returns Unsupported \
             via the trait impl gate."
        );
    }
}
