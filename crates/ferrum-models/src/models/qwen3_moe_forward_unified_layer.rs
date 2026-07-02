//! One Qwen3-MoE unified forward transformer layer.

use ferrum_interfaces::kv_dtype::KvDtypeKind;
use ferrum_kernels::backend::{Backend, BackendPagedKv, MoeLlmBackend};

use super::qwen3_moe::Qwen3MoeModel;
use super::qwen3_moe_forward_unified::{
    unified_stage_finish, unified_stage_start, UnifiedLayerProfile,
};

pub(crate) struct UnifiedLayerParams<'a, B: Backend> {
    pub ctx: &'a mut B::Context,
    pub li: usize,
    pub residual: &'a mut B::Buffer,
    pub m_total: usize,
    pub num_seqs: usize,
    pub max_q_len: usize,
    pub max_kv_len: usize,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub h: usize,
    pub inter: usize,
    pub eps: f32,
    pub qk_mode: i32,
    pub top_k: usize,
    pub n_exp: usize,
    pub norm_topk_prob: bool,
    pub use_vllm_tiled_q4: bool,
    pub tile_q4_count: usize,
    pub prof: Option<&'a mut UnifiedLayerProfile>,
}

impl<B, K> Qwen3MoeModel<B, K>
where
    B: MoeLlmBackend + BackendPagedKv,
    K: KvDtypeKind,
{
    /// One transformer layer for the unified forward. Mirrors Llama's
    /// `unified_forward_layer` for steps 1-6 (shared attention path) and
    /// dispatches MoE via `moe_forward_bucketed` for the FFN block.
    /// Uses LEGACY scratch fields (sized for `max_tokens`) — no per-call
    /// realloc, no buffer duplication.
    pub(crate) fn unified_forward_layer(&mut self, params: UnifiedLayerParams<'_, B>) {
        let UnifiedLayerParams {
            ctx,
            li,
            residual,
            m_total,
            num_seqs,
            max_q_len,
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
            tile_q4_count,
            mut prof,
        } = params;
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
        let use_fa_layout_varlen = self.use_vllm_paged_attn
            && (self.runtime_env.fa_layout_varlen
                || self.runtime_env.fa2_direct_ffi
                || self.runtime_env.fa2_source);
        let fa_pool_ptr = if use_fa_layout_varlen {
            let pools = self
                .paged_fa_pools
                .as_mut()
                .expect("FA-layout varlen or FA2 requires paged_fa_pools");
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
            let seq_lens_buf = self
                .scratch
                .unified_seq_lens
                .as_ref()
                .expect("unified_seq_lens missing");
            let bt_buf = self
                .scratch
                .unified_block_tables
                .as_ref()
                .expect("unified_block_tables missing");
            let t_attn = unified_stage_start::<B>(ctx, prof_enabled);
            if self.use_vllm_paged_attn {
                if let Some((fa_k_ptr, fa_v_ptr)) = fa_pool_ptr {
                    let (fa_pool_k, fa_pool_v) = unsafe { (&mut *fa_k_ptr, &mut *fa_v_ptr) };
                    if self.runtime_env.fa2_direct_ffi || self.runtime_env.fa2_source {
                        let lse_buf = self
                            .scratch
                            .unified_attn_lse
                            .as_mut()
                            .expect("unified_attn_lse missing");
                        B::paged_varlen_attention_fa2_ffi(
                            ctx,
                            &self.scratch.q_head_major,
                            fa_pool_k,
                            fa_pool_v,
                            &mut self.scratch.attn_head_major_out,
                            lse_buf,
                            cu_seqlens_buf,
                            seq_lens_buf,
                            bt_buf,
                            num_seqs,
                            m_total,
                            max_q_len,
                            max_kv_len,
                            nh,
                            nkv,
                            hd,
                            block_size,
                            max_blocks_per_seq,
                        )
                        .expect("Qwen3Moe unified: paged_varlen_attention_fa2_ffi");
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
                            0,
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
                    0,
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
        crate::moe::moe_forward_bucketed::<B>(crate::moe::dispatch::MoeForwardBucketedParams {
            ctx,
            x: &self.scratch.norm_out,
            router_logits: &self.scratch.router_logits,
            out: &mut self.scratch.moe_out,
            batch: m_total,
            hidden_size: h,
            expert_intermediate: inter,
            num_experts: n_exp,
            top_k,
            norm_topk_prob,
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
        })
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
