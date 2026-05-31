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

use ferrum_bench_core::{global_profile, profile_fields_from_json};
use ferrum_interfaces::kv_dtype::KvDtypeKind;
use ferrum_kernels::backend::{Backend, BackendPagedKv, MoeLlmBackend};

use super::qwen3_moe::Qwen3MoeModel;
use super::qwen3_moe_forward_unified_layer::UnifiedLayerParams;

#[derive(Default)]
pub(crate) struct UnifiedLayerProfile {
    pub(super) input_norm_us: u64,
    pub(super) qkv_us: u64,
    pub(super) split_cache_us: u64,
    pub(super) attn_us: u64,
    pub(super) o_proj_us: u64,
    pub(super) post_norm_us: u64,
    pub(super) router_us: u64,
    pub(super) moe_us: u64,
    pub(super) residual_add_us: u64,
    pub(super) final_norm_us: u64,
    pub(super) sample_gather_us: u64,
    pub(super) lm_head_us: u64,
    pub(super) readback_us: u64,
}

static UNIFIED_LAYER_PROF_CALLS: AtomicU64 = AtomicU64::new(0);

pub(super) fn unified_stage_start<B: Backend>(
    ctx: &mut B::Context,
    enabled: bool,
) -> Option<Instant> {
    if enabled {
        B::sync(ctx);
        Some(Instant::now())
    } else {
        None
    }
}

pub(super) fn unified_stage_finish<B: Backend>(
    ctx: &mut B::Context,
    start: Option<Instant>,
) -> u64 {
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
    every: u64,
) {
    let call = UNIFIED_LAYER_PROF_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
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
    let profile = global_profile();
    if profile.is_enabled() {
        let _ = profile.push_event(
            "unified_layer_prof",
            profile_fields_from_json(serde_json::json!({
                "call": call,
                "m": m_total,
                "seqs": num_seqs,
                "sampled": num_sampled,
                "layers": num_layers,
            })),
            profile_fields_from_json(serde_json::json!({
                "layer_sum": layer_sum,
                "final_sum": final_sum,
                "input_norm": prof.input_norm_us,
                "qkv": prof.qkv_us,
                "split_cache": prof.split_cache_us,
                "attn": prof.attn_us,
                "o_proj": prof.o_proj_us,
                "post_norm": prof.post_norm_us,
                "router": prof.router_us,
                "moe": prof.moe_us,
                "residual_add": prof.residual_add_us,
                "final_norm": prof.final_norm_us,
                "sample_gather": prof.sample_gather_us,
                "lm_head": prof.lm_head_us,
                "readback": prof.readback_us,
            })),
            false,
        );
    }
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
        let seq_lens: Vec<u32> = pos_offsets
            .iter()
            .zip(q_lens.iter())
            .map(|(&pos, &q_len)| pos + q_len as u32)
            .collect();
        let max_q_len = q_lens.iter().copied().max().unwrap_or(0);
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
        {
            let sl = self
                .scratch
                .unified_seq_lens
                .as_mut()
                .expect("unified_seq_lens missing");
            B::write_typed::<u32>(&mut ctx, sl, &seq_lens);
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
            self.use_vllm_paged_attn && self.runtime_env.vllm_varlen_tiled_q4 && m_total > num_seqs;
        let mut tile_q4_count = 0usize;
        if use_vllm_tiled_q4 {
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
        let mut layer_prof = if self
            .runtime_env
            .unified_layer_prof_selected(m_total, num_seqs)
        {
            Some(UnifiedLayerProfile::default())
        } else {
            None
        };
        for li in 0..num_layers {
            self.unified_forward_layer(UnifiedLayerParams {
                ctx: &mut ctx,
                li,
                residual: &mut residual,
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
                prof: layer_prof.as_mut(),
            });
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
            if self.runtime_env.unified_greedy_argmax {
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
            log_unified_layer_profile(
                prof,
                m_total,
                num_seqs,
                num_sampled,
                num_layers,
                self.runtime_env.unified_layer_prof_every,
            );
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
}
