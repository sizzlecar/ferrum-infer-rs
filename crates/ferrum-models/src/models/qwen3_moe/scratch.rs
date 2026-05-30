use super::*;

/// Per-layer MoE state: router linear (small) + per-expert MLP stack.
pub struct Qwen3MoeLayerState<B: QuantLlmBackend + BackendMoeFused> {
    /// Router projection `[hidden] → [num_experts]` — tiny, never sparse,
    /// always runs the full GEMV.
    pub router: Box<dyn ferrum_quantization::Linear<B>>,
    /// Per-expert weight stack. Each entry's `gate_up` is the fused
    /// `[gate; up]` projection; `down` is the post-SwiGLU output proj.
    pub experts: ExpertStack<B>,
}

/// Reusable scratch buffers for the MoE forward path. All sized at
/// allocation time and reused across layers / forward calls.
pub struct Qwen3MoeScratch<B: QuantLlmBackend + BackendMoeFused> {
    /// See [`crate::models::llama_family::LlamaFamilyScratch`] for the
    /// attention scratch — we re-use those names verbatim.
    pub residual: Option<B::Buffer>,
    pub norm_out: B::Buffer,
    pub qkv_out: B::Buffer,
    pub q_buf: B::Buffer,
    pub k_buf: B::Buffer,
    pub v_buf: B::Buffer,
    pub q_head_major: B::Buffer,
    pub k_head_major: B::Buffer,
    pub v_head_major: B::Buffer,
    pub attn_head_major_out: B::Buffer,
    pub attn_flat: B::Buffer,
    pub o_proj_out: B::Buffer,

    // ── MoE-specific scratch ─────────────────────────────────────────
    /// Router logits for the whole batch: `[max_tokens, num_experts]`.
    pub router_logits: B::Buffer,
    /// Per-(token, expert) gate||up projection output — `[2 * expert_inter]`.
    pub gate_up_buf: B::Buffer,
    /// SiLU(gate) * up scratch — `[expert_inter]`.
    pub silu_buf: B::Buffer,
    /// Per-(token, expert) down-projection output — `[hidden]`.
    pub down_buf: B::Buffer,
    /// Per-token input row scratch — `[hidden]`. Holds the post-RMSNorm
    /// activation slice that the per-(expert) gate_up gemv reads, kept
    /// stable across the entire top_k loop for one token.
    pub x_single: B::Buffer,
    /// Per-token output accumulator — `[hidden]`. Holds the running
    /// `Σ_k weight_k · expert_k(x[b])` sum that grows across the top_k
    /// loop and is flushed to `moe_out[b]` once per token.
    pub acc_buf: B::Buffer,
    /// MoE output `[max_tokens, hidden]`. Zeroed each forward.
    pub moe_out: B::Buffer,
    /// Pre-allocated `[hidden]` zero scratch — `acc_buf` is reset to
    /// this each token without going through `B::from_slice` on the
    /// hot path.
    pub zero_hidden: B::Buffer,

    // ── MoE batched-fast-path scratch (Metal `gemv_q*kw_moe_id_f32` /
    //    `gemm_q*kw_moe_id_f32`) ─────────────────────────────────────
    //
    // Sized for `max_tokens * top_k * X` so the same buffers cover both
    // decode (m=1, uses the first `top_k * X` slice) and prefill
    // (m>1, uses the full `max_tokens * top_k * X`). Decode-only
    // workloads pay no extra memory because `max_tokens` was 1 there.
    /// `[max_tokens * top_k * expert_inter]` — gate gemm output per pair.
    pub gate_out_stacked: B::Buffer,
    /// `[max_tokens * top_k * expert_inter]` — up gemm output per pair.
    pub up_out_stacked: B::Buffer,
    /// `[max_tokens * top_k * expert_inter]` — SiLU(gate)·up per pair.
    pub silu_stacked: B::Buffer,
    /// `[max_tokens * top_k * hidden]` — down gemm output per pair.
    pub down_out_stacked: B::Buffer,

    // ── Bucketed CUDA path scratch (shared with stacked Metal where
    //    layout matches; used by `moe_forward_bucketed`).
    /// `[max_tokens * top_k * hidden]` — input gather output: per-pair
    /// row gathered from `x[batch, hidden]` via embedding_lookup.
    pub x_packed_bucket: B::Buffer,
    /// `[max_tokens * top_k * 2 * expert_inter]` — gate_up Marlin output
    /// per pair (gate cols then up cols). Distinct from `gate_out_stacked`
    /// + `up_out_stacked` which the Metal path keeps separate.
    pub gate_up_packed_bucket: B::Buffer,
    /// `[top_k]` i32 expert IDs for the current token (decode reuses;
    /// prefill writes per-pair indices into `ids_2d` instead).
    pub ids_buf: B::Buffer,
    /// `[top_k]` f32 router combine weights for the current decode
    /// token. Decode hot-path uses `write_f32_into` to update.
    pub weights_buf: B::Buffer,
    /// `[max_tokens * top_k]` i32 — flat selected-expert IDs from the
    /// GPU router for the prefill batch. Consumed by `compute_ids_tpe_gpu`
    /// to bucket pairs by expert into `tpe_buf` / `ids_2d`.
    pub selected_ids_buf: B::Buffer,
    /// `[3]` u32 indirect-dispatch args (`grid_x, grid_y, grid_z`) for
    /// the gate / up MoE GEMM. Written by `compute_ids_tpe_gpu` so the
    /// consumer GEMM grid covers exactly `max(tpe[e])` columns instead
    /// of the worst-case `tokens * top_k`.
    pub gate_up_args_buf: B::Buffer,
    /// Same shape as `gate_up_args_buf` but for the down MoE GEMM
    /// (different `grid_y` because down's `M = hidden_size` vs gate/up's
    /// `M = expert_intermediate_size`).
    pub down_args_buf: B::Buffer,
    /// `[num_experts * max_per_expert_max]` i32 — per-expert pair
    /// index lists for prefill 2-D mul_mm_id. `max_per_expert_max`
    /// is bounded by `max_tokens * top_k` (worst-case: one expert
    /// gets every pair). Sized at scratch alloc time.
    pub ids_2d: B::Buffer,
    /// `[num_experts]` i32 — `tpe[e]` = number of pairs assigned to
    /// expert `e`. Companion to `ids_2d`.
    pub tpe_buf: B::Buffer,
    /// `[max_tokens * top_k]` f32 — combine weights per pair, in
    /// natural `[batch, top_k]` layout for `weighted_sum_batched`.
    pub weights_2d: B::Buffer,

    // ── Device-side routing scratch for graph-capturable MoE path ────
    //
    // Output of `B::moe_build_pairs_by_token` invoked on device-side
    // `selected_ids_buf` (which `B::route_topk_softmax` fills). When
    // these are populated, the bucketed-MoE forward can run without
    // any host round-trip — the prerequisite for CUDA Graph capture.
    /// `[max_tokens * top_k]` i32 — sorted-by-expert position of each
    /// (b, k) pair (row index into `down_packed` that `moe_combine`
    /// reads).
    pub route_pairs_dev: B::Buffer,
    /// `[max_tokens * top_k]` i32 — inverse of `route_pairs_dev`: for
    /// each packed row, the original token b. Used by gather
    /// (`embedding_lookup` x → x_packed before phase 1).
    pub route_packed_idx_dev: B::Buffer,
    /// `[num_experts + 1]` i32 — exclusive prefix sum of tokens-per-
    /// expert. Phase 1/3 dispatcher consults to compute each expert's
    /// row slice in the packed buffers.
    pub route_expert_offsets_dev: B::Buffer,
    /// `[max_tokens * top_k]` f32 — pair_weights from `route_topk_softmax`
    /// in float precision. Separate from `weights_2d` (which is F16 for
    /// the legacy per-pair path's `copy_slice` consumption) — the route
    /// kernel writes `float* out_weights`, so we need F32-byte capacity.
    pub route_pair_weights_dev: B::Buffer,

    // ── Device-side vLLM marlin_moe routing buffers ─────────────────
    //
    // Outputs of `B::moe_align_block_size` invoked on device-side
    // `selected_ids_buf`. Layout matches vLLM's marlin_moe_wna16
    // kernel input — same shape as host `vllm_routing` builder
    // produces, but built entirely on-device so the GEMM phases can
    // be captured.
    /// `[max_tokens * top_k + num_experts * VLLM_MOE_BLOCK_SIZE]` i32
    /// — flat list of pair indices in `[0, batch*top_k)`, sorted by
    /// expert and padded with sentinel inside each expert group to
    /// `VLLM_MOE_BLOCK_SIZE=16` boundary. Worst-case sized assuming
    /// each expert needs up to (block_size-1) padding.
    pub route_sorted_tokens_dev: B::Buffer,
    /// `[sorted_tokens / VLLM_MOE_BLOCK_SIZE + num_experts]` i32 —
    /// which expert each block of `sorted_tokens` belongs to.
    pub route_block_ids_dev: B::Buffer,
    /// `[1]` i32 — actual padded token count (`total_padded`). Read
    /// by `gemm_phase_vllm` as `num_tokens_post_padded[0]`.
    pub route_total_post_pad_dev: B::Buffer,

    // ── Final-token / lm_head outputs ────────────────────────────────
    pub last_hidden: B::Buffer,
    pub last_normed: B::Buffer,
    pub logits: B::Buffer,
    pub batch_logits: B::Buffer,

    // ── Per-item single-token buffers for decode_batch (Phase 4b) ────
    //
    // The batched-decode path runs M GEMMs at m=M (qkv_proj / o_proj /
    // router / MoE expert mul_mm_id) but attention stays a per-item loop
    // (each cache_id has its own contiguous K/V buffer — no way to fan
    // M items into a single attention dispatch without paged KV). These
    // 1-token-shaped scratches hold the per-item slice during the loop:
    // `copy_slice` extracts q/k/v from the batched buffers, qk_norm_rope
    // writes head-major into _single, kv_cache_append + flash_attention
    // run on it, then copy_slice writes back into attn_flat[i*q_dim].
    //
    // None until `enable_batched_decode_scratch` is called from
    // `ensure_kv` once we know we'll be doing multi-seq decode.
    pub q_single: Option<B::Buffer>,
    pub k_single: Option<B::Buffer>,
    pub v_single: Option<B::Buffer>,
    pub q_head_major_single: Option<B::Buffer>,
    pub k_head_major_single: Option<B::Buffer>,
    pub v_head_major_single: Option<B::Buffer>,
    pub attn_head_major_single: Option<B::Buffer>,

    // ── Paged batched dispatch scratch ──────────────────────────────────
    //
    // Mirrors the same fields on `LlamaFamilyScratch`. `Some` only when
    // `FERRUM_METAL_PAGED_KV=1` and `enable_paged_batch` was called once
    // we know the pool dimensions. Sized for `FERRUM_PAGED_MAX_SEQS ×
    // q_dim` so the multi-seq decode path can fan in M items' Q into a
    // single batched buffer for one `paged_decode_attention(num_seqs=M)`
    // call instead of running M sequential m=1 attentions.
    pub paged_batch_q: Option<B::Buffer>,
    pub paged_batch_o: Option<B::Buffer>,
    pub paged_batch_block_tables: Option<B::Buffer>,
    pub paged_batch_context_lens: Option<B::Buffer>,
    /// Per-seq pos_offset buffer for the batched
    /// `split_qkv_norm_rope_into_paged_cache_varlen` path. Eliminates the
    /// per-item dispatch loop in `forward_layer_batched_decode`.
    pub paged_batch_pos_offsets: Option<B::Buffer>,
    /// `[0, 1, 2, ..., max_seqs]` — pre-filled cumulative seq-len array
    /// for batched decode where every seq contributes q_len=1. Constant
    /// across the lifetime of the scratch.
    pub paged_batch_cu_seqlens_q: Option<B::Buffer>,
    pub paged_max_blocks_per_seq: usize,

    pub max_tokens: usize,

    /// Allocation-free host scratch for the bucketed MoE forward path.
    /// Holds RouterOutput / softmax buffer / MoeBucketPlan reused across
    /// every layer (~10 ms / token reclaimed at c=32 / Qwen3-MoE 30B-A3B).
    pub moe_route_scratch: crate::moe::MoeRouteScratch,

    // ── Unified mixed-batch INDEX buffers ─────────────────────────────
    //
    // vLLM-style: small index tensors written per call. The big
    // activation tensors (residual/norm_out/qkv_out/moe_out) are
    // SHARED with legacy decode/prefill paths — sized for max_tokens,
    // pre-allocated at scratch construction, never grown on demand.
    /// `[max_seqs + 1]` u32 — cumulative q-token counts across items.
    pub unified_cu_seqlens_q: Option<B::Buffer>,
    /// `[max_seqs]` u32 — per-item starting absolute KV position.
    pub unified_pos_offsets: Option<B::Buffer>,
    /// `[max_seqs]` u32 — per-item final KV length after this varlen append.
    pub unified_seq_lens: Option<B::Buffer>,
    /// `[max_seqs * max_blocks_per_seq]` u32 — stacked block tables.
    pub unified_block_tables: Option<B::Buffer>,
    /// `[num_heads * max_tokens]` f32 — FA2 softmax LSE scratch.
    pub unified_attn_lse: Option<B::Buffer>,
    /// Compact q4 tile list for opt-in vLLM-layout varlen attention tiling.
    pub unified_tile_q4_seqs: Option<B::Buffer>,
    pub unified_tile_q4_starts: Option<B::Buffer>,
    /// `[max_seqs, hidden]` — gather of last-token rows for lm_head.
    pub unified_packed_normed: Option<B::Buffer>,
    /// `[max_seqs, vocab]` — per-final-token logits from lm_head.
    pub unified_packed_logits: Option<B::Buffer>,
}

impl<B: QuantLlmBackend + BackendMoeFused> Qwen3MoeScratch<B> {
    pub(super) fn alloc(cfg: &Qwen3MoeConfig, max_tokens: usize) -> Self {
        let h = cfg.base.hidden_size;
        let q_dim = cfg.base.num_heads * cfg.base.head_dim;
        let kv_dim = cfg.base.num_kv_heads * cfg.base.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let t = max_tokens;
        let inter = cfg.expert_intermediate_size;
        let n_exp = cfg.num_experts;
        let vocab = cfg.base.vocab_size;
        Self {
            residual: Some(B::alloc(t * h)),
            norm_out: B::alloc(t * h),
            qkv_out: B::alloc(t * qkv_dim),
            q_buf: B::alloc(t * q_dim),
            k_buf: B::alloc(t * kv_dim),
            v_buf: B::alloc(t * kv_dim),
            q_head_major: B::alloc(cfg.base.num_heads * t * cfg.base.head_dim),
            k_head_major: B::alloc(cfg.base.num_kv_heads * t * cfg.base.head_dim),
            v_head_major: B::alloc(cfg.base.num_kv_heads * t * cfg.base.head_dim),
            attn_head_major_out: B::alloc(cfg.base.num_heads * t * cfg.base.head_dim),
            attn_flat: B::alloc(t * q_dim),
            o_proj_out: B::alloc(t * h),
            router_logits: B::alloc(t * n_exp),
            gate_up_buf: B::alloc(2 * inter),
            silu_buf: B::alloc(inter),
            down_buf: B::alloc(h),
            x_single: B::alloc(h),
            acc_buf: B::alloc(h),
            moe_out: B::alloc(t * h),
            zero_hidden: B::from_slice(&vec![0.0f32; h]),
            gate_out_stacked: B::alloc(t * cfg.num_experts_per_tok * inter),
            up_out_stacked: B::alloc(t * cfg.num_experts_per_tok * inter),
            silu_stacked: B::alloc(t * cfg.num_experts_per_tok * inter),
            down_out_stacked: B::alloc(t * cfg.num_experts_per_tok * h),
            x_packed_bucket: B::alloc(t * cfg.num_experts_per_tok * h),
            gate_up_packed_bucket: B::alloc(t * cfg.num_experts_per_tok * 2 * inter),
            ids_buf: B::from_slice_typed::<i32>(&vec![0i32; cfg.num_experts_per_tok]),
            weights_buf: B::from_slice(&vec![0.0f32; cfg.num_experts_per_tok]),
            selected_ids_buf: B::from_slice_typed::<i32>(&vec![0i32; t * cfg.num_experts_per_tok]),
            // 3 u32s per indirect args buffer; allocated as 3 i32s so we
            // can reuse `from_slice_i32`. The kernel writes them as
            // `device uint *` and the bit pattern is consumed by
            // `dispatch_thread_groups_indirect`.
            gate_up_args_buf: B::from_slice_typed::<i32>(&[0i32, 0, 0]),
            down_args_buf: B::from_slice_typed::<i32>(&[0i32, 0, 0]),
            ids_2d: B::from_slice_typed::<i32>(&vec![0i32; n_exp * t * cfg.num_experts_per_tok]),
            tpe_buf: B::from_slice_typed::<i32>(&vec![0i32; n_exp]),
            weights_2d: B::from_slice(&vec![0.0f32; t * cfg.num_experts_per_tok]),
            // Device-side routing scratch (graph-capturable MoE path).
            route_pairs_dev: B::from_slice_typed::<i32>(&vec![0i32; t * cfg.num_experts_per_tok]),
            route_packed_idx_dev: B::from_slice_typed::<i32>(&vec![
                0i32;
                t * cfg.num_experts_per_tok
            ]),
            route_expert_offsets_dev: B::from_slice_typed::<i32>(&vec![0i32; n_exp + 1]),
            route_pair_weights_dev: B::from_slice_typed::<f32>(&vec![
                0.0f32;
                t * cfg.num_experts_per_tok
            ]),
            // moe_align_block_size outputs — worst-case sized for the
            // largest `moe_block_size` the dispatch path can pick, so the
            // dynamic picker in `dispatch.rs::pick_moe_block_size` is free
            // to go up to 64 without re-allocating scratch per-iter.
            //
            // sorted_tokens capacity = t*top_k + n_exp * MOE_BLOCK_SIZE_MAX
            // (each active expert pads at most MAX-1 extra rows past m_e).
            // block_ids capacity = ceil(t*top_k / MIN_BLOCK_SIZE) + n_exp + 1
            // (worst case: smallest block_size used → most blocks). Min is
            // 16 today; if a smaller is ever introduced, bump this.
            route_sorted_tokens_dev: B::from_slice_typed::<i32>(&vec![
                0i32;
                t * cfg.num_experts_per_tok
                    + n_exp * crate::moe::dispatch::MOE_BLOCK_SIZE_MAX
            ]),
            route_block_ids_dev: B::from_slice_typed::<i32>(&vec![
                0i32;
                t * cfg.num_experts_per_tok / 16
                    + n_exp
                    + 1
            ]),
            route_total_post_pad_dev: B::from_slice_typed::<i32>(&[0i32]),
            last_hidden: B::alloc(h),
            last_normed: B::alloc(h),
            logits: B::alloc(vocab),
            batch_logits: B::alloc(t * vocab),
            // Lazily-allocated; `enable_batched_decode_scratch` populates
            // these the first time decode_batch is called with M > 1.
            q_single: None,
            k_single: None,
            v_single: None,
            q_head_major_single: None,
            k_head_major_single: None,
            v_head_major_single: None,
            attn_head_major_single: None,
            // Lazily-allocated; `enable_paged_batch` populates these when
            // FERRUM_METAL_PAGED_KV=1 + we know the pool dimensions.
            paged_batch_q: None,
            paged_batch_o: None,
            paged_batch_block_tables: None,
            paged_batch_context_lens: None,
            paged_batch_pos_offsets: None,
            paged_batch_cu_seqlens_q: None,
            paged_max_blocks_per_seq: 0,
            max_tokens: t,
            moe_route_scratch: crate::moe::MoeRouteScratch::new(),
            // Unified small index buffers — allocated once by ensure.
            unified_cu_seqlens_q: None,
            unified_pos_offsets: None,
            unified_seq_lens: None,
            unified_block_tables: None,
            unified_attn_lse: None,
            unified_tile_q4_seqs: None,
            unified_tile_q4_starts: None,
            unified_packed_normed: None,
            unified_packed_logits: None,
        }
    }

    /// Allocate small per-call index buffers for the unified mixed-batch
    /// forward. Idempotent. The BIG activation tensors (residual / norm_out
    /// / qkv_out / moe_out) are shared with the legacy paths and sized
    /// for `max_tokens` at scratch construction — no realloc here.
    pub(crate) fn ensure_unified_scratch(
        &mut self,
        cfg: &Qwen3MoeConfig,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    ) {
        if self.unified_cu_seqlens_q.is_some() {
            return;
        }
        let h = cfg.base.hidden_size;
        let v = cfg.base.vocab_size;
        self.unified_cu_seqlens_q = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs + 1,
        ));
        self.unified_pos_offsets = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs,
        ));
        self.unified_seq_lens = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs,
        ));
        self.unified_block_tables = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs * max_blocks_per_seq,
        ));
        self.unified_attn_lse = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::F32,
            cfg.base.num_heads * self.max_tokens,
        ));
        self.unified_tile_q4_seqs = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            self.max_tokens,
        ));
        self.unified_tile_q4_starts = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            self.max_tokens,
        ));
        self.unified_packed_normed = Some(B::alloc(max_seqs * h));
        self.unified_packed_logits = Some(B::alloc(max_seqs * v));
    }

    /// Allocate scratch for paged batched dispatch. Mirrors
    /// `LlamaFamilyScratch::enable_paged_batch`. Idempotent.
    pub(super) fn enable_paged_batch(
        &mut self,
        cfg: &Qwen3MoeConfig,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    ) {
        if self.paged_batch_q.is_some() {
            return;
        }
        let q_dim = cfg.base.num_heads * cfg.base.head_dim;
        self.paged_batch_q = Some(B::alloc(max_seqs * q_dim));
        self.paged_batch_o = Some(B::alloc(max_seqs * q_dim));
        self.paged_batch_block_tables = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs * max_blocks_per_seq,
        ));
        self.paged_batch_context_lens = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs,
        ));
        self.paged_batch_pos_offsets = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs,
        ));
        // cu_seqlens_q is constant [0, 1, 2, ..., max_seqs] for batched
        // decode (q_len=1 per seq) — pre-fill once via a "context" we can
        // borrow temporarily; if the writer needs ctx, we lazy-init at
        // first call instead.
        self.paged_batch_cu_seqlens_q = Some(B::alloc_typed(
            ferrum_kernels::backend::Dtype::U32,
            max_seqs + 1,
        ));
        self.paged_max_blocks_per_seq = max_blocks_per_seq;
    }

    /// Allocate the per-item single-token scratch buffers used by
    /// `forward_layer_batched_decode`. Idempotent.
    pub(super) fn enable_batched_decode_scratch(&mut self, cfg: &Qwen3MoeConfig) {
        if self.q_single.is_some() {
            return;
        }
        let q_dim = cfg.base.num_heads * cfg.base.head_dim;
        let kv_dim = cfg.base.num_kv_heads * cfg.base.head_dim;
        self.q_single = Some(B::alloc(q_dim));
        self.k_single = Some(B::alloc(kv_dim));
        self.v_single = Some(B::alloc(kv_dim));
        self.q_head_major_single = Some(B::alloc(q_dim));
        self.k_head_major_single = Some(B::alloc(kv_dim));
        self.v_head_major_single = Some(B::alloc(kv_dim));
        self.attn_head_major_single = Some(B::alloc(q_dim));
    }
}
