//! `Qwen3MoeModel<B>` — Qwen3-MoE family decoder (Qwen3-30B-A3B and friends).
//!
//! Architectural delta vs [`LlamaFamilyModel`]:
//!   * Each transformer layer's FFN is a top-K MoE block instead of a
//!     fused `gate_up_proj → silu → down_proj` MLP.
//!     - One small router linear (`[hidden] → [num_experts]`) picks
//!       top-K experts per token.
//!     - Each expert is itself a fused `gate_up + down` MLP with the
//!       same SwiGLU + RMSNorm structure as the dense path, just with
//!       `expert_intermediate_size` (typically much smaller than the
//!       dense `intermediate_size`).
//!     - Output is the weight-summed combination of the K selected
//!       expert outputs.
//!   * Attention path is unchanged from dense Qwen3 (GQA + QK-norm + RoPE).
//!
//! Implementation re-uses the dense layer's attention machinery
//! verbatim — RMSNorm, fused QKV, QK-norm + RoPE, KV cache append,
//! flash attention, O-projection, residual + post-norm. The only new
//! code is the MoE FFN block at the tail of each layer's forward.
//!
//! Memory model: experts are loaded as `QuantLinear<B>` per expert,
//! slicing the on-disk 3-D `ffn_{gate,up,down}_exps.weight` tensors
//! byte-wise so weights stay compressed (Q4_K / Q6_K). For a 32 GB
//! Mac to run Qwen3-30B-A3B at all, this is non-negotiable: an
//! eager-fp32 expert stack would weigh ~110 GB.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::OnceLock;

use ferrum_interfaces::kv_dtype::{KvDtypeKind, KvFp16};
use ferrum_kernels::backend::{
    Backend, BackendGraph, BackendMoeFused, BackendPagedKv, BackendQuantGguf, BackendQuantMarlin,
    KvCache, LlmBackend, MoeLlmBackend, QuantLlmBackend,
};
use ferrum_quantization::WeightLoader;
use ferrum_types::{FerrumError, Result};

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyLayer, RopeCache};
use crate::moe::{moe_forward, ExpertStack};
use crate::moe_config::Qwen3MoeConfig;

// Decode-side per-op profile counters — same names as the dense path
// so existing tooling (`FERRUM_DECODE_OP_PROFILE=1` log scrapers) keeps
// working without a separate switch for MoE.
pub(crate) static ATTN_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_CALLS: AtomicU64 = AtomicU64::new(0);

// Fine-grained decode-only counters, populated by
// `moe_forward_stacked_decode_impl` when FERRUM_DECODE_OP_PROFILE is set.
// Each is per-layer summed over the layers in one decode token; drained
// at the bottom of `decode_internal`.
pub(crate) static DEC_ROUTE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_GATE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_UP_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_SILU_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_WSUM_US: AtomicU64 = AtomicU64::new(0);
// Single-shot per decode token (not per-layer).
pub(crate) static DEC_EMBED_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_FINAL_NORM_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_LM_HEAD_US: AtomicU64 = AtomicU64::new(0);

// MoE batched-prefill sub-stage counters (gate / up / down mul_mm_id +
// silu + weighted_sum + host topk). Same FERRUM_DECODE_OP_PROFILE gate.
pub(crate) static MOE_PREFILL_HOST_TOPK_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_HOST_TOPK_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_GATE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_GATE_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_UP_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_UP_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_SILU_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_SILU_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_WSUM_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_WSUM_CALLS: AtomicU64 = AtomicU64::new(0);

// MoE batched-DECODE sub-stage counters (small-m path that uses the
// batched-pair GEMV in place of the per-token loop).
pub(crate) static MOE_BATCHED_DECODE_ROUTE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_GATE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_UP_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_SILU_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_WSUM_US: AtomicU64 = AtomicU64::new(0);

// Coarse stage counters for `forward_layer_batched_decode` so we can
// see where the time goes without per-op instrumentation. Summed
// across all layers in one decode_batch_internal call.
pub(crate) static BD_DENSE_US: AtomicU64 = AtomicU64::new(0); // rms_norm + qkv_proj + split_qkv + o_proj + fused_add_rms_norm
pub(crate) static BD_ATTN_PERITEM_US: AtomicU64 = AtomicU64::new(0); // the for-i in 0..m attention loop (incl. plumbing)
pub(crate) static BD_MOE_US: AtomicU64 = AtomicU64::new(0); // router + MoE FFN + residual add
pub(crate) static BD_LAYER_CALLS: AtomicU64 = AtomicU64::new(0);

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
}

impl<B: QuantLlmBackend + BackendMoeFused> Qwen3MoeScratch<B> {
    fn alloc(cfg: &Qwen3MoeConfig, max_tokens: usize) -> Self {
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
            ids_buf: B::from_slice_i32(&vec![0i32; cfg.num_experts_per_tok]),
            weights_buf: B::from_slice(&vec![0.0f32; cfg.num_experts_per_tok]),
            selected_ids_buf: B::from_slice_i32(&vec![0i32; t * cfg.num_experts_per_tok]),
            // 3 u32s per indirect args buffer; allocated as 3 i32s so we
            // can reuse `from_slice_i32`. The kernel writes them as
            // `device uint *` and the bit pattern is consumed by
            // `dispatch_thread_groups_indirect`.
            gate_up_args_buf: B::from_slice_i32(&[0i32, 0, 0]),
            down_args_buf: B::from_slice_i32(&[0i32, 0, 0]),
            ids_2d: B::from_slice_i32(&vec![0i32; n_exp * t * cfg.num_experts_per_tok]),
            tpe_buf: B::from_slice_i32(&vec![0i32; n_exp]),
            weights_2d: B::from_slice(&vec![0.0f32; t * cfg.num_experts_per_tok]),
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
        }
    }

    /// Allocate scratch for paged batched dispatch. Mirrors
    /// `LlamaFamilyScratch::enable_paged_batch`. Idempotent.
    fn enable_paged_batch(
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
        self.paged_batch_block_tables = Some(B::alloc_u32(max_seqs * max_blocks_per_seq));
        self.paged_batch_context_lens = Some(B::alloc_u32(max_seqs));
        self.paged_batch_pos_offsets = Some(B::alloc_u32(max_seqs));
        // cu_seqlens_q is constant [0, 1, 2, ..., max_seqs] for batched
        // decode (q_len=1 per seq) — pre-fill once via a "context" we can
        // borrow temporarily; if the writer needs ctx, we lazy-init at
        // first call instead.
        self.paged_batch_cu_seqlens_q = Some(B::alloc_u32(max_seqs + 1));
        self.paged_max_blocks_per_seq = max_blocks_per_seq;
    }

    /// Allocate the per-item single-token scratch buffers used by
    /// `forward_layer_batched_decode`. Idempotent.
    fn enable_batched_decode_scratch(&mut self, cfg: &Qwen3MoeConfig) {
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

/// Qwen3-MoE decoder model.
///
/// Holds the same per-layer attention weights as [`LlamaFamilyModel`]
/// plus a [`Qwen3MoeLayerState`] per layer for the MoE FFN. Routing,
/// expert dispatch, and weighted combine all happen inside
/// [`moe_forward`]; this struct only owns the storage and orchestrates
/// the per-layer call sequence.
pub struct Qwen3MoeModel<B: MoeLlmBackend, K: KvDtypeKind = KvFp16> {
    pub cfg: Qwen3MoeConfig,
    pub runtime_cfg: LlmRuntimeConfig,

    pub embed: B::Buffer,
    /// Per-layer attention weights (re-uses dense `LlamaFamilyLayer`).
    pub attn_layers: Vec<LlamaFamilyLayer<B>>,
    /// Per-layer MoE state (router + expert stack).
    pub moe_layers: Vec<Qwen3MoeLayerState<B>>,
    pub final_norm_w: B::Buffer,
    pub lm_head: Box<dyn ferrum_quantization::Linear<B>>,

    pub rope: RopeCache<B>,
    pub scratch: Qwen3MoeScratch<B>,

    pub kv_caches: HashMap<String, Vec<KvCache<B, K>>>,
    kv_free_pool: Vec<Vec<KvCache<B, K>>>,

    // ── Paged-KV multi-seq state ────────────────────────────────────────
    //
    // Mirrors `LlamaFamilyModel`. Only populated when
    // `FERRUM_METAL_PAGED_KV=1`. Kv_caches entries become metadata-only
    // views (block_table + context_lens) into the shared `paged_pools`.
    pub paged_pools: Option<Vec<(B::Buffer, B::Buffer)>>,
    pub paged_block_alloc: Option<std::sync::Mutex<crate::common::paged_pool::BlockAllocator>>,
    // Paged-batch dispatch dimensions. Pinned at the first `ensure_kv`
    // when paged-KV is on. Stored on the model (not on scratch) so
    // `ensure_scratch`'s realloc — which wipes scratch's
    // `paged_batch_block_tables`/`paged_batch_q`/etc. — can re-call
    // `enable_paged_batch` with the same dims afterwards. Without this
    // re-init the next forward enters `forward_layer_batched_decode`
    // and panics on `paged_batch_block_tables missing`.
    pub paged_dims: Option<(usize, usize)>, // (max_seqs, max_blocks_per_seq)
}

impl<B: MoeLlmBackend, K: KvDtypeKind> Qwen3MoeModel<B, K> {
    /// Build a Qwen3-MoE model from a generic `WeightLoader<B>` plus a
    /// GGUF reader for the experts (which `WeightLoader` doesn't model
    /// directly — its API is rank-2 only).
    ///
    /// `loader` provides: token embedding, attention projections, layer
    /// norms, lm_head — all the rank-2 weights.
    /// `gguf` provides: the rank-3 expert tensors, sliced per-expert
    /// inside [`ExpertStack::load_from_gguf`].
    pub fn new(
        cfg: Qwen3MoeConfig,
        loader: &dyn WeightLoader<B>,
        gguf: &ferrum_quantization::gguf::GgufFile,
    ) -> Result<Self> {
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let rope = build_rope_cache::<B>(&cfg.base);
        let scratch = Qwen3MoeScratch::alloc(&cfg, 1);

        let embed = loader.load_tensor("model.embed_tokens.weight")?;

        let mut attn_layers = Vec::with_capacity(cfg.base.num_layers);
        let mut moe_layers = Vec::with_capacity(cfg.base.num_layers);
        for li in 0..cfg.base.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w =
                loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;

            // Dense gate_up_proj / down_proj are absent in MoE GGUFs —
            // we synthesise stub Linears so the LlamaFamilyLayer struct
            // type-checks. They're never invoked because forward_layer
            // calls the MoE path. Cheap: tiny zero-sized DenseLinears.
            let gate_up_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(2 * cfg.expert_intermediate_size, cfg.base.hidden_size);
            let down_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(cfg.base.hidden_size, cfg.expert_intermediate_size);

            let (q_norm_w, k_norm_w) = if cfg.base.has_qk_norm {
                let q = loader
                    .load_tensor(&format!("{prefix}.self_attn.q_norm.weight"))
                    .ok();
                let k = loader
                    .load_tensor(&format!("{prefix}.self_attn.k_norm.weight"))
                    .ok();
                (q, k)
            } else {
                (None, None)
            };

            attn_layers.push(LlamaFamilyLayer {
                input_ln_w,
                qkv_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_ln_w,
                gate_up_proj,
                down_proj,
            });

            // Router lives at `model.layers.{li}.mlp.router.weight` in
            // ferrum-name space (see ferrum_to_gguf mapping). It's a
            // plain rank-2 linear so the standard loader path covers
            // it without going through the MoE-specific GGUF helper.
            let router = loader.load_linear(&format!("{prefix}.mlp.router"))?;
            if router.in_features() != cfg.base.hidden_size {
                return Err(FerrumError::model(format!(
                    "router layer {li}: in_features {} != hidden {}",
                    router.in_features(),
                    cfg.base.hidden_size
                )));
            }
            if router.out_features() != cfg.num_experts {
                return Err(FerrumError::model(format!(
                    "router layer {li}: out_features {} != num_experts {}",
                    router.out_features(),
                    cfg.num_experts
                )));
            }

            let experts = ExpertStack::<B>::load_from_gguf(
                gguf,
                li,
                cfg.num_experts,
                cfg.base.hidden_size,
                cfg.expert_intermediate_size,
            )?;

            moe_layers.push(Qwen3MoeLayerState { router, experts });
        }

        let final_norm_w = loader.load_tensor("model.norm.weight")?;
        let lm_head = if loader.has_tensor("lm_head.weight") {
            loader.load_linear("lm_head")?
        } else {
            // Tied embeddings — same as dense path.
            tracing::info!(
                "Qwen3MoeModel: tied embeddings — loading model.embed_tokens.weight as lm_head"
            );
            loader.load_linear("model.embed_tokens")?
        };

        let runtime_cfg = cfg.base.to_runtime();
        Ok(Self {
            cfg,
            runtime_cfg,
            embed,
            attn_layers,
            moe_layers,
            final_norm_w,
            lm_head,
            rope,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_block_alloc: None,
            paged_dims: None,
        })
    }

    /// Build from a HuggingFace safetensors model directory (GPTQ-INT4
    /// expected). Mirrors [`Self::new`] but with a STACKED expert loader:
    /// reads all `num_experts` experts' raw GPTQ tensors per layer once,
    /// concats along N host-side, single `B::load_gptq` repacks the
    /// whole thing into one Marlin tile per (layer, role).
    ///
    /// 128 experts × 48 layers × 3 projs would otherwise trigger 18 432
    /// per-call Marlin repacks (~30+ minute cold start at ~100 ms each
    /// on Llama-MoE shapes). The stacked path drops that to 96 repacks
    /// — one per (layer × {gate_up, down}) — and dispatch slices per
    /// expert via `B::gemm_gptq_with_offset`.
    pub fn new_safetensors(
        cfg: Qwen3MoeConfig,
        loader: &ferrum_quantization::NativeSafetensorsLoader<B>,
    ) -> Result<Self> {
        use ferrum_quantization::WeightLoader as _;
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let rope = build_rope_cache::<B>(&cfg.base);
        let scratch = Qwen3MoeScratch::alloc(&cfg, 1);
        let embed = loader.load_tensor("model.embed_tokens.weight")?;
        let mut attn_layers = Vec::with_capacity(cfg.base.num_layers);
        let mut moe_layers = Vec::with_capacity(cfg.base.num_layers);
        for li in 0..cfg.base.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w =
                loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let gate_up_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(2 * cfg.expert_intermediate_size, cfg.base.hidden_size);
            let down_proj: Box<dyn ferrum_quantization::Linear<B>> =
                stub_linear::<B>(cfg.base.hidden_size, cfg.expert_intermediate_size);
            let (q_norm_w, k_norm_w) = if cfg.base.has_qk_norm {
                let q = loader
                    .load_tensor(&format!("{prefix}.self_attn.q_norm.weight"))
                    .ok();
                let k = loader
                    .load_tensor(&format!("{prefix}.self_attn.k_norm.weight"))
                    .ok();
                (q, k)
            } else {
                (None, None)
            };
            attn_layers.push(LlamaFamilyLayer {
                input_ln_w,
                qkv_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_ln_w,
                gate_up_proj,
                down_proj,
            });

            // Router: rank-2 linear, standard load.
            let router = loader.load_linear(&format!("{prefix}.mlp.gate"))?;
            if router.in_features() != cfg.base.hidden_size {
                return Err(ferrum_types::FerrumError::model(format!(
                    "router layer {li}: in_features {} != hidden {}",
                    router.in_features(),
                    cfg.base.hidden_size
                )));
            }
            if router.out_features() != cfg.num_experts {
                return Err(ferrum_types::FerrumError::model(format!(
                    "router layer {li}: out_features {} != num_experts {}",
                    router.out_features(),
                    cfg.num_experts
                )));
            }

            // Stacked GPTQ Marlin load via per-expert-repack-then-concat
            // (B::load_gptq_stacked). Each expert's Marlin-packed bytes
            // are contiguous in the GPU buffer, so offset GEMM
            // dispatches via pointer offset alone — no stride magic.
            let expert_prefix = format!("{prefix}.mlp.experts.{{e}}.");
            let probe_split =
                loader.has_tensor(&format!("{prefix}.mlp.experts.0.gate_proj.qweight"));
            let gate_up_projs: &[&str] = if probe_split {
                &["gate_proj", "up_proj"]
            } else {
                &["gate_up_proj"]
            };
            let (gate_up_store, gate_up_n_per_expert, gate_up_k) =
                loader.load_stacked_gptq_experts(&expert_prefix, cfg.num_experts, gate_up_projs)?;
            let (down_store, down_n_per_expert, down_k) = loader.load_stacked_gptq_experts(
                &expert_prefix,
                cfg.num_experts,
                &["down_proj"],
            )?;
            let gate_up_arc = std::sync::Arc::new(gate_up_store);
            let down_arc = std::sync::Arc::new(down_store);

            let mut gate_up: Vec<Box<dyn ferrum_quantization::Linear<B>>> =
                Vec::with_capacity(cfg.num_experts);
            let mut down: Vec<Box<dyn ferrum_quantization::Linear<B>>> =
                Vec::with_capacity(cfg.num_experts);
            for e in 0..cfg.num_experts {
                gate_up.push(Box::new(
                    ferrum_quantization::StackedExpertLinear::<B>::new(
                        gate_up_arc.clone(),
                        e * gate_up_n_per_expert,
                        gate_up_n_per_expert,
                        gate_up_k,
                    )?,
                ));
                down.push(Box::new(
                    ferrum_quantization::StackedExpertLinear::<B>::new(
                        down_arc.clone(),
                        e * down_n_per_expert,
                        down_n_per_expert,
                        down_k,
                    )?,
                ));
            }
            // Wrap raw Marlin tile stores in trait objects (Phase C step 3) —
            // dispatch goes through `gu_store.gemm_phase_*` / `zero_workspace`
            // instead of `B::moe_gemm_phase_*` / `B::marlin_zero_stacked_workspace`.
            let gate_up_marlin =
                <B as ferrum_kernels::backend::BackendQuantMarlin>::make_marlin_expert_stack(
                    gate_up_arc,
                    cfg.num_experts,
                    gate_up_n_per_expert,
                    gate_up_k,
                )?;
            let down_marlin =
                <B as ferrum_kernels::backend::BackendQuantMarlin>::make_marlin_expert_stack(
                    down_arc,
                    cfg.num_experts,
                    down_n_per_expert,
                    down_k,
                )?;
            let experts = crate::moe::ExpertStack::<B> {
                gate_up,
                down,
                gate_stacked: None,
                up_stacked: None,
                down_stacked: None,
                gate_up_marlin_stack: Some(gate_up_marlin),
                down_marlin_stack: Some(down_marlin),
            };
            moe_layers.push(Qwen3MoeLayerState { router, experts });

            if li == 0 || li.is_multiple_of(8) || li == cfg.base.num_layers - 1 {
                tracing::info!(
                    "Qwen3MoeModel safetensors: layer {li}/{} loaded \
                     (stacked: gate_up={}x{} k={}, down={}x{} k={})",
                    cfg.base.num_layers,
                    cfg.num_experts,
                    gate_up_n_per_expert,
                    gate_up_k,
                    cfg.num_experts,
                    down_n_per_expert,
                    down_k,
                );
            }
        }

        let final_norm_w = loader.load_tensor("model.norm.weight")?;
        let lm_head = if loader.has_tensor("lm_head.weight") {
            loader.load_linear("lm_head")?
        } else {
            tracing::info!(
                "Qwen3MoeModel safetensors: tied embeddings — using model.embed_tokens as lm_head"
            );
            loader.load_linear("model.embed_tokens")?
        };

        let runtime_cfg = cfg.base.to_runtime();
        Ok(Self {
            cfg,
            runtime_cfg,
            embed,
            attn_layers,
            moe_layers,
            final_norm_w,
            lm_head,
            rope,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_block_alloc: None,
            paged_dims: None,
        })
    }

    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            {
                let mut ctx = B::new_context();
                B::reset_all_graphs(&mut ctx);
            }
            self.scratch = Qwen3MoeScratch::alloc(&self.cfg, tokens);
            // Realloc wiped paged_batch_*. Re-enable using the dims
            // pinned at first ensure_kv. Without this, the next
            // `forward_layer_batched_decode` panics on
            // `paged_batch_block_tables missing` (regression manifests
            // at c≥16 when batch growth triggers scratch realloc
            // between `ensure_kv` and the batched-decode entry point).
            if let Some((max_seqs, max_blocks_per_seq)) = self.paged_dims {
                self.scratch
                    .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
            }
        }
    }

    pub(crate) fn ensure_kv(&mut self, cache_id: &str) {
        if self.kv_caches.contains_key(cache_id) {
            return;
        }
        let nkv = self.cfg.base.num_kv_heads;
        let hd = self.cfg.base.head_dim;
        // 512 in 0.7.2 — same value the published bench used to hit 79
        // tok/s at c=16 on this exact MoE model. See
        // `LlamaFamilyModel::ensure_kv` for the full rationale.
        let model_max = self.cfg.base.max_seq_len;
        const DEFAULT_KV_CAPACITY: usize = 512;
        let max = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY));

        // Paged-KV mode: `FERRUM_METAL_PAGED_KV=1` switches caches into
        // block-table-indirect layout. Mirrors LlamaFamilyModel's path so
        // the existing `paged_decode_attention` Metal kernel can fire
        // once at num_seqs=m for batched decode (replacing the per-item
        // attention loop that currently dominates `attn_peritem` in the
        // c=16 profile).
        // Default ON when the backend supports paged-KV (Metal). Users
        // can force off with `FERRUM_METAL_PAGED_KV=0`. The flag was
        // opt-in pre-0.7.2; flipping the default so default `ferrum
        // serve` matches the bench-quality numbers without requiring
        // env-var knowledge.
        let paged = std::env::var("FERRUM_METAL_PAGED_KV")
            .map(|v| v != "0")
            .unwrap_or_else(|_| B::supports_paged_kv());
        const PAGED_BLOCK_SIZE: usize = 16;

        // Default 32: covers c=16 burst with 2× headroom for the
        // fresh-cache-id-per-request pattern that bench/server harnesses
        // use. Pool memory unchanged from pre-0.7.2 default because
        // DEFAULT_KV_CAPACITY dropped 4096 → 2048 in lockstep.
        let max_seqs = std::env::var("FERRUM_PAGED_MAX_SEQS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32);
        let max_blocks_per_seq = max.div_ceil(PAGED_BLOCK_SIZE);
        let total_pool_blocks = max_seqs * max_blocks_per_seq;

        // Lazy-allocate the shared paged pools on the first paged
        // ensure_kv call.
        if paged && self.paged_pools.is_none() {
            let mut pools = Vec::with_capacity(self.cfg.base.num_layers);
            for _ in 0..self.cfg.base.num_layers {
                let pool_floats = total_pool_blocks * nkv * PAGED_BLOCK_SIZE * hd;
                pools.push((B::alloc(pool_floats), B::alloc(pool_floats)));
            }
            self.paged_pools = Some(pools);
            self.paged_block_alloc = Some(std::sync::Mutex::new(
                crate::common::paged_pool::BlockAllocator::new(total_pool_blocks as u32),
            ));
        }
        if paged {
            self.scratch
                .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
            // Pin dims on the model so `ensure_scratch`'s realloc can
            // re-call `enable_paged_batch` after wiping scratch.
            self.paged_dims = Some((max_seqs, max_blocks_per_seq));
        }

        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.base.num_layers)
                .map(|_| {
                    if paged {
                        // Paged mode: cache holds metadata only. K/V are
                        // 1-element placeholders. Real data lives in
                        // `self.paged_pools[li].{k,v}`.
                        let mut block_table = B::alloc_u32(max_blocks_per_seq);
                        let _ = &mut block_table; // suppress unused-mut on backends that no-op write_u32
                        let mut context_lens = B::alloc_u32(1);
                        let mut bt_ctx = B::new_context();
                        B::write_u32(&mut bt_ctx, &mut context_lens, &[0u32]);
                        B::sync(&mut bt_ctx);
                        KvCache {
                            k: B::alloc(1),
                            v: B::alloc(1),
                            len: 0,
                            capacity: max_blocks_per_seq * PAGED_BLOCK_SIZE,
                            num_kv_heads: nkv,
                            head_dim: hd,
                            block_size: PAGED_BLOCK_SIZE,
                            block_table: Some(block_table),
                            context_lens: Some(context_lens),
                            paged_block_indices: Vec::new(),
                            _kv_dtype: std::marker::PhantomData,
                        }
                    } else {
                        KvCache {
                            k: B::alloc(nkv * max * hd),
                            v: B::alloc(nkv * max * hd),
                            len: 0,
                            capacity: max,
                            num_kv_heads: nkv,
                            head_dim: hd,
                            block_size: 0,
                            block_table: None,
                            context_lens: None,
                            paged_block_indices: Vec::new(),
                            _kv_dtype: std::marker::PhantomData,
                        }
                    }
                })
                .collect()
        });

        // Allocate physical blocks for THIS cache_id from the shared pool.
        if paged {
            let alloc_arc = self
                .paged_block_alloc
                .as_ref()
                .expect("paged_block_alloc must be initialised when paged=true");
            let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            let block_indices = match alloc.allocate_n(max_blocks_per_seq) {
                Ok(idx) => idx,
                Err(e) => {
                    drop(alloc);
                    self.kv_free_pool.push(caches);
                    eprintln!(
                        "[ferrum] paged KV pool exhausted on ensure_kv for \
                         cache_id={cache_id:?}: {e}. Increase \
                         FERRUM_PAGED_MAX_SEQS (currently {max_seqs}) or \
                         throttle concurrent requests.",
                    );
                    return;
                }
            };
            let mut padded = block_indices.clone();
            padded.resize(max_blocks_per_seq, 0);
            let mut ctx_tmp = B::new_context();
            for c in caches.iter_mut() {
                if let Some(bt) = c.block_table.as_mut() {
                    B::write_u32(&mut ctx_tmp, bt, &padded);
                }
                c.paged_block_indices = block_indices.clone();
            }
            B::sync(&mut ctx_tmp);
        }

        for c in caches.iter_mut() {
            c.len = 0;
            if let Some(cl) = c.context_lens.as_mut() {
                let mut ctx_tmp = B::new_context();
                B::write_u32(&mut ctx_tmp, cl, &[0u32]);
                B::sync(&mut ctx_tmp);
            }
        }
        self.kv_caches.insert(cache_id.to_string(), caches);
    }

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

        let attn_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };

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
            B::split_qkv_norm_rope_into_paged_cache(
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
            B::write_u32(ctx, cl_buf, &[final_kv_len]);
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

        if let Some(t0) = attn_t0 {
            B::sync(ctx);
            ATTN_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
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
        let moe_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };

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
            && std::env::var("FERRUM_MOE_BUCKETED").as_deref() != Ok("0");

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
                ctx,
                &self.scratch.norm_out,
                &self.scratch.router_logits,
                &mut self.scratch.moe_out,
                tokens,
                h,
                self.cfg.expert_intermediate_size,
                self.cfg.num_experts,
                self.cfg.num_experts_per_tok,
                self.cfg.norm_topk_prob,
                &moe_layer.experts,
                &mut self.scratch.x_packed_bucket,
                &mut self.scratch.gate_up_packed_bucket,
                &mut self.scratch.silu_stacked,
                &mut self.scratch.down_out_stacked,
                &mut self.scratch.moe_route_scratch,
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
            moe_forward::<B>(
                ctx,
                &self.scratch.norm_out,
                &self.scratch.router_logits,
                &mut self.scratch.moe_out,
                tokens,
                h,
                self.cfg.expert_intermediate_size,
                self.cfg.num_experts,
                self.cfg.num_experts_per_tok,
                self.cfg.norm_topk_prob,
                &moe_layer.experts,
                &mut self.scratch.x_single,
                &mut self.scratch.acc_buf,
                &mut self.scratch.gate_up_buf,
                &mut self.scratch.silu_buf,
                &mut self.scratch.down_buf,
                &self.scratch.zero_hidden,
            )?;
        }

        // 12. residual += moe_out (skipped on decode fast path — already
        //     accumulated by `weighted_sum_residual_stacked`).
        if !decode_fast_path {
            B::add_inplace(ctx, residual, &self.scratch.moe_out, tokens * h);
        }

        if let Some(t0) = moe_t0 {
            B::sync(ctx);
            MOE_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
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

    /// Prefill: process `tokens` prompt tokens, return last-token logits.
    pub fn prefill_internal(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0);
        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

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
        let prefill_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(&mut ctx);
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
            Some(std::time::Instant::now())
        } else {
            None
        };

        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        B::embedding_lookup(&mut ctx, &self.embed, tokens, &mut residual, h);

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

        B::sync(&mut ctx);
        if let Some(t0) = prefill_t0 {
            let total_us = t0.elapsed().as_micros() as u64;
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
        B::to_vec(&self.scratch.logits, vocab)
    }

    /// Decode: 1 token at position `pos`, return next-step logits.
    pub fn decode_internal(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let h = self.cfg.base.hidden_size;
        let vocab = self.cfg.base.vocab_size;
        let mut ctx = B::new_context();

        let decode_t0 = if std::env::var("FERRUM_MOE_PROFILE").is_ok() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // FERRUM_DECODE_OP_PROFILE gates the per-stage breakdown emitted
        // at the bottom of every decode token. Reuses the same atomic
        // counters that `forward_layer` already populates (ATTN_TIME_US,
        // MOE_TIME_US — drained here per-token instead of per-prefill).
        let stage_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(&mut ctx);
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
            Some(std::time::Instant::now())
        } else {
            None
        };
        let prof = stage_t0.is_some();
        let mark = |ctx: &mut B::Context, c: &AtomicU64, t0: std::time::Instant| {
            if prof {
                B::sync(ctx);
                c.fetch_add(
                    t0.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        };
        let mt0 = std::time::Instant::now();

        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let t0 = std::time::Instant::now();
        B::embedding_lookup(&mut ctx, &self.embed, &[token], &mut residual, h);
        mark(&mut ctx, &DEC_EMBED_US, t0);
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

        let t0 = std::time::Instant::now();
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.base.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );
        mark(&mut ctx, &DEC_FINAL_NORM_US, t0);

        let t0 = std::time::Instant::now();
        self.lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );
        mark(&mut ctx, &DEC_LM_HEAD_US, t0);

        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);

        // FERRUM_DECODE_OP_PROFILE: per-token decode breakdown.
        if let Some(t0) = stage_t0 {
            use std::sync::atomic::Ordering;
            let total_us = t0.elapsed().as_micros() as u64;
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
            }
        }

        B::to_vec(&self.scratch.logits, vocab)
    }

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
        let m = batch.len();
        if m == 0 {
            return Vec::new();
        }
        if m == 1 {
            let (cid, tok, pos) = &batch[0];
            return vec![self.decode_internal(cid, *tok, *pos)];
        }

        let prof_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
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
        let mut ctx = B::new_context();

        // 0. Embed all M tokens into residual [M, H]
        let tokens: Vec<u32> = batch.iter().map(|(_, t, _)| *t).collect();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        B::embedding_lookup(&mut ctx, &self.embed, &tokens, &mut residual, h);

        // 1..num_layers: batched forward for each layer
        for li in 0..self.cfg.base.num_layers {
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

        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);

        let all = B::to_vec(&self.scratch.batch_logits, m * vocab);

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
            }
        }

        (0..m)
            .map(|i| all[i * vocab..(i + 1) * vocab].to_vec())
            .collect()
    }

    /// One transformer layer over M items: GEMMs at m=M, per-item
    /// attention loop, MoE FFN at m=M via the prefill batched path.
    /// Mirrors `LlamaFamilyModel::forward_layer_batched_decode` minus
    /// the paged branch.
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

        let prof = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
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

            // Step 1: gather per-seq metadata on host (cache.len before
            // append, full block_tables stack, post-append context_lens),
            // bump cache.len, build cu_seqlens_q.
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
                let cache = &mut caches[li];
                pos_offsets_host[i] = cache.len as u32; // RoPE position = pre-append cache len
                let blocks = &cache.paged_block_indices;
                let n_to_copy = blocks.len().min(max_blocks_per_seq);
                stacked_bt[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
                    .copy_from_slice(&blocks[..n_to_copy]);
                cache.len += 1;
                stacked_cl[i] = cache.len as u32;
            }

            // Step 2: upload all per-seq metadata.
            let bt_buf = self
                .scratch
                .paged_batch_block_tables
                .as_mut()
                .expect("paged_batch_block_tables missing");
            B::write_u32(ctx, bt_buf, &stacked_bt);
            let cl_buf = self
                .scratch
                .paged_batch_context_lens
                .as_mut()
                .expect("paged_batch_context_lens missing");
            B::write_u32(ctx, cl_buf, &stacked_cl);
            let pos_buf = self
                .scratch
                .paged_batch_pos_offsets
                .as_mut()
                .expect("paged_batch_pos_offsets missing");
            B::write_u32(ctx, pos_buf, &pos_offsets_host);
            let cu_buf = self
                .scratch
                .paged_batch_cu_seqlens_q
                .as_mut()
                .expect("paged_batch_cu_seqlens_q missing");
            B::write_u32(ctx, cu_buf, &cu_seqlens_host);

            // Step 3: write K/V into the shared pool + RoPE'd Q into
            // paged_batch_q at offset i × q_dim. Two code paths:
            //
            //   - CUDA (`B::supports_varlen_qkv() == true`): ONE batched
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

            if B::supports_varlen_qkv() {
                let bt_ptr_raw =
                    self.scratch.paged_batch_block_tables.as_ref().unwrap() as *const B::Buffer;
                let pos_ptr_raw =
                    self.scratch.paged_batch_pos_offsets.as_ref().unwrap() as *const B::Buffer;
                let cu_ptr_raw =
                    self.scratch.paged_batch_cu_seqlens_q.as_ref().unwrap() as *const B::Buffer;
                let bt_safe: &B::Buffer = unsafe { &*bt_ptr_raw };
                let pos_safe: &B::Buffer = unsafe { &*pos_ptr_raw };
                let cu_safe: &B::Buffer = unsafe { &*cu_ptr_raw };
                B::split_qkv_norm_rope_into_paged_cache_varlen(
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
                // Per-item fallback. Element size for qkv / Q output is the
                // backend's `Buffer` payload type — FP16 on CUDA / Metal,
                // FP32 on CPU. The Metal MoE path lives on FP16 buffers
                // (`half::f16`), matching what llama_family's batched-
                // decode path uses.
                let qkv_stride_bytes = (qkv_stride * std::mem::size_of::<half::f16>()) as u64;
                let q_head_major_size_bytes = (q_dim * std::mem::size_of::<half::f16>()) as u64;
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
                    // pos_offsets_host[i] is pre-append cache.len (captured
                    // before the increment in step 1). The per-item kernel
                    // uses it for BOTH RoPE position AND K/V write offset.
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
                        pos_i, // cache_len = pre-append
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
        //   * `FERRUM_MOE_BATCH_THRESHOLD` (default 8) — m above which
        //     the LEGACY non-experimental path uses the prefill GEMM.
        //     Shared with `decode_batch`'s engine-level gate, so users
        //     who set it to a small value to engage batched decode
        //     don't accidentally also push the inner FFN to GEMM.
        //   * `FERRUM_MOE_PREFILL_THRESHOLD` (default 32) — m above
        //     which the EXPERIMENTAL batched-decode path defers to the
        //     prefill GEMM path. Mirrors llama.cpp's `ne21_mm_id_min=32`
        //     GEMV→GEMM boundary.
        let legacy_prefill_threshold: usize = std::env::var("FERRUM_MOE_BATCH_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let new_prefill_threshold: usize = std::env::var("FERRUM_MOE_PREFILL_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        // 0.7.2: default to ON when paged-KV is also on (which is now
        // the default for Metal). The historical regression for this
        // flag (-19% c=4 / -36% c=16) was measured in the pre-paged-KV
        // world where `forward_layer_batched_decode`'s per-item
        // copy_slice × m × 6 attention dispatches cost more than the
        // batched MoE FFN saved. Once paged-KV is on, attention runs as
        // one `paged_decode_attention(num_seqs=m)` dispatch, the
        // plumbing cost drops, and the batched MoE GEMV's win net out
        // to ~+50% at c=16. `FERRUM_MOE_BATCHED_DECODE=0` forces off.
        let new_batched_default = stacked_path_available && B::supports_batched_moe_gemv();
        let new_batched_enabled = new_batched_default
            && std::env::var("FERRUM_MOE_BATCHED_DECODE")
                .map(|v| v != "0")
                .unwrap_or(true);

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
            && std::env::var("FERRUM_MOE_BUCKETED").as_deref() != Ok("0")
        {
            // CUDA Marlin bucketed path (decode_batch m ≥ 1 entry).
            crate::moe::moe_forward_bucketed::<B>(
                ctx,
                &self.scratch.norm_out,
                &self.scratch.router_logits,
                &mut self.scratch.moe_out,
                m,
                h,
                self.cfg.expert_intermediate_size,
                self.cfg.num_experts,
                self.cfg.num_experts_per_tok,
                self.cfg.norm_topk_prob,
                &moe_layer.experts,
                &mut self.scratch.x_packed_bucket,
                &mut self.scratch.gate_up_packed_bucket,
                &mut self.scratch.silu_stacked,
                &mut self.scratch.down_out_stacked,
                &mut self.scratch.moe_route_scratch,
            )?;
        } else {
            // Backend without stacked variants — fall back to the legacy
            // per-(token, expert) host-routed path.
            moe_forward::<B>(
                ctx,
                &self.scratch.norm_out,
                &self.scratch.router_logits,
                &mut self.scratch.moe_out,
                m,
                h,
                self.cfg.expert_intermediate_size,
                self.cfg.num_experts,
                self.cfg.num_experts_per_tok,
                self.cfg.norm_topk_prob,
                &moe_layer.experts,
                &mut self.scratch.x_single,
                &mut self.scratch.acc_buf,
                &mut self.scratch.gate_up_buf,
                &mut self.scratch.silu_buf,
                &mut self.scratch.down_buf,
                &self.scratch.zero_hidden,
            )?;
        }

        // 11. residual += moe_out [M, H]
        B::add_inplace(ctx, residual, &self.scratch.moe_out, m * h);

        // Close MoE-block instrumentation (router + FFN + residual add).
        stage_end(moe_t0, ctx, &BD_MOE_US);

        Ok(())
    }
}

impl<B: MoeLlmBackend, K: KvDtypeKind> DecoderOnlyLLM for Qwen3MoeModel<B, K> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        // Eager scratch + KV cache grow + a 1-token forward warmup so
        // the first real prefill / decode doesn't pay the cold-start
        // ~25-MTLBuffer scratch alloc + ~96-MTLBuffer KV alloc + Metal
        // pipeline-state first-bind costs (~265 ms total on Qwen3-MoE
        // 30B-A3B / M1 Max). Mirrors what llama-bench's --warmup does
        // (which runs a same-shape forward before the timer).
        self.ensure_scratch(max_tokens);
        self.ensure_kv(cache_id);

        // Warmup forward through all 48 layers under a scratch cache_id
        // so the real `cache_id` starts at pos_offset=0. Token 0 is
        // valid for any tokenizer (BOS or pad).
        const WARMUP_CACHE: &str = "__ferrum_warmup__";
        let _ = self.prefill_internal(WARMUP_CACHE, &[0u32]);
        // Drop the warmup KV cache slot — real cache_id is unaffected.
        if let Some(caches) = self.kv_caches.remove(WARMUP_CACHE) {
            self.kv_free_pool.push(caches);
        }
    }

    fn kv_capacity(&self) -> usize {
        // Mirror the bound `ensure_kv` will use when allocating the cache.
        let model_max = self.cfg.base.max_seq_len;
        const DEFAULT_KV_CAPACITY: usize = 512;
        std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY))
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.prefill_internal(cache_id, tokens)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.decode_internal(cache_id, token, pos)
    }

    // decode_batch is gated to use the batched path only when it's a
    // measurable win. The crossover depends on M:
    //
    //   - At low M (≤ ~8) the per-item `decode_internal` loop wins
    //     because: (a) it stays at scratch offset 0 (no copy_slice
    //     overhead), (b) it preserves the cross-layer rms_norm fusion
    //     fast path (`weighted_sum_residual_norm_stacked`).
    //   - At high M (≥ ~12) the batched path wins because the dense
    //     GEMM batching (qkv_proj, o_proj, router, lm_head at m=M) and
    //     the prefill-batched MoE dispatch (one `gemm_quant_moe_id` for
    //     all tokens) amortise the ~48-dispatch lost-fusion penalty.
    //
    // Default opted out of FERRUM_MOE_BATCHED. When opted in, the
    // batched path engages only at M ≥ FERRUM_MOE_BATCH_THRESHOLD
    // (default 12). Below that we still go per-item.
    //
    // Empirical note 2026-05-02: a follow-up PR added a batched MoE
    // GEMV kernel (`gemv_quant_moe_id_batched`) that holds MoE
    // dispatch count flat as concurrency scales. Wiring it through
    // `decode_batch_internal` regressed throughput by 19% (c=4) /
    // 36% (c=16) — `forward_layer_batched_decode`'s per-item
    // attention plumbing (copy_slice × m × 6 dispatches) costs more
    // than the MoE save. The batched MoE kernel is shipped as opt-in
    // infrastructure (`FERRUM_MOE_BATCHED_DECODE=1`); flipping it on
    // by default has to wait until the attention plumbing is fixed.
    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        let m = batch.len();
        // Default ON in 0.7.2+. The threshold (default 8) keeps small-m
        // requests on the per-token loop where it still wins on this
        // hardware — see docs/bench/macos-2026-05-02 for the crossover
        // measurements (c=4 batched 39 < per_token 42; c=8 batched 59 >
        // per_token 47). `FERRUM_MOE_BATCHED=0` forces the legacy loop.
        let opted_in = std::env::var("FERRUM_MOE_BATCHED")
            .map(|v| v != "0")
            .unwrap_or(true);
        let threshold = std::env::var("FERRUM_MOE_BATCH_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(8);
        if opted_in && m >= threshold {
            self.decode_batch_internal(batch)
        } else {
            batch
                .iter()
                .map(|(cid, tok, p)| self.decode(cid, *tok, *p))
                .collect()
        }
    }

    fn release(&mut self, cache_id: &str) {
        // qwen3_moe doesn't currently use the batched-graph capture path,
        // so single-key reset is sufficient.
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_graph(&mut ctx, 0);
        B::sync(&mut ctx);
        if let Some(mut caches) = self.kv_caches.remove(cache_id) {
            // Paged mode: return the cache_id's blocks to the shared
            // allocator so other sequences can reuse them. Without this,
            // every request consumes max_blocks_per_seq blocks
            // permanently — pool exhausts after FERRUM_PAGED_MAX_SEQS
            // requests and subsequent ensure_kv panics with
            // "scratch residual missing" (the cascade panic from a
            // failed ensure_kv path leaving scratch poisoned).
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_all_graphs(&mut ctx);
        B::sync(&mut ctx);
        self.kv_caches.clear();
        self.kv_free_pool.clear();
    }
}

/// Build a stub Linear<B> with the given shape but zero weights. Used to
/// fill the dense `gate_up_proj` / `down_proj` slots in `LlamaFamilyLayer`
/// for MoE models — those slots are never invoked because the MoE FFN
/// path runs through `moe_layer.experts` instead. The stub's only purpose
/// is to satisfy the struct's type signature with minimal memory cost.
fn stub_linear<B: QuantLlmBackend + BackendMoeFused>(
    out_features: usize,
    in_features: usize,
) -> Box<dyn ferrum_quantization::Linear<B>> {
    // Zero-init: out_features * in_features f32. For a 30B-A3B layer
    // this is 2*768*2048 = 3.1M f32 = 12 MB → fine; per-layer overhead
    // ≈ 12 MB × 48 = 576 MB. Marginal vs the experts (~16 GB).
    let zeros = vec![0.0f32; out_features * in_features];
    Box::new(ferrum_quantization::DenseLinear::<B>::from_rows(
        &zeros,
        out_features,
        in_features,
    ))
}

fn build_rope_cache<B: QuantLlmBackend + BackendMoeFused>(cfg: &LlamaFamilyConfig) -> RopeCache<B> {
    let hd = cfg.head_dim;
    let half = hd / 2;
    let max = cfg.max_seq_len;
    let mut cos = vec![0.0f32; max * half];
    let mut sin = vec![0.0f32; max * half];
    for pos in 0..max {
        for i in 0..half {
            let freq = 1.0f64 / cfg.rope_theta.powf((2 * i) as f64 / hd as f64);
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    RopeCache {
        cos: B::from_slice(&cos),
        sin: B::from_slice(&sin),
    }
}
