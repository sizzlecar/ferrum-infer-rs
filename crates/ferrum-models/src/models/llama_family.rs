//! Llama-family decoder model as explicit code.
//!
//! Covers all "standard Llama-style GQA + SwiGLU + RoPE" decoders:
//!   - Llama / Llama-2 / Llama-3  (no QK-norm)
//!   - Qwen2 / Qwen2.5            (no QK-norm, structurally Llama)
//!   - Qwen3                      (QK-norm per head, larger rope_theta)
//!   - Mistral                    (sliding-window attention — not yet
//!                                 supported on the forward path; will
//!                                 require an `AttnConfig.sliding_window`
//!                                 field + shader support in Phase D)
//!
//! Variant differences are controlled by `LlamaFamilyConfig::has_qk_norm`
//! and RoPE theta. Weight loading accepts both fused (`qkv_proj`,
//! `gate_up_proj`) and split (`q_proj`+`k_proj`+`v_proj`,
//! `gate_proj`+`up_proj`) projection layouts — the loader (e.g.
//! `CandleVarBuilderLoader`) fuses split weights on load so model code
//! sees a uniform `qkv_proj` / `gate_up_proj` Linear.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

use ferrum_kernels::backend::{Backend, KvCache, MAX_LAYERS_FOR_GRAPH};

/// Graph cache key for the single-item decode path (`decode_internal`).
/// Distinct from any `m_padded`-based key used by the batched path.
const SINGLE_ITEM_GRAPH_KEY: u64 = 0;

/// Diag counters for the batched graph dispatcher (replay vs eager).
static BATCHED_GRAPH_REPLAY_COUNT: AtomicU64 = AtomicU64::new(0);
static BATCHED_GRAPH_EAGER_COUNT: AtomicU64 = AtomicU64::new(0);

static ATTN_TIME_US: AtomicU64 = AtomicU64::new(0);
static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
static QKR_TIME_US: AtomicU64 = AtomicU64::new(0);
static QKR_CALLS: AtomicU64 = AtomicU64::new(0);
static MATMUL_TIME_US: AtomicU64 = AtomicU64::new(0);
static MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static NORM_TIME_US: AtomicU64 = AtomicU64::new(0);
static NORM_CALLS: AtomicU64 = AtomicU64::new(0);
static OTHER_TIME_US: AtomicU64 = AtomicU64::new(0);
static OTHER_CALLS: AtomicU64 = AtomicU64::new(0);
use ferrum_quantization::{Linear, WeightLoader};
use ferrum_types::Result;

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};

/// Full Qwen3 architecture config (everything the model code needs, not just
/// the engine-facing subset in `LlmRuntimeConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaFamilyConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Whether the checkpoint has `q_norm` / `k_norm` per layer. All known
    /// Qwen3 checkpoints do; some derivatives may strip them.
    pub has_qk_norm: bool,
    /// Sliding-window attention size. `0` disables (full causal).
    /// Mistral v0.1 sets 4096; Mistral v0.2+ removed the limitation (0).
    pub sliding_window: usize,
}

impl LlamaFamilyConfig {
    pub fn to_runtime(&self) -> LlmRuntimeConfig {
        LlmRuntimeConfig {
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
        }
    }

    /// Build config from a `ModelDefinition`, shared field extraction.
    /// Variant-specific constructors below set `has_qk_norm` and fall back
    /// to different `rope_theta` defaults when the checkpoint doesn't set one.
    fn from_def_base(def: &crate::definition::ModelDefinition) -> LlamaFamilyConfigBase {
        let num_kv_heads = def.num_key_value_heads.unwrap_or(def.num_attention_heads);
        let head_dim = def
            .extra_params
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(def.hidden_size / def.num_attention_heads);
        // Mistral / Gemma: "sliding_window" may be null (v0.2+) or a positive
        // integer (v0.1). Non-null value passes through; missing/null → 0.
        let sliding_window = def
            .extra_params
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        LlamaFamilyConfigBase {
            hidden_size: def.hidden_size,
            intermediate_size: def.intermediate_size,
            num_heads: def.num_attention_heads,
            num_kv_heads,
            head_dim,
            num_layers: def.num_hidden_layers,
            vocab_size: def.vocab_size,
            max_seq_len: def.max_position_embeddings,
            rms_norm_eps: def.norm_eps as f32,
            rope_theta_opt: def.rope_theta,
            sliding_window,
        }
    }

    fn from_base(b: LlamaFamilyConfigBase, rope_default: f64, has_qk_norm: bool) -> Self {
        Self {
            hidden_size: b.hidden_size,
            intermediate_size: b.intermediate_size,
            num_heads: b.num_heads,
            num_kv_heads: b.num_kv_heads,
            head_dim: b.head_dim,
            num_layers: b.num_layers,
            vocab_size: b.vocab_size,
            max_seq_len: b.max_seq_len,
            rms_norm_eps: b.rms_norm_eps,
            rope_theta: b.rope_theta_opt.unwrap_or(rope_default),
            has_qk_norm,
            sliding_window: b.sliding_window,
        }
    }

    /// Qwen3: QK-norm on, rope_theta default 1e6.
    pub fn qwen3_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 1_000_000.0, true)
    }

    /// Llama / Llama-2 / Llama-3: no QK-norm; rope_theta varies by version
    /// (10k for Llama-2, 500k for Llama-3.1+) — use the checkpoint value or
    /// fall back to the most common modern value.
    pub fn llama_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 500_000.0, false)
    }

    /// Qwen2 / Qwen2.5: structurally Llama; no QK-norm; rope_theta default 1e6.
    pub fn qwen2_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 1_000_000.0, false)
    }

    /// Mistral: no QK-norm; `rope_theta` commonly 10_000 (v0.1 / v0.2).
    /// Picks up `sliding_window` from the checkpoint's config.json
    /// (Mistral v0.1: 4096; Mistral v0.2+: 0 / null).
    pub fn mistral_from_def(def: &crate::definition::ModelDefinition) -> Self {
        Self::from_base(Self::from_def_base(def), 10_000.0, false)
    }
}

struct LlamaFamilyConfigBase {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    max_seq_len: usize,
    rms_norm_eps: f32,
    rope_theta_opt: Option<f64>,
    sliding_window: usize,
}

/// Per-layer weights. `Box<dyn Linear<B>>` means each projection can be
/// Dense / GPTQ / AWQ / GGUF without the surrounding code caring.
pub struct LlamaFamilyLayer<B: Backend> {
    pub input_ln_w: B::Buffer,
    pub qkv_proj: Box<dyn Linear<B>>,
    /// QK-norm weight per head: `[head_dim]`. Optional for non-Qwen3 derivatives.
    pub q_norm_w: Option<B::Buffer>,
    pub k_norm_w: Option<B::Buffer>,
    pub o_proj: Box<dyn Linear<B>>,
    pub post_ln_w: B::Buffer,
    pub gate_up_proj: Box<dyn Linear<B>>,
    pub down_proj: Box<dyn Linear<B>>,
}

/// Precomputed RoPE cos/sin tables (shape `[max_seq, head_dim / 2]` each).
pub struct RopeCache<B: Backend> {
    pub cos: B::Buffer,
    pub sin: B::Buffer,
}

/// Reusable per-layer scratch buffers sized for `max_tokens` tokens of a
/// single forward pass (prefill or decode step).
///
/// Sized lazily on first use so tiny decode steps don't pay for prefill-sized
/// buffers. Grows monotonically when a larger prefill arrives.
pub struct LlamaFamilyScratch<B: Backend> {
    /// Residual stream — wrapped in Option so decode_internal can
    /// `.take()` it without needing an alloc placeholder.
    ///
    /// Why this matters for graph capture: the old pattern was
    /// `mem::replace(&mut scratch.residual, B::alloc(1))` which creates a
    /// 1-element buffer at every decode step. When graph capture is on,
    /// that alloc-during-capture + drop-after-capture pair surfaces as
    /// cuMemFreeAsync(INVALID_VALUE) because the free tries to release a
    /// pointer the captured graph may still reference. Option::take leaves
    /// None and moves the real buffer into a local — no spurious alloc.
    pub residual: Option<B::Buffer>,
    pub norm_out: B::Buffer,
    pub qkv_out: B::Buffer,
    // ── Per-item scratch for batched decode path ──────────────────────
    // decode_batch_internal runs tokens=M batched ops for the GEMM-heavy
    // half (norm, qkv_proj, split_qkv, o_proj, post_norm, gate_up, silu,
    // down, residual_add) but must loop per-item for rope + KV append +
    // attention (each item has its own KV cache at a different kv_len).
    // These single-item buffers hold item i's slice during that loop.
    /// Item-scope q_buf slice, sized `q_dim`.
    pub q_single: B::Buffer,
    pub k_single: B::Buffer,
    pub v_single: B::Buffer,
    pub q_head_major_single: B::Buffer,
    pub k_head_major_single: B::Buffer,
    pub v_head_major_single: B::Buffer,
    pub attn_head_major_single: B::Buffer,
    pub attn_flat_single: B::Buffer,
    /// Batched logits output, sized `max_tokens * vocab_size`. Used only
    /// in decode_batch; prefill/single-decode use the regular `logits`.
    pub batch_logits: B::Buffer,
    /// Token-major Q/K/V right after `split_qkv`. Stride: heads * hd per row.
    pub q_buf: B::Buffer,
    pub k_buf: B::Buffer,
    pub v_buf: B::Buffer,
    /// Head-major Q produced by `qk_norm_rope` — fed into `flash_attention`.
    pub q_head_major: B::Buffer,
    /// Head-major K/V staging — produced by `qk_norm_rope`, consumed by
    /// `kv_cache_append_head_major` (no reuse after append).
    pub k_head_major: B::Buffer,
    pub v_head_major: B::Buffer,
    /// Head-major attention output from `flash_attention`.
    pub attn_head_major_out: B::Buffer,
    /// Token-major attention output after `transpose_head_to_token`.
    pub attn_flat: B::Buffer,
    pub o_proj_out: B::Buffer,
    pub gate_up_out: B::Buffer,
    pub silu_out: B::Buffer,
    pub mlp_out: B::Buffer,
    /// Paged batched dispatch scratch (Phase 4b). Sized for
    /// `FERRUM_PAGED_MAX_SEQS × q_dim` so multi-seq decode can fan
    /// in M items' Q into a single buffer for one batched
    /// `paged_decode_attention(num_seqs=M)` call. `None` when paged
    /// mode is off.
    pub paged_batch_q: Option<B::Buffer>,
    pub paged_batch_o: Option<B::Buffer>,
    /// Stacked per-seq block tables for batched paged dispatch.
    /// Layout: `[max_M, max_blocks_per_seq]` u32. Written
    /// host-side per decode_batch step.
    pub paged_batch_block_tables: Option<B::Buffer>,
    /// Stacked per-seq context lengths for batched paged dispatch
    /// (`[max_M]` u32).
    pub paged_batch_context_lens: Option<B::Buffer>,
    /// `max_blocks_per_seq` value baked into the stacked block_tables
    /// stride. Set when `paged_batch_block_tables` is allocated.
    pub paged_max_blocks_per_seq: usize,
    /// Per-item RoPE positions for the batched-decode path (`[max_M]`
    /// i32 / u32). Written host-side once per batched-decode step from
    /// each request's `pos` field, read by the batched
    /// `qk_norm_rope_batched_per_item` CUDA kernel.
    pub batch_positions: B::Buffer,
    /// Per-item input token ids for the batched-decode path (`[max_M]`
    /// u32). Written once per call before forward; read by the
    /// graph-capture-friendly `embedding_lookup_batched_dyn` variant.
    pub batch_tokens: B::Buffer,
    /// Per-item KV-cache length BEFORE this step's kv_append
    /// (`[max_M]` u32). Used by `kv_cache_append_batched_per_cache_dyn`
    /// to write at the right slot for graph replay.
    pub batch_kv_lens_pre: B::Buffer,
    /// Per-item KV-cache length AFTER this step's kv_append
    /// (`[max_M]` u32 = pre + 1). Used by
    /// `flash_attention_batched_per_cache_dyn` for the attention
    /// reduce window length.
    pub batch_kv_lens_post: B::Buffer,
    /// Output buffers for the batched per-item qk_norm_rope kernel.
    /// Same shape as q_buf / k_buf / v_buf — separate so the kernel
    /// API can take `&input` and `&mut output` without aliasing.
    pub q_normed_batched: B::Buffer,
    pub k_normed_batched: B::Buffer,
    pub v_normed_batched: B::Buffer,

    // ── Unified mixed-batch scratch (chunked-prefill path; Step 5b) ─────
    // Buffers sized for `M_total = sum(items[i].q_tokens.len())`. Grown
    // on demand by `ensure_unified_scratch(M_total_max)`. Used only by
    // the new `unified_forward_internal`; legacy `forward_layer_batched_decode`
    // continues to use the per-item-stride scratch above.
    pub unified_capacity: usize, // current allocated M_total slots
    pub unified_residual: Option<B::Buffer>,
    pub unified_norm_out: Option<B::Buffer>,
    pub unified_qkv_out: Option<B::Buffer>,
    pub unified_packed_q: Option<B::Buffer>,
    pub unified_attn_out: Option<B::Buffer>,
    pub unified_o_proj_out: Option<B::Buffer>,
    pub unified_gate_up_out: Option<B::Buffer>,
    pub unified_silu_out: Option<B::Buffer>,
    pub unified_mlp_out: Option<B::Buffer>,
    /// Per-item index buffers (i32-stored-as-f16): cu_seqlens_q is
    /// length `max_seqs+1`, pos_offsets is `max_seqs`, block_tables is
    /// `max_seqs * max_blocks_per_seq`. Sized once at first use to
    /// match `paged_batch_*` capacity since they share `max_seqs`.
    pub unified_cu_seqlens_q: Option<B::Buffer>,
    pub unified_pos_offsets: Option<B::Buffer>,
    pub unified_block_tables: Option<B::Buffer>,
    /// Packed last-token hidden states for is_final_chunk items
    /// (`[num_sampled, h]`). Used as input to lm_head.
    pub unified_packed_normed: Option<B::Buffer>,
    /// Packed logits output (`[num_sampled, vocab]`).
    pub unified_packed_logits: Option<B::Buffer>,
    /// Last token's hidden state (`[h]`). For prefill this is populated via
    /// `copy_slice(residual, (seq_len-1)*h, ..)`; for decode `residual` already
    /// holds only 1 row so `last_hidden` is unused on that path.
    pub last_hidden: B::Buffer,
    /// Final-norm output for the last token (`[h]`).
    pub last_normed: B::Buffer,
    /// lm_head logits (`[vocab]`).
    pub logits: B::Buffer,
    /// The max tokens-per-step this scratch has been sized for.
    pub max_tokens: usize,
}

impl<B: Backend> LlamaFamilyScratch<B> {
    fn alloc(cfg: &LlamaFamilyConfig, max_tokens: usize) -> Self {
        let h = cfg.hidden_size;
        let im = cfg.intermediate_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let t = max_tokens;
        Self {
            residual: Some(B::alloc(t * h)),
            norm_out: B::alloc(t * h),
            qkv_out: B::alloc(t * qkv_dim),
            q_buf: B::alloc(t * q_dim),
            k_buf: B::alloc(t * kv_dim),
            v_buf: B::alloc(t * kv_dim),
            q_head_major: B::alloc(cfg.num_heads * t * cfg.head_dim),
            k_head_major: B::alloc(cfg.num_kv_heads * t * cfg.head_dim),
            v_head_major: B::alloc(cfg.num_kv_heads * t * cfg.head_dim),
            attn_head_major_out: B::alloc(cfg.num_heads * t * cfg.head_dim),
            attn_flat: B::alloc(t * q_dim),
            o_proj_out: B::alloc(t * h),
            gate_up_out: B::alloc(t * 2 * im),
            silu_out: B::alloc(t * im),
            mlp_out: B::alloc(t * h),
            last_hidden: B::alloc(h),
            last_normed: B::alloc(h),
            logits: B::alloc(cfg.vocab_size),
            q_single: B::alloc(q_dim),
            k_single: B::alloc(kv_dim),
            v_single: B::alloc(kv_dim),
            q_head_major_single: B::alloc(q_dim),
            k_head_major_single: B::alloc(kv_dim),
            v_head_major_single: B::alloc(kv_dim),
            attn_head_major_single: B::alloc(q_dim),
            attn_flat_single: B::alloc(q_dim),
            batch_logits: B::alloc(t * cfg.vocab_size),
            // Paged batched dispatch scratch. None until `enable_paged_batch`
            // is called from `ensure_kv` once the model knows max_seqs +
            // max_blocks_per_seq. This avoids paying the alloc cost when
            // paged mode is off.
            paged_batch_q: None,
            paged_batch_o: None,
            paged_batch_block_tables: None,
            paged_batch_context_lens: None,
            paged_max_blocks_per_seq: 0,
            batch_positions: B::alloc_u32(t.max(1)),
            batch_tokens: B::alloc_u32(t.max(1)),
            batch_kv_lens_pre: B::alloc_u32(t.max(1)),
            batch_kv_lens_post: B::alloc_u32(t.max(1)),
            q_normed_batched: B::alloc(t * q_dim),
            k_normed_batched: B::alloc(t * kv_dim),
            v_normed_batched: B::alloc(t * kv_dim),
            unified_capacity: 0,
            unified_residual: None,
            unified_norm_out: None,
            unified_qkv_out: None,
            unified_packed_q: None,
            unified_attn_out: None,
            unified_o_proj_out: None,
            unified_gate_up_out: None,
            unified_silu_out: None,
            unified_mlp_out: None,
            unified_cu_seqlens_q: None,
            unified_pos_offsets: None,
            unified_block_tables: None,
            unified_packed_normed: None,
            unified_packed_logits: None,
            max_tokens: t,
        }
    }

    /// Grow unified-path scratch buffers to accommodate `m_total` query
    /// tokens. Called lazily from `unified_forward_internal` so single-
    /// path workloads (no chunked prefill) don't pay the alloc cost.
    fn ensure_unified_scratch(
        &mut self,
        cfg: &LlamaFamilyConfig,
        m_total: usize,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    ) {
        if m_total <= self.unified_capacity
            && self.unified_residual.is_some()
            && self.unified_cu_seqlens_q.is_some()
        {
            return;
        }
        let cap = m_total.max(self.unified_capacity).max(1);
        let h = cfg.hidden_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let im = cfg.intermediate_size;
        let v = cfg.vocab_size;
        self.unified_residual = Some(B::alloc(cap * h));
        self.unified_norm_out = Some(B::alloc(cap * h));
        self.unified_qkv_out = Some(B::alloc(cap * qkv_dim));
        self.unified_packed_q = Some(B::alloc(cap * q_dim));
        self.unified_attn_out = Some(B::alloc(cap * q_dim));
        self.unified_o_proj_out = Some(B::alloc(cap * h));
        self.unified_gate_up_out = Some(B::alloc(cap * 2 * im));
        self.unified_silu_out = Some(B::alloc(cap * im));
        self.unified_mlp_out = Some(B::alloc(cap * h));
        if self.unified_cu_seqlens_q.is_none() {
            self.unified_cu_seqlens_q = Some(B::alloc_u32(max_seqs + 1));
            self.unified_pos_offsets = Some(B::alloc_u32(max_seqs));
            self.unified_block_tables = Some(B::alloc_u32(max_seqs * max_blocks_per_seq));
            self.unified_packed_normed = Some(B::alloc(max_seqs * h));
            self.unified_packed_logits = Some(B::alloc(max_seqs * v));
        }
        self.unified_capacity = cap;
    }

    /// Allocate scratch for batched paged dispatch (Phase 4b). Called
    /// lazily from `ensure_kv` once paged mode is enabled and we know
    /// the pool dimensions. Idempotent.
    fn enable_paged_batch(
        &mut self,
        cfg: &LlamaFamilyConfig,
        max_seqs: usize,
        max_blocks_per_seq: usize,
    ) {
        if self.paged_batch_q.is_some() {
            return;
        }
        let q_dim = cfg.num_heads * cfg.head_dim;
        self.paged_batch_q = Some(B::alloc(max_seqs * q_dim));
        self.paged_batch_o = Some(B::alloc(max_seqs * q_dim));
        self.paged_batch_block_tables = Some(B::alloc_u32(max_seqs * max_blocks_per_seq));
        self.paged_batch_context_lens = Some(B::alloc_u32(max_seqs));
        self.paged_max_blocks_per_seq = max_blocks_per_seq;
    }
}

/// Qwen3 model — decoder-only LLM, one per (backend, weights) combination.
///
/// Holds all parameters, scratch space, RoPE cache, and per-sequence KV caches.
pub struct LlamaFamilyModel<B: Backend> {
    pub cfg: LlamaFamilyConfig,
    pub runtime_cfg: LlmRuntimeConfig,

    /// Token embedding table. `None` for backbone-only models (e.g. the
    /// Qwen3-TTS Talker, which embeds inputs externally and feeds via
    /// `prefill_from_embeds`).
    pub embed: Option<B::Buffer>,
    pub layers: Vec<LlamaFamilyLayer<B>>,
    pub final_norm_w: B::Buffer,
    /// LM output head. `None` for backbone-only models.
    pub lm_head: Option<Box<dyn Linear<B>>>,

    pub rope: RopeCache<B>,
    pub scratch: LlamaFamilyScratch<B>,

    /// Per-sequence KV caches, one `Vec<KvCache<B>>` of length `num_layers`.
    ///
    /// Two layouts overlay this same map:
    /// - **Contiguous mode** (default): each cache holds its own
    ///   `[num_kv_heads, capacity, head_dim]` k/v buffers.
    /// - **Paged mode** (`FERRUM_METAL_PAGED_KV=1`): k/v are unused
    ///   placeholders; the real K/V live in [`Self::paged_pools`] and
    ///   the cache's `block_table` + `context_lens` index into them.
    pub kv_caches: HashMap<String, Vec<KvCache<B>>>,
    /// Free pool of pre-allocated KV cache slots. Released caches return
    /// here instead of being dropped, so their device pointers stay valid
    /// across requests — critical for graph capture (pointers baked into
    /// the captured graph would otherwise dangle).
    kv_free_pool: Vec<Vec<KvCache<B>>>,

    // ── Paged-KV multi-seq state (Phase 4) ─────────────────────────────
    //
    // Only populated when `FERRUM_METAL_PAGED_KV=1`. When set, every
    // `kv_caches` entry becomes a "view" into the shared pool: its
    // `k` / `v` buffers are placeholders; reads / writes go through
    // `paged_pools[layer].k` / `.v` indexed via the cache's
    // `block_table`. Multiple cache_ids share the same pool, with
    // disjoint physical block sets owned by `paged_block_alloc`.
    //
    /// Shared K/V pools, one per layer. Sized at model load time for the
    /// configured `MAX_RUNNING_REQUESTS × max_blocks_per_seq` blocks.
    pub paged_pools: Option<Vec<(B::Buffer, B::Buffer)>>,
    /// Block allocator hands out physical block indices from the pool.
    /// `Mutex` because the engine can call `ensure_kv` / `release_kv`
    /// from multiple threads in concurrent serving.
    pub paged_block_alloc: Option<std::sync::Mutex<crate::common::paged_pool::BlockAllocator>>,

    // ── Graph capture state (CUDA only; harmless no-op on other backends) ──
    /// Count of eager decode steps run so far. After `GRAPH_WARMUP`, the
    /// next step captures the decode flow as a graph.
    graph_warmup: usize,
    /// True if capture was attempted but failed (e.g. backend doesn't
    /// support graph capture). Stops further attempts, falls back to eager.
    graph_capture_failed: bool,
    /// Same warmup counter for the batched-decode path.
    batched_graph_warmup: usize,
    /// True if batched graph capture failed.
    batched_graph_failed: bool,
    /// Set of `m_padded` values (as u64 graph keys) for which a batched
    /// graph has been captured. Multi-slot via cuda.rs's HashMap-keyed
    /// graph cache — different batch shapes don't thrash a single slot.
    batched_graph_keys_seen: std::collections::HashSet<u64>,
    /// Cache IDs for which device-pointer scratch is currently populated.
    /// Populate only re-runs when the batch composition changes (new
    /// requests joined / requests finished). Hot-path optimization:
    /// avoids 3 sync cuMemcpyHtoD_v2's per decode token (~5% TPOT).
    batched_pointers_for: Option<Vec<String>>,
}

impl<B: Backend> LlamaFamilyModel<B> {
    /// Build a Qwen3 model from weights provided by the loader.
    ///
    /// The loader decides per-projection whether to instantiate DenseLinear,
    /// GptqLinear, etc. — this code doesn't care.
    pub fn new(cfg: LlamaFamilyConfig, loader: &dyn WeightLoader<B>) -> Result<Self> {
        // Invalidate any graph from a previously-loaded model. The captured
        // graph references the old model's scratch buffers; a fresh model
        // gets fresh scratch, so reusing the graph would read/write freed
        // pointers. Matters for test suites where multiple models coexist.
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let rope = build_rope_cache::<B>(&cfg);
        let scratch = LlamaFamilyScratch::alloc(&cfg, 1); // decode-sized; prefill resizes

        // Embedding: plain tensor (no projection math, just lookup).
        let embed = loader.load_tensor("model.embed_tokens.weight")?;

        // Per-layer weights.
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for li in 0..cfg.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w =
                loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let gate_up_proj = loader.load_linear(&format!("{prefix}.mlp.gate_up_proj"))?;
            let down_proj = loader.load_linear(&format!("{prefix}.mlp.down_proj"))?;

            let (q_norm_w, k_norm_w) = if cfg.has_qk_norm {
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

            layers.push(LlamaFamilyLayer {
                input_ln_w,
                qkv_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_ln_w,
                gate_up_proj,
                down_proj,
            });
        }

        let final_norm_w = loader.load_tensor("model.norm.weight")?;

        // LM head: either dedicated `lm_head.weight` or tied to embedding.
        // Many models (Qwen3-4B, Llama-3.2-1B, some Qwen2.5) use TIED
        // embeddings — lm_head shares weights with model.embed_tokens. When
        // no dedicated lm_head tensor exists, re-load the embed tensor as a
        // DenseLinear. This duplicates the buffer (memory cost = vocab*h*2
        // bytes, e.g. ~770MB for Qwen3-4B) but keeps the Linear trait's
        // owned-weights invariant. Sharing via Arc is a future optimisation.
        let lm_head = if loader.has_tensor("lm_head.weight") {
            loader.load_linear("lm_head")?
        } else {
            tracing::info!(
                "LlamaFamilyModel: tied embeddings — loading model.embed_tokens.weight as lm_head"
            );
            let as_linear = loader.load_linear("model.embed_tokens")?;
            // Sanity check: shape must be [vocab, hidden].
            if as_linear.out_features() != cfg.vocab_size
                || as_linear.in_features() != cfg.hidden_size
            {
                return Err(ferrum_types::FerrumError::model(format!(
                    "tied embed shape mismatch: got [{}, {}], expected [{}, {}]",
                    as_linear.out_features(),
                    as_linear.in_features(),
                    cfg.vocab_size,
                    cfg.hidden_size
                )));
            }
            as_linear
        };

        let runtime_cfg = cfg.to_runtime();
        Ok(Self {
            cfg,
            runtime_cfg,
            embed: Some(embed),
            layers,
            final_norm_w,
            lm_head: Some(lm_head),
            rope,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_block_alloc: None,
            graph_warmup: 0,
            graph_capture_failed: false,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            batched_pointers_for: None,
        })
    }

    /// Build a backbone-only Qwen3 transformer stack (no embed, no lm_head).
    ///
    /// Intended for composing the transformer inside a larger model where
    /// embedding and output-head logic differs from the standard LLM path —
    /// e.g. Qwen3-TTS Talker uses dual text/codec embeddings with a projection
    /// MLP, and a codec_head output. The caller drives forward via
    /// `prefill_from_embeds` / `decode_from_embed`.
    ///
    /// Loader must provide: per-layer weights under `model.layers.{i}.*` and
    /// the final `model.norm.weight`. `model.embed_tokens` and `lm_head`
    /// are NOT read.
    pub fn new_backbone_only(cfg: LlamaFamilyConfig, loader: &dyn WeightLoader<B>) -> Result<Self> {
        // See `new` — invalidate stale graph referring to prior model's scratch.
        {
            let mut ctx = B::new_context();
            B::reset_all_graphs(&mut ctx);
        }
        let rope = build_rope_cache::<B>(&cfg);
        let scratch = LlamaFamilyScratch::alloc(&cfg, 1);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for li in 0..cfg.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w =
                loader.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let gate_up_proj = loader.load_linear(&format!("{prefix}.mlp.gate_up_proj"))?;
            let down_proj = loader.load_linear(&format!("{prefix}.mlp.down_proj"))?;

            let (q_norm_w, k_norm_w) = if cfg.has_qk_norm {
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

            layers.push(LlamaFamilyLayer {
                input_ln_w,
                qkv_proj,
                q_norm_w,
                k_norm_w,
                o_proj,
                post_ln_w,
                gate_up_proj,
                down_proj,
            });
        }

        let final_norm_w = loader.load_tensor("model.norm.weight")?;

        let runtime_cfg = cfg.to_runtime();
        Ok(Self {
            cfg,
            runtime_cfg,
            embed: None,
            layers,
            final_norm_w,
            lm_head: None,
            rope,
            scratch,
            kv_caches: HashMap::new(),
            kv_free_pool: Vec::new(),
            paged_pools: None,
            paged_block_alloc: None,
            graph_warmup: 0,
            graph_capture_failed: false,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            batched_pointers_for: None,
        })
    }

    /// Grow scratch buffers if `tokens` exceeds the current sizing.
    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            // Any captured decode graph holds pointers to the old scratch
            // buffers; those are about to be freed. Invalidate ALL captured
            // graphs (both single-item and per-m_padded batched) — every
            // captured kernel-arg pointer into scratch is stale.
            {
                let mut ctx = B::new_context();
                B::reset_all_graphs(&mut ctx);
            }
            self.scratch = LlamaFamilyScratch::alloc(&self.cfg, tokens);
            self.graph_warmup = 0;
            self.graph_capture_failed = false;
            self.batched_graph_keys_seen.clear();
            self.batched_graph_warmup = 0;
            self.batched_graph_failed = false;
        }
    }

    /// Ensure per-layer KV caches exist for `cache_id`, pre-allocated to
    /// `max_seq_len` slots per head. Enables the in-place
    /// `kv_cache_append_head_major` path — no realloc per layer.
    pub(crate) fn ensure_kv(&mut self, cache_id: &str) {
        if self.kv_caches.contains_key(cache_id) {
            return;
        }
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        // KV capacity defaults to a chat-friendly 4096 to keep the working
        // set sane on a 32 GB Mac (a 48-layer / 4-kv-head / 128-head_dim
        // model spends ~786 MB on 4096 KV slots, vs ~6 GB on the model's
        // declared 32K which would push the 17 GB Qwen3-30B-A3B model into
        // swap). `FERRUM_KV_CAPACITY=N` overrides; clamp to the model's
        // declared max so we never lie to the model about its window.
        let model_max = self.cfg.max_seq_len;
        // 512 in 0.7.2 — matches the value used in
        // docs/bench/macos-2026-05-02 to get the published numbers.
        // pre-0.7.2 default of 4096 was safe only because paged-KV was
        // opt-in (pool wasn't allocated). With paged-KV now on by
        // default + MAX_SEQS=32, the pool occupies physical memory:
        // ~3 GB on Qwen3-30B-A3B Q4_K_M leaves 18 GB weights + 3 GB pool
        // = 21 GB, fits comfortably on a 32 GB Mac. Long-context users
        // can `FERRUM_KV_CAPACITY=4096` and accept lower max_seqs.
        const DEFAULT_KV_CAPACITY: usize = 512;
        let max = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY));

        // Paged-KV mode: `FERRUM_METAL_PAGED_KV=1` switches every cache
        // for this model into block-table-indirect layout. Kernels from
        // PR #68 (decode read) + PR #69 (decode write) handle the
        // indirect addressing; the LlamaFamily decode path below picks
        // them up automatically by checking `cache.block_size > 0`.
        //
        // Pool sizing: round capacity up to a multiple of block_size,
        // identity-assign logical→physical block. Memory footprint is
        // the same as contiguous (within block_size rounding); the
        // benefit only shows up under multi-seq sharing in Phase 4+.
        // Default ON when the backend supports paged-KV (Metal). Users
        // can force off with `FERRUM_METAL_PAGED_KV=0`. The flag was
        // opt-in pre-0.7.2; flipping the default so default `ferrum
        // serve` matches the bench-quality numbers without requiring
        // env-var knowledge.
        let paged = std::env::var("FERRUM_METAL_PAGED_KV")
            .map(|v| v != "0")
            .unwrap_or_else(|_| B::supports_paged_kv());
        const PAGED_BLOCK_SIZE: usize = 16;

        // Phase 4 shared-pool sizing. The pool sees ALL concurrent
        // sequences; per-cache_id state just owns indices into it.
        // Default 32: covers c=16 burst with 2× headroom for the
        // fresh-cache-id-per-request pattern that bench/server harnesses
        // use. Pool memory is `max_seqs × max_blocks_per_seq` total
        // blocks — we lowered DEFAULT_KV_CAPACITY to 2048 so this 2× max_seqs
        // bump keeps the pool footprint identical to the pre-0.7.2 default.
        let max_seqs = std::env::var("FERRUM_PAGED_MAX_SEQS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32);
        let max_blocks_per_seq = max.div_ceil(PAGED_BLOCK_SIZE);
        let total_pool_blocks = max_seqs * max_blocks_per_seq;

        // Lazy-allocate the shared paged pools on the FIRST paged
        // ensure_kv call. Pools are big — for Llama-8B (8 kv_heads,
        // head_dim=128) at 16 seqs × 256 blocks × 16 slots = 65536 KV
        // slots: 65536 * 8 * 128 * 4 = 256 MB per layer × 32 layers
        // = 8 GB total. Sized this large only because `max_seqs=16`
        // is the default; lower it via env to shrink.
        if paged && self.paged_pools.is_none() {
            let mut pools = Vec::with_capacity(self.cfg.num_layers);
            for _ in 0..self.cfg.num_layers {
                let pool_floats = total_pool_blocks * nkv * PAGED_BLOCK_SIZE * hd;
                pools.push((B::alloc(pool_floats), B::alloc(pool_floats)));
            }
            self.paged_pools = Some(pools);
            self.paged_block_alloc = Some(std::sync::Mutex::new(
                crate::common::paged_pool::BlockAllocator::new(total_pool_blocks as u32),
            ));
        }
        // Phase 4b: ensure batched-dispatch scratch is allocated whenever
        // paged is on. Idempotent — re-init is a no-op if already
        // sized. Has to live outside the `paged_pools.is_none()` branch
        // because `ensure_scratch` may have replaced the scratch struct
        // since the pools were first allocated.
        if paged {
            self.scratch
                .enable_paged_batch(&self.cfg, max_seqs, max_blocks_per_seq);
        }

        // Try pool first — reused buffers have stable device pointers,
        // so a captured decode graph can be replayed for this request too.
        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.num_layers)
                .map(|_| {
                    if paged {
                        // Paged mode: cache holds metadata only. K/V
                        // are 1-element placeholders (allocated cheaply
                        // since Backend::alloc requires a non-zero
                        // size on most backends). The real data lives
                        // in `self.paged_pools[li].{k,v}`.
                        let mut block_table = B::alloc_u32(max_blocks_per_seq);
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
                        }
                    }
                })
                .collect()
        });

        // Allocate physical blocks for THIS cache_id from the shared
        // pool. We allocate all `max_blocks_per_seq` upfront for
        // simplicity (matches contig's "pre-alloc to capacity"
        // semantics); a smarter Phase 4b can grow on demand to save
        // pool occupancy.
        if paged {
            let alloc_arc = self
                .paged_block_alloc
                .as_ref()
                .expect("paged_block_alloc must be initialised when paged=true");
            // Recover from a previously-poisoned mutex instead of panicking
            // (poison just means a prior holder panicked; the BlockAllocator
            // state is still intact since allocate_n is fail-safe).
            let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
            let block_indices = match alloc.allocate_n(max_blocks_per_seq) {
                Ok(idx) => idx,
                Err(e) => {
                    // Pool exhaustion is a back-pressure signal, not a crash.
                    // Drop the lock, return the cache to the free pool, and
                    // bail before inserting it into kv_caches. The downstream
                    // call will then fail with a clean per-request error
                    // ("ensure_kv must be called before ...") instead of
                    // dragging every other in-flight request down with it.
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
            // Write the block table to each layer's cache. All layers
            // share the same logical→physical mapping for this seq.
            // Also stash the host-side index list so release_kv can
            // return them to the allocator without a device readback.
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

        // Reset logical length; buffers stay. No need to zero the memory —
        // the kv_cache_append writes new K/V in place, and attention only
        // reads up to `cache_len`.
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

    /// Run one transformer layer. Mutates `residual` in place.
    ///
    /// `pos_offset` is the absolute position of token 0 in this batch
    /// (decode: `pos`; prefill: 0). `tokens` is the batch size.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_layer(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        cache_id: &str,
        residual: &mut B::Buffer,
        pos_offset: usize,
        tokens: usize,
    ) {
        let layer = &self.layers[li];
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        // 1. Input RMSNorm
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::rms_norm(
            ctx,
            residual,
            &layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            tokens,
            h,
        );
        if let Some(t0) = _t0 {
            B::sync(ctx);
            NORM_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 2. Fused QKV projection (Linear dispatches to Dense/GPTQ/AWQ/GGUF)
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        layer.qkv_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.qkv_out,
            tokens,
        );
        if let Some(t0) = _t0 {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 3-5. Fused split-QKV + QK-norm + RoPE + cache-write.
        //
        // Single Metal dispatch replaces the (split_qkv → 3× qk_norm_rope
        // → kv_cache_append_head_major) five-launch chain on the decode
        // hot path. Reads qkv_out once, writes Q to head-major scratch
        // and K/V straight into the pre-allocated KV cache slot at
        // `cache_len + tok`. Saves 4 dispatches per layer when the
        // backend implements the fused kernel; CPU and other backends
        // keep using the unfused chain via the Unsupported fallbacks.
        //
        // qk_mode: 1 = norm + RoPE (Qwen3); 2 = RoPE only (Llama).
        // V always passes apply_norm=0.
        let qk_mode: i32 = if cfg.has_qk_norm { 1 } else { 2 };
        let dummy = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy);

        // Grab the per-layer KV cache up front so the deepest fusion can
        // write K/V straight into it.
        //
        // Paged mode: also need this layer's shared pool buffers
        // (self.paged_pools[li]). The pool is a separate field from
        // kv_caches, so we take a raw pointer to its (k, v) here while
        // we still hold &mut self, then deref via unsafe inside the
        // paged dispatch below. Safety: paged_pools is allocated once
        // and never resized; we don't touch self.paged_pools while the
        // pointer is in use.
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

        // Defense in depth: refuse to write past the KV buffer. The
        // graceful path is the caller pre-checking via `kv_capacity()`
        // and either compacting or refusing the request; this panic only
        // fires when that contract is broken (and silent overflow would
        // otherwise corrupt the cache + adjacent allocations).
        if cache_len_before + tokens > cache_capacity {
            panic!(
                "KV cache overflow on layer {li}: would write tokens [{cache_len_before}..{}) but capacity is {cache_capacity} (cache_id={cache_id:?}). Increase FERRUM_KV_CAPACITY or call /clear in the REPL.",
                cache_len_before + tokens
            );
        }

        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Paged-KV path: when the cache was allocated with paged
        // metadata (`block_size > 0`), use the paged write kernel
        // which fans out into the block pool via `block_table`.
        // Falls back to contiguous if Backend doesn't implement it.
        let used_qkv_into_cache = if cache.block_size > 0 {
            let bt = cache
                .block_table
                .as_ref()
                .expect("paged cache missing block_table");
            let num_blocks_per_seq = cache.capacity / cache.block_size;
            // Paged mode: K/V live in the shared pool, not cache.k/.v.
            let (pool_k_ptr, pool_v_ptr) =
                paged_pool_ptr.expect("paged_pools must be allocated when block_size > 0");
            // SAFETY: paged_pools is allocated once and never resized;
            // we do not touch self.paged_pools concurrently.
            let pool_k = unsafe { &mut *pool_k_ptr };
            let pool_v = unsafe { &mut *pool_v_ptr };
            B::split_qkv_norm_rope_into_paged_cache(
                ctx,
                &self.scratch.qkv_out,
                0, // qkv_byte_offset: single-seq dispatch reads from start
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                &mut self.scratch.q_head_major,
                0, // q_out_byte_offset: writes to start of head-major scratch
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
            // (PR #47 path).
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
        if let Some(t0) = _t0 {
            B::sync(ctx);
            QKR_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            QKR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        cache.len += tokens;
        let kv_len = cache.len;
        let kv_stride = cache.capacity;

        // 6. Flash attention.
        //    Paged path: when the cache uses block layout, dispatch the
        //    paged_decode_attention kernel; for q_len > 1 (prefill),
        //    iterate token-by-token (kernel only handles q_len=1 right
        //    now — Phase 4 will add a paged Q-tiled path).
        //    Contiguous path: existing flash_attention dispatch.
        let _attn_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        if cache.block_size > 0 {
            let bt = cache
                .block_table
                .as_ref()
                .expect("paged cache missing block_table");
            let cl_buf = cache
                .context_lens
                .as_mut()
                .expect("paged cache missing context_lens");
            let num_blocks_per_seq = cache.capacity / cache.block_size;
            // Paged mode: K/V come from the shared pool.
            let (pool_k_ptr, pool_v_ptr) =
                paged_pool_ptr.expect("paged_pools must be allocated when block_size > 0");
            // SAFETY: same as the write-side above; pool buffers are
            // allocated-once and never moved while we hold the pointer.
            let pool_k = unsafe { &*pool_k_ptr };
            let pool_v = unsafe { &*pool_v_ptr };
            // Single dispatch handles both decode (q_len=1) and causal
            // prefill (q_len>1). The kernel computes per-token causal
            // limit as `context_lens[seq] - (q_len - 1 - q_token_idx)`,
            // so we set context_lens to the FINAL kv_len after this
            // batch's writes.
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
                1, // num_seqs (single-seq dispatch; multi-seq is fan-in via forward_layer_batched, Phase 4b)
                nh,
                nkv,
                hd,
                cache.block_size,
                num_blocks_per_seq,
                tokens, // q_len
            )
            .expect("paged_decode_attention");
        } else {
            //    `causal` is always true for decoder-only LLMs — every query must
            //    mask out future tokens. Sliding-window models (Mistral v0.1) narrow
            //    the lower bound via `sliding_window`.
            let attn_cfg = ferrum_kernels::backend::AttnConfig {
                num_heads: nh,
                num_kv_heads: nkv,
                head_dim: hd,
                causal: true,
                scale: 1.0 / (hd as f32).sqrt(),
                kv_seq_stride: kv_stride,
                sliding_window: cfg.sliding_window,
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
        if let Some(t0) = _attn_t0 {
            B::sync(ctx);
            ATTN_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 7. Untranspose head-major → token-major for O-proj input.
        //
        // For tokens=1 the head-major and token-major layouts collapse
        // to the same flat [heads * head_dim] vector, so the dispatch is
        // an identity memcpy — skip it and point o_proj at the
        // head-major buffer directly.
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
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
        if let Some(t0) = _t0 {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 8. O projection
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        layer
            .o_proj
            .forward(ctx, attn_token_major, &mut self.scratch.o_proj_out, tokens);
        if let Some(t0) = _t0 {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 9. Fused residual-add + post-attention RMSNorm.
        //    Writes the new residual back into `residual` and the normed
        //    value into `norm_out`.
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::fused_add_rms_norm(
            ctx,
            residual,
            &self.scratch.o_proj_out,
            &layer.post_ln_w,
            eps,
            &mut self.scratch.norm_out,
            tokens,
            h,
        );
        if let Some(t0) = _t0 {
            B::sync(ctx);
            NORM_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 10. Fused gate+up projection
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        layer.gate_up_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.gate_up_out,
            tokens,
        );
        if let Some(t0) = _t0 {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 11. SwiGLU: silu(gate) * up
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::fused_silu_mul_split(
            ctx,
            &self.scratch.gate_up_out,
            &mut self.scratch.silu_out,
            tokens,
            im,
        );
        if let Some(t0) = _t0 {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 12. Down projection
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        layer.down_proj.forward(
            ctx,
            &self.scratch.silu_out,
            &mut self.scratch.mlp_out,
            tokens,
        );
        if let Some(t0) = _t0 {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 13. Final residual add
        let _t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        B::add_inplace(ctx, residual, &self.scratch.mlp_out, tokens * h);
        if let Some(t0) = _t0 {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Multi-position decode-verify: run one forward pass over `tokens`
    /// starting at the cache's current end position, write their K/V
    /// into the KV cache, and return logits for ALL `tokens.len()`
    /// positions as a flat `Vec<f32>` of length `seq_len * vocab_size`.
    ///
    /// Used by speculative decoding: target receives
    /// `[last_token, draft_0, ..., draft_{N-1}]` (N+1 inputs) and produces
    /// N+1 logit rows in a single forward instead of N+1 sequential
    /// decode() calls. Positions are implicit — the model looks up
    /// `pos_offset = cache.len` the same way prefill_internal does, so
    /// chunked prefill semantics carry over for free.
    pub fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "forward_verify called with empty tokens");
        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;

        let pos_offset = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(|c| c.len)
            .unwrap_or(0);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed = self
            .embed
            .as_ref()
            .expect("forward_verify called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, tokens, &mut residual, h);

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos_offset, seq_len);
        }

        // RMSNorm on ALL seq_len positions (prefill_internal only norms
        // the last one; verify needs the full grid).
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.norm_out,
            seq_len,
            h,
        );

        // LM head applied to all positions → `seq_len * vocab` logits.
        // Reuses the existing `batch_logits` scratch (sized max_tokens *
        // vocab) so no extra allocation.
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("forward_verify called on backbone-only model (no lm_head)");
        lm_head.forward(
            &mut ctx,
            &self.scratch.norm_out,
            &mut self.scratch.batch_logits,
            seq_len,
        );

        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.batch_logits, seq_len * vocab)
    }

    /// Prefill: process `tokens` prompt tokens in a single batch, return
    /// `[vocab_size]` logits for the last position.
    ///
    /// Supports incremental prefill: if the KV cache for `cache_id` already
    /// contains earlier tokens, the new chunk's positions are computed as
    /// `[kv_len, kv_len + tokens.len())` so RoPE and causal masking stay
    /// aligned. Used by the engine's chunked-prefill path.
    pub fn prefill_internal(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "prefill called with empty token list");
        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        // Starting position for this chunk — 0 for a fresh prefill, kv_len
        // for the second+ chunk of a split prefill.
        let pos_offset = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
            .map(|c| c.len)
            .unwrap_or(0);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let mut ctx = B::new_context();

        // Move `residual` out of `scratch` to work around the borrow checker:
        // `forward_layer` re-borrows `&mut self` to reach `self.layers` /
        // `self.kv_caches`, which would conflict with an outstanding
        // `&mut self.scratch.residual`. Use Option::take to move it out
        // (no placeholder alloc → no transient cuMemFreeAsync that could
        // corrupt stream pool state after graph ops on Blackwell).
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("prefill_internal called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, tokens, &mut residual, h);

        let prefill_profile = std::env::var("FERRUM_PREFILL_OP_PROFILE").is_ok();
        let prefill_t0 = if prefill_profile {
            B::sync(&mut ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos_offset, seq_len);
        }

        if let Some(t0) = prefill_t0 {
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
            eprintln!(
                "[prefill-profile] tokens={} layers total={} ms",
                seq_len,
                total_us / 1000
            );
            let bucket = |label: &str, n: u64, us: u64| {
                if n > 0 {
                    eprintln!(
                        "[prefill-profile] {label}: {} calls {} ms (avg {} us)",
                        n,
                        us / 1000,
                        us / n
                    );
                }
            };
            bucket("flash_attn", attn_n, attn_us);
            bucket("qk_norm_rope", qkr_n, qkr_us);
            bucket("matmuls", mm_n, mm_us);
            bucket("norms", norm_n, norm_us);
            bucket("other", other_n, other_us);
        }

        // Take the last token's hidden state: residual[(seq_len-1)*h .. seq_len*h]
        B::copy_slice(
            &mut ctx,
            &residual,
            (seq_len - 1) * h,
            &mut self.scratch.last_hidden,
            0,
            h,
        );

        // Final RMSNorm on the last hidden.
        B::rms_norm(
            &mut ctx,
            &self.scratch.last_hidden,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );

        // LM head (m=1 — triggers GEMV on MetalBackend).
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("prefill_internal called on backbone-only model (no lm_head)");
        lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );

        // Sync ctx before to_vec: on Metal, `to_vec` just reads the shared
        // buffer's CPU pointer without flushing the command buffer, so the
        // GPU must complete all pending work first or we read stale/random
        // data. CUDA's to_vec does an internal stream.synchronize, making
        // the call redundant there (~50µs/step cost), but correctness on
        // Metal requires the explicit flush here.
        B::sync(&mut ctx);

        // Restore residual into scratch for reuse on the next call.
        self.scratch.residual = Some(residual);

        B::to_vec(&self.scratch.logits, vocab)
    }

    /// Decode: process 1 token at position `pos`, return `[vocab_size]` logits.
    pub fn decode_internal(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;

        // Context creation is cheap (CUDA reuses the process-global stream).
        // The captured graph lives in a process-global slot, not on ctx.
        let mut ctx = B::new_context();

        // Graph capture is opt-in via FERRUM_CUDA_GRAPH=1. Replay is currently
        // single-request-only on Blackwell + CUDA 12.8 (see
        // docs/phase-e-cuda-status.md). In pure eager mode, we skip the
        // per-step device-state memcpy_htod trio entirely.
        const GRAPH_WARMUP: usize = 3;
        let graph_enabled = std::env::var("FERRUM_CUDA_GRAPH").is_ok();

        if graph_enabled {
            // Refresh device-side dynamic state (token/pos/kv_len) before
            // replay — captured graph reads these from device buffers.
            B::set_decode_state(&mut ctx, token, pos);

            // Fast path: graph replay (if available). Single-item path
            // uses key=SINGLE_ITEM_GRAPH_KEY (0) — separate from the
            // batched path's m_padded keys.
            match B::replay_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY) {
                Ok(true) => {
                    B::sync(&mut ctx);
                    return B::to_vec(&self.scratch.logits, vocab);
                }
                Ok(false) => { /* no graph yet, fall through to eager */ }
                Err(_) => { /* backend error or unsupported, eager */ }
            }
        }

        let should_capture =
            graph_enabled && !self.graph_capture_failed && self.graph_warmup >= GRAPH_WARMUP;

        if should_capture {
            B::set_dev_state_mode(&mut ctx, true);
            if B::begin_graph_capture(&mut ctx).is_err() {
                self.graph_capture_failed = true;
                B::set_dev_state_mode(&mut ctx, false);
            }
        }

        // Eager forward (records into graph if capture is active).
        // mem::replace needs a placeholder. B::alloc(0) was our choice but
        // cuMemAllocFromPoolAsync(stream, 0) can return CUDA_ERROR_INVALID_VALUE
        // on Blackwell after graph replay corrupts the pool state. Size-1 is
        // always valid and costs 2 bytes of transient VRAM per decode step.
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        let embed = self
            .embed
            .as_ref()
            .expect("decode_internal called on backbone-only model (no embed)");
        B::embedding_lookup(&mut ctx, embed, &[token], &mut residual, h);

        // Per-layer wall-time profile (env-gated, off by default — adds
        // a B::sync between layers which serializes the pipeline). Helps
        // localise non-matmul bottlenecks during perf work.
        let layer_profile = std::env::var("FERRUM_DECODE_LAYER_PROFILE").is_ok();
        let mut layer_times = if layer_profile {
            Some(Vec::with_capacity(self.cfg.num_layers))
        } else {
            None
        };

        for li in 0..self.cfg.num_layers {
            if layer_profile {
                let t0 = std::time::Instant::now();
                self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1);
                B::sync(&mut ctx);
                let elapsed_us = t0.elapsed().as_micros() as u64;
                if let Some(v) = layer_times.as_mut() {
                    v.push(elapsed_us);
                }
            } else {
                self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1);
            }
        }
        if let Some(times) = layer_times.take() {
            let sum: u64 = times.iter().sum();
            let avg = sum / times.len() as u64;
            let mn = *times.iter().min().unwrap_or(&0);
            let mx = *times.iter().max().unwrap_or(&0);
            eprintln!(
                "[layer-profile] {} layers total={} ms avg={} us min={} us max={} us",
                times.len(),
                sum / 1000,
                avg,
                mn,
                mx
            );
            for (i, t) in times.iter().enumerate() {
                eprint!("L{i}={}ms ", t / 1000);
                if (i + 1) % 6 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
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
            eprintln!(
                "[op-profile] flash_attn: {} calls {} ms (avg {} us)",
                attn_n,
                attn_us / 1000,
                if attn_n > 0 { attn_us / attn_n } else { 0 }
            );
            eprintln!(
                "[op-profile] qk_norm_rope: {} calls {} ms (avg {} us)",
                qkr_n,
                qkr_us / 1000,
                if qkr_n > 0 { qkr_us / qkr_n } else { 0 }
            );
            eprintln!(
                "[op-profile] matmuls (Linear::forward): {} calls {} ms (avg {} us)",
                mm_n,
                mm_us / 1000,
                if mm_n > 0 { mm_us / mm_n } else { 0 }
            );
            eprintln!(
                "[op-profile] norms (rms+fused_add_rms): {} calls {} ms (avg {} us)",
                norm_n,
                norm_us / 1000,
                if norm_n > 0 { norm_us / norm_n } else { 0 }
            );
            eprintln!(
                "[op-profile] other (split_qkv, kv_append, transpose, silu, add): {} calls {} ms (avg {} us)",
                other_n, other_us / 1000, if other_n > 0 { other_us / other_n } else { 0 }
            );
        }

        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );

        let lm_head = self
            .lm_head
            .as_ref()
            .expect("decode_internal called on backbone-only model (no lm_head)");
        lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );

        if should_capture && !self.graph_capture_failed {
            if B::end_graph_capture(&mut ctx, SINGLE_ITEM_GRAPH_KEY).is_err() {
                self.graph_capture_failed = true;
            } else {
                // Stream capture mode RECORDS ops into the graph without
                // executing them. scratch.logits still holds the previous
                // step's value. Replay the just-captured graph once to
                // actually execute and produce this step's logits. Without
                // this, the capture step's to_vec returns stale logits,
                // yielding a 1-token offset in the generated sequence.
                if B::replay_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY).is_err() {
                    self.graph_capture_failed = true;
                }
            }
            B::set_dev_state_mode(&mut ctx, false);
        } else {
            self.graph_warmup += 1;
        }

        // Sync ctx before to_vec: on Metal, `to_vec` just reads the shared
        // buffer's CPU pointer without flushing the command buffer, so the
        // GPU must complete all pending work first or we read stale/random
        // data. CUDA's to_vec does an internal stream.synchronize, making
        // the call redundant there (~50µs/step cost), but correctness on
        // Metal requires the explicit flush here.
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);

        B::to_vec(&self.scratch.logits, vocab)
    }

    /// Prefill with pre-computed embeddings instead of token IDs.
    ///
    /// Used by models that embed inputs outside the LLM (e.g. Qwen3-TTS
    /// mixes text-embedding + codec-embedding before feeding the LM).
    /// Skips `final_norm` + `lm_head`; returns the last position's pre-norm
    /// hidden state. Caller applies its own output head.
    ///
    /// `embeds` is row-major `[seq_len * hidden_size]`, f32.
    pub fn prefill_from_embeds(
        &mut self,
        cache_id: &str,
        embeds: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(
            embeds.len(),
            seq_len * h,
            "embeds length {} != seq_len * hidden_size {}",
            embeds.len(),
            seq_len * h
        );
        assert!(seq_len > 0, "prefill_from_embeds called with zero length");

        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        // Upload embeds → residual[0 .. seq_len*h].
        let embed_buf = B::from_slice(embeds);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, seq_len * h);

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, 0, seq_len);
        }

        B::copy_slice(
            &mut ctx,
            &residual,
            (seq_len - 1) * h,
            &mut self.scratch.last_hidden,
            0,
            h,
        );
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.last_hidden, h)
    }

    /// Decode with a single pre-computed embedding (shape `[hidden]`).
    /// Returns the pre-norm hidden state for the position `pos`. Caller
    /// applies final norm + its own output head.
    pub fn decode_from_embed(&mut self, cache_id: &str, embed: &[f32], pos: u32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(
            embed.len(),
            h,
            "embed length {} != hidden_size {}",
            embed.len(),
            h
        );

        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed_buf = B::from_slice(embed);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, h);

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1);
        }

        B::copy_slice(&mut ctx, &residual, 0, &mut self.scratch.last_hidden, 0, h);
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.last_hidden, h)
    }

    /// Variant of `prefill_from_embeds` that applies `final_norm` to every
    /// position and returns the whole `[seq_len * hidden_size]` vector.
    /// Accepts `pos_offset` so callers can continue an existing sequence
    /// (e.g. Qwen3-TTS voice-clone: one prefill for the role prefix, a
    /// follow-up prefill for the reference-audio ICL block, then
    /// autoregressive decoding — all against the same KV cache).
    ///
    /// Used by TTS where `forward_step` in the candle-based wrapper is
    /// expected to return **post-norm all-positions** hidden state so
    /// `codec_head` can be applied on candle side.
    pub fn prefill_all_post_norm(
        &mut self,
        cache_id: &str,
        embeds: &[f32],
        seq_len: usize,
        pos_offset: usize,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(
            embeds.len(),
            seq_len * h,
            "embeds length {} != seq_len * hidden_size {}",
            embeds.len(),
            seq_len * h
        );
        assert!(seq_len > 0);

        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed_buf = B::from_slice(embeds);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, seq_len * h);

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos_offset, seq_len);
        }

        // Apply final_norm over all seq_len positions → scratch.norm_out.
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.norm_out,
            seq_len,
            h,
        );
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.norm_out, seq_len * h)
    }

    /// Decode-side companion to `prefill_all_post_norm`. Runs a single-token
    /// decode step at `pos`, applies `final_norm`, and returns the post-norm
    /// hidden state `[hidden_size]`.
    pub fn decode_post_norm_from_embed(
        &mut self,
        cache_id: &str,
        embed: &[f32],
        pos: u32,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        assert_eq!(embed.len(), h);

        self.ensure_scratch(1);
        self.ensure_kv(cache_id);

        let mut ctx = B::new_context();
        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");

        let embed_buf = B::from_slice(embed);
        B::copy_slice(&mut ctx, &embed_buf, 0, &mut residual, 0, h);

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1);
        }

        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.last_normed,
            1,
            h,
        );
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);
        B::to_vec(&self.scratch.last_normed, h)
    }

    /// Batched decode: process M concurrent requests at potentially different
    /// positions in one forward pass. GEMM-heavy ops (qkv_proj, o_proj,
    /// gate_up, down) run with m=M for natural batching; rope + KV append +
    /// attention loop per-item (each has its own KV cache at a different
    /// kv_len, and potentially different pos).
    ///
    /// Returns M logit vectors in the same order as `batch`.
    pub fn decode_batch_internal(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        let m = batch.len();
        if m == 0 {
            return Vec::new();
        }
        if m == 1 {
            let (cid, tok, pos) = &batch[0];
            return vec![self.decode_internal(cid, *tok, *pos)];
        }

        // Ensure all caches exist and scratch is sized for M tokens.
        for (cid, _, _) in batch {
            self.ensure_kv(cid);
        }
        self.ensure_scratch(m);
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
            .map(|(cid, _, _)| {
                self.kv_caches.get(cid).expect("kv_caches missing")[0].len as u32
            })
            .collect();
        let kv_post: Vec<u32> = kv_pre.iter().map(|&x| x + 1).collect();
        B::write_u32(&mut ctx, &mut self.scratch.batch_positions, &positions);
        B::write_u32(&mut ctx, &mut self.scratch.batch_kv_lens_pre, &kv_pre);
        B::write_u32(&mut ctx, &mut self.scratch.batch_kv_lens_post, &kv_post);

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

        // ── Phase 4d: CUDA-graph replay path ─────────────────────────
        // gated on FERRUM_BATCHED_GRAPH=1; skipped on backends without
        // graph support (begin_graph_capture returns Err).
        let graph_enabled = std::env::var("FERRUM_BATCHED_GRAPH")
            .map(|v| v != "0")
            .unwrap_or(false);
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
            let trace = std::env::var("FERRUM_BATCHED_TRACE").is_ok();
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
            let batched_profile = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
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
                self.forward_layer_batched_decode(&mut ctx, li, batch, &mut residual, m);
                tracesync!(format!("after layer {}", li));
            }
            let _t0_norm = if batched_profile {
                B::sync(&mut ctx);
                Some(std::time::Instant::now())
            } else {
                None
            };
            B::rms_norm(
                &mut ctx,
                &residual,
                &self.final_norm_w,
                self.cfg.rms_norm_eps,
                &mut self.scratch.norm_out,
                m,
                h,
            );
            if let Some(t0) = _t0_norm {
                B::sync(&mut ctx);
                NORM_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
            lm_head.forward(
                &mut ctx,
                &self.scratch.norm_out,
                &mut self.scratch.batch_logits,
                m,
            );
            if let Some(t0) = _t0_lm {
                B::sync(&mut ctx);
                MATMUL_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
            let caches = self
                .kv_caches
                .get_mut(cid)
                .expect("kv_caches missing");
            for li in 0..num_layers {
                caches[li].len += 1;
            }
        }

        // Sync before to_vec (Metal: no internal sync on buffer read).
        B::sync(&mut ctx);
        self.scratch.residual = Some(residual);

        // Extract M logit vectors from the flat buffer.
        let all = B::to_vec(&self.scratch.batch_logits, m * vocab);
        (0..m)
            .map(|i| all[i * vocab..(i + 1) * vocab].to_vec())
            .collect()
    }

    /// One transformer layer over M items, GEMMs batched + per-item attention.
    fn forward_layer_batched_decode(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        batch: &[(String, u32, u32)],
        residual: &mut B::Buffer,
        m: usize,
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

        let layer = &self.layers[li];
        let qk_mode: i32 = if cfg.has_qk_norm { 1 } else { 2 };
        let dummy_w = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy_w);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy_w);

        let _bp = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();

        // 1. rms_norm [M, H]  → norm_out
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        B::rms_norm(
            ctx,
            residual,
            &layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m,
            h,
        );
        if let Some(t0) = _t {
            B::sync(ctx);
            NORM_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 2. qkv_proj (GEMM m=M): norm_out [M, H] → qkv_out [M, QKV]
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        layer
            .qkv_proj
            .forward(ctx, &self.scratch.norm_out, &mut self.scratch.qkv_out, m);
        if let Some(t0) = _t {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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

            // Take block_table buffer ptrs ahead of the dispatch loop —
            // we need both per-cache block_table (to write into) and
            // self.scratch.paged_batch_q (to write Q stacks into).
            let q_head_major_size_bytes = (q_dim * std::mem::size_of::<f32>()) as u64;
            let qkv_stride_bytes = (qkv_stride * std::mem::size_of::<f32>()) as u64;
            let _t_qkr = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
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
                QKR_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                QKR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            // Step 2: bump cache.len and build the stacked block_tables +
            // context_lens host-side, then upload to device scratch.
            let mut stacked_bt: Vec<u32> = vec![0u32; m * max_blocks_per_seq];
            let mut stacked_cl: Vec<u32> = vec![0u32; m];
            for (i, (cache_id, _, _)) in batch.iter().enumerate() {
                let caches = self
                    .kv_caches
                    .get_mut(cache_id)
                    .expect("paged batched: cache not present");
                let cache = &mut caches[li];
                cache.len += 1;
                let len = cache.len as u32;
                stacked_cl[i] = len;
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
            B::write_u32(ctx, bt_buf, &stacked_bt);
            let cl_buf = self
                .scratch
                .paged_batch_context_lens
                .as_mut()
                .expect("paged_batch_context_lens missing");
            B::write_u32(ctx, cl_buf, &stacked_cl);

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
            let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
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
                ATTN_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
            return self.forward_layer_batched_decode_post_attn(ctx, li, residual, m);
        }

        // 3. split_qkv [M, QKV] → q_buf [M, Q], k_buf [M, KV], v_buf [M, KV]
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
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
            OTHER_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
        let _t_qkr_contig = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        let q_batched = B::qk_norm_rope_batched_per_item(
            ctx,
            &self.scratch.q_buf,
            q_norm_w,
            &self.rope.cos,
            &self.rope.sin,
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
            &self.rope.cos,
            &self.rope.sin,
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
            &self.rope.cos,
            &self.rope.sin,
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
            QKR_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
        let _t_kvapp = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
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
            OTHER_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // kv_lens_host no longer used; flash_attn reads
        // scratch.batch_kv_lens_post (also pre-populated).
        let _ = kv_lens_host;

        // 6. Per-item loop: only runs when the batched paths are NOT
        //    in effect, OR when batched_kv_append failed (Err fallback).
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
                    &self.rope.cos,
                    &self.rope.sin,
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
                    &self.rope.cos,
                    &self.rope.sin,
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
                    &self.rope.cos,
                    &self.rope.sin,
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
            // cache.len bump moved to decode_batch_internal post-forward
            // for graph-replay correctness. flash_attn below uses
            // cache.len + 1 directly.
            let kv_len = cache.len + 1;
            let kv_stride = cache.capacity;
            kv_lens_host.push(kv_len as u32);

            // Per-item flash_attn ONLY when batched qkr fallback is in
            // use. Otherwise the batched flash_attn after the loop
            // covers it in one launch. Q comes from the already-normed
            // q_head_major_single populated by per-item qk_norm_rope.
            if !use_batched_qkr {
                let attn_cfg = ferrum_kernels::backend::AttnConfig {
                    num_heads: nh,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    causal: true,
                    scale: 1.0 / (hd as f32).sqrt(),
                    kv_seq_stride: kv_stride,
                    sliding_window: cfg.sliding_window,
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
            let _t_attn = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
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
                li,
            );
            if let Some(t0) = _t_attn {
                B::sync(ctx);
                ATTN_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
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
                        sliding_window: cfg.sliding_window,
                    };
                    B::flash_attention(
                        ctx,
                        &self.scratch.q_head_major_single,
                        &cache.k,
                        &cache.v,
                        &mut self.scratch.attn_head_major_single,
                        1,
                        1,
                        cache.len,
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

        self.forward_layer_batched_decode_post_attn(ctx, li, residual, m);
    }

    fn forward_layer_batched_decode_post_attn(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        residual: &mut B::Buffer,
        m: usize,
    ) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let layer = &self.layers[li];
        let _bp = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();

        // 7. o_proj (GEMM m=M): attn_flat [M, Q] → o_proj_out [M, H]
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        layer.o_proj.forward(
            ctx,
            &self.scratch.attn_flat,
            &mut self.scratch.o_proj_out,
            m,
        );
        if let Some(t0) = _t {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 8. Fused residual add + post-attention RMSNorm.
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        B::fused_add_rms_norm(
            ctx,
            residual,
            &self.scratch.o_proj_out,
            &layer.post_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m,
            h,
        );
        if let Some(t0) = _t {
            B::sync(ctx);
            NORM_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            NORM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 9. gate_up_proj (GEMM m=M)
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        layer.gate_up_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.gate_up_out,
            m,
        );
        if let Some(t0) = _t {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 10. SwiGLU
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        B::fused_silu_mul_split(
            ctx,
            &self.scratch.gate_up_out,
            &mut self.scratch.silu_out,
            m,
            im,
        );
        if let Some(t0) = _t {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 11. down_proj (GEMM m=M)
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        layer
            .down_proj
            .forward(ctx, &self.scratch.silu_out, &mut self.scratch.mlp_out, m);
        if let Some(t0) = _t {
            B::sync(ctx);
            MATMUL_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            MATMUL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 12. Residual add
        let _t = if _bp { B::sync(ctx); Some(std::time::Instant::now()) } else { None };
        B::add_inplace(ctx, residual, &self.scratch.mlp_out, m * h);
        if let Some(t0) = _t {
            B::sync(ctx);
            OTHER_TIME_US.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            OTHER_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
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

        // Per-item bookkeeping (host).
        let q_lens: Vec<usize> = items.iter().map(|it| it.1.len()).collect();
        let mut cu_seqlens_q: Vec<u32> = Vec::with_capacity(num_seqs + 1);
        cu_seqlens_q.push(0);
        for &l in &q_lens {
            let prev = *cu_seqlens_q.last().unwrap();
            cu_seqlens_q.push(prev + l as u32);
        }
        let m_total = *cu_seqlens_q.last().unwrap() as usize;
        let pos_offsets: Vec<u32> = items.iter().map(|it| it.2 as u32).collect();
        // Causal max over (pos_offset + q_len) — shared-mem size for the
        // varlen attn kernel needs to fit the longest reachable kv_pos.
        let max_kv_len: usize = items
            .iter()
            .map(|it| it.2 + it.1.len())
            .max()
            .unwrap_or(0);

        // Ensure all items' KV caches exist.
        for (cid, _, _, _) in items {
            self.ensure_kv(cid);
        }

        // Concatenated input tokens for one embedding_lookup.
        let all_tokens: Vec<u32> = items.iter().flat_map(|it| it.1.iter().copied()).collect();
        debug_assert_eq!(all_tokens.len(), m_total);

        // Paged path requirements.
        let pools_present = self.paged_pools.is_some();
        if !pools_present {
            // Caller should have routed to fallback; defensive.
            panic!(
                "unified_forward_internal called without paged_pools — caller must check"
            );
        }
        let max_blocks_per_seq = self.scratch.paged_max_blocks_per_seq;
        let block_size = 16usize; // matches PAGED_BLOCK_SIZE in ensure_kv

        // Make sure scratch fits this batch.
        // max_seqs uses paged_batch's existing limit so unified shares
        // the cap; if items.len() exceeds it, panic loudly (engine
        // shouldn't ever produce that).
        let max_seqs = num_seqs;
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
            B::write_u32(&mut ctx, csq, &cu_seqlens_q);
        }
        {
            let po = self
                .scratch
                .unified_pos_offsets
                .as_mut()
                .expect("unified_pos_offsets missing");
            B::write_u32(&mut ctx, po, &pos_offsets);
        }
        // Stack per-seq block tables host-side, then upload.
        {
            let mut stacked: Vec<u32> =
                vec![0u32; num_seqs * max_blocks_per_seq];
            for (i, (cid, _, _, _)) in items.iter().enumerate() {
                let caches = self
                    .kv_caches
                    .get(cid)
                    .expect("kv cache missing for unified item");
                let cache0 = &caches[0];
                let blocks = &cache0.paged_block_indices;
                let n_to_copy = blocks.len().min(max_blocks_per_seq);
                stacked[i * max_blocks_per_seq..i * max_blocks_per_seq + n_to_copy]
                    .copy_from_slice(&blocks[..n_to_copy]);
            }
            let bt = self
                .scratch
                .unified_block_tables
                .as_mut()
                .expect("unified_block_tables missing");
            B::write_u32(&mut ctx, bt, &stacked);
        }

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

        // Final norm on the WHOLE M_total then extract last-token rows
        // for is_final_chunk items via per-item copy_slice into a packed
        // buffer; lm_head runs on the packed [num_sampled, h].
        let final_norm_w = &self.final_norm_w;
        // Reuse unified_norm_out as final-norm output (it's free post-layer).
        let mut norm_out = self
            .scratch
            .unified_norm_out
            .take()
            .expect("unified_norm_out missing");
        B::rms_norm(&mut ctx, &residual, final_norm_w, eps, &mut norm_out, m_total, h);

        // Identify per-item last-token global indices for is_final_chunk items.
        let final_indices: Vec<(usize, usize)> = items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.3)
            .map(|(orig_idx, _)| {
                let item = &items[orig_idx];
                let last_token_local = item.1.len() - 1;
                let global = (cu_seqlens_q[orig_idx] as usize) + last_token_local;
                (orig_idx, global)
            })
            .collect();
        let num_sampled = final_indices.len();
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("unified_forward_internal called on backbone-only model");

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

        // Restore unified_residual / unified_norm_out so a subsequent
        // call's `take()` finds them. norm_out was written by final
        // rms_norm; residual is post-final-layer state we no longer need.
        self.scratch.unified_residual = Some(residual);
        // norm_out is consumed by the per-item copy_slice below before
        // being put back.

        let mut out: Vec<Option<Vec<f32>>> = (0..items.len()).map(|_| None).collect();

        if num_sampled > 0 {
            // Pack final-chunk hidden states into a contiguous buffer
            // [num_sampled, h] via per-item copy_slice. Inexpensive — at
            // c=16 worst case 16 small copies.
            let packed_normed = self
                .scratch
                .unified_packed_normed
                .as_mut()
                .expect("unified_packed_normed missing");
            for (j, &(_, global)) in final_indices.iter().enumerate() {
                B::copy_slice(
                    &mut ctx,
                    &norm_out,
                    global * h,
                    packed_normed,
                    j * h,
                    h,
                );
            }
            // lm_head [num_sampled, h] → packed_logits [num_sampled, vocab]
            let packed_logits = self
                .scratch
                .unified_packed_logits
                .as_mut()
                .expect("unified_packed_logits missing");
            lm_head.forward(&mut ctx, packed_normed, packed_logits, num_sampled);
            // Sync + readback.
            B::sync(&mut ctx);
            let logits_flat = B::to_vec(packed_logits, num_sampled * vocab);
            for (j, &(orig_idx, _)) in final_indices.iter().enumerate() {
                let row = logits_flat[j * vocab..(j + 1) * vocab].to_vec();
                out[orig_idx] = Some(row);
            }
        } else {
            B::sync(&mut ctx);
        }

        // Restore norm_out for next call.
        self.scratch.unified_norm_out = Some(norm_out);

        out
    }

    /// One transformer layer for the unified mixed-batch forward.
    /// Mirrors `forward_layer_batched_decode` paged path but operates
    /// on `[M_total, *]` tensors and uses `paged_varlen_attention`.
    #[allow(clippy::too_many_arguments)]
    fn unified_forward_layer(
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

        // 1. rms_norm [M_total, h] → unified_norm_out
        {
            let norm_out = self
                .scratch
                .unified_norm_out
                .as_mut()
                .expect("unified_norm_out missing");
            B::rms_norm(ctx, residual, &layer.input_ln_w, eps, norm_out, m_total, h);
        }

        // 2. qkv_proj GEMM (m=M_total): unified_norm_out → unified_qkv_out
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
            layer.qkv_proj.forward(ctx, norm_out, qkv_out, m_total);
        }

        // 3. Per-item split_qkv_norm_rope_into_paged_cache. Writes K/V
        // into seq's paged pool slot AND stores RoPE'd Q at packed_q
        // offset. Handles per-item q_len + pos_offset (the same kernel
        // already supports tokens > 1 per call — used by prefill_internal).
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
        let q_head_major_size_bytes = (q_dim * std::mem::size_of::<f32>()) as u64;
        let qkv_stride_bytes = (qkv_stride * std::mem::size_of::<f32>()) as u64;

        for (i, (cid, _q_tokens, pos_offset, _)) in items.iter().enumerate() {
            let q_token_offset = cu_seqlens_q[i] as u64;
            let cache_len_before;
            let bt_safe: &B::Buffer;
            {
                let caches = self
                    .kv_caches
                    .get(cid)
                    .expect("paged unified: cache not present");
                let cache = &caches[li];
                cache_len_before = cache.len;
                let bt = cache
                    .block_table
                    .as_ref()
                    .expect("paged unified: block_table missing");
                let block_table_ref = bt as *const B::Buffer;
                // SAFETY: bt is read-only; we don't mutate kv_caches between
                // this raw deref and the call.
                bt_safe = unsafe { &*block_table_ref };
            }
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
            B::split_qkv_norm_rope_into_paged_cache(
                ctx,
                qkv_out,
                q_token_offset * qkv_stride_bytes,
                q_norm_w,
                k_norm_w,
                &self.rope.cos,
                &self.rope.sin,
                packed_q,
                q_token_offset * q_head_major_size_bytes,
                pool_k,
                pool_v,
                bt_safe,
                q_lens[i],
                nh,
                nkv,
                hd,
                *pos_offset,
                eps,
                qk_mode,
                cache_len_before,
                block_size,
                max_blocks_per_seq,
            )
            .expect("paged unified: split_qkv_norm_rope_into_paged_cache");
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
            .expect("paged_varlen_attention");
        }

        // 5. o_proj (M_total): attn_out → o_proj_out
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
            layer.o_proj.forward(ctx, attn_out, o_proj_out, m_total);
        }

        // 6. fused_add_rms_norm: residual = residual + o_proj_out;
        //    norm_out = rms_norm(new_residual, post_ln_w)
        {
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
        }

        // 7. gate_up_proj
        {
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
            layer
                .gate_up_proj
                .forward(ctx, norm_out, gate_up_out, m_total);
        }

        // 8. SwiGLU
        {
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
        }

        // 9. down_proj
        {
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
            layer.down_proj.forward(ctx, silu_out, mlp_out, m_total);
        }

        // 10. residual_add
        {
            let mlp_out = self
                .scratch
                .unified_mlp_out
                .as_ref()
                .expect("unified_mlp_out missing");
            B::add_inplace(ctx, residual, mlp_out, m_total * h);
        }
    }
}

impl<B: Backend> DecoderOnlyLLM for LlamaFamilyModel<B> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        // Eager scratch + KV cache grow + a 1-token forward warmup —
        // see the Qwen3MoeModel::prepare comment for the rationale.
        // Without the warmup forward, the first real prefill pays
        // Metal pipeline first-bind costs inside the timer window.
        self.ensure_scratch(max_tokens);
        self.ensure_kv(cache_id);

        const WARMUP_CACHE: &str = "__ferrum_warmup__";
        let _ = self.prefill_internal(WARMUP_CACHE, &[0u32]);
        // Release via the same path as `release` so paged blocks
        // return to the shared allocator. Otherwise warmup leaks
        // 256 blocks (the full per-seq quota) into the pool.
        if let Some(mut caches) = self.kv_caches.remove(WARMUP_CACHE) {
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

    fn kv_capacity(&self) -> usize {
        // Mirror the bound `ensure_kv` will use when allocating the cache.
        let model_max = self.cfg.max_seq_len;
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

    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        self.decode_batch_internal(batch)
    }

    /// Unified mixed-batch forward (chunked-prefill API).
    ///
    /// When paged KV is on, dispatches to `unified_forward_internal`
    /// which runs ONE `[M_total, hidden]` forward through the layer
    /// loop + `paged_varlen_attention` kernel — the chunked-prefill
    /// perf path.
    ///
    /// On the contig-KV path (CUDA default, no paged pool), returns
    /// `Err(unsupported)` so the executor's fallback handles each item
    /// via existing `prefill()` / `decode_batch()`.
    fn unified_forward(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> std::result::Result<Vec<Option<Vec<f32>>>, ferrum_types::FerrumError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if self.paged_pools.is_none() {
            return Err(ferrum_types::FerrumError::unsupported(
                "LlamaFamilyModel::unified_forward: paged KV required; \
                 enable via FERRUM_METAL_PAGED_KV=1 (Metal) or paged-on-CUDA \
                 opt-in. Engine will fall back to per-item dispatch.",
            ));
        }
        Ok(self.unified_forward_internal(items))
    }

    fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        // Delegate to the inherent implementation on LlamaFamilyModel.
        LlamaFamilyModel::<B>::forward_verify(self, cache_id, tokens)
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        if let Some(caches) = self.kv_caches.get_mut(cache_id) {
            for c in caches.iter_mut() {
                if new_len < c.len {
                    c.len = new_len;
                }
            }
        }
        // Single-item graph captures kv-cache pointers directly; truncate
        // doesn't change pointer but the captured kv_len-bumping pattern
        // is fragile against state walks. Force single-item re-capture;
        // batched path is layout-agnostic (per-call cache_lens write).
        let mut ctx = B::new_context();
        B::reset_graph(&mut ctx, SINGLE_ITEM_GRAPH_KEY);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
    }

    fn release(&mut self, cache_id: &str) {
        // KV buffers go to kv_free_pool (not dropped) → captured graph's
        // pointer args stay valid. Batched graph reads cache pointers from
        // BATCHED_SCRATCH populated per-call, freed cache_ids drop out
        // naturally. Drain in-flight kernels, then DON'T invalidate the
        // graph slot — keeping it across releases avoids the thrashing
        // pattern (every request completion = forced re-capture) that
        // killed graph mode performance at c=16.
        //
        // Phase 4d earlier hit a multi-replay SIGSEGV when keeping the
        // slot alive; Phase 8 (perm-aware Marlin) replaced cuBLAS dense
        // GEMMs with Marlin in the captured path, which appears to have
        // resolved the underlying instability. Verified 2026-05-04: c=4
        // graph mode 16/16 ok with slot kept alive across releases (was
        // 0/16 before Phase 8).
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        // Single-item path captures kv-cache pointers directly into the
        // graph; force re-capture if the next request's pool-recycled
        // buffer differs. Batched path uses BATCHED_SCRATCH indirection
        // and survives buffer reuse without re-capture.
        self.graph_warmup = 0;
        self.graph_capture_failed = false;

        // Return the cache's buffers to the free pool instead of dropping.
        // Pointers stay stable for the next request's captured graph.
        // Paged mode: also free the cache's blocks back to the shared
        // allocator so other sequences can reuse them.
        if let Some(mut caches) = self.kv_caches.remove(cache_id) {
            if let Some(alloc_arc) = self.paged_block_alloc.as_ref() {
                let mut alloc = alloc_arc.lock().unwrap_or_else(|p| p.into_inner());
                // All caches share the same block_indices set (one per
                // cache_id), so freeing once via the first layer's cache
                // is enough.
                if let Some(c0) = caches.first() {
                    if !c0.paged_block_indices.is_empty() {
                        alloc.free(&c0.paged_block_indices);
                    }
                }
                // Clear the host-side mirror on every layer so a
                // free-pool reuse re-allocates fresh blocks.
                for c in caches.iter_mut() {
                    c.paged_block_indices.clear();
                }
            }
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        // Hard reset: drop all caches AND the pool, invalidate ALL graphs.
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_all_graphs(&mut ctx);
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
        self.batched_graph_keys_seen.clear();
        self.batched_graph_warmup = 0;
        self.batched_graph_failed = false;
        self.kv_caches.clear();
        self.kv_free_pool.clear();
    }
}

fn build_rope_cache<B: Backend>(cfg: &LlamaFamilyConfig) -> RopeCache<B> {
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
