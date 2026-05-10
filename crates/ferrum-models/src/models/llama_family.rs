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

use ferrum_interfaces::kv_dtype::{KvDtypeKind, KvFp16, KvInt8};
use ferrum_kernels::backend::{
    Backend, BackendGraph, BackendInt8KvOps, BackendKvDtype, BackendMoeFused, BackendPagedKv,
    BackendQuantGguf, BackendQuantMarlin, KvCache, KvCacheQuant, LlmBackend, MoeLlmBackend,
    QuantLlmBackend, MAX_LAYERS_FOR_GRAPH,
};

/// Per-layer KV cache element (Dim 5).
///
/// Wraps both the FP16 (`KvCache<B, KvFp16>`) and INT8
/// (`KvCacheQuant<B, KvInt8>`) cache layouts so a single model struct
/// can carry either depending on the model's `K` type marker.
///
/// `K` is *not* on the enum: every `LlamaFamilyModel<B, K>` instance
/// only ever constructs **one** variant for the lifetime of the model,
/// chosen by `K::NAME` in `ensure_kv`. Carrying an actual `K` parameter
/// here would force `K = KvInt8` for the `Int8` variant and force
/// every `K` value to satisfy `BackendKvDtype<K>` for the `Fp16` arm —
/// painful for callers that just have a `K: KvDtypeKind` bound. Instead
/// the variants name concrete dtypes; the `B: BackendKvDtype<KvInt8>`
/// bound is satisfied by stub impls on CpuBackend/MetalBackend so the
/// enum type is well-formed everywhere.
pub enum LayerKvCache<B: MoeLlmBackend + BackendInt8KvOps> {
    Fp16(KvCache<B, KvFp16>),
    Int8(KvCacheQuant<B, KvInt8>),
}

impl<B: MoeLlmBackend + BackendInt8KvOps> LayerKvCache<B> {
    pub fn len(&self) -> usize {
        match self {
            LayerKvCache::Fp16(kv) => kv.len,
            LayerKvCache::Int8(kv) => kv.len,
        }
    }

    pub fn set_len(&mut self, new_len: usize) {
        match self {
            LayerKvCache::Fp16(kv) => kv.len = new_len,
            LayerKvCache::Int8(kv) => kv.len = new_len,
        }
    }

    pub fn capacity(&self) -> usize {
        match self {
            LayerKvCache::Fp16(kv) => kv.capacity,
            LayerKvCache::Int8(kv) => kv.capacity,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            LayerKvCache::Fp16(kv) => kv.block_size,
            LayerKvCache::Int8(kv) => kv.block_size,
        }
    }

    pub fn num_kv_heads(&self) -> usize {
        match self {
            LayerKvCache::Fp16(kv) => kv.num_kv_heads,
            LayerKvCache::Int8(kv) => kv.num_kv_heads,
        }
    }

    pub fn head_dim(&self) -> usize {
        match self {
            LayerKvCache::Fp16(kv) => kv.head_dim,
            LayerKvCache::Int8(kv) => kv.head_dim,
        }
    }

    pub fn block_table(&self) -> Option<&B::Buffer> {
        match self {
            LayerKvCache::Fp16(kv) => kv.block_table.as_ref(),
            LayerKvCache::Int8(kv) => kv.block_table.as_ref(),
        }
    }

    pub fn context_lens(&self) -> Option<&B::Buffer> {
        match self {
            LayerKvCache::Fp16(kv) => kv.context_lens.as_ref(),
            LayerKvCache::Int8(kv) => kv.context_lens.as_ref(),
        }
    }

    pub fn context_lens_mut(&mut self) -> Option<&mut B::Buffer> {
        match self {
            LayerKvCache::Fp16(kv) => kv.context_lens.as_mut(),
            LayerKvCache::Int8(kv) => kv.context_lens.as_mut(),
        }
    }

    pub fn paged_block_indices(&self) -> &[u32] {
        match self {
            LayerKvCache::Fp16(kv) => &kv.paged_block_indices,
            LayerKvCache::Int8(kv) => &kv.paged_block_indices,
        }
    }

    pub fn paged_block_indices_mut(&mut self) -> &mut Vec<u32> {
        match self {
            LayerKvCache::Fp16(kv) => &mut kv.paged_block_indices,
            LayerKvCache::Int8(kv) => &mut kv.paged_block_indices,
        }
    }

    pub fn block_table_mut(&mut self) -> Option<&mut B::Buffer> {
        match self {
            LayerKvCache::Fp16(kv) => kv.block_table.as_mut(),
            LayerKvCache::Int8(kv) => kv.block_table.as_mut(),
        }
    }

    /// Whether this cache is in paged mode (`block_size > 0`).
    pub fn is_paged(&self) -> bool {
        self.block_size() > 0
    }

    /// Borrow the FP16 cache (panics if this is the INT8 variant). Used
    /// by paths that haven't been generalized to the enum yet — caller
    /// must guarantee the model's `K = KvFp16`.
    pub fn as_fp16_mut(&mut self) -> &mut KvCache<B, KvFp16> {
        match self {
            LayerKvCache::Fp16(kv) => kv,
            LayerKvCache::Int8(_) => {
                panic!("LayerKvCache::as_fp16_mut called on Int8 variant")
            }
        }
    }

    pub fn as_fp16(&self) -> &KvCache<B, KvFp16> {
        match self {
            LayerKvCache::Fp16(kv) => kv,
            LayerKvCache::Int8(_) => {
                panic!("LayerKvCache::as_fp16 called on Int8 variant")
            }
        }
    }

    pub fn as_int8_mut(&mut self) -> &mut KvCacheQuant<B, KvInt8> {
        match self {
            LayerKvCache::Int8(kv) => kv,
            LayerKvCache::Fp16(_) => {
                panic!("LayerKvCache::as_int8_mut called on Fp16 variant")
            }
        }
    }

    pub fn as_int8(&self) -> &KvCacheQuant<B, KvInt8> {
        match self {
            LayerKvCache::Int8(kv) => kv,
            LayerKvCache::Fp16(_) => {
                panic!("LayerKvCache::as_int8 called on Fp16 variant")
            }
        }
    }
}

/// Graph cache key for the single-item decode path (`decode_internal`).
/// Distinct from any `m_padded`-based key used by the batched path.
pub(crate) const SINGLE_ITEM_GRAPH_KEY: u64 = 0;

/// Diag counters for the batched graph dispatcher (replay vs eager).
pub(crate) static BATCHED_GRAPH_REPLAY_COUNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static BATCHED_GRAPH_EAGER_COUNT: AtomicU64 = AtomicU64::new(0);

pub(crate) static ATTN_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static QKR_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static QKR_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MATMUL_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static NORM_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static NORM_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static OTHER_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static OTHER_CALLS: AtomicU64 = AtomicU64::new(0);
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
pub struct LlamaFamilyLayer<B: QuantLlmBackend + BackendMoeFused> {
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
pub struct RopeCache<B: QuantLlmBackend + BackendMoeFused> {
    pub cos: B::Buffer,
    pub sin: B::Buffer,
}

/// Reusable per-layer scratch buffers sized for `max_tokens` tokens of a
/// single forward pass (prefill or decode step).
///
/// Sized lazily on first use so tiny decode steps don't pay for prefill-sized
/// buffers. Grows monotonically when a larger prefill arrives.
pub struct LlamaFamilyScratch<B: QuantLlmBackend + BackendMoeFused> {
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
    /// Engine-side max concurrent sequences (= `FERRUM_PAGED_MAX_SEQS`).
    /// Pinned at the first `enable_paged_batch` so the unified-forward
    /// scratch sizes (`unified_cu_seqlens_q`, `unified_pos_offsets`,
    /// `unified_block_tables`, `unified_packed_*`) are big enough for
    /// any subsequent batch up to that bound.
    pub paged_max_seqs: usize,
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

impl<B: QuantLlmBackend + BackendMoeFused> LlamaFamilyScratch<B> {
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
            paged_max_seqs: 0,
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
    pub(crate) fn ensure_unified_scratch(
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
        self.paged_max_seqs = max_seqs;
    }
}

/// Qwen3 model — decoder-only LLM, one per (backend, weights) combination.
///
/// Holds all parameters, scratch space, RoPE cache, and per-sequence KV caches.
///
/// `B: BackendGraph + BackendQuantMarlin + BackendQuantGguf` because the decode hot path uses CUDA Graph capture/replay
/// when the backend supports it; non-graph backends (Metal/CPU) inherit no-op
/// defaults, so this bound is satisfied by every concrete `Backend`.
///
/// `K: KvDtypeKind = KvFp16` (Dim 5): selects the KV cache element type.
/// `K = KvFp16` constructs only `LayerKvCache::Fp16(...)` variants;
/// `K = KvInt8` constructs only `LayerKvCache::Int8(...)` variants. The
/// `B: BackendKvDtype<KvInt8>` bound is structurally required by the
/// enum and is satisfied by all backends via stub impls (Cpu/Metal) or
/// the real impl (CUDA).
pub struct LlamaFamilyModel<B: MoeLlmBackend + BackendInt8KvOps, K: KvDtypeKind = KvFp16> {
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
    pub kv_caches: HashMap<String, Vec<LayerKvCache<B>>>,
    /// Free pool of pre-allocated KV cache slots. Released caches return
    /// here instead of being dropped, so their device pointers stay valid
    /// across requests — critical for graph capture (pointers baked into
    /// the captured graph would otherwise dangle).
    kv_free_pool: Vec<Vec<LayerKvCache<B>>>,
    /// Phantom marker — selects which `LayerKvCache` variant `ensure_kv`
    /// constructs. `K::NAME` is checked at runtime but folds to a const
    /// after monomorphization.
    _kv_dtype: std::marker::PhantomData<K>,

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
    /// CUDA-graph state for the unified_forward path. Mirrors the
    /// `batched_graph_*` triple but keyed on `(m_total, num_seqs)`
    /// so different concurrency levels each get their own cached
    /// graph instead of thrashing a single slot.
    pub(crate) unified_graph_warmup: usize,
    pub(crate) unified_graph_failed: bool,
    pub(crate) unified_graph_keys_seen: std::collections::HashSet<u64>,
}

impl<B: MoeLlmBackend + BackendInt8KvOps, K: KvDtypeKind> LlamaFamilyModel<B, K> {
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
            _kv_dtype: std::marker::PhantomData,
            paged_pools: None,
            paged_block_alloc: None,
            graph_warmup: 0,
            graph_capture_failed: false,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            batched_pointers_for: None,
            unified_graph_warmup: 0,
            unified_graph_failed: false,
            unified_graph_keys_seen: std::collections::HashSet::new(),
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
            _kv_dtype: std::marker::PhantomData,
            paged_pools: None,
            paged_block_alloc: None,
            graph_warmup: 0,
            graph_capture_failed: false,
            batched_graph_warmup: 0,
            batched_graph_failed: false,
            batched_graph_keys_seen: std::collections::HashSet::new(),
            batched_pointers_for: None,
            unified_graph_warmup: 0,
            unified_graph_failed: false,
            unified_graph_keys_seen: std::collections::HashSet::new(),
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
            self.unified_graph_keys_seen.clear();
            self.unified_graph_warmup = 0;
            self.unified_graph_failed = false;
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
        // K::NAME selects which `LayerKvCache` variant to construct:
        //   - "fp16" → FP16 path (existing).
        //   - "int8" → INT8 paged path; requires paged=true and
        //     `B: BackendInt8KvOps` (the factory cascade only routes
        //     `(CUDA, Int8)` here so this branch only ever runs on CUDA).
        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.num_layers)
                .map(|_| {
                    if K::NAME == "int8" {
                        if !paged {
                            panic!(
                                "INT8 KV requires paged mode; set FERRUM_METAL_PAGED_KV=1 \
                                 (cross-backend env var) or use FP16 KV"
                            );
                        }
                        // INT8 paged layer: K/V/scales pools allocated
                        // by the backend; block_table mirrors the FP16
                        // path (alloc_u32 on Self::Buffer to match the
                        // u32-as-f16 storage trick).
                        LayerKvCache::Int8(B::alloc_paged_int8_layer(
                            max_blocks_per_seq,
                            PAGED_BLOCK_SIZE,
                            nkv,
                            hd,
                        ))
                    } else if paged {
                        // FP16 paged: cache holds metadata only. K/V
                        // are 1-element placeholders. Real data lives
                        // in `self.paged_pools[li].{k,v}`.
                        let block_table = B::alloc_u32(max_blocks_per_seq);
                        let mut context_lens = B::alloc_u32(1);
                        let mut bt_ctx = B::new_context();
                        B::write_u32(&mut bt_ctx, &mut context_lens, &[0u32]);
                        B::sync(&mut bt_ctx);
                        LayerKvCache::Fp16(KvCache {
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
                        })
                    } else {
                        LayerKvCache::Fp16(KvCache {
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
                        })
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
                if let Some(bt) = c.block_table_mut() {
                    B::write_u32(&mut ctx_tmp, bt, &padded);
                }
                *c.paged_block_indices_mut() = block_indices.clone();
            }
            B::sync(&mut ctx_tmp);
        }

        // Reset logical length; buffers stay. No need to zero the memory —
        // the kv_cache_append writes new K/V in place, and attention only
        // reads up to `cache_len`.
        for c in caches.iter_mut() {
            c.set_len(0);
            if let Some(cl) = c.context_lens_mut() {
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
        // K=KvFp16 path uses the existing fused launchers that work over
        // `KvCache<B, KvFp16>`; INT8 forwards through this method are
        // re-routed at the public entry (`prefill_internal` / `decode_internal`)
        // before reaching here. `as_fp16_mut` panics on the Int8 variant —
        // that's a contract bug if it ever fires here.
        let cache = caches[li].as_fp16_mut();
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
            .map(|c| c.len())
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
            .map(|c| c.len())
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
            .map(|(cid, _, _)| self.kv_caches.get(cid).expect("kv_caches missing")[0].len() as u32)
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
            lm_head.forward(
                &mut ctx,
                &self.scratch.norm_out,
                &mut self.scratch.batch_logits,
                m,
            );
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
                let new_len = caches[li].len() + 1;
                caches[li].set_len(new_len);
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
}

impl<B: MoeLlmBackend + BackendInt8KvOps, K: KvDtypeKind> DecoderOnlyLLM
    for LlamaFamilyModel<B, K>
{
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
                    if !c0.paged_block_indices().is_empty() {
                        alloc.free(c0.paged_block_indices());
                    }
                }
                for c in caches.iter_mut() {
                    c.paged_block_indices_mut().clear();
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
        // Backend capability gate: unified path requires varlen QKV +
        // paged_varlen_attention. Backends without them (Metal as of
        // 2026-05-09, CPU) should drop back to per-item dispatch via
        // forward_layer / decode_batch_internal.
        if !B::supports_varlen_qkv() {
            return Err(ferrum_types::FerrumError::unsupported(
                "LlamaFamilyModel::unified_forward: backend lacks varlen \
                 QKV kernels. Engine will fall back to per-item dispatch.",
            ));
        }
        // Touch the first item's KV cache so `ensure_kv` lazily allocates
        // `paged_pools` when paged is enabled (env or backend default).
        self.ensure_kv(&items[0].0);
        if self.paged_pools.is_none() {
            return Err(ferrum_types::FerrumError::unsupported(
                "LlamaFamilyModel::unified_forward: paged KV required; \
                 enable via FERRUM_METAL_PAGED_KV=1 (cross-backend env). \
                 Engine will fall back to per-item dispatch.",
            ));
        }
        // ensure_kv all items up front and surface pool-exhaust as a clean
        // per-request error (`ResourceExhausted`) instead of panicking
        // inside `unified_forward_internal` — a panic kills the tokio
        // worker and dangles every cache_id still in flight, which then
        // permanently exhausts the pool.
        for (cid, _, _, _) in items {
            self.ensure_kv(cid);
            if !self.kv_caches.contains_key(cid) {
                return Err(ferrum_types::FerrumError::resource_exhausted(format!(
                    "paged KV pool exhausted for cache_id={cid:?}; back off"
                )));
            }
        }
        Ok(self.unified_forward_internal(items))
    }

    fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        // Delegate to the inherent implementation on LlamaFamilyModel.
        LlamaFamilyModel::<B, K>::forward_verify(self, cache_id, tokens)
    }

    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        if let Some(caches) = self.kv_caches.get_mut(cache_id) {
            for c in caches.iter_mut() {
                if new_len < c.len() {
                    c.set_len(new_len);
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
                    if !c0.paged_block_indices().is_empty() {
                        alloc.free(c0.paged_block_indices());
                    }
                }
                // Clear the host-side mirror on every layer so a
                // free-pool reuse re-allocates fresh blocks.
                for c in caches.iter_mut() {
                    c.paged_block_indices_mut().clear();
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
