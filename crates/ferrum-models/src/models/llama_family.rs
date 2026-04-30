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

use ferrum_kernels::backend::{Backend, KvCache};

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
            max_tokens: t,
        }
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
    pub kv_caches: HashMap<String, Vec<KvCache<B>>>,
    /// Free pool of pre-allocated KV cache slots. Released caches return
    /// here instead of being dropped, so their device pointers stay valid
    /// across requests — critical for graph capture (pointers baked into
    /// the captured graph would otherwise dangle).
    kv_free_pool: Vec<Vec<KvCache<B>>>,

    // ── Graph capture state (CUDA only; harmless no-op on other backends) ──
    /// Count of eager decode steps run so far. After `GRAPH_WARMUP`, the
    /// next step captures the decode flow as a graph.
    graph_warmup: usize,
    /// True if capture was attempted but failed (e.g. backend doesn't
    /// support graph capture). Stops further attempts, falls back to eager.
    graph_capture_failed: bool,
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
            B::reset_graph(&mut ctx);
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
            graph_warmup: 0,
            graph_capture_failed: false,
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
            B::reset_graph(&mut ctx);
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
            graph_warmup: 0,
            graph_capture_failed: false,
        })
    }

    /// Grow scratch buffers if `tokens` exceeds the current sizing.
    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            // Any captured decode graph holds pointers to the old scratch
            // buffers; those are about to be freed. Invalidate first so the
            // next decode falls back to eager + re-captures with fresh ptrs.
            // Critical for multi-turn chat (turn N+1's prefill may grow scratch).
            {
                let mut ctx = B::new_context();
                B::reset_graph(&mut ctx);
            }
            self.scratch = LlamaFamilyScratch::alloc(&self.cfg, tokens);
            self.graph_warmup = 0;
            self.graph_capture_failed = false;
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
        const DEFAULT_KV_CAPACITY: usize = 4096;
        let max = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY));

        // Try pool first — reused buffers have stable device pointers,
        // so a captured decode graph can be replayed for this request too.
        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.num_layers)
                .map(|_| KvCache {
                    k: B::alloc(nkv * max * hd),
                    v: B::alloc(nkv * max * hd),
                    len: 0,
                    capacity: max,
                    num_kv_heads: nkv,
                    head_dim: hd,
                })
                .collect()
        });
        // Reset logical length; buffers stay. No need to zero the memory —
        // the kv_cache_append writes new K/V in place, and attention only
        // reads up to `cache_len`.
        for c in caches.iter_mut() {
            c.len = 0;
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
        let used_qkv_into_cache = B::split_qkv_norm_rope_into_cache(
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
        .is_ok();
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

        // 6. Flash attention over strided cache.
        let _attn_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };
        //    `causal` is always true for decoder-only LLMs — every query must
        //    mask out future tokens. (The `tokens > 1` heuristic from the old
        //    path only worked because single-token decode trivially "attends"
        //    to one position.) Sliding-window models (Mistral v0.1) narrow
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

            // Fast path: graph replay (if available).
            match B::replay_last_graph(&mut ctx) {
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
            if B::end_graph_capture(&mut ctx).is_err() {
                self.graph_capture_failed = true;
            } else {
                // Stream capture mode RECORDS ops into the graph without
                // executing them. scratch.logits still holds the previous
                // step's value. Replay the just-captured graph once to
                // actually execute and produce this step's logits. Without
                // this, the capture step's to_vec returns stale logits,
                // yielding a 1-token offset in the generated sequence.
                if B::replay_last_graph(&mut ctx).is_err() {
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

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let mut ctx = B::new_context();

        // 0. Embed all M tokens into residual [M, H]
        let tokens: Vec<u32> = batch.iter().map(|(_, t, _)| *t).collect();
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

        // 1..num_layers: batched forward for each layer
        for li in 0..self.cfg.num_layers {
            self.forward_layer_batched_decode(&mut ctx, li, batch, &mut residual, m);
        }

        // Final RMSNorm on [M, H] → norm_out [M, H]
        B::rms_norm(
            &mut ctx,
            &residual,
            &self.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.scratch.norm_out,
            m,
            h,
        );

        // LM head with m=M → batch_logits [M, vocab]
        let lm_head = self
            .lm_head
            .as_ref()
            .expect("decode_batch_internal called on backbone-only model (no lm_head)");
        lm_head.forward(
            &mut ctx,
            &self.scratch.norm_out,
            &mut self.scratch.batch_logits,
            m,
        );

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

        // 1. rms_norm [M, H]  → norm_out
        B::rms_norm(
            ctx,
            residual,
            &layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            m,
            h,
        );

        // 2. qkv_proj (GEMM m=M): norm_out [M, H] → qkv_out [M, QKV]
        layer
            .qkv_proj
            .forward(ctx, &self.scratch.norm_out, &mut self.scratch.qkv_out, m);

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

        // 4-6. Per-item loop for rope + kv_append + attention.
        //      Each item has its own cache_id + pos + kv_len.
        for (i, (cache_id, _token, pos)) in batch.iter().enumerate() {
            let pos_i = *pos as usize;

            // Extract item i's Q/K/V from batched buffers.
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

            // qk_norm_rope with tokens=1, per-item pos.
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
                &self.scratch.k_head_major_single,
                &self.scratch.v_head_major_single,
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

            // Untranspose head-major → token-major + inject into batched
            // attn_flat[M, Q]. For tokens=1 the head-major and
            // token-major layouts are byte-identical (both flat to
            // [heads * head_dim] = [q_dim] floats), so we skip the
            // transpose dispatch entirely and copy attn_head_major_single
            // straight into the per-item slot. Saves 1 dispatch per
            // batch-item per layer.
            B::copy_slice(
                ctx,
                &self.scratch.attn_head_major_single,
                0,
                &mut self.scratch.attn_flat,
                i * q_dim,
                q_dim,
            );
        }

        // 7. o_proj (GEMM m=M): attn_flat [M, Q] → o_proj_out [M, H]
        layer.o_proj.forward(
            ctx,
            &self.scratch.attn_flat,
            &mut self.scratch.o_proj_out,
            m,
        );

        // 8. Fused residual add + post-attention RMSNorm.
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

        // 9. gate_up_proj (GEMM m=M)
        layer.gate_up_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.gate_up_out,
            m,
        );

        // 10. SwiGLU
        B::fused_silu_mul_split(
            ctx,
            &self.scratch.gate_up_out,
            &mut self.scratch.silu_out,
            m,
            im,
        );

        // 11. down_proj (GEMM m=M)
        layer
            .down_proj
            .forward(ctx, &self.scratch.silu_out, &mut self.scratch.mlp_out, m);

        // 12. Residual add
        B::add_inplace(ctx, residual, &self.scratch.mlp_out, m * h);
    }
}

impl<B: Backend> DecoderOnlyLLM for LlamaFamilyModel<B> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn kv_capacity(&self) -> usize {
        // Mirror the bound `ensure_kv` will use when allocating the cache.
        let model_max = self.cfg.max_seq_len;
        const DEFAULT_KV_CAPACITY: usize = 4096;
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
        // Captured graph expects a specific cache layout; roll it back too.
        let mut ctx = B::new_context();
        B::reset_graph(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
    }

    fn release(&mut self, cache_id: &str) {
        // Sync + drop graph BEFORE touching cache buffers. The graph was
        // actively running replays up to this point; destroying the graph
        // while the allocator pool still has in-flight references from the
        // graph's kernels corrupts stream state. Sync first to drain, then
        // destroy graph, then sync again to ensure cleanup completes.
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_graph(&mut ctx);
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;

        // Return the cache's buffers to the free pool instead of dropping.
        // Pointers stay stable for the next request's captured graph.
        if let Some(caches) = self.kv_caches.remove(cache_id) {
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        // Hard reset: drop all caches AND the pool, invalidate graph.
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_graph(&mut ctx);
        B::sync(&mut ctx);
        self.graph_warmup = 0;
        self.graph_capture_failed = false;
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
