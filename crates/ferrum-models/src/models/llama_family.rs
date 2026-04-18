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

use ferrum_kernels::backend::{Backend, KvCache};
use ferrum_quantization::{Linear, WeightLoader};
use ferrum_types::Result;

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};

/// Full Qwen3 architecture config (everything the model code needs, not just
/// the engine-facing subset in `LlmRuntimeConfig`).
#[derive(Clone, Debug)]
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
    /// `.take()` it (no alloc-placeholder needed for mem::replace, avoiding
    /// a transient CudaSlice drop that corrupts the stream pool state after
    /// graph capture on Blackwell).
    pub residual: Option<B::Buffer>,
    pub norm_out: B::Buffer,
    pub qkv_out: B::Buffer,
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

    pub embed: B::Buffer,
    pub layers: Vec<LlamaFamilyLayer<B>>,
    pub final_norm_w: B::Buffer,
    pub lm_head: Box<dyn Linear<B>>,

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
            embed,
            layers,
            final_norm_w,
            lm_head,
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
            self.scratch = LlamaFamilyScratch::alloc(&self.cfg, tokens);
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
        let max = self.cfg.max_seq_len;

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
        B::rms_norm(
            ctx,
            residual,
            &layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            tokens,
            h,
        );

        // 2. Fused QKV projection (Linear dispatches to Dense/GPTQ/AWQ/GGUF)
        layer.qkv_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.qkv_out,
            tokens,
        );

        // 3. Split fused QKV → token-major Q/K/V
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

        // 4. Fused QK-norm + RoPE + transpose to head-major
        //    Qwen3: mode=1 (norm + rope). Non-QK-norm variants: mode=2 (rope only).
        //    V always uses mode=0 (transpose only).
        let qk_mode: i32 = if cfg.has_qk_norm { 1 } else { 2 };
        let dummy = &layer.input_ln_w;
        let q_norm_w = layer.q_norm_w.as_ref().unwrap_or(dummy);
        let k_norm_w = layer.k_norm_w.as_ref().unwrap_or(dummy);

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
            dummy, // unused in mode 0
            &self.rope.cos,
            &self.rope.sin,
            &mut self.scratch.v_head_major,
            tokens,
            nkv,
            hd,
            pos_offset,
            eps,
            0, // transpose only
        );

        // 5. Append K/V to pre-allocated head-major cache
        let caches = self
            .kv_caches
            .get_mut(cache_id)
            .expect("ensure_kv must be called before forward_layer");
        let cache = &mut caches[li];
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
        cache.len += tokens;
        let kv_len = cache.len;
        let kv_stride = cache.capacity;

        // 6. Flash attention over strided cache.
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

        // 7. Untranspose head-major → token-major for O-proj input
        B::transpose_head_to_token(
            ctx,
            &self.scratch.attn_head_major_out,
            &mut self.scratch.attn_flat,
            tokens,
            nh,
            hd,
        );

        // 8. O projection
        layer.o_proj.forward(
            ctx,
            &self.scratch.attn_flat,
            &mut self.scratch.o_proj_out,
            tokens,
        );

        // 9. Fused residual-add + post-attention RMSNorm.
        //    Writes the new residual back into `residual` and the normed
        //    value into `norm_out`.
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

        // 10. Fused gate+up projection
        layer.gate_up_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.gate_up_out,
            tokens,
        );

        // 11. SwiGLU: silu(gate) * up
        B::fused_silu_mul_split(
            ctx,
            &self.scratch.gate_up_out,
            &mut self.scratch.silu_out,
            tokens,
            im,
        );

        // 12. Down projection
        layer.down_proj.forward(
            ctx,
            &self.scratch.silu_out,
            &mut self.scratch.mlp_out,
            tokens,
        );

        // 13. Final residual add
        B::add_inplace(ctx, residual, &self.scratch.mlp_out, tokens * h);
    }

    /// Prefill: process `tokens` prompt tokens in a single batch, return
    /// `[vocab_size]` logits for the last position.
    pub fn prefill_internal(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "prefill called with empty token list");
        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

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
            .expect("scratch residual missing");
        B::embedding_lookup(&mut ctx, &self.embed, tokens, &mut residual, h);

        for li in 0..self.cfg.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, 0, seq_len);
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
        self.lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );

        // to_vec() syncs internally before dtoh — no explicit B::sync needed.

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
            eprintln!("[MODEL] pre-replay pos={pos} token={token}");
            match B::replay_last_graph(&mut ctx) {
                Ok(true) => {
                    eprintln!("[MODEL] replay ok, pre-sync");
                    B::sync(&mut ctx);
                    eprintln!("[MODEL] post-sync, calling to_vec");
                    let out = B::to_vec(&self.scratch.logits, vocab);
                    eprintln!("[MODEL] to_vec done, len={}", out.len());
                    return out;
                }
                Ok(false) => {
                    eprintln!("[MODEL] no graph yet, eager");
                }
                Err(e) => {
                    eprintln!("[MODEL] replay err: {e:?}");
                }
            }
        }

        let should_capture = graph_enabled
            && !self.graph_capture_failed
            && self.graph_warmup >= GRAPH_WARMUP;

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
            .expect("scratch residual missing");
        B::embedding_lookup(&mut ctx, &self.embed, &[token], &mut residual, h);

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

        self.lm_head.forward(
            &mut ctx,
            &self.scratch.last_normed,
            &mut self.scratch.logits,
            1,
        );

        if should_capture && !self.graph_capture_failed {
            if B::end_graph_capture(&mut ctx).is_err() {
                self.graph_capture_failed = true;
            }
            B::set_dev_state_mode(&mut ctx, false);
        } else {
            self.graph_warmup += 1;
        }

        // to_vec() syncs internally before dtoh — no explicit B::sync needed.
        self.scratch.residual = Some(residual);

        B::to_vec(&self.scratch.logits, vocab)
    }
}

impl<B: Backend> DecoderOnlyLLM for LlamaFamilyModel<B> {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.runtime_cfg
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.prefill_internal(cache_id, tokens)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.decode_internal(cache_id, token, pos)
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
