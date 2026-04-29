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

use ferrum_kernels::backend::{Backend, KvCache};
use ferrum_quantization::WeightLoader;
use ferrum_types::{FerrumError, Result};

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyLayer, RopeCache};
use crate::moe::{moe_forward, ExpertStack};
use crate::moe_config::Qwen3MoeConfig;

// Decode-side per-op profile counters — same names as the dense path
// so existing tooling (`FERRUM_DECODE_OP_PROFILE=1` log scrapers) keeps
// working without a separate switch for MoE.
static ATTN_TIME_US: AtomicU64 = AtomicU64::new(0);
static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_TIME_US: AtomicU64 = AtomicU64::new(0);
static MOE_CALLS: AtomicU64 = AtomicU64::new(0);

/// Per-layer MoE state: router linear (small) + per-expert MLP stack.
pub struct Qwen3MoeLayerState<B: Backend> {
    /// Router projection `[hidden] → [num_experts]` — tiny, never sparse,
    /// always runs the full GEMV.
    pub router: Box<dyn ferrum_quantization::Linear<B>>,
    /// Per-expert weight stack. Each entry's `gate_up` is the fused
    /// `[gate; up]` projection; `down` is the post-SwiGLU output proj.
    pub experts: ExpertStack<B>,
}

/// Reusable scratch buffers for the MoE forward path. All sized at
/// allocation time and reused across layers / forward calls.
pub struct Qwen3MoeScratch<B: Backend> {
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

    // ── Final-token / lm_head outputs ────────────────────────────────
    pub last_hidden: B::Buffer,
    pub last_normed: B::Buffer,
    pub logits: B::Buffer,
    pub batch_logits: B::Buffer,

    pub max_tokens: usize,
}

impl<B: Backend> Qwen3MoeScratch<B> {
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
            last_hidden: B::alloc(h),
            last_normed: B::alloc(h),
            logits: B::alloc(vocab),
            batch_logits: B::alloc(t * vocab),
            max_tokens: t,
        }
    }
}

/// Qwen3-MoE decoder model.
///
/// Holds the same per-layer attention weights as [`LlamaFamilyModel`]
/// plus a [`Qwen3MoeLayerState`] per layer for the MoE FFN. Routing,
/// expert dispatch, and weighted combine all happen inside
/// [`moe_forward`]; this struct only owns the storage and orchestrates
/// the per-layer call sequence.
pub struct Qwen3MoeModel<B: Backend> {
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

    pub kv_caches: HashMap<String, Vec<KvCache<B>>>,
    kv_free_pool: Vec<Vec<KvCache<B>>>,
}

impl<B: Backend> Qwen3MoeModel<B> {
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
            B::reset_graph(&mut ctx);
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
        })
    }

    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            {
                let mut ctx = B::new_context();
                B::reset_graph(&mut ctx);
            }
            self.scratch = Qwen3MoeScratch::alloc(&self.cfg, tokens);
        }
    }

    pub(crate) fn ensure_kv(&mut self, cache_id: &str) {
        if self.kv_caches.contains_key(cache_id) {
            return;
        }
        let nkv = self.cfg.base.num_kv_heads;
        let hd = self.cfg.base.head_dim;
        let model_max = self.cfg.base.max_seq_len;
        let max = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|cap| cap.min(model_max))
            .unwrap_or(model_max);

        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.base.num_layers)
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
        for c in caches.iter_mut() {
            c.len = 0;
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
        let moe_layer = &self.moe_layers[li];

        let attn_t0 = if std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok() {
            B::sync(ctx);
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 1. Input RMSNorm
        B::rms_norm(
            ctx,
            residual,
            &attn_layer.input_ln_w,
            eps,
            &mut self.scratch.norm_out,
            tokens,
            h,
        );

        // 2. Fused QKV
        attn_layer.qkv_proj.forward(
            ctx,
            &self.scratch.norm_out,
            &mut self.scratch.qkv_out,
            tokens,
        );

        // 3. split QKV
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

        // 4. QK-norm + RoPE → head-major Q/K, transpose-only V
        let qk_mode: i32 = if cfg_base.has_qk_norm { 1 } else { 2 };
        let dummy = &attn_layer.input_ln_w;
        let q_norm_w = attn_layer.q_norm_w.as_ref().unwrap_or(dummy);
        let k_norm_w = attn_layer.k_norm_w.as_ref().unwrap_or(dummy);
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

        // 5. KV append + 6. flash attention.
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

        if let Some(t0) = attn_t0 {
            B::sync(ctx);
            ATTN_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 7. transpose head-major → token-major.
        B::transpose_head_to_token(
            ctx,
            &self.scratch.attn_head_major_out,
            &mut self.scratch.attn_flat,
            tokens,
            nh,
            hd,
        );

        // 8. O-proj.
        attn_layer.o_proj.forward(
            ctx,
            &self.scratch.attn_flat,
            &mut self.scratch.o_proj_out,
            tokens,
        );

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
        //     `moe_forward` writes acc_buf → moe_out[b] once per token,
        //     so we don't need to pre-zero `moe_out` (every byte will be
        //     overwritten by the per-token copy_slice from acc_buf).
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

        // 12. residual += moe_out
        B::add_inplace(ctx, residual, &self.scratch.moe_out, tokens * h);

        if let Some(t0) = moe_t0 {
            B::sync(ctx);
            MOE_TIME_US.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MOE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
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

        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        B::embedding_lookup(&mut ctx, &self.embed, tokens, &mut residual, h);

        for li in 0..self.cfg.base.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos_offset, seq_len)
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

        let mut residual = self
            .scratch
            .residual
            .take()
            .expect("scratch residual missing (previous call didn't restore)");
        B::embedding_lookup(&mut ctx, &self.embed, &[token], &mut residual, h);

        for li in 0..self.cfg.base.num_layers {
            self.forward_layer(&mut ctx, li, cache_id, &mut residual, pos as usize, 1)
                .expect("forward_layer");
        }

        B::rms_norm(
            &mut ctx,
            &residual,
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
        self.scratch.residual = Some(residual);

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
        }

        B::to_vec(&self.scratch.logits, vocab)
    }
}

impl<B: Backend> DecoderOnlyLLM for Qwen3MoeModel<B> {
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
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_graph(&mut ctx);
        B::sync(&mut ctx);
        if let Some(caches) = self.kv_caches.remove(cache_id) {
            self.kv_free_pool.push(caches);
        }
    }

    fn reset(&mut self) {
        let mut ctx = B::new_context();
        B::sync(&mut ctx);
        B::reset_graph(&mut ctx);
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
fn stub_linear<B: Backend>(
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
