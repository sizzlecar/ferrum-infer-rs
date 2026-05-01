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

// Fine-grained decode-only counters, populated by
// `moe_forward_stacked_decode_impl` when FERRUM_DECODE_OP_PROFILE is set.
// Each is per-layer summed over the layers in one decode token; drained
// at the bottom of `decode_internal`.
static DEC_ROUTE_US: AtomicU64 = AtomicU64::new(0);
static DEC_GATE_US: AtomicU64 = AtomicU64::new(0);
static DEC_UP_US: AtomicU64 = AtomicU64::new(0);
static DEC_SILU_US: AtomicU64 = AtomicU64::new(0);
static DEC_DOWN_US: AtomicU64 = AtomicU64::new(0);
static DEC_WSUM_US: AtomicU64 = AtomicU64::new(0);
// Single-shot per decode token (not per-layer).
static DEC_EMBED_US: AtomicU64 = AtomicU64::new(0);
static DEC_FINAL_NORM_US: AtomicU64 = AtomicU64::new(0);
static DEC_LM_HEAD_US: AtomicU64 = AtomicU64::new(0);

// MoE batched-prefill sub-stage counters (gate / up / down mul_mm_id +
// silu + weighted_sum + host topk). Same FERRUM_DECODE_OP_PROFILE gate.
static MOE_PREFILL_HOST_TOPK_US: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_HOST_TOPK_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_GATE_US: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_GATE_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_UP_US: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_UP_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_SILU_US: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_SILU_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_DOWN_US: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_WSUM_US: AtomicU64 = AtomicU64::new(0);
static MOE_PREFILL_WSUM_CALLS: AtomicU64 = AtomicU64::new(0);

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
            gate_out_stacked: B::alloc(t * cfg.num_experts_per_tok * inter),
            up_out_stacked: B::alloc(t * cfg.num_experts_per_tok * inter),
            silu_stacked: B::alloc(t * cfg.num_experts_per_tok * inter),
            down_out_stacked: B::alloc(t * cfg.num_experts_per_tok * h),
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
        // See `LlamaFamilyModel::ensure_kv` for the rationale on the 4096
        // chat-friendly default and how `FERRUM_KV_CAPACITY` overrides it.
        let model_max = self.cfg.base.max_seq_len;
        const DEFAULT_KV_CAPACITY: usize = 4096;
        let max = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|cap| cap.min(model_max))
            .unwrap_or_else(|| model_max.min(DEFAULT_KV_CAPACITY));

        let mut caches = self.kv_free_pool.pop().unwrap_or_else(|| {
            (0..self.cfg.base.num_layers)
                .map(|_| KvCache {
                    k: B::alloc(nkv * max * hd),
                    v: B::alloc(nkv * max * hd),
                    len: 0,
                    capacity: max,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    // Paged-KV not yet wired for Qwen3-MoE — keeps the
                    // contiguous decode path. Phase 4+ may enable it
                    // here too once the MoE expert dispatch's KV reads
                    // also go through block_table indirection.
                    block_size: 0,
                    block_table: None,
                    context_lens: None,
                    paged_block_indices: Vec::new(),
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
        // K/V directly into the cache slot. Skips both the head-major
        // K/V scratch buffers and the `kv_cache_append_head_major`
        // dispatch on the decode hot path.
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

        if stacked_path_available {
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
        moe_forward_stacked_decode_impl::<B>(
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
        moe_forward_batched_prefill_impl::<B>(
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
        }

        B::to_vec(&self.scratch.logits, vocab)
    }
}

impl<B: Backend> DecoderOnlyLLM for Qwen3MoeModel<B> {
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

/// Batched MoE FFN — decode (m=1) and per-token-prefill (m>1 looped).
///
/// Three batched `gemv_quant_moe_id` dispatches per token: gate (broadcast
/// activation), up (broadcast activation), down (per-slot activation —
/// each expert sees its own silu·up). The per-(token, expert) outer loop
/// shrinks from `top_k * 4` dispatches per layer to **3 batched + 1
/// silu_mul_split + 1 weighted_sum_dispatch_loop**.
///
/// For prefill (m > 1) we loop over tokens externally — each token's
/// router output drives a single batched call. Still much faster than
/// the per-(token, expert) per-Linear path because the gemvs are batched.
///
/// Free function (not a method) so the caller can split the borrow on
/// `self` between `moe_layers[li]` (immutable) and `scratch` (mutable).
#[allow(clippy::too_many_arguments)]
fn moe_forward_stacked_decode_impl<B: Backend>(
    ctx: &mut B::Context,
    moe_layer: &Qwen3MoeLayerState<B>,
    scratch: &mut Qwen3MoeScratch<B>,
    h: usize,
    inter: usize,
    top_k: usize,
    n_exp: usize,
    norm_topk_prob: bool,
    tokens: usize,
    residual: &mut B::Buffer,
    // If `Some`, fold the NEXT layer's leading rms_norm into the
    // weighted-sum-residual tail using `weighted_sum_residual_norm_stacked`.
    next_norm_w: Option<&B::Buffer>,
    eps: f32,
) -> Result<()> {
    // GPU-side routing: one Metal launch reads router_logits and writes
    // selected ids + combine weights directly into device-side scratch
    // buffers. Eliminates the per-layer `B::sync + B::to_vec(router_logits)
    // + host route()` round trip — the dominant remaining cost in the
    // decode hot path (~10% of total decode latency).
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

    let t0 = stage_t0();
    B::route_topk_softmax(
        ctx,
        &scratch.router_logits,
        &mut scratch.ids_buf,
        &mut scratch.weights_buf,
        tokens,
        n_exp,
        top_k,
        norm_topk_prob,
    )?;
    stage_end(t0, ctx, &DEC_ROUTE_US);

    let gate_stacked = moe_layer.experts.gate_stacked.as_ref().unwrap();
    let up_stacked = moe_layer.experts.up_stacked.as_ref().unwrap();
    let down_stacked = moe_layer.experts.down_stacked.as_ref().unwrap();

    // moe_forward_stacked_decode_impl is only called when `tokens == 1`
    // (the branch in `forward_layer` routes prefill m>1 through
    // `moe_forward_batched_prefill_impl` instead). The for-b loop and
    // the copy norm_out[b*h] → x_single were vestigial scaffolding;
    // for tokens=1 norm_out[0..h] IS the activation row, and we can
    // pass it straight to the gemv kernel via src1_stride=0 broadcast.
    debug_assert_eq!(
        tokens, 1,
        "moe_forward_stacked_decode_impl expects tokens=1 (prefill goes through moe_forward_batched_prefill_impl)"
    );
    let _ = tokens; // silence unused-warning when assertion is compiled out

    {
        // ids_buf and weights_buf populated by the GPU router above —
        // no host writes needed here in the decode path.

        // 1. Batched gate gemv — broadcast input across top_k slots.
        //    src1 = norm_out (which has hidden floats at offset 0),
        //    src1_stride=0 → all slots read the same row.
        let t0 = stage_t0();
        B::gemv_quant_moe_id(
            ctx,
            &scratch.norm_out,
            gate_stacked,
            &scratch.ids_buf,
            &mut scratch.gate_out_stacked,
            top_k,
            0, // broadcast
        )?;
        stage_end(t0, ctx, &DEC_GATE_US);

        // 2. Batched up gemv — also broadcast.
        let t0 = stage_t0();
        B::gemv_quant_moe_id(
            ctx,
            &scratch.norm_out,
            up_stacked,
            &scratch.ids_buf,
            &mut scratch.up_out_stacked,
            top_k,
            0,
        )?;
        stage_end(t0, ctx, &DEC_UP_US);

        // 3. Stacked SiLU·gate → silu_stacked. Single dispatch covers
        //    all top_k slots — replaces the per-slot loop's
        //    (3 copy_slice + 1 silu_mul) × 8 = 32 dispatches.
        let t0 = stage_t0();
        B::silu_mul_stacked(
            ctx,
            &scratch.gate_out_stacked,
            &scratch.up_out_stacked,
            &mut scratch.silu_stacked,
            top_k,
            inter,
        )?;
        stage_end(t0, ctx, &DEC_SILU_US);

        // 4. Batched down gemv — per-slot input via src1_stride = inter.
        //    silu_stacked[k * inter ..] is the activation row for slot k.
        let t0 = stage_t0();
        B::gemv_quant_moe_id(
            ctx,
            &scratch.silu_stacked,
            down_stacked,
            &scratch.ids_buf,
            &mut scratch.down_out_stacked,
            top_k,
            inter,
        )?;
        stage_end(t0, ctx, &DEC_DOWN_US);

        // 5. Fused weighted-sum + residual-add (+ optional next-layer
        //    rms_norm). Two paths:
        //
        //    * `next_norm_w = Some(_)` (cross-layer fusion): one kernel
        //      computes residual[i] += Σ_k w[k] · down[k, i] AND
        //      norm_out[i] = residual[i] · scale · next_norm_w[i].
        //      The next layer's leading rms_norm is skipped. Saves an
        //      additional dispatch per layer transition.
        //    * `next_norm_w = None` (last layer): just residual-add.
        let t0 = stage_t0();
        if let Some(nnw) = next_norm_w {
            B::weighted_sum_residual_norm_stacked(
                ctx,
                &scratch.down_out_stacked,
                &scratch.weights_buf,
                residual,
                nnw,
                &mut scratch.norm_out,
                top_k,
                h,
                eps,
            )?;
        } else {
            B::weighted_sum_residual_stacked(
                ctx,
                &scratch.down_out_stacked,
                &scratch.weights_buf,
                residual,
                top_k,
                h,
            )?;
        }
        stage_end(t0, ctx, &DEC_WSUM_US);
    }

    Ok(())
}

/// Batched MoE FFN for prefill (m > 1).
///
/// One pass through the expert dispatch — replaces the per-token loop
/// with three batched 2-D mul_mm_id dispatches (gate, up, down) where
/// each expert's slab of (token, slot) pairs runs as one gemm tile.
/// Per-layer dispatch count: ~6 (router + 3 mul_mm_id + silu + wsum)
/// independent of `tokens`. Compare to the decode-style stacked path
/// that emits ~10 per token.
///
/// Free function so the caller can split the borrow on `self` between
/// `moe_layers[li]` (immutable) and `scratch` (mutable).
#[allow(clippy::too_many_arguments)]
fn moe_forward_batched_prefill_impl<B: Backend>(
    ctx: &mut B::Context,
    moe_layer: &Qwen3MoeLayerState<B>,
    scratch: &mut Qwen3MoeScratch<B>,
    h: usize,
    inter: usize,
    top_k: usize,
    n_exp: usize,
    norm_topk_prob: bool,
    tokens: usize,
) -> Result<()> {
    let prof = std::env::var("FERRUM_DECODE_OP_PROFILE").is_ok();
    let stage_t0 = || -> Option<std::time::Instant> {
        if prof {
            Some(std::time::Instant::now())
        } else {
            None
        }
    };
    let stage_end =
        |t0: Option<std::time::Instant>, ctx: &mut B::Context, us: &AtomicU64, n: &AtomicU64| {
            if let Some(t) = t0 {
                B::sync(ctx);
                us.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        };

    // GPU-side routing: keep the whole pipeline device-resident. Two
    // dispatches replace the per-layer `B::sync + to_vec(router_logits)
    // + host route() + host compute_ids_tpe + write_back` round trip.
    //
    //   1. `route_topk_softmax` writes selected expert IDs (flat
    //      `[batch, top_k]`) into `selected_ids_buf` and the post-renorm
    //      combine weights directly into `weights_2d`.
    //   2. `compute_ids_tpe_gpu` buckets those pairs into `tpe_buf` and
    //      `ids_2d` using device-side atomic_fetch_add slot claims. The
    //      `ids_2d` row stride is the worst-case `tokens * top_k`; the
    //      consumer GEMM stops at `tpe[e]` so the over-strided columns
    //      cost only launch overhead, not real compute.
    //
    // `FERRUM_MOE_HOST_TOPK=1`        → legacy CPU softmax+topk+bucket
    // `FERRUM_MOE_DIRECT_DISPATCH=1`  → GPU topk but worst-case GEMM grid
    // (default)                       → GPU topk + indirect-dispatched GEMM
    //                                    (grid sized from max(tpe[e]))
    let use_gpu_topk = std::env::var("FERRUM_MOE_HOST_TOPK").as_deref() != Ok("1");
    let use_indirect_dispatch =
        use_gpu_topk && std::env::var("FERRUM_MOE_DIRECT_DISPATCH").as_deref() != Ok("1");
    let max_per_expert = if use_gpu_topk {
        let t0 = stage_t0();
        B::route_topk_softmax(
            ctx,
            &scratch.router_logits,
            &mut scratch.selected_ids_buf,
            &mut scratch.weights_2d,
            tokens,
            n_exp,
            top_k,
            norm_topk_prob,
        )?;
        B::compute_ids_tpe_gpu(
            ctx,
            &scratch.selected_ids_buf,
            &mut scratch.tpe_buf,
            &mut scratch.ids_2d,
            &mut scratch.gate_up_args_buf,
            &mut scratch.down_args_buf,
            tokens,
            n_exp,
            top_k,
            inter,
            h,
        )?;
        stage_end(
            t0,
            ctx,
            &MOE_PREFILL_HOST_TOPK_US,
            &MOE_PREFILL_HOST_TOPK_CALLS,
        );
        // Worst-case ids row stride; matches `dispatch_compute_ids_tpe`.
        tokens * top_k
    } else {
        use ferrum_kernels::moe_host::compute_ids_tpe;
        let t0 = stage_t0();
        B::sync(ctx);
        let logits_host = B::to_vec(&scratch.router_logits, tokens * n_exp);
        let route = crate::moe::router::route(&logits_host, tokens, n_exp, top_k, norm_topk_prob);
        let (tpe_host, ids_host, max_per_expert) =
            compute_ids_tpe(&route.expert_ids, n_exp, tokens, top_k);
        B::write_i32_into(&mut scratch.tpe_buf, &tpe_host);
        B::write_i32_into(&mut scratch.ids_2d, &ids_host);
        B::write_f32_into(&mut scratch.weights_2d, &route.expert_weights);
        stage_end(
            t0,
            ctx,
            &MOE_PREFILL_HOST_TOPK_US,
            &MOE_PREFILL_HOST_TOPK_CALLS,
        );
        max_per_expert
    };

    let gate_stacked = moe_layer.experts.gate_stacked.as_ref().unwrap();
    let up_stacked = moe_layer.experts.up_stacked.as_ref().unwrap();
    let down_stacked = moe_layer.experts.down_stacked.as_ref().unwrap();

    // 1. Batched gate gemm — one launch covers all (token, expert) pairs.
    //    src1 layout: [batch, ne11=1, K] (broadcast: each pair reads its
    //    token's row, slot index ignored).
    //    dst layout:  [batch, top_k, expert_inter] — natural.
    let t0 = stage_t0();
    if use_indirect_dispatch {
        B::gemm_quant_moe_id_indirect(
            ctx,
            &scratch.norm_out,
            gate_stacked,
            &scratch.ids_2d,
            &scratch.tpe_buf,
            &mut scratch.gate_out_stacked,
            &scratch.gate_up_args_buf,
            1, // ne11 = 1: broadcast
            top_k,
            max_per_expert,
            tokens,
        )?;
    } else {
        B::gemm_quant_moe_id(
            ctx,
            &scratch.norm_out,
            gate_stacked,
            &scratch.ids_2d,
            &scratch.tpe_buf,
            &mut scratch.gate_out_stacked,
            1,
            top_k,
            max_per_expert,
            tokens,
        )?;
    }
    stage_end(t0, ctx, &MOE_PREFILL_GATE_US, &MOE_PREFILL_GATE_CALLS);

    // 2. Batched up gemm — same shape as gate.
    let t0 = stage_t0();
    if use_indirect_dispatch {
        B::gemm_quant_moe_id_indirect(
            ctx,
            &scratch.norm_out,
            up_stacked,
            &scratch.ids_2d,
            &scratch.tpe_buf,
            &mut scratch.up_out_stacked,
            &scratch.gate_up_args_buf,
            1,
            top_k,
            max_per_expert,
            tokens,
        )?;
    } else {
        B::gemm_quant_moe_id(
            ctx,
            &scratch.norm_out,
            up_stacked,
            &scratch.ids_2d,
            &scratch.tpe_buf,
            &mut scratch.up_out_stacked,
            1,
            top_k,
            max_per_expert,
            tokens,
        )?;
    }
    stage_end(t0, ctx, &MOE_PREFILL_UP_US, &MOE_PREFILL_UP_CALLS);

    // 3. SiLU·gate over [tokens * top_k, expert_inter] flat layout.
    let total_pairs = tokens * top_k;
    let t0 = stage_t0();
    B::silu_mul_batched(
        ctx,
        &scratch.gate_out_stacked,
        &scratch.up_out_stacked,
        &mut scratch.silu_stacked,
        total_pairs,
        inter,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_SILU_US, &MOE_PREFILL_SILU_CALLS);

    // 4. Batched down gemm — src1 is [batch, top_k, expert_inter] from
    //    silu_stacked. ne11 = top_k → each pair reads its own row.
    let t0 = stage_t0();
    if use_indirect_dispatch {
        B::gemm_quant_moe_id_indirect(
            ctx,
            &scratch.silu_stacked,
            down_stacked,
            &scratch.ids_2d,
            &scratch.tpe_buf,
            &mut scratch.down_out_stacked,
            &scratch.down_args_buf,
            top_k, // ne11 = top_k: per-slot
            top_k,
            max_per_expert,
            tokens,
        )?;
    } else {
        B::gemm_quant_moe_id(
            ctx,
            &scratch.silu_stacked,
            down_stacked,
            &scratch.ids_2d,
            &scratch.tpe_buf,
            &mut scratch.down_out_stacked,
            top_k,
            top_k,
            max_per_expert,
            tokens,
        )?;
    }
    stage_end(t0, ctx, &MOE_PREFILL_DOWN_US, &MOE_PREFILL_DOWN_CALLS);

    // 5. Per-batch weighted sum: moe_out[b, h] = Σ_k w[b,k] · down[b,k,h]
    let t0 = stage_t0();
    B::weighted_sum_batched(
        ctx,
        &scratch.down_out_stacked,
        &scratch.weights_2d,
        &mut scratch.moe_out,
        tokens,
        top_k,
        h,
    )?;
    stage_end(t0, ctx, &MOE_PREFILL_WSUM_US, &MOE_PREFILL_WSUM_CALLS);

    Ok(())
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
