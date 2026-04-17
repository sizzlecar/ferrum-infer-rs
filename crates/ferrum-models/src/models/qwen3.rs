//! Qwen3 model as explicit code (Phase B2: structure + weight loading).
//!
//! Phase B3 will add `forward_layer`, B4 adds `prefill`/`decode` + KV cache,
//! B5 adds `impl DecoderOnlyLLM`.
//!
//! Architecture notes (Qwen3 0.6B / 1.7B / 4B):
//!   - GQA (num_heads > num_kv_heads), QK-norm (RMS per head) unique to Qwen3
//!   - Fused QKV projection in safetensors (`qkv_proj`)
//!   - Fused gate+up MLP (`gate_up_proj`)
//!   - RoPE with large theta (typically 1e6 for 32k context)

use std::collections::HashMap;

use ferrum_kernels::backend::{Backend, KvCache};
use ferrum_quantization::{Linear, WeightLoader};
use ferrum_types::Result;

use crate::common::LlmRuntimeConfig;

/// Full Qwen3 architecture config (everything the model code needs, not just
/// the engine-facing subset in `LlmRuntimeConfig`).
#[derive(Clone, Debug)]
pub struct Qwen3Config {
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
}

impl Qwen3Config {
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
}

/// Per-layer weights. `Box<dyn Linear<B>>` means each projection can be
/// Dense / GPTQ / AWQ / GGUF without the surrounding code caring.
pub struct Qwen3Layer<B: Backend> {
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
pub struct Qwen3Scratch<B: Backend> {
    pub residual: B::Buffer,
    pub norm_out: B::Buffer,
    pub qkv_out: B::Buffer,
    pub q_buf: B::Buffer,
    pub k_buf: B::Buffer,
    pub v_buf: B::Buffer,
    pub attn_head_major: B::Buffer, // [nh, tokens, hd] flash_attn output
    pub attn_flat: B::Buffer,       // [tokens, nh, hd]
    pub k_head_major: B::Buffer,    // staging for kv_cache_append
    pub v_head_major: B::Buffer,
    pub o_proj_out: B::Buffer,
    pub gate_up_out: B::Buffer,
    pub silu_out: B::Buffer,
    pub mlp_out: B::Buffer,
    pub final_norm_out: B::Buffer,
    pub logits: B::Buffer,
    /// The max tokens-per-step this scratch has been sized for.
    pub max_tokens: usize,
}

impl<B: Backend> Qwen3Scratch<B> {
    fn alloc(cfg: &Qwen3Config, max_tokens: usize) -> Self {
        let h = cfg.hidden_size;
        let im = cfg.intermediate_size;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let t = max_tokens;
        Self {
            residual: B::alloc(t * h),
            norm_out: B::alloc(t * h),
            qkv_out: B::alloc(t * qkv_dim),
            q_buf: B::alloc(t * q_dim),
            k_buf: B::alloc(t * kv_dim),
            v_buf: B::alloc(t * kv_dim),
            attn_head_major: B::alloc(cfg.num_heads * t * cfg.head_dim),
            attn_flat: B::alloc(t * q_dim),
            k_head_major: B::alloc(cfg.num_kv_heads * t * cfg.head_dim),
            v_head_major: B::alloc(cfg.num_kv_heads * t * cfg.head_dim),
            o_proj_out: B::alloc(t * h),
            gate_up_out: B::alloc(t * 2 * im),
            silu_out: B::alloc(t * im),
            mlp_out: B::alloc(t * h),
            final_norm_out: B::alloc(t * h),
            logits: B::alloc(t * cfg.vocab_size),
            max_tokens: t,
        }
    }
}

/// Qwen3 model — decoder-only LLM, one per (backend, weights) combination.
///
/// Holds all parameters, scratch space, RoPE cache, and per-sequence KV caches.
pub struct Qwen3Model<B: Backend> {
    pub cfg: Qwen3Config,
    pub runtime_cfg: LlmRuntimeConfig,

    pub embed: B::Buffer,
    pub layers: Vec<Qwen3Layer<B>>,
    pub final_norm_w: B::Buffer,
    pub lm_head: Box<dyn Linear<B>>,

    pub rope: RopeCache<B>,
    pub scratch: Qwen3Scratch<B>,

    /// Per-sequence KV caches, one `Vec<KvCache<B>>` of length `num_layers`.
    pub kv_caches: HashMap<String, Vec<KvCache<B>>>,
}

impl<B: Backend> Qwen3Model<B> {
    /// Build a Qwen3 model from weights provided by the loader.
    ///
    /// The loader decides per-projection whether to instantiate DenseLinear,
    /// GptqLinear, etc. — this code doesn't care.
    pub fn new(cfg: Qwen3Config, loader: &dyn WeightLoader<B>) -> Result<Self> {
        let rope = build_rope_cache::<B>(&cfg);
        let scratch = Qwen3Scratch::alloc(&cfg, 1); // decode-sized; prefill resizes

        // Embedding: plain tensor (no projection math, just lookup).
        let embed = loader.load_tensor("model.embed_tokens.weight")?;

        // Per-layer weights.
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for li in 0..cfg.num_layers {
            let prefix = format!("model.layers.{li}");
            let input_ln_w = loader.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let qkv_proj = loader.load_linear(&format!("{prefix}.self_attn.qkv_proj"))?;
            let o_proj = loader.load_linear(&format!("{prefix}.self_attn.o_proj"))?;
            let post_ln_w = loader
                .load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
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

            layers.push(Qwen3Layer {
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
        let lm_head = if loader.has_tensor("lm_head.weight") {
            loader.load_linear("lm_head")?
        } else {
            // Tied embedding: use a DenseLinear over the embedding buffer.
            // TODO(B3): we need a way to share the embed buffer with a Linear.
            // For now, require dedicated lm_head; tied-embed support lands with
            // the first model that actually requires it.
            return Err(ferrum_types::FerrumError::model(
                "lm_head.weight not found and tied-embedding Linear not yet supported",
            ));
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
        })
    }

    /// Grow scratch buffers if `tokens` exceeds the current sizing.
    pub(crate) fn ensure_scratch(&mut self, tokens: usize) {
        if self.scratch.max_tokens < tokens {
            self.scratch = Qwen3Scratch::alloc(&self.cfg, tokens);
        }
    }

    /// Ensure per-layer KV caches exist for `cache_id`. Allocation itself is
    /// deferred to the first `layer_forward` (backends pre-allocate
    /// `max_seq_len` slots on first append).
    pub(crate) fn ensure_kv(&mut self, cache_id: &str) {
        if self.kv_caches.contains_key(cache_id) {
            return;
        }
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let caches = (0..self.cfg.num_layers)
            .map(|_| KvCache {
                k: B::alloc(0),
                v: B::alloc(0),
                len: 0,
                capacity: 0,
                num_kv_heads: nkv,
                head_dim: hd,
            })
            .collect();
        self.kv_caches.insert(cache_id.to_string(), caches);
    }
}

fn build_rope_cache<B: Backend>(cfg: &Qwen3Config) -> RopeCache<B> {
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
