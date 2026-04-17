//! ModelRunner — unified model execution over any Backend.
//!
//! Handles embedding, N × layer_forward, final norm, lm_head.
//! Manages per-sequence KV caches.

use std::collections::HashMap;

use super::layer_forward::layer_forward;
use super::{
    Backend, KvCache, LayerScratch, LayerWeights, ModelWeights, RopeConfig, TransformerConfig,
};

/// Pre-allocated scratch buffers sized for max batch/tokens.
struct RunnerBuffers<B: Backend> {
    /// Residual stream: [max_tokens, hidden]
    residual: B::Buffer,
    /// Per-layer scratch (reused across layers)
    scratch: LayerScratch<B>,
    /// Final norm output
    norm_out: B::Buffer,
    /// Logits output: [max_tokens, vocab]
    logits: B::Buffer,
}

/// Precomputed cos/sin tables for RoPE.
struct RopeTable<B: Backend> {
    cos: B::Buffer,
    sin: B::Buffer,
}

/// Unified model runner: `ModelRunner<CpuBackend>`, `ModelRunner<CudaBackend>`, etc.
pub struct ModelRunner<B: Backend> {
    cfg: TransformerConfig,
    weights: ModelWeights<B>,
    buffers: RunnerBuffers<B>,
    rope_table: RopeTable<B>,
    /// Per-sequence KV caches: cache_id → per-layer KV caches.
    kv_caches: HashMap<String, Vec<KvCache<B>>>,
}

impl<B: Backend> ModelRunner<B> {
    /// Create a new runner with pre-loaded weights.
    pub fn new(cfg: TransformerConfig, weights: ModelWeights<B>) -> Self {
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv_dim = q_dim + 2 * kv_dim;

        // Precompute RoPE cos/sin tables
        let rope_table = build_rope_table::<B>(&cfg.rope);

        // Pre-allocate scratch (sized for 1 token; prefill will need resizing)
        let max_tokens = 1;
        let buffers = alloc_buffers::<B>(max_tokens, h, q_dim, kv_dim, qkv_dim, im, cfg.vocab_size);

        ModelRunner {
            cfg,
            weights,
            buffers,
            rope_table,
            kv_caches: HashMap::new(),
        }
    }

    /// Ensure scratch buffers are large enough for `tokens` tokens.
    fn ensure_scratch(&mut self, tokens: usize) {
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let im = self.cfg.intermediate_size;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv_dim = q_dim + 2 * kv_dim;
        let vocab = self.cfg.vocab_size;

        // Check if current scratch is big enough
        let needed = tokens * h;
        if B::to_vec(&self.buffers.residual, 0).len() >= needed {
            return;
        }

        self.buffers = alloc_buffers::<B>(tokens, h, q_dim, kv_dim, qkv_dim, im, vocab);
    }

    /// Ensure KV caches exist for a sequence.
    fn ensure_kv(&mut self, cache_id: &str) {
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

    /// Prefill: process all prompt tokens at once, return logits for last token.
    ///
    /// Returns `[vocab_size]` logits.
    pub fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        self.ensure_scratch(seq_len);
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let mut ctx = B::new_context();

        B::embedding_lookup(
            &mut ctx,
            &self.weights.embed,
            tokens,
            &mut self.buffers.residual,
            h,
        );

        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let kv_caches = self.kv_caches.get_mut(cache_id).unwrap();
        for li in 0..self.cfg.num_layers {
            B::layer_forward_fused(
                &mut ctx,
                &self.cfg,
                &self.weights.layers[li],
                &mut kv_caches[li],
                &mut self.buffers.scratch,
                &mut self.buffers.residual,
                &positions,
                &self.rope_table.cos,
                &self.rope_table.sin,
                seq_len,
            );
        }

        B::rms_norm(
            &mut ctx,
            &self.buffers.residual,
            &self.weights.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.buffers.norm_out,
            seq_len,
            h,
        );

        // Sync before reading results back to CPU
        B::sync(&mut ctx);

        let all_hidden = B::to_vec(&self.buffers.norm_out, seq_len * h);
        let last_hidden = B::from_slice(&all_hidden[(seq_len - 1) * h..seq_len * h]);
        let mut logits_buf = B::alloc(vocab);
        B::gemm(
            &mut ctx,
            &last_hidden,
            &self.weights.lm_head_w,
            &mut logits_buf,
            1,
            vocab,
            h,
        );
        B::sync(&mut ctx);

        B::to_vec(&logits_buf, vocab)
    }

    /// Decode: process one token, return logits.
    ///
    /// `pos` is the position of this token in the sequence (= number of previously generated tokens).
    /// Returns `[vocab_size]` logits.
    pub fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.ensure_kv(cache_id);

        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let mut ctx = B::new_context();

        B::embedding_lookup(
            &mut ctx,
            &self.weights.embed,
            &[token],
            &mut self.buffers.residual,
            h,
        );

        let positions = [pos];

        let kv_caches = self.kv_caches.get_mut(cache_id).unwrap();
        for li in 0..self.cfg.num_layers {
            B::layer_forward_fused(
                &mut ctx,
                &self.cfg,
                &self.weights.layers[li],
                &mut kv_caches[li],
                &mut self.buffers.scratch,
                &mut self.buffers.residual,
                &positions,
                &self.rope_table.cos,
                &self.rope_table.sin,
                1,
            );
        }

        B::rms_norm(
            &mut ctx,
            &self.buffers.residual,
            &self.weights.final_norm_w,
            self.cfg.rms_norm_eps,
            &mut self.buffers.norm_out,
            1,
            h,
        );
        B::gemm(
            &mut ctx,
            &self.buffers.norm_out,
            &self.weights.lm_head_w,
            &mut self.buffers.logits,
            1,
            vocab,
            h,
        );
        B::sync(&mut ctx);

        B::to_vec(&self.buffers.logits, vocab)
    }

    /// Release KV caches for a completed sequence.
    pub fn release(&mut self, cache_id: &str) {
        self.kv_caches.remove(cache_id);
    }

    /// Reset all state (KV caches + scratch).
    pub fn reset(&mut self) {
        self.kv_caches.clear();
    }

    /// Get config reference.
    pub fn config(&self) -> &TransformerConfig {
        &self.cfg
    }
}

// ── RoPE table ───────────────────────────────────────────────────────────

/// Convert CPU weights to Metal weights (f32 Vec → Metal shared buffer).
#[cfg(feature = "metal")]
pub fn convert_weights_to_metal(
    cpu: &ModelWeights<super::cpu::CpuBackend>,
) -> ModelWeights<super::metal::MetalBackend> {
    use super::metal::MetalBackend;
    ModelWeights {
        embed: MetalBackend::from_slice(&cpu.embed),
        layers: cpu
            .layers
            .iter()
            .map(|l| LayerWeights {
                input_ln_w: MetalBackend::from_slice(&l.input_ln_w),
                qkv_proj_w: MetalBackend::from_slice(&l.qkv_proj_w),
                o_proj_w: MetalBackend::from_slice(&l.o_proj_w),
                post_ln_w: MetalBackend::from_slice(&l.post_ln_w),
                gate_up_proj_w: MetalBackend::from_slice(&l.gate_up_proj_w),
                down_proj_w: MetalBackend::from_slice(&l.down_proj_w),
                q_norm_w: l.q_norm_w.as_ref().map(|w| MetalBackend::from_slice(w)),
                k_norm_w: l.k_norm_w.as_ref().map(|w| MetalBackend::from_slice(w)),
            })
            .collect(),
        final_norm_w: MetalBackend::from_slice(&cpu.final_norm_w),
        lm_head_w: MetalBackend::from_slice(&cpu.lm_head_w),
    }
}

fn alloc_buffers<B: Backend>(
    tokens: usize,
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    qkv_dim: usize,
    im: usize,
    vocab: usize,
) -> RunnerBuffers<B> {
    RunnerBuffers {
        residual: B::alloc(tokens * h),
        scratch: LayerScratch {
            norm_out: B::alloc(tokens * h),
            qkv_out: B::alloc(tokens * qkv_dim),
            q_buf: B::alloc(tokens * q_dim),
            k_buf: B::alloc(tokens * kv_dim),
            v_buf: B::alloc(tokens * kv_dim),
            attn_out: B::alloc(tokens * q_dim),
            o_proj_out: B::alloc(tokens * h),
            gate_up_out: B::alloc(tokens * 2 * im),
            silu_out: B::alloc(tokens * im),
            mlp_out: B::alloc(tokens * h),
        },
        norm_out: B::alloc(tokens * h),
        logits: B::alloc(tokens * vocab),
    }
}

fn build_rope_table<B: Backend>(cfg: &RopeConfig) -> RopeTable<B> {
    let hd = cfg.head_dim;
    let half = hd / 2;
    let max_seq = cfg.max_seq_len;

    let mut cos = vec![0.0f32; max_seq * half];
    let mut sin = vec![0.0f32; max_seq * half];

    for pos in 0..max_seq {
        for i in 0..half {
            let freq = 1.0f64 / cfg.theta.powf((2 * i) as f64 / hd as f64);
            let angle = pos as f64 * freq;
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }

    RopeTable {
        cos: B::from_slice(&cos),
        sin: B::from_slice(&sin),
    }
}
