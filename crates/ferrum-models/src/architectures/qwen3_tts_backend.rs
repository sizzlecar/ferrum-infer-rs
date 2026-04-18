//! Qwen3-TTS Talker using the `Backend<B>` trait — the Model-as-Code port
//! that replaces `ferrum_attention::FusedTransformer` for Phase F.
//!
//! Why this exists: the old `qwen3_tts.rs` uses `FusedTransformer` which
//! only has Metal + CPU backends. Its CUDA module is a stub that falls back
//! to CPU, and on Linux that CPU path uses naive fp64 O(n³) matmul. Through
//! 20 decoder layers + ~128 decode steps the accumulated rounding diverges
//! enough from training numerics to select wrong codec tokens, producing
//! YouTube-outro garbage instead of the target text.
//!
//! This file reuses `LlamaFamilyModel<B>` (via `new_backbone_only`) for the
//! transformer stack — inheriting CUDA/Metal/CPU kernels, KV cache pooling,
//! and batched decode — and adds TTS-specific wiring on top: dual text /
//! codec embeddings, a text projection MLP, and the codec output head.

use ferrum_kernels::backend::Backend;
use ferrum_quantization::loader::WeightLoader;
use ferrum_quantization::traits::Linear;
use ferrum_quantization::PrefixedLoader;
use ferrum_types::Result;
use std::collections::HashMap;

use crate::architectures::qwen3_tts::TalkerConfig;
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};

/// Qwen3-TTS Talker, Model-as-Code implementation over `Backend<B>`.
///
/// Ownership model: owns a backbone `LlamaFamilyModel<B>` configured for
/// 20-layer Qwen3 shape (hidden=1024, heads=16/2, head_dim=64) plus the
/// TTS-specific head/tail layers (text embedding + projection, codec
/// embedding, codec head).
///
/// Forward flow per step:
///   text_ids → text_embed → silu(fc1) → fc2 → mixed_embeds
///   codec_ids → codec_embed → mixed_embeds
///   mixed_embeds → backbone.{prefill,decode}_from_embed → hidden
///   hidden → final_norm (via backbone.final_norm_w) → codec_head → logits
pub struct Qwen3TtsTalker<B: Backend> {
    pub cfg: TalkerConfig,

    /// Transformer backbone — only `layers`, `final_norm_w`, `scratch`,
    /// `kv_caches`, and `rope` are used. `embed` and `lm_head` are `None`
    /// because TTS embeds externally and applies `codec_head` separately.
    pub backbone: LlamaFamilyModel<B>,

    /// Text token embedding table: `[text_vocab * text_hidden]`.
    /// Qwen3-TTS: text_vocab=151936, text_hidden=2048.
    pub text_embedding: B::Buffer,

    /// Text projection: `text_hidden -> text_hidden` (linear_fc1) then SiLU
    /// then `text_hidden -> hidden` (linear_fc2). Both have bias.
    pub text_proj_fc1: Box<dyn Linear<B>>,
    pub text_proj_fc2: Box<dyn Linear<B>>,

    /// Codec token embedding table: `[vocab * hidden]`. Qwen3-TTS has
    /// vocab=3072, hidden=1024.
    pub codec_embedding: B::Buffer,

    /// Output head: `hidden -> vocab` (no bias on codec_head).
    pub codec_head: Box<dyn Linear<B>>,

    /// Per-sequence position tracking — each call to `prefill` / `decode`
    /// advances the cached position. The backbone manages its own KV cache
    /// under the same cache_id.
    positions: HashMap<String, u32>,
}

impl<B: Backend> Qwen3TtsTalker<B> {
    /// Build a Qwen3-TTS Talker from weights. Uses `PrefixedLoader` with
    /// `"talker."` prefix internally so the backbone can reuse its standard
    /// `model.layers.{i}.*` / `model.norm.weight` lookups.
    pub fn new(cfg: TalkerConfig, loader: &dyn WeightLoader<B>) -> Result<Self> {
        let backbone_cfg = LlamaFamilyConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_hidden_layers,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            rope_theta: cfg.rope_theta,
            has_qk_norm: true,
            sliding_window: 0,
        };

        // Backbone transformer — loads model.layers.{i}.* and model.norm
        // under the `talker.` prefix.
        let talker_loader = PrefixedLoader::new(loader, "talker.");
        let backbone = LlamaFamilyModel::<B>::new_backbone_only(backbone_cfg, &talker_loader)?;

        // Dual embeddings.
        let text_embedding = loader.load_tensor("talker.model.text_embedding.weight")?;
        let codec_embedding = loader.load_tensor("talker.model.codec_embedding.weight")?;

        // Text projection: 2048 → 2048 → 1024, both with bias.
        let text_proj_fc1 = loader.load_linear("talker.text_projection.linear_fc1")?;
        let text_proj_fc2 = loader.load_linear("talker.text_projection.linear_fc2")?;

        // Output head: 1024 → 3072, no bias (Qwen3-TTS stores as linear_no_bias).
        let codec_head = loader.load_linear("talker.codec_head")?;

        Ok(Self {
            cfg,
            backbone,
            text_embedding,
            text_proj_fc1,
            text_proj_fc2,
            codec_embedding,
            codec_head,
            positions: HashMap::new(),
        })
    }

    /// Reset per-sequence state so the next call starts from position 0.
    pub fn reset(&mut self, cache_id: &str) {
        self.positions.remove(cache_id);
        self.backbone.kv_caches.remove(cache_id);
    }

    /// Clear all sessions.
    pub fn reset_all(&mut self) {
        self.positions.clear();
        self.backbone.kv_caches.clear();
    }

    /// Embed a text token as `[hidden]` f32 — goes through text_embedding →
    /// silu(fc1) → fc2. Runs once per text token at prefill time; not in
    /// the decode hot loop, so the SiLU CPU roundtrip is acceptable.
    fn embed_text_token(&mut self, token: u32) -> Vec<f32> {
        let text_hidden = self.cfg.text_hidden_size;
        let hidden = self.cfg.hidden_size;
        let mut ctx = B::new_context();

        // text_embedding[token] → [text_hidden]
        let mut embed_out = B::alloc(text_hidden);
        B::embedding_lookup(
            &mut ctx,
            &self.text_embedding,
            &[token],
            &mut embed_out,
            text_hidden,
        );

        // fc1: text_hidden → text_hidden (+ bias)
        let mut fc1_out = B::alloc(text_hidden);
        self.text_proj_fc1
            .forward(&mut ctx, &embed_out, &mut fc1_out, 1);

        // SiLU(x) = x * sigmoid(x). Done on CPU — prefill-only path.
        B::sync(&mut ctx);
        let fc1_host = B::to_vec(&fc1_out, text_hidden);
        let silu_host: Vec<f32> = fc1_host
            .iter()
            .map(|&x| x * (1.0f32 / (1.0f32 + (-x).exp())))
            .collect();
        let silu_dev = B::from_slice(&silu_host);

        // fc2: text_hidden → hidden (+ bias)
        let mut fc2_out = B::alloc(hidden);
        self.text_proj_fc2
            .forward(&mut ctx, &silu_dev, &mut fc2_out, 1);
        B::sync(&mut ctx);

        B::to_vec(&fc2_out, hidden)
    }

    /// Embed a codec token as `[hidden]` f32.
    fn embed_codec_token(&mut self, token: u32) -> Vec<f32> {
        let hidden = self.cfg.hidden_size;
        let mut ctx = B::new_context();
        let mut out = B::alloc(hidden);
        B::embedding_lookup(
            &mut ctx,
            &self.codec_embedding,
            &[token],
            &mut out,
            hidden,
        );
        B::sync(&mut ctx);
        B::to_vec(&out, hidden)
    }

    /// Prefill with a mixed text / codec token sequence. Each input is
    /// `(token_id, is_text)` — text tokens route through text_embedding +
    /// projection; codec tokens route through codec_embedding.
    ///
    /// Returns `[vocab_size]` logits for the last position, ready for
    /// sampling the first codec token.
    pub fn prefill(
        &mut self,
        cache_id: &str,
        tokens: &[(u32, bool)],
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let seq_len = tokens.len();

        // Build [seq_len * hidden] mixed embedding on host (embeds are
        // tiny-per-token; this stays out of the hot decode loop).
        let mut mixed = Vec::with_capacity(seq_len * h);
        for (tok, is_text) in tokens {
            let emb = if *is_text {
                self.embed_text_token(*tok)
            } else {
                self.embed_codec_token(*tok)
            };
            mixed.extend(emb);
        }

        // Backbone transformer + last-pos hidden extract (no final norm,
        // no output head — we apply those ourselves below).
        let pre_norm_hidden = self.backbone.prefill_from_embeds(cache_id, &mixed, seq_len);

        self.positions.insert(cache_id.to_string(), seq_len as u32);

        // Final norm + codec head.
        self.apply_head(&pre_norm_hidden)
    }

    /// Decode one codec token — advances the sequence by 1.
    pub fn decode_codec(&mut self, cache_id: &str, token: u32) -> Vec<f32> {
        let pos = *self.positions.get(cache_id).unwrap_or(&0);
        let embed = self.embed_codec_token(token);
        let pre_norm = self.backbone.decode_from_embed(cache_id, &embed, pos);
        self.positions.insert(cache_id.to_string(), pos + 1);
        self.apply_head(&pre_norm)
    }

    /// Apply final_norm + codec_head on a `[hidden]` f32 vector, return
    /// `[vocab_size]` logits.
    fn apply_head(&mut self, hidden_f32: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        debug_assert_eq!(hidden_f32.len(), h);

        let mut ctx = B::new_context();
        let hidden_buf = B::from_slice(hidden_f32);
        let mut normed = B::alloc(h);
        B::rms_norm(
            &mut ctx,
            &hidden_buf,
            &self.backbone.final_norm_w,
            self.cfg.rms_norm_eps as f32,
            &mut normed,
            1,
            h,
        );

        let mut logits = B::alloc(vocab);
        self.codec_head.forward(&mut ctx, &normed, &mut logits, 1);
        B::sync(&mut ctx);
        B::to_vec(&logits, vocab)
    }

    /// Expose hidden state for the last position (after final_norm, before
    /// codec_head). SubTalker needs this to run its own transformer.
    pub fn last_hidden_normed(&mut self, cache_id: &str) -> Vec<f32> {
        // The backbone's scratch.last_hidden holds the last prefill/decode's
        // pre-norm hidden. Re-apply final_norm to get the post-norm vector.
        let h = self.cfg.hidden_size;
        let mut ctx = B::new_context();
        let mut normed = B::alloc(h);
        B::rms_norm(
            &mut ctx,
            &self.backbone.scratch.last_hidden,
            &self.backbone.final_norm_w,
            self.cfg.rms_norm_eps as f32,
            &mut normed,
            1,
            h,
        );
        B::sync(&mut ctx);
        let _ = cache_id; // cache_id not used yet; reserved for per-session variants
        B::to_vec(&normed, h)
    }
}
