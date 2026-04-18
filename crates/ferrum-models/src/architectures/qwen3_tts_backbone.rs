//! `TalkerBackboneBackend<B>` — trait-object-ready wrapper that routes the
//! Qwen3-TTS Talker transformer stack through `Backend<B>` kernels instead
//! of `ferrum_attention::FusedTransformer`.
//!
//! Why this exists: on Linux with `--features cuda` (no Metal),
//! `ferrum_attention::FusedTransformer`'s CUDA module is a stub and its
//! CPU fallback uses naive fp64 O(n³) matmul — produces wrong codec tokens
//! (YouTube-outro garbage) after 20 layers × ~128 decode steps. Routing
//! through `LlamaFamilyModel<B>` fixes both perf and correctness; CUDA
//! parity against CPU is validated in
//! `crates/ferrum-models/tests/qwen3_tts_backend_smoke.rs`.
//!
//! This wrapper is a narrow drop-in point: the candle-based
//! `Qwen3TTSTalker` keeps its embeddings / projection / codec_head, and
//! swaps only the transformer stack.

use ferrum_kernels::backend::Backend;
use ferrum_quantization::loader::WeightLoader;
use ferrum_quantization::PrefixedLoader;
use ferrum_types::Result;

use crate::architectures::qwen3_tts::TalkerConfig;
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};

/// Object-safe trait so the candle-based talker can hold
/// `Box<dyn TalkerBackboneForward>` without threading the Backend type
/// through its public signature.
pub trait TalkerBackboneForward: Send + Sync {
    /// Run the transformer stack on `input [seq_len, hidden]` and return
    /// **post-norm** hidden states for every position (`[seq_len * hidden]`).
    fn forward(&mut self, input_f32: &[f32], seq_len: usize) -> Vec<f32>;

    /// Forget the KV cache + position counter so a fresh sequence can start.
    fn reset(&mut self);
}

/// Backbone adapter — wraps `LlamaFamilyModel<B>` loaded as a backbone-only
/// (no embed / no lm_head) plus a per-sequence position counter.
pub struct TalkerBackboneBackend<B: Backend> {
    backbone: LlamaFamilyModel<B>,
    cache_id: String,
    pos: usize,
}

impl<B: Backend> TalkerBackboneBackend<B> {
    /// Build from a TTS model-directory loader. Uses `PrefixedLoader`
    /// with `"talker."` so `LlamaFamilyModel::new_backbone_only` picks up
    /// `talker.model.layers.*` and `talker.model.norm.weight`.
    pub fn new(cfg: &TalkerConfig, loader: &dyn WeightLoader<B>) -> Result<Self> {
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
        let prefixed = PrefixedLoader::new(loader, "talker.");
        let backbone = LlamaFamilyModel::<B>::new_backbone_only(backbone_cfg, &prefixed)?;
        Ok(Self {
            backbone,
            cache_id: "tts-talker".to_string(),
            pos: 0,
        })
    }

    /// SubTalker variant — expects weights under `talker.code_predictor.` and
    /// smaller dims (`code_predictor_*`).
    pub fn new_code_predictor(
        cfg: &TalkerConfig,
        loader: &dyn WeightLoader<B>,
    ) -> Result<Self> {
        let cp_h = cfg.code_predictor_hidden_size;
        let backbone_cfg = LlamaFamilyConfig {
            hidden_size: cp_h,
            intermediate_size: cp_h * 3,
            num_heads: cfg.code_predictor_num_heads,
            num_kv_heads: cfg.code_predictor_num_kv_heads,
            // head_dim is explicit in code_predictor_config, NOT hidden/num_heads.
            // Qwen3-TTS: hidden=1024, num_heads=16, head_dim=128 → q_dim=2048,
            // kv_dim=1024 (GQA). The old `cp_h/num_heads=64` undersized scratch
            // by half; cuBLAS GEMM wrote 2*4096 floats into a 2*2048-sized
            // qkv_out, corrupting adjacent CUDA memory and causing NaN cascades
            // in SubTalker layer 2+ on the 2nd predict() invocation.
            head_dim: cfg.code_predictor_head_dim,
            num_layers: cfg.code_predictor_num_layers,
            vocab_size: cfg.code_predictor_vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps as f32,
            rope_theta: cfg.rope_theta,
            has_qk_norm: true,
            sliding_window: 0,
        };
        let prefixed = PrefixedLoader::new(loader, "talker.code_predictor.");
        let backbone = LlamaFamilyModel::<B>::new_backbone_only(backbone_cfg, &prefixed)?;
        Ok(Self {
            backbone,
            cache_id: "tts-subtalker".to_string(),
            pos: 0,
        })
    }
}

impl<B: Backend> TalkerBackboneForward for TalkerBackboneBackend<B> {
    fn forward(&mut self, input_f32: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.backbone.cfg.hidden_size;
        assert_eq!(
            input_f32.len(),
            seq_len * h,
            "TalkerBackboneBackend: input len {} != seq_len * hidden {}",
            input_f32.len(),
            seq_len * h
        );

        // The upstream candle talker calls forward_step for both prefill
        // (multi-token) and decode (single-token). We route to the matching
        // backbone method based on seq_len, but both respect `self.pos` so
        // a multi-stage prefill (role prefix then ICL block then decode)
        // keeps the KV cache and positions aligned.
        tracing::debug!(
            "TalkerBackboneBackend::forward cache={} seq_len={} pos_offset={}",
            self.cache_id,
            seq_len,
            self.pos
        );
        let out = if seq_len == 1 {
            self.backbone
                .decode_post_norm_from_embed(&self.cache_id, input_f32, self.pos as u32)
        } else {
            self.backbone
                .prefill_all_post_norm(&self.cache_id, input_f32, seq_len, self.pos)
        };
        self.pos += seq_len;
        out
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.backbone.kv_caches.remove(&self.cache_id);
    }
}
