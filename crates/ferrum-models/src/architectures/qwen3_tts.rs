//! Qwen3-TTS Talker model — generates speech codec tokens from text.
//!
//! Architecture: Qwen3 backbone (20 layers, 1024 hidden, 16 heads, 2 KV heads)
//! with text projection (2048→1024) and SubTalker code predictor (31 codebooks).
//!
//! candle loads weights from safetensors; forward pass is ours for Metal/CPU.

use candle_core::{DType, Device as CandleDevice, IndexOp, Module, Tensor, D};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use tracing::info;

use super::repeat_kv;

// ── Config ──────────────────────────────────────────────────────────────

/// Talker LM config (from config.json talker_config section).
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    pub vocab_size: usize,        // 3072 (codec token vocabulary)
    pub hidden_size: usize,       // 1024
    pub intermediate_size: usize, // 2816
    pub num_hidden_layers: usize, // 20
    pub num_attention_heads: usize, // 16
    pub num_key_value_heads: usize, // 2
    pub head_dim: usize,          // 64
    pub max_position_embeddings: usize, // 32768
    pub rope_theta: f64,          // 1000000.0
    pub rms_norm_eps: f64,        // 1e-6
    pub text_vocab_size: usize,   // 151936
    pub text_hidden_size: usize,  // 2048
    pub num_code_groups: usize,   // 32
    pub codec_eos_token_id: u32,  // 4198
    pub codec_pad_id: u32,        // 4196
    pub codec_bos_id: u32,        // 4197
    pub codec_think_id: u32,      // 4202
    pub codec_nothink_id: u32,    // 4203
    pub codec_think_bos_id: u32,  // 4204
    pub codec_think_eos_id: u32,  // 4205
    pub tts_bos_token_id: u32,    // 151672
    pub tts_eos_token_id: u32,    // 151673
    pub tts_pad_token_id: u32,    // 151671
    pub code_predictor_vocab_size: usize, // 2048
    pub code_predictor_hidden_size: usize, // 1024
    pub code_predictor_num_layers: usize,  // typically 4
    pub code_predictor_num_heads: usize,   // 16
    pub code_predictor_num_kv_heads: usize, // 2
    /// Speaker ID mapping (speaker_name → token_id)
    pub spk_id: HashMap<String, Vec<u32>>,
    /// Language ID mapping (language_name → token_id)
    pub codec_language_id: HashMap<String, u32>,
}

impl TalkerConfig {
    /// Parse from the config.json's talker_config section.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let tc = v
            .get("talker_config")
            .ok_or_else(|| FerrumError::model("missing talker_config"))?;

        let get_usize = |key: &str, default: usize| -> usize {
            tc.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            tc.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let get_u32 = |key: &str, default: u32| -> u32 {
            tc.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default)
        };

        let mut spk_id = HashMap::new();
        if let Some(obj) = tc.get("spk_id").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Some(arr) = v.as_array() {
                    let ids: Vec<u32> = arr.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
                    spk_id.insert(k.clone(), ids);
                }
            }
        }

        let mut codec_language_id = HashMap::new();
        if let Some(obj) = tc.get("codec_language_id").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Some(id) = v.as_u64() {
                    codec_language_id.insert(k.clone(), id as u32);
                }
            }
        }

        Ok(Self {
            vocab_size: get_usize("vocab_size", 3072),
            hidden_size: get_usize("hidden_size", 1024),
            intermediate_size: get_usize("intermediate_size", 3072),
            num_hidden_layers: get_usize("num_hidden_layers", 28),
            num_attention_heads: get_usize("num_attention_heads", 16),
            num_key_value_heads: get_usize("num_key_value_heads", 8),
            head_dim: get_usize("head_dim", 128),
            max_position_embeddings: get_usize("max_position_embeddings", 32768),
            rope_theta: get_f64("rope_theta", 1000000.0),
            rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
            text_vocab_size: get_usize("text_vocab_size", 151936),
            text_hidden_size: get_usize("text_hidden_size", 2048),
            num_code_groups: get_usize("num_code_groups", 16),
            codec_eos_token_id: get_u32("codec_eos_token_id", 2150),
            codec_pad_id: get_u32("codec_pad_id", 2148),
            codec_bos_id: get_u32("codec_bos_id", 2149),
            codec_think_id: get_u32("codec_think_id", 2154),
            codec_nothink_id: get_u32("codec_nothink_id", 2155),
            codec_think_bos_id: get_u32("codec_think_bos_id", 2156),
            codec_think_eos_id: get_u32("codec_think_eos_id", 2157),
            tts_bos_token_id: v.get("tts_bos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(151672),
            tts_eos_token_id: v.get("tts_eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(151673),
            tts_pad_token_id: v.get("tts_pad_token_id").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(151671),
            code_predictor_vocab_size: {
                let cp = tc.get("code_predictor_config");
                cp.and_then(|c| c.get("vocab_size")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(2048)
            },
            code_predictor_hidden_size: {
                let cp = tc.get("code_predictor_config");
                cp.and_then(|c| c.get("hidden_size")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(1024)
            },
            code_predictor_num_layers: {
                let cp = tc.get("code_predictor_config");
                cp.and_then(|c| c.get("num_hidden_layers")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(5)
            },
            code_predictor_num_heads: {
                let cp = tc.get("code_predictor_config");
                cp.and_then(|c| c.get("num_attention_heads")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(16)
            },
            code_predictor_num_kv_heads: {
                let cp = tc.get("code_predictor_config");
                cp.and_then(|c| c.get("num_key_value_heads")).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(8)
            },
            spk_id,
            codec_language_id,
        })
    }
}

// ── Rotary Embedding ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &TalkerConfig, dev: &CandleDevice) -> candle_core::Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── RMSNorm (manual ops for Metal compatibility) ────────────────────────

#[derive(Debug, Clone)]
struct ManualRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ManualRmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for ManualRmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(x.dtype())?;
        normed.broadcast_mul(&self.weight)
    }
}

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<ManualRmsNorm> {
    let w = vb.get(size, "weight")?;
    Ok(ManualRmsNorm::new(w, eps))
}

// ── MLP ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl MLP {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            intermediate_size: i,
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = x.apply(&self.gate_proj)?.silu()?;
        let up = x.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ── Self-Attention with KV Cache ────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: ManualRmsNorm,
    k_norm: ManualRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        cfg: &TalkerConfig,
        rotary: RotaryEmbedding,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let q_proj = candle_nn::linear_no_bias(h, nh * hd, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(nh * hd, h, vb.pp("o_proj"))?;
        let q_norm = rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            rotary,
            kv_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, pos_offset: usize) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let hd = self.head_dim;

        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        // Reshape: [b, seq, heads*hd] → [b, heads, seq, hd]
        let q = q
            .reshape((b, seq_len, self.num_heads, hd))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, hd))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, hd))?
            .transpose(1, 2)?;

        // QK norm
        let q = q.apply(&self.q_norm)?;
        let k = k.apply(&self.k_norm)?;

        // RoPE
        let (q, k) = self.rotary.apply(&q, &k, pos_offset)?;

        // KV cache
        let (k, v) = if let Some((prev_k, prev_v)) = &self.kv_cache {
            (
                Tensor::cat(&[prev_k, &k], 2)?,
                Tensor::cat(&[prev_v, &v], 2)?,
            )
        } else {
            (k, v)
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat KV heads
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // Scaled dot-product attention
        let scale = (hd as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        // Causal mask: only needed for seq_len > 1 (prefill)
        let attn = if seq_len > 1 {
            let kv_len = pos_offset + seq_len;
            let mask_data: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..kv_len)
                        .map(move |j| if j <= pos_offset + i { 0f32 } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, kv_len), x.device())?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };
        // Manual softmax for Metal compatibility
        let attn = {
            let max = attn.max_keepdim(D::Minus1)?;
            let shifted = attn.broadcast_sub(&max)?;
            let exp = shifted.exp()?;
            let sum = exp.sum_keepdim(D::Minus1)?;
            exp.broadcast_div(&sum)?
        };
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        out.apply(&self.o_proj)
    }

    fn reset_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ── Transformer Layer ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TransformerLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: ManualRmsNorm,
    post_attention_layernorm: ManualRmsNorm,
}

impl TransformerLayer {
    fn new(
        cfg: &TalkerConfig,
        rotary: RotaryEmbedding,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, pos_offset: usize) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let x = x.apply(&self.input_layernorm)?;
        let x = self.self_attn.forward(&x, pos_offset)?;
        let x = (residual + x)?;
        let residual = x.clone();
        let x = x.apply(&self.post_attention_layernorm)?;
        let x = x.apply(&self.mlp)?;
        residual + x
    }

    fn reset_cache(&mut self) {
        self.self_attn.reset_cache();
    }
}

// ── Text Projection (ResizeMLP: text_hidden → hidden) ───────────────────

#[derive(Debug, Clone)]
struct TextProjection {
    linear1: Linear,
    linear2: Linear,
}

impl TextProjection {
    fn new(text_hidden: usize, hidden: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        // Keys: talker.text_projection.linear_fc1.weight/bias, linear_fc2.weight/bias
        let linear1 = candle_nn::linear(text_hidden, text_hidden, vb.pp("linear_fc1"))?;
        let linear2 = candle_nn::linear(text_hidden, hidden, vb.pp("linear_fc2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.apply(&self.linear1)?.silu()?.apply(&self.linear2)
    }
}

// ── Talker Model (main LM) ─────────────────────────────────────────────

/// Qwen3-TTS Talker: text → speech codec tokens.
pub struct Qwen3TTSTalker {
    text_embedding: Embedding,
    text_projection: TextProjection,
    codec_embedding: Embedding,
    layers: Vec<TransformerLayer>,
    norm: ManualRmsNorm,
    codec_head: Linear,
    config: TalkerConfig,
    device: CandleDevice,
    tokens_generated: usize,
}

impl Qwen3TTSTalker {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder, device: CandleDevice) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = vb.pp("talker").pp("model");

        let text_embedding = candle_nn::embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            model_vb.pp("text_embedding"),
        )
        .map_err(|e| FerrumError::model(format!("text_embedding: {e}")))?;

        let text_projection = TextProjection::new(
            cfg.text_hidden_size,
            cfg.hidden_size,
            vb.pp("talker").pp("text_projection"),
        )
        .map_err(|e| FerrumError::model(format!("text_projection: {e}")))?;

        let codec_embedding = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            model_vb.pp("codec_embedding"),
        )
        .map_err(|e| FerrumError::model(format!("codec_embedding: {e}")))?;

        let rotary = RotaryEmbedding::new(dtype, cfg, &device)
            .map_err(|e| FerrumError::model(format!("rotary: {e}")))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = TransformerLayer::new(cfg, rotary.clone(), model_vb.pp(format!("layers.{i}")))
                .map_err(|e| FerrumError::model(format!("layer {i}: {e}")))?;
            layers.push(layer);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model_vb.pp("norm"))
            .map_err(|e| FerrumError::model(format!("norm: {e}")))?;

        let codec_head = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("talker").pp("codec_head"),
        )
        .map_err(|e| FerrumError::model(format!("codec_head: {e}")))?;

        info!(
            "Qwen3TTSTalker loaded: hidden={}, layers={}, heads={}/{}, vocab={}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.vocab_size,
        );

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            config: cfg.clone(),
            device,
            tokens_generated: 0,
        })
    }

    /// Embed text token IDs through text_embedding + text_projection.
    pub fn embed_text(&self, text_ids: &Tensor) -> Result<Tensor> {
        let embeds = text_ids
            .apply(&self.text_embedding)
            .map_err(|e| FerrumError::model(format!("text_embed: {e}")))?;
        self.text_projection
            .forward(&embeds)
            .map_err(|e| FerrumError::model(format!("text_proj: {e}")))
    }

    /// Embed codec token IDs through codec_embedding.
    pub fn embed_codec(&self, codec_ids: &Tensor) -> Result<Tensor> {
        codec_ids
            .apply(&self.codec_embedding)
            .map_err(|e| FerrumError::model(format!("codec_embed: {e}")))
    }

    /// Forward one step through transformer layers. Returns hidden states.
    pub fn forward_step(&mut self, input_embeds: &Tensor) -> Result<Tensor> {
        let pos_offset = self.tokens_generated;
        let seq_len = input_embeds
            .dim(1)
            .map_err(|e| FerrumError::model(format!("dim: {e}")))?;

        let mut hidden = input_embeds.clone();
        for layer in &mut self.layers {
            hidden = layer
                .forward(&hidden, pos_offset)
                .map_err(|e| FerrumError::model(format!("layer forward: {e}")))?;
        }
        hidden = hidden
            .apply(&self.norm)
            .map_err(|e| FerrumError::model(format!("norm: {e}")))?;

        self.tokens_generated += seq_len;
        Ok(hidden)
    }

    /// Get logits from hidden states.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden
            .apply(&self.codec_head)
            .map_err(|e| FerrumError::model(format!("codec_head: {e}")))
    }

    pub fn reset(&mut self) {
        self.tokens_generated = 0;
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }

    pub fn config(&self) -> &TalkerConfig {
        &self.config
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }
}

// ── SubTalker (Code Predictor) ──────────────────────────────────────────
//
// Predicts codec tokens 1..num_code_groups-1 given the talker hidden state
// and the first codec token embedding. 5-layer transformer with per-codebook
// lm_head and embedding.

pub struct SubTalker {
    layers: Vec<TransformerLayer>,
    norm: ManualRmsNorm,
    /// Per-codebook embeddings: codec_embedding[i] maps token → hidden for codebook i+1
    pub codec_embeddings: Vec<Embedding>,
    /// Per-codebook prediction heads: lm_head[i] maps hidden → logits for codebook i+1
    lm_heads: Vec<Linear>,
    /// Projection from talker hidden to subtalker hidden (if sizes differ)
    projection: Option<Linear>,
    num_code_groups: usize,
    tokens_generated: usize,
}

impl SubTalker {
    pub fn load(cfg: &TalkerConfig, vb: VarBuilder, device: CandleDevice) -> Result<Self> {
        let dtype = vb.dtype();
        let cp_vb = vb.pp("talker").pp("code_predictor");
        let model_vb = cp_vb.pp("model");

        let cp_cfg = TalkerConfig {
            hidden_size: cfg.code_predictor_hidden_size,
            intermediate_size: cfg.code_predictor_hidden_size * 3, // ~3072
            num_hidden_layers: cfg.code_predictor_num_layers,
            num_attention_heads: cfg.code_predictor_num_heads,
            num_key_value_heads: cfg.code_predictor_num_kv_heads,
            ..cfg.clone()
        };

        let rotary = RotaryEmbedding::new(dtype, &cp_cfg, &device)
            .map_err(|e| FerrumError::model(format!("subtalker rotary: {e}")))?;

        let mut layers = Vec::new();
        for i in 0..cfg.code_predictor_num_layers {
            layers.push(
                TransformerLayer::new(&cp_cfg, rotary.clone(), model_vb.pp(format!("layers.{i}")))
                    .map_err(|e| FerrumError::model(format!("subtalker layer {i}: {e}")))?,
            );
        }

        let norm = rms_norm(cfg.code_predictor_hidden_size, cfg.rms_norm_eps, model_vb.pp("norm"))
            .map_err(|e| FerrumError::model(format!("subtalker norm: {e}")))?;

        // Per-codebook embeddings (num_code_groups - 1)
        let n_extra = cfg.num_code_groups - 1;
        let mut codec_embeddings = Vec::new();
        for i in 0..n_extra {
            codec_embeddings.push(
                candle_nn::embedding(
                    cfg.code_predictor_vocab_size,
                    cfg.hidden_size, // embedding dim = talker hidden size
                    model_vb.pp(format!("codec_embedding.{i}")),
                )
                .map_err(|e| FerrumError::model(format!("subtalker codec_embedding.{i}: {e}")))?,
            );
        }

        // Per-codebook lm_heads
        let mut lm_heads = Vec::new();
        for i in 0..n_extra {
            lm_heads.push(
                candle_nn::linear_no_bias(
                    cfg.code_predictor_hidden_size,
                    cfg.code_predictor_vocab_size,
                    cp_vb.pp(format!("lm_head.{i}")),
                )
                .map_err(|e| FerrumError::model(format!("subtalker lm_head.{i}: {e}")))?,
            );
        }

        // Projection if hidden sizes differ
        let projection = if cfg.hidden_size != cfg.code_predictor_hidden_size {
            Some(
                candle_nn::linear(
                    cfg.hidden_size,
                    cfg.code_predictor_hidden_size,
                    cp_vb.pp("small_to_mtp_projection"),
                )
                .map_err(|e| FerrumError::model(format!("subtalker projection: {e}")))?,
            )
        } else {
            None
        };

        info!(
            "SubTalker loaded: layers={}, heads={}/{}, codebooks={}",
            cfg.code_predictor_num_layers,
            cfg.code_predictor_num_heads,
            cfg.code_predictor_num_kv_heads,
            n_extra,
        );

        Ok(Self {
            layers,
            norm,
            codec_embeddings,
            lm_heads,
            projection,
            num_code_groups: cfg.num_code_groups,
            tokens_generated: 0,
        })
    }

    /// Predict codec tokens 1..num_code_groups-1 given talker hidden state and first codec token.
    ///
    /// - `talker_hidden`: [1, 1, hidden_size] — last hidden from main talker
    /// - `first_codec_embed`: [1, 1, hidden_size] — embedding of first codec token (from main talker's codec_embedding)
    ///
    /// Returns Vec of (num_code_groups - 1) token IDs.
    pub fn predict(
        &mut self,
        talker_hidden: &Tensor,
        first_codec_embed: &Tensor,
        temperature: f32,
        top_k: usize,
    ) -> Result<Vec<u32>> {
        self.reset();

        // Input: [talker_hidden, first_codec_embed] → [1, 2, hidden]
        let input = Tensor::cat(&[talker_hidden, first_codec_embed], 1)
            .map_err(|e| FerrumError::model(format!("subtalker cat: {e}")))?;

        // Project if needed
        let input = if let Some(proj) = &self.projection {
            input
                .apply(proj)
                .map_err(|e| FerrumError::model(format!("subtalker proj: {e}")))?
        } else {
            input
        };

        // Prefill
        let mut hidden = self.forward_layers(&input)?;

        let n_extra = self.num_code_groups - 1;
        let mut predicted_tokens = Vec::with_capacity(n_extra);

        for i in 0..n_extra {
            // Get logits from last position using lm_head[i]
            let last = hidden
                .narrow(1, hidden.dim(1).unwrap() - 1, 1)
                .map_err(|e| FerrumError::model(format!("narrow: {e}")))?;
            let logits = last
                .apply(&self.lm_heads[i])
                .map_err(|e| FerrumError::model(format!("lm_head.{i}: {e}")))?;

            // Sample
            let logits_vec = logits
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| FerrumError::model(format!("logits vec: {e}")))?;

            let token = crate::executor::tts_executor::sample_token(&logits_vec, temperature, top_k, 1.0);
            predicted_tokens.push(token);

            // If not last, embed and forward for next step
            if i < n_extra - 1 {
                let token_tensor = Tensor::new(&[token], hidden.device())
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?;
                let embed = token_tensor
                    .apply(&self.codec_embeddings[i])
                    .map_err(|e| FerrumError::model(format!("codec_embed.{i}: {e}")))?;
                // Project if needed
                let embed = if let Some(proj) = &self.projection {
                    embed.apply(proj).map_err(|e| FerrumError::model(format!("proj: {e}")))?
                } else {
                    embed
                };
                hidden = self.forward_layers(&embed)?;
            }
        }

        Ok(predicted_tokens)
    }

    fn forward_layers(&mut self, input: &Tensor) -> Result<Tensor> {
        let pos_offset = self.tokens_generated;
        let seq_len = input.dim(1).map_err(|e| FerrumError::model(format!("dim: {e}")))?;

        let mut h = input.clone();
        for layer in &mut self.layers {
            h = layer
                .forward(&h, pos_offset)
                .map_err(|e| FerrumError::model(format!("subtalker layer: {e}")))?;
        }
        h = h
            .apply(&self.norm)
            .map_err(|e| FerrumError::model(format!("subtalker norm: {e}")))?;

        self.tokens_generated += seq_len;
        Ok(h)
    }

    pub fn reset(&mut self) {
        self.tokens_generated = 0;
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }
}
