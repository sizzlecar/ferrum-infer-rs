//! Qwen3 architecture implementation
//!
//! Qwen3 is architecturally similar to Qwen2 but uses an explicit `head_dim`
//! that differs from `hidden_size / num_attention_heads`. This module provides
//! a custom implementation that supports this configuration.

use candle_core::{DType, Device as CandleDevice, Module, Result as CandleResult, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::{debug, info};

// Re-use candle's with_tracing utilities (except RmsNorm — candle's custom op
// lacks a Metal kernel; we use rms_norm_slow which is built from basic tensor
// ops that Metal supports).
use candle_transformers::models::with_tracing::{linear_no_bias, Linear};

/// RmsNorm implementation.
///
/// On CUDA: uses `candle_nn::RmsNorm` which dispatches to a single fused CUDA kernel.
/// On other backends (Metal/CPU): uses a manual implementation with basic tensor ops
/// because the custom op lacks a Metal kernel.
#[derive(Debug, Clone)]
struct RmsNorm {
    #[cfg(feature = "cuda")]
    inner: candle_nn::RmsNorm,
    #[cfg(not(feature = "cuda"))]
    weight: Tensor,
    #[cfg(not(feature = "cuda"))]
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let inner = candle_nn::rms_norm(size, eps, vb)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let weight = vb.get(size, "weight")?;
            Ok(Self { weight, eps })
        }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        #[cfg(feature = "cuda")]
        {
            self.inner.forward(x)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let x_dtype = x.dtype();
            let internal_dtype = match x_dtype {
                DType::F16 | DType::BF16 => DType::F32,
                d => d,
            };
            let hidden_size = x.dim(D::Minus1)?;
            let x = x.to_dtype(internal_dtype)?;
            let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
            x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &CandleDevice) -> CandleResult<Self> {
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

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        // CUDA: fused rope custom op. Other backends: basic tensor ops (no Metal kernel).
        #[cfg(feature = "cuda")]
        let (q_embed, k_embed) = {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            (q_embed, k_embed)
        };
        #[cfg(not(feature = "cuda"))]
        let (q_embed, k_embed) = {
            let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
            (q_embed, k_embed)
        };
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// Pre-allocated KV cache that avoids O(n) Tensor::cat copies per decode token.
///
/// Stores K/V in FlashAttention layout `[batch, max_seq, kv_heads, head_dim]` contiguous.
/// - Prefill: allocates buffer sized to prompt_len × 8 (capped at max_position_embeddings)
/// - Decode: `slice_set` writes O(1) new data; `narrow` provides zero-copy view
/// - FlashAttention only requires last-dim stride=1, so narrow views work directly
#[derive(Debug, Clone)]
struct PreAllocKvCache {
    k: Tensor, // [batch, max_len, kv_heads, head_dim] contiguous
    v: Tensor,
    current_len: usize,
    max_len: usize,
}

impl PreAllocKvCache {
    /// Allocate cache. K/V must be in [batch, seq, kv_heads, head_dim] layout, contiguous.
    fn new(
        k_init: &Tensor,
        v_init: &Tensor,
        max_len: usize,
    ) -> CandleResult<Self> {
        let (b, seq, h, d) = k_init.dims4()?;
        // Allocate [batch, max_len, kv_heads, head_dim]
        let k_buf = Tensor::zeros((b, max_len, h, d), k_init.dtype(), k_init.device())?;
        let v_buf = Tensor::zeros((b, max_len, h, d), v_init.dtype(), v_init.device())?;
        // Copy initial data
        k_buf.slice_set(k_init, 1, 0)?;
        v_buf.slice_set(v_init, 1, 0)?;
        Ok(Self {
            k: k_buf,
            v: v_buf,
            current_len: seq,
            max_len,
        })
    }

    /// Append new K/V at current position. Returns (k_view, v_view) for attention.
    /// Both views are [batch, current_len+new_len, kv_heads, head_dim] with last-dim stride=1.
    fn append(
        &mut self,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> CandleResult<(Tensor, Tensor)> {
        let new_len = k_new.dims4()?.1;
        let total = self.current_len + new_len;

        if total > self.max_len {
            // Grow buffer (double or fit)
            let new_max = (self.max_len * 2).max(total);
            let (b, _, h, d) = self.k.dims4()?;
            let k_new_buf = Tensor::zeros((b, new_max, h, d), self.k.dtype(), self.k.device())?;
            let v_new_buf = Tensor::zeros((b, new_max, h, d), self.v.dtype(), self.v.device())?;
            // Copy existing data
            let k_old = self.k.narrow(1, 0, self.current_len)?;
            let v_old = self.v.narrow(1, 0, self.current_len)?;
            k_new_buf.slice_set(&k_old.contiguous()?, 1, 0)?;
            v_new_buf.slice_set(&v_old.contiguous()?, 1, 0)?;
            self.k = k_new_buf;
            self.v = v_new_buf;
            self.max_len = new_max;
        }

        // Write new data at current_len
        self.k.slice_set(k_new, 1, self.current_len)?;
        self.v.slice_set(v_new, 1, self.current_len)?;
        self.current_len = total;

        // Return views over the valid range
        let k_view = self.k.narrow(1, 0, total)?;
        let v_view = self.v.narrow(1, 0, total)?;
        Ok((k_view, v_view))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<PreAllocKvCache>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            max_position_embeddings: cfg.max_position_embeddings,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        // Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Qwen3: apply QK-Norm before RoPE
        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // KV cache: new K/V are [batch, kv_heads, new_len, head_dim] after RoPE
        // Convert to FlashAttention layout [batch, new_len, kv_heads, head_dim] for cache
        let k_for_cache = key_states.transpose(1, 2)?.contiguous()?;
        let v_for_cache = value_states.transpose(1, 2)?.contiguous()?;

        // Update pre-allocated cache (O(1) write) or create it (first call)
        let (k_view, v_view) = match &mut self.kv_cache {
            Some(cache) => cache.append(&k_for_cache, &v_for_cache)?,
            None => {
                // First call: allocate cache sized to seq_len*8 (capped)
                let alloc_len = (q_len * 8)
                    .max(2048)
                    .min(self.max_position_embeddings);
                let cache = PreAllocKvCache::new(&k_for_cache, &v_for_cache, alloc_len)?;
                let k_v = cache.k.narrow(1, 0, cache.current_len)?;
                let v_v = cache.v.narrow(1, 0, cache.current_len)?;
                self.kv_cache = Some(cache);
                (k_v, v_v)
            }
        };
        // k_view, v_view: [batch, total_seq, kv_heads, head_dim] — zero-copy narrow views

        let attn_output = {
            #[cfg(feature = "cuda")]
            {
                // FlashAttention-2 expects [batch, seq, heads, head_dim]
                // k_view/v_view are already in that layout from the cache
                // query needs transpose back: [batch, heads, q_len, dim] -> [batch, q_len, heads, dim]
                let q = query_states.transpose(1, 2)?.contiguous()?;

                let scale = 1f32 / (self.head_dim as f32).sqrt();
                let causal = attention_mask.is_some() || q_len > 1;

                // flash_attn requires F16 or BF16
                let target_dtype = q.dtype();
                let compute_dtype = match target_dtype {
                    DType::F16 | DType::BF16 => target_dtype,
                    _ => DType::F16,
                };
                let q = if q.dtype() != compute_dtype { q.to_dtype(compute_dtype)? } else { q };
                let k = if k_view.dtype() != compute_dtype { k_view.to_dtype(compute_dtype)? } else { k_view };
                let v = if v_view.dtype() != compute_dtype { v_view.to_dtype(compute_dtype)? } else { v_view };

                let out = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?;
                let out = if out.dtype() != target_dtype { out.to_dtype(target_dtype)? } else { out };
                // Output [batch, q_len, heads, head_dim] -> [batch, heads, q_len, head_dim]
                out.transpose(1, 2)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                // Standard attention: needs [batch, heads, seq, head_dim]
                // Convert k_view/v_view from [batch, seq, heads, dim] to [batch, heads, seq, dim]
                let key_states = k_view.transpose(1, 2)?;
                let value_states = v_view.transpose(1, 2)?;
                let key_states =
                    candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?
                        .contiguous()?;
                let value_states =
                    candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
                        .contiguous()?;

                let scale = 1f64 / f64::sqrt(self.head_dim as f64);
                let attn_weights =
                    (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

                let attn_weights = match attention_mask {
                    None => attn_weights,
                    Some(mask) => attn_weights.broadcast_add(mask)?,
                };
                let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
                attn_weights.matmul(&value_states)?
            }
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: CandleDevice,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> CandleResult<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        _attn_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        } else if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> CandleResult<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset, None)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }
}

// ============================================================================
// Qwen3ModelWrapper — bridge between ferrum engine and Qwen3 model
// ============================================================================

pub struct Qwen3ModelWrapper {
    model: Mutex<ModelForCausalLM>,
    config: Config,
    device: CandleDevice,
    dtype: DType,
}

impl Qwen3ModelWrapper {
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Creating Qwen3 model from weights...");

        let head_dim = config
            .extra_params
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let tie_word_embeddings = config
            .extra_params
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let sliding_window = config
            .extra_params
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let max_window_layers = config
            .extra_params
            .get("max_window_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(config.num_hidden_layers as u64) as usize;

        let use_sliding_window = config
            .extra_params
            .get("use_sliding_window")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let qwen3_config = Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            head_dim,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta.unwrap_or(1000000.0),
            rms_norm_eps: config.norm_eps,
            tie_word_embeddings,
            sliding_window,
            max_window_layers,
            use_sliding_window,
            hidden_act: Activation::Silu,
        };

        debug!(
            "Qwen3 config: hidden={}, layers={}, heads={}, kv_heads={}, head_dim={}",
            qwen3_config.hidden_size,
            qwen3_config.num_hidden_layers,
            qwen3_config.num_attention_heads,
            qwen3_config.num_key_value_heads,
            qwen3_config.head_dim,
        );

        let model = ModelForCausalLM::new(&qwen3_config, vb)
            .map_err(|e| FerrumError::model(format!("Failed to create Qwen3 model: {}", e)))?;

        info!("Qwen3 model created successfully");

        Ok(Self {
            model: Mutex::new(model),
            config: qwen3_config,
            device,
            dtype,
        })
    }

    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut model = self.model.lock();
        model
            .forward(input_ids, 0)
            .map_err(|e| FerrumError::model(format!("Prefill forward failed: {}", e)))
    }

    pub fn forward_decode(&self, token_id: &Tensor, pos: usize) -> Result<Tensor> {
        let mut model = self.model.lock();
        model
            .forward(token_id, pos)
            .map_err(|e| FerrumError::model(format!("Decode forward failed: {}", e)))
    }

    pub fn reset_cache(&self) -> Result<()> {
        self.model.lock().clear_kv_cache();
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    pub fn candle_device(&self) -> &CandleDevice {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
