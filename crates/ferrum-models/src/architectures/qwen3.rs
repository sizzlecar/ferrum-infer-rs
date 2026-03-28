//! Qwen3 architecture implementation
//!
//! Qwen3 is architecturally similar to Qwen2 but uses an explicit `head_dim`
//! that differs from `hidden_size / num_attention_heads`. This module provides
//! a custom implementation that supports this configuration.

use candle_core::{DType, Device as CandleDevice, Module, Result as CandleResult, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

// Use candle_nn::Linear directly (not with_tracing wrapper) so we can
// access weight() for CUDA decode runner weight extraction.
use candle_nn::Linear;

/// Load a no-bias linear layer from a VarBuilder.
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> CandleResult<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// RmsNorm implementation.
///
/// On CUDA: uses `candle_nn::RmsNorm` which dispatches to a single fused CUDA kernel.
/// On other backends (Metal/CPU): uses a manual implementation with basic tensor ops
/// because the custom op lacks a Metal kernel.
///
/// Weight and eps are always accessible for use by fused custom CUDA kernels.
#[derive(Debug, Clone)]
struct RmsNorm {
    #[cfg(feature = "cuda")]
    inner: candle_nn::RmsNorm,
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        #[cfg(feature = "cuda")]
        {
            let inner = candle_nn::RmsNorm::new(weight.clone(), eps);
            Ok(Self { inner, weight, eps })
        }
        #[cfg(not(feature = "cuda"))]
        {
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
            rms_norm_slow(x, &self.weight, self.eps)
        }
    }
}

#[cfg(feature = "cuda")]
fn fused_add_rms_norm_compat(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> CandleResult<(Tensor, Tensor)> {
    if input.dims() != residual.dims() {
        candle_core::bail!(
            "fused_add_rms_norm_compat shape mismatch: input {:?}, residual {:?}",
            input.dims(),
            residual.dims()
        );
    }

    let dims = input.dims();
    match dims.len() {
        2 => ferrum_cuda_kernels::fused_add_rms_norm(input, residual, weight, eps),
        3 => {
            let batch_size = dims[0];
            let seq_len = dims[1];
            let hidden_size = dims[2];
            let flat_shape = (batch_size * seq_len, hidden_size);
            let view_shape = (batch_size, seq_len, hidden_size);

            let input_2d = input.reshape(flat_shape)?;
            let residual_2d = residual.reshape(flat_shape)?;
            let (normalized, residual_updated) =
                ferrum_cuda_kernels::fused_add_rms_norm(&input_2d, &residual_2d, weight, eps)?;

            Ok((
                normalized.reshape(view_shape)?,
                residual_updated.reshape(view_shape)?,
            ))
        }
        _ => candle_core::bail!(
            "fused_add_rms_norm_compat unsupported input shape: {:?}",
            dims
        ),
    }
}

/// Manual RMS norm using basic tensor ops (Metal/CPU compatible).
fn rms_norm_slow(x: &Tensor, weight: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)
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

/// MLP with fused gate+up projection: 2 matmuls → 1.
///
/// During decode, input activations are read once instead of twice from GPU memory.
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_up_proj: Linear, // fused [intermediate*2, hidden]
    down_proj: Linear,
    intermediate_size: usize,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        // Load gate and up weights separately, then fuse
        let gate_w = vb
            .pp("gate_proj")
            .get((intermediate_sz, hidden_sz), "weight")?;
        let up_w = vb
            .pp("up_proj")
            .get((intermediate_sz, hidden_sz), "weight")?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?; // [intermediate*2, hidden]
        let gate_up_proj = Linear::new(gate_up_w, None);
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: intermediate_sz,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let gate_up = xs.apply(&self.gate_up_proj)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        // Fused SiLU+mul: 2 kernel launches → 1, reads gate+up once instead of twice
        #[cfg(feature = "cuda")]
        let hidden = ferrum_cuda_kernels::fused_silu_mul(&gate.contiguous()?, &up.contiguous()?)?;
        #[cfg(not(feature = "cuda"))]
        let hidden = (gate.apply(&self.act_fn)? * up)?;
        hidden.apply(&self.down_proj)
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
    fn new(k_init: &Tensor, v_init: &Tensor, max_len: usize) -> CandleResult<Self> {
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
    fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> CandleResult<(Tensor, Tensor)> {
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

/// Attention with fused QKV projection: 3 matmuls → 1.
///
/// During decode, input activations are read once instead of three times from GPU memory.
#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Linear, // fused [q_dim + kv_dim*2, hidden]
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    q_dim: usize,  // num_heads * head_dim
    kv_dim: usize, // num_kv_heads * head_dim
    max_position_embeddings: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_caches: HashMap<String, PreAllocKvCache>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Load Q/K/V weights separately, then fuse into one matrix
        let q_w = vb.pp("q_proj").get((q_dim, hidden_sz), "weight")?;
        let k_w = vb.pp("k_proj").get((kv_dim, hidden_sz), "weight")?;
        let v_w = vb.pp("v_proj").get((kv_dim, hidden_sz), "weight")?;
        let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?; // [q_dim+kv_dim*2, hidden]
        let qkv_proj = Linear::new(qkv_w, None);
        let o_proj = linear_no_bias(q_dim, hidden_sz, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            q_dim,
            kv_dim,
            max_position_embeddings: cfg.max_position_embeddings,
            rotary_emb,
            kv_caches: HashMap::new(),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_key: &str,
    ) -> CandleResult<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Fused QKV projection: single matmul instead of 3 separate ones.
        // Input activations are read from GPU memory once instead of three times.
        let qkv = self.qkv_proj.forward(xs)?;
        let query_states = qkv.narrow(D::Minus1, 0, self.q_dim)?;
        let key_states = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
        let value_states = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;

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
        // Each sequence gets its own KV cache keyed by cache_key.
        let (k_view, v_view) = if let Some(cache) = self.kv_caches.get_mut(cache_key) {
            cache.append(&k_for_cache, &v_for_cache)?
        } else {
            // First call for this sequence: allocate cache
            let alloc_len = (q_len * 8).max(2048).min(self.max_position_embeddings);
            let cache = PreAllocKvCache::new(&k_for_cache, &v_for_cache, alloc_len)?;
            let k_v = cache.k.narrow(1, 0, cache.current_len)?;
            let v_v = cache.v.narrow(1, 0, cache.current_len)?;
            self.kv_caches.insert(cache_key.to_string(), cache);
            (k_v, v_v)
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
                let q = if q.dtype() != compute_dtype {
                    q.to_dtype(compute_dtype)?
                } else {
                    q
                };
                let k = if k_view.dtype() != compute_dtype {
                    k_view.to_dtype(compute_dtype)?
                } else {
                    k_view
                };
                let v = if v_view.dtype() != compute_dtype {
                    v_view.to_dtype(compute_dtype)?
                } else {
                    v_view
                };

                let out = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?;
                let out = if out.dtype() != target_dtype {
                    out.to_dtype(target_dtype)?
                } else {
                    out
                };
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
                let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

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
        self.kv_caches.clear();
    }

    fn clear_kv_cache_for(&mut self, cache_key: &str) {
        self.kv_caches.remove(cache_key);
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
        cache_key: &str,
    ) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset, cache_key)?;
        // Fused residual-add + RMS norm: 2 kernel launches → 1, saves 1 memory round-trip
        #[cfg(feature = "cuda")]
        let (xs, residual) = {
            let (normalized, residual_updated) = fused_add_rms_norm_compat(
                &xs,
                residual,
                &self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps as f32,
            )?;
            (normalized.apply(&self.mlp)?, residual_updated)
        };
        #[cfg(not(feature = "cuda"))]
        let (xs, residual) = {
            let sum = (xs + residual)?;
            let xs = sum
                .apply(&self.post_attention_layernorm)?
                .apply(&self.mlp)?;
            (xs, sum)
        };
        &residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }

    fn clear_kv_cache_for(&mut self, cache_key: &str) {
        self.self_attn.clear_kv_cache_for(cache_key)
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
        cache_key: &str,
    ) -> CandleResult<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset, cache_key)?
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    pub fn clear_kv_cache_for(&mut self, cache_key: &str) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache_for(cache_key)
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
            Linear::new(base_model.embed_tokens.embeddings().clone(), None)
        } else if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        cache_key: &str,
    ) -> CandleResult<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset, None, cache_key)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }

    pub fn clear_kv_cache_for(&mut self, cache_key: &str) {
        self.base_model.clear_kv_cache_for(cache_key)
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
    /// Model directory path (for loading GPTQ weights directly from safetensors).
    model_dir: Option<std::path::PathBuf>,
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
            model_dir: None,
        })
    }

    /// Set model directory path (for GPTQ weight loading).
    pub fn set_model_dir(&mut self, path: std::path::PathBuf) {
        self.model_dir = Some(path);
    }

    pub fn forward_prefill(&self, input_ids: &Tensor, cache_key: &str) -> Result<Tensor> {
        let mut model = self.model.lock();
        model.clear_kv_cache_for(cache_key);
        model
            .forward(input_ids, 0, cache_key)
            .map_err(|e| FerrumError::model(format!("Prefill forward failed: {}", e)))
    }

    pub fn forward_decode(&self, token_id: &Tensor, pos: usize, cache_key: &str) -> Result<Tensor> {
        let mut model = self.model.lock();
        model
            .forward(token_id, pos, cache_key)
            .map_err(|e| FerrumError::model(format!("Decode forward failed: {}", e)))
    }

    /// Release KV cache for a completed sequence, freeing GPU memory.
    pub fn release_cache(&self, cache_key: &str) {
        self.model.lock().clear_kv_cache_for(cache_key);
    }

    pub fn reset_all_caches(&self) {
        self.model.lock().clear_kv_cache();
    }

    /// Export KV cache data for a sequence, for use by CudaDecodeRunner.
    ///
    /// Returns per-layer (k_tensor, v_tensor, current_len, max_len) where K/V are
    /// [batch=1, max_len, kv_heads, head_dim] contiguous candle Tensors.
    /// Returns None if the sequence has no KV cache.
    #[cfg(feature = "cuda")]
    pub fn export_kv_cache(&self, cache_key: &str) -> Option<Vec<(Tensor, Tensor, usize, usize)>> {
        let model = self.model.lock();
        let mut result = Vec::new();
        for layer in &model.base_model.layers {
            if let Some(cache) = layer.self_attn.kv_caches.get(cache_key) {
                result.push((
                    cache.k.clone(),
                    cache.v.clone(),
                    cache.current_len,
                    cache.max_len,
                ));
            } else {
                return None;
            }
        }
        Some(result)
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

    /// Create a CUDA decode runner by extracting weight pointers from the model.
    ///
    /// The runner bypasses candle for the decode hot path, using cuBLAS + custom
    /// CUDA kernels with pre-allocated buffers.
    #[cfg(feature = "cuda")]
    pub fn create_decode_runner(
        &self,
    ) -> Result<ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner> {
        use ferrum_cuda_kernels::decode_buffers::ModelDims;
        use ferrum_cuda_kernels::weight_store::{
            GpuWeight, LayerWeights, LinearWeight, Qwen3Weights,
        };

        let model = self.model.lock();
        let cfg = &self.config;

        // Create non-blocking stream — required for CUDA Graph capture.
        // All weights and buffers will be copied to this stream.
        let cuda_device = self
            .device
            .as_cuda_device()
            .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
        // Sync candle's stream FIRST — ensure all weight tensors are
        // fully materialized on GPU before we copy them cross-stream.
        let candle_stream = cuda_device.cuda_stream();
        candle_stream
            .synchronize()
            .map_err(|e| FerrumError::model(format!("candle stream sync: {e}")))?;

        let rs = candle_stream
            .context()
            .new_stream()
            .map_err(|e| FerrumError::model(format!("new_stream: {e}")))?;

        // Detect GPTQ quantization
        let qconfig = self.model_dir.as_ref().and_then(|dir| {
            crate::loader::QuantizeConfig::from_model_dir(dir)
                .ok()
                .flatten()
        });
        let is_gptq = qconfig.is_some();

        // Load GPTQ packed weights if quantized
        let gptq_weights = if let (Some(dir), Some(ref qc)) = (&self.model_dir, &qconfig) {
            Some(
                crate::loader::load_gptq_weights(dir, qc)
                    .map_err(|e| FerrumError::model(format!("GPTQ load: {e}")))?,
            )
        } else {
            None
        };

        let dims = ModelDims {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_attention_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            vocab_size: cfg.vocab_size,
            num_layers: cfg.num_hidden_layers,
            max_seq_len: cfg.max_position_embeddings,
            quantized: is_gptq,
            max_batch_size: std::env::var("FERRUM_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
        };

        /// Helper: try to load GPTQ INT4 weight, fall back to FP16 from candle tensor.
        fn load_linear_weight(
            prefix: &str,
            candle_tensor: &candle_core::Tensor,
            gptq: Option<&std::collections::HashMap<String, crate::loader::GptqLayerWeights>>,
            stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        ) -> std::result::Result<LinearWeight, FerrumError> {
            use ferrum_cuda_kernels::weight_store::GpuQuantWeight;

            if let Some(gptq_map) = gptq {
                if let Some(gw) = gptq_map.get(prefix) {
                    // Try Marlin (fused, 3.9x) if dimensions are compatible
                    let use_marlin = ferrum_cuda_kernels::marlin::is_available()
                        && gw.k % 128 == 0
                        && gw.n % 256 == 0
                        && (gw.group_size == 128 || gw.group_size == gw.k)
                        && std::env::var("FERRUM_NO_MARLIN").is_err();

                    if use_marlin {
                        use ferrum_cuda_kernels::marlin::{
                            repack_gptq_to_marlin, repack_scales_to_marlin, MarlinWeight,
                        };
                        let marlin_qw = repack_gptq_to_marlin(&gw.qweight, gw.k, gw.n);
                        // TODO: debug scale permutation. For now try raw scales.
                        let marlin_scales = gw.scales.clone();
                        let gs_marlin = if gw.group_size == gw.k {
                            -1i32
                        } else {
                            gw.group_size as i32
                        };
                        let qweight = stream
                            .clone_htod(&marlin_qw)
                            .map_err(|e| FerrumError::model(format!("{prefix} marlin qw: {e}")))?;
                        let scales = stream
                            .clone_htod(bytemuck::cast_slice::<half::f16, u8>(&marlin_scales))
                            .map_err(|e| {
                                FerrumError::model(format!("{prefix} marlin scales: {e}"))
                            })?;
                        let scales_f16: candle_core::cuda_backend::cudarc::driver::CudaSlice<
                            half::f16,
                        > = unsafe { std::mem::transmute(scales) };
                        let ws_size = (gw.n / 128) * 16;
                        let workspace = stream
                            .clone_htod(&vec![0i32; ws_size])
                            .map_err(|e| FerrumError::model(format!("{prefix} marlin ws: {e}")))?;
                        tracing::warn!(
                            "Marlin weight: {prefix} K={} N={} gs={gs_marlin}",
                            gw.k,
                            gw.n,
                        );
                        return Ok(LinearWeight::Marlin(MarlinWeight {
                            qweight,
                            scales: scales_f16,
                            workspace,
                            k: gw.k,
                            n: gw.n,
                            group_size: gs_marlin,
                        }));
                    }

                    // Fallback: dequant + cuBLAS path
                    let qweight = stream
                        .clone_htod(&gw.qweight)
                        .map_err(|e| FerrumError::model(format!("{prefix} qweight upload: {e}")))?;
                    let scales = stream
                        .clone_htod(bytemuck::cast_slice::<half::f16, u8>(&gw.scales))
                        .map_err(|e| FerrumError::model(format!("{prefix} scales upload: {e}")))?;
                    let scales_f16: candle_core::cuda_backend::cudarc::driver::CudaSlice<
                        half::f16,
                    > = unsafe { std::mem::transmute(scales) };
                    let qzeros = if let Some(ref qz) = gw.qzeros {
                        Some(stream.clone_htod(qz).map_err(|e| {
                            FerrumError::model(format!("{prefix} qzeros upload: {e}"))
                        })?)
                    } else {
                        None
                    };
                    tracing::warn!(
                        "INT4 dequant weight: {prefix} K={} N={} gs={}",
                        gw.k,
                        gw.n,
                        gw.group_size
                    );
                    return Ok(LinearWeight::Int4(GpuQuantWeight {
                        qweight,
                        scales: scales_f16,
                        qzeros,
                        k: gw.k,
                        n: gw.n,
                        group_size: gw.group_size,
                        symmetric: gw.symmetric,
                    }));
                }
            }
            // Fall back to FP16
            Ok(LinearWeight::Fp16(
                GpuWeight::from_tensor(candle_tensor, stream)
                    .map_err(|e| FerrumError::model(format!("{prefix}: {e}")))?,
            ))
        }

        // Extract weights — copy to runner's non-blocking stream
        let embed_table = GpuWeight::from_tensor(model.base_model.embed_tokens.embeddings(), &rs)
            .map_err(|e| FerrumError::model(format!("embed: {e}")))?;

        let gptq_ref = gptq_weights.as_ref();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (li, layer) in model.base_model.layers.iter().enumerate() {
            let prefix = format!("model.layers.{li}");
            let lw = LayerWeights {
                input_ln_w: GpuWeight::from_tensor(&layer.input_layernorm.weight, &rs)
                    .map_err(|e| FerrumError::model(format!("input_ln: {e}")))?,
                qkv_w: load_linear_weight(
                    &format!("{prefix}.self_attn.qkv_proj"),
                    layer.self_attn.qkv_proj.weight(),
                    gptq_ref,
                    &rs,
                )?,
                q_norm_w: Some(
                    GpuWeight::from_tensor(&layer.self_attn.q_norm.weight, &rs)
                        .map_err(|e| FerrumError::model(format!("q_norm: {e}")))?,
                ),
                k_norm_w: Some(
                    GpuWeight::from_tensor(&layer.self_attn.k_norm.weight, &rs)
                        .map_err(|e| FerrumError::model(format!("k_norm: {e}")))?,
                ),
                o_w: load_linear_weight(
                    &format!("{prefix}.self_attn.o_proj"),
                    layer.self_attn.o_proj.weight(),
                    gptq_ref,
                    &rs,
                )?,
                post_ln_w: GpuWeight::from_tensor(&layer.post_attention_layernorm.weight, &rs)
                    .map_err(|e| FerrumError::model(format!("post_ln: {e}")))?,
                gate_up_w: load_linear_weight(
                    &format!("{prefix}.mlp.gate_up_proj"),
                    layer.mlp.gate_up_proj.weight(),
                    gptq_ref,
                    &rs,
                )?,
                down_w: load_linear_weight(
                    &format!("{prefix}.mlp.down_proj"),
                    layer.mlp.down_proj.weight(),
                    gptq_ref,
                    &rs,
                )?,
            };
            layers.push(lw);
        }

        let final_norm_w = GpuWeight::from_tensor(&model.base_model.norm.weight, &rs)
            .map_err(|e| FerrumError::model(format!("final_norm: {e}")))?;
        let lm_head_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(model.lm_head.weight(), &rs)
                .map_err(|e| FerrumError::model(format!("lm_head: {e}")))?,
        );
        let rope_cos =
            GpuWeight::from_tensor(&model.base_model.layers[0].self_attn.rotary_emb.cos, &rs)
                .map_err(|e| FerrumError::model(format!("rope_cos: {e}")))?;
        let rope_sin =
            GpuWeight::from_tensor(&model.base_model.layers[0].self_attn.rotary_emb.sin, &rs)
                .map_err(|e| FerrumError::model(format!("rope_sin: {e}")))?;

        let weights = Qwen3Weights {
            embed_table,
            layers,
            final_norm_w,
            lm_head_w,
            rope_cos,
            rope_sin,
        };

        // Synchronize the runner stream to ensure all weight D2D copies
        // from candle's default stream are complete before use.
        rs.synchronize()
            .map_err(|e| FerrumError::model(format!("stream sync after weight copy: {e}")))?;

        ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner::new(
            weights,
            dims,
            cuda_device.clone(),
            rs,
        )
        .map_err(|e| FerrumError::model(format!("CudaDecodeRunner: {e}")))
    }
}

// ======================== TransformerWeights implementation ========================

use ferrum_interfaces::transformer::{TransformerConfig, TransformerWeights};
use ferrum_interfaces::TensorRef;

/// Cached weight TensorRefs for the TransformerWeights trait.
/// Pre-wrapped at construction time — each call returns Arc::clone() (cheap).
struct Qwen3WeightCache {
    config: TransformerConfig,
    embed: TensorRef,
    layers: Vec<Qwen3LayerWeightCache>,
    final_norm: TensorRef,
    lm_head: TensorRef,
    rope_cos: TensorRef,
    rope_sin: TensorRef,
}

struct Qwen3LayerWeightCache {
    input_norm: TensorRef,
    qkv: TensorRef,
    q_norm: TensorRef,
    k_norm: TensorRef,
    o_proj: TensorRef,
    post_norm: TensorRef,
    gate_up: TensorRef,
    down: TensorRef,
}

fn wrap(t: &Tensor) -> TensorRef {
    Arc::new(crate::tensor_wrapper::CandleTensorWrapper::new(t.clone()))
}

impl Qwen3WeightCache {
    fn from_model(model: &ModelForCausalLM, cfg: &Config) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in &model.base_model.layers {
            layers.push(Qwen3LayerWeightCache {
                input_norm: wrap(&layer.input_layernorm.weight),
                qkv: wrap(layer.self_attn.qkv_proj.weight()),
                q_norm: wrap(&layer.self_attn.q_norm.weight),
                k_norm: wrap(&layer.self_attn.k_norm.weight),
                o_proj: wrap(layer.self_attn.o_proj.weight()),
                post_norm: wrap(&layer.post_attention_layernorm.weight),
                gate_up: wrap(layer.mlp.gate_up_proj.weight()),
                down: wrap(layer.mlp.down_proj.weight()),
            });
        }

        Self {
            config: TransformerConfig {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attention_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                intermediate_size: cfg.intermediate_size,
                vocab_size: cfg.vocab_size,
                max_seq_len: cfg.max_position_embeddings,
                rms_norm_eps: cfg.rms_norm_eps as f32,
                has_qk_norm: true, // Qwen3 has Q/K head normalization
            },
            embed: wrap(model.base_model.embed_tokens.embeddings()),
            layers,
            final_norm: wrap(&model.base_model.norm.weight),
            lm_head: wrap(model.lm_head.weight()),
            rope_cos: wrap(&model.base_model.layers[0].self_attn.rotary_emb.cos),
            rope_sin: wrap(&model.base_model.layers[0].self_attn.rotary_emb.sin),
        }
    }
}

impl TransformerWeights for Qwen3WeightCache {
    fn config(&self) -> &TransformerConfig {
        &self.config
    }
    fn embed_weight(&self) -> TensorRef {
        self.embed.clone()
    }
    fn layer_input_norm_weight(&self, layer: usize) -> TensorRef {
        self.layers[layer].input_norm.clone()
    }
    fn layer_qkv_weight(&self, layer: usize) -> TensorRef {
        self.layers[layer].qkv.clone()
    }
    fn layer_q_norm_weight(&self, layer: usize) -> Option<TensorRef> {
        Some(self.layers[layer].q_norm.clone())
    }
    fn layer_k_norm_weight(&self, layer: usize) -> Option<TensorRef> {
        Some(self.layers[layer].k_norm.clone())
    }
    fn layer_o_weight(&self, layer: usize) -> TensorRef {
        self.layers[layer].o_proj.clone()
    }
    fn layer_post_norm_weight(&self, layer: usize) -> TensorRef {
        self.layers[layer].post_norm.clone()
    }
    fn layer_gate_up_weight(&self, layer: usize) -> TensorRef {
        self.layers[layer].gate_up.clone()
    }
    fn layer_down_weight(&self, layer: usize) -> TensorRef {
        self.layers[layer].down.clone()
    }
    fn final_norm_weight(&self) -> TensorRef {
        self.final_norm.clone()
    }
    fn lm_head_weight(&self) -> TensorRef {
        self.lm_head.clone()
    }
    fn rope_cos(&self) -> TensorRef {
        self.rope_cos.clone()
    }
    fn rope_sin(&self) -> TensorRef {
        self.rope_sin.clone()
    }
}
