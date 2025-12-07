//! Metal-accelerated Qwen2 model implementation
//!
//! Qwen2 is architecturally similar to LLaMA with:
//! - RMS Normalization (same as LLaMA)
//! - Rotary Position Embeddings (same as LLaMA)  
//! - Group Query Attention (same as LLaMA)
//! - SiLU activation in MLP (same as LLaMA)
//! - Optional bias in attention and MLP layers

use crate::metal::{MetalContext, RmsNormOps};
use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};
use candle_nn::ops;
use ferrum_types::{FerrumError, Result};
use ferrum_models::architectures::qwen2::Qwen2ModelWrapper;
use std::sync::Arc;
use tracing::{debug, info, warn};

// Re-use MetalRmsNorm from metal_llama
use super::metal_llama::MetalRmsNorm;

/// Qwen2 model configuration
#[derive(Debug, Clone)]
pub struct MetalQwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub use_bias: bool,
    pub tie_word_embeddings: bool,  // Whether lm_head shares weights with embed_tokens
}

impl Default for MetalQwen2Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            use_bias: true,
            tie_word_embeddings: false,
        }
    }
}

/// Rotary Position Embedding for Qwen2
/// 
/// CRITICAL: All RoPE computations (inv_freq, cos, sin, rotation) are done in F32
/// to avoid half-precision numerical instability. Only the final result is cast
/// back to the target dtype.
struct Qwen2RotaryEmbedding {
    cos_cache: Tensor,  // Stored in F32 for precision
    sin_cache: Tensor,  // Stored in F32 for precision
    head_dim: usize,
    target_dtype: DType,  // Original dtype to cast back to
}

impl Qwen2RotaryEmbedding {
    fn new(config: &MetalQwen2Config, dtype: DType, device: &CandleDevice) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_position_embeddings;

        // CRITICAL FIX: All RoPE calculations in F32
        // Half-precision theta/cos/sin calculations cause massive numerical errors
        
        // Compute inverse frequencies in f64, store as f32
        // inv_freq[i] = 1 / (theta ^ (2*i / head_dim))
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| {
                let exponent = (i as f64) / (head_dim as f64);
                let theta_pow = (config.rope_theta as f64).powf(exponent);
                (1.0f64 / theta_pow) as f32
            })
            .collect();
        let inv_freq_len = inv_freq.len();

        // Keep inv_freq in F32 (NOT target dtype)
        let inv_freq_tensor = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)
            .map_err(|e| FerrumError::model(format!("inv_freq tensor failed: {}", e)))?;
        // F32 is the default dtype for from_vec with f32

        // Compute position indices in F32
        let t = Tensor::arange(0u32, max_seq_len as u32, device)
            .map_err(|e| FerrumError::model(format!("arange failed: {}", e)))?
            .to_dtype(DType::F32)  // F32, not target dtype!
            .map_err(|e| FerrumError::model(format!("t dtype failed: {}", e)))?
            .reshape((max_seq_len, 1))
            .map_err(|e| FerrumError::model(format!("t reshape failed: {}", e)))?;

        // Compute freqs = t * inv_freq in F32
        let freqs = t
            .matmul(&inv_freq_tensor)
            .map_err(|e| FerrumError::model(format!("freqs matmul failed: {}", e)))?;

        // cos/sin in F32 - CRITICAL for numerical stability
        let cos_cache = freqs
            .cos()
            .map_err(|e| FerrumError::model(format!("cos failed: {}", e)))?;

        let sin_cache = freqs
            .sin()
            .map_err(|e| FerrumError::model(format!("sin failed: {}", e)))?;

        info!("RoPE cache (F32 for precision): cos shape={:?}, sin shape={:?}, head_dim={}, target_dtype={:?}", 
            cos_cache.shape(), sin_cache.shape(), head_dim, dtype);
        
        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            target_dtype: dtype,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        // CRITICAL FIX: All RoPE operations in F32 for numerical stability
        // Convert input to F32, do rotation, convert back to target dtype
        
        let original_dtype = q.dtype();
        
        // Ensure contiguous first
        let q = q.contiguous()
            .map_err(|e| FerrumError::model(format!("q contiguous failed: {}", e)))?;
        let k = k.contiguous()
            .map_err(|e| FerrumError::model(format!("k contiguous failed: {}", e)))?;
        
        // Convert to F32 for rotation computation
        let q_f32 = if original_dtype != DType::F32 {
            q.to_dtype(DType::F32)
                .map_err(|e| FerrumError::model(format!("q to F32 failed: {}", e)))?
        } else {
            q
        };
        let k_f32 = if original_dtype != DType::F32 {
            k.to_dtype(DType::F32)
                .map_err(|e| FerrumError::model(format!("k to F32 failed: {}", e)))?
        } else {
            k
        };
        
        // Apply RoPE in F32
        let q_rot_f32 = self.rope_slow(&q_f32, start_pos)?;
        let k_rot_f32 = self.rope_slow(&k_f32, start_pos)?;
        
        // Convert back to original dtype
        let q_rot = if original_dtype != DType::F32 {
            q_rot_f32.to_dtype(original_dtype)
                .map_err(|e| FerrumError::model(format!("q_rot to target dtype failed: {}", e)))?
                .contiguous()
                .map_err(|e| FerrumError::model(format!("q_rot contiguous failed: {}", e)))?
        } else {
            q_rot_f32.contiguous()
                .map_err(|e| FerrumError::model(format!("q_rot contiguous failed: {}", e)))?
        };
        let k_rot = if original_dtype != DType::F32 {
            k_rot_f32.to_dtype(original_dtype)
                .map_err(|e| FerrumError::model(format!("k_rot to target dtype failed: {}", e)))?
                .contiguous()
                .map_err(|e| FerrumError::model(format!("k_rot contiguous failed: {}", e)))?
        } else {
            k_rot_f32.contiguous()
                .map_err(|e| FerrumError::model(format!("k_rot contiguous failed: {}", e)))?
        };
        
        Ok((q_rot, k_rot))
    }

    /// Manual RoPE implementation - ALL operations in F32
    /// This matches candle_nn::rope_slow (HALF ROTATION, NOT interleaved)
    /// Qwen2 uses half rotation: split into first half and second half, not interleaved pairs
    fn rope_slow(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        // x shape: [batch, num_heads, seq_len, head_dim]
        // x is already in F32 (ensured by apply())
        let (_b_sz, _n_head, seq_len, n_embd) = x.dims4()
            .map_err(|e| FerrumError::model(format!("dims4 failed: {}", e)))?;

        // cos/sin cache shape: [max_seq_len, head_dim/2]
        // For half rotation, we need to duplicate: [seq_len, head_dim/2] -> [seq_len, head_dim]
        let cos = self.cos_cache
            .narrow(0, start_pos, seq_len)
            .map_err(|e| FerrumError::model(format!("cos narrow failed: {}", e)))?;
        let sin = self.sin_cache
            .narrow(0, start_pos, seq_len)
            .map_err(|e| FerrumError::model(format!("sin narrow failed: {}", e)))?;
        
        // Duplicate cos and sin to match head_dim: [seq, dim/2] -> [seq, dim]
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)
            .map_err(|e| FerrumError::model(format!("cos cat failed: {}", e)))?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)
            .map_err(|e| FerrumError::model(format!("sin cat failed: {}", e)))?;
        
        // Add batch and head dims: [seq, dim] -> [1, 1, seq, dim]
        let cos = cos.unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("cos unsqueeze1 failed: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("cos unsqueeze2 failed: {}", e)))?;
        let sin = sin.unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("sin unsqueeze1 failed: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("sin unsqueeze2 failed: {}", e)))?;
        
        // rotate_half: split x into first half and second half, negate second half for rotation
        // x = [x1, x2] -> rotate_half(x) = [-x2, x1]
        let x1 = x.narrow(D::Minus1, 0, n_embd / 2)
            .map_err(|e| FerrumError::model(format!("x1 narrow failed: {}", e)))?;
        let x2 = x.narrow(D::Minus1, n_embd / 2, n_embd / 2)
            .map_err(|e| FerrumError::model(format!("x2 narrow failed: {}", e)))?;
        let x2_neg = x2.neg()
            .map_err(|e| FerrumError::model(format!("x2 neg failed: {}", e)))?;
        let rotated = Tensor::cat(&[&x2_neg, &x1], D::Minus1)
            .map_err(|e| FerrumError::model(format!("rotated cat failed: {}", e)))?;
        
        // Apply rotation: x * cos + rotate_half(x) * sin
        let result = x.broadcast_mul(&cos)
            .map_err(|e| FerrumError::model(format!("x*cos failed: {}", e)))?
            .broadcast_add(&rotated.broadcast_mul(&sin)
                .map_err(|e| FerrumError::model(format!("rotated*sin failed: {}", e)))?)
            .map_err(|e| FerrumError::model(format!("result add failed: {}", e)))?;
        
        Ok(result)
    }
}

/// Qwen2 Attention layer with Metal RMS Norm
struct MetalQwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MetalQwen2Attention {
    fn load(vb: VarBuilder, config: &MetalQwen2Config) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        // Log VarBuilder dtype
        debug!("Attention VarBuilder dtype: {:?}", vb.dtype());

        // Qwen2.5: Q/K/V have bias, O_proj has no bias
        // Try with bias first, fallback to no bias for compatibility
        let q_proj = linear(config.hidden_size, config.hidden_size, vb.pp("q_proj"))
            .or_else(|_| linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q_proj")))
            .map_err(|e| FerrumError::model(format!("q_proj load failed: {}", e)))?;

        // Log weight dtype for debugging
        debug!("q_proj weight dtype: {:?}, shape: {:?}", q_proj.weight().dtype(), q_proj.weight().shape());

        let k_proj = linear(config.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))
            .or_else(|_| linear_no_bias(config.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj")))
            .map_err(|e| FerrumError::model(format!("k_proj load failed: {}", e)))?;

        let v_proj = linear(config.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))
            .or_else(|_| linear_no_bias(config.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj")))
            .map_err(|e| FerrumError::model(format!("v_proj load failed: {}", e)))?;

        let o_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("o_proj"))
            .or_else(|_| linear(config.hidden_size, config.hidden_size, vb.pp("o_proj")))
            .map_err(|e| FerrumError::model(format!("o_proj load failed: {}", e)))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope: &Qwen2RotaryEmbedding,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()
            .map_err(|e| FerrumError::model(format!("dims3 failed: {}", e)))?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)
            .map_err(|e| FerrumError::model(format!("q_proj forward failed: {}", e)))?;
        let k = self.k_proj.forward(x)
            .map_err(|e| FerrumError::model(format!("k_proj forward failed: {}", e)))?;
        let v = self.v_proj.forward(x)
            .map_err(|e| FerrumError::model(format!("v_proj forward failed: {}", e)))?;

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| FerrumError::model(format!("q reshape failed: {}", e)))?
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("q transpose failed: {}", e)))?;

        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| FerrumError::model(format!("k reshape failed: {}", e)))?
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("k transpose failed: {}", e)))?;

        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| FerrumError::model(format!("v reshape failed: {}", e)))?
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("v transpose failed: {}", e)))?;

        // Apply rotary embeddings (already in F32)
        let (q, k) = rope.apply(&q, &k, start_pos)?;

        // Update KV cache
        let (k, v) = if let Some((prev_k, prev_v)) = &self.kv_cache {
            let k = Tensor::cat(&[prev_k, &k], 2)
                .map_err(|e| FerrumError::model(format!("k cat failed: {}", e)))?;
            let v = Tensor::cat(&[prev_v, &v], 2)
                .map_err(|e| FerrumError::model(format!("v cat failed: {}", e)))?;
            (k, v)
        } else {
            (k, v)
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(2, 3)
            .map_err(|e| FerrumError::model(format!("k transpose failed: {}", e)))?)
            .map_err(|e| FerrumError::model(format!("attn matmul failed: {}", e)))?;
        
        let attn = (attn_scores * scale)
            .map_err(|e| FerrumError::model(format!("attn scale failed: {}", e)))?;

        // Apply mask
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)
                .map_err(|e| FerrumError::model(format!("mask add failed: {}", e)))?
        } else {
            attn
        };

        // Softmax (manual implementation for Metal compatibility)
        let attn = {
            let max_val = attn.max_keepdim(D::Minus1)
                .map_err(|e| FerrumError::model(format!("softmax max failed: {}", e)))?;
            let diff = attn.broadcast_sub(&max_val)
                .map_err(|e| FerrumError::model(format!("softmax sub failed: {}", e)))?;
            let num = diff.exp()
                .map_err(|e| FerrumError::model(format!("softmax exp failed: {}", e)))?;
            let den = num.sum_keepdim(D::Minus1)
                .map_err(|e| FerrumError::model(format!("softmax sum failed: {}", e)))?;
            num.broadcast_div(&den)
                .map_err(|e| FerrumError::model(format!("softmax div failed: {}", e)))?
        };

        // attn @ v
        let output = attn.matmul(&v)
            .map_err(|e| FerrumError::model(format!("attn v matmul failed: {}", e)))?
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("output transpose failed: {}", e)))?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| FerrumError::model(format!("output reshape failed: {}", e)))?;

        self.o_proj.forward(&output)
            .map_err(|e| FerrumError::model(format!("o_proj forward failed: {}", e)))
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()
            .map_err(|e| FerrumError::model(format!("dims4 failed: {}", e)))?;

        // Use cat like the official implementation (faster and more compatible)
        let xs_vec: Vec<&Tensor> = vec![x; n_rep];
        Tensor::cat(&xs_vec, 2)
            .map_err(|e| FerrumError::model(format!("cat failed: {}", e)))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
            .map_err(|e| FerrumError::model(format!("reshape failed: {}", e)))
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Qwen2 MLP layer
struct MetalQwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MetalQwen2MLP {
    fn load(vb: VarBuilder, config: &MetalQwen2Config) -> Result<Self> {
        // Qwen2 MLP typically doesn't use bias
        let gate_proj = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| FerrumError::model(format!("gate_proj load failed: {}", e)))?;
        let up_proj = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))
            .map_err(|e| FerrumError::model(format!("up_proj load failed: {}", e)))?;
        let down_proj = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))
            .map_err(|e| FerrumError::model(format!("down_proj load failed: {}", e)))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Match official implementation exactly: gate.apply(act_fn) * up
        let gate = self.gate_proj.forward(x)
            .map_err(|e| FerrumError::model(format!("gate_proj forward failed: {}", e)))?;
        // Use Candle's silu for better compatibility
        let gate_silu = gate.silu()
            .map_err(|e| FerrumError::model(format!("gate silu failed: {}", e)))?;
        
        let up = self.up_proj.forward(x)
            .map_err(|e| FerrumError::model(format!("up_proj forward failed: {}", e)))?;
        let hidden = (gate_silu * up)
            .map_err(|e| FerrumError::model(format!("gate * up failed: {}", e)))?;
        self.down_proj.forward(&hidden)
            .map_err(|e| FerrumError::model(format!("down_proj forward failed: {}", e)))
    }
}

/// Qwen2 Decoder Layer with Metal acceleration
struct MetalQwen2DecoderLayer {
    self_attn: MetalQwen2Attention,
    mlp: MetalQwen2MLP,
    input_layernorm: MetalRmsNorm,
    post_attention_layernorm: MetalRmsNorm,
}

impl MetalQwen2DecoderLayer {
    fn load(
        vb: VarBuilder,
        config: &MetalQwen2Config,
        metal_ops: Option<Arc<RmsNormOps>>,
    ) -> Result<Self> {
        let self_attn = MetalQwen2Attention::load(vb.pp("self_attn"), config)?;
        let mlp = MetalQwen2MLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = MetalRmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
            metal_ops.clone(),
        )?;
        let post_attention_layernorm = MetalRmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
            metal_ops,
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
        x: &Tensor,
        rope: &Qwen2RotaryEmbedding,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm + attention + residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&x, rope, start_pos, mask)?;
        let x = (residual + attn_out)
            .map_err(|e| FerrumError::model(format!("residual add 1 failed: {}", e)))?;

        // Pre-norm + MLP + residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x)?;
        (residual + mlp_out)
            .map_err(|e| FerrumError::model(format!("residual add 2 failed: {}", e)))
    }

    #[allow(dead_code)]
    fn forward_debug(
        &mut self,
        x: &Tensor,
        rope: &Qwen2RotaryEmbedding,
        start_pos: usize,
        mask: Option<&Tensor>,
        _layer_idx: usize,
    ) -> Result<Tensor> {
        // Simplified debug forward - same as regular forward
        self.forward(x, rope, start_pos, mask)
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

/// Metal-accelerated Qwen2 Model
pub struct MetalQwen2Model {
    embed_tokens: Embedding,
    layers: Vec<MetalQwen2DecoderLayer>,
    norm: MetalRmsNorm,
    lm_head: Linear,
    rope: Qwen2RotaryEmbedding,
    config: MetalQwen2Config,
    device: CandleDevice,
    dtype: DType,
}

impl MetalQwen2Model {
    pub fn load(
        vb: VarBuilder,
        config: MetalQwen2Config,
        device: &CandleDevice,
        metal_context: Option<Arc<MetalContext>>,
    ) -> Result<Self> {
        info!("🔨 Loading Metal-accelerated Qwen2 model...");

        // Get dtype from VarBuilder
        let dtype = vb.dtype();

        // Initialize Metal RMS Norm ops if available
        // DISABLED FOR DEBUGGING: Force use Candle's native RMS Norm
        let metal_ops: Option<Arc<RmsNormOps>> = None;
        info!("Using Candle native RMS Norm (Metal kernel disabled for debugging)");
        let _ = metal_context; // suppress unused warning

        // Create rotary embeddings
        let rope = Qwen2RotaryEmbedding::new(&config, dtype, device)?;

        // Load embedding layer
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))
            .map_err(|e| FerrumError::model(format!("embed_tokens load failed: {}", e)))?;

        // Load decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            debug!("Loading Qwen2 layer {}/{}", i + 1, config.num_hidden_layers);
            let layer = MetalQwen2DecoderLayer::load(
                vb.pp(format!("model.layers.{}", i)),
                &config,
                metal_ops.clone(),
            )?;
            layers.push(layer);
        }

        // Load final norm
        let norm = MetalRmsNorm::load(
            vb.pp("model.norm"),
            config.hidden_size,
            config.rms_norm_eps,
            metal_ops,
        )?;

        // Load LM head: prefer dedicated lm_head weight if present, else fall back to tied embedding.
        let lm_head = match linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head")) {
            Ok(head) => {
                info!("Using dedicated lm_head weight");
                head
            }
            Err(e) => {
                info!("lm_head weight missing ({}), falling back to embed_tokens weight", e);
                let embed_weight = vb
                    .pp("model.embed_tokens")
                    .get((config.vocab_size, config.hidden_size), "weight")
                    .map_err(|e| FerrumError::model(format!("embed weight for lm_head: {}", e)))?;
                info!("Embed weight shape: {:?}", embed_weight.shape());
                Linear::new(embed_weight, None)
            }
        };

        info!("✅ Metal Qwen2 model loaded with {} layers", config.num_hidden_layers);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config,
            device: device.clone(),
            dtype,
        })
    }

    /// Forward pass for prefill
    pub fn forward_prefill(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)
            .map_err(|e| FerrumError::model(format!("dim failed: {}", e)))?;

        // Embed tokens
        let mut x = self.embed_tokens.forward(input_ids)
            .map_err(|e| FerrumError::model(format!("embed forward failed: {}", e)))?;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len)?;

        // Apply decoder layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == 0 {
                // Debug first layer
                x = layer.forward_debug(&x, &self.rope, 0, Some(&mask), i)?;
            } else {
                x = layer.forward(&x, &self.rope, 0, Some(&mask))?;
            }
        }

        // Final norm
        x = self.norm.forward(&x)?;

        // LM head (only last token for efficiency)
        let last_hidden = x.i((.., seq_len - 1.., ..))
            .map_err(|e| FerrumError::model(format!("last hidden slice failed: {}", e)))?;
        
        self.lm_head.forward(&last_hidden)
            .map_err(|e| FerrumError::model(format!("lm_head forward failed: {}", e)))
    }

    /// Forward pass for decode (single token)
    pub fn forward_decode(&mut self, token_id: &Tensor, pos: usize) -> Result<Tensor> {
        // Embed token
        let mut x = self.embed_tokens.forward(token_id)
            .map_err(|e| FerrumError::model(format!("embed forward failed: {}", e)))?;

        // No mask needed for single token decode with KV cache
        for layer in &mut self.layers {
            x = layer.forward(&x, &self.rope, pos, None)?;
        }

        // Final norm
        x = self.norm.forward(&x)?;

        // LM head
        self.lm_head.forward(&x)
            .map_err(|e| FerrumError::model(format!("lm_head forward failed: {}", e)))
    }

    /// Clear KV cache
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j <= i { 0.0 } else { f32::NEG_INFINITY }
                })
            })
            .collect();

        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), &self.device)
            .map_err(|e| FerrumError::model(format!("mask tensor failed: {}", e)))?
            .to_dtype(self.dtype)
            .map_err(|e| FerrumError::model(format!("mask dtype failed: {}", e)))
    }

    pub fn config(&self) -> &MetalQwen2Config {
        &self.config
    }
}

impl MetalQwen2Config {
    /// Create config from ModelDefinition
    pub fn from_model_def(model_def: &ferrum_models::ModelDefinition) -> Self {
        // Read tie_word_embeddings from extra_params (original config.json)
        let tie_word_embeddings = model_def.extra_params
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        Self {
            vocab_size: model_def.vocab_size,
            hidden_size: model_def.hidden_size,
            intermediate_size: model_def.intermediate_size,
            num_hidden_layers: model_def.num_hidden_layers,
            num_attention_heads: model_def.num_attention_heads,
            num_key_value_heads: model_def.num_key_value_heads.unwrap_or(model_def.num_attention_heads),
            max_position_embeddings: model_def.max_position_embeddings,
            rms_norm_eps: model_def.norm_eps,
            rope_theta: model_def.rope_theta.unwrap_or(1000000.0) as f32,
            use_bias: true,
            tie_word_embeddings,
        }
    }
}

// ============================================================================
// Metal Qwen2 Executor
// ============================================================================

use async_trait::async_trait;
use ferrum_interfaces::kv_cache::CacheHandleStats;
use ferrum_interfaces::model_executor::{
    DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
    ExecutorState, ExecutorStatus, MemoryRequirements, AttentionType,
};
use ferrum_interfaces::{KvCacheHandle, ModelExecutor, PrefillInput, PrefillOutput, TensorRef, BlockTable};
use ferrum_types::{DataType, Device, ModelInfo};
use half::f16;

/// Simple tensor implementation for executor outputs
#[derive(Debug)]
struct SimpleTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl SimpleTensor {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl ferrum_interfaces::TensorLike for SimpleTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        DataType::FP32
    }

    fn device(&self) -> Device {
        Device::Metal
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn view(&self, start: &[usize], end: &[usize]) -> Result<TensorRef> {
        if start.len() != end.len() || start.len() != self.shape.len() {
            return Err(FerrumError::backend("Invalid view dimensions"));
        }
        let mut stride = 1usize;
        let mut offset = 0usize;
        for dim in (0..self.shape.len()).rev() {
            offset += start[dim] * stride;
            stride *= self.shape[dim];
        }
        let count: usize = start.iter().zip(end.iter()).map(|(s, e)| e - s).product();
        let end_offset = offset + count;
        let data = self
            .data
            .get(offset..end_offset)
            .ok_or_else(|| FerrumError::backend("View slice out of range"))?
            .to_vec();
        Ok(Arc::new(SimpleTensor::new(
            data,
            end.iter().zip(start.iter()).map(|(e, s)| e - s).collect(),
        )))
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorRef> {
        let new_numel: usize = shape.iter().product();
        if new_numel != self.numel() {
            return Err(FerrumError::backend("Reshape numel mismatch"));
        }
        Ok(Arc::new(SimpleTensor::new(self.data.clone(), shape.to_vec())))
    }

    fn to_cpu(&self) -> Result<TensorRef> {
        Ok(Arc::new(SimpleTensor::new(self.data.clone(), self.shape.clone())))
    }

    fn to_device(&self, _device: &Device) -> Result<TensorRef> {
        Ok(Arc::new(SimpleTensor::new(self.data.clone(), self.shape.clone())))
    }

    fn to_dtype(&self, dtype: DataType) -> Result<TensorRef> {
        match dtype {
            DataType::FP32 | DataType::FP16 | DataType::BF16 | DataType::FP8 => {
                Ok(Arc::new(SimpleTensor::new(self.data.clone(), self.shape.clone())))
            }
            _ => Err(FerrumError::backend("Unsupported dtype conversion")),
        }
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        Ok(self.data.iter().map(|x| *x as u32).collect())
    }
}

/// Dummy KV cache handle for Metal Qwen2 executor
#[derive(Debug, Clone)]
struct DummyKvCache {
    block_table: BlockTable,
}

impl DummyKvCache {
    fn new(block_size: usize) -> Self {
        Self {
            block_table: BlockTable::new(block_size),
        }
    }
    
    fn with_length(block_size: usize, sequence_length: usize) -> Self {
        let mut block_table = BlockTable::new(block_size);
        block_table.sequence_length = sequence_length;
        Self { block_table }
    }
}

impl KvCacheHandle for DummyKvCache {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Device {
        Device::Metal
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn num_heads(&self) -> usize {
        0
    }

    fn head_dim(&self) -> usize {
        0
    }

    fn key_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn stats(&self) -> CacheHandleStats {
        CacheHandleStats {
            memory_bytes: 0,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.block_table.sequence_length,
            utilization: 0.0,
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        "metal_qwen2_dummy_cache".to_string()
    }
}

/// Metal-accelerated Qwen2 Executor
pub struct MetalQwen2Executor {
    model: MetalQwen2Model,
    info: ModelInfo,
    device: CandleDevice,
    status: ExecutorStatus,
    cpu_ref: Option<Qwen2ModelWrapper>,
}

impl MetalQwen2Executor {
    /// Create a new Metal Qwen2 executor
    pub fn new(
        model: MetalQwen2Model,
        model_info: ModelInfo,
        device: CandleDevice,
        cpu_ref: Option<Qwen2ModelWrapper>,
    ) -> Self {
        info!(
            "Created MetalQwen2Executor for model: {}, cpu_ref={}",
            model_info.model_id,
            cpu_ref.is_some()
        );

        let status = ExecutorStatus {
            state: ExecutorState::Ready,
            is_ready: true,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: 0,
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: 0,
                used_bytes: 0,
                peak_bytes: 0,
                utilization_percent: 0.0,
            },
            last_operation: None,
        };

        Self {
            model,
            info: model_info,
            device,
            status,
            cpu_ref,
        }
    }

    /// Load a Metal Qwen2 model from path
    pub async fn from_path(
        model_path: &str,
        model_def: &ferrum_models::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Loading Metal Qwen2 model from: {}", model_path);

        // Initialize Metal context
        let metal_context = {
            let mut ctx = MetalContext::new()?;
            ctx.load_shader_library()?;
            Some(Arc::new(ctx))
        };

        // Create Qwen2 config from model definition
        let config = MetalQwen2Config::from_model_def(model_def);

        // Load weights
        let loader = ferrum_models::SafeTensorsLoader::new(model_path);
        let vb = loader.load_varbuilder(&device, dtype)?;

        // Create model
        let model = MetalQwen2Model::load(vb, config.clone(), &device, metal_context)?;

        // Optional CPU reference model for debug comparison
        let cpu_ref = if std::env::var("FERRUM_QWEN2_COMPARE_CPU")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            info!("FERRUM_QWEN2_COMPARE_CPU=1: loading CPU reference model for comparison");
            let loader_cpu = ferrum_models::SafeTensorsLoader::new(model_path);
            let vb_cpu = loader_cpu.load_varbuilder(&CandleDevice::Cpu, DType::F32)?;
            let cpu_model =
                ferrum_models::Qwen2ModelWrapper::from_varbuilder(vb_cpu, model_def, CandleDevice::Cpu, DType::F32)
                    .map_err(|e| FerrumError::model(format!("CPU ref load failed: {}", e)))?;
            Some(cpu_model)
        } else {
            None
        };

        // Create model info
        let model_info = model_def.to_model_info(model_path.to_string());

        Ok(Self::new(model, model_info, device, cpu_ref))
    }

    /// Extract token IDs from TensorRef
    fn extract_token_ids(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        if let Ok(v) = tensor.to_vec_u32() {
            return Ok(v);
        }
        if let Ok(vf) = tensor.to_vec_f32() {
            return Ok(vf.into_iter().map(|x| x as u32).collect());
        }
        Err(FerrumError::backend("Unable to extract token ids from tensor"))
    }
}

#[async_trait]
impl ModelExecutor for MetalQwen2Executor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let token_ids = self.extract_token_ids(&input.input_ids)?;
        let seq_len = token_ids.len();

        debug!("Metal Qwen2 prefill: {} tokens", seq_len);

        // Get mutable reference for model operations
        let model_ptr = &self.model as *const MetalQwen2Model as *mut MetalQwen2Model;
        
        // Clear KV cache before new prefill (important for multi-turn conversations)
        unsafe { (*model_ptr).clear_cache() };

        // Convert token IDs to tensor
        let input_tensor = Tensor::from_vec(token_ids.clone(), (1, seq_len), &self.device)
            .map_err(|e| FerrumError::internal(format!("Failed to create input tensor: {}", e)))?;

        // Forward pass
        let logits = unsafe { (*model_ptr).forward_prefill(&input_tensor)? };

        // Optional CPU reference comparison
        if let Some(ref cpu_model) = self.cpu_ref {
            if seq_len <= 64 {
                let input_cpu = Tensor::from_vec(token_ids.clone(), (1, seq_len), &CandleDevice::Cpu)
                    .map_err(|e| FerrumError::internal(format!("cpu input tensor failed: {}", e)))?;
                if let Ok(cpu_logits) = cpu_model.forward_prefill(&input_cpu) {
                    if let (Ok(metal_vals), Ok(cpu_vals)) = (
                        logits.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()),
                        cpu_logits.flatten_all().and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.to_vec1::<f32>()),
                    ) {
                        let diff: f32 = metal_vals
                            .iter()
                            .zip(cpu_vals.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            .sqrt();
                        let mut metal_top: Vec<(usize, f32)> =
                            metal_vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                        metal_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let mut cpu_top: Vec<(usize, f32)> =
                            cpu_vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                        cpu_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        info!(
                            "CPU vs Metal logits L2 diff: {:.4}; top_metal={:?}; top_cpu={:?}; tokens 17/19/28 metal=({:.4},{:.4},{:.4}) cpu=({:.4},{:.4},{:.4})",
                            diff,
                            &metal_top[..5.min(metal_top.len())],
                            &cpu_top[..5.min(cpu_top.len())],
                            metal_vals[17], metal_vals[19], metal_vals[28],
                            cpu_vals[17], cpu_vals[19], cpu_vals[28],
                        );
                    }
                }
            }
        }

        debug!(
            "prefill logits shape={:?}, dtype={:?}",
            logits.dims(),
            logits.dtype()
        );

        // Get logits - shape should be [1, 1, vocab_size] (already sliced to last token)
        let logits_dims = logits.dims();
        let vocab_size = *logits_dims.last().unwrap_or(&0);

        // Get logits on CPU
        let logits_cpu = logits
            .to_device(&CandleDevice::Cpu)
            .map_err(|e| FerrumError::internal(format!("To CPU failed: {}", e)))?;

        let last_logits = logits_cpu
            .flatten_all()
            .map_err(|e| FerrumError::internal(format!("Flatten error: {}", e)))?;

        let dtype = last_logits.dtype();

        // Convert to f32
        let logits_vec: Vec<f32> = if dtype == DType::F32 {
            last_logits
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("To vec f32 error: {}", e)))?
        } else if dtype == DType::F16 {
            let v16: Vec<f16> = last_logits
                .to_vec1()
                .map_err(|e| FerrumError::internal(format!("To vec f16 error: {}", e)))?;
            v16.iter().map(|v| f32::from(*v)).collect()
        } else {
            last_logits
                .to_dtype(DType::F32)
                .unwrap_or(last_logits)
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("Fallback to vec f32 error: {}", e)))?
        };

        // Create output tensor
        let output_tensor: TensorRef = Arc::new(SimpleTensor::new(
            logits_vec,
            vec![1, vocab_size],
        ));

        // Create KV cache handle with correct sequence length
        let kv_cache: Arc<dyn KvCacheHandle> = Arc::new(DummyKvCache::with_length(16, seq_len));

        Ok(PrefillOutput::new(output_tensor, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let token_ids = self.extract_token_ids(&input.input_ids)?;
        let position = input.kv_cache.num_tokens();

        debug!("Metal Qwen2 decode: position {}", position);

        // Convert token ID to tensor
        let token_tensor = Tensor::from_vec(token_ids, (1, 1), &self.device)
            .map_err(|e| FerrumError::internal(format!("Failed to create token tensor: {}", e)))?;

        // Get mutable reference for forward pass
        let model_ptr = &self.model as *const MetalQwen2Model as *mut MetalQwen2Model;
        let logits = unsafe { (*model_ptr).forward_decode(&token_tensor, position)? };

        debug!(
            "decode logits shape={:?}, dtype={:?}",
            logits.dims(),
            logits.dtype()
        );

        // Get logits on CPU
        let logits_cpu = logits
            .to_device(&CandleDevice::Cpu)
            .map_err(|e| FerrumError::internal(format!("To CPU failed: {}", e)))?;

        let last_logits = logits_cpu
            .flatten_all()
            .map_err(|e| FerrumError::internal(format!("Flatten error: {}", e)))?;

        let dtype = last_logits.dtype();

        let logits_vec: Vec<f32> = if dtype == DType::F32 {
            last_logits
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("To vec f32 error: {}", e)))?
        } else if dtype == DType::F16 {
            let v16: Vec<f16> = last_logits
                .to_vec1()
                .map_err(|e| FerrumError::internal(format!("To vec f16 error: {}", e)))?;
            v16.iter().map(|v| f32::from(*v)).collect()
        } else {
            last_logits
                .to_dtype(DType::F32)
                .unwrap_or(last_logits)
                .to_vec1::<f32>()
                .map_err(|e| FerrumError::internal(format!("Fallback to vec f32 error: {}", e)))?
        };

        let vocab_size = logits_vec.len();
        let output_tensor: TensorRef = Arc::new(SimpleTensor::new(
            logits_vec,
            vec![1, vocab_size],
        ));

        // Create new kv_cache with incremented sequence length
        let new_length = position + 1;
        let kv_cache: Arc<dyn KvCacheHandle> = Arc::new(DummyKvCache::with_length(16, new_length));

        Ok(DecodeOutput::new(output_tensor, kv_cache))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.model.config().max_position_embeddings,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32, DataType::FP16],
            supported_devices: vec![Device::Metal],
            memory_requirements: MemoryRequirements {
                parameter_memory: 0,
                activation_memory_per_token: 0,
                kv_cache_memory_per_token: 0,
                overhead_memory: 0,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        self.status.clone()
    }
}

