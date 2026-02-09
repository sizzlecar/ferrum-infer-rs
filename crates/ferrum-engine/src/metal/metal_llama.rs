//! Metal-accelerated LLaMA model implementation
//!
//! This module provides a LLaMA model that uses custom Metal kernels
//! for operations like RMS Norm, instead of relying on Candle's built-in.

use crate::metal::{MetalContext, RmsNormOps};
use candle_core::{DType, Device as CandleDevice, IndexOp, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Metal-accelerated RMS Normalization layer
pub struct MetalRmsNorm {
    weight: Tensor,
    eps: f32,
    metal_ops: Option<Arc<RmsNormOps>>,
}

impl MetalRmsNorm {
    pub fn new(weight: Tensor, eps: f64, metal_ops: Option<Arc<RmsNormOps>>) -> Self {
        Self {
            weight,
            eps: eps as f32,
            metal_ops,
        }
    }

    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        eps: f64,
        metal_ops: Option<Arc<RmsNormOps>>,
    ) -> Result<Self> {
        let weight = vb
            .get(hidden_size, "weight")
            .map_err(|e| FerrumError::model(format!("Failed to load rms_norm weight: {}", e)))?;
        Ok(Self::new(weight, eps, metal_ops))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // If Metal kernel is available and input is on Metal, use Metal kernel
        if let Some(ref metal_ops) = self.metal_ops {
            if matches!(x.device(), CandleDevice::Metal(_)) {
                return self.forward_metal(x, metal_ops);
            }
        }

        // Default: use Candle implementation (works on any device including Metal)
        self.forward_candle(x)
    }

    fn forward_metal(&self, x: &Tensor, metal_ops: &RmsNormOps) -> Result<Tensor> {
        let dims = x.dims();
        let hidden_size = *dims.last().unwrap();
        let batch_size: usize = dims[..dims.len() - 1].iter().product();

        // Flatten to 2D for processing
        let x_flat = x
            .flatten_all()
            .map_err(|e| FerrumError::internal(format!("Flatten failed: {}", e)))?;

        // Convert to f32 vec for Metal - handle FP16 inputs
        let x_data: Vec<f32> = match x_flat.dtype() {
            DType::F32 => x_flat
                .to_vec1()
                .map_err(|e| FerrumError::internal(format!("To vec f32 failed: {}", e)))?,
            DType::F16 => {
                debug!("forward_metal: converting FP16 input to FP32");
                let v16: Vec<f16> = x_flat
                    .to_vec1()
                    .map_err(|e| FerrumError::internal(format!("To vec f16 failed: {}", e)))?;
                v16.iter().map(|v| f32::from(*v)).collect()
            }
            dtype => {
                warn!(
                    "forward_metal: unexpected dtype {:?}, casting to F32",
                    dtype
                );
                let converted = x_flat
                    .to_dtype(DType::F32)
                    .map_err(|e| FerrumError::internal(format!("Dtype cast failed: {}", e)))?;
                converted.to_vec1().map_err(|e| {
                    FerrumError::internal(format!("To vec after cast failed: {}", e))
                })?
            }
        };

        // Convert weight to f32 - handle FP16 weights
        let weight_data: Vec<f32> = match self.weight.dtype() {
            DType::F32 => self
                .weight
                .to_vec1()
                .map_err(|e| FerrumError::internal(format!("Weight to vec f32 failed: {}", e)))?,
            DType::F16 => {
                debug!("forward_metal: converting FP16 weight to FP32");
                let v16: Vec<f16> = self.weight.to_vec1().map_err(|e| {
                    FerrumError::internal(format!("Weight to vec f16 failed: {}", e))
                })?;
                v16.iter().map(|v| f32::from(*v)).collect()
            }
            dtype => {
                warn!(
                    "forward_metal: unexpected weight dtype {:?}, casting to F32",
                    dtype
                );
                let converted = self.weight.to_dtype(DType::F32).map_err(|e| {
                    FerrumError::internal(format!("Weight dtype cast failed: {}", e))
                })?;
                converted.to_vec1().map_err(|e| {
                    FerrumError::internal(format!("Weight to vec after cast failed: {}", e))
                })?
            }
        };

        // Execute Metal RMS Norm (result is f32)
        let result = metal_ops.forward(&x_data, &weight_data, batch_size, hidden_size, self.eps)?;

        // Convert back to Tensor with original dtype
        let result_tensor = Tensor::from_vec(result, dims, x.device())
            .map_err(|e| FerrumError::internal(format!("Tensor from vec failed: {}", e)))?;

        // Convert back to original dtype if needed
        let original_dtype = x.dtype();
        if original_dtype != DType::F32 {
            debug!(
                "forward_metal: converting result back to {:?}",
                original_dtype
            );
            result_tensor.to_dtype(original_dtype).map_err(|e| {
                FerrumError::internal(format!("Result dtype conversion failed: {}", e))
            })
        } else {
            Ok(result_tensor)
        }
    }

    fn forward_candle(&self, x: &Tensor) -> Result<Tensor> {
        // Matching candle_nn::ops::rms_norm_slow exactly:
        // For FP16/BF16, compute in F32 for numerical stability
        let x_dtype = x.dtype();
        let hidden_size = x.dims().last().copied().unwrap_or(1);

        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let x = x
            .to_dtype(internal_dtype)
            .map_err(|e| FerrumError::internal(format!("x to internal dtype failed: {}", e)))?;

        // norm_x = sum(x^2) / hidden_size
        let norm_x = x
            .sqr()
            .map_err(|e| FerrumError::internal(format!("Sqr failed: {}", e)))?
            .sum_keepdim(D::Minus1)
            .map_err(|e| FerrumError::internal(format!("Sum failed: {}", e)))?
            .affine(1.0 / hidden_size as f64, 0.0)
            .map_err(|e| FerrumError::internal(format!("Affine failed: {}", e)))?;

        // x_normed = x / sqrt(norm_x + eps)
        let x_normed = x
            .broadcast_div(
                &(norm_x + self.eps as f64)
                    .map_err(|e| FerrumError::internal(format!("Add eps failed: {}", e)))?
                    .sqrt()
                    .map_err(|e| FerrumError::internal(format!("Sqrt failed: {}", e)))?,
            )
            .map_err(|e| FerrumError::internal(format!("Div failed: {}", e)))?;

        // Convert back to original dtype and apply weight
        x_normed
            .to_dtype(x_dtype)
            .map_err(|e| {
                FerrumError::internal(format!("x_normed to original dtype failed: {}", e))
            })?
            .broadcast_mul(&self.weight)
            .map_err(|e| FerrumError::internal(format!("Mul weight failed: {}", e)))
    }
}

/// Metal-accelerated Rotary Position Embedding
pub struct MetalRotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    head_dim: usize,
}

impl MetalRotaryEmbedding {
    pub fn new(
        max_seq_len: usize,
        head_dim: usize,
        theta: f32,
        device: &CandleDevice,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute frequencies: theta^(-2i/d) for i in 0..d/2
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)
            .map_err(|e| FerrumError::internal(format!("inv_freq tensor failed: {}", e)))?;

        // Compute positions
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, max_seq_len, device)
            .map_err(|e| FerrumError::internal(format!("positions tensor failed: {}", e)))?;

        // freqs = positions * inv_freq^T -> [max_seq_len, half_dim]
        let freqs = positions
            .unsqueeze(1)
            .map_err(|e| FerrumError::internal(format!("unsqueeze failed: {}", e)))?
            .broadcast_mul(
                &inv_freq
                    .unsqueeze(0)
                    .map_err(|e| FerrumError::internal(format!("unsqueeze failed: {}", e)))?,
            )
            .map_err(|e| FerrumError::internal(format!("broadcast_mul failed: {}", e)))?;

        let cos_cache = freqs
            .cos()
            .map_err(|e| FerrumError::internal(format!("cos failed: {}", e)))?;
        let sin_cache = freqs
            .sin()
            .map_err(|e| FerrumError::internal(format!("sin failed: {}", e)))?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Apply rotary embedding to query/key tensors
    /// x shape: [batch, seq_len, num_heads, head_dim]
    pub fn forward(&self, x: &Tensor, position_ids: &[usize]) -> Result<Tensor> {
        let half_dim = self.head_dim / 2;
        let _dims = x.dims();
        let target_dtype = x.dtype();

        // Get cos/sin for positions and convert to input dtype if needed
        let cos = self
            .cos_cache
            .i((position_ids[0], ..))
            .map_err(|e| FerrumError::internal(format!("cos index failed: {}", e)))?;
        let cos = if cos.dtype() != target_dtype {
            cos.to_dtype(target_dtype)
                .map_err(|e| FerrumError::internal(format!("cos dtype conversion failed: {}", e)))?
        } else {
            cos
        };

        let sin = self
            .sin_cache
            .i((position_ids[0], ..))
            .map_err(|e| FerrumError::internal(format!("sin index failed: {}", e)))?;
        let sin = if sin.dtype() != target_dtype {
            sin.to_dtype(target_dtype)
                .map_err(|e| FerrumError::internal(format!("sin dtype conversion failed: {}", e)))?
        } else {
            sin
        };

        // Split x into two halves
        let x1 = x
            .narrow(D::Minus1, 0, half_dim)
            .map_err(|e| FerrumError::internal(format!("narrow x1 failed: {}", e)))?;
        let x2 = x
            .narrow(D::Minus1, half_dim, half_dim)
            .map_err(|e| FerrumError::internal(format!("narrow x2 failed: {}", e)))?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = x1
            .broadcast_mul(&cos)
            .map_err(|e| FerrumError::internal(format!("mul cos failed: {}", e)))?
            .broadcast_sub(
                &x2.broadcast_mul(&sin)
                    .map_err(|e| FerrumError::internal(format!("mul sin failed: {}", e)))?,
            )
            .map_err(|e| FerrumError::internal(format!("sub failed: {}", e)))?;

        let rotated_x2 = x1
            .broadcast_mul(&sin)
            .map_err(|e| FerrumError::internal(format!("mul sin failed: {}", e)))?
            .broadcast_add(
                &x2.broadcast_mul(&cos)
                    .map_err(|e| FerrumError::internal(format!("mul cos failed: {}", e)))?,
            )
            .map_err(|e| FerrumError::internal(format!("add failed: {}", e)))?;

        Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1)
            .map_err(|e| FerrumError::internal(format!("cat failed: {}", e)))
    }
}

/// Metal LLaMA model configuration
#[derive(Clone, Debug)]
pub struct MetalLlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

/// Metal-accelerated LLaMA Attention layer
pub struct MetalLlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: MetalRotaryEmbedding,
}

impl MetalLlamaAttention {
    pub fn load(vb: VarBuilder, config: &MetalLlamaConfig, device: &CandleDevice) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / num_heads;

        let q_proj = linear_no_bias(config.hidden_size, num_heads * head_dim, vb.pp("q_proj"))
            .map_err(|e| FerrumError::model(format!("q_proj load failed: {}", e)))?;
        let k_proj = linear_no_bias(config.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))
            .map_err(|e| FerrumError::model(format!("k_proj load failed: {}", e)))?;
        let v_proj = linear_no_bias(config.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))
            .map_err(|e| FerrumError::model(format!("v_proj load failed: {}", e)))?;
        let o_proj = linear_no_bias(num_heads * head_dim, config.hidden_size, vb.pp("o_proj"))
            .map_err(|e| FerrumError::model(format!("o_proj load failed: {}", e)))?;

        let rotary_emb = MetalRotaryEmbedding::new(
            config.max_position_embeddings,
            head_dim,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &[usize],
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| FerrumError::internal(format!("dims3 failed: {}", e)))?;

        // Project Q, K, V
        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| FerrumError::internal(format!("q_proj forward failed: {}", e)))?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| FerrumError::internal(format!("k_proj forward failed: {}", e)))?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| FerrumError::internal(format!("v_proj forward failed: {}", e)))?;

        // Reshape for multi-head attention
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| FerrumError::internal(format!("q reshape failed: {}", e)))?;
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| FerrumError::internal(format!("k reshape failed: {}", e)))?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| FerrumError::internal(format!("v reshape failed: {}", e)))?;

        // Apply rotary embeddings
        let q = self.rotary_emb.forward(&q, position_ids)?;
        let k = self.rotary_emb.forward(&k, position_ids)?;

        // Concatenate with KV cache if present
        let (k, v) = if let Some((k_cache, v_cache)) = kv_cache {
            let k = Tensor::cat(&[k_cache, &k], 1)
                .map_err(|e| FerrumError::internal(format!("k concat failed: {}", e)))?;
            let v = Tensor::cat(&[v_cache, &v], 1)
                .map_err(|e| FerrumError::internal(format!("v concat failed: {}", e)))?;
            (k, v)
        } else {
            (k, v)
        };

        // Transpose for attention: [batch, num_heads, seq_len, head_dim]
        let q = q
            .transpose(1, 2)
            .map_err(|e| FerrumError::internal(format!("q transpose failed: {}", e)))?;
        let k = k
            .transpose(1, 2)
            .map_err(|e| FerrumError::internal(format!("k transpose failed: {}", e)))?;
        let v = v
            .transpose(1, 2)
            .map_err(|e| FerrumError::internal(format!("v transpose failed: {}", e)))?;

        // Handle GQA by repeating KV heads
        debug!("Before repeat_kv: k={:?}, v={:?}", k.dims(), v.dims());
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;
        debug!("After repeat_kv: k={:?}, v={:?}", k.dims(), v.dims());

        // Compute attention scores - ensure tensors are contiguous for Metal matmul
        let scale = (self.head_dim as f64).sqrt();
        let q = q
            .contiguous()
            .map_err(|e| FerrumError::internal(format!("q contiguous failed: {}", e)))?;
        let k = k
            .contiguous()
            .map_err(|e| FerrumError::internal(format!("k contiguous failed: {}", e)))?;
        let k_t = k
            .transpose(D::Minus2, D::Minus1)
            .map_err(|e| FerrumError::internal(format!("k transpose failed: {}", e)))?
            .contiguous()
            .map_err(|e| FerrumError::internal(format!("k_t contiguous failed: {}", e)))?;
        debug!("Attention matmul: q={:?}, k_t={:?}", q.dims(), k_t.dims());
        let attn_weights = q
            .matmul(&k_t)
            .map_err(|e| FerrumError::internal(format!("matmul failed: {}", e)))?
            .affine(1.0 / scale, 0.0)
            .map_err(|e| FerrumError::internal(format!("affine failed: {}", e)))?;

        // Apply causal mask (TODO: proper implementation)
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)
            .map_err(|e| FerrumError::internal(format!("softmax failed: {}", e)))?;

        // Apply attention to values
        let attn_output = attn_weights
            .matmul(&v)
            .map_err(|e| FerrumError::internal(format!("attn matmul failed: {}", e)))?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| FerrumError::internal(format!("transpose failed: {}", e)))?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| FerrumError::internal(format!("reshape failed: {}", e)))?;

        // Output projection
        let output = self
            .o_proj
            .forward(&attn_output)
            .map_err(|e| FerrumError::internal(format!("o_proj forward failed: {}", e)))?;

        // Return output and updated KV cache
        let k_cache = k
            .transpose(1, 2)
            .map_err(|e| FerrumError::internal(format!("k cache transpose failed: {}", e)))?;
        let v_cache = v
            .transpose(1, 2)
            .map_err(|e| FerrumError::internal(format!("v cache transpose failed: {}", e)))?;

        Ok((output, k_cache, v_cache))
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x
            .dims4()
            .map_err(|e| FerrumError::internal(format!("dims4 failed: {}", e)))?;

        // Use repeat instead of expand to ensure contiguous memory
        // Candle's expand may create a view which Metal matmul doesn't handle well
        let x_unsqueezed = x
            .unsqueeze(2)
            .map_err(|e| FerrumError::internal(format!("unsqueeze failed: {}", e)))?;

        // Repeat along dimension 2 n_rep times
        let repeated: Vec<Tensor> = (0..n_rep).map(|_| x_unsqueezed.clone()).collect();
        let x_repeated = Tensor::cat(&repeated, 2)
            .map_err(|e| FerrumError::internal(format!("cat repeat failed: {}", e)))?;

        // Reshape to merge heads
        x_repeated
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
            .map_err(|e| FerrumError::internal(format!("reshape failed: {}", e)))
    }
}

/// Metal-accelerated LLaMA MLP
pub struct MetalLlamaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MetalLlamaMlp {
    pub fn load(vb: VarBuilder, config: &MetalLlamaConfig) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )
        .map_err(|e| FerrumError::model(format!("gate_proj load failed: {}", e)))?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )
        .map_err(|e| FerrumError::model(format!("up_proj load failed: {}", e)))?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )
        .map_err(|e| FerrumError::model(format!("down_proj load failed: {}", e)))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = self
            .gate_proj
            .forward(x)
            .map_err(|e| FerrumError::internal(format!("gate_proj forward failed: {}", e)))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| FerrumError::internal(format!("silu failed: {}", e)))?;

        let up = self
            .up_proj
            .forward(x)
            .map_err(|e| FerrumError::internal(format!("up_proj forward failed: {}", e)))?;

        let hidden = gate
            .mul(&up)
            .map_err(|e| FerrumError::internal(format!("mul failed: {}", e)))?;

        self.down_proj
            .forward(&hidden)
            .map_err(|e| FerrumError::internal(format!("down_proj forward failed: {}", e)))
    }
}

/// Metal-accelerated LLaMA Decoder Layer
pub struct MetalLlamaDecoderLayer {
    self_attn: MetalLlamaAttention,
    mlp: MetalLlamaMlp,
    input_layernorm: MetalRmsNorm,
    post_attention_layernorm: MetalRmsNorm,
}

impl MetalLlamaDecoderLayer {
    pub fn load(
        vb: VarBuilder,
        config: &MetalLlamaConfig,
        device: &CandleDevice,
        metal_ops: Option<Arc<RmsNormOps>>,
    ) -> Result<Self> {
        let self_attn = MetalLlamaAttention::load(vb.pp("self_attn"), config, device)?;
        let mlp = MetalLlamaMlp::load(vb.pp("mlp"), config)?;
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

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &[usize],
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Pre-normalization (LLaMA style)
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;

        // Self attention
        let (attn_output, k_cache, v_cache) =
            self.self_attn
                .forward(&hidden_states, position_ids, kv_cache)?;

        // Residual connection
        let hidden_states = residual
            .add(&attn_output)
            .map_err(|e| FerrumError::internal(format!("residual add failed: {}", e)))?;

        // Pre-normalization for MLP
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // MLP
        let mlp_output = self.mlp.forward(&hidden_states)?;

        // Residual connection
        let hidden_states = residual
            .add(&mlp_output)
            .map_err(|e| FerrumError::internal(format!("mlp residual add failed: {}", e)))?;

        Ok((hidden_states, k_cache, v_cache))
    }
}

/// Metal-accelerated LLaMA Model
pub struct MetalLlamaModel {
    embed_tokens: Embedding,
    layers: Vec<MetalLlamaDecoderLayer>,
    norm: MetalRmsNorm,
    lm_head: Linear,
    config: MetalLlamaConfig,
}

impl MetalLlamaModel {
    pub fn load(
        vb: VarBuilder,
        config: MetalLlamaConfig,
        device: &CandleDevice,
        metal_context: Option<Arc<MetalContext>>,
    ) -> Result<Self> {
        info!("🔨 Loading Metal-accelerated LLaMA model...");

        // Initialize Metal RMS Norm ops if available
        let metal_ops = if let Some(ctx) = metal_context {
            match RmsNormOps::new(ctx) {
                Ok(ops) => {
                    info!("✅ Metal RMS Norm acceleration enabled");
                    Some(Arc::new(ops))
                }
                Err(e) => {
                    debug!("Metal RMS Norm not available: {}, using CPU fallback", e);
                    None
                }
            }
        } else {
            None
        };

        // Load embedding layer
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )
        .map_err(|e| FerrumError::model(format!("embed_tokens load failed: {}", e)))?;

        // Load decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            debug!("Loading layer {}/{}", i + 1, config.num_hidden_layers);
            let layer = MetalLlamaDecoderLayer::load(
                vb.pp(format!("model.layers.{}", i)),
                &config,
                device,
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

        // Load LM head
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))
            .map_err(|e| FerrumError::model(format!("lm_head load failed: {}", e)))?;

        info!(
            "✅ Metal LLaMA model loaded with {} layers",
            config.num_hidden_layers
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids
            .dims2()
            .map_err(|e| FerrumError::internal(format!("dims2 failed: {}", e)))?;

        // Get embeddings
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(|e| FerrumError::internal(format!("embedding forward failed: {}", e)))?;

        // Position IDs
        let position_ids: Vec<usize> = (start_pos..start_pos + seq_len).collect();

        // Forward through layers
        for layer in &self.layers {
            let (output, _, _) = layer.forward(&hidden_states, &position_ids, None)?;
            hidden_states = output;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // LM head
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(|e| FerrumError::internal(format!("lm_head forward failed: {}", e)))?;

        Ok(logits)
    }

    pub fn config(&self) -> &MetalLlamaConfig {
        &self.config
    }
}
