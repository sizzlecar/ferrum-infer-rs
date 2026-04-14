//! Mimi-based speech tokenizer encoder for Qwen3-TTS ICL voice cloning.
//!
//! Takes raw 24kHz audio and outputs codec token indices [T, 16]
//! (1 semantic + 15 acoustic codebooks).
//!
//! Architecture: CausalConv encoder (15 layers, 960x downsample) →
//! Transformer (8 layers, sliding window) → Split RVQ quantizer.
//!
//! Loaded from speech_tokenizer/model.safetensors.

use candle_core::{DType, Device as CandleDevice, IndexOp, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use ferrum_types::{FerrumError, Result};
use tracing::info;

// ── Config ──────────────────────────────────────────────────────────────

const HIDDEN_SIZE: usize = 512;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const INTERMEDIATE_SIZE: usize = 2048;
const NUM_TRANSFORMER_LAYERS: usize = 8;
const ROPE_THETA: f64 = 10000.0;
const LN_EPS: f64 = 1e-5;
const SLIDING_WINDOW: usize = 250;
const SEMANTIC_CODEBOOK_SIZE: usize = 2048;
const ACOUSTIC_CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;
const NUM_ACOUSTIC_CODEBOOKS: usize = 31;
const NUM_OUTPUT_CODEBOOKS: usize = 16; // 1 semantic + 15 acoustic

// ── Manual softmax ──────────────────────────────────────────────────────

fn softmax_last_dim(x: &Tensor) -> candle_core::Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}

// ── Manual LayerNorm ────────────────────────────────────────────────────

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = diff.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let norm = norm.to_dtype(x_dtype)?;
        norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

// ── Manual Linear (no bias) ─────────────────────────────────────────────

struct LinearNoBias {
    weight: Tensor,
}

impl LinearNoBias {
    fn load(in_: usize, out: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get((out, in_), "weight")?;
        Ok(Self { weight })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let wt = self.weight.t()?;
        if x.dims().len() == 3 {
            let b = x.dim(0)?;
            x.matmul(&wt.broadcast_left(b)?)
        } else {
            x.matmul(&wt)
        }
    }
}

// ── CausalConv1d ────────────────────────────────────────────────────────

/// Causal convolution with left-padding only.
///
/// For stride=1: pad_left = (kernel_size - 1) * dilation
/// For stride>1 (downsample): pad_left = kernel_size - stride
struct CausalConv {
    conv: Conv1d,
    pad_left: usize,
    kernel_size: usize,
    stride: usize,
}

impl CausalConv {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let pad_left = if stride > 1 {
            kernel_size - stride
        } else {
            (kernel_size - 1) * dilation
        };
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let cv = vb.pp("conv");
        let w = cv.get((out_ch, in_ch, kernel_size), "weight")?;
        let b = cv.get(out_ch, "bias").ok();
        Ok(Self {
            conv: Conv1d::new(w, b, cfg),
            pad_left,
            kernel_size,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Match Python MimiConv1d: add extra right padding for strided convs
        // to ensure output = ceil(n_frames) instead of floor.
        let length = x.dim(2)?;
        let n_frames_f =
            (length + self.pad_left - self.kernel_size) as f64 / self.stride as f64 + 1.0;
        let n_frames_ceil = n_frames_f.ceil() as usize;
        let ideal_length = (n_frames_ceil - 1) * self.stride + self.kernel_size - self.pad_left;
        let extra_right = ideal_length.saturating_sub(length);
        let x = x.pad_with_zeros(2, self.pad_left, extra_right)?;
        self.conv.forward(&x)
    }
}

// ── Encoder ResBlock ────────────────────────────────────────────────────

/// ResBlock: ELU → Conv(k=3, dilation) → ELU → Conv(k=1) + skip.
///
/// block.0 = ELU (no weights)
/// block.1 = Conv(in→in/2, k=3, d=dilation)
/// block.2 = ELU (no weights)
/// block.3 = Conv(in/2→in, k=1)
struct EncoderResBlock {
    conv1: CausalConv, // block.1
    conv2: CausalConv, // block.3
}

impl EncoderResBlock {
    fn load(channels: usize, dilation: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let half = channels / 2;
        let conv1 = CausalConv::load(channels, half, 3, 1, dilation, vb.pp("block.1"))?;
        let conv2 = CausalConv::load(half, channels, 1, 1, 1, vb.pp("block.3"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        // block.0: ELU
        let h = x.elu(1.0)?;
        // block.1: Conv(k=3, dilation)
        let h = self.conv1.forward(&h)?;
        // block.2: ELU
        let h = h.elu(1.0)?;
        // block.3: Conv(k=1)
        let h = self.conv2.forward(&h)?;
        residual + h
    }
}

// ── Encoder Conv Stack ──────────────────────────────────────────────────

/// All 15 encoder conv layers in sequence.
enum EncoderLayer {
    Conv(CausalConv),
    ResBlock(EncoderResBlock),
    Elu,
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            EncoderLayer::Conv(c) => c.forward(x),
            EncoderLayer::ResBlock(r) => r.forward(x),
            EncoderLayer::Elu => x.elu(1.0),
        }
    }
}

struct EncoderConvStack {
    layers: Vec<EncoderLayer>,
}

impl EncoderConvStack {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(15);

        // Layer 0: Conv1d(1→64, k=7, s=1)
        layers.push(EncoderLayer::Conv(CausalConv::load(
            1,
            64,
            7,
            1,
            1,
            vb.pp("layers.0"),
        )?));

        // Layer 1: ResBlock(64, dilation=1)
        layers.push(EncoderLayer::ResBlock(EncoderResBlock::load(
            64,
            1,
            vb.pp("layers.1"),
        )?));

        // Layer 2: ELU
        layers.push(EncoderLayer::Elu);

        // Layer 3: Conv1d(64→128, k=8, s=4) downsample
        layers.push(EncoderLayer::Conv(CausalConv::load(
            64,
            128,
            8,
            4,
            1,
            vb.pp("layers.3"),
        )?));

        // Layer 4: ResBlock(128, dilation=1)
        layers.push(EncoderLayer::ResBlock(EncoderResBlock::load(
            128,
            1,
            vb.pp("layers.4"),
        )?));

        // Layer 5: ELU
        layers.push(EncoderLayer::Elu);

        // Layer 6: Conv1d(128→256, k=10, s=5) downsample
        layers.push(EncoderLayer::Conv(CausalConv::load(
            128,
            256,
            10,
            5,
            1,
            vb.pp("layers.6"),
        )?));

        // Layer 7: ResBlock(256, dilation=1)
        layers.push(EncoderLayer::ResBlock(EncoderResBlock::load(
            256,
            1,
            vb.pp("layers.7"),
        )?));

        // Layer 8: ELU
        layers.push(EncoderLayer::Elu);

        // Layer 9: Conv1d(256→512, k=12, s=6) downsample
        layers.push(EncoderLayer::Conv(CausalConv::load(
            256,
            512,
            12,
            6,
            1,
            vb.pp("layers.9"),
        )?));

        // Layer 10: ResBlock(512, dilation=1)
        layers.push(EncoderLayer::ResBlock(EncoderResBlock::load(
            512,
            1,
            vb.pp("layers.10"),
        )?));

        // Layer 11: ELU
        layers.push(EncoderLayer::Elu);

        // Layer 12: Conv1d(512→1024, k=16, s=8) downsample
        layers.push(EncoderLayer::Conv(CausalConv::load(
            512,
            1024,
            16,
            8,
            1,
            vb.pp("layers.12"),
        )?));

        // Layer 13: ELU
        layers.push(EncoderLayer::Elu);

        // Layer 14: Conv1d(1024→512, k=3, s=1) project to hidden
        layers.push(EncoderLayer::Conv(CausalConv::load(
            1024,
            512,
            3,
            1,
            1,
            vb.pp("layers.14"),
        )?));

        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i <= 3 || i >= 13 {
                let vals: Vec<f32> = h
                    .narrow(0, 0, 1)?
                    .narrow(1, 0, 1)?
                    .narrow(2, 0, 5.min(h.dim(2)?))?
                    .flatten_all()?
                    .to_vec1()?;
                tracing::info!(
                    "  conv layer {}: shape={:?}, first 5: {:?}",
                    i,
                    h.dims(),
                    vals
                );
            }
        }
        Ok(h)
    }
}

// ── Rotary Position Embedding ───────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor, // [max_len, head_dim/2]
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(max_len: usize, dtype: DType, dev: &CandleDevice) -> candle_core::Result<Self> {
        let inv_freq: Vec<f32> = (0..HEAD_DIM)
            .step_by(2)
            .map(|i| 1.0f32 / ROPE_THETA.powf(i as f64 / HEAD_DIM as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
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

// ── LayerScale ──────────────────────────────────────────────────────────

/// Learnable per-channel scale applied after attention or MLP.
struct LayerScale {
    scale: Tensor, // [hidden_size]
}

impl LayerScale {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let scale = vb.get(HIDDEN_SIZE, "scale")?;
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

// ── Encoder Transformer Layer ───────────────────────────────────────────

struct EncoderTransformerLayer {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    q_proj: LinearNoBias,
    k_proj: LinearNoBias,
    v_proj: LinearNoBias,
    o_proj: LinearNoBias,
    fc1: LinearNoBias,
    fc2: LinearNoBias,
    attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
}

impl EncoderTransformerLayer {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let input_layernorm = LayerNorm::load(HIDDEN_SIZE, LN_EPS, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            LayerNorm::load(HIDDEN_SIZE, LN_EPS, vb.pp("post_attention_layernorm"))?;

        let sa = vb.pp("self_attn");
        let q_proj = LinearNoBias::load(HIDDEN_SIZE, HIDDEN_SIZE, sa.pp("q_proj"))?;
        let k_proj = LinearNoBias::load(HIDDEN_SIZE, HIDDEN_SIZE, sa.pp("k_proj"))?;
        let v_proj = LinearNoBias::load(HIDDEN_SIZE, HIDDEN_SIZE, sa.pp("v_proj"))?;
        let o_proj = LinearNoBias::load(HIDDEN_SIZE, HIDDEN_SIZE, sa.pp("o_proj"))?;

        let mlp_vb = vb.pp("mlp");
        let fc1 = LinearNoBias::load(HIDDEN_SIZE, INTERMEDIATE_SIZE, mlp_vb.pp("fc1"))?;
        let fc2 = LinearNoBias::load(INTERMEDIATE_SIZE, HIDDEN_SIZE, mlp_vb.pp("fc2"))?;

        let attn_layer_scale = LayerScale::load(vb.pp("self_attn_layer_scale"))?;
        let mlp_layer_scale = LayerScale::load(vb.pp("mlp_layer_scale"))?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            fc1,
            fc2,
            attn_layer_scale,
            mlp_layer_scale,
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding) -> candle_core::Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        // ── Self-attention ──
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;

        let q = self.q_proj.forward(&h)?; // [B, T, H]
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;

        // Reshape to [B, num_heads, T, head_dim]
        let q = q.reshape((b, t, NUM_HEADS, HEAD_DIM))?.transpose(1, 2)?;
        let k = k.reshape((b, t, NUM_HEADS, HEAD_DIM))?.transpose(1, 2)?;
        let v = v.reshape((b, t, NUM_HEADS, HEAD_DIM))?.transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = rope.apply(&q, &k)?;

        // Scaled dot-product attention with sliding window
        let scale = (HEAD_DIM as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?;

        // Build causal sliding window mask
        let attn = if t > 1 {
            let mask = build_sliding_window_mask(t, SLIDING_WINDOW, x.device())?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, T, T]
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?; // [B, H, T, D]

        // Reshape back to [B, T, hidden]
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, t, HIDDEN_SIZE))?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        let attn_out = self.attn_layer_scale.forward(&attn_out)?;
        let h = (residual + attn_out)?;

        // ── MLP ──
        let residual = h.clone();
        let m = self.post_attention_layernorm.forward(&h)?;
        let m = self.fc1.forward(&m)?;
        let m = m.gelu()?;
        let m = self.fc2.forward(&m)?;
        let m = self.mlp_layer_scale.forward(&m)?;
        residual + m
    }
}

/// Build a causal sliding window attention mask.
///
/// Returns a [T, T] tensor where positions outside the causal window
/// are set to -inf and positions inside are 0.
fn build_sliding_window_mask(
    seq_len: usize,
    window: usize,
    dev: &CandleDevice,
) -> candle_core::Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            // Causal: j <= i (cannot attend to future)
            // Sliding window: i - j < window (cannot attend too far back)
            let allowed = j <= i && (i - j) < window;
            if !allowed {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), dev)
}

// ── Encoder Transformer ─────────────────────────────────────────────────

struct EncoderTransformer {
    layers: Vec<EncoderTransformerLayer>,
    rope: RotaryEmbedding,
}

impl EncoderTransformer {
    fn load(dev: &CandleDevice, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(NUM_TRANSFORMER_LAYERS);
        for i in 0..NUM_TRANSFORMER_LAYERS {
            layers.push(EncoderTransformerLayer::load(vb.pp(format!("layers.{i}")))?);
        }
        // Pre-compute RoPE for up to 8192 positions (generous upper bound)
        let rope = RotaryEmbedding::new(8192, DType::F32, dev)?;
        Ok(Self { layers, rope })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T] → transpose to [B, T, C] for transformer
        let mut h = x.transpose(1, 2)?;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope)?;
        }
        // Back to [B, C, T]
        h.transpose(1, 2)
    }
}

// ── EuclideanCodebook (encoder-side: quantize) ──────────────────────────

/// Codebook for nearest-neighbor quantization.
///
/// Stores `embed_sum` and `cluster_usage`; effective embedding is
/// `embed_sum / clamp(cluster_usage, eps)[:, None]`.
struct EuclideanCodebook {
    embed_sum: Tensor,     // [codebook_size, dim]
    cluster_usage: Tensor, // [codebook_size]
    eps: f64,
}

impl EuclideanCodebook {
    fn load(codebook_size: usize, dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let embed_sum = vb.get((codebook_size, dim), "embed_sum")?;
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?;
        Ok(Self {
            embed_sum,
            cluster_usage,
            eps: 1e-5,
        })
    }

    /// Get the effective embedding table: embed_sum / clamp(usage, eps).
    fn embedding(&self) -> candle_core::Result<Tensor> {
        let usage = self.cluster_usage.clamp(self.eps as f32, f32::MAX)?;
        self.embed_sum.broadcast_div(&usage.unsqueeze(1)?)
    }

    /// Quantize: find the nearest codebook entry for each input vector.
    ///
    /// Input: [B, T, dim]
    /// Output: (indices [B, T], quantized [B, T, dim])
    fn quantize(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let emb = self.embedding()?; // [K, D]
        let (b, t, d) = x.dims3()?;
        let _k = emb.dim(0)?;

        // L2 distance: ||x - e||^2 = ||x||^2 - 2*x*e^T + ||e||^2
        // Flatten to [B*T, D] for matmul
        let flat = x.reshape((b * t, d))?;

        // ||x||^2: [B*T, 1]
        let x_sq = flat.sqr()?.sum_keepdim(D::Minus1)?;

        // ||e||^2: [1, K]
        let e_sq = emb.sqr()?.sum_keepdim(D::Minus1)?.t()?;

        // x * e^T: [B*T, K]
        let xe = flat.matmul(&emb.t()?)?;

        // dist = ||x||^2 - 2*x*e^T + ||e||^2
        let dist = (x_sq.broadcast_add(&e_sq)? - (xe * 2.0)?)?;

        // argmin over codebook dimension
        let indices = dist.argmin(D::Minus1)?; // [B*T]

        // Look up quantized vectors
        let quantized = emb.index_select(&indices, 0)?; // [B*T, D]
        let indices = indices.reshape((b, t))?;
        let quantized = quantized.reshape((b, t, d))?;

        Ok((indices, quantized))
    }
}

// ── VQ Layer (single codebook) ──────────────────────────────────────────

struct VectorQuantizationLayer {
    codebook: EuclideanCodebook,
}

impl VectorQuantizationLayer {
    fn load(codebook_size: usize, dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let codebook = EuclideanCodebook::load(codebook_size, dim, vb.pp("codebook"))?;
        Ok(Self { codebook })
    }

    /// Quantize input, return (index [B, T], quantized [B, T, dim]).
    fn quantize(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        self.codebook.quantize(x)
    }
}

// ── RVQ (residual vector quantization, encoder-side) ────────────────────

struct ResidualVQ {
    layers: Vec<VectorQuantizationLayer>,
}

impl ResidualVQ {
    fn load(
        num_layers: usize,
        codebook_size: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(VectorQuantizationLayer::load(
                codebook_size,
                dim,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    /// Quantize with residual refinement.
    ///
    /// Input: [B, T, dim]
    /// Output: indices [num_layers, B, T]
    fn quantize(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_b, _t, _d) = x.dims3()?;
        let mut residual = x.clone();
        let mut all_indices = Vec::with_capacity(self.layers.len());

        for (li, layer) in self.layers.iter().enumerate() {
            let (indices, quantized) = layer.quantize(&residual)?;
            // Debug: dump first 2 layers for frame 0
            if li < 2 {
                if let Ok(idx) = indices.flatten_all().and_then(|t| t.to_vec1::<u32>()) {
                    info!("  RVQ layer {}: frame 0 idx={}", li, idx[0]);
                }
                if let Ok(r) = residual
                    .narrow(0, 0, 1)
                    .and_then(|t| t.narrow(1, 0, 1))
                    .and_then(|t| t.narrow(2, 0, 5))
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                {
                    info!("  RVQ layer {} residual first 5: {:?}", li, r);
                }
            }
            all_indices.push(indices);
            residual = (residual - quantized)?;
        }

        // Stack: list of [B, T] → [num_layers, B, T]
        Tensor::stack(&all_indices, 0)
    }
}

// ── Split RVQ Quantizer (semantic + acoustic) ───────────────────────────

/// Encoder-side quantizer: projects hidden→codebook_dim, quantizes via
/// semantic RVQ (1 codebook, 4096 entries) + acoustic RVQ (31 codebooks,
/// 2048 entries), outputs first 16 codebook indices.
struct SplitQuantizer {
    semantic_input_proj: Conv1d,  // [256, 512, 1]
    semantic_output_proj: Conv1d, // [512, 256, 1]
    semantic_rvq: ResidualVQ,

    acoustic_input_proj: Conv1d, // [256, 512, 1]
    #[allow(dead_code)] // loaded but unused during encode
    acoustic_output_proj: Conv1d, // [512, 256, 1]
    acoustic_rvq: ResidualVQ,
}

impl SplitQuantizer {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let conv1d_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };

        // Semantic sub-quantizer
        let sem_vb = vb.pp("semantic_residual_vector_quantizer");
        let sem_inp_w = sem_vb
            .pp("input_proj")
            .get((CODEBOOK_DIM, HIDDEN_SIZE, 1), "weight")?;
        let sem_inp_b = sem_vb.pp("input_proj").get(CODEBOOK_DIM, "bias").ok();
        let semantic_input_proj = Conv1d::new(sem_inp_w, sem_inp_b, conv1d_cfg);

        let sem_out_w = sem_vb
            .pp("output_proj")
            .get((HIDDEN_SIZE, CODEBOOK_DIM, 1), "weight")?;
        let sem_out_b = sem_vb.pp("output_proj").get(HIDDEN_SIZE, "bias").ok();
        let semantic_output_proj = Conv1d::new(sem_out_w, sem_out_b, conv1d_cfg);

        let semantic_rvq = ResidualVQ::load(1, SEMANTIC_CODEBOOK_SIZE, CODEBOOK_DIM, sem_vb)?;

        // Acoustic sub-quantizer
        let aco_vb = vb.pp("acoustic_residual_vector_quantizer");
        let aco_inp_w = aco_vb
            .pp("input_proj")
            .get((CODEBOOK_DIM, HIDDEN_SIZE, 1), "weight")?;
        let aco_inp_b = aco_vb.pp("input_proj").get(CODEBOOK_DIM, "bias").ok();
        let acoustic_input_proj = Conv1d::new(aco_inp_w, aco_inp_b, conv1d_cfg);

        let aco_out_w = aco_vb
            .pp("output_proj")
            .get((HIDDEN_SIZE, CODEBOOK_DIM, 1), "weight")?;
        let aco_out_b = aco_vb.pp("output_proj").get(HIDDEN_SIZE, "bias").ok();
        let acoustic_output_proj = Conv1d::new(aco_out_w, aco_out_b, conv1d_cfg);

        let acoustic_rvq = ResidualVQ::load(
            NUM_ACOUSTIC_CODEBOOKS,
            ACOUSTIC_CODEBOOK_SIZE,
            CODEBOOK_DIM,
            aco_vb,
        )?;

        Ok(Self {
            semantic_input_proj,
            semantic_output_proj,
            semantic_rvq,
            acoustic_input_proj,
            acoustic_output_proj,
            acoustic_rvq,
        })
    }

    /// Quantize encoder hidden states to codec token indices.
    ///
    /// Input: hidden [B, hidden_size, T]
    /// Output: indices [T, 16] (1 semantic + 15 acoustic)
    fn quantize(&self, hidden: &Tensor) -> candle_core::Result<Vec<Vec<u32>>> {
        let _b = hidden.dim(0)?;
        let t = hidden.dim(2)?;

        // Step 1: Semantic quantization
        // Project: [B, 512, T] → [B, 256, T]
        let sem_proj = self.semantic_input_proj.forward(hidden)?;
        let sem_input = sem_proj.transpose(1, 2)?; // [B, T, 256]
        let sem_indices = self.semantic_rvq.quantize(&sem_input)?; // [1, B, T]

        // Step 2: Acoustic quantization
        // Python passes the SAME hidden to both sub-quantizers (no residual between them).
        let aco_proj = self.acoustic_input_proj.forward(hidden)?; // same hidden!

        // Debug: dump acoustic input proj frame 0
        if let Ok(vals) = aco_proj
            .narrow(0, 0, 1)
            .and_then(|t| t.narrow(2, 0, 1))
            .and_then(|t| t.narrow(1, 0, 10))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
        {
            info!("Acoustic input proj frame 0 first 10: {:?}", vals);
        }

        // Transpose: [B, T, 256]
        let aco_input = aco_proj.transpose(1, 2)?;
        // Quantize all 31 acoustic codebooks → [31, B, T]
        let aco_indices = self.acoustic_rvq.quantize(&aco_input)?;

        // Step 3: Take first 16 codebook indices (1 semantic + 15 acoustic)
        // sem_indices: [1, B, T], aco_indices: [31, B, T]
        // Take first 15 from acoustic
        let aco_first15 = aco_indices.narrow(0, 0, NUM_OUTPUT_CODEBOOKS - 1)?;
        // Concat: [16, B, T]
        let all_indices = Tensor::cat(&[&sem_indices, &aco_first15], 0)?;

        // Convert to Vec<Vec<u32>>: output shape [T, 16]
        // all_indices is [16, B, T], take B=0
        let indices_2d = all_indices.i((.., 0, ..))?; // [16, T]
        let indices_2d = indices_2d.t()?; // [T, 16]
        let indices_2d = indices_2d.to_dtype(DType::U32)?;

        let mut result = Vec::with_capacity(t);
        for ti in 0..t {
            let row = indices_2d.i(ti)?; // [16]
            let row_vec: Vec<u32> = row.to_vec1()?;
            result.push(row_vec);
        }
        Ok(result)
    }
}

// ── SpeechTokenizerEncoder (top-level) ──────────────────────────────────

/// Mimi-based speech tokenizer encoder: raw 24kHz PCM → codec tokens.
///
/// Pipeline: CausalConv stack (960x downsample) → Transformer (8 layers)
/// → Split RVQ quantizer → [T, 16] token indices.
pub struct SpeechTokenizerEncoder {
    conv_stack: candle_transformers::models::mimi::seanet::SeaNetEncoder,
    transformer:
        parking_lot::Mutex<candle_transformers::models::mimi::transformer::ProjectedTransformer>,
    downsample: candle_transformers::models::mimi::conv::ConvDownsample1d,
    quantizer: candle_transformers::models::mimi::quantization::SplitResidualVectorQuantizer,
    device: CandleDevice,
}

impl SpeechTokenizerEncoder {
    /// Load weights from VarBuilder with `encoder.` prefix.
    pub fn load(vb: VarBuilder, device: CandleDevice) -> Result<Self> {
        let mimi_cfg = candle_transformers::models::mimi::Config::v0_1(Some(NUM_OUTPUT_CODEBOOKS));

        let conv_stack = candle_transformers::models::mimi::seanet::SeaNetEncoder::new(
            &mimi_cfg.seanet,
            vb.pp("encoder"),
        )
        .map_err(|e| FerrumError::model(format!("encoder conv stack: {e}")))?;
        let transformer =
            candle_transformers::models::mimi::transformer::ProjectedTransformer::new(
                mimi_cfg.seanet.dimension,
                &[mimi_cfg.seanet.dimension],
                &mimi_cfg.transformer,
                vb.pp("encoder_transformer"),
            )
            .map_err(|e| FerrumError::model(format!("encoder transformer: {e}")))?;
        let transformer = parking_lot::Mutex::new(transformer);

        let downsample_stride = 2usize;
        let downsample = candle_transformers::models::mimi::conv::ConvDownsample1d::new(
            downsample_stride,
            mimi_cfg.seanet.dimension,
            true, // causal
            true, // learnt
            vb.pp("downsample"),
        )
        .map_err(|e| FerrumError::model(format!("encoder downsample: {e}")))?;

        // Use candle's SplitResidualVectorQuantizer (matches reference project exactly)
        let quantizer =
            candle_transformers::models::mimi::quantization::SplitResidualVectorQuantizer::new(
                CODEBOOK_DIM,           // 256
                Some(HIDDEN_SIZE),      // 512
                Some(HIDDEN_SIZE),      // 512
                NUM_OUTPUT_CODEBOOKS,   // 16
                SEMANTIC_CODEBOOK_SIZE, // 2048
                vb.pp("quantizer"),
            )
            .map_err(|e| FerrumError::model(format!("encoder quantizer: {e}")))?;

        info!(
            "SpeechTokenizerEncoder loaded: conv=15 layers (960x ds) + 2x downsample, \
             transformer={} layers (h={}, heads={}), \
             RVQ=1x{}+{}x{} → {} codebooks",
            NUM_TRANSFORMER_LAYERS,
            HIDDEN_SIZE,
            NUM_HEADS,
            SEMANTIC_CODEBOOK_SIZE,
            NUM_ACOUSTIC_CODEBOOKS,
            ACOUSTIC_CODEBOOK_SIZE,
            NUM_OUTPUT_CODEBOOKS,
        );

        Ok(Self {
            conv_stack,
            transformer,
            downsample,
            quantizer,
            device,
        })
    }

    /// Encode 24kHz mono PCM audio to codec token indices.
    ///
    /// Input: raw PCM samples at 24kHz sample rate.
    /// Output: `[T, 16]` where T = ceil(num_samples / 1920), each row
    /// contains 16 codebook indices (1 semantic + 15 acoustic).
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
        let num_samples = pcm.len();
        info!(
            "SpeechTokenizerEncoder: encoding {} samples ({:.2}s @ 24kHz)",
            num_samples,
            num_samples as f64 / 24000.0,
        );

        // Shape: [1, 1, num_samples]
        let input = Tensor::from_vec(pcm.to_vec(), (1, 1, num_samples), &self.device)
            .map_err(|e| FerrumError::model(format!("input tensor: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| FerrumError::model(format!("input dtype: {e}")))?;

        // Conv encoder: [1, 1, N] → [1, 512, T'] via candle SeaNetEncoder
        let conv_out = input
            .apply(&self.conv_stack)
            .map_err(|e| FerrumError::model(format!("conv encoder: {e}")))?;

        // Transformer: [1, 512, T] → [1, 512, T] via candle Mimi transformer
        let mut transformer = self.transformer.lock();
        let hidden = transformer
            .forward(&conv_out)
            .map_err(|e| FerrumError::model(format!("encoder transformer: {e}")))?;
        let hidden = &hidden[0]; // ProjectedTransformer returns Vec<Tensor>

        // Downsample: [1, 512, T] → [1, 512, T/2]
        let hidden = hidden
            .apply(&self.downsample)
            .map_err(|e| FerrumError::model(format!("encoder downsample: {e}")))?;

        let t_ds = hidden
            .dim(2)
            .map_err(|e| FerrumError::model(format!("downsample dim: {e}")))?;
        info!("After downsample: T={}", t_ds);

        // Quantize: [1, 512, T/2] → codes [1, 16, T/2] via candle SplitRVQ
        let codes = self
            .quantizer
            .encode(&hidden)
            .map_err(|e| FerrumError::model(format!("quantizer encode: {e}")))?;

        // Convert [1, 16, T] → Vec<Vec<u32>> as [T, 16]
        let codes = codes
            .squeeze(0)
            .map_err(|e| FerrumError::model(format!("squeeze: {e}")))?
            .transpose(0, 1)
            .map_err(|e| FerrumError::model(format!("transpose: {e}")))?
            .to_dtype(DType::U32)
            .map_err(|e| FerrumError::model(format!("to_u32: {e}")))?;

        let t = codes
            .dim(0)
            .map_err(|e| FerrumError::model(format!("codes dim: {e}")))?;
        let k = codes
            .dim(1)
            .map_err(|e| FerrumError::model(format!("codes dim1: {e}")))?;
        info!("SpeechTokenizerEncoder: {} frames, {} codebooks", t, k);

        let mut result = Vec::with_capacity(t);
        for ti in 0..t {
            let row: Vec<u32> = codes
                .i(ti)
                .and_then(|r| r.to_vec1())
                .map_err(|e| FerrumError::model(format!("codes row: {e}")))?;
            result.push(row);
        }
        Ok(result)
    }
}
