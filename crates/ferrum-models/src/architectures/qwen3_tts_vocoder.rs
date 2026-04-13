//! Qwen3-TTS Vocoder — converts codec tokens to audio waveform.
//!
//! Architecture: SplitRVQ decode → CausalConv → Transformer (8 layers) →
//! Upsampling (2×2×8×5×4×3 = 1920×) → Waveform @ 24kHz.
//!
//! Loaded from speech_tokenizer/model.safetensors.

use candle_core::{DType, Device as CandleDevice, IndexOp, Module, Tensor, D};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, RmsNorm, VarBuilder,
};
use ferrum_types::{FerrumError, Result};
use tracing::info;

// ── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VocoderConfig {
    pub codebook_size: usize,          // 2048
    pub codebook_dim: usize,           // 256 (but split into 128 per sub-quantizer)
    pub num_quantizers: usize,         // 16
    pub latent_dim: usize,             // 1024
    pub hidden_size: usize,            // 1024
    pub num_hidden_layers: usize,      // 8
    pub num_attention_heads: usize,    // 16
    pub num_key_value_heads: usize,    // 16
    pub intermediate_size: usize,      // 3072
    pub rms_norm_eps: f64,             // 1e-5
    pub rope_theta: f64,               // 10000.0
    pub sliding_window: usize,         // 72
    pub decoder_dim: usize,            // 1536
    pub upsample_rates: Vec<usize>,    // [8, 5, 4, 3]
    pub upsampling_ratios: Vec<usize>, // [2, 2]
    pub output_sample_rate: usize,     // 24000
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            codebook_size: 2048,
            codebook_dim: 512,
            num_quantizers: 16,
            latent_dim: 1024,
            hidden_size: 512,
            num_hidden_layers: 8,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            intermediate_size: 1024,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: 72,
            decoder_dim: 1536,
            upsample_rates: vec![8, 5, 4, 3],
            upsampling_ratios: vec![2, 2],
            output_sample_rate: 24000,
        }
    }
}

// ── EuclideanCodebook ───────────────────────────────────────────────────

/// Codebook that stores embedding_sum and cluster_usage, decodes via normalized lookup.
struct EuclideanCodebook {
    embedding_sum: Tensor, // [codebook_size, dim]
    cluster_usage: Tensor, // [codebook_size]
    eps: f64,
}

impl EuclideanCodebook {
    fn load(dim: usize, codebook_size: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let embedding_sum = vb.get((codebook_size, dim), "embedding_sum")?;
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?;
        Ok(Self {
            embedding_sum,
            cluster_usage,
            eps: 1e-5,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        // embedding = embedding_sum / clamp(cluster_usage, min=eps)[:, None]
        let usage = self.cluster_usage.clamp(self.eps as f32, f32::MAX)?;
        let embedding = self.embedding_sum.broadcast_div(&usage.unsqueeze(1)?)?;
        // F.embedding(codes, embedding)
        let flat = codes.flatten_all()?;
        let looked_up = embedding.index_select(&flat, 0)?;
        // Reshape back to codes shape + dim
        let codes_shape = codes.dims();
        let dim = embedding.dim(1)?;
        let mut shape: Vec<usize> = codes_shape.to_vec();
        shape.push(dim);
        looked_up.reshape(shape)
    }
}

// ── VectorQuantization ──────────────────────────────────────────────────

struct VectorQuantization {
    codebook: EuclideanCodebook,
    project_out: Option<Linear>, // codebook_dim → dim (if different)
}

impl VectorQuantization {
    fn load(
        dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let codebook = EuclideanCodebook::load(codebook_dim, codebook_size, vb.pp("_codebook"))?;
        let project_out = if codebook_dim != dim {
            Some(candle_nn::linear(codebook_dim, dim, vb.pp("project_out"))?)
        } else {
            None
        };
        Ok(Self {
            codebook,
            project_out,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        // codes: [B, T] → quantized: [B, dim, T]
        let quantized = self.codebook.decode(codes)?; // [B, T, codebook_dim]
        let quantized = if let Some(proj) = &self.project_out {
            quantized.apply(proj)? // [B, T, dim]
        } else {
            quantized
        };
        quantized.transpose(1, 2) // [B, dim, T]
    }
}

// ── ResidualVectorQuantization ──────────────────────────────────────────

struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    fn load(
        num_quantizers: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_quantizers {
            layers.push(VectorQuantization::load(
                dim,
                codebook_size,
                codebook_dim,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        // codes: [num_quantizers, B, T]
        let mut quantized: Option<Tensor> = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_codes = codes.i(idx)?; // [B, T]
            let q = layer.decode(&layer_codes)?; // [B, dim, T]
            quantized = Some(match quantized {
                Some(prev) => (prev + q)?,
                None => q,
            });
        }
        quantized.ok_or_else(|| candle_core::Error::Msg("empty RVQ".into()))
    }
}

// ── ResidualVectorQuantizer (with input/output projection) ──────────────

struct ResidualVectorQuantizer {
    vq: ResidualVectorQuantization,
    output_proj: Option<Conv1d>, // dimension → output_dimension (1x1 conv)
}

impl ResidualVectorQuantizer {
    fn load(
        n_q: usize,
        dimension: usize,
        codebook_size: usize,
        codebook_dim: usize,
        output_dimension: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let vq = ResidualVectorQuantization::load(
            n_q,
            dimension,
            codebook_size,
            codebook_dim,
            vb.pp("vq"),
        )?;
        let output_proj = if output_dimension != dimension {
            let cfg = Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            let w = vb
                .pp("output_proj")
                .get((output_dimension, dimension, 1), "weight")?;
            let b = vb.pp("output_proj").get(output_dimension, "bias").ok();
            Some(Conv1d::new(w, b, cfg))
        } else {
            None
        };
        Ok(Self { vq, output_proj })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        // codes: [B, K, T] → [K, B, T]
        let codes = codes.transpose(0, 1)?;
        let quantized = self.vq.decode(&codes)?; // [B, dim, T]
        if let Some(proj) = &self.output_proj {
            proj.forward(&quantized)
        } else {
            Ok(quantized)
        }
    }
}

// ── SplitResidualVectorQuantizer ────────────────────────────────────────

struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer, // semantic (1 codebook)
    rvq_rest: ResidualVectorQuantizer,  // acoustic (15 codebooks)
    n_q_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    fn load(cfg: &VocoderConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        // dimension = codebook_dim // 2 = 256 (internal RVQ dimension)
        // codebook entries are 256-dim, but input/output projected to codebook_dim (512)
        let dim = cfg.codebook_dim / 2; // 256
        let n_q_semantic = 1;
        let n_q_acoustic = cfg.num_quantizers - n_q_semantic;

        let rvq_first = ResidualVectorQuantizer::load(
            n_q_semantic,
            dim,
            cfg.codebook_size,
            dim,
            cfg.codebook_dim, // output_dimension = 512
            vb.pp("rvq_first"),
        )?;
        let rvq_rest = ResidualVectorQuantizer::load(
            n_q_acoustic,
            dim,
            cfg.codebook_size,
            dim,
            cfg.codebook_dim, // output_dimension = 512
            vb.pp("rvq_rest"),
        )?;

        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q_semantic,
        })
    }

    fn decode(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        // codes: [B, K, T] where K = num_quantizers
        let semantic = codes.narrow(1, 0, self.n_q_semantic)?;
        let quantized = self.rvq_first.decode(&semantic)?;
        let k = codes.dim(1)?;
        if k > self.n_q_semantic {
            let acoustic = codes.narrow(1, self.n_q_semantic, k - self.n_q_semantic)?;
            let q_rest = self.rvq_rest.decode(&acoustic)?;
            Ok((quantized + q_rest)?)
        } else {
            Ok(quantized)
        }
    }
}

// ── CausalConv1d ────────────────────────────────────────────────────────

struct CausalConv {
    conv: Conv1d,
    padding: usize,
    stride: usize,
    kernel_size: usize,
}

impl CausalConv {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let padding = effective_kernel - stride;
        let cfg = Conv1dConfig {
            padding: 0, // we pad manually
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let w = vb
            .pp("conv")
            .get((out_ch, in_ch / groups, kernel_size), "weight")?;
        let b = vb.pp("conv").get(out_ch, "bias").ok();
        Ok(Self {
            conv: Conv1d::new(w, b, cfg),
            padding,
            stride,
            kernel_size: effective_kernel,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Left-pad for causal behavior
        let x = x.pad_with_zeros(2, self.padding, 0)?;
        self.conv.forward(&x)
    }
}

// ── CausalTransConv1d ───────────────────────────────────────────────────

struct CausalTransConv {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalTransConv {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            stride,
            dilation: 1,
            output_padding: 0,
            groups: 1,
        };
        let w = vb.pp("conv").get((in_ch, out_ch, kernel_size), "weight")?;
        let b = vb.pp("conv").get(out_ch, "bias").ok();
        let right_pad = kernel_size - stride;
        Ok(Self {
            conv: ConvTranspose1d::new(w, b, cfg),
            right_pad,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.conv.forward(&x)?;
        if self.right_pad > 0 {
            let len = y.dim(2)?;
            y.narrow(2, 0, len - self.right_pad)
        } else {
            Ok(y)
        }
    }
}

// ── SnakeBeta activation ────────────────────────────────────────────────

struct SnakeBeta {
    alpha: Tensor, // [C]
    beta: Tensor,  // [C]
}

impl SnakeBeta {
    fn load(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        let beta = vb.get(channels, "beta")?;
        Ok(Self { alpha, beta })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // x: [B, C, T]
        // SnakeBeta(x) = x + (1/beta) * sin^2(x * alpha)
        let alpha = self.alpha.exp()?.unsqueeze(0)?.unsqueeze(2)?; // [1, C, 1]
        let beta = self.beta.exp()?.unsqueeze(0)?.unsqueeze(2)?; // [1, C, 1]
        let sin_val = (x.broadcast_mul(&alpha))?.sin()?;
        let sin_sq = (&sin_val * &sin_val)?;
        let eps = 1e-9f64;
        x + sin_sq.broadcast_div(&(beta + eps)?)
    }
}

// ── ConvNeXtBlock ───────────────────────────────────────────────────────

struct ConvNeXtBlock {
    dwconv: CausalConv,
    norm: candle_nn::LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let dwconv = CausalConv::load(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = candle_nn::linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let h = self.dwconv.forward(x)?; // [B, C, T]
        let h = h.transpose(1, 2)?; // [B, T, C]
        let h = h.apply(&self.norm)?;
        let h = h.apply(&self.pwconv1)?.gelu()?;
        let h = h.apply(&self.pwconv2)?;
        let gamma = self.gamma.unsqueeze(0)?; // [1, C]
        let h = h.broadcast_mul(&gamma)?;
        let h = h.transpose(1, 2)?; // [B, C, T]
        (residual + h)
    }
}

// ── DecoderResidualUnit ─────────────────────────────────────────────────

struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv,
    act2: SnakeBeta,
    conv2: CausalConv,
}

impl DecoderResidualUnit {
    fn load(dim: usize, dilation: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let act1 = SnakeBeta::load(dim, vb.pp("act1"))?;
        let conv1 = CausalConv::load(dim, dim, 7, dilation, 1, 1, vb.pp("conv1"))?;
        let act2 = SnakeBeta::load(dim, vb.pp("act2"))?;
        let conv2 = CausalConv::load(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?;
        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        (residual + h)
    }
}

// ── DecoderBlock (upsample + 3 residual units) ─────────────────────────

struct DecoderBlock {
    snake: SnakeBeta,
    upsample: CausalTransConv,
    residual_units: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    fn load(cfg: &VocoderConfig, layer_idx: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let in_dim = cfg.decoder_dim / (1 << layer_idx);
        let out_dim = cfg.decoder_dim / (1 << (layer_idx + 1));
        let rate = cfg.upsample_rates[layer_idx];

        let snake = SnakeBeta::load(in_dim, vb.pp("block.0"))?;
        let upsample = CausalTransConv::load(in_dim, out_dim, 2 * rate, rate, vb.pp("block.1"))?;

        let mut residual_units = Vec::new();
        for (i, &dilation) in [1usize, 3, 9].iter().enumerate() {
            residual_units.push(DecoderResidualUnit::load(
                out_dim,
                dilation,
                vb.pp(format!("block.{}", i + 2)),
            )?);
        }

        Ok(Self {
            snake,
            upsample,
            residual_units,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = self.snake.forward(x)?;
        h = self.upsample.forward(&h)?;
        for unit in &self.residual_units {
            h = unit.forward(&h)?;
        }
        Ok(h)
    }
}

// ── Vocoder Pre-Transformer (FusedTransformer with layer_scale) ─────────
//
// Uses ferrum_attention::FusedTransformer for Metal/CPU acceleration.
// 8 layers, hidden=512, heads=16 (no GQA), head_dim=64, intermediate=1024.
// Includes layer_scale (learnable per-sublayer scalar weights).

/// Vocoder pre-transformer using FusedTransformer (Metal/CPU, no candle overhead).
/// input_proj(1024→512) → 8 fused layers with layer_scale → norm → output_proj(512→1024)
struct VocoderPreTransformer {
    input_proj_w: Vec<f32>,  // [512, 1024]
    input_proj_b: Vec<f32>,  // [512]
    output_proj_w: Vec<f32>, // [1024, 512]
    output_proj_b: Vec<f32>, // [1024]
    fused: ferrum_attention::FusedTransformer,
    hidden: usize,
    latent: usize,
}

impl VocoderPreTransformer {
    fn load(cfg: &VocoderConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size; // 512
        let lat = cfg.latent_dim; // 1024
        let nh = cfg.num_attention_heads; // 16
        let hd = 64usize; // head_dim

        // Extract projection weights to raw f32
        let to_vec = |t: &Tensor| -> candle_core::Result<Vec<f32>> {
            t.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
        };
        let get_w = |pp: &VarBuilder, shape: candle_core::Shape, name: &str| -> candle_core::Result<Vec<f32>> {
            to_vec(&pp.get(shape, name)?)
        };

        let input_proj_w = get_w(&vb.pp("input_proj"), (h, lat).into(), "weight")?;
        let input_proj_b = get_w(&vb.pp("input_proj"), h.into(), "bias")?;
        let output_proj_w = get_w(&vb.pp("output_proj"), (lat, h).into(), "weight")?;
        let output_proj_b = get_w(&vb.pp("output_proj"), lat.into(), "bias")?;

        // Extract transformer layer weights with layer_scale
        let mut fused_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lv = vb.pp(format!("layers.{i}"));
            let av = lv.pp("self_attn");
            let mv = lv.pp("mlp");
            let attn_scale = to_vec(&lv.pp("self_attn_layer_scale").get(h, "scale")?)?;
            let mlp_scale = to_vec(&lv.pp("mlp_layer_scale").get(h, "scale")?)?;
            fused_layers.push(ferrum_attention::LayerWeights {
                input_ln_w: get_w(&lv.pp("input_layernorm"), h.into(), "weight")?,
                q_proj_w: get_w(&av.pp("q_proj"), (nh * hd, h).into(), "weight")?,
                k_proj_w: get_w(&av.pp("k_proj"), (nh * hd, h).into(), "weight")?,
                v_proj_w: get_w(&av.pp("v_proj"), (nh * hd, h).into(), "weight")?,
                o_proj_w: get_w(&av.pp("o_proj"), (h, nh * hd).into(), "weight")?,
                q_norm_w: vec![], // empty = skip QK-norm (vocoder has none)
                k_norm_w: vec![], // empty = skip QK-norm
                post_ln_w: get_w(&lv.pp("post_attention_layernorm"), h.into(), "weight")?,
                gate_proj_w: get_w(&mv.pp("gate_proj"), (cfg.intermediate_size, h).into(), "weight")?,
                up_proj_w: get_w(&mv.pp("up_proj"), (cfg.intermediate_size, h).into(), "weight")?,
                down_proj_w: get_w(&mv.pp("down_proj"), (h, cfg.intermediate_size).into(), "weight")?,
                attn_layer_scale: Some(attn_scale),
                mlp_layer_scale: Some(mlp_scale),
            });
        }
        let norm_w = get_w(&vb.pp("norm"), h.into(), "weight")?;

        let fused = ferrum_attention::FusedTransformer::new(
            ferrum_attention::TransformerConfig {
                hidden_size: h,
                intermediate_size: cfg.intermediate_size,
                num_heads: nh,
                num_kv_heads: nh, // no GQA in vocoder
                head_dim: hd,
                num_layers: cfg.num_hidden_layers,
                rms_norm_eps: cfg.rms_norm_eps,
                rope_theta: cfg.rope_theta,
                max_position_embeddings: 4096,
            },
            fused_layers,
            norm_w,
        );

        Ok(Self { input_proj_w, input_proj_b, output_proj_w, output_proj_b, fused, hidden: h, latent: lat })
    }

    /// Forward: [B, T, latent_dim] → [B, T, latent_dim]. All on CPU/Metal via FusedTransformer.
    fn forward(&mut self, x: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = x.dim(1)?;
        let h = self.hidden;
        let lat = self.latent;

        // Extract to raw f32
        let x_data: Vec<f32> = x.to_device(&candle_core::Device::Cpu)?
            .to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        // Input projection: [T, 1024] @ [512, 1024]^T + bias → [T, 512]
        let mut proj = vec![0.0f32; seq_len * h];
        #[cfg(target_os = "macos")]
        {
            extern "C" {
                fn cblas_sgemm(order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32,
                    alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32,
                    beta: f32, c: *mut f32, ldc: i32);
            }
            unsafe {
                cblas_sgemm(101, 111, 112, seq_len as i32, h as i32, lat as i32,
                    1.0, x_data.as_ptr(), lat as i32,
                    self.input_proj_w.as_ptr(), lat as i32,
                    0.0, proj.as_mut_ptr(), h as i32);
            }
        }
        // Add bias
        for t in 0..seq_len {
            for j in 0..h { proj[t * h + j] += self.input_proj_b[j]; }
        }

        // FusedTransformer: 8 layers
        let transformed = self.fused.forward(&proj, seq_len);

        // Output projection: [T, 512] @ [1024, 512]^T + bias → [T, 1024]
        let mut out = vec![0.0f32; seq_len * lat];
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" {
                fn cblas_sgemm(order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32,
                    alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32,
                    beta: f32, c: *mut f32, ldc: i32);
            }
            cblas_sgemm(101, 111, 112, seq_len as i32, lat as i32, h as i32,
                1.0, transformed.as_ptr(), h as i32,
                self.output_proj_w.as_ptr(), h as i32,
                0.0, out.as_mut_ptr(), lat as i32);
        }
        for t in 0..seq_len {
            for j in 0..lat { out[t * lat + j] += self.output_proj_b[j]; }
        }

        Tensor::from_vec(out, (1, seq_len, lat), x.device())
    }
}

// ── Vocoder (top-level) ─────────────────────────────────────────────────

/// Qwen3-TTS Vocoder: codec tokens → audio waveform.
pub struct Qwen3TTSVocoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv,
    pre_transformer: VocoderPreTransformer,
    upsample_blocks: Vec<(CausalTransConv, ConvNeXtBlock)>,
    decoder_first_conv: CausalConv,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_final_snake: SnakeBeta,
    decoder_final_conv: CausalConv,
    config: VocoderConfig,
}

impl Qwen3TTSVocoder {
    pub fn load(cfg: &VocoderConfig, vb: VarBuilder) -> Result<Self> {
        let decoder_vb = vb.pp("decoder");

        let quantizer = SplitResidualVectorQuantizer::load(cfg, decoder_vb.pp("quantizer"))
            .map_err(|e| FerrumError::model(format!("quantizer: {e}")))?;

        // pre_conv: codebook_dim (512, from RVQ output) → latent_dim (1024)
        let pre_conv = CausalConv::load(
            cfg.codebook_dim,
            cfg.latent_dim,
            3,
            1,
            1,
            1,
            decoder_vb.pp("pre_conv"),
        )
        .map_err(|e| FerrumError::model(format!("pre_conv: {e}")))?;

        let pre_transformer = VocoderPreTransformer::load(cfg, decoder_vb.pp("pre_transformer"))
            .map_err(|e| FerrumError::model(format!("pre_transformer: {e}")))?;
        info!("Vocoder pre_transformer loaded: {} layers, hidden={}", cfg.num_hidden_layers, cfg.hidden_size);

        // Upsampling stages (before decoder)
        let mut upsample_blocks = Vec::new();
        for (i, &ratio) in cfg.upsampling_ratios.iter().enumerate() {
            let trans_conv = CausalTransConv::load(
                cfg.latent_dim,
                cfg.latent_dim,
                ratio,
                ratio,
                decoder_vb.pp(format!("upsample.{i}.0")),
            )
            .map_err(|e| FerrumError::model(format!("upsample.{i}.0: {e}")))?;
            let conv_next =
                ConvNeXtBlock::load(cfg.latent_dim, decoder_vb.pp(format!("upsample.{i}.1")))
                    .map_err(|e| FerrumError::model(format!("upsample.{i}.1: {e}")))?;
            upsample_blocks.push((trans_conv, conv_next));
        }

        // Decoder stages
        let decoder_first_conv = CausalConv::load(
            cfg.latent_dim,
            cfg.decoder_dim,
            7,
            1,
            1,
            1,
            decoder_vb.pp("decoder.0"),
        )
        .map_err(|e| FerrumError::model(format!("decoder.0: {e}")))?;

        let num_rates = cfg.upsample_rates.len();
        let mut decoder_blocks = Vec::new();
        for i in 0..num_rates {
            decoder_blocks.push(
                DecoderBlock::load(cfg, i, decoder_vb.pp(format!("decoder.{}", i + 1)))
                    .map_err(|e| FerrumError::model(format!("decoder.{}: {e}", i + 1)))?,
            );
        }

        let output_dim = cfg.decoder_dim / (1 << num_rates);
        let decoder_final_snake = SnakeBeta::load(
            output_dim,
            decoder_vb.pp(format!("decoder.{}", num_rates + 1)),
        )
        .map_err(|e| FerrumError::model(format!("decoder final snake: {e}")))?;

        let decoder_final_conv = CausalConv::load(
            output_dim,
            1,
            7,
            1,
            1,
            1,
            decoder_vb.pp(format!("decoder.{}", num_rates + 2)),
        )
        .map_err(|e| FerrumError::model(format!("decoder final conv: {e}")))?;

        info!(
            "Qwen3TTSVocoder loaded: codebook={}x{}, upsample={:?}+{:?}, decoder_dim={}",
            cfg.num_quantizers,
            cfg.codebook_size,
            cfg.upsampling_ratios,
            cfg.upsample_rates,
            cfg.decoder_dim,
        );

        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample_blocks,
            decoder_first_conv,
            decoder_blocks,
            decoder_final_snake,
            decoder_final_conv,
            config: cfg.clone(),
        })
    }

    /// Decode codec tokens to audio waveform.
    ///
    /// Input: codes [B, num_quantizers, T] (u32 codec tokens)
    /// Output: waveform [B, 1, T * upsample_total] clamped to [-1, 1]
    pub fn decode(&mut self, codes: &Tensor) -> Result<Tensor> {
        // RVQ decode: [B, K, T] → [B, codebook_dim, T]
        let hidden = self
            .quantizer
            .decode(codes)
            .map_err(|e| FerrumError::model(format!("quantizer decode: {e}")))?;

        // Pre-conv: [B, codebook_dim, T] → [B, latent_dim, T]
        let hidden = self
            .pre_conv
            .forward(&hidden)
            .map_err(|e| FerrumError::model(format!("pre_conv: {e}")))?;

        // Pre-transformer: [B, 1024, T] → transpose → 8-layer transformer → transpose back
        let hidden = hidden
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("transpose: {e}")))?; // [B, T, 1024]
        let hidden = self.pre_transformer.forward(&hidden)
            .map_err(|e| FerrumError::model(format!("pre_transformer: {e}")))?; // [B, T, 1024]
        let hidden = hidden
            .transpose(1, 2)
            .map_err(|e| FerrumError::model(format!("transpose back: {e}")))?; // [B, 1024, T]

        // Upsampling: (2, 2)
        let mut hidden = hidden;
        for (trans_conv, conv_next) in &self.upsample_blocks {
            hidden = trans_conv
                .forward(&hidden)
                .map_err(|e| FerrumError::model(format!("upsample trans: {e}")))?;
            hidden = conv_next
                .forward(&hidden)
                .map_err(|e| FerrumError::model(format!("upsample convnext: {e}")))?;
        }

        // Decoder: conv7 → [DecoderBlock × 4] → snake → conv7 → clamp
        let mut wav = self
            .decoder_first_conv
            .forward(&hidden)
            .map_err(|e| FerrumError::model(format!("decoder first conv: {e}")))?;

        for (i, block) in self.decoder_blocks.iter().enumerate() {
            wav = block
                .forward(&wav)
                .map_err(|e| FerrumError::model(format!("decoder block {i}: {e}")))?;
        }

        wav = self
            .decoder_final_snake
            .forward(&wav)
            .map_err(|e| FerrumError::model(format!("final snake: {e}")))?;
        wav = self
            .decoder_final_conv
            .forward(&wav)
            .map_err(|e| FerrumError::model(format!("final conv: {e}")))?;

        // Clamp to [-1, 1]
        wav.clamp(-1.0f32, 1.0f32)
            .map_err(|e| FerrumError::model(format!("clamp: {e}")))
    }

    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> usize {
        self.config.output_sample_rate
    }
}
