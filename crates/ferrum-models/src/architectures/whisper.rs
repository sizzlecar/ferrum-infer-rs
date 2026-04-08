//! Whisper ASR model — custom forward pass.
//!
//! candle loads weights only; forward pass is ours so Metal/CUDA work
//! without depending on candle-nn's LayerNorm CustomOp.

use candle_core::{DType, Device as CandleDevice, IndexOp, Module, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use candle_transformers::models::whisper::{self, Config};
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use tracing::info;

// ── Manual softmax (works on CPU/Metal/CUDA) ────────────────────────────

fn softmax_last_dim(x: &Tensor) -> candle_core::Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}

// ── Manual LayerNorm (pure tensor ops — works on CPU/Metal/CUDA) ─────────

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

// ── Linear (weight + optional bias) ─────────────────────────────────────

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn load(in_: usize, out: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get((out, in_), "weight")?;
        let bias = vb.get(out, "bias").ok();
        Ok(Self { weight, bias })
    }

    fn load_no_bias(in_: usize, out: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get((out, in_), "weight")?;
        Ok(Self { weight, bias: None })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let wt = self.weight.t()?;
        // Broadcast weight for batched matmul: (out, in) → (batch, in, out)
        let y = if x.dims().len() == 3 {
            let b = x.dim(0)?;
            x.matmul(&wt.broadcast_left(b)?)?
        } else {
            x.matmul(&wt)?
        };
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

// ── Multi-Head Attention ────────────────────────────────────────────────

struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let query = Linear::load(n_state, n_state, vb.pp("q_proj"))?;
        let value = Linear::load(n_state, n_state, vb.pp("v_proj"))?;
        let key = Linear::load_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = Linear::load(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_cache: bool,
    ) -> candle_core::Result<Tensor> {
        let q = self.query.forward(x)?;
        let (k, v) = match xa {
            None => (self.key.forward(x)?, self.value.forward(x)?),
            Some(xa_t) => {
                if flush_cache {
                    self.kv_cache = None;
                }
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone())
                } else {
                    let k = self.key.forward(xa_t)?;
                    let v = self.value.forward(xa_t)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            }
        };
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        self.out.forward(&wv)
    }

    fn reshape_head(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        x.reshape((b, t, self.n_head, c / self.n_head))?
            .transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (_, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            let mask = mask.i((0..n_ctx, 0..n_ctx))?;
            qk = qk.broadcast_add(&mask)?;
        }
        let w = softmax_last_dim(&qk)?;
        w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ── Residual Attention Block ────────────────────────────────────────────

struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(
        n_state: usize,
        n_head: usize,
        cross_attn: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = LayerNorm::load(n_state, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let ca = if cross_attn {
            let ca_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
            let ca_ln = LayerNorm::load(n_state, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
            Some((ca_attn, ca_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = Linear::load(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = Linear::load(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = LayerNorm::load(n_state, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn: ca,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv: bool,
    ) -> candle_core::Result<Tensor> {
        let a = self
            .attn
            .forward(&self.attn_ln.forward(x)?, None, mask, flush_kv)?;
        let mut x = (x + a)?;
        if let Some((ref mut ca, ref ln)) = self.cross_attn {
            x = (&x + ca.forward(&ln.forward(&x)?, xa, None, flush_kv)?)?;
        }
        let mlp = self
            .mlp_linear2
            .forward(&self.mlp_linear1.forward(&self.mlp_ln.forward(&x)?)?.gelu()?)?;
        x + mlp
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        if let Some((ref mut ca, _)) = self.cross_attn {
            ca.reset_kv_cache();
        }
    }
}

// ── Sinusoidal positional encoding ──────────────────────────────────────

fn sinusoids(length: usize, channels: usize, device: &CandleDevice) -> candle_core::Result<Tensor> {
    let max_timescale = 10000f32;
    let log_inc = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv: Vec<f32> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_inc)).exp())
        .collect();
    let inv = Tensor::new(inv.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled = (arange.broadcast_as(sh)? * inv.broadcast_as(sh)?)?;
    Tensor::cat(&[scaled.sin()?, scaled.cos()?], 1)
}

// ── Audio Encoder ───────────────────────────────────────────────────────

struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let n = cfg.d_model;
        let h = cfg.encoder_attention_heads;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = {
            let w = vb.pp("conv1").get((n, cfg.num_mel_bins, 3), "weight")?;
            let b = vb.pp("conv1").get(n, "bias")?;
            Conv1d::new(w, Some(b), cfg1)
        };
        let conv2 = {
            let w = vb.pp("conv2").get((n, n, 3), "weight")?;
            let b = vb.pp("conv2").get(n, "bias")?;
            Conv1d::new(w, Some(b), cfg2)
        };
        let pe = sinusoids(cfg.max_source_positions, n, vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n, h, false, vb.pp(format!("layers.{i}")))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ln_post = LayerNorm::load(n, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding: pe,
            blocks,
            ln_post,
        })
    }

    fn forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let (_, seq_len, _) = x.dims3()?;
        let pe = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&pe)?;
        for block in &mut self.blocks {
            x = block.forward(&x, None, None, flush)?;
        }
        self.ln_post.forward(&x)
    }
}

// ── Text Decoder ────────────────────────────────────────────────────────

struct TextDecoder {
    token_embedding: Tensor, // (vocab, d_model)
    positional_embedding: Tensor, // (max_target_positions, d_model)
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let n = cfg.d_model;
        let h = cfg.decoder_attention_heads;
        let ctx = cfg.max_target_positions;
        let token_embedding = vb.get((cfg.vocab_size, n), "embed_tokens.weight")?;
        let positional_embedding = vb.get((ctx, n), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n, h, true, vb.pp(format!("layers.{i}")))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ln = LayerNorm::load(n, 1e-5, vb.pp("layer_norm"))?;
        let mask_data: Vec<f32> = (0..ctx)
            .flat_map(|i| (0..ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask_data, (ctx, ctx), vb.device())?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    fn forward(
        &mut self,
        tokens: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        let seq_len = tokens.dim(D::Minus1)?;
        // Embedding lookup: tokens [batch, seq] → embeddings [batch, seq, d_model]
        let flat_tokens = tokens.flatten_all()?;
        let te = self.token_embedding.index_select(&flat_tokens, 0)?;
        let te = te.reshape((tokens.dim(0)?, seq_len, self.token_embedding.dim(1)?))?;
        let pe = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = te.broadcast_add(&pe)?;
        for block in &mut self.blocks {
            x = block.forward(&x, Some(xa), Some(&self.mask), flush)?;
        }
        self.ln.forward(&x)
    }

    fn final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let b = x.dim(0)?;
        let w = self.token_embedding.broadcast_left(b)?;
        x.matmul(&w.t()?)
    }

    fn reset_kv_cache(&mut self) {
        for block in &mut self.blocks {
            block.reset_kv_cache();
        }
    }
}

// ── Top-level Whisper wrapper ───────────────────────────────────────────

pub struct WhisperModelWrapper {
    encoder: Mutex<AudioEncoder>,
    decoder: Mutex<TextDecoder>,
    config: Config,
    mel_filters: Vec<f32>,
    device: CandleDevice,
    #[allow(dead_code)]
    dtype: DType,
}

impl WhisperModelWrapper {
    /// Load from VarBuilder + config.
    pub fn new(
        vb: VarBuilder,
        config: Config,
        mel_filters: Vec<f32>,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!(
            "Loading Whisper (d_model={}, encoder_layers={}, decoder_layers={})",
            config.d_model, config.encoder_layers, config.decoder_layers
        );
        let enc_vb = vb.pp("model.encoder");
        let dec_vb = vb.pp("model.decoder");
        let encoder = AudioEncoder::load(enc_vb, &config)
            .map_err(|e| FerrumError::model(format!("encoder load: {e}")))?;
        let decoder = TextDecoder::load(dec_vb, &config)
            .map_err(|e| FerrumError::model(format!("decoder load: {e}")))?;
        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            config,
            mel_filters,
            device,
            dtype,
        })
    }

    /// Load from model directory.
    pub fn from_model_dir(
        model_dir: &std::path::Path,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .map_err(|e| FerrumError::model(format!("read config: {e}")))?,
        )
        .map_err(|e| FerrumError::model(format!("parse config: {e}")))?;

        let mel_bytes = match config.num_mel_bins {
            128 => include_bytes!("mel_filters128.bin").as_slice(),
            _ => include_bytes!("mel_filters80.bin").as_slice(),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        for (i, chunk) in mel_bytes.chunks_exact(4).enumerate() {
            mel_filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let safetensors: Vec<_> = std::fs::read_dir(model_dir)
            .map_err(|e| FerrumError::model(format!("read dir: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        if safetensors.is_empty() {
            return Err(FerrumError::model("No safetensors files found"));
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors, dtype, &device)
                .map_err(|e| FerrumError::model(format!("load weights: {e}")))?
        };

        Self::new(vb, config, mel_filters, device, dtype)
    }

    /// PCM → mel spectrogram tensor.
    /// Pads or truncates PCM to exactly 30 seconds (N_SAMPLES = 480000) as Whisper expects.
    pub fn pcm_to_mel_tensor(&self, pcm: &[f32]) -> Result<Tensor> {
        // Whisper expects exactly 480000 samples (30 seconds at 16kHz)
        let n_samples = whisper::N_SAMPLES;
        let padded: std::borrow::Cow<[f32]> = if pcm.len() >= n_samples {
            std::borrow::Cow::Borrowed(&pcm[..n_samples])
        } else {
            let mut buf = pcm.to_vec();
            buf.resize(n_samples, 0.0);
            std::borrow::Cow::Owned(buf)
        };
        let mel = whisper::audio::pcm_to_mel(&self.config, &padded, &self.mel_filters);
        let mel_len = mel.len() / self.config.num_mel_bins;
        // Truncate mel to N_FRAMES (3000) if STFT padding produced extra frames
        let mel_len = mel_len.min(whisper::N_FRAMES);
        let mel_trimmed: Vec<f32> = (0..self.config.num_mel_bins)
            .flat_map(|bin| {
                let start = bin * (mel.len() / self.config.num_mel_bins);
                mel[start..start + mel_len].iter().copied()
            })
            .collect();
        Tensor::from_vec(mel_trimmed, (1, self.config.num_mel_bins, mel_len), &self.device)
            .map_err(|e| FerrumError::model(format!("mel tensor: {e}")))
    }

    /// Full transcription: PCM → token IDs.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        language_token: u32,
        task_token: u32,
        no_timestamps_token: u32,
        eot_token: u32,
        sot_token: u32,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mel = self.pcm_to_mel_tensor(pcm)?;

        // Encode
        let encoder_out = {
            let mut enc = self.encoder.lock();
            enc.blocks.iter_mut().for_each(|b| b.reset_kv_cache());
            enc.forward(&mel, true)
                .map_err(|e| FerrumError::model(format!("encode: {e}")))?
        };

        // Decode
        let mut dec = self.decoder.lock();
        dec.reset_kv_cache();

        let mut tokens: Vec<u32> =
            vec![sot_token, language_token, task_token, no_timestamps_token];
        let mut result: Vec<u32> = Vec::new();

        for _ in 0..max_tokens {
            let t = Tensor::new(tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| FerrumError::model(format!("token tensor: {e}")))?;

            let logits = dec
                .forward(&t, &encoder_out, false)
                .map_err(|e| FerrumError::model(format!("decode: {e}")))?;

            // final_linear on last position
            let last_pos = logits
                .dim(1)
                .map_err(|e| FerrumError::model(format!("dim: {e}")))?
                - 1;
            let last_hidden = logits
                .i((.., last_pos..last_pos + 1))
                .map_err(|e| FerrumError::model(format!("slice: {e}")))?;
            let logits_out = dec
                .final_linear(&last_hidden)
                .map_err(|e| FerrumError::model(format!("final_linear: {e}")))?;

            let next_token = logits_out
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .and_then(|t| t.argmax(0))
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| FerrumError::model(format!("argmax: {e}")))?;

            if next_token == eot_token {
                break;
            }

            result.push(next_token);
            tokens = vec![next_token];
        }

        Ok(result)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }
}
