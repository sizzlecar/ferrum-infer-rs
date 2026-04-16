//! Custom Llama architecture with public fields and per-request KV cache.
//!
//! Based on candle's Llama implementation but restructured for:
//! - Public weight access (CUDA runner weight extraction)
//! - Per-request KV cache keyed by cache_id (concurrent serving)
//! - export_kv_cache() for CUDA runner KV migration
//! - create_decode_runner() for CUDA decode path

use candle_core::{DType, Device as CandleDevice, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{self, Module, VarBuilder};

// Use candle_nn types directly
type Linear = candle_nn::Linear;
type Embedding = candle_nn::Embedding;
type RmsNorm = candle_nn::RmsNorm;

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> CandleResult<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(w, None))
}

/// RmsNorm with public weight access for CUDA runner extraction.
pub struct RmsNormWithWeight {
    norm: RmsNorm,
    pub weight: Tensor,
}

impl Module for RmsNormWithWeight {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.norm.forward(xs)
    }
}

fn rms_norm_with_weight(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<RmsNormWithWeight> {
    let w = vb.get(size, "weight")?;
    Ok(RmsNormWithWeight {
        norm: RmsNorm::new(w.clone(), eps),
        weight: w,
    })
}
use ferrum_types::{FerrumError, Result};
use std::collections::HashMap;
use tracing::{debug, info};

// ======================== Pre-allocated KV Cache ========================

/// Pre-allocated KV cache for a single sequence (all layers).
pub struct PreAllocKvCache {
    /// Per-layer K cache: [max_len, num_kv_heads, head_dim]
    pub k_caches: Vec<Tensor>,
    /// Per-layer V cache: [max_len, num_kv_heads, head_dim]
    pub v_caches: Vec<Tensor>,
    pub current_len: usize,
    pub max_len: usize,
}

impl PreAllocKvCache {
    pub fn new(
        num_layers: usize,
        max_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &CandleDevice,
    ) -> CandleResult<Self> {
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(Tensor::zeros(
                (max_len, num_kv_heads, head_dim),
                dtype,
                device,
            )?);
            v_caches.push(Tensor::zeros(
                (max_len, num_kv_heads, head_dim),
                dtype,
                device,
            )?);
        }
        Ok(Self {
            k_caches,
            v_caches,
            current_len: 0,
            max_len,
        })
    }
}

// ======================== Rotary Embedding ========================

pub struct RotaryEmbedding {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, dtype: DType, device: &CandleDevice) -> CandleResult<Self> {
        let head_dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_t = Tensor::new(inv_freq, device)?;
        let positions = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let angles = positions.matmul(&inv_freq_t.reshape((1, inv_freq_t.elem_count()))?)?;
        let cos = angles.cos()?.to_dtype(dtype)?;
        let sin = angles.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply(&self, x: &Tensor, pos: usize) -> CandleResult<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

// ======================== Model Components ========================

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub head_dim: usize,
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        layer_idx: usize,
        rotary: &RotaryEmbedding,
        kv_cache: &mut PreAllocKvCache,
    ) -> CandleResult<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let q = rotary.apply(&q, pos)?;
        let k = rotary.apply(&k, pos)?;

        // Update KV cache: [seq_len, num_kv_heads, head_dim]
        let k_for_cache = k.transpose(1, 2)?.contiguous()?; // [b, seq, nkv, hd]
        let v_for_cache = v.transpose(1, 2)?.contiguous()?;
        let k_squeezed = k_for_cache.squeeze(0)?; // [seq, nkv, hd]
        let v_squeezed = v_for_cache.squeeze(0)?;

        // Write into pre-allocated cache using slice_set (in-place)
        let start = kv_cache.current_len;
        let valid_len = start + seq_len;
        kv_cache.k_caches[layer_idx].slice_set(&k_squeezed, 0, start)?;
        kv_cache.v_caches[layer_idx].slice_set(&v_squeezed, 0, start)?;

        // Read back full KV for attention: [b, nkv, valid_len, hd]
        let k_full = kv_cache.k_caches[layer_idx]
            .narrow(0, 0, valid_len)?
            .unsqueeze(0)?
            .transpose(1, 2)?;
        let v_full = kv_cache.v_caches[layer_idx]
            .narrow(0, 0, valid_len)?
            .unsqueeze(0)?
            .transpose(1, 2)?;

        // GQA repeat
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        let k_full = crate::architectures::repeat_kv(k_full, n_rep)?;
        let v_full = crate::architectures::repeat_kv(v_full, n_rep)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let att = q
            .to_dtype(DType::F32)?
            .matmul(&k_full.to_dtype(DType::F32)?.t()?)?;
        let att = (att / scale)?;

        // Causal mask for prefill (seq_len > 1)
        let att = if seq_len > 1 {
            let mask: Vec<u8> = (0..seq_len)
                .flat_map(|i| (0..valid_len).map(move |j| u8::from(j > i + start)))
                .collect();
            let mask = Tensor::from_slice(&mask, (1, 1, seq_len, valid_len), x.device())?
                .broadcast_as(att.shape())?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(att.shape())?;
            mask.where_cond(&neg_inf, &att)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att
            .matmul(&v_full.to_dtype(DType::F32)?.contiguous()?)?
            .to_dtype(x.dtype())?;

        let y =
            y.transpose(1, 2)?
                .reshape((b, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&y)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> CandleResult<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, q_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }
}

pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> CandleResult<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
}

pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: Mlp,
    pub input_layernorm: RmsNormWithWeight,
    pub post_attention_layernorm: RmsNormWithWeight,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        layer_idx: usize,
        rotary: &RotaryEmbedding,
        kv_cache: &mut PreAllocKvCache,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self
            .self_attn
            .forward(&x, pos, layer_idx, rotary, kv_cache)?
            + residual)?;
        let residual = &x;
        let x = (self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?
            + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> CandleResult<Self> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), cfg)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            input_layernorm: rms_norm_with_weight(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm_with_weight(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }
}

// ======================== Full Model ========================

pub struct Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNormWithWeight,
    pub lm_head: Linear,
    pub rotary_emb: RotaryEmbedding,
    pub config: Config,
    /// Per-request KV caches, keyed by cache_id
    kv_caches: HashMap<String, PreAllocKvCache>,
}

impl Model {
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &CandleDevice,
    ) -> CandleResult<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let norm = rms_norm_with_weight(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let layers: Vec<DecoderLayer> = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load(vb.pp(format!("model.layers.{i}")), cfg))
            .collect::<CandleResult<_>>()?;
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: cfg.clone(),
            kv_caches: HashMap::new(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pos: usize,
        cache_key: &str,
    ) -> CandleResult<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Ensure KV cache exists for this request
        if !self.kv_caches.contains_key(cache_key) {
            let kv = PreAllocKvCache::new(
                self.config.num_hidden_layers,
                self.config.max_position_embeddings,
                self.config.num_key_value_heads,
                self.config.head_dim,
                DType::F16,
                &input_ids.device(),
            )?;
            self.kv_caches.insert(cache_key.to_string(), kv);
        }
        let kv_cache = self.kv_caches.get_mut(cache_key).unwrap();

        let mut x = self.embed_tokens.forward(input_ids)?;
        for (li, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, pos, li, &self.rotary_emb, kv_cache)?;
        }
        let x = self.norm.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;

        // Update KV cache length
        kv_cache.current_len += seq_len;

        logits.to_dtype(DType::F32)
    }

    pub fn clear_kv_cache_for(&mut self, cache_key: &str) {
        self.kv_caches.remove(cache_key);
    }

    /// Export KV cache for CUDA runner migration.
    /// Returns per-layer (K_tensor, V_tensor, current_len, max_len).
    pub fn export_kv_cache(&self, cache_key: &str) -> Option<Vec<(Tensor, Tensor, usize, usize)>> {
        let kv = self.kv_caches.get(cache_key)?;
        Some(
            kv.k_caches
                .iter()
                .zip(kv.v_caches.iter())
                .map(|(k, v)| (k.clone(), v.clone(), kv.current_len, kv.max_len))
                .collect(),
        )
    }

    pub fn release_cache(&self, _cache_key: &str) {
        // Intentionally no-op: kv_caches is &self, can't mutate.
        // Use clear_kv_cache_for with &mut self instead.
    }
}

// ======================== Model Wrapper ========================

pub struct LlamaModelWrapper {
    pub(crate) model: parking_lot::Mutex<Model>,
    config: Config,
    device: CandleDevice,
    dtype: DType,
    pub model_dir: Option<std::path::PathBuf>,
}

impl LlamaModelWrapper {
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("Creating Llama model from weights...");

        let head_dim = config
            .extra_params
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let cfg = Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            rms_norm_eps: config.norm_eps,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            max_position_embeddings: config.max_position_embeddings,
            tie_word_embeddings: config
                .extra_params
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            head_dim,
        };

        debug!(
            "Llama config: hidden={}, layers={}, heads={}, kv_heads={}, head_dim={}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim
        );

        let model = Model::load(vb, &cfg, dtype, &device)
            .map_err(|e| FerrumError::model(format!("Failed to load Llama model: {}", e)))?;

        info!("Llama model created successfully");

        Ok(Self {
            model: parking_lot::Mutex::new(model),
            config: cfg,
            device,
            dtype,
            model_dir: None,
        })
    }

    pub fn forward_prefill(&self, input_ids: &Tensor, cache_key: &str) -> Result<Tensor> {
        let mut model = self.model.lock();
        model.clear_kv_cache_for(cache_key);
        model
            .forward(input_ids, 0, cache_key)
            .map_err(|e| FerrumError::model(format!("Prefill failed: {}", e)))
    }

    pub fn forward_decode(&self, token_id: &Tensor, pos: usize, cache_key: &str) -> Result<Tensor> {
        let mut model = self.model.lock();
        model
            .forward(token_id, pos, cache_key)
            .map_err(|e| FerrumError::model(format!("Decode failed: {}", e)))
    }

    pub fn export_kv_cache(&self, cache_key: &str) -> Option<Vec<(Tensor, Tensor, usize, usize)>> {
        self.model.lock().export_kv_cache(cache_key)
    }

    pub fn release_cache(&self, cache_key: &str) {
        self.model.lock().clear_kv_cache_for(cache_key);
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

    pub fn set_model_dir(&mut self, dir: std::path::PathBuf) {
        self.model_dir = Some(dir);
    }

    /// Create CUDA decode runner by extracting weights directly from the model.
    #[cfg(feature = "cuda")]
    pub fn create_decode_runner(&self) -> Result<ferrum_kernels::cuda_decode::CudaDecodeRunner> {
        use ferrum_kernels::decode_buffers::ModelDims;
        use ferrum_kernels::weight_store::{
            GpuWeight, LayerWeights, LinearWeight, TransformerGpuWeights,
        };

        let model = self.model.lock();
        let cfg = &self.config;

        let cuda_device = self
            .device
            .as_cuda_device()
            .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
        let candle_stream = cuda_device.cuda_stream();
        candle_stream
            .synchronize()
            .map_err(|e| FerrumError::model(format!("sync: {e}")))?;
        let rs = candle_stream
            .context()
            .new_stream()
            .map_err(|e| FerrumError::model(format!("new_stream: {e}")))?;

        let embed_table = GpuWeight::from_tensor(model.embed_tokens.embeddings(), &rs)
            .map_err(|e| FerrumError::model(format!("embed: {e}")))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (li, layer) in model.layers.iter().enumerate() {
            // Fuse Q+K+V → QKV
            let qkv_fused = candle_core::Tensor::cat(
                &[
                    layer.self_attn.q_proj.weight(),
                    layer.self_attn.k_proj.weight(),
                    layer.self_attn.v_proj.weight(),
                ],
                0,
            )
            .map_err(|e| FerrumError::model(format!("qkv cat L{li}: {e}")))?;

            // Fuse gate+up
            let gate_up_fused = candle_core::Tensor::cat(
                &[layer.mlp.gate_proj.weight(), layer.mlp.up_proj.weight()],
                0,
            )
            .map_err(|e| FerrumError::model(format!("gate_up cat L{li}: {e}")))?;

            layers.push(LayerWeights {
                input_ln_w: GpuWeight::from_tensor(&layer.input_layernorm.weight, &rs)
                    .map_err(|e| FerrumError::model(format!("input_ln: {e}")))?,
                qkv_w: LinearWeight::Fp16(
                    GpuWeight::from_tensor(&qkv_fused, &rs)
                        .map_err(|e| FerrumError::model(format!("qkv: {e}")))?,
                ),
                q_norm_w: None,
                k_norm_w: None,
                o_w: LinearWeight::Fp16(
                    GpuWeight::from_tensor(layer.self_attn.o_proj.weight(), &rs)
                        .map_err(|e| FerrumError::model(format!("o: {e}")))?,
                ),
                post_ln_w: GpuWeight::from_tensor(&layer.post_attention_layernorm.weight, &rs)
                    .map_err(|e| FerrumError::model(format!("post_ln: {e}")))?,
                gate_up_w: LinearWeight::Fp16(
                    GpuWeight::from_tensor(&gate_up_fused, &rs)
                        .map_err(|e| FerrumError::model(format!("gate_up: {e}")))?,
                ),
                down_w: LinearWeight::Fp16(
                    GpuWeight::from_tensor(layer.mlp.down_proj.weight(), &rs)
                        .map_err(|e| FerrumError::model(format!("down: {e}")))?,
                ),
            });
        }

        let final_norm_w = GpuWeight::from_tensor(&model.norm.weight, &rs)
            .map_err(|e| FerrumError::model(format!("final_norm: {e}")))?;
        let lm_head_w = LinearWeight::Fp16(
            GpuWeight::from_tensor(model.lm_head.weight(), &rs)
                .map_err(|e| FerrumError::model(format!("lm_head: {e}")))?,
        );
        let rope_cos = GpuWeight::from_tensor(&model.rotary_emb.cos, &rs)
            .map_err(|e| FerrumError::model(format!("rope_cos: {e}")))?;
        let rope_sin = GpuWeight::from_tensor(&model.rotary_emb.sin, &rs)
            .map_err(|e| FerrumError::model(format!("rope_sin: {e}")))?;

        let weights = TransformerGpuWeights {
            embed_table,
            layers,
            final_norm_w,
            lm_head_w,
            rope_cos,
            rope_sin,
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
            quantized: false,
            max_batch_size: std::env::var("FERRUM_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
        };

        rs.synchronize()
            .map_err(|e| FerrumError::model(format!("sync: {e}")))?;

        ferrum_kernels::cuda_decode::CudaDecodeRunner::new(weights, dims, cuda_device.clone(), rs)
            .map_err(|e| FerrumError::model(format!("CudaDecodeRunner: {e}")))
    }
}
