//! ferrum-attention: Fused flash attention and transformer for Metal and CPU.
//!
//! Single-kernel attention (QK^T + softmax + attn@V) with no intermediate buffer
//! materialization. Full transformer layer with all ops fused on GPU.

pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

/// Attention configuration.
pub struct AttentionParams {
    pub batch: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub pos_offset: usize,
}

/// Run fused attention on CPU.
pub fn attention_cpu(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    params: &AttentionParams,
) {
    cpu::fused_attention(q, k, v, out, params);
}

/// Run fused attention on best available backend.
pub fn attention(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    params: &AttentionParams,
) {
    #[cfg(feature = "metal")]
    {
        if metal::is_available() {
            metal::fused_attention(q, k, v, out, params);
            return;
        }
    }
    cpu::fused_attention(q, k, v, out, params);
}

// ── Fused Transformer ───────────────────────────────────────────────────

/// Transformer layer configuration.
#[derive(Clone)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

/// Per-layer weights as flat f32 vectors (extracted from safetensors once at init).
pub struct LayerWeights {
    pub input_ln_w: Vec<f32>,
    pub q_proj_w: Vec<f32>,
    pub k_proj_w: Vec<f32>,
    pub v_proj_w: Vec<f32>,
    pub o_proj_w: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub post_ln_w: Vec<f32>,
    pub gate_proj_w: Vec<f32>,
    pub up_proj_w: Vec<f32>,
    pub down_proj_w: Vec<f32>,
}

/// A complete N-layer fused transformer. All ops bypass candle.
pub struct FusedTransformer {
    cfg: TransformerConfig,
    cos: Vec<f32>,
    sin: Vec<f32>,
    norm_w: Vec<f32>,

    #[cfg(feature = "metal")]
    metal_state: Option<MetalTransformerState>,

    // CPU state
    cpu_layers: Vec<LayerWeights>,
    cpu_kv: Vec<cpu::transformer::CpuKvCache>,
    tokens_generated: usize,
}

#[cfg(feature = "metal")]
struct MetalTransformerState {
    pipes: metal::pipelines::MetalPipelines,
    weights: Vec<metal::transformer::MetalLayerWeights>,
    kv: Vec<metal::transformer::MetalKvCache>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    metal_cfg: metal::transformer::MetalTransformerConfig,
}

impl FusedTransformer {
    /// Create from pre-extracted layer weights.
    pub fn new(cfg: TransformerConfig, layers: Vec<LayerWeights>, norm_w: Vec<f32>) -> Self {
        // Precompute cos/sin
        let hd = cfg.head_dim;
        let half = hd / 2;
        let max_seq = cfg.max_position_embeddings.min(32768);
        let mut cos = vec![0.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let freq = 1.0f64 / cfg.rope_theta.powf((2 * i) as f64 / hd as f64);
                let angle = pos as f64 * freq;
                cos[pos * half + i] = angle.cos() as f32;
                sin[pos * half + i] = angle.sin() as f32;
            }
        }

        let n = layers.len();
        let cpu_kv = (0..n).map(|_| cpu::transformer::CpuKvCache::new()).collect();

        #[cfg(feature = "metal")]
        let metal_state = {
            if let Some(device) = ::metal::Device::system_default() {
                let pipes = metal::pipelines::MetalPipelines::new(&device);
                let weights: Vec<_> = layers.iter().map(|lw| {
                    metal::transformer::MetalLayerWeights {
                        input_ln_w: pipes.buffer_from_data(&lw.input_ln_w),
                        q_proj_w: pipes.buffer_from_data(&lw.q_proj_w),
                        k_proj_w: pipes.buffer_from_data(&lw.k_proj_w),
                        v_proj_w: pipes.buffer_from_data(&lw.v_proj_w),
                        o_proj_w: pipes.buffer_from_data(&lw.o_proj_w),
                        q_norm_w: lw.q_norm_w.clone(),
                        k_norm_w: lw.k_norm_w.clone(),
                        post_ln_w: pipes.buffer_from_data(&lw.post_ln_w),
                        gate_proj_w: pipes.buffer_from_data(&lw.gate_proj_w),
                        up_proj_w: pipes.buffer_from_data(&lw.up_proj_w),
                        down_proj_w: pipes.buffer_from_data(&lw.down_proj_w),
                    }
                }).collect();
                let kv = (0..n).map(|_| metal::transformer::MetalKvCache::new()).collect();
                let metal_cfg = metal::transformer::MetalTransformerConfig {
                    hidden_size: cfg.hidden_size,
                    intermediate_size: cfg.intermediate_size,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    rms_norm_eps: cfg.rms_norm_eps as f32,
                };
                Some(MetalTransformerState { pipes, weights, kv, cos: cos.clone(), sin: sin.clone(), metal_cfg })
            } else {
                None
            }
        };

        FusedTransformer {
            cfg,
            cos,
            sin,
            norm_w,
            #[cfg(feature = "metal")]
            metal_state,
            cpu_layers: layers,
            cpu_kv,
            tokens_generated: 0,
        }
    }

    /// Forward: input [tokens, hidden] → output [tokens, hidden] (f32 vecs).
    pub fn forward(&mut self, input: &[f32], tokens: usize) -> Vec<f32> {
        let pos_offset = self.tokens_generated;
        let h = self.cfg.hidden_size;

        #[cfg(feature = "metal")]
        if let Some(ref mut ms) = self.metal_state {
            let input_buf = ms.pipes.buffer_from_data(input);
            let mut buf = input_buf;
            for li in 0..ms.weights.len() {
                buf = metal::transformer::metal_layer_forward(
                    &ms.pipes, &buf, tokens, &ms.weights[li], &ms.metal_cfg,
                    &mut ms.kv[li], pos_offset, &ms.cos, &ms.sin,
                );
            }
            let hidden = metal::pipelines::MetalPipelines::read_buffer(&buf, tokens * h);
            self.tokens_generated += tokens;
            return self.final_rms_norm(&hidden, tokens);
        }

        // CPU fallback
        let mut hidden = input.to_vec();
        for li in 0..self.cpu_layers.len() {
            hidden = cpu::transformer::cpu_layer_forward(
                &hidden, tokens, &self.cpu_layers[li], &self.cfg,
                &self.cos, &self.sin, &mut self.cpu_kv[li], pos_offset,
            );
        }
        self.tokens_generated += tokens;
        self.final_rms_norm(&hidden, tokens)
    }

    fn final_rms_norm(&self, hidden: &[f32], tokens: usize) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let mut out = vec![0.0f32; tokens * h];
        for t in 0..tokens {
            let row = &hidden[t * h..(t + 1) * h];
            let o = &mut out[t * h..(t + 1) * h];
            let mut var = 0.0f64;
            for &v in row { let v64 = v as f64; var += v64 * v64; }
            let inv = 1.0 / (var / h as f64 + self.cfg.rms_norm_eps).sqrt();
            for i in 0..h { o[i] = (row[i] as f64 * inv) as f32 * self.norm_w[i]; }
        }
        out
    }

    pub fn reset(&mut self) {
        self.tokens_generated = 0;
        for kv in &mut self.cpu_kv {
            *kv = cpu::transformer::CpuKvCache::new();
        }
        #[cfg(feature = "metal")]
        if let Some(ref mut ms) = self.metal_state {
            for kv in &mut ms.kv {
                kv.reset();
            }
        }
    }
}
