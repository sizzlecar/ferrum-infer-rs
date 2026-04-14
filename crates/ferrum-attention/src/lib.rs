//! ferrum-attention: Fused flash attention and transformer for Metal and CPU.
//!
//! Single-kernel attention (QK^T + softmax + attn@V) with no intermediate buffer
//! materialization. Full transformer layer with all ops fused on GPU.

pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

/// Opaque GPU buffer type (Metal Buffer when available, placeholder otherwise).
#[cfg(feature = "metal")]
pub type GpuBuffer = ::metal::Buffer;
#[cfg(not(feature = "metal"))]
pub type GpuBuffer = Vec<f32>; // fallback: CPU storage

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
pub fn attention_cpu(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], params: &AttentionParams) {
    cpu::fused_attention(q, k, v, out, params);
}

/// Run fused attention on best available backend.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], params: &AttentionParams) {
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
    /// Optional layer_scale for attention (vocoder transformer uses this, talker doesn't)
    pub attn_layer_scale: Option<Vec<f32>>,
    /// Optional layer_scale for MLP
    pub mlp_layer_scale: Option<Vec<f32>>,
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
    /// true = always use CPU path (skips Metal even if available).
    /// Auto-set for small models where Metal sync overhead > compute benefit.
    use_cpu: bool,
}

#[cfg(feature = "metal")]
struct MetalTransformerState {
    pipes: metal::pipelines::MetalPipelines,
    weights: Vec<metal::transformer::MetalLayerWeights>,
    kv: Vec<metal::transformer::MetalKvCache>,
    cos_buf: ::metal::Buffer,
    sin_buf: ::metal::Buffer,
    metal_cfg: metal::transformer::MetalTransformerConfig,
    scratch: Option<metal::transformer::LayerScratch>,
    max_scratch_tokens: usize,
    input_buf: Option<::metal::Buffer>,
    input_buf_size: usize,
    /// GPU-resident final norm weight for forward_gpu
    norm_w_buf: ::metal::Buffer,
    /// Reusable output buffer for forward_gpu (avoids alloc per call)
    norm_out_buf: Option<::metal::Buffer>,
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
        let cpu_kv = (0..n)
            .map(|_| cpu::transformer::CpuKvCache::new())
            .collect();

        // Backend selection:
        //   FERRUM_FUSED_CPU=1  → force CPU
        //   FERRUM_FUSED_CPU=1  → force CPU
        //   FERRUM_FUSED_METAL=1 → force Metal
        //   otherwise → auto: Metal for large models (28-layer talker), CPU for small (SubTalker/vocoder)
        let use_cpu = if std::env::var("FERRUM_FUSED_CPU").as_deref() == Ok("1") {
            true
        } else if std::env::var("FERRUM_FUSED_METAL").as_deref() == Ok("1") {
            false
        } else {
            // Metal for talker (28) + vocoder (8), CPU for SubTalker (5) only
            // All Metal: total pipeline is faster even though SubTalker per-step is slower,
            // because GPU pipeline amortizes overhead across the full decode loop.
            false
        };

        #[cfg(feature = "metal")]
        let metal_state = {
            if let Some(device) = ::metal::Device::system_default() {
                let pipes = metal::pipelines::MetalPipelines::new(&device);
                let weights: Vec<_> = layers
                    .iter()
                    .map(|lw| {
                        metal::transformer::MetalLayerWeights {
                            input_ln_w: pipes.buffer_from_data(&lw.input_ln_w),
                            q_proj_w: pipes.buffer_from_data(&lw.q_proj_w),
                            k_proj_w: pipes.buffer_from_data(&lw.k_proj_w),
                            v_proj_w: pipes.buffer_from_data(&lw.v_proj_w),
                            o_proj_w: pipes.buffer_from_data(&lw.o_proj_w),
                            q_norm_w: if lw.q_norm_w.is_empty() {
                                pipes.buffer_from_data(&[1.0f32]) // dummy, won't be used
                            } else {
                                pipes.buffer_from_data(&lw.q_norm_w)
                            },
                            k_norm_w: if lw.k_norm_w.is_empty() {
                                pipes.buffer_from_data(&[1.0f32])
                            } else {
                                pipes.buffer_from_data(&lw.k_norm_w)
                            },
                            post_ln_w: pipes.buffer_from_data(&lw.post_ln_w),
                            gate_proj_w: pipes.buffer_from_data(&lw.gate_proj_w),
                            up_proj_w: pipes.buffer_from_data(&lw.up_proj_w),
                            down_proj_w: pipes.buffer_from_data(&lw.down_proj_w),
                            has_qk_norm: !lw.q_norm_w.is_empty(),
                            attn_scale: lw
                                .attn_layer_scale
                                .as_ref()
                                .map(|s| pipes.buffer_from_data(s)),
                            mlp_scale: lw
                                .mlp_layer_scale
                                .as_ref()
                                .map(|s| pipes.buffer_from_data(s)),
                        }
                    })
                    .collect();
                let kv_max_len = cfg.max_position_embeddings.min(4096);
                let kv = (0..n)
                    .map(|_| {
                        metal::transformer::MetalKvCache::new(
                            &pipes,
                            cfg.num_kv_heads,
                            cfg.head_dim,
                            kv_max_len,
                        )
                    })
                    .collect();
                let metal_cfg = metal::transformer::MetalTransformerConfig {
                    hidden_size: cfg.hidden_size,
                    intermediate_size: cfg.intermediate_size,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    rms_norm_eps: cfg.rms_norm_eps as f32,
                };
                let cos_buf = pipes.buffer_from_data(&cos);
                let sin_buf = pipes.buffer_from_data(&sin);
                let norm_w_buf = pipes.buffer_from_data(&norm_w);
                Some(MetalTransformerState {
                    pipes,
                    weights,
                    kv,
                    cos_buf,
                    sin_buf,
                    metal_cfg,
                    scratch: None,
                    max_scratch_tokens: 0,
                    input_buf: None,
                    input_buf_size: 0,
                    norm_w_buf,
                    norm_out_buf: None,
                })
            } else {
                None
            }
        };

        let backend = if use_cpu {
            "CPU (Accelerate)"
        } else {
            "Metal+Accelerate"
        };
        // Log backend selection (visible with RUST_LOG=info)
        #[cfg(feature = "metal")]
        tracing::info!(
            "FusedTransformer: backend={backend}, hidden={}, layers={n}",
            cfg.hidden_size
        );
        #[cfg(not(feature = "metal"))]
        tracing::info!(
            "FusedTransformer: backend=CPU, hidden={}, layers={n}",
            cfg.hidden_size
        );

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
            use_cpu,
        }
    }

    /// Forward: input [tokens, hidden] → output [tokens, hidden] (f32 vecs).
    pub fn forward(&mut self, input: &[f32], tokens: usize) -> Vec<f32> {
        let pos_offset = self.tokens_generated;
        let h = self.cfg.hidden_size;

        #[cfg(feature = "metal")]
        if !self.use_cpu {
            if let Some(ref mut ms) = self.metal_state {
                // Allocate/resize scratch buffers if needed
                if ms.scratch.is_none() || ms.max_scratch_tokens < tokens {
                    ms.scratch = Some(metal::transformer::LayerScratch::new(
                        &ms.pipes,
                        tokens,
                        h,
                        ms.metal_cfg.intermediate_size,
                        ms.metal_cfg.num_heads,
                        ms.metal_cfg.num_kv_heads,
                        ms.metal_cfg.head_dim,
                    ));
                    ms.max_scratch_tokens = tokens;
                }
                let scratch = ms.scratch.as_ref().unwrap();

                // Reuse or allocate input buffer (shared memory = zero-copy write on Apple Silicon)
                let needed = tokens * h;
                if ms.input_buf.is_none() || ms.input_buf_size < needed {
                    ms.input_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h))); // preallocate for up to 128 tokens
                    ms.input_buf_size = needed.max(128 * h);
                }
                let input_buf = ms.input_buf.as_ref().unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr(),
                        input_buf.contents() as *mut f32,
                        needed,
                    );
                }

                let cmd = ms.pipes.queue.new_command_buffer();

                // Layer 0: input from input_buf
                metal::transformer::metal_layer_forward_v2(
                    cmd,
                    &ms.pipes,
                    &input_buf,
                    tokens,
                    &ms.weights[0],
                    &ms.metal_cfg,
                    &mut ms.kv[0],
                    pos_offset,
                    &ms.cos_buf,
                    &ms.sin_buf,
                    scratch,
                );

                // Layers 1..N: input from scratch.output (ping-pong via copy)
                for li in 1..ms.weights.len() {
                    // Copy scratch.output to input_buf for next layer
                    let enc = cmd.new_blit_command_encoder();
                    enc.copy_from_buffer(
                        &scratch.output,
                        0,
                        &input_buf,
                        0,
                        (tokens * h * 4) as u64,
                    );
                    enc.end_encoding();

                    metal::transformer::metal_layer_forward_v2(
                        cmd,
                        &ms.pipes,
                        &input_buf,
                        tokens,
                        &ms.weights[li],
                        &ms.metal_cfg,
                        &mut ms.kv[li],
                        pos_offset,
                        &ms.cos_buf,
                        &ms.sin_buf,
                        scratch,
                    );
                }

                // Single commit+wait for all layers
                cmd.commit();
                cmd.wait_until_completed();

                let hidden =
                    metal::pipelines::MetalPipelines::read_buffer(&scratch.output, tokens * h);
                self.tokens_generated += tokens;
                return self.final_rms_norm(&hidden, tokens);
            }
        } // !self.use_cpu

        // CPU path (Accelerate sgemm + SIMD element-wise)
        let mut hidden = input.to_vec();
        for li in 0..self.cpu_layers.len() {
            hidden = cpu::transformer::cpu_layer_forward(
                &hidden,
                tokens,
                &self.cpu_layers[li],
                &self.cfg,
                &self.cos,
                &self.sin,
                &mut self.cpu_kv[li],
                pos_offset,
            );
        }
        self.tokens_generated += tokens;
        self.final_rms_norm(&hidden, tokens)
    }

    /// Forward on GPU, returning Metal Buffer directly (zero CPU transfer).
    /// Input: raw f32 slice (will be copied to GPU once).
    /// Output: Metal Buffer containing normed hidden [tokens, hidden].
    /// Falls back to CPU path if Metal not available.
    #[cfg(feature = "metal")]
    pub fn forward_gpu(
        &mut self,
        input: &[f32],
        tokens: usize,
    ) -> Option<(::metal::Buffer, usize)> {
        let pos_offset = self.tokens_generated;
        let h = self.cfg.hidden_size;

        if self.use_cpu {
            return None;
        }

        let ms = self.metal_state.as_mut()?;

        // Allocate scratch
        if ms.scratch.is_none() || ms.max_scratch_tokens < tokens {
            ms.scratch = Some(metal::transformer::LayerScratch::new(
                &ms.pipes,
                tokens,
                h,
                ms.metal_cfg.intermediate_size,
                ms.metal_cfg.num_heads,
                ms.metal_cfg.num_kv_heads,
                ms.metal_cfg.head_dim,
            ));
            ms.max_scratch_tokens = tokens;
        }
        let scratch = ms.scratch.as_ref().unwrap();

        // Input buffer
        let needed = tokens * h;
        if ms.input_buf.is_none() || ms.input_buf_size < needed {
            ms.input_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h)));
            ms.input_buf_size = needed.max(128 * h);
        }
        let input_buf = ms.input_buf.as_ref().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), input_buf.contents() as *mut f32, needed);
        }

        let cmd = ms.pipes.queue.new_command_buffer();

        // All transformer layers
        metal::transformer::metal_layer_forward_v2(
            cmd,
            &ms.pipes,
            input_buf,
            tokens,
            &ms.weights[0],
            &ms.metal_cfg,
            &mut ms.kv[0],
            pos_offset,
            &ms.cos_buf,
            &ms.sin_buf,
            scratch,
        );
        for li in 1..ms.weights.len() {
            let enc = cmd.new_blit_command_encoder();
            enc.copy_from_buffer(&scratch.output, 0, input_buf, 0, (tokens * h * 4) as u64);
            enc.end_encoding();
            metal::transformer::metal_layer_forward_v2(
                cmd,
                &ms.pipes,
                input_buf,
                tokens,
                &ms.weights[li],
                &ms.metal_cfg,
                &mut ms.kv[li],
                pos_offset,
                &ms.cos_buf,
                &ms.sin_buf,
                scratch,
            );
        }

        // Final RMSNorm on GPU
        if ms.norm_out_buf.is_none() {
            ms.norm_out_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h)));
        }
        let norm_out = ms.norm_out_buf.as_ref().unwrap();
        {
            let enc = cmd.new_compute_command_encoder();
            ms.pipes.rms_norm_enc(
                enc,
                &scratch.output,
                &ms.norm_w_buf,
                norm_out,
                tokens,
                h,
                self.cfg.rms_norm_eps as f32,
            );
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        self.tokens_generated += tokens;

        // Return buffer with normed hidden (stays on GPU)
        let result = ms.pipes.buffer_empty(tokens * h);
        // Copy norm_out to result (so caller owns it)
        let cmd2 = ms.pipes.queue.new_command_buffer();
        let enc = cmd2.new_blit_command_encoder();
        enc.copy_from_buffer(norm_out, 0, &result, 0, (tokens * h * 4) as u64);
        enc.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        Some((result, tokens * h))
    }

    /// Forward + lm_head + argmax in ONE command buffer, ZERO extra allocs.
    /// Returns (token_index, norm_hidden as Vec<f32>).
    /// Pre-allocated buffers reused across calls.
    #[cfg(feature = "metal")]
    pub fn forward_and_argmax(
        &mut self,
        input_buf: &GpuBuffer,
        tokens: usize,
        lm_weights_buf: &GpuBuffer,
        vocab_size: usize,
    ) -> Option<(u32, Vec<f32>)> {
        let pos_offset = self.tokens_generated;
        let h = self.cfg.hidden_size;
        if self.use_cpu {
            return None;
        }

        let ms = self.metal_state.as_mut()?;

        // Ensure scratch allocated
        if ms.scratch.is_none() || ms.max_scratch_tokens < tokens {
            ms.scratch = Some(metal::transformer::LayerScratch::new(
                &ms.pipes,
                tokens,
                h,
                ms.metal_cfg.intermediate_size,
                ms.metal_cfg.num_heads,
                ms.metal_cfg.num_kv_heads,
                ms.metal_cfg.head_dim,
            ));
            ms.max_scratch_tokens = tokens;
        }
        let scratch = ms.scratch.as_ref().unwrap();
        let needed = tokens * h;
        if ms.input_buf.is_none() || ms.input_buf_size < needed {
            ms.input_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h)));
            ms.input_buf_size = needed.max(128 * h);
        }
        let int_buf = ms.input_buf.as_ref().unwrap();
        if ms.norm_out_buf.is_none() {
            ms.norm_out_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h)));
        }
        let norm_out = ms.norm_out_buf.as_ref().unwrap();

        // === SINGLE command buffer: layers + norm + lm_head + argmax ===
        let cmd = ms.pipes.queue.new_command_buffer();

        metal::transformer::metal_layer_forward_v2(
            cmd,
            &ms.pipes,
            input_buf,
            tokens,
            &ms.weights[0],
            &ms.metal_cfg,
            &mut ms.kv[0],
            pos_offset,
            &ms.cos_buf,
            &ms.sin_buf,
            scratch,
        );
        for li in 1..ms.weights.len() {
            let enc = cmd.new_blit_command_encoder();
            enc.copy_from_buffer(&scratch.output, 0, int_buf, 0, (needed * 4) as u64);
            enc.end_encoding();
            metal::transformer::metal_layer_forward_v2(
                cmd,
                &ms.pipes,
                int_buf,
                tokens,
                &ms.weights[li],
                &ms.metal_cfg,
                &mut ms.kv[li],
                pos_offset,
                &ms.cos_buf,
                &ms.sin_buf,
                scratch,
            );
        }

        // RMSNorm
        {
            let enc = cmd.new_compute_command_encoder();
            ms.pipes.rms_norm_enc(
                enc,
                &scratch.output,
                &ms.norm_w_buf,
                norm_out,
                tokens,
                h,
                self.cfg.rms_norm_eps as f32,
            );
            enc.end_encoding();
        }

        // lm_head GEMM (reuse int_buf as logits since it's ≥ vocab_size for small models)
        // For safety, use a dedicated buffer only if needed
        let logits_buf = if ms.input_buf_size >= vocab_size {
            // Can't reuse int_buf — it might be read by norm. Use scratch.ln_out as temp.
            &scratch.gate_buf // gate_buf is large enough (intermediate_size ≥ vocab_size)
        } else {
            &scratch.gate_buf
        };
        {
            let enc = cmd.new_compute_command_encoder();
            ms.pipes
                .gemm_v2(enc, norm_out, lm_weights_buf, logits_buf, 1, vocab_size, h);
            enc.end_encoding();
        }

        // Argmax (reuse scratch.up_buf for result — just need 1 u32 = 4 bytes)
        let result_ptr = scratch.up_buf.contents() as *mut u32;
        {
            let enc = cmd.new_compute_command_encoder();
            #[repr(C)]
            struct P {
                n: i32,
            }
            let p = P {
                n: vocab_size as i32,
            };
            let p_buf = ms.pipes.device.new_buffer_with_data(
                &p as *const _ as *const std::ffi::c_void,
                4,
                ::metal::MTLResourceOptions::StorageModeShared,
            );
            enc.set_compute_pipeline_state(ms.pipes.pipeline("argmax_f32"));
            enc.set_buffer(0, Some(logits_buf), 0);
            enc.set_buffer(1, Some(&scratch.up_buf), 0);
            enc.set_buffer(2, Some(&p_buf), 0);
            enc.dispatch_thread_groups(
                ::metal::MTLSize::new(1, 1, 1),
                ::metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();
        self.tokens_generated += tokens;

        // Read results from shared memory (zero-copy on Apple Silicon)
        let token = unsafe { *result_ptr };
        let hidden_vec = metal::pipelines::MetalPipelines::read_buffer(norm_out, needed);

        Some((token, hidden_vec))
    }

    /// Forward on GPU from a Metal Buffer input. Zero CPU transfer.
    /// Returns normed hidden as Metal Buffer.
    #[cfg(feature = "metal")]
    pub fn forward_gpu_buffer(
        &mut self,
        input_buf: &::metal::Buffer,
        tokens: usize,
    ) -> Option<::metal::Buffer> {
        let pos_offset = self.tokens_generated;
        let h = self.cfg.hidden_size;
        if self.use_cpu {
            return None;
        }
        let ms = self.metal_state.as_mut()?;

        if ms.scratch.is_none() || ms.max_scratch_tokens < tokens {
            ms.scratch = Some(metal::transformer::LayerScratch::new(
                &ms.pipes,
                tokens,
                h,
                ms.metal_cfg.intermediate_size,
                ms.metal_cfg.num_heads,
                ms.metal_cfg.num_kv_heads,
                ms.metal_cfg.head_dim,
            ));
            ms.max_scratch_tokens = tokens;
        }
        let scratch = ms.scratch.as_ref().unwrap();

        let cmd = ms.pipes.queue.new_command_buffer();

        // All transformer layers (input from caller's buffer)
        metal::transformer::metal_layer_forward_v2(
            cmd,
            &ms.pipes,
            input_buf,
            tokens,
            &ms.weights[0],
            &ms.metal_cfg,
            &mut ms.kv[0],
            pos_offset,
            &ms.cos_buf,
            &ms.sin_buf,
            scratch,
        );
        // Use ms.input_buf as intermediate for layers 1..N
        let needed = tokens * h;
        if ms.input_buf.is_none() || ms.input_buf_size < needed {
            ms.input_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h)));
            ms.input_buf_size = needed.max(128 * h);
        }
        let int_buf = ms.input_buf.as_ref().unwrap();

        for li in 1..ms.weights.len() {
            let enc = cmd.new_blit_command_encoder();
            enc.copy_from_buffer(&scratch.output, 0, int_buf, 0, (tokens * h * 4) as u64);
            enc.end_encoding();
            metal::transformer::metal_layer_forward_v2(
                cmd,
                &ms.pipes,
                int_buf,
                tokens,
                &ms.weights[li],
                &ms.metal_cfg,
                &mut ms.kv[li],
                pos_offset,
                &ms.cos_buf,
                &ms.sin_buf,
                scratch,
            );
        }

        // Final RMSNorm on GPU
        if ms.norm_out_buf.is_none() {
            ms.norm_out_buf = Some(ms.pipes.buffer_empty(needed.max(128 * h)));
        }
        let norm_out = ms.norm_out_buf.as_ref().unwrap();
        {
            let enc = cmd.new_compute_command_encoder();
            ms.pipes.rms_norm_enc(
                enc,
                &scratch.output,
                &ms.norm_w_buf,
                norm_out,
                tokens,
                h,
                self.cfg.rms_norm_eps as f32,
            );
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();
        self.tokens_generated += tokens;

        // Return copy of norm output
        let result = ms.pipes.buffer_empty(tokens * h);
        let cmd2 = ms.pipes.queue.new_command_buffer();
        let enc = cmd2.new_blit_command_encoder();
        enc.copy_from_buffer(norm_out, 0, &result, 0, (tokens * h * 4) as u64);
        enc.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        Some(result)
    }

    /// Forward on GPU with GPU-side norm, returns Vec<f32>.
    /// Avoids CPU-side RMSNorm but still transfers output to CPU.
    #[cfg(feature = "metal")]
    pub fn forward_gpu_to_vec(&mut self, input: &[f32], tokens: usize) -> Option<Vec<f32>> {
        let h = self.cfg.hidden_size;
        let (buf, _) = self.forward_gpu(input, tokens)?;
        Some(metal::pipelines::MetalPipelines::read_buffer(
            &buf,
            tokens * h,
        ))
    }

    fn final_rms_norm(&self, hidden: &[f32], tokens: usize) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let mut out = vec![0.0f32; tokens * h];
        for t in 0..tokens {
            let row = &hidden[t * h..(t + 1) * h];
            let o = &mut out[t * h..(t + 1) * h];
            // vDSP_dotpr for sum-of-squares (same SIMD path as PyTorch on macOS)
            let sum_sq;
            #[cfg(feature = "metal")]
            {
                extern "C" {
                    fn vDSP_dotpr(
                        a: *const f32,
                        a_stride: i32,
                        b: *const f32,
                        b_stride: i32,
                        result: *mut f32,
                        n: u64,
                    );
                }
                let mut dot = 0.0f32;
                unsafe {
                    vDSP_dotpr(row.as_ptr(), 1, row.as_ptr(), 1, &mut dot, h as u64);
                }
                sum_sq = dot;
            }
            #[cfg(not(feature = "metal"))]
            {
                let mut v = 0.0f32;
                for &val in row {
                    v += val * val;
                }
                sum_sq = v;
            }
            let inv = 1.0f32 / (sum_sq / h as f32 + eps).sqrt();
            for i in 0..h {
                o[i] = row[i] * inv * self.norm_w[i];
            }
        }
        out
    }

    /// Create a Metal buffer from f32 data (shared memory, zero-copy on Apple Silicon).
    /// Returns None if Metal not available.
    /// Create a GPU buffer from f32 data. Returns None if Metal not available.
    pub fn create_gpu_buffer(&self, data: &[f32]) -> Option<GpuBuffer> {
        #[cfg(feature = "metal")]
        {
            let ms = self.metal_state.as_ref()?;
            Some(ms.pipes.buffer_from_data(data))
        }
        #[cfg(not(feature = "metal"))]
        {
            Some(data.to_vec())
        }
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
