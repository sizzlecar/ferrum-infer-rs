//! All-Metal transformer layer — single command buffer, zero CPU-GPU sync.
//! GEMM via simdgroup_multiply_accumulate (64x32 tiles), all ops on GPU.

use metal::*;
use super::pipelines::MetalPipelines;
use crate::AttentionParams;

pub struct MetalLayerWeights {
    pub input_ln_w: Buffer,
    pub q_proj_w: Buffer,
    pub k_proj_w: Buffer,
    pub v_proj_w: Buffer,
    pub o_proj_w: Buffer,
    pub q_norm_w: Buffer,
    pub k_norm_w: Buffer,
    pub post_ln_w: Buffer,
    pub gate_proj_w: Buffer,
    pub up_proj_w: Buffer,
    pub down_proj_w: Buffer,
    /// False = skip QK-norm in qk_norm_rope kernel (vocoder has no QK-norm)
    pub has_qk_norm: bool,
}

pub struct MetalTransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

/// GPU-resident KV cache with pre-allocated buffers.
pub struct MetalKvCache {
    pub k_buf: Buffer,   // [nkv, max_len, hd]
    pub v_buf: Buffer,   // [nkv, max_len, hd]
    pub len: usize,
    pub max_len: usize,
}

impl MetalKvCache {
    pub fn new(pipes: &MetalPipelines, nkv: usize, hd: usize, max_len: usize) -> Self {
        let size = nkv * max_len * hd;
        Self {
            k_buf: pipes.buffer_empty(size),
            v_buf: pipes.buffer_empty(size),
            len: 0,
            max_len,
        }
    }
    pub fn reset(&mut self) { self.len = 0; }
}

/// Pre-allocated scratch buffers for one layer forward (reused across layers).
pub struct LayerScratch {
    pub ln_out: Buffer,
    pub q_buf: Buffer,
    pub k_buf: Buffer,
    pub v_buf: Buffer,
    pub q_ready: Buffer,
    pub k_ready: Buffer,
    pub v_ready: Buffer,
    pub attn_out: Buffer,
    pub attn_flat: Buffer,
    pub o_out: Buffer,
    pub hidden: Buffer,
    pub post_ln: Buffer,
    pub gate_buf: Buffer,
    pub up_buf: Buffer,
    pub silu_out: Buffer,
    pub mlp_out: Buffer,
    pub output: Buffer,
}

impl LayerScratch {
    pub fn new(pipes: &MetalPipelines, tokens: usize, h: usize, im: usize, nh: usize, nkv: usize, hd: usize) -> Self {
        Self {
            ln_out: pipes.buffer_empty(tokens * h),
            q_buf: pipes.buffer_empty(tokens * nh * hd),
            k_buf: pipes.buffer_empty(tokens * nkv * hd),
            v_buf: pipes.buffer_empty(tokens * nkv * hd),
            q_ready: pipes.buffer_empty(nh * tokens * hd),
            k_ready: pipes.buffer_empty(nkv * tokens * hd),
            v_ready: pipes.buffer_empty(nkv * tokens * hd),
            attn_out: pipes.buffer_empty(nh * tokens * hd),
            attn_flat: pipes.buffer_empty(tokens * nh * hd),
            o_out: pipes.buffer_empty(tokens * h),
            hidden: pipes.buffer_empty(tokens * h),
            post_ln: pipes.buffer_empty(tokens * h),
            gate_buf: pipes.buffer_empty(tokens * im),
            up_buf: pipes.buffer_empty(tokens * im),
            silu_out: pipes.buffer_empty(tokens * im),
            mlp_out: pipes.buffer_empty(tokens * h),
            output: pipes.buffer_empty(tokens * h),
        }
    }
}

/// Run one transformer layer entirely on Metal.
/// Encodes into `cmd` WITHOUT commit — caller commits after all layers.
/// Output is written to `s.output`. Caller should use `&s.output` as input to the next layer.
pub fn metal_layer_forward_v2(
    cmd: &CommandBufferRef,
    pipes: &MetalPipelines,
    input: &Buffer,
    tokens: usize,
    w: &MetalLayerWeights,
    cfg: &MetalTransformerConfig,
    kv_cache: &mut MetalKvCache,
    pos_offset: usize,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    s: &LayerScratch,
) {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;

    // Encoder 1: RMSNorm
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.rms_norm_enc(enc, input, &w.input_ln_w, &s.ln_out, tokens, h, cfg.rms_norm_eps);
        enc.end_encoding();
    }

    // Encoder 2: Q/K/V GEMM (3 independent dispatches, 1 encoder)
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.gemm_v2(enc, &s.ln_out, &w.q_proj_w, &s.q_buf, tokens, nh * hd, h);
        pipes.gemm_v2(enc, &s.ln_out, &w.k_proj_w, &s.k_buf, tokens, nkv * hd, h);
        pipes.gemm_v2(enc, &s.ln_out, &w.v_proj_w, &s.v_buf, tokens, nkv * hd, h);
        enc.end_encoding();
    }

    // Encoder 3: QK-norm + RoPE + transpose (3 independent dispatches)
    {
        let enc = cmd.new_compute_command_encoder();
        // Q/K: mode 1 (norm+RoPE) if has_qk_norm, mode 2 (RoPE only) if not
        let qk_mode: i32 = if w.has_qk_norm { 1 } else { 2 };
        pipes.qk_norm_rope(enc, &s.q_buf, &w.q_norm_w, cos_buf, sin_buf, &s.q_ready,
            tokens, nh, hd, pos_offset, cfg.rms_norm_eps, qk_mode);
        pipes.qk_norm_rope(enc, &s.k_buf, &w.k_norm_w, cos_buf, sin_buf, &s.k_ready,
            tokens, nkv, hd, pos_offset, cfg.rms_norm_eps, qk_mode);
        // V: mode 0 (transpose only, no norm, no RoPE)
        pipes.qk_norm_rope(enc, &s.v_buf, &w.k_norm_w, cos_buf, sin_buf, &s.v_ready,
            tokens, nkv, hd, pos_offset, cfg.rms_norm_eps, 0); // mode 0: transpose only
        enc.end_encoding();
    }

    // Encoder 4: KV cache append
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.kv_cache_append(enc, &s.k_ready, &kv_cache.k_buf,
            nkv, hd, kv_cache.len, tokens, kv_cache.max_len);
        pipes.kv_cache_append(enc, &s.v_ready, &kv_cache.v_buf,
            nkv, hd, kv_cache.len, tokens, kv_cache.max_len);
        enc.end_encoding();
    }
    let kv_len = kv_cache.len + tokens;
    kv_cache.len = kv_len;

    // Encoder 5: Flash attention (GQA handled internally)
    {
        let params = AttentionParams {
            batch: 1, num_heads: nh, num_kv_heads: nkv,
            q_len: tokens, kv_len, head_dim: hd,
            causal: tokens > 1, pos_offset,
        };
        // flash_attn creates its own encoder; kv_seq_stride=max_len for GPU cache
        pipes.flash_attn_v2(cmd, &s.q_ready, &kv_cache.k_buf, &kv_cache.v_buf, &s.attn_out,
            &params, kv_cache.max_len);
    }

    // Encoder 6: Untranspose
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.transpose_out(enc, &s.attn_out, &s.attn_flat, tokens, nh, hd);
        enc.end_encoding();
    }

    // Encoder 7: O projection GEMM
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.gemm_v2(enc, &s.attn_flat, &w.o_proj_w, &s.o_out, tokens, h, nh * hd);
        enc.end_encoding();
    }

    // Encoder 8: Residual add
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.add_enc(enc, input, &s.o_out, &s.hidden, tokens * h);
        enc.end_encoding();
    }

    // Encoder 9: Post-attention RMSNorm
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.rms_norm_enc(enc, &s.hidden, &w.post_ln_w, &s.post_ln, tokens, h, cfg.rms_norm_eps);
        enc.end_encoding();
    }

    // Encoder 10: Gate/Up GEMM (2 independent dispatches)
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.gemm_v2(enc, &s.post_ln, &w.gate_proj_w, &s.gate_buf, tokens, im, h);
        pipes.gemm_v2(enc, &s.post_ln, &w.up_proj_w, &s.up_buf, tokens, im, h);
        enc.end_encoding();
    }

    // Encoder 11: SiLU × gate
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.silu_mul_enc(enc, &s.gate_buf, &s.up_buf, &s.silu_out, tokens * im);
        enc.end_encoding();
    }

    // Encoder 12: Down projection GEMM
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.gemm_v2(enc, &s.silu_out, &w.down_proj_w, &s.mlp_out, tokens, h, im);
        enc.end_encoding();
    }

    // Encoder 13: Final residual add
    {
        let enc = cmd.new_compute_command_encoder();
        pipes.add_enc(enc, &s.hidden, &s.mlp_out, &s.output, tokens * h);
        enc.end_encoding();
    }
    // Output is in s.output. Caller reads it after commit+wait.
}
