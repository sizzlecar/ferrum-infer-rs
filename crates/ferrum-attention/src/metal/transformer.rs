//! Full transformer layer on Metal — all ops in GPU, no CPU roundtrip.

use metal::*;
use super::pipelines::MetalPipelines;
use crate::AttentionParams;

/// Pre-extracted weights for one transformer layer, stored as Metal buffers.
pub struct MetalLayerWeights {
    pub input_ln_w: Buffer,    // [hidden]
    pub q_proj_w: Buffer,      // [nh*hd, hidden]
    pub k_proj_w: Buffer,      // [nkv*hd, hidden]
    pub v_proj_w: Buffer,      // [nkv*hd, hidden]
    pub o_proj_w: Buffer,      // [hidden, nh*hd]
    pub q_norm_w: Buffer,      // [hd]
    pub k_norm_w: Buffer,      // [hd]
    pub post_ln_w: Buffer,     // [hidden]
    pub gate_proj_w: Buffer,   // [intermediate, hidden]
    pub up_proj_w: Buffer,     // [intermediate, hidden]
    pub down_proj_w: Buffer,   // [hidden, intermediate]
}

pub struct MetalTransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

/// KV cache stored as Metal buffers.
pub struct MetalKvCache {
    pub k: Option<Buffer>,  // [nkv, cached_len, hd]
    pub v: Option<Buffer>,
    pub len: usize,
}

impl MetalKvCache {
    pub fn new() -> Self { Self { k: None, v: None, len: 0 } }
    pub fn reset(&mut self) { self.k = None; self.v = None; self.len = 0; }
}

/// Run one transformer layer entirely on Metal.
///
/// input: Metal buffer [tokens, hidden] (contiguous f32)
/// Returns: output Metal buffer [tokens, hidden]
pub fn metal_layer_forward(
    pipes: &MetalPipelines,
    input: &Buffer,
    tokens: usize,
    w: &MetalLayerWeights,
    cfg: &MetalTransformerConfig,
    kv_cache: &mut MetalKvCache,
    pos_offset: usize,
    cos_buf: &Buffer,  // [max_seq, hd/2] precomputed
    sin_buf: &Buffer,
) -> Buffer {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;

    let cmd = pipes.queue.new_command_buffer();

    // 1. Input LayerNorm → ln_out[tokens, h]
    let ln_out = pipes.buffer_empty(tokens * h);
    pipes.rms_norm(cmd, input, &w.input_ln_w, &ln_out, tokens, h, cfg.rms_norm_eps);

    // 2. Q/K/V projections
    let q_buf = pipes.buffer_empty(tokens * nh * hd);
    let k_buf = pipes.buffer_empty(tokens * nkv * hd);
    let v_buf = pipes.buffer_empty(tokens * nkv * hd);
    pipes.gemm(cmd, &ln_out, &w.q_proj_w, &q_buf, tokens, nh * hd, h);
    pipes.gemm(cmd, &ln_out, &w.k_proj_w, &k_buf, tokens, nkv * hd, h);
    pipes.gemm(cmd, &ln_out, &w.v_proj_w, &v_buf, tokens, nkv * hd, h);

    // Commit and wait — needed because subsequent ops need reshape/RoPE on CPU
    // TODO: implement QK-norm + RoPE as Metal kernels to avoid this sync
    cmd.commit();
    cmd.wait_until_completed();

    // 3. QK-norm + RoPE (CPU for now — these are element-wise, fast)
    let mut q_data = MetalPipelines::read_buffer(&q_buf, tokens * nh * hd);
    let mut k_data = MetalPipelines::read_buffer(&k_buf, tokens * nkv * hd);
    let v_data = MetalPipelines::read_buffer(&v_buf, tokens * nkv * hd);

    // Reshape [tokens, heads*hd] → [heads, tokens, hd] and apply QK norm + RoPE
    let mut q_r = vec![0.0f32; nh * tokens * hd];
    let mut k_r = vec![0.0f32; nkv * tokens * hd];
    let mut v_r = vec![0.0f32; nkv * tokens * hd];

    // Transpose and apply QK norm
    let cos_size = (cos_buf.length() / 4) as usize;
    let sin_size = (sin_buf.length() / 4) as usize;
    let cos_data = MetalPipelines::read_buffer(cos_buf, cos_size);
    let sin_data = MetalPipelines::read_buffer(sin_buf, sin_size);
    let q_norm_w = MetalPipelines::read_buffer(&w.q_norm_w, hd);
    let k_norm_w = MetalPipelines::read_buffer(&w.k_norm_w, hd);

    transpose_and_norm(&q_data, &mut q_r, tokens, nh, hd, &q_norm_w, cfg.rms_norm_eps as f64);
    transpose_and_norm(&k_data, &mut k_r, tokens, nkv, hd, &k_norm_w, cfg.rms_norm_eps as f64);
    // V: just transpose, no norm
    for t in 0..tokens {
        for hi in 0..nkv {
            for d in 0..hd {
                v_r[hi * tokens * hd + t * hd + d] = v_data[t * nkv * hd + hi * hd + d];
            }
        }
    }

    // RoPE
    apply_rope(&mut q_r, nh, tokens, hd, &cos_data, &sin_data, pos_offset);
    apply_rope(&mut k_r, nkv, tokens, hd, &cos_data, &sin_data, pos_offset);

    // 4. KV cache update
    let (full_k, full_v, kv_len) = update_kv_cache(kv_cache, &k_r, &v_r, nkv, tokens, hd, pipes);

    // 5. Flash attention (on Metal)
    let attn_params = AttentionParams {
        batch: 1, num_heads: nh, num_kv_heads: nkv,
        q_len: tokens, kv_len, head_dim: hd,
        causal: tokens > 1, pos_offset,
    };

    // GQA: repeat K/V for attention
    let n_rep = nh / nkv;
    let k_rep = repeat_kv_data(&full_k, nkv, n_rep, kv_len, hd);
    let v_rep = repeat_kv_data(&full_v, nkv, n_rep, kv_len, hd);

    let q_metal = pipes.buffer_from_data(&q_r);
    let k_metal = pipes.buffer_from_data(&k_rep);
    let v_metal = pipes.buffer_from_data(&v_rep);
    let attn_out_metal = pipes.buffer_empty(nh * tokens * hd);

    let cmd2 = pipes.queue.new_command_buffer();
    pipes.flash_attn(cmd2, &q_metal, &k_metal, &v_metal, &attn_out_metal, &attn_params);
    cmd2.commit();
    cmd2.wait_until_completed();

    // 6. Transpose [heads, tokens, hd] → [tokens, heads*hd] and O projection
    let attn_out = MetalPipelines::read_buffer(&attn_out_metal, nh * tokens * hd);
    let mut attn_flat = vec![0.0f32; tokens * nh * hd];
    for t in 0..tokens {
        for hi in 0..nh {
            for d in 0..hd {
                attn_flat[t * nh * hd + hi * hd + d] = attn_out[hi * tokens * hd + t * hd + d];
            }
        }
    }

    let attn_flat_buf = pipes.buffer_from_data(&attn_flat);
    let o_out = pipes.buffer_empty(tokens * h);

    let cmd3 = pipes.queue.new_command_buffer();
    pipes.gemm(cmd3, &attn_flat_buf, &w.o_proj_w, &o_out, tokens, h, nh * hd);

    // 7. Residual add: hidden = input + o_out
    let hidden = pipes.buffer_empty(tokens * h);
    pipes.add(cmd3, input, &o_out, &hidden, tokens * h);

    // 8. Post-attention LayerNorm
    let post_ln = pipes.buffer_empty(tokens * h);
    pipes.rms_norm(cmd3, &hidden, &w.post_ln_w, &post_ln, tokens, h, cfg.rms_norm_eps);

    // 9. MLP: gate, up, silu_mul, down
    let gate_buf = pipes.buffer_empty(tokens * im);
    let up_buf = pipes.buffer_empty(tokens * im);
    pipes.gemm(cmd3, &post_ln, &w.gate_proj_w, &gate_buf, tokens, im, h);
    pipes.gemm(cmd3, &post_ln, &w.up_proj_w, &up_buf, tokens, im, h);

    let silu_out = pipes.buffer_empty(tokens * im);
    pipes.silu_mul(cmd3, &gate_buf, &up_buf, &silu_out, tokens * im);

    let mlp_out = pipes.buffer_empty(tokens * h);
    pipes.gemm(cmd3, &silu_out, &w.down_proj_w, &mlp_out, tokens, h, im);

    // 10. Residual add: output = hidden + mlp_out
    let output = pipes.buffer_empty(tokens * h);
    pipes.add(cmd3, &hidden, &mlp_out, &output, tokens * h);

    cmd3.commit();
    cmd3.wait_until_completed();

    output
}

// ── Helper functions ────────────────────────────────────────────────────

fn transpose_and_norm(
    flat: &[f32],  // [tokens, heads*hd]
    out: &mut [f32], // [heads, tokens, hd]
    tokens: usize, heads: usize, hd: usize,
    norm_w: &[f32], eps: f64,
) {
    for t in 0..tokens {
        for hi in 0..heads {
            // Extract this head's slice
            let src_base = t * heads * hd + hi * hd;
            let dst_base = hi * tokens * hd + t * hd;

            // RMS norm on head_dim
            let mut var_sum = 0.0f64;
            for d in 0..hd {
                let v = flat[src_base + d] as f64;
                var_sum += v * v;
            }
            let inv_rms = 1.0 / (var_sum / hd as f64 + eps).sqrt();
            for d in 0..hd {
                out[dst_base + d] = (flat[src_base + d] as f64 * inv_rms) as f32 * norm_w[d];
            }
        }
    }
}

fn apply_rope(
    data: &mut [f32], heads: usize, seq: usize, hd: usize,
    cos: &[f32], sin: &[f32], offset: usize,
) {
    let half = hd / 2;
    for h in 0..heads {
        for s in 0..seq {
            let pos = offset + s;
            let base = h * seq * hd + s * hd;
            for i in 0..half {
                let c = cos[pos * half + i];
                let si = sin[pos * half + i];
                let x0 = data[base + i];
                let x1 = data[base + half + i];
                data[base + i] = x0 * c - x1 * si;
                data[base + half + i] = x1 * c + x0 * si;
            }
        }
    }
}

fn repeat_kv_data(kv: &[f32], nkv: usize, n_rep: usize, seq: usize, hd: usize) -> Vec<f32> {
    if n_rep == 1 { return kv.to_vec(); }
    let nh = nkv * n_rep;
    let mut out = vec![0.0f32; nh * seq * hd];
    for kh in 0..nkv {
        let src = &kv[kh * seq * hd..(kh + 1) * seq * hd];
        for r in 0..n_rep {
            let dst_h = kh * n_rep + r;
            out[dst_h * seq * hd..(dst_h + 1) * seq * hd].copy_from_slice(src);
        }
    }
    out
}

fn update_kv_cache(
    cache: &mut MetalKvCache,
    new_k: &[f32], new_v: &[f32],
    nkv: usize, new_len: usize, hd: usize,
    _pipes: &MetalPipelines,
) -> (Vec<f32>, Vec<f32>, usize) {
    if cache.len == 0 {
        cache.len = new_len;
        cache.k = Some(_pipes.buffer_from_data(new_k));
        cache.v = Some(_pipes.buffer_from_data(new_v));
        let k = new_k.to_vec();
        let v = new_v.to_vec();
        (k, v, new_len)
    } else {
        let old_len = cache.len;
        let total = old_len + new_len;
        let old_k = cache.k.as_ref().map(|b| MetalPipelines::read_buffer(b, nkv * old_len * hd)).unwrap_or_default();
        let old_v = cache.v.as_ref().map(|b| MetalPipelines::read_buffer(b, nkv * old_len * hd)).unwrap_or_default();

        let mut full_k = vec![0.0f32; nkv * total * hd];
        let mut full_v = vec![0.0f32; nkv * total * hd];
        for h in 0..nkv {
            // Copy old
            full_k[h * total * hd..h * total * hd + old_len * hd]
                .copy_from_slice(&old_k[h * old_len * hd..(h + 1) * old_len * hd]);
            // Copy new
            full_k[h * total * hd + old_len * hd..h * total * hd + total * hd]
                .copy_from_slice(&new_k[h * new_len * hd..(h + 1) * new_len * hd]);
            // V
            full_v[h * total * hd..h * total * hd + old_len * hd]
                .copy_from_slice(&old_v[h * old_len * hd..(h + 1) * old_len * hd]);
            full_v[h * total * hd + old_len * hd..h * total * hd + total * hd]
                .copy_from_slice(&new_v[h * new_len * hd..(h + 1) * new_len * hd]);
        }

        cache.len = total;
        cache.k = Some(_pipes.buffer_from_data(&full_k));
        cache.v = Some(_pipes.buffer_from_data(&full_v));

        (full_k, full_v, total)
    }
}
