//! Full transformer layer — GEMM on Accelerate, attention + element-wise on Metal.
//! All data in Metal shared buffers (zero-copy on Apple Silicon unified memory).

use metal::*;
use super::pipelines::MetalPipelines;
use crate::AttentionParams;

pub struct MetalLayerWeights {
    pub input_ln_w: Buffer,
    pub q_proj_w: Buffer,
    pub k_proj_w: Buffer,
    pub v_proj_w: Buffer,
    pub o_proj_w: Buffer,
    pub q_norm_w: Vec<f32>,  // small, keep on CPU
    pub k_norm_w: Vec<f32>,
    pub post_ln_w: Buffer,
    pub gate_proj_w: Buffer,
    pub up_proj_w: Buffer,
    pub down_proj_w: Buffer,
}

pub struct MetalTransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

pub struct MetalKvCache {
    pub k: Vec<f32>,  // [nkv, cached_len, hd]
    pub v: Vec<f32>,
    pub len: usize,
}

impl MetalKvCache {
    pub fn new() -> Self { Self { k: Vec::new(), v: Vec::new(), len: 0 } }
    pub fn reset(&mut self) { self.k.clear(); self.v.clear(); self.len = 0; }
}

/// Run one transformer layer. Input/output are Metal shared buffers.
pub fn metal_layer_forward(
    pipes: &MetalPipelines,
    input: &Buffer,
    tokens: usize,
    w: &MetalLayerWeights,
    cfg: &MetalTransformerConfig,
    kv_cache: &mut MetalKvCache,
    pos_offset: usize,
    cos: &[f32],
    sin: &[f32],
) -> Buffer {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;

    // 1. RMSNorm (Metal kernel)
    let ln_out = pipes.buffer_empty(tokens * h);
    {
        let cmd = pipes.queue.new_command_buffer();
        pipes.rms_norm(cmd, input, &w.input_ln_w, &ln_out, tokens, h, cfg.rms_norm_eps);
        cmd.commit();
        cmd.wait_until_completed();
    }

    // 2. Q/K/V projections (Accelerate sgemm on shared buffers — zero-copy)
    let q_buf = pipes.buffer_empty(tokens * nh * hd);
    let k_buf = pipes.buffer_empty(tokens * nkv * hd);
    let v_buf = pipes.buffer_empty(tokens * nkv * hd);
    let dummy_cmd = pipes.queue.new_command_buffer();
    pipes.gemm(dummy_cmd, &ln_out, &w.q_proj_w, &q_buf, tokens, nh * hd, h);
    pipes.gemm(dummy_cmd, &ln_out, &w.k_proj_w, &k_buf, tokens, nkv * hd, h);
    pipes.gemm(dummy_cmd, &ln_out, &w.v_proj_w, &v_buf, tokens, nkv * hd, h);
    // No commit needed — cblas_sgemm runs synchronously on CPU

    // 3. QK-norm + RoPE (CPU, fast element-wise)
    let q_data = read_buf(&q_buf, tokens * nh * hd);
    let k_data = read_buf(&k_buf, tokens * nkv * hd);
    let v_data = read_buf(&v_buf, tokens * nkv * hd);

    let mut q_r = transpose_and_norm(&q_data, tokens, nh, hd, &w.q_norm_w, cfg.rms_norm_eps as f64);
    let mut k_r = transpose_and_norm(&k_data, tokens, nkv, hd, &w.k_norm_w, cfg.rms_norm_eps as f64);
    let v_r = transpose_no_norm(&v_data, tokens, nkv, hd);

    apply_rope(&mut q_r, nh, tokens, hd, cos, sin, pos_offset);
    apply_rope(&mut k_r, nkv, tokens, hd, cos, sin, pos_offset);

    // 4. KV cache
    update_kv(kv_cache, &k_r, &v_r, nkv, tokens, hd);
    let kv_len = kv_cache.len;

    // 5. Flash attention (Metal kernel)
    let n_rep = nh / nkv;
    let k_rep = repeat_kv(&kv_cache.k, nkv, n_rep, kv_len, hd);
    let v_rep = repeat_kv(&kv_cache.v, nkv, n_rep, kv_len, hd);

    let q_metal = pipes.buffer_from_data(&q_r);
    let k_metal = pipes.buffer_from_data(&k_rep);
    let v_metal = pipes.buffer_from_data(&v_rep);
    let attn_out_metal = pipes.buffer_empty(nh * tokens * hd);
    {
        let params = AttentionParams {
            batch: 1, num_heads: nh, num_kv_heads: nkv,
            q_len: tokens, kv_len, head_dim: hd,
            causal: tokens > 1, pos_offset,
        };
        let cmd = pipes.queue.new_command_buffer();
        pipes.flash_attn(cmd, &q_metal, &k_metal, &v_metal, &attn_out_metal, &params);
        cmd.commit();
        cmd.wait_until_completed();
    }

    // 6. Untranspose + O projection (Accelerate sgemm)
    let attn_out = read_buf(&attn_out_metal, nh * tokens * hd);
    let attn_flat = untranspose(&attn_out, tokens, nh, hd);
    let attn_flat_buf = pipes.buffer_from_data(&attn_flat);
    let o_out = pipes.buffer_empty(tokens * h);
    let dummy = pipes.queue.new_command_buffer();
    pipes.gemm(dummy, &attn_flat_buf, &w.o_proj_w, &o_out, tokens, h, nh * hd);

    // 7-10. Residual + PostLN + MLP + Residual (Metal kernels + Accelerate GEMM)
    let hidden = pipes.buffer_empty(tokens * h);
    let post_ln = pipes.buffer_empty(tokens * h);
    let gate_buf = pipes.buffer_empty(tokens * im);
    let up_buf = pipes.buffer_empty(tokens * im);
    let silu_out = pipes.buffer_empty(tokens * im);
    let mlp_out = pipes.buffer_empty(tokens * h);
    let output = pipes.buffer_empty(tokens * h);

    {
        let cmd = pipes.queue.new_command_buffer();
        pipes.add(cmd, input, &o_out, &hidden, tokens * h);
        pipes.rms_norm(cmd, &hidden, &w.post_ln_w, &post_ln, tokens, h, cfg.rms_norm_eps);
        cmd.commit();
        cmd.wait_until_completed();
    }

    // MLP GEMM (Accelerate)
    let dummy = pipes.queue.new_command_buffer();
    pipes.gemm(dummy, &post_ln, &w.gate_proj_w, &gate_buf, tokens, im, h);
    pipes.gemm(dummy, &post_ln, &w.up_proj_w, &up_buf, tokens, im, h);

    {
        let cmd = pipes.queue.new_command_buffer();
        pipes.silu_mul(cmd, &gate_buf, &up_buf, &silu_out, tokens * im);
        cmd.commit();
        cmd.wait_until_completed();
    }

    let dummy = pipes.queue.new_command_buffer();
    pipes.gemm(dummy, &silu_out, &w.down_proj_w, &mlp_out, tokens, h, im);

    {
        let cmd = pipes.queue.new_command_buffer();
        pipes.add(cmd, &hidden, &mlp_out, &output, tokens * h);
        cmd.commit();
        cmd.wait_until_completed();
    }

    output
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn read_buf(buf: &Buffer, len: usize) -> Vec<f32> {
    MetalPipelines::read_buffer(buf, len)
}

fn transpose_and_norm(flat: &[f32], tokens: usize, heads: usize, hd: usize, w: &[f32], eps: f64) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * tokens * hd];
    for t in 0..tokens {
        for hi in 0..heads {
            let src = t * heads * hd + hi * hd;
            let dst = hi * tokens * hd + t * hd;
            let mut v = 0.0f64;
            for d in 0..hd { let f = flat[src+d] as f64; v += f * f; }
            let inv = 1.0 / (v / hd as f64 + eps).sqrt();
            for d in 0..hd { out[dst+d] = (flat[src+d] as f64 * inv) as f32 * w[d]; }
        }
    }
    out
}

fn transpose_no_norm(flat: &[f32], tokens: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * tokens * hd];
    for t in 0..tokens { for hi in 0..heads { for d in 0..hd {
        out[hi * tokens * hd + t * hd + d] = flat[t * heads * hd + hi * hd + d];
    }}}
    out
}

fn untranspose(data: &[f32], tokens: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * heads * hd];
    for t in 0..tokens { for hi in 0..heads { for d in 0..hd {
        out[t * heads * hd + hi * hd + d] = data[hi * tokens * hd + t * hd + d];
    }}}
    out
}

fn apply_rope(data: &mut [f32], heads: usize, seq: usize, hd: usize, cos: &[f32], sin: &[f32], offset: usize) {
    let half = hd / 2;
    for h in 0..heads { for s in 0..seq {
        let pos = offset + s;
        let base = h * seq * hd + s * hd;
        for i in 0..half {
            let c = cos[pos * half + i]; let si = sin[pos * half + i];
            let x0 = data[base + i]; let x1 = data[base + half + i];
            data[base + i] = x0 * c - x1 * si;
            data[base + half + i] = x1 * c + x0 * si;
        }
    }}
}

fn repeat_kv(kv: &[f32], nkv: usize, n_rep: usize, seq: usize, hd: usize) -> Vec<f32> {
    if n_rep == 1 { return kv.to_vec(); }
    let nh = nkv * n_rep;
    let mut out = vec![0.0f32; nh * seq * hd];
    for kh in 0..nkv { for r in 0..n_rep {
        let dst = kh * n_rep + r;
        out[dst*seq*hd..(dst+1)*seq*hd].copy_from_slice(&kv[kh*seq*hd..(kh+1)*seq*hd]);
    }}
    out
}

fn update_kv(cache: &mut MetalKvCache, k: &[f32], v: &[f32], nkv: usize, new: usize, hd: usize) {
    if cache.len == 0 {
        cache.k = k.to_vec();
        cache.v = v.to_vec();
        cache.len = new;
    } else {
        let old = cache.len;
        let total = old + new;
        let mut fk = vec![0.0f32; nkv * total * hd];
        let mut fv = vec![0.0f32; nkv * total * hd];
        for h in 0..nkv {
            fk[h*total*hd..h*total*hd+old*hd].copy_from_slice(&cache.k[h*old*hd..(h+1)*old*hd]);
            fk[h*total*hd+old*hd..h*total*hd+total*hd].copy_from_slice(&k[h*new*hd..(h+1)*new*hd]);
            fv[h*total*hd..h*total*hd+old*hd].copy_from_slice(&cache.v[h*old*hd..(h+1)*old*hd]);
            fv[h*total*hd+old*hd..h*total*hd+total*hd].copy_from_slice(&v[h*new*hd..(h+1)*new*hd]);
        }
        cache.k = fk;
        cache.v = fv;
        cache.len = total;
    }
}
