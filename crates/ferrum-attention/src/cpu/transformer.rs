//! CPU transformer layer using Accelerate sgemm.

use crate::{TransformerConfig, LayerWeights, AttentionParams};

pub struct CpuKvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

impl CpuKvCache {
    pub fn new() -> Self { Self { k: Vec::new(), v: Vec::new(), len: 0 } }
}

pub fn cpu_layer_forward(
    input: &[f32], tokens: usize, w: &LayerWeights, cfg: &TransformerConfig,
    cos: &[f32], sin: &[f32], kv_cache: &mut CpuKvCache, pos_offset: usize,
) -> Vec<f32> {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;

    // 1. Input LayerNorm
    let ln_out = rms_norm(input, &w.input_ln_w, tokens, h, eps);

    // 2. Q/K/V projections
    let q = matmul_at_bt(&ln_out, &w.q_proj_w, tokens, nh * hd, h);
    let k = matmul_at_bt(&ln_out, &w.k_proj_w, tokens, nkv * hd, h);
    let v = matmul_at_bt(&ln_out, &w.v_proj_w, tokens, nkv * hd, h);

    // 3. Transpose + QK norm + RoPE
    let mut q_r = transpose_and_norm(&q, tokens, nh, hd, &w.q_norm_w, eps);
    let mut k_r = transpose_and_norm(&k, tokens, nkv, hd, &w.k_norm_w, eps);
    let v_r = transpose_no_norm(&v, tokens, nkv, hd);

    apply_rope(&mut q_r, nh, tokens, hd, cos, sin, pos_offset);
    apply_rope(&mut k_r, nkv, tokens, hd, cos, sin, pos_offset);

    // 4. KV cache
    let (full_k, full_v, kv_len) = update_kv(kv_cache, &k_r, &v_r, nkv, tokens, hd);

    // 5. Fused attention
    let n_rep = nh / nkv;
    let k_rep = repeat_kv(&full_k, nkv, n_rep, kv_len, hd);
    let v_rep = repeat_kv(&full_v, nkv, n_rep, kv_len, hd);
    let params = AttentionParams {
        batch: 1, num_heads: nh, num_kv_heads: nkv,
        q_len: tokens, kv_len, head_dim: hd,
        causal: tokens > 1, pos_offset,
    };
    let mut attn_out = vec![0.0f32; nh * tokens * hd];
    super::fused_attention(&q_r, &k_rep, &v_rep, &mut attn_out, &params);

    // 6. Transpose back + O projection
    let attn_flat = untranspose(&attn_out, tokens, nh, hd);
    let o_out = matmul_at_bt(&attn_flat, &w.o_proj_w, tokens, h, nh * hd);

    // 7. Residual add
    let mut hidden = add_vecs(input, &o_out);

    // 8. Post LayerNorm + MLP
    let post_ln = rms_norm(&hidden, &w.post_ln_w, tokens, h, eps);
    let gate = matmul_at_bt(&post_ln, &w.gate_proj_w, tokens, im, h);
    let up = matmul_at_bt(&post_ln, &w.up_proj_w, tokens, im, h);
    let silu_out = silu_mul(&gate, &up);
    let mlp_out = matmul_at_bt(&silu_out, &w.down_proj_w, tokens, h, im);

    // 9. Residual add
    add_vecs(&hidden, &mlp_out)
}

// ── Helpers ─────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32);
}

fn matmul_at_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(101, 111, 112, m as i32, n as i32, k as i32,
            1.0, a.as_ptr(), k as i32, b.as_ptr(), k as i32,
            0.0, c.as_mut_ptr(), n as i32);
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..k { s += a[i*k+p] as f64 * b[j*k+p] as f64; }
            c[i*n+j] = s as f32;
        }
    }
    c
}

fn rms_norm(x: &[f32], w: &[f32], tokens: usize, dim: usize, eps: f64) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * dim];
    for t in 0..tokens {
        let row = &x[t*dim..(t+1)*dim];
        let o = &mut out[t*dim..(t+1)*dim];
        let mut v = 0.0f64;
        for &val in row { let f = val as f64; v += f * f; }
        let inv = 1.0 / (v / dim as f64 + eps).sqrt();
        for i in 0..dim { o[i] = (row[i] as f64 * inv) as f32 * w[i]; }
    }
    out
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

fn update_kv(cache: &mut CpuKvCache, k: &[f32], v: &[f32], nkv: usize, new: usize, hd: usize) -> (Vec<f32>, Vec<f32>, usize) {
    if cache.len == 0 {
        cache.k = k.to_vec(); cache.v = v.to_vec(); cache.len = new;
        (k.to_vec(), v.to_vec(), new)
    } else {
        let old = cache.len; let total = old + new;
        let mut fk = vec![0.0f32; nkv * total * hd];
        let mut fv = vec![0.0f32; nkv * total * hd];
        for h in 0..nkv {
            fk[h*total*hd..h*total*hd+old*hd].copy_from_slice(&cache.k[h*old*hd..(h+1)*old*hd]);
            fk[h*total*hd+old*hd..h*total*hd+total*hd].copy_from_slice(&k[h*new*hd..(h+1)*new*hd]);
            fv[h*total*hd..h*total*hd+old*hd].copy_from_slice(&cache.v[h*old*hd..(h+1)*old*hd]);
            fv[h*total*hd+old*hd..h*total*hd+total*hd].copy_from_slice(&v[h*new*hd..(h+1)*new*hd]);
        }
        cache.k = fk.clone(); cache.v = fv.clone(); cache.len = total;
        (fk, fv, total)
    }
}

fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(g, u)| {
        let s = g / (1.0 + (-g).exp());
        s * u
    }).collect()
}
