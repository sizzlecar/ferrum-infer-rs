//! CPU backend using Accelerate (macOS) / portable fallback (Linux).

use super::{AttnConfig, Backend};

pub struct CpuBackend;

// ── BLAS bindings (macOS Accelerate) ──────────────────────────────────

#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn vDSP_dotpr(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        result: *mut f32,
        n: u64,
    );
}

impl Backend for CpuBackend {
    type Buffer = Vec<f32>;

    fn gemm(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        // C[m,n] = A[m,k] @ B[n,k]^T (B stored as [n, k])
        debug_assert!(a.len() >= m * k, "gemm: a too small");
        debug_assert!(b.len() >= n * k, "gemm: b too small");
        debug_assert!(out.len() >= m * n, "gemm: out too small");

        #[cfg(target_os = "macos")]
        unsafe {
            cblas_sgemm(
                101,
                111,
                112, // RowMajor, NoTrans, Trans
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                out.as_mut_ptr(),
                n as i32,
            );
        }

        #[cfg(not(target_os = "macos"))]
        {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for p in 0..k {
                        sum += a[i * k + p] as f64 * b[j * k + p] as f64;
                    }
                    out[i * n + j] = sum as f32;
                }
            }
        }
    }

    fn rms_norm(
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        for t in 0..tokens {
            let row = &x[t * dim..(t + 1) * dim];
            let o = &mut out[t * dim..(t + 1) * dim];
            let sum_sq = dot_product(row, row);
            let inv = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
            for i in 0..dim {
                o[i] = row[i] * inv * w[i];
            }
        }
    }

    fn fused_add_rms_norm(
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        for t in 0..tokens {
            let off = t * dim;
            // residual += x
            for i in 0..dim {
                residual[off + i] += x[off + i];
            }
            // rms norm on updated residual
            let row = &residual[off..off + dim];
            let o = &mut out[off..off + dim];
            let sum_sq = dot_product(row, row);
            let inv = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
            for i in 0..dim {
                o[i] = row[i] * inv * w[i];
            }
        }
    }

    fn rope(
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        positions: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let half = head_dim / 2;
        let tokens = positions.len();
        apply_rope_impl(q, tokens, num_heads, head_dim, half, cos, sin, positions);
        apply_rope_impl(k, tokens, num_kv_heads, head_dim, half, cos, sin, positions);
    }

    fn decode_attention(
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        out: &mut Self::Buffer,
        kv_len: usize,
        cfg: &AttnConfig,
    ) {
        // Single-token attention: q [num_heads, head_dim] × k_cache [num_kv_heads, kv_len, head_dim]
        cpu_attention(q, k_cache, v_cache, out, 1, 1, kv_len, false, 0, cfg);
    }

    fn flash_attention(
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        pos_offset: usize,
        cfg: &AttnConfig,
    ) {
        cpu_attention(
            q, k, v, out, batch, q_len, kv_len, cfg.causal, pos_offset, cfg,
        );
    }

    fn silu_mul(gate: &Self::Buffer, up: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        for i in 0..len {
            let g = gate[i];
            let s = g / (1.0 + (-g).exp());
            out[i] = s * up[i];
        }
    }

    fn add(a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        for i in 0..len {
            out[i] = a[i] + b[i];
        }
    }

    fn copy(src: &Self::Buffer, dst: &mut Self::Buffer, len: usize) {
        dst[..len].copy_from_slice(&src[..len]);
    }

    fn embedding_lookup(table: &Self::Buffer, ids: &[u32], out: &mut Self::Buffer, dim: usize) {
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * dim;
            out[i * dim..(i + 1) * dim].copy_from_slice(&table[src..src + dim]);
        }
    }

    fn alloc(len: usize) -> Self::Buffer {
        vec![0.0f32; len]
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        buf[..len].to_vec()
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        data.to_vec()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_os = "macos")]
    {
        let mut result = 0.0f32;
        unsafe {
            vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, a.len() as u64);
        }
        result
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
}

fn apply_rope_impl(
    data: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    half: usize,
    cos: &[f32],
    sin: &[f32],
    positions: &[u32],
) {
    // data layout: [tokens, heads, head_dim]
    for t in 0..tokens {
        let pos = positions[t] as usize;
        for h in 0..heads {
            let base = t * heads * head_dim + h * head_dim;
            for i in 0..half {
                let c = cos[pos * half + i];
                let s = sin[pos * half + i];
                let x0 = data[base + i];
                let x1 = data[base + half + i];
                data[base + i] = x0 * c - x1 * s;
                data[base + half + i] = x1 * c + x0 * s;
            }
        }
    }
}

/// CPU attention supporting both prefill (q_len > 1) and decode (q_len == 1).
/// Handles GQA (num_heads != num_kv_heads).
///
/// Layout:
///   Q: [batch, num_heads, q_len, head_dim]
///   K: [batch, num_kv_heads, kv_len, head_dim]
///   V: [batch, num_kv_heads, kv_len, head_dim]
///   out: [batch, num_heads, q_len, head_dim]
fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    batch: usize,
    q_len: usize,
    kv_len: usize,
    causal: bool,
    pos_offset: usize,
    cfg: &AttnConfig,
) {
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let n_rep = nh / nkv;
    let scale = cfg.scale;

    for b in 0..batch {
        for h in 0..nh {
            let kv_h = h / n_rep;

            let q_off = (b * nh + h) * q_len * d;
            let k_off = (b * nkv + kv_h) * kv_len * d;
            let v_off = (b * nkv + kv_h) * kv_len * d;
            let o_off = (b * nh + h) * q_len * d;

            for qi in 0..q_len {
                // Compute scores: Q[qi] · K[ki] for all ki
                let attend_len = if causal {
                    (pos_offset + qi + 1).min(kv_len)
                } else {
                    kv_len
                };

                // Online softmax + weighted V accumulation
                let mut max_score = f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; d];

                for ki in 0..attend_len {
                    // dot(q[qi], k[ki])
                    let mut dot = 0.0f32;
                    for di in 0..d {
                        dot += q[q_off + qi * d + di] * k[k_off + ki * d + di];
                    }
                    let score = dot * scale;

                    if score > max_score {
                        // Rescale existing accumulator
                        let correction = (max_score - score).exp();
                        for di in 0..d {
                            acc[di] *= correction;
                        }
                        sum_exp *= correction;
                        max_score = score;
                    }
                    let w = (score - max_score).exp();
                    sum_exp += w;
                    for di in 0..d {
                        acc[di] += w * v[v_off + ki * d + di];
                    }
                }

                // Normalize
                if sum_exp > 0.0 {
                    let inv = 1.0 / sum_exp;
                    for di in 0..d {
                        out[o_off + qi * d + di] = acc[di] * inv;
                    }
                }
            }
        }
    }
}
