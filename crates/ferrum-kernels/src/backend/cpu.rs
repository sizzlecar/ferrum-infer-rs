//! CPU backend using Accelerate (macOS) / portable fallback (Linux).
//! Context = () — all ops execute immediately, no batching needed.

use super::{AttnConfig, Backend};
use ferrum_types::{FerrumError, Result};

pub struct CpuBackend;

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

/// CPU-side GPTQ store — dequantized f32 weights in row-major [n, k] layout.
/// Trades memory for simplicity: repack once at load, then run normal GEMM.
pub struct CpuGptqStore {
    pub weight_f32: Vec<f32>, // [n, k] row-major
    pub k: usize,
    pub n: usize,
}

impl Backend for CpuBackend {
    type Buffer = Vec<f32>;
    type Context = ();
    type GptqStore = CpuGptqStore;

    fn new_context() -> Self::Context {}
    fn sync(_ctx: &mut Self::Context) {}

    fn load_gptq(
        qweight: &[i32],
        scales: &[f32],
        qzeros: &[i32],
        _g_idx: Option<&[i32]>,
        bits: u32,
        group_size: usize,
        k: usize,
        n: usize,
    ) -> Result<Self::GptqStore> {
        if bits != 4 {
            return Err(FerrumError::unsupported(format!(
                "CPU GPTQ: only bits=4 supported (got {bits})"
            )));
        }
        let num_groups = k / group_size;
        // Unpack GPTQ [K/8, N] i32 → int4 values, dequantize per-group:
        //   w_f16 = (q - zero) * scale
        // Write to [n, k] row-major (matches DenseLinear convention).
        let mut w = vec![0.0f32; n * k];
        let packed_rows = k / 8;
        for pr in 0..packed_rows {
            for col in 0..n {
                let packed = qweight[pr * n + col] as u32;
                for bi in 0..8 {
                    let ki = pr * 8 + bi;
                    let q = ((packed >> (bi * 4)) & 0xF) as i32;
                    let grp = ki / group_size;
                    let scale = scales[grp * n + col];
                    // qzeros [num_groups, N/8] i32 packs 8 zero-values per int32
                    let z_packed = qzeros[grp * (n / 8) + (col / 8)] as u32;
                    let zero = (((z_packed >> ((col % 8) * 4)) & 0xF) as i32) + 1;
                    let val = (q - zero) as f32 * scale;
                    w[col * k + ki] = val;
                }
            }
        }
        let _ = num_groups; // informational only
        Ok(CpuGptqStore {
            weight_f32: w,
            k,
            n,
        })
    }

    fn gemm_gptq(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        weight: &Self::GptqStore,
        out: &mut Self::Buffer,
        m: usize,
    ) -> Result<()> {
        // Just run normal GEMM with dequantized weights.
        // out[m, n] = a[m, k] @ w[n, k]^T — same contract as B::gemm.
        Self::gemm(ctx, a, &weight.weight_f32, out, m, weight.n, weight.k);
        Ok(())
    }

    fn gemm(
        _ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert!(
            a.len() >= m * k,
            "gemm: a too small len={} m={m} k={k}",
            a.len()
        );
        assert!(
            b.len() >= n * k,
            "gemm: b too small len={} n={n} k={k}",
            b.len()
        );
        assert!(
            out.len() >= m * n,
            "gemm: out too small len={} m={m} n={n}",
            out.len()
        );
        #[cfg(target_os = "macos")]
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
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
        _ctx: &mut Self::Context,
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
        _ctx: &mut Self::Context,
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
            for i in 0..dim {
                residual[off + i] += x[off + i];
            }
            let row = &residual[off..off + dim];
            let o = &mut out[off..off + dim];
            let sum_sq = dot_product(row, row);
            let inv = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
            for i in 0..dim {
                o[i] = row[i] * inv * w[i];
            }
        }
    }

    fn flash_attention(
        _ctx: &mut Self::Context,
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

    fn copy_slice(
        _ctx: &mut Self::Context,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    ) {
        dst[dst_offset..dst_offset + len].copy_from_slice(&src[src_offset..src_offset + len]);
    }

    fn embedding_lookup(
        _ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * dim;
            out[i * dim..(i + 1) * dim].copy_from_slice(&table[src..src + dim]);
        }
    }

    fn split_qkv(
        _ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let qkv_dim = q_dim + 2 * kv_dim;
        for t in 0..tokens {
            let base = t * qkv_dim;
            q[t * q_dim..(t + 1) * q_dim].copy_from_slice(&qkv[base..base + q_dim]);
            k[t * kv_dim..(t + 1) * kv_dim]
                .copy_from_slice(&qkv[base + q_dim..base + q_dim + kv_dim]);
            v[t * kv_dim..(t + 1) * kv_dim]
                .copy_from_slice(&qkv[base + q_dim + kv_dim..base + qkv_dim]);
        }
    }

    fn fused_silu_mul_split(
        _ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        for t in 0..tokens {
            for i in 0..im {
                let g = gate_up[t * 2 * im + i];
                let u = gate_up[t * 2 * im + im + i];
                out[t * im + i] = (g / (1.0 + (-g).exp())) * u;
            }
        }
    }

    fn qk_norm_rope(
        _ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) {
        let half = head_dim / 2;
        let cos_len = cos.len();
        let sin_len = sin.len();
        debug_assert_eq!(cos_len, sin_len);

        for t in 0..tokens {
            let pos = pos_offset + t;
            for h in 0..heads {
                // input row: [t, h, :]  stride = heads * head_dim
                let src_off = (t * heads + h) * head_dim;
                // output row: [h, t, :]  stride = tokens * head_dim
                let dst_off = (h * tokens + t) * head_dim;

                // Mode 0: plain transpose.
                if mode == 0 {
                    for i in 0..head_dim {
                        output[dst_off + i] = input[src_off + i];
                    }
                    continue;
                }

                // Optional RMS norm (mode 1 only).
                let scale = if mode == 1 {
                    let mut sum_sq = 0.0f32;
                    for i in 0..head_dim {
                        sum_sq += input[src_off + i] * input[src_off + i];
                    }
                    1.0f32 / (sum_sq / head_dim as f32 + eps).sqrt()
                } else {
                    1.0
                };

                // Apply (norm?) + RoPE to halves, write to head-major output.
                for i in 0..half {
                    let (x0_raw, x1_raw) = (input[src_off + i], input[src_off + i + half]);
                    let (x0, x1) = if mode == 1 {
                        (
                            x0_raw * scale * norm_w[i],
                            x1_raw * scale * norm_w[i + half],
                        )
                    } else {
                        (x0_raw, x1_raw)
                    };
                    let c = cos[pos * half + i];
                    let s = sin[pos * half + i];
                    output[dst_off + i] = x0 * c - x1 * s;
                    output[dst_off + i + half] = x1 * c + x0 * s;
                }
            }
        }
    }

    fn kv_cache_append_head_major(
        _ctx: &mut Self::Context,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cache_len: usize,
        cache_capacity: usize,
        new_k_head_major: &Self::Buffer,
        new_v_head_major: &Self::Buffer,
        new_tokens: usize,
        nkv: usize,
        hd: usize,
    ) {
        debug_assert!(cache_len + new_tokens <= cache_capacity);
        debug_assert_eq!(cache_k.len(), nkv * cache_capacity * hd);
        debug_assert_eq!(cache_v.len(), nkv * cache_capacity * hd);
        debug_assert_eq!(new_k_head_major.len(), nkv * new_tokens * hd);
        debug_assert_eq!(new_v_head_major.len(), nkv * new_tokens * hd);

        for h in 0..nkv {
            let dst_base = h * cache_capacity * hd + cache_len * hd;
            let src_base = h * new_tokens * hd;
            cache_k[dst_base..dst_base + new_tokens * hd]
                .copy_from_slice(&new_k_head_major[src_base..src_base + new_tokens * hd]);
            cache_v[dst_base..dst_base + new_tokens * hd]
                .copy_from_slice(&new_v_head_major[src_base..src_base + new_tokens * hd]);
        }
    }

    fn transpose_head_to_token(
        _ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        for h in 0..heads {
            for t in 0..tokens {
                let s = (h * tokens + t) * dim;
                let d = (t * heads + h) * dim;
                dst[d..d + dim].copy_from_slice(&src[s..s + dim]);
            }
        }
    }

    fn add_inplace(
        _ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        len: usize,
    ) {
        for i in 0..len {
            residual[i] += x[i];
        }
    }

    fn add_bias(
        _ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) {
        debug_assert_eq!(bias.len(), cols);
        for r in 0..rows {
            let off = r * cols;
            for c in 0..cols {
                data[off + c] += bias[c];
            }
        }
    }

    fn layer_norm(
        _ctx: &mut Self::Context,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        debug_assert_eq!(gamma.len(), dim);
        debug_assert_eq!(beta.len(), dim);
        for t in 0..tokens {
            let off = t * dim;
            // Compute mean + variance over `dim` in f64 for stability.
            let mut mean = 0.0f64;
            for i in 0..dim {
                mean += x[off + i] as f64;
            }
            mean /= dim as f64;
            let mut var = 0.0f64;
            for i in 0..dim {
                let d = x[off + i] as f64 - mean;
                var += d * d;
            }
            var /= dim as f64;
            let inv = 1.0f32 / ((var as f32) + eps).sqrt();
            let mean_f32 = mean as f32;
            for i in 0..dim {
                out[off + i] = (x[off + i] - mean_f32) * inv * gamma[i] + beta[i];
            }
        }
    }

    fn gelu(_ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
        // Uses f64 for erf accuracy (matches torch.nn.functional.gelu default).
        for i in 0..len {
            let xi = x[i];
            out[i] = 0.5 * xi * (1.0 + libm_erf(xi / std::f32::consts::SQRT_2));
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

// ── Helpers ──────────────────────────────────────────────────────────────

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
        a.iter().zip(b).map(|(x, y)| x * y).sum()
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
    // Per-head KV stride: 0 (the default) means contiguous (legacy
    // `kv_cache_append` path reallocates each layer). A non-zero value means
    // the cache is pre-allocated to `kv_seq_stride` rows per head but only
    // the first `kv_len` are valid — we skip the rest via `attend_len`.
    let kv_stride = if cfg.kv_seq_stride > 0 {
        cfg.kv_seq_stride
    } else {
        kv_len
    };

    for b in 0..batch {
        for h in 0..nh {
            let kv_h = h / n_rep;
            let q_off = (b * nh + h) * q_len * d;
            let k_off = (b * nkv + kv_h) * kv_stride * d;
            let v_off = (b * nkv + kv_h) * kv_stride * d;
            let o_off = (b * nh + h) * q_len * d;

            for qi in 0..q_len {
                let attend_end = if causal {
                    (pos_offset + qi + 1).min(kv_len)
                } else {
                    kv_len
                };
                let attend_start = if causal && cfg.sliding_window > 0 {
                    attend_end.saturating_sub(cfg.sliding_window)
                } else {
                    0
                };
                let mut max_score = f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; d];

                for ki in attend_start..attend_end {
                    let mut dot = 0.0f32;
                    for di in 0..d {
                        dot += q[q_off + qi * d + di] * k[k_off + ki * d + di];
                    }
                    let score = dot * scale;
                    if score > max_score {
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

/// Minimal error-function approximation (Abramowitz & Stegun 7.1.26),
/// max error ~1.5e-7 which is comfortably below f32 round-off noise.
fn libm_erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
