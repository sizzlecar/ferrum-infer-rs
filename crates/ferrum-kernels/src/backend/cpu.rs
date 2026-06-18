//! CPU backend using Accelerate (macOS) / portable fallback (Linux).
//! Context = () — all ops execute immediately, no batching needed.

use half::f16;

use super::{AttnConfig, Backend};
use ferrum_types::{FerrumError, Result};

// ── Q4_K_M block layout ────────────────────────────────────────────────
//
// Mirrors GGML / candle's `BlockQ4K`. Used by `load_q4_k` to dequant raw
// GGUF block bytes to fp32 row-major weights on CPU.

const Q4_K_QK: usize = 256;
const Q4_K_SCALE_SIZE: usize = 12;
const Q4_K_BLOCK_BYTES: usize = 4 + Q4_K_SCALE_SIZE + Q4_K_QK / 2; // 144

/// Bit-unpacker matching candle's `quantized::utils::get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Port of candle's CPU `BlockQ4K::to_float`. Bit-identical output for
/// identical input — the test in `q4_k.rs` verifies our Metal kernel
/// also matches.
fn dequant_q4_k_cpu(bytes: &[u8], n_blocks: usize) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), n_blocks * Q4_K_BLOCK_BYTES);
    let mut out = Vec::with_capacity(n_blocks * Q4_K_QK);
    for b in 0..n_blocks {
        let off = b * Q4_K_BLOCK_BYTES;
        let d = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
        let scales = &bytes[off + 4..off + 4 + Q4_K_SCALE_SIZE];
        let qs = &bytes[off + 4 + Q4_K_SCALE_SIZE..off + Q4_K_BLOCK_BYTES];

        let mut is = 0usize;
        for j in (0..Q4_K_QK).step_by(64) {
            let q_chunk = &qs[j / 2..j / 2 + 32];
            let (sc1, mn1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * mn1 as f32;
            let (sc2, mn2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * mn2 as f32;
            for q in q_chunk {
                out.push(d1 * (q & 0xF) as f32 - m1);
            }
            for q in q_chunk {
                out.push(d2 * (q >> 4) as f32 - m2);
            }
            is += 2;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn validate_gated_delta_rule_shape(
    query_len: usize,
    key_len: usize,
    value_len_actual: usize,
    g_len: usize,
    beta_len: usize,
    initial_state_len: usize,
    out_len: usize,
    final_state_len: usize,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<()> {
    if tokens == 0 || key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule shape must be positive, got tokens={tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim}"
        )));
    }
    if value_heads % key_heads != 0 {
        return Err(FerrumError::model(format!(
            "gated_delta_rule value_heads {value_heads} must be divisible by key_heads {key_heads}"
        )));
    }

    for (label, actual, expected) in [
        ("query", query_len, tokens * key_heads * key_dim),
        ("key", key_len, tokens * key_heads * key_dim),
        ("value", value_len_actual, tokens * value_heads * value_dim),
        ("g", g_len, tokens * value_heads),
        ("beta", beta_len, tokens * value_heads),
        (
            "initial_state",
            initial_state_len,
            value_heads * value_dim * key_dim,
        ),
        ("out", out_len, tokens * value_heads * value_dim),
        (
            "final_state",
            final_state_len,
            value_heads * value_dim * key_dim,
        ),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_delta_rule {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_linear_attention_prepare_shape(
    mixed_qkv_raw_len: usize,
    conv_weight_len: usize,
    a_raw_len: usize,
    b_raw_len: usize,
    a_log_len: usize,
    dt_bias_len: usize,
    query_len: usize,
    key_len: usize,
    value_len_actual: usize,
    g_len: usize,
    beta_len: usize,
    tokens: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> Result<()> {
    if tokens == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || conv_kernel == 0
    {
        return Err(FerrumError::model(format!(
            "linear_attention_prepare shape must be positive, got tokens={tokens} key_heads={key_heads} value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel}"
        )));
    }

    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    for (label, actual, expected) in [
        ("mixed_qkv_raw", mixed_qkv_raw_len, tokens * conv_channels),
        ("conv_weight", conv_weight_len, conv_channels * conv_kernel),
        ("a_raw", a_raw_len, tokens * value_heads),
        ("b_raw", b_raw_len, tokens * value_heads),
        ("a_log", a_log_len, value_heads),
        ("dt_bias", dt_bias_len, value_heads),
        ("query", query_len, tokens * qk_total),
        ("key", key_len, tokens * qk_total),
        ("value", value_len_actual, tokens * value_total),
        ("g", g_len, tokens * value_heads),
        ("beta", beta_len, tokens * value_heads),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "linear_attention_prepare {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_gated_rms_norm_shape(
    core_len: usize,
    z_len: usize,
    weight_len: usize,
    out_len: usize,
    tokens: usize,
    heads: usize,
    dim: usize,
) -> Result<()> {
    if tokens == 0 || heads == 0 || dim == 0 {
        return Err(FerrumError::model(format!(
            "gated_rms_norm shape must be positive, got tokens={tokens} heads={heads} dim={dim}"
        )));
    }
    let expected = tokens * heads * dim;
    for (label, actual, expected) in [
        ("core", core_len, expected),
        ("z", z_len, expected),
        ("weight", weight_len, dim),
        ("out", out_len, expected),
    ] {
        if actual < expected {
            return Err(FerrumError::model(format!(
                "gated_rms_norm {label} length {actual} < expected {expected}"
            )));
        }
    }
    Ok(())
}

pub struct CpuBackend;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    unsafe fn cblas_sgemm(
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

/// CPU-side container for any GGUF k-quant flavour. Each variant holds
/// the dense fp32 weights post-eager-dequant — CPU isn't the bench
/// target so we don't pay the complexity of on-the-fly dequant here;
/// the variant tag exists so `gemm_quant` can route consistently.
///
/// New k-quant types (Q5_K / Q6_K / Q8_0) become new variants — no
/// trait churn, just a new arm in `load_quant` and `gemm_quant`.
pub enum CpuQuantStore {
    Q4K {
        weights: Vec<f32>, // [n_rows, n_cols] row-major
        n_rows: usize,
        n_cols: usize,
    },
}

impl Backend for CpuBackend {
    type Buffer = Vec<f32>;
    type Context = ();
    // type GptqStore: removed in Phase C step 4e. CpuGptqStore is now
    // a private (crate-internal) detail of CpuMarlinExpertStack.

    type Timer = crate::backend::timer::CpuTimer;
    fn make_timer() -> Self::Timer {
        crate::backend::timer::CpuTimer::new()
    }

    fn new_context() -> Self::Context {}
    fn sync(_ctx: &mut Self::Context) {}
    fn activation_elem_size_bytes() -> usize {
        std::mem::size_of::<f32>()
    }

    /// Phase D step 2+3: typed alloc. CPU Buffer is Vec<f32> — bytes
    /// are dtype-erased, so we size the underlying Vec to hold `n`
    /// elements of `dtype` (bit-cast at read/write time).
    fn alloc_typed(dtype: crate::backend::Dtype, n: usize) -> Self::Buffer {
        // f32 storage; for i8 we round up to 4-byte boundary so the
        // Vec<f32> length covers all i8 elements.
        let bytes = n * dtype.bytes_per_elem();
        let f32_len = bytes.div_ceil(4);
        vec![0.0f32; f32_len]
    }

    /// Phase D step 2+3: typed upload. Bit-cast host data into f32
    /// words (CPU buffer is dtype-erased Vec<f32>, see alloc_typed).
    fn from_slice_typed<T: crate::backend::HostDtype>(data: &[T]) -> Self::Buffer {
        let bytes = data.len() * std::mem::size_of::<T>();
        let f32_len = bytes.div_ceil(4);
        let mut out = vec![0.0f32; f32_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                out.as_mut_ptr() as *mut u8,
                bytes,
            );
        }
        out
    }

    /// Phase D step 2+3: typed in-place write. Bit-cast bytes into
    /// the dtype-erased f32 storage.
    fn write_typed<T: crate::backend::HostDtype>(
        _ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        data: &[T],
    ) {
        let bytes = data.len() * std::mem::size_of::<T>();
        debug_assert!(
            bytes <= dst.len() * 4,
            "CpuBackend::write_typed: src bytes {} > dst bytes {}",
            bytes,
            dst.len() * 4
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                dst.as_mut_ptr() as *mut u8,
                bytes,
            );
        }
    }

    fn fused_silu_mul_split_strided(
        _ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        in_row_offset: usize,
        out: &mut Self::Buffer,
        out_row_offset: usize,
        tokens: usize,
        intermediate: usize,
    ) {
        let in_per_row = 2 * intermediate;
        let in_start = in_row_offset * in_per_row;
        let out_start = out_row_offset * intermediate;
        for r in 0..tokens {
            for c in 0..intermediate {
                let g = gate_up[in_start + r * in_per_row + c];
                let u = gate_up[in_start + r * in_per_row + intermediate + c];
                let silu = g / (1.0 + (-g).exp());
                out[out_start + r * intermediate + c] = silu * u;
            }
        }
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

    #[allow(clippy::too_many_arguments)]
    fn recurrent_gated_delta_rule_f32(
        _ctx: &mut Self::Context,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        g: &Self::Buffer,
        beta: &Self::Buffer,
        initial_state: &Self::Buffer,
        out: &mut Self::Buffer,
        final_state: &mut Self::Buffer,
        tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> Result<()> {
        validate_gated_delta_rule_shape(
            query.len(),
            key.len(),
            value.len(),
            g.len(),
            beta.len(),
            initial_state.len(),
            out.len(),
            final_state.len(),
            tokens,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        )?;

        let repeat_factor = value_heads / key_heads;
        let state_len = value_heads * value_dim * key_dim;
        final_state[..state_len].copy_from_slice(&initial_state[..state_len]);

        for token in 0..tokens {
            for value_head in 0..value_heads {
                let key_head = value_head / repeat_factor;
                let mut q_inv = 1.0;
                let mut k_inv = 1.0;
                if use_qk_l2norm {
                    let mut q_norm = 0.0;
                    let mut k_norm = 0.0;
                    for kd in 0..key_dim {
                        let qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
                        q_norm += query[qk_idx] * query[qk_idx];
                        k_norm += key[qk_idx] * key[qk_idx];
                    }
                    q_inv = (q_norm + 1e-6).sqrt().recip();
                    k_inv = (k_norm + 1e-6).sqrt().recip();
                }
                let gate_idx = token * value_heads + value_head;
                let decay = g[gate_idx].exp();
                let beta_t = beta[gate_idx];
                for vd in 0..value_dim {
                    let state_base = (value_head * value_dim + vd) * key_dim;
                    let mut kv_mem = 0.0;
                    for kd in 0..key_dim {
                        let qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
                        let state_idx = state_base + kd;
                        final_state[state_idx] *= decay;
                        kv_mem += final_state[state_idx] * (key[qk_idx] * k_inv);
                    }
                    let value_idx = ((token * value_heads + value_head) * value_dim) + vd;
                    let delta = (value[value_idx] - kv_mem) * beta_t;
                    for kd in 0..key_dim {
                        let qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
                        final_state[state_base + kd] += delta * (key[qk_idx] * k_inv);
                    }
                    let mut acc = 0.0;
                    for kd in 0..key_dim {
                        let qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
                        acc += final_state[state_base + kd] * (query[qk_idx] * q_inv * scale);
                    }
                    out[value_idx] = acc;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_prepare_f32(
        _ctx: &mut Self::Context,
        mixed_qkv_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        a_raw: &Self::Buffer,
        b_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        validate_linear_attention_prepare_shape(
            mixed_qkv_raw.len(),
            conv_weight.len(),
            a_raw.len(),
            b_raw.len(),
            a_log.len(),
            dt_bias.len(),
            query.len(),
            key.len(),
            value.len(),
            g.len(),
            beta.len(),
            tokens,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
        )?;

        let qk_total = key_heads * key_dim;
        let value_total = value_heads * value_dim;
        let conv_channels = 2 * qk_total + value_total;
        let pad = conv_kernel - 1;
        for token in 0..tokens {
            for channel in 0..conv_channels {
                let mut acc = 0.0;
                for kernel_idx in 0..conv_kernel {
                    let padded = token + kernel_idx;
                    if padded >= pad {
                        let src_token = padded - pad;
                        if src_token < tokens {
                            acc += mixed_qkv_raw[src_token * conv_channels + channel]
                                * conv_weight[channel * conv_kernel + kernel_idx];
                        }
                    }
                }
                let conv = silu(acc);
                if channel < qk_total {
                    query[token * qk_total + channel] = conv;
                } else if channel < 2 * qk_total {
                    key[token * qk_total + (channel - qk_total)] = conv;
                } else {
                    value[token * value_total + (channel - 2 * qk_total)] = conv;
                }
            }

            for value_head in 0..value_heads {
                let gate_idx = token * value_heads + value_head;
                g[gate_idx] =
                    -a_log[value_head].exp() * softplus(a_raw[gate_idx] + dt_bias[value_head]);
                beta[gate_idx] = sigmoid(b_raw[gate_idx]);
            }
        }

        if apply_qk_l2norm {
            for row in 0..tokens * key_heads {
                let base = row * key_dim;
                let mut q_sum = 0.0;
                let mut k_sum = 0.0;
                for d in 0..key_dim {
                    q_sum += query[base + d] * query[base + d];
                    k_sum += key[base + d] * key[base + d];
                }
                let q_inv = (q_sum + 1e-6).sqrt().recip();
                let k_inv = (k_sum + 1e-6).sqrt().recip();
                for d in 0..key_dim {
                    query[base + d] *= q_inv;
                    key[base + d] *= k_inv;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_rms_norm_f32(
        _ctx: &mut Self::Context,
        core: &Self::Buffer,
        z: &Self::Buffer,
        weight: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        validate_gated_rms_norm_shape(
            core.len(),
            z.len(),
            weight.len(),
            out.len(),
            tokens,
            heads,
            dim,
        )?;

        for row in 0..tokens * heads {
            let base = row * dim;
            let mut sum_sq = 0.0;
            for d in 0..dim {
                let x = core[base + d];
                sum_sq += x * x;
            }
            let inv = (sum_sq / dim as f32 + eps).sqrt().recip();
            for d in 0..dim {
                out[base + d] = core[base + d] * inv * weight[d] * silu(z[base + d]);
            }
        }
        Ok(())
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

    fn fused_gelu_tanh_mul_split(
        _ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_56;
        for t in 0..tokens {
            for i in 0..im {
                let g = gate_up[t * 2 * im + i];
                let u = gate_up[t * 2 * im + im + i];
                let inner = SQRT_2_OVER_PI * (g + 0.044715 * g * g * g);
                out[t * im + i] = 0.5 * g * (1.0 + inner.tanh()) * u;
            }
        }
    }

    fn scale_inplace(_ctx: &mut Self::Context, buf: &mut Self::Buffer, scale: f32, len: usize) {
        for x in buf[..len].iter_mut() {
            *x *= scale;
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

                if mode == 3 {
                    // GGUF LLaMA / llama.cpp interleaved RoPE layout.
                    for i in 0..half {
                        let j = 2 * i;
                        let x0 = input[src_off + j];
                        let x1 = input[src_off + j + 1];
                        let c = cos[pos * half + i];
                        let s = sin[pos * half + i];
                        output[dst_off + j] = x0 * c - x1 * s;
                        output[dst_off + j + 1] = x1 * c + x0 * s;
                    }
                } else {
                    // Apply (norm?) + half-split RoPE to head-major output.
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
        // The source buffers may be sized for `max_tokens` (the prefill-
        // sized scratch) while only the first `nkv * new_tokens * hd`
        // entries are valid for this call. Allow >= so reusing scratch
        // across prefill and decode doesn't trip the assert.
        debug_assert!(new_k_head_major.len() >= nkv * new_tokens * hd);
        debug_assert!(new_v_head_major.len() >= nkv * new_tokens * hd);

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

    fn scaled_add_inplace(
        _ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        scale: f32,
        len: usize,
    ) {
        for i in 0..len {
            dst[i] += scale * src[i];
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

#[allow(dead_code)]
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
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

// CPU has no graph-capture analogue; inherit BackendGraph defaults.
impl crate::backend::BackendGraph for CpuBackend {}

// CPU has no multi-rank collectives; inherit BackendCollective defaults.
impl crate::backend::BackendCollective for CpuBackend {}

/// Dequant raw GPTQ tensors → row-major `[n, k]` f32. Shared between
/// the per-tensor `load_gptq` and the MoE `load_gptq_stacked` impls.
fn cpu_dequant_gptq(
    qweight: &[i32],
    scales: &[f32],
    qzeros: &[i32],
    bits: u32,
    group_size: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if bits != 4 {
        return Err(FerrumError::unsupported(format!(
            "CPU GPTQ: only bits=4 supported (got {bits})"
        )));
    }
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
                let z_packed = qzeros[grp * (n / 8) + (col / 8)] as u32;
                let zero = (((z_packed >> ((col % 8) * 4)) & 0xF) as i32) + 1;
                let val = (q - zero) as f32 * scale;
                w[col * k + ki] = val;
            }
        }
    }
    Ok(w)
}

impl crate::backend::BackendQuantMarlin for CpuBackend {
    fn load_gptq(
        qweight: &[i32],
        scales: &[f32],
        qzeros: &[i32],
        _g_idx: Option<&[i32]>,
        bias_host: Option<&[f32]>,
        bits: u32,
        group_size: usize,
        k: usize,
        n: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        let w = cpu_dequant_gptq(qweight, scales, qzeros, bits, group_size, k, n)?;
        // Phase 3e/2: dequantized weights become a CpuGptqLinear that
        // owns the (out_features, in_features) f32 matrix and runs
        // through the existing Self::gemm CPU path.
        Ok(Box::new(crate::quant_linear::cpu_dequant::CpuGptqLinear {
            weight_f32: w,
            bias: bias_host.map(|b| b.to_vec()),
            in_features: k,
            out_features: n,
        }))
    }
    fn load_gptq_stacked(
        qweights: &[&[i32]],
        scales: &[&[f32]],
        qzeros: &[&[i32]],
        _g_idx: Option<&[i32]>,
        bits: u32,
        group_size: usize,
        k: usize,
        n_per_expert: usize,
    ) -> Result<std::sync::Arc<dyn crate::MarlinExpertStack<Self>>> {
        // Phase 3e/2 addition: dequant each expert independently, concat
        // along N (rows in [n, k] layout). Used by MoE parity tests.
        let num_experts = qweights.len();
        if scales.len() != num_experts || qzeros.len() != num_experts {
            return Err(FerrumError::model(format!(
                "load_gptq_stacked: input slice lengths disagree (qw {num_experts}, sc {}, qz {})",
                scales.len(),
                qzeros.len()
            )));
        }
        let total_n = num_experts * n_per_expert;
        let mut all_w = Vec::with_capacity(total_n * k);
        for ((qw_e, sc_e), qz_e) in qweights.iter().zip(scales.iter()).zip(qzeros.iter()) {
            let w_e = cpu_dequant_gptq(qw_e, sc_e, qz_e, bits, group_size, k, n_per_expert)?;
            all_w.extend_from_slice(&w_e);
        }
        let store = std::sync::Arc::new(CpuGptqStore {
            weight_f32: all_w,
            k,
            n: total_n,
        });
        Ok(std::sync::Arc::new(
            crate::quant_linear::cpu_marlin_stack::CpuMarlinExpertStack::new(
                store,
                num_experts,
                n_per_expert,
                k,
            ),
        ))
    }
    // Phase C step 4b: make_stacked_expert_linear inlined into
    // CpuMarlinExpertStack::make_expert_linear.
    // Phase C step 4e: make_marlin_expert_stack subsumed by load_gptq_stacked.
    // gemm_gptq_with_offset_strided body moved to free function
    // cpu_gemm_gptq_with_offset_strided below — called by
    // CpuMarlinExpertStack::gemm_phase_batched.
}

/// Free-function form of the deleted
/// `BackendQuantMarlin::gemm_gptq_with_offset_strided` (Phase C step 4e).
/// Single caller: `CpuMarlinExpertStack::gemm_phase_batched`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_gemm_gptq_with_offset_strided(
    _ctx: &mut <CpuBackend as Backend>::Context,
    input: &<CpuBackend as Backend>::Buffer,
    in_row_offset: usize,
    weight: &CpuGptqStore,
    expert_offset: usize,
    expert_n: usize,
    output: &mut <CpuBackend as Backend>::Buffer,
    out_row_offset: usize,
    m: usize,
    k: usize,
) -> Result<()> {
    if expert_offset + expert_n > weight.n {
        return Err(FerrumError::model(format!(
            "cpu_gemm_gptq_with_offset_strided OOB: offset {expert_offset} + n {expert_n} > stacked_n {}",
            weight.n
        )));
    }
    if k != weight.k {
        return Err(FerrumError::model(format!(
            "cpu_gemm_gptq_with_offset_strided k mismatch: arg {k} vs weight.k {}",
            weight.k
        )));
    }
    let in_start = in_row_offset * k;
    let in_end = (in_row_offset + m) * k;
    let out_start = out_row_offset * expert_n;
    let out_end = (out_row_offset + m) * expert_n;
    let row_start = expert_offset * k;
    let row_end = (expert_offset + expert_n) * k;
    let weight_slice = weight.weight_f32[row_start..row_end].to_vec();
    let in_slice = input[in_start..in_end].to_vec();
    let mut out_slice = vec![0.0f32; m * expert_n];
    let mut ctx_local = ();
    CpuBackend::gemm(
        &mut ctx_local,
        &in_slice,
        &weight_slice,
        &mut out_slice,
        m,
        expert_n,
        k,
    );
    output[out_start..out_end].copy_from_slice(&out_slice);
    Ok(())
}

impl crate::backend::BackendQuantGguf for CpuBackend {
    fn load_quant(
        kind: super::GgufQuantType,
        bytes: &[u8],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        use super::GgufQuantType;
        let store = match kind {
            GgufQuantType::Q4K => {
                let total_elems = n_rows * n_cols;
                if total_elems % Q4_K_QK != 0 {
                    return Err(FerrumError::model(format!(
                        "load_quant Q4K: elements {total_elems} not a multiple of {Q4_K_QK}"
                    )));
                }
                let n_blocks = total_elems / Q4_K_QK;
                let expected = n_blocks * Q4_K_BLOCK_BYTES;
                if bytes.len() != expected {
                    return Err(FerrumError::model(format!(
                        "load_quant Q4K: bytes {} != expected {} ({n_blocks} × {Q4_K_BLOCK_BYTES})",
                        bytes.len(),
                        expected
                    )));
                }
                CpuQuantStore::Q4K {
                    weights: dequant_q4_k_cpu(bytes, n_blocks),
                    n_rows,
                    n_cols,
                }
            }
            other => {
                return Err(FerrumError::unsupported(format!(
                    "CPU load_quant: {other:?} not yet implemented"
                )));
            }
        };
        // Phase 3e/3: dispatch via CpuGgufLinear::forward instead of
        // a trait method.
        Ok(Box::new(crate::quant_linear::cpu_gguf::CpuGgufLinear {
            store,
            in_features: n_cols,
            out_features: n_rows,
        }))
    }
}

// CPU has no paged-KV path; inherit unsupported defaults.
impl crate::backend::BackendPagedKv for CpuBackend {}

// CPU has no native MoE dispatch; inherit unsupported defaults.
impl crate::backend::BackendMoeFused for CpuBackend {}

// CPU: existing KV cache path treats fp16 buffer as f32 internally; mark as KvFp16 for compatibility.
impl crate::backend::BackendKvDtype<crate::backend::KvFp16> for CpuBackend {
    type KvBuffer = <Self as crate::backend::Backend>::Buffer;
    type KvScales = ();
}
