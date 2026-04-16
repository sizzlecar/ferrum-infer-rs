//! Generic transformer layer forward — one function, all backends.
//!
//! This is the core compute loop shared by prefill and decode.
//! The only branch is attention dispatch (decode_attention vs flash_attention).

use super::{AttnConfig, Backend, KvCache, LayerScratch, LayerWeights, TransformerConfig};

/// Forward one transformer layer.
///
/// **Data layout:**
///   - `residual`: `[tokens, hidden_size]` — updated in-place (residual connections)
///   - `LayerWeights`: fused QKV `[q_dim + 2*kv_dim, hidden]`, fused gate_up `[2*inter, hidden]`
///   - KV cache: `[num_kv_heads, kv_len, head_dim]`
///
/// **Flow:**
///   1. RMS Norm → 2. QKV GEMM → 3. (optional QK norm) → 4. RoPE →
///   5. KV cache append → 6. Attention → 7. O-proj GEMM → 8. Residual + Norm →
///   9. Gate/Up GEMM → 10. SiLU*mul → 11. Down GEMM → 12. Residual add
pub fn layer_forward<B: Backend>(
    cfg: &TransformerConfig,
    weights: &LayerWeights<B>,
    kv: &mut KvCache<B>,
    scratch: &mut LayerScratch<B>,
    residual: &mut B::Buffer,
    positions: &[u32],
    cos: &B::Buffer,
    sin: &B::Buffer,
    tokens: usize,
) {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let qkv_dim = q_dim + 2 * kv_dim;

    // 1. Input RMS Norm
    B::rms_norm(residual, &weights.input_ln_w, eps, &mut scratch.norm_out, tokens, h);

    // 2. Fused QKV projection: [tokens, hidden] @ [q_dim+2*kv_dim, hidden]^T
    B::gemm(
        &scratch.norm_out,
        &weights.qkv_proj_w,
        &mut scratch.qkv_out,
        tokens,
        qkv_dim,
        h,
    );

    // 3. Split Q, K, V and optionally apply QK norm
    //    QKV layout: [tokens, q_dim + kv_dim + kv_dim]
    //    After split, we need to work with:
    //      Q: [tokens, num_heads, head_dim] (= [tokens, q_dim])
    //      K: [tokens, num_kv_heads, head_dim] (= [tokens, kv_dim])
    //      V: [tokens, num_kv_heads, head_dim] (= [tokens, kv_dim])
    //
    //    For QK norm + RoPE + KV cache append, we reuse scratch buffers.
    //    The split is implicit via offset arithmetic in the backend.

    // 4. Split QKV into dedicated buffers, then apply RoPE
    split_qkv_all::<B>(
        &scratch.qkv_out,
        &mut scratch.q_buf,
        &mut scratch.k_buf,
        &mut scratch.v_buf,
        tokens,
        q_dim,
        kv_dim,
    );

    // Apply RoPE to Q and K
    B::rope(
        &mut scratch.q_buf,
        &mut scratch.k_buf,
        cos,
        sin,
        positions,
        nh,
        nkv,
        hd,
    );

    // 5. KV cache append
    kv_cache_append::<B>(kv, &scratch.k_buf, &scratch.v_buf, tokens, nkv, hd);
    let kv_len = kv.len;

    // 6. Attention
    let attn_cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        causal: true,
        scale: 1.0 / (hd as f32).sqrt(),
    };

    if tokens == 1 {
        // Decode: single-query attention against full KV cache
        B::decode_attention(
            &scratch.q_buf,
            &kv.k,
            &kv.v,
            &mut scratch.attn_out,
            kv_len,
            &attn_cfg,
        );
    } else {
        // Prefill: full-sequence flash attention
        B::flash_attention(
            &scratch.q_buf,
            &kv.k,
            &kv.v,
            &mut scratch.attn_out,
            1,       // batch=1
            tokens,  // q_len
            kv_len,  // kv_len
            positions[0] as usize,
            &attn_cfg,
        );
    }

    // 7. O-projection: [tokens, q_dim] @ [hidden, q_dim]^T → [tokens, hidden]
    B::gemm(
        &scratch.attn_out,
        &weights.o_proj_w,
        &mut scratch.o_proj_out,
        tokens,
        h,
        q_dim,
    );

    // 8. Fused residual add + post-attention RMS norm
    //    residual += o_proj_out, then norm(residual) → norm_out
    B::fused_add_rms_norm(
        residual,
        &scratch.o_proj_out,
        &weights.post_ln_w,
        eps,
        &mut scratch.norm_out,
        tokens,
        h,
    );

    // 9. Fused gate+up projection: [tokens, hidden] @ [2*inter, hidden]^T → [tokens, 2*inter]
    B::gemm(
        &scratch.norm_out,
        &weights.gate_up_proj_w,
        &mut scratch.gate_up_out,
        tokens,
        2 * im,
        h,
    );

    // 10. SiLU(gate) * up
    //     gate_up_out layout: [tokens, 2*inter] → gate = [tokens, inter], up = [tokens, inter]
    //     Split is implicit: gate = gate_up_out[..tokens*im], up = gate_up_out[tokens*im..]
    //     But they're interleaved per token: [gate_0 | up_0 | gate_1 | up_1 | ...]
    //     Actually with fused weight [gate_w; up_w], output is [tokens, 2*im] where
    //     each row is [gate_out(im) | up_out(im)].
    silu_mul_split::<B>(
        &scratch.gate_up_out,
        &mut scratch.silu_out,
        tokens,
        im,
    );

    // 11. Down projection: [tokens, inter] @ [hidden, inter]^T → [tokens, hidden]
    B::gemm(
        &scratch.silu_out,
        &weights.down_proj_w,
        &mut scratch.mlp_out,
        tokens,
        h,
        im,
    );

    // 12. Final residual add: residual += mlp_out
    //     (Next layer will do its own norm at step 1)
    add_inplace::<B>(residual, &scratch.mlp_out, tokens * h);
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Split fused QKV output into separate Q, K, V buffers.
///
/// QKV layout per token: [Q(q_dim) | K(kv_dim) | V(kv_dim)]
/// Output: Q [tokens, q_dim], K [tokens, kv_dim], V [tokens, kv_dim]
fn split_qkv_all<B: Backend>(
    qkv: &B::Buffer,
    q_out: &mut B::Buffer,
    k_out: &mut B::Buffer,
    v_out: &mut B::Buffer,
    tokens: usize,
    q_dim: usize,
    kv_dim: usize,
) {
    let qkv_dim = q_dim + 2 * kv_dim;
    let qkv_vec = B::to_vec(qkv, tokens * qkv_dim);
    let mut q_vec = vec![0.0f32; tokens * q_dim];
    let mut k_vec = vec![0.0f32; tokens * kv_dim];
    let mut v_vec = vec![0.0f32; tokens * kv_dim];
    for t in 0..tokens {
        let base = t * qkv_dim;
        q_vec[t * q_dim..(t + 1) * q_dim]
            .copy_from_slice(&qkv_vec[base..base + q_dim]);
        k_vec[t * kv_dim..(t + 1) * kv_dim]
            .copy_from_slice(&qkv_vec[base + q_dim..base + q_dim + kv_dim]);
        v_vec[t * kv_dim..(t + 1) * kv_dim]
            .copy_from_slice(&qkv_vec[base + q_dim + kv_dim..base + qkv_dim]);
    }
    // Write into existing buffers (preserve allocation size)
    let q_buf = B::from_slice(&q_vec);
    let k_buf = B::from_slice(&k_vec);
    let v_buf = B::from_slice(&v_vec);
    B::copy(&q_buf, q_out, tokens * q_dim);
    B::copy(&k_buf, k_out, tokens * kv_dim);
    B::copy(&v_buf, v_out, tokens * kv_dim);
}

/// Append new K, V tokens to the KV cache.
///
/// KV cache layout: `[num_kv_heads, kv_len, head_dim]` (contiguous per head).
/// New K/V: `[tokens, num_kv_heads, head_dim]` (token-major from QKV projection).
///
/// We need to transpose new data to head-major and append to each head's sequence.
fn kv_cache_append<B: Backend>(
    kv: &mut KvCache<B>,
    new_k: &B::Buffer,
    new_v: &B::Buffer,
    new_tokens: usize,
    nkv: usize,
    hd: usize,
) {
    let old_len = kv.len;
    let new_len = old_len + new_tokens;

    // Read existing cache and new data
    let old_k = B::to_vec(&kv.k, nkv * old_len * hd);
    let old_v = B::to_vec(&kv.v, nkv * old_len * hd);
    let nk = B::to_vec(new_k, new_tokens * nkv * hd);
    let nv = B::to_vec(new_v, new_tokens * nkv * hd);

    // Build new cache: [nkv, new_len, hd]
    let mut full_k = vec![0.0f32; nkv * new_len * hd];
    let mut full_v = vec![0.0f32; nkv * new_len * hd];

    for h in 0..nkv {
        // Copy old cache for this head
        if old_len > 0 {
            let dst_off = h * new_len * hd;
            let src_off = h * old_len * hd;
            full_k[dst_off..dst_off + old_len * hd]
                .copy_from_slice(&old_k[src_off..src_off + old_len * hd]);
            full_v[dst_off..dst_off + old_len * hd]
                .copy_from_slice(&old_v[src_off..src_off + old_len * hd]);
        }
        // Append new tokens (transpose from token-major to head-major)
        for t in 0..new_tokens {
            let src = t * nkv * hd + h * hd;
            let dst = h * new_len * hd + (old_len + t) * hd;
            full_k[dst..dst + hd].copy_from_slice(&nk[src..src + hd]);
            full_v[dst..dst + hd].copy_from_slice(&nv[src..src + hd]);
        }
    }

    kv.k = B::from_slice(&full_k);
    kv.v = B::from_slice(&full_v);
    kv.len = new_len;
}

/// SiLU(gate) * up from fused [tokens, 2*inter] output.
///
/// Input layout per token: [gate(inter) | up(inter)]
fn silu_mul_split<B: Backend>(
    gate_up: &B::Buffer,
    out: &mut B::Buffer,
    tokens: usize,
    im: usize,
) {
    let data = B::to_vec(gate_up, tokens * 2 * im);
    let mut gate = vec![0.0f32; tokens * im];
    let mut up = vec![0.0f32; tokens * im];
    for t in 0..tokens {
        gate[t * im..(t + 1) * im].copy_from_slice(&data[t * 2 * im..t * 2 * im + im]);
        up[t * im..(t + 1) * im].copy_from_slice(&data[t * 2 * im + im..(t + 1) * 2 * im]);
    }
    let gate_buf = B::from_slice(&gate);
    let up_buf = B::from_slice(&up);
    B::silu_mul(&gate_buf, &up_buf, out, tokens * im);
}

/// residual += x (in-place)
fn add_inplace<B: Backend>(residual: &mut B::Buffer, x: &B::Buffer, len: usize) {
    let r = B::to_vec(residual, len);
    let xv = B::to_vec(x, len);
    let mut result = vec![0.0f32; len];
    for i in 0..len {
        result[i] = r[i] + xv[i];
    }
    *residual = B::from_slice(&result);
}
