//! Generic transformer layer forward — one function, all backends.
//!
//! All data manipulation (split, transpose, cache append) goes through Backend
//! methods — no CPU round-trips via to_vec/from_slice.

use super::{AttnConfig, Backend, KvCache, LayerScratch, LayerWeights, TransformerConfig};

/// Forward one transformer layer.
pub fn layer_forward<B: Backend>(
    ctx: &mut B::Context,
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
    B::rms_norm(
        ctx,
        residual,
        &weights.input_ln_w,
        eps,
        &mut scratch.norm_out,
        tokens,
        h,
    );

    // 2. Fused QKV projection
    B::gemm(
        ctx,
        &scratch.norm_out,
        &weights.qkv_proj_w,
        &mut scratch.qkv_out,
        tokens,
        qkv_dim,
        h,
    );

    // 3. Split QKV into separate buffers (Backend method, no CPU round-trip)
    B::split_qkv(
        ctx,
        &scratch.qkv_out,
        &mut scratch.q_buf,
        &mut scratch.k_buf,
        &mut scratch.v_buf,
        tokens,
        q_dim,
        kv_dim,
    );

    // 4. Optional QK norm
    if cfg.has_qk_norm {
        if let Some(ref w) = weights.q_norm_w {
            B::qk_norm(ctx, &mut scratch.q_buf, w, tokens, nh, hd, eps);
        }
        if let Some(ref w) = weights.k_norm_w {
            B::qk_norm(ctx, &mut scratch.k_buf, w, tokens, nkv, hd, eps);
        }
    }

    // 5. RoPE
    B::rope(
        ctx,
        &mut scratch.q_buf,
        &mut scratch.k_buf,
        cos,
        sin,
        positions,
        nh,
        nkv,
        hd,
    );

    // 6. KV cache append (Backend method handles transpose)
    let (new_cache_k, new_cache_v) = B::kv_cache_append(
        ctx,
        &mut kv.k,
        &mut kv.v,
        kv.len,
        &scratch.k_buf,
        &scratch.v_buf,
        tokens,
        nkv,
        hd,
    );
    kv.k = new_cache_k;
    kv.v = new_cache_v;
    kv.len += tokens;
    let kv_len = kv.len;

    // 7. Attention
    let attn_cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        causal: true,
        scale: 1.0 / (hd as f32).sqrt(),
    };

    if tokens == 1 {
        B::decode_attention(
            ctx,
            &scratch.q_buf,
            &kv.k,
            &kv.v,
            &mut scratch.attn_out,
            kv_len,
            &attn_cfg,
        );
    } else {
        // Prefill: transpose Q to head-major for flash attention
        B::transpose_token_to_head(ctx, &scratch.q_buf, &mut scratch.o_proj_out, tokens, nh, hd);
        B::flash_attention(
            ctx,
            &scratch.o_proj_out, // Q in head-major
            &kv.k,
            &kv.v,
            &mut scratch.attn_out,
            1,
            tokens,
            kv_len,
            positions[0] as usize,
            &attn_cfg,
        );
        // Transpose output back to token-major
        // attn_out is [nh, tokens, hd], need [tokens, nh, hd]
        let mut temp = B::alloc(tokens * q_dim);
        B::transpose_head_to_token(ctx, &scratch.attn_out, &mut temp, tokens, nh, hd);
        B::copy(ctx, &temp, &mut scratch.attn_out, tokens * q_dim);
    }

    // 8. O-projection
    B::gemm(
        ctx,
        &scratch.attn_out,
        &weights.o_proj_w,
        &mut scratch.o_proj_out,
        tokens,
        h,
        q_dim,
    );

    // 9. Fused residual add + post-attention RMS norm
    B::fused_add_rms_norm(
        ctx,
        residual,
        &scratch.o_proj_out,
        &weights.post_ln_w,
        eps,
        &mut scratch.norm_out,
        tokens,
        h,
    );

    // 10. Fused gate+up projection
    B::gemm(
        ctx,
        &scratch.norm_out,
        &weights.gate_up_proj_w,
        &mut scratch.gate_up_out,
        tokens,
        2 * im,
        h,
    );

    // 11. SiLU(gate) * up (Backend method, no CPU split)
    B::fused_silu_mul_split(ctx, &scratch.gate_up_out, &mut scratch.silu_out, tokens, im);

    // 12. Down projection
    B::gemm(
        ctx,
        &scratch.silu_out,
        &weights.down_proj_w,
        &mut scratch.mlp_out,
        tokens,
        h,
        im,
    );

    // 13. Residual add (in-place)
    B::add_inplace(ctx, residual, &scratch.mlp_out, tokens * h);
}
