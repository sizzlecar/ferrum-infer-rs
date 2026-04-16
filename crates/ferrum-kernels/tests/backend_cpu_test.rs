//! CPU Backend correctness tests.
//!
//! These tests verify every Backend method against known reference values.
//! They run on all platforms (Linux CI + macOS dev) with zero GPU dependency.

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::{AttnConfig, Backend};

// ── Helpers ──────────────────────────────────────────────────────────────

fn assert_close(a: &[f32], b: &[f32], atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: length mismatch {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= atol,
            "{msg}[{i}]: {x} vs {y} (diff={})",
            (x - y).abs()
        );
    }
}

fn is_finite_nonzero(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0)
}

/// Simple pseudo-random sequence for reproducible tests.
fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

// ── GEMM ─────────────────────────────────────────────────────────────────

#[test]
fn test_gemm_identity() {
    // A[2,3] @ B[3,3]^T where B = I → out = A
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        1.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, // row 1
        0.0, 0.0, 1.0, // row 2
    ];
    let mut out = vec![0.0f32; 6];
    CpuBackend::gemm(&a, &b, &mut out, 2, 3, 3);
    assert_close(&out, &a, 1e-6, "gemm identity");
}

#[test]
fn test_gemm_small() {
    // A[1,2] = [1, 2], B[1,2] = [3, 4] → C[1,1] = 1*3 + 2*4 = 11
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let mut out = vec![0.0f32; 1];
    CpuBackend::gemm(&a, &b, &mut out, 1, 1, 2);
    assert_close(&out, &[11.0], 1e-5, "gemm small");
}

#[test]
fn test_gemm_rectangular() {
    // A[2,3] @ B[2,3]^T = C[2,2]
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3
    let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3
    let mut out = vec![0.0f32; 4];
    CpuBackend::gemm(&a, &b, &mut out, 2, 2, 3);
    // C = [[1,0],[0,1]]
    assert_close(&out, &[1.0, 0.0, 0.0, 1.0], 1e-6, "gemm rectangular");
}

// ── RMS Norm ─────────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_ones() {
    // x = [1,1,1,1], w = [1,1,1,1], eps = 0
    // rms = sqrt(mean(1^2)) = 1, out = x * w / rms = [1,1,1,1]
    let x = vec![1.0; 4];
    let w = vec![1.0; 4];
    let mut out = vec![0.0f32; 4];
    CpuBackend::rms_norm(&x, &w, 0.0, &mut out, 1, 4);
    assert_close(&out, &[1.0, 1.0, 1.0, 1.0], 1e-6, "rms_norm ones");
}

#[test]
fn test_rms_norm_scale() {
    // x = [2, 2], w = [0.5, 0.5], eps = 0
    // rms = sqrt(mean(4,4)) = 2, out = [2*0.5/2, 2*0.5/2] = [0.5, 0.5]
    let x = vec![2.0, 2.0];
    let w = vec![0.5, 0.5];
    let mut out = vec![0.0f32; 2];
    CpuBackend::rms_norm(&x, &w, 0.0, &mut out, 1, 2);
    assert_close(&out, &[0.5, 0.5], 1e-5, "rms_norm scale");
}

#[test]
fn test_rms_norm_multi_token() {
    let x = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens, dim=2
    let w = vec![1.0, 1.0];
    let mut out = vec![0.0f32; 4];
    CpuBackend::rms_norm(&x, &w, 1e-6, &mut out, 2, 2);
    // Each token normalized independently
    // Token 0: x=[1,0], sum_sq=1, mean_sq=0.5, rms=sqrt(0.5+eps), inv=1/rms
    // Token 1: x=[0,1], same
    let inv = 1.0 / (0.5f32 + 1e-6).sqrt();
    assert_close(&out, &[inv, 0.0, 0.0, inv], 1e-5, "rms_norm multi");
}

// ── Fused Add + RMS Norm ─────────────────────────────────────────────────

#[test]
fn test_fused_add_rms_norm() {
    let mut residual = vec![1.0, 1.0, 1.0, 1.0];
    let x = vec![1.0, 1.0, 1.0, 1.0];
    let w = vec![1.0; 4];
    let mut out = vec![0.0f32; 4];
    CpuBackend::fused_add_rms_norm(&mut residual, &x, &w, 0.0, &mut out, 1, 4);
    // residual = [2,2,2,2], rms = 2, out = [2/2, ...] = [1,1,1,1]
    assert_close(&residual, &[2.0; 4], 1e-6, "fused residual");
    assert_close(&out, &[1.0; 4], 1e-6, "fused norm out");
}

// ── RoPE ─────────────────────────────────────────────────────────────────

#[test]
fn test_rope_position_zero() {
    // At position 0, cos=1 sin=0 for all frequencies → no rotation
    let head_dim = 4;
    let half = 2;
    let cos = vec![1.0; 4 * half]; // max_seq=4
    let sin = vec![0.0; 4 * half];
    let mut q = vec![1.0, 2.0, 3.0, 4.0]; // 1 token, 1 head, dim=4
    let mut k = vec![5.0, 6.0, 7.0, 8.0];
    let positions = vec![0u32];
    CpuBackend::rope(&mut q, &mut k, &cos, &sin, &positions, 1, 1, head_dim);
    assert_close(&q, &[1.0, 2.0, 3.0, 4.0], 1e-6, "rope q pos0");
    assert_close(&k, &[5.0, 6.0, 7.0, 8.0], 1e-6, "rope k pos0");
}

#[test]
fn test_rope_rotation() {
    // Position 1, theta=1.0, head_dim=2 → freq = 1.0, angle = 1.0
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
    let c = 1.0f32.cos();
    let s = 1.0f32.sin();
    let cos = vec![1.0, c]; // pos 0, pos 1 (half=1)
    let sin = vec![0.0, s];
    let mut q = vec![1.0, 0.0]; // 1 token, 1 head, dim=2
    let mut k = vec![0.0; 2];
    let positions = vec![1u32];
    CpuBackend::rope(&mut q, &mut k, &cos, &sin, &positions, 1, 1, 2);
    // q[0] = 1*cos - 0*sin = cos, q[1] = 0*cos + 1*sin = sin
    assert_close(&q, &[c, s], 1e-5, "rope rotation");
}

// ── SiLU Mul ─────────────────────────────────────────────────────────────

#[test]
fn test_silu_mul() {
    let gate = vec![0.0, 1.0, -1.0, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0f32; 4];
    CpuBackend::silu_mul(&gate, &up, &mut out, 4);

    // silu(0) = 0, silu(1) = 1/(1+e^-1) ≈ 0.7311, silu(-1) = -1/(1+e^1) ≈ -0.2689
    assert!((out[0]).abs() < 1e-6, "silu(0)={}", out[0]);
    assert!((out[1] - 0.7311).abs() < 1e-3, "silu(1)={}", out[1]);
    assert!((out[2] - (-0.2689)).abs() < 1e-3, "silu(-1)={}", out[2]);
}

// ── Decode Attention ─────────────────────────────────────────────────────

#[test]
fn test_decode_attention_single_kv() {
    // 1 head, 1 KV token, dim=2: attention is trivially copying V
    let q = vec![1.0, 0.0]; // [num_heads=1, head_dim=2]
    let k = vec![1.0, 0.0]; // [num_kv_heads=1, kv_len=1, head_dim=2]
    let v = vec![3.0, 7.0];
    let mut out = vec![0.0f32; 2];
    let cfg = AttnConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        causal: false,
        scale: 1.0 / (2.0f32).sqrt(),
    };
    CpuBackend::decode_attention(&q, &k, &v, &mut out, 1, &cfg);
    assert_close(&out, &[3.0, 7.0], 1e-5, "decode attn single kv");
}

#[test]
fn test_decode_attention_gqa() {
    // 2 Q heads sharing 1 KV head, 1 KV token
    let q = vec![1.0, 0.0, 0.0, 1.0]; // [num_heads=2, head_dim=2]
    let k = vec![1.0, 0.0]; // [num_kv_heads=1, kv_len=1, head_dim=2]
    let v = vec![3.0, 7.0];
    let mut out = vec![0.0f32; 4];
    let cfg = AttnConfig {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        causal: false,
        scale: 1.0 / (2.0f32).sqrt(),
    };
    CpuBackend::decode_attention(&q, &k, &v, &mut out, 1, &cfg);
    // Both Q heads attend to the single KV → both get V
    assert_close(&out, &[3.0, 7.0, 3.0, 7.0], 1e-5, "decode attn gqa");
}

// ── Flash Attention (Prefill) ────────────────────────────────────────────

#[test]
fn test_flash_attention_causal() {
    // 1 batch, 1 head, 2 tokens, dim=4
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [1,1,2,4]
    let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [1,1,2,4]
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut out = vec![0.0f32; 8];
    let cfg = AttnConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        causal: true,
        scale: 1.0 / 2.0, // 1/sqrt(4)
    };
    CpuBackend::flash_attention(&q, &k, &v, &mut out, 1, 2, 2, 0, &cfg);

    // Token 0 only attends to token 0 → copies V[0]
    assert_close(&out[0..4], &[1.0, 2.0, 3.0, 4.0], 1e-4, "flash causal t0");

    // Token 1 attends to both: scores = [0, 0.5] * scale = [0, 0.25]
    // softmax([0, 0.5]) = online softmax result
    assert!(is_finite_nonzero(&out[4..8]), "flash causal t1 should be non-zero");
    // Weighted average of V[0] and V[1], skewed toward V[1]
    assert!(out[4] > 1.0 && out[4] < 5.0, "flash causal t1[0]={}", out[4]);
}

// ── Embedding Lookup ─────────────────────────────────────────────────────

#[test]
fn test_embedding_lookup() {
    // Vocab=3, dim=2
    let table = vec![
        0.1, 0.2, // token 0
        0.3, 0.4, // token 1
        0.5, 0.6, // token 2
    ];
    let ids = vec![2u32, 0, 1];
    let mut out = vec![0.0f32; 6];
    CpuBackend::embedding_lookup(&table, &ids, &mut out, 2);
    assert_close(
        &out,
        &[0.5, 0.6, 0.1, 0.2, 0.3, 0.4],
        1e-6,
        "embedding",
    );
}

// ── Add ──────────────────────────────────────────────────────────────────

#[test]
fn test_add() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let mut out = vec![0.0f32; 3];
    CpuBackend::add(&a, &b, &mut out, 3);
    assert_close(&out, &[5.0, 7.0, 9.0], 1e-6, "add");
}

// ── Buffer Management ────────────────────────────────────────────────────

#[test]
fn test_alloc_and_from_slice() {
    let buf = CpuBackend::alloc(4);
    assert_eq!(buf, vec![0.0; 4]);

    let buf2 = CpuBackend::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(CpuBackend::to_vec(&buf2, 3), vec![1.0, 2.0, 3.0]);
}

// ── Integration: mini transformer layer ──────────────────────────────────

#[test]
fn test_mini_layer_forward() {
    // Simulate a single transformer layer with tiny dimensions:
    // hidden=8, heads=2, kv_heads=2, head_dim=4, intermediate=16
    let h = 8;
    let nh = 2;
    let nkv = 2;
    let hd = 4;
    let im = 16;
    let eps = 1e-5f32;

    // Random weights (deterministic)
    let input_ln_w = pseudo_random(1, h);
    let qkv_w = pseudo_random(2, (nh * hd + 2 * nkv * hd) * h); // [q_dim+2*kv_dim, hidden]
    let o_proj_w = pseudo_random(3, h * (nh * hd));
    let post_ln_w = pseudo_random(4, h);
    let gate_up_w = pseudo_random(5, 2 * im * h);
    let down_w = pseudo_random(6, h * im);

    // Input: 1 token
    let input = pseudo_random(42, h);
    let mut residual = input.clone();

    // 1. RMS Norm
    let mut norm_out = CpuBackend::alloc(h);
    CpuBackend::rms_norm(&residual, &input_ln_w, eps, &mut norm_out, 1, h);
    assert!(is_finite_nonzero(&norm_out), "norm_out should be finite nonzero");

    // 2. QKV projection
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let qkv_dim = q_dim + 2 * kv_dim;
    let mut qkv_out = CpuBackend::alloc(qkv_dim);
    CpuBackend::gemm(&norm_out, &qkv_w, &mut qkv_out, 1, qkv_dim, h);
    assert!(is_finite_nonzero(&qkv_out), "qkv should be finite nonzero");

    // 3. Split Q, K, V
    let q = qkv_out[..q_dim].to_vec();
    let k = qkv_out[q_dim..q_dim + kv_dim].to_vec();
    let v = qkv_out[q_dim + kv_dim..].to_vec();

    // 4. Decode attention (1 token, kv_len=1)
    let mut attn_out = CpuBackend::alloc(q_dim);
    let cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        causal: false,
        scale: 1.0 / (hd as f32).sqrt(),
    };
    CpuBackend::decode_attention(&q, &k, &v, &mut attn_out, 1, &cfg);
    assert!(is_finite_nonzero(&attn_out), "attn_out should be finite nonzero");

    // 5. O projection
    let mut o_out = CpuBackend::alloc(h);
    CpuBackend::gemm(&attn_out, &o_proj_w, &mut o_out, 1, h, q_dim);

    // 6. Fused add + norm (residual += o_out, then norm)
    let mut post_norm = CpuBackend::alloc(h);
    CpuBackend::fused_add_rms_norm(&mut residual, &o_out, &post_ln_w, eps, &mut post_norm, 1, h);

    // 7. MLP: gate_up projection
    let mut gate_up_out = CpuBackend::alloc(2 * im);
    CpuBackend::gemm(&post_norm, &gate_up_w, &mut gate_up_out, 1, 2 * im, h);
    let gate = gate_up_out[..im].to_vec();
    let up = gate_up_out[im..].to_vec();

    // 8. SiLU + mul
    let mut silu_out = CpuBackend::alloc(im);
    CpuBackend::silu_mul(&gate, &up, &mut silu_out, im);

    // 9. Down projection
    let mut mlp_out = CpuBackend::alloc(h);
    CpuBackend::gemm(&silu_out, &down_w, &mut mlp_out, 1, h, im);

    // 10. Final residual add
    let mut final_out = CpuBackend::alloc(h);
    CpuBackend::add(&residual, &mlp_out, &mut final_out, h);
    assert!(is_finite_nonzero(&final_out), "layer output should be finite nonzero");

    // Verify output differs from input (something computed)
    let diff: f32 = input.iter().zip(&final_out).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.01, "layer output should differ from input, diff={diff}");
}

// ── Determinism ──────────────────────────────────────────────────────────

#[test]
fn test_deterministic() {
    // Same inputs → same outputs across two runs
    let x = pseudo_random(99, 64);
    let w = pseudo_random(100, 64);
    let mut out1 = CpuBackend::alloc(64);
    let mut out2 = CpuBackend::alloc(64);
    CpuBackend::rms_norm(&x, &w, 1e-5, &mut out1, 1, 64);
    CpuBackend::rms_norm(&x, &w, 1e-5, &mut out2, 1, 64);
    assert_close(&out1, &out2, 0.0, "determinism: rms_norm");
}
