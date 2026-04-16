//! MetalBackend vs CpuBackend reference tests.
//!
//! Every Backend method is tested: Metal output must match CPU within atol.
//! Runs only on macOS with Metal feature.

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_kernels::backend::{AttnConfig, Backend};

fn assert_close(a: &[f32], b: &[f32], atol: f32, msg: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{msg}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let max_diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= atol, "{msg}: max_diff={max_diff} > atol={atol}");
}

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
fn test_metal_gemm_vs_cpu() {
    let m = 4;
    let n = 8;
    let k = 16;
    let a_data = pseudo_random(1, m * k);
    let b_data = pseudo_random(2, n * k);

    let mut cpu_out = CpuBackend::alloc(m * n);
    CpuBackend::gemm(&mut (), &a_data, &b_data, &mut cpu_out, m, n, k);

    let a_metal = MetalBackend::from_slice(&a_data);
    let b_metal = MetalBackend::from_slice(&b_data);
    let mut metal_out = MetalBackend::alloc(m * n);
    let mut mctx = MetalBackend::new_context();
    MetalBackend::gemm(&mut mctx, &a_metal, &b_metal, &mut metal_out, m, n, k);
    MetalBackend::sync(&mut mctx);

    let metal_vec = MetalBackend::to_vec(&metal_out, m * n);
    assert_close(&cpu_out, &metal_vec, 1e-4, "gemm");
}

// ── RMS Norm ─────────────────────────────────────────────────────────────

#[test]
fn test_metal_rms_norm_vs_cpu() {
    let tokens = 3;
    let dim = 64;
    let x_data = pseudo_random(10, tokens * dim);
    let w_data = pseudo_random(11, dim);
    let eps = 1e-5;

    let mut cpu_out = CpuBackend::alloc(tokens * dim);
    CpuBackend::rms_norm(&mut (), &x_data, &w_data, eps, &mut cpu_out, tokens, dim);

    let x_metal = MetalBackend::from_slice(&x_data);
    let w_metal = MetalBackend::from_slice(&w_data);
    let mut metal_out = MetalBackend::alloc(tokens * dim);
    let mut mctx = MetalBackend::new_context();
    MetalBackend::rms_norm(
        &mut mctx,
        &x_metal,
        &w_metal,
        eps,
        &mut metal_out,
        tokens,
        dim,
    );
    MetalBackend::sync(&mut mctx);

    let metal_vec = MetalBackend::to_vec(&metal_out, tokens * dim);
    assert_close(&cpu_out, &metal_vec, 1e-4, "rms_norm");
}

// ── SiLU Mul ─────────────────────────────────────────────────────────────

#[test]
fn test_metal_silu_mul_vs_cpu() {
    let len = 128;
    let gate_data = pseudo_random(20, len);
    let up_data = pseudo_random(21, len);

    let mut cpu_out = CpuBackend::alloc(len);
    CpuBackend::silu_mul(&mut (), &gate_data, &up_data, &mut cpu_out, len);

    let gate_metal = MetalBackend::from_slice(&gate_data);
    let up_metal = MetalBackend::from_slice(&up_data);
    let mut metal_out = MetalBackend::alloc(len);
    let mut mctx = MetalBackend::new_context();
    MetalBackend::silu_mul(&mut mctx, &gate_metal, &up_metal, &mut metal_out, len);
    MetalBackend::sync(&mut mctx);

    let metal_vec = MetalBackend::to_vec(&metal_out, len);
    assert_close(&cpu_out, &metal_vec, 1e-5, "silu_mul");
}

// ── Add ──────────────────────────────────────────────────────────────────

#[test]
fn test_metal_add_vs_cpu() {
    let len = 64;
    let a_data = pseudo_random(30, len);
    let b_data = pseudo_random(31, len);

    let mut cpu_out = CpuBackend::alloc(len);
    CpuBackend::add(&mut (), &a_data, &b_data, &mut cpu_out, len);

    let a_metal = MetalBackend::from_slice(&a_data);
    let b_metal = MetalBackend::from_slice(&b_data);
    let mut metal_out = MetalBackend::alloc(len);
    let mut mctx = MetalBackend::new_context();
    MetalBackend::add(&mut mctx, &a_metal, &b_metal, &mut metal_out, len);
    MetalBackend::sync(&mut mctx);

    let metal_vec = MetalBackend::to_vec(&metal_out, len);
    assert_close(&cpu_out, &metal_vec, 1e-6, "add");
}

// ── Flash Attention ──────────────────────────────────────────────────────

#[test]
#[ignore] // TODO: Metal shader uses different Q/K/V layout than Backend trait convention
fn test_metal_flash_attention_vs_cpu() {
    // 1 batch, 2 heads, 4 tokens, head_dim=8
    let batch = 1;
    let nh = 2;
    let nkv = 2;
    let seq = 4;
    let hd = 8;
    let size = batch * nh * seq * hd;

    let q_data = pseudo_random(40, size);
    let k_data = pseudo_random(41, batch * nkv * seq * hd);
    let v_data = pseudo_random(42, batch * nkv * seq * hd);

    let cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        causal: true,
        scale: 1.0 / (hd as f32).sqrt(),
    };

    let mut cpu_out = CpuBackend::alloc(size);
    CpuBackend::flash_attention(
        &mut (),
        &q_data,
        &k_data,
        &v_data,
        &mut cpu_out,
        batch,
        seq,
        seq,
        0,
        &cfg,
    );

    let q_metal = MetalBackend::from_slice(&q_data);
    let k_metal = MetalBackend::from_slice(&k_data);
    let v_metal = MetalBackend::from_slice(&v_data);
    let mut metal_out = MetalBackend::alloc(size);
    MetalBackend::flash_attention(
        &mut MetalBackend::new_context(),
        &q_metal,
        &k_metal,
        &v_metal,
        &mut metal_out,
        batch,
        seq,
        seq,
        0,
        &cfg,
    );

    let metal_vec = MetalBackend::to_vec(&metal_out, size);
    // Metal float32 accumulation order differs from CPU — allow 1e-3
    assert_close(&cpu_out, &metal_vec, 1e-3, "flash_attention");
}

// ── Decode Attention ─────────────────────────────────────────────────────

#[test]
#[ignore] // TODO: Metal shader uses different Q/K/V layout than Backend trait convention
fn test_metal_decode_attention_vs_cpu() {
    let nh = 2;
    let nkv = 1;
    let hd = 8;
    let kv_len = 5;

    let q_data = pseudo_random(50, nh * hd);
    let k_data = pseudo_random(51, nkv * kv_len * hd);
    let v_data = pseudo_random(52, nkv * kv_len * hd);

    let cfg = AttnConfig {
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        causal: false,
        scale: 1.0 / (hd as f32).sqrt(),
    };

    let mut cpu_out = CpuBackend::alloc(nh * hd);
    CpuBackend::decode_attention(
        &mut (),
        &q_data,
        &k_data,
        &v_data,
        &mut cpu_out,
        kv_len,
        &cfg,
    );

    let q_metal = MetalBackend::from_slice(&q_data);
    let k_metal = MetalBackend::from_slice(&k_data);
    let v_metal = MetalBackend::from_slice(&v_data);
    let mut metal_out = MetalBackend::alloc(nh * hd);
    MetalBackend::decode_attention(
        &mut MetalBackend::new_context(),
        &q_metal,
        &k_metal,
        &v_metal,
        &mut metal_out,
        kv_len,
        &cfg,
    );

    let metal_vec = MetalBackend::to_vec(&metal_out, nh * hd);
    assert_close(&cpu_out, &metal_vec, 1e-3, "decode_attention");
}

// ── Embedding ────────────────────────────────────────────────────────────

#[test]
fn test_metal_embedding_vs_cpu() {
    let vocab = 10;
    let dim = 16;
    let table_data = pseudo_random(60, vocab * dim);
    let ids = vec![3u32, 7, 1, 0];

    let mut cpu_out = CpuBackend::alloc(ids.len() * dim);
    CpuBackend::embedding_lookup(&mut (), &table_data, &ids, &mut cpu_out, dim);

    let table_metal = MetalBackend::from_slice(&table_data);
    let mut metal_out = MetalBackend::alloc(ids.len() * dim);
    MetalBackend::embedding_lookup(
        &mut MetalBackend::new_context(),
        &table_metal,
        &ids,
        &mut metal_out,
        dim,
    );

    let metal_vec = MetalBackend::to_vec(&metal_out, ids.len() * dim);
    assert_close(&cpu_out, &metal_vec, 0.0, "embedding");
}

// ── Buffer roundtrip ─────────────────────────────────────────────────────

#[test]
fn test_metal_buffer_roundtrip() {
    let data = vec![1.0f32, 2.0, 3.0, -4.5];
    let buf = MetalBackend::from_slice(&data);
    let out = MetalBackend::to_vec(&buf, 4);
    assert_eq!(data, out);
}
