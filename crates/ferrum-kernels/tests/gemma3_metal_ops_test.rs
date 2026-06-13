//! Metal-vs-CPU microbenches for the ops Gemma 3 newly exercises:
//! GeGLU split kernel, qk_norm_rope at head_dim=256, and sliding-window
//! flash attention at seq < window. Each compares Metal output against
//! the CPU backend on identical inputs (CPU verified against HF).
#![cfg(all(feature = "metal", target_os = "macos"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_kernels::backend::{AttnConfig, Backend};

fn pseudo(n: usize, seed: f32) -> Vec<f32> {
    // Deterministic, smooth-ish values in [-2, 2].
    (0..n)
        .map(|i| ((i as f32 * 0.7301 + seed) % 4.0) - 2.0)
        .collect()
}

fn assert_close(metal: &[f32], cpu: &[f32], tol: f32, what: &str) {
    assert_eq!(metal.len(), cpu.len(), "{what}: length");
    let mut max_abs = 0f32;
    for (i, (m, c)) in metal.iter().zip(cpu).enumerate() {
        assert!(
            m.is_finite(),
            "{what}: metal[{i}] not finite: {m} (cpu={c})"
        );
        max_abs = max_abs.max((m - c).abs());
    }
    assert!(max_abs < tol, "{what}: max_abs={max_abs} >= tol={tol}");
}

#[test]
fn gelu_tanh_mul_split_metal_matches_cpu() {
    let tokens = 3;
    let im = 257; // off the power-of-two path on purpose
    let mut gate_up = pseudo(tokens * 2 * im, 0.13);
    // Real Gemma gate activations reach ±60+; Metal fast-math tanh NaNs
    // there without clamping (caught live on gemma3:1b layer 0).
    gate_up[0] = 61.5;
    gate_up[1] = -58.0;
    gate_up[2] = 173.0;
    gate_up[im] = 2.0;
    gate_up[im + 1] = -3.0;
    gate_up[im + 2] = 0.5;

    let mut mctx = <MetalBackend as Backend>::new_context();
    let mgu = <MetalBackend as Backend>::from_slice(&gate_up);
    let mut mout = <MetalBackend as Backend>::alloc(tokens * im);
    <MetalBackend as Backend>::fused_gelu_tanh_mul_split(&mut mctx, &mgu, &mut mout, tokens, im);
    <MetalBackend as Backend>::sync(&mut mctx);
    let metal = <MetalBackend as Backend>::to_vec(&mout, tokens * im);

    let mut cctx = <CpuBackend as Backend>::new_context();
    let cgu = <CpuBackend as Backend>::from_slice(&gate_up);
    let mut cout = <CpuBackend as Backend>::alloc(tokens * im);
    <CpuBackend as Backend>::fused_gelu_tanh_mul_split(&mut cctx, &cgu, &mut cout, tokens, im);
    let cpu = <CpuBackend as Backend>::to_vec(&cout, tokens * im);

    assert_close(&metal, &cpu, 1e-4, "gelu_tanh_mul_split");
}

#[test]
fn qk_norm_rope_head_dim_256_metal_matches_cpu() {
    let tokens = 4;
    let heads = 2;
    let head_dim = 256; // Gemma3-1B; existing models all use 128
    let half = head_dim / 2;
    let max_pos = 64;
    let input = pseudo(tokens * heads * head_dim, 0.31);
    let norm_w = pseudo(head_dim, 0.77)
        .iter()
        .map(|x| 1.0 + x * 0.05)
        .collect::<Vec<_>>();
    let mut cos = vec![0f32; max_pos * half];
    let mut sin = vec![0f32; max_pos * half];
    for p in 0..max_pos {
        for i in 0..half {
            let freq = 1.0_f64 / 10_000f64.powf((2 * i) as f64 / head_dim as f64);
            cos[p * half + i] = ((p as f64) * freq).cos() as f32;
            sin[p * half + i] = ((p as f64) * freq).sin() as f32;
        }
    }

    let run_qk = |metal: bool| -> Vec<f32> {
        macro_rules! go {
            ($B:ty) => {{
                let mut ctx = <$B as Backend>::new_context();
                let inp = <$B as Backend>::from_slice(&input);
                let w = <$B as Backend>::from_slice(&norm_w);
                let c = <$B as Backend>::from_slice(&cos);
                let s = <$B as Backend>::from_slice(&sin);
                let mut out = <$B as Backend>::alloc(heads * tokens * head_dim);
                <$B as Backend>::qk_norm_rope(
                    &mut ctx, &inp, &w, &c, &s, &mut out, tokens, heads, head_dim, 5, 1e-6, 1,
                );
                <$B as Backend>::sync(&mut ctx);
                <$B as Backend>::to_vec(&out, heads * tokens * head_dim)
            }};
        }
        if metal {
            go!(MetalBackend)
        } else {
            go!(CpuBackend)
        }
    };

    assert_close(&run_qk(true), &run_qk(false), 2e-3, "qk_norm_rope hd=256");
}

#[test]
fn transpose_head_to_token_gemma3_shape_matches_cpu() {
    // Gemma3-1B post-attention untranspose: heads=4, head_dim=256,
    // tokens=15 — head_dim 256 is new territory (all prior models: 128).
    let tokens = 15;
    let heads = 4;
    let head_dim = 256;
    let input = pseudo(heads * tokens * head_dim, 0.59);

    macro_rules! go {
        ($B:ty) => {{
            let mut ctx = <$B as Backend>::new_context();
            let inp = <$B as Backend>::from_slice(&input);
            let mut out = <$B as Backend>::alloc(tokens * heads * head_dim);
            <$B as Backend>::transpose_head_to_token(
                &mut ctx, &inp, &mut out, tokens, heads, head_dim,
            );
            <$B as Backend>::sync(&mut ctx);
            <$B as Backend>::to_vec(&out, tokens * heads * head_dim)
        }};
    }
    let metal = go!(MetalBackend);
    let cpu = go!(CpuBackend);
    assert_close(&metal, &cpu, 1e-6, "transpose_head_to_token 4x15x256");
}

#[test]
fn sliding_window_prefill_seq_below_window_matches_cpu() {
    // seq=9 < window=512: the window must be a strict no-op vs full causal.
    let tokens = 9;
    let heads = 2;
    let kv_heads = 1;
    let head_dim = 64;
    let q = pseudo(heads * tokens * head_dim, 0.11);
    let k = pseudo(kv_heads * tokens * head_dim, 0.47);
    let v = pseudo(kv_heads * tokens * head_dim, 0.89);

    let run = |metal: bool, window: usize| -> Vec<f32> {
        macro_rules! go {
            ($B:ty) => {{
                let mut ctx = <$B as Backend>::new_context();
                let qb = <$B as Backend>::from_slice(&q);
                let kb = <$B as Backend>::from_slice(&k);
                let vb = <$B as Backend>::from_slice(&v);
                let mut out = <$B as Backend>::alloc(heads * tokens * head_dim);
                let cfg = AttnConfig {
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim,
                    causal: true,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    kv_seq_stride: tokens,
                    sliding_window: window,
                };
                <$B as Backend>::flash_attention(
                    &mut ctx, &qb, &kb, &vb, &mut out, 1, tokens, tokens, 0, &cfg,
                );
                <$B as Backend>::sync(&mut ctx);
                <$B as Backend>::to_vec(&out, heads * tokens * head_dim)
            }};
        }
        if metal {
            go!(MetalBackend)
        } else {
            go!(CpuBackend)
        }
    };

    let cpu_full = run(false, 0);
    let cpu_win = run(false, 512);
    assert_close(&cpu_win, &cpu_full, 1e-6, "cpu window-noop sanity");
    let metal_win = run(true, 512);
    assert_close(
        &metal_win,
        &cpu_full,
        2e-3,
        "metal sliding_window=512 seq=9",
    );
}

#[test]
fn flash_attention_gemma3_shape_matches_cpu() {
    // Gemma3-1B attention shape: 4 query heads, 1 kv head (GQA 4:1),
    // head_dim=256, prefill seq 15, local window 512. All prior Metal
    // models run head_dim=128.
    let tokens = 15;
    let heads = 4;
    let kv_heads = 1;
    let head_dim = 256;
    let q = pseudo(heads * tokens * head_dim, 0.21);
    let k = pseudo(kv_heads * tokens * head_dim, 0.43);
    let v = pseudo(kv_heads * tokens * head_dim, 0.67);

    let run = |metal: bool, window: usize| -> Vec<f32> {
        macro_rules! go {
            ($B:ty) => {{
                let mut ctx = <$B as Backend>::new_context();
                let qb = <$B as Backend>::from_slice(&q);
                let kb = <$B as Backend>::from_slice(&k);
                let vb = <$B as Backend>::from_slice(&v);
                let mut out = <$B as Backend>::alloc(heads * tokens * head_dim);
                let cfg = AttnConfig {
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim,
                    causal: true,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    kv_seq_stride: tokens,
                    sliding_window: window,
                };
                <$B as Backend>::flash_attention(
                    &mut ctx, &qb, &kb, &vb, &mut out, 1, tokens, tokens, 0, &cfg,
                );
                <$B as Backend>::sync(&mut ctx);
                <$B as Backend>::to_vec(&out, heads * tokens * head_dim)
            }};
        }
        if metal {
            go!(MetalBackend)
        } else {
            go!(CpuBackend)
        }
    };

    let cpu = run(false, 512);
    let metal = run(true, 512);
    assert_close(&metal, &cpu, 2e-3, "flash_attention gemma3 4/1x15x256 w512");
}
