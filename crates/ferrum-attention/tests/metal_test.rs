//! Tests for Metal flash attention kernel.
//! Compares Metal output against CPU reference.

use ferrum_attention::{attention_cpu, AttentionParams};

#[cfg(target_os = "macos")]
use ferrum_attention::metal;

#[cfg(target_os = "macos")]
fn assert_close(a: &[f32], b: &[f32], atol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    assert!(
        max_diff < atol,
        "{label}: max diff {max_diff} at idx {max_idx} (a={}, b={}), atol={atol}",
        a[max_idx],
        b[max_idx]
    );
}

#[test]
#[cfg(target_os = "macos")]
fn test_metal_flash_attn_causal_small() {
    // 1 batch, 1 head, 4 tokens, dim=64
    let sq = 4;
    let sk = 4;
    let d = 64;
    let n = sq * d;

    // Random-ish Q, K, V
    let q: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.1).sin() * 0.5)).collect();
    let k: Vec<f32> = (0..sk * d)
        .map(|i| ((i as f32 * 0.07 + 1.0).cos() * 0.5))
        .collect();
    let v: Vec<f32> = (0..sk * d)
        .map(|i| ((i as f32 * 0.13 + 2.0).sin() * 0.3))
        .collect();

    let params = AttentionParams {
        batch: 1,
        num_heads: 1,
        num_kv_heads: 1,
        q_len: sq,
        kv_len: sk,
        head_dim: d,
        causal: true,
        pos_offset: 0,
    };

    let mut out_cpu = vec![0.0f32; n];
    attention_cpu(&q, &k, &v, &mut out_cpu, &params);

    let mut out_metal = vec![0.0f32; n];
    metal::fused_attention(&q, &k, &v, &mut out_metal, &params);

    assert_close(&out_cpu, &out_metal, 1e-4, "causal_small");
}

#[test]
#[cfg(target_os = "macos")]
fn test_metal_flash_attn_prefill_73() {
    // Matches TTS prefill: 1 batch, 16 heads, 73 tokens, head_dim=128
    let b = 1;
    let nh = 16;
    let nkv = 8;
    let sq = 73;
    let sk = 73;
    let d = 128;

    let q: Vec<f32> = (0..b * nh * sq * d)
        .map(|i| ((i as f32 * 0.01).sin() * 0.1))
        .collect();
    let k: Vec<f32> = (0..b * nkv * sk * d)
        .map(|i| ((i as f32 * 0.007 + 1.0).cos() * 0.1))
        .collect();
    let v: Vec<f32> = (0..b * nkv * sk * d)
        .map(|i| ((i as f32 * 0.013 + 2.0).sin() * 0.1))
        .collect();

    let params = AttentionParams {
        batch: b,
        num_heads: nh,
        num_kv_heads: nkv,
        q_len: sq,
        kv_len: sk,
        head_dim: d,
        causal: true,
        pos_offset: 0,
    };

    let mut out_cpu = vec![0.0f32; b * nh * sq * d];
    attention_cpu(&q, &k, &v, &mut out_cpu, &params);

    let mut out_metal = vec![0.0f32; b * nh * sq * d];
    metal::fused_attention(&q, &k, &v, &mut out_metal, &params);

    assert_close(&out_cpu, &out_metal, 1e-3, "prefill_73");
    eprintln!("prefill_73: Metal vs CPU max diff < 1e-3 ✓");
}

#[test]
#[cfg(target_os = "macos")]
fn test_metal_flash_attn_decode() {
    // Decode step: 1 query token, 74 KV tokens
    let b = 1;
    let nh = 16;
    let nkv = 8;
    let sq = 1;
    let sk = 74;
    let d = 128;

    let q: Vec<f32> = (0..b * nh * sq * d)
        .map(|i| ((i as f32 * 0.02).sin() * 0.2))
        .collect();
    let k: Vec<f32> = (0..b * nkv * sk * d)
        .map(|i| ((i as f32 * 0.005).cos() * 0.2))
        .collect();
    let v: Vec<f32> = (0..b * nkv * sk * d)
        .map(|i| ((i as f32 * 0.011 + 3.0).sin() * 0.2))
        .collect();

    let params = AttentionParams {
        batch: b,
        num_heads: nh,
        num_kv_heads: nkv,
        q_len: sq,
        kv_len: sk,
        head_dim: d,
        causal: false,
        pos_offset: 73,
    };

    let mut out_cpu = vec![0.0f32; b * nh * sq * d];
    attention_cpu(&q, &k, &v, &mut out_cpu, &params);

    let mut out_metal = vec![0.0f32; b * nh * sq * d];
    metal::fused_attention(&q, &k, &v, &mut out_metal, &params);

    assert_close(&out_cpu, &out_metal, 1e-4, "decode");
}

#[test]
#[cfg(target_os = "macos")]
fn bench_metal_flash_attn_prefill() {
    // Benchmark: 73 tokens, 16 heads, 128 dim
    let b = 1;
    let nh = 16;
    let nkv = 8;
    let sq = 73;
    let d = 128;

    let q: Vec<f32> = vec![0.1; b * nh * sq * d];
    let k: Vec<f32> = vec![0.1; b * nkv * sq * d];
    let v: Vec<f32> = vec![0.1; b * nkv * sq * d];

    let params = AttentionParams {
        batch: b,
        num_heads: nh,
        num_kv_heads: nkv,
        q_len: sq,
        kv_len: sq,
        head_dim: d,
        causal: true,
        pos_offset: 0,
    };

    // Warmup
    let mut out = vec![0.0f32; b * nh * sq * d];
    metal::fused_attention(&q, &k, &v, &mut out, &params);

    let start = std::time::Instant::now();
    let iters = 100;
    for _ in 0..iters {
        metal::fused_attention(&q, &k, &v, &mut out, &params);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "Metal flash attn prefill (73 tok, 16 heads, 128 dim): {:.2} ms/iter ({} iters in {:.1}ms)",
        elapsed.as_secs_f64() * 1000.0 / iters as f64,
        iters,
        elapsed.as_secs_f64() * 1000.0,
    );
}
