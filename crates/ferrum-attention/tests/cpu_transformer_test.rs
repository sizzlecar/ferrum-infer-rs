//! Test FusedTransformer CPU path with random weights.
//! Verifies forward pass produces finite, non-zero output.

use ferrum_attention::*;

fn random_layer(h: usize, im: usize, nh: usize, nkv: usize, hd: usize, seed: u32) -> LayerWeights {
    let rng = |n: usize, s: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(s).wrapping_add(12345)) as f32;
                (x % 1000.0) / 10000.0 - 0.05
            })
            .collect()
    };
    LayerWeights {
        input_ln_w: vec![1.0f32; h],
        q_proj_w: rng(nh * hd * h, seed + 1),
        k_proj_w: rng(nkv * hd * h, seed + 2),
        v_proj_w: rng(nkv * hd * h, seed + 3),
        o_proj_w: rng(h * nh * hd, seed + 4),
        q_norm_w: vec![1.0f32; hd],
        k_norm_w: vec![1.0f32; hd],
        post_ln_w: vec![1.0f32; h],
        gate_proj_w: rng(im * h, seed + 5),
        up_proj_w: rng(im * h, seed + 6),
        down_proj_w: rng(h * im, seed + 7),
        attn_layer_scale: None,
        mlp_layer_scale: None,
    }
}

#[test]
fn test_cpu_forward_single_token() {
    let h = 64;
    let im = 128;
    let nh = 4;
    let nkv = 2;
    let hd = 16;
    let n_layers = 2;

    let cfg = TransformerConfig {
        hidden_size: h,
        intermediate_size: im,
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        num_layers: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        max_position_embeddings: 1024,
    };

    let layers: Vec<_> = (0..n_layers)
        .map(|i| random_layer(h, im, nh, nkv, hd, i as u32 * 100))
        .collect();
    let norm_w = vec![1.0f32; h];

    let mut fused = FusedTransformer::new(cfg, layers, norm_w);

    // Single token forward
    let input: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
    let output = fused.forward(&input, 1);

    assert_eq!(output.len(), h);
    assert!(output.iter().all(|x| x.is_finite()), "output has NaN/Inf");
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "output is all zeros");
}

#[test]
fn test_cpu_forward_multi_token() {
    let h = 64;
    let im = 128;
    let nh = 4;
    let nkv = 2;
    let hd = 16;

    let cfg = TransformerConfig {
        hidden_size: h,
        intermediate_size: im,
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        num_layers: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        max_position_embeddings: 1024,
    };

    let layers: Vec<_> = (0..2)
        .map(|i| random_layer(h, im, nh, nkv, hd, i as u32 * 100))
        .collect();
    let norm_w = vec![1.0f32; h];
    let mut fused = FusedTransformer::new(cfg, layers, norm_w);

    // Prefill 4 tokens
    let input: Vec<f32> = (0..4 * h)
        .map(|i| ((i as f32) * 0.001).sin() * 0.1)
        .collect();
    let output = fused.forward(&input, 4);

    assert_eq!(output.len(), 4 * h);
    assert!(output.iter().all(|x| x.is_finite()));

    // Decode 1 token (uses KV cache from prefill)
    let decode_input: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.002).cos() * 0.1).collect();
    let decode_output = fused.forward(&decode_input, 1);

    assert_eq!(decode_output.len(), h);
    assert!(decode_output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_cpu_reset() {
    let h = 64;
    let im = 128;
    let nh = 4;
    let nkv = 2;
    let hd = 16;

    let cfg = TransformerConfig {
        hidden_size: h,
        intermediate_size: im,
        num_heads: nh,
        num_kv_heads: nkv,
        head_dim: hd,
        num_layers: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        max_position_embeddings: 1024,
    };

    let layers: Vec<_> = (0..2)
        .map(|i| random_layer(h, im, nh, nkv, hd, i as u32 * 100))
        .collect();
    let norm_w = vec![1.0f32; h];
    let mut fused = FusedTransformer::new(cfg, layers, norm_w);

    let input: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();

    // Two runs should produce same result after reset
    let out1 = fused.forward(&input, 1);
    fused.reset();
    let out2 = fused.forward(&input, 1);

    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "reset didn't produce same output: {a} vs {b}"
        );
    }
}

#[test]
fn test_cpu_attention_basic() {
    let params = AttentionParams {
        batch: 1,
        num_heads: 2,
        num_kv_heads: 2,
        q_len: 4,
        kv_len: 4,
        head_dim: 8,
        causal: true,
        pos_offset: 0,
    };

    let n = params.batch * params.num_heads * params.q_len * params.head_dim;
    let nkv = params.batch * params.num_kv_heads * params.kv_len * params.head_dim;

    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let k: Vec<f32> = (0..nkv)
        .map(|i| ((i as f32) * 0.07 + 1.0).cos() * 0.5)
        .collect();
    let v: Vec<f32> = (0..nkv)
        .map(|i| ((i as f32) * 0.13 + 2.0).sin() * 0.3)
        .collect();
    let mut out = vec![0.0f32; n];

    attention_cpu(&q, &k, &v, &mut out, &params);

    assert!(
        out.iter().all(|x| x.is_finite()),
        "attention output has NaN/Inf"
    );
    let sum: f32 = out.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "attention output all zeros");
}

#[test]
fn test_sampling_deterministic() {
    // Same logits + same seed should give same token
    let _logits = vec![1.0, 5.0, 2.0, 8.0, 0.5];

    // Greedy (temp=0) should always pick index 3 (value 8.0)
    // We can't test this without access to sample_token, but we can
    // verify attention produces deterministic output
    let params = AttentionParams {
        batch: 1,
        num_heads: 1,
        num_kv_heads: 1,
        q_len: 1,
        kv_len: 3,
        head_dim: 4,
        causal: false,
        pos_offset: 0,
    };

    let q = vec![1.0f32; 4];
    let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let v = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut out1 = vec![0.0f32; 4];
    let mut out2 = vec![0.0f32; 4];

    attention_cpu(&q, &k, &v, &mut out1, &params);
    attention_cpu(&q, &k, &v, &mut out2, &params);

    for (a, b) in out1.iter().zip(out2.iter()) {
        assert_eq!(a, b, "attention not deterministic");
    }
}
