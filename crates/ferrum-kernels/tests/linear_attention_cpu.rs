use ferrum_kernels::backend::{cpu::CpuBackend, Backend};

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

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *e).abs() <= tol,
            "idx={idx} actual={a} expected={e} tol={tol}"
        );
    }
}

#[test]
fn linear_attention_prepare_cpu_splits_gates_and_l2_normalizes_qk() {
    let tokens = 2;
    let key_heads = 1;
    let value_heads = 2;
    let key_dim = 2;
    let value_dim = 1;
    let conv_kernel = 2;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;

    let mixed_qkv_raw = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut conv_weight = Vec::new();
    for _ in 0..conv_channels {
        conv_weight.extend([0.0, 1.0]);
    }
    let a_raw = vec![0.25, -0.5, 1.0, -1.25];
    let b_raw = vec![-1.0, 0.0, 0.5, 1.5];
    let a_log = vec![2.0f32.ln(), 0.5f32.ln()];
    let dt_bias = vec![0.1, -0.2];

    let mut ctx = CpuBackend::new_context();
    let mut query = vec![0.0; tokens * qk_total];
    let mut key = vec![0.0; tokens * qk_total];
    let mut value = vec![0.0; tokens * value_total];
    let mut g = vec![0.0; tokens * value_heads];
    let mut beta = vec![0.0; tokens * value_heads];

    CpuBackend::linear_attention_prepare_f32(
        &mut ctx,
        &mixed_qkv_raw,
        &conv_weight,
        &a_raw,
        &b_raw,
        &a_log,
        &dt_bias,
        &mut query,
        &mut key,
        &mut value,
        &mut g,
        &mut beta,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let raw_query = [silu(1.0), silu(2.0), silu(7.0), silu(8.0)];
    let raw_key = [silu(3.0), silu(4.0), silu(9.0), silu(10.0)];
    let mut expected_query = raw_query;
    let mut expected_key = raw_key;
    for token in 0..tokens {
        let base = token * key_dim;
        let q_norm = (expected_query[base] * expected_query[base]
            + expected_query[base + 1] * expected_query[base + 1]
            + 1e-6)
            .sqrt();
        let k_norm = (expected_key[base] * expected_key[base]
            + expected_key[base + 1] * expected_key[base + 1]
            + 1e-6)
            .sqrt();
        expected_query[base] /= q_norm;
        expected_query[base + 1] /= q_norm;
        expected_key[base] /= k_norm;
        expected_key[base + 1] /= k_norm;
    }
    let expected_value = vec![silu(5.0), silu(6.0), silu(11.0), silu(12.0)];
    let expected_g = vec![
        -2.0 * softplus(0.25 + 0.1),
        -0.5 * softplus(-0.5 - 0.2),
        -2.0 * softplus(1.0 + 0.1),
        -0.5 * softplus(-1.25 - 0.2),
    ];
    let expected_beta = vec![sigmoid(-1.0), sigmoid(0.0), sigmoid(0.5), sigmoid(1.5)];

    assert_close(&query, &expected_query, 1e-6);
    assert_close(&key, &expected_key, 1e-6);
    assert_close(&value, &expected_value, 1e-6);
    assert_close(&g, &expected_g, 1e-6);
    assert_close(&beta, &expected_beta, 1e-6);
}

#[test]
fn linear_attention_decode_prepare_cpu_updates_conv_state_and_splits_current_token() {
    let key_heads = 1;
    let value_heads = 2;
    let key_dim = 2;
    let value_dim = 1;
    let conv_kernel = 3;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = conv_kernel - 1;

    let mixed_qkv_raw = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let conv_state = vec![
        1.0, 2.0, //
        3.0, 4.0, //
        5.0, 6.0, //
        7.0, 8.0, //
        9.0, 10.0, //
        11.0, 12.0,
    ];
    let mut conv_weight = Vec::new();
    for channel in 0..conv_channels {
        conv_weight.extend([
            0.1 + channel as f32 * 0.01,
            0.2 + channel as f32 * 0.01,
            0.3 + channel as f32 * 0.01,
        ]);
    }
    let a_raw = vec![0.25, -0.5];
    let b_raw = vec![-1.0, 1.5];
    let a_log = vec![2.0f32.ln(), 0.5f32.ln()];
    let dt_bias = vec![0.1, -0.2];

    let mut ctx = CpuBackend::new_context();
    let mut query = vec![0.0; qk_total];
    let mut key = vec![0.0; qk_total];
    let mut value = vec![0.0; value_total];
    let mut g = vec![0.0; value_heads];
    let mut beta = vec![0.0; value_heads];
    let mut next_conv_state = vec![0.0; conv_channels * state_len];

    CpuBackend::linear_attention_decode_prepare_f32(
        &mut ctx,
        &mixed_qkv_raw,
        &conv_weight,
        &conv_state,
        &a_raw,
        &b_raw,
        &a_log,
        &dt_bias,
        &mut query,
        &mut key,
        &mut value,
        &mut g,
        &mut beta,
        &mut next_conv_state,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        false,
    )
    .unwrap();

    let mut conv = vec![0.0; conv_channels];
    let mut expected_next_state = vec![0.0; conv_channels * state_len];
    for channel in 0..conv_channels {
        let state_base = channel * state_len;
        let weight_base = channel * conv_kernel;
        let acc = conv_state[state_base] * conv_weight[weight_base]
            + conv_state[state_base + 1] * conv_weight[weight_base + 1]
            + mixed_qkv_raw[channel] * conv_weight[weight_base + 2];
        conv[channel] = silu(acc);
        expected_next_state[state_base] = conv_state[state_base + 1];
        expected_next_state[state_base + 1] = mixed_qkv_raw[channel];
    }
    let expected_g = vec![-2.0 * softplus(0.25 + 0.1), -0.5 * softplus(-0.5 - 0.2)];
    let expected_beta = vec![sigmoid(-1.0), sigmoid(1.5)];

    assert_close(&query, &conv[0..qk_total], 1e-6);
    assert_close(&key, &conv[qk_total..2 * qk_total], 1e-6);
    assert_close(&value, &conv[2 * qk_total..], 1e-6);
    assert_close(&g, &expected_g, 1e-6);
    assert_close(&beta, &expected_beta, 1e-6);
    assert_close(&next_conv_state, &expected_next_state, 1e-6);
}

#[test]
fn gated_rms_norm_cpu_matches_norm_before_gate() {
    let tokens = 2;
    let heads = 1;
    let dim = 3;
    let eps = 1e-6;
    let core = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let z = vec![0.5, -0.25, 1.0, 1.5, -1.0, 0.25];
    let weight = vec![0.5, 1.5, -2.0];
    let mut out = vec![0.0; tokens * heads * dim];
    let mut ctx = CpuBackend::new_context();

    CpuBackend::gated_rms_norm_f32(
        &mut ctx, &core, &z, &weight, &mut out, tokens, heads, dim, eps,
    )
    .unwrap();

    let mut expected = vec![0.0; out.len()];
    for row in 0..tokens * heads {
        let base = row * dim;
        let sum_sq = core[base] * core[base]
            + core[base + 1] * core[base + 1]
            + core[base + 2] * core[base + 2];
        let inv = (sum_sq / dim as f32 + eps).sqrt().recip();
        for d in 0..dim {
            expected[base + d] = core[base + d] * inv * weight[d] * silu(z[base + d]);
        }
    }

    assert_close(&out, &expected, 1e-6);
}
