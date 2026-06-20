#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{
    cpu::CpuBackend, cuda::gated_delta_rule, cuda::CudaBackend, Backend, Dtype,
};
use half::f16;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *e).abs() <= tol,
            "idx={idx} actual={a} expected={e} tol={tol}"
        );
    }
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

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn to_f16_vec(values: &[f32]) -> Vec<f16> {
    values.iter().map(|value| f16::from_f32(*value)).collect()
}

fn quantize_to_f16_f32(values: &[f32]) -> Vec<f32> {
    to_f16_vec(values)
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

#[test]
fn recurrent_gated_delta_rule_cuda_matches_cpu_reference() {
    let tokens = 3;
    let key_heads = 1;
    let value_heads = 2;
    let key_dim = 2;
    let value_dim = 2;
    let query = vec![1.0, 0.5, -0.25, 0.75, 0.5, -1.0];
    let key = vec![0.5, -0.5, 1.0, 0.25, -0.75, 0.5];
    let value = vec![
        2.0, -1.0, 0.5, 3.0, //
        -0.5, 1.5, 2.5, -2.0, //
        1.0, 2.0, -1.0, 0.25,
    ];
    let g = vec![0.0, -0.25, -0.5, 0.0, 0.125, -0.125];
    let beta = vec![0.5, 0.75, 0.25, 1.0, 0.6, 0.4];
    let initial_state = vec![0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.4, -0.3];
    let out_len = tokens * value_heads * value_dim;
    let state_len = value_heads * value_dim * key_dim;

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_out = vec![0.0; out_len];
    let mut expected_state = vec![0.0; state_len];
    CpuBackend::recurrent_gated_delta_rule_f32(
        &mut cpu_ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_state,
        &mut expected_out,
        &mut expected_state,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        true,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let q_dev = CudaBackend::from_slice_typed(&query);
    let k_dev = CudaBackend::from_slice_typed(&key);
    let v_dev = CudaBackend::from_slice_typed(&value);
    let g_dev = CudaBackend::from_slice_typed(&g);
    let beta_dev = CudaBackend::from_slice_typed(&beta);
    let state0_dev = CudaBackend::from_slice_typed(&initial_state);
    let mut out_dev = CudaBackend::alloc_typed(Dtype::F32, out_len);
    let mut state_dev = CudaBackend::alloc_typed(Dtype::F32, state_len);
    CudaBackend::recurrent_gated_delta_rule_f32(
        &mut cuda_ctx,
        &q_dev,
        &k_dev,
        &v_dev,
        &g_dev,
        &beta_dev,
        &state0_dev,
        &mut out_dev,
        &mut state_dev,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        true,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(&CudaBackend::to_vec(&out_dev, out_len), &expected_out, 2e-5);
    assert_close(
        &CudaBackend::to_vec(&state_dev, state_len),
        &expected_state,
        2e-5,
    );
}

#[test]
fn recurrent_gated_delta_rule_batch_cuda_matches_cpu_reference() {
    let batch = 3;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let out_len = batch * value_heads * value_dim;
    let state_len = batch * value_heads * value_dim * key_dim;
    let query: Vec<f32> = (0..batch * key_heads * key_dim)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.125)
        .collect();
    let key: Vec<f32> = (0..batch * key_heads * key_dim)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
        .collect();
    let value: Vec<f32> = (0..out_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.2)
        .collect();
    let g: Vec<f32> = (0..batch * value_heads)
        .map(|i| -0.2 + (i as f32 % 5.0) * 0.05)
        .collect();
    let beta: Vec<f32> = (0..batch * value_heads)
        .map(|i| 0.2 + (i as f32 % 7.0) * 0.07)
        .collect();
    let initial_states: Vec<f32> = (0..state_len)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.03)
        .collect();

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_out = vec![0.0; out_len];
    let mut expected_state = vec![0.0; state_len];
    CpuBackend::recurrent_gated_delta_rule_batch_f32(
        &mut cpu_ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_states,
        &mut expected_out,
        &mut expected_state,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        false,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let q_dev = CudaBackend::from_slice_typed(&query);
    let k_dev = CudaBackend::from_slice_typed(&key);
    let v_dev = CudaBackend::from_slice_typed(&value);
    let g_dev = CudaBackend::from_slice_typed(&g);
    let beta_dev = CudaBackend::from_slice_typed(&beta);
    let state0_dev = CudaBackend::from_slice_typed(&initial_states);
    let mut out_dev = CudaBackend::alloc_typed(Dtype::F32, out_len);
    let mut state_dev = CudaBackend::alloc_typed(Dtype::F32, state_len);
    CudaBackend::recurrent_gated_delta_rule_batch_f32(
        &mut cuda_ctx,
        &q_dev,
        &k_dev,
        &v_dev,
        &g_dev,
        &beta_dev,
        &state0_dev,
        &mut out_dev,
        &mut state_dev,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        false,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(&CudaBackend::to_vec(&out_dev, out_len), &expected_out, 2e-5);
    assert_close(
        &CudaBackend::to_vec(&state_dev, state_len),
        &expected_state,
        2e-5,
    );
}

#[test]
fn recurrent_gated_delta_rule_varlen_tiled_cuda_matches_cpu_reference() {
    let batch = 2;
    let tokens = 5;
    let key_heads = 1;
    let value_heads = 2;
    let key_dim = 128;
    let value_dim = 128;
    let cu_seqlens = vec![0u32, 2, 5];
    let qk_len = tokens * key_heads * key_dim;
    let value_len = tokens * value_heads * value_dim;
    let gate_len = tokens * value_heads;
    let state_len = batch * value_heads * value_dim * key_dim;
    let query: Vec<f32> = (0..qk_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.01)
        .collect();
    let key: Vec<f32> = (0..qk_len)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.01)
        .collect();
    let value: Vec<f32> = (0..value_len)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.005)
        .collect();
    let g: Vec<f32> = (0..gate_len)
        .map(|i| -0.01 - (i as f32 % 5.0) * 0.005)
        .collect();
    let beta: Vec<f32> = (0..gate_len)
        .map(|i| 0.2 + (i as f32 % 7.0) * 0.03)
        .collect();
    let initial_states: Vec<f32> = (0..state_len)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.001)
        .collect();

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_out = vec![0.0; value_len];
    let mut expected_state = vec![0.0; state_len];
    let cu_cpu = CpuBackend::from_slice_typed(&cu_seqlens);
    CpuBackend::recurrent_gated_delta_rule_varlen_f32(
        &mut cpu_ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_states,
        &cu_cpu,
        &mut expected_out,
        &mut expected_state,
        batch,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        false,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let q_dev = CudaBackend::from_slice_typed(&query);
    let k_dev = CudaBackend::from_slice_typed(&key);
    let v_dev = CudaBackend::from_slice_typed(&value);
    let g_dev = CudaBackend::from_slice_typed(&g);
    let beta_dev = CudaBackend::from_slice_typed(&beta);
    let state0_dev = CudaBackend::from_slice_typed(&initial_states);
    let cu_dev = CudaBackend::from_slice_typed(&cu_seqlens);
    let mut out_dev = CudaBackend::alloc_typed(Dtype::F32, value_len);
    let mut state_dev = CudaBackend::alloc_typed(Dtype::F32, state_len);
    CudaBackend::recurrent_gated_delta_rule_varlen_f32(
        &mut cuda_ctx,
        &q_dev,
        &k_dev,
        &v_dev,
        &g_dev,
        &beta_dev,
        &state0_dev,
        &cu_dev,
        &mut out_dev,
        &mut state_dev,
        batch,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        false,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&out_dev, value_len),
        &expected_out,
        1e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&state_dev, state_len),
        &expected_state,
        1e-4,
    );
}

fn packed_indexed_expected(
    mixed_qkv: &[f32],
    ba_raw: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    initial_slots: &[f32],
    slot_indices: &[u32],
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = value_heads * value_dim * key_dim;
    let mut expected_out = vec![0.0; batch * value_total];
    let mut expected_slots = initial_slots.to_vec();
    let mut cpu_ctx = CpuBackend::new_context();

    for row in 0..batch {
        let slot = slot_indices[row] as usize;
        assert!(slot < max_slots);
        let mixed_base = row * conv_channels;
        let query = &mixed_qkv[mixed_base..mixed_base + qk_total];
        let key = &mixed_qkv[mixed_base + qk_total..mixed_base + 2 * qk_total];
        let value = &mixed_qkv[mixed_base + 2 * qk_total..mixed_base + conv_channels];
        let ba_base = row * 2 * value_heads;
        let mut g = vec![0.0; value_heads];
        let mut beta = vec![0.0; value_heads];
        for value_head in 0..value_heads {
            let b_raw = ba_raw[ba_base + value_head];
            let a_raw = ba_raw[ba_base + value_heads + value_head];
            g[value_head] = -a_log[value_head].exp() * softplus(a_raw + dt_bias[value_head]);
            beta[value_head] = sigmoid(b_raw);
        }

        let state_base = slot * state_len;
        let out_base = row * value_total;
        let mut final_state = vec![0.0; state_len];
        CpuBackend::recurrent_gated_delta_rule_f32(
            &mut cpu_ctx,
            query,
            key,
            value,
            &g,
            &beta,
            &expected_slots[state_base..state_base + state_len],
            &mut expected_out[out_base..out_base + value_total],
            &mut final_state,
            1,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            true,
            (key_dim as f32).sqrt().recip(),
        )
        .unwrap();
        expected_slots[state_base..state_base + state_len].copy_from_slice(&final_state);
    }

    (expected_out, expected_slots)
}

#[test]
fn recurrent_gated_delta_rule_batch_indexed_packed_cuda_matches_cpu_reference() {
    let batch = 3;
    let max_slots = 5;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 4;
    let value_dim = 5;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = max_slots * value_heads * value_dim * key_dim;
    let mixed_qkv: Vec<f32> = (0..batch * conv_channels)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.013)
        .collect();
    let ba_raw: Vec<f32> = (0..batch * 2 * value_heads)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.07)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.25 + i as f32 * 0.05).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect();
    let initial_slots: Vec<f32> = (0..state_len)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.002)
        .collect();
    let slot_indices = vec![2u32, 0, 4];
    let (expected_out, expected_slots) = packed_indexed_expected(
        &mixed_qkv,
        &ba_raw,
        &a_log,
        &dt_bias,
        &initial_slots,
        &slot_indices,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    );

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkv);
    let ba_dev = CudaBackend::from_slice_typed(&ba_raw);
    let a_log_dev = CudaBackend::from_slice_typed(&a_log);
    let dt_bias_dev = CudaBackend::from_slice_typed(&dt_bias);
    let mut state_dev = CudaBackend::from_slice_typed(&initial_slots);
    let slot_dev = CudaBackend::from_slice_typed(&slot_indices);
    let mut out_dev = CudaBackend::alloc_typed(Dtype::F32, batch * value_total);
    gated_delta_rule::recurrent_gated_delta_rule_batch_indexed_packed_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &ba_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut state_dev,
        &slot_dev,
        &mut out_dev,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&out_dev, batch * value_total),
        &expected_out,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&state_dev, state_len),
        &expected_slots,
        2e-5,
    );
}

#[test]
fn recurrent_gated_delta_rule_batch_indexed_packed_cuda_accepts_f16_ba() {
    let batch = 2;
    let max_slots = 3;
    let key_heads = 1;
    let value_heads = 2;
    let key_dim = 8;
    let value_dim = 7;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = max_slots * value_heads * value_dim * key_dim;
    let mixed_qkv: Vec<f32> = (0..batch * conv_channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.009)
        .collect();
    let ba_raw_f32: Vec<f32> = (0..batch * 2 * value_heads)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.05)
        .collect();
    let ba_expected = quantize_to_f16_f32(&ba_raw_f32);
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.35 + i as f32 * 0.06).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.1 + i as f32 * 0.04).collect();
    let initial_slots: Vec<f32> = (0..state_len)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.003)
        .collect();
    let slot_indices = vec![1u32, 2];
    let (expected_out, expected_slots) = packed_indexed_expected(
        &mixed_qkv,
        &ba_expected,
        &a_log,
        &dt_bias,
        &initial_slots,
        &slot_indices,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    );

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkv);
    let ba_dev = CudaBackend::from_slice_typed(&to_f16_vec(&ba_raw_f32));
    let a_log_dev = CudaBackend::from_slice_typed(&a_log);
    let dt_bias_dev = CudaBackend::from_slice_typed(&dt_bias);
    let mut state_dev = CudaBackend::from_slice_typed(&initial_slots);
    let slot_dev = CudaBackend::from_slice_typed(&slot_indices);
    let mut out_dev = CudaBackend::alloc_typed(Dtype::F32, batch * value_total);
    gated_delta_rule::recurrent_gated_delta_rule_batch_indexed_packed_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &ba_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut state_dev,
        &slot_dev,
        &mut out_dev,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&out_dev, batch * value_total),
        &expected_out,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&state_dev, state_len),
        &expected_slots,
        2e-5,
    );
}
