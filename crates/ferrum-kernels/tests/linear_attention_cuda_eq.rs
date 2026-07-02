#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cpu::CpuBackend, cuda::CudaBackend, Backend, Dtype};
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
fn linear_attention_prepare_cuda_matches_cpu_reference() {
    let tokens = 3;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 3;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let mixed_qkv_raw: Vec<f32> = (0..tokens * conv_channels)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.125)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 3 {
            0 => -0.25,
            1 => 0.5,
            _ => 0.75,
        })
        .collect();
    let a_raw: Vec<f32> = (0..tokens * value_heads)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
        .collect();
    let b_raw: Vec<f32> = (0..tokens * value_heads)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.5 + i as f32 * 0.25).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.1).collect();

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_q = vec![0.0; tokens * qk_total];
    let mut expected_k = vec![0.0; tokens * qk_total];
    let mut expected_v = vec![0.0; tokens * value_total];
    let mut expected_g = vec![0.0; tokens * value_heads];
    let mut expected_beta = vec![0.0; tokens * value_heads];
    CpuBackend::linear_attention_prepare_f32(
        &mut cpu_ctx,
        &mixed_qkv_raw,
        &conv_weight,
        &a_raw,
        &b_raw,
        &a_log,
        &dt_bias,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_g,
        &mut expected_beta,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkv_raw);
    let conv_dev = CudaBackend::from_slice_typed(&conv_weight);
    let a_dev = CudaBackend::from_slice_typed(&a_raw);
    let b_dev = CudaBackend::from_slice_typed(&b_raw);
    let a_log_dev = CudaBackend::from_slice_typed(&a_log);
    let dt_bias_dev = CudaBackend::from_slice_typed(&dt_bias);
    let mut q_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * qk_total);
    let mut k_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * qk_total);
    let mut v_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * value_total);
    let mut g_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * value_heads);
    let mut beta_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * value_heads);

    CudaBackend::linear_attention_prepare_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &a_dev,
        &b_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut q_dev,
        &mut k_dev,
        &mut v_dev,
        &mut g_dev,
        &mut beta_dev,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&q_dev, tokens * qk_total),
        &expected_q,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&k_dev, tokens * qk_total),
        &expected_k,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&v_dev, tokens * value_total),
        &expected_v,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&g_dev, tokens * value_heads),
        &expected_g,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&beta_dev, tokens * value_heads),
        &expected_beta,
        2e-5,
    );
}

#[test]
fn linear_attention_prepare_cuda_accepts_f16_inputs_and_writes_f32() {
    let tokens = 3;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 3;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let mixed_qkv_raw: Vec<f32> = (0..tokens * conv_channels)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.125)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 3 {
            0 => -0.25,
            1 => 0.5,
            _ => 0.75,
        })
        .collect();
    let a_raw: Vec<f32> = (0..tokens * value_heads)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
        .collect();
    let b_raw: Vec<f32> = (0..tokens * value_heads)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.5 + i as f32 * 0.25).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.1).collect();

    let mixed_qkv_expected = quantize_to_f16_f32(&mixed_qkv_raw);
    let conv_weight_expected = quantize_to_f16_f32(&conv_weight);
    let a_expected = quantize_to_f16_f32(&a_raw);
    let b_expected = quantize_to_f16_f32(&b_raw);
    let a_log_expected = quantize_to_f16_f32(&a_log);
    let dt_bias_expected = quantize_to_f16_f32(&dt_bias);

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_q = vec![0.0; tokens * qk_total];
    let mut expected_k = vec![0.0; tokens * qk_total];
    let mut expected_v = vec![0.0; tokens * value_total];
    let mut expected_g = vec![0.0; tokens * value_heads];
    let mut expected_beta = vec![0.0; tokens * value_heads];
    CpuBackend::linear_attention_prepare_f32(
        &mut cpu_ctx,
        &mixed_qkv_expected,
        &conv_weight_expected,
        &a_expected,
        &b_expected,
        &a_log_expected,
        &dt_bias_expected,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_g,
        &mut expected_beta,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&to_f16_vec(&mixed_qkv_raw));
    let conv_dev = CudaBackend::from_slice_typed(&to_f16_vec(&conv_weight));
    let a_dev = CudaBackend::from_slice_typed(&to_f16_vec(&a_raw));
    let b_dev = CudaBackend::from_slice_typed(&to_f16_vec(&b_raw));
    let a_log_dev = CudaBackend::from_slice_typed(&to_f16_vec(&a_log));
    let dt_bias_dev = CudaBackend::from_slice_typed(&to_f16_vec(&dt_bias));
    let mut q_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * qk_total);
    let mut k_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * qk_total);
    let mut v_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * value_total);
    let mut g_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * value_heads);
    let mut beta_dev = CudaBackend::alloc_typed(Dtype::F32, tokens * value_heads);

    CudaBackend::linear_attention_prepare_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &a_dev,
        &b_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut q_dev,
        &mut k_dev,
        &mut v_dev,
        &mut g_dev,
        &mut beta_dev,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&q_dev, tokens * qk_total),
        &expected_q,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&k_dev, tokens * qk_total),
        &expected_k,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&v_dev, tokens * value_total),
        &expected_v,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&g_dev, tokens * value_heads),
        &expected_g,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&beta_dev, tokens * value_heads),
        &expected_beta,
        5e-4,
    );
}

#[test]
fn linear_attention_prepare_varlen_packed_cuda_matches_cpu_reference() {
    let batch = 2;
    let total_tokens = 5;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 4;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let qkvz_width = conv_channels + value_total;
    let ba_width = 2 * value_heads;
    let conv_state_len = conv_channels * (conv_kernel - 1);

    let mixed_qkvz_raw: Vec<f32> = (0..total_tokens * qkvz_width)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.03125)
        .collect();
    let ba_raw: Vec<f32> = (0..total_tokens * ba_width)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.0625)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 5 {
            0 => -0.25,
            1 => 0.125,
            2 => 0.5,
            3 => -0.375,
            _ => 0.75,
        })
        .collect();
    let initial_conv_states: Vec<f32> = (0..batch * conv_state_len)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.015)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.5 + i as f32 * 0.25).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.1).collect();
    let cu_seqlens = vec![0u32, 3, 5];
    let token_seq_indices = vec![0u32, 0, 0, 1, 1];

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_q = vec![0.0; total_tokens * qk_total];
    let mut expected_k = vec![0.0; total_tokens * qk_total];
    let mut expected_v = vec![0.0; total_tokens * value_total];
    let mut expected_z = vec![0.0; total_tokens * value_total];
    let mut expected_g = vec![0.0; total_tokens * value_heads];
    let mut expected_beta = vec![0.0; total_tokens * value_heads];
    let mut expected_final_conv_states = vec![0.0; batch * conv_state_len];
    CpuBackend::linear_attention_prepare_varlen_packed_qkvz_ba_f32(
        &mut cpu_ctx,
        &mixed_qkvz_raw,
        &ba_raw,
        &conv_weight,
        &initial_conv_states,
        &a_log,
        &dt_bias,
        &CpuBackend::from_slice_typed(&cu_seqlens),
        &CpuBackend::from_slice_typed(&token_seq_indices),
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_z,
        &mut expected_g,
        &mut expected_beta,
        &mut expected_final_conv_states,
        batch,
        total_tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkvz_raw);
    let ba_dev = CudaBackend::from_slice_typed(&ba_raw);
    let conv_dev = CudaBackend::from_slice_typed(&conv_weight);
    let initial_conv_dev = CudaBackend::from_slice_typed(&initial_conv_states);
    let a_log_dev = CudaBackend::from_slice_typed(&a_log);
    let dt_bias_dev = CudaBackend::from_slice_typed(&dt_bias);
    let cu_dev = CudaBackend::from_slice_typed(&cu_seqlens);
    let token_rows_dev = CudaBackend::from_slice_typed(&token_seq_indices);
    let mut q_dev = CudaBackend::alloc_typed(Dtype::F32, total_tokens * qk_total);
    let mut k_dev = CudaBackend::alloc_typed(Dtype::F32, total_tokens * qk_total);
    let mut v_dev = CudaBackend::alloc_typed(Dtype::F32, total_tokens * value_total);
    let mut z_dev = CudaBackend::alloc_typed(Dtype::F32, total_tokens * value_total);
    let mut g_dev = CudaBackend::alloc_typed(Dtype::F32, total_tokens * value_heads);
    let mut beta_dev = CudaBackend::alloc_typed(Dtype::F32, total_tokens * value_heads);
    let mut final_conv_dev = CudaBackend::alloc_typed(Dtype::F32, batch * conv_state_len);
    CudaBackend::linear_attention_prepare_varlen_packed_qkvz_ba_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &ba_dev,
        &conv_dev,
        &initial_conv_dev,
        &a_log_dev,
        &dt_bias_dev,
        &cu_dev,
        &token_rows_dev,
        &mut q_dev,
        &mut k_dev,
        &mut v_dev,
        &mut z_dev,
        &mut g_dev,
        &mut beta_dev,
        &mut final_conv_dev,
        batch,
        total_tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&q_dev, total_tokens * qk_total),
        &expected_q,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&k_dev, total_tokens * qk_total),
        &expected_k,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&v_dev, total_tokens * value_total),
        &expected_v,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&z_dev, total_tokens * value_total),
        &expected_z,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&g_dev, total_tokens * value_heads),
        &expected_g,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&beta_dev, total_tokens * value_heads),
        &expected_beta,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&final_conv_dev, batch * conv_state_len),
        &expected_final_conv_states,
        2e-5,
    );
}

#[test]
fn linear_attention_decode_prepare_cuda_matches_cpu_reference() {
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 4;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = conv_kernel - 1;
    let mixed_qkv_raw: Vec<f32> = (0..conv_channels)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.125)
        .collect();
    let conv_state: Vec<f32> = (0..conv_channels * state_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.05)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 4 {
            0 => -0.125,
            1 => 0.25,
            2 => 0.5,
            _ => 0.75,
        })
        .collect();
    let a_raw: Vec<f32> = (0..value_heads)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
        .collect();
    let b_raw: Vec<f32> = (0..value_heads)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.5 + i as f32 * 0.25).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.1).collect();

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_q = vec![0.0; qk_total];
    let mut expected_k = vec![0.0; qk_total];
    let mut expected_v = vec![0.0; value_total];
    let mut expected_g = vec![0.0; value_heads];
    let mut expected_beta = vec![0.0; value_heads];
    let mut expected_next_conv_state = vec![0.0; conv_channels * state_len];
    CpuBackend::linear_attention_decode_prepare_f32(
        &mut cpu_ctx,
        &mixed_qkv_raw,
        &conv_weight,
        &conv_state,
        &a_raw,
        &b_raw,
        &a_log,
        &dt_bias,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_g,
        &mut expected_beta,
        &mut expected_next_conv_state,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkv_raw);
    let conv_dev = CudaBackend::from_slice_typed(&conv_weight);
    let conv_state_dev = CudaBackend::from_slice_typed(&conv_state);
    let a_dev = CudaBackend::from_slice_typed(&a_raw);
    let b_dev = CudaBackend::from_slice_typed(&b_raw);
    let a_log_dev = CudaBackend::from_slice_typed(&a_log);
    let dt_bias_dev = CudaBackend::from_slice_typed(&dt_bias);
    let mut q_dev = CudaBackend::alloc_typed(Dtype::F32, qk_total);
    let mut k_dev = CudaBackend::alloc_typed(Dtype::F32, qk_total);
    let mut v_dev = CudaBackend::alloc_typed(Dtype::F32, value_total);
    let mut g_dev = CudaBackend::alloc_typed(Dtype::F32, value_heads);
    let mut beta_dev = CudaBackend::alloc_typed(Dtype::F32, value_heads);
    let mut next_conv_state_dev = CudaBackend::alloc_typed(Dtype::F32, conv_channels * state_len);

    CudaBackend::linear_attention_decode_prepare_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &conv_state_dev,
        &a_dev,
        &b_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut q_dev,
        &mut k_dev,
        &mut v_dev,
        &mut g_dev,
        &mut beta_dev,
        &mut next_conv_state_dev,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(&CudaBackend::to_vec(&q_dev, qk_total), &expected_q, 2e-5);
    assert_close(&CudaBackend::to_vec(&k_dev, qk_total), &expected_k, 2e-5);
    assert_close(&CudaBackend::to_vec(&v_dev, value_total), &expected_v, 2e-5);
    assert_close(&CudaBackend::to_vec(&g_dev, value_heads), &expected_g, 2e-5);
    assert_close(
        &CudaBackend::to_vec(&beta_dev, value_heads),
        &expected_beta,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&next_conv_state_dev, conv_channels * state_len),
        &expected_next_conv_state,
        2e-5,
    );
}

#[test]
fn linear_attention_decode_prepare_cuda_accepts_f16_projection_inputs() {
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 4;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = conv_kernel - 1;
    let mixed_qkv_raw: Vec<f32> = (0..conv_channels)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.125)
        .collect();
    let conv_state: Vec<f32> = (0..conv_channels * state_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.05)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 4 {
            0 => -0.125,
            1 => 0.25,
            2 => 0.5,
            _ => 0.75,
        })
        .collect();
    let a_raw: Vec<f32> = (0..value_heads)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
        .collect();
    let b_raw: Vec<f32> = (0..value_heads)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.5 + i as f32 * 0.25).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.1).collect();

    let mixed_qkv_expected = quantize_to_f16_f32(&mixed_qkv_raw);
    let conv_weight_expected = quantize_to_f16_f32(&conv_weight);
    let a_expected = quantize_to_f16_f32(&a_raw);
    let b_expected = quantize_to_f16_f32(&b_raw);
    let a_log_expected = quantize_to_f16_f32(&a_log);
    let dt_bias_expected = quantize_to_f16_f32(&dt_bias);

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_q = vec![0.0; qk_total];
    let mut expected_k = vec![0.0; qk_total];
    let mut expected_v = vec![0.0; value_total];
    let mut expected_g = vec![0.0; value_heads];
    let mut expected_beta = vec![0.0; value_heads];
    let mut expected_next_conv_state = vec![0.0; conv_channels * state_len];
    CpuBackend::linear_attention_decode_prepare_f32(
        &mut cpu_ctx,
        &mixed_qkv_expected,
        &conv_weight_expected,
        &conv_state,
        &a_expected,
        &b_expected,
        &a_log_expected,
        &dt_bias_expected,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_g,
        &mut expected_beta,
        &mut expected_next_conv_state,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&to_f16_vec(&mixed_qkv_raw));
    let conv_dev = CudaBackend::from_slice_typed(&to_f16_vec(&conv_weight));
    let conv_state_dev = CudaBackend::from_slice_typed(&conv_state);
    let a_dev = CudaBackend::from_slice_typed(&to_f16_vec(&a_raw));
    let b_dev = CudaBackend::from_slice_typed(&to_f16_vec(&b_raw));
    let a_log_dev = CudaBackend::from_slice_typed(&to_f16_vec(&a_log));
    let dt_bias_dev = CudaBackend::from_slice_typed(&to_f16_vec(&dt_bias));
    let mut q_dev = CudaBackend::alloc_typed(Dtype::F32, qk_total);
    let mut k_dev = CudaBackend::alloc_typed(Dtype::F32, qk_total);
    let mut v_dev = CudaBackend::alloc_typed(Dtype::F32, value_total);
    let mut g_dev = CudaBackend::alloc_typed(Dtype::F32, value_heads);
    let mut beta_dev = CudaBackend::alloc_typed(Dtype::F32, value_heads);
    let mut next_conv_state_dev = CudaBackend::alloc_typed(Dtype::F32, conv_channels * state_len);

    CudaBackend::linear_attention_decode_prepare_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &conv_state_dev,
        &a_dev,
        &b_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut q_dev,
        &mut k_dev,
        &mut v_dev,
        &mut g_dev,
        &mut beta_dev,
        &mut next_conv_state_dev,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(&CudaBackend::to_vec(&q_dev, qk_total), &expected_q, 5e-4);
    assert_close(&CudaBackend::to_vec(&k_dev, qk_total), &expected_k, 5e-4);
    assert_close(&CudaBackend::to_vec(&v_dev, value_total), &expected_v, 5e-4);
    assert_close(&CudaBackend::to_vec(&g_dev, value_heads), &expected_g, 5e-4);
    assert_close(
        &CudaBackend::to_vec(&beta_dev, value_heads),
        &expected_beta,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&next_conv_state_dev, conv_channels * state_len),
        &expected_next_conv_state,
        5e-4,
    );
}

#[test]
fn linear_attention_decode_prepare_batch_cuda_matches_cpu_reference() {
    let batch = 3;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 4;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let state_len = conv_kernel - 1;
    let conv_state_len = conv_channels * state_len;
    let mixed_qkv_raw: Vec<f32> = (0..batch * conv_channels)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.125)
        .collect();
    let conv_states: Vec<f32> = (0..batch * conv_state_len)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.05)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 4 {
            0 => -0.125,
            1 => 0.25,
            2 => 0.5,
            _ => 0.75,
        })
        .collect();
    let a_raw: Vec<f32> = (0..batch * value_heads)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.2)
        .collect();
    let b_raw: Vec<f32> = (0..batch * value_heads)
        .map(|i| ((i as f32 % 5.0) - 2.0) * 0.3)
        .collect();
    let a_log: Vec<f32> = (0..value_heads)
        .map(|i| (0.5 + i as f32 * 0.25).ln())
        .collect();
    let dt_bias: Vec<f32> = (0..value_heads).map(|i| -0.2 + i as f32 * 0.1).collect();

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected_q = vec![0.0; batch * qk_total];
    let mut expected_k = vec![0.0; batch * qk_total];
    let mut expected_v = vec![0.0; batch * value_total];
    let mut expected_g = vec![0.0; batch * value_heads];
    let mut expected_beta = vec![0.0; batch * value_heads];
    let mut expected_next = vec![0.0; batch * conv_state_len];
    CpuBackend::linear_attention_decode_prepare_batch_f32(
        &mut cpu_ctx,
        &mixed_qkv_raw,
        &conv_weight,
        &conv_states,
        &a_raw,
        &b_raw,
        &a_log,
        &dt_bias,
        &mut expected_q,
        &mut expected_k,
        &mut expected_v,
        &mut expected_g,
        &mut expected_beta,
        &mut expected_next,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkv_raw);
    let conv_dev = CudaBackend::from_slice_typed(&conv_weight);
    let conv_states_dev = CudaBackend::from_slice_typed(&conv_states);
    let a_dev = CudaBackend::from_slice_typed(&a_raw);
    let b_dev = CudaBackend::from_slice_typed(&b_raw);
    let a_log_dev = CudaBackend::from_slice_typed(&a_log);
    let dt_bias_dev = CudaBackend::from_slice_typed(&dt_bias);
    let mut q_dev = CudaBackend::alloc_typed(Dtype::F32, batch * qk_total);
    let mut k_dev = CudaBackend::alloc_typed(Dtype::F32, batch * qk_total);
    let mut v_dev = CudaBackend::alloc_typed(Dtype::F32, batch * value_total);
    let mut g_dev = CudaBackend::alloc_typed(Dtype::F32, batch * value_heads);
    let mut beta_dev = CudaBackend::alloc_typed(Dtype::F32, batch * value_heads);
    let mut next_dev = CudaBackend::alloc_typed(Dtype::F32, batch * conv_state_len);
    CudaBackend::linear_attention_decode_prepare_batch_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &conv_states_dev,
        &a_dev,
        &b_dev,
        &a_log_dev,
        &dt_bias_dev,
        &mut q_dev,
        &mut k_dev,
        &mut v_dev,
        &mut g_dev,
        &mut beta_dev,
        &mut next_dev,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
        true,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&q_dev, batch * qk_total),
        &expected_q,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&k_dev, batch * qk_total),
        &expected_k,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&v_dev, batch * value_total),
        &expected_v,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&g_dev, batch * value_heads),
        &expected_g,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&beta_dev, batch * value_heads),
        &expected_beta,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&next_dev, batch * conv_state_len),
        &expected_next,
        2e-5,
    );
}

#[allow(clippy::too_many_arguments)]
fn packed_qkvz_to_mixed_expected(
    mixed_qkvz_raw: &[f32],
    conv_weight: &[f32],
    initial_slots: &[f32],
    slot_indices: &[u32],
    batch: usize,
    max_slots: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let qkvz_width = conv_channels + value_total;
    let state_len = conv_kernel - 1;
    let conv_state_len = conv_channels * state_len;
    let mut expected_mixed = vec![0.0; batch * conv_channels];
    let mut expected_z = vec![0.0; batch * value_total];
    let mut expected_slots = initial_slots.to_vec();
    let mut cpu_ctx = CpuBackend::new_context();
    let conv_weight = conv_weight.to_vec();
    let a_raw = vec![0.0; value_heads];
    let b_raw = vec![0.0; value_heads];
    let a_log = vec![0.0; value_heads];
    let dt_bias = vec![0.0; value_heads];

    for row in 0..batch {
        let slot = slot_indices[row] as usize;
        assert!(slot < max_slots);
        let raw_base = row * qkvz_width;
        let state_base = slot * conv_state_len;
        let mixed_qkv_raw = mixed_qkvz_raw[raw_base..raw_base + conv_channels].to_vec();
        let conv_state = expected_slots[state_base..state_base + conv_state_len].to_vec();
        let mut query = vec![0.0; qk_total];
        let mut key = vec![0.0; qk_total];
        let mut value = vec![0.0; value_total];
        let mut g = vec![0.0; value_heads];
        let mut beta = vec![0.0; value_heads];
        let mut next_state = vec![0.0; conv_state_len];
        CpuBackend::linear_attention_decode_prepare_f32(
            &mut cpu_ctx,
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
            &mut next_state,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            false,
        )
        .unwrap();

        let mixed_base = row * conv_channels;
        expected_mixed[mixed_base..mixed_base + qk_total].copy_from_slice(&query);
        expected_mixed[mixed_base + qk_total..mixed_base + 2 * qk_total].copy_from_slice(&key);
        expected_mixed[mixed_base + 2 * qk_total..mixed_base + conv_channels]
            .copy_from_slice(&value);
        expected_z[row * value_total..(row + 1) * value_total]
            .copy_from_slice(&mixed_qkvz_raw[raw_base + conv_channels..raw_base + qkvz_width]);
        expected_slots[state_base..state_base + conv_state_len].copy_from_slice(&next_state);
    }

    (expected_mixed, expected_z, expected_slots)
}

#[test]
fn linear_attention_decode_prepare_batch_indexed_packed_to_mixed_cuda_matches_reference() {
    let batch = 3;
    let max_slots = 5;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let conv_kernel = 4;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let qkvz_width = conv_channels + value_total;
    let conv_state_len = conv_channels * (conv_kernel - 1);
    let mixed_qkvz_raw: Vec<f32> = (0..batch * qkvz_width)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.03125)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| match i % 5 {
            0 => -0.25,
            1 => 0.125,
            2 => 0.5,
            3 => -0.375,
            _ => 0.75,
        })
        .collect();
    let initial_slots: Vec<f32> = (0..max_slots * conv_state_len)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.015)
        .collect();
    let slot_indices = vec![2u32, 0, 4];
    let (expected_mixed, expected_z, expected_slots) = packed_qkvz_to_mixed_expected(
        &mixed_qkvz_raw,
        &conv_weight,
        &initial_slots,
        &slot_indices,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    );

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&mixed_qkvz_raw);
    let conv_dev = CudaBackend::from_slice_typed(&conv_weight);
    let mut state_dev = CudaBackend::from_slice_typed(&initial_slots);
    let slot_dev = CudaBackend::from_slice_typed(&slot_indices);
    let mut mixed_out = CudaBackend::alloc_typed(Dtype::F32, batch * conv_channels);
    let mut z_out = CudaBackend::alloc_typed(Dtype::F32, batch * value_total);
    CudaBackend::linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &mut state_dev,
        &slot_dev,
        &mut mixed_out,
        &mut z_out,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&mixed_out, batch * conv_channels),
        &expected_mixed,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&z_out, batch * value_total),
        &expected_z,
        2e-5,
    );
    assert_close(
        &CudaBackend::to_vec(&state_dev, max_slots * conv_state_len),
        &expected_slots,
        2e-5,
    );
}

#[test]
fn linear_attention_decode_prepare_batch_indexed_packed_to_mixed_cuda_accepts_f16_inputs() {
    let batch = 2;
    let max_slots = 3;
    let key_heads = 1;
    let value_heads = 2;
    let key_dim = 4;
    let value_dim = 3;
    let conv_kernel = 3;
    let qk_total = key_heads * key_dim;
    let value_total = value_heads * value_dim;
    let conv_channels = 2 * qk_total + value_total;
    let qkvz_width = conv_channels + value_total;
    let conv_state_len = conv_channels * (conv_kernel - 1);
    let mixed_qkvz_raw: Vec<f32> = (0..batch * qkvz_width)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.02)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_channels * conv_kernel)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.05)
        .collect();
    let initial_slots: Vec<f32> = (0..max_slots * conv_state_len)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.01)
        .collect();
    let slot_indices = vec![1u32, 2];
    let mixed_expected = quantize_to_f16_f32(&mixed_qkvz_raw);
    let conv_expected = quantize_to_f16_f32(&conv_weight);
    let (expected_mixed, expected_z, expected_slots) = packed_qkvz_to_mixed_expected(
        &mixed_expected,
        &conv_expected,
        &initial_slots,
        &slot_indices,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    );

    let mut cuda_ctx = CudaBackend::new_context();
    let mixed_dev = CudaBackend::from_slice_typed(&to_f16_vec(&mixed_qkvz_raw));
    let conv_dev = CudaBackend::from_slice_typed(&to_f16_vec(&conv_weight));
    let mut state_dev = CudaBackend::from_slice_typed(&initial_slots);
    let slot_dev = CudaBackend::from_slice_typed(&slot_indices);
    let mut mixed_out = CudaBackend::alloc_typed(Dtype::F32, batch * conv_channels);
    let mut z_out = CudaBackend::alloc_typed(Dtype::F32, batch * value_total);
    CudaBackend::linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32(
        &mut cuda_ctx,
        &mixed_dev,
        &conv_dev,
        &mut state_dev,
        &slot_dev,
        &mut mixed_out,
        &mut z_out,
        batch,
        max_slots,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(
        &CudaBackend::to_vec(&mixed_out, batch * conv_channels),
        &expected_mixed,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&z_out, batch * value_total),
        &expected_z,
        5e-4,
    );
    assert_close(
        &CudaBackend::to_vec(&state_dev, max_slots * conv_state_len),
        &expected_slots,
        5e-4,
    );
}

#[test]
fn gated_rms_norm_cuda_matches_cpu_reference() {
    let tokens = 4;
    let heads = 3;
    let dim = 5;
    let len = tokens * heads * dim;
    let core: Vec<f32> = (0..len).map(|i| ((i as f32 % 13.0) - 6.0) * 0.1).collect();
    let z: Vec<f32> = (0..len).map(|i| ((i as f32 % 7.0) - 3.0) * 0.2).collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.25).collect();
    let eps = 1e-6;

    let mut cpu_ctx = CpuBackend::new_context();
    let mut expected = vec![0.0; len];
    CpuBackend::gated_rms_norm_f32(
        &mut cpu_ctx,
        &core,
        &z,
        &weight,
        &mut expected,
        tokens,
        heads,
        dim,
        eps,
    )
    .unwrap();

    let mut cuda_ctx = CudaBackend::new_context();
    let core_dev = CudaBackend::from_slice_typed(&core);
    let z_dev = CudaBackend::from_slice_typed(&z);
    let weight_dev = CudaBackend::from_slice_typed(&weight);
    let mut out_dev = CudaBackend::alloc_typed(Dtype::F32, len);
    CudaBackend::gated_rms_norm_f32(
        &mut cuda_ctx,
        &core_dev,
        &z_dev,
        &weight_dev,
        &mut out_dev,
        tokens,
        heads,
        dim,
        eps,
    )
    .unwrap();
    CudaBackend::sync(&mut cuda_ctx);

    assert_close(&CudaBackend::to_vec(&out_dev, len), &expected, 2e-5);
}
