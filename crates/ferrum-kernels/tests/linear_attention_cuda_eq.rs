#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cpu::CpuBackend, cuda::CudaBackend, Backend, Dtype};

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
