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
