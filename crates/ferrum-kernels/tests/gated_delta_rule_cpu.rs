use ferrum_kernels::backend::{cpu::CpuBackend, Backend};

#[test]
fn recurrent_gated_delta_rule_cpu_applies_decay_and_state_update() {
    let mut ctx = CpuBackend::new_context();
    let query = vec![1.0, 1.0];
    let key = vec![1.0, 1.0];
    let value = vec![2.0, 4.0];
    let g = vec![0.5f32.ln(), 0.5f32.ln()];
    let beta = vec![1.0, 1.0];
    let initial_state = vec![0.0];
    let mut out = vec![0.0; 2];
    let mut final_state = vec![0.0; 1];

    CpuBackend::recurrent_gated_delta_rule_f32(
        &mut ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_state,
        &mut out,
        &mut final_state,
        2,
        1,
        1,
        1,
        1,
        false,
        1.0,
    )
    .unwrap();

    assert_eq!(out, vec![2.0, 4.0]);
    assert_eq!(final_state, vec![4.0]);
}

#[test]
fn recurrent_gated_delta_rule_cpu_repeats_key_heads_over_value_heads() {
    let mut ctx = CpuBackend::new_context();
    let query = vec![2.0];
    let key = vec![3.0];
    let value = vec![5.0, 7.0];
    let g = vec![0.0, 0.0];
    let beta = vec![1.0, 1.0];
    let initial_state = vec![0.0, 0.0];
    let mut out = vec![0.0; 2];
    let mut final_state = vec![0.0; 2];

    CpuBackend::recurrent_gated_delta_rule_f32(
        &mut ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_state,
        &mut out,
        &mut final_state,
        1,
        1,
        2,
        1,
        1,
        false,
        1.0,
    )
    .unwrap();

    assert_eq!(out, vec![30.0, 42.0]);
    assert_eq!(final_state, vec![15.0, 21.0]);
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
fn recurrent_gated_delta_rule_batch_cpu_matches_repeated_single_token_updates() {
    let batch = 3;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 3;
    let value_dim = 2;
    let state_len = value_heads * value_dim * key_dim;
    let out_len = value_heads * value_dim;
    let query: Vec<f32> = (0..batch * key_heads * key_dim)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.125)
        .collect();
    let key: Vec<f32> = (0..batch * key_heads * key_dim)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.1)
        .collect();
    let value: Vec<f32> = (0..batch * value_heads * value_dim)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.2)
        .collect();
    let g: Vec<f32> = (0..batch * value_heads)
        .map(|i| -0.2 + (i as f32 % 5.0) * 0.05)
        .collect();
    let beta: Vec<f32> = (0..batch * value_heads)
        .map(|i| 0.2 + (i as f32 % 7.0) * 0.07)
        .collect();
    let initial_states: Vec<f32> = (0..batch * state_len)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.03)
        .collect();

    let mut ctx = CpuBackend::new_context();
    let mut out = vec![0.0; batch * out_len];
    let mut final_states = vec![0.0; batch * state_len];
    CpuBackend::recurrent_gated_delta_rule_batch_f32(
        &mut ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_states,
        &mut out,
        &mut final_states,
        batch,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        false,
        (key_dim as f32).sqrt().recip(),
    )
    .unwrap();

    let mut expected_out = vec![0.0; batch * out_len];
    let mut expected_states = vec![0.0; batch * state_len];
    for row in 0..batch {
        let row_q = query[row * key_heads * key_dim..(row + 1) * key_heads * key_dim].to_vec();
        let row_k = key[row * key_heads * key_dim..(row + 1) * key_heads * key_dim].to_vec();
        let row_v = value[row * out_len..(row + 1) * out_len].to_vec();
        let row_g = g[row * value_heads..(row + 1) * value_heads].to_vec();
        let row_beta = beta[row * value_heads..(row + 1) * value_heads].to_vec();
        let row_state = initial_states[row * state_len..(row + 1) * state_len].to_vec();
        let mut row_out = vec![0.0; out_len];
        let mut row_final = vec![0.0; state_len];
        CpuBackend::recurrent_gated_delta_rule_f32(
            &mut ctx,
            &row_q,
            &row_k,
            &row_v,
            &row_g,
            &row_beta,
            &row_state,
            &mut row_out,
            &mut row_final,
            1,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            false,
            (key_dim as f32).sqrt().recip(),
        )
        .unwrap();
        expected_out[row * out_len..(row + 1) * out_len].copy_from_slice(&row_out);
        expected_states[row * state_len..(row + 1) * state_len].copy_from_slice(&row_final);
    }

    assert_close(&out, &expected_out, 1e-6);
    assert_close(&final_states, &expected_states, 1e-6);
}

#[test]
fn recurrent_delta_external_qk_norm_matches_internal_norm() {
    let tokens = 1;
    let key_heads = 2;
    let value_heads = 4;
    let key_dim = 4;
    let value_dim = 3;
    let query = vec![0.2, -0.7, 1.1, 0.4, -0.5, 0.3, 0.9, -1.2];
    let key = vec![-0.4, 0.8, 0.6, -0.3, 1.0, -0.2, 0.5, 0.7];
    let value = (0..value_heads * value_dim)
        .map(|index| index as f32 * 0.11 - 0.45)
        .collect::<Vec<_>>();
    let g = vec![-0.1, -0.3, -0.05, -0.25];
    let beta = vec![0.2, 0.7, 0.4, 0.9];
    let state_len = value_heads * value_dim * key_dim;
    let initial_state = (0..state_len)
        .map(|index| index as f32 * 0.007 - 0.12)
        .collect::<Vec<_>>();
    let output_len = value_heads * value_dim;
    let scale = (key_dim as f32).sqrt().recip();

    let mut ctx = CpuBackend::new_context();
    let mut internal_out = vec![0.0; output_len];
    let mut internal_state = vec![0.0; state_len];
    CpuBackend::recurrent_gated_delta_rule_f32(
        &mut ctx,
        &query,
        &key,
        &value,
        &g,
        &beta,
        &initial_state,
        &mut internal_out,
        &mut internal_state,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        true,
        scale,
    )
    .unwrap();

    let mut normalized_query = query.clone();
    let mut normalized_key = key.clone();
    for head in 0..key_heads {
        let range = head * key_dim..(head + 1) * key_dim;
        let query_sum = normalized_query[range.clone()]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let key_sum = normalized_key[range.clone()]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let query_inv = (query_sum + 1.0e-6).sqrt().recip();
        let key_inv = (key_sum + 1.0e-6).sqrt().recip();
        for value in &mut normalized_query[range.clone()] {
            *value *= query_inv;
        }
        for value in &mut normalized_key[range] {
            *value *= key_inv;
        }
    }

    let mut external_out = vec![0.0; output_len];
    let mut external_state = vec![0.0; state_len];
    CpuBackend::recurrent_gated_delta_rule_f32(
        &mut ctx,
        &normalized_query,
        &normalized_key,
        &value,
        &g,
        &beta,
        &initial_state,
        &mut external_out,
        &mut external_state,
        tokens,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        false,
        scale,
    )
    .unwrap();

    assert_close(&external_out, &internal_out, 1e-6);
    assert_close(&external_state, &internal_state, 1e-6);
}
