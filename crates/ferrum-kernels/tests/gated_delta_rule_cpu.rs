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
