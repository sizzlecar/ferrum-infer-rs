use ferrum_kernels::backend::{cpu::CpuBackend, Backend, Dtype};

#[test]
fn activation_add_to_f32_shadow_default_matches_convert_then_add() {
    let mut ctx = CpuBackend::new_context();
    let src = CpuBackend::from_slice(&[0.5, -1.25, 2.0, 4.5]);
    let mut residual = CpuBackend::from_slice_typed::<f32>(&[1.0, 2.0, -3.0, 0.25]);
    let mut scratch = CpuBackend::alloc_typed(Dtype::F32, 4);

    CpuBackend::activation_add_to_f32_shadow(&mut ctx, &src, &mut residual, &mut scratch, 4);

    assert_eq!(
        CpuBackend::to_vec(&residual, 4),
        vec![1.5, 0.75, -1.0, 4.75]
    );
}
