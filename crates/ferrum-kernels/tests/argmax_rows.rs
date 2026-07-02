//! CUDA argmax row kernel tests.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!       --test argmax_rows -- --nocapture

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cuda::CudaBackend, Backend, Dtype};
use half::f16;

#[test]
fn argmax_rows_f16_masked_skips_invalid_tokens() {
    let mut ctx = CudaBackend::new_context();
    let logits = vec![
        f16::from_f32(1.0),
        f16::from_f32(9.0),
        f16::from_f32(4.0),
        f16::from_f32(3.0),
        f16::from_f32(0.0),
        f16::from_f32(5.0),
        f16::from_f32(7.0),
        f16::from_f32(8.0),
    ];
    let logits_dev = CudaBackend::from_slice_typed::<f16>(&logits);
    let mut mask_dev = CudaBackend::alloc_typed(Dtype::I8, 4);
    CudaBackend::write_typed::<i8>(&mut ctx, &mut mask_dev, &[1, 0, 1, 1]);

    let raw = CudaBackend::argmax_rows_f16(&mut ctx, &logits_dev, 2, 4).unwrap();
    assert_eq!(raw, vec![1, 3]);

    let masked =
        CudaBackend::argmax_rows_f16_masked(&mut ctx, &logits_dev, &mask_dev, 3, 2, 4).unwrap();
    assert_eq!(masked, vec![2, 2]);
}

#[test]
fn argmax_rows_f16_masked_returns_sentinel_without_finite_valid_token() {
    let mut ctx = CudaBackend::new_context();
    let logits = vec![
        f16::from_f32(9.0),
        f16::from_f32(f32::NAN),
        f16::from_f32(f32::NAN),
        f16::from_f32(8.0),
    ];
    let logits_dev = CudaBackend::from_slice_typed::<f16>(&logits);
    let mut mask_dev = CudaBackend::alloc_typed(Dtype::I8, 4);
    CudaBackend::write_typed::<i8>(&mut ctx, &mut mask_dev, &[0, 1, 1, 0]);

    let masked =
        CudaBackend::argmax_rows_f16_masked(&mut ctx, &logits_dev, &mask_dev, 4, 1, 4).unwrap();
    assert_eq!(masked, vec![u32::MAX]);
}
