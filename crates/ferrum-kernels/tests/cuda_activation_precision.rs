//! CUDA activation precision checks for residual updates.
//!
//! Gemma3 sandwich norms can produce finite residual and branch values whose
//! sum exceeds the FP16 finite range. The default CUDA activation path is still
//! FP16, so this test documents the overflow and pins the F32 typed-buffer
//! path as finite.

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cuda::CudaBackend, Backend, Dtype};

const RESIDUAL: f32 = 40_416.0;
const BRANCH: f32 = 42_912.0;

#[test]
fn default_f16_add_inplace_overflows_gemma3_sandwich_scale() {
    let mut ctx = CudaBackend::new_context();
    let mut residual = CudaBackend::from_slice(&[RESIDUAL, 1.0]);
    let branch = CudaBackend::from_slice(&[BRANCH, 2.0]);

    CudaBackend::add_inplace(&mut ctx, &mut residual, &branch, 2);
    CudaBackend::sync(&mut ctx);

    let out = CudaBackend::to_vec(&residual, 2);
    assert!(
        out[0].is_infinite() && out[0].is_sign_positive(),
        "expected FP16 residual add to overflow, got {}",
        out[0]
    );
    assert!(
        (out[1] - 3.0).abs() < 1e-3,
        "small lane changed: {}",
        out[1]
    );
}

#[test]
fn f32_add_inplace_keeps_gemma3_sandwich_scale_finite() {
    let mut ctx = CudaBackend::new_context();
    let mut residual = CudaBackend::from_slice_typed::<f32>(&[RESIDUAL, 1.0]);
    let branch = CudaBackend::from_slice_typed::<f32>(&[BRANCH, 2.0]);

    assert_eq!(residual.dtype(), Dtype::F32);
    assert_eq!(branch.dtype(), Dtype::F32);

    CudaBackend::add_inplace(&mut ctx, &mut residual, &branch, 2);
    CudaBackend::sync(&mut ctx);

    let out = CudaBackend::to_vec(&residual, 2);
    assert!(
        out[0].is_finite(),
        "F32 residual add should preserve finite sum, got {}",
        out[0]
    );
    assert!(
        (out[0] - (RESIDUAL + BRANCH)).abs() < 1e-3,
        "sum lane: {}",
        out[0]
    );
    assert!((out[1] - 3.0).abs() < 1e-6, "small lane: {}", out[1]);
}
