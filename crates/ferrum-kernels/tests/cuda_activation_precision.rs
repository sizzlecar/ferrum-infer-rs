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

#[test]
fn f32_rms_norm_uses_f32_kernel() {
    let mut ctx = CudaBackend::new_context();
    let input = CudaBackend::from_slice_typed::<f32>(&[3.0, 4.0, 1.0, 2.0]);
    let weight = CudaBackend::from_slice_typed::<f32>(&[1.5, 0.5]);
    let mut out = CudaBackend::alloc_typed(Dtype::F32, 4);

    CudaBackend::rms_norm(&mut ctx, &input, &weight, 1e-6, &mut out, 2, 2);
    CudaBackend::sync(&mut ctx);

    assert_eq!(out.dtype(), Dtype::F32);
    let got = CudaBackend::to_vec(&out, 4);
    let mut expected = Vec::new();
    for row in [[3.0f32, 4.0], [1.0, 2.0]] {
        let inv_rms = ((row[0] * row[0] + row[1] * row[1]) / 2.0 + 1e-6)
            .sqrt()
            .recip();
        expected.push(row[0] * inv_rms * 1.5);
        expected.push(row[1] * inv_rms * 0.5);
    }

    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "idx={idx} got={g} expected={e} all={got:?}"
        );
    }
}

#[test]
fn f32_shadow_sandwich_norm_hooks_match_reference() {
    let mut ctx = CudaBackend::new_context();
    let activation = CudaBackend::from_slice(&[3.0, 4.0, 1.0, 2.0]);
    let weight = CudaBackend::from_slice(&[1.5, 0.5]);
    let mut shadow = CudaBackend::alloc_typed(Dtype::F32, 4);
    let mut branch = CudaBackend::alloc_typed(Dtype::F32, 4);
    let mut activation_out = CudaBackend::alloc_typed(Dtype::F16, 4);

    CudaBackend::activation_to_f32_shadow(&mut ctx, &activation, &mut shadow, 4);
    CudaBackend::rms_norm_activation_to_f32(
        &mut ctx,
        &activation,
        &weight,
        1e-6,
        &mut branch,
        2,
        2,
    );
    CudaBackend::rms_norm_f32_to_activation(
        &mut ctx,
        &shadow,
        &weight,
        1e-6,
        &mut activation_out,
        2,
        2,
    );
    CudaBackend::sync(&mut ctx);

    assert_eq!(shadow.dtype(), Dtype::F32);
    assert_eq!(branch.dtype(), Dtype::F32);
    assert_eq!(activation_out.dtype(), Dtype::F16);
    assert_eq!(CudaBackend::to_vec(&shadow, 4), vec![3.0, 4.0, 1.0, 2.0]);

    let mut expected = Vec::new();
    for row in [[3.0f32, 4.0], [1.0, 2.0]] {
        let inv_rms = ((row[0] * row[0] + row[1] * row[1]) / 2.0 + 1e-6)
            .sqrt()
            .recip();
        expected.push(row[0] * inv_rms * 1.5);
        expected.push(row[1] * inv_rms * 0.5);
    }

    for (idx, (got, expected)) in CudaBackend::to_vec(&branch, 4)
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (got - expected).abs() < 1e-5,
            "branch idx={idx} got={got} expected={expected}"
        );
    }
    for (idx, (got, expected)) in CudaBackend::to_vec(&activation_out, 4)
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (got - expected).abs() < 2e-3,
            "activation idx={idx} got={got} expected={expected}"
        );
    }
}
