//! Equivalence test: triton in-place `residual_add_inplace_triton` matches
//! the out-of-place `residual_add` (.cu PTX). The native code path has no
//! F32 in-place variant, so we compare against the out-of-place result.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_residual_add_inplace_eq -- --nocapture
//!
//! Skipped (compiles to no tests) without the feature combo.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::{residual_add, residual_add_inplace_triton};

const ROWS: usize = 32;
const COLS: usize = 1024;

fn make_inputs(dev: &Device) -> (Tensor, Tensor) {
    let a: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let b: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as f32) * 0.0091).cos())
        .collect();
    let a = Tensor::from_vec(a, (ROWS, COLS), dev).unwrap();
    let b = Tensor::from_vec(b, (ROWS, COLS), dev).unwrap();
    (a.to_dtype(DType::F32).unwrap(), b.to_dtype(DType::F32).unwrap())
}

#[test]
fn residual_add_inplace_triton_matches_native_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let (a, b) = make_inputs(&dev);

    // Reference: out-of-place sum via the .cu kernel.
    let out_native = residual_add(&a, &b).unwrap();

    // Triton in-place mutates `a`. Make a fresh copy so we don't touch the
    // tensor we just used as a reference operand.
    let a_copy = a.copy().unwrap();
    residual_add_inplace_triton(&a_copy, &b).unwrap();

    let v_native: Vec<f32> = out_native.flatten_all().unwrap().to_vec1().unwrap();
    let v_triton: Vec<f32> = a_copy.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(v_native.len(), v_triton.len(), "length mismatch");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut argmax = 0usize;
    for (i, (&x, &y)) in v_native.iter().zip(v_triton.iter()).enumerate() {
        let diff = (x - y).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = i;
        }
        let denom = x.abs().max(y.abs()).max(1e-12);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }

    println!(
        "residual_add_inplace eq:  max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}  \
         (worst @ idx {argmax}: native={} triton={})",
        v_native[argmax], v_triton[argmax]
    );

    assert!(
        max_abs < 1e-4,
        "max_abs_diff {max_abs} exceeds tolerance 1e-4 — triton-rs port diverges"
    );
}
