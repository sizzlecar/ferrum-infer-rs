//! Equivalence test: ferrum's hand-written `rms_norm` (.cu PTX) vs the
//! triton-rs-compiled `rms_norm_triton` produce numerically equivalent
//! output on the same input.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_rms_norm_eq -- --nocapture
//!
//! Skipped (compiles to no tests) without the feature combo.

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::{rms_norm, rms_norm_triton};

const ROWS: usize = 32;
const ROW_SIZE: usize = 1024;
const EPS: f32 = 1e-6;

fn make_inputs(dev: &Device) -> (Tensor, Tensor) {
    let total: Vec<f32> = (0..ROWS * ROW_SIZE)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let weights: Vec<f32> = (0..ROW_SIZE).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let input = Tensor::from_vec(total, (ROWS, ROW_SIZE), dev).unwrap();
    let weight = Tensor::from_vec(weights, ROW_SIZE, dev).unwrap();
    (input.to_dtype(DType::F32).unwrap(), weight.to_dtype(DType::F32).unwrap())
}

#[test]
fn rms_norm_triton_matches_native_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let (input, weight) = make_inputs(&dev);

    let out_native = rms_norm(&input, &weight, EPS).unwrap();
    let out_triton = rms_norm_triton(&input, &weight, EPS).unwrap();

    let v_native: Vec<f32> = out_native.flatten_all().unwrap().to_vec1().unwrap();
    let v_triton: Vec<f32> = out_triton.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(v_native.len(), v_triton.len(), "length mismatch");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut argmax = 0usize;
    for (i, (&a, &b)) in v_native.iter().zip(v_triton.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = i;
        }
        let denom = a.abs().max(b.abs()).max(1e-12);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }

    println!(
        "rms_norm eq:  max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}  \
         (worst @ idx {argmax}: native={} triton={})",
        v_native[argmax], v_triton[argmax]
    );

    // Tolerance: both kernels accumulate in fp32 with the same algorithm
    // (sum-of-squares row reduction), so we expect agreement to a few ULPs.
    // Triton's internal fp32 ops use fused-multiply-add aggressively, which
    // can perturb the last bit of mantissa. 1e-5 absolute should be easy.
    assert!(
        max_abs < 1e-4,
        "max_abs_diff {max_abs} exceeds tolerance 1e-4 — triton-rs port diverges"
    );
}
