//! Equivalence test: ferrum's `fused_silu_mul` (.cu PTX) vs the
//! triton-rs-compiled `fused_silu_mul_triton` produce numerically equivalent
//! output on the same input.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_fused_silu_mul_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::{fused_silu_mul, fused_silu_mul_triton};

const ROWS: usize = 32;
const COLS: usize = 1024;

fn make_inputs(dev: &Device) -> (Tensor, Tensor) {
    let g: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let u: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as f32) * 0.0091).cos())
        .collect();
    let g = Tensor::from_vec(g, (ROWS, COLS), dev).unwrap();
    let u = Tensor::from_vec(u, (ROWS, COLS), dev).unwrap();
    (g.to_dtype(DType::F32).unwrap(), u.to_dtype(DType::F32).unwrap())
}

#[test]
fn fused_silu_mul_triton_matches_native_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let (g, u) = make_inputs(&dev);

    let out_native = fused_silu_mul(&g, &u).unwrap();
    let out_triton = fused_silu_mul_triton(&g, &u).unwrap();

    let v_native: Vec<f32> = out_native.flatten_all().unwrap().to_vec1().unwrap();
    let v_triton: Vec<f32> = out_triton.flatten_all().unwrap().to_vec1().unwrap();

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
        "fused_silu_mul eq:  max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}  \
         (worst @ idx {argmax}: native={} triton={})",
        v_native[argmax], v_triton[argmax]
    );

    // SiLU uses fp32 sigmoid; both paths share the same algorithm, so
    // disagreement comes only from FMA ordering.
    assert!(
        max_abs < 1e-4,
        "max_abs_diff {max_abs} exceeds tolerance 1e-4 — triton-rs port diverges"
    );
}
