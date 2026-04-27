//! Equivalence test: triton-rs-compiled `softmax_triton` matches a CPU
//! reference implementation within tolerance.
//!
//! There is no Rust-level native wrapper for the .cu softmax kernel, so
//! the reference path is a CPU softmax computed in `f64` and cast back to
//! `f32`.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_softmax_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::softmax_triton;

const ROWS: usize = 32;
const COLS: usize = 1024;

fn cpu_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row.iter().map(|&v| ((v - max_val) as f64).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for c in 0..cols {
            out[r * cols + c] = (exps[c] / sum) as f32;
        }
    }
    out
}

#[test]
fn softmax_triton_matches_cpu_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");

    let v: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as f32) * 0.0173).sin() * 5.0)
        .collect();
    let x = Tensor::from_vec(v.clone(), (ROWS, COLS), &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let out_triton = softmax_triton(&x).unwrap();
    let v_triton: Vec<f32> = out_triton.flatten_all().unwrap().to_vec1().unwrap();
    let v_cpu = cpu_softmax(&v, ROWS, COLS);

    assert_eq!(v_cpu.len(), v_triton.len(), "length mismatch");

    let mut max_abs = 0.0f32;
    let mut argmax = 0usize;
    for (i, (&a, &b)) in v_cpu.iter().zip(v_triton.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = i;
        }
    }

    // Sanity: each row should sum to 1.
    for r in 0..ROWS {
        let s: f64 = v_triton[r * COLS..(r + 1) * COLS]
            .iter()
            .map(|&v| v as f64)
            .sum();
        assert!(
            (s - 1.0).abs() < 1e-3,
            "softmax row {r} sums to {s} (expected ~1.0)"
        );
    }

    println!(
        "softmax eq vs CPU:  max_abs_diff={max_abs:.3e}  \
         (worst @ idx {argmax}: cpu={} triton={})",
        v_cpu[argmax], v_triton[argmax]
    );

    assert!(
        max_abs < 1e-4,
        "softmax max_abs_diff {max_abs} exceeds tolerance 1e-4"
    );
}
