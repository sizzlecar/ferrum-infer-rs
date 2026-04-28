//! Equivalence test: triton-rs-compiled `layer_norm_triton` matches a CPU
//! reference implementation within tolerance.
//!
//! There is no Rust-level native wrapper for the .cu LayerNorm kernel, so
//! the reference path is a straightforward CPU LN computed in `f64` and
//! cast back to `f32`.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_layer_norm_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::layer_norm_triton;

const ROWS: usize = 32;
const DIM: usize = 1024;
const EPS: f32 = 1e-5;

fn cpu_layer_norm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
        let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for c in 0..dim {
            let normed = (row[c] as f64 - mean) * inv_std;
            out[r * dim + c] = (normed * gamma[c] as f64 + beta[c] as f64) as f32;
        }
    }
    out
}

#[test]
fn layer_norm_triton_matches_cpu_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");

    let x_v: Vec<f32> = (0..ROWS * DIM)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let g_v: Vec<f32> = (0..DIM).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let b_v: Vec<f32> = (0..DIM).map(|i| (i as f32) * -0.0007).collect();

    let x = Tensor::from_vec(x_v.clone(), (ROWS, DIM), &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let gamma = Tensor::from_vec(g_v.clone(), DIM, &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let beta = Tensor::from_vec(b_v.clone(), DIM, &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let out_triton = layer_norm_triton(&x, &gamma, &beta, EPS).unwrap();
    let v_triton: Vec<f32> = out_triton.flatten_all().unwrap().to_vec1().unwrap();
    let v_cpu = cpu_layer_norm(&x_v, &g_v, &b_v, ROWS, DIM, EPS);

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

    println!(
        "layer_norm eq vs CPU:  max_abs_diff={max_abs:.3e}  \
         (worst @ idx {argmax}: cpu={} triton={})",
        v_cpu[argmax], v_triton[argmax]
    );

    // Welford-style reductions in fp32 with FMA can drift up to a few 1e-5
    // off the f64 reference; 1e-4 is comfortable headroom.
    assert!(
        max_abs < 1e-4,
        "layer_norm max_abs_diff {max_abs} exceeds tolerance 1e-4"
    );
}
