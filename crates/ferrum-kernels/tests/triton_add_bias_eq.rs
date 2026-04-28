//! Equivalence test: triton-rs-compiled `add_bias_triton` (in-place)
//! matches a CPU reference (`data[r, c] += bias[c]`).
//!
//! There is no Rust-level native wrapper for the .cu add_bias kernel.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_add_bias_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::add_bias_triton;

const ROWS: usize = 32;
const COLS: usize = 1024;

fn cpu_add_bias(data: &[f32], bias: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] += bias[c];
        }
    }
    out
}

#[test]
fn add_bias_triton_matches_cpu_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");

    let data_v: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let bias_v: Vec<f32> = (0..COLS).map(|i| (i as f32) * 0.001 - 0.5).collect();

    let data = Tensor::from_vec(data_v.clone(), (ROWS, COLS), &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let bias = Tensor::from_vec(bias_v.clone(), COLS, &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    add_bias_triton(&data, &bias).unwrap();

    let v_triton: Vec<f32> = data.flatten_all().unwrap().to_vec1().unwrap();
    let v_cpu = cpu_add_bias(&data_v, &bias_v, ROWS, COLS);

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
        "add_bias eq vs CPU:  max_abs_diff={max_abs:.3e}  \
         (worst @ idx {argmax}: cpu={} triton={})",
        v_cpu[argmax], v_triton[argmax]
    );

    // Pure element-wise add — fp32 should be bit-exact in principle.
    assert!(
        max_abs < 1e-4,
        "add_bias max_abs_diff {max_abs} exceeds tolerance 1e-4"
    );
}
