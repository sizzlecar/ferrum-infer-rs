//! Equivalence test: triton-rs-compiled `gelu_triton` matches a CPU
//! reference implementation (erf-based GELU) within tolerance.
//!
//! There is no Rust-level native wrapper for the .cu GELU kernel; the
//! reference is computed in `f64` and cast to `f32`.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_gelu_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::gelu_triton;

const N: usize = 32 * 1024;

/// libm-style erf via series — but std::f64 has no erf. We approximate
/// using a high-quality rational (Abramowitz & Stegun 7.1.26 in f64), then
/// cast to f32. Tolerance below absorbs the approximation error.
fn erf_f64(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

fn cpu_gelu(x: &[f32]) -> Vec<f32> {
    let inv_sqrt_2 = 0.7071067811865475_f64;
    x.iter()
        .map(|&v| {
            let v = v as f64;
            (0.5 * v * (1.0 + erf_f64(v * inv_sqrt_2))) as f32
        })
        .collect()
}

#[test]
fn gelu_triton_matches_cpu_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");

    let v: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.0173).sin() * 3.0).collect();
    let x = Tensor::from_vec(v.clone(), N, &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let out_triton = gelu_triton(&x).unwrap();
    let v_triton: Vec<f32> = out_triton.flatten_all().unwrap().to_vec1().unwrap();
    let v_cpu = cpu_gelu(&v);

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
        "gelu eq vs CPU:  max_abs_diff={max_abs:.3e}  \
         (worst @ idx {argmax}: cpu={} triton={})",
        v_cpu[argmax], v_triton[argmax]
    );

    // The CPU reference uses an A&S 7.1.26 erf approximation (~1.5e-7 max
    // error), and Triton's `__nv_erff` (single-precision libdevice erf)
    // adds another ~1e-7. Headroom: 5e-4 covers both layers comfortably.
    assert!(
        max_abs < 5e-4,
        "gelu max_abs_diff {max_abs} exceeds tolerance 5e-4"
    );
}
