//! Equivalence test: ferrum's `fused_add_rms_norm` (.cu PTX) vs the
//! triton-rs-compiled `fused_add_rms_norm_triton` produce numerically
//! equivalent output on the same input. Both `(normalized, residual_out)`
//! tensors are compared element-wise.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!       --test triton_fused_add_rms_norm_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use candle_core::{DType, Device, Tensor};
use ferrum_kernels::{fused_add_rms_norm, fused_add_rms_norm_triton};

const TOKENS: usize = 32;
const HIDDEN: usize = 1024;
const EPS: f32 = 1e-6;

fn make_inputs(dev: &Device) -> (Tensor, Tensor, Tensor) {
    let inp: Vec<f32> = (0..TOKENS * HIDDEN)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let res: Vec<f32> = (0..TOKENS * HIDDEN)
        .map(|i| ((i as f32) * 0.0091).cos())
        .collect();
    let w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let inp = Tensor::from_vec(inp, (TOKENS, HIDDEN), dev).unwrap();
    let res = Tensor::from_vec(res, (TOKENS, HIDDEN), dev).unwrap();
    let w = Tensor::from_vec(w, HIDDEN, dev).unwrap();
    (
        inp.to_dtype(DType::F32).unwrap(),
        res.to_dtype(DType::F32).unwrap(),
        w.to_dtype(DType::F32).unwrap(),
    )
}

fn cmp(label: &str, native: &[f32], triton: &[f32]) {
    assert_eq!(native.len(), triton.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut argmax = 0usize;
    for (i, (&x, &y)) in native.iter().zip(triton.iter()).enumerate() {
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
        "{label} eq:  max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}  \
         (worst @ idx {argmax}: native={} triton={})",
        native[argmax], triton[argmax]
    );
    assert!(
        max_abs < 1e-4,
        "{label}: max_abs_diff {max_abs} exceeds tolerance 1e-4 — triton-rs port diverges"
    );
}

#[test]
fn fused_add_rms_norm_triton_matches_native_within_tolerance() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    let (inp, res, w) = make_inputs(&dev);

    let (n_norm, n_res) = fused_add_rms_norm(&inp, &res, &w, EPS).unwrap();
    let (t_norm, t_res) = fused_add_rms_norm_triton(&inp, &res, &w, EPS).unwrap();

    let nn: Vec<f32> = n_norm.flatten_all().unwrap().to_vec1().unwrap();
    let tn: Vec<f32> = t_norm.flatten_all().unwrap().to_vec1().unwrap();
    cmp("fused_add_rms_norm.normalized", &nn, &tn);

    let nr: Vec<f32> = n_res.flatten_all().unwrap().to_vec1().unwrap();
    let tr: Vec<f32> = t_res.flatten_all().unwrap().to_vec1().unwrap();
    cmp("fused_add_rms_norm.residual_out", &nr, &tr);
}
