//! GPTQ parity: CPU dequant-GEMM vs CUDA Marlin on synthesised weights.
//!
//! Run (CPU only, always):
//!   cargo test -p ferrum-quantization --test gptq_parity_test cpu_selfcheck
//!
//! Run (GPU, needs --features cuda):
//!   cargo test -p ferrum-quantization --features cuda --release \
//!     --test gptq_parity_test cuda_vs_cpu -- --ignored --nocapture
//!
//! Uses randomly-generated GPTQ tensors (not a real model) so the test
//! doesn't depend on HF cache. Validates that the CUDA Marlin pipeline
//! (repack + upload + marlin_gemm) produces numerically-equivalent
//! output to the CPU dequant-then-GEMM path.

use ferrum_kernels::backend::Backend;
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::Linear;
use ferrum_quantization::GptqLinear;

/// Tiny deterministic PRNG so we don't need rand crate.
fn rnd_u32(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

fn rnd_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = (rnd_u32(state) & 0x00FF_FFFF) as f32 / 16_777_216.0; // [0, 1)
    lo + u * (hi - lo)
}

struct SyntheticGptq {
    k: usize,
    n: usize,
    bits: u32,
    group_size: usize,
    qweight: Vec<i32>,   // [K/8, N]
    scales: Vec<f32>,    // [K/group, N]
    qzeros: Vec<i32>,    // [K/group, N/8]
}

fn make_synthetic(k: usize, n: usize, group_size: usize, seed: u64) -> SyntheticGptq {
    assert_eq!(k % 8, 0);
    assert_eq!(n % 8, 0);
    assert_eq!(k % group_size, 0);
    let mut rs = seed;
    let num_groups = k / group_size;
    let mut qweight = vec![0i32; (k / 8) * n];
    for qw in qweight.iter_mut() {
        *qw = rnd_u32(&mut rs) as i32;
    }
    let mut scales = vec![0f32; num_groups * n];
    for s in scales.iter_mut() {
        *s = rnd_f32(&mut rs, 0.01, 0.1);
    }
    let mut qzeros = vec![0i32; num_groups * (n / 8)];
    for qz in qzeros.iter_mut() {
        // Choose zero-codes in [0, 16) so each packed int4 is 0..15.
        let mut word: u32 = 0;
        for bi in 0..8 {
            word |= ((rnd_u32(&mut rs) & 0xF) as u32) << (bi * 4);
        }
        *qz = word as i32;
    }
    SyntheticGptq { k, n, bits: 4, group_size, qweight, scales, qzeros }
}

/// CPU-side self-check: feed the GPTQ tensors through CpuBackend's
/// load_gptq (which dequantizes to f32) + run a GEMM, compare to a
/// from-scratch dequant+matmul reference. No GPU required.
#[test]
fn cpu_selfcheck() {
    let k = 256;
    let n = 128;
    let gs = 128;
    let syn = make_synthetic(k, n, gs, 0xDEADBEEF);

    let linear = GptqLinear::<CpuBackend>::from_raw(
        &syn.qweight, &syn.scales, &syn.qzeros,
        None, syn.bits, syn.group_size, syn.k, syn.n,
    )
    .expect("CPU load_gptq");

    // Reference: dequantize independently, run cblas gemm.
    let ref_w = dequant_reference(&syn);
    let input: Vec<f32> = (0..2 * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let m = 2;
    let mut out_linear = vec![0.0f32; m * n];
    let mut ctx = <CpuBackend as Backend>::new_context();
    linear.forward(&mut ctx, &input, &mut out_linear, m);

    let mut out_ref = vec![0.0f32; m * n];
    <CpuBackend as Backend>::gemm(&mut ctx, &input, &ref_w, &mut out_ref, m, n, k);

    let max_diff = out_linear.iter().zip(&out_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max_diff < 1e-3, "CPU GPTQ selfcheck drift: {max_diff}");
}

fn dequant_reference(syn: &SyntheticGptq) -> Vec<f32> {
    // Same math as CpuGptqStore's load, from-scratch for cross-verification.
    let mut w = vec![0f32; syn.n * syn.k];
    let packed_rows = syn.k / 8;
    for pr in 0..packed_rows {
        for col in 0..syn.n {
            let packed = syn.qweight[pr * syn.n + col] as u32;
            for bi in 0..8 {
                let ki = pr * 8 + bi;
                let q = ((packed >> (bi * 4)) & 0xF) as i32;
                let grp = ki / syn.group_size;
                let scale = syn.scales[grp * syn.n + col];
                let z_packed = syn.qzeros[grp * (syn.n / 8) + (col / 8)] as u32;
                let zero = (((z_packed >> ((col % 8) * 4)) & 0xF) as i32) + 1;
                w[col * syn.k + ki] = (q - zero) as f32 * scale;
            }
        }
    }
    w
}

/// CUDA Marlin vs CPU dequant reference.
/// Requires `--features cuda` AND a working GPU. Marked #[ignore] so it
/// doesn't run in default CPU builds.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_vs_cpu() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    use half::f16;

    // Marlin constraints: K % 128 == 0, N % 256 == 0.
    let k = 512;
    let n = 256;
    let gs = 128;
    let syn = make_synthetic(k, n, gs, 0xC0FFEE);

    // CPU reference (fp32).
    let cpu_linear = GptqLinear::<CpuBackend>::from_raw(
        &syn.qweight, &syn.scales, &syn.qzeros,
        None, syn.bits, syn.group_size, syn.k, syn.n,
    )
    .expect("CPU load_gptq");

    // CUDA Marlin path.
    let cuda_linear = GptqLinear::<CudaBackend>::from_raw(
        &syn.qweight, &syn.scales, &syn.qzeros,
        None, syn.bits, syn.group_size, syn.k, syn.n,
    )
    .expect("CUDA load_gptq (Marlin repack + upload)");

    let m = 2;
    let input_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.003).cos()).collect();
    let input_f16: Vec<f32> = input_f32.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // CPU out
    let mut cpu_out = vec![0.0f32; m * n];
    let mut cpu_ctx = <CpuBackend as Backend>::new_context();
    cpu_linear.forward(&mut cpu_ctx, &input_f16, &mut cpu_out, m);

    // CUDA out
    let mut cuda_ctx = <CudaBackend as Backend>::new_context();
    let input_dev = CudaBackend::from_slice(&input_f32);
    let mut out_dev = CudaBackend::alloc(m * n);
    cuda_linear.forward(&mut cuda_ctx, &input_dev, &mut out_dev, m);
    <CudaBackend as Backend>::sync(&mut cuda_ctx);
    let cuda_out = CudaBackend::to_vec(&out_dev, m * n);

    // Marlin is fp16-accum. CPU fp32. Expect modest drift.
    let max_diff = cpu_out
        .iter()
        .zip(&cuda_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let rel_err = max_diff / cpu_out.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-6);
    eprintln!(
        "CUDA↔CPU GPTQ: max|diff|={max_diff:.4}, rel={rel_err:.4}"
    );
    // fp16 GEMM typical relative error < 1% over small dims.
    assert!(
        rel_err < 0.05,
        "GPTQ CUDA/CPU mismatch too large: rel_err={rel_err}"
    );
}
