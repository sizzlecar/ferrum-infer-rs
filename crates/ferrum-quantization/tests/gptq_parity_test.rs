//! GPTQ parity: CPU dequant-GEMM vs CUDA Marlin on synthesised weights.
//!
//! Run (CPU only, always):
//!   cargo test -p ferrum-quantization --test gptq_parity_test cpu_selfcheck
//!
//! Run (GPU, needs --features cuda):
//!   cargo test -p ferrum-quantization --features cuda --release \
//!     --test gptq_parity_test cuda_vs_cpu -- --ignored --nocapture
//!   cargo test -p ferrum-quantization --features cuda --release \
//!     --test gptq_parity_test cuda_desc_act_vs_cpu_reference -- --ignored --nocapture
//!   cargo test -p ferrum-quantization --features cuda --release \
//!     --test gptq_parity_test cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference \
//!     -- --ignored --nocapture
//!
//! Uses randomly-generated GPTQ tensors (not a real model) so the test
//! doesn't depend on HF cache. Validates that the CUDA Marlin pipeline
//! (repack + upload + marlin_gemm) produces numerically-equivalent
//! output to the CPU dequant-then-GEMM path.

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
#[cfg(feature = "cuda")]
use ferrum_kernels::backend::BackendQuantMarlin;
use ferrum_kernels::Linear;
use ferrum_quantization::GptqLinear;

/// Tiny deterministic PRNG so we don't need rand crate.
fn rnd_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
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
    qweight: Vec<i32>, // [K/8, N]
    scales: Vec<f32>,  // [K/group, N]
    qzeros: Vec<i32>,  // [K/group, N/8]
}

fn make_desc_act_g_idx(k: usize, group_size: usize) -> Vec<i32> {
    assert_eq!(k % group_size, 0);
    let num_groups = k / group_size;
    assert!(num_groups >= 4, "test g_idx pattern expects >=4 groups");

    // Deliberately non-monotonic but balanced: every quant group appears
    // exactly group_size times, so full-K act-order kernels can legally
    // sort by g_idx and recover contiguous full groups.
    (0..k).map(|i| (i % num_groups) as i32).collect()
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
            word |= (rnd_u32(&mut rs) & 0xF) << (bi * 4);
        }
        *qz = word as i32;
    }
    SyntheticGptq {
        k,
        n,
        bits: 4,
        group_size,
        qweight,
        scales,
        qzeros,
    }
}

fn make_synthetic_symmetric(k: usize, n: usize, group_size: usize, seed: u64) -> SyntheticGptq {
    let mut syn = make_synthetic(k, n, group_size, seed);
    for qz in syn.qzeros.iter_mut() {
        // Marlin's sym=true/no-zp path has an implicit int4 zero point of 8.
        // GPTQ qzeros stores zero-1, packed along N, so code 7 encodes zero=8.
        *qz = 0x7777_7777u32 as i32;
    }
    syn
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
        &syn.qweight,
        &syn.scales,
        &syn.qzeros,
        None,
        None,
        syn.bits,
        syn.group_size,
        syn.k,
        syn.n,
    )
    .expect("CPU load_gptq");

    // Reference: dequantize independently, run cblas gemm.
    let ref_w = dequant_reference(&syn);
    let input: Vec<f32> = (0..2 * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let m = 2;
    let mut out_linear = vec![0.0f32; m * n];
    <CpuBackend as Backend>::new_context();
    linear.forward(&mut (), &input, &mut out_linear, m);

    let mut out_ref = vec![0.0f32; m * n];
    <CpuBackend as Backend>::gemm(&mut (), &input, &ref_w, &mut out_ref, m, n, k);

    let max_diff = out_linear
        .iter()
        .zip(&out_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
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

fn dequant_reference_with_g_idx(syn: &SyntheticGptq, g_idx: &[i32]) -> Vec<f32> {
    assert_eq!(g_idx.len(), syn.k);
    let num_groups = syn.k / syn.group_size;
    let mut w = vec![0f32; syn.n * syn.k];
    let packed_rows = syn.k / 8;
    for pr in 0..packed_rows {
        for col in 0..syn.n {
            let packed = syn.qweight[pr * syn.n + col] as u32;
            for bi in 0..8 {
                let ki = pr * 8 + bi;
                let q = ((packed >> (bi * 4)) & 0xF) as i32;
                let grp = g_idx[ki] as usize;
                assert!(grp < num_groups);
                let scale = syn.scales[grp * syn.n + col];
                let z_packed = syn.qzeros[grp * (syn.n / 8) + (col / 8)] as u32;
                let zero = (((z_packed >> ((col % 8) * 4)) & 0xF) as i32) + 1;
                w[col * syn.k + ki] = (q - zero) as f32 * scale;
            }
        }
    }
    w
}

fn argsort_g_idx(g_idx: &[i32]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..g_idx.len()).collect();
    perm.sort_by_key(|&i| g_idx[i]);
    perm
}

fn permute_qweight_rows_for_test(qweight: &[i32], perm: &[usize], k: usize, n: usize) -> Vec<i32> {
    assert_eq!(perm.len(), k);
    assert_eq!(qweight.len(), (k / 8) * n);

    let packed_rows = k / 8;
    let mut unpacked = vec![0u8; k * n];
    for pr in 0..packed_rows {
        for col in 0..n {
            let packed = qweight[pr * n + col] as u32;
            for bi in 0..8 {
                unpacked[(pr * 8 + bi) * n + col] = ((packed >> (bi * 4)) & 0xF) as u8;
            }
        }
    }

    let mut sorted = vec![0u8; k * n];
    for (dst_row, &src_row) in perm.iter().enumerate() {
        let dst = dst_row * n;
        let src = src_row * n;
        sorted[dst..dst + n].copy_from_slice(&unpacked[src..src + n]);
    }

    let mut packed = vec![0i32; packed_rows * n];
    for pr in 0..packed_rows {
        for col in 0..n {
            let mut word = 0u32;
            for bi in 0..8 {
                word |= (sorted[(pr * 8 + bi) * n + col] as u32) << (bi * 4);
            }
            packed[pr * n + col] = word as i32;
        }
    }
    packed
}

#[test]
fn desc_act_reference_uses_g_idx_for_scale_lookup() {
    let syn = make_synthetic(512, 256, 128, 0x51A7E5);
    let g_idx = make_desc_act_g_idx(syn.k, syn.group_size);
    let mut counts = vec![0usize; syn.k / syn.group_size];
    for &g in &g_idx {
        counts[g as usize] += 1;
    }
    assert!(counts.iter().all(|&count| count == syn.group_size));

    let sequential = dequant_reference(&syn);
    let desc_act = dequant_reference_with_g_idx(&syn, &g_idx);
    let max_diff = sequential
        .iter()
        .zip(&desc_act)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    assert!(
        max_diff > 1e-3,
        "synthetic desc_act fixture must exercise non-sequential scale lookup"
    );
}

#[test]
fn symmetric_qzeros_encode_marlin_zero_point() {
    let syn = make_synthetic_symmetric(512, 256, 128, 0x51A7_E501);
    assert!(syn.qzeros.iter().all(|&qz| qz as u32 == 0x7777_7777));

    let w = dequant_reference(&syn);
    let has_positive = w.iter().any(|&x| x > 0.0);
    let has_negative = w.iter().any(|&x| x < 0.0);
    assert!(
        has_positive && has_negative,
        "sym=true qzeros fixture should dequantize around zero"
    );
}

#[test]
fn desc_act_perm_gather_is_equivalent_to_g_idx_reference() {
    let syn = make_synthetic(512, 256, 128, 0x5A17_0A7);
    let g_idx = make_desc_act_g_idx(syn.k, syn.group_size);
    let perm = argsort_g_idx(&g_idx);

    let ref_w = dequant_reference_with_g_idx(&syn, &g_idx);
    let permuted_qweight = permute_qweight_rows_for_test(&syn.qweight, &perm, syn.k, syn.n);
    let sorted_syn = SyntheticGptq {
        k: syn.k,
        n: syn.n,
        bits: syn.bits,
        group_size: syn.group_size,
        qweight: permuted_qweight,
        scales: syn.scales.clone(),
        qzeros: syn.qzeros.clone(),
    };
    let sorted_w = dequant_reference(&sorted_syn);

    let m = 3;
    let input: Vec<f32> = (0..m * syn.k).map(|i| (i as f32 * 0.0041).sin()).collect();
    let mut gathered = vec![0.0f32; input.len()];
    for row in 0..m {
        for (dst_k, &src_k) in perm.iter().enumerate() {
            gathered[row * syn.k + dst_k] = input[row * syn.k + src_k];
        }
    }

    let mut out_ref = vec![0.0f32; m * syn.n];
    <CpuBackend as Backend>::gemm(&mut (), &input, &ref_w, &mut out_ref, m, syn.n, syn.k);

    let mut out_perm = vec![0.0f32; m * syn.n];
    <CpuBackend as Backend>::gemm(
        &mut (),
        &gathered,
        &sorted_w,
        &mut out_perm,
        m,
        syn.n,
        syn.k,
    );

    let max_diff = out_ref
        .iter()
        .zip(&out_perm)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "desc_act perm/gather transform diverged from g_idx reference: {max_diff}"
    );
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
    let syn = make_synthetic_symmetric(k, n, gs, 0xC0FFEE);

    // CPU reference (fp32).
    let cpu_linear = GptqLinear::<CpuBackend>::from_raw(
        &syn.qweight,
        &syn.scales,
        &syn.qzeros,
        None,
        None,
        syn.bits,
        syn.group_size,
        syn.k,
        syn.n,
    )
    .expect("CPU load_gptq");

    // CUDA Marlin path.
    let cuda_linear = GptqLinear::<CudaBackend>::from_raw(
        &syn.qweight,
        &syn.scales,
        &syn.qzeros,
        None,
        None,
        syn.bits,
        syn.group_size,
        syn.k,
        syn.n,
    )
    .expect("CUDA load_gptq (Marlin repack + upload)");

    let m = 2;
    let input_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.003).cos()).collect();
    let input_f16: Vec<f32> = input_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

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
    let rel_err = max_diff
        / cpu_out
            .iter()
            .map(|x| x.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
    eprintln!("CUDA↔CPU GPTQ: max|diff|={max_diff:.4}, rel={rel_err:.4}");
    // fp16 GEMM typical relative error < 1% over small dims.
    assert!(
        rel_err < 0.05,
        "GPTQ CUDA/CPU mismatch too large: rel_err={rel_err}"
    );
}

/// CUDA Marlin desc_act=true path vs CPU reference using g_idx scale lookup.
///
/// This is the smallest paid-GPU diagnostic for the W2 Gemma3-27B failure:
/// Gemma3's GPTQ checkpoint is desc_act=true/static_groups=false, and its
/// first-layer residuals explode before final logits become all-NaN.
#[cfg(feature = "cuda")]
fn run_cuda_desc_act_vs_cpu_reference_shape(label: &str, k: usize, n: usize, m: usize, seed: u64) {
    use ferrum_kernels::backend::cuda::CudaBackend;
    use half::f16;

    let gs = 128;
    let syn = make_synthetic_symmetric(k, n, gs, seed);
    let g_idx = make_desc_act_g_idx(k, gs);

    let ref_w = dequant_reference_with_g_idx(&syn, &g_idx);
    let input_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0041).sin()).collect();
    let input_f16: Vec<f32> = input_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let mut cpu_out = vec![0.0f32; m * n];
    <CpuBackend as Backend>::gemm(&mut (), &input_f16, &ref_w, &mut cpu_out, m, n, k);

    let cuda_linear = GptqLinear::<CudaBackend>::from_raw(
        &syn.qweight,
        &syn.scales,
        &syn.qzeros,
        Some(&g_idx),
        None,
        syn.bits,
        syn.group_size,
        syn.k,
        syn.n,
    )
    .expect("CUDA load_gptq desc_act (Marlin repack + upload)");

    let mut cuda_ctx = <CudaBackend as Backend>::new_context();
    let input_dev = CudaBackend::from_slice(&input_f32);
    let mut out_dev = CudaBackend::alloc(m * n);
    cuda_linear.forward(&mut cuda_ctx, &input_dev, &mut out_dev, m);
    <CudaBackend as Backend>::sync(&mut cuda_ctx);
    let cuda_out = CudaBackend::to_vec(&out_dev, m * n);

    let nonfinite = cuda_out.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        nonfinite, 0,
        "CUDA desc_act output contains non-finite values"
    );

    let max_diff = cpu_out
        .iter()
        .zip(&cuda_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let rel_err = max_diff
        / cpu_out
            .iter()
            .map(|x| x.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
    eprintln!("CUDA↔CPU GPTQ desc_act ({label}): max|diff|={max_diff:.4}, rel={rel_err:.4}");
    assert!(
        rel_err < 0.05,
        "GPTQ CUDA desc_act/CPU mismatch too large for {label}: rel_err={rel_err}"
    );
}

#[cfg(feature = "cuda")]
fn read_f32_bin(path: &std::path::Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        raw.len() % 4,
        0,
        "{} byte length is not f32-aligned",
        path.display()
    );
    raw.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

#[cfg(feature = "cuda")]
fn run_real_gptq_prefix_vs_cpu_reference(label: &str, prefix: &str, input_path: &std::path::Path) {
    use ferrum_kernels::backend::cuda::CudaBackend;
    use ferrum_quantization::NativeSafetensorsLoader;
    use half::f16;

    let model_dir = std::env::var("FERRUM_GPTQ_PARITY_MODEL_DIR")
        .expect("set FERRUM_GPTQ_PARITY_MODEL_DIR to a GPTQ safetensors snapshot");
    let loader =
        NativeSafetensorsLoader::<CudaBackend>::open(&model_dir).expect("open GPTQ model dir");
    let qcfg = loader
        .quant_config_ref()
        .expect("quantize_config.json required for real GPTQ parity")
        .clone();
    let (qweight, scales, qzeros, g_idx, k, n) = loader
        .read_gptq_raw(prefix)
        .expect("read real GPTQ tensors");
    let g_idx = g_idx.expect("real desc_act GPTQ tensor must include g_idx");
    let input_f32 = read_f32_bin(input_path);
    assert_eq!(
        input_f32.len() % k,
        0,
        "{} input len {} is not divisible by K={k}",
        input_path.display(),
        input_f32.len()
    );
    let m = input_f32.len() / k;
    let input_f16: Vec<f32> = input_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let syn = SyntheticGptq {
        k,
        n,
        bits: qcfg.bits,
        group_size: qcfg.group_size,
        qweight: qweight.clone(),
        scales: scales.clone(),
        qzeros: qzeros.clone(),
    };
    let ref_w = dequant_reference_with_g_idx(&syn, &g_idx);
    let mut cpu_out = vec![0.0f32; m * n];
    <CpuBackend as Backend>::gemm(&mut (), &input_f16, &ref_w, &mut cpu_out, m, n, k);

    let cuda_linear = GptqLinear::<CudaBackend>::from_raw(
        &qweight,
        &scales,
        &qzeros,
        Some(&g_idx),
        None,
        qcfg.bits,
        qcfg.group_size,
        k,
        n,
    )
    .expect("CUDA load real GPTQ prefix");

    let mut cuda_ctx = <CudaBackend as Backend>::new_context();
    let input_dev = CudaBackend::from_slice(&input_f32);
    let mut out_dev = CudaBackend::alloc(m * n);
    cuda_linear.forward(&mut cuda_ctx, &input_dev, &mut out_dev, m);
    <CudaBackend as Backend>::sync(&mut cuda_ctx);
    let cuda_out = CudaBackend::to_vec(&out_dev, m * n);

    let cuda_nonfinite = cuda_out.iter().filter(|x| !x.is_finite()).count();
    let cpu_nonfinite = cpu_out.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(cpu_nonfinite, 0, "CPU reference contains non-finite values");
    assert_eq!(cuda_nonfinite, 0, "CUDA output contains non-finite values");

    let max_diff = cpu_out
        .iter()
        .zip(&cuda_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let cuda_max = cuda_out.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let rel_err = max_diff / cpu_max.max(1e-6);
    eprintln!(
        "CUDA↔CPU real GPTQ desc_act ({label}): M={m} K={k} N={n} \
         cpu_max={cpu_max:.4} cuda_max={cuda_max:.4} max|diff|={max_diff:.4} rel={rel_err:.4}"
    );
    assert!(
        rel_err < 0.05,
        "real GPTQ CUDA desc_act/CPU mismatch too large for {label}: rel_err={rel_err}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_desc_act_vs_cpu_reference() {
    run_cuda_desc_act_vs_cpu_reference_shape("small", 512, 256, 3, 0xD35C_AC7);
}

/// Same synthetic desc_act parity as `cuda_desc_act_vs_cpu_reference`,
/// but with Gemma3-27B q_proj's real GPTQ dimensions from
/// `circulus/gemma-3-27b-it-gptq`:
///   g_idx [5376], qzeros [42, 512] => K=5376, N=4096, group_size=128.
/// This catches shape/tile bugs that the tiny diagnostic can miss.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_desc_act_gemma3_qproj_shape_vs_cpu_reference() {
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-q_proj K5376 N4096",
        5376,
        4096,
        2,
        0x6E33_A027,
    );
}

/// Same diagnostic over the other layer0 attention projection shapes observed
/// in `circulus/gemma-3-27b-it-gptq`.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_desc_act_gemma3_attention_shapes_vs_cpu_reference() {
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-k_proj K5376 N2048",
        5376,
        2048,
        2,
        0x6E33_A02B,
    );
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-v_proj K5376 N2048",
        5376,
        2048,
        2,
        0x6E33_A02C,
    );
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-o_proj K4096 N5376",
        4096,
        5376,
        2,
        0x6E33_A02D,
    );
}

/// Same diagnostic over Gemma3-27B layer0 MLP projection shapes. W2 CUDA
/// evidence currently shows finite attention parity but large layer0 FFN
/// amplification before layer8 overflows, so keep this as a narrow native-CUDA
/// diagnostic before running another product smoke.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_desc_act_gemma3_mlp_shapes_vs_cpu_reference() {
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-gate_proj K5376 N21504",
        5376,
        21504,
        2,
        0x6E33_A031,
    );
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-up_proj K5376 N21504",
        5376,
        21504,
        2,
        0x6E33_A032,
    );
    run_cuda_desc_act_vs_cpu_reference_shape(
        "gemma3-down_proj K21504 N5376",
        21504,
        5376,
        2,
        0x6E33_A033,
    );
}

/// Real-tensor version of the MLP diagnostic. Requires:
///
/// - `FERRUM_GPTQ_PARITY_MODEL_DIR`: local HF snapshot directory for
///   `circulus/gemma-3-27b-it-gptq`.
/// - `FERRUM_GPTQ_PARITY_DUMP_DIR`: optional op-dump dir containing
///   `layer_00_pre_mlp_norm.bin` and `layer_00_act_mul.bin`.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_real_gemma3_layer0_mlp_vs_cpu_reference() {
    let dump_dir = std::env::var("FERRUM_GPTQ_PARITY_DUMP_DIR")
        .unwrap_or_else(|_| "/workspace/w2/gates_early/op_dump_layer0".to_string());
    let dump_dir = std::path::PathBuf::from(dump_dir);
    let pre_mlp = dump_dir.join("layer_00_pre_mlp_norm.bin");
    let act_mul = dump_dir.join("layer_00_act_mul.bin");

    run_real_gptq_prefix_vs_cpu_reference(
        "layer0 gate_proj",
        "model.layers.0.mlp.gate_proj",
        &pre_mlp,
    );
    run_real_gptq_prefix_vs_cpu_reference("layer0 up_proj", "model.layers.0.mlp.up_proj", &pre_mlp);
    run_real_gptq_prefix_vs_cpu_reference(
        "layer0 down_proj",
        "model.layers.0.mlp.down_proj",
        &act_mul,
    );
}

/// CUDA offset GEMM parity vs per-expert dedicated GptqLinear.
/// Builds N synthetic experts as a stacked store, runs each via
/// `gemm_gptq_with_offset` on the stacked tile, and compares against
/// the per-expert dedicated GptqLinear (which uses the unstrided
/// `marlin_gemm`). Catches workspace mutex aliasing between experts —
/// if the offset variant's per-expert workspace ranges overlap, this
/// test will produce wrong results or hang.
///
/// Marlin tile constraints force k % 128 == 0, n % 64 == 0 per expert.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_stacked_offset_vs_per_expert() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    use ferrum_kernels::backend::{Backend, BackendMoeFused, BackendPagedKv, BackendQuantMarlin};

    let k = 256;
    let n_per = 128;
    let gs = 128;
    let num_experts = 4;

    // Build N independent experts.
    let experts: Vec<SyntheticGptq> = (0..num_experts)
        .map(|e| make_synthetic(k, n_per, gs, 0xCAFE0000 + e as u64))
        .collect();

    // Per-expert dedicated GptqLinears (each gets its own MarlinWeight).
    let per_expert: Vec<GptqLinear<CudaBackend>> = experts
        .iter()
        .map(|syn| {
            GptqLinear::<CudaBackend>::from_raw(
                &syn.qweight,
                &syn.scales,
                &syn.qzeros,
                None,
                None,
                syn.bits,
                syn.group_size,
                syn.k,
                syn.n,
            )
            .expect("per-expert load_gptq")
        })
        .collect();

    // Stacked store: use the new per-expert-repack-then-concat API.
    // Each expert's packed bytes are CONTIGUOUS in the resulting
    // store, so offset GEMM dispatches via pointer offset alone.
    let qw_refs: Vec<&[i32]> = experts.iter().map(|e| e.qweight.as_slice()).collect();
    let sc_refs: Vec<&[f32]> = experts.iter().map(|e| e.scales.as_slice()).collect();
    let qz_refs: Vec<&[i32]> = experts.iter().map(|e| e.qzeros.as_slice()).collect();
    // Phase C step 4e: load_gptq_stacked now returns the trait-object
    // MarlinExpertStack directly (no intermediate GptqStore type).
    let stacked = <CudaBackend as BackendQuantMarlin>::load_gptq_stacked(
        &qw_refs, &sc_refs, &qz_refs, None, 4, gs, k, n_per,
    )
    .expect("stacked load_gptq");

    let m = 2;
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0027).sin()).collect();
    let mut ctx = <CudaBackend as Backend>::new_context();

    // Run per-expert reference + offset variant for each expert; compare.
    for (e, lin) in per_expert.iter().enumerate() {
        let input_dev = CudaBackend::from_slice(&input);
        let mut ref_out_dev = CudaBackend::alloc(m * n_per);
        lin.forward(&mut ctx, &input_dev, &mut ref_out_dev, m);
        <CudaBackend as Backend>::sync(&mut ctx);
        let ref_out = CudaBackend::to_vec(&ref_out_dev, m * n_per);

        let input_dev_off = CudaBackend::from_slice(&input);
        let mut off_out_dev = CudaBackend::alloc(m * n_per);
        let off_lin = ferrum_quantization::StackedExpertLinear::<CudaBackend>::new(
            stacked.clone(),
            e * n_per,
            n_per,
        )
        .expect("StackedExpertLinear::new");
        ferrum_kernels::Linear::<CudaBackend>::forward(
            &off_lin,
            &mut ctx,
            &input_dev_off,
            &mut off_out_dev,
            m,
        );
        <CudaBackend as Backend>::sync(&mut ctx);
        let off_out = CudaBackend::to_vec(&off_out_dev, m * n_per);

        let max_diff = ref_out
            .iter()
            .zip(&off_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mag = ref_out
            .iter()
            .map(|x| x.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
        let rel = max_diff / mag;
        eprintln!("expert {e}: per-expert vs offset: max|diff|={max_diff:.4} rel={rel:.4}");
        assert!(
            rel < 0.05,
            "CUDA offset GEMM disagrees with per-expert (expert {e}): rel={rel}"
        );
    }
}

/// Stacked-vs-per-expert layout parity (CPU): build two synthetic
/// "experts", run them as (a) two independent `GptqLinear`s, (b) one
/// big stacked `GptqLinear` indexed by `gemm_gptq_with_offset`. Both
/// should produce identical output for each expert's column slice.
///
/// Validates the row-major concat layout used by
/// `NativeSafetensorsLoader::load_stacked_gptq_experts` and the
/// offset arithmetic in `Backend::gemm_gptq_with_offset`.
#[test]
fn cpu_stacked_vs_per_expert_parity() {
    use ferrum_kernels::backend::cpu::CpuBackend;
    use ferrum_kernels::backend::{Backend, BackendQuantMarlin};

    let k = 256;
    let n_per = 128; // per expert
    let gs = 128;
    let num_experts = 4;

    // Make num_experts independent synthetic GPTQ tensors.
    let experts: Vec<SyntheticGptq> = (0..num_experts)
        .map(|e| make_synthetic(k, n_per, gs, 0xA00DBA11 + e as u64))
        .collect();

    // (a) Per-expert path: load each as its own GptqLinear, run
    //     gemm_gptq for each.
    let m = 3;
    let input: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.0017).sin()).collect();
    let mut per_expert_outs: Vec<Vec<f32>> = Vec::with_capacity(num_experts);
    for syn in &experts {
        let lin = GptqLinear::<CpuBackend>::from_raw(
            &syn.qweight,
            &syn.scales,
            &syn.qzeros,
            None,
            None,
            syn.bits,
            syn.group_size,
            syn.k,
            syn.n,
        )
        .expect("CPU load_gptq per-expert");
        let mut out = vec![0.0f32; m * n_per];
        let mut ctx = <CpuBackend as Backend>::new_context();
        lin.forward(&mut ctx, &input, &mut out, m);
        per_expert_outs.push(out);
    }

    // (b) Stacked path: Phase 3e/2 uses the proper
    //     `load_gptq_stacked` API rather than abusing `load_gptq` on
    //     hand-concatenated tensors.
    let qweights: Vec<&[i32]> = experts.iter().map(|s| s.qweight.as_slice()).collect();
    let scales: Vec<&[f32]> = experts.iter().map(|s| s.scales.as_slice()).collect();
    let qzeros: Vec<&[i32]> = experts.iter().map(|s| s.qzeros.as_slice()).collect();
    let stacked_store = <CpuBackend as BackendQuantMarlin>::load_gptq_stacked(
        &qweights, &scales, &qzeros, None, 4, gs, k, n_per,
    )
    .expect("stacked load_gptq_stacked");

    // Phase C step 4e: load_gptq_stacked returns Arc<dyn MarlinExpertStack>
    // directly — `stacked_store` is already the trait object, no separate
    // make_marlin_expert_stack wrap.
    let stacked_stack = stacked_store;
    for (e, ref_out) in per_expert_outs.iter().enumerate() {
        let mut stacked_out = vec![0.0f32; m * n_per];
        let mut ctx = <CpuBackend as Backend>::new_context();
        let stacked_lin = ferrum_quantization::StackedExpertLinear::<CpuBackend>::new(
            stacked_stack.clone(),
            e * n_per,
            n_per,
        )
        .expect("StackedExpertLinear::new");
        ferrum_kernels::Linear::<CpuBackend>::forward(
            &stacked_lin,
            &mut ctx,
            &input,
            &mut stacked_out,
            m,
        );
        let max_diff = ref_out
            .iter()
            .zip(&stacked_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "stacked vs per-expert GPTQ drift on expert {e}: {max_diff}"
        );
    }
}
