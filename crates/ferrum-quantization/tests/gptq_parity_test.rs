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
    let stacked = <CudaBackend as BackendQuantMarlin>::load_gptq_stacked(
        &qw_refs, &sc_refs, &qz_refs, None, 4, gs, k, n_per,
    )
    .expect("stacked load_gptq");

    let m = 2;
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0027).sin()).collect();
    let mut ctx = <CudaBackend as Backend>::new_context();

    // Run per-expert reference + offset variant for each expert; compare.
    let stacked_arc = std::sync::Arc::new(stacked);
    for (e, lin) in per_expert.iter().enumerate() {
        let input_dev = CudaBackend::from_slice(&input);
        let mut ref_out_dev = CudaBackend::alloc(m * n_per);
        lin.forward(&mut ctx, &input_dev, &mut ref_out_dev, m);
        <CudaBackend as Backend>::sync(&mut ctx);
        let ref_out = CudaBackend::to_vec(&ref_out_dev, m * n_per);

        let input_dev_off = CudaBackend::from_slice(&input);
        let mut off_out_dev = CudaBackend::alloc(m * n_per);
        // Phase C step 4b: StackedExpertLinear takes a MarlinExpertStack
        // trait object. Wrap the raw GptqStoreCuda first.
        let off_stack = <CudaBackend as ferrum_kernels::backend::BackendQuantMarlin>::make_marlin_expert_stack(
            stacked_arc.clone(),
            num_experts,
            n_per,
            k,
        )
        .expect("make_marlin_expert_stack");
        let off_lin = ferrum_quantization::StackedExpertLinear::<CudaBackend>::new(
            off_stack,
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

    // Slice-and-compare: for each expert, the stacked-store sliced view
    // must equal the per-expert dedicated GEMM. Phase 3e/2: per-expert
    // dispatch goes through `StackedExpertLinear<CpuBackend>::forward`.
    let stacked_arc = std::sync::Arc::new(stacked_store);
    // Phase C step 4b: wrap the raw GptqStore in a MarlinExpertStack
    // trait object first; StackedExpertLinear::new takes the stack
    // directly (no longer routes through B::make_stacked_expert_linear).
    let stacked_stack = <CpuBackend as ferrum_kernels::backend::BackendQuantMarlin>::make_marlin_expert_stack(
        stacked_arc.clone(),
        num_experts,
        n_per,
        k,
    )
    .expect("make_marlin_expert_stack");
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
