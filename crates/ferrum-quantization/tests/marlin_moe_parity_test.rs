//! Stage 11 — fused MoE Marlin parity test (CUDA only).
//!
//! Builds N synthetic experts, stacks them via `Backend::load_gptq_stacked`
//! (the production loader), then compares:
//!
//!   reference: per-expert `gemm_gptq_with_offset` (existing path, used by
//!              `moe_forward_bucketed` via `moe_gemm_phase_batched`)
//!   fused:     `marlin::marlin_gemm_moe` ONE call covering all experts
//!
//! Both should produce bit-identical (or rel < 1e-4) outputs in each
//! expert's row range of the packed buffer.
//!
//! Run:
//!   cargo test -p ferrum-quantization --features cuda,marlin --release \
//!     --test marlin_moe_parity_test cuda_marlin_moe_fused -- --ignored --nocapture

#[cfg(feature = "cuda")]
mod synth {
    pub fn rnd_u32(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }
    pub fn rnd_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = (rnd_u32(state) & 0x00FF_FFFF) as f32 / 16_777_216.0;
        lo + u * (hi - lo)
    }
    pub struct SyntheticGptq {
        pub k: usize,
        pub n: usize,
        pub group_size: usize,
        pub qweight: Vec<i32>,
        pub scales: Vec<f32>,
        pub qzeros: Vec<i32>,
    }
    pub fn make(k: usize, n: usize, gs: usize, seed: u64) -> SyntheticGptq {
        assert_eq!(k % 8, 0);
        assert_eq!(n % 8, 0);
        assert_eq!(k % gs, 0);
        let mut s = seed;
        let groups = k / gs;
        let mut qweight = vec![0i32; (k / 8) * n];
        for w in qweight.iter_mut() {
            *w = rnd_u32(&mut s) as i32;
        }
        let mut scales = vec![0f32; groups * n];
        for x in scales.iter_mut() {
            *x = rnd_f32(&mut s, 0.01, 0.1);
        }
        let mut qzeros = vec![0i32; groups * (n / 8)];
        for qz in qzeros.iter_mut() {
            let mut word: u32 = 0;
            for bi in 0..8 {
                word |= (rnd_u32(&mut s) & 0xF) << (bi * 4);
            }
            *qz = word as i32;
        }
        SyntheticGptq {
            k,
            n,
            group_size: gs,
            qweight,
            scales,
            qzeros,
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_marlin_moe_fused_vs_per_expert() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    use ferrum_kernels::backend::{Backend, BackendMoeFused, BackendPagedKv, BackendQuantMarlin};

    // Marlin tile constraints: k % 128 == 0, n_per % 64 == 0, n_per % 256 ==
    // 0 for some specs. Use a shape Qwen3-30B-A3B-like: k=2048 hidden,
    // n_per=768 (== 2 * intermediate / 4 in practice ~ 512).
    let k: usize = 256;
    let n_per: usize = 256; // n_per % 256 == 0 satisfies all CALL_IF specs
    let gs: usize = 128;
    let num_experts: usize = 4;

    // Synthetic experts.
    let experts: Vec<synth::SyntheticGptq> = (0..num_experts)
        .map(|e| synth::make(k, n_per, gs, 0xCAFE0000 + e as u64))
        .collect();

    // Per-expert variable m: bucket choice of thread_m_blocks=1 (max m=16).
    let tokens_per_expert: Vec<usize> = vec![16, 8, 12, 4]; // varying m_e
    let total_tokens: usize = tokens_per_expert.iter().sum();
    let mut a_row_offsets: Vec<i32> = Vec::with_capacity(num_experts);
    {
        let mut acc = 0usize;
        for &m_e in &tokens_per_expert {
            a_row_offsets.push(acc as i32);
            acc += m_e;
        }
    }
    let prob_m_bucket: i32 = 16; // bucket-wide max → thread_m_blocks=1

    // Stacked store: this is what production uses.
    let qw_refs: Vec<&[i32]> = experts.iter().map(|e| e.qweight.as_slice()).collect();
    let sc_refs: Vec<&[f32]> = experts.iter().map(|e| e.scales.as_slice()).collect();
    let qz_refs: Vec<&[i32]> = experts.iter().map(|e| e.qzeros.as_slice()).collect();
    let stacked = <CudaBackend as BackendQuantMarlin>::load_gptq_stacked(
        &qw_refs, &sc_refs, &qz_refs, None, 4, gs, k, n_per,
    )
    .expect("stacked load_gptq");

    // Synthetic A_packed [total_tokens, k] f32 (CudaBackend stores f16
    // internally — `from_slice` does the f32→f16 convert).
    let a_data: Vec<f32> = (0..total_tokens * k)
        .map(|i| ((i as f32) * 0.0027).sin())
        .collect();
    let mut ctx = <CudaBackend as Backend>::new_context();
    let a_dev = CudaBackend::from_slice(&a_data);

    // ── Reference: per-expert gemm_gptq_with_offset ──────────────────────
    // Each expert e's input rows = a_dev[a_row_offsets[e] .. + tokens_per_expert[e]],
    // output rows = c_ref[a_row_offsets[e] .. + tokens_per_expert[e]].
    // The Backend gemm_gptq_with_offset takes a single contiguous input
    // slice (not strided), so we extract per-expert sub-buffers.
    let mut c_ref = vec![0f32; total_tokens * n_per];
    for e in 0..num_experts {
        let m_e = tokens_per_expert[e];
        if m_e == 0 {
            continue;
        }
        let row_start = a_row_offsets[e] as usize;
        let a_e: Vec<f32> = a_data[row_start * k..(row_start + m_e) * k].to_vec();
        let a_e_dev = CudaBackend::from_slice(&a_e);
        let mut out_e_dev = CudaBackend::alloc(m_e * n_per);
        <CudaBackend as BackendQuantMarlin>::gemm_gptq_with_offset(
            &mut ctx,
            &a_e_dev,
            &stacked,
            e * n_per,
            n_per,
            &mut out_e_dev,
            m_e,
        )
        .expect("gemm_gptq_with_offset");
        <CudaBackend as Backend>::sync(&mut ctx);
        let out_e = CudaBackend::to_vec(&out_e_dev, m_e * n_per);
        c_ref[row_start * n_per..(row_start + m_e) * n_per].copy_from_slice(&out_e);
    }

    // ── Fused: marlin_gemm_moe ONE call ──────────────────────────────────
    // Need: a_dev (already), MarlinWeight (= stacked when no triton-kernels),
    //       c_dev, a_row_offsets dev, tokens_per_expert dev.
    //
    // Without `triton-kernels` feature, GptqStoreCuda is a transparent
    // alias for MarlinWeight, so `stacked` IS the MarlinWeight directly
    // (see crates/ferrum-kernels/src/backend/cuda.rs:279).
    let mw: &ferrum_kernels::marlin::MarlinWeight = &stacked;
    let mut c_fused_dev = <CudaBackend as Backend>::alloc(total_tokens * n_per);

    // Upload offsets/tokens as i32. Use raw cudarc since the Backend's
    // typed buffer is f16-only.
    let stream = ctx.stream.clone();
    let row_off_dev = stream
        .clone_htod(&a_row_offsets)
        .expect("upload a_row_offsets");
    let tok_dev = stream
        .clone_htod(
            &tokens_per_expert
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        )
        .expect("upload tokens_per_expert");

    // Pre-zero stacked workspace (mirrors production).
    let _ = <CudaBackend as BackendQuantMarlin>::marlin_zero_stacked_workspace(&mut ctx, &stacked);

    ferrum_kernels::marlin::marlin_gemm_moe(
        &stream,
        &a_dev,
        mw,
        &mut c_fused_dev,
        &row_off_dev,
        &tok_dev,
        None, // identity active_expert_ids = [0..num_experts)
        num_experts as i32,
        prob_m_bucket,
        n_per as i32,
        num_experts as i32,
    )
    .expect("marlin_gemm_moe");
    <CudaBackend as Backend>::sync(&mut ctx);
    let c_fused = CudaBackend::to_vec(&c_fused_dev, total_tokens * n_per);

    // Compare per-expert.
    for e in 0..num_experts {
        let m_e = tokens_per_expert[e];
        if m_e == 0 {
            continue;
        }
        let row_start = a_row_offsets[e] as usize;
        let slice_ref = &c_ref[row_start * n_per..(row_start + m_e) * n_per];
        let slice_fused = &c_fused[row_start * n_per..(row_start + m_e) * n_per];
        let max_diff = slice_ref
            .iter()
            .zip(slice_fused)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mag = slice_ref
            .iter()
            .map(|x| x.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
        let rel = max_diff / mag;
        eprintln!(
            "expert {e}: m_e={m_e}, row_start={row_start}, \
             per-expert vs fused: max|diff|={max_diff:.4} rel={rel:.4}"
        );
        assert!(
            rel < 1e-3,
            "fused MoE Marlin disagrees with per-expert (expert {e}): rel={rel}"
        );
    }
}
