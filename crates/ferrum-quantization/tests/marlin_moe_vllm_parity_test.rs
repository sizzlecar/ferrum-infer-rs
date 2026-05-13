//! Stage 14d — vLLM marlin_moe_wna16 vs per-expert parity (CUDA only).
//!
//! Adapter test: feed both paths the same synthetic 4-expert problem and
//! compare per-expert output rows in the pre-gathered C buffer.
//!
//! Post Phase B-D: BOTH paths now route through the
//! `MarlinExpertStack` trait — `make_expert_linear` for the per-expert
//! reference, `gemm_phase_vllm` for the fused vLLM kernel. The stack
//! is one and the same (`load_gptq_stacked` returns the trait object).
//! The "two formats" the old test set up (IST-DASLab via load_gptq_stacked
//! + vLLM Marlin via load_stacked_gptq_vllm_marlin) collapsed into one
//! when `gemm_phase_vllm` was hoisted onto MarlinExpertStack.
//!
//! Run on RTX 4090:
//!   cargo test -p ferrum-quantization --features cuda,vllm-moe-marlin --release \
//!     --test marlin_moe_vllm_parity_test cuda_marlin_moe_vllm \
//!     -- --ignored --nocapture

#[cfg(all(feature = "cuda", feature = "vllm-moe-marlin"))]
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

#[cfg(all(feature = "cuda", feature = "vllm-moe-marlin"))]
#[test]
#[ignore]
fn cuda_marlin_moe_vllm_vs_per_expert() {
    use ferrum_kernels::backend::cuda::CudaBackend;
    use ferrum_kernels::backend::{Backend, BackendQuantMarlin};

    let k: usize = 256;
    let n_per: usize = 256;
    let gs: usize = 128;
    let num_experts: usize = 4;

    let mut experts: Vec<synth::SyntheticGptq> = (0..num_experts)
        .map(|e| synth::make(k, n_per, gs, 0xCAFE0000 + e as u64))
        .collect();

    // Optional: replace all per-channel scales with a single constant.
    // Defeats any scale-permutation difference between paths.
    if std::env::var("FERRUM_PARITY_UNIT_SCALES").is_ok() {
        eprintln!("DEBUG: forcing all scales = 0.05");
        for ex in experts.iter_mut() {
            for s in ex.scales.iter_mut() {
                *s = 0.05;
            }
        }
    }

    // Pre-gathered layout: each expert's rows contiguous.
    let tokens_per_expert: Vec<usize> = vec![16, 8, 12, 4];
    let total_tokens: usize = tokens_per_expert.iter().sum();
    let mut a_row_offsets: Vec<usize> = Vec::with_capacity(num_experts);
    {
        let mut acc = 0usize;
        for &m_e in &tokens_per_expert {
            a_row_offsets.push(acc);
            acc += m_e;
        }
    }

    // Single stacked store: post Phase B-D the trait-object stack
    // exposes both `make_expert_linear` (per-expert ref path) and
    // `gemm_phase_vllm` (fused vLLM marlin_moe_wna16 path). No need
    // for two separate weight formats anymore.
    let qw_refs: Vec<&[i32]> = experts.iter().map(|e| e.qweight.as_slice()).collect();
    let sc_refs: Vec<&[f32]> = experts.iter().map(|e| e.scales.as_slice()).collect();
    let qz_refs: Vec<&[i32]> = experts.iter().map(|e| e.qzeros.as_slice()).collect();
    let stacked = <CudaBackend as BackendQuantMarlin>::load_gptq_stacked(
        &qw_refs, &sc_refs, &qz_refs, None, 4, gs, k, n_per,
    )
    .expect("stacked load_gptq");

    let a_data: Vec<f32> = (0..total_tokens * k)
        .map(|i| ((i as f32) * 0.0027).sin())
        .collect();
    let mut ctx = <CudaBackend as Backend>::new_context();
    let a_dev = CudaBackend::from_slice(&a_data);

    let force_expert0 = std::env::var("FERRUM_PARITY_FORCE_EXPERT0").is_ok();

    // Reference: per-expert via the trait `make_expert_linear`.
    let mut c_ref = vec![0f32; total_tokens * n_per];
    for e in 0..num_experts {
        let m_e = tokens_per_expert[e];
        if m_e == 0 {
            continue;
        }
        let row_start = a_row_offsets[e];
        let a_e: Vec<f32> = a_data[row_start * k..(row_start + m_e) * k].to_vec();
        let a_e_dev = CudaBackend::from_slice(&a_e);
        let mut out_e_dev = CudaBackend::alloc(m_e * n_per);
        // In force-expert0 mode the vLLM path will route every block to
        // expert 0, so the reference must also use expert 0 weights for
        // all rows (same offset for every expert).
        let weight_expert = if force_expert0 { 0 } else { e };
        let view = stacked
            .clone()
            .make_expert_linear(weight_expert * n_per, n_per, None)
            .expect("make_expert_linear");
        view.forward(&mut ctx, &a_e_dev, &mut out_e_dev, m_e);
        <CudaBackend as Backend>::sync(&mut ctx);
        let out_e = CudaBackend::to_vec(&out_e_dev, m_e * n_per);
        c_ref[row_start * n_per..(row_start + m_e) * n_per].copy_from_slice(&out_e);
    }

    // ── vLLM-port path: fed our pre-gathered A as if it were
    // size_m=total_tokens unique inputs with top_k=1. sorted_token_ids
    // = identity so each output tile (16 rows) corresponds to a
    // contiguous chunk of pre-gathered rows. expert_ids built per tile.
    //
    // moe_block_size=16 → BLOCK_M aligned padding. Each expert's m_e
    // is rounded up to next multiple of 16 in the padded layout.
    let moe_block_size: i32 = 16;
    let mb = moe_block_size as usize;
    let mut padded_tokens_per_expert: Vec<usize> = Vec::with_capacity(num_experts);
    let mut padded_offsets: Vec<usize> = Vec::with_capacity(num_experts + 1);
    let mut acc = 0usize;
    for &m_e in &tokens_per_expert {
        padded_offsets.push(acc);
        let pe = m_e.div_ceil(mb) * mb;
        padded_tokens_per_expert.push(pe);
        acc += pe;
    }
    padded_offsets.push(acc);
    let total_padded = acc;
    let total_blocks = total_padded / mb;
    let sentinel = total_tokens as i32;

    // sorted_token_ids[i] = pre-gathered row index OR sentinel for padding.
    let mut sorted_token_ids = vec![sentinel; total_padded];
    for e in 0..num_experts {
        let m_e = tokens_per_expert[e];
        let p_off = padded_offsets[e];
        let real_off = a_row_offsets[e];
        for i in 0..m_e {
            sorted_token_ids[p_off + i] = (real_off + i) as i32;
        }
    }

    // expert_ids[block] = expert index for that 16-row block.
    let mut expert_ids = vec![0i32; total_blocks];
    for e in 0..num_experts {
        let blocks_for_e = padded_tokens_per_expert[e] / mb;
        let block_start = padded_offsets[e] / mb;
        for b in 0..blocks_for_e {
            expert_ids[block_start + b] = e as i32;
        }
    }
    // Optional debug override: route every block to expert 0 to isolate
    // per-expert stride bugs from kernel-config bugs.
    if force_expert0 {
        eprintln!("DEBUG: forcing expert_ids = [0; total_blocks]");
        for b in 0..total_blocks {
            expert_ids[b] = 0;
        }
    }
    let num_tokens_past_padded = vec![total_padded as i32];

    // Allocate zeroed output per vLLM contract: [size_m * top_k, n].
    // The kernel uses atomic-add when c_tmp is None, so caller must
    // pre-zero. `Backend::alloc` doesn't zero; bounce through a zero
    // host buffer (parity test → not perf path → cost doesn't matter).
    let size_m = total_tokens;
    let top_k: usize = 1;
    let mut c_vllm_dev: <CudaBackend as Backend>::Buffer =
        CudaBackend::from_slice(&vec![0f32; size_m * top_k * n_per]);

    // Upload index buffers through the typed-buffer API (Phase B-2).
    let st_buf = CudaBackend::from_slice_typed::<i32>(&sorted_token_ids);
    let eid_buf = CudaBackend::from_slice_typed::<i32>(&expert_ids);
    let npp_buf = CudaBackend::from_slice_typed::<i32>(&num_tokens_past_padded);

    // Zero workspace via the trait method (replaces deleted
    // `BackendQuantMarlin::marlin_zero_stacked_workspace`).
    stacked.zero_workspace(&mut ctx).expect("zero_workspace");

    // Run fused vLLM marlin_moe_wna16 via the trait method.
    stacked
        .gemm_phase_vllm(
            &mut ctx,
            &a_dev,
            &st_buf,
            &eid_buf,
            &npp_buf,
            &mut c_vllm_dev,
            size_m,
            mb,
            top_k,
        )
        .expect("gemm_phase_vllm");
    <CudaBackend as Backend>::sync(&mut ctx);
    let c_vllm = CudaBackend::to_vec(&c_vllm_dev, size_m * top_k * n_per);

    // Compare per-expert. Soft-mode first: print every expert's diagnostics
    // even if expert 0 fails, then assert at the end.
    let mut max_rel_overall = 0f32;
    for e in 0..num_experts {
        let m_e = tokens_per_expert[e];
        if m_e == 0 {
            continue;
        }
        let row_start = a_row_offsets[e];
        let slice_ref = &c_ref[row_start * n_per..(row_start + m_e) * n_per];
        let slice_vllm = &c_vllm[row_start * n_per..(row_start + m_e) * n_per];
        let max_diff = slice_ref
            .iter()
            .zip(slice_vllm)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mag = slice_ref
            .iter()
            .map(|x| x.abs())
            .fold(0f32, f32::max)
            .max(1e-6);
        let rel = max_diff / mag;
        max_rel_overall = max_rel_overall.max(rel);

        let avg_ref: f32 = slice_ref.iter().map(|x| x.abs()).sum::<f32>() / slice_ref.len() as f32;
        let avg_vllm: f32 =
            slice_vllm.iter().map(|x| x.abs()).sum::<f32>() / slice_vllm.len() as f32;

        // Sorted-set test: are ref/vllm the same multiset (just permuted)?
        let mut ref_sorted: Vec<f32> = slice_ref.to_vec();
        let mut vllm_sorted: Vec<f32> = slice_vllm.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vllm_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let max_sorted_diff = ref_sorted
            .iter()
            .zip(vllm_sorted.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        eprintln!(
            "expert {e}: m_e={m_e}, row_start={row_start}, \
             max|diff|={max_diff:.4} rel={rel:.4} sorted-diff={max_sorted_diff:.4} \
             avg|ref|={avg_ref:.4} avg|vllm|={avg_vllm:.4}"
        );
        eprintln!("  ref[0..6] = {:?}", &slice_ref[..6.min(slice_ref.len())]);
        eprintln!(
            "  vllm[0..6] = {:?}",
            &slice_vllm[..6.min(slice_vllm.len())]
        );
    }
    assert!(
        max_rel_overall < 1e-2,
        "vLLM marlin_moe_wna16 disagrees: max rel across experts = {max_rel_overall}"
    );
}
