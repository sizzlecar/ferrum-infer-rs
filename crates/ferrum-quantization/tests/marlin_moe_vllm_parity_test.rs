//! Stage 14d — vLLM marlin_moe_wna16 vs Stage 12.1 fused parity (CUDA only).
//!
//! Adapter test: feed both paths the same synthetic 4-expert problem and
//! compare per-expert output rows in the pre-gathered C buffer.
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
    use ferrum_kernels::backend::Backend;

    let k: usize = 256;
    let n_per: usize = 256;
    let gs: usize = 128;
    let num_experts: usize = 4;

    let experts: Vec<synth::SyntheticGptq> = (0..num_experts)
        .map(|e| synth::make(k, n_per, gs, 0xCAFE0000 + e as u64))
        .collect();

    // Pre-gathered layout: each expert's rows contiguous.
    let tokens_per_expert: Vec<usize> = vec![16, 8, 12, 4];
    let total_tokens: usize = tokens_per_expert.iter().sum();
    let mut a_row_offsets: Vec<i32> = Vec::with_capacity(num_experts);
    {
        let mut acc = 0usize;
        for &m_e in &tokens_per_expert {
            a_row_offsets.push(acc as i32);
            acc += m_e;
        }
    }

    // Stacked store (production loader path).
    let qw_refs: Vec<&[i32]> = experts.iter().map(|e| e.qweight.as_slice()).collect();
    let sc_refs: Vec<&[f32]> = experts.iter().map(|e| e.scales.as_slice()).collect();
    let qz_refs: Vec<&[i32]> = experts.iter().map(|e| e.qzeros.as_slice()).collect();
    let stacked = <CudaBackend as Backend>::load_gptq_stacked(
        &qw_refs, &sc_refs, &qz_refs, None, 4, gs, k, n_per,
    )
    .expect("stacked load_gptq");
    let mw: &ferrum_kernels::marlin::MarlinWeight = &stacked;

    let a_data: Vec<f32> = (0..total_tokens * k)
        .map(|i| ((i as f32) * 0.0027).sin())
        .collect();
    let mut ctx = <CudaBackend as Backend>::new_context();
    let a_dev = CudaBackend::from_slice(&a_data);

    // Reference: per-expert via existing path.
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
        <CudaBackend as Backend>::gemm_gptq_with_offset(
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
        let real_off = a_row_offsets[e] as usize;
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
    let num_tokens_past_padded = vec![total_padded as i32];

    // Allocate output per vLLM contract: [size_m * top_k, n].
    let size_m = total_tokens;
    let top_k = 1;
    let mut c_vllm_dev = <CudaBackend as Backend>::alloc(size_m * top_k * n_per);

    // Upload index buffers.
    let stream = ctx.stream.clone();
    let st_dev = stream
        .clone_htod(&sorted_token_ids)
        .expect("upload sorted_token_ids");
    let eid_dev = stream.clone_htod(&expert_ids).expect("upload expert_ids");
    let npp_dev = stream
        .clone_htod(&num_tokens_past_padded)
        .expect("upload num_tokens_past_padded");

    let _ = <CudaBackend as Backend>::marlin_zero_stacked_workspace(&mut ctx, &stacked);

    ferrum_kernels::marlin::marlin_gemm_moe_vllm(
        &stream,
        &a_dev,
        mw,
        &mut c_vllm_dev,
        None, // c_tmp
        &st_dev,
        &eid_dev,
        &npp_dev,
        None, // topk_weights
        moe_block_size,
        top_k as i32,
        false, // mul_topk_weights
        false, // is_ep
        size_m as i32,
        n_per as i32,
        k as i32,
    )
    .expect("marlin_gemm_moe_vllm");
    <CudaBackend as Backend>::sync(&mut ctx);
    let c_vllm = CudaBackend::to_vec(&c_vllm_dev, size_m * top_k * n_per);

    // Compare per-expert.
    for e in 0..num_experts {
        let m_e = tokens_per_expert[e];
        if m_e == 0 {
            continue;
        }
        let row_start = a_row_offsets[e] as usize;
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
        eprintln!(
            "expert {e}: m_e={m_e}, row_start={row_start}, \
             per-expert vs vLLM: max|diff|={max_diff:.4} rel={rel:.4}"
        );
        assert!(
            rel < 1e-2,
            "vLLM marlin_moe_wna16 disagrees with per-expert (expert {e}): rel={rel}"
        );
    }
}
