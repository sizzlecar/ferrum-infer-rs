//! Stage 15d-pretest: triton-rs `launch_fused_moe_w4a16_triton` matches
//! a CPU reference for a small synthetic 4-expert problem.
//!
//! The kernel reads expert id from `expert_ids[pid_m]`, gathers M-axis
//! input rows via `sorted_token_ids`, and produces the same dequant+
//! GEMM result as a per-expert CPU implementation would.
//!
//! Run on RTX 4090:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!     --test triton_fused_moe_eq -- --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use cudarc::driver::{CudaContext, CudaSlice};
use ferrum_kernels::triton_fused_moe::{
    fn_name, launch_fused_moe_w4a16_triton, TritonStackedGptqWeight, BM, FUSED_MOE_W4A16_PTX,
};

const NUM_EXPERTS: usize = 4;
const TOKENS_PER_EXPERT: [usize; NUM_EXPERTS] = [16, 8, 12, 4];
const K: usize = 256;
const N: usize = 64;
const G: usize = 64; // group_size

fn build_one_expert(seed: u64) -> (Vec<i32>, Vec<half::f16>, Vec<i32>) {
    use half::f16;
    let mut s = seed;
    let mut rng_u32 = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };

    let w_int: Vec<u8> = (0..K * N).map(|_| (rng_u32() & 0xF) as u8).collect();
    let mut qw: Vec<i32> = vec![0; (K / 8) * N];
    for pk in 0..K / 8 {
        for n in 0..N {
            let mut packed: u32 = 0;
            for i in 0..8 {
                let v = w_int[(pk * 8 + i) * N + n] as u32;
                packed |= v << (i * 4);
            }
            qw[pk * N + n] = packed as i32;
        }
    }

    let scales: Vec<f16> = (0..(K / G) * N)
        .map(|_| f16::from_f32(0.01 + ((rng_u32() & 0xFF) as f32) * 0.001))
        .collect();

    let z_int: Vec<u8> = (0..(K / G) * N).map(|_| (rng_u32() & 0xF) as u8).collect();
    let mut qz: Vec<i32> = vec![0; (K / G) * (N / 8)];
    for kg in 0..K / G {
        for pn in 0..N / 8 {
            let mut packed: u32 = 0;
            for j in 0..8 {
                let v = z_int[kg * N + pn * 8 + j] as u32;
                packed |= v << (j * 4);
            }
            qz[kg * (N / 8) + pn] = packed as i32;
        }
    }

    (qw, scales, qz)
}

/// Per-expert dequant + GEMM in f32, written to dst rows [row_start..row_start+m_e].
fn cpu_per_expert_gemm(
    a: &[half::f16],
    qw: &[i32],
    scales: &[half::f16],
    qz: &[i32],
    row_start: usize,
    m_e: usize,
    dst: &mut [half::f16],
) {
    use half::f16;
    let mut deq = vec![0f32; K * N];
    for k in 0..K {
        for n in 0..N {
            let pk = k / 8;
            let shift = (k % 8) * 4;
            let qw_v = qw[pk * N + n] as u32;
            let nibble = ((qw_v >> shift) & 0xF) as i32;

            let kg = k / G;
            let pn = n / 8;
            let z_shift = (n % 8) * 4;
            let qz_v = qz[kg * (N / 8) + pn] as u32;
            let zero = (((qz_v >> z_shift) & 0xF) as i32) + 1;

            let scale = scales[kg * N + n].to_f32();
            deq[k * N + n] = (nibble - zero) as f32 * scale;
        }
    }
    for m in 0..m_e {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += a[(row_start + m) * K + k].to_f32() * deq[k * N + n];
            }
            dst[(row_start + m) * N + n] = f16::from_f32(acc);
        }
    }
}

#[test]
fn triton_fused_moe_matches_cpu_per_expert() {
    use half::f16;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Build per-expert tensors (still as f16 scales — caller-side type
    // matches what `load_stacked_gptq_raw` expects from f32 scales, but
    // here we have f16 already; use the alternative loader path below).
    let mut qw_per: Vec<Vec<i32>> = Vec::new();
    let mut sc_per_f16: Vec<Vec<f16>> = Vec::new();
    let mut qz_per: Vec<Vec<i32>> = Vec::new();
    for e in 0..NUM_EXPERTS {
        let (qw, sc, qz) = build_one_expert(0xCAFE_0000 + e as u64);
        qw_per.push(qw);
        sc_per_f16.push(sc);
        qz_per.push(qz);
    }

    // Pre-gathered A laid out by expert order.
    let total_tokens: usize = TOKENS_PER_EXPERT.iter().sum();
    let a: Vec<f16> = (0..total_tokens * K)
        .map(|i| f16::from_f32(((i as f32) * 0.0017).sin()))
        .collect();

    // ── CPU reference: per-expert GEMM into c_ref ──
    let mut c_ref = vec![f16::from_f32(0.0); total_tokens * N];
    let mut row_start = 0usize;
    for e in 0..NUM_EXPERTS {
        let m_e = TOKENS_PER_EXPERT[e];
        cpu_per_expert_gemm(
            &a,
            &qw_per[e],
            &sc_per_f16[e],
            &qz_per[e],
            row_start,
            m_e,
            &mut c_ref,
        );
        row_start += m_e;
    }

    // ── GPU path: stack tensors + build sorted_token_ids/expert_ids ──
    let mut qw_flat: Vec<i32> = Vec::new();
    for v in &qw_per {
        qw_flat.extend_from_slice(v);
    }
    let mut sc_flat: Vec<f16> = Vec::new();
    for v in &sc_per_f16 {
        sc_flat.extend_from_slice(v);
    }
    let mut qz_flat: Vec<i32> = Vec::new();
    for v in &qz_per {
        qz_flat.extend_from_slice(v);
    }

    let qw_dev: CudaSlice<i32> = stream.clone_htod(&qw_flat).unwrap();
    let sc_dev: CudaSlice<f16> = stream.clone_htod(&sc_flat).unwrap();
    let qz_dev: CudaSlice<i32> = stream.clone_htod(&qz_flat).unwrap();
    let weight = TritonStackedGptqWeight {
        qweight: qw_dev,
        scales: sc_dev,
        qzeros: qz_dev,
        num_experts: NUM_EXPERTS,
        k: K,
        n: N,
        group_size: G as i32,
    };

    // sorted_token_ids: pad each expert's m_e to BM=16. Sentinel = total_tokens.
    let mb = BM as usize;
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_ids: Vec<i32> = Vec::new();
    let mut row_start = 0usize;
    for e in 0..NUM_EXPERTS {
        let m_e = TOKENS_PER_EXPERT[e];
        let pe = ((m_e + mb - 1) / mb) * mb;
        for i in 0..m_e {
            sorted_token_ids.push((row_start + i) as i32);
        }
        for _ in m_e..pe {
            sorted_token_ids.push(total_tokens as i32); // sentinel
        }
        for _ in 0..(pe / mb) {
            expert_ids.push(e as i32);
        }
        row_start += m_e;
    }
    let num_padded = sorted_token_ids.len() as i32;
    let st_dev: CudaSlice<i32> = stream.clone_htod(&sorted_token_ids).unwrap();
    let eid_dev: CudaSlice<i32> = stream.clone_htod(&expert_ids).unwrap();

    let a_dev: CudaSlice<f16> = stream.clone_htod(&a).unwrap();
    let mut c_dev: CudaSlice<f16> = stream.alloc_zeros::<f16>(total_tokens * N).unwrap();

    // Load the kernel module + function.
    let func = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(
            FUSED_MOE_W4A16_PTX.to_string(),
        ))
        .unwrap()
        .load_function(fn_name())
        .unwrap();

    launch_fused_moe_w4a16_triton(
        &stream,
        &func,
        &a_dev,
        &weight,
        &mut c_dev,
        &st_dev,
        &eid_dev,
        num_padded,
        total_tokens as i32,
    )
    .expect("triton fused_moe launch");
    stream.synchronize().expect("sync");

    let c_gpu: Vec<f16> = stream.memcpy_dtov(&c_dev).unwrap();

    // Per-expert error breakdown — first pass diagnostic, no assert,
    // surfaces all 4 expert results so we can see error pattern.
    let mut row_start = 0usize;
    let mut all_max_rel = 0f32;
    let mut all_max_abs = 0f32;
    for e in 0..NUM_EXPERTS {
        let m_e = TOKENS_PER_EXPERT[e];
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut sum_ref = 0f32;
        let mut sum_gpu = 0f32;
        for m in 0..m_e {
            for n in 0..N {
                let g = c_gpu[(row_start + m) * N + n].to_f32();
                let r = c_ref[(row_start + m) * N + n].to_f32();
                let abs = (g - r).abs();
                let rel = abs / r.abs().max(1e-3);
                if abs > max_abs {
                    max_abs = abs;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
                sum_ref += r.abs();
                sum_gpu += g.abs();
            }
        }
        let avg_ref = sum_ref / (m_e * N) as f32;
        let avg_gpu = sum_gpu / (m_e * N) as f32;
        eprintln!(
            "expert {e}: m_e={m_e} row_start={row_start} max|diff|={max_abs:.4} \
             rel={max_rel:.4} avg|ref|={avg_ref:.3} avg|gpu|={avg_gpu:.3}"
        );
        if max_rel > all_max_rel {
            all_max_rel = max_rel;
        }
        if max_abs > all_max_abs {
            all_max_abs = max_abs;
        }
        row_start += m_e;
    }
    eprintln!(
        "── overall: max|diff|={all_max_abs:.4} max_rel={all_max_rel:.4}"
    );
    // Hard threshold: catch order-of-magnitude wrong layouts (rel ≈ 1).
    assert!(
        all_max_abs < 1.0,
        "triton fused_moe MAJOR mismatch: max|diff|={all_max_abs:.4}"
    );
}
