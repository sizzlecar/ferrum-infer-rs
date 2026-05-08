//! Triton fused MoE microbench at Qwen3-30B-A3B shape.
//!
//! Compares the Triton fused MoE kernel against a per-expert Marlin
//! reference at the actual production GEMM shapes:
//!   - gate_up: m_avg=2.5 (varying per expert), n=1536, k=2048
//!   - down:    m_avg=2.5, n=2048, k=768
//!
//! With E_active = 100, top_k=8, c=32 → ~256 (token, expert) pairs.
//!
//! Run on RTX 4090:
//!   cargo test -p ferrum-kernels --features cuda,triton-kernels --release \
//!     --test triton_fused_moe_bench -- --ignored --nocapture

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use cudarc::driver::{CudaContext, CudaSlice};
use ferrum_kernels::triton_fused_moe::{
    fn_name, launch_fused_moe_w4a16_triton, TritonStackedGptqWeight, BM, FUSED_MOE_W4A16_PTX,
};
use std::time::Instant;

const NUM_EXPERTS: usize = 100; // active experts in a typical layer at c=32
const K: usize = 2048;
const N_GATE_UP: usize = 1536; // 2 * intermediate_size for Qwen3-MoE
const G: usize = 128;
const TOP_K: i32 = 8;
const SIZE_M: usize = 32; // c=32 unique tokens

const N_ITERS: usize = 10;

fn make_random_gptq_tensor(k: usize, n: usize, seed: u64) -> (Vec<i32>, Vec<half::f16>, Vec<i32>) {
    use half::f16;
    let mut s = seed;
    let mut rng_u32 = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };

    // qweight [K/8, N] random
    let mut qw = vec![0i32; (k / 8) * n];
    for v in qw.iter_mut() {
        *v = rng_u32() as i32;
    }
    // scales [K/G, N]
    let scales: Vec<f16> = (0..(k / G) * n)
        .map(|_| f16::from_f32(0.01 + ((rng_u32() & 0xFF) as f32) * 0.0001))
        .collect();
    // qzeros [K/G, N/8]
    let mut qz = vec![0i32; (k / G) * (n / 8)];
    for v in qz.iter_mut() {
        *v = rng_u32() as i32;
    }
    (qw, scales, qz)
}

#[test]
#[ignore]
fn triton_fused_moe_bench_qwen3_shape() {
    use half::f16;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    eprintln!(
        "─── Triton fused MoE bench (Qwen3-30B-A3B shape) ───\n\
         E_active={NUM_EXPERTS} top_k={TOP_K} K={K} n_gate_up={N_GATE_UP} group_size={G}\n\
         size_m={SIZE_M} (input tokens; total pairs = size_m × top_k = {})",
        SIZE_M * (TOP_K as usize)
    );

    // Build stacked weights: NUM_EXPERTS × (K, N_GATE_UP).
    let mut qw_flat: Vec<i32> = Vec::with_capacity(NUM_EXPERTS * (K / 8) * N_GATE_UP);
    let mut sc_flat: Vec<f16> = Vec::with_capacity(NUM_EXPERTS * (K / G) * N_GATE_UP);
    let mut qz_flat: Vec<i32> = Vec::with_capacity(NUM_EXPERTS * (K / G) * (N_GATE_UP / 8));
    for e in 0..NUM_EXPERTS {
        let (qw, sc, qz) = make_random_gptq_tensor(K, N_GATE_UP, 0xCAFE0000 + e as u64);
        qw_flat.extend_from_slice(&qw);
        sc_flat.extend_from_slice(&sc);
        qz_flat.extend_from_slice(&qz);
    }
    let qw_dev: CudaSlice<i32> = stream.clone_htod(&qw_flat).unwrap();
    let sc_dev: CudaSlice<f16> = stream.clone_htod(&sc_flat).unwrap();
    let qz_dev: CudaSlice<i32> = stream.clone_htod(&qz_flat).unwrap();
    eprintln!(
        "weights uploaded: qw={:.1} MB scales={:.1} MB qzeros={:.1} MB",
        (qw_flat.len() * 4) as f32 / 1024.0 / 1024.0,
        (sc_flat.len() * 2) as f32 / 1024.0 / 1024.0,
        (qz_flat.len() * 4) as f32 / 1024.0 / 1024.0,
    );

    let weight = TritonStackedGptqWeight {
        qweight: qw_dev,
        scales: sc_dev,
        qzeros: qz_dev,
        num_experts: NUM_EXPERTS,
        k: K,
        n: N_GATE_UP,
        group_size: G as i32,
    };

    // sorted_token_ids + expert_ids: simulate the c=32 routing.
    // Each expert gets ~2-3 tokens; the average is total_pairs / num_experts.
    let total_pairs = SIZE_M * (TOP_K as usize);
    let mb = BM as usize;
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_ids: Vec<i32> = Vec::new();
    let mut idx = 0i32;
    for e in 0..NUM_EXPERTS {
        // Distribute pairs roughly evenly. Some experts get 2, some 3.
        let m_e = if e < (total_pairs % NUM_EXPERTS) {
            (total_pairs / NUM_EXPERTS) + 1
        } else {
            total_pairs / NUM_EXPERTS
        };
        for i in 0..m_e {
            sorted_token_ids.push(idx + i as i32);
        }
        idx += m_e as i32;
        // Pad to BM.
        let pad = (mb - (m_e % mb)) % mb;
        for _ in 0..pad {
            sorted_token_ids.push(total_pairs as i32); // sentinel
        }
        // expert_ids: one entry per BM-row tile.
        let blocks_for_e = ((m_e + mb - 1) / mb).max(1);
        for _ in 0..blocks_for_e {
            expert_ids.push(e as i32);
        }
    }
    let num_padded = sorted_token_ids.len() as i32;
    eprintln!("num_padded={num_padded} ({} tiles)", num_padded / mb as i32);

    let st_dev: CudaSlice<i32> = stream.clone_htod(&sorted_token_ids).unwrap();
    let eid_dev: CudaSlice<i32> = stream.clone_htod(&expert_ids).unwrap();

    // Synthetic A: [size_m, K]
    let a: Vec<f16> = (0..SIZE_M * K)
        .map(|i| f16::from_f32(((i as f32) * 0.0017).sin()))
        .collect();
    let a_dev: CudaSlice<f16> = stream.clone_htod(&a).unwrap();
    let mut c_dev: CudaSlice<f16> = stream
        .alloc_zeros::<f16>(SIZE_M * (TOP_K as usize) * N_GATE_UP)
        .unwrap();

    let func = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(
            FUSED_MOE_W4A16_PTX.to_string(),
        ))
        .unwrap()
        .load_function(fn_name())
        .unwrap();

    // Warmup.
    for _ in 0..3 {
        launch_fused_moe_w4a16_triton(
            &stream,
            &func,
            &a_dev,
            &weight,
            &mut c_dev,
            &st_dev,
            &eid_dev,
            num_padded,
            SIZE_M as i32,
        )
        .expect("triton fused_moe warmup");
    }
    stream.synchronize().expect("warmup sync");

    // Bench.
    let t0 = Instant::now();
    for _ in 0..N_ITERS {
        launch_fused_moe_w4a16_triton(
            &stream,
            &func,
            &a_dev,
            &weight,
            &mut c_dev,
            &st_dev,
            &eid_dev,
            num_padded,
            SIZE_M as i32,
        )
        .expect("triton fused_moe");
    }
    stream.synchronize().expect("bench sync");
    let elapsed = t0.elapsed();

    let per_iter_us = elapsed.as_micros() as f64 / N_ITERS as f64;
    eprintln!(
        "── Triton fused MoE: {N_ITERS} iters in {:.1} ms = {:.1} µs/iter",
        elapsed.as_secs_f64() * 1000.0,
        per_iter_us
    );
    eprintln!(
        "Compare to Stage 12.1 fused gate_up at c=32: ~8.8 ms / 48 layers \
         = ~183 µs/layer."
    );
    eprintln!("If Triton ≪ 183 µs, this path is faster than Marlin per-expert.");
}
