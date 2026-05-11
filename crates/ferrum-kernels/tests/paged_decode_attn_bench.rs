//! Paged decode attention microbench at Qwen3-MoE / RTX 4090 shape.
//!
//! Reports per-launch GPU time (us) + achieved DRAM bandwidth (% of peak).
//! Used to decide between (A) tightening the current kernel, (B) porting
//! vLLM PagedAttn V2, or (C) FlashAttention-2 production decode.
//!
//! Run on RTX 4090:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!     --test paged_decode_attn_bench bench_paged_decode_attn \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_kernels::backend::{Backend, BackendPagedKv};
use std::time::Instant;

#[test]
#[ignore]
fn bench_paged_decode_attn() {
    // Qwen3-MoE / -30B-A3B
    const NUM_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;

    // Decode parameters: c=32, kv_len=256 (typical mid-conversation).
    const NUM_SEQS: usize = 32;
    const KV_LEN: usize = 256;
    const MAX_BLOCKS_PER_SEQ: usize = 32; // covers up to 512 tokens; we fill 256
    const N_ITERS: usize = 200;
    const N_WARMUP: usize = 20;

    let ctx_handle = CudaContext::new(0).expect("CUDA context");
    let stream = ctx_handle.default_stream();

    // Q: [num_seqs, num_heads, head_dim] (q_len=1)
    let q_n = NUM_SEQS * NUM_HEADS * HEAD_DIM;
    let q_data: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.0017).sin()).collect();
    let q = CudaBackend::from_slice(&q_data);

    // KV pool: [total_blocks, num_kv_heads, block_size, head_dim] f16
    // total_blocks = num_seqs * max_blocks_per_seq (over-alloc)
    let total_blocks = NUM_SEQS * MAX_BLOCKS_PER_SEQ;
    let kv_n = total_blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM;
    let k_data: Vec<f32> = (0..kv_n).map(|i| ((i as f32) * 0.0011).cos()).collect();
    let v_data: Vec<f32> = (0..kv_n).map(|i| ((i as f32) * 0.0009).sin()).collect();
    let k_pool = CudaBackend::from_slice(&k_data);
    let v_pool = CudaBackend::from_slice(&v_data);

    // block_tables: [num_seqs, max_blocks_per_seq] i32
    let mut bt_host: Vec<i32> = Vec::with_capacity(NUM_SEQS * MAX_BLOCKS_PER_SEQ);
    for s in 0..NUM_SEQS {
        for b in 0..MAX_BLOCKS_PER_SEQ {
            bt_host.push((s * MAX_BLOCKS_PER_SEQ + b) as i32);
        }
    }
    let block_tables_i32 = stream.clone_htod(&bt_host).unwrap();

    // valid_kv_lens: [num_seqs] i32, all = KV_LEN
    let kvl_host = vec![KV_LEN as i32; NUM_SEQS];
    let valid_kv_lens_i32 = stream.clone_htod(&kvl_host).unwrap();

    // Reinterpret i32 device buffers as f16 (Self::Buffer) for the trait API.
    // Wrap in CudaBuf::F16 — the trait expects &Self::Buffer = &CudaBuf;
    // downstream forwarders extract via .as_f16() and the underlying
    // CudaSlice<f16> view aliases the i32 bytes (legacy type-tunnel kept
    // until Phase B-3 migrates these to typed CudaBuf::I32 storage).
    use cudarc::driver::sys::CUdeviceptr;
    use cudarc::driver::CudaSlice;
    use cudarc::driver::DevicePtr;
    use ferrum_kernels::backend::CudaBuf;
    use half::f16;
    let (bt_ptr, _g0) = block_tables_i32.device_ptr(&stream);
    let (kvl_ptr, _g1) = valid_kv_lens_i32.device_ptr(&stream);
    let bt_f16_inner: CudaSlice<f16> =
        unsafe { stream.upgrade_device_ptr(bt_ptr as CUdeviceptr, 0) };
    let kvl_f16_inner: CudaSlice<f16> =
        unsafe { stream.upgrade_device_ptr(kvl_ptr as CUdeviceptr, 0) };
    let bt_f16 = CudaBuf::from_f16(bt_f16_inner);
    let kvl_f16 = CudaBuf::from_f16(kvl_f16_inner);

    let mut out_dev = CudaBackend::alloc(NUM_SEQS * NUM_HEADS * HEAD_DIM);
    let mut ctx = <CudaBackend as Backend>::new_context();

    // Warmup
    for _ in 0..N_WARMUP {
        <CudaBackend as BackendPagedKv>::paged_decode_attention(
            &mut ctx,
            &q,
            &k_pool,
            &v_pool,
            &mut out_dev,
            &bt_f16,
            &kvl_f16,
            NUM_SEQS,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            MAX_BLOCKS_PER_SEQ,
            1, // q_len = 1
        )
        .unwrap();
    }
    <CudaBackend as Backend>::sync(&mut ctx);

    // Measured run
    let t0 = Instant::now();
    for _ in 0..N_ITERS {
        <CudaBackend as BackendPagedKv>::paged_decode_attention(
            &mut ctx,
            &q,
            &k_pool,
            &v_pool,
            &mut out_dev,
            &bt_f16,
            &kvl_f16,
            NUM_SEQS,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            MAX_BLOCKS_PER_SEQ,
            1,
        )
        .unwrap();
    }
    <CudaBackend as Backend>::sync(&mut ctx);
    let elapsed = t0.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / N_ITERS as f64;

    // KV bytes per iter: each query attends to KV_LEN tokens of K + V.
    // GQA — kv heads are shared across query heads, so each kv head's
    // (K, V) block range is read num_q_heads/num_kv_heads times. We bound
    // by the read-once layout (best case) and amplified (worst case).
    let kv_bytes_optimal: u64 = (NUM_SEQS * NUM_KV_HEADS * 2 * KV_LEN * HEAD_DIM * 2) as u64;
    let kv_bytes_amplified: u64 = (NUM_SEQS * NUM_HEADS * 2 * KV_LEN * HEAD_DIM * 2) as u64;
    let bw_optimal_gbs = kv_bytes_optimal as f64 / (per_iter_us * 1e3);
    let bw_amplified_gbs = kv_bytes_amplified as f64 / (per_iter_us * 1e3);
    // RTX 4090 peak DRAM: 1008 GB/s
    const PEAK_GBS: f64 = 1008.0;

    eprintln!("── paged_decode_attn @ Qwen3-MoE c={NUM_SEQS} kv_len={KV_LEN} ──");
    eprintln!(
        "  shape: heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} dim={HEAD_DIM} block={BLOCK_SIZE}"
    );
    eprintln!(
        "  per-launch: {per_iter_us:.1} µs ({} iters in {:.1} ms)",
        N_ITERS,
        elapsed.as_secs_f64() * 1000.0
    );
    eprintln!(
        "  KV bytes optimal:    {:.1} MB  → BW {:.0} GB/s  ({:.1}% of {:.0} peak)",
        kv_bytes_optimal as f64 / 1e6,
        bw_optimal_gbs,
        bw_optimal_gbs / PEAK_GBS * 100.0,
        PEAK_GBS,
    );
    eprintln!(
        "  KV bytes amplified:  {:.1} MB  → BW {:.0} GB/s  ({:.1}% of {:.0} peak)",
        kv_bytes_amplified as f64 / 1e6,
        bw_amplified_gbs,
        bw_amplified_gbs / PEAK_GBS * 100.0,
        PEAK_GBS,
    );

    // Per-forward attention budget at 48 layers:
    let attn_per_forward_us = per_iter_us * 48.0;
    eprintln!(
        "  per-forward (48 layers): {:.2} ms",
        attn_per_forward_us / 1e3
    );

    std::mem::forget(bt_f16);
    std::mem::forget(kvl_f16);
}
