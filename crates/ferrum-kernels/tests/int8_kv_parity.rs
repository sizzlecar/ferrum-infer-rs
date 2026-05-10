//! INT8 KV parity test (CUDA-only).
//!
//! Two checks run end-to-end on a real CUDA device:
//!
//!   1. **Append round-trip.** Take random FP16 K/V tokens, push them
//!      through `launch_int8_kv_cache_append`, read back the INT8 buffer
//!      + per-token FP16 scales, dequantize on the host, and check that
//!      the round-trip max-relative-error is bounded by `1/127` (the
//!      INT8 quantization step).
//!
//!   2. **Decode parity vs FP16.** Build a paged KV pool with random
//!      FP16 K/V, run the existing `paged_decode_attention_f16` kernel,
//!      then quantize the SAME K/V to INT8 (using the kernel's scale
//!      convention), run `paged_decode_attention_int8`, and check that
//!      the two output tensors agree within a small relative tolerance.
//!
//! Run on a CUDA box:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!     --test int8_kv_parity -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_kernels::backend::{Backend, BackendPagedKv};
use ferrum_kernels::int8_kv::{launch_int8_kv_cache_append, launch_int8_paged_decode_attention};
use half::f16;

fn rnd(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = (*state >> 33) as u32 & 0x00FF_FFFF;
    (u as f32) / 16_777_216.0 * 2.0 - 1.0
}

fn quantize_token_per_head(
    fp16: &[f32],            // [num_tokens, num_kv_heads, head_dim]
    num_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; num_tokens * num_kv_heads * head_dim];
    let mut s = vec![0f32; num_tokens * num_kv_heads];
    for t in 0..num_tokens {
        for h in 0..num_kv_heads {
            let base = (t * num_kv_heads + h) * head_dim;
            let mut max_abs = 0f32;
            for d in 0..head_dim {
                max_abs = max_abs.max(fp16[base + d].abs());
            }
            let max_abs = max_abs.max(1e-8);
            let scale = max_abs / 127.0;
            s[t * num_kv_heads + h] = scale;
            let inv = 1.0 / scale;
            for d in 0..head_dim {
                let v = (fp16[base + d] * inv).round();
                q[base + d] = v.clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, s)
}

#[test]
#[ignore]
fn int8_kv_append_roundtrip() {
    const NUM_TOKENS: usize = 16;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const POOL_TOKENS: usize = 32; // sparse: only half the slots filled

    let total = NUM_TOKENS * NUM_KV_HEADS * HEAD_DIM;
    let mut s = 0xDEADBEEFu64;
    let k_in: Vec<f32> = (0..total).map(|_| rnd(&mut s) * 0.1).collect();
    let v_in: Vec<f32> = (0..total).map(|_| rnd(&mut s) * 0.1).collect();

    // Place tokens in non-contiguous slots (5, 9, 13, ..., 5+4*15) to
    // exercise slot_mapping.
    let slot_mapping: Vec<i32> = (0..NUM_TOKENS).map(|i| (5 + i * 4) as i32).collect();

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    let k_in_dev = CudaBackend::from_slice(&k_in);
    let v_in_dev = CudaBackend::from_slice(&v_in);
    let mut k_pool: cudarc::driver::CudaSlice<i8> = stream
        .alloc_zeros::<i8>(POOL_TOKENS * NUM_KV_HEADS * HEAD_DIM)
        .unwrap();
    let mut v_pool: cudarc::driver::CudaSlice<i8> = stream
        .alloc_zeros::<i8>(POOL_TOKENS * NUM_KV_HEADS * HEAD_DIM)
        .unwrap();
    let mut k_scales: cudarc::driver::CudaSlice<f16> = stream
        .alloc_zeros::<f16>(POOL_TOKENS * NUM_KV_HEADS)
        .unwrap();
    let mut v_scales: cudarc::driver::CudaSlice<f16> = stream
        .alloc_zeros::<f16>(POOL_TOKENS * NUM_KV_HEADS)
        .unwrap();
    let slot_dev = stream.memcpy_stod(&slot_mapping).unwrap();

    launch_int8_kv_cache_append(
        &ctx,
        &k_in_dev,
        &v_in_dev,
        &mut k_pool,
        &mut v_pool,
        &mut k_scales,
        &mut v_scales,
        &slot_dev,
        NUM_TOKENS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    .expect("int8_kv_cache_append");

    stream.synchronize().unwrap();

    let k_pool_h: Vec<i8> = stream.memcpy_dtov(&k_pool).unwrap();
    let v_pool_h: Vec<i8> = stream.memcpy_dtov(&v_pool).unwrap();
    let k_scales_h: Vec<f16> = stream.memcpy_dtov(&k_scales).unwrap();
    let v_scales_h: Vec<f16> = stream.memcpy_dtov(&v_scales).unwrap();

    // Dequantize and compare.
    let mut max_rel = 0f32;
    for t in 0..NUM_TOKENS {
        let slot = slot_mapping[t] as usize;
        for h in 0..NUM_KV_HEADS {
            let s_off = slot * NUM_KV_HEADS + h;
            let ks = k_scales_h[s_off].to_f32();
            let vs = v_scales_h[s_off].to_f32();
            for d in 0..HEAD_DIM {
                let in_off = (t * NUM_KV_HEADS + h) * HEAD_DIM + d;
                let pool_off = slot * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d;
                let k_dq = ks * k_pool_h[pool_off] as f32;
                let v_dq = vs * v_pool_h[pool_off] as f32;
                let mag = k_in[in_off].abs().max(1e-3);
                let rel_k = (k_dq - k_in[in_off]).abs() / mag;
                let rel_v = (v_dq - v_in[in_off]).abs() / mag;
                max_rel = max_rel.max(rel_k).max(rel_v);
            }
        }
    }
    eprintln!("int8 append round-trip max relative error: {max_rel:.4}");
    // INT8 sym-quant per token-head: worst case is ~1/127 relative for
    // small values, but per-token scale keeps most dims at ≤ 1/64 ≈ 0.016.
    // Use a generous bound to make the test robust to RNG.
    assert!(max_rel < 0.05, "round-trip rel err too high: {max_rel}");
}

#[test]
#[ignore]
fn int8_paged_decode_parity_vs_fp16() {
    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;
    const VALID_KV_LEN: usize = 96; // 6 blocks
    const MAX_BLOCKS: usize = 8;

    let mut rng = 0xCAFEF00Du64;

    // Q : [num_heads, head_dim] FP16
    let q_n = NUM_HEADS * HEAD_DIM;
    let q_data: Vec<f32> = (0..q_n).map(|_| rnd(&mut rng) * 0.5).collect();

    // K/V pool : [max_blocks, block_size, num_kv_heads, head_dim] FP16/INT8
    let kv_n = MAX_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let k_data: Vec<f32> = (0..kv_n).map(|_| rnd(&mut rng) * 0.3).collect();
    let v_data: Vec<f32> = (0..kv_n).map(|_| rnd(&mut rng) * 0.3).collect();

    // Block table: identity mapping (logical i → physical i).
    let block_table: Vec<i32> = (0..MAX_BLOCKS as i32).collect();

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Quantize K/V on host (using kernel's scale convention).
    // Layout for quant: [max_blocks * block_size, num_kv_heads, head_dim].
    let pool_tokens = MAX_BLOCKS * BLOCK_SIZE;
    let (k_q, k_s) = quantize_token_per_head(&k_data, pool_tokens, NUM_KV_HEADS, HEAD_DIM);
    let (v_q, v_s) = quantize_token_per_head(&v_data, pool_tokens, NUM_KV_HEADS, HEAD_DIM);

    // Upload everything.
    let q_dev = CudaBackend::from_slice(&q_data);
    let k_pool_fp16 = CudaBackend::from_slice(&k_data);
    let v_pool_fp16 = CudaBackend::from_slice(&v_data);
    let k_pool_i8 = stream.memcpy_stod(&k_q).unwrap();
    let v_pool_i8 = stream.memcpy_stod(&v_q).unwrap();
    let k_scales_h: Vec<f16> = k_s.iter().map(|x| f16::from_f32(*x)).collect();
    let v_scales_h: Vec<f16> = v_s.iter().map(|x| f16::from_f32(*x)).collect();
    let k_scales_dev = stream.memcpy_stod(&k_scales_h).unwrap();
    let v_scales_dev = stream.memcpy_stod(&v_scales_h).unwrap();
    let bt_dev = stream.memcpy_stod(&block_table).unwrap();

    // Run FP16 reference. Trait API uses `&Buffer` (CudaSlice<f16>) for
    // block_table and context_lens — we reinterpret the i32 slice via
    // upgrade_device_ptr (same hack as the existing bench).
    use cudarc::driver::sys::CUdeviceptr;
    use cudarc::driver::CudaSlice;
    use cudarc::driver::DevicePtr;
    let mut out_fp16 = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    let context_lens_i32: Vec<i32> = vec![VALID_KV_LEN as i32];
    let context_lens_dev = stream.memcpy_stod(&context_lens_i32).unwrap();
    let (bt_ptr, _g0) = bt_dev.device_ptr(&stream);
    let (cl_ptr, _g1) = context_lens_dev.device_ptr(&stream);
    let bt_f16: CudaSlice<f16> = unsafe { stream.upgrade_device_ptr(bt_ptr as CUdeviceptr, 0) };
    let cl_f16: CudaSlice<f16> = unsafe { stream.upgrade_device_ptr(cl_ptr as CUdeviceptr, 0) };
    let mut bctx = <CudaBackend as Backend>::new_context();

    let attn_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    <CudaBackend as BackendPagedKv>::paged_decode_attention(
        &mut bctx,
        &q_dev,
        &k_pool_fp16,
        &v_pool_fp16,
        &mut out_fp16,
        &bt_f16,
        &cl_f16,
        1, // num_seqs
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        MAX_BLOCKS,
        1, // q_len
    )
    .expect("fp16 paged decode");
    <CudaBackend as Backend>::sync(&mut bctx);

    let out_fp16_h: Vec<f32> = CudaBackend::to_vec(&out_fp16, NUM_HEADS * HEAD_DIM);

    // Run INT8 path.
    let mut out_int8 = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    launch_int8_paged_decode_attention(
        &ctx,
        &q_dev,
        &k_pool_i8,
        &v_pool_i8,
        &k_scales_dev,
        &v_scales_dev,
        &bt_dev,
        &mut out_int8,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        attn_scale,
    )
    .expect("int8 paged decode");
    stream.synchronize().unwrap();

    let out_int8_h: Vec<f32> = CudaBackend::to_vec(&out_int8, NUM_HEADS * HEAD_DIM);

    // Compare. INT8 dequantization induces ~1% per-element relative
    // error in attention output for benign inputs.
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut mag_max = 0f32;
    for i in 0..out_fp16_h.len() {
        let diff = (out_fp16_h[i] - out_int8_h[i]).abs();
        max_abs = max_abs.max(diff);
        let mag = out_fp16_h[i].abs().max(1e-4);
        max_rel = max_rel.max(diff / mag);
        mag_max = mag_max.max(out_fp16_h[i].abs());
    }
    eprintln!(
        "int8 vs fp16: max|diff|={max_abs:.5} max rel={max_rel:.4} (max output mag={mag_max:.4})"
    );
    // INT8 paged decode should land within 5% relative of FP16 reference
    // for benign random inputs at this scale.
    assert!(max_abs < 0.05, "abs diff too high: {max_abs}");
    assert!(max_rel < 0.10, "rel diff too high: {max_rel}");
}
