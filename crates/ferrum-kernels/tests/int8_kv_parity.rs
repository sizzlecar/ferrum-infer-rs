//! INT8 KV parity test (CUDA-only).
//!
//! Two checks run end-to-end on a real CUDA device:
//!
//!   1. **Append round-trip.** Take random FP16 K/V tokens, push them
//!      through `launch_int8_kv_cache_append`, read back the INT8 buffer
//!      + per-token FP16 scales, dequantize on the host, and check that
//!      the round-trip max-relative-error is bounded — INT8 sym-quant is
//!      lossy by ~1/127 of the per-token max.
//!
//!   2. **Decode parity vs host reference.** Build a paged KV pool with
//!      random FP16 K/V, compute attention output on the host (FP32
//!      reference), quantize the same K/V to INT8 (using the kernel's
//!      scale convention), run `paged_decode_attention_int8`, and check
//!      that the GPU INT8 output agrees with the FP32 reference within
//!      a small relative tolerance (the only loss is the quant step).
//!
//! Run on a CUDA box:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!     --test int8_kv_parity -- --ignored --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_kernels::backend::Backend;
use ferrum_kernels::int8_kv::{launch_int8_kv_cache_append, launch_int8_paged_decode_attention};
use half::f16;

fn rnd(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = (*state >> 33) as u32 & 0x00FF_FFFF;
    (u as f32) / 16_777_216.0 * 2.0 - 1.0
}

/// Per-token per-kv-head symmetric INT8 quantization. Mirrors the
/// kernel's `s = max(|x|)/127` convention exactly.
fn quantize_token_per_head(
    fp16: &[f32], // [pool_tokens, num_kv_heads, head_dim]
    pool_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; pool_tokens * num_kv_heads * head_dim];
    let mut s = vec![0f32; pool_tokens * num_kv_heads];
    for t in 0..pool_tokens {
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
    const POOL_TOKENS: usize = 64; // sparse: half-filled, room for slot up to 60

    let total = NUM_TOKENS * NUM_KV_HEADS * HEAD_DIM;
    let mut s = 0xDEADBEEFu64;
    let k_in: Vec<f32> = (0..total).map(|_| rnd(&mut s) * 0.1).collect();
    let v_in: Vec<f32> = (0..total).map(|_| rnd(&mut s) * 0.1).collect();

    // Place tokens in non-contiguous slots within the pool [0..POOL_TOKENS).
    // 3 + 4*i for i in 0..16 → max = 3 + 60 = 63.
    let slot_mapping: Vec<i32> = (0..NUM_TOKENS).map(|i| (3 + i * 4) as i32).collect();

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

    // Dequantize and compare. INT8 sym-quant is lossy by ~1/127 of the
    // per-token-head max — assert a generous bound.
    let mut max_abs = 0f32;
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
                max_abs = max_abs.max((k_dq - k_in[in_off]).abs());
                max_abs = max_abs.max((v_dq - v_in[in_off]).abs());
            }
        }
    }
    eprintln!("int8 append round-trip max abs error: {max_abs:.5}");
    // Inputs are scaled by 0.1 so per-token max ≈ 0.1; quantization
    // step is max/127 ≈ 0.0008. Allow 2× margin.
    assert!(max_abs < 0.002, "round-trip abs err too high: {max_abs}");
}

/// Pure-Rust reference: paged decode attention in FP32, mirroring the
/// kernel's algorithm. Used to validate the GPU INT8 output.
#[allow(clippy::too_many_arguments)]
fn host_ref_paged_decode(
    q: &[f32],         // [num_q_heads, head_dim]
    k_pool: &[f32],    // [pool_tokens * num_kv_heads * head_dim]
    v_pool: &[f32],    // same
    block_table: &[i32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    valid_kv_len: usize,
    block_size: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; num_q_heads * head_dim];
    let q_per_kv = num_q_heads / num_kv_heads;
    let kv_stride = num_kv_heads * head_dim;
    let block_stride = block_size * kv_stride;
    for qh in 0..num_q_heads {
        let kv_head = qh / q_per_kv;
        // 1. Q·K^T scores
        let mut scores = vec![0f32; valid_kv_len];
        for kv_pos in 0..valid_kv_len {
            let logical_block = kv_pos / block_size;
            let slot = kv_pos % block_size;
            let physical = block_table[logical_block] as usize;
            let k_base = physical * block_stride + slot * kv_stride + kv_head * head_dim;
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[qh * head_dim + d] * k_pool[k_base + d];
            }
            scores[kv_pos] = dot * scale;
        }
        // 2. softmax
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in &mut scores {
            *s = (*s - max).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for s in &mut scores {
            *s *= inv;
        }
        // 3. weighted V sum
        for d in 0..head_dim {
            let mut acc = 0f32;
            for kv_pos in 0..valid_kv_len {
                let logical_block = kv_pos / block_size;
                let slot = kv_pos % block_size;
                let physical = block_table[logical_block] as usize;
                let v_base = physical * block_stride + slot * kv_stride + kv_head * head_dim;
                acc += scores[kv_pos] * v_pool[v_base + d];
            }
            out[qh * head_dim + d] = acc;
        }
    }
    out
}

#[test]
#[ignore]
fn int8_paged_decode_parity_vs_host_ref() {
    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;
    const VALID_KV_LEN: usize = 96; // 6 blocks
    const MAX_BLOCKS: usize = 8;

    let mut rng = 0xCAFEF00Du64;

    let q_n = NUM_HEADS * HEAD_DIM;
    let q_data: Vec<f32> = (0..q_n).map(|_| rnd(&mut rng) * 0.5).collect();

    let pool_tokens = MAX_BLOCKS * BLOCK_SIZE;
    let kv_n = pool_tokens * NUM_KV_HEADS * HEAD_DIM;
    let k_data: Vec<f32> = (0..kv_n).map(|_| rnd(&mut rng) * 0.3).collect();
    let v_data: Vec<f32> = (0..kv_n).map(|_| rnd(&mut rng) * 0.3).collect();

    // Block table: identity mapping (logical i → physical i).
    let block_table: Vec<i32> = (0..MAX_BLOCKS as i32).collect();

    // Host reference (FP32).
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let out_ref = host_ref_paged_decode(
        &q_data,
        &k_data,
        &v_data,
        &block_table,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    );

    // Quantize K/V on host (using kernel's scale convention).
    let (k_q, k_s) = quantize_token_per_head(&k_data, pool_tokens, NUM_KV_HEADS, HEAD_DIM);
    let (v_q, v_s) = quantize_token_per_head(&v_data, pool_tokens, NUM_KV_HEADS, HEAD_DIM);

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    let q_dev = CudaBackend::from_slice(&q_data);
    let k_pool_i8 = stream.memcpy_stod(&k_q).unwrap();
    let v_pool_i8 = stream.memcpy_stod(&v_q).unwrap();
    let k_scales_h: Vec<f16> = k_s.iter().map(|x| f16::from_f32(*x)).collect();
    let v_scales_h: Vec<f16> = v_s.iter().map(|x| f16::from_f32(*x)).collect();
    let k_scales_dev = stream.memcpy_stod(&k_scales_h).unwrap();
    let v_scales_dev = stream.memcpy_stod(&v_scales_h).unwrap();
    let bt_dev = stream.memcpy_stod(&block_table).unwrap();

    let mut out_dev = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    launch_int8_paged_decode_attention(
        &ctx,
        &q_dev,
        &k_pool_i8,
        &v_pool_i8,
        &k_scales_dev,
        &v_scales_dev,
        &bt_dev.as_view(),
        &mut out_dev,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    )
    .expect("int8 paged decode");
    stream.synchronize().unwrap();

    let out_int8: Vec<f32> = CudaBackend::to_vec(&out_dev, NUM_HEADS * HEAD_DIM);

    // Compare via cosine similarity + max absolute error against the
    // full output max — per-element relative error is meaningless when
    // outputs are near zero (16% rel on a 0.0002 abs diff is fine if
    // the max output magnitude is 0.06).
    let mut max_abs = 0f32;
    let mut sse = 0f64;
    let mut sum_a = 0f64;
    let mut sum_b = 0f64;
    let mut dot = 0f64;
    let mut max_mag = 0f32;
    for i in 0..out_ref.len() {
        let diff = (out_int8[i] - out_ref[i]).abs();
        max_abs = max_abs.max(diff);
        sse += (diff as f64) * (diff as f64);
        max_mag = max_mag.max(out_ref[i].abs());
        sum_a += (out_ref[i] as f64) * (out_ref[i] as f64);
        sum_b += (out_int8[i] as f64) * (out_int8[i] as f64);
        dot += (out_ref[i] as f64) * (out_int8[i] as f64);
    }
    let rmse = (sse / out_ref.len() as f64).sqrt() as f32;
    let cosine = dot / (sum_a.sqrt() * sum_b.sqrt() + 1e-12);
    let rel_to_mag = max_abs / max_mag.max(1e-6);
    eprintln!(
        "int8 vs host-ref: max|diff|={max_abs:.5} rmse={rmse:.5} cos={cosine:.5} \
         rel-to-max-mag={rel_to_mag:.4} (max output mag={max_mag:.4})"
    );
    // Cosine ≈ 1 means INT8 reproduces the FP32 reference vector
    // direction nearly perfectly. The remaining absolute error is the
    // INT8 quantization noise floor (~max(|K|)/127 propagated through
    // attention, ≈ 0.0003 at this shape).
    assert!(cosine > 0.999, "cosine similarity too low: {cosine}");
    assert!(rel_to_mag < 0.02, "max abs / max mag too high: {rel_to_mag}");
}

/// End-to-end kernel composition test that mirrors how a model decode
/// loop would use INT8 KV: take FP16 K/V tokens, run `int8_kv_cache_append`
/// to populate the INT8 paged pool, then run `paged_decode_attention_int8`
/// reading from that pool. Compare to a pure FP32 host reference computed
/// directly from the original FP16 K/V (no quantization step in the
/// reference). Cosine similarity should still be ~0.999 — the only loss
/// is the INT8 round-trip.
#[test]
#[ignore]
fn int8_kv_append_then_decode_e2e() {
    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;
    const VALID_KV_LEN: usize = 64; // 4 blocks
    const MAX_BLOCKS: usize = 8;

    let mut rng = 0xBEEF_F00Du64;

    // Q
    let q_n = NUM_HEADS * HEAD_DIM;
    let q_data: Vec<f32> = (0..q_n).map(|_| rnd(&mut rng) * 0.5).collect();

    // FP16 K/V tokens that will be appended into the INT8 pool. One
    // token at a time mirrors a decode-loop append; here we batch all
    // VALID_KV_LEN tokens in a single launch (the kernel handles
    // multi-token batches via the (token, kv_head) grid).
    let kv_in_n = VALID_KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    let k_in_data: Vec<f32> = (0..kv_in_n).map(|_| rnd(&mut rng) * 0.3).collect();
    let v_in_data: Vec<f32> = (0..kv_in_n).map(|_| rnd(&mut rng) * 0.3).collect();

    // Slot mapping: append into slots 0..VALID_KV_LEN (contiguous, simplest case).
    let slot_mapping: Vec<i32> = (0..VALID_KV_LEN as i32).collect();

    // Block table: identity (logical i → physical i).
    let block_table: Vec<i32> = (0..MAX_BLOCKS as i32).collect();

    // Host reference: build a paged pool laid out exactly as the
    // append kernel would produce, then run the FP32 reference.
    let pool_tokens = MAX_BLOCKS * BLOCK_SIZE;
    let mut k_pool_ref = vec![0f32; pool_tokens * NUM_KV_HEADS * HEAD_DIM];
    let mut v_pool_ref = vec![0f32; pool_tokens * NUM_KV_HEADS * HEAD_DIM];
    for t in 0..VALID_KV_LEN {
        let slot = slot_mapping[t] as usize;
        for h in 0..NUM_KV_HEADS {
            for d in 0..HEAD_DIM {
                let in_off = (t * NUM_KV_HEADS + h) * HEAD_DIM + d;
                let pool_off = slot * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d;
                k_pool_ref[pool_off] = k_in_data[in_off];
                v_pool_ref[pool_off] = v_in_data[in_off];
            }
        }
    }
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let out_ref = host_ref_paged_decode(
        &q_data,
        &k_pool_ref,
        &v_pool_ref,
        &block_table,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    );

    // GPU path: append → decode.
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    let q_dev = CudaBackend::from_slice(&q_data);
    let k_in_dev = CudaBackend::from_slice(&k_in_data);
    let v_in_dev = CudaBackend::from_slice(&v_in_data);
    let mut k_pool: cudarc::driver::CudaSlice<i8> = stream
        .alloc_zeros::<i8>(pool_tokens * NUM_KV_HEADS * HEAD_DIM)
        .unwrap();
    let mut v_pool: cudarc::driver::CudaSlice<i8> = stream
        .alloc_zeros::<i8>(pool_tokens * NUM_KV_HEADS * HEAD_DIM)
        .unwrap();
    let mut k_scales: cudarc::driver::CudaSlice<f16> = stream
        .alloc_zeros::<f16>(pool_tokens * NUM_KV_HEADS)
        .unwrap();
    let mut v_scales: cudarc::driver::CudaSlice<f16> = stream
        .alloc_zeros::<f16>(pool_tokens * NUM_KV_HEADS)
        .unwrap();
    let slot_dev = stream.memcpy_stod(&slot_mapping).unwrap();
    let bt_dev = stream.memcpy_stod(&block_table).unwrap();

    launch_int8_kv_cache_append(
        &ctx,
        &k_in_dev,
        &v_in_dev,
        &mut k_pool,
        &mut v_pool,
        &mut k_scales,
        &mut v_scales,
        &slot_dev,
        VALID_KV_LEN,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    .expect("kv append");

    let mut out_dev = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    launch_int8_paged_decode_attention(
        &ctx,
        &q_dev,
        &k_pool,
        &v_pool,
        &k_scales,
        &v_scales,
        &bt_dev.as_view(),
        &mut out_dev,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    )
    .expect("int8 paged decode");
    stream.synchronize().unwrap();

    let out_int8: Vec<f32> = CudaBackend::to_vec(&out_dev, NUM_HEADS * HEAD_DIM);

    let mut sum_a = 0f64;
    let mut sum_b = 0f64;
    let mut dot = 0f64;
    let mut max_abs = 0f32;
    let mut max_mag = 0f32;
    for i in 0..out_ref.len() {
        max_abs = max_abs.max((out_int8[i] - out_ref[i]).abs());
        max_mag = max_mag.max(out_ref[i].abs());
        sum_a += (out_ref[i] as f64) * (out_ref[i] as f64);
        sum_b += (out_int8[i] as f64) * (out_int8[i] as f64);
        dot += (out_ref[i] as f64) * (out_int8[i] as f64);
    }
    let cosine = dot / (sum_a.sqrt() * sum_b.sqrt() + 1e-12);
    eprintln!(
        "append→decode e2e: max|diff|={max_abs:.5} cos={cosine:.5} (max output mag={max_mag:.4})"
    );
    // Same noise floor as the standalone decode parity test — the
    // kernel pair composes cleanly.
    assert!(cosine > 0.999, "cosine similarity too low: {cosine}");
    let rel_to_mag = max_abs / max_mag.max(1e-6);
    assert!(rel_to_mag < 0.02, "rel-to-max-mag too high: {rel_to_mag}");
}

/// Same flow as `int8_kv_append_then_decode_e2e` but driven through
/// the `KvCacheQuant<CudaBackend, KvInt8>` abstraction (constructor +
/// typed buffers). This is what a model decode loop would use — it
/// no longer touches cudarc primitives directly.
#[test]
#[ignore]
fn kv_cache_quant_int8_e2e() {
    use ferrum_kernels::backend::{KvCacheQuant, KvInt8};

    const NUM_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;
    const VALID_KV_LEN: usize = 64;
    const MAX_BLOCKS: usize = 8;

    let mut rng = 0xCAFE_FACEu64;

    // Random Q + K/V tokens (FP16 inputs).
    let q_n = NUM_HEADS * HEAD_DIM;
    let q_data: Vec<f32> = (0..q_n).map(|_| rnd(&mut rng) * 0.5).collect();
    let kv_in_n = VALID_KV_LEN * NUM_KV_HEADS * HEAD_DIM;
    let k_in_data: Vec<f32> = (0..kv_in_n).map(|_| rnd(&mut rng) * 0.3).collect();
    let v_in_data: Vec<f32> = (0..kv_in_n).map(|_| rnd(&mut rng) * 0.3).collect();

    let slot_mapping: Vec<i32> = (0..VALID_KV_LEN as i32).collect();

    // Block table: identity (logical i → physical i) for the first
    // ceil(VALID_KV_LEN/BLOCK_SIZE) blocks.
    let block_table: Vec<i32> = (0..MAX_BLOCKS as i32).collect();

    // Host reference (FP32, no quant).
    let pool_tokens = MAX_BLOCKS * BLOCK_SIZE;
    let mut k_pool_ref = vec![0f32; pool_tokens * NUM_KV_HEADS * HEAD_DIM];
    let mut v_pool_ref = vec![0f32; pool_tokens * NUM_KV_HEADS * HEAD_DIM];
    for t in 0..VALID_KV_LEN {
        let slot = slot_mapping[t] as usize;
        for h in 0..NUM_KV_HEADS {
            for d in 0..HEAD_DIM {
                let in_off = (t * NUM_KV_HEADS + h) * HEAD_DIM + d;
                let pool_off = slot * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d;
                k_pool_ref[pool_off] = k_in_data[in_off];
                v_pool_ref[pool_off] = v_in_data[in_off];
            }
        }
    }
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let out_ref = host_ref_paged_decode(
        &q_data,
        &k_pool_ref,
        &v_pool_ref,
        &block_table,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    );

    // Drive the GPU side through the high-level KvCacheQuant cache.
    let mut cache: KvCacheQuant<CudaBackend, KvInt8> =
        KvCacheQuant::new_paged_cuda(MAX_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM);

    let ctx_handle = CudaContext::new(0).expect("CUDA context");
    let stream = ctx_handle.default_stream();

    // Upload the FP16 K/V tokens + Q + slot mapping + block table.
    let q_dev = CudaBackend::from_slice(&q_data);
    let k_in_dev = CudaBackend::from_slice(&k_in_data);
    let v_in_dev = CudaBackend::from_slice(&v_in_data);
    let slot_dev = stream.memcpy_stod(&slot_mapping).unwrap();
    let bt_dev = stream.memcpy_stod(&block_table).unwrap();

    // Append.
    launch_int8_kv_cache_append(
        &ctx_handle,
        &k_in_dev,
        &v_in_dev,
        cache.k.buffer_mut(),
        cache.v.buffer_mut(),
        cache.k_scales.buffer_mut(),
        cache.v_scales.buffer_mut(),
        &slot_dev,
        VALID_KV_LEN,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    .expect("kv append (cache)");

    // Decode.
    let mut out_dev = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    launch_int8_paged_decode_attention(
        &ctx_handle,
        &q_dev,
        cache.k.buffer(),
        cache.v.buffer(),
        cache.k_scales.buffer(),
        cache.v_scales.buffer(),
        &bt_dev.as_view(),
        &mut out_dev,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    )
    .expect("int8 paged decode (cache)");
    stream.synchronize().unwrap();

    let out_int8: Vec<f32> = CudaBackend::to_vec(&out_dev, NUM_HEADS * HEAD_DIM);

    let mut sum_a = 0f64;
    let mut sum_b = 0f64;
    let mut dot = 0f64;
    let mut max_abs = 0f32;
    let mut max_mag = 0f32;
    for i in 0..out_ref.len() {
        max_abs = max_abs.max((out_int8[i] - out_ref[i]).abs());
        max_mag = max_mag.max(out_ref[i].abs());
        sum_a += (out_ref[i] as f64) * (out_ref[i] as f64);
        sum_b += (out_int8[i] as f64) * (out_int8[i] as f64);
        dot += (out_ref[i] as f64) * (out_int8[i] as f64);
    }
    let cosine = dot / (sum_a.sqrt() * sum_b.sqrt() + 1e-12);
    eprintln!(
        "KvCacheQuant<CudaBackend, KvInt8> e2e: cos={cosine:.5} max|diff|={max_abs:.5} (mag={max_mag:.4})"
    );
    assert!(cosine > 0.999, "cosine similarity too low: {cosine}");
}
