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

use cudarc::driver::{sys, CudaStream};
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_kernels::backend::{Backend, BackendCompletion, BackendInt8KvOps};
use ferrum_kernels::int8_kv::{launch_int8_kv_cache_append, launch_int8_paged_decode_attention};
use half::f16;
use std::sync::{Arc, Mutex};

// Backend contexts in this process share a CUDA stream. Keep another test's
// work out of the stream while the ownership check captures it.
static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Capture only to inspect stream ownership, then discard without replaying.
/// A NULL-stream escape cannot pass by happening to finish before readback.
/// Allocations and uploads are complete before this helper is entered.
fn assert_launches_use_stream(
    stream: &Arc<CudaStream>,
    enqueue: impl FnOnce() -> Result<(), String>,
) {
    assert!(
        !stream.cu_stream().is_null(),
        "test must observe a non-NULL stream"
    );
    stream.context().bind_to_thread().unwrap();
    stream.synchronize().unwrap();
    let begin = unsafe {
        sys::cuStreamBeginCapture_v2(
            stream.cu_stream(),
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    };
    assert_eq!(
        begin,
        sys::CUresult::CUDA_SUCCESS,
        "begin stream inspection"
    );
    let encoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(enqueue));
    let mut graph = std::ptr::null_mut();
    let end = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
    let mut node_count = 0;
    let mut kernel_count = 0;
    let inspected = || {
        assert_eq!(end, sys::CUresult::CUDA_SUCCESS, "end stream inspection");
        assert!(!graph.is_null());
        let status = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut node_count) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
        let mut nodes = vec![std::ptr::null_mut(); node_count];
        let status = unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut node_count) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
        for node in nodes {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            let status = unsafe { sys::cuGraphNodeGetType(node, &mut kind) };
            assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
            if kind == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                kernel_count += 1;
            }
        }
    };
    // No graph executable or global graph key is created. Cleanup also runs
    // when a launch or inspection fails, before any captured buffers drop.
    let inspected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(inspected));
    if !graph.is_null() {
        let status = unsafe { sys::cuGraphDestroy(graph) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS);
    }
    if let Err(panic) = inspected {
        std::panic::resume_unwind(panic);
    }
    encoded.unwrap().expect("capture both INT8 launches");
    assert_eq!(
        kernel_count, 2,
        "append and attention must use the observed stream"
    );
}

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
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
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

    let mut ctx = CudaBackend::new_context();
    let stream = ctx.stream.clone();

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
    let slot_dev = stream.clone_htod(&slot_mapping).unwrap();

    launch_int8_kv_cache_append(
        &stream,
        k_in_dev.as_f16(),
        v_in_dev.as_f16(),
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

    assert!(matches!(
        CudaBackend::observe_completion(&mut ctx),
        BackendCompletion::Succeeded
    ));

    let k_pool_h: Vec<i8> = stream.clone_dtoh(&k_pool).unwrap();
    let v_pool_h: Vec<i8> = stream.clone_dtoh(&v_pool).unwrap();
    let k_scales_h: Vec<f16> = stream.clone_dtoh(&k_scales).unwrap();
    let v_scales_h: Vec<f16> = stream.clone_dtoh(&v_scales).unwrap();

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
    q: &[f32],      // [num_q_heads, head_dim]
    k_pool: &[f32], // [pool_tokens * num_kv_heads * head_dim]
    v_pool: &[f32], // same
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
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
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

    let mut ctx = CudaBackend::new_context();
    let stream = ctx.stream.clone();

    let q_dev = CudaBackend::from_slice(&q_data);
    let k_pool_i8 = stream.clone_htod(&k_q).unwrap();
    let v_pool_i8 = stream.clone_htod(&v_q).unwrap();
    let k_scales_h: Vec<f16> = k_s.iter().map(|x| f16::from_f32(*x)).collect();
    let v_scales_h: Vec<f16> = v_s.iter().map(|x| f16::from_f32(*x)).collect();
    let k_scales_dev = stream.clone_htod(&k_scales_h).unwrap();
    let v_scales_dev = stream.clone_htod(&v_scales_h).unwrap();
    let bt_dev = stream.clone_htod(&block_table).unwrap();

    let mut out_dev = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    launch_int8_paged_decode_attention(
        &stream,
        q_dev.as_f16(),
        &k_pool_i8,
        &v_pool_i8,
        &k_scales_dev,
        &v_scales_dev,
        &bt_dev.as_view(),
        out_dev.as_f16_mut(),
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    )
    .expect("int8 paged decode");
    assert!(matches!(
        CudaBackend::observe_completion(&mut ctx),
        BackendCompletion::Succeeded
    ));

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
    assert!(
        rel_to_mag < 0.02,
        "max abs / max mag too high: {rel_to_mag}"
    );
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
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
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
    let mut ctx = CudaBackend::new_context();
    let stream = ctx.stream.clone();

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
    let slot_dev = stream.clone_htod(&slot_mapping).unwrap();
    let bt_dev = stream.clone_htod(&block_table).unwrap();

    let mut out_dev = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    let mut enqueue = || {
        launch_int8_kv_cache_append(
            &stream,
            k_in_dev.as_f16(),
            v_in_dev.as_f16(),
            &mut k_pool,
            &mut v_pool,
            &mut k_scales,
            &mut v_scales,
            &slot_dev,
            VALID_KV_LEN,
            NUM_KV_HEADS,
            HEAD_DIM,
        )?;
        launch_int8_paged_decode_attention(
            &stream,
            q_dev.as_f16(),
            &k_pool,
            &v_pool,
            &k_scales,
            &v_scales,
            &bt_dev.as_view(),
            out_dev.as_f16_mut(),
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            VALID_KV_LEN,
            BLOCK_SIZE,
            scale,
        )
    };
    assert_launches_use_stream(&stream, &mut enqueue);
    // The captured graph was never run. This eager submission must provide
    // the actual write/read result observed through the caller's context.
    enqueue().expect("append then decode on the caller stream");
    assert!(matches!(
        CudaBackend::observe_completion(&mut ctx),
        BackendCompletion::Succeeded
    ));

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
/// the actual `BackendInt8KvOps` entrypoints and a single backend context.
/// Completion must cover the append's slot upload, cache writes, and attention
/// reads; there is no NULL-stream synchronization between these operations.
#[test]
#[ignore]
fn kv_cache_quant_int8_e2e() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    use ferrum_kernels::backend::{KvCacheQuant, KvInt8, QuantizedPagedKvPool};

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
    let mut pool = QuantizedPagedKvPool::<CudaBackend, KvInt8>::new_cuda(
        MAX_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    );

    let mut ctx = CudaBackend::new_context();

    // Upload the FP16 K/V tokens + Q. The production append method creates its
    // slot mapping on ctx.stream from the cache's host block-index mirror.
    let q_dev = CudaBackend::from_slice(&q_data);
    let k_in_dev = CudaBackend::from_slice(&k_in_data);
    let v_in_dev = CudaBackend::from_slice(&v_in_data);
    cache.paged_block_indices = block_table.iter().map(|&block| block as u32).collect();
    CudaBackend::write_typed::<u32>(
        &mut ctx,
        cache.block_table.as_mut().unwrap(),
        &cache.paged_block_indices,
    );
    CudaBackend::write_typed::<u32>(
        &mut ctx,
        cache.context_lens.as_mut().unwrap(),
        &[VALID_KV_LEN as u32],
    );

    // Append.
    CudaBackend::int8_kv_append_paged(
        &mut ctx,
        &k_in_dev,
        &v_in_dev,
        &mut pool,
        &cache.paged_block_indices,
        0,
        VALID_KV_LEN,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    .expect("kv append (cache)");

    // Decode.
    let mut out_dev = CudaBackend::alloc(NUM_HEADS * HEAD_DIM);
    CudaBackend::int8_paged_decode_attention(
        &mut ctx,
        &q_dev,
        &pool,
        cache.block_table.as_ref().unwrap(),
        &cache.paged_block_indices,
        &mut out_dev,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        VALID_KV_LEN,
        BLOCK_SIZE,
        scale,
    )
    .expect("int8 paged decode (cache)");
    assert!(matches!(
        CudaBackend::observe_completion(&mut ctx),
        BackendCompletion::Succeeded
    ));

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
    let rel_to_mag = max_abs / max_mag.max(1e-6);
    assert!(rel_to_mag < 0.02, "rel-to-max-mag too high: {rel_to_mag}");
}

/// Distinct sequence tables address a single physical pool, including blocks
/// beyond either sequence's logical capacity and an aliased complete prefix.
#[test]
#[ignore]
fn int8_shared_pool_nonzero_blocks_and_prefix_alias_match_host_reference() {
    let _guard = CUDA_TEST_LOCK.lock().unwrap();
    use ferrum_kernels::backend::{KvCacheQuant, KvInt8, QuantizedPagedKvPool};
    const BS: usize = 16;
    const HD: usize = 64;
    const KV: usize = 2;
    const Q: usize = 4;
    let mut ctx = CudaBackend::new_context();
    let mut pool = QuantizedPagedKvPool::<CudaBackend, KvInt8>::new_cuda(3, BS, KV, HD);
    let mut a = KvCacheQuant::<CudaBackend, KvInt8>::new_paged_cuda(2, BS, KV, HD);
    let mut b = KvCacheQuant::<CudaBackend, KvInt8>::new_paged_cuda(2, BS, KV, HD);
    let mut alias = KvCacheQuant::<CudaBackend, KvInt8>::new_paged_cuda(2, BS, KV, HD);
    a.paged_block_indices = vec![0, 1];
    b.paged_block_indices = vec![2];
    alias.paged_block_indices = vec![0];
    for cache in [&mut a, &mut b, &mut alias] {
        CudaBackend::write_typed::<u32>(
            &mut ctx,
            cache.block_table.as_mut().unwrap(),
            &cache.paged_block_indices,
        );
    }
    assert!(b.paged_block_indices[0] as usize * BS >= b.capacity);
    assert_eq!(pool.k.buffer().len(), 3 * BS * KV * HD);
    let mut rng = 0x9137_bca1;
    let q: Vec<f32> = (0..Q * HD).map(|_| rnd(&mut rng) * 0.25).collect();
    let a_k: Vec<f32> = (0..17 * KV * HD).map(|_| rnd(&mut rng) * 0.4).collect();
    let a_v: Vec<f32> = (0..17 * KV * HD).map(|_| rnd(&mut rng) * 0.4).collect();
    let b_k: Vec<f32> = (0..KV * HD).map(|_| rnd(&mut rng) * 0.7).collect();
    let b_v: Vec<f32> = (0..KV * HD).map(|_| rnd(&mut rng) * 0.7).collect();
    let mut expected_k = vec![0.0; 3 * BS * KV * HD];
    let mut expected_v = expected_k.clone();
    expected_k[..a_k.len()].copy_from_slice(&a_k);
    expected_v[..a_v.len()].copy_from_slice(&a_v);
    expected_k[2 * BS * KV * HD..(2 * BS + 1) * KV * HD].copy_from_slice(&b_k);
    expected_v[2 * BS * KV * HD..(2 * BS + 1) * KV * HD].copy_from_slice(&b_v);
    let q_device = CudaBackend::from_slice(&q);
    let append = |ctx: &mut _, pool: &mut _, indices: &[u32], k: &[f32], v: &[f32], count| {
        let k_device = CudaBackend::from_slice(k);
        let v_device = CudaBackend::from_slice(v);
        CudaBackend::int8_kv_append_paged(
            ctx, &k_device, &v_device, pool, indices, 0, count, BS, KV, HD,
        )
        .unwrap();
        assert!(matches!(
            CudaBackend::observe_completion(ctx),
            BackendCompletion::Succeeded
        ));
    };
    let decode = |ctx: &mut _, pool: &_, cache: &KvCacheQuant<CudaBackend, KvInt8>, count| {
        let mut output = CudaBackend::alloc(Q * HD);
        CudaBackend::int8_paged_decode_attention(
            ctx,
            &q_device,
            pool,
            cache.block_table.as_ref().unwrap(),
            &cache.paged_block_indices,
            &mut output,
            Q,
            KV,
            HD,
            count,
            BS,
            (HD as f32).sqrt().recip(),
        )
        .unwrap();
        assert!(matches!(
            CudaBackend::observe_completion(ctx),
            BackendCompletion::Succeeded
        ));
        CudaBackend::to_vec(&output, Q * HD)
    };
    append(&mut ctx, &mut pool, &a.paged_block_indices, &a_k, &a_v, 17);
    let a_before = decode(&mut ctx, &pool, &a, 17);
    append(&mut ctx, &mut pool, &b.paged_block_indices, &b_k, &b_v, 1);
    let a_after = decode(&mut ctx, &pool, &a, 17);
    assert_eq!(
        a_before, a_after,
        "B must not overwrite A's physical blocks"
    );
    for (cache, count) in [(&a, 17), (&b, 1), (&alias, 16)] {
        let actual = decode(&mut ctx, &pool, cache, count);
        let table: Vec<i32> = cache
            .paged_block_indices
            .iter()
            .map(|&id| id as i32)
            .collect();
        let expected = host_ref_paged_decode(
            &q,
            &expected_k,
            &expected_v,
            &table,
            Q,
            KV,
            HD,
            count,
            BS,
            (HD as f32).sqrt().recip(),
        );
        let magnitude = expected.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(actual.iter().all(|x| x.is_finite()));
        let maximum_error = actual
            .iter()
            .zip(&expected)
            .fold(0.0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(
            maximum_error < 0.02 * magnitude.max(1e-6),
            "shared INT8 attention error={maximum_error}, magnitude={magnitude}"
        );
    }
}
