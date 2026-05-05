//! Equivalence test: per-item `kv_cache_append_head_major_f16` (m calls)
//! vs `kv_cache_append_batched_per_cache_f16` (one call). Both should
//! write the SAME bytes to each item's cache buffer.
//!
//! Bisects the m≥2 batched_decode garbage bug after batched flash_attn
//! was fixed (head-major layout) and verified by integration test:
//! per-item kv_append + new batched flash_attn → CORRECT joke output.
//! But default (batched kv_append + new batched flash_attn) → still
//! GARBAGE. So the cache CONTENT must differ between paths.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!       --test kv_cache_append_batched_eq -- --nocapture

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cuda::CudaBackend, Backend};

const M: usize = 4;
const NKV: usize = 8;
const HD: usize = 128;
const CAPACITY: usize = 64;
const CACHE_LEN: usize = 4;

fn det_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as f32) / (u64::MAX as f32) * 2.0 - 1.0) * 0.3
        })
        .collect()
}

#[test]
fn kv_append_batched_matches_per_item_head_major() {
    let mut ctx = CudaBackend::new_context();

    // new_data = [m, NKV, HD] item-major (output of qk_norm_rope_batched
    // for K). Same source for both paths.
    let new_data_h = det_f32(1, M * NKV * HD);
    let new_data_dev = CudaBackend::from_slice(&new_data_h);

    // ── Per-item path ─────────────────────────────────────────────────
    // For each item i: copy_slice new_data[i*NKV*HD..(i+1)*NKV*HD] →
    // single scratch [NKV, 1, HD] (head-major-equivalent for tokens=1),
    // then call kv_cache_append_head_major.
    let mut per_item_caches: Vec<_> = (0..M)
        .map(|_| {
            let mut buf = CudaBackend::alloc(NKV * CAPACITY * HD);
            // pre-fill with sentinel so unmodified slots stay distinct
            let sentinel = vec![0.0f32; NKV * CAPACITY * HD];
            // The simplest way to seed is from_slice + write; but
            // we'll use B::sync after each append to read back.
            let mut seed_buf = CudaBackend::from_slice(&sentinel);
            CudaBackend::copy_slice(&mut ctx, &seed_buf, 0, &mut buf, 0, NKV * CAPACITY * HD);
            let _ = seed_buf;
            buf
        })
        .collect();

    let mut single_scratch = CudaBackend::alloc(NKV * HD);
    for i in 0..M {
        CudaBackend::copy_slice(
            &mut ctx,
            &new_data_dev,
            i * NKV * HD,
            &mut single_scratch,
            0,
            NKV * HD,
        );
        let v_dummy = CudaBackend::alloc(NKV * CAPACITY * HD);
        // We only test K append; pass single_scratch as both K and V
        // sources (V cache is throwaway, just needs same shape).
        let mut v_dummy_cache = CudaBackend::alloc(NKV * CAPACITY * HD);
        CudaBackend::kv_cache_append_head_major(
            &mut ctx,
            &mut per_item_caches[i],
            &mut v_dummy_cache,
            CACHE_LEN,
            CAPACITY,
            &single_scratch,
            &v_dummy,
            1, // new_tokens
            NKV,
            HD,
        );
    }
    CudaBackend::sync(&mut ctx);
    let per_item_cache_0_h = CudaBackend::to_vec(&per_item_caches[0], NKV * CAPACITY * HD);
    let per_item_cache_1_h = CudaBackend::to_vec(&per_item_caches[1], NKV * CAPACITY * HD);

    // ── Batched path ──────────────────────────────────────────────────
    let mut batched_caches: Vec<_> = (0..M)
        .map(|_| {
            let zero = vec![0.0f32; NKV * CAPACITY * HD];
            let seed_buf = CudaBackend::from_slice(&zero);
            let mut buf = CudaBackend::alloc(NKV * CAPACITY * HD);
            CudaBackend::copy_slice(&mut ctx, &seed_buf, 0, &mut buf, 0, NKV * CAPACITY * HD);
            let _ = seed_buf;
            buf
        })
        .collect();

    let cache_lens_h: Vec<u32> = vec![CACHE_LEN as u32; M];
    let mut cache_lens_dev = CudaBackend::alloc_u32(M);
    CudaBackend::write_u32(&mut ctx, &mut cache_lens_dev, &cache_lens_h);

    let cache_refs: Vec<&_> = batched_caches.iter().collect();
    CudaBackend::kv_cache_append_batched_per_cache(
        &mut ctx,
        &cache_refs,
        &new_data_dev,
        &cache_lens_dev,
        CAPACITY,
        M,
        NKV,
        HD,
        0, // slot
    )
    .expect("batched kv_append");
    CudaBackend::sync(&mut ctx);
    let batched_cache_0_h = CudaBackend::to_vec(&batched_caches[0], NKV * CAPACITY * HD);
    let batched_cache_1_h = CudaBackend::to_vec(&batched_caches[1], NKV * CAPACITY * HD);

    // ── Compare cache contents ────────────────────────────────────────
    fn diff_stats(a: &[f32], b: &[f32], label: &str) {
        let mut max_abs = 0.0f32;
        let mut argmax = 0usize;
        let mut nz_per = 0usize;
        let mut nz_bat = 0usize;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
            if x.abs() > 1e-6 {
                nz_per += 1;
            }
            if y.abs() > 1e-6 {
                nz_bat += 1;
            }
        }
        let kv_head = argmax / (CAPACITY * HD);
        let kv_pos = (argmax % (CAPACITY * HD)) / HD;
        let dim = argmax % HD;
        println!(
            "[{label}] max_abs_diff={max_abs:.3e}  worst @ kv_head={kv_head} kv_pos={kv_pos} dim={dim}\n  \
             per_item={}  batched={}\n  \
             per_item nonzero count: {nz_per}  batched nonzero count: {nz_bat}",
            a[argmax], b[argmax]
        );
    }
    diff_stats(&per_item_cache_0_h, &batched_cache_0_h, "item 0 cache");
    diff_stats(&per_item_cache_1_h, &batched_cache_1_h, "item 1 cache");

    // Tolerance: byte identical writes expected.
    let max0 = per_item_cache_0_h
        .iter()
        .zip(batched_cache_0_h.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max1 = per_item_cache_1_h
        .iter()
        .zip(batched_cache_1_h.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max0 < 1e-3, "item 0 cache diverges: max={max0}");
    assert!(max1 < 1e-3, "item 1 cache diverges: max={max1}");
}
