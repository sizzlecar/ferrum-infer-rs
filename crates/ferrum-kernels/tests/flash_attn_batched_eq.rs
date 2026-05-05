//! Equivalence test: per-item `flash_attention` (decode_attention_head_major_f16)
//! called m times vs `flash_attention_batched_per_cache`
//! (batched_decode_attention_f16) called once for m items.
//!
//! Bisects the m≥2 batched_decode garbage bug after the qkr kernel was
//! ruled out (qk_norm_rope_batched_eq passes 0.0 max diff). A
//! suspected layout mismatch between batched kv_append (head-major) and
//! batched flash_attn (was token-major) was patched to head-major; this
//! test verifies the patch (or finds the next bug).
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!       --test flash_attn_batched_eq -- --nocapture

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cuda::CudaBackend, AttnConfig, Backend};

const M: usize = 2;
const NH: usize = 32;
const NKV: usize = 8;
const HD: usize = 128;
const CAPACITY: usize = 64;
const VALID_KV: usize = 5;
const SCALE: f32 = 0.0883883476; // 1/sqrt(128)

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
fn flash_attn_batched_matches_per_item() {
    let mut ctx = CudaBackend::new_context();

    // Q: [m, NH, HD] item-major
    let q_h = det_f32(1, M * NH * HD);
    let q_dev = CudaBackend::from_slice(&q_h);

    // K and V caches per item (head-major [NKV, CAPACITY, HD])
    let k0_h = det_f32(10, NKV * CAPACITY * HD);
    let v0_h = det_f32(11, NKV * CAPACITY * HD);
    let k1_h = det_f32(20, NKV * CAPACITY * HD);
    let v1_h = det_f32(21, NKV * CAPACITY * HD);
    let k0_dev = CudaBackend::from_slice(&k0_h);
    let v0_dev = CudaBackend::from_slice(&v0_h);
    let k1_dev = CudaBackend::from_slice(&k1_h);
    let v1_dev = CudaBackend::from_slice(&v1_h);

    // Per-item: split Q + run head-major decode_attention per item
    let mut per_item_concat = vec![0.0f32; M * NH * HD];
    let mut q_single_scratch = CudaBackend::alloc(NH * HD);
    let mut out_single_scratch = CudaBackend::alloc(NH * HD);
    let attn_cfg = AttnConfig {
        num_heads: NH,
        num_kv_heads: NKV,
        head_dim: HD,
        causal: true,
        scale: SCALE,
        kv_seq_stride: CAPACITY,
        sliding_window: 0,
    };
    for i in 0..M {
        CudaBackend::copy_slice(
            &mut ctx,
            &q_dev,
            i * NH * HD,
            &mut q_single_scratch,
            0,
            NH * HD,
        );
        let k_ref = if i == 0 { &k0_dev } else { &k1_dev };
        let v_ref = if i == 0 { &v0_dev } else { &v1_dev };
        CudaBackend::flash_attention(
            &mut ctx,
            &q_single_scratch,
            k_ref,
            v_ref,
            &mut out_single_scratch,
            1,            // batch
            1,            // q_len
            VALID_KV,     // kv_len
            VALID_KV - 1, // pos_offset (irrelevant for head-major decode)
            &attn_cfg,
        );
        CudaBackend::sync(&mut ctx);
        let out_h = CudaBackend::to_vec(&out_single_scratch, NH * HD);
        per_item_concat[i * NH * HD..(i + 1) * NH * HD].copy_from_slice(&out_h);
    }

    // Batched: 1 call for m items
    let kv_lens_h: Vec<u32> = vec![VALID_KV as u32; M];
    let mut kv_lens_dev = CudaBackend::alloc_u32(M);
    CudaBackend::write_u32(&mut ctx, &mut kv_lens_dev, &kv_lens_h);
    let mut out_batched_dev = CudaBackend::alloc(M * NH * HD);
    let k_caches: Vec<&_> = vec![&k0_dev, &k1_dev];
    let v_caches: Vec<&_> = vec![&v0_dev, &v1_dev];

    CudaBackend::flash_attention_batched_per_cache(
        &mut ctx,
        &q_dev,
        &k_caches,
        &v_caches,
        &kv_lens_dev,
        &mut out_batched_dev,
        NH,
        NKV,
        HD,
        SCALE,
        VALID_KV, // max_valid_kv (post-bump)
        CAPACITY,
        0, // slot
    )
    .expect("flash_attention_batched_per_cache");
    CudaBackend::sync(&mut ctx);
    let batched_concat = CudaBackend::to_vec(&out_batched_dev, M * NH * HD);

    // Compare elementwise
    let mut max_abs = 0.0f32;
    let mut argmax = 0usize;
    let mut mismatches = 0usize;
    for (i, (a, b)) in per_item_concat
        .iter()
        .zip(batched_concat.iter())
        .enumerate()
    {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = i;
        }
        if diff > 5e-3 {
            mismatches += 1;
        }
    }

    let item = argmax / (NH * HD);
    let head = (argmax % (NH * HD)) / HD;
    let dim = argmax % HD;

    println!(
        "flash_attn batched eq:\n  \
         max_abs_diff={max_abs:.3e}\n  \
         worst @ item={item} head={head} dim={dim}: per_item={} batched={}\n  \
         mismatches > 5e-3: {mismatches} / {} elements\n  \
         per_item[0..5]: {:?}\n  \
         batched[0..5]: {:?}",
        per_item_concat[argmax],
        batched_concat[argmax],
        per_item_concat.len(),
        &per_item_concat[..5],
        &batched_concat[..5],
    );

    assert!(
        max_abs < 5e-3,
        "max_abs_diff {max_abs} exceeds 5e-3 — batched flash_attn diverges"
    );
}
