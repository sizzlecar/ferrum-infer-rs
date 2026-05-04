//! Equivalence test: per-item `qk_norm_rope_transpose_f16` (called m
//! times) vs `qk_norm_rope_batched_decode_f16` (one call for m items)
//! must produce numerically equivalent output for the same Q input.
//!
//! Bisects the m≥2 batched_decode garbage bug: c=4 produces gibberish
//! tokens; FERRUM_FORCE_PER_ITEM=1 in the engine restores correct
//! output. This test isolates the batched qkr kernel without spinning
//! up a model server (avoids the 4-min model-load+rebuild cycle).
//!
//! Layout note. Per-item kernel TRANSPOSES:
//!   input:  [tokens, heads, head_dim] (token-major)
//!   output: [heads, tokens, head_dim] (head-major)
//! Batched kernel does NOT transpose:
//!   input:  [m, heads, head_dim]
//!   output: [m, heads, head_dim] (item-major)
//!
//! For tokens=1 (per-item case), head-major [heads, 1, hd] and
//! item-major [1, heads, hd] are byte-identical for the i-th item.
//! Per-item kernel called once with tokens=1 outputs HEADS*HD floats
//! that match the same offsets as batched output for that item.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda --release \
//!       --test qk_norm_rope_batched_eq -- --nocapture

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{cuda::CudaBackend, Backend};

const M: usize = 4;
const HEADS: usize = 32;
const HD: usize = 128;
const HALF_D: usize = HD / 2;
const MAX_SEQ: usize = 64;
const EPS: f32 = 1e-6;
const POSITION: usize = 4;

fn det_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f32) / (u64::MAX as f32) * 2.0 - 1.0
        })
        .map(|x| x * 0.3)
        .collect()
}

fn make_rope_table(seed: u64) -> Vec<f32> {
    // [MAX_SEQ, HALF_D] — values in [-1, 1] like real cos/sin
    det_f32(seed, MAX_SEQ * HALF_D)
}

#[test]
fn qkr_batched_matches_per_item_mode_2() {
    // mode=2: RoPE only (Llama / Mistral, no QK-norm)
    run_eq(2, "Llama-style (mode=2, RoPE only)");
}

#[test]
fn qkr_batched_matches_per_item_mode_1() {
    // mode=1: per-head RMSNorm + RoPE (Qwen3 with QK-norm)
    run_eq(1, "Qwen3-style (mode=1, QK-norm + RoPE)");
}

fn run_eq(mode: i32, label: &str) {
    let mut ctx = CudaBackend::new_context();

    // Inputs (deterministic random):
    //   input  [m, heads, head_dim] — same for both kernels
    //   norm_w [head_dim]
    //   cos    [max_seq, head_dim/2]
    //   sin    [max_seq, head_dim/2]
    let input_h = det_f32(1, M * HEADS * HD);
    let norm_w_h = det_f32(2, HD);
    let cos_h = make_rope_table(3);
    let sin_h = make_rope_table(4);

    let input_dev = CudaBackend::from_slice(&input_h);
    let norm_w_dev = CudaBackend::from_slice(&norm_w_h);
    let cos_dev = CudaBackend::from_slice(&cos_h);
    let sin_dev = CudaBackend::from_slice(&sin_h);

    let mut out_batched_dev = CudaBackend::alloc(M * HEADS * HD);

    // Positions buffer (device i32 / u32 wire-compatible).
    let positions_h: Vec<u32> = vec![POSITION as u32; M];
    let mut positions_dev = CudaBackend::alloc_u32(M);
    CudaBackend::write_u32(&mut ctx, &mut positions_dev, &positions_h);

    // ── Per-item: m calls, each with tokens=1, copying item i's slice ─
    let mut per_item_scratch = CudaBackend::alloc(HEADS * HD);
    let mut input_single_scratch = CudaBackend::alloc(HEADS * HD);
    let mut per_item_concat = vec![0.0f32; M * HEADS * HD];
    for i in 0..M {
        CudaBackend::copy_slice(
            &mut ctx,
            &input_dev,
            i * HEADS * HD,
            &mut input_single_scratch,
            0,
            HEADS * HD,
        );
        CudaBackend::qk_norm_rope(
            &mut ctx,
            &input_single_scratch,
            &norm_w_dev,
            &cos_dev,
            &sin_dev,
            &mut per_item_scratch,
            1, // tokens
            HEADS,
            HD,
            POSITION,
            EPS,
            mode,
        );
        CudaBackend::sync(&mut ctx);
        let item_h = CudaBackend::to_vec(&per_item_scratch, HEADS * HD);
        per_item_concat[i * HEADS * HD..(i + 1) * HEADS * HD].copy_from_slice(&item_h);
    }

    // ── Batched: 1 call for m items ──────────────────────────────────
    CudaBackend::qk_norm_rope_batched_per_item(
        &mut ctx,
        &input_dev,
        &norm_w_dev,
        &cos_dev,
        &sin_dev,
        &mut out_batched_dev,
        &positions_dev,
        M,
        HEADS,
        HD,
        EPS,
        mode,
    )
    .expect("qk_norm_rope_batched_per_item");
    CudaBackend::sync(&mut ctx);
    let batched_concat = CudaBackend::to_vec(&out_batched_dev, M * HEADS * HD);

    // ── Compare elementwise ──────────────────────────────────────────
    assert_eq!(per_item_concat.len(), batched_concat.len());

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut argmax = 0usize;
    let mut mismatches = 0usize;
    for (idx, (a, b)) in per_item_concat.iter().zip(batched_concat.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            argmax = idx;
        }
        let denom = a.abs().max(b.abs()).max(1e-12);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        if diff > 1e-2 {
            mismatches += 1;
        }
    }

    let item = argmax / (HEADS * HD);
    let head = (argmax % (HEADS * HD)) / HD;
    let dim = argmax % HD;

    println!(
        "[{label}] eq stats:\n  \
         max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}\n  \
         worst @ item={item} head={head} dim={dim}: per_item={} batched={}\n  \
         mismatches > 1e-2: {mismatches} / {} elements",
        per_item_concat[argmax],
        batched_concat[argmax],
        per_item_concat.len()
    );

    // Tolerance: same fp32 math, fp16 storage. A few ULPs.
    assert!(
        max_abs < 5e-3,
        "[{label}] max_abs_diff {max_abs} exceeds 5e-3 tolerance — \
         batched kernel diverges from per-item"
    );
}
