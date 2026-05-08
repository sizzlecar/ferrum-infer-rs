//! Equivalence test for the GPU-side `moe_align_block_size` kernel.
//! Builds a synthetic per-pair expert assignment, runs the kernel,
//! D2Hs the outputs, and compares against a Rust reference impl.
//!
//! Run with:
//!   cargo test -p ferrum-kernels --features cuda,marlin --release \
//!       --test moe_align_block_size_eq -- --nocapture
//!
//! Skipped (no tests compiled) without the cuda feature.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice};
use ferrum_kernels::backend::{cuda::CudaState, Backend};

const NUM_EXPERTS: usize = 128;
const TOP_K: usize = 8;
const BATCH: usize = 32;
const BLOCK_SIZE: usize = 16; // matches Marlin's M_BLOCK

fn reference_align(
    expert_ids: &[i32],
    num_experts: usize,
    block_size: usize,
) -> (Vec<i32>, Vec<i32>, i32) {
    // Reference: count → ceil to block_size → cumsum → fill sorted.
    let n_pairs = expert_ids.len();
    let sentinel = n_pairs as i32;

    let mut counts = vec![0i32; num_experts];
    for &e in expert_ids {
        if e >= 0 && (e as usize) < num_experts {
            counts[e as usize] += 1;
        }
    }
    let mut counts_padded = vec![0i32; num_experts];
    for e in 0..num_experts {
        counts_padded[e] = ((counts[e] + block_size as i32 - 1) / block_size as i32)
            * block_size as i32;
    }
    let mut offsets = vec![0i32; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts_padded[e];
    }
    let total_post_pad = offsets[num_experts];

    let mut sorted = vec![sentinel; total_post_pad as usize];
    let mut cursors: Vec<usize> = offsets[..num_experts].iter().map(|&v| v as usize).collect();
    for (p, &e) in expert_ids.iter().enumerate() {
        if e >= 0 && (e as usize) < num_experts {
            let slot = cursors[e as usize];
            cursors[e as usize] += 1;
            sorted[slot] = p as i32;
        }
    }

    let total_blocks = (total_post_pad / block_size as i32) as usize;
    let mut block_ids = vec![0i32; total_blocks];
    for b in 0..total_blocks {
        let row = b as i32 * block_size as i32;
        for ei in 0..num_experts {
            if offsets[ei] <= row && row < offsets[ei + 1] {
                block_ids[b] = ei as i32;
                break;
            }
        }
    }

    (sorted, block_ids, total_post_pad)
}

#[test]
fn moe_align_matches_reference_qwen3moe_shape() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0xABCD);
    let n_pairs = BATCH * TOP_K;
    let expert_ids_host: Vec<i32> = (0..n_pairs)
        .map(|_| rng.gen_range(0..NUM_EXPERTS as i32))
        .collect();

    // Sentinel padding = num_experts × ceil(n_pairs / block_size) × block_size
    // is the worst-case upper bound; in practice total_post_pad is much
    // smaller. We size to that upper bound.
    let sorted_max =
        num_experts_max_padded(n_pairs, NUM_EXPERTS, BLOCK_SIZE);

    let (ref_sorted, ref_block_ids, ref_total) =
        reference_align(&expert_ids_host, NUM_EXPERTS, BLOCK_SIZE);

    // GPU side
    let mut ctx = <ferrum_kernels::backend::cuda::CudaBackend as Backend>::new_context();
    // upload expert_ids as i32 buffer (use alloc_u32 + write_u32)
    let mut expert_ids_dev =
        <ferrum_kernels::backend::cuda::CudaBackend as Backend>::alloc_u32(n_pairs);
    let expert_ids_u32: Vec<u32> = expert_ids_host.iter().map(|&v| v as u32).collect();
    <ferrum_kernels::backend::cuda::CudaBackend as Backend>::write_u32(
        &mut ctx,
        &mut expert_ids_dev,
        &expert_ids_u32,
    );

    let mut sorted_dev =
        <ferrum_kernels::backend::cuda::CudaBackend as Backend>::alloc_u32(sorted_max);
    let mut block_ids_dev = <ferrum_kernels::backend::cuda::CudaBackend as Backend>::alloc_u32(
        sorted_max / BLOCK_SIZE,
    );
    let mut total_dev =
        <ferrum_kernels::backend::cuda::CudaBackend as Backend>::alloc_u32(1);

    <ferrum_kernels::backend::cuda::CudaBackend as Backend>::moe_align_block_size(
        &mut ctx,
        &expert_ids_dev,
        &mut sorted_dev,
        &mut block_ids_dev,
        &mut total_dev,
        n_pairs,
        NUM_EXPERTS,
        BLOCK_SIZE,
        sorted_max,
    )
    .expect("moe_align_block_size launch");

    // D2H the outputs
    let stream = ctx.stream.clone();
    let total_view = unsafe {
        sorted_dev
            .transmute::<i32>(sorted_max)
            .expect("sorted transmute")
    };
    let mut sorted_host = vec![0i32; sorted_max];
    stream
        .memcpy_dtoh(&total_view, sorted_host.as_mut_slice())
        .expect("dtoh sorted");

    let block_view = unsafe {
        block_ids_dev
            .transmute::<i32>(sorted_max / BLOCK_SIZE)
            .expect("block_ids transmute")
    };
    let mut block_ids_host = vec![0i32; sorted_max / BLOCK_SIZE];
    stream
        .memcpy_dtoh(&block_view, block_ids_host.as_mut_slice())
        .expect("dtoh block_ids");

    let total_view2 = unsafe { total_dev.transmute::<i32>(1).expect("total transmute") };
    let mut total_host = vec![0i32; 1];
    stream
        .memcpy_dtoh(&total_view2, total_host.as_mut_slice())
        .expect("dtoh total");
    stream.synchronize().expect("sync");

    assert_eq!(
        total_host[0], ref_total,
        "total_tokens_post_pad mismatch: GPU={} ref={}",
        total_host[0], ref_total
    );

    // Compare sorted ids prefix [0, total_post_pad) — the rest is
    // sentinel padding which the reference also fills with `n_pairs`.
    let used = ref_total as usize;
    assert_eq!(
        &sorted_host[..used],
        &ref_sorted[..],
        "sorted_token_ids prefix mismatch"
    );

    // Compare block_ids for the active prefix.
    let total_blocks = used / BLOCK_SIZE;
    assert_eq!(
        &block_ids_host[..total_blocks],
        &ref_block_ids[..],
        "block_ids mismatch"
    );

    eprintln!(
        "moe_align ok: n_pairs={n_pairs} total_post_pad={ref_total} \
         total_blocks={total_blocks}"
    );
}

fn num_experts_max_padded(n_pairs: usize, num_experts: usize, block_size: usize) -> usize {
    // Upper bound: every expert padded individually to block_size, regardless
    // of count. = num_experts × ceil(n_pairs / block_size) × block_size.
    let max_blocks_per_expert = (n_pairs + block_size - 1) / block_size;
    num_experts * max_blocks_per_expert * block_size
}
