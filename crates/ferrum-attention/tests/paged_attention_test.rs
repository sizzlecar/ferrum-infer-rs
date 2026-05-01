//! Correctness tests for the paged-KV decode attention kernel.
//!
//! Approach: build a contiguous KV with known data, run the existing
//! `flash_attn_decode_f32` kernel as the reference, then re-arrange the
//! same data into a paged cache and run `flash_attn_decode_paged_f32`,
//! and assert the outputs match within tolerance. This isolates the
//! block-table indirection — the math (online softmax, cross-simdgroup
//! combine) is identical between the two kernels.

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_attention::metal::pipelines::{
    MetalPipelines, PagedAttnDispatchParams, PagedAttnQLayout,
};
use metal::{Device, MTLResourceOptions};
use std::ffi::c_void;

fn assert_close(a: &[f32], b: &[f32], atol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    assert!(
        max_diff < atol,
        "{label}: max diff {max_diff} at idx {max_idx} (a={}, b={}), atol={atol}",
        a[max_idx],
        b[max_idx]
    );
}

fn make_buffer(device: &Device, data_bytes: &[u8]) -> metal::Buffer {
    device.new_buffer_with_data(
        data_bytes.as_ptr() as *const c_void,
        data_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buffer_from_f32(device: &Device, data: &[f32]) -> metal::Buffer {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    make_buffer(device, bytes)
}

fn buffer_from_u32(device: &Device, data: &[u32]) -> metal::Buffer {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    make_buffer(device, bytes)
}

fn read_f32(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

/// Reference: run the contiguous-KV decode kernel and read its output.
fn run_contiguous(
    pipes: &MetalPipelines,
    q: &[f32],
    k_contig: &[f32],
    v_contig: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Vec<f32> {
    let device = &pipes.device;
    let q_buf = buffer_from_f32(device, q);
    let k_buf = buffer_from_f32(device, k_contig);
    let v_buf = buffer_from_f32(device, v_contig);
    let o_buf = device.new_buffer((q.len() * 4) as u64, MTLResourceOptions::StorageModeShared);

    let cmd = pipes.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    // Match the FlashAttnParams layout in flash_attn.metal verbatim.
    #[repr(C)]
    struct P {
        batch: i32,
        num_heads: i32,
        num_kv_heads: i32,
        q_len: i32,
        kv_len: i32,
        head_dim: i32,
        scale: f32,
        causal: i32,
        pos_offset: i32,
        kv_seq_stride: i32,
        sliding_window: i32,
    }
    let params = P {
        batch: 1,
        num_heads: num_heads as i32,
        num_kv_heads: num_kv_heads as i32,
        q_len: 1,
        kv_len: kv_len as i32,
        head_dim: head_dim as i32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        causal: 1,
        pos_offset: kv_len as i32 - 1,
        kv_seq_stride: 0,
        sliding_window: 0,
    };

    enc.set_compute_pipeline_state(pipes.pipeline("flash_attn_decode_f32"));
    enc.set_buffer(0, Some(&q_buf), 0);
    enc.set_buffer(1, Some(&k_buf), 0);
    enc.set_buffer(2, Some(&v_buf), 0);
    enc.set_buffer(3, Some(&o_buf), 0);
    enc.set_bytes(
        4,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );
    let grid = metal::MTLSize::new(1, num_heads as u64, 1);
    let tg = metal::MTLSize::new(32, 32, 1);
    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    read_f32(&o_buf, q.len())
}

/// Build a paged KV cache from a contiguous one.
///
/// Contiguous layout: `[num_kv_heads, kv_len, head_dim]`
/// Paged layout: `[num_blocks, num_kv_heads, block_size, head_dim]`
///
/// Returns (paged_buf, block_table) where block_table[i] = physical
/// index of the i-th logical block. We assign blocks in-order for the
/// test; a real allocator would interleave.
fn pack_into_paged(
    contig: &[f32],
    num_kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    block_size: usize,
) -> (Vec<f32>, Vec<u32>) {
    let num_blocks = kv_len.div_ceil(block_size);
    let mut paged = vec![0.0f32; num_blocks * num_kv_heads * block_size * head_dim];
    let mut table = Vec::with_capacity(num_blocks);
    for logical_block in 0..num_blocks {
        let physical_block = logical_block; // identity assignment for the test
        table.push(physical_block as u32);
        for slot in 0..block_size {
            let token_idx = logical_block * block_size + slot;
            if token_idx >= kv_len {
                break;
            }
            for kvh in 0..num_kv_heads {
                let src_off = kvh * kv_len * head_dim + token_idx * head_dim;
                let dst_off = physical_block * num_kv_heads * block_size * head_dim
                    + kvh * block_size * head_dim
                    + slot * head_dim;
                paged[dst_off..dst_off + head_dim]
                    .copy_from_slice(&contig[src_off..src_off + head_dim]);
            }
        }
    }
    (paged, table)
}

fn run_paged(
    pipes: &MetalPipelines,
    q: &[f32],
    k_paged: &[f32],
    v_paged: &[f32],
    block_table: &[u32],
    context_len: u32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    let device = &pipes.device;
    let q_buf = buffer_from_f32(device, q);
    let k_buf = buffer_from_f32(device, k_paged);
    let v_buf = buffer_from_f32(device, v_paged);
    let o_buf = device.new_buffer((q.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
    let bt_buf = buffer_from_u32(device, block_table);
    let cl_buf = buffer_from_u32(device, &[context_len]);

    let cmd = pipes.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    pipes.paged_decode_attention_on_encoder(
        enc,
        &q_buf,
        &k_buf,
        &v_buf,
        &o_buf,
        &bt_buf,
        &cl_buf,
        &PagedAttnDispatchParams {
            num_seqs: 1,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks_per_seq: block_table.len(),
            q_len: 1,
            q_layout: PagedAttnQLayout::TokenMajor,
        },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    read_f32(&o_buf, q.len())
}

#[test]
fn paged_decode_matches_contiguous_small() {
    // 8 query heads / 2 kv heads (GQA factor 4), 50 KV positions,
    // block_size=16 → 4 blocks (the last is partial).
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 128;
    let kv_len = 50;
    let block_size = 16;

    let q: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| (i as f32 * 0.013).sin() * 0.2)
        .collect();
    let k: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
        .map(|i| (i as f32 * 0.0091).cos() * 0.25)
        .collect();
    let v: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
        .map(|i| (i as f32 * 0.0117 + 1.7).sin() * 0.25)
        .collect();

    let device = Device::system_default().expect("no Metal device");
    let pipes = MetalPipelines::new(&device);

    let out_contig = run_contiguous(
        &pipes,
        &q,
        &k,
        &v,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
    );

    let (k_paged, table) = pack_into_paged(&k, num_kv_heads, kv_len, head_dim, block_size);
    let (v_paged, _) = pack_into_paged(&v, num_kv_heads, kv_len, head_dim, block_size);
    let out_paged = run_paged(
        &pipes,
        &q,
        &k_paged,
        &v_paged,
        &table,
        kv_len as u32,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    );

    // Both kernels do f32 online softmax with the same loop schedule
    // (same SDPA_BN=32 simdgroup distribution); only the K/V address
    // computation differs. Numerical results should match to high
    // precision.
    assert_close(&out_contig, &out_paged, 1e-4, "paged vs contiguous (small)");
}

#[test]
fn paged_decode_matches_contiguous_long_context() {
    // Long-context regime: kv_len=2049 > 32 (SDPA_BN), each simdgroup
    // walks ~64 KV positions. Block size 16 → 129 blocks, last partial.
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 128;
    let kv_len = 2049;
    let block_size = 16;

    let q: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| (i as f32 * 0.005 + 0.3).sin() * 0.15)
        .collect();
    let k: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
        .map(|i| (i as f32 * 0.00071).cos() * 0.18)
        .collect();
    let v: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
        .map(|i| (i as f32 * 0.00097 + 2.1).sin() * 0.18)
        .collect();

    let device = Device::system_default().expect("no Metal device");
    let pipes = MetalPipelines::new(&device);

    let out_contig = run_contiguous(
        &pipes,
        &q,
        &k,
        &v,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
    );

    let (k_paged, table) = pack_into_paged(&k, num_kv_heads, kv_len, head_dim, block_size);
    let (v_paged, _) = pack_into_paged(&v, num_kv_heads, kv_len, head_dim, block_size);
    let out_paged = run_paged(
        &pipes,
        &q,
        &k_paged,
        &v_paged,
        &table,
        kv_len as u32,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    );

    // Slightly looser tolerance for long context: f32 sumexp
    // accumulation accumulates more rounding error.
    assert_close(
        &out_contig,
        &out_paged,
        5e-4,
        "paged vs contiguous (long ctx)",
    );
}

#[test]
fn paged_decode_handles_shuffled_block_table() {
    // The block_table doesn't have to be identity; that's the whole
    // point of paged caching. Build a contiguous KV, scatter into a
    // larger paged pool with NON-identity block assignment, run paged
    // attention, compare.
    let num_heads: usize = 8;
    let num_kv_heads: usize = 2;
    let head_dim: usize = 128;
    let kv_len: usize = 50;
    let block_size: usize = 16;
    let num_logical_blocks = kv_len.div_ceil(block_size); // 4
                                                          // Allocate a bigger physical pool so we can shuffle assignments.
    let num_physical_blocks = 8;
    // Permutation: logical 0->5, 1->2, 2->7, 3->1.
    let permutation = [5u32, 2, 7, 1];
    assert_eq!(permutation.len(), num_logical_blocks);

    let q: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| (i as f32 * 0.029).sin() * 0.2)
        .collect();
    let k: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
        .map(|i| (i as f32 * 0.011).cos() * 0.2)
        .collect();
    let v: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
        .map(|i| (i as f32 * 0.013 + 0.5).sin() * 0.2)
        .collect();

    // Build pool with the given permutation.
    let mut k_pool = vec![0.0f32; num_physical_blocks * num_kv_heads * block_size * head_dim];
    let mut v_pool = vec![0.0f32; num_physical_blocks * num_kv_heads * block_size * head_dim];
    for (logical_block, &physical_block) in permutation.iter().enumerate() {
        for slot in 0..block_size {
            let token_idx = logical_block * block_size + slot;
            if token_idx >= kv_len {
                break;
            }
            for kvh in 0..num_kv_heads {
                let src_off = kvh * kv_len * head_dim + token_idx * head_dim;
                let dst_off = physical_block as usize * num_kv_heads * block_size * head_dim
                    + kvh * block_size * head_dim
                    + slot * head_dim;
                k_pool[dst_off..dst_off + head_dim]
                    .copy_from_slice(&k[src_off..src_off + head_dim]);
                v_pool[dst_off..dst_off + head_dim]
                    .copy_from_slice(&v[src_off..src_off + head_dim]);
            }
        }
    }

    let device = Device::system_default().expect("no Metal device");
    let pipes = MetalPipelines::new(&device);

    let out_contig = run_contiguous(
        &pipes,
        &q,
        &k,
        &v,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_len,
    );
    let out_paged = run_paged(
        &pipes,
        &q,
        &k_pool,
        &v_pool,
        &permutation,
        kv_len as u32,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
    );

    assert_close(
        &out_contig,
        &out_paged,
        1e-4,
        "paged vs contiguous (shuffled block table)",
    );
}
