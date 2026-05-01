//! Correctness tests for the paged-KV variant of split_qkv_norm_rope.
//!
//! Approach: build a fused QKV input + RoPE tables, run the EXISTING
//! contiguous `split_qkv_norm_rope_into_cache` to populate a contiguous
//! KV cache, then run the NEW paged variant with the same inputs to
//! populate a paged pool. Walk the paged pool via the block_table and
//! verify each (head, position) slice equals the contiguous cache's
//! corresponding slice. This isolates the paged-dst computation from
//! the math (norm + RoPE), which is identical between the two kernels.

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_attention::metal::pipelines::MetalPipelines;
use metal::{Device, MTLResourceOptions};
use std::ffi::c_void;

fn buffer_from_f32(device: &Device, data: &[f32]) -> metal::Buffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buffer_from_u32(device: &Device, data: &[u32]) -> metal::Buffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn empty_buffer(device: &Device, num_floats: usize) -> metal::Buffer {
    device.new_buffer(
        (num_floats * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_f32(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

#[derive(Clone, Copy)]
struct Cfg {
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    cache_len: usize,
    cache_capacity: usize,
    block_size: usize,
    max_seq_len: usize,
    qk_mode: i32, // 0 = no qk-norm, 1 = qk-norm enabled
    eps: f32,
}

fn make_input_data(c: &Cfg) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let qkv_dim = c.q_heads * c.head_dim + 2 * c.kv_heads * c.head_dim;
    let qkv: Vec<f32> = (0..c.tokens * qkv_dim)
        .map(|i| (i as f32 * 0.011 + 0.2).sin() * 0.3)
        .collect();
    let q_norm_w: Vec<f32> = (0..c.head_dim)
        .map(|i| 1.0 + (i as f32 * 0.05).cos() * 0.1)
        .collect();
    let k_norm_w: Vec<f32> = (0..c.head_dim)
        .map(|i| 1.0 + (i as f32 * 0.07).sin() * 0.1)
        .collect();
    // RoPE tables: cos/sin for each position up to cache_len + tokens.
    let half_d = c.head_dim / 2;
    let cos: Vec<f32> = (0..c.max_seq_len * half_d)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let sin: Vec<f32> = (0..c.max_seq_len * half_d)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    (qkv, q_norm_w, k_norm_w, cos, sin)
}

/// Run the existing contiguous `split_qkv_norm_rope_into_cache` and
/// return (q_head_major, cache_k, cache_v) buffers.
#[allow(clippy::too_many_arguments)]
fn run_contiguous(
    pipes: &MetalPipelines,
    c: &Cfg,
    qkv: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = &pipes.device;
    let qkv_buf = buffer_from_f32(device, qkv);
    let q_norm_buf = buffer_from_f32(device, q_norm_w);
    let k_norm_buf = buffer_from_f32(device, k_norm_w);
    let cos_buf = buffer_from_f32(device, cos);
    let sin_buf = buffer_from_f32(device, sin);
    let q_out = empty_buffer(device, c.q_heads * c.tokens * c.head_dim);
    let cache_k = empty_buffer(device, c.kv_heads * c.cache_capacity * c.head_dim);
    let cache_v = empty_buffer(device, c.kv_heads * c.cache_capacity * c.head_dim);

    let cmd = pipes.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    pipes.split_qkv_norm_rope_into_cache(
        enc,
        &qkv_buf,
        &q_norm_buf,
        &k_norm_buf,
        &cos_buf,
        &sin_buf,
        &q_out,
        &cache_k,
        &cache_v,
        c.tokens,
        c.q_heads,
        c.kv_heads,
        c.head_dim,
        c.cache_len, // pos_offset = cache_len
        c.eps,
        c.qk_mode,
        c.cache_len,
        c.cache_capacity,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    (
        read_f32(&q_out, c.q_heads * c.tokens * c.head_dim),
        read_f32(&cache_k, c.kv_heads * c.cache_capacity * c.head_dim),
        read_f32(&cache_v, c.kv_heads * c.cache_capacity * c.head_dim),
    )
}

/// Run the NEW paged `split_qkv_norm_rope_into_paged_cache` and return
/// (q_head_major, k_pool, v_pool, block_table). The pool is sized for
/// `num_blocks` physical blocks; we use a non-identity assignment to
/// stress the indirection logic.
#[allow(clippy::too_many_arguments)]
fn run_paged(
    pipes: &MetalPipelines,
    c: &Cfg,
    qkv: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    cos: &[f32],
    sin: &[f32],
    block_table: &[u32],
    num_physical_blocks: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = &pipes.device;
    let qkv_buf = buffer_from_f32(device, qkv);
    let q_norm_buf = buffer_from_f32(device, q_norm_w);
    let k_norm_buf = buffer_from_f32(device, k_norm_w);
    let cos_buf = buffer_from_f32(device, cos);
    let sin_buf = buffer_from_f32(device, sin);
    let q_out = empty_buffer(device, c.q_heads * c.tokens * c.head_dim);
    let pool_floats = num_physical_blocks * c.kv_heads * c.block_size * c.head_dim;
    let cache_k = empty_buffer(device, pool_floats);
    let cache_v = empty_buffer(device, pool_floats);
    let bt_buf = buffer_from_u32(device, block_table);

    let cmd = pipes.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    pipes.split_qkv_norm_rope_into_paged_cache(
        enc,
        &qkv_buf,
        &q_norm_buf,
        &k_norm_buf,
        &cos_buf,
        &sin_buf,
        &q_out,
        &cache_k,
        &cache_v,
        &bt_buf,
        c.tokens,
        c.q_heads,
        c.kv_heads,
        c.head_dim,
        c.cache_len, // pos_offset = cache_len
        c.eps,
        c.qk_mode,
        c.cache_len,
        c.block_size,
        block_table.len(),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    (
        read_f32(&q_out, c.q_heads * c.tokens * c.head_dim),
        read_f32(&cache_k, pool_floats),
        read_f32(&cache_v, pool_floats),
    )
}

/// Walk the contiguous and paged caches at the same logical positions
/// and assert per-element equality. Returns the maximum absolute diff
/// observed (for diagnostic).
fn compare_caches(
    c: &Cfg,
    contig_k: &[f32],
    contig_v: &[f32],
    paged_k: &[f32],
    paged_v: &[f32],
    block_table: &[u32],
    label: &str,
) {
    let mut max_diff = 0.0f32;
    let mut max_loc = (0usize, 0usize, 0usize, 0usize);
    for kvh in 0..c.kv_heads {
        for tok in 0..c.tokens {
            let global_slot = c.cache_len + tok;
            let logical_block = global_slot / c.block_size;
            let slot_in_block = global_slot % c.block_size;
            let physical_block = block_table[logical_block] as usize;
            for d in 0..c.head_dim {
                let contig_off = kvh * c.cache_capacity * c.head_dim
                    + global_slot * c.head_dim
                    + d;
                let paged_off = physical_block * c.kv_heads * c.block_size * c.head_dim
                    + kvh * c.block_size * c.head_dim
                    + slot_in_block * c.head_dim
                    + d;
                let dk = (contig_k[contig_off] - paged_k[paged_off]).abs();
                let dv = (contig_v[contig_off] - paged_v[paged_off]).abs();
                let dmax = dk.max(dv);
                if dmax > max_diff {
                    max_diff = dmax;
                    max_loc = (kvh, tok, d, physical_block);
                }
            }
        }
    }
    assert!(
        max_diff < 1e-5,
        "{label}: max diff {max_diff} at (kvh={}, tok={}, d={}, phys_block={}); paged write doesn't match contiguous reference",
        max_loc.0,
        max_loc.1,
        max_loc.2,
        max_loc.3,
    );
}

#[test]
fn paged_kv_append_matches_contiguous_qk_norm() {
    // Qwen3-style: QK-norm enabled (qk_mode=1).
    let c = Cfg {
        tokens: 5,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
        cache_len: 11,
        cache_capacity: 64,
        block_size: 16,
        max_seq_len: 128,
        qk_mode: 1,
        eps: 1e-6,
    };

    let (qkv, qn, kn, cos, sin) = make_input_data(&c);
    let device = Device::system_default().expect("no Metal device");
    let pipes = MetalPipelines::new(&device);

    // Contiguous reference.
    let (q_contig, k_contig, v_contig) = run_contiguous(&pipes, &c, &qkv, &qn, &kn, &cos, &sin);

    // Paged variant with non-identity block table.
    // cache_len=11, tokens=5 → writes to global slots [11..16). With
    // block_size=16, that's logical block 0 (slots 11-15). So we only
    // need block_table[0] populated, but allocate room for more.
    let num_physical_blocks = 8;
    let block_table: Vec<u32> = (0..4).map(|i| ((i + 3) % num_physical_blocks) as u32).collect();
    let (q_paged, k_paged, v_paged) = run_paged(
        &pipes,
        &c,
        &qkv,
        &qn,
        &kn,
        &cos,
        &sin,
        &block_table,
        num_physical_blocks,
    );

    // Q output should be byte-identical (same head-major scratch layout).
    for (i, (a, b)) in q_contig.iter().zip(q_paged.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Q output mismatch at idx {i}: contig={a}, paged={b}",
        );
    }

    // K/V should match per-position via block_table indirection.
    compare_caches(
        &c,
        &k_contig,
        &v_contig,
        &k_paged,
        &v_paged,
        &block_table,
        "paged kvc (qk_norm)",
    );
}

#[test]
fn paged_kv_append_matches_contiguous_no_qk_norm() {
    // Llama-3.x-style: qk_mode=2 (RoPE only, no QK-norm).
    let c = Cfg {
        tokens: 7,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
        cache_len: 33,
        cache_capacity: 64,
        block_size: 16,
        max_seq_len: 128,
        qk_mode: 2,
        eps: 1e-6,
    };

    let (qkv, qn, kn, cos, sin) = make_input_data(&c);
    let device = Device::system_default().expect("no Metal device");
    let pipes = MetalPipelines::new(&device);

    let (q_contig, k_contig, v_contig) = run_contiguous(&pipes, &c, &qkv, &qn, &kn, &cos, &sin);

    // tokens 33..40 span logical blocks 2 (slots 32-39) and possibly 3.
    // 40 is exclusive. 33→ block 2 slot 1, 39 → block 2 slot 7. All in block 2.
    let num_physical_blocks = 6;
    let block_table: Vec<u32> = vec![5, 1, 4, 0, 2, 3]; // permutation
    let (q_paged, k_paged, v_paged) = run_paged(
        &pipes,
        &c,
        &qkv,
        &qn,
        &kn,
        &cos,
        &sin,
        &block_table,
        num_physical_blocks,
    );

    for (i, (a, b)) in q_contig.iter().zip(q_paged.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Q output mismatch at idx {i}: contig={a}, paged={b}",
        );
    }
    compare_caches(
        &c,
        &k_contig,
        &v_contig,
        &k_paged,
        &v_paged,
        &block_table,
        "paged kvc (no qk_norm)",
    );
}

#[test]
fn paged_kv_append_spans_block_boundary() {
    // Stress: tokens span TWO logical blocks. cache_len=14, tokens=5
    // writes to global slots [14..19). With block_size=16: block 0
    // gets slots 14-15 (2 tokens), block 1 gets slots 16-18 (3 tokens).
    let c = Cfg {
        tokens: 5,
        q_heads: 8,
        kv_heads: 2,
        head_dim: 128,
        cache_len: 14,
        cache_capacity: 64,
        block_size: 16,
        max_seq_len: 128,
        qk_mode: 1,
        eps: 1e-6,
    };

    let (qkv, qn, kn, cos, sin) = make_input_data(&c);
    let device = Device::system_default().expect("no Metal device");
    let pipes = MetalPipelines::new(&device);

    let (q_contig, k_contig, v_contig) = run_contiguous(&pipes, &c, &qkv, &qn, &kn, &cos, &sin);

    let num_physical_blocks = 8;
    // Logical block 0 → physical 5; logical block 1 → physical 2.
    let block_table: Vec<u32> = vec![5, 2, 7, 1];
    let (q_paged, k_paged, v_paged) = run_paged(
        &pipes,
        &c,
        &qkv,
        &qn,
        &kn,
        &cos,
        &sin,
        &block_table,
        num_physical_blocks,
    );

    for (i, (a, b)) in q_contig.iter().zip(q_paged.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Q output mismatch at idx {i}: contig={a}, paged={b}",
        );
    }
    compare_caches(
        &c,
        &k_contig,
        &v_contig,
        &k_paged,
        &v_paged,
        &block_table,
        "paged kvc (cross-block)",
    );
}
