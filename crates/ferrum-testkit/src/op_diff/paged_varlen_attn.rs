//! `paged_varlen_attention` op-diff harness — **PARTIAL: planning stub**.
//!
//! The full op needs:
//!   - cu_seqlens (cumulative seq lengths, `[batch+1]` u32)
//!   - paged K/V cache (`[num_blocks, num_heads, block_size, head_dim]`)
//!   - block tables (`[batch, max_blocks_per_seq]` i32)
//!   - Q tensor (`[total_tokens, num_heads, head_dim]`)
//!   - softmax_scale, num_kv_heads (for GQA)
//!
//! The CPU reference impl exists in `crates/ferrum-kernels/src/backend/cpu.rs`
//! but reconstructing valid paged tables from scratch is ~200 lines of
//! setup. Punting for a follow-up: needs a shared `paged_kv_fixture()`
//! helper in `ferrum-testkit::fixtures` to produce a valid block_table
//! + cu_seqlens pair from a sequence length list, which can be reused
//! by `paged_decode_attention` op-diff too.
//!
//! See PR #206's "Known gaps" section.

#![allow(dead_code)]

pub struct PagedVarlenAttnOp {
    pub batch: usize,
    pub seq_lens: Vec<usize>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
}

// impl OpUnderTest for PagedVarlenAttnOp — pending paged_kv_fixture helper.
