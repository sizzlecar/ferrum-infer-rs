//! Source-built FlashAttention-2 bridge.
//!
//! This module is compiled only with the `fa2-source` cargo feature. The C ABI
//! symbol is linked from a static library built by `build.rs` from the Ferrum
//! FA2 shim plus vendored FlashAttention source templates, so runtime does not
//! need vLLM, Torch, Python, or a user-provided shim `.so`.

use std::ffi::{c_char, c_int, c_void};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use ferrum_types::Result;
use half::f16;

use super::fa2_ffi::call_paged_varlen_fn;

extern "C" {
    fn ferrum_fa2_paged_varlen_fwd(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        lse: *mut c_void,
        cu_seqlens_q: *const c_void,
        seq_lens: *const c_void,
        block_tables: *const c_void,
        num_seqs: c_int,
        total_q_tokens: c_int,
        max_q_len: c_int,
        max_kv_len: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        stream: *mut c_void,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> c_int;
}

pub(crate) struct Fa2SourcePagedVarlenArgs<'a> {
    pub(crate) stream: &'a Arc<CudaStream>,
    pub(crate) q: &'a CudaSlice<f16>,
    pub(crate) k_pool: &'a CudaSlice<f16>,
    pub(crate) v_pool: &'a CudaSlice<f16>,
    pub(crate) out: &'a mut CudaSlice<f16>,
    pub(crate) lse: &'a mut CudaSlice<f32>,
    pub(crate) cu_seqlens_q: &'a CudaSlice<u32>,
    pub(crate) seq_lens: &'a CudaSlice<u32>,
    pub(crate) block_tables: &'a CudaSlice<u32>,
    pub(crate) num_seqs: usize,
    pub(crate) total_q_tokens: usize,
    pub(crate) max_q_len: usize,
    pub(crate) max_kv_len: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) block_size: usize,
    pub(crate) max_blocks_per_seq: usize,
}

pub(crate) fn paged_varlen_attention_fa2_source(args: Fa2SourcePagedVarlenArgs<'_>) -> Result<()> {
    let Fa2SourcePagedVarlenArgs {
        stream,
        q,
        k_pool,
        v_pool,
        out,
        lse,
        cu_seqlens_q,
        seq_lens,
        block_tables,
        num_seqs,
        total_q_tokens,
        max_q_len,
        max_kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
    } = args;

    unsafe {
        call_paged_varlen_fn(
            ferrum_fa2_paged_varlen_fwd,
            stream,
            q,
            k_pool,
            v_pool,
            out,
            lse,
            cu_seqlens_q,
            seq_lens,
            block_tables,
            num_seqs,
            total_q_tokens,
            max_q_len,
            max_kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
        )
    }
}
