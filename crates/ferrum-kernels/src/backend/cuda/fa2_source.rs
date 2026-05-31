//! Source-built FlashAttention-2 bridge.
//!
//! This module is compiled only with the `fa2-source` cargo feature. The C ABI
//! symbol is linked from a static library built by `build.rs` from
//! FlashAttention source templates, so runtime does not need vLLM, Torch,
//! Python, or a user-provided shim `.so`.

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

#[allow(clippy::too_many_arguments)]
pub fn paged_varlen_attention_fa2_source(
    stream: &Arc<CudaStream>,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    lse: &mut CudaSlice<f32>,
    cu_seqlens_q: &CudaSlice<u32>,
    seq_lens: &CudaSlice<u32>,
    block_tables: &CudaSlice<u32>,
    num_seqs: usize,
    total_q_tokens: usize,
    max_q_len: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
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
