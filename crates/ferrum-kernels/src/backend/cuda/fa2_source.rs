//! Compatibility stub for the removed source-built FlashAttention-2 bridge.
//!
//! The old `fa2-source` feature compiled FlashAttention/CUTLASS inputs from the
//! main repository into `libfa2_source.a`. That bulk source has moved out of the
//! normal Ferrum build domain; FA2 must be provided by a native operator artifact
//! before this runtime path can be selected again.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use ferrum_types::{FerrumError, Result};
use half::f16;

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

pub(crate) fn paged_varlen_attention_fa2_source(_args: Fa2SourcePagedVarlenArgs<'_>) -> Result<()> {
    Err(FerrumError::unsupported(
        "FERRUM_FA2_SOURCE selected the removed source-linked FA2 path; provide a \
         Ferrum native operator artifact for FA2 or unset FERRUM_FA2_SOURCE",
    ))
}
