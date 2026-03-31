//! Tensor Parallel Decode Group.
//!
//! Coordinates N CudaDecodeRunners (one per GPU) for tensor-parallel decode.
//! Each GPU holds 1/N of attention heads and MLP width.
//! NCCL all-reduce after O projection and down projection (Megatron-LM pattern).
//!
//! Feature-gated: only available with `tensor-parallel` feature.

#[cfg(feature = "tensor-parallel")]
use crate::cuda_decode::CudaDecodeRunner;
#[cfg(feature = "tensor-parallel")]
use crate::nccl_comm::NcclRank;
#[cfg(feature = "tensor-parallel")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "tensor-parallel")]
use std::sync::Arc;

/// Tensor parallel decode group.
///
/// Holds one CudaDecodeRunner + NcclRank per GPU.
/// Orchestrates the decode pipeline with all-reduce at the right points.
#[cfg(feature = "tensor-parallel")]
pub struct TpDecodeGroup {
    /// Per-rank runners (index = rank)
    runners: Vec<CudaDecodeRunner>,
    /// Per-rank NCCL communicators
    nccl: Vec<NcclRank>,
    /// Number of ranks (GPUs)
    world_size: usize,
}

#[cfg(feature = "tensor-parallel")]
impl TpDecodeGroup {
    /// Create a new TP decode group.
    ///
    /// `runners` and `nccl` must be in rank order (index 0 = rank 0).
    pub fn new(runners: Vec<CudaDecodeRunner>, nccl: Vec<NcclRank>) -> candle_core::Result<Self> {
        let world_size = runners.len();
        if nccl.len() != world_size {
            return Err(candle_core::Error::Msg(format!(
                "runners ({}) != nccl ({})",
                world_size,
                nccl.len()
            )));
        }
        Ok(Self {
            runners,
            nccl,
            world_size,
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    // TODO: Phase 3 — implement tp_decode_step
    // The decode pipeline per layer:
    //   1. [all ranks] embed + rms_norm + QKV GEMM (column-parallel: each rank's qkv is smaller)
    //   2. [all ranks] Q/K norm + RoPE + KV append + attention (local heads only)
    //   3. [all ranks] O GEMM (row-parallel: output is partial hidden_size)
    //   4. [NCCL] all_reduce(o_proj_out) — sum partial outputs
    //   5. [all ranks] fused_add_rms_norm (on full hidden_size)
    //   6. [all ranks] gate_up GEMM (column-parallel) + SiLU + down GEMM (row-parallel)
    //   7. [NCCL] all_reduce(down_out) — sum partial outputs
    //   8. [all ranks] fused_add_rms_norm
    // After all layers:
    //   9. [all ranks] final_norm + lm_head (replicated)
}
