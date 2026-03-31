//! NCCL communication primitives for tensor parallelism.
//!
//! Thin wrapper around cudarc's NCCL bindings providing:
//! - Communicator initialization across GPU ranks
//! - In-place FP16 all-reduce (sum)
//! - Stream-aware async operations (CUDA graph compatible)
//!
//! Feature-gated: only available with `tensor-parallel` feature.

#[cfg(feature = "tensor-parallel")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "tensor-parallel")]
use cudarc::nccl::safe::{Comm, Id};
#[cfg(feature = "tensor-parallel")]
use std::sync::Arc;

/// NCCL communicator for one GPU rank in a tensor parallel group.
#[cfg(feature = "tensor-parallel")]
pub struct NcclRank {
    comm: Comm,
    rank: usize,
    world_size: usize,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "tensor-parallel")]
impl NcclRank {
    /// Initialize NCCL communicators for all ranks.
    ///
    /// Must be called from separate threads (one per GPU), all with the same `id`.
    /// Each thread calls with its own `rank` and `stream`.
    pub fn init(
        id: &Id,
        rank: usize,
        world_size: usize,
        stream: Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        let comm = Comm::from_rank(stream.device().clone(), rank, world_size, *id)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL init rank {rank}: {e}")))?;
        Ok(Self {
            comm,
            rank,
            world_size,
            stream,
        })
    }

    /// Generate a unique NCCL ID (call once on rank 0, broadcast to others).
    pub fn unique_id() -> candle_core::Result<Id> {
        Id::new().map_err(|e| candle_core::Error::Msg(format!("NCCL unique_id: {e}")))
    }

    /// In-place all-reduce (sum) on FP16 buffer.
    ///
    /// After this call, `buf` on every rank contains the element-wise sum
    /// of all ranks' input buffers. The operation is async on `self.stream`.
    pub fn all_reduce_f16_inplace(
        &self,
        buf: &mut CudaSlice<half::f16>,
    ) -> candle_core::Result<()> {
        self.comm
            .all_reduce(buf, &self.stream)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL all_reduce: {e}")))
    }

    /// Synchronize the NCCL stream (wait for all pending ops).
    pub fn sync(&self) -> candle_core::Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("NCCL stream sync: {e}")))
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }
}
