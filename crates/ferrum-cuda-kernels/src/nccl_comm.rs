//! NCCL communication primitives for tensor parallelism.

#[cfg(feature = "tensor-parallel")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "tensor-parallel")]
use cudarc::nccl::safe::{Comm, Id};
#[cfg(feature = "tensor-parallel")]
use std::sync::Arc;

#[cfg(feature = "tensor-parallel")]
pub struct NcclRank {
    comm: Comm,
    rank: usize,
    world_size: usize,
}

// Safety: NcclRank is accessed via Mutex, one thread at a time.
// *mut ncclComm is not Send by default but NCCL comms are safe
// when used from the thread that created them.
#[cfg(feature = "tensor-parallel")]
unsafe impl Send for NcclRank {}
#[cfg(feature = "tensor-parallel")]
unsafe impl Sync for NcclRank {}

#[cfg(feature = "tensor-parallel")]
impl NcclRank {
    pub fn init(
        id: &Id,
        rank: usize,
        world_size: usize,
        stream: Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        // Comm::from_rank takes the stream and owns it internally
        let comm = Comm::from_rank(stream, rank, world_size, *id)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL init rank {rank}: {e:?}")))?;
        Ok(Self {
            comm,
            rank,
            world_size,
        })
    }

    pub fn unique_id() -> candle_core::Result<Id> {
        Id::new().map_err(|e| candle_core::Error::Msg(format!("NCCL unique_id: {e:?}")))
    }

    /// In-place all-reduce (sum). Comm holds its own stream.
    pub fn all_reduce_f16_inplace(
        &self,
        buf: &mut CudaSlice<half::f16>,
    ) -> candle_core::Result<()> {
        self.comm
            .all_reduce_in_place(buf, &cudarc::nccl::ReduceOp::Sum)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL all_reduce: {e:?}")))?;
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}
