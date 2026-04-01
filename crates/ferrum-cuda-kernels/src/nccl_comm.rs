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
    stream: Arc<CudaStream>,
}

#[cfg(feature = "tensor-parallel")]
impl NcclRank {
    pub fn init(
        id: &Id,
        rank: usize,
        world_size: usize,
        stream: Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        let device = stream.device().clone();
        let comm = Comm::from_rank(device, rank, world_size, *id)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL init rank {rank}: {e:?}")))?;
        Ok(Self {
            comm,
            rank,
            world_size,
            stream,
        })
    }

    pub fn unique_id() -> candle_core::Result<Id> {
        Id::new().map_err(|e| candle_core::Error::Msg(format!("NCCL unique_id: {e:?}")))
    }

    pub fn all_reduce_f16_inplace(
        &self,
        buf: &mut CudaSlice<half::f16>,
    ) -> candle_core::Result<()> {
        self.comm
            .all_reduce(buf, &self.stream, cudarc::nccl::ReduceOp::Sum)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL all_reduce: {e:?}")))?;
        Ok(())
    }

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
