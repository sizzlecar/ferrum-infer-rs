//! NCCL communication primitives for tensor parallelism.

#[cfg(feature = "tensor-parallel")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "tensor-parallel")]
use cudarc::nccl::safe::Comm;

#[cfg(feature = "tensor-parallel")]
pub struct NcclRank {
    comm: Comm,
    rank: usize,
    world_size: usize,
}

#[cfg(feature = "tensor-parallel")]
unsafe impl Send for NcclRank {}
#[cfg(feature = "tensor-parallel")]
unsafe impl Sync for NcclRank {}

#[cfg(feature = "tensor-parallel")]
impl NcclRank {
    /// Init all comms at once using ncclCommInitAll (single thread, no deadlock).
    pub fn init_all(
        streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>>,
    ) -> candle_core::Result<Vec<Self>> {
        let comms = Comm::from_devices(streams)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL init_all: {e:?}")))?;
        let ws = comms.len();
        Ok(comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| NcclRank {
                comm,
                rank,
                world_size: ws,
            })
            .collect())
    }

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
