//! `BackendCollective for CudaBackend` — TP / multi-GPU collectives.
//!
//! The immutable context fixes world size and rank. The retained backend
//! constructor rejects multi-rank policies because it has no collective owner.
//! Single-rank collectives preserve the ordinary no-op behavior.

use super::CudaBackend;
use crate::backend::{Backend, BackendCollective, ReduceOp};
impl BackendCollective for CudaBackend {
    fn world_size(ctx: &Self::Context) -> usize {
        ctx.configuration.collective.world_size
    }

    fn rank(ctx: &Self::Context) -> usize {
        ctx.configuration.collective.rank
    }

    fn all_reduce(ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize, op: ReduceOp) {
        // Only Sum is supported for now (the NCCL wrapper is sum-only).
        if !matches!(op, ReduceOp::Sum) {
            tracing::warn!(
                "CudaBackend::all_reduce: op {op:?} not implemented (only Sum); skipping"
            );
            return;
        }
        // Single-rank path: no-op.
        if Self::world_size(ctx) <= 1 {
            return;
        }
        unreachable!("CudaBackend construction rejects a missing multi-rank collective owner");
    }

    fn all_gather(
        _ctx: &mut Self::Context,
        _local: &Self::Buffer,
        _global: &mut Self::Buffer,
        _local_len: usize,
    ) {
        // Phase E-TP: no NCCL wrapper for all_gather yet.
    }

    fn broadcast(_ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize, _src_rank: usize) {
        // Phase E-TP: no NCCL wrapper for broadcast yet.
    }
}
