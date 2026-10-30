//! Reclaim unused driver-pool backing before evicting resident model weights.

use std::sync::Arc;

use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DriverError};

use super::{allocation_pool_snapshot, CudaAllocationPoolSnapshot, CudaDeviceRuntime};

mod pool;
pub(super) use pool::TensorAllocationPool;

impl CudaDeviceRuntime {
    pub(super) fn tensor_allocation_pool(&self) -> Option<&TensorAllocationPool> {
        self.tensor_eager
            .as_ref()
            .and_then(|owner| owner.allocation_pool.as_ref())
    }

    pub(super) fn allocation_pool_snapshot(
        &self,
    ) -> Result<Option<CudaAllocationPoolSnapshot>, DriverError> {
        match self.tensor_allocation_pool() {
            Some(pool) => pool.snapshot().map(Some),
            None => allocation_pool_snapshot(&self.context),
        }
    }
}

pub(super) fn allocate(
    context: &CudaContext,
    stream: &Arc<CudaStream>,
    pool: Option<&TensorAllocationPool>,
    bytes: usize,
) -> Result<CudaSlice<u8>, DriverError> {
    let attempt = || match pool {
        Some(pool) => pool.allocate(bytes),
        // SAFETY: the admitted caller initializes the returned byte storage.
        None => unsafe { stream.alloc::<u8>(bytes) },
    };
    let snapshot = || match pool {
        Some(pool) => pool.snapshot().map(Some),
        None => allocation_pool_snapshot(context),
    };
    // SAFETY: the admitted caller initializes the returned byte storage before
    // any inference reads it, just as for the ordinary allocation path.
    match attempt() {
        Ok(buffer) => return Ok(buffer),
        Err(error)
            if error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY && context.has_async_alloc() => {}
        Err(error) => return Err(error),
    }

    let before = snapshot();
    // A retired CudaSlice queues its free on this allocation stream. Even when
    // stream query reports completion, synchronization updates the driver's
    // allocator reuse state. Reserved-minus-used alone cannot promise a large
    // allocation: freed blocks can still require that update before remapping.
    // Synchronize only on allocation OOM; normal allocations and capacity
    // observations must retain their fast path. No resident owner is retired.
    stream.synchronize()?;
    // A private pool deliberately keeps idle backing across synchronization.
    // Explicitly return only its unused pages before retrying under pressure.
    // The legacy current-pool path preserves its existing synchronization retry.
    if let Some(pool) = pool {
        pool.trim_idle()?;
    }
    let after = snapshot();
    tracing::debug!(
        allocation_bytes = bytes,
        pool_before = ?before,
        pool_after = ?after,
        private_pool = pool.is_some(),
        "CUDA allocation pressure reclaimed idle pool backing; retrying once"
    );
    // SAFETY: same admitted uninitialized allocation as above. A real capacity
    // shortage still returns to the existing transactional residency recovery.
    // Other driver errors are never retried, and this retry cannot loop.
    attempt()
}
