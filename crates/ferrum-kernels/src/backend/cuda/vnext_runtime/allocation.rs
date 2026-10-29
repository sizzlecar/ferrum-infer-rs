//! Reclaim unused driver-pool backing before evicting resident model weights.

use std::sync::Arc;

use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DriverError};

use super::allocation_pool_snapshot;

pub(super) fn allocate(
    context: &CudaContext,
    stream: &Arc<CudaStream>,
    bytes: usize,
) -> Result<CudaSlice<u8>, DriverError> {
    // SAFETY: the admitted caller initializes the returned byte storage before
    // any inference reads it, just as for the ordinary allocation path.
    match unsafe { stream.alloc::<u8>(bytes) } {
        Ok(buffer) => return Ok(buffer),
        Err(error)
            if error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY && context.has_async_alloc() => {}
        Err(error) => return Err(error),
    }

    let before = allocation_pool_snapshot(context);
    // A retired CudaSlice queues its free on this allocation stream. Even when
    // stream query reports completion, synchronization updates the driver's
    // allocator reuse state. Reserved-minus-used alone cannot promise a large
    // allocation: freed blocks can still require that update before remapping.
    // Synchronize only on allocation OOM; normal allocations and capacity
    // observations must retain their fast path. No resident owner is retired.
    stream.synchronize()?;
    let after = allocation_pool_snapshot(context);
    tracing::debug!(
        allocation_bytes = bytes,
        pool_before = ?before,
        pool_after = ?after,
        "CUDA allocation pressure synchronized pool reuse; retrying once"
    );
    // SAFETY: same admitted uninitialized allocation as above. A real capacity
    // shortage still returns to the existing transactional residency recovery.
    // Other driver errors are never retried, and this retry cannot loop.
    unsafe { stream.alloc::<u8>(bytes) }
}
