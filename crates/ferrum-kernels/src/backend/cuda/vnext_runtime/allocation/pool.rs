//! Per-tensor-runtime driver cache. No device current/default pool is modified.
use std::sync::Arc;

use cudarc::driver::{result, sys, CudaSlice, CudaStream, DriverError};
use ferrum_runtime::{DeviceCapacityOwner, DeviceCapacityRetention};

use super::super::{allocation_pool_snapshot_at, CudaAllocationPoolSnapshot};

pub(in crate::backend::cuda::vnext_runtime) struct TensorAllocationPool {
    pool: sys::CUmemoryPool,
    stream: Arc<CudaStream>,
    // A shared bootstrap claim, not an additional memory allowance. Unknown
    // native cleanup must retain it together with the context and pool.
    bootstrap: Option<DeviceCapacityRetention>,
    // Auto construction may have lowered the initial bootstrap ceiling. Keep
    // the final policy owner as well when native cleanup is indeterminate.
    capacity: Option<DeviceCapacityOwner>,
}

// CUDA driver pool operations are thread-safe. Allocations and queued frees
// use one retained allocation stream; each operation binds its live context.
unsafe impl Send for TensorAllocationPool {}
unsafe impl Sync for TensorAllocationPool {}

impl TensorAllocationPool {
    pub(in crate::backend::cuda::vnext_runtime) fn new(
        stream: Arc<CudaStream>,
        release_threshold_bytes: u64,
        bootstrap: DeviceCapacityRetention,
        capacity: DeviceCapacityOwner,
    ) -> Result<Self, DriverError> {
        let context = stream.context();
        context.bind_to_thread()?;
        if !context.has_async_alloc() || context.is_event_tracking() || release_threshold_bytes == 0
        {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        // CUDA requires all unspecified/reserved properties to be zero.
        let mut properties = unsafe { std::mem::zeroed::<sys::CUmemPoolProps>() };
        properties.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        properties.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = context.cu_device();
        // SAFETY: properties describe the live stream's physical CUDA device.
        let pool = unsafe { result::mem_pool::create(&properties)? };
        let owner = Self {
            pool,
            stream,
            bootstrap: Some(bootstrap),
            capacity: Some(capacity),
        };
        let mut threshold = release_threshold_bytes;
        // This changes when unused backing is returned, not admission or a
        // reservation. The value is bounded by this runtime's resolved ceiling.
        unsafe {
            result::mem_pool::set_attribute(
                owner.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )?;
        }
        tracing::info!(
            policy = "private_capacity_bounded_v1",
            release_threshold_bytes,
            "CUDA tensor allocator cache selected"
        );
        Ok(owner)
    }

    pub(in crate::backend::cuda::vnext_runtime) fn snapshot(
        &self,
    ) -> Result<CudaAllocationPoolSnapshot, DriverError> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: this owner retains the private pool and its context.
        unsafe { allocation_pool_snapshot_at(self.pool) }
    }

    pub(in crate::backend::cuda::vnext_runtime) fn allocate(
        &self,
        bytes: usize,
    ) -> Result<CudaSlice<u8>, DriverError> {
        self.stream.context().bind_to_thread()?;
        // SAFETY: this is the same uninitialized admitted byte allocation as
        // CudaStream::alloc. Event tracking is disabled by the owning runtime;
        // upgrade creates no fallible events. CudaSlice queues free on this
        // stream. CudaAllocation keeps TensorEagerState (and this pool) alive
        // until after its CudaSlice drops.
        unsafe {
            let pointer = result::mem_pool::alloc_async(self.pool, bytes, self.stream.cu_stream())?;
            Ok(self.stream.upgrade_device_ptr(pointer, bytes))
        }
    }

    pub(in crate::backend::cuda::vnext_runtime) fn trim_idle(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        // The caller synchronizes queued frees first. CUDA only releases
        // unused pages; every live model/request allocation stays untouched.
        unsafe { result::mem_pool::trim_to(self.pool, 0) }
    }
}

impl Drop for TensorAllocationPool {
    fn drop(&mut self) {
        let result = self.stream.synchronize().and_then(|()| {
            // SAFETY: the final TensorEagerState owner follows all buffers and
            // queued frees. Successful synchronization proves their retirement.
            unsafe { result::mem_pool::destroy(self.pool) }
        });
        if let Err(error) = result {
            tracing::error!(
                ?error,
                "CUDA tensor allocator cleanup failed; retaining native ownership"
            );
            std::mem::forget(Arc::clone(&self.stream));
            if let Some(bootstrap) = self.bootstrap.take() {
                std::mem::forget(bootstrap);
            }
            if let Some(capacity) = self.capacity.take() {
                std::mem::forget(capacity);
            }
        }
    }
}

#[cfg(test)]
mod tests;
