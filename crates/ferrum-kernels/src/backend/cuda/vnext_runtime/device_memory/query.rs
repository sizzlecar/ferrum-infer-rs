use std::sync::Arc;

use cudarc::driver::{sys, CudaContext};
use ferrum_types::device_memory::{
    CudaMemoryDomains, CudaMemoryReading as Reading, CudaMemoryUnavailable as Unavailable,
};

use super::super::nvml::Nvml;
use super::allocation::AllocationTracker;

pub(super) struct MemoryQueries {
    pub context: Arc<CudaContext>,
    pub tracker: Arc<AllocationTracker>,
    pub uuid: [u8; 16],
    pub nvml: Result<Nvml, Unavailable>,
}

impl MemoryQueries {
    pub fn sample(&self) -> Result<CudaMemoryDomains, String> {
        let runtime_requested_allocation_bytes = self.tracker.current()?;
        let (cuda_driver_reported_free_bytes, cuda_driver_reported_total_bytes) =
            match self.context.mem_get_info() {
                Ok((free, total)) => (Reading::known(free as u64), Reading::known(total as u64)),
                Err(error) => {
                    let unavailable = Reading::unavailable(Unavailable::QueryFailed {
                        api: "cuMemGetInfo_v2".to_owned(),
                        code: error.0 as i64,
                    });
                    (unavailable.clone(), unavailable)
                }
            };
        let (default_pool_used_bytes, default_pool_reserved_bytes) = self.pool();
        let nvml_process_used_bytes = match &self.nvml {
            Ok(nvml) => nvml.process_memory(self.uuid, std::process::id()),
            Err(reason) => Reading::unavailable(reason.clone()),
        };
        let device_memory = match &self.nvml {
            Ok(nvml) => nvml.device_memory(self.uuid),
            Err(reason) => Err(reason.clone()),
        };
        let (
            nvml_device_free_bytes,
            nvml_device_used_bytes,
            nvml_device_reserved_bytes,
            nvml_device_total_bytes,
        ) = match device_memory {
            Ok(memory) => (
                Reading::known(memory.free),
                Reading::known(memory.used),
                Reading::known(memory.reserved),
                Reading::known(memory.total),
            ),
            Err(reason) => {
                let missing = Reading::unavailable(reason);
                (missing.clone(), missing.clone(), missing.clone(), missing)
            }
        };
        Ok(CudaMemoryDomains {
            runtime_requested_allocation_bytes,
            default_pool_used_bytes,
            default_pool_reserved_bytes,
            cuda_driver_reported_free_bytes,
            cuda_driver_reported_total_bytes,
            nvml_device_free_bytes,
            nvml_device_used_bytes,
            nvml_device_reserved_bytes,
            nvml_device_total_bytes,
            nvml_process_used_bytes,
        })
    }

    fn pool(&self) -> (Reading, Reading) {
        if !self.context.has_async_alloc() {
            let unsupported = Reading::unavailable(Unavailable::Unsupported);
            return (unsupported.clone(), unsupported);
        }
        if let Err(error) = self.context.bind_to_thread() {
            let unavailable = Reading::unavailable(Unavailable::QueryFailed {
                api: "cuCtxSetCurrent".to_owned(),
                code: error.0 as i64,
            });
            return (unavailable.clone(), unavailable);
        }
        let mut pool = std::ptr::null_mut();
        // SAFETY: exact retained context's device, one writable output handle;
        // no allocation, free, synchronization or device work is requested.
        let status = unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, self.context.cu_device()) };
        if status != sys::CUresult::CUDA_SUCCESS || pool.is_null() {
            let unavailable = Reading::unavailable(Unavailable::QueryFailed {
                api: "cuDeviceGetDefaultMemPool".to_owned(),
                code: status as i64,
            });
            return (unavailable.clone(), unavailable);
        }
        let query = |attribute| {
            let mut bytes = 0_u64;
            // Both selected attributes have the documented uint64_t output.
            let status = unsafe {
                sys::cuMemPoolGetAttribute(pool, attribute, (&mut bytes as *mut u64).cast())
            };
            if status == sys::CUresult::CUDA_SUCCESS {
                Reading::known(bytes)
            } else {
                Reading::unavailable(Unavailable::QueryFailed {
                    api: "cuMemPoolGetAttribute".to_owned(),
                    code: status as i64,
                })
            }
        };
        (
            query(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
            query(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        )
    }
}
