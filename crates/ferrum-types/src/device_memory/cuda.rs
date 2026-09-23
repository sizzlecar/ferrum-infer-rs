//! Distinct CUDA memory domains. None is a substitute for another domain.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CudaMemoryReading {
    Known { bytes: u64 },
    Unavailable { reason: CudaMemoryUnavailable },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CudaMemoryUnavailable {
    Unsupported,
    LibraryUnavailable,
    SymbolUnavailable,
    QueryFailed { api: String, code: i64 },
    DeviceIdentityMismatch,
    ProcessNotListed,
    ProcessAccountingUnavailable,
    AmbiguousProcess,
    CapacityExceeded,
    QueryChanged,
}

impl CudaMemoryReading {
    pub fn known(bytes: u64) -> Self {
        Self::Known { bytes }
    }

    pub fn unavailable(reason: CudaMemoryUnavailable) -> Self {
        Self::Unavailable { reason }
    }
}

/// These observations are sampled independently; they are not one atomic
/// snapshot and must not be added together. NVML device values include other
/// processes. CUDA driver memory availability can differ from physical free
/// memory under WDDM/WSL; it is not a safe substitute for NVML device free.
/// Default-pool values exclude non-pool allocations. NVML can report
/// unavailable process accounting (not zero), notably under WDDM/WSL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaMemoryDomains {
    /// Actual backing allocation requests made through this runtime, including
    /// alignment padding, retained once until the last owning handle is dropped.
    /// Excludes native-library/context allocations and cached pool reservations.
    /// This is allocation accounting, not physical residency / Peak VRAM.
    pub runtime_requested_allocation_bytes: u64,
    pub default_pool_used_bytes: CudaMemoryReading,
    pub default_pool_reserved_bytes: CudaMemoryReading,
    pub cuda_driver_reported_free_bytes: CudaMemoryReading,
    pub cuda_driver_reported_total_bytes: CudaMemoryReading,
    pub nvml_device_free_bytes: CudaMemoryReading,
    pub nvml_device_used_bytes: CudaMemoryReading,
    pub nvml_device_reserved_bytes: CudaMemoryReading,
    pub nvml_device_total_bytes: CudaMemoryReading,
    pub nvml_process_used_bytes: CudaMemoryReading,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_memory_unavailable_is_not_a_zero_measurement() {
        let zero = CudaMemoryReading::known(0);
        let unavailable =
            CudaMemoryReading::unavailable(CudaMemoryUnavailable::ProcessAccountingUnavailable);
        assert_ne!(zero, unavailable);
        for value in [zero, unavailable] {
            let json = serde_json::to_string(&value).unwrap();
            assert_eq!(
                serde_json::from_str::<CudaMemoryReading>(&json).unwrap(),
                value
            );
        }
    }
}
