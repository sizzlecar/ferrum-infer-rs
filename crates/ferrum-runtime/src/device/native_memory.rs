use serde::Serialize;

/// Physical availability sampled at an idle stage boundary. This is neither
/// an allocation permit nor a guarantee against concurrent native allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceMemoryAvailability {
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// Unused bytes in the allocator pool, already excluded from free_bytes.
    pub reusable_pool_bytes: u64,
}

impl DeviceMemoryAvailability {
    pub fn available_bytes(self) -> u64 {
        self.free_bytes
            .saturating_add(self.reusable_pool_bytes)
            .min(self.total_bytes)
    }
}

/// The native counter's attribution boundary, independent of plan ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeviceNativeMemoryScope {
    /// Resources reported by the backend's device handle. Other users of that
    /// native device in this process may contribute to the same counter.
    DeviceHandle,
}

/// Native resource accounting, never an admission limit or resident-memory
/// measurement. Unsupported backends return `None`, not an invented zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceNativeMemorySnapshot {
    /// Identifies the native API/metric, so unlike counters are not compared.
    pub source: &'static str,
    pub scope: DeviceNativeMemoryScope,
    pub allocated_bytes: u64,
    /// Highest sampled value since this runtime was constructed. This is not
    /// a per-request peak; unsampled native/driver transients may be missed.
    pub observed_peak_allocated_bytes: u64,
    /// Includes allocation-boundary and explicit snapshot samples; saturates
    /// at u64::MAX. Sampling never resets a shared runtime's observations.
    pub samples: u64,
}
