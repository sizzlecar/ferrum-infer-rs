//! Startup memory observations, taken before model weights are allocated.
//!
//! Availability already accounts for existing allocations. Callers must not
//! subtract the process footprint again, and must budget future weights once.

use ferrum_types::{Device, DeviceMemorySnapshot, FerrumError, Result};

use super::cpu::vnext_runtime::{host_memory_available, host_memory_capacity};

/// Observe the capacity and currently available bytes on the selected device.
/// This is a point-in-time planning input, not a reservation against other
/// processes. Unsupported devices and failed probes return an error.
pub fn probe_device_memory(device: &Device) -> Result<DeviceMemorySnapshot> {
    match device {
        Device::CPU => probe_host_memory(),
        #[cfg(feature = "cuda")]
        Device::CUDA(ordinal) => {
            let context = cudarc::driver::CudaContext::new(*ordinal)
                .map_err(|error| FerrumError::device(format!("probe CUDA {ordinal}: {error}")))?;
            // This method binds the selected context before querying the driver,
            // so CUDA_VISIBLE_DEVICES and nonzero ordinals are respected.
            let (available, capacity) = context.mem_get_info().map_err(|error| {
                FerrumError::device(format!("read CUDA {ordinal} free memory: {error}"))
            })?;
            snapshot(capacity as u64, available as u64, "cuda_mem_get_info")
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        Device::Metal => {
            // Match the backend's system-default device without compiling
            // pipelines or allocating model resources during discovery.
            let device = metal::Device::system_default()
                .ok_or_else(|| FerrumError::device("no default Metal device"))?;
            let host = probe_host_memory()?;
            let recommended = device.recommended_max_working_set_size();
            let allocated = device.current_allocated_size() as u64;
            let (capacity, available) = metal_memory_bounds(
                recommended,
                allocated,
                host.capacity_bytes,
                host.available_bytes,
            );
            snapshot(
                capacity,
                available,
                &format!("metal_recommended_minus_allocated_and_{}", host.source),
            )
        }
        device => Err(FerrumError::unsupported(format!(
            "memory discovery is unavailable for {device} in this build"
        ))),
    }
}

fn probe_host_memory() -> Result<DeviceMemorySnapshot> {
    let capacity = host_memory_capacity()
        .map_err(|error| FerrumError::device(format!("read host memory capacity: {error}")))?;
    let available = host_memory_available()
        .map_err(|error| FerrumError::device(format!("read host available memory: {error}")))?;
    snapshot(
        capacity,
        available,
        if cfg!(any(target_os = "macos", target_os = "ios")) {
            "host_free_inactive_pages_with_process_limits"
        } else {
            "host_available_with_process_limits"
        },
    )
}

fn snapshot(capacity: u64, available: u64, source: &str) -> Result<DeviceMemorySnapshot> {
    if capacity == 0 {
        return Err(FerrumError::device(
            "memory discovery reported zero capacity",
        ));
    }
    Ok(DeviceMemorySnapshot {
        capacity_bytes: capacity,
        available_bytes: available.min(capacity),
        source: source.into(),
    })
}

#[cfg(any(feature = "metal", test))]
fn metal_memory_bounds(
    recommended: u64,
    allocated: u64,
    host_capacity: u64,
    host_available: u64,
) -> (u64, u64) {
    (
        recommended.min(host_capacity),
        recommended.saturating_sub(allocated).min(host_available),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_uses_independent_limits_without_double_charging_allocations() {
        // The host observation already includes these existing GPU allocations.
        assert_eq!(metal_memory_bounds(80, 20, 100, 70), (80, 60));
        assert_eq!(metal_memory_bounds(80, 20, 100, 40), (80, 40));
        assert_eq!(metal_memory_bounds(80, 90, 100, 40), (80, 0));
    }

    #[test]
    fn observations_preserve_exhaustion_and_bound_availability() {
        assert_eq!(snapshot(100, 0, "test").unwrap().available_bytes, 0);
        assert_eq!(snapshot(100, 200, "test").unwrap().available_bytes, 100);
        assert!(snapshot(0, 0, "test").is_err());
    }

    #[test]
    fn unsupported_backend_does_not_invent_a_memory_budget() {
        assert!(probe_device_memory(&Device::ROCm(0)).is_err());
    }

    #[test]
    fn host_observation_has_an_addressable_capacity() {
        let observation = probe_device_memory(&Device::CPU).unwrap();
        assert!(observation.capacity_bytes > 0);
        assert!(observation.capacity_bytes <= usize::MAX as u64);
        assert!(observation.available_bytes <= observation.capacity_bytes);
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_observation_respects_the_device_working_set() {
        let device = metal::Device::system_default().expect("Metal test requires a default GPU");
        let observation = probe_device_memory(&Device::Metal).unwrap();
        assert!(observation.capacity_bytes > 0);
        assert!(observation.capacity_bytes <= device.recommended_max_working_set_size());
        assert!(observation.available_bytes <= observation.capacity_bytes);
    }
}
