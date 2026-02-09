//! Device Management
//!
//! Device discovery, capability detection, and resource monitoring
//! for multi-GPU environments.

use ferrum_types::{Device, FerrumError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use tracing::{info, warn};

/// Device capability information
#[derive(Debug, Clone)]
pub struct DeviceCapability {
    /// Compute capability (e.g., 8.0 for A100)
    pub compute_capability: (u32, u32),
    /// Total memory in bytes
    pub total_memory: usize,
    /// Number of SMs (for CUDA) or compute units
    pub num_compute_units: u32,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Warp size (32 for CUDA, varies for others)
    pub warp_size: u32,
    /// Whether tensor cores are available
    pub has_tensor_cores: bool,
    /// Whether unified memory is supported
    pub unified_memory: bool,
    /// Maximum shared memory per block
    pub max_shared_memory: usize,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,
    /// Peak FLOPS (TF32)
    pub peak_tflops: f32,
}

impl Default for DeviceCapability {
    fn default() -> Self {
        Self {
            compute_capability: (0, 0),
            total_memory: 0,
            num_compute_units: 0,
            max_threads_per_block: 1024,
            warp_size: 32,
            has_tensor_cores: false,
            unified_memory: false,
            max_shared_memory: 48 * 1024, // 48KB default
            memory_bandwidth: 0.0,
            peak_tflops: 0.0,
        }
    }
}

impl DeviceCapability {
    /// Create capability info for CPU
    pub fn cpu() -> Self {
        Self {
            compute_capability: (0, 0),
            total_memory: 0,      // To be filled from system info
            num_compute_units: 0, // CPU cores
            max_threads_per_block: 1,
            warp_size: 1,
            has_tensor_cores: false,
            unified_memory: true,
            max_shared_memory: 0,
            memory_bandwidth: 0.0,
            peak_tflops: 0.0,
        }
    }

    /// Create capability info for Apple Silicon (Metal)
    pub fn apple_silicon(total_memory: usize, gpu_cores: u32) -> Self {
        Self {
            compute_capability: (1, 0), // Metal version indicator
            total_memory,
            num_compute_units: gpu_cores,
            max_threads_per_block: 1024,
            warp_size: 32,           // SIMD width
            has_tensor_cores: false, // No tensor cores, but has AMX
            unified_memory: true,
            max_shared_memory: 32 * 1024,
            memory_bandwidth: 200.0, // Varies by chip
            peak_tflops: 10.0,       // Varies by chip
        }
    }

    /// Check if device can run a model of given size
    pub fn can_fit_model(&self, model_size_bytes: usize) -> bool {
        // Leave some memory for KV cache and intermediate activations
        let usable_memory = self.total_memory * 70 / 100; // 70% for model
        model_size_bytes <= usable_memory
    }
}

/// Device information with runtime state
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device identifier
    pub device: Device,
    /// Device name
    pub name: String,
    /// Device capabilities
    pub capability: DeviceCapability,
    /// Current memory usage in bytes
    pub used_memory: usize,
    /// Whether device is available
    pub is_available: bool,
    /// Current utilization (0.0 - 1.0)
    pub utilization: f32,
}

impl DeviceInfo {
    /// Create info for CPU
    pub fn cpu() -> Self {
        Self {
            device: Device::CPU,
            name: "CPU".to_string(),
            capability: DeviceCapability::cpu(),
            used_memory: 0,
            is_available: true,
            utilization: 0.0,
        }
    }

    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.capability
            .total_memory
            .saturating_sub(self.used_memory)
    }
}

/// Device manager for multi-GPU coordination
pub struct DeviceManager {
    /// All discovered devices
    devices: RwLock<HashMap<Device, DeviceInfo>>,
    /// Primary device (first available GPU or CPU)
    primary_device: RwLock<Device>,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        Self {
            devices: RwLock::new(HashMap::new()),
            primary_device: RwLock::new(Device::CPU),
        }
    }

    /// Discover available devices
    pub fn discover_devices(&self) -> Result<()> {
        info!("Discovering available devices...");
        let mut devices = self.devices.write();
        devices.clear();

        // Always add CPU
        let cpu_info = DeviceInfo::cpu();
        devices.insert(Device::CPU, cpu_info);

        // Try to detect GPUs
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            if let Ok(metal_info) = self.detect_metal_devices() {
                for info in metal_info {
                    info!("Found Metal device: {}", info.name);
                    *self.primary_device.write() = info.device.clone();
                    devices.insert(info.device.clone(), info);
                }
            }
        }

        // CUDA device detection would go here
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_info) = self.detect_cuda_devices() {
                for (idx, info) in cuda_info.into_iter().enumerate() {
                    info!("Found CUDA device {}: {}", idx, info.name);
                    if idx == 0 {
                        *self.primary_device.write() = info.device.clone();
                    }
                    devices.insert(info.device.clone(), info);
                }
            }
        }

        info!("Discovered {} device(s)", devices.len());
        Ok(())
    }

    /// Get all available devices
    pub fn get_devices(&self) -> Vec<DeviceInfo> {
        self.devices.read().values().cloned().collect()
    }

    /// Get device info by ID
    pub fn get_device(&self, device: &Device) -> Option<DeviceInfo> {
        self.devices.read().get(device).cloned()
    }

    /// Get primary device
    pub fn primary_device(&self) -> Device {
        self.primary_device.read().clone()
    }

    /// Set primary device
    pub fn set_primary_device(&self, device: Device) -> Result<()> {
        if self.devices.read().contains_key(&device) {
            *self.primary_device.write() = device;
            Ok(())
        } else {
            Err(FerrumError::not_found(format!(
                "Device {:?} not found",
                device
            )))
        }
    }

    /// Get available GPU devices
    pub fn get_gpu_devices(&self) -> Vec<DeviceInfo> {
        self.devices
            .read()
            .values()
            .filter(|info| Self::is_gpu_device(&info.device))
            .cloned()
            .collect()
    }

    /// Get total available GPU memory across all devices
    pub fn total_gpu_memory(&self) -> usize {
        self.devices
            .read()
            .values()
            .filter(|info| Self::is_gpu_device(&info.device))
            .map(|info| info.capability.total_memory)
            .sum()
    }

    /// Check if a device is a GPU (CUDA, ROCm, or Metal)
    fn is_gpu_device(device: &Device) -> bool {
        match device {
            Device::CPU => false,
            Device::CUDA(_) => true,
            Device::ROCm(_) => true,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => true,
        }
    }

    /// Update memory usage for a device
    pub fn update_memory_usage(&self, device: &Device, used_bytes: usize) {
        if let Some(info) = self.devices.write().get_mut(device) {
            info.used_memory = used_bytes;
        }
    }

    /// Find devices that can fit a model of given size
    pub fn find_devices_for_model(&self, model_size_bytes: usize) -> Vec<DeviceInfo> {
        self.devices
            .read()
            .values()
            .filter(|info| info.capability.can_fit_model(model_size_bytes))
            .cloned()
            .collect()
    }

    /// Select optimal devices for parallel execution
    pub fn select_devices_for_parallelism(
        &self,
        model_size_bytes: usize,
        min_devices: usize,
        max_devices: usize,
    ) -> Vec<DeviceInfo> {
        let gpu_devices: Vec<_> = self.get_gpu_devices();

        // If we have enough GPUs that can individually fit the model, use those
        let fitting_devices: Vec<_> = gpu_devices
            .iter()
            .filter(|d| d.capability.can_fit_model(model_size_bytes))
            .cloned()
            .collect();

        if fitting_devices.len() >= min_devices {
            return fitting_devices.into_iter().take(max_devices).collect();
        }

        // Otherwise, select devices for model parallelism based on total memory
        let mut sorted_devices = gpu_devices;
        sorted_devices.sort_by(|a, b| b.capability.total_memory.cmp(&a.capability.total_memory));

        let mut selected = Vec::new();
        let mut total_memory = 0usize;

        for device in sorted_devices {
            if selected.len() >= max_devices {
                break;
            }
            total_memory += device.capability.total_memory;
            selected.push(device);

            // Check if we have enough memory for model + overhead
            let required = model_size_bytes + model_size_bytes / 4; // 25% overhead
            if total_memory >= required && selected.len() >= min_devices {
                break;
            }
        }

        // Fall back to CPU if no GPUs available
        if selected.is_empty() {
            warn!("No suitable GPUs found, falling back to CPU");
            if let Some(cpu) = self.get_device(&Device::CPU) {
                selected.push(cpu);
            }
        }

        selected
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    fn detect_metal_devices(&self) -> Result<Vec<DeviceInfo>> {
        // In actual implementation, this would use metal-rs to detect devices
        // For now, return a placeholder for the default GPU
        Ok(vec![DeviceInfo {
            device: Device::Metal,
            name: "Apple GPU".to_string(),
            capability: DeviceCapability::apple_silicon(16 * 1024 * 1024 * 1024, 76), // Placeholder
            used_memory: 0,
            is_available: true,
            utilization: 0.0,
        }])
    }

    #[cfg(feature = "cuda")]
    fn detect_cuda_devices(&self) -> Result<Vec<DeviceInfo>> {
        // Would use CUDA runtime API to detect devices
        Err(FerrumError::unsupported(
            "CUDA detection not yet implemented",
        ))
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        let manager = Self::new();
        let _ = manager.discover_devices();
        manager
    }
}

/// Global device manager
static GLOBAL_DEVICE_MANAGER: std::sync::OnceLock<DeviceManager> = std::sync::OnceLock::new();

/// Get the global device manager
pub fn global_device_manager() -> &'static DeviceManager {
    GLOBAL_DEVICE_MANAGER.get_or_init(DeviceManager::default)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_capability_default() {
        let cap = DeviceCapability::default();
        assert_eq!(cap.compute_capability, (0, 0));
        assert!(!cap.has_tensor_cores);
    }

    #[test]
    fn test_device_info_cpu() {
        let info = DeviceInfo::cpu();
        assert_eq!(info.device, Device::CPU);
        assert!(info.is_available);
    }

    #[test]
    fn test_device_manager_discover() {
        let manager = DeviceManager::new();
        manager.discover_devices().unwrap();

        let devices = manager.get_devices();
        assert!(!devices.is_empty()); // At least CPU should be there
    }

    #[test]
    fn test_can_fit_model() {
        let mut cap = DeviceCapability::default();
        cap.total_memory = 16 * 1024 * 1024 * 1024; // 16GB

        assert!(cap.can_fit_model(10 * 1024 * 1024 * 1024)); // 10GB model
        assert!(!cap.can_fit_model(15 * 1024 * 1024 * 1024)); // 15GB model (exceeds 70%)
    }
}
