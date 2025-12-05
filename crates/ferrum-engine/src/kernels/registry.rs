//! Kernel Registry
//!
//! Provides runtime discovery and selection of optimal kernels
//! based on hardware capabilities and workload characteristics.

use ferrum_types::{Device, FerrumError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use super::attention::{
    create_attention_info, AttentionConfig, AttentionKernel, AttentionType, FlashAttentionInfo,
    StandardAttentionInfo,
};
use super::fused::{CpuFusedOpsInfo, FusedOps, FusedOpsConfig};

/// Kernel information for registry
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Unique kernel identifier
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Supported devices
    pub supported_devices: Vec<Device>,
    /// Feature requirements (e.g., "flash_attention", "fused_rope")
    pub features: Vec<String>,
    /// Priority (higher = preferred)
    pub priority: i32,
}

impl KernelInfo {
    /// Create new kernel info
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            supported_devices: vec![Device::CPU],
            features: vec![],
            priority: 0,
        }
    }

    /// Add supported device
    pub fn with_device(mut self, device: Device) -> Self {
        if !self.supported_devices.contains(&device) {
            self.supported_devices.push(device);
        }
        self
    }

    /// Add feature
    pub fn with_feature(mut self, feature: impl Into<String>) -> Self {
        self.features.push(feature.into());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Kernel performance hint
#[derive(Debug, Clone, Copy)]
pub struct PerformanceHint {
    /// Expected batch size
    pub batch_size: usize,
    /// Expected sequence length
    pub seq_len: usize,
    /// Whether memory is constrained
    pub memory_constrained: bool,
    /// Whether latency is critical
    pub latency_critical: bool,
}

impl Default for PerformanceHint {
    fn default() -> Self {
        Self {
            batch_size: 1,
            seq_len: 512,
            memory_constrained: false,
            latency_critical: true,
        }
    }
}

impl PerformanceHint {
    /// Create hint for prefill phase
    pub fn for_prefill(batch_size: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            memory_constrained: seq_len > 2048,
            latency_critical: true,
        }
    }

    /// Create hint for decode phase
    pub fn for_decode(batch_size: usize) -> Self {
        Self {
            batch_size,
            seq_len: 1,
            memory_constrained: false,
            latency_critical: true,
        }
    }
}

/// Attention kernel factory type
type AttentionFactory =
    Box<dyn Fn(&AttentionConfig) -> Result<Box<dyn AttentionKernel>> + Send + Sync>;

/// Fused ops factory type
type FusedOpsFactory =
    Box<dyn Fn(&FusedOpsConfig) -> Result<Arc<dyn FusedOps>> + Send + Sync>;

/// Kernel Registry
///
/// Manages available kernel implementations and provides
/// automatic selection based on hardware and workload.
pub struct KernelRegistry {
    /// Registered attention kernel factories
    attention_factories: RwLock<HashMap<String, (KernelInfo, AttentionFactory)>>,
    /// Registered fused operation implementations
    fused_factories: RwLock<HashMap<String, (KernelInfo, FusedOpsFactory)>>,
}

impl KernelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            attention_factories: RwLock::new(HashMap::new()),
            fused_factories: RwLock::new(HashMap::new()),
        }
    }

    /// Create registry with default kernels
    pub fn with_defaults() -> Self {
        let registry = Self::new();
        registry.register_defaults();
        registry
    }

    /// Register default kernel implementations
    fn register_defaults(&self) {
        // Register standard attention
        self.register_attention_kernel(
            "standard",
            KernelInfo::new("standard", "Standard multi-head attention")
                .with_device(Device::CPU)
                .with_priority(0),
            |config| Ok(Box::new(StandardAttentionInfo::new(config.clone()))),
        );

        // Register flash attention
        self.register_attention_kernel(
            "flash",
            KernelInfo::new("flash", "Flash Attention (memory efficient)")
                .with_device(Device::CPU)
                .with_feature("flash_attention")
                .with_priority(10),
            |config| Ok(Box::new(FlashAttentionInfo::new(config.clone()))),
        );

        // Register CPU fused ops
        self.register_fused_ops(
            "cpu",
            KernelInfo::new("cpu_fused", "CPU fused operations")
                .with_device(Device::CPU)
                .with_feature("fused_rope")
                .with_feature("fused_qkv")
                .with_priority(0),
            |config| Ok(Arc::new(CpuFusedOpsInfo::new(config.clone()))),
        );

        info!("Registered default kernels");
    }

    /// Register an attention kernel
    pub fn register_attention_kernel<F>(&self, name: &str, info: KernelInfo, factory: F)
    where
        F: Fn(&AttentionConfig) -> Result<Box<dyn AttentionKernel>> + Send + Sync + 'static,
    {
        debug!("Registering attention kernel: {}", name);
        self.attention_factories
            .write()
            .insert(name.to_string(), (info, Box::new(factory)));
    }

    /// Register fused operations
    pub fn register_fused_ops<F>(&self, name: &str, info: KernelInfo, factory: F)
    where
        F: Fn(&FusedOpsConfig) -> Result<Arc<dyn FusedOps>> + Send + Sync + 'static,
    {
        debug!("Registering fused ops: {}", name);
        self.fused_factories
            .write()
            .insert(name.to_string(), (info, Box::new(factory)));
    }

    /// Get an attention kernel by name
    pub fn get_attention_kernel(
        &self,
        name: &str,
        config: &AttentionConfig,
    ) -> Result<Box<dyn AttentionKernel>> {
        let factories = self.attention_factories.read();
        let (_, factory) = factories
            .get(name)
            .ok_or_else(|| FerrumError::not_found(format!("Attention kernel not found: {}", name)))?;
        factory(config)
    }

    /// Get fused operations by name
    pub fn get_fused_ops(&self, name: &str, config: &FusedOpsConfig) -> Result<Arc<dyn FusedOps>> {
        let factories = self.fused_factories.read();
        let (_, factory) = factories
            .get(name)
            .ok_or_else(|| FerrumError::not_found(format!("Fused ops not found: {}", name)))?;
        factory(config)
    }

    /// Select best attention kernel for device and performance hint
    pub fn select_attention_kernel(
        &self,
        config: &AttentionConfig,
        hint: &PerformanceHint,
    ) -> Result<Box<dyn AttentionKernel>> {
        let factories = self.attention_factories.read();

        // Find best kernel based on priority and device support
        let mut best: Option<(&str, i32)> = None;

        for (name, (info, _)) in factories.iter() {
            if !info.supported_devices.contains(&config.device) {
                continue;
            }

            let priority = if hint.memory_constrained
                && info.features.contains(&"flash_attention".to_string())
            {
                info.priority + 20 // Boost flash attention for memory-constrained scenarios
            } else if hint.batch_size > 8
                && info.features.contains(&"flash_attention".to_string())
            {
                info.priority + 10 // Boost for large batches
            } else {
                info.priority
            };

            if best.is_none() || priority > best.unwrap().1 {
                best = Some((name, priority));
            }
        }

        let (best_name, _) = best.ok_or_else(|| {
            FerrumError::not_found(format!(
                "No suitable attention kernel for device {:?}",
                config.device
            ))
        })?;

        debug!(
            "Selected attention kernel: {} for device {:?}",
            best_name, config.device
        );

        let best_name = best_name.to_string();
        drop(factories);
        self.get_attention_kernel(&best_name, config)
    }

    /// Select best fused ops for device
    pub fn select_fused_ops(
        &self,
        config: &FusedOpsConfig,
    ) -> Result<Arc<dyn FusedOps>> {
        let factories = self.fused_factories.read();

        let mut best: Option<(&str, i32)> = None;

        for (name, (info, _)) in factories.iter() {
            if !info.supported_devices.contains(&config.device) {
                continue;
            }

            if best.is_none() || info.priority > best.unwrap().1 {
                best = Some((name, info.priority));
            }
        }

        let (best_name, _) = best.ok_or_else(|| {
            FerrumError::not_found(format!(
                "No suitable fused ops for device {:?}",
                config.device
            ))
        })?;

        debug!(
            "Selected fused ops: {} for device {:?}",
            best_name, config.device
        );

        let best_name = best_name.to_string();
        drop(factories);
        self.get_fused_ops(&best_name, config)
    }

    /// List available attention kernels
    pub fn list_attention_kernels(&self) -> Vec<KernelInfo> {
        self.attention_factories
            .read()
            .values()
            .map(|(info, _)| info.clone())
            .collect()
    }

    /// List available fused ops
    pub fn list_fused_ops(&self) -> Vec<KernelInfo> {
        self.fused_factories
            .read()
            .values()
            .map(|(info, _)| info.clone())
            .collect()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Global kernel registry
static GLOBAL_KERNEL_REGISTRY: std::sync::OnceLock<KernelRegistry> = std::sync::OnceLock::new();

/// Get the global kernel registry
pub fn global_kernel_registry() -> &'static KernelRegistry {
    GLOBAL_KERNEL_REGISTRY.get_or_init(KernelRegistry::with_defaults)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = KernelRegistry::with_defaults();
        let kernels = registry.list_attention_kernels();
        assert!(!kernels.is_empty());
    }

    #[test]
    fn test_kernel_info_builder() {
        let info = KernelInfo::new("test", "Test kernel")
            .with_device(Device::CPU)
            .with_feature("test_feature")
            .with_priority(5);

        assert_eq!(info.name, "test");
        assert_eq!(info.priority, 5);
        assert!(info.features.contains(&"test_feature".to_string()));
    }

    #[test]
    fn test_kernel_selection() {
        let registry = KernelRegistry::with_defaults();
        let config = AttentionConfig::standard(32, 128);
        let hint = PerformanceHint::default();

        let kernel = registry.select_attention_kernel(&config, &hint);
        assert!(kernel.is_ok());
    }

    #[test]
    fn test_memory_constrained_selection() {
        let registry = KernelRegistry::with_defaults();
        let config = AttentionConfig::standard(32, 128);
        let hint = PerformanceHint {
            memory_constrained: true,
            ..Default::default()
        };

        let kernel = registry.select_attention_kernel(&config, &hint);
        assert!(kernel.is_ok());
        // Should prefer flash attention when memory constrained
        let kernel = kernel.unwrap();
        assert_eq!(kernel.attention_type(), AttentionType::Flash);
    }

    #[test]
    fn test_fused_ops_selection() {
        let registry = KernelRegistry::with_defaults();
        let config = FusedOpsConfig::default();
        let fused = registry.select_fused_ops(&config);
        assert!(fused.is_ok());
    }

    #[test]
    fn test_performance_hint_for_prefill() {
        let hint = PerformanceHint::for_prefill(4, 2048);
        assert_eq!(hint.batch_size, 4);
        assert_eq!(hint.seq_len, 2048);
    }

    #[test]
    fn test_performance_hint_for_decode() {
        let hint = PerformanceHint::for_decode(8);
        assert_eq!(hint.batch_size, 8);
        assert_eq!(hint.seq_len, 1);
    }
}
