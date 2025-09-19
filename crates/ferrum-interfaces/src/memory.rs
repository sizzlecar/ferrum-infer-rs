//! Memory management interfaces for device memory operations
//!
//! This module provides device memory management abstractions, separate from
//! KV cache management. It handles raw memory allocation, transfers, and
//! memory pool management across different devices.

use ferrum_types::{Device, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Device memory manager for raw memory operations
#[async_trait]
pub trait DeviceMemoryManager: Send + Sync {
    /// Allocate memory on device
    async fn allocate(&self, size: usize, device: &Device) -> Result<MemoryHandle>;
    
    /// Allocate aligned memory
    async fn allocate_aligned(
        &self,
        size: usize,
        alignment: usize,
        device: &Device,
    ) -> Result<MemoryHandle>;
    
    /// Deallocate memory
    async fn deallocate(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Copy memory between handles
    async fn copy(
        &self,
        src: MemoryHandle,
        dst: MemoryHandle,
        size: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()>;
    
    /// Copy memory between devices asynchronously
    async fn copy_async(
        &self,
        transfer: MemoryTransfer,
        stream: Option<StreamHandle>,
    ) -> Result<()>;
    
    /// Get memory information for device
    async fn memory_info(&self, device: &Device) -> Result<MemoryInfo>;
    
    /// Get handle information
    fn handle_info(&self, handle: MemoryHandle) -> Option<MemoryHandleInfo>;
    
    /// Set memory pool configuration
    async fn configure_pool(&self, device: &Device, config: MemoryPoolConfig) -> Result<()>;
    
    /// Defragment memory (if supported)
    async fn defragment(&self, device: &Device) -> Result<DefragmentationStats>;
    
    /// Set memory pressure callback
    fn set_pressure_callback(
        &self,
        callback: Box<dyn Fn(MemoryPressure) + Send + Sync>,
    );
}

/// Memory handle representing allocated memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryHandle(pub u64);

impl MemoryHandle {
    /// Create new memory handle
    pub fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get handle ID
    pub fn id(&self) -> u64 {
        self.0
    }
    
    /// Check if handle is valid (non-zero)
    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
}

/// Stream handle for asynchronous operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHandle(pub u64);

impl StreamHandle {
    /// Create new stream handle
    pub fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get default stream (usually synchronous)
    pub fn default() -> Self {
        Self(0)
    }
}

/// Memory transfer specification
#[derive(Debug, Clone)]
pub struct MemoryTransfer {
    /// Source memory handle
    pub src: MemoryHandle,
    /// Destination memory handle
    pub dst: MemoryHandle,
    /// Number of bytes to transfer
    pub size: usize,
    /// Offset in source memory
    pub src_offset: usize,
    /// Offset in destination memory
    pub dst_offset: usize,
}

impl MemoryTransfer {
    /// Create new memory transfer
    pub fn new(src: MemoryHandle, dst: MemoryHandle, size: usize) -> Self {
        Self {
            src,
            dst,
            size,
            src_offset: 0,
            dst_offset: 0,
        }
    }
    
    /// Set source offset
    pub fn with_src_offset(mut self, offset: usize) -> Self {
        self.src_offset = offset;
        self
    }
    
    /// Set destination offset
    pub fn with_dst_offset(mut self, offset: usize) -> Self {
        self.dst_offset = offset;
        self
    }
}

/// Memory information for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory available on device (bytes)
    pub total_bytes: u64,
    /// Currently used memory (bytes)
    pub used_bytes: u64,
    /// Free memory available (bytes)
    pub free_bytes: u64,
    /// Memory reserved by the runtime/driver (bytes)
    pub reserved_bytes: u64,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Memory fragmentation ratio (0.0 - 1.0)
    pub fragmentation_ratio: f32,
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: Option<f32>,
}

impl MemoryInfo {
    /// Calculate memory utilization percentage
    pub fn utilization_percent(&self) -> f32 {
        if self.total_bytes > 0 {
            (self.used_bytes as f32 / self.total_bytes as f32) * 100.0
        } else {
            0.0
        }
    }
    
    /// Check if memory is under pressure
    pub fn pressure_level(&self) -> MemoryPressure {
        let utilization = self.utilization_percent();
        
        if utilization >= 95.0 {
            MemoryPressure::Critical
        } else if utilization >= 85.0 {
            MemoryPressure::High
        } else if utilization >= 70.0 {
            MemoryPressure::Medium
        } else {
            MemoryPressure::Low
        }
    }
}

/// Information about a memory handle
#[derive(Debug, Clone)]
pub struct MemoryHandleInfo {
    /// Memory handle
    pub handle: MemoryHandle,
    /// Size in bytes
    pub size: usize,
    /// Device where memory is allocated
    pub device: Device,
    /// Memory alignment
    pub alignment: usize,
    /// Allocation timestamp
    pub allocated_at: std::time::Instant,
    /// Whether memory is currently mapped
    pub is_mapped: bool,
    /// Memory type/usage hint
    pub memory_type: MemoryType,
}

/// Memory types for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// General purpose memory
    General,
    /// Memory optimized for tensor operations
    Tensor,
    /// Memory for KV cache
    Cache,
    /// Temporary/scratch memory
    Temporary,
    /// Pinned/page-locked memory for fast transfers
    Pinned,
    /// Mapped memory (shared between devices)
    Mapped,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// Low pressure - plenty of memory available
    Low,
    /// Medium pressure - should be conservative
    Medium,
    /// High pressure - consider cleanup/eviction
    High,
    /// Critical pressure - must free memory or reject requests
    Critical,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes
    pub initial_size: u64,
    /// Maximum pool size in bytes (None for unlimited)
    pub max_size: Option<u64>,
    /// Growth increment when expanding pool
    pub growth_increment: u64,
    /// Enable automatic pool expansion
    pub enable_auto_expansion: bool,
    /// Memory alignment for pool allocations
    pub alignment: usize,
    /// Pre-allocate entire pool upfront
    pub pre_allocate: bool,
    /// Enable pool statistics tracking
    pub enable_stats: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024 * 1024, // 1GB
            max_size: None,
            growth_increment: 512 * 1024 * 1024, // 512MB
            enable_auto_expansion: true,
            alignment: 256,
            pre_allocate: false,
            enable_stats: true,
        }
    }
}

/// Defragmentation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefragmentationStats {
    /// Memory freed by defragmentation (bytes)
    pub memory_freed: u64,
    /// Number of memory blocks moved
    pub blocks_moved: usize,
    /// Time taken for defragmentation
    pub time_taken_ms: u64,
    /// Fragmentation ratio before defragmentation
    pub fragmentation_before: f32,
    /// Fragmentation ratio after defragmentation
    pub fragmentation_after: f32,
}

/// Advanced memory operations
#[async_trait]
pub trait AdvancedMemoryManager: DeviceMemoryManager {
    /// Map memory for direct CPU access
    async fn map_memory(
        &self,
        handle: MemoryHandle,
        access: MemoryAccess,
    ) -> Result<*mut u8>;
    
    /// Unmap previously mapped memory
    async fn unmap_memory(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Create memory mapping between devices
    async fn create_mapping(
        &self,
        src_device: &Device,
        dst_device: &Device,
        size: usize,
    ) -> Result<(MemoryHandle, MemoryHandle)>;
    
    /// Enable memory prefetching
    async fn prefetch(
        &self,
        handle: MemoryHandle,
        target_device: &Device,
    ) -> Result<()>;
    
    /// Get memory access pattern statistics
    fn access_stats(&self, handle: MemoryHandle) -> Option<MemoryAccessStats>;
    
    /// Set memory usage hints
    async fn set_usage_hint(
        &self,
        handle: MemoryHandle,
        hint: MemoryUsageHint,
    ) -> Result<()>;
}

/// Memory access modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccess {
    /// Read-only access
    ReadOnly,
    /// Write-only access
    WriteOnly,
    /// Read-write access
    ReadWrite,
}

/// Memory usage hints for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryUsageHint {
    /// Memory will be accessed sequentially
    Sequential,
    /// Memory will be accessed randomly
    Random,
    /// Memory will be read frequently
    ReadMostly,
    /// Memory will be written frequently
    WriteMostly,
    /// Memory is temporary and can be freed aggressively
    Temporary,
    /// Memory should be kept resident
    Resident,
}

/// Memory access pattern statistics
#[derive(Debug, Clone)]
pub struct MemoryAccessStats {
    /// Total number of reads
    pub read_count: u64,
    /// Total number of writes
    pub write_count: u64,
    /// Average read size
    pub avg_read_size: usize,
    /// Average write size
    pub avg_write_size: usize,
    /// Last access timestamp
    pub last_access: std::time::Instant,
    /// Access pattern type (detected)
    pub pattern_type: AccessPatternType,
}

/// Detected access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPatternType {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Burst access pattern
    Burst,
    /// Mixed access pattern
    Mixed,
    /// Unknown/undetected pattern
    Unknown,
}

/// Stream manager for asynchronous operations
#[async_trait]
pub trait StreamManager: Send + Sync {
    /// Create new compute stream
    async fn create_stream(&self, device: &Device) -> Result<StreamHandle>;
    
    /// Destroy stream
    async fn destroy_stream(&self, stream: StreamHandle) -> Result<()>;
    
    /// Synchronize stream (wait for all operations to complete)
    async fn synchronize_stream(&self, stream: StreamHandle) -> Result<()>;
    
    /// Check if stream operations are complete
    async fn is_stream_ready(&self, stream: StreamHandle) -> Result<bool>;
    
    /// Get default stream for device
    fn default_stream(&self, device: &Device) -> StreamHandle;
    
    /// Record synchronization point
    async fn record_event(&self, stream: StreamHandle) -> Result<EventHandle>;
    
    /// Wait for event on stream
    async fn wait_event(&self, stream: StreamHandle, event: EventHandle) -> Result<()>;
}

/// Event handle for stream synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventHandle(pub u64);

/// Memory manager factory
#[async_trait]
pub trait MemoryManagerFactory: Send + Sync {
    /// Create memory manager for device
    async fn create_memory_manager(
        &self,
        device: &Device,
        config: &MemoryManagerConfig,
    ) -> Result<Box<dyn DeviceMemoryManager>>;
    
    /// Create advanced memory manager
    async fn create_advanced_memory_manager(
        &self,
        device: &Device,
        config: &MemoryManagerConfig,
    ) -> Result<Box<dyn AdvancedMemoryManager>>;
    
    /// Create stream manager
    async fn create_stream_manager(
        &self,
        device: &Device,
    ) -> Result<Box<dyn StreamManager>>;
}

/// Memory manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagerConfig {
    /// Memory pool configurations per memory type
    pub pool_configs: HashMap<MemoryType, MemoryPoolConfig>,
    /// Enable memory tracking and statistics
    pub enable_tracking: bool,
    /// Enable automatic garbage collection
    pub enable_auto_gc: bool,
    /// Garbage collection trigger threshold
    pub gc_threshold: f32,
    /// Enable memory debugging
    pub enable_debug: bool,
    /// Maximum number of concurrent transfers
    pub max_concurrent_transfers: usize,
}

impl Default for MemoryManagerConfig {
    fn default() -> Self {
        let mut pool_configs = HashMap::new();
        pool_configs.insert(MemoryType::General, MemoryPoolConfig::default());
        
        Self {
            pool_configs,
            enable_tracking: true,
            enable_auto_gc: true,
            gc_threshold: 0.85,
            enable_debug: false,
            max_concurrent_transfers: 4,
        }
    }
}

/// Global memory monitor for system-wide memory tracking
pub trait GlobalMemoryMonitor: Send + Sync {
    /// Get memory information across all devices
    fn global_memory_info(&self) -> HashMap<Device, MemoryInfo>;
    
    /// Get total system memory pressure
    fn global_memory_pressure(&self) -> MemoryPressure;
    
    /// Register memory manager for monitoring
    fn register_manager(&mut self, device: Device, manager: &dyn DeviceMemoryManager);
    
    /// Unregister memory manager
    fn unregister_manager(&mut self, device: &Device);
    
    /// Set global memory pressure callback
    fn set_global_pressure_callback(
        &mut self,
        callback: Box<dyn Fn(HashMap<Device, MemoryPressure>) + Send + Sync>,
    );
    
    /// Force global garbage collection
    async fn global_gc(&self) -> Result<HashMap<Device, DefragmentationStats>>;
}

/// Memory allocation strategy
pub trait AllocationStrategy: Send + Sync {
    /// Select best device for allocation
    fn select_device(
        &self,
        size: usize,
        requirements: &AllocationRequirements,
        available_devices: &[Device],
        memory_info: &HashMap<Device, MemoryInfo>,
    ) -> Option<Device>;
    
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Requirements for memory allocation
#[derive(Debug, Clone)]
pub struct AllocationRequirements {
    /// Preferred devices in order
    pub preferred_devices: Vec<Device>,
    /// Memory type hint
    pub memory_type: MemoryType,
    /// Required alignment
    pub alignment: Option<usize>,
    /// Whether allocation is time-critical
    pub is_critical: bool,
    /// Expected lifetime
    pub expected_lifetime: Option<std::time::Duration>,
}

/// Best-fit allocation strategy
pub struct BestFitStrategy;

impl AllocationStrategy for BestFitStrategy {
    fn select_device(
        &self,
        size: usize,
        requirements: &AllocationRequirements,
        available_devices: &[Device],
        memory_info: &HashMap<Device, MemoryInfo>,
    ) -> Option<Device> {
        let mut best_device = None;
        let mut best_score = f32::NEG_INFINITY;
        
        for device in available_devices {
            if let Some(info) = memory_info.get(device) {
                // Check if device has enough memory
                if info.free_bytes < size as u64 {
                    continue;
                }
                
                // Prefer devices with just enough memory (best fit)
                let waste_ratio = (info.free_bytes - size as u64) as f32 / info.total_bytes as f32;
                let utilization = info.utilization_percent() / 100.0;
                
                // Score based on minimal waste and moderate utilization
                let score = 1.0 - waste_ratio - (utilization - 0.5).abs() * 0.5;
                
                // Bonus for preferred devices
                let preference_bonus = requirements.preferred_devices
                    .iter()
                    .position(|d| d == device)
                    .map(|pos| 1.0 / (pos as f32 + 1.0))
                    .unwrap_or(0.0) * 0.2;
                
                let final_score = score + preference_bonus;
                
                if final_score > best_score {
                    best_score = final_score;
                    best_device = Some(device.clone());
                }
            }
        }
        
        best_device
    }
    
    fn name(&self) -> &str {
        "best_fit"
    }
}
