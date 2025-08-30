//! Type definitions for runtime operations
//!
//! This module defines the core types used throughout the runtime system.

use ferrum_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;


/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device identifier
    pub device: Device,
    
    /// Device name
    pub name: String,
    
    /// Total memory in bytes
    pub total_memory: usize,
    
    /// Available memory in bytes
    pub available_memory: usize,
    
    /// Compute capability
    pub compute_capability: ComputeCapability,
    
    /// Number of compute units/SMs
    pub compute_units: u32,
    
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    
    /// Maximum block dimensions
    pub max_block_dims: (u32, u32, u32),
    
    /// Maximum grid dimensions
    pub max_grid_dims: (u32, u32, u32),
    
    /// Shared memory per block
    pub shared_memory_per_block: usize,
    
    /// Warp/wavefront size
    pub warp_size: u32,
    
    /// Clock rate in MHz
    pub clock_rate: u32,
    
    /// Memory clock rate in MHz
    pub memory_clock_rate: u32,
    
    /// Memory bus width in bits
    pub memory_bus_width: u32,
    
    /// Whether device supports unified memory
    pub unified_memory: bool,
}

/// Memory information for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total: usize,
    
    /// Free memory in bytes
    pub free: usize,
    
    /// Used memory in bytes
    pub used: usize,
    
    /// Reserved memory in bytes
    pub reserved: usize,
    
    /// Memory pool size in bytes
    pub pool_size: Option<usize>,
    
    /// Memory fragmentation ratio (0.0 - 1.0)
    pub fragmentation: f32,
}

/// Stream handle for asynchronous operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamHandle(pub u64);

/// Kernel handle for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelHandle(pub u64);

/// Compute capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Major version
    pub major: u32,
    
    /// Minor version
    pub minor: u32,
    
    /// Supported data types
    pub supported_dtypes: Vec<String>,
    
    /// Supported operations
    pub supported_ops: Vec<String>,
    
    /// Maximum tensor dimensions
    pub max_tensor_dims: u32,
    
    /// Supports tensor cores
    pub tensor_cores: bool,
    
    /// Supports mixed precision
    pub mixed_precision: bool,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Default device
    pub default_device: Device,
    
    /// Memory pool configurations per device
    pub memory_pools: HashMap<Device, MemoryPoolConfig>,
    
    /// Number of streams per device
    pub streams_per_device: u32,
    
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    
    /// Enable kernel caching
    pub enable_kernel_caching: bool,
    
    /// Kernel cache directory
    pub kernel_cache_dir: Option<String>,
    
    /// Enable profiling
    pub enable_profiling: bool,
    
    /// Synchronization mode
    pub sync_mode: SynchronizationMode,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    
    /// Maximum pool size in bytes
    pub max_size: usize,
    
    /// Growth factor when expanding pool
    pub growth_factor: f32,
    
    /// Enable memory defragmentation
    pub enable_defragmentation: bool,
    
    /// Defragmentation threshold
    pub defrag_threshold: f32,
}

/// Execution context for compute operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionContext(pub u64);

/// Memory transfer specification
#[derive(Debug, Clone)]
pub struct MemoryTransfer {
    /// Source memory handle
    pub src: MemoryHandle,
    
    /// Destination memory handle
    pub dst: MemoryHandle,
    
    /// Number of bytes to transfer
    pub size: usize,
    
    /// Source offset in bytes
    pub src_offset: usize,
    
    /// Destination offset in bytes
    pub dst_offset: usize,
    
    /// Transfer direction
    pub direction: TransferDirection,
}

/// Memory transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferDirection {
    /// Host to Device
    HostToDevice,
    
    /// Device to Host
    DeviceToHost,
    
    /// Device to Device (same device)
    DeviceToDevice,
    
    /// Peer to Peer (different devices)
    PeerToPeer,
}

/// Synchronization point for stream coordination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SynchronizationPoint(pub u64);

/// Synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynchronizationMode {
    /// Automatic synchronization
    Auto,
    
    /// Manual synchronization
    Manual,
    
    /// Blocking operations
    Blocking,
}

/// Memory handle for GPU memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryHandle(pub u64);

/// Runtime statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeStats {
    /// Total memory allocations
    pub total_allocations: u64,
    
    /// Total memory deallocations
    pub total_deallocations: u64,
    
    /// Current memory usage in bytes
    pub current_memory_usage: usize,
    
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    
    /// Total kernel launches
    pub total_kernel_launches: u64,
    
    /// Total memory transfers
    pub total_memory_transfers: u64,
    
    /// Total bytes transferred
    pub total_bytes_transferred: u64,
    
    /// Average kernel execution time in microseconds
    pub avg_kernel_time_us: f64,
    
    /// Average memory transfer time in microseconds
    pub avg_transfer_time_us: f64,
}

/// Profiling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingInfo {
    /// Operation name
    pub name: String,
    
    /// Start time (as microseconds since epoch)
    pub start_time_us: u64,
    
    /// Duration in microseconds
    pub duration_us: u64,
    
    /// Device used
    pub device: Device,
    
    /// Stream used
    pub stream: StreamHandle,
    
    /// Memory usage during operation
    pub memory_usage: usize,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Error types specific to runtime operations
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Device not available: {device:?}")]
    DeviceNotAvailable { device: Device },
    
    #[error("Out of memory on device: {device:?}")]
    OutOfMemory { device: Device },
    
    #[error("Invalid memory handle: {handle:?}")]
    InvalidMemoryHandle { handle: MemoryHandle },
    
    #[error("Invalid stream handle: {handle:?}")]
    InvalidStreamHandle { handle: StreamHandle },
    
    #[error("Kernel compilation failed: {error}")]
    KernelCompilationFailed { error: String },
    
    #[error("Kernel execution failed: {error}")]
    KernelExecutionFailed { error: String },
    
    #[error("Synchronization failed: {error}")]
    SynchronizationFailed { error: String },
    
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            default_device: Device::CPU,
            memory_pools: HashMap::new(),
            streams_per_device: 4,
            enable_memory_optimization: true,
            enable_kernel_caching: true,
            kernel_cache_dir: None,
            enable_profiling: false,
            sync_mode: SynchronizationMode::Auto,
        }
    }
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024, // 256MB
            max_size: 8 * 1024 * 1024 * 1024, // 8GB
            growth_factor: 1.5,
            enable_defragmentation: true,
            defrag_threshold: 0.3,
        }
    }
}
