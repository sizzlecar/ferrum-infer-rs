//! Core runtime traits
//!
//! This module defines the abstract interfaces for runtime execution,
//! GPU operations, and hardware management.

use async_trait::async_trait;
use ferrum_core::{Result, Device, DataType, Tensor};
use crate::types::*;
use std::sync::Arc;

/// Main runtime trait for managing execution environment
#[async_trait]
pub trait Runtime: Send + Sync {
    /// Initialize the runtime
    async fn initialize(&mut self) -> Result<()>;
    
    /// Shutdown the runtime gracefully
    async fn shutdown(&mut self) -> Result<()>;
    
    /// Get device manager
    fn device_manager(&self) -> Arc<dyn DeviceManager>;
    
    /// Get memory manager
    fn memory_manager(&self) -> Arc<dyn MemoryManager>;
    
    /// Get stream manager
    fn stream_manager(&self) -> Arc<dyn StreamManager>;
    
    /// Get kernel executor
    fn kernel_executor(&self) -> Arc<dyn KernelExecutor>;
    
    /// Get runtime configuration
    fn config(&self) -> &RuntimeConfig;
    
    /// Check if runtime supports a specific device
    fn supports_device(&self, device: &Device) -> bool;
}

/// Device management trait
#[async_trait]
pub trait DeviceManager: Send + Sync {
    /// Get available devices
    async fn get_devices(&self) -> Result<Vec<DeviceInfo>>;
    
    /// Get device information
    async fn get_device_info(&self, device: &Device) -> Result<DeviceInfo>;
    
    /// Set current device
    async fn set_device(&self, device: &Device) -> Result<()>;
    
    /// Get current device
    fn current_device(&self) -> Device;
    
    /// Check device availability
    async fn is_device_available(&self, device: &Device) -> bool;
    
    /// Get device utilization
    async fn get_device_utilization(&self, device: &Device) -> Result<f32>;
    
    /// Synchronize device
    async fn synchronize_device(&self, device: &Device) -> Result<()>;
}

/// Memory management trait for GPU/CPU memory operations
#[async_trait]
pub trait MemoryManager: Send + Sync {
    /// Allocate memory on device
    async fn allocate(&self, size: usize, device: &Device) -> Result<MemoryHandle>;
    
    /// Deallocate memory
    async fn deallocate(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Copy memory between devices
    async fn copy(&self, transfer: MemoryTransfer) -> Result<()>;
    
    /// Copy memory asynchronously
    async fn copy_async(&self, transfer: MemoryTransfer, stream: StreamHandle) -> Result<()>;
    
    /// Get memory information
    async fn get_memory_info(&self, device: &Device) -> Result<MemoryInfo>;
    
    /// Set memory pool size
    async fn set_memory_pool_size(&self, device: &Device, size: usize) -> Result<()>;
    
    /// Get memory handle information
    fn get_handle_info(&self, handle: MemoryHandle) -> Option<MemoryHandleInfo>;
}

/// Stream management for asynchronous operations
#[async_trait]
pub trait StreamManager: Send + Sync {
    /// Create a new stream
    async fn create_stream(&self, device: &Device) -> Result<StreamHandle>;
    
    /// Destroy a stream
    async fn destroy_stream(&self, stream: StreamHandle) -> Result<()>;
    
    /// Synchronize stream
    async fn synchronize_stream(&self, stream: StreamHandle) -> Result<()>;
    
    /// Check if stream is ready
    async fn is_stream_ready(&self, stream: StreamHandle) -> Result<bool>;
    
    /// Get default stream for device
    fn get_default_stream(&self, device: &Device) -> StreamHandle;
    
    /// Record synchronization point
    async fn record_event(&self, stream: StreamHandle) -> Result<SynchronizationPoint>;
    
    /// Wait for synchronization point
    async fn wait_event(&self, stream: StreamHandle, event: SynchronizationPoint) -> Result<()>;
}

/// Kernel execution trait for custom GPU operations
#[async_trait]
pub trait KernelExecutor: Send + Sync {
    /// Load kernel from source
    async fn load_kernel(&self, source: &str, name: &str) -> Result<KernelHandle>;
    
    /// Execute kernel
    async fn execute_kernel(
        &self,
        kernel: KernelHandle,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        args: &[KernelArg],
        stream: StreamHandle,
    ) -> Result<()>;
    
    /// Get kernel information
    fn get_kernel_info(&self, kernel: KernelHandle) -> Option<KernelInfo>;
    
    /// Unload kernel
    async fn unload_kernel(&self, kernel: KernelHandle) -> Result<()>;
}

/// Tensor operations trait
#[async_trait]
pub trait TensorOps: Send + Sync {
    /// Create tensor on device
    async fn create_tensor(
        &self,
        shape: &[usize],
        dtype: DataType,
        device: &Device,
    ) -> Result<Tensor>;
    
    /// Copy tensor to device
    async fn to_device(&self, tensor: &Tensor, device: &Device) -> Result<Tensor>;
    
    /// Copy tensor to CPU
    async fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor>;
    
    /// Reshape tensor
    async fn reshape(&self, tensor: &Tensor, shape: &[usize]) -> Result<Tensor>;
    
    /// Matrix multiplication
    async fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    
    /// Element-wise addition
    async fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    
    /// Apply softmax
    async fn softmax(&self, tensor: &Tensor, dim: i32) -> Result<Tensor>;
    
    /// Apply layer normalization
    async fn layer_norm(
        &self,
        tensor: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor>;
    
    /// Apply RMS normalization
    async fn rms_norm(&self, tensor: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    
    /// Apply ReLU activation
    async fn relu(&self, tensor: &Tensor) -> Result<Tensor>;
    
    /// Apply GELU activation
    async fn gelu(&self, tensor: &Tensor) -> Result<Tensor>;
    
    /// Apply SiLU activation
    async fn silu(&self, tensor: &Tensor) -> Result<Tensor>;
}

/// Compute backend trait for different hardware backends
#[async_trait]
pub trait ComputeBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Get supported devices
    async fn get_supported_devices(&self) -> Result<Vec<Device>>;
    
    /// Get compute capabilities
    async fn get_compute_capabilities(&self, device: &Device) -> Result<ComputeCapability>;
    
    /// Create execution context
    async fn create_context(&self, device: &Device) -> Result<ExecutionContext>;
    
    /// Destroy execution context
    async fn destroy_context(&self, context: ExecutionContext) -> Result<()>;
    
    /// Check if backend is available
    fn is_available(&self) -> bool;
    
    /// Get backend version
    fn version(&self) -> String;
}

/// Memory handle for tracking allocated memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryHandle(pub u64);

/// Memory handle information
#[derive(Debug, Clone)]
pub struct MemoryHandleInfo {
    pub handle: MemoryHandle,
    pub size: usize,
    pub device: Device,
    pub alignment: usize,
    pub allocated_at: std::time::Instant,
}

/// Kernel argument types
#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(MemoryHandle),
    Scalar(ScalarValue),
    LocalMemory(usize),
}

/// Scalar values for kernel arguments
#[derive(Debug, Clone)]
pub enum ScalarValue {
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

/// Kernel information
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub name: String,
    pub max_threads_per_block: u32,
    pub shared_memory_size: usize,
    pub register_count: u32,
}
