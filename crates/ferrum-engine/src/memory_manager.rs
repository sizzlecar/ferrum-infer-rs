//! GPU/CPU memory management implementation

use async_trait::async_trait;
use ferrum_core::{
    MemoryManager, MemoryHandle, MemoryUsage, MemoryPressure,
    Result, Error,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use parking_lot::RwLock;
use tracing::{info, warn, debug};

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Total GPU memory in bytes
    pub gpu_memory_bytes: usize,
    
    /// Total CPU memory for swapping in bytes
    pub cpu_memory_bytes: usize,
    
    /// GPU memory fraction to use
    pub gpu_memory_fraction: f32,
    
    /// Enable memory swapping
    pub enable_swapping: bool,
    
    /// Memory pressure threshold (0.0 - 1.0)
    pub pressure_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            gpu_memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            cpu_memory_bytes: 32 * 1024 * 1024 * 1024, // 32GB
            gpu_memory_fraction: 0.9,
            enable_swapping: true,
            pressure_threshold: 0.85,
        }
    }
}

/// GPU memory manager
pub struct GpuMemoryManager {
    config: MemoryConfig,
    allocations: Arc<RwLock<HashMap<MemoryHandle, Allocation>>>,
    gpu_used: Arc<AtomicUsize>,
    cpu_used: Arc<AtomicUsize>,
    handle_counter: Arc<AtomicU64>,
    pressure_callbacks: Arc<RwLock<Vec<Box<dyn Fn(MemoryPressure) + Send + Sync>>>>,
}

/// Memory allocation information
struct Allocation {
    handle: MemoryHandle,
    size: usize,
    location: MemoryLocation,
    allocated_at: std::time::Instant,
}

/// Memory location
#[derive(Debug, Clone, Copy, PartialEq)]
enum MemoryLocation {
    GPU,
    CPU,
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    pub fn new(config: MemoryConfig) -> Self {
        let available_gpu = (config.gpu_memory_bytes as f32 * config.gpu_memory_fraction) as usize;
        let available_cpu = config.cpu_memory_bytes;
        
        info!("Initializing GpuMemoryManager with {} GB GPU, {} GB CPU memory",
              available_gpu / (1024 * 1024 * 1024),
              available_cpu / (1024 * 1024 * 1024));
        
        Self {
            config,
            allocations: Arc::new(RwLock::new(HashMap::new())),
            gpu_used: Arc::new(AtomicUsize::new(0)),
            cpu_used: Arc::new(AtomicUsize::new(0)),
            handle_counter: Arc::new(AtomicU64::new(0)),
            pressure_callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Get available GPU memory
    fn available_gpu_memory(&self) -> usize {
        let total = (self.config.gpu_memory_bytes as f32 * self.config.gpu_memory_fraction) as usize;
        let used = self.gpu_used.load(Ordering::Relaxed);
        total.saturating_sub(used)
    }
    
    /// Get available CPU memory
    fn available_cpu_memory(&self) -> usize {
        let used = self.cpu_used.load(Ordering::Relaxed);
        self.config.cpu_memory_bytes.saturating_sub(used)
    }
    
    /// Calculate memory pressure
    fn calculate_pressure(&self) -> MemoryPressure {
        let gpu_usage = self.gpu_used.load(Ordering::Relaxed) as f32
            / (self.config.gpu_memory_bytes as f32 * self.config.gpu_memory_fraction);
        
        if gpu_usage < 0.5 {
            MemoryPressure::Low
        } else if gpu_usage < 0.75 {
            MemoryPressure::Medium
        } else if gpu_usage < self.config.pressure_threshold {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        }
    }
    
    /// Trigger pressure callbacks if needed
    fn check_and_trigger_pressure(&self) {
        let pressure = self.calculate_pressure();
        
        if pressure >= MemoryPressure::High {
            let callbacks = self.pressure_callbacks.read();
            for callback in callbacks.iter() {
                callback(pressure);
            }
        }
    }
    
    /// Generate next memory handle
    fn next_handle(&self) -> MemoryHandle {
        let id = self.handle_counter.fetch_add(1, Ordering::Relaxed);
        MemoryHandle(id)
    }
}

#[async_trait]
impl MemoryManager for GpuMemoryManager {
    async fn allocate(&self, size: usize) -> Result<MemoryHandle> {
        // Check if we have enough GPU memory
        if size > self.available_gpu_memory() {
            // Try CPU memory if swapping is enabled
            if self.config.enable_swapping && size <= self.available_cpu_memory() {
                warn!("GPU memory insufficient, allocating {} bytes on CPU", size);
                
                let handle = self.next_handle();
                let allocation = Allocation {
                    handle,
                    size,
                    location: MemoryLocation::CPU,
                    allocated_at: std::time::Instant::now(),
                };
                
                self.allocations.write().insert(handle, allocation);
                self.cpu_used.fetch_add(size, Ordering::Relaxed);
                
                debug!("Allocated {} bytes on CPU with handle {:?}", size, handle);
                return Ok(handle);
            } else {
                return Err(Error::oom(format!(
                    "Cannot allocate {} bytes, available GPU: {}, CPU: {}",
                    size,
                    self.available_gpu_memory(),
                    self.available_cpu_memory()
                )));
            }
        }
        
        // Allocate on GPU
        let handle = self.next_handle();
        let allocation = Allocation {
            handle,
            size,
            location: MemoryLocation::GPU,
            allocated_at: std::time::Instant::now(),
        };
        
        self.allocations.write().insert(handle, allocation);
        self.gpu_used.fetch_add(size, Ordering::Relaxed);
        
        debug!("Allocated {} bytes on GPU with handle {:?}", size, handle);
        
        // Check memory pressure
        self.check_and_trigger_pressure();
        
        Ok(handle)
    }
    
    async fn deallocate(&self, handle: MemoryHandle) -> Result<()> {
        let mut allocations = self.allocations.write();
        
        if let Some(allocation) = allocations.remove(&handle) {
            match allocation.location {
                MemoryLocation::GPU => {
                    self.gpu_used.fetch_sub(allocation.size, Ordering::Relaxed);
                    debug!("Deallocated {} bytes from GPU", allocation.size);
                }
                MemoryLocation::CPU => {
                    self.cpu_used.fetch_sub(allocation.size, Ordering::Relaxed);
                    debug!("Deallocated {} bytes from CPU", allocation.size);
                }
            }
            Ok(())
        } else {
            Err(Error::not_found(format!("Memory handle {:?} not found", handle)))
        }
    }
    
    fn get_memory_usage(&self) -> MemoryUsage {
        let gpu_used = self.gpu_used.load(Ordering::Relaxed);
        let cpu_used = self.cpu_used.load(Ordering::Relaxed);
        let gpu_total = (self.config.gpu_memory_bytes as f32 * self.config.gpu_memory_fraction) as usize;
        
        MemoryUsage {
            total_bytes: gpu_total + self.config.cpu_memory_bytes,
            used_bytes: gpu_used + cpu_used,
            free_bytes: (gpu_total - gpu_used) + (self.config.cpu_memory_bytes - cpu_used),
            gpu_memory_bytes: Some(gpu_used),
            cpu_memory_bytes: Some(cpu_used),
        }
    }
    
    async fn swap_out(&self, handle: MemoryHandle) -> Result<()> {
        if !self.config.enable_swapping {
            return Err(Error::unsupported("Memory swapping is disabled"));
        }
        
        let mut allocations = self.allocations.write();
        
        if let Some(allocation) = allocations.get_mut(&handle) {
            if allocation.location != MemoryLocation::GPU {
                return Err(Error::invalid_request("Memory is not on GPU"));
            }
            
            // Check CPU memory availability
            if allocation.size > self.available_cpu_memory() {
                return Err(Error::oom("Insufficient CPU memory for swap"));
            }
            
            // Update counters
            self.gpu_used.fetch_sub(allocation.size, Ordering::Relaxed);
            self.cpu_used.fetch_add(allocation.size, Ordering::Relaxed);
            allocation.location = MemoryLocation::CPU;
            
            info!("Swapped out {} bytes from GPU to CPU", allocation.size);
            Ok(())
        } else {
            Err(Error::not_found(format!("Memory handle {:?} not found", handle)))
        }
    }
    
    async fn swap_in(&self, handle: MemoryHandle) -> Result<()> {
        if !self.config.enable_swapping {
            return Err(Error::unsupported("Memory swapping is disabled"));
        }
        
        let mut allocations = self.allocations.write();
        
        if let Some(allocation) = allocations.get_mut(&handle) {
            if allocation.location != MemoryLocation::CPU {
                return Err(Error::invalid_request("Memory is not on CPU"));
            }
            
            // Check GPU memory availability
            if allocation.size > self.available_gpu_memory() {
                return Err(Error::oom("Insufficient GPU memory for swap"));
            }
            
            // Update counters
            self.cpu_used.fetch_sub(allocation.size, Ordering::Relaxed);
            self.gpu_used.fetch_add(allocation.size, Ordering::Relaxed);
            allocation.location = MemoryLocation::GPU;
            
            info!("Swapped in {} bytes from CPU to GPU", allocation.size);
            
            // Check memory pressure
            self.check_and_trigger_pressure();
            
            Ok(())
        } else {
            Err(Error::not_found(format!("Memory handle {:?} not found", handle)))
        }
    }
    
    fn set_pressure_callback(&self, callback: Box<dyn Fn(MemoryPressure) + Send + Sync>) {
        self.pressure_callbacks.write().push(callback);
    }
}
