//! Parallel Executor
//!
//! Provides execution coordination for multi-GPU inference.

use async_trait::async_trait;
use ferrum_interfaces::ModelExecutor;
use ferrum_types::{Device, FerrumError, Result};
use std::sync::Arc;
use tracing::{debug, info};

use super::config::{ParallelConfig, ParallelismType};
use super::device::{DeviceInfo, DeviceManager};
use super::tensor_parallel::{TensorParallelConfig, TensorParallelGroup};

/// Parallel executor trait
#[async_trait]
pub trait ParallelExecutor: Send + Sync {
    /// Get the parallelism configuration
    fn config(&self) -> &ParallelConfig;

    /// Get local rank
    fn rank(&self) -> usize;

    /// Get world size
    fn world_size(&self) -> usize;

    /// Check if this is the master rank
    fn is_master(&self) -> bool {
        self.rank() == 0
    }

    /// Get the local device
    fn device(&self) -> Device;

    /// Barrier synchronization across all ranks
    async fn barrier(&self) -> Result<()>;

    /// All-reduce operation (sum by default)
    async fn all_reduce(&self, data: &mut [f32]) -> Result<()>;

    /// All-gather operation
    async fn all_gather(&self, local_data: &[f32]) -> Result<Vec<f32>>;

    /// Broadcast from master to all ranks
    async fn broadcast(&self, data: &mut [f32]) -> Result<()>;
}

/// Single-GPU executor (no parallelism)
pub struct SingleGpuExecutor {
    config: ParallelConfig,
    device: Device,
}

impl SingleGpuExecutor {
    pub fn new(device: Device) -> Self {
        Self {
            config: ParallelConfig::single_gpu(device.clone()),
            device,
        }
    }
}

#[async_trait]
impl ParallelExecutor for SingleGpuExecutor {
    fn config(&self) -> &ParallelConfig {
        &self.config
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    async fn barrier(&self) -> Result<()> {
        // No-op for single GPU
        Ok(())
    }

    async fn all_reduce(&self, _data: &mut [f32]) -> Result<()> {
        // No-op for single GPU
        Ok(())
    }

    async fn all_gather(&self, local_data: &[f32]) -> Result<Vec<f32>> {
        // Just return copy of local data
        Ok(local_data.to_vec())
    }

    async fn broadcast(&self, _data: &mut [f32]) -> Result<()> {
        // No-op for single GPU
        Ok(())
    }
}

/// Simulated multi-GPU executor for development/testing
///
/// Uses threading to simulate multiple GPUs within a single process.
/// Useful for testing parallel algorithms without actual multi-GPU hardware.
pub struct SimulatedParallelExecutor {
    config: ParallelConfig,
    rank: usize,
    /// Shared buffers for simulated communication
    shared_buffers: Arc<parking_lot::RwLock<Vec<Vec<f32>>>>,
    /// Barrier for synchronization
    barrier: Arc<std::sync::Barrier>,
}

impl SimulatedParallelExecutor {
    /// Create a simulated parallel executor
    pub fn new(
        config: ParallelConfig,
        rank: usize,
        shared_buffers: Arc<parking_lot::RwLock<Vec<Vec<f32>>>>,
        barrier: Arc<std::sync::Barrier>,
    ) -> Self {
        Self {
            config,
            rank,
            shared_buffers,
            barrier,
        }
    }
}

#[async_trait]
impl ParallelExecutor for SimulatedParallelExecutor {
    fn config(&self) -> &ParallelConfig {
        &self.config
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.config.world_size()
    }

    fn device(&self) -> Device {
        self.config.devices.get(self.rank).cloned().unwrap_or(Device::CPU)
    }

    async fn barrier(&self) -> Result<()> {
        self.barrier.wait();
        Ok(())
    }

    async fn all_reduce(&self, data: &mut [f32]) -> Result<()> {
        // Store local data
        {
            let mut buffers = self.shared_buffers.write();
            buffers[self.rank] = data.to_vec();
        }

        // Wait for all ranks
        self.barrier.wait();

        // Compute sum
        {
            let buffers = self.shared_buffers.read();
            for i in 0..data.len() {
                let mut sum = 0.0f32;
                for rank_data in buffers.iter() {
                    if i < rank_data.len() {
                        sum += rank_data[i];
                    }
                }
                data[i] = sum;
            }
        }

        // Sync after reduction
        self.barrier.wait();
        Ok(())
    }

    async fn all_gather(&self, local_data: &[f32]) -> Result<Vec<f32>> {
        // Store local data
        {
            let mut buffers = self.shared_buffers.write();
            buffers[self.rank] = local_data.to_vec();
        }

        // Wait for all ranks
        self.barrier.wait();

        // Gather all data
        let result = {
            let buffers = self.shared_buffers.read();
            buffers.iter().flatten().copied().collect()
        };

        // Sync after gather
        self.barrier.wait();
        Ok(result)
    }

    async fn broadcast(&self, data: &mut [f32]) -> Result<()> {
        if self.rank == 0 {
            // Master stores data
            let mut buffers = self.shared_buffers.write();
            buffers[0] = data.to_vec();
        }

        // Wait for master
        self.barrier.wait();

        // Non-master ranks copy data
        if self.rank != 0 {
            let buffers = self.shared_buffers.read();
            data.copy_from_slice(&buffers[0]);
        }

        // Sync after broadcast
        self.barrier.wait();
        Ok(())
    }
}

/// Factory for creating parallel executors
pub struct ParallelExecutorFactory;

impl ParallelExecutorFactory {
    /// Create executor based on configuration
    pub fn create(config: &ParallelConfig, rank: usize) -> Result<Box<dyn ParallelExecutor>> {
        match config.parallelism_type {
            ParallelismType::None => {
                let device = config.devices.first().cloned().unwrap_or(Device::CPU);
                Ok(Box::new(SingleGpuExecutor::new(device)))
            }
            ParallelismType::Tensor | ParallelismType::Pipeline | ParallelismType::Data | ParallelismType::Hybrid => {
                // For now, return a simulated executor
                // In production, this would create actual distributed executors
                info!("Creating simulated parallel executor (rank {})", rank);
                let world_size = config.world_size();
                let shared_buffers = Arc::new(parking_lot::RwLock::new(vec![Vec::new(); world_size]));
                let barrier = Arc::new(std::sync::Barrier::new(world_size));

                Ok(Box::new(SimulatedParallelExecutor::new(
                    config.clone(),
                    rank,
                    shared_buffers,
                    barrier,
                )))
            }
        }
    }

    /// Create executors for all ranks
    pub fn create_all(config: &ParallelConfig) -> Result<Vec<Box<dyn ParallelExecutor>>> {
        let world_size = config.world_size();
        let mut executors = Vec::with_capacity(world_size);

        // Create shared state for simulated executors
        let shared_buffers = Arc::new(parking_lot::RwLock::new(vec![Vec::new(); world_size]));
        let barrier = Arc::new(std::sync::Barrier::new(world_size));

        for rank in 0..world_size {
            let executor: Box<dyn ParallelExecutor> = match config.parallelism_type {
                ParallelismType::None => {
                    let device = config.devices.first().cloned().unwrap_or(Device::CPU);
                    Box::new(SingleGpuExecutor::new(device))
                }
                _ => Box::new(SimulatedParallelExecutor::new(
                    config.clone(),
                    rank,
                    Arc::clone(&shared_buffers),
                    Arc::clone(&barrier),
                )),
            };
            executors.push(executor);
        }

        Ok(executors)
    }
}

/// Helper to select parallelism strategy based on model and hardware
pub struct ParallelStrategySelector;

impl ParallelStrategySelector {
    /// Select optimal parallelism strategy
    pub fn select(
        model_size_bytes: usize,
        num_layers: usize,
        device_manager: &DeviceManager,
    ) -> ParallelConfig {
        let gpu_devices = device_manager.get_gpu_devices();

        if gpu_devices.is_empty() {
            debug!("No GPUs available, using CPU");
            return ParallelConfig::single_gpu(Device::CPU);
        }

        // Check if model fits on a single GPU
        let first_gpu = &gpu_devices[0];
        if first_gpu.capability.can_fit_model(model_size_bytes) {
            debug!("Model fits on single GPU");
            return ParallelConfig::single_gpu(first_gpu.device.clone());
        }

        // Need multi-GPU parallelism
        let devices: Vec<_> = gpu_devices.iter().map(|d| d.device.clone()).collect();
        let num_gpus = devices.len();

        // Calculate total GPU memory
        let total_memory: usize = gpu_devices.iter().map(|d| d.capability.total_memory).sum();

        if total_memory >= model_size_bytes {
            // Tensor parallelism if model can be sharded across available GPUs
            if num_gpus >= 2 && num_gpus <= 8 {
                debug!("Using tensor parallelism with {} GPUs", num_gpus);
                return ParallelConfig::tensor_parallel(devices);
            }
        }

        // Pipeline parallelism for many GPUs or very large models
        if num_gpus >= 2 {
            debug!("Using pipeline parallelism with {} stages", num_gpus);
            return ParallelConfig::pipeline_parallel(devices);
        }

        // Fallback to single GPU with memory optimizations
        ParallelConfig::single_gpu(first_gpu.device.clone())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_single_gpu_executor() {
        let executor = SingleGpuExecutor::new(Device::CPU);

        assert_eq!(executor.rank(), 0);
        assert_eq!(executor.world_size(), 1);
        assert!(executor.is_master());

        // Operations should be no-ops
        executor.barrier().await.unwrap();

        let mut data = vec![1.0, 2.0, 3.0];
        executor.all_reduce(&mut data).await.unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]); // Unchanged

        let gathered = executor.all_gather(&[1.0, 2.0]).await.unwrap();
        assert_eq!(gathered, vec![1.0, 2.0]);
    }

    #[test]
    fn test_parallel_executor_factory() {
        let config = ParallelConfig::single_gpu(Device::CPU);
        let executor = ParallelExecutorFactory::create(&config, 0).unwrap();
        assert_eq!(executor.world_size(), 1);
    }

    #[test]
    fn test_strategy_selector_single_gpu() {
        let manager = DeviceManager::new();
        manager.discover_devices().unwrap();

        // Small model should use single GPU
        let config = ParallelStrategySelector::select(
            1024 * 1024 * 100, // 100MB model
            32,
            &manager,
        );

        // On a machine with GPUs, might get GPU; otherwise CPU
        assert_eq!(config.world_size(), 1);
    }
}


