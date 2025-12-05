//! Parallel Configuration
//!
//! Configuration types for multi-GPU parallelism.

use ferrum_types::Device;
use serde::{Deserialize, Serialize};

/// Type of parallelism strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismType {
    /// No parallelism (single GPU)
    None,
    /// Tensor parallelism: split tensors across GPUs
    Tensor,
    /// Pipeline parallelism: split layers across GPUs
    Pipeline,
    /// Data parallelism: replicate model, split batches
    Data,
    /// Hybrid: combination of tensor and pipeline parallelism
    Hybrid,
}

impl Default for ParallelismType {
    fn default() -> Self {
        Self::None
    }
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Type of parallelism to use
    pub parallelism_type: ParallelismType,
    /// Devices to use for parallel execution
    pub devices: Vec<Device>,
    /// Number of tensor parallel ranks
    pub tensor_parallel_size: usize,
    /// Number of pipeline parallel stages
    pub pipeline_parallel_size: usize,
    /// Whether to enable memory optimization
    pub enable_memory_optimization: bool,
    /// Communication backend (nccl, gloo, etc.)
    pub communication_backend: String,
    /// Maximum chunk size for all-reduce operations
    pub max_chunk_size: usize,
    /// Enable overlapping communication with computation
    pub overlap_communication: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            parallelism_type: ParallelismType::None,
            devices: vec![Device::CPU],
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            enable_memory_optimization: true,
            communication_backend: "cpu".to_string(),
            max_chunk_size: 1024 * 1024, // 1MB
            overlap_communication: true,
        }
    }
}

impl ParallelConfig {
    /// Create config for single GPU
    pub fn single_gpu(device: Device) -> Self {
        Self {
            parallelism_type: ParallelismType::None,
            devices: vec![device],
            ..Default::default()
        }
    }

    /// Create config for tensor parallelism
    pub fn tensor_parallel(devices: Vec<Device>) -> Self {
        let size = devices.len();
        Self {
            parallelism_type: ParallelismType::Tensor,
            devices,
            tensor_parallel_size: size,
            pipeline_parallel_size: 1,
            ..Default::default()
        }
    }

    /// Create config for pipeline parallelism
    pub fn pipeline_parallel(devices: Vec<Device>) -> Self {
        let size = devices.len();
        Self {
            parallelism_type: ParallelismType::Pipeline,
            devices,
            tensor_parallel_size: 1,
            pipeline_parallel_size: size,
            ..Default::default()
        }
    }

    /// Create config for data parallelism
    pub fn data_parallel(devices: Vec<Device>) -> Self {
        Self {
            parallelism_type: ParallelismType::Data,
            devices,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            ..Default::default()
        }
    }

    /// Create hybrid config (tensor + pipeline)
    pub fn hybrid(devices: Vec<Device>, tp_size: usize, pp_size: usize) -> Self {
        Self {
            parallelism_type: ParallelismType::Hybrid,
            devices,
            tensor_parallel_size: tp_size,
            pipeline_parallel_size: pp_size,
            ..Default::default()
        }
    }

    /// Get the world size (total number of ranks)
    pub fn world_size(&self) -> usize {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }

    /// Check if parallelism is enabled
    pub fn is_parallel(&self) -> bool {
        self.parallelism_type != ParallelismType::None && self.world_size() > 1
    }

    /// Get device for a specific rank
    pub fn device_for_rank(&self, rank: usize) -> Option<&Device> {
        self.devices.get(rank)
    }

    /// Calculate tensor parallel rank from global rank
    pub fn tp_rank(&self, global_rank: usize) -> usize {
        global_rank % self.tensor_parallel_size
    }

    /// Calculate pipeline parallel rank from global rank
    pub fn pp_rank(&self, global_rank: usize) -> usize {
        global_rank / self.tensor_parallel_size
    }
}

/// Layer distribution configuration for pipeline parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDistribution {
    /// Layer assignments per pipeline stage
    pub stage_layers: Vec<LayerRange>,
    /// Memory requirements per stage (in bytes)
    pub stage_memory: Vec<usize>,
}

/// Range of layers assigned to a pipeline stage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LayerRange {
    /// First layer (inclusive)
    pub start: usize,
    /// Last layer (exclusive)
    pub end: usize,
}

impl LayerRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn contains(&self, layer: usize) -> bool {
        layer >= self.start && layer < self.end
    }
}

impl LayerDistribution {
    /// Create even distribution of layers across stages
    pub fn even_distribution(num_layers: usize, num_stages: usize) -> Self {
        let layers_per_stage = num_layers / num_stages;
        let remainder = num_layers % num_stages;

        let mut stage_layers = Vec::with_capacity(num_stages);
        let mut start = 0;

        for stage in 0..num_stages {
            let extra = if stage < remainder { 1 } else { 0 };
            let end = start + layers_per_stage + extra;
            stage_layers.push(LayerRange::new(start, end));
            start = end;
        }

        Self {
            stage_layers,
            stage_memory: vec![0; num_stages], // To be filled based on actual model
        }
    }

    /// Get stage for a given layer
    pub fn stage_for_layer(&self, layer: usize) -> Option<usize> {
        self.stage_layers.iter().position(|range| range.contains(layer))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_config() {
        let config = ParallelConfig::single_gpu(Device::CPU);
        assert_eq!(config.parallelism_type, ParallelismType::None);
        assert_eq!(config.world_size(), 1);
        assert!(!config.is_parallel());
    }

    #[test]
    fn test_tensor_parallel_config() {
        let config = ParallelConfig::tensor_parallel(vec![Device::CUDA(0), Device::CUDA(1)]);
        assert_eq!(config.parallelism_type, ParallelismType::Tensor);
        assert_eq!(config.tensor_parallel_size, 2);
        assert_eq!(config.world_size(), 2);
        assert!(config.is_parallel());
    }

    #[test]
    fn test_hybrid_config() {
        let config = ParallelConfig::hybrid(
            vec![Device::CUDA(0), Device::CUDA(1), Device::CUDA(2), Device::CUDA(3)],
            2, // tp_size
            2, // pp_size
        );
        assert_eq!(config.world_size(), 4);
        assert_eq!(config.tp_rank(0), 0);
        assert_eq!(config.tp_rank(1), 1);
        assert_eq!(config.tp_rank(2), 0);
        assert_eq!(config.tp_rank(3), 1);
        assert_eq!(config.pp_rank(0), 0);
        assert_eq!(config.pp_rank(1), 0);
        assert_eq!(config.pp_rank(2), 1);
        assert_eq!(config.pp_rank(3), 1);
    }

    #[test]
    fn test_layer_distribution() {
        let dist = LayerDistribution::even_distribution(32, 4);
        assert_eq!(dist.stage_layers.len(), 4);
        assert_eq!(dist.stage_layers[0].start, 0);
        assert_eq!(dist.stage_layers[0].end, 8);
        assert_eq!(dist.stage_layers[3].start, 24);
        assert_eq!(dist.stage_layers[3].end, 32);
    }

    #[test]
    fn test_layer_distribution_uneven() {
        let dist = LayerDistribution::even_distribution(33, 4);
        // 33 layers / 4 stages = 8 per stage with 1 remainder
        // First stage gets extra layer
        assert_eq!(dist.stage_layers[0].len(), 9);
        assert_eq!(dist.stage_layers[1].len(), 8);
    }

    #[test]
    fn test_stage_for_layer() {
        let dist = LayerDistribution::even_distribution(32, 4);
        assert_eq!(dist.stage_for_layer(0), Some(0));
        assert_eq!(dist.stage_for_layer(7), Some(0));
        assert_eq!(dist.stage_for_layer(8), Some(1));
        assert_eq!(dist.stage_for_layer(31), Some(3));
        assert_eq!(dist.stage_for_layer(32), None);
    }
}

