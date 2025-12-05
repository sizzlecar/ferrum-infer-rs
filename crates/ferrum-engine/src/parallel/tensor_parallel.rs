//! Tensor Parallelism
//!
//! Implements tensor parallelism for splitting model weights and
//! computations across multiple GPUs.
//!
//! ## Weight Distribution
//!
//! For a Linear layer with weight W of shape [out_features, in_features]:
//! - Column-parallel: Split along out_features, each GPU has W[:, start:end]
//! - Row-parallel: Split along in_features, each GPU has W[start:end, :]
//!
//! ## Communication Patterns
//!
//! - Column-parallel → Row-parallel: All-Reduce
//! - Row-parallel → Column-parallel: All-Gather

use ferrum_types::{Device, FerrumError, Result};
use std::sync::Arc;
use tracing::debug;

/// Tensor parallel configuration
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// World size (number of tensor parallel ranks)
    pub world_size: usize,
    /// Local rank in tensor parallel group
    pub rank: usize,
    /// Device for this rank
    pub device: Device,
    /// Whether to use sequence parallelism
    pub sequence_parallel: bool,
    /// Whether to reduce scatter for efficiency
    pub reduce_scatter: bool,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            device: Device::CPU,
            sequence_parallel: false,
            reduce_scatter: true,
        }
    }
}

impl TensorParallelConfig {
    /// Create config for a specific rank
    pub fn new(world_size: usize, rank: usize, device: Device) -> Self {
        Self {
            world_size,
            rank,
            device,
            ..Default::default()
        }
    }

    /// Check if tensor parallelism is enabled
    pub fn is_parallel(&self) -> bool {
        self.world_size > 1
    }

    /// Calculate shard size for a dimension
    pub fn shard_size(&self, dim_size: usize) -> usize {
        assert!(
            dim_size % self.world_size == 0,
            "Dimension {} must be divisible by world size {}",
            dim_size,
            self.world_size
        );
        dim_size / self.world_size
    }

    /// Calculate offset for this rank's shard
    pub fn shard_offset(&self, dim_size: usize) -> usize {
        self.shard_size(dim_size) * self.rank
    }

    /// Get the range for this rank's shard
    pub fn shard_range(&self, dim_size: usize) -> (usize, usize) {
        let size = self.shard_size(dim_size);
        let start = size * self.rank;
        (start, start + size)
    }
}

/// Tensor parallel group for collective operations
pub struct TensorParallelGroup {
    /// Configuration for this group
    config: TensorParallelConfig,
    /// Devices in this group
    devices: Vec<Device>,
}

impl TensorParallelGroup {
    /// Create a new tensor parallel group
    pub fn new(devices: Vec<Device>, rank: usize) -> Result<Self> {
        if devices.is_empty() {
            return Err(FerrumError::config("No devices provided for tensor parallel group"));
        }
        if rank >= devices.len() {
            return Err(FerrumError::config(format!(
                "Rank {} >= num devices {}",
                rank,
                devices.len()
            )));
        }

        let config = TensorParallelConfig::new(devices.len(), rank, devices[rank].clone());

        Ok(Self { config, devices })
    }

    /// Get configuration
    pub fn config(&self) -> &TensorParallelConfig {
        &self.config
    }

    /// Get all devices in group
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get local rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Get local device
    pub fn device(&self) -> &Device {
        &self.config.device
    }

    /// Check if this rank is the master (rank 0)
    pub fn is_master(&self) -> bool {
        self.config.rank == 0
    }
}

/// Type of parallelism for a layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerParallelType {
    /// Column-parallel (split output features)
    ColumnParallel,
    /// Row-parallel (split input features)
    RowParallel,
    /// No parallelism (replicated)
    Replicated,
}

/// Weight sharding specification
#[derive(Debug, Clone)]
pub struct WeightShard {
    /// Original tensor name
    pub name: String,
    /// Parallel type
    pub parallel_type: LayerParallelType,
    /// Dimension to shard along
    pub shard_dim: usize,
    /// Local shard range (start, end)
    pub shard_range: (usize, usize),
}

impl WeightShard {
    /// Create column-parallel shard specification
    pub fn column_parallel(name: impl Into<String>, dim_size: usize, config: &TensorParallelConfig) -> Self {
        Self {
            name: name.into(),
            parallel_type: LayerParallelType::ColumnParallel,
            shard_dim: 0, // Output dimension
            shard_range: config.shard_range(dim_size),
        }
    }

    /// Create row-parallel shard specification
    pub fn row_parallel(name: impl Into<String>, dim_size: usize, config: &TensorParallelConfig) -> Self {
        Self {
            name: name.into(),
            parallel_type: LayerParallelType::RowParallel,
            shard_dim: 1, // Input dimension
            shard_range: config.shard_range(dim_size),
        }
    }

    /// Create replicated weight specification
    pub fn replicated(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parallel_type: LayerParallelType::Replicated,
            shard_dim: 0,
            shard_range: (0, 0), // Full tensor
        }
    }
}

/// Tensor parallel layer mapping for transformer models
#[derive(Debug, Clone)]
pub struct TransformerParallelMapping {
    /// Number of attention heads per rank
    pub heads_per_rank: usize,
    /// Number of KV heads per rank
    pub kv_heads_per_rank: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Hidden dimension per rank
    pub hidden_per_rank: usize,
    /// Intermediate dimension per rank (for MLP)
    pub intermediate_per_rank: usize,
}

impl TransformerParallelMapping {
    /// Create mapping for a transformer model
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        tp_size: usize,
    ) -> Result<Self> {
        // Validate divisibility
        if num_heads % tp_size != 0 {
            return Err(FerrumError::config(format!(
                "num_heads {} must be divisible by tp_size {}",
                num_heads, tp_size
            )));
        }
        if num_kv_heads % tp_size != 0 {
            return Err(FerrumError::config(format!(
                "num_kv_heads {} must be divisible by tp_size {}",
                num_kv_heads, tp_size
            )));
        }
        if intermediate_dim % tp_size != 0 {
            return Err(FerrumError::config(format!(
                "intermediate_dim {} must be divisible by tp_size {}",
                intermediate_dim, tp_size
            )));
        }

        Ok(Self {
            heads_per_rank: num_heads / tp_size,
            kv_heads_per_rank: num_kv_heads / tp_size,
            head_dim,
            hidden_per_rank: hidden_dim, // Hidden dim is not sharded
            intermediate_per_rank: intermediate_dim / tp_size,
        })
    }

    /// Get Q projection output dimension per rank
    pub fn q_proj_size(&self) -> usize {
        self.heads_per_rank * self.head_dim
    }

    /// Get K projection output dimension per rank
    pub fn k_proj_size(&self) -> usize {
        self.kv_heads_per_rank * self.head_dim
    }

    /// Get V projection output dimension per rank
    pub fn v_proj_size(&self) -> usize {
        self.kv_heads_per_rank * self.head_dim
    }

    /// Get O projection input dimension per rank
    pub fn o_proj_in_size(&self) -> usize {
        self.heads_per_rank * self.head_dim
    }

    /// Get weight shards for attention layer
    pub fn attention_weight_shards(
        &self,
        layer_idx: usize,
        config: &TensorParallelConfig,
    ) -> Vec<WeightShard> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        vec![
            // Q, K, V are column-parallel
            WeightShard::column_parallel(
                format!("{}.q_proj.weight", prefix),
                self.q_proj_size() * config.world_size,
                config,
            ),
            WeightShard::column_parallel(
                format!("{}.k_proj.weight", prefix),
                self.k_proj_size() * config.world_size,
                config,
            ),
            WeightShard::column_parallel(
                format!("{}.v_proj.weight", prefix),
                self.v_proj_size() * config.world_size,
                config,
            ),
            // O is row-parallel
            WeightShard::row_parallel(
                format!("{}.o_proj.weight", prefix),
                self.o_proj_in_size() * config.world_size,
                config,
            ),
        ]
    }

    /// Get weight shards for MLP layer
    pub fn mlp_weight_shards(
        &self,
        layer_idx: usize,
        config: &TensorParallelConfig,
    ) -> Vec<WeightShard> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        vec![
            // Gate and up projections are column-parallel
            WeightShard::column_parallel(
                format!("{}.gate_proj.weight", prefix),
                self.intermediate_per_rank * config.world_size,
                config,
            ),
            WeightShard::column_parallel(
                format!("{}.up_proj.weight", prefix),
                self.intermediate_per_rank * config.world_size,
                config,
            ),
            // Down projection is row-parallel
            WeightShard::row_parallel(
                format!("{}.down_proj.weight", prefix),
                self.intermediate_per_rank * config.world_size,
                config,
            ),
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_parallel_config() {
        let config = TensorParallelConfig::new(4, 2, Device::CUDA(2));
        assert!(config.is_parallel());
        assert_eq!(config.shard_size(128), 32);
        assert_eq!(config.shard_offset(128), 64);
        assert_eq!(config.shard_range(128), (64, 96));
    }

    #[test]
    fn test_tensor_parallel_group() {
        let devices = vec![Device::CUDA(0), Device::CUDA(1), Device::CUDA(2), Device::CUDA(3)];
        let group = TensorParallelGroup::new(devices, 1).unwrap();

        assert_eq!(group.world_size(), 4);
        assert_eq!(group.rank(), 1);
        assert_eq!(group.device(), &Device::CUDA(1));
        assert!(!group.is_master());
    }

    #[test]
    fn test_transformer_parallel_mapping() {
        // Test with invalid intermediate_dim (not divisible by tp_size)
        let mapping = TransformerParallelMapping::new(
            32,     // num_heads
            8,      // num_kv_heads
            128,    // head_dim
            4096,   // hidden_dim
            11009,  // intermediate_dim (11009 % 4 = 1, not divisible)
            4,      // tp_size
        );

        // 11009 is not divisible by 4, so this should fail
        assert!(mapping.is_err());

        // Use a divisible intermediate_dim (11008 is divisible by 4)
        let mapping = TransformerParallelMapping::new(32, 8, 128, 4096, 11008, 4);
        assert!(mapping.is_ok());

        let mapping = mapping.unwrap();
        assert_eq!(mapping.heads_per_rank, 8);
        assert_eq!(mapping.kv_heads_per_rank, 2);
    }

    #[test]
    fn test_weight_shard() {
        let config = TensorParallelConfig::new(4, 1, Device::CUDA(1));
        let shard = WeightShard::column_parallel("test.weight", 4096, &config);

        assert_eq!(shard.parallel_type, LayerParallelType::ColumnParallel);
        assert_eq!(shard.shard_range, (1024, 2048));
    }
}

