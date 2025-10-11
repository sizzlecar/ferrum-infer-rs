//! Tensor abstraction with zero-copy and device-aware semantics
//!
//! This module provides the core tensor interface that abstracts over different
//! ML frameworks (Candle, ONNX Runtime, etc.) while maintaining zero-copy
//! semantics and device information.

use ferrum_types::{DataType, Device, Result};
use std::sync::Arc;

/// Core tensor trait for zero-copy, device-aware operations
pub trait TensorLike: Send + Sync + std::fmt::Debug {
    /// Get tensor shape
    fn shape(&self) -> &[usize];

    /// Get tensor data type
    fn dtype(&self) -> DataType;

    /// Get device where tensor resides
    fn device(&self) -> Device;

    /// Get total number of elements
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Get number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Check if tensor is scalar (0-dimensional)
    fn is_scalar(&self) -> bool {
        self.shape().is_empty()
    }

    /// Check if tensor is contiguous in memory
    fn is_contiguous(&self) -> bool;

    /// Get size in bytes for this tensor
    fn size_bytes(&self) -> usize {
        self.numel() * self.dtype().size_bytes()
    }

    /// Create a view/slice of this tensor
    fn view(&self, start: &[usize], end: &[usize]) -> Result<TensorRef>;

    /// Reshape tensor to new shape (must have same number of elements)
    fn reshape(&self, shape: &[usize]) -> Result<TensorRef>;

    /// Convert tensor to CPU device
    fn to_cpu(&self) -> Result<TensorRef>;

    /// Convert tensor to specific device  
    fn to_device(&self, device: &Device) -> Result<TensorRef>;

    /// Convert tensor to specific data type
    fn to_dtype(&self, dtype: DataType) -> Result<TensorRef>;
    
    /// Extract tensor data as Vec<f32> (for logits sampling)
    /// This is a convenience method for backends that need to extract data
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        // Default implementation returns error - backends should override
        Err(crate::FerrumError::model(
            "to_vec_f32 not implemented for this tensor backend"
        ))
    }
}

/// Reference-counted tensor handle for zero-copy sharing
pub type TensorRef = Arc<dyn TensorLike>;

/// Tensor factory for creating tensors on specific backends
pub trait TensorFactory: Send + Sync {
    /// 创建指定形状/数据类型的空张量（`[MVP]`）
    fn empty(&self, shape: &[usize], dtype: DataType, device: Device) -> Result<TensorRef>;
    /// 基于已有张量创建零填充张量（`[MVP]`）
    fn zeros_like(&self, tensor: &TensorRef) -> Result<TensorRef>;
    /// 通过 slice 数据创建张量（`[MVP]`）
    fn from_slice(
        &self,
        data: &[f32],
        shape: &[usize],
        dtype: DataType,
        device: Device,
    ) -> Result<TensorRef>;
    /// 迁移张量到目标设备（`[MVP]`）
    fn to_device(&self, tensor: &TensorRef, device: Device) -> Result<TensorRef>;
    /// 执行窄视图操作（`[MVP]`）
    fn narrow(
        &self,
        tensor: &TensorRef,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorRef>;
    /// reshape 张量（`[MVP]`）
    fn reshape(&self, tensor: &TensorRef, shape: &[usize]) -> Result<TensorRef>;

    /// Create tensor filled with zeros（`[Phase 2+]` 可选实现）
    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef>;

    /// Create tensor filled with ones（`[Phase 2+]`）
    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef>;

    /// Create tensor from uniform random distribution（`[Phase 2+]`）
    fn uniform(
        &self,
        shape: &[usize],
        low: f32,
        high: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef>;

    /// Create tensor from normal distribution（`[Phase 2+]`）
    fn normal(
        &self,
        shape: &[usize],
        mean: f32,
        std: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef>;

    /// Create tensor from existing tensor reference (may involve copying)
    fn from_tensor(&self, tensor: &TensorRef, device: &Device) -> Result<TensorRef>;
}

/// Basic tensor operations
pub trait TensorOps: Send + Sync {
    /// Matrix multiplication
    fn matmul(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef>;

    /// Element-wise addition
    fn add(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef>;

    /// Element-wise subtraction  
    fn sub(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef>;

    /// Element-wise multiplication
    fn mul(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef>;

    /// Element-wise division
    fn div(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef>;

    /// Apply softmax along specified dimension
    fn softmax(&self, tensor: &TensorRef, dim: i32) -> Result<TensorRef>;

    /// Apply layer normalization
    fn layer_norm(
        &self,
        input: &TensorRef,
        weight: &TensorRef,
        bias: Option<&TensorRef>,
        eps: f32,
    ) -> Result<TensorRef>;

    /// Apply RMS normalization  
    fn rms_norm(&self, input: &TensorRef, weight: &TensorRef, eps: f32) -> Result<TensorRef>;

    /// Apply ReLU activation
    fn relu(&self, tensor: &TensorRef) -> Result<TensorRef>;

    /// Apply GELU activation
    fn gelu(&self, tensor: &TensorRef) -> Result<TensorRef>;

    /// Apply SiLU (Swish) activation
    fn silu(&self, tensor: &TensorRef) -> Result<TensorRef>;

    /// Concatenate tensors along specified dimension
    fn concat(&self, tensors: &[&TensorRef], dim: usize) -> Result<TensorRef>;

    /// Split tensor along specified dimension
    fn split(&self, tensor: &TensorRef, sizes: &[usize], dim: usize) -> Result<Vec<TensorRef>>;

    /// Transpose tensor dimensions
    fn transpose(&self, tensor: &TensorRef, dim0: usize, dim1: usize) -> Result<TensorRef>;

    /// Permute tensor dimensions
    fn permute(&self, tensor: &TensorRef, dims: &[usize]) -> Result<TensorRef>;
}

/// GPU-specific tensor operations
#[async_trait::async_trait]
pub trait AsyncTensorOps: TensorOps {
    /// Asynchronous matrix multiplication
    async fn matmul_async(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef>;

    /// Asynchronous softmax
    async fn softmax_async(&self, tensor: &TensorRef, dim: i32) -> Result<TensorRef>;

    /// Synchronize all pending operations
    async fn synchronize(&self) -> Result<()>;
}

/// Tensor batch operations for efficient processing
pub trait TensorBatchOps: Send + Sync {
    /// Batch matrix multiplication for multiple pairs
    fn batch_matmul(
        &self,
        a_batch: &[&TensorRef],
        b_batch: &[&TensorRef],
    ) -> Result<Vec<TensorRef>>;

    /// Stack tensors along new dimension  
    fn stack(&self, tensors: &[&TensorRef], dim: usize) -> Result<TensorRef>;

    /// Unstack tensor along specified dimension
    fn unstack(&self, tensor: &TensorRef, dim: usize) -> Result<Vec<TensorRef>>;

    /// Pad tensors in batch to same shape
    fn pad_batch(&self, tensors: &[&TensorRef], target_shape: &[usize]) -> Result<Vec<TensorRef>>;
}

/// Device-specific tensor memory management
pub trait TensorMemoryManager: Send + Sync {
    /// Pre-allocate tensor of given shape for reuse
    fn preallocate(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef>;

    /// Clear tensor data (set to zeros) without deallocation
    fn clear(&self, tensor: &TensorRef) -> Result<()>;

    /// Get memory usage statistics
    fn memory_stats(&self) -> TensorMemoryStats;

    /// Force garbage collection of unused tensors
    fn gc(&self) -> Result<()>;
}

/// Tensor memory usage statistics
#[derive(Debug, Clone)]
pub struct TensorMemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Currently used memory in bytes  
    pub used_memory: usize,
    /// Number of active tensor references
    pub active_tensors: usize,
    /// Peak memory usage
    pub peak_memory: usize,
}

/// Tensor data access for interop
pub trait TensorDataAccess {
    /// Get read-only access to raw data (CPU only)
    /// Returns None if tensor is not on CPU or data is not contiguous
    fn data_f32(&self) -> Option<&[f32]>;

    /// Get read-only access to raw data as bytes
    fn data_bytes(&self) -> Option<&[u8]>;

    /// Copy tensor data to a Vec<f32> (may involve device-to-host transfer)
    fn to_vec_f32(&self) -> Result<Vec<f32>>;

    /// Copy tensor data to a Vec<u8>
    fn to_vec_u8(&self) -> Result<Vec<u8>>;
}

/// Utility functions for tensor operations
pub mod utils {
    use super::*;

    /// Calculate output shape for matrix multiplication
    pub fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(ferrum_types::FerrumError::backend(
                "Matrix multiplication requires at least 2D tensors",
            ));
        }

        let a_rows = a_shape[a_shape.len() - 2];
        let a_cols = a_shape[a_shape.len() - 1];
        let b_rows = b_shape[b_shape.len() - 2];
        let b_cols = b_shape[b_shape.len() - 1];

        if a_cols != b_rows {
            return Err(ferrum_types::FerrumError::backend(format!(
                "Matrix dimensions mismatch: {} vs {}",
                a_cols, b_rows
            )));
        }

        let mut output_shape = a_shape[..a_shape.len() - 2].to_vec();
        output_shape.push(a_rows);
        output_shape.push(b_cols);

        Ok(output_shape)
    }

    /// Check if shapes are broadcastable
    pub fn are_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
        let max_ndim = shape1.len().max(shape2.len());

        for i in 0..max_ndim {
            let dim1 = shape1.get(shape1.len() - 1 - i).copied().unwrap_or(1);
            let dim2 = shape2.get(shape2.len() - 1 - i).copied().unwrap_or(1);

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Calculate output shape after broadcasting
    pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
        if !are_broadcastable(shape1, shape2) {
            return None;
        }

        let max_ndim = shape1.len().max(shape2.len());
        let mut output_shape = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim1 = shape1.get(shape1.len() - 1 - i).copied().unwrap_or(1);
            let dim2 = shape2.get(shape2.len() - 1 - i).copied().unwrap_or(1);

            output_shape.push(dim1.max(dim2));
        }

        output_shape.reverse();
        Some(output_shape)
    }
}
