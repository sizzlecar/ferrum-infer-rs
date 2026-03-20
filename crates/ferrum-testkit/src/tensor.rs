//! Mock tensor and tensor factory for testing without any ML backend.

use ferrum_interfaces::{TensorFactory, TensorRef};
use ferrum_types::{DataType, Device, FerrumError, Result};
use std::sync::Arc;

/// A mock tensor that stores shape and optional f32 data.
/// No GPU, no Candle — pure Rust.
#[derive(Clone)]
pub struct MockTensor {
    shape: Vec<usize>,
    dtype: DataType,
    device: Device,
    data_f32: Vec<f32>,
}

impl MockTensor {
    /// Create a zero-filled tensor with given shape.
    pub fn zeros(shape: &[usize], dtype: DataType) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            dtype,
            device: Device::CPU,
            data_f32: vec![0.0; numel],
        }
    }

    /// Create a tensor from f32 data.
    pub fn from_f32(data: Vec<f32>, shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            dtype: DataType::FP32,
            device: Device::CPU,
            data_f32: data,
        }
    }

    /// Create a tensor from u32 token IDs (stored as f32 internally).
    pub fn from_u32(data: &[u32], shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            dtype: DataType::FP32,
            device: Device::CPU,
            data_f32: data.iter().map(|&v| v as f32).collect(),
        }
    }

    /// Wrap as TensorRef.
    pub fn into_ref(self) -> TensorRef {
        Arc::new(self)
    }
}

impl std::fmt::Debug for MockTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl ferrum_interfaces::TensorLike for MockTensor {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn view(&self, start: &[usize], end: &[usize]) -> Result<TensorRef> {
        let new_shape: Vec<usize> = start
            .iter()
            .zip(end.iter())
            .map(|(s, e)| e - s)
            .collect();

        // Compute strides for the original shape (row-major)
        let ndim = self.shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // Copy the viewed region
        let new_numel: usize = new_shape.iter().product();
        let mut data = Vec::with_capacity(new_numel);
        let mut coords = start.to_vec();
        loop {
            // Compute flat index
            let flat: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
            data.push(self.data_f32[flat]);

            // Increment coords (innermost first)
            let mut dim = ndim - 1;
            loop {
                coords[dim] += 1;
                if coords[dim] < end[dim] {
                    break;
                }
                coords[dim] = start[dim];
                if dim == 0 {
                    // All done
                    return Ok(MockTensor {
                        shape: new_shape,
                        dtype: self.dtype,
                        device: self.device.clone(),
                        data_f32: data,
                    }
                    .into_ref());
                }
                dim -= 1;
            }
        }
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorRef> {
        let new_numel: usize = shape.iter().product();
        if new_numel != self.data_f32.len() {
            return Err(FerrumError::backend(format!(
                "Cannot reshape {} elements to {:?}",
                self.data_f32.len(),
                shape
            )));
        }
        Ok(MockTensor {
            shape: shape.to_vec(),
            dtype: self.dtype,
            device: self.device.clone(),
            data_f32: self.data_f32.clone(),
        }
        .into_ref())
    }

    fn to_cpu(&self) -> Result<TensorRef> {
        Ok(self.clone().into_ref())
    }

    fn to_device(&self, _device: &Device) -> Result<TensorRef> {
        Ok(self.clone().into_ref())
    }

    fn to_dtype(&self, dtype: DataType) -> Result<TensorRef> {
        Ok(MockTensor {
            shape: self.shape.clone(),
            dtype,
            device: self.device.clone(),
            data_f32: self.data_f32.clone(),
        }
        .into_ref())
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.data_f32.clone())
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        Ok(self.data_f32.iter().map(|&v| v as u32).collect())
    }

    fn argmax_last_dim_u32(&self) -> Result<u32> {
        self.data_f32
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| FerrumError::backend("Empty tensor"))
    }
}

/// Mock tensor factory implementing TensorFactory without any ML backend.
pub struct MockTensorFactory;

impl TensorFactory for MockTensorFactory {
    fn empty(&self, shape: &[usize], dtype: DataType, _device: Device) -> Result<TensorRef> {
        Ok(MockTensor::zeros(shape, dtype).into_ref())
    }

    fn zeros_like(&self, tensor: &TensorRef) -> Result<TensorRef> {
        Ok(MockTensor::zeros(tensor.shape(), tensor.dtype()).into_ref())
    }

    fn from_slice(
        &self,
        data: &[f32],
        shape: &[usize],
        _dtype: DataType,
        _device: Device,
    ) -> Result<TensorRef> {
        Ok(MockTensor::from_f32(data.to_vec(), shape).into_ref())
    }

    fn to_device(&self, tensor: &TensorRef, _device: Device) -> Result<TensorRef> {
        Ok(MockTensor::zeros(tensor.shape(), tensor.dtype()).into_ref())
    }

    fn narrow(
        &self,
        tensor: &TensorRef,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorRef> {
        let mut new_shape = tensor.shape().to_vec();
        if dim < new_shape.len() {
            new_shape[dim] = length;
        }
        let _ = start; // mock ignores actual data slicing
        Ok(MockTensor::zeros(&new_shape, tensor.dtype()).into_ref())
    }

    fn reshape(&self, tensor: &TensorRef, shape: &[usize]) -> Result<TensorRef> {
        tensor.reshape(shape)
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, _device: &Device) -> Result<TensorRef> {
        Ok(MockTensor::zeros(shape, dtype).into_ref())
    }

    fn ones(&self, shape: &[usize], _dtype: DataType, _device: &Device) -> Result<TensorRef> {
        let numel: usize = shape.iter().product();
        Ok(MockTensor::from_f32(vec![1.0; numel], shape).into_ref())
    }

    fn uniform(
        &self,
        shape: &[usize],
        _low: f32,
        _high: f32,
        dtype: DataType,
        _device: &Device,
    ) -> Result<TensorRef> {
        Ok(MockTensor::zeros(shape, dtype).into_ref())
    }

    fn normal(
        &self,
        shape: &[usize],
        _mean: f32,
        _std: f32,
        dtype: DataType,
        _device: &Device,
    ) -> Result<TensorRef> {
        Ok(MockTensor::zeros(shape, dtype).into_ref())
    }

    fn from_tensor(&self, tensor: &TensorRef, _device: &Device) -> Result<TensorRef> {
        Ok(MockTensor::zeros(tensor.shape(), tensor.dtype()).into_ref())
    }
}
