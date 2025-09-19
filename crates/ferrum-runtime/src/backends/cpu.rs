//! CPU backend implementation using ndarray

use crate::{ComputeBackend, DeviceMemoryManager, TensorFactory, TensorLike, TensorOps, TensorRef};
use ferrum_interfaces::backend::{BackendCapabilities, BackendStatus};
use ferrum_types::{DataType, Device, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// CPU tensor implementation using ndarray
#[derive(Debug, Clone)]
pub struct CpuTensor {
    data: ndarray::ArrayD<f32>,
    dtype: DataType,
}

impl CpuTensor {
    pub fn new(data: ndarray::ArrayD<f32>, dtype: DataType) -> Self {
        Self { data, dtype }
    }

    pub fn data(&self) -> &ndarray::ArrayD<f32> {
        &self.data
    }

    pub fn into_data(self) -> ndarray::ArrayD<f32> {
        self.data
    }
}

impl TensorLike for CpuTensor {
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn is_contiguous(&self) -> bool {
        self.data.is_standard_layout()
    }

    fn view(&self, start: &[usize], end: &[usize]) -> Result<TensorRef> {
        let mut slice_info = Vec::new();
        for (&s, &e) in start.iter().zip(end.iter()) {
            slice_info.push(ndarray::Slice::from(s..e));
        }
        
        let view = self.data.slice(ndarray::SliceInfo::new(slice_info).unwrap());
        Ok(Arc::new(CpuTensor::new(view.to_owned(), self.dtype)))
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorRef> {
        let reshaped = self.data.clone().into_shape(shape)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Reshape error: {}", e)))?;
        Ok(Arc::new(CpuTensor::new(reshaped, self.dtype)))
    }

    fn to_cpu(&self) -> Result<TensorRef> {
        Ok(Arc::new(self.clone()))
    }

    fn to_device(&self, device: &Device) -> Result<TensorRef> {
        if matches!(device, Device::CPU) {
            Ok(Arc::new(self.clone()))
        } else {
            Err(ferrum_types::FerrumError::backend(
                "CPU backend cannot move tensors to non-CPU devices"
            ))
        }
    }

    fn to_dtype(&self, dtype: DataType) -> Result<TensorRef> {
        // For simplicity, only support F32 in CPU backend
        if matches!(dtype, DataType::F32) {
            Ok(Arc::new(CpuTensor::new(self.data.clone(), dtype)))
        } else {
            Err(ferrum_types::FerrumError::backend(
                format!("CPU backend dtype conversion not implemented for {:?}", dtype)
            ))
        }
    }
}

/// CPU tensor factory
pub struct CpuTensorFactory;

impl TensorFactory for CpuTensorFactory {
    fn create_tensor(
        &self,
        data: &[f32],
        shape: &[usize],
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        if !matches!(device, Device::CPU) {
            return Err(ferrum_types::FerrumError::backend(
                "CPU factory can only create CPU tensors"
            ));
        }
        
        let array = ndarray::ArrayD::from_shape_vec(shape, data.to_vec())
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Array creation error: {}", e)))?;
        
        Ok(Arc::new(CpuTensor::new(array, dtype)))
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        if !matches!(device, Device::CPU) {
            return Err(ferrum_types::FerrumError::backend(
                "CPU factory can only create CPU tensors"
            ));
        }
        
        let array = ndarray::ArrayD::zeros(shape);
        Ok(Arc::new(CpuTensor::new(array, dtype)))
    }

    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        if !matches!(device, Device::CPU) {
            return Err(ferrum_types::FerrumError::backend(
                "CPU factory can only create CPU tensors"
            ));
        }
        
        let array = ndarray::ArrayD::ones(shape);
        Ok(Arc::new(CpuTensor::new(array, dtype)))
    }

    fn uniform(
        &self,
        shape: &[usize],
        low: f32,
        high: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        if !matches!(device, Device::CPU) {
            return Err(ferrum_types::FerrumError::backend(
                "CPU factory can only create CPU tensors"
            ));
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rng.gen_range(low..high)).collect();
        
        let array = ndarray::ArrayD::from_shape_vec(shape, data)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Array creation error: {}", e)))?;
        
        Ok(Arc::new(CpuTensor::new(array, dtype)))
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: f32,
        std: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        if !matches!(device, Device::CPU) {
            return Err(ferrum_types::FerrumError::backend(
                "CPU factory can only create CPU tensors"
            ));
        }
        
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = rand::thread_rng();
        
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        
        let array = ndarray::ArrayD::from_shape_vec(shape, data)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Array creation error: {}", e)))?;
        
        Ok(Arc::new(CpuTensor::new(array, dtype)))
    }

    fn from_tensor(&self, tensor: &TensorRef, device: &Device) -> Result<TensorRef> {
        tensor.to_device(device)
    }
}

/// CPU tensor operations
pub struct CpuTensorOps;

impl TensorOps for CpuTensorOps {
    fn matmul(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_cpu = get_cpu_tensor(a)?;
        let b_cpu = get_cpu_tensor(b)?;
        
        // Simple 2D matrix multiplication for now
        let a_data = &a_cpu.data;
        let b_data = &b_cpu.data;
        
        if a_data.ndim() != 2 || b_data.ndim() != 2 {
            return Err(ferrum_types::FerrumError::backend(
                "CPU matmul only supports 2D matrices currently"
            ));
        }
        
        let result = a_data.dot(b_data);
        Ok(Arc::new(CpuTensor::new(result.into_dyn(), DataType::F32)))
    }

    fn add(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_cpu = get_cpu_tensor(a)?;
        let b_cpu = get_cpu_tensor(b)?;
        
        let result = &a_cpu.data + &b_cpu.data;
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn sub(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_cpu = get_cpu_tensor(a)?;
        let b_cpu = get_cpu_tensor(b)?;
        
        let result = &a_cpu.data - &b_cpu.data;
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn mul(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_cpu = get_cpu_tensor(a)?;
        let b_cpu = get_cpu_tensor(b)?;
        
        let result = &a_cpu.data * &b_cpu.data;
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn div(&self, a: &TensorRef, b: &TensorRef) -> Result<TensorRef> {
        let a_cpu = get_cpu_tensor(a)?;
        let b_cpu = get_cpu_tensor(b)?;
        
        let result = &a_cpu.data / &b_cpu.data;
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn softmax(&self, tensor: &TensorRef, dim: i32) -> Result<TensorRef> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let data = &cpu_tensor.data;
        
        // Simple softmax implementation along last dimension
        let mut result = data.clone();
        
        // Apply softmax along the specified dimension
        let dim = if dim < 0 {
            (data.ndim() as i32 + dim) as usize
        } else {
            dim as usize
        };
        
        for mut lane in result.lanes_mut(ndarray::Axis(dim)) {
            let max_val = lane.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            lane.mapv_inplace(|x| (x - max_val).exp());
            let sum = lane.sum();
            lane.mapv_inplace(|x| x / sum);
        }
        
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn layer_norm(
        &self,
        input: &TensorRef,
        weight: &TensorRef,
        bias: Option<&TensorRef>,
        eps: f32,
    ) -> Result<TensorRef> {
        let input_cpu = get_cpu_tensor(input)?;
        let weight_cpu = get_cpu_tensor(weight)?;
        let bias_cpu = bias.map(|b| get_cpu_tensor(b)).transpose()?;
        
        let input_data = &input_cpu.data;
        let mut result = input_data.clone();
        
        // Simple layer norm implementation - normalize last dimension
        let last_axis = ndarray::Axis(input_data.ndim() - 1);
        
        for mut lane in result.lanes_mut(last_axis) {
            let mean = lane.mean().unwrap();
            let var = lane.mapv(|x| (x - mean).powi(2)).mean().unwrap();
            let std = (var + eps).sqrt();
            
            lane.mapv_inplace(|x| (x - mean) / std);
            
            // Apply weight and bias
            lane *= &weight_cpu.data.view();
            if let Some(bias_cpu) = &bias_cpu {
                lane += &bias_cpu.data.view();
            }
        }
        
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn rms_norm(&self, input: &TensorRef, weight: &TensorRef, eps: f32) -> Result<TensorRef> {
        let input_cpu = get_cpu_tensor(input)?;
        let weight_cpu = get_cpu_tensor(weight)?;
        
        let input_data = &input_cpu.data;
        let mut result = input_data.clone();
        
        // Simple RMS norm implementation
        let last_axis = ndarray::Axis(input_data.ndim() - 1);
        
        for mut lane in result.lanes_mut(last_axis) {
            let rms = (lane.mapv(|x| x * x).mean().unwrap() + eps).sqrt();
            lane.mapv_inplace(|x| x / rms);
            lane *= &weight_cpu.data.view();
        }
        
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn relu(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let result = cpu_tensor.data.mapv(|x| x.max(0.0));
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn gelu(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let result = cpu_tensor.data.mapv(|x| {
            0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        });
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn silu(&self, tensor: &TensorRef) -> Result<TensorRef> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let result = cpu_tensor.data.mapv(|x| x / (1.0 + (-x).exp()));
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn concat(&self, tensors: &[&TensorRef], dim: usize) -> Result<TensorRef> {
        if tensors.is_empty() {
            return Err(ferrum_types::FerrumError::backend("Cannot concat empty tensor list"));
        }
        
        let cpu_tensors: Result<Vec<_>> = tensors.iter()
            .map(|t| get_cpu_tensor(t))
            .collect();
        let cpu_tensors = cpu_tensors?;
        
        let arrays: Vec<_> = cpu_tensors.iter().map(|t| t.data.view()).collect();
        let result = ndarray::concatenate(ndarray::Axis(dim), &arrays)
            .map_err(|e| ferrum_types::FerrumError::backend(format!("Concat error: {}", e)))?;
        
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn split(&self, tensor: &TensorRef, sizes: &[usize], dim: usize) -> Result<Vec<TensorRef>> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let data = &cpu_tensor.data;
        
        let mut results = Vec::new();
        let mut offset = 0;
        
        for &size in sizes {
            let end = offset + size;
            let slice = data.slice_axis(ndarray::Axis(dim), ndarray::Slice::from(offset..end));
            results.push(Arc::new(CpuTensor::new(slice.to_owned(), DataType::F32)) as TensorRef);
            offset = end;
        }
        
        Ok(results)
    }

    fn transpose(&self, tensor: &TensorRef, dim0: usize, dim1: usize) -> Result<TensorRef> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let result = cpu_tensor.data.clone().swap_axes(dim0, dim1);
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }

    fn permute(&self, tensor: &TensorRef, dims: &[usize]) -> Result<TensorRef> {
        let cpu_tensor = get_cpu_tensor(tensor)?;
        let result = cpu_tensor.data.clone().permuted_axes(dims);
        Ok(Arc::new(CpuTensor::new(result, DataType::F32)))
    }
}

/// CPU backend implementation
pub struct CpuBackend {
    tensor_factory: Arc<CpuTensorFactory>,
    tensor_ops: Arc<CpuTensorOps>,
    memory_manager: Arc<crate::memory::MemoryPool>,
}

impl CpuBackend {
    pub fn new() -> Self {
        let memory_manager = Arc::new(crate::memory::MemoryPool::new(
            Device::CPU,
            crate::memory::MemoryPoolConfig::default(),
        ));
        
        Self {
            tensor_factory: Arc::new(CpuTensorFactory),
            tensor_ops: Arc::new(CpuTensorOps),
            memory_manager,
        }
    }
}

#[async_trait]
impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        use ferrum_types::DataType::*;
        
        BackendCapabilities {
            supported_dtypes: vec![F32],
            supported_devices: vec![Device::CPU],
            max_tensor_dims: 8,
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: false,
            supports_flash_attention: false,
            supports_paged_attention: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            max_batch_size: 64,
            max_sequence_length: 8192,
            memory_alignment: 64,
            supports_custom_kernels: false,
            supports_cuda_graphs: false,
            extra_capabilities: HashMap::new(),
        }
    }

    fn tensor_ops(&self) -> &dyn TensorOps {
        self.tensor_ops.as_ref()
    }

    fn tensor_factory(&self) -> &dyn TensorFactory {
        self.tensor_factory.as_ref()
    }

    fn memory_manager(&self) -> &dyn DeviceMemoryManager {
        self.memory_manager.as_ref()
    }

    fn kernel_executor(&self) -> Option<&dyn crate::KernelExecutor> {
        None
    }

    async fn initialize(&mut self, _device: &Device) -> Result<()> {
        Ok(())
    }

    fn supports_device(&self, device: &Device) -> bool {
        matches!(device, Device::CPU)
    }

    fn version(&self) -> String {
        "1.0.0".to_string()
    }

    async fn synchronize(&self, _device: &Device) -> Result<()> {
        // CPU operations are synchronous
        Ok(())
    }

    fn status(&self) -> BackendStatus {
        BackendStatus {
            is_initialized: true,
            is_ready: true,
            active_devices: vec![Device::CPU],
            memory_usage: HashMap::new(),
            operations_completed: 0,
            last_error: None,
            backend_specific: HashMap::new(),
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Helper function
fn get_cpu_tensor(tensor: &TensorRef) -> Result<&CpuTensor> {
    // This is a simplified implementation - in practice we'd need proper downcasting
    // For now, we'll assume all tensors in CPU backend are CpuTensor
    unimplemented!("Proper tensor downcasting not implemented in this simplified version")
}
