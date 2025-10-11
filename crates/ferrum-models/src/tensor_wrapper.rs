//! Candle Tensor wrapper implementing TensorLike

use candle_core::Tensor;
use ferrum_interfaces::TensorLike;
use ferrum_types::{DataType, Device, FerrumError, Result};
use std::any::Any;

/// Wrapper for Candle Tensor to implement TensorLike
#[derive(Debug, Clone)]
pub struct CandleTensorWrapper {
    tensor: Tensor,
}

impl CandleTensorWrapper {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }
    
    pub fn inner(&self) -> &Tensor {
        &self.tensor
    }
    
    pub fn into_inner(self) -> Tensor {
        self.tensor
    }
}

impl TensorLike for CandleTensorWrapper {
    fn shape(&self) -> &[usize] {
        self.tensor.dims()
    }
    
    fn dtype(&self) -> DataType {
        match self.tensor.dtype() {
            candle_core::DType::F32 => DataType::FP32,
            candle_core::DType::F16 => DataType::FP16,
            candle_core::DType::BF16 => DataType::BF16,
            _ => DataType::FP32,
        }
    }
    
    fn device(&self) -> Device {
        match self.tensor.device() {
            candle_core::Device::Cpu => Device::CPU,
            candle_core::Device::Cuda(_) => Device::CUDA(0),
            candle_core::Device::Metal(_) => {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                return Device::Metal;
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                Device::CPU
            }
        }
    }
    
    fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }
    
    fn view(&self, _start: &[usize], _end: &[usize]) -> Result<ferrum_interfaces::TensorRef> {
        // TODO: Implement tensor slicing
        Err(FerrumError::model("Tensor view not yet implemented"))
    }
    
    fn reshape(&self, shape: &[usize]) -> Result<ferrum_interfaces::TensorRef> {
        let reshaped = self.tensor
            .reshape(shape)
            .map_err(|e| FerrumError::model(format!("Reshape failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(reshaped)))
    }
    
    fn to_cpu(&self) -> Result<ferrum_interfaces::TensorRef> {
        if matches!(self.tensor.device(), candle_core::Device::Cpu) {
            return Ok(std::sync::Arc::new(self.clone()));
        }
        
        let cpu_tensor = self.tensor
            .to_device(&candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("to_cpu failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(cpu_tensor)))
    }
    
    fn to_device(&self, device: &Device) -> Result<ferrum_interfaces::TensorRef> {
        let candle_device = match device {
            Device::CPU => candle_core::Device::Cpu,
            Device::CUDA(id) => candle_core::Device::new_cuda(*id)
                .map_err(|e| FerrumError::device(format!("CUDA device error: {}", e)))?,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => candle_core::Device::new_metal(0)
                .map_err(|e| FerrumError::device(format!("Metal device error: {}", e)))?,
            Device::ROCm(_) => {
                return Err(FerrumError::device("ROCm not supported yet"));
            }
        };
        
        let device_tensor = self.tensor
            .to_device(&candle_device)
            .map_err(|e| FerrumError::model(format!("to_device failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(device_tensor)))
    }
    
    fn to_dtype(&self, dtype: DataType) -> Result<ferrum_interfaces::TensorRef> {
        let candle_dtype = match &dtype {
            DataType::FP32 => candle_core::DType::F32,
            DataType::FP16 => candle_core::DType::F16,
            DataType::BF16 => candle_core::DType::BF16,
            _ => return Err(FerrumError::model(format!("Unsupported dtype: {:?}", dtype))),
        };
        
        let converted = self.tensor
            .to_dtype(candle_dtype)
            .map_err(|e| FerrumError::model(format!("to_dtype failed: {}", e)))?;
        Ok(std::sync::Arc::new(CandleTensorWrapper::new(converted)))
    }
}

